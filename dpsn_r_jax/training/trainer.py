import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
import optax
from flax import struct, traverse_util
import flax.core as core
from typing import Any, Callable
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.training.sparse_adam import sparse_adam_update


class TrainState(train_state.TrainState):
    rng: Any
    pool_m: jnp.ndarray
    pool_v: jnp.ndarray
    window_size: int = struct.field(pytree_node=False)
    learning_rate_fn: Callable[[int], float] = struct.field(pytree_node=False)
    # ── Precision routing ──────────────────────────────────────────────────────
    # sigma_anneal_fn(step) → float in (0, 1]: multiplier applied to sigma_max.
    # Returns 1.0 at step 0 (broad, easy learning), decays to ~0 at anneal_steps
    # (precise, exact retrieval).  A constant 1.0 fn is set when annealing is off.
    sigma_anneal_fn: Callable[[int], float] = struct.field(pytree_node=False)


def _make_sigma_anneal_fn(sigma_anneal_steps: int, sigma_target_ratio: float):
    """Build a cosine decay schedule for sigma_max_scale.

    sigma_max_scale = 1.0 at step 0,
                    = sigma_target_ratio at step >= sigma_anneal_steps

    Uses cosine annealing for smooth decay (avoids abrupt gradient changes).
    """
    if sigma_anneal_steps <= 0:
        return lambda step: 1.0

    def fn(step):
        t  = jnp.minimum(step, sigma_anneal_steps) / sigma_anneal_steps
        cos = 0.5 * (1 + jnp.cos(jnp.pi * t))   # 1 → 0 (cosine)
        return sigma_target_ratio + (1.0 - sigma_target_ratio) * cos

    return fn


def create_train_state(rng, config, learning_rate_fn=None):
    model = DPSNR(config)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables["params"]

    flat_params = traverse_util.flatten_dict(params)
    pool_key = ("pool", "params_storage")
    pool_params = flat_params[pool_key]

    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    # If no schedule provided, create a constant schedule
    if learning_rate_fn is None:
        from dpsn_r_jax.training.lr_schedules import create_constant_schedule
        learning_rate_fn = create_constant_schedule(config.learning_rate)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate_fn)
    )
    opt_state = tx.init(dense_params)

    pool_m = jnp.zeros_like(pool_params)
    pool_v = jnp.zeros_like(pool_params)

    # ── Sigma annealing schedule ────────────────────────────────────────────────
    # sigma_target is expressed as a fraction of sigma_max:
    #   sigma_target_ratio = sigma_target / sigma_max
    sigma_target_ratio = config.sigma_target / max(config.sigma_max, 1e-8)
    sigma_anneal_fn = _make_sigma_anneal_fn(
        config.sigma_anneal_steps, sigma_target_ratio
    )

    return TrainState(
        step=jnp.array(0, dtype=jnp.int32),
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        rng=rng,
        pool_m=pool_m,
        pool_v=pool_v,
        window_size=config.max_k,
        learning_rate_fn=learning_rate_fn,
        sigma_anneal_fn=sigma_anneal_fn,
    )


import functools


def chunked_lm_loss(
    hidden: jnp.ndarray,
    labels: jnp.ndarray,
    decode_fn,
    pad_token_id: int,
    chunk_size: int,
) -> jnp.ndarray:
    """Compute LM cross-entropy loss without ever materialising the full (B, T, V) logits.

    The LM head (decode_fn) is applied to `chunk_size` samples at a time via
    jax.lax.scan with a checkpointed body.  XLA only keeps (chunk_size, T, V)
    logits in HBM at any moment — typically ~0.4 GB instead of ~13 GB for
    BS=256 on TPU v5e-8.

    Args:
        hidden:       (B, T, D) hidden states from encode_to_hidden.
        labels:       (B, T)    integer token ids (same as input_ids).
        decode_fn:    Callable (chunk_h: (C, T, D)) -> (C, T, V) logits.
        pad_token_id: Padding token id (masked out of loss).
        chunk_size:   Number of samples to decode at once.

    Returns:
        Scalar loss averaged over non-padding tokens.
    """
    B, T, D = hidden.shape

    # Pad batch to a multiple of chunk_size if needed
    remainder = B % chunk_size
    if remainder != 0:
        pad = chunk_size - remainder
        hidden = jnp.concatenate(
            [hidden, jnp.zeros((pad, T, D), dtype=hidden.dtype)], axis=0
        )
        labels = jnp.concatenate(
            [labels, jnp.zeros((pad, T), dtype=labels.dtype)], axis=0
        )
        B_padded = B + pad
    else:
        B_padded = B

    n_chunks = B_padded // chunk_size
    hidden_chunks = hidden.reshape(n_chunks, chunk_size, T, D)
    labels_chunks = labels.reshape(n_chunks, chunk_size, T)

    def scan_body(carry, chunk):
        chunk_h, chunk_l = chunk
        # Apply LM head: (chunk_size, T, D) → (chunk_size, T, V)
        chunk_logits = decode_fn(chunk_h)
        # Always compute loss in float32 for stability
        chunk_logits = chunk_logits.astype(jnp.float32)

        shift_logits = chunk_logits[:, :-1, :]
        shift_labels = chunk_l[:, 1:]

        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )
        mask = (shift_labels != pad_token_id).astype(jnp.float32)
        return carry, (per_token_loss * mask, mask)

    # ── Checkpoint the ENTIRE scan body, not just decode_fn ───────────────
    # With jax.checkpoint(decode_fn) only, scan backward still stores
    # chunk_logits (the OUTPUT of decode_fn) across ALL chunks, producing a
    # f32[n_chunks, chunk_size, T-1, V] buffer — that's as big as the full
    # batch logits and defeats the purpose.
    #
    # By checkpointing the whole body, scan stores only the tiny per-chunk
    # INPUTS (chunk_h, chunk_l) and RECOMPUTES the body (decode + cross-entropy)
    # during backward.  Peak logit memory = one chunk at a time.
    _, (weighted_losses, masks) = jax.lax.scan(
        jax.checkpoint(scan_body),
        None,
        (hidden_chunks, labels_chunks),
    )

    total_loss = weighted_losses.sum()
    total_mask = masks.sum()
    return total_loss / (total_mask + 1e-9)


@functools.partial(jax.jit, static_argnames=["pad_token_id", "precision_loss_weight", "sigma_anneal_steps", "use_bf16", "loss_chunk_size"], donate_argnums=(0,))
def train_step(state, batch, pad_token_id=0,
               precision_loss_weight=0.0, sigma_anneal_steps=0,
               use_bf16=False, loss_chunk_size=0):
    """One training step with precision routing support.

    New vs original:
      - Computes sigma_max_scale from state.sigma_anneal_fn(step).
      - Passes scale into model so retrieval tightens progressively.
      - Optionally adds a precision auxiliary loss: weight * mean_sigma.
        The weight is linearly ramped from 0 → precision_loss_weight over
        sigma_anneal_steps, so the model only gets penalised for broad sigma
        once it has had time to learn coarse routing first.
      - Returns (state, loss, mean_sigma) — mean_sigma is logged.

    Args:
        state:                   TrainState
        batch:                   (B, T) integer token ids
        pad_token_id:            ignored positions
        precision_loss_weight:   max weight for sigma penalty (0 = disabled)
        sigma_anneal_steps:      steps over which precision weight is ramped in
        use_bf16:                if True, cast params & batch to bfloat16 for
                                 the forward pass (halves activation memory)

    Returns:
        new_state, loss (float), mean_sigma (float)
    """
    print("Compiling train_step for XLA...", flush=True)
    dropout_rng, new_rng = random.split(state.rng)

    # ── Current sigma scale from annealing schedule ────────────────────────────
    sigma_scale = state.sigma_anneal_fn(state.step)

    # ── Precision loss ramp weight ─────────────────────────────────────────────
    # Linearly ramp from 0 → precision_loss_weight over sigma_anneal_steps.
    # This ensures the model learns coarse routing before being penalised for
    # imprecision.
    if sigma_anneal_steps > 0 and precision_loss_weight > 0.0:
        ramp = jnp.minimum(1.0, (state.step + 1) / sigma_anneal_steps)
        effective_precision_weight = precision_loss_weight * ramp
    else:
        ramp = 0.0
        effective_precision_weight = 0.0

    def loss_fn(params):
        # ── BFloat16 mixed precision ───────────────────────────────────────
        # Cast params to bf16 for the forward pass only.  Optimizer state
        # stays in float32 for numerical stability.  All activations and
        # logits are computed in bf16, halving their memory footprint.
        if use_bf16:
            compute_params = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16), params
            )
        else:
            compute_params = params

        if loss_chunk_size > 0:
            # ── Chunked LM loss path ───────────────────────────────────────
            # Run encode+reasoning (no LM head) → tiny (B, T, D) hidden state.
            # Then apply LM head in sub-batches of `loss_chunk_size` so the
            # peak (chunk_size, T, V) logits tensor is tiny.
            state_hidden, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params},
                batch,
                deterministic=False,
                sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
                method=lambda mod, *a, **kw: mod.encode_to_hidden(*a, **kw),
            )

            def decode_fn(chunk_h):
                return state.apply_fn(
                    {"params": compute_params},
                    chunk_h,
                    method=lambda mod, h: mod.controller.decode(h),
                )

            lm_loss = chunked_lm_loss(
                state_hidden, batch, decode_fn, pad_token_id, loss_chunk_size
            )
            lm_loss = lm_loss.astype(jnp.float32)
        else:
            # ── Standard path: full (B, T, V) logits ──────────────────────
            logits, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params},
                batch,
                deterministic=False,
                sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
            )

            # Cast logits back to float32 for numerically stable loss computation
            logits = logits.astype(jnp.float32)

            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]

            lm_loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            mask = (shift_labels != pad_token_id).astype(jnp.float32)
            lm_loss = (lm_loss * mask).sum() / (mask.sum() + 1e-9)

        # ── Precision auxiliary loss ────────────────────────────────────────
        # Penalises broad sigma (large = imprecise retrieval).
        # The model is rewarded for using narrower, more targeted windows.
        precision_loss = effective_precision_weight * jnp.float32(mean_sigma)

        total_loss = lm_loss + precision_loss
        return total_loss, (indices, mean_sigma)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (indices, mean_sigma)), grads = grad_fn(state.params)

    pool_key = ("pool", "params_storage")
    flat_params = traverse_util.flatten_dict(state.params)
    flat_grads = traverse_util.flatten_dict(grads)

    pool_params = jnp.asarray(flat_params[pool_key])
    pool_grads = jnp.asarray(flat_grads[pool_key])

    dense_flat_grads = {k: v for k, v in flat_grads.items() if k != pool_key}
    dense_grads = traverse_util.unflatten_dict(dense_flat_grads)

    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    updates, new_opt_state = state.tx.update(dense_grads, state.opt_state, dense_params)
    new_dense_params = optax.apply_updates(dense_params, updates)

    # ── Sparse Adam pool update ────────────────────────────────────────────────
    # indices shape: (heads*B, max_loops) — same sparse logic whether 1D or 2D pool.
    # For 2D pool, flat_start indices are already flattened to 1D; same treatment.
    B_times_H, L = indices.shape
    W = state.window_size

    # Expand all window offsets: (B*H, L, W) → flatten to (B*H*L*W,)
    # jnp.unique is NOT needed here: .at[idx].set() handles duplicate indices
    # correctly (last write wins), which is acceptable for Adam moment updates
    # on nearby pool vectors.  Removing unique() saves a full sort per step.
    offsets     = jnp.arange(W)                                    # (W,)
    flat_touched = (
        indices[:, :, None] + offsets[None, None, :]               # (B*H, L, W)
    ).reshape(-1)                                                   # (B*H*L*W,)

    # Clip to valid pool range (handles both 1D and 2D pool flat indices)
    pool_size   = pool_params.reshape(-1, pool_params.shape[-1]).shape[0]
    safe_indices_raw = jnp.clip(flat_touched, 0, pool_size - 1)

    # ── Bug #3 Fix: Sort indices to enable coalesced HBM reads ────────────────
    # XLA scatter/gather on unsorted indices results in random HBM accesses
    # (irregular memory access pattern = throttled bandwidth).  Sorting once
    # here (O(N log N) on a small index vector) unlocks sequential HBM reads
    # in the subsequent gather and scatter, which is vastly cheaper at scale.
    sort_order   = jnp.argsort(safe_indices_raw)
    safe_indices = safe_indices_raw[sort_order]

    # Reshape pool for flat indexing (handles both 1D and 2D pool storage)
    pool_flat = pool_params.reshape(-1, pool_params.shape[-1])
    pool_m_flat = state.pool_m.reshape(-1, state.pool_m.shape[-1])
    pool_v_flat = state.pool_v.reshape(-1, state.pool_v.shape[-1])
    pool_grads_flat = pool_grads.reshape(-1, pool_grads.shape[-1])

    p_slice = pool_flat[safe_indices]
    g_slice = pool_grads_flat[safe_indices]
    m_slice = pool_m_flat[safe_indices]
    v_slice = pool_v_flat[safe_indices]

    current_lr = state.learning_rate_fn(state.step + 1)

    pool_grad_norm = jnp.sqrt(jnp.sum(pool_grads**2) + 1e-9)
    pool_grad_scale = jnp.minimum(1.0, 1.0 / pool_grad_norm)
    clipped_g_slice = g_slice * pool_grad_scale

    new_p_s, new_m_s, new_v_s = sparse_adam_update(
        p_slice,
        clipped_g_slice,
        m_slice,
        v_slice,
        state.step + 1,
        lr=current_lr,
    )

    # Scatter updates back into the flat pool arrays.
    # Duplicates in safe_indices write the same value twice — harmless.
    new_pool_flat  = pool_flat.at[safe_indices].set(new_p_s)
    new_pool_m_flat = pool_m_flat.at[safe_indices].set(new_m_s)
    new_pool_v_flat = pool_v_flat.at[safe_indices].set(new_v_s)

    # Reshape back to original pool shape (works for both 1D and 2D storage)
    new_pool_params = new_pool_flat.reshape(pool_params.shape)
    new_pool_m = new_pool_m_flat.reshape(state.pool_m.shape)
    new_pool_v = new_pool_v_flat.reshape(state.pool_v.shape)

    new_flat_params = traverse_util.flatten_dict(new_dense_params)
    new_flat_params[pool_key] = new_pool_params
    new_params = traverse_util.unflatten_dict(new_flat_params)

    state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        pool_m=new_pool_m,
        pool_v=new_pool_v,
        rng=new_rng,
    )

    return state, loss, mean_sigma


# ── Gradient Accumulation ──────────────────────────────────────────────────────
@functools.partial(
    jax.jit,
    static_argnames=[
        "pad_token_id", "precision_loss_weight", "sigma_anneal_steps",
        "use_bf16", "loss_chunk_size", "grad_accum_steps",
    ],
    donate_argnums=(0,),
)
def grad_accum_step(
    state,
    micro_batches,          # (grad_accum_steps, micro_B, T)
    pad_token_id=0,
    precision_loss_weight=0.0,
    sigma_anneal_steps=0,
    use_bf16=False,
    loss_chunk_size=0,
    grad_accum_steps=1,
):
    """Gradient-accumulation training step.

    Runs `grad_accum_steps` forward+backward passes on consecutive micro-batches,
    averages the gradients, then applies ONE optimizer update.  This lets you
    train with an effective batch of (grad_accum_steps × micro_batch_size) while
    only ever holding one micro-batch of activations in HBM at a time.

    Args:
        state:              TrainState
        micro_batches:      (grad_accum_steps, micro_B, T) — pre-split by caller
        pad_token_id:       ignored positions
        grad_accum_steps:   number of micro-batches to accumulate (static)
        (all other args):   same as train_step

    Returns:
        new_state, avg_loss (float), avg_mean_sigma (float)
    """
    jax.debug.print("Tracing grad_accum_step for XLA...")
    dropout_rng, new_rng = random.split(state.rng)
    sigma_scale = state.sigma_anneal_fn(state.step)

    if sigma_anneal_steps > 0 and precision_loss_weight > 0.0:
        ramp = jnp.minimum(1.0, (state.step + 1) / sigma_anneal_steps)
        effective_precision_weight = precision_loss_weight * ramp
    else:
        effective_precision_weight = 0.0

    def loss_fn_micro(params, micro_batch):
        """Forward+loss for a single micro-batch. Returns (loss, aux)."""
        if use_bf16:
            compute_params = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.bfloat16), params
            )
        else:
            compute_params = params

        if loss_chunk_size > 0:
            state_hidden, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params},
                micro_batch,
                deterministic=False,
                sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
                method=lambda mod, *a, **kw: mod.encode_to_hidden(*a, **kw),
            )
            def decode_fn(chunk_h):
                return state.apply_fn(
                    {"params": compute_params},
                    chunk_h,
                    method=lambda mod, h: mod.controller.decode(h),
                )
            lm_loss = chunked_lm_loss(
                state_hidden, micro_batch, decode_fn, pad_token_id, loss_chunk_size
            ).astype(jnp.float32)
        else:
            logits, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params},
                micro_batch,
                deterministic=False,
                sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
            )
            logits = logits.astype(jnp.float32)
            shift_logits = logits[:, :-1, :]
            shift_labels = micro_batch[:, 1:]
            per_token = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            mask = (shift_labels != pad_token_id).astype(jnp.float32)
            lm_loss = (per_token * mask).sum() / (mask.sum() + 1e-9)

        precision_loss = effective_precision_weight * jnp.float32(mean_sigma)
        total_loss = lm_loss + precision_loss
        return total_loss, (indices, mean_sigma)

    grad_fn_micro = jax.value_and_grad(loss_fn_micro, has_aux=True)

    # ── Accumulate gradients over all micro-batches ────────────────────────
    # Use jax.lax.scan so XLA sees a single kernel rather than N unrolled ones.
    def scan_body(carry, micro_batch):
        acc_grads, acc_loss, acc_sigma = carry
        (loss, (indices, mean_sigma)), grads = grad_fn_micro(state.params, micro_batch)
        # Sum gradients (we divide by grad_accum_steps at the end)
        acc_grads   = jax.tree_util.tree_map(jnp.add, acc_grads, grads)
        acc_loss    = acc_loss  + loss
        acc_sigma   = acc_sigma + mean_sigma
        return (acc_grads, acc_loss, acc_sigma), indices

    # Initialise accumulators with zeros shaped like params
    zero_grads   = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init_carry   = (zero_grads, jnp.float32(0.0), jnp.float32(0.0))

    (summed_grads, total_loss, total_sigma), all_indices = jax.lax.scan(
        scan_body,                   # no jax.checkpoint needed; grad is taken inside!
        init_carry,
        micro_batches,               # (grad_accum_steps, micro_B, T)
        unroll=4,                    # Let XLA overlap FSDP network syncs!
    )

    # Average over accumulation steps
    scale       = 1.0 / grad_accum_steps
    avg_grads   = jax.tree_util.tree_map(lambda g: g * scale, summed_grads)
    avg_loss    = total_loss  * scale
    avg_sigma   = total_sigma * scale

    # Use ALL micro-batch indices for the sparse pool update, not just the last one
    # all_indices shape: (grad_accum_steps, heads*micro_B, max_loops)
    indices = all_indices.reshape(-1, all_indices.shape[-1])

    # ── Optimizer update (identical to train_step) ────────────────────────
    pool_key      = ("pool", "params_storage")
    flat_params   = traverse_util.flatten_dict(state.params)
    flat_grads    = traverse_util.flatten_dict(avg_grads)

    pool_params   = jnp.asarray(flat_params[pool_key])
    pool_grads    = jnp.asarray(flat_grads[pool_key])

    dense_flat_grads  = {k: v for k, v in flat_grads.items()  if k != pool_key}
    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_grads   = traverse_util.unflatten_dict(dense_flat_grads)
    dense_params  = traverse_util.unflatten_dict(dense_flat_params)

    updates, new_opt_state = state.tx.update(dense_grads, state.opt_state, dense_params)
    new_dense_params = optax.apply_updates(dense_params, updates)

    W             = state.window_size
    offsets       = jnp.arange(W)
    flat_touched  = (
        indices[:, :, None] + offsets[None, None, :]
    ).reshape(-1)
    pool_size     = pool_params.reshape(-1, pool_params.shape[-1]).shape[0]
    safe_indices_raw = jnp.clip(flat_touched, 0, pool_size - 1)

    # ── Bug #3 Fix: Sort indices to enable coalesced HBM reads ────────────────
    sort_order   = jnp.argsort(safe_indices_raw)
    safe_indices = safe_indices_raw[sort_order]

    pool_flat     = pool_params.reshape(-1, pool_params.shape[-1])
    pool_m_flat   = state.pool_m.reshape(-1, state.pool_m.shape[-1])
    pool_v_flat   = state.pool_v.reshape(-1, state.pool_v.shape[-1])
    pool_grads_flat = pool_grads.reshape(-1, pool_grads.shape[-1])

    p_slice = pool_flat[safe_indices]
    g_slice = pool_grads_flat[safe_indices]
    m_slice = pool_m_flat[safe_indices]
    v_slice = pool_v_flat[safe_indices]

    current_lr      = state.learning_rate_fn(state.step + 1)
    pool_grad_norm  = jnp.sqrt(jnp.sum(pool_grads**2) + 1e-9)
    pool_grad_scale = jnp.minimum(1.0, 1.0 / pool_grad_norm)
    clipped_g_slice = g_slice * pool_grad_scale

    new_p_s, new_m_s, new_v_s = sparse_adam_update(
        p_slice, clipped_g_slice, m_slice, v_slice, state.step + 1, lr=current_lr
    )

    new_pool_flat   = pool_flat.at[safe_indices].set(new_p_s)
    new_pool_m_flat = pool_m_flat.at[safe_indices].set(new_m_s)
    new_pool_v_flat = pool_v_flat.at[safe_indices].set(new_v_s)

    new_pool_params = new_pool_flat.reshape(pool_params.shape)
    new_pool_m      = new_pool_m_flat.reshape(state.pool_m.shape)
    new_pool_v      = new_pool_v_flat.reshape(state.pool_v.shape)

    new_flat_params          = traverse_util.flatten_dict(new_dense_params)
    new_flat_params[pool_key] = new_pool_params
    new_params               = traverse_util.unflatten_dict(new_flat_params)

    state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        pool_m=new_pool_m,
        pool_v=new_pool_v,
        rng=new_rng,
    )
    return state, avg_loss, avg_sigma
