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
    # sigma_anneal_fn(step) -> float in (0, 1]: multiplier applied to sigma_max.
    sigma_anneal_fn: Callable[[int], float] = struct.field(pytree_node=False)


def _make_sigma_anneal_fn(sigma_anneal_steps: int, sigma_target_ratio: float):
    """Build a cosine decay schedule for sigma_max_scale."""
    if sigma_anneal_steps <= 0:
        return lambda step: 1.0

    def fn(step):
        t   = jnp.minimum(step, sigma_anneal_steps) / sigma_anneal_steps
        cos = 0.5 * (1 + jnp.cos(jnp.pi * t))
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


def chunked_lm_loss(hidden, labels, decode_fn, pad_token_id, chunk_size):
    """LM loss without ever materialising the full (B, T, V) logits tensor."""
    B, T, D = hidden.shape

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

    n_chunks      = B_padded // chunk_size
    hidden_chunks = hidden.reshape(n_chunks, chunk_size, T, D)
    labels_chunks = labels.reshape(n_chunks, chunk_size, T)

    def scan_body(carry, chunk):
        chunk_h, chunk_l = chunk
        chunk_logits = decode_fn(chunk_h).astype(jnp.float32)
        shift_logits = chunk_logits[:, :-1, :]
        shift_labels = chunk_l[:, 1:]
        per_token_loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )
        mask = (shift_labels != pad_token_id).astype(jnp.float32)
        return carry, (per_token_loss * mask, mask)

    _, (weighted_losses, masks) = jax.lax.scan(
        jax.checkpoint(scan_body),
        None,
        (hidden_chunks, labels_chunks),
    )
    return weighted_losses.sum() / (masks.sum() + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Shared optimizer update — called from train_step and _finalize_grad_accum.
# Python helper (not JIT by itself); gets traced inside the JIT context of
# its callers.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_optimizer_update(state, grads, indices, new_rng):
    """Dense AdamW update + sparse Adam pool update, returns new_state."""
    pool_key = ("pool", "params_storage")
    flat_params = traverse_util.flatten_dict(state.params)
    flat_grads  = traverse_util.flatten_dict(grads)

    pool_params = jnp.asarray(flat_params[pool_key])
    pool_grads  = jnp.asarray(flat_grads[pool_key])

    dense_flat_grads  = {k: v for k, v in flat_grads.items()  if k != pool_key}
    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_grads  = traverse_util.unflatten_dict(dense_flat_grads)
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    updates, new_opt_state = state.tx.update(dense_grads, state.opt_state, dense_params)
    new_dense_params = optax.apply_updates(dense_params, updates)

    # ── Sparse Adam pool update ────────────────────────────────────────────
    W            = state.window_size
    offsets      = jnp.arange(W)
    flat_touched = (
        indices[:, :, None] + offsets[None, None, :]
    ).reshape(-1)

    pool_size        = pool_params.reshape(-1, pool_params.shape[-1]).shape[0]
    safe_indices_raw = jnp.clip(flat_touched, 0, pool_size - 1)

    # Sort for coalesced HBM reads (Bug #3 fix)
    sort_order   = jnp.argsort(safe_indices_raw)
    safe_indices = safe_indices_raw[sort_order]

    pool_flat       = pool_params.reshape(-1, pool_params.shape[-1])
    pool_m_flat     = state.pool_m.reshape(-1, state.pool_m.shape[-1])
    pool_v_flat     = state.pool_v.reshape(-1, state.pool_v.shape[-1])
    pool_grads_flat = pool_grads.reshape(-1, pool_grads.shape[-1])

    p_slice = pool_flat[safe_indices]
    g_slice = pool_grads_flat[safe_indices]
    m_slice = pool_m_flat[safe_indices]
    v_slice = pool_v_flat[safe_indices]

    current_lr      = state.learning_rate_fn(state.step + 1)
    pool_grad_norm  = jnp.sqrt(jnp.sum(pool_grads ** 2) + 1e-9)
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

    new_flat_params           = traverse_util.flatten_dict(new_dense_params)
    new_flat_params[pool_key] = new_pool_params
    new_params                = traverse_util.unflatten_dict(new_flat_params)

    return state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        pool_m=new_pool_m,
        pool_v=new_pool_v,
        rng=new_rng,
    )


# ─────────────────────────────────────────────────────────────────────────────
# train_step
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(
    jax.jit,
    static_argnames=[
        "pad_token_id", "precision_loss_weight",
        "sigma_anneal_steps", "use_bf16", "loss_chunk_size",
    ],
    donate_argnums=(0,),
)
def train_step(
    state, batch, pad_token_id=0,
    precision_loss_weight=0.0, sigma_anneal_steps=0,
    use_bf16=False, loss_chunk_size=0,
):
    """Single training step. Returns (new_state, loss, mean_sigma)."""
    print("Compiling train_step for XLA...", flush=True)
    dropout_rng, new_rng = random.split(state.rng)
    sigma_scale = state.sigma_anneal_fn(state.step)

    if sigma_anneal_steps > 0 and precision_loss_weight > 0.0:
        ramp = jnp.minimum(1.0, (state.step + 1) / sigma_anneal_steps)
        effective_precision_weight = precision_loss_weight * ramp
    else:
        effective_precision_weight = 0.0

    def loss_fn(params):
        compute_params = (
            jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
            if use_bf16 else params
        )
        if loss_chunk_size > 0:
            state_hidden, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params}, batch,
                deterministic=False, sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
                method=lambda mod, *a, **kw: mod.encode_to_hidden(*a, **kw),
            )
            def decode_fn(chunk_h):
                return state.apply_fn(
                    {"params": compute_params}, chunk_h,
                    method=lambda mod, h: mod.controller.decode(h),
                )
            lm_loss = chunked_lm_loss(
                state_hidden, batch, decode_fn, pad_token_id, loss_chunk_size
            ).astype(jnp.float32)
        else:
            logits, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params}, batch,
                deterministic=False, sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
            )
            logits = logits.astype(jnp.float32)
            shift_logits = logits[:, :-1, :]
            shift_labels = batch[:, 1:]
            lm_loss = optax.softmax_cross_entropy_with_integer_labels(
                shift_logits, shift_labels
            )
            mask = (shift_labels != pad_token_id).astype(jnp.float32)
            lm_loss = (lm_loss * mask).sum() / (mask.sum() + 1e-9)

        return (
            lm_loss + effective_precision_weight * jnp.float32(mean_sigma),
            (indices, mean_sigma),
        )

    (loss, (indices, mean_sigma)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    new_state = _apply_optimizer_update(state, grads, indices, new_rng)
    return new_state, loss, mean_sigma


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Accumulation — Python-loop design
#
# ROOT CAUSE of the previous 220-330 GB RAM blowup:
#   Embedding jax.value_and_grad inside jax.lax.scan forces XLA to lower the
#   ENTIRE differentiated loop (forward + backward, ~35 000 HLO ops per step)
#   into a single fused graph.  XLA's optimization passes (CSE, algebraic
#   simplification) are super-linear in HLO node count, so even with unroll=1
#   the compiler repeatedly allocates and frees hundreds of GB trying to
#   simplify the massive graph — and often never finishes.
#
# THE FIX — three separate compilation units:
#   1. _compute_micro_grads  (JIT)  — forward+backward for ONE micro-batch
#   2. _finalize_grad_accum  (JIT)  — average gradients + one optimizer update
#   3. grad_accum_step       (Python) — plain for-loop calling (1) N times
#
# JAX's *asynchronous dispatch* means each _compute_micro_grads call is
# enqueued on the device immediately and the host Python loop proceeds without
# blocking.  The device overlaps successive micro-batch computations, so there
# is zero throughput penalty compared to the scan approach.
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(
    jax.jit,
    static_argnames=["pad_token_id", "use_bf16", "loss_chunk_size"],
)
def _compute_micro_grads(
    state,
    micro_batch,
    dropout_rng,
    sigma_scale,
    effective_precision_weight,
    pad_token_id: int = 0,
    use_bf16: bool = False,
    loss_chunk_size: int = 0,
):
    """JIT: forward+backward for ONE micro-batch. No accumulation, no optimizer.

    Returns:
        grads, loss, mean_sigma, indices
    """
    def loss_fn(params):
        compute_params = (
            jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
            if use_bf16 else params
        )
        if loss_chunk_size > 0:
            state_hidden, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params}, micro_batch,
                deterministic=False, sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
                method=lambda mod, *a, **kw: mod.encode_to_hidden(*a, **kw),
            )
            def decode_fn(chunk_h):
                return state.apply_fn(
                    {"params": compute_params}, chunk_h,
                    method=lambda mod, h: mod.controller.decode(h),
                )
            lm_loss = chunked_lm_loss(
                state_hidden, micro_batch, decode_fn, pad_token_id, loss_chunk_size
            ).astype(jnp.float32)
        else:
            logits, (_, indices, mean_sigma) = state.apply_fn(
                {"params": compute_params}, micro_batch,
                deterministic=False, sigma_max_scale=sigma_scale,
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

        return (
            lm_loss + effective_precision_weight * jnp.float32(mean_sigma),
            (indices, mean_sigma),
        )

    (loss, (indices, mean_sigma)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    return grads, loss, mean_sigma, indices


@functools.partial(
    jax.jit,
    static_argnames=["grad_accum_steps"],
    donate_argnums=(0,),
)
def _finalize_grad_accum(state, summed_grads, all_indices, new_rng, grad_accum_steps: int):
    """JIT: scale summed gradients by 1/N, then run the optimizer update."""
    scale     = jnp.float32(1.0 / grad_accum_steps)
    avg_grads = jax.tree_util.tree_map(lambda g: g * scale, summed_grads)
    return _apply_optimizer_update(state, avg_grads, all_indices, new_rng)


def grad_accum_step(
    state,
    micro_batches,
    pad_token_id=0,
    precision_loss_weight=0.0,
    sigma_anneal_steps=0,
    use_bf16=False,
    loss_chunk_size=0,
    grad_accum_steps=1,
):
    """Gradient accumulation via Python loop (no lax.scan, no compile blowup).

    Args:
        state:            TrainState
        micro_batches:    (grad_accum_steps, micro_B, T) JAX array
        grad_accum_steps: number of micro-batches to accumulate (Python int)
        (others):         same as train_step

    Returns:
        new_state, avg_loss, avg_mean_sigma
    """
    dropout_rng, new_rng = random.split(state.rng)
    sigma_scale = state.sigma_anneal_fn(state.step)

    if sigma_anneal_steps > 0 and precision_loss_weight > 0.0:
        ramp  = jnp.minimum(jnp.float32(1.0),
                             (state.step + 1) / jnp.float32(sigma_anneal_steps))
        eff_pw = jnp.float32(precision_loss_weight) * ramp
    else:
        eff_pw = jnp.float32(0.0)

    # ── Enqueue N micro-batch forward+backward calls asynchronously ────────
    acc_grads        = None
    total_loss       = jnp.float32(0.0)
    total_sigma      = jnp.float32(0.0)
    all_indices_list = []

    for i in range(grad_accum_steps):
        micro_batch = micro_batches[i]              # slice: (micro_B, T)
        step_rng    = random.fold_in(dropout_rng, i)  # unique dropout mask

        grads, loss, mean_sigma, indices = _compute_micro_grads(
            state, micro_batch, step_rng, sigma_scale, eff_pw,
            pad_token_id=pad_token_id,
            use_bf16=use_bf16,
            loss_chunk_size=loss_chunk_size,
        )

        # All values are on-device JAX futures; additions are also async
        acc_grads = (
            grads if acc_grads is None
            else jax.tree_util.tree_map(jnp.add, acc_grads, grads)
        )
        total_loss  = total_loss  + loss
        total_sigma = total_sigma + mean_sigma
        all_indices_list.append(indices)

    # Concatenate pool indices across all micro-batches → (N*B*H, max_loops)
    all_indices = jnp.concatenate(all_indices_list, axis=0)

    # ── One JIT-compiled optimizer update ─────────────────────────────────
    new_state = _finalize_grad_accum(
        state, acc_grads, all_indices, new_rng,
        grad_accum_steps=grad_accum_steps,
    )

    scale     = jnp.float32(1.0 / grad_accum_steps)
    avg_loss  = total_loss  * scale
    avg_sigma = total_sigma * scale

    return new_state, avg_loss, avg_sigma
