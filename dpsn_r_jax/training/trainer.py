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
from dpsn_r_jax.kernels import sparse_adam_pallas


class TrainState(train_state.TrainState):
    rng: Any
    pool_m: jnp.ndarray
    pool_v: jnp.ndarray
    window_size: int = struct.field(pytree_node=False)


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

    pool_m = jnp.zeros_like(pool_params, dtype=jnp.float32)
    pool_v = jnp.zeros_like(pool_params, dtype=jnp.float32)

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
    """LM loss chunked along the *sequence* (T) dimension.

    Why T, not B:
      Full logits shape is (B, T, V).  For xxl with T=8192 and V=50257 even
      B=1 gives 1×8192×50257×4 = 1.6 GB.  Chunking along T reduces peak to
      B × chunk_size × V.  With chunk_size=128 that is ~51 MB/device — a 64×
      reduction vs. materialising the full tensor.

      The old code chunked along B: with loss_chunk_size=128 and B<128 it
      padded the batch UP to 128, making memory usage worse, not better.
    """
    B, T, D = hidden.shape

    # Shift once before chunking: predict token i+1 from position i.
    h   = hidden[:, :-1, :]   # (B, T-1, D)
    tgt = labels[:, 1:]       # (B, T-1)
    T1  = T - 1

    # Pad T1 to a multiple of chunk_size so lax.scan sees equal-sized slices.
    remainder = T1 % chunk_size
    if remainder != 0:
        pad_len = chunk_size - remainder
        h   = jnp.concatenate(
            [h,   jnp.zeros((B, pad_len, D),              dtype=h.dtype)],   axis=1
        )
        tgt = jnp.concatenate(
            [tgt, jnp.full((B, pad_len), pad_token_id,    dtype=tgt.dtype)], axis=1
        )
        T1_padded = T1 + pad_len
    else:
        T1_padded = T1

    n_chunks = T1_padded // chunk_size

    # Reshape for lax.scan — leading axis must be the scan axis (n_chunks).
    # h_chunks:   (n_chunks, B, chunk_size, D)
    # tgt_chunks: (n_chunks, B, chunk_size)
    h_chunks   = h.reshape(B, n_chunks, chunk_size, D).transpose(1, 0, 2, 3)
    tgt_chunks = tgt.reshape(B, n_chunks, chunk_size).transpose(1, 0, 2)

    def scan_body(carry, x):
        chunk_h, chunk_tgt = x                                 # (B, chunk_size, D/int)
        logits = decode_fn(chunk_h).astype(jnp.float32)       # (B, chunk_size, V)
        loss   = optax.softmax_cross_entropy_with_integer_labels(logits, chunk_tgt)
        mask   = (chunk_tgt != pad_token_id).astype(jnp.float32)
        return carry, (loss * mask, mask)

    _, (weighted_losses, masks) = jax.lax.scan(
        jax.checkpoint(scan_body),
        None,
        (h_chunks, tgt_chunks),
    )
    # weighted_losses / masks: (n_chunks, B, chunk_size)
    return weighted_losses.sum() / jnp.maximum(masks.sum(), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Shared optimizer update — called from train_step and _finalize_grad_accum.
# Python helper (not JIT by itself); gets traced inside the JIT context of
# its callers.
# ─────────────────────────────────────────────────────────────────────────────

def _apply_optimizer_update(state, grads, indices, new_rng, current_lr):
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

    import jax.profiler

    # ── Dense AdamW update ─────────────────────────────────────────────────
    with jax.profiler.TraceAnnotation("Optimizer_Dense_AdamW"):
        updates, new_opt_state = state.tx.update(dense_grads, state.opt_state, dense_params)
        new_dense_params = optax.apply_updates(dense_params, updates)

    # ── Sparse Adam pool update ────────────────────────────────────────────
    with jax.profiler.TraceAnnotation("Optimizer_Sparse_Adam_Pool"):
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

        pool_grad_norm  = jnp.sqrt(jnp.sum(pool_grads ** 2) + 1e-9)
        pool_grad_scale = jnp.minimum(jnp.float32(1.0), jnp.float32(1.0) / pool_grad_norm)

        new_p_s, new_m_s, new_v_s = sparse_adam_pallas(
            p_slice, g_slice, m_slice, v_slice,
            lr=current_lr,
            step=state.step + 1,
            grad_scale=pool_grad_scale,
        )

        # Cast back to the pool's native dtype (bfloat16 after Opt-1) before
        # scattering — prevents a FutureWarning (→ error in future JAX) from
        # an implicit float32 → bfloat16 narrowing inside lax.scatter.
        pool_dtype      = pool_flat.dtype
        new_pool_flat   = pool_flat.at[safe_indices].set(new_p_s.astype(pool_dtype))
        new_pool_m_flat = pool_m_flat.at[safe_indices].set(new_m_s.astype(pool_m_flat.dtype))
        new_pool_v_flat = pool_v_flat.at[safe_indices].set(new_v_s.astype(pool_v_flat.dtype))

        new_pool_params = new_pool_flat.reshape(pool_params.shape)
        new_pool_m      = new_pool_m_flat.reshape(state.pool_m.shape)
        new_pool_v      = new_pool_v_flat.reshape(state.pool_v.shape)

    new_flat_params           = traverse_util.flatten_dict(new_dense_params)
    new_flat_params[pool_key] = new_pool_params
    new_params                = traverse_util.unflatten_dict(new_flat_params)

    new_state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        pool_m=new_pool_m,
        pool_v=new_pool_v,
        rng=new_rng,
    )
    return new_state, pool_grad_norm


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
    state, batch, current_lr, sigma_scale, pad_token_id=0,
    precision_loss_weight=0.0, sigma_anneal_steps=0,
    use_bf16=False, loss_chunk_size=0,
):
    """Single training step. Returns (new_state, loss, mean_sigma)."""
    print("Compiling train_step for XLA...", flush=True)
    dropout_rng, new_rng = random.split(state.rng)

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
            with jax.profiler.TraceAnnotation("Forward_encode_to_hidden"):
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
            with jax.profiler.TraceAnnotation("Forward_chunked_lm_loss"):
                lm_loss = chunked_lm_loss(
                    state_hidden, batch, decode_fn, pad_token_id, loss_chunk_size
                ).astype(jnp.float32)
        else:
            with jax.profiler.TraceAnnotation("Forward_full_model"):
                logits, (_, indices, mean_sigma) = state.apply_fn(
                    {"params": compute_params}, batch,
                    deterministic=False, sigma_max_scale=sigma_scale,
                    rngs={"dropout": dropout_rng},
                )
            with jax.profiler.TraceAnnotation("Forward_lm_loss"):
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

    import jax.profiler
    with jax.profiler.TraceAnnotation("Loss_and_Backprop"):
        (loss, (indices, mean_sigma)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.params)

    new_state, pool_grad_norm = _apply_optimizer_update(state, grads, indices, new_rng, current_lr)
    return new_state, loss, mean_sigma, pool_grad_norm


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Accumulation — JAX Native lax.scan Design
#
# Previously, this was a Python 'for' loop dispatching N separate jax.jit calls
# and thousands of jnp.add operations per step. That caused a massive Host
# Dispatch overhead (~1.4s) while the TPU sat 99% idle.
# By using jax.lax.scan (with NO unroll), XLA perfectly compiles this down to
# a single TPU kernel that executes entirely on-device, preserving fast compile
# times and enabling 100% TPU utilization.
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(
    jax.jit,
    static_argnames=[
        "pad_token_id", "precision_loss_weight",
        "sigma_anneal_steps", "use_bf16", "loss_chunk_size", "grad_accum_steps"
    ],
    donate_argnums=(0,),
)
def grad_accum_step(
    state,
    micro_batches,
    current_lr,
    sigma_scale,
    pad_token_id=0,
    precision_loss_weight=0.0,
    sigma_anneal_steps=0,
    use_bf16=False,
    loss_chunk_size=0,
    grad_accum_steps=1,
):
    """Gradient accumulation via fully JIT-compiled jax.lax.scan."""
    print("Compiling grad_accum_step for XLA...", flush=True)
    dropout_rng, new_rng = random.split(state.rng)

    if sigma_anneal_steps > 0 and precision_loss_weight > 0.0:
        ramp = jnp.minimum(1.0, (state.step + 1) / jnp.float32(sigma_anneal_steps))
        effective_precision_weight = jnp.float32(precision_loss_weight) * ramp
    else:
        effective_precision_weight = jnp.float32(0.0)

    # 1. Define loss function operating on a single micro_batch
    def loss_fn(params, micro_batch, step_rng):
        compute_params = (
            jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
            if use_bf16 else params
        )
        if loss_chunk_size > 0:
            with jax.profiler.TraceAnnotation("Forward_encode_to_hidden"):
                state_hidden, (_, indices, mean_sigma) = state.apply_fn(
                    {"params": compute_params}, micro_batch,
                    deterministic=False, sigma_max_scale=sigma_scale,
                    rngs={"dropout": step_rng},
                    method=lambda mod, *a, **kw: mod.encode_to_hidden(*a, **kw),
                )
            def decode_fn(chunk_h):
                return state.apply_fn(
                    {"params": compute_params}, chunk_h,
                    method=lambda mod, h: mod.controller.decode(h),
                )
            with jax.profiler.TraceAnnotation("Forward_chunked_lm_loss"):
                lm_loss = chunked_lm_loss(
                    state_hidden, micro_batch, decode_fn, pad_token_id, loss_chunk_size
                ).astype(jnp.float32)
        else:
            with jax.profiler.TraceAnnotation("Forward_full_model"):
                logits, (_, indices, mean_sigma) = state.apply_fn(
                    {"params": compute_params}, micro_batch,
                    deterministic=False, sigma_max_scale=sigma_scale,
                    rngs={"dropout": step_rng},
                )
            with jax.profiler.TraceAnnotation("Forward_lm_loss"):
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

    # 2. Get value and grad function
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # 3. Initialize PyTree of zero gradients
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)

    # 4. Define scan body
    def scan_body(carry, i):
        acc_grads, acc_loss, acc_sigma, current_rng = carry
        micro_batch = micro_batches[i]
        
        # Split rng for this specific micro-batch forward pass
        step_rng, next_rng = random.split(current_rng)

        import jax.profiler
        with jax.profiler.TraceAnnotation("Microbatch_Forward_Backward"):
            (loss, (indices, mean_sigma)), grads = grad_fn(state.params, micro_batch, step_rng)
            
        new_acc_grads = jax.tree_util.tree_map(jnp.add, acc_grads, grads)
        
        return (new_acc_grads, acc_loss + loss, acc_sigma + mean_sigma, next_rng), indices

    # 5. Execute lax.scan entirely on-device
    init_carry = (zero_grads, jnp.float32(0.0), jnp.float32(0.0), dropout_rng)
    
    (acc_grads, total_loss, total_sigma, _), all_indices = jax.lax.scan(
        scan_body, 
        init_carry, 
        jnp.arange(grad_accum_steps)
    )

    # 6. Finalize gradients and update optimizer
    scale = jnp.float32(1.0 / grad_accum_steps)
    avg_grads = jax.tree_util.tree_map(lambda g: g * scale, acc_grads)
    avg_loss = total_loss * scale
    avg_sigma = total_sigma * scale

    # flatten indices from (grad_accum_steps, micro_B, max_loops) to (B, max_loops)
    all_indices = jnp.reshape(all_indices, (-1, all_indices.shape[-1]))

    new_state, pool_grad_norm = _apply_optimizer_update(state, avg_grads, all_indices, new_rng, current_lr)
    return new_state, avg_loss, avg_sigma, pool_grad_norm


# ─────────────────────────────────────────────────────────────────────────────
# forward_only_step — forward pass only, no grad, no optimizer update.
# Used by the host-side component timer in main.py to measure forward-pass
# wall time independently of backward + optimizer time.
# NOTE: no donate_argnums so state is preserved for the real training step.
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(
    jax.jit,
    static_argnames=["use_bf16", "loss_chunk_size"],
)
def forward_only_step(state, batch, sigma_scale, use_bf16=False, loss_chunk_size=0):
    """Forward pass only (no grad, no optimizer) — for component timing.

    Returns (output, aux) where output is either state_hidden (chunked path)
    or logits (full path).  Caller should call jax.block_until_ready(output)
    to force synchronization before measuring elapsed time.
    """
    dropout_rng, _ = random.split(state.rng)
    compute_params = (
        jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), state.params)
        if use_bf16 else state.params
    )
    if loss_chunk_size > 0:
        with jax.profiler.TraceAnnotation("Profile_Forward_encode_to_hidden"):
            state_hidden, aux = state.apply_fn(
                {"params": compute_params}, batch,
                deterministic=True, sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
                method=lambda mod, *a, **kw: mod.encode_to_hidden(*a, **kw),
            )
        return state_hidden, aux
    else:
        with jax.profiler.TraceAnnotation("Profile_Forward_full"):
            logits, aux = state.apply_fn(
                {"params": compute_params}, batch,
                deterministic=True, sigma_max_scale=sigma_scale,
                rngs={"dropout": dropout_rng},
            )
        return logits, aux
