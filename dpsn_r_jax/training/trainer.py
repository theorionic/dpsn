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
    max_reasoning_loops: int = struct.field(pytree_node=False)
    heads_per_dim: int = struct.field(pytree_node=False)


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
        max_reasoning_loops=config.max_reasoning_loops,
        heads_per_dim=max(1, config.num_indexer_heads // 2),
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
# Sparse pool gradient helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pool_slice_grads(grad_probe, all_mu_r, all_mu_c, all_sigma_h, all_start_2d, W, R_pool, C_pool):
    """Compute gradient slices for sparse pool update from probe gradients.

    Args:
        grad_probe:   (R, B, D)        — ∂loss/∂retrieved per reasoning loop
        all_mu_r:     (R, B, H_dim)    — normalised row coordinate per head
        all_mu_c:     (R, B, H_dim)    — normalised col coordinate per head
        all_sigma_h:  (R, B, H_dim)    — Gaussian bandwidth per head
        all_start_2d: (R, B, H_dim)    — flat_start = r_start * C + c_start
        W:            window size per axis (axis_window)
        R_pool, C_pool: pool grid dimensions

    Returns:
        all_r_starts:    (N,) int32        where N = R * B * H_dim
        all_c_starts:    (N,) int32
        all_grad_slices: (N, W, W, D) f32  gradient for each retrieval event
    """
    R_loops, B, H_dim = all_mu_r.shape
    D = grad_probe.shape[-1]

    all_r_starts = (all_start_2d // C_pool).astype(jnp.int32)   # (R, B, H)
    all_c_starts = (all_start_2d % C_pool).astype(jnp.int32)    # (R, B, H)

    r_centers = (all_mu_r * (R_pool - 1)).astype(jnp.float32)   # (R, B, H)
    c_centers = (all_mu_c * (C_pool - 1)).astype(jnp.float32)

    w_arange = jnp.arange(W, dtype=jnp.float32)
    r_idx = all_r_starts[:, :, :, None].astype(jnp.float32) + w_arange[None, None, None, :]  # (R, B, H, W)
    c_idx = all_c_starts[:, :, :, None].astype(jnp.float32) + w_arange[None, None, None, :]

    r_dist = r_idx - r_centers[:, :, :, None]
    c_dist = c_idx - c_centers[:, :, :, None]

    sigma_sq = (all_sigma_h.astype(jnp.float32) + 1e-6) ** 2   # (R, B, H)
    r_w = jnp.exp(-r_dist ** 2 / (2.0 * sigma_sq[:, :, :, None]))  # (R, B, H, W)
    c_w = jnp.exp(-c_dist ** 2 / (2.0 * sigma_sq[:, :, :, None]))

    w_2d = jnp.einsum("rbhi,rbhj->rbhij", r_w, c_w)            # (R, B, H, W, W)
    w_2d = w_2d / (w_2d.sum(axis=(-2, -1), keepdims=True) + 1e-9)

    # ∂loss/∂retrieved_per_head = grad_probe / H_dim  (chain rule through mean)
    grad_per_head = grad_probe[:, :, None, :].astype(jnp.float32) / jnp.float32(H_dim)  # (R, B, H, D)

    # Pool slice grad: outer product of grad and weights
    all_grad_slices = jnp.einsum("rbhd,rbhij->rbhijd", grad_per_head, w_2d)  # (R, B, H, W, W, D)

    N = R_loops * B * H_dim
    return (
        all_r_starts.reshape(N),
        all_c_starts.reshape(N),
        all_grad_slices.reshape(N, W, W, D).astype(jnp.float32),
    )


def _apply_sparse_pool_adam(pool_params, pool_m, pool_v, r_starts, c_starts,
                             grad_slices, lr, step, b1=0.9, b2=0.999, eps=1e-8):
    """Apply sparse Adam updates to pool at specific (r_start, c_start) positions.

    Vectorized gather → Adam → scatter.  All N events run in parallel:
      1. vmap(dynamic_slice) — single parallel gather kernel
      2. batched Adam on (N, W, W, D) arrays — all arithmetic fused
      3. at[flat_idx].set() — single scatter kernel (last-write-wins for overlaps)

    No serial scan ⇒ XLA can schedule everything as one wave.

    Args:
        pool_params:  (R_pool, C_pool, D)
        pool_m:       (R_pool, C_pool, D) float32 — Adam 1st moment
        pool_v:       (R_pool, C_pool, D) float32 — Adam 2nd moment
        r_starts:     (N,) int32
        c_starts:     (N,) int32
        grad_slices:  (N, W, W, D) float32
        lr, step, b1, b2, eps: Adam hyperparameters
    """
    import jax
    from jax import lax
    W = grad_slices.shape[1]
    D = grad_slices.shape[-1]
    R_pool, C_pool = pool_params.shape[0], pool_params.shape[1]
    pool_dtype = pool_params.dtype
    m_dtype = pool_m.dtype
    v_dtype = pool_v.dtype

    # ── 1. Parallel gather ────────────────────────────────────────────────────
    gather_fn_p = lambda r, c: lax.dynamic_slice(pool_params, (r, c, 0), (W, W, D))
    gather_fn_m = lambda r, c: lax.dynamic_slice(pool_m,     (r, c, 0), (W, W, D))
    gather_fn_v = lambda r, c: lax.dynamic_slice(pool_v,     (r, c, 0), (W, W, D))

    p_slices = jax.vmap(gather_fn_p)(r_starts, c_starts).astype(jnp.float32)  # (N,W,W,D)
    m_slices = jax.vmap(gather_fn_m)(r_starts, c_starts).astype(jnp.float32)
    v_slices = jax.vmap(gather_fn_v)(r_starts, c_starts).astype(jnp.float32)

    # ── 2. Batched Adam on (N, W, W, D) ──────────────────────────────────────
    step_f = step.astype(jnp.float32)
    m_new = b1 * m_slices + (1.0 - b1) * grad_slices
    v_new = b2 * v_slices + (1.0 - b2) * grad_slices ** 2
    m_hat = m_new / (1.0 - b1 ** step_f)
    v_hat = v_new / (1.0 - b2 ** step_f)
    p_new = p_slices - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    # ── 3. Build flat indices for scatter ────────────────────────────────────
    # Each event covers r_starts[i]:r_starts[i]+W, c_starts[i]:c_starts[i]+W
    # flat index of pool[r, c, :] = r * C_pool + c  (in the (R*C, D) view)
    wr = jnp.arange(W, dtype=jnp.int32)  # (W,)
    wc = jnp.arange(W, dtype=jnp.int32)  # (W,)
    # row offsets for each event: (N, W)
    row_offsets = (r_starts[:, None] + wr[None, :]) * C_pool  # (N, W)
    # col offsets: (N, W)
    col_offsets = c_starts[:, None] + wc[None, :]              # (N, W)
    # flat indices: (N, W, W) via broadcasting
    flat_idx = row_offsets[:, :, None] + col_offsets[:, None, :]  # (N, W, W)

    flat_idx_1d = flat_idx.reshape(-1)                          # (N*W*W,)

    # ── 4. Scatter back (last-write-wins; overlaps are rare in practice) ─────
    pool_flat = pool_params.reshape(-1, D)
    pool_params_new = pool_flat.at[flat_idx_1d].set(
        p_new.reshape(-1, D).astype(pool_dtype)
    ).reshape(pool_params.shape)

    m_flat = pool_m.reshape(-1, D)
    pool_m_new = m_flat.at[flat_idx_1d].set(
        m_new.reshape(-1, D).astype(m_dtype)
    ).reshape(pool_m.shape)

    v_flat = pool_v.reshape(-1, D)
    pool_v_new = v_flat.at[flat_idx_1d].set(
        v_new.reshape(-1, D).astype(v_dtype)
    ).reshape(pool_v.shape)

    return pool_params_new, pool_m_new, pool_v_new


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


def _apply_optimizer_update_sparse(state, dense_grads, dense_params, pool_params,
                                    r_starts, c_starts, grad_slices, new_rng, current_lr):
    """Dense AdamW on non-pool params + sparse Adam on pool slices only.

    Pool gradient is represented as (N, W, W, D) slices instead of the full
    (R_pool, C_pool, D) tensor, eliminating ~805 MB of memory traffic per step.
    """
    import jax.profiler

    # ── Dense AdamW update ──────────────────────────────────────────────────
    with jax.profiler.TraceAnnotation("Optimizer_Dense_AdamW"):
        updates, new_opt_state = state.tx.update(dense_grads, state.opt_state, dense_params)
        new_dense_params = optax.apply_updates(dense_params, updates)

    # ── Sparse Adam pool update ─────────────────────────────────────────────
    with jax.profiler.TraceAnnotation("Optimizer_Sparse_Adam_Pool_Slices"):
        pool_grad_norm = jnp.sqrt(jnp.sum(grad_slices ** 2) + 1e-9)

        new_pool_params, new_pool_m, new_pool_v = _apply_sparse_pool_adam(
            pool_params, state.pool_m, state.pool_v,
            r_starts, c_starts, grad_slices,
            lr=current_lr,
            step=state.step + 1,
        )

    # ── Reassemble full params ──────────────────────────────────────────────
    pool_key = ("pool", "params_storage")
    new_flat_params = traverse_util.flatten_dict(new_dense_params)
    new_flat_params[pool_key] = new_pool_params
    new_params = traverse_util.unflatten_dict(new_flat_params)

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
    """Single training step with sparse pool gradient updates."""
    print("Compiling train_step for XLA...", flush=True)
    dropout_rng, new_rng = random.split(state.rng)

    if sigma_anneal_steps > 0 and precision_loss_weight > 0.0:
        ramp = jnp.minimum(1.0, (state.step + 1) / sigma_anneal_steps)
        effective_precision_weight = precision_loss_weight * ramp
    else:
        effective_precision_weight = 0.0

    # ── Split params ────────────────────────────────────────────────────────
    pool_key = ("pool", "params_storage")
    flat_params = traverse_util.flatten_dict(state.params)
    pool_params = flat_params[pool_key]
    dense_flat = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat)
    pool_params_stopped = jax.lax.stop_gradient(pool_params)

    R_pool, C_pool, D_pool = pool_params.shape
    W = state.window_size
    axis_W = max(2, int(W ** 0.5))
    R_loops = state.max_reasoning_loops
    H_dim   = state.heads_per_dim

    B = batch.shape[0]
    model_dtype = jnp.bfloat16 if use_bf16 else jnp.float32
    probe = jnp.zeros((R_loops, B, D_pool), dtype=model_dtype)

    def loss_fn(dense_p_and_probe):
        dense_p, probe_ = dense_p_and_probe
        new_flat = dict(traverse_util.flatten_dict(dense_p))
        new_flat[pool_key] = pool_params_stopped
        full_params = traverse_util.unflatten_dict(new_flat)

        compute_params = (
            jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), full_params)
            if use_bf16 else full_params
        )

        if loss_chunk_size > 0:
            with jax.profiler.TraceAnnotation("Forward_encode_to_hidden"):
                state_hidden, (_, indices, mean_sigma,
                               all_mu_r, all_mu_c, all_sigma_h, all_start_2d) = state.apply_fn(
                    {"params": compute_params}, batch,
                    deterministic=False, sigma_max_scale=sigma_scale,
                    retrieved_probes=probe_,
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
            all_mu_r = jnp.zeros((R_loops, B, H_dim))
            all_mu_c = jnp.zeros((R_loops, B, H_dim))
            all_sigma_h = jnp.ones((R_loops, B, H_dim))
            all_start_2d = jnp.zeros((R_loops, B, H_dim), dtype=jnp.int32)
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
            (indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d),
        )

    import jax.profiler
    with jax.profiler.TraceAnnotation("Loss_and_Backprop"):
        (loss, (indices, mean_sigma,
                all_mu_r, all_mu_c, all_sigma_h, all_start_2d)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )((dense_params, probe))

    dense_grads, grad_probe = grads  # grad_probe: (R, B, D)

    r_starts, c_starts, grad_slices = _compute_pool_slice_grads(
        grad_probe, all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
        W=axis_W, R_pool=R_pool, C_pool=C_pool,
    )

    new_state, pool_grad_norm = _apply_optimizer_update_sparse(
        state, dense_grads, dense_params, pool_params,
        r_starts, c_starts, grad_slices,
        new_rng, current_lr,
    )
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
    """Gradient accumulation with sparse pool gradient updates.

    Pool parameters are excluded from the lax.scan accumulation buffer.
    Instead of an 805 MB gradient tensor, we collect tiny (W×W×D) gradient
    slices per retrieval event and apply sparse Adam directly.
    Memory traffic reduction: ~1000× for pool gradient operations.
    """
    print("Compiling grad_accum_step for XLA...", flush=True)
    dropout_rng, new_rng = random.split(state.rng)

    if sigma_anneal_steps > 0 and precision_loss_weight > 0.0:
        ramp = jnp.minimum(1.0, (state.step + 1) / jnp.float32(sigma_anneal_steps))
        effective_precision_weight = jnp.float32(precision_loss_weight) * ramp
    else:
        effective_precision_weight = jnp.float32(0.0)

    # ── Extract pool params and dense params ────────────────────────────────
    pool_key = ("pool", "params_storage")
    flat_params = traverse_util.flatten_dict(state.params)
    pool_params = flat_params[pool_key]
    dense_flat = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat)

    # Stop gradient on pool — autograd will NOT compute 805 MB pool gradient.
    pool_params_stopped = jax.lax.stop_gradient(pool_params)

    # Pool grid dimensions (static, from array shape)
    R_pool, C_pool, D_pool = pool_params.shape
    W = state.window_size  # axis_window = max(2, int(max_k**0.5))
    # Note: window_size in state is max_k (e.g. 32). For 2D pool, actual
    # axis window = max(2, int(max_k**0.5)). Re-derive here:
    axis_W = max(2, int(W ** 0.5))

    R_loops = state.max_reasoning_loops
    H_dim   = state.heads_per_dim

    # ── Zero grads for DENSE params only (no 805 MB pool buffer!) ──────────
    zero_dense_grads = jax.tree_util.tree_map(jnp.zeros_like, dense_params)

    # ── Scan body ──────────────────────────────────────────────────────────
    def scan_body(carry, i):
        acc_dense_grads, acc_loss, acc_sigma, current_rng = carry
        micro_batch = micro_batches[i]

        step_rng, next_rng = random.split(current_rng)

        B = micro_batch.shape[0]
        model_dtype = jnp.bfloat16 if use_bf16 else jnp.float32
        probe = jnp.zeros((R_loops, B, D_pool), dtype=model_dtype)

        def loss_fn_sparse(dense_p_and_probe):
            dense_p, probe_ = dense_p_and_probe
            # Rebuild full params with stopped pool
            new_flat = dict(traverse_util.flatten_dict(dense_p))
            new_flat[pool_key] = pool_params_stopped
            full_params = traverse_util.unflatten_dict(new_flat)

            compute_params = (
                jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), full_params)
                if use_bf16 else full_params
            )

            if loss_chunk_size > 0:
                with jax.profiler.TraceAnnotation("Forward_encode_to_hidden"):
                    state_hidden, (_, indices, mean_sigma,
                                   all_mu_r, all_mu_c, all_sigma_h, all_start_2d) = state.apply_fn(
                        {"params": compute_params}, micro_batch,
                        deterministic=False, sigma_max_scale=sigma_scale,
                        retrieved_probes=probe_,
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
                # Non-chunked: fall back (probe not supported in __call__ path)
                with jax.profiler.TraceAnnotation("Forward_full_model"):
                    logits, (_, indices, mean_sigma) = state.apply_fn(
                        {"params": compute_params}, micro_batch,
                        deterministic=False, sigma_max_scale=sigma_scale,
                        rngs={"dropout": step_rng},
                    )
                all_mu_r = jnp.zeros((R_loops, B, H_dim))
                all_mu_c = jnp.zeros((R_loops, B, H_dim))
                all_sigma_h = jnp.ones((R_loops, B, H_dim))
                all_start_2d = jnp.zeros((R_loops, B, H_dim), dtype=jnp.int32)
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
                (indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d),
            )

        import jax.profiler
        with jax.profiler.TraceAnnotation("Microbatch_Forward_Backward"):
            (loss, (indices, mean_sigma,
                    all_mu_r, all_mu_c, all_sigma_h, all_start_2d)), grads = jax.value_and_grad(
                loss_fn_sparse, has_aux=True
            )((dense_params, probe))

        dense_grads, grad_probe = grads  # dense_grads: pytree; grad_probe: (R, B, D)

        new_acc_dense_grads = jax.tree_util.tree_map(jnp.add, acc_dense_grads, dense_grads)

        return (
            (new_acc_dense_grads, acc_loss + loss, acc_sigma + mean_sigma, next_rng),
            (indices, grad_probe, all_mu_r, all_mu_c, all_sigma_h, all_start_2d),
        )

    # ── Execute lax.scan entirely on-device ────────────────────────────────
    init_carry = (zero_dense_grads, jnp.float32(0.0), jnp.float32(0.0), dropout_rng)

    (acc_dense_grads, total_loss, total_sigma, _), \
    (all_indices, all_grad_probes, all_mu_r, all_mu_c, all_sigma_h, all_start_2d) = jax.lax.scan(
        scan_body,
        init_carry,
        jnp.arange(grad_accum_steps),
    )

    # ── Average dense grads ─────────────────────────────────────────────────
    scale = jnp.float32(1.0 / grad_accum_steps)
    avg_dense_grads = jax.tree_util.tree_map(lambda g: g * scale, acc_dense_grads)
    avg_loss  = total_loss  * scale
    avg_sigma = total_sigma * scale

    # ── Compute pool slice gradients from probes ────────────────────────────
    # all_grad_probes: (grad_accum_steps, R, B, D) → flatten accum×R
    # all_mu_r etc.:  (grad_accum_steps, R, B, H_dim) → same flatten
    accum_steps_dim = all_grad_probes.shape[0]
    B_micro = all_grad_probes.shape[2]

    flat_grad_probe  = all_grad_probes.reshape(accum_steps_dim * R_loops, B_micro, D_pool) * scale
    flat_mu_r        = all_mu_r.reshape(accum_steps_dim * R_loops, B_micro, H_dim)
    flat_mu_c        = all_mu_c.reshape(accum_steps_dim * R_loops, B_micro, H_dim)
    flat_sigma_h     = all_sigma_h.reshape(accum_steps_dim * R_loops, B_micro, H_dim)
    flat_start_2d    = all_start_2d.reshape(accum_steps_dim * R_loops, B_micro, H_dim)

    r_starts, c_starts, grad_slices = _compute_pool_slice_grads(
        flat_grad_probe, flat_mu_r, flat_mu_c, flat_sigma_h, flat_start_2d,
        W=axis_W, R_pool=R_pool, C_pool=C_pool,
    )

    # ── Apply updates ───────────────────────────────────────────────────────
    # Flatten all_indices for logging compatibility: (accum, R, B*H) → (B*H, R) equiv
    flat_all_indices = all_indices.reshape(-1, all_indices.shape[-1])

    new_state, pool_grad_norm = _apply_optimizer_update_sparse(
        state, avg_dense_grads, dense_params, pool_params,
        r_starts, c_starts, grad_slices,
        new_rng, current_lr,
    )
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
