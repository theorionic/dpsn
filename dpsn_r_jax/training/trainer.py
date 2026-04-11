import jax
import jax.numpy as jnp
import jax.profiler
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


def get_training_phase(global_step: int, config) -> int:
    """Return the current training phase (0–3) based on step count and config.

    Phase 0  (joint)        : all components train — default when steps are 0.
    Phase 1  (controller)   : steps [0, phase1_steps)
    Phase 2  (direct pool)  : steps [phase1_steps, phase1_steps+phase2_steps)
    Phase 3  (spatial pool) : steps >= phase1_steps+phase2_steps

    If phase1_steps == phase2_steps == 0, always returns 0 (joint training).
    If config.training_phase is set to a non-zero value, that value is returned
    as a manual override (ignoring step-based thresholds entirely).
    """
    if config.training_phase != 0:
        return config.training_phase  # manual pin via yaml

    if config.phase1_steps == 0 and config.phase2_steps == 0:
        return 0  # no schedule configured → joint training

    if global_step < config.phase1_steps:
        return 1
    if global_step < config.phase1_steps + config.phase2_steps:
        return 2
    return 3


# Component groups used by _phase_stop_gradient.
# Keys match the top-level parameter subtree names in the Flax params dict.
_CONTROLLER_KEYS = frozenset({
    "controller", "indexer", "retrieval_integrator", "acc",
    "pool_cross_attn", "prefetch_query_attn", "prefetch_query_proj",
})
_DIRECT_POOL_KEYS = frozenset({"direct_pool"})


def _phase_stop_gradient(dense_params: dict, phase: int) -> dict:
    """Apply stop_gradient to parameter subtrees that should not train.

    Called inside loss_fn so it runs at XLA trace time.  Because
    ``phase`` is a static Python int (captured in the closure from
    train_step's static_argnames), the if-branches are resolved at
    trace time — zero runtime overhead.

    Phase 0: no-op (all dense params train).
    Phase 1 (controller only): freeze direct_pool.
    Phase 2 (direct pool only): freeze everything except direct_pool.
    Phase 3 (spatial pool only): freeze controller group AND direct_pool;
             the spatial pool is handled separately via sparse Adam.
    """
    if phase == 0:
        return dense_params

    flat = traverse_util.flatten_dict(dense_params)

    if phase == 1:
        # Controller + indexer train; direct pool is frozen.
        flat = {k: (jax.lax.stop_gradient(v) if k[0] in _DIRECT_POOL_KEYS else v)
                for k, v in flat.items()}

    elif phase == 2:
        # Only direct_pool trains; everything else is frozen.
        flat = {k: (v if k[0] in _DIRECT_POOL_KEYS else jax.lax.stop_gradient(v))
                for k, v in flat.items()}

    elif phase == 3:
        # Spatial pool trains (via sparse Adam); dense params are frozen.
        frozen = _CONTROLLER_KEYS | _DIRECT_POOL_KEYS
        flat = {k: (jax.lax.stop_gradient(v) if k[0] in frozen else jax.lax.stop_gradient(v))
                for k, v in flat.items()}
        # All dense params frozen in phase 3 — spatial pool is the only learner.

    return traverse_util.unflatten_dict(flat)


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


def _routing_diversity_loss(
    all_mu_r: jnp.ndarray,
    all_mu_c: jnp.ndarray,
    weight: float,
    n_bins: int = 32,
) -> jnp.ndarray:
    """Soft-histogram entropy loss to discourage indexer collapse.

    Discretises the [0, 1] coordinate space into n_bins bins and computes
    a soft occupancy via Gaussian kernels.  The loss penalises the gap to
    maximum entropy (log n_bins), pushing the routing distribution toward
    uniform coverage of the pool grid.

    Design choices
    --------------
    * Bins cover [0.0, 1.0] (not [0.05, 0.95]) because coord_margin=0.0
      now — matching bin range to the actual mu range is critical for the
      gradient to have the right sign at the grid edges.
    * sigma_bin = 3.0 / n_bins (vs old 1.5 / n_bins): wider kernel ensures
      that a fully-collapsed distribution still generates large entropy
      gradients for ALL bins, giving a strong push away from corners even
      at the start of training when routing is maximally degenerate.
    * Both row and column axes are regularised independently and summed,
      so 2-D collapse (same row AND same col) is twice as penalised.

    Args:
        all_mu_r: (R, B, H) or any shape — normalised row coordinates in [0, 1].
        all_mu_c: (R, B, H) or any shape — normalised col coordinates in [0, 1].
        weight:   Scalar coefficient.  0 = disabled.
        n_bins:   Number of histogram bins (default 32).

    Returns:
        Scalar loss value (jnp.float32).
    """
    if weight == 0.0:
        return jnp.float32(0.0)

    mu_r = all_mu_r.reshape(-1).astype(jnp.float32)   # (N,)
    mu_c = all_mu_c.reshape(-1).astype(jnp.float32)   # (N,)

    # Bin centres span the FULL [0, 1] to match coord_margin = 0.
    bin_centers = jnp.linspace(jnp.float32(0.0), jnp.float32(1.0), n_bins)
    # Wider kernel: 3 / K instead of 1.5 / K.
    # At K=32, sigma_bin ≈ 0.094, so a coord at 0.0 still deposits ~10% of its
    # mass on the bin at 0.5 — giving a global gradient even during total collapse.
    sigma_bin = jnp.float32(3.0 / n_bins)

    def _axis_entropy(mu: jnp.ndarray) -> jnp.ndarray:
        # Soft assignment: (N, K)
        w = jnp.exp(
            jnp.float32(-0.5) * ((mu[:, None] - bin_centers[None, :]) / sigma_bin) ** 2
        )
        w = w / (w.sum(axis=-1, keepdims=True) + jnp.float32(1e-8))
        p = jnp.mean(w, axis=0) + jnp.float32(1e-8)  # (K,) soft histogram
        p = p / p.sum()
        return -jnp.sum(p * jnp.log(p))              # scalar entropy

    max_entropy = jnp.log(jnp.float32(n_bins))
    gap_r = max_entropy - _axis_entropy(mu_r)
    gap_c = max_entropy - _axis_entropy(mu_c)
    return jnp.float32(weight) * (gap_r + gap_c)


def _cross_loop_diversity_loss(
    all_mu_r: jnp.ndarray,
    all_mu_c: jnp.ndarray,
    weight: float,
) -> jnp.ndarray:
    """Penalise each batch item routing to the same pool coordinate in every
    reasoning loop iteration (cross-loop / per-item collapse).

    The existing _routing_diversity_loss measures entropy across the flattened
    (loops × batch × heads) distribution — it catches global collapse (all
    values pile at one coordinate) but misses the case where each batch item
    individually repeats the same coordinate across loops while different items
    happen to go to different places (giving adequate global entropy).

    This loss fills that gap: for each (batch item, head) pair it computes the
    variance of the routed coordinate across the R loop iterations.  If an item
    always goes to the same place, its per-item variance is 0 → high loss.

    Form:  weight * exp(-mean_per_item_variance / 0.01)
      At full collapse (var=0):   loss = weight          (maximum penalty)
      At var=0.01 (modest spread): loss ≈ 0.37 * weight
      At var >> 0.01:             loss → 0               (no penalty)
    The exponential form bounds the loss to [0, weight] — no gradient explosion.

    Args:
        all_mu_r: (R, B, H) — row coordinates across loops, batch items, heads.
        all_mu_c: (R, B, H) — col coordinates.
        weight:   Scalar coefficient.  0 = disabled.
                  Recommended: same as routing_diversity_weight (they are summed).

    Returns:
        Scalar loss value (jnp.float32).
    """
    if weight == 0.0:
        return jnp.float32(0.0)

    R = all_mu_r.shape[0]
    if R < 2:
        return jnp.float32(0.0)

    # Per-(batch item, head) variance across the R loop iterations.
    var_r = jnp.var(all_mu_r.astype(jnp.float32), axis=0)  # (B, H)
    var_c = jnp.var(all_mu_c.astype(jnp.float32), axis=0)  # (B, H)
    mean_var = (var_r.mean() + var_c.mean()) * jnp.float32(0.5)

    # Exponential penalty: strong near 0, fades as variance grows.
    return jnp.float32(weight) * jnp.exp(-mean_var / jnp.float32(0.01))


def _boundary_repulsion_loss(
    all_mu_r: jnp.ndarray,
    all_mu_c: jnp.ndarray,
    weight: float,
    margin: float = 0.15,
) -> jnp.ndarray:
    """Exponential repulsion from grid boundaries (mu near 0 or 1).

    Directly targets 4-corner collapse: when tanh saturates, mu reaches exactly
    0 or 1, making the four corners (0,0), (0,1), (1,0), (1,1) strong gradient
    attractors.  This loss pushes mu away from boundaries with a force that falls
    off exponentially beyond `margin` — leaving the interior unaffected.

    Loss per coordinate:
        exp(-mu / margin) + exp(-(1 - mu) / margin)
    At mu=0:      1.0 + exp(-1/margin)  ≈ 1.0   (full repulsion)
    At mu=margin: exp(-1) ≈ 0.37                 (half-strength)
    At mu=0.5:    ≈ 0                             (negligible)

    Args:
        all_mu_r:  (R, B, H) — row coordinates in [0, 1].
        all_mu_c:  (R, B, H) — col coordinates in [0, 1].
        weight:    Scalar coefficient. 0 = disabled.
        margin:    Distance from boundary at which repulsion is ~37% strength.
                   Default 0.05 = 5% of the pool edge (row ~51 on a 1024-grid).

    Returns:
        Scalar loss value (jnp.float32).
    """
    if weight == 0.0:
        return jnp.float32(0.0)

    mu = jnp.concatenate([
        all_mu_r.reshape(-1).astype(jnp.float32),
        all_mu_c.reshape(-1).astype(jnp.float32),
    ])
    m = jnp.float32(margin)
    repulsion = (
        jnp.exp(-mu / m) + jnp.exp(-(jnp.float32(1.0) - mu) / m)
    )
    return jnp.float32(weight) * repulsion.mean()


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

    # dots_with_no_batch_dims_saveable: keeps Dense matmul output in memory,
    # only recomputes softmax/CE — avoids re-running the expensive LM head matmul.
    _, (weighted_losses, masks) = jax.lax.scan(
        jax.checkpoint(
            scan_body,
            policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        ),
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

    # ── 2. Fused Adam via Pallas kernel (or JAX fallback on multi-chip) ───────
    from dpsn_r_jax.kernels import sparse_adam_pallas as _sap
    N_flat = p_slices.shape[0] * p_slices.shape[1] * p_slices.shape[2]  # N*W*W
    D = p_slices.shape[-1]
    pool_grad_norm = jnp.sqrt(jnp.sum(grad_slices ** 2) + 1e-9)
    grad_scale = jnp.minimum(jnp.float32(1.0), jnp.float32(1.0) / pool_grad_norm)
    p_new_flat, m_new_flat, v_new_flat = _sap(
        p_slices.reshape(N_flat, D),
        grad_slices.reshape(N_flat, D),
        m_slices.reshape(N_flat, D),
        v_slices.reshape(N_flat, D),
        lr=jnp.float32(lr) if not hasattr(lr, 'shape') else lr,
        step=step,
        grad_scale=grad_scale,
        b1=b1, b2=b2, eps=eps,
    )
    p_new = p_new_flat.reshape(p_slices.shape)
    m_new = m_new_flat.reshape(m_slices.shape)
    v_new = v_new_flat.reshape(v_slices.shape)

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
        safe_indices     = jnp.clip(flat_touched, 0, pool_size - 1)
        # argsort removed: TPU gather/scatter do not benefit from sorted indices
        # (HBM access is burst-coalesced by XLA, not index-order dependent),
        # and argsort is O(N log N) — pure overhead on every step.

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
        "prefetch_reasoning", "prefetch_size", "routing_diversity_weight",
        "training_phase",
    ],
    donate_argnums=(0,),
)
def train_step(
    state, batch, current_lr, sigma_scale, pad_token_id=0,
    precision_loss_weight=0.0, sigma_anneal_steps=0,
    use_bf16=False, loss_chunk_size=0,
    prefetch_reasoning=False, prefetch_size=0,
    seq_pack_ids=None, routing_diversity_weight=0.0,
    training_phase=0,
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

    # Prefetch path: one (B, K, D) probe for all loops (candidates fetched once).
    # Non-prefetch path: (R, B, D) per-loop probe.
    if prefetch_reasoning:
        K     = prefetch_size * prefetch_size
        probe = jnp.zeros((B, K, D_pool), dtype=model_dtype)
    else:
        probe = jnp.zeros((R_loops, B, D_pool), dtype=model_dtype)

    def loss_fn(dense_p_and_probe):
        dense_p, probe_ = dense_p_and_probe
        # Phase-aware freeze: stop_gradient on components that shouldn't train.
        # training_phase is a static int (resolved at XLA trace time, zero overhead).
        dense_p = _phase_stop_gradient(dense_p, training_phase)
        new_flat = dict(traverse_util.flatten_dict(dense_p))
        new_flat[pool_key] = pool_params_stopped
        full_params = traverse_util.unflatten_dict(new_flat)

        compute_params = (
            jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), full_params)
            if use_bf16 else full_params
        )

        if loss_chunk_size > 0:
            with jax.profiler.TraceAnnotation("Forward_encode_to_hidden"):
                encode_kwargs = dict(
                    deterministic=False,
                    sigma_max_scale=sigma_scale,
                    rngs={"dropout": dropout_rng},
                )
                if prefetch_reasoning:
                    encode_kwargs["candidates_probe"] = probe_
                else:
                    encode_kwargs["retrieved_probes"] = probe_
                if seq_pack_ids is not None:
                    encode_kwargs["seq_pack_ids"] = seq_pack_ids

                state_hidden, (_, indices, mean_sigma,
                               all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
                               pf_r_start, pf_c_start) = state.apply_fn(
                    {"params": compute_params}, batch,
                    **encode_kwargs,
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
                _fwd_kwargs = dict(
                    deterministic=False, sigma_max_scale=sigma_scale,
                    rngs={"dropout": dropout_rng},
                )
                if seq_pack_ids is not None:
                    _fwd_kwargs["seq_pack_ids"] = seq_pack_ids
                logits, (_, indices, mean_sigma, _hidden) = state.apply_fn(
                    {"params": compute_params}, batch,
                    **_fwd_kwargs,
                )
            all_mu_r     = jnp.zeros((R_loops, B, H_dim))
            all_mu_c     = jnp.zeros((R_loops, B, H_dim))
            all_sigma_h  = jnp.ones((R_loops, B, H_dim))
            all_start_2d = jnp.zeros((R_loops, B, H_dim), dtype=jnp.int32)
            pf_r_start   = jnp.zeros((B,), dtype=jnp.int32)
            pf_c_start   = jnp.zeros((B,), dtype=jnp.int32)
            with jax.profiler.TraceAnnotation("Forward_lm_loss"):
                logits = logits.astype(jnp.float32)
                shift_logits = logits[:, :-1, :]
                shift_labels = batch[:, 1:]
                lm_loss = optax.softmax_cross_entropy_with_integer_labels(
                    shift_logits, shift_labels
                )
                mask = (shift_labels != pad_token_id).astype(jnp.float32)
                lm_loss = (lm_loss * mask).sum() / (mask.sum() + 1e-9)

        # Routing diversity: soft histogram entropy for uniform coverage.
        diversity_loss = _routing_diversity_loss(
            all_mu_r, all_mu_c, routing_diversity_weight
        )
        # Cross-loop diversity: penalise each batch item repeating the same
        # coordinate across reasoning loop iterations (the collapse mode where
        # global entropy looks fine but every item picks the same place each loop).
        cross_loop_loss = _cross_loop_diversity_loss(
            all_mu_r, all_mu_c, routing_diversity_weight
        )
        # Boundary repulsion: directly penalise mu near 0 or 1 to prevent
        # 4-corner collapse (tanh saturation → indexer locks to grid corners).
        # Weight 2.0x routing_diversity_weight + wider margin (0.15) gives a
        # much stronger push away from corners to break the positive-feedback loop.
        boundary_loss = _boundary_repulsion_loss(
            all_mu_r, all_mu_c, routing_diversity_weight * 2.0
        )
        return (
            lm_loss + effective_precision_weight * jnp.float32(mean_sigma)
            + diversity_loss + cross_loop_loss + boundary_loss,
            (indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
             pf_r_start, pf_c_start),
        )

    with jax.profiler.TraceAnnotation("Loss_and_Backprop"):
        (loss, (indices, mean_sigma,
                all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
                pf_r_start, pf_c_start)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )((dense_params, probe))

    dense_grads, grad_probe = grads

    # ── Diagnostic: log exact grad_probe magnitude (inside JIT via debug.print) ─
#    jax.debug.print(
#       "[POOL_GRAD_DIAG] grad_probe max={mx} l2={l2}",
#        mx=jnp.max(jnp.abs(grad_probe.astype(jnp.float32))),
#        l2=jnp.sqrt(jnp.sum(grad_probe.astype(jnp.float32) ** 2)),
#    )

    # ── Pool gradient computation ────────────────────────────────────────────
    if prefetch_reasoning:
        # grad_probe: (B, K, D) = ∂loss/∂candidates; reshape to (B, PS, PS, D)
        PS          = prefetch_size
        grad_slices = grad_probe.reshape(B, PS, PS, D_pool).astype(jnp.float32)
        r_starts    = pf_r_start   # (B,) — one patch per batch element
        c_starts    = pf_c_start   # (B,)
    else:
        r_starts, c_starts, grad_slices = _compute_pool_slice_grads(
            grad_probe, all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
            W=axis_W, R_pool=R_pool, C_pool=C_pool,
        )

    # Phase 1 and 2: spatial pool should NOT be updated.
    # Zero out grad_slices so sparse Adam writes nothing (training_phase is
    # a static int so this branch is resolved at XLA trace time).
    if training_phase in (1, 2):
        grad_slices = jnp.zeros_like(grad_slices)

    new_state, pool_grad_norm = _apply_optimizer_update_sparse(
        state, dense_grads, dense_params, pool_params,
        r_starts, c_starts, grad_slices,
        new_rng, current_lr,
    )
    return new_state, loss, mean_sigma, pool_grad_norm, all_mu_r, all_mu_c


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
        "sigma_anneal_steps", "use_bf16", "loss_chunk_size", "grad_accum_steps",
        "prefetch_reasoning", "prefetch_size", "routing_diversity_weight",
        "training_phase",
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
    prefetch_reasoning=False,
    prefetch_size=0,
    seq_pack_ids=None,
    routing_diversity_weight=0.0,
    training_phase=0,
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
        if prefetch_reasoning:
            K     = prefetch_size * prefetch_size
            probe = jnp.zeros((B, K, D_pool), dtype=model_dtype)
        else:
            probe = jnp.zeros((R_loops, B, D_pool), dtype=model_dtype)

        def loss_fn_sparse(dense_p_and_probe):
            dense_p, probe_ = dense_p_and_probe
            # Phase-aware freeze (static int → resolved at trace time)
            dense_p = _phase_stop_gradient(dense_p, training_phase)
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
                    encode_kwargs = dict(
                        deterministic=False,
                        sigma_max_scale=sigma_scale,
                        rngs={"dropout": step_rng},
                    )
                    if prefetch_reasoning:
                        encode_kwargs["candidates_probe"] = probe_
                    else:
                        encode_kwargs["retrieved_probes"] = probe_
                    if seq_pack_ids is not None:
                        encode_kwargs["seq_pack_ids"] = seq_pack_ids

                    state_hidden, (_, indices, mean_sigma,
                                   all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
                                   pf_r_start, pf_c_start) = state.apply_fn(
                        {"params": compute_params}, micro_batch,
                        **encode_kwargs,
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
                    _fwd_kwargs = dict(
                        deterministic=False, sigma_max_scale=sigma_scale,
                        rngs={"dropout": step_rng},
                    )
                    if seq_pack_ids is not None:
                        _fwd_kwargs["seq_pack_ids"] = seq_pack_ids
                    logits, (_, indices, mean_sigma, _hidden) = state.apply_fn(
                        {"params": compute_params}, micro_batch,
                        **_fwd_kwargs,
                    )
                all_mu_r     = jnp.zeros((R_loops, B, H_dim))
                all_mu_c     = jnp.zeros((R_loops, B, H_dim))
                all_sigma_h  = jnp.ones((R_loops, B, H_dim))
                all_start_2d = jnp.zeros((R_loops, B, H_dim), dtype=jnp.int32)
                pf_r_start   = jnp.zeros((B,), dtype=jnp.int32)
                pf_c_start   = jnp.zeros((B,), dtype=jnp.int32)
                with jax.profiler.TraceAnnotation("Forward_lm_loss"):
                    logits = logits.astype(jnp.float32)
                    shift_logits = logits[:, :-1, :]
                    shift_labels = micro_batch[:, 1:]
                    per_token = optax.softmax_cross_entropy_with_integer_labels(
                        shift_logits, shift_labels
                    )
                    mask = (shift_labels != pad_token_id).astype(jnp.float32)
                    lm_loss = (per_token * mask).sum() / (mask.sum() + 1e-9)

            # Routing diversity: soft histogram entropy (same as train_step).
            diversity_loss = _routing_diversity_loss(
                all_mu_r, all_mu_c, routing_diversity_weight
            )
            # Cross-loop collapse: penalise each batch item repeating the same
            # coordinate across reasoning loop iterations.
            cross_loop_loss = _cross_loop_diversity_loss(
                all_mu_r, all_mu_c, routing_diversity_weight
            )
            # Boundary repulsion: prevent 4-corner collapse (tanh saturation).
            # Weight 2.0x + wider margin (0.15) for stronger anti-collapse force.
            boundary_loss = _boundary_repulsion_loss(
                all_mu_r, all_mu_c, routing_diversity_weight * 2.0
            )
            return (
                lm_loss + effective_precision_weight * jnp.float32(mean_sigma)
                + diversity_loss + cross_loop_loss + boundary_loss,
                (indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
                 pf_r_start, pf_c_start),
            )

        with jax.profiler.TraceAnnotation("Microbatch_Forward_Backward"):
            (loss, (indices, mean_sigma,
                    all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
                    pf_r_start, pf_c_start)), grads = jax.value_and_grad(
                loss_fn_sparse, has_aux=True
            )((dense_params, probe))

        dense_grads, grad_probe = grads

        new_acc_dense_grads = jax.tree_util.tree_map(jnp.add, acc_dense_grads, dense_grads)

        return (
            (new_acc_dense_grads, acc_loss + loss, acc_sigma + mean_sigma, next_rng),
            (indices, grad_probe, all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
             pf_r_start, pf_c_start),
        )

    # ── Execute lax.scan entirely on-device ────────────────────────────────
    init_carry = (zero_dense_grads, jnp.float32(0.0), jnp.float32(0.0), dropout_rng)

    (acc_dense_grads, total_loss, total_sigma, _), \
    (all_indices, all_grad_probes, all_mu_r, all_mu_c, all_sigma_h, all_start_2d,
     all_pf_r, all_pf_c) = jax.lax.scan(
        scan_body,
        init_carry,
        jnp.arange(grad_accum_steps),
    )

    # ── Average dense grads ─────────────────────────────────────────────────
    scale = jnp.float32(1.0 / grad_accum_steps)
    avg_dense_grads = jax.tree_util.tree_map(lambda g: g * scale, acc_dense_grads)
    avg_loss  = total_loss  * scale
    avg_sigma = total_sigma * scale

    # ── Compute pool slice gradients ─────────────────────────────────────────
    if prefetch_reasoning:
        # all_grad_probes: (accum, B, K, D) → treat each (accum, B) as one event
        # all_pf_r/c:      (accum, B) int32 — patch positions per micro-batch
        PS          = prefetch_size
        accum_B     = grad_accum_steps * all_grad_probes.shape[1]
        grad_slices = (all_grad_probes * scale).reshape(accum_B, PS, PS, D_pool).astype(jnp.float32)
        r_starts    = all_pf_r.reshape(accum_B)
        c_starts    = all_pf_c.reshape(accum_B)
    else:
        # all_grad_probes: (accum, R, B, D) → flatten accum×R
        accum_steps_dim = all_grad_probes.shape[0]
        B_micro         = all_grad_probes.shape[2]

        flat_grad_probe = all_grad_probes.reshape(accum_steps_dim * R_loops, B_micro, D_pool) * scale
        flat_mu_r       = all_mu_r.reshape(accum_steps_dim * R_loops, B_micro, H_dim)
        flat_mu_c       = all_mu_c.reshape(accum_steps_dim * R_loops, B_micro, H_dim)
        flat_sigma_h    = all_sigma_h.reshape(accum_steps_dim * R_loops, B_micro, H_dim)
        flat_start_2d   = all_start_2d.reshape(accum_steps_dim * R_loops, B_micro, H_dim)

        r_starts, c_starts, grad_slices = _compute_pool_slice_grads(
            flat_grad_probe, flat_mu_r, flat_mu_c, flat_sigma_h, flat_start_2d,
            W=axis_W, R_pool=R_pool, C_pool=C_pool,
        )

    # ── Apply updates ───────────────────────────────────────────────────────
    # Flatten all_indices for logging compatibility: (accum, R, B*H) → (B*H, R) equiv
    flat_all_indices = all_indices.reshape(-1, all_indices.shape[-1])

    # Phase 1 and 2: spatial pool should NOT be updated.
    if training_phase in (1, 2):
        grad_slices = jnp.zeros_like(grad_slices)

    new_state, pool_grad_norm = _apply_optimizer_update_sparse(
        state, avg_dense_grads, dense_params, pool_params,
        r_starts, c_starts, grad_slices,
        new_rng, current_lr,
    )
    return new_state, avg_loss, avg_sigma, pool_grad_norm, all_mu_r, all_mu_c


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


# ─────────────────────────────────────────────────────────────────────────────
# eval_step — forward pass + scalar loss, no grad, no optimizer update.
# Used for validation loss computation.
# ─────────────────────────────────────────────────────────────────────────────

@functools.partial(
    jax.jit,
    static_argnames=["pad_token_id", "use_bf16", "loss_chunk_size"],
)
def eval_step(state, batch, sigma_scale, pad_token_id=0, use_bf16=False, loss_chunk_size=0):
    """Forward pass with scalar loss — for validation. No gradients, no optimizer."""
    dropout_rng, _ = random.split(state.rng)
    compute_params = (
        jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), state.params)
        if use_bf16 else state.params
    )

    if loss_chunk_size > 0:
        state_hidden, _ = state.apply_fn(
            {"params": compute_params}, batch,
            deterministic=True, sigma_max_scale=sigma_scale,
            rngs={"dropout": dropout_rng},
            method=lambda mod, *a, **kw: mod.encode_to_hidden(*a, **kw),
        )
        def decode_fn(chunk_h):
            return state.apply_fn(
                {"params": compute_params}, chunk_h,
                method=lambda mod, h: mod.controller.decode(h),
            )
        loss = chunked_lm_loss(
            state_hidden, batch, decode_fn, pad_token_id, loss_chunk_size
        ).astype(jnp.float32)
    else:
        logits, _ = state.apply_fn(
            {"params": compute_params}, batch,
            deterministic=True, sigma_max_scale=sigma_scale,
            rngs={"dropout": dropout_rng},
        )
        logits = logits.astype(jnp.float32)
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        lm_loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)
        mask = (shift_labels != pad_token_id).astype(jnp.float32)
        loss = (lm_loss * mask).sum() / (mask.sum() + 1e-9)

    return loss
