"""
Pallas TPU kernels for DPSN-R — fused, memory-bandwidth-optimised operations.

Three kernels are provided, each targeting a different HBM bottleneck:

1. sparse_adam_pallas   (trainer.py)
   Replaces the 7-separate-kernel chain produced by JAX for the Adam element-wise
   ops on pre-gathered pool slices.  All of  m_new / v_new / m_hat / v_hat / p_new
   are computed in a single TPU pass — no intermediate (N, D) tensors ever touch HBM.

2. pool_retrieve_1d_pallas   (memory.py — CoordinateMassivePool)
   Fuses three previously separate ops into one kernel per batch sample:
     dynamic_slice (HBM gather)  →  Gaussian weight computation  →  weighted dot
   The (B, W, D) selected-vectors buffer and the (B, W) weights buffer are
   eliminated; only the final (B, D) aggregated output is written to HBM.

3. pool_retrieve_2d_pallas   (memory.py — CoordinateMassivePool2D)
   Same as (2) but for the 2-D grid pool.  Additionally eliminates the (B, W, W)
   Gaussian weight matrix from the outer-product step.

Fallback behaviour
──────────────────
Each function checks _is_tpu() at call time and falls back to a pure-JAX
implementation on GPU / CPU.  The fallbacks are numerically identical to the
existing code, so switching the flag (or running without a TPU) is safe.

Interpret mode
──────────────
Set  DPSN_PALLAS_INTERPRET=1  to force interpret=True on every pallas_call.
This runs the kernel logic as ordinary JAX on any backend — useful for unit
tests and debugging without physical TPU access.
"""

from __future__ import annotations

import functools
import os
from typing import Tuple

import jax
import jax.numpy as jnp

# ── Pallas availability ───────────────────────────────────────────────────────
try:
    import jax.experimental.pallas as pl
    _PALLAS_AVAILABLE = True
except ImportError:               # pragma: no cover
    _PALLAS_AVAILABLE = False


def _is_tpu() -> bool:
    try:
        return jax.default_backend() == "tpu"
    except Exception:
        return False


def _interpret_mode() -> bool:
    """Force interpret=True via env-var — useful for tests without a real TPU."""
    return os.environ.get("DPSN_PALLAS_INTERPRET", "0") == "1"


def _use_pallas() -> bool:
    """Return True only when Pallas kernels are safe to call.

    Mosaic TPU kernels cannot be auto-partitioned by XLA's GSPMD, so they fail
    inside jit with a multi-chip mesh unless explicitly wrapped in shard_map.
    We therefore restrict to single-chip TPU by default.

    To enable on multi-chip (after wrapping calls in shard_map yourself):
        export DPSN_PALLAS_FORCE=1
    """
    if _interpret_mode():
        return _PALLAS_AVAILABLE  # interpret mode always safe (pure Python)
    if not _PALLAS_AVAILABLE or not _is_tpu():
        return False
    # Multi-chip guard: disable unless the user has explicitly opted in.
    if jax.device_count() > 1 and os.environ.get("DPSN_PALLAS_FORCE", "0") != "1":
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1 — Fused Sparse Adam
# ─────────────────────────────────────────────────────────────────────────────
#
# The pure-JAX path in _apply_optimizer_update() (trainer.py) does:
#
#   m_new  = b1 * m + (1 - b1) * g         # 3 ops → 3 kernel dispatches
#   v_new  = b2 * v + (1 - b2) * (g * g)   # 4 ops → 4 kernel dispatches
#   m_hat  = m_new / (1 - b1 ** step)       # 3 ops
#   v_hat  = v_new / (1 - b2 ** step)       # 3 ops
#   p_new  = p - lr * m_hat / (√v + ε)      # 4 ops
#                                            ──── 17 HBM reads/writes for (N, D)
#
# This Pallas kernel collapses all 17 operations into ONE pass:
#   read p, g, m, v once  →  compute in registers  →  write p, m, v once
#                                            ──── 7 HBM accesses total
# ─────────────────────────────────────────────────────────────────────────────

def _adam_fused_body(
    # Per-tile refs — shape (block_n, D) each
    p_ref, g_ref, m_ref, v_ref,
    # Dynamic scalar refs — 1-element arrays (shape (1,))
    lr_ref, step_ref, grad_scale_ref,
    # Output refs (same shapes as p / m / v)
    p_out_ref, m_out_ref, v_out_ref,
    # Static hyperparameters — embedded via functools.partial (no recompile risk)
    *, b1: float, b2: float, eps: float,
):
    """One tile of the fused Adam update.  All intermediates stay in registers."""
    lr         = lr_ref[0].astype(jnp.float32)
    step       = step_ref[0].astype(jnp.float32)
    grad_scale = grad_scale_ref[0].astype(jnp.float32)

    # Cast to float32 for all arithmetic (p/m/v might be stored as bfloat16)
    p = p_ref[...].astype(jnp.float32)
    g = g_ref[...].astype(jnp.float32) * grad_scale
    m = m_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)

    m_new = b1 * m + (1.0 - b1) * g
    v_new = b2 * v + (1.0 - b2) * (g * g)
    m_hat = m_new / (1.0 - b1 ** step)
    v_hat = v_new / (1.0 - b2 ** step)
    p_new = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    # Cast back to storage dtype before writing
    p_out_ref[...] = p_new.astype(p_ref.dtype)
    m_out_ref[...] = m_new.astype(m_ref.dtype)
    v_out_ref[...] = v_new.astype(v_ref.dtype)


def sparse_adam_pallas(
    p_slice:    jnp.ndarray,          # (N, D) — pre-gathered pool params
    g_slice:    jnp.ndarray,          # (N, D) — pre-gathered pool gradients
    m_slice:    jnp.ndarray,          # (N, D) — first moment at touched indices
    v_slice:    jnp.ndarray,          # (N, D) — second moment at touched indices
    lr:         jnp.ndarray,          # scalar float32
    step:       jnp.ndarray,          # scalar int32 (1-indexed)
    grad_scale: jnp.ndarray,          # scalar float32 (clipping multiplier)
    b1:         float = 0.9,
    b2:         float = 0.999,
    eps:        float = 1e-8,
    block_n:    int   = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fused Adam update on pre-gathered pool slices.

    All seven Adam element-wise operations are executed in a single Pallas
    kernel; no intermediate (N, D) tensors are materialised in HBM.

    The caller remains responsible for the final scatter-back step:
        pool_flat.at[safe_indices].set(p_new)

    Args:
        p_slice:    Current parameter values at the N touched pool indices.
        g_slice:    Gradients at those indices.
        m_slice:    First moment (Adam m) at those indices.
        v_slice:    Second moment (Adam v) at those indices.
        lr:         Learning rate scalar (JAX array — traces through JIT).
        step:       Training step, 1-indexed (JAX array).
        grad_scale: Gradient norm clipping multiplier (JAX array).
        b1, b2:     Adam momentum decay factors (Python floats — static).
        eps:        Adam epsilon (Python float — static).
        block_n:    Tile size along the N dimension.  Must divide N evenly
                    after padding; default 64 works for most pool sizes.

    Returns:
        (p_new, m_new, v_new), each (N, D) in the same dtype as the inputs.
    """
    if not _use_pallas():
        return _sparse_adam_jax_fallback(
            p_slice, g_slice, m_slice, v_slice, lr, step, grad_scale, b1, b2, eps
        )

    N, D = p_slice.shape

    # Pad N to a multiple of block_n so the grid divides evenly.
    pad_n = (-N) % block_n
    if pad_n:
        _pad = lambda x: jnp.pad(x, ((0, pad_n), (0, 0)))
        p_slice = _pad(p_slice)
        g_slice = _pad(g_slice)
        m_slice = _pad(m_slice)
        v_slice = _pad(v_slice)
    N_padded = N + pad_n

    # Wrap static hyperparams so the kernel function has no free Python vars
    kernel = functools.partial(_adam_fused_body, b1=b1, b2=b2, eps=eps)
    tile   = (block_n, D)
    t_spec = lambda i: pl.BlockSpec(tile, lambda i: (i * block_n, 0))
    s_spec = pl.BlockSpec((1,), lambda i: (0,))   # scalar: same element for all tiles

    p_new, m_new, v_new = pl.pallas_call(
        kernel,
        out_shape=[
            jax.ShapeDtypeStruct(p_slice.shape, p_slice.dtype),
            jax.ShapeDtypeStruct(m_slice.shape, m_slice.dtype),
            jax.ShapeDtypeStruct(v_slice.shape, v_slice.dtype),
        ],
        grid_spec=pl.GridSpec(
            grid=(N_padded // block_n,),
            in_specs=[
                pl.BlockSpec(tile, lambda i: (i * block_n, 0)),   # p
                pl.BlockSpec(tile, lambda i: (i * block_n, 0)),   # g
                pl.BlockSpec(tile, lambda i: (i * block_n, 0)),   # m
                pl.BlockSpec(tile, lambda i: (i * block_n, 0)),   # v
                s_spec,                                            # lr
                s_spec,                                            # step
                s_spec,                                            # grad_scale
            ],
            out_specs=[
                pl.BlockSpec(tile, lambda i: (i * block_n, 0)),
                pl.BlockSpec(tile, lambda i: (i * block_n, 0)),
                pl.BlockSpec(tile, lambda i: (i * block_n, 0)),
            ],
        ),
        interpret=_interpret_mode(),
    )(
        p_slice, g_slice, m_slice, v_slice,
        lr.reshape(1).astype(jnp.float32),
        step.reshape(1).astype(jnp.float32),
        grad_scale.reshape(1).astype(jnp.float32),
    )

    # Strip padding rows added above
    if pad_n:
        p_new = p_new[:N]
        m_new = m_new[:N]
        v_new = v_new[:N]

    return p_new, m_new, v_new


def _sparse_adam_jax_fallback(p, g, m, v, lr, step, grad_scale, b1, b2, eps):
    """Pure-JAX Adam — numerically identical, used on non-TPU backends."""
    g_clipped = g * grad_scale
    m_new = b1 * m + (1.0 - b1) * g_clipped
    v_new = b2 * v + (1.0 - b2) * (g_clipped * g_clipped)
    m_hat = m_new / (1.0 - b1 ** step.astype(jnp.float32))
    v_hat = v_new / (1.0 - b2 ** step.astype(jnp.float32))
    p_new = p.astype(jnp.float32) - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return p_new.astype(p.dtype), m_new.astype(m.dtype), v_new.astype(v.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2 — Fused 1-D Pool Retrieve
# ─────────────────────────────────────────────────────────────────────────────
#
# The pure-JAX path in CoordinateMassivePool._retrieve() does:
#   selected = vmap(dynamic_slice)(pool, starts)   # → (B, W, D) in HBM
#   weights  = softmax(gaussian(distances, sigma)) # → (B, W)   in HBM
#   agg      = einsum("bw,bwd->bd", weights, selected)  # → (B, D)
#
# This kernel fuses all three steps for each batch sample:
#   1. Load (W, D) window from HBM into registers  (1 read, no intermediate)
#   2. Compute Gaussian weights in registers
#   3. Dot-product → (D,) output
#   4. Write (D,) to HBM output buffer
#
# The (B, W, D) selected tensor and (B, W) weights tensor are never written to HBM.
# ─────────────────────────────────────────────────────────────────────────────

def _pool1d_body(
    pool_ref,    # (Total, D) bf16  — full pool, BlockSpec=None (dynamic access)
    start_ref,   # (1,)       int32 — window start for this batch sample
    mu_ref,      # (1,)       float32
    sigma_ref,   # (1,)       float32
    out_ref,     # (1, D)     float32
    *, W: int, Total: int,
):
    """Kernel body: one batch sample.  Fuses gather + Gaussian + dot."""
    start = start_ref[0]
    mu_b  = mu_ref[0].astype(jnp.float32)
    sig_b = sigma_ref[0].astype(jnp.float32)
    D     = pool_ref.shape[1]

    # ── 1. Dynamic gather: load exactly W×D values from HBM ──────────────────
    window = pool_ref[pl.dslice(start, W), :]   # (W, D) bfloat16

    # ── 2. Gaussian weights in registers ─────────────────────────────────────
    center = mu_b * jnp.float32(Total - 1)
    pos    = jnp.arange(W, dtype=jnp.float32) + start.astype(jnp.float32)
    dist   = pos - center                        # (W,)
    s_sq   = (sig_b + jnp.float32(1e-6)) ** 2
    w      = jnp.exp(-dist * dist / (2.0 * s_sq)) + jnp.float32(1e-6)  # (W,)
    w      = w / jnp.sum(w)                      # normalise

    # ── 3. Weighted dot-product → (D,) ───────────────────────────────────────
    agg = jnp.dot(w, window.astype(jnp.float32))  # (D,) float32

    out_ref[0, :] = agg


def _pool1d_pallas_impl(pool, mu, sigma, start_indices, W, Total):
    """Actual Pallas kernel call (or JAX fallback). No autodiff wrapping."""
    if not _use_pallas():
        return _pool1d_jax_fallback(pool, mu, sigma, start_indices, W, Total)

    B, D = mu.shape[0], pool.shape[1]
    kernel = functools.partial(_pool1d_body, W=W, Total=Total)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, D), jnp.float32),
        grid_spec=pl.GridSpec(
            grid=(B,),
            in_specs=[
                pl.BlockSpec((Total, D), lambda i: (0, 0)),
                pl.BlockSpec((1,), lambda i: (i,)),
                pl.BlockSpec((1,), lambda i: (i,)),
                pl.BlockSpec((1,), lambda i: (i,)),
            ],
            out_specs=pl.BlockSpec((1, D), lambda i: (i, 0)),
        ),
        interpret=_interpret_mode(),
    )(pool, start_indices, mu, sigma)


# custom_vjp: Pallas runs in the forward pass; pure-JAX VJP in the backward.
# nondiff_argnums covers W and Total (Python ints — never traced by JAX).
@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def pool_retrieve_1d_pallas(
    pool:          jnp.ndarray,   # (Total, D) bfloat16 — full pool storage
    mu:            jnp.ndarray,   # (B,)       float32  — normalised coordinate
    sigma:         jnp.ndarray,   # (B,)       float32  — retrieval bandwidth
    start_indices: jnp.ndarray,   # (B,)       int32    — pre-clipped window starts
    W:             int,           # window size (static)
    Total:         int,           # pool size   (static)
) -> jnp.ndarray:                 # (B, D)     float32
    """Fused gather + Gaussian weighting + weighted sum for the 1-D pool.

    Replaces three separate operations in CoordinateMassivePool._retrieve():
      vmap(dynamic_slice) + gaussian weights + einsum

    The intermediate (B, W, D) and (B, W) tensors are never written to HBM.
    Supports reverse-mode autodiff via custom_vjp (backward uses pure JAX).
    """
    return _pool1d_pallas_impl(pool, mu, sigma, start_indices, W, Total)


def _pool1d_fwd(pool, mu, sigma, start_indices, W, Total):
    out = _pool1d_pallas_impl(pool, mu, sigma, start_indices, W, Total)
    return out, (pool, mu, sigma, start_indices)


def _pool1d_bwd(W, Total, res, g):
    pool, mu, sigma, start_indices = res
    # Re-run VJP through the pure-JAX equivalent — correct gradients, no Pallas needed.
    _, vjp_fn = jax.vjp(
        lambda p, m, s: _pool1d_jax_fallback(p, m, s, start_indices, W, Total),
        pool, mu, sigma,
    )
    d_pool, d_mu, d_sigma = vjp_fn(g)
    # start_indices is int32 — no gradient
    return d_pool, d_mu, d_sigma, jnp.zeros_like(start_indices)


pool_retrieve_1d_pallas.defvjp(_pool1d_fwd, _pool1d_bwd)


def _pool1d_jax_fallback(pool, mu, sigma, start_indices, W, Total):
    """Pure-JAX 1-D pool retrieve — identical semantics to the existing code."""
    from jax import lax

    def slice_fn(start):
        return lax.dynamic_slice(pool, (start, 0), (W, pool.shape[1]))

    selected = jax.vmap(slice_fn)(start_indices)                    # (B, W, D)
    center   = mu * jnp.float32(Total - 1)
    rel_idx  = jnp.arange(W)[None, :] + start_indices[:, None]
    dist     = rel_idx.astype(jnp.float32) - center[:, None]
    s_sq     = (sigma[:, None].astype(jnp.float32) + 1e-6) ** 2
    w        = jnp.exp(-dist * dist / (2.0 * s_sq)) + 1e-6
    w        = w / jnp.sum(w, axis=-1, keepdims=True)
    return jnp.einsum("bw,bwd->bd", w, selected.astype(jnp.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3 — Fused 2-D Pool Retrieve
# ─────────────────────────────────────────────────────────────────────────────
#
# The pure-JAX path in CoordinateMassivePool2D._weight_and_aggregate() does:
#   windows = vmap(dynamic_slice_2d)(pool, r_starts, c_starts) # → (B, W, W, D) HBM
#   r_w     = gaussian(r_dist, sigma)                          # → (B, W)
#   c_w     = gaussian(c_dist, sigma)                          # → (B, W)
#   w_2d    = einsum("bi,bj->bij", r_w, c_w)                  # → (B, W, W) HBM
#   agg     = einsum("bij,bijd->bd", w_2d, windows)            # → (B, D)
#
# This kernel eliminates BOTH the (B, W, W, D) windows buffer AND the (B, W, W)
# weight matrix.  For a typical W=16 window that's:
#   16×16×512 bf16 = 262 KB   saved per batch sample per retrieval head
#   16×16     f32  =   4 KB   saved per batch sample per retrieval head
# ─────────────────────────────────────────────────────────────────────────────

def _pool2d_body(
    pool_ref,     # (R, C, D) bf16 — full pool
    r_start_ref,  # (1,)      int32
    c_start_ref,  # (1,)      int32
    r_center_ref, # (1,)      float32
    c_center_ref, # (1,)      float32
    sigma_ref,    # (1,)      float32
    out_ref,      # (1, D)    float32
    *, W: int, R: int, C: int,
):
    """Kernel body: one batch sample.  Fuses 2-D gather + Gaussian + dot."""
    r_s   = r_start_ref[0]
    c_s   = c_start_ref[0]
    r_cen = r_center_ref[0].astype(jnp.float32)
    c_cen = c_center_ref[0].astype(jnp.float32)
    sig   = sigma_ref[0].astype(jnp.float32)
    D     = pool_ref.shape[2]

    # ── 1. Dynamic gather: load W×W×D values from HBM ────────────────────────
    window = pool_ref[pl.dslice(r_s, W), pl.dslice(c_s, W), :]  # (W, W, D) bf16

    # ── 2. Separable 2-D Gaussian weights in registers ───────────────────────
    s_sq  = (sig + jnp.float32(1e-6)) ** 2
    r_pos = jnp.arange(W, dtype=jnp.float32) + r_s.astype(jnp.float32)
    c_pos = jnp.arange(W, dtype=jnp.float32) + c_s.astype(jnp.float32)
    r_w   = jnp.exp(-(r_pos - r_cen) ** 2 / (2.0 * s_sq))      # (W,)
    c_w   = jnp.exp(-(c_pos - c_cen) ** 2 / (2.0 * s_sq))      # (W,)

    # Outer product → (W, W) — kept in registers, never written to HBM
    w_2d  = jnp.outer(r_w, c_w) + jnp.float32(1e-6)            # (W, W)
    w_2d  = w_2d / jnp.sum(w_2d)                                # normalise

    # ── 3. Weighted sum (W*W,) × (W*W, D) → (D,) ────────────────────────────
    w_flat  = w_2d.reshape(-1)                                   # (W*W,)
    win_f32 = window.astype(jnp.float32).reshape(-1, D)          # (W*W, D)
    agg     = jnp.dot(w_flat, win_f32)                           # (D,)

    out_ref[0, :] = agg


def _pool2d_pallas_impl(pool, r_start, c_start, r_center, c_center, sigma, W, R, C):
    """Actual Pallas kernel call (or JAX fallback). No autodiff wrapping."""
    if not _use_pallas():
        return _pool2d_jax_fallback(pool, r_start, c_start, r_center, c_center, sigma, W, C)

    B, D = sigma.shape[0], pool.shape[2]
    kernel = functools.partial(_pool2d_body, W=W, R=R, C=C)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((B, D), jnp.float32),
        grid_spec=pl.GridSpec(
            grid=(B,),
            in_specs=[
                pl.BlockSpec((R, C, D), lambda i: (0, 0, 0)),
                pl.BlockSpec((1,), lambda i: (i,)),
                pl.BlockSpec((1,), lambda i: (i,)),
                pl.BlockSpec((1,), lambda i: (i,)),
                pl.BlockSpec((1,), lambda i: (i,)),
                pl.BlockSpec((1,), lambda i: (i,)),
            ],
            out_specs=pl.BlockSpec((1, D), lambda i: (i, 0)),
        ),
        interpret=_interpret_mode(),
    )(pool, r_start, c_start, r_center, c_center, sigma)


# custom_vjp: Pallas forward, pure-JAX backward.
# nondiff_argnums covers W, R, C (Python ints).
@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def pool_retrieve_2d_pallas(
    pool:     jnp.ndarray,   # (R, C, D)  bfloat16 — full 2-D pool storage
    r_start:  jnp.ndarray,   # (B,)       int32
    c_start:  jnp.ndarray,   # (B,)       int32
    r_center: jnp.ndarray,   # (B,)       float32
    c_center: jnp.ndarray,   # (B,)       float32
    sigma:    jnp.ndarray,   # (B,)       float32
    W:        int,
    R:        int,
    C:        int,
) -> jnp.ndarray:            # (B, D)     float32
    """Fused 2-D gather + separable Gaussian weighting + weighted sum.

    Replaces the vmap(dynamic_slice) + gaussian outer-product + einsum chain
    in CoordinateMassivePool2D._weight_and_aggregate().

    The (B, W, W, D) windows buffer and the (B, W, W) weight matrix are both
    eliminated — only the (B, D) output is written to HBM.
    Supports reverse-mode autodiff via custom_vjp (backward uses pure JAX).
    """
    return _pool2d_pallas_impl(pool, r_start, c_start, r_center, c_center, sigma, W, R, C)


def _pool2d_fwd(pool, r_start, c_start, r_center, c_center, sigma, W, R, C):
    out = _pool2d_pallas_impl(pool, r_start, c_start, r_center, c_center, sigma, W, R, C)
    return out, (pool, r_start, c_start, r_center, c_center, sigma)


def _pool2d_bwd(W, R, C, res, g):
    pool, r_start, c_start, r_center, c_center, sigma = res
    _, vjp_fn = jax.vjp(
        lambda p, rc, cc, s: _pool2d_jax_fallback(p, r_start, c_start, rc, cc, s, W, C),
        pool, r_center, c_center, sigma,
    )
    d_pool, d_r_center, d_c_center, d_sigma = vjp_fn(g)
    # r_start / c_start are int32 — no gradient
    return (
        d_pool,
        jnp.zeros_like(r_start),
        jnp.zeros_like(c_start),
        d_r_center,
        d_c_center,
        d_sigma,
    )


pool_retrieve_2d_pallas.defvjp(_pool2d_fwd, _pool2d_bwd)


def _pool2d_jax_fallback(pool, r_start, c_start, r_center, c_center, sigma, W, C):
    """Pure-JAX 2-D pool retrieve — identical semantics to the existing code."""
    from jax import lax

    D = pool.shape[2]

    def fetch_window(r_s, c_s):
        return lax.dynamic_slice(pool, (r_s, c_s, 0), (W, W, D))

    windows = jax.vmap(fetch_window)(r_start, c_start)                   # (B, W, W, D)
    s_sq    = (sigma.astype(jnp.float32) + 1e-6) ** 2
    r_idx   = jnp.arange(W)[None, :] + r_start[:, None]
    c_idx   = jnp.arange(W)[None, :] + c_start[:, None]
    r_w     = jnp.exp(-(r_idx.astype(jnp.float32) - r_center[:, None]) ** 2 / (2.0 * s_sq[:, None]))
    c_w     = jnp.exp(-(c_idx.astype(jnp.float32) - c_center[:, None]) ** 2 / (2.0 * s_sq[:, None]))
    w_2d    = jnp.einsum("bi,bj->bij", r_w, c_w) + 1e-6
    w_2d    = w_2d / jnp.sum(w_2d, axis=(-2, -1), keepdims=True)
    return jnp.einsum("bij,bijd->bd", w_2d, windows.astype(jnp.float32))
