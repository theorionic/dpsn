"""TPU HBM memory debugging utilities for DPSNR training.

Two independent information sources:
  1. jax.devices()[...].memory_stats()  ─ live TPU allocator stats
  2. Parameter/state tree analysis       ─ what's statically allocated

Usage:
    from dpsn_r_jax.utils.memory_debug import print_tpu_memory, print_param_memory

    print_param_memory(state, config, batch_size)   # call once after init
    print_tpu_memory("before train_step")           # call anywhere
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any


# ── helpers ───────────────────────────────────────────────────────────────────

def _bytes(arr: Any) -> int:
    """Return byte count of a JAX/numpy array."""
    try:
        return arr.size * arr.dtype.itemsize
    except Exception:
        return 0


def _tree_bytes(tree: Any) -> int:
    return sum(_bytes(x) for x in jax.tree_util.tree_leaves(tree))


def _gb(n_bytes: int | float) -> float:
    return n_bytes / (1024 ** 3)


# ── Public API ─────────────────────────────────────────────────────────────────

def print_tpu_memory(label: str = "") -> None:
    """Print live TPU HBM allocator stats for every device.

    Uses jax.devices()[i].memory_stats() which returns a dict with keys:
        bytes_in_use       — currently allocated HBM
        bytes_limit        — total HBM on the chip
        peak_bytes_in_use  — high-water mark since last reset
    """
    sep = "─" * 60
    tag = f" [{label}]" if label else ""
    print(f"\n{sep}")
    print(f"  TPU HBM Usage{tag}")
    print(sep)

    total_in_use   = 0
    total_limit    = 0
    total_peak     = 0

    for dev in jax.devices():
        try:
            stats = dev.memory_stats()
        except Exception:
            print(f"  {dev}: memory_stats() not available on this backend")
            continue

        in_use = stats.get("bytes_in_use", 0)
        limit  = stats.get("bytes_limit", 0)
        peak   = stats.get("peak_bytes_in_use", 0)
        pct    = (in_use / limit * 100) if limit else 0.0

        total_in_use += in_use
        total_limit  += limit
        total_peak   += peak

        print(f"  {str(dev):<20}  in_use: {_gb(in_use):6.2f} GB  "
              f"peak: {_gb(peak):6.2f} GB  "
              f"limit: {_gb(limit):6.2f} GB  "
              f"used: {pct:5.1f}%")

    if total_limit:
        print(sep)
        total_pct = total_in_use / total_limit * 100
        print(f"  {'TOTAL':<20}  in_use: {_gb(total_in_use):6.2f} GB  "
              f"peak: {_gb(total_peak):6.2f} GB  "
              f"limit: {_gb(total_limit):6.2f} GB  "
              f"used: {total_pct:5.1f}%")
    print(f"{sep}\n")


def print_param_memory(state: Any, config: Any, batch_size: int, loss_chunk_size: int = 0) -> None:
    """Print a per-component HBM breakdown for params, optimizer, and activations.

    Args:
        state:           The TrainState object.
        config:          DPSNRConfig.
        batch_size:      Current batch size (used for activation estimates).
        loss_chunk_size: If > 0, logits peak is per-chunk; 0 = full batch.
    """
    sep  = "═" * 72
    sep2 = "─" * 72
    bf16 = getattr(config, "use_bf16", False)
    dtype_str   = "bfloat16" if bf16 else "float32"
    act_factor  = 2 if bf16 else 4   # bytes per element for activations
    param_factor= 4                  # params always float32

    print(f"\n{sep}")
    print(f"  DPSNR HBM Memory Breakdown  (batch={batch_size}, dtype={dtype_str})")
    print(sep)

    # ── 1. Parameters ─────────────────────────────────────────────────────────
    print(f"  {'PARAMETERS (always float32)':}")
    print(sep2)

    params = state.params
    components = {
        "TinyController (transformer)": params.get("controller", {}),
        "LearnedIndexer  (MLP)":        params.get("indexer", {}),
        "CoordinateMassivePool (2D)":   params.get("pool", {}),
        "AdaptiveCompute (ACC)":        params.get("acc", {}),
        "RetrievalIntegrator (MLP)":    params.get("retrieval_integrator", {}),
    }

    total_param_bytes = 0
    for name, subtree in components.items():
        b = _tree_bytes(subtree)
        total_param_bytes += b
        if b > 0:
            print(f"  {name:<40}  {_gb(b):7.3f} GB  ({b // (1024**2):,} MB)")

    print(sep2)
    print(f"  {'Total Parameters':<40}  {_gb(total_param_bytes):7.3f} GB")

    # ── 2. Optimizer / momentum state ─────────────────────────────────────────
    print(f"\n  {'OPTIMIZER STATE (float32)':}")
    print(sep2)

    pool_m_bytes = _tree_bytes(state.pool_m)
    pool_v_bytes = _tree_bytes(state.pool_v)

    try:
        opt_bytes = _tree_bytes(state.opt_state)
    except Exception:
        opt_bytes = 0

    dense_adam_bytes = max(0, opt_bytes)   # AdamW m+v for controller/indexer/acc
    print(f"  {'pool_m (sparse Adam 1st moment)':<40}  {_gb(pool_m_bytes):7.3f} GB")
    print(f"  {'pool_v (sparse Adam 2nd moment)':<40}  {_gb(pool_v_bytes):7.3f} GB")
    print(f"  {'AdamW opt_state (dense params m+v)':<40}  {_gb(dense_adam_bytes):7.3f} GB")
    total_opt_bytes = pool_m_bytes + pool_v_bytes + dense_adam_bytes
    print(sep2)
    print(f"  {'Total Optimizer State':<40}  {_gb(total_opt_bytes):7.3f} GB")

    # ── 3. Data prefetch buffers (on-device) ──────────────────────────────────
    # DevicePrefetchIterator: prefetch_depth=2 batches of (B, T) int32
    T = config.max_seq_len
    prefetch_depth = 2
    data_bytes = prefetch_depth * batch_size * T * 4   # int32 = 4 bytes
    print(f"\n  {'DATA BUFFERS (on-device prefetch)':}")
    print(sep2)
    print(f"  {'DevicePrefetchIterator (depth=2)':<40}  {_gb(data_bytes):7.3f} GB  "
          f"({batch_size} × {T} × int32 × {prefetch_depth})")

    # ── 4. Activation estimates (scales with batch_size) ─────────────────────
    print(f"\n  ACTIVATION ESTIMATES ({dtype_str}, fwd+bwd)")
    print(sep2)
    B  = batch_size
    T  = config.max_seq_len
    D  = config.controller_hidden_dim
    L  = config.controller_num_layers
    V  = config.vocab_size
    R  = config.max_reasoning_loops
    ff = int(D * config.controller_ff_multiplier)

    def est(nelems: int, label: str, note: str = "") -> int:
        b = nelems * act_factor * 2   # ×2 for fwd+bwd
        suffix = f"  ({note})" if note else ""
        print(f"  {label:<40}  {_gb(b):7.3f} GB{suffix}")
        return b

    a_embed   = est(B * T * D,     "Embeddings")
    a_attn_kv = est(B * L * T * D, "Attention QKV cache (all layers)",
                    f"remat saves ~{L-1}/{L} layers")
    a_ffn     = est(B * L * T * ff,"FFN intermediates (all layers)",
                    "remat saves ~{}/{} layers".format(L-1, L))

    # Logits: with chunked loss only one chunk is live at a time
    if loss_chunk_size > 0:
        full_logits_bytes  = B * T * V * act_factor * 2
        chunk_logits_bytes = loss_chunk_size * T * V * act_factor * 2
        saving_gb = _gb(full_logits_bytes - chunk_logits_bytes)
        a_logits = chunk_logits_bytes
        print(f"  {'Logits peak  (chunked, per chunk)':<40}  {_gb(a_logits):7.3f} GB"
              f"  (chunk={loss_chunk_size}×{T}×{V}, saves {saving_gb:.2f} GB vs full)")
    else:
        a_logits  = est(B * T * V, "Logits  (B×T×V  — LARGEST)",
                        f"{B}×{T}×{V}")

    a_reason  = est(R * B * T * D, "Reasoning loop states (NOT remated)",
                    f"{R} loops × (B,T,D)")

    total_act = a_embed + a_attn_kv + a_ffn + a_logits + a_reason
    print(sep2)
    note = "(full remat would save most of attention/FFN)" if not getattr(config, "gradient_checkpointing", False) else ""
    print(f"  {'Total Activations (est.)':<40}  {_gb(total_act):7.3f} GB  {note}")

    # ── 5. Grand total ────────────────────────────────────────────────────────
    grand_total = total_param_bytes + total_opt_bytes + data_bytes + total_act
    print(f"\n{sep}")
    print(f"  {'GRAND TOTAL  (estimated)':<40}  {_gb(grand_total):7.3f} GB")
    n_chips = len(jax.devices())
    per_chip = (total_param_bytes - _tree_bytes(params.get("pool", {}))) / 1   # replicated
    pool_sharded = _tree_bytes(params.get("pool", {})) / n_chips
    activations_sharded = (a_logits + a_attn_kv + a_ffn + a_embed + a_reason) / n_chips
    opt_sharded = (pool_m_bytes + pool_v_bytes) / n_chips  + dense_adam_bytes  # dense replicated
    per_chip_est = (per_chip + pool_sharded + activations_sharded + opt_sharded + data_bytes / n_chips)
    print(f"  {'Per-chip estimate ({} chips)'.format(n_chips):<40}  {_gb(per_chip_est):7.3f} GB")
    print(f"  {'TPU v5e-8 total HBM':<40}  128.000 GB  ({n_chips} chips × 16 GB)")
    print(f"{sep}\n")
