from dpsn_r_jax.config import DPSNRConfig


def calculate_flops(config: DPSNRConfig, batch_size: int) -> float:
    """Estimates FLOPs per training step for the DPSNR architecture.

    Uses the standard 2× multiply-add convention for Dense/matmul ops:
        Dense(in, out) operating on (*, in) → 2 * in * out multiply-adds.

    Total FLOPs ≈ 3 × forward_FLOPs (approximate backward pass cost).

    Component breakdown
    ───────────────────
    1. TinyController encoder   — L transformer layers
    2. LearnedIndexer           — attention pooling + MLP + H-head projection
    3. Pool retrieval           — windowed dynamic_slice + Gaussian weights + avg
    4. Retrieval integrator     — 2-layer MLP on (B, T, 2D) → (B, T, D)
    5. AdaptiveComputeController — gated state transform + halt MLP
    6. LM Head (decoder)        — (B, T, D) × (D, V) projection

    NOTE: Pool retrieval uses a *windowed* slice of W vectors — it is NOT a
    full (d, pool_total_vectors) matmul.  The old `2*d*p` formula was wrong
    by up to 8 000× for large pool configs.  Correct cost: 2 * W * d per head.
    """
    n       = config.max_seq_len
    d       = config.controller_hidden_dim
    l       = config.controller_num_layers
    v       = config.vocab_size
    r       = config.max_reasoning_loops
    ff_mult = config.controller_ff_multiplier
    H       = config.num_indexer_heads  # number of retrieval heads

    # CoordinateMassivePool2D: axis_window = max(2, int(max_k**0.5)) per axis;
    # total retrieved = axis_window² per head-pair.
    axis_w = max(2, int(config.max_k ** 0.5))
    W = axis_w * axis_w          # vectors actually retrieved per head-pair
    n_head_pairs = max(1, H // 2)
    effective_H = n_head_pairs   # number of parallel pool calls after vmap

    # ─────────────────────────────────────────────────────────────────────────
    # 1. TinyController Encoder  (per sequence, per layer)
    #    • QKV projections:  3 separate (D→D) = 3 × 2 n d² but fused as one
    #      QKV block (D → 3D): 2 n d (3d) = 6 n d²
    #    • Output projection (D→D):               2 n d²
    #    • Attention scores: Q @ Kᵀ  →            2 n² d
    #    • Attention output: scores @ V  →         2 n² d
    #    • FFN up   (D → ff_dim):                  2 n d (ff_mult d)
    #    • FFN down (ff_dim → D):                  2 n (ff_mult d) d
    # ─────────────────────────────────────────────────────────────────────────
    # Attention window: local-window attention cost is O(n × window), not O(n²).
    # Using n² here would overcount by n/window (e.g. 32× for n=8192, window=256).
    attn_window = getattr(config, 'attn_window_size', n)  # full-attn if not set

    ff_dim = ff_mult * d
    encoder_fwd = l * (
        6 * n * d**2               # QKV projection (fused)
        + 2 * n * d**2             # output projection
        + 2 * n * attn_window * d  # Q @ Kᵀ  (local window)
        + 2 * n * attn_window * d  # scores @ V
        + 2 * n * d * ff_dim       # FFN up
        + 2 * n * ff_dim * d       # FFN down
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. LearnedIndexer (per reasoning loop — called once before scan warm-up,
    #    then once per loop iteration via reasoning_step)
    #    • Attention score (D→1 projection per token): 2 n d
    #    • Attention-weighted pooling (soft sum): n d  (just a weighted sum)
    #    • Trunk Dense 1 (D→D):                   2 d²
    #    • Trunk Dense 2 (D→D/2):                 d²
    #    • mu head (D/2 → H):                     d H
    #    • sigma head (D/2 → H):                  d H
    # ─────────────────────────────────────────────────────────────────────────
    indexer_fwd = (
        2 * n * d             # attention score projection
        + n * d               # weighted sum
        + 2 * d**2            # trunk dense 1
        + d**2                # trunk dense 2 (D → D/2)
        + d * H               # mu head
        + d * H               # sigma head
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Pool retrieval  (per head, per loop)
    #    dynamic_slice fetches W vectors — essentially W*D memory reads.
    #    Gaussian weight computation:  ~4W ops (subtract, square, exp, normalize)
    #    Weighted aggregation:          2 W D multiply-adds
    #    ∴ dominant cost ≈ 2 W D per head-call
    # ─────────────────────────────────────────────────────────────────────────
    pool_fwd = effective_H * (
        4 * W                 # Gaussian: dist², exp, normalize (cheap, memory-bound)
        + 2 * W * d           # weighted sum: Σ w_i · v_i
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Retrieval integrator  — nn.Sequential([Dense(D), gelu, Dense(D), LayerNorm])
    #    Input: (B, T, 2D) → Dense(2D→D) + Dense(D→D)
    #    • First Dense (2D→D):    2 n (2d) d = 4 n d²
    #    • Second Dense (D→D):    2 n d²
    # ─────────────────────────────────────────────────────────────────────────
    integrator_fwd = 4 * n * d**2 + 2 * n * d**2   # = 6 n d²

    # ─────────────────────────────────────────────────────────────────────────
    # 5. AdaptiveComputeController (per loop)
    #    Input pairs: (B, T, D) × 2  →  concat → (B, T, 2D)
    #    • state_gate  Dense(2D→D):           4 n d²
    #    • state_transform Dense(D→D):        2 n d²
    #    • halt_net Dense(D→D/4) + Dense(D/4→1):  n d²/2 + n d/4  ≈ n d²/2
    #    • loop_embed lookup:   negligible
    # ─────────────────────────────────────────────────────────────────────────
    acc_fwd = (
        4 * n * d**2          # state_gate
        + 2 * n * d**2        # state_transform
        + n * d**2 // 2       # halt_net dense layers (approx)
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. LM Head (Decoder): Dense(D→V) over (B, T, D)
    #    Note: vocab_size should be a multiple of 128 for efficient MXU use.
    # ─────────────────────────────────────────────────────────────────────────
    decoder_fwd = 2 * n * d * v

    # ─────────────────────────────────────────────────────────────────────────
    # Per-loop FLOPs (reasoning_step, executed `r` times via lax.scan)
    # ─────────────────────────────────────────────────────────────────────────
    loop_fwd = r * (indexer_fwd + pool_fwd + integrator_fwd + acc_fwd)

    # ─────────────────────────────────────────────────────────────────────────
    # Total: Forward + Backward (≈ 2× forward for activations/gradients)
    # The classic "3×" rule (Chinchilla / PaLM) assumes remat; without remat
    # it's closer to 2×. We use 3× to match convention and account for
    # optimiser overhead (Adam moment updates, etc.).
    # ─────────────────────────────────────────────────────────────────────────
    total_fwd   = encoder_fwd + decoder_fwd + loop_fwd
    total_flops = 3 * total_fwd * batch_size

    return float(total_flops)


def estimate_hbm_bytes_per_step(
    config: DPSNRConfig,
    batch_size: int,
    n_chips: int,
    tp_size: int = 1,
    gradient_checkpointing: bool = False,
) -> dict:
    """Estimate HBM bytes transferred per training step.

    This is a structural estimate based on parameter counts and activation
    sizes — NOT a hardware counter measurement.  Actual XLA traffic depends
    on op fusion, remat, and XLA memory planning.  Use it as a roofline
    lower-bound indicator, not a precise figure.

    Returns a dict with:
        total_bytes       : total bytes across all chips (forward+backward+opt)
        breakdown         : per-phase byte counts for attribution
        mbu_fraction      : total_bytes / (step_time × peak_bw × n_chips)
                            call site must pass step_time to complete this.
    """
    import math

    dp_size       = max(1, n_chips // tp_size)
    B_per_dp      = max(1, batch_size // dp_size)   # sequences per DP replica
    T             = config.max_seq_len
    D             = config.controller_hidden_dim
    L             = config.controller_num_layers
    V             = config.vocab_size
    ff_mult       = config.controller_ff_multiplier

    # ── Parameter byte counts ─────────────────────────────────────────────────
    # With TP, each chip holds (params / tp_size) bytes; we account for all chips
    # by using full param count × dp_size (dp_size independent replicas).
    attn_params   = 4 * D * D * L           # Q,K,V,O per layer
    ffn_params    = 2 * int(D * ff_mult) * D * L
    emb_params    = V * D                   # token embedding (+ LM head shared)
    head_params   = V * D                   # LM head (separate if not tied)
    ctrl_params   = attn_params + ffn_params + emb_params + head_params

    pool_params   = config.pool_grid_rows * config.pool_grid_cols * D

    # Bytes: bf16 = 2, f32 = 4
    ctrl_bf16     = ctrl_params * 2
    ctrl_f32      = ctrl_params * 4
    pool_bf16     = pool_params * 2
    pool_f32      = pool_params * 4

    # ── Forward pass HBM reads ────────────────────────────────────────────────
    # Each of dp_size replicas reads its local copy of controller weights.
    fwd_ctrl_rd   = ctrl_bf16 * dp_size

    # Pool fetch (once for prefetch, or per-loop for standard path)
    if getattr(config, 'prefetch_reasoning', False):
        PS            = getattr(config, 'prefetch_size', 8)
        fwd_pool_rd   = B_per_dp * dp_size * PS * PS * D * 2   # bf16
    else:
        axis_w        = max(2, int(getattr(config, 'max_k', 25) ** 0.5))
        SW            = axis_w * getattr(config, 'pool_super_window_factor', 2)
        fwd_pool_rd   = (B_per_dp * dp_size * SW * SW * D * 2 *
                         config.max_reasoning_loops)

    # Activation writes: B_per_dp × T × D per layer (bf16)
    act_per_layer = B_per_dp * T * D * 2
    if gradient_checkpointing:
        n_stored      = max(1, int(math.sqrt(L)))  # ≈ sqrt(L) checkpoint layers
        fwd_act_wr    = act_per_layer * n_stored * dp_size
    else:
        fwd_act_wr    = act_per_layer * L * dp_size

    # ── Backward pass HBM traffic ─────────────────────────────────────────────
    if gradient_checkpointing:
        # Recompute: re-read weights + re-run from checkpoints; then actual bwd
        bwd_ctrl_rd   = ctrl_bf16 * dp_size * 2   # recompute + backward
        bwd_act_rw    = fwd_act_wr * 2             # read checkpoints, write grad-acts
    else:
        bwd_ctrl_rd   = ctrl_bf16 * dp_size        # read once for backward
        bwd_act_rw    = fwd_act_wr                 # stored activations read once

    bwd_grad_wr       = ctrl_f32 * dp_size         # accumulated weight gradients (f32)

    # ── Optimizer HBM traffic ─────────────────────────────────────────────────
    # Dense AdamW: read+write params(bf16) + m(f32) + v(f32) per dp replica
    opt_ctrl          = (ctrl_bf16 + ctrl_f32 * 2) * 2 * dp_size   # ×2 for read+write

    # Pool sparse Adam: vmap gather reads touched slices, but at[].set() on
    # the full flattened pool forces XLA to touch pool_m and pool_v fully.
    # Conservative: assume full pool_m and pool_v read+write each step.
    opt_pool          = (pool_bf16 + pool_f32 * 2) * 2             # ×2 read+write (pool shared)

    # ── Totals ────────────────────────────────────────────────────────────────
    fwd_total  = fwd_ctrl_rd + fwd_pool_rd + fwd_act_wr
    bwd_total  = bwd_ctrl_rd + bwd_act_rw + bwd_grad_wr
    opt_total  = opt_ctrl + opt_pool
    total      = fwd_total + bwd_total + opt_total

    return {
        "total_bytes"    : total,
        "fwd_bytes"      : fwd_total,
        "bwd_bytes"      : bwd_total,
        "opt_bytes"      : opt_total,
        "ctrl_param_bytes": ctrl_bf16,
        "pool_param_bytes": pool_bf16,
    }


# Peak HBM bandwidth per chip (bytes/s) for known TPU generations.
# Used by the roofline metric in main.py.
TPU_HBM_BW_PER_CHIP = {
    "v5e": 819e9,
    "v4":  1200e9,
    "v3":  900e9,
    "v2":  700e9,
}
TPU_PEAK_TFLOPS_BF16_PER_CHIP = {
    "v5e": 197.0,
    "v4":  275.0,
    "v3":  123.0,
    "v2":  45.0,
}


def roofline_metrics(
    config: DPSNRConfig,
    batch_size: int,
    n_chips: int,
    tp_size: int,
    step_time_s: float,
    actual_tflops: float,
    gradient_checkpointing: bool = False,
    tpu_gen: str = "v5e",
) -> dict:
    """Compute roofline-model metrics for a completed training step.

    Returns:
        mfu          : compute utilisation  (actual / peak TFLOPS)
        mbu          : HBM bandwidth util   (estimated bytes / peak bandwidth)
        bottleneck   : "Compute", "HBM-BW", or "Latency/Sync"
        ideal_ms     : time step would take if MXU at 100% (lower bound)
        stall_ms     : step_time - ideal_ms  (time MXU was not doing useful work)
        bw_GB_s      : estimated HBM bandwidth being used (GB/s, all chips)
        breakdown    : per-phase byte counts
    """
    peak_tflops_per_chip = TPU_PEAK_TFLOPS_BF16_PER_CHIP.get(tpu_gen, 197.0)
    peak_bw_per_chip     = TPU_HBM_BW_PER_CHIP.get(tpu_gen, 819e9)

    peak_tflops_total    = peak_tflops_per_chip * n_chips   # TFLOPS across all chips
    peak_bw_total        = peak_bw_per_chip * n_chips        # bytes/s across all chips

    mfu = actual_tflops / peak_tflops_total                  # fraction [0, 1]

    hbm_est = estimate_hbm_bytes_per_step(
        config, batch_size, n_chips, tp_size, gradient_checkpointing
    )
    mbu = hbm_est["total_bytes"] / (step_time_s * peak_bw_total)  # fraction [0, 1]

    bw_GB_s = hbm_est["total_bytes"] / step_time_s / 1e9  # actual estimated GB/s (all chips)

    # Ideal time if compute was the only bottleneck (100% MXU)
    from dpsn_r_jax.utils.metrics import calculate_flops
    flops = calculate_flops(config, batch_size)
    ideal_s  = flops / (peak_tflops_total * 1e12)
    ideal_ms = ideal_s * 1000.0
    stall_ms = max(0.0, step_time_s * 1000.0 - ideal_ms)

    # Bottleneck classification:
    #   MFU > 50%              → genuinely compute-saturated
    #   MBU > MFU × 1.5       → HBM is the limit (MXU starves waiting for data)
    #   both < 15%             → latency/sync dominates (sequential ops, comms)
    if mfu > 0.50:
        bottleneck = "Compute"
    elif mbu > mfu * 1.5:
        bottleneck = "HBM-BW"
    elif mfu < 0.10 and mbu < 0.15:
        bottleneck = "Latency/Sync"
    else:
        bottleneck = "Mixed"

    return {
        "mfu"        : mfu,
        "mbu"        : mbu,
        "bottleneck" : bottleneck,
        "ideal_ms"   : ideal_ms,
        "stall_ms"   : stall_ms,
        "bw_GB_s"    : bw_GB_s,
        "breakdown"  : hbm_est,
    }


def summarise_flops(config: DPSNRConfig, batch_size: int) -> None:
    """Pretty-print a per-component FLOP breakdown for a given config.

    Useful for verifying the estimate before a long training run:
        from dpsn_r_jax.utils.metrics import summarise_flops
        summarise_flops(config, batch_size=256)
    """
    n       = config.max_seq_len
    d       = config.controller_hidden_dim
    l       = config.controller_num_layers
    v       = config.vocab_size
    r       = config.max_reasoning_loops
    ff_mult = config.controller_ff_multiplier
    H       = config.num_indexer_heads

    axis_w = max(2, int(config.max_k ** 0.5))
    W = axis_w * axis_w
    effective_H = max(1, H // 2)

    attn_window = getattr(config, 'attn_window_size', n)
    ff_dim = ff_mult * d
    encoder_fwd = l * (
        6*n*d**2 + 2*n*d**2
        + 2*n*attn_window*d + 2*n*attn_window*d
        + 2*n*d*ff_dim + 2*n*ff_dim*d
    )
    indexer_fwd = 2*n*d + n*d + 2*d**2 + d**2 + d*H + d*H
    pool_fwd    = effective_H * (4*W + 2*W*d)
    integrator  = 6*n*d**2
    acc_fwd     = 4*n*d**2 + 2*n*d**2 + n*d**2 // 2
    loop_fwd    = r * (indexer_fwd + pool_fwd + integrator + acc_fwd)
    decoder_fwd = 2*n*d*v
    total_fwd   = encoder_fwd + decoder_fwd + loop_fwd
    total_step  = 3 * total_fwd * batch_size

    T = 1e12
    print("\n" + "=" * 58)
    print(f"  FLOP Breakdown  (B={batch_size}, n={n}, d={d}, r={r})")
    print("=" * 58)
    print(f"  {'TinyController encoder':<32} {encoder_fwd*batch_size*3/T:>8.3f} TFLOPS-eq")
    print(f"  {'Reasoning loop × ' + str(r):<32} {loop_fwd*batch_size*3/T:>8.3f} TFLOPS-eq")
    print(f"    {'LearnedIndexer':<30} {r*indexer_fwd*batch_size*3/T:>8.3f}")
    print(f"    {f'Pool retrieval (W={W}, H={effective_H})':<30} {r*pool_fwd*batch_size*3/T:>8.3f}")
    print(f"    {'Integrator':<30} {r*integrator*batch_size*3/T:>8.3f}")
    print(f"    {'ACC':<30} {r*acc_fwd*batch_size*3/T:>8.3f}")
    print(f"  {'LM head decoder':<32} {decoder_fwd*batch_size*3/T:>8.3f} TFLOPS-eq")
    print("-" * 58)
    print(f"  {'TOTAL per step':<32} {total_step/T:>8.3f} TFLOPS-eq")
    print(f"  (TPU peak utilisation = measured_TFLOPS / {total_step/T:.3f})")
    print("=" * 58 + "\n")
