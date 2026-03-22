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
