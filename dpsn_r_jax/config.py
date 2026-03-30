from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class PoolConfig:
    total_vectors: int
    hidden_dim: int


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning DPSNR model."""

    # Data paths
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    template: str = "alpaca"
    template_path: Optional[str] = None

    # Training hyperparameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    num_train_epochs: int = 3
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 1

    # LR scheduler
    lr_scheduler_type: str = "cosine"  # linear, cosine, constant, constant_with_warmup

    # Model freezing
    freeze_controller: bool = False
    freeze_pool: bool = True
    freeze_indexer: bool = False

    # Checkpoint
    load_pretrained: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None

    # Evaluation
    evaluation_strategy: str = "steps"  # no, steps, epoch
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3

    # Logging
    logging_steps: int = 10
    report_to: str = "tensorboard"  # tensorboard, wandb, none


@dataclass
class DPSNRConfig:
    vocab_size: int = 24
    controller_hidden_dim: int = 64
    controller_num_layers: int = 2
    controller_num_heads: int = 2
    # GQA: number of KV heads shared across query heads.
    # 0 = full MHA (num_kv_heads == num_heads).
    # Typical: num_heads // 4  (e.g. 12 Q heads → 3 KV heads, 4x KV memory savings).
    controller_num_kv_heads: int = 0
    controller_ff_multiplier: float = 2.0
    max_seq_len: int = 64
    dropout: float = 0.1
    pool_total_vectors: int = 1000
    pool_hidden_dim: int = 64
    librarian_hidden_dim: int = 32
    max_reasoning_loops: int = 4
    min_reasoning_loops: int = 1
    halt_threshold: float = 0.99
    min_k: int = 4
    max_k: int = 32
    num_clusters_to_search: int = 4
    hf_dataset_name: Optional[str] = None
    hf_tokenizer_name: Optional[str] = None
    streaming: bool = True
    pad_token_id: int = 0
    max_steps: Optional[int] = None
    generation_steps: Optional[int] = None
    generation_max_tokens: int = 20
    generation_prompts: Optional[list[str]] = None
    learning_rate: float = 3e-4
    num_workers: int = 4
    gradient_checkpointing: bool = False
    use_bf16: bool = False
    # LM head chunking: compute cross-entropy in sub-batches of this size to
    # avoid ever materialising the full (B, T, vocab) logits tensor.
    # 0 = disabled (standard path).  Recommended: 16 for TPU v5e-8.
    loss_chunk_size: int = 0
    # ── Indexer improvements ───────────────────────────────────────────────
    num_indexer_heads: int = 1      # Multi-head pool queries per reasoning step
    sigma_min: float = 0.01         # Minimum retrieval bandwidth (sharp/precise)
    sigma_max: float = 5.0          # Maximum retrieval bandwidth (broad/soft)
    # MLP width inside LearnedIndexer.  0 = use controller_hidden_dim (default,
    # backward-compat).  Set to a large value to give the indexer its own
    # independent parameter budget (e.g. 10240 → ~62M params with D=1024).
    indexer_hidden_dim: int = 0
    finetune: Optional[FineTuningConfig] = None

    # ── Precision Routing ──────────────────────────────────────────────────
    # 2D Grid Pool: instead of (N, D), pool is (rows, cols, D).
    # Precision per coordinate: 1/sqrt(N) instead of 1/N  → huge improvement.
    pool_grid_rows: int = 512   # rows × cols = total pool vectors
    pool_grid_cols: int = 512

    # Sigma annealing: sigma_max decays from its initial value to sigma_target
    # over sigma_anneal_steps training steps.  0 disables annealing.
    # Effect: routing starts broad (easy to learn), ends precise (accurate retrieval).
    sigma_anneal_steps: int = 0
    sigma_target: float = 0.05

    # Precision auxiliary loss: small weight penalising large sigma values.
    # The model is rewarded for using precise, narrow retrievals.
    # weight is linearly ramped from 0 → precision_loss_weight over sigma_anneal_steps.
    precision_loss_weight: float = 0.0

    # ── Prefetch Reasoning (SRAM buffer design) ────────────────────────────
    # When True, the reasoning loop fetches pool vectors ONCE before the scan
    # and passes them as a lax.scan carry (XLA keeps the buffer in on-chip
    # SRAM).  Every reasoning iteration then reads from SRAM via
    # dot-product attention instead of issuing a new HBM dynamic_slice.
    #
    # Hardware benefit:
    #   Original design  : 1 HBM fetch per reasoning loop iteration
    #                      (100 ns latency × max_reasoning_loops)
    #   Prefetch design  : 1 HBM fetch total, then SRAM reads (~1 ns each)
    #
    # SRAM cost estimate per chip (batch sharded, B_per_chip = B / n_chips):
    #   B_per_chip × prefetch_size² × hidden_dim × 2 bytes (bf16)
    #   e.g. xxl, 8 chips, B=32: 4 × 64² × 1024 × 2 = 33 MB  ← fits in 128 MB
    #
    # Tradeoff: pool retrieval coordinates are computed once (initial state)
    # and fixed for all reasoning loops.  The model compensates by attending
    # softly over prefetch_size² candidate vectors instead of a hard Gaussian
    # slice, which is at least as expressive for broad sigma values.
    #
    # Recommended: prefetch_size=64 (4096 candidates, ~33 MB/chip on xxl).
    # Use 128 only on smaller configs where SRAM headroom allows.
    prefetch_reasoning: bool = False
    prefetch_size: int = 64   # per-axis size; total candidates = prefetch_size²

    # ── Splash Attention (Pallas TPU kernel) ───────────────────────────────
    # When True, FlashCausalSelfAttention uses splash_attention (Pallas TPU)
    # instead of Flax's nn.dot_product_attention.
    # Requirements: TPU backend, seq_len >= 128 and divisible by 128.
    # Falls back to standard attention automatically if conditions aren't met.
    use_flash_attention: bool = True

    # ── Sliding Window (Local) Attention ───────────────────────────────────
    # Each token attends to only the nearest `attn_window_size` past tokens
    # instead of the full sequence.  0 = full causal attention.
    #
    # Why: the controller is architecturally a *local context encoder* —
    # long-range reasoning is handled by the pool + reasoning loop.
    # Full O(T²) attention at T=8192 is wasteful and contradicts this design.
    # With window=512 the controller is O(T × 512) — 16× cheaper at T=8192.
    #
    # The pool's reasoning loop captures everything beyond the window, so
    # nothing is lost semantically.
    attn_window_size: int = 0

    # ── SRAM Super-Window pre-fetching (Opt-2) ─────────────────────────────
    # Before the reasoning loop, fetch pool_super_window_factor × window_size
    # vectors from HBM in one pass.  The XLA lax.scan carry mechanism then
    # holds this wider tensor in on-chip SRAM across all reasoning iterations,
    # reducing per-iteration HBM latency from ~100 ns to ~1 ns.
    # Set to 1 to disable. Recommended: 2 (since 2x means 2x2 = 4x the total vectors).
    pool_super_window_factor: int = 2

    @classmethod
    def from_yaml(cls, path: str) -> "DPSNRConfig":
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_config)


def get_model_config(name: str) -> DPSNRConfig:
    """
    Returns a predefined configuration based on the name.

    Sizes:
    - tiny: ~6M params (Debug/CPU)
    - base: ~125M params (TPU v3-8 / GPU)
    - large: ~350M params (TPU Pod Slice)
    - xl: ~1.5B params (Large TPU Pod - Massive Pool)
    - xxl: ~2.8B params (TPU v5e-8) — 2D pool mandatory at this scale

    Precision-routing variants (same param count, better pool addressing):
    - precise_tiny:  tiny  + 2D pool + sigma annealing (for quick experiments)
    - precise_large: large + 2D pool + sigma annealing + precision loss

    Pool-dominant design (tiny controller + massive pool, custom 8K tokenizer):
    - mini_pool: ~84M controller + ~63M indexer + ~1.07B pool  (TPU v5e-8)
                 8192-vocab tokenizer keeps controller embedding small;
                 oversized indexer acts as a deep feature extractor for precise
                 pool addressing; pool carries the bulk of world knowledge.
    """
    if name == "tiny":
        return DPSNRConfig(
            vocab_size=24,
            controller_hidden_dim=32,
            controller_num_layers=2,
            controller_num_heads=2,
            controller_ff_multiplier=2.0,
            max_seq_len=64,
            dropout=0.0,
            pool_total_vectors=100,
            pool_hidden_dim=32,
            pool_grid_rows=10,
            pool_grid_cols=10,
            librarian_hidden_dim=16,
            max_reasoning_loops=2,
            min_reasoning_loops=1,
            halt_threshold=0.5,
            min_k=2,
            max_k=10,
            num_clusters_to_search=2,
            learning_rate=1e-3,
        )

    elif name == "base":
        return DPSNRConfig(
            # vocab_size=50304 is the next multiple of 128 above GPT-2's 50257.
            # Setting it explicitly avoids XLA's internal padding of the LM head
            # matmul (hidden_dim, vocab_size), which wastes MXU cycles on TPU.
            # Same trick used by GPT-NeoX and Llama tokenizers.
            vocab_size=50304,  # 393 × 128 = 50304
            controller_hidden_dim=512,
            controller_num_layers=6,
            controller_num_heads=8,
            max_seq_len=512,
            pool_total_vectors=65536,  # 256 × 256
            pool_hidden_dim=512,
            pool_grid_rows=256,
            pool_grid_cols=256,
            max_reasoning_loops=4,
            learning_rate=6e-4,
        )

    elif name == "large":
        return DPSNRConfig(
            vocab_size=50304,  # 393 × 128, MXU-aligned
            controller_hidden_dim=768,
            controller_num_layers=12,
            controller_num_heads=12,
            max_seq_len=4096,
            pool_total_vectors=262144,  # 512 × 512
            pool_hidden_dim=768,
            pool_grid_rows=512,
            pool_grid_cols=512,
            max_reasoning_loops=6,
            learning_rate=3e-4,
            attn_window_size=512,   # O(T×512) vs O(T²); pool handles long-range
        )

    elif name == "xl":
        return DPSNRConfig(
            vocab_size=50304,  # 393 × 128, MXU-aligned
            controller_hidden_dim=1024,
            controller_num_layers=16,
            controller_num_heads=16,
            max_seq_len=8192,
            pool_total_vectors=1048576,  # 1024 × 1024
            pool_hidden_dim=1024,
            pool_grid_rows=1024,
            pool_grid_cols=1024,
            max_reasoning_loops=8,
            learning_rate=2e-4,
            attn_window_size=512,   # 512/8192 = 6% local; pool handles the rest
        )

    # ── Precision Routing Variants ─────────────────────────────────────────────
    elif name == "precise_tiny":
        return DPSNRConfig(
            # Same capacity as tiny, but with full precision routing stack
            vocab_size=24,
            controller_hidden_dim=32,
            controller_num_layers=2,
            controller_num_heads=2,
            controller_ff_multiplier=2.0,
            max_seq_len=64,
            dropout=0.0,
            pool_total_vectors=100,    # 10×10 grid
            pool_hidden_dim=32,
            librarian_hidden_dim=16,
            max_reasoning_loops=2,
            halt_threshold=0.5,
            min_k=2,
            max_k=10,
            learning_rate=1e-3,
            num_indexer_heads=2,
            sigma_min=0.01,
            sigma_max=5.0,             # starts broad
            # 2D pool: 10×10 = 100 vectors; 10x easier to address precisely
            pool_grid_rows=10,
            pool_grid_cols=10,
            # Sigma annealing: broad → precise over 5 000 steps
            sigma_anneal_steps=5_000,
            sigma_target=0.05,
            # Precision loss: penalise large sigma (ramped in over annealing period)
            precision_loss_weight=0.01,
            use_flash_attention=True,
        )

    elif name == "precise_large":
        return DPSNRConfig(
            # Same 340M params as large, but with full precision routing
            vocab_size=50304,  # 393 × 128, MXU-aligned
            controller_hidden_dim=768,
            controller_num_layers=12,
            controller_num_heads=6,   # head_dim=128 → MXU-aligned (was 12→head_dim=64, 50% MXU waste)
            max_seq_len=4096,
            pool_total_vectors=262144,
            pool_hidden_dim=768,
            max_reasoning_loops=4,    # reduced from 6 → saves ~33% pool gradient scatter cost
            learning_rate=3e-4,
            attn_window_size=512,
            num_indexer_heads=4,
            sigma_min=0.01,
            sigma_max=5.0,
            # 2D pool: 512×512 = 262 144 vectors
            # Each coordinate only needs 1/512 ≈ 0.2% precision (vs 1/262144 = 0.0004%)
            pool_grid_rows=512,
            pool_grid_cols=512,
            # Anneal sigma over first 50 000 steps: broad → precise routing
            sigma_anneal_steps=50_000,
            sigma_target=0.05,
            # Precision loss: small penalty on broad sigma
            precision_loss_weight=0.01,
            use_flash_attention=True,
            # SRAM prefetch reasoning: fetch pool vectors ONCE → all loops read from SRAM.
            # 1 HBM fetch instead of max_reasoning_loops fetches per step.
            # 16×16 = 256 candidates; SRAM cost = B/chip × 256 × 768 × 2 B ≈ 1.5 MB/chip.
            prefetch_reasoning=True,
            prefetch_size=16,          # per-axis; total K = 16² = 256 SRAM candidates
            pool_super_window_factor=1,  # disable Opt-2 (superseded by prefetch path)
        )

    elif name == "xxl":
        # ~2.82B total parameters — designed for TPU v5e-8 (16 GB/chip × 8 chips)
        #
        # Parameter breakdown:
        #   Pool  : 1536 × 1536 × 1024 = 2,415,919,104  (~2.42B)  — stored bfloat16
        #   Controller : hidden=1024, 24 layers, 16 heads  (~405M)
        #   Total                                          (~2.82B)
        #
        # Per-chip HBM estimate (8 chips):
        #   Controller params + Adam m/v (replicated) : ~4.86 GB/chip
        #   Pool params + Adam m/v (sharded)          : ~3.02 GB/chip
        #   Static total                              : ~7.88 GB/chip
        #   Available for activations                 : ~7 GB/chip
        #   → batch=4/chip (32 total) is comfortable with flash-attn + grad-ckpt
        #
        # 2D pool is NOT optional here:
        #   1D addressing would need 1/2,400,000 coordinate precision (unlearnable).
        #   2D addressing needs only 1/1536 per axis — well within gradient descent.
        #
        # Recommended training settings:
        #   batch_size=32 (4/chip), gradient_checkpointing=True,
        #   use_bf16=True, loss_chunk_size=128
        return DPSNRConfig(
            vocab_size=50304,              # 393 × 128, MXU-aligned (GPT-2 vocab)
            controller_hidden_dim=1024,
            controller_num_layers=24,
            controller_num_heads=16,
            controller_ff_multiplier=4.0,
            max_seq_len=8192,
            attn_window_size=256,   # 256/8192 = 3% local; pool handles the rest
            # 2D pool: 768 × 768 × 1024 ≈ 605M params (stored in bfloat16)
            # Reduced from 1536×1536 to fit pool-gradient (f32) on single device:
            #   1536²×1024×4 = 9.6 GB — exceeds 15.75 GB HBM when unsharded by GSPMD.
            #   768²×1024×4  = 2.4 GB — fits with room for params + activations.
            # Coordinate precision: 1/768 per axis — still fully learnable.
            pool_grid_rows=768,
            pool_grid_cols=768,
            pool_hidden_dim=1024,
            pool_total_vectors=768 * 768,  # 589,824 — kept for compatibility
            max_reasoning_loops=8,
            min_reasoning_loops=2,
            # Multi-head indexer: 4 concurrent pool queries per reasoning step
            num_indexer_heads=4,
            sigma_min=0.005,
            sigma_max=5.0,
            max_k=64,
            # Sigma annealing: broad → precise over 100k steps
            sigma_anneal_steps=100_000,
            sigma_target=0.02,
            precision_loss_weight=0.01,
            # Memory optimisations — all required for v5e-8
            gradient_checkpointing=True,
            use_bf16=True,
            loss_chunk_size=128,
            use_flash_attention=True,
            pool_super_window_factor=2,
            learning_rate=1e-4,
        )

    elif name == "mini_pool":
        # ── Pool-dominant design for custom 8K-vocab tokenizer ─────────────────
        #
        # Philosophy: keep the controller tiny (cheap to run, easy to train)
        # and push almost all capacity into the pool + a deep indexer that can
        # precisely address it.  The small 8 192-token vocabulary reduces the
        # embedding / LM-head cost so the controller stays under 100 M params
        # even at D=1024.
        #
        # Parameter breakdown:
        #   Controller  : D=1024, 6 layers, 8 heads  →  ~84 M
        #     Embedding : 8 192 × 1 024              =   8.4 M
        #     6 × layer : (4+8) × 1 024²             =  75.5 M (tied LM head: +0)
        #
        #   Indexer MLP : D_in=1024 → 10 240 → 5 120 → num_heads
        #     Dense(10240): 1 024 × 10 240 + 10 240  =  10.5 M
        #     Dense(5120) : 10 240 × 5 120 +  5 120  =  52.5 M
        #     Total                                  ≈  63 M   (> 50 M target)
        #
        #   Pool        : 1 024 × 1 024 × 1 024      =  1.07 B (> 1 B target)
        #     Stored bfloat16 → 2.15 GB HBM (sharded across 8 chips: 269 MB/chip)
        #
        # Per-chip HBM estimate (8 × TPU v5e chips, 16 GB each):
        #   Controller + Adam m/v (replicated) : ~1.0 GB/chip
        #   Indexer    + Adam m/v (replicated) : ~0.76 GB/chip
        #   Pool       + Adam m/v (sharded)    : ~0.81 GB/chip
        #   Static total                       : ~2.6 GB/chip
        #   Available for activations          : ~13 GB/chip  ← very comfortable
        #   → batch=16/chip (128 total) is achievable
        #
        # Training notes:
        #   - Train a BPE/Unigram tokenizer on your domain corpus first, target
        #     8 192 tokens (8192 = 64 × 128, MXU-aligned — no padding wasted).
        #   - use_bf16=True is required for the pool to stay in bfloat16.
        #   - loss_chunk_size=64 keeps the (B, T, 8192) logits tensor from
        #     ever being fully materialised on-chip.
        #   - attn_window_size=512 makes the controller O(T×512) instead of O(T²)
        #     at seq_len=4096; the pool handles all long-range recall.
        return DPSNRConfig(
            vocab_size=8192,               # 64 × 128, MXU-aligned for 8K tokenizer
            controller_hidden_dim=1024,
            controller_num_layers=6,
            controller_num_heads=8,        # head_dim = 128, MXU-aligned
            controller_ff_multiplier=4.0,
            max_seq_len=4096,
            attn_window_size=512,          # local attention; pool covers long-range
            dropout=0.0,
            # ── Indexer: deep MLP trunk independent of controller width ──────
            # indexer_hidden_dim=10240 gives ~63 M indexer params (> 50 M target)
            indexer_hidden_dim=10240,
            num_indexer_heads=8,           # 8 independent (µ_r, µ_c, σ) per step
            sigma_min=0.005,
            sigma_max=5.0,
            sigma_anneal_steps=200_000,  # anneal over 40% of a 500k run
            sigma_target=0.5,            # never get tighter than this — keeps indexer exploring for new data types
            precision_loss_weight=0.0,   # disabled: let indexer explore freely; enable after coverage >10%
            # ── Pool: 1024 × 1024 × 1024 = 1.07 B parameters ────────────────
            pool_grid_rows=1024,
            pool_grid_cols=1024,
            pool_hidden_dim=1024,          # kept for config compat; model uses controller_hidden_dim
            pool_total_vectors=1024 * 1024,
            max_k=64,
            max_reasoning_loops=6,
            min_reasoning_loops=2,
            # ── Memory & compute optimisations ────────────────────────────────
            gradient_checkpointing=True,
            use_bf16=True,
            use_flash_attention=True,
            loss_chunk_size=64,            # 8192-vocab × B × T in chunks of 64
            pool_super_window_factor=2,
            # ── Learning rate ─────────────────────────────────────────────────
            learning_rate=2e-4,
        )

    else:
        raise ValueError(f"Unknown config name: {name}")


def get_tiny_config():
    return get_model_config("tiny")
