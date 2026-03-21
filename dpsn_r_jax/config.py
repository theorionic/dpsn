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
    finetune: Optional[FineTuningConfig] = None

    # ── Precision Routing ──────────────────────────────────────────────────
    # 2D Grid Pool: instead of (N, D), pool is (rows, cols, D).
    # Precision per coordinate: 1/sqrt(N) instead of 1/N  → huge improvement.
    use_2d_pool: bool = False
    pool_grid_rows: int = 512   # rows × cols must equal pool_total_vectors
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

    # ── Gradient Checkpointing granularity ────────────────────────────────
    # Checkpoint every N controller layers instead of every layer.
    # N=1 (default): every layer is rematerialised — maximum memory savings,
    #   but causes 24 HBM read/write round-trips per backward pass.
    # N=4: only layers 0,4,8,… are checkpointed — 6× fewer activations saved,
    #   trading ~10% more peak memory for far less HBM traffic.
    # Set to 1 to revert to the original behaviour.
    controller_checkpoint_interval: int = 1

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
    # Set to 1 to disable. For 1D pools, 4-8 is good.
    # For 2D pools, recommended: 2 (since 2x means 2x2 = 4x the total vectors).
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
            pool_total_vectors=65536,  # 2^16 vectors
            pool_hidden_dim=512,
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
            pool_total_vectors=262144,  # 2^18 vectors (~200M params in pool)
            pool_hidden_dim=768,
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
            pool_total_vectors=1048576,  # 2^20 vectors (~1.1B params in pool)
            pool_hidden_dim=1024,
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
            pool_total_vectors=100,    # 10×10 grid when use_2d_pool=True
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
            use_2d_pool=True,
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
            controller_num_heads=12,
            max_seq_len=4096,
            pool_total_vectors=262144,
            pool_hidden_dim=768,
            max_reasoning_loops=6,
            learning_rate=3e-4,
            attn_window_size=512,
            num_indexer_heads=4,
            sigma_min=0.01,
            sigma_max=5.0,
            # 2D pool: 512×512 = 262 144 vectors
            # Each coordinate only needs 1/512 ≈ 0.2% precision (vs 1/262144 = 0.0004%)
            use_2d_pool=True,
            pool_grid_rows=512,
            pool_grid_cols=512,
            # Anneal sigma over first 50 000 steps: broad → precise routing
            sigma_anneal_steps=50_000,
            sigma_target=0.05,
            # Precision loss: small penalty on broad sigma
            precision_loss_weight=0.01,
            use_flash_attention=True,
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
            use_2d_pool=True,
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
            controller_checkpoint_interval=4,
            learning_rate=1e-4,
        )

    else:
        raise ValueError(f"Unknown config name: {name}")


def get_tiny_config():
    return get_model_config("tiny")
