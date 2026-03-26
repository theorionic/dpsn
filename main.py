import argparse
import json
import os
import signal
import sys
import time
import warnings

# Suppress annoying Kaggle/Colab PyTorch XLA vs TensorFlow warning
warnings.filterwarnings("ignore", message=".*tensorflow.*conflict with.*torch-xla.*")
# Suppress Transparent Hugepages warning
warnings.filterwarnings("ignore", message=".*Transparent hugepages are not enabled.*")
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import orbax.checkpoint
from flax.training import orbax_utils
import jax.numpy as jnp
import optax

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print(
        "For TensorBoard logging without PyTorch, install tensorboardX: pip install tensorboardX"
    )
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:

        class SummaryWriter:
            def __init__(self, log_dir=None):
                pass

            def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
                pass

            def close(self):
                pass


from dpsn_r_jax.config import DPSNRConfig, get_model_config
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.data.dataset import (
    ChunkedHFDataset,
    MultiprocessingHFDataset,
    SyntheticReasoningDataset,
    BackgroundGenerator,
)
from dpsn_r_jax.data.tokenizer import get_tokenizer
from dpsn_r_jax.data.grain_loader import (
    get_grain_loader,
    expand_npy_paths,
    get_single_npy_grain_loader,
    release_npy_loader,
)
from dpsn_r_jax.data.prefetch import DevicePrefetchIterator
from dpsn_r_jax.data.ram_cache import TokenizedRAMCache
from dpsn_r_jax.utils.generation import generate, generate_fast, clear_generation_cache
from dpsn_r_jax.utils.metrics import calculate_flops
from dpsn_r_jax.utils.memory_debug import print_tpu_memory, print_param_memory


def log_pool_utilization(state):
    touched_mask = jnp.any(state.pool_v > 0, axis=-1)
    num_touched = jnp.sum(touched_mask)
    total_vectors = touched_mask.size
    percentage = (num_touched / total_vectors) * 100
    print(
        f"Pool Utilization: {percentage:.2f}% ({int(num_touched)} / {total_vectors} vectors touched)"
    )
    return float(percentage)


def _grain_state_path(args) -> str:
    """Resolve the grain_state.json path (in checkpoint_dir if set)."""
    if args.checkpoint_dir:
        return os.path.join(os.path.abspath(args.checkpoint_dir), "grain_state.json")
    return os.path.abspath(args.resume_data_path)


def _save_grain_state(path: str, step: int, dataset) -> None:
    """Save data loader position to grain_state.json."""
    # Unwrap DevicePrefetchIterator → ChunkedHFDataset
    inner = getattr(dataset, 'data_source', dataset)
    if hasattr(inner, 'get_state'):
        state = inner.get_state()
    else:
        rows_consumed = getattr(inner, '_rows_consumed', 0)
        state = {"dataset_idx": 0, "sample_idx": int(rows_consumed), "rows_consumed": int(rows_consumed)}
    state["step"] = int(step)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[grain_state] Saved: step={int(step)}, rows_consumed={state.get('rows_consumed', '?')}")


def _load_grain_state(path: str) -> dict:
    """Load grain_state.json; returns {} if not found."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    # ── TPU Megacore: must be set before JAX initialises devices ─────────────
    import os as _os
    if jax.default_backend() == "tpu":
        _os.environ.setdefault("TPU_MEGACORE", "megacore")

    parser = argparse.ArgumentParser(description="Train DPSNR Model")
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny config for testing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xl", "precise_tiny", "precise_large", "xxl", "mini_pool"],
        help="Model configuration size (precise_* variants enable 2D pool + sigma annealing)",
    )
    parser.add_argument(
        "--sigma_anneal_steps",
        type=int,
        default=None,
        help="Override sigma annealing steps (0 = disabled; default from config)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--dataset_size", type=int, default=500, help="Dataset size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max training steps"
    )
    parser.add_argument(
        "--hf_dataset", type=str, default=None, help="HuggingFace dataset name (legacy)"
    )
    parser.add_argument(
        "--hf_datasets",
        nargs="+",
        default=None,
        help="List of HuggingFace dataset paths to stream sequentially",
    )
    parser.add_argument("--hf_subset", type=str, default=None, help="Dataset subset")
    parser.add_argument(
        "--hf_text_column",
        type=str,
        nargs="+",
        default=["text"],
        help="Column name for text content",
    )
    parser.add_argument(
        "--hf_tokenizer", type=str, default=None, help="HuggingFace tokenizer"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help=(
            "Download the HF dataset in fixed-size chunks of N rows (e.g. 10000). "
            "Each chunk is fully downloaded, tokenized with multiple processes, "
            "and shuffled in RAM before training; the next chunk is prefetched in "
            "the background so there is no stall at chunk boundaries. "
            "0 = disabled (uses the default streaming row-by-row mode)."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to dataset files/directories",
    )
    parser.add_argument(
        "--resume_data_path",
        type=str,
        default="grain_state.json",
        help="Path to save/load data loader state",
    )
    parser.add_argument(
        "--generation_steps", type=int, default=None, help="Generate text every N steps"
    )
    parser.add_argument(
        "--generation_max_tokens", type=int, default=20, help="Max tokens to generate"
    )
    # Checkpoint args
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--custom_prompts",
        nargs="+",
        default=None,
        help="Custom prompts for generation",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision (halves activation memory)",
    )
    parser.add_argument(
        "--loss_chunk_size",
        type=int,
        default=0,
        help="Chunk size for chunked cross-entropy loss (0=disabled). "
             "Avoids materialising full (B,T,V) logits; recommended: 16 for TPU v5e-8.",
    )
    parser.add_argument(
        "--num_kv_heads",
        type=int,
        default=0,
        help="GQA: number of KV heads shared across query heads (0 = full MHA). "
             "Must evenly divide --controller_num_heads. "
             "Typical: num_heads // 4 (e.g. 3 for precise_large with 12 heads).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size × grad_accum_steps. "
             "Use this to simulate large batches (e.g. 512) with a small physical batch "
             "(e.g. 64) that fits in HBM. batch_size must be divisible by grad_accum_steps.",
    )
    parser.add_argument(
        "--resume_data",
        action="store_true",
        help="Resume data loader from the checkpointed step",
    )
    parser.add_argument(
        "--skip_batches",
        type=int,
        default=0,
        help="Manually skip N batches of data",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
        ],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--ram_cache_gb",
        type=float,
        default=0,
        help="Pre-tokenize and cache this many GB in RAM before training (0=disabled)",
    )
    parser.add_argument(
        "--prefill_pct",
        type=float,
        default=0.1,
        help="Fraction of RAM cache to prefill before training starts (0.0-1.0)",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default=None,
        help="Directory to save detailed TensorBoard XLA traces. E.g., /tmp/tensorboard",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        nargs=2,
        default=[10, 20],
        help="Start step and end step for XLA profiling. Default: 10 20",
    )
    parser.add_argument(
        "--profile_detailed",
        action="store_true",
        help="Force synchronizations to print exact ms timings for Fetch, Dispatch, and TPU Execution every step.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Log metrics to TensorBoard and print to stdout every N steps (default: 50).",
    )
    parser.add_argument(
        "--timing_interval",
        type=int,
        default=0,
        help="Run forward_only_step every N steps to measure fwd/bwd time split (0=disabled). "
             "Loads a second JIT program — disable on memory-constrained configs like xxl.",
    )
    parser.add_argument(
        "--profile_components",
        action="store_true",
        help=(
            "Print per-component wall-clock timing breakdown every LOG_INTERVAL steps. "
            "Shows how the 1.3s step time is split across: "
            "TinyController / LearnedIndexer / CoordinateMassivePool / "
            "RetrievalIntegrator / AdaptiveComputeController / LM-head decode. "
            "Uses jax.debug.callback(ordered=True) so it works inside jit+lax.scan. "
            "Implies --profile_detailed (forces block_until_ready for accurate totals). "
            "Use for diagnosis only — adds ~1-2ms host overhead per step."
        ),
    )
    parser.add_argument(
        "--profile_components_interval",
        type=int,
        default=None,
        help="Print component breakdown every N steps (default: same as LOG_INTERVAL=200).",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help=(
            "Tensor-parallel size for pool feature sharding. "
            "Must divide device_count evenly. "
            "tp_size=1 (default): 1-D data-parallel mesh, original behaviour. "
            "tp_size=4 on v5e-8: 2×4 mesh (dp=2, tp=4) — pool features split "
            "4-way across TP chips, batch split 2-way across DP groups. "
            "Eliminates the 1.2 GB pool all-gather in the forward pass and "
            "reduces pool gradient all-reduce from 1.2 GB → 300 MB."
        ),
    )
    parser.add_argument(
        "--xla_cache_dir",
        type=str,
        default=None,
        help="Directory to persist JAX/XLA JIT compilation cache. "
             "First run compiles and saves artifacts; subsequent runs skip recompilation. "
             "Cache is hardware- and JAX-version-specific.",
    )
    parser.add_argument(
        "--prefetch_reasoning",
        action="store_true",
        help=(
            "Enable prefetch-once SRAM reasoning. "
            "Instead of fetching pool vectors from HBM on every reasoning loop "
            "iteration, this fetches a patch_size×patch_size region ONCE before "
            "the scan and passes it as a lax.scan carry so XLA keeps it in "
            "on-chip SRAM (~1 ns reads vs ~100 ns HBM). "
            "Each iteration then uses scaled dot-product attention over the "
            "SRAM-resident candidates instead of dynamic_slice from HBM. "
            "The retrieval_integrator and ACC modules are reused unchanged. "
            "SRAM cost: B_per_chip × prefetch_size² × D × 2 bytes "
            "(e.g. xxl/8-chip/B=32: 4 × 64² × 1024 × 2 = 33 MB per chip). "
            "Combine with --prefetch_size to control the candidate pool size."
        ),
    )
    parser.add_argument(
        "--prefetch_size",
        type=int,
        default=64,
        help=(
            "Per-axis size of the pre-fetched pool patch (total = prefetch_size²). "
            "Only used when --prefetch_reasoning is set. "
            "Recommended values: "
            "  64  → 4 096 candidates, ~33 MB/chip on xxl (safe default). "
            " 128  → 16 384 candidates, ~134 MB/chip on xxl (tight, use only on "
            "         smaller configs like base/large where D is smaller). "
            "Larger patch = broader pool coverage per step, higher SRAM use."
        ),
    )
    parser.add_argument(
        "--profile_model",
        action="store_true",
        help=(
            "Run a fine-grained wall-clock profiler on every model component before "
            "training starts. Prints a breakdown table showing how step time is split "
            "across: TinyController / LearnedIndexer / Pool retrieve / "
            "Retrieval integrator / ACC / LM head decoder / full forward / "
            "full train step. Also derives backward+optimizer time and lax.scan overhead. "
            "Works on v5e-8 (multi-device) via jax.block_until_ready — not ctimer. "
            "Each component is JIT-compiled separately and timed with warmup runs."
        ),
    )
    parser.add_argument(
        "--profile_model_runs",
        type=int,
        default=10,
        help=(
            "Number of timed runs per component when --profile_model is set. "
            "Median over this many runs is reported. "
            "3–5 is fast; 10+ gives more stable estimates. Default: 10."
        ),
    )
    parser.add_argument(
        "--profile_model_warmup",
        type=int,
        default=3,
        help=(
            "Number of warmup runs (compiled, not timed) before profiling each "
            "component. Ensures XLA has fully pipelined the computation. Default: 3."
        ),
    )
    parser.add_argument(
        "--pack_sequences",
        action="store_true",
        help=(
            "Enable sequence packing: bin-pack multiple variable-length sequences "
            "into single max_seq_len bins using first-fit-decreasing, then apply "
            "a block-diagonal causal attention mask so packed sequences cannot "
            "attend across boundaries. Improves TPU utilisation when training on "
            "short-sequence datasets. Requires the dataset to return varying-length "
            "sequences. Incompatible with --loss_chunk_size (chunked loss path) "
            "and splash attention (falls back to standard dot-product attention)."
        ),
    )

    args = parser.parse_args()

    # ── Component timing setup ───────────────────────────────────────────────
    # Import the singleton that dpsnr.py's jax.debug.callbacks write into.
    from dpsn_r_jax.utils.component_timer import ctimer as _ctimer
    if args.profile_components:
        _ctimer.enable()
        print(
            "[COMPONENT TIMER] Enabled — will print internal model timing every "
            f"{args.profile_components_interval or 'LOG_INTERVAL'} steps.\n"
            "  Marks captured: encode_start → controller → warmup → "
            "reasoning_loop×N(indexer→pool→integrator→acc) → decode\n"
            "  jax.debug.print fires inside lax.scan to show per-iter stats.\n"
        )

    # ── JAX/XLA Persistent Compilation Cache ────────────────────────────────
    if args.xla_cache_dir:
        os.makedirs(args.xla_cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", args.xla_cache_dir)
        print(f"[XLA CACHE] Persistent JIT cache ENABLED → {args.xla_cache_dir}")
        print(f"[XLA CACHE] First run: compiles + saves artifacts (slow). "
              f"Subsequent runs: loads from cache (fast).")
        print(f"[XLA CACHE] Cache is specific to this JAX version + hardware topology. "
              f"Clear the dir after upgrading JAX or changing device count.")
    else:
        print("[XLA CACHE] No --xla_cache_dir set — recompiling from scratch every run. "
              "Pass --xla_cache_dir /tmp/xla_cache to skip recompilation on restarts.")

    # Initialize TensorBoard writer
    log_dir = None
    if args.checkpoint_dir:
        log_dir = os.path.join(args.checkpoint_dir, "runs")
    writer = SummaryWriter(log_dir=log_dir)

    if args.tiny:
        print("Using TINY config (via flag)...")
        config = get_model_config("tiny")
    elif args.config:
        print(f"Using {args.config.upper()} config...")
        config = get_model_config(args.config)
    else:
        config = DPSNRConfig()

    if args.gradient_checkpointing:
        config.gradient_checkpointing = True

    if args.bf16:
        config.use_bf16 = True

    if args.prefetch_reasoning:
        config.prefetch_reasoning = True
        config.prefetch_size = args.prefetch_size
        _sram_mb = (args.prefetch_size ** 2 * config.controller_hidden_dim * 2) / 1e6
        _n_chips = jax.device_count()
        _b_per_chip = max(1, args.batch_size // _n_chips)
        _sram_per_chip_mb = _sram_mb * _b_per_chip
        print(
            f"[PREFETCH REASONING] Enabled\n"
            f"  Candidates  : {args.prefetch_size}×{args.prefetch_size} "
            f"= {args.prefetch_size**2:,} vectors\n"
            f"  SRAM/chip   : ~{_sram_per_chip_mb:.0f} MB  "
            f"(limit 128 MB — {'OK' if _sram_per_chip_mb < 100 else 'WARNING: tight'})\n"
            f"  HBM fetches : 1 per step  (was {config.max_reasoning_loops} per step)\n"
            f"  Loop style  : dot-product attention over SRAM candidates "
            f"(integrator + ACC reused)"
        )
        if _sram_per_chip_mb > 110:
            print(
                f"[PREFETCH REASONING] WARNING: {_sram_per_chip_mb:.0f} MB/chip "
                f"may exceed VMEM (128 MB). Consider --prefetch_size 32 or 48."
            )

    if args.loss_chunk_size > 0:
        config.loss_chunk_size = args.loss_chunk_size

    if args.num_kv_heads > 0:
        assert config.controller_num_heads % args.num_kv_heads == 0, (
            f"--num_kv_heads {args.num_kv_heads} must evenly divide "
            f"controller_num_heads {config.controller_num_heads}."
        )
        config.controller_num_kv_heads = args.num_kv_heads

    # Create device mesh - handles 1 to N devices automatically
    _tp_size = args.tp_size
    _n_dev   = jax.device_count()
    assert _n_dev % _tp_size == 0, (
        f"--tp_size {_tp_size} must divide device_count {_n_dev} evenly."
    )
    _dp_size = _n_dev // _tp_size

    if _tp_size == 1:
        # ── 1-D mesh (original behaviour) ────────────────────────────────────
        # Single "shard" axis used for data parallelism.
        # Pool params are row-sharded along "shard".
        devices = mesh_utils.create_device_mesh((_n_dev,))
        mesh = Mesh(devices, axis_names=("shard",))
        _dp_axis = "shard"   # batch sharding axis
        _tp_axis = "shard"   # pool feature sharding axis (same → row-shard)
        _pool_spec_fn = lambda ndim: PartitionSpec(*("shard",) + (None,) * (ndim - 1))
    else:
        # ── 2-D mesh (dp × tp) ───────────────────────────────────────────────
        # dp axis: data parallelism (batch split).
        # tp axis: tensor/feature parallelism for pool (feature-dim split).
        #
        # Example for v5e-8 with --tp_size 4:
        #   mesh shape (2, 4) → dp=2, tp=4
        #   batch=24 → 12 samples per dp group
        #   pool bf16[768,768,1024] → 768×768×256 per tp chip (300 MB vs 1.2 GB)
        #   pool forward: each chip slices its 256-feature sub-block — no all-gather
        #   pool grad all-reduce: 300 MB within dp group (vs 1.2 GB with 1-D mesh)
        devices = mesh_utils.create_device_mesh((_tp_size, _dp_size))
        mesh = Mesh(devices, axis_names=("tp", "dp"))
        _dp_axis = "dp"
        _tp_axis = "tp"
        _pool_spec_fn = lambda ndim: PartitionSpec(*((None,) * (ndim - 1) + ("tp",)))

    # Register the mesh so FlashCausalSelfAttention can wrap splash_attention
    # in shard_map for multi-device TPU runs (avoids GSPMD auto-partition error).
    from dpsn_r_jax.models.layers import set_mesh as _set_mesh
    _set_mesh(mesh)
    # Register the mesh in the kernels module so pool Pallas kernels can use
    # shard_map for TP multi-chip runs (each chip runs the kernel on its local
    # feature-sharded slice of the pool).
    from dpsn_r_jax.kernels import set_mesh as _set_kernels_mesh
    _set_kernels_mesh(mesh)

    # Sharding Rules:
    # 1. Batch: Split along 'shard' axis (Data Parallelism)
    # 2. Pool Params: Split along 'shard' axis (Model Parallelism)
    # 3. Other Params: Replicated (None)

    batch_sharding = NamedSharding(mesh, PartitionSpec(_dp_axis, None))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def get_sharding_rule(path, param):
        """
        Determines where a parameter should live based on its path in the PyTree.

        Strategy:
          tp_size=1 (1-D mesh, "shard"):
            Pool → row-sharded PartitionSpec("shard", None, None)
            Rest → replicated

          tp_size>1 (2-D mesh, "dp" × "tp"):
            Pool → feature-sharded PartitionSpec(None, None, "tp")
              Each chip holds (rows, cols, hidden_dim/tp_size).
              lax.dynamic_slice in memory.py uses params_storage.shape[-1]
              (the local feature count), so no all-gather is needed before
              the slice.  XLA GSPMD auto-inserts a small all-gather on the
              (B, local_D) pool output to assemble the full (B, hidden_dim).
              Pool gradient all-reduce: (rows, cols, hidden_dim/tp_size) within
              the dp group — tp_size× smaller than the 1-D mesh case.
            Rest → replicated (controller weights small enough to replicate)
        """
        if "pool" in path:
            if _tp_size == 1:
                # 1-D mesh: shard along first dim (row-sharding), guard divisibility
                if param.shape[0] % _n_dev == 0:
                    spec = PartitionSpec(*("shard",) + (None,) * (param.ndim - 1))
                else:
                    spec = PartitionSpec(*((None,) * param.ndim))
            else:
                # 2-D mesh: shard along LAST dim (feature-sharding) on tp axis
                # Guard: hidden_dim must be divisible by tp_size
                if param.shape[-1] % _tp_size == 0:
                    spec = _pool_spec_fn(param.ndim)
                else:
                    spec = PartitionSpec(*((None,) * param.ndim))  # fallback replicated
            return NamedSharding(mesh, spec)

        # Replicate everything else (Controller, Indexer, ACC, LayerNorm, biases…)
        return replicated_sharding

    print(f"Distributed Mesh: {mesh}")
    if _tp_size == 1:
        print(f"Sharding Strategy: 1-D mesh — Pool row-sharded, Rest replicated")
    else:
        print(
            f"Sharding Strategy: 2-D mesh dp={_dp_size}×tp={_tp_size} — "
            f"Pool feature-sharded (last dim / {_tp_size}), Rest replicated"
        )

    if args.hf_dataset:
        config.hf_dataset_name = args.hf_dataset
    if args.hf_tokenizer:
        config.hf_tokenizer_name = args.hf_tokenizer

    config.num_workers = args.num_workers

    if (
        args.max_steps is None
        and hasattr(config, "max_steps")
        and config.max_steps is not None
    ):
        print(f"Using max_steps from config: {config.max_steps}")
        args.max_steps = config.max_steps

    if args.generation_steps is not None:
        config.generation_steps = args.generation_steps
    config.generation_max_tokens = args.generation_max_tokens

    tokenizer_name = config.hf_tokenizer_name or "numeric"
    tokenizer = get_tokenizer(tokenizer_name)

    if hasattr(tokenizer, "vocab_size"):
        config.vocab_size = tokenizer.vocab_size
    elif hasattr(tokenizer, "__len__"):
        config.vocab_size = len(tokenizer)
        
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id

    # Initialize Model
    model = DPSNR(config)

    # Initialize State (Distributed)
    rng = jax.random.PRNGKey(0)

    # 1. Create abstract parameters (no memory usage)
    # We need a dummy input to trace the init function
    dummy_input = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)

    print("Initializing distributed model state...")

    # JIT-compile the initialization with the sharding constraints
    # This ensures parameters are created directly on the correct devices
    @jax.jit
    def init_model(rng, input_ids):
        print("Compiling init_model in main.py for XLA...", flush=True)
        return model.init(rng, input_ids)

    # Get abstract PyTree of variables (shapes/types only)
    abstract_variables = jax.eval_shape(init_model, rng, dummy_input)

    # Create a matching PyTree of Sharding objects
    sharding_tree = jax.tree_util.tree_map_with_path(
        get_sharding_rule, abstract_variables
    )

    # --- CHECKPOINT SETUP ---
    checkpoint_manager = None
    if args.checkpoint_dir:
        abs_checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(
            abs_checkpoint_dir, orbax.checkpoint.PyTreeCheckpointer(), options
        )

    # Initialize state with sharding constraints
    # We first create the raw variables distributedly
    variables = jax.lax.with_sharding_constraint(
        init_model(rng, dummy_input), sharding_tree
    )

    # Create TrainState (using the sharded variables)
    from dpsn_r_jax.training.trainer import TrainState, _make_sigma_anneal_fn
    from dpsn_r_jax.training.lr_schedules import get_scheduler
    from flax import traverse_util

    params = variables["params"]
    flat_params = traverse_util.flatten_dict(params)
    pool_key = ("pool", "params_storage")
    pool_params = flat_params[pool_key]
    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    # Calculate total training steps for LR schedule
    steps_per_epoch_calc = max(1, args.dataset_size // args.batch_size)
    total_steps = steps_per_epoch_calc * args.epochs
    if args.max_steps:
        total_steps = args.max_steps

    # Create learning rate schedule
    lr_schedule = get_scheduler(
        scheduler_type=args.lr_scheduler_type,
        learning_rate=config.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )

    # Initialize optimizer with the LR schedule so it decays properly each step
    # Gradient clipping prevents training instability from large gradient spikes
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule)
    )
    opt_state = tx.init(dense_params)

    # Allocate Adam moments for the pool directly sharded across chips.
    # We must NOT use jnp.zeros_like here: if pool_params somehow landed on a
    # single device (e.g. due to a PartitionSpec rank mismatch on older JAX),
    # zeros_like would try to create the full tensor (4.5 GB for xxl) on that
    # one chip and OOM.  Using jit+out_shardings forces XLA to allocate each
    # shard directly on its target device without ever forming the full array.
    _num_shards = jax.device_count()
    if _tp_size == 1:
        # 1-D mesh: moments sharded along first dim, same as pool params
        _moment_spec = (
            PartitionSpec(*("shard",) + (None,) * (pool_params.ndim - 1))
            if pool_params.shape[0] % _num_shards == 0
            else PartitionSpec(*((None,) * pool_params.ndim))
        )
    else:
        # 2-D mesh: moments sharded along last dim (tp axis), same as pool params
        _moment_spec = (
            _pool_spec_fn(pool_params.ndim)
            if pool_params.shape[-1] % _tp_size == 0
            else PartitionSpec(*((None,) * pool_params.ndim))
        )
    _pool_moment_sharding = NamedSharding(mesh, _moment_spec)
    _make_pool_zeros = jax.jit(
        lambda: jnp.zeros(pool_params.shape, dtype=pool_params.dtype),
        out_shardings=_pool_moment_sharding,
    )
    pool_m = _make_pool_zeros()
    pool_v = _make_pool_zeros()

    sigma_target_ratio = config.sigma_target / max(config.sigma_max, 1e-8)
    sigma_anneal_fn = _make_sigma_anneal_fn(
        getattr(config, 'sigma_anneal_steps', 0), sigma_target_ratio
    )

    state = TrainState(
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
    )

    # ── FIX DOUBLE COMPILATION ────────────────────────────────────────────────
    # JAX jit caches based on PyTree structure AND Sharding. `step`, `rng`, and
    # `opt_state.count` initialize as SingleDeviceSharding(CpuDevice(id=0)).
    # The first grad_accum_step returns them upgraded to NamedSharding(mesh).
    # Since Input != Output sharding, JAX forced a 20-minute recompile on Step 1.
    # We fix this by proactively promoting all SingleDevice arrays to the Mesh.
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    
    def bound_to_mesh(x):
        if hasattr(x, "sharding") and not isinstance(x.sharding, NamedSharding):
            return jax.device_put(x, replicated_sharding)
        return x

    state = jax.tree_util.tree_map(bound_to_mesh, state)


    # ── PRE-WARM JIT BEFORE CHECKPOINT RESTORE ───────────────────────────────
    # Problem: on resume, checkpoint_manager.restore() temporarily doubles HBM
    # usage (old random params + new checkpoint params = ~6.2 GB on mini_pool).
    # When the first train_step is called after restore, XLA tries to load the
    # compiled program (9.82 GB) at the bottom of HBM — but only 9.8 GB is
    # available (16 - 6.2), failing by ~20 MB.
    #
    # Fix: trigger JIT compilation NOW (with random params) before restore so
    # the XLA program claims the HBM bottom first. The checkpoint is then loaded
    # into the space above the compiled program. On subsequent training steps
    # the program is already loaded — no re-allocation needed.
    if args.resume and checkpoint_manager and checkpoint_manager.latest_step() is not None:
        print("Pre-warming JIT compilation before checkpoint restore (resume OOM fix)...")
        from dpsn_r_jax.training.trainer import train_step as _warmup_step
        _warmup_batch = jax.device_put(
            jnp.zeros((args.batch_size, config.max_seq_len), dtype=jnp.int32),
            batch_sharding,
        )
        _warmup_sigma = float(getattr(config, 'sigma_target', 0.05) / max(getattr(config, 'sigma_max', 5.0), 1e-8))
        _warmup_state, _, _, _ = _warmup_step(
            state, _warmup_batch, float(config.learning_rate), _warmup_sigma,
            config.pad_token_id,
            precision_loss_weight=0.0,
            sigma_anneal_steps=0,
            use_bf16=getattr(config, 'use_bf16', False),
            loss_chunk_size=getattr(config, 'loss_chunk_size', 0),
            prefetch_reasoning=getattr(config, 'prefetch_reasoning', False),
            prefetch_size=getattr(config, 'prefetch_size', 0),
        )
        del _warmup_batch, _warmup_state
        jax.effects_barrier()
        print("JIT pre-warm complete. Loading checkpoint...")

    # RESTORE CHECKPOINT IF REQUESTED
    if args.resume and checkpoint_manager:
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"Resuming from checkpoint step {latest_step}...")
            # We must pass the target 'state' so Orbax knows the sharding layout
            state = checkpoint_manager.restore(latest_step, items=state)
            global_step = latest_step
        else:
            # Fallback for direct directory path
            abs_checkpoint_dir = os.path.abspath(args.checkpoint_dir)
            target_path = None
            if os.path.exists(os.path.join(abs_checkpoint_dir, "default")):
                target_path = os.path.join(abs_checkpoint_dir, "default")
            elif os.path.exists(os.path.join(abs_checkpoint_dir, "_METADATA")):
                target_path = abs_checkpoint_dir

            if target_path:
                print(f"Resuming directly from checkpoint path: {target_path}")
                state = orbax.checkpoint.PyTreeCheckpointer().restore(
                    target_path, items=state
                )
                # Try to extract step from path if possible
                try:
                    step_str = os.path.basename(
                        os.path.dirname(target_path)
                        if target_path.endswith("default")
                        else target_path
                    )
                    global_step = jnp.array(int(step_str), dtype=jnp.int32)
                except ValueError:
                    global_step = jnp.array(0, dtype=jnp.int32)
            else:
                print("No checkpoint found to resume from. Starting from scratch.")
                global_step = jnp.array(0, dtype=jnp.int32)

    else:
        global_step = jnp.array(0, dtype=jnp.int32)

    # Data Loader Initialization
    if args.skip_batches > 0:
        loader_start_step = args.skip_batches
    elif args.resume_data:
        loader_start_step = global_step
    else:
        loader_start_step = 0

    # Resolve grain_state path and load if resuming data position
    _grain_state_file = _grain_state_path(args)
    _grain_skip_rows = 0
    _grain_hf_state = None
    if args.resume_data:
        _gs = _load_grain_state(_grain_state_file)
        if _gs:
            _grain_hf_state = _gs.get("hf_state")         # O(1) seek via state_dict
            _grain_skip_rows = _gs.get("rows_consumed", _gs.get("sample_idx", 0))
            method = "hf_state (O(1) seek)" if _grain_hf_state else "skip_rows (row replay)"
            print(
                f"[grain_state] Loaded: step={_gs.get('step', '?')}, "
                f"rows_consumed={_grain_skip_rows:,}, method={method}"
            )
        else:
            print("[grain_state] No grain_state.json found — data will start from the beginning.")

    # ── Detect sequential NPY mode ─────────────────────────────────────────
    npy_files = expand_npy_paths(args.dataset_path) if args.dataset_path else []
    use_sequential_npy = len(npy_files) > 0

    if use_sequential_npy:
        print(f"\nSequential NPY mode: {len(npy_files)} files detected.")
        print(f"Files will be loaded ONE AT A TIME to minimize RAM usage.")
        for i, f in enumerate(npy_files):
            print(f"  [{i+1}/{len(npy_files)}] {os.path.basename(f)}")
    else:
        # Resolve the primary dataset name: --hf_dataset takes precedence,
        # falling back to the first entry in --hf_datasets.
        primary_hf = args.hf_dataset or (args.hf_datasets[0] if getattr(args, "hf_datasets", None) else None)

        if getattr(args, "chunk_size", 0) > 0 and primary_hf:
            # ── Chunk-based mode (recommended for TPU): ──────────────────────
            # Downloads `chunk_size` rows at once via the HF streaming iterator,
            # tokenizes them in parallel across `num_workers` CPU cores,
            # shuffles the whole chunk in RAM, then serves batches at memory
            # speed.  A background thread keeps the next chunk ready so there
            # is zero training stall at chunk boundaries.
            print(
                f"Loading HF dataset (CHUNK mode): '{primary_hf}' | "
                f"chunk_size={args.chunk_size:,} rows | "
                f"{args.num_workers} tokenizer workers"
            )
            dataset = ChunkedHFDataset(
                dataset_name=primary_hf,
                tokenizer_name=tokenizer_name,
                chunk_size=args.chunk_size,
                subset=args.hf_subset,
                split="train",
                seq_len=config.max_seq_len,
                batch_size=args.batch_size,
                num_tokenizer_workers=args.num_workers,
                text_columns=args.hf_text_column or None,
                hf_state=_grain_hf_state,       # O(1) seek if available
                skip_rows=_grain_skip_rows,      # row-replay fallback
            )
        else:
            # Fallback: original loader path for non-chunked HF or non-NPY datasets
            grain_loader = get_grain_loader(
                args.dataset_path, args, start_step=loader_start_step
            )

            if grain_loader:
                print(f"Using Google Grain data loader (start_step={loader_start_step}).")

                class GrainWrapper:
                    def __init__(self, loader):
                        self.loader = loader
                        self.iterator = iter(loader)

                    def get_batch(self, batch_size=None):
                        try:
                            batch = next(self.iterator)
                        except StopIteration:
                            self.iterator = iter(self.loader)
                            batch = next(self.iterator)
                        return batch["input_ids"]

                dataset = GrainWrapper(grain_loader)
            elif primary_hf:
                # ── Legacy streaming mode: row-by-row, N parallel worker processes ──
                print(
                    f"Loading HF dataset (streaming mode): '{primary_hf}' "
                    f"(subset: {args.hf_subset}) with {args.num_workers} workers"
                )
                dataset = MultiprocessingHFDataset(
                    dataset_name=primary_hf,
                    tokenizer_name=tokenizer_name,
                    subset=args.hf_subset,
                    seq_len=config.max_seq_len,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    prefetch_batches=100,
                )
            else:
                print("Generating synthetic sorting dataset...")
                dataset = SyntheticReasoningDataset(
                    size=args.dataset_size, seq_len=config.max_seq_len
                )

    def count_params(tree):
        return sum(x.size for x in jax.tree_util.tree_leaves(tree))

    p = state.params
    breakdown = {
        "TinyController (CEO)": count_params(p["controller"]),
        "LearnedIndexer (Archivist)": count_params(p["indexer"]),
        "CoordinateMassivePool (Library)": count_params(p["pool"]),
        "ReasoningEngine": count_params(p["acc"])
        + count_params(p["retrieval_integrator"]),
    }
    total_params = count_params(p)

    print("\n" + "=" * 50)
    print(f"{'Component':<35} | {'Parameters':>12}")
    print("-" * 50)
    for name, size in breakdown.items():
        print(f"{name:<35} | {size:>12,}")
    print("-" * 50)
    print(f"{'Total Parameters':<35} | {total_params:>12,}")
    print("=" * 50 + "\n")

    # ── Memory breakdown (static: params + optimizer + activation estimates) ──
    print_param_memory(state, config, args.batch_size,
                       loss_chunk_size=getattr(config, 'loss_chunk_size', 0))
    print_tpu_memory("after model init (before first train_step compile)")

    from dpsn_r_jax.training.trainer import train_step, grad_accum_step, forward_only_step
    from dpsn_r_jax.utils.metrics import summarise_flops, roofline_metrics

    # ── Choose training function based on gradient accumulation ──────────────
    _grad_accum = getattr(args, "grad_accum_steps", 1)
    if _grad_accum > 1:
        assert args.batch_size % _grad_accum == 0, (
            f"--batch_size ({args.batch_size}) must be divisible by "
            f"--grad_accum_steps ({_grad_accum})"
        )
        _micro_batch = args.batch_size // _grad_accum
        print(
            f"Gradient accumulation ENABLED: "
            f"micro_batch={_micro_batch} × accum={_grad_accum} "
            f"= effective batch {args.batch_size}"
        )
        distributed_train_step = None   # not used when accumulating
    else:
        _micro_batch = args.batch_size
        distributed_train_step = train_step

    # ── Sequence packing collator (--pack_sequences) ─────────────────────────
    if args.pack_sequences:
        from dpsn_r_jax.data.packing_collator import PackingCollator
        _packing_collator = PackingCollator(
            max_seq_len=config.max_seq_len,
            pad_token_id=config.pad_token_id,
        )
        print(
            f"[PACK SEQUENCES] Enabled — first-fit-decreasing bin packing, "
            f"max_seq_len={config.max_seq_len}, pad_token_id={config.pad_token_id}. "
            f"Block-diagonal causal mask applied (splash attention disabled for packed batches)."
        )
    else:
        _packing_collator = None

    flops_per_step = calculate_flops(config, args.batch_size)
    summarise_flops(config, args.batch_size)  # print breakdown once at startup

    # ── Model component profiler (--profile_model) ───────────────────────────
    # Runs BEFORE the training loop so it doesn't interfere with step timing.
    # Uses synthetic zero batches — timing is independent of actual token values.
    # Works on v5e-8 (multi-device): uses jax.block_until_ready, NOT ctimer.
    if args.profile_model:
        from dpsn_r_jax.utils.model_profiler import run_model_profile
        _profile_batch = jax.device_put(
            jnp.zeros((args.batch_size, config.max_seq_len), dtype=jnp.int32),
            batch_sharding,
        )
        print(
            f"\n[MODEL PROFILER] --profile_model enabled\n"
            f"  warmup={args.profile_model_warmup} runs, "
            f"timed={args.profile_model_runs} runs per component\n"
            f"  Batch: {args.batch_size} × {config.max_seq_len} (synthetic zeros)\n"
            f"  NOTE: first component will trigger JIT compilation — "
            f"subsequent ones use the compiled cache.\n"
        )
        run_model_profile(
            model=model,
            state=state,
            config=config,
            sample_batch=_profile_batch,
            batch_sharding=batch_sharding,
            replicated_sharding=replicated_sharding,
            warmup=args.profile_model_warmup,
            runs=args.profile_model_runs,
            step=0,
        )
        print("[MODEL PROFILER] Done. Proceeding to training...\n")

    # For infinite HF streaming/chunked datasets, steps_per_epoch from
    # dataset_size is meaningless.  Use max_steps as the epoch length so the
    # training loop runs until the step cap, otherwise fall back to the size
    # estimate (for local .npy and synthetic datasets).
    is_hf_streaming = (
        not use_sequential_npy
        and (args.hf_dataset or args.hf_datasets)
    )
    if is_hf_streaming and args.max_steps:
        steps_per_epoch = args.max_steps
    else:
        steps_per_epoch = max(1, args.dataset_size // args.batch_size)

    # Define test samples for generation
    test_samples = ["Sort: 5 2 8 1 ->", "Sort: 10 3 7 ->", "Sort: 1 1 1 ->"]

    # ── Build data pipeline (non-sequential path) ──────────────────────────
    # Track whether the dataset already manages its own background prefetch
    # so we don't double-wrap with BackgroundGenerator.
    _dataset_is_chunked = isinstance(dataset, ChunkedHFDataset) if not use_sequential_npy else False

    if not use_sequential_npy:
        if args.ram_cache_gb > 0:
            cache_source = dataset
            hf_name = None

            if args.hf_datasets:
                hf_name = args.hf_datasets[0]
            elif args.hf_dataset:
                hf_name = args.hf_dataset

            if hf_name:
                cache_workers = max(16, args.num_workers * 4)
                print(f"\nUsing {cache_workers} parallel workers for fast HF cache fill "
                      f"(dataset: {hf_name})...")
                cache_source = MultiprocessingHFDataset(
                    dataset_name=hf_name,
                    tokenizer_name=tokenizer_name,
                    subset=args.hf_subset,
                    seq_len=config.max_seq_len,
                    batch_size=args.batch_size,
                    num_workers=cache_workers,
                    prefetch_batches=500,
                )

            print(f"\nEnabling RAM Cache: {args.ram_cache_gb:.1f} GB, "
                  f"prefill {args.prefill_pct*100:.0f}% before training")
            dataset = TokenizedRAMCache(
                data_source=cache_source,
                batch_size=args.batch_size,
                seq_len=config.max_seq_len,
                cache_size_gb=args.ram_cache_gb,
                prefill_pct=args.prefill_pct,
            )
            _dataset_is_chunked = False  # TokenizedRAMCache doesn't self-prefetch

        elif not _dataset_is_chunked:
            # Only wrap non-self-prefetching datasets.  ChunkedHFDataset already
            # has its own background thread — adding BackgroundGenerator on top
            # would add an extra queue hop and ~1 ms of latency per batch.
            dataset = BackgroundGenerator(dataset, args.batch_size, prefetch_size=50)

        else:
            print(
                "[ChunkedHFDataset] Skipping BackgroundGenerator wrap — "
                "ChunkedHFDataset self-prefetches via its own background thread."
            )

        # ChunkedHFDataset serves batches from RAM; use a deeper on-device prefetch
        # buffer so XLA always has the next batch staged on TPU HBM.
        _prefetch_depth = 4 if _dataset_is_chunked else 2
        dataset = DevicePrefetchIterator(
            data_source=dataset,
            batch_size=args.batch_size,
            sharding=batch_sharding,
            prefetch_depth=_prefetch_depth,
        )

    LOG_INTERVAL = args.log_interval  # sync + log every N steps; lower = more block_until_ready stalls
    # Component timing: run forward_only_step every TIMING_INTERVAL steps and
    # compare to the avg step time to estimate forward vs backward+optimizer split.
    # Set to 0 to disable.  First run is skipped (includes JIT compilation).
    # NOTE: forward_only_step is a separate JIT program — on memory-constrained
    # configs (e.g. xxl on v5e-8) it will OOM when loaded alongside grad_accum_step.
    # Use --timing_interval 0 (default) to disable.
    TIMING_INTERVAL = args.timing_interval if args.timing_interval > 0 else 0

    tokens_per_sec = 0.0
    tflops = 0.0
    total_data_wait_time_interval = 0.0

    # ── Helper: run training steps on a data pipeline ──────────────────────
    def _run_training_steps(dataset_pipeline, state, global_step, epoch,
                            steps_per_epoch, start_time, total_data_wait_time_interval,
                            file_label=""):
        """Run training steps using the given data pipeline.
        Returns (state, global_step, epoch_loss, start_time, total_data_wait_time_interval, hit_max_steps).
        """
        # Bug #1 Fix: accumulate loss as a JAX future (no blocking DtH transfer).
        # We only call float() / block_until_ready() inside the LOG_INTERVAL block,
        # keeping the TPU pipeline bubble-free between logging events.
        epoch_loss = jnp.zeros((), dtype=jnp.float32)
        last_loss = jnp.zeros((), dtype=jnp.float32)       # holds the last step's loss future
        last_sigma = jnp.zeros((), dtype=jnp.float32)      # holds the last step's mean_sigma future
        last_grad_norm = jnp.zeros((), dtype=jnp.float32)  # holds the last step's pool grad norm
        hit_max_steps = False
        last_batch = None          # saved for component timing
        _timing_compiled = False   # skip first timing call (includes JIT compile)

        for step in range(steps_per_epoch):
            if _stop_requested[0]:
                break

            if args.max_steps and global_step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}). Stopping training.")
                hit_max_steps = True
                break

            if args.profile_dir and global_step == args.profile_steps[0]:
                print(f"\n[PROFILER] 🟢 STARTING JAX XLA PROFILER AT STEP {global_step} 🟢")
                jax.profiler.start_trace(args.profile_dir)

            data_start_time = time.time()
            try:
                batch = dataset_pipeline.get_batch(args.batch_size)
            except (StopIteration, Exception) as e:
                if isinstance(e, StopIteration) or "StopIteration" in str(type(e).__name__):
                    break  # This file is exhausted
                raise
            current_data_wait_time = time.time() - data_start_time
            total_data_wait_time_interval += current_data_wait_time

            # ── Sequence packing (--pack_sequences) ────────────────────────────
            # PackingCollator returns packed_ids and seq_pack_ids. The latter is
            # placed on device as a (T,) int32 array and passed to train_step so
            # the model applies a block-diagonal causal attention mask.
            # Note: batch from pipeline is already (B, T); packing re-bins rows.
            _seq_pack_ids = None
            if args.pack_sequences:
                import numpy as _np
                _rows = [_np.array(batch[i]) for i in range(batch.shape[0])]
                _packed, _spi = _packing_collator(_rows)
                # _packed: (B_packed, T), _spi: (B_packed, T)
                batch = jax.device_put(
                    jnp.array(_packed, dtype=jnp.int32), batch_sharding
                )
                # seq_pack_ids is (T,) — broadcast over batch; replicated.
                _seq_pack_ids = jax.device_put(
                    jnp.array(_spi[0], dtype=jnp.int32), replicated_sharding
                )

            last_batch = batch   # save for component timing
            dispatch_start_time = time.time()

            # .astype() stays on-device — no D2H sync.
            # jnp.float32(jax_scalar) calls __float__ → np.asarray → D2H stall.
            current_lr_val = lr_schedule(global_step).astype(jnp.float32)
            sigma_scale_val = sigma_anneal_fn(global_step).astype(jnp.float32)

            if _grad_accum > 1:
                # ── Gradient accumulation path ────────────────────────────────
                # Caller splits (B, T) batch into (_grad_accum, micro_B, T) and
                # hands it to the JIT-compiled grad_accum_step.
                micro_batch_size = args.batch_size // _grad_accum
                # Stack micro-batches: (accum, micro_B, T)
                # batch arrives as (B, T) sharded along the dp axis.
                # After reshape to (accum, micro_B, T), dim 0 = accum_steps (e.g. 4)
                # which is smaller than device count (e.g. 8) → IndivisibleError.
                # Fix: explicitly reshard so the dp axis moves to dim 1
                # (the micro-batch dim).  micro_batch_size must be divisible by
                # dp device count; the assertion below catches mis-configurations.
                assert micro_batch_size % jax.device_count() == 0, (
                    f"micro_batch_size ({micro_batch_size}) must be divisible by "
                    f"device count ({jax.device_count()}). "
                    f"Increase --batch_size or decrease --grad_accum_steps."
                )
                # batch arrives sharded along the dp axis.
                # .reshape() propagates that sharding to the output — dim 0
                # becomes accum_steps (e.g. 4) which is < device_count (8)
                # → IndivisibleError even before device_put runs.
                # Fix: strip the dp axis first (replicate), then reshape,
                # then place the dp shard on dim 1 (micro_batch_size must be
                # divisible by dp device count).
                _batch_rep = jax.device_put(batch, replicated_sharding)
                micro_batches = jax.device_put(
                    _batch_rep.reshape(_grad_accum, micro_batch_size, config.max_seq_len),
                    NamedSharding(mesh, PartitionSpec(None, _dp_axis, None)),
                )
                state, loss, mean_sigma, pool_grad_norm = grad_accum_step(
                    state, micro_batches, current_lr_val, sigma_scale_val,
                    pad_token_id=config.pad_token_id,
                    precision_loss_weight=getattr(config, 'precision_loss_weight', 0.0),
                    sigma_anneal_steps=getattr(config, 'sigma_anneal_steps', 0),
                    use_bf16=getattr(config, 'use_bf16', False),
                    loss_chunk_size=getattr(config, 'loss_chunk_size', 0),
                    grad_accum_steps=_grad_accum,
                    prefetch_reasoning=getattr(config, 'prefetch_reasoning', False),
                    prefetch_size=getattr(config, 'prefetch_size', 0),
                    seq_pack_ids=_seq_pack_ids,
                )
            else:
                # ── Standard single-step path ─────────────────────────────────
                state, loss, mean_sigma, pool_grad_norm = distributed_train_step(
                    state, batch, current_lr_val, sigma_scale_val, config.pad_token_id,
                    precision_loss_weight=getattr(config, 'precision_loss_weight', 0.0),
                    sigma_anneal_steps=getattr(config, 'sigma_anneal_steps', 0),
                    use_bf16=getattr(config, 'use_bf16', False),
                    loss_chunk_size=getattr(config, 'loss_chunk_size', 0),
                    prefetch_reasoning=getattr(config, 'prefetch_reasoning', False),
                    prefetch_size=getattr(config, 'prefetch_size', 0),
                    seq_pack_ids=_seq_pack_ids,
                )

            # Bug #1 Fix: append JAX futures — NO float() / .item() here!
            # These are enqueued as async device operations; the TPU keeps running.
            epoch_loss     = epoch_loss + loss
            last_loss      = loss
            last_sigma     = mean_sigma
            last_grad_norm = pool_grad_norm
            dispatch_time = time.time() - dispatch_start_time

            # ── Detailed per-step timing breakdown (--profile_detailed) ──────
            # Forces a host/device sync every step — this WILL reduce throughput
            # but gives exact latency for every phase. Use only for diagnosis.
            #
            # Phases:
            #   DataFetch    = time blocked waiting for the data pipeline to
            #                  return a batch (CPU workers / prefetch queue).
            #   HostDispatch = time for Python/XLA to trace & enqueue the JIT
            #                  computation on the accelerator (should be ~1-5ms;
            #                  high values indicate Python overhead or re-tracing).
            #   TPU-Exec     = actual accelerator compute time measured by
            #                  blocking until the loss scalar is ready.  Low
            #                  TFLOPS usually means this is dominated by memory
            #                  bandwidth (pool scatter/gather) rather than FLOPs.
            if args.profile_detailed:
                _sync_t0 = time.time()
                jax.block_until_ready(loss)
                tpu_exec_ms = (time.time() - _sync_t0) * 1000.0

                data_ms     = current_data_wait_time * 1000.0
                dispatch_ms = dispatch_time          * 1000.0
                total_ms    = data_ms + dispatch_ms + tpu_exec_ms
                _step_tag   = int(global_step) + 1  # +1: global_step increments below

                bottleneck = "DATA" if data_ms > tpu_exec_ms else "COMPUTE"
                print(
                    f"[STEP {_step_tag:>6}] "
                    f"DataFetch={data_ms:7.2f}ms  "
                    f"HostDispatch={dispatch_ms:6.2f}ms  "
                    f"TPU-Exec={tpu_exec_ms:8.2f}ms  "
                    f"StepTotal={total_ms:8.2f}ms  "
                    f"Loss={float(loss):.4f}  "
                    f"[bottleneck={bottleneck}]",
                    flush=True,
                )

            global_step += 1

            if args.profile_dir and global_step == args.profile_steps[1]:
                print(f"\n[PROFILER] 🛑 STOPPING JAX XLA PROFILER AT STEP {global_step} 🛑")
                jax.profiler.stop_trace()
                print(f"Profile saved to {args.profile_dir}. View using: tensorboard --logdir={args.profile_dir}\n")

            # Save Checkpoint
            if (
                checkpoint_manager
                and args.save_interval
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving checkpoint at step {global_step}...")
                checkpoint_manager.save(global_step, state)
                _save_grain_state(_grain_state_file, global_step, dataset)

            if step % LOG_INTERVAL == 0:
                # Bug #1 Fix: ONE blocking sync per LOG_INTERVAL steps.
                # block_until_ready() stalls the host until the TPU has
                # finished this step, then we do ALL float() conversions
                # at once.  Between logging events the TPU runs freely.
                last_loss.block_until_ready()
                current_time = time.time()
                elapsed = current_time - start_time

                steps_in_interval = LOG_INTERVAL if step > 0 else 1
                avg_step_time = elapsed / steps_in_interval
                avg_data_wait = total_data_wait_time_interval / steps_in_interval

                active_tpu_time = max(0.0001, avg_step_time - avg_data_wait)

                tokens_per_sec  = (args.batch_size * config.max_seq_len) / avg_step_time
                steps_per_sec   = 1.0 / avg_step_time
                tflops = flops_per_step / active_tpu_time / 1e12

                # All float() calls happen AFTER block_until_ready — single sync point
                loss_val  = float(last_loss)
                sigma_val = float(last_sigma)
                ppl_val   = float(jnp.exp(last_loss))
                sigma_scale_val = float(sigma_anneal_fn(global_step))
                current_lr = float(lr_schedule(global_step))
                grad_norm_val = float(last_grad_norm)

                writer.add_scalar("Loss/train", loss_val, global_step)
                writer.add_scalar("PPL/train", ppl_val, global_step)
                writer.add_scalar("LR", current_lr, global_step)
                writer.add_scalar("Routing/mean_sigma", sigma_val, global_step)
                writer.add_scalar("Routing/sigma_scale", sigma_scale_val, global_step)
                writer.add_scalar("Perf/TPS", tokens_per_sec, global_step)
                writer.add_scalar("Perf/SPS", steps_per_sec, global_step)
                writer.add_scalar("Perf/TFLOPS", tflops, global_step)
                writer.add_scalar("Perf/DataWaitTime_s", avg_data_wait, global_step)
                writer.add_scalar("GradNorm/pool", grad_norm_val, global_step)

                # TPU memory utilization — aggregate across all devices
                _mem_in_use = _mem_limit = 0
                for _dev in jax.devices():
                    try:
                        _ms = _dev.memory_stats()
                        _mem_in_use += _ms.get("bytes_in_use", 0)
                        _mem_limit   += _ms.get("bytes_limit", 0)
                    except Exception:
                        pass
                if _mem_limit > 0:
                    writer.add_scalar("HW/TPU_mem_pct", _mem_in_use / _mem_limit * 100, global_step)
                    writer.add_scalar("HW/TPU_mem_GB",  _mem_in_use / 1e9, global_step)

                precision_tag = "broad" if sigma_val > 1.0 else ("precise" if sigma_val < 0.1 else "narrowing")
                print(
                    f"{file_label}Epoch {epoch + 1} | Step {step} | Global Step {global_step} | "
                    f"Loss: {loss_val:.4f} | PPL: {ppl_val:.4f} | LR: {current_lr:.2e} | "
                    f"sigma={sigma_val:.3f} ({precision_tag}) | GradNorm: {grad_norm_val:.4f} | "
                    f"TPS: {tokens_per_sec:.0f} | SPS: {steps_per_sec:.3f} | "
                    f"TFLOPS: {tflops:.4f} | DataWait: {avg_data_wait:.3f}s"
                )

                # ── Roofline: MXU compute utilisation vs HBM bandwidth pressure ───
                # MFU  = actual_TFLOPS / peak_TFLOPS  → how busy the MXU is
                # ~MBU = estimated_bytes / (step_time × peak_HBM_BW) → how hard
                #        HBM is working.  If MBU >> MFU the MXU is stalling waiting
                #        for data.  If both are low, the bottleneck is latency /
                #        sequential kernel scheduling, not raw throughput.
                # NOTE: MBU is a structural estimate from param counts — not a
                #        hardware counter.  Treat it as a directional indicator.
                try:
                    _n_chips  = jax.device_count()
                    _tp_size  = getattr(args, 'tp_size', 1)
                    _gc       = getattr(args, 'gradient_checkpointing', False)
                    _rl = roofline_metrics(
                        config, args.batch_size, _n_chips, _tp_size,
                        avg_step_time, tflops,
                        gradient_checkpointing=_gc,
                        tpu_gen="v5e",
                    )
                    _mfu_pct  = _rl["mfu"]  * 100
                    _mbu_pct  = _rl["mbu"]  * 100
                    _btn      = _rl["bottleneck"]
                    _bw_GBs   = _rl["bw_GB_s"]
                    _ideal_ms = _rl["ideal_ms"]
                    _stall_ms = _rl["stall_ms"]
                    print(
                        f"[ROOFLINE] MFU: {_mfu_pct:.1f}% | ~MBU: {_mbu_pct:.1f}% | "
                        f"Bottleneck: {_btn} | "
                        f"~HBM: {_bw_GBs:.0f} GB/s of {819*_n_chips:.0f} GB/s peak | "
                        f"Ideal: {_ideal_ms:.0f}ms | Stall: {_stall_ms:.0f}ms"
                    )
                    writer.add_scalar("Roofline/MFU_pct",       _mfu_pct,  global_step)
                    writer.add_scalar("Roofline/MBU_pct_est",   _mbu_pct,  global_step)
                    writer.add_scalar("Roofline/HBM_GB_s_est",  _bw_GBs,   global_step)
                    writer.add_scalar("Roofline/Stall_ms_est",  _stall_ms, global_step)
                except Exception as _e:
                    pass  # never crash training over a metric

            # ── Component timing: forward vs backward+optimizer split ─────────
            # Runs a no-grad forward pass (forward_only_step) every TIMING_INTERVAL
            # steps, syncs with block_until_ready, and compares to the avg full-step
            # time to estimate the forward vs backward+optimizer time split.
            # First invocation is skipped because it includes JIT compile time.
            if (
                TIMING_INTERVAL > 0
                and last_batch is not None
                and global_step > 0
                and global_step % TIMING_INTERVAL == 0
            ):
                _sigma_for_timing = jnp.float32(sigma_anneal_fn(global_step))
                _fwd_t0 = time.time()
                _fwd_out, _ = forward_only_step(
                    state, last_batch,
                    sigma_scale=_sigma_for_timing,
                    use_bf16=getattr(config, 'use_bf16', False),
                    loss_chunk_size=getattr(config, 'loss_chunk_size', 0),
                )
                jax.block_until_ready(_fwd_out)
                _fwd_ms = (time.time() - _fwd_t0) * 1000.0

                if not _timing_compiled:
                    # First call includes JIT compilation — record but label clearly
                    _timing_compiled = True
                    print(
                        f"\n[TIMING] Step {global_step} | First timing call includes JIT compile "
                        f"— forward_only compile+exec: {_fwd_ms:.1f}ms (not representative)"
                    )
                else:
                    # avg_step_time is in seconds (computed earlier in this block)
                    _total_ms = avg_step_time * 1000.0
                    _bwd_opt_ms = max(0.0, _total_ms - _fwd_ms)
                    _fwd_pct   = 100.0 * _fwd_ms   / (_total_ms + 1e-6)
                    _bwd_pct   = 100.0 * _bwd_opt_ms / (_total_ms + 1e-6)
                    print(
                        f"\n[TIMING] Step {global_step} Component Breakdown\n"
                        f"  Forward  (controller+reasoning+decode): {_fwd_ms:7.1f}ms  ({_fwd_pct:.0f}%)\n"
                        f"  Backward + Optimizer (est.):            {_bwd_opt_ms:7.1f}ms  ({_bwd_pct:.0f}%)\n"
                        f"  Total step (avg over {LOG_INTERVAL} steps):       {_total_ms:7.1f}ms\n"
                        f"  [Use --profile_dir to get per-layer XLA trace for "
                        f"controller/reasoning/pool breakdown]"
                    )
                    writer.add_scalar("Timing/Forward_ms",          _fwd_ms,    global_step)
                    writer.add_scalar("Timing/Backward_Optim_ms",   _bwd_opt_ms, global_step)
                    writer.add_scalar("Timing/Total_step_ms",        _total_ms,  global_step)
                    writer.add_scalar("Timing/Forward_pct",          _fwd_pct,   global_step)

            # ── Internal component timing (--profile_components) ─────────────
            # Prints the per-stage breakdown captured by jax.debug.callback marks
            # inside dpsnr.py: controller → warmup → reasoning_loop×N
            # (indexer → pool → integrator → acc) → decode.
            # block_until_ready() ensures all callbacks have fired before we print.
            _comp_interval = args.profile_components_interval or LOG_INTERVAL
            if (
                args.profile_components
                and global_step > 1          # skip step 0 (includes JIT compile)
                and global_step % _comp_interval == 0
            ):
                # Re-run a clean forward pass so ctimer marks come from a steady
                # state step (not mixed with the backward pass ops).
                _ctimer.reset()
                _sigma_comp = jnp.float32(sigma_anneal_fn(global_step))
                _comp_out, _ = forward_only_step(
                    state, last_batch,
                    sigma_scale=_sigma_comp,
                    use_bf16=getattr(config, 'use_bf16', False),
                    loss_chunk_size=getattr(config, 'loss_chunk_size', 0),
                )
                # block_until_ready: flush all jax.debug.callbacks before printing
                jax.block_until_ready(_comp_out)
                _comp_fwd_ms = (_fwd_ms if 'last_batch' in dir() and _timing_compiled
                                else 0.0)
                _ctimer.print_summary(
                    step=int(global_step),
                    total_step_ms=avg_step_time * 1000.0,
                )
                _ctimer.reset()  # clear for next interval

            # Periodic Generation
            if (
                config.generation_steps
                and global_step > 0
                and global_step % config.generation_steps == 0
            ):
                print(f"\n--- Generation at step {global_step} ---")

                if args.custom_prompts:
                    prompts_to_use = args.custom_prompts
                elif config.generation_prompts:
                    prompts_to_use = config.generation_prompts
                elif args.hf_dataset:
                    prompts_to_use = ["The quick brown fox", "Once upon a time"]
                else:
                    prompts_to_use = test_samples

                if not args.custom_prompts and not config.generation_prompts:
                    prompts_to_use = prompts_to_use[:3]

                for prompt in prompts_to_use:
                    print(f"Input: {prompt}")
                    output = generate_fast(
                        state,
                        prompt,
                        tokenizer,
                        max_len=config.generation_max_tokens,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        max_seq_len=config.max_seq_len,
                        verbose=True,
                    )
                    print(f"Output: {output}")
                print("---------------------------------------")

            # Reset start_time for next interval
            if step % LOG_INTERVAL == 0:
                start_time = time.time()
                total_data_wait_time_interval = 0.0

        return state, global_step, epoch_loss, start_time, total_data_wait_time_interval, hit_max_steps

    # ── Graceful interrupt handler ────────────────────────────────────────────
    # Ctrl+C sets the flag; the inner step loop checks it and breaks cleanly,
    # letting the current JAX dispatch finish before we save the checkpoint.
    _stop_requested = [False]

    def _sigint_handler(signum, frame):
        if not _stop_requested[0]:
            print("\n[Interrupted] Ctrl+C received — finishing current step then saving checkpoint...")
            _stop_requested[0] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    start_time = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    #  SEQUENTIAL NPY FILE TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    if use_sequential_npy:
        import gc

        class GrainWrapperSingleFile:
            """Wraps a single-file grain DataLoader; raises StopIteration at EOF."""
            def __init__(self, loader):
                self.loader = loader
                self.iterator = iter(loader)

            def get_batch(self, batch_size=None):
                batch = next(self.iterator)  # raises StopIteration at end
                return batch["input_ids"]

        for epoch in range(args.epochs):
            epoch_loss_total = 0
            epoch_steps = 0
            hit_max_steps = False

            for file_idx, npy_path in enumerate(npy_files):
                if args.max_steps and global_step >= args.max_steps:
                    hit_max_steps = True
                    break

                file_label = f"[File {file_idx+1}/{len(npy_files)}] "
                print(f"\n{'='*60}")
                print(f"{file_label}Loading {os.path.basename(npy_path)}...")
                print(f"{'='*60}")

                # Create loader for just this one file
                result = get_single_npy_grain_loader(npy_path, args, config=config)
                if result is None:
                    print(f"{file_label}Failed to load, skipping.")
                    continue
                single_loader, num_records = result

                # Build the async pipeline: GrainWrapper → BackgroundGenerator → DevicePrefetch
                grain_source = GrainWrapperSingleFile(single_loader)
                bg_source = BackgroundGenerator(grain_source, args.batch_size, prefetch_size=50)
                pipeline = DevicePrefetchIterator(
                    data_source=bg_source,
                    batch_size=args.batch_size,
                    sharding=batch_sharding,
                    prefetch_depth=2,
                )

                # Calculate steps for this file
                file_steps = max(1, num_records // args.batch_size)

                state, global_step, file_loss, start_time, total_data_wait_time_interval, hit_max_steps = (
                    _run_training_steps(
                        pipeline, state, global_step, epoch,
                        file_steps, start_time, total_data_wait_time_interval,
                        file_label=file_label,
                    )
                )

                epoch_loss_total += file_loss
                epoch_steps += file_steps

                # ── Tear down pipeline & free RAM ──────────────────────────
                print(f"{file_label}Finished {os.path.basename(npy_path)}. Releasing memory...")
                pipeline.stop()
                bg_source.stop()
                release_npy_loader(single_loader)
                del pipeline, bg_source, grain_source, single_loader
                gc.collect()

                if hit_max_steps or _stop_requested[0]:
                    break

            if hit_max_steps or _stop_requested[0]:
                break

            if epoch_steps > 0:
                avg_loss = epoch_loss_total / epoch_steps
                avg_ppl = jnp.exp(avg_loss)
                print(
                    f"\nEpoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} | Avg PPL: {avg_ppl:.4f}"
                )
            pool_util = log_pool_utilization(state)
            writer.add_scalar("Pool/Utilization", pool_util, global_step)

            if checkpoint_manager:
                print(f"Saving checkpoint at end of epoch {epoch + 1} (step {global_step})...")
                checkpoint_manager.save(global_step, state)
                _save_grain_state(_grain_state_file, global_step, dataset)

    # ══════════════════════════════════════════════════════════════════════════
    #  ORIGINAL SINGLE-PIPELINE TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    else:
        for epoch in range(args.epochs):
            state, global_step, epoch_loss, start_time, total_data_wait_time_interval, hit_max_steps = (
                _run_training_steps(
                    dataset, state, global_step, epoch,
                    steps_per_epoch, start_time, total_data_wait_time_interval,
                )
            )

            if hit_max_steps or _stop_requested[0]:
                break

            avg_loss = epoch_loss / steps_per_epoch
            avg_ppl = jnp.exp(avg_loss)
            print(
                f"Epoch {epoch + 1} Complete | Avg Loss: {avg_loss:.4f} | Avg PPL: {avg_ppl:.4f}"
            )
            pool_util = log_pool_utilization(state)
            writer.add_scalar("Pool/Utilization", pool_util, global_step)

            if checkpoint_manager:
                print(f"Saving checkpoint at end of epoch {epoch + 1} (step {global_step})...")
                checkpoint_manager.save(global_step, state)
                _save_grain_state(_grain_state_file, global_step, dataset)

        # ── Gracefully shut down ChunkedHFDataset background thread ──────────
        # The DevicePrefetchIterator wraps the dataset; reach through to stop().
        if _dataset_is_chunked:
            _inner = getattr(dataset, 'data_source', dataset)
            if hasattr(_inner, 'stop'):
                _inner.stop()

    # ── Interrupted: save checkpoint and exit cleanly ─────────────────────────
    if _stop_requested[0]:
        if checkpoint_manager:
            print(f"[Interrupted] Saving checkpoint at step {global_step}...")
            checkpoint_manager.save(global_step, state)
            _save_grain_state(_grain_state_file, global_step, dataset)
            print(f"[Interrupted] Checkpoint saved. Exiting.")
        else:
            print("[Interrupted] No checkpoint_dir configured — checkpoint not saved. Exiting.")
        writer.close()
        sys.exit(0)

    # Generation Test
    print("\nVerifying model generation...")

    if args.hf_dataset or args.hf_datasets:
        prompt = "The quick brown fox"
        print(f"Input: {prompt}")
        output = generate(
            state, prompt, tokenizer, temperature=0.7, repetition_penalty=1.2
        )
        print(f"Output: {output}")
    else:
        test_samples = ["Sort: 5 2 8 1 ->", "Sort: 10 3 7 ->", "Sort: 1 1 1 ->"]

        for prompt in test_samples:
            print(f"Input: {prompt}")
            output = generate(
                state, prompt, tokenizer, temperature=0.7, repetition_penalty=1.2
            )
            print(f"Output: {output}")
            print("-" * 20)

    pool_util = log_pool_utilization(state)
    writer.add_scalar("Pool/Utilization", pool_util, global_step)
    writer.close()


if __name__ == "__main__":
    main()
