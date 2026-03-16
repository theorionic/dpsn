import argparse
import os
import time
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
from dpsn_r_jax.utils.generation import generate, clear_generation_cache
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


def main():
    parser = argparse.ArgumentParser(description="Train DPSNR Model")
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny config for testing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xl", "precise_tiny", "precise_large"],
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

    args = parser.parse_args()

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

    if args.loss_chunk_size > 0:
        config.loss_chunk_size = args.loss_chunk_size

    # Create device mesh - handles 1 to N devices automatically
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    # We use 'data' axis for data parallelism and 'pool' axis for model parallelism of the pool
    # Since we have a 1D mesh, we map 'data' to the single axis
    # For complex setups on 2D meshes (e.g. 4x8), this would need adjustment,
    # but for 1D array of devices, we use the single axis for both or mix them.
    # Here we define a single axis name 'shard'.
    mesh = Mesh(devices, axis_names=("shard",))

    # Sharding Rules:
    # 1. Batch: Split along 'shard' axis (Data Parallelism)
    # 2. Pool Params: Split along 'shard' axis (Model Parallelism)
    # 3. Other Params: Replicated (None)

    batch_sharding = NamedSharding(mesh, PartitionSpec("shard", None))
    # Pool is usually (num_vectors, dim), we split num_vectors
    pool_sharding = NamedSharding(mesh, PartitionSpec("shard", None))
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    def get_sharding_rule(path, param):
        """
        Determines where a parameter should live based on its path in the PyTree.
        path: tuple of strings (e.g., ('params', 'pool', 'vectors'))
        param: the actual parameter array (for shape inspection if needed)
        """
        # If it's part of the massive pool, shard it!
        # Path usually looks like ('params', 'pool', ...)
        if "pool" in path:
            # We shard the first dimension (total_vectors)
            return pool_sharding

        # Everything else (Controller, Router, etc.) is REPLICATED
        return replicated_sharding

    print(f"Distributed Mesh: {mesh}")
    print(f"Sharding Strategy: Pool -> Sharded, Rest -> Replicated")

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

    pool_m = jnp.zeros_like(pool_params)
    pool_v = jnp.zeros_like(pool_params)

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
        learning_rate_fn=lr_schedule,
        sigma_anneal_fn=sigma_anneal_fn,
    )

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

    # ── Detect sequential NPY mode ─────────────────────────────────────────
    npy_files = expand_npy_paths(args.dataset_path) if args.dataset_path else []
    use_sequential_npy = len(npy_files) > 0

    if use_sequential_npy:
        print(f"\nSequential NPY mode: {len(npy_files)} files detected.")
        print(f"Files will be loaded ONE AT A TIME to minimize RAM usage.")
        for i, f in enumerate(npy_files):
            print(f"  [{i+1}/{len(npy_files)}] {os.path.basename(f)}")
    else:
        # Fallback: original loader path for non-NPY datasets
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
        elif args.hf_dataset or args.hf_datasets:
            # Resolve the primary dataset name: --hf_dataset takes precedence,
            # falling back to the first entry in --hf_datasets.
            primary_hf = args.hf_dataset or (args.hf_datasets[0] if args.hf_datasets else None)

            if getattr(args, "chunk_size", 0) > 0:
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
                )
            else:
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

    from dpsn_r_jax.training.trainer import train_step

    # train_step is already JIT-compiled with static_argnames in trainer.py!
    # We just need to ensure inputs are sharded correctly before entering.
    distributed_train_step = train_step

    flops_per_step = calculate_flops(config, args.batch_size)

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

    LOG_INTERVAL = 200  # sync + log every N steps; lower = more block_until_ready stalls

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
        epoch_loss = 0
        hit_max_steps = False

        for step in range(steps_per_epoch):
            if args.max_steps and global_step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}). Stopping training.")
                hit_max_steps = True
                break

            data_start_time = time.time()
            try:
                batch = dataset_pipeline.get_batch(args.batch_size)
            except (StopIteration, Exception) as e:
                if isinstance(e, StopIteration) or "StopIteration" in str(type(e).__name__):
                    break  # This file is exhausted
                raise
            current_data_wait_time = time.time() - data_start_time
            total_data_wait_time_interval += current_data_wait_time

            state, loss, mean_sigma = distributed_train_step(
                state, batch, config.pad_token_id,
                precision_loss_weight=getattr(config, 'precision_loss_weight', 0.0),
                sigma_anneal_steps=getattr(config, 'sigma_anneal_steps', 0),
                use_bf16=getattr(config, 'use_bf16', False),
                loss_chunk_size=getattr(config, 'loss_chunk_size', 0),
            )

            epoch_loss += loss
            global_step += 1

            # TensorBoard logging
            writer.add_scalar("Loss/train", float(loss), global_step)
            writer.add_scalar("PPL/train", float(jnp.exp(loss)), global_step)
            current_lr = state.learning_rate_fn(global_step)
            writer.add_scalar("LR", current_lr, global_step)
            writer.add_scalar("Routing/mean_sigma", float(mean_sigma), global_step)
            sigma_scale = float(state.sigma_anneal_fn(global_step))
            writer.add_scalar("Routing/sigma_scale", sigma_scale, global_step)

            # Save Checkpoint
            if (
                checkpoint_manager
                and args.save_interval
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving checkpoint at step {global_step}...")
                checkpoint_manager.save(global_step, state)

            if step % LOG_INTERVAL == 0:
                loss.block_until_ready()
                current_time = time.time()
                elapsed = current_time - start_time

                steps_in_interval = LOG_INTERVAL if step > 0 else 1
                avg_step_time = elapsed / steps_in_interval
                avg_data_wait = total_data_wait_time_interval / steps_in_interval

                active_tpu_time = max(0.0001, avg_step_time - avg_data_wait)

                tokens_per_sec = (args.batch_size * config.max_seq_len) / avg_step_time
                tflops = flops_per_step / active_tpu_time / 1e12

                writer.add_scalar("Perf/TPS", tokens_per_sec, global_step)
                writer.add_scalar("Perf/TFLOPS", tflops, global_step)
                writer.add_scalar("Perf/DataWaitTime_s", avg_data_wait, global_step)

                ppl = jnp.exp(loss)
                sigma_scale = float(state.sigma_anneal_fn(global_step))
                precision_tag = "broad" if float(mean_sigma) > 1.0 else ("precise" if float(mean_sigma) < 0.1 else "narrowing")
                print(
                    f"{file_label}Epoch {epoch + 1} | Step {step} | Global Step {global_step} | "
                    f"Loss: {loss:.4f} | PPL: {ppl:.4f} | LR: {current_lr:.2e} | "
                    f"sigma={float(mean_sigma):.3f} ({precision_tag}) | "
                    f"TPS: {tokens_per_sec:.0f} | TFLOPS: {tflops:.4f} | DataWait: {avg_data_wait:.3f}s"
                )

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
                    output = generate(
                        state,
                        prompt,
                        tokenizer,
                        max_len=config.generation_max_tokens,
                        temperature=0.7,
                        repetition_penalty=1.2,
                    )
                    print(f"Output: {output}")
                clear_generation_cache()
                print("---------------------------------------")

            # Reset start_time for next interval
            if step % LOG_INTERVAL == 0:
                start_time = time.time()
                total_data_wait_time_interval = 0.0

        return state, global_step, epoch_loss, start_time, total_data_wait_time_interval, hit_max_steps

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

                if hit_max_steps:
                    break

            if hit_max_steps:
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

            if hit_max_steps:
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

        # ── Gracefully shut down ChunkedHFDataset background thread ──────────
        # The DevicePrefetchIterator wraps the dataset; reach through to stop().
        if _dataset_is_chunked:
            _inner = getattr(dataset, 'data_source', dataset)
            if hasattr(_inner, 'stop'):
                _inner.stop()

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
