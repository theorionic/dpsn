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
    HFStreamingDataset,
    SyntheticReasoningDataset,
    BackgroundGenerator,
)
from dpsn_r_jax.data.tokenizer import get_tokenizer
from dpsn_r_jax.data.grain_loader import get_grain_loader
from dpsn_r_jax.utils.generation import generate
from dpsn_r_jax.utils.metrics import calculate_flops


def log_pool_utilization(state):
    touched_mask = jnp.any(state.pool_v > 0, axis=-1)
    num_touched = jnp.sum(touched_mask)
    total_vectors = state.pool_v.shape[0]
    percentage = (num_touched / total_vectors) * 100
    print(
        f"Pool Utilization: {percentage:.2f}% ({int(num_touched)} / {total_vectors} vectors touched)"
    )
    return float(percentage)


def sync_checkpoints(local_dir, remote_dest):
    """Syncs local checkpoints to a remote destination using rclone."""
    try:
        from rclone_python import rclone

        print(f"Syncing checkpoints to remote: {remote_dest}...")
        rclone.sync(local_dir, remote_dest)
        print("Checkpoint sync complete.")
    except ImportError:
        print(
            "WARNING: rclone-python not installed. Skipping sync.\n"
            "To enable remote checkpoints, install: pip install rclone-python"
        )
    except Exception as e:
        print(f"ERROR: Failed to sync checkpoints: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train DPSNR Model")
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny config for testing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="base",
        choices=["tiny", "base", "large", "xl"],
        help="Model configuration size",
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
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps for scheduler",
    )
    parser.add_argument(
        "--rclone_remote",
        type=str,
        default=None,
        help="Rclone remote destination (e.g., 'gdrive:dpsn_checkpoints')",
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

    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

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
    from dpsn_r_jax.training.trainer import TrainState
    from flax import traverse_util

    params = variables["params"]
    flat_params = traverse_util.flatten_dict(params)
    pool_key = ("pool", "params_storage")
    pool_params = flat_params[pool_key]
    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    # Use learning rate schedule
    total_steps = (
        args.max_steps
        if args.max_steps
        else args.epochs * (args.dataset_size // args.batch_size)
    )
    if args.warmup_steps > 0:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=total_steps,
            end_value=0.1 * config.learning_rate,
        )
    else:
        lr_schedule = config.learning_rate

    tx = optax.adamw(lr_schedule)
    opt_state = tx.init(dense_params)

    pool_m = jnp.zeros_like(pool_params)
    pool_v = jnp.zeros_like(pool_params)

    state = TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        rng=rng,
        pool_m=pool_m,
        pool_v=pool_v,
        window_size=config.max_k,
        learning_rate=config.learning_rate,
    )

    # RESTORE CHECKPOINT IF REQUESTED
    if args.resume and checkpoint_manager:
        latest_step = checkpoint_manager.latest_step()
        if latest_step is not None:
            print(f"Resuming from checkpoint step {latest_step}...")
            try:
                state = checkpoint_manager.restore(latest_step, items=state)
            except ValueError as e:
                if "tree structures do not match" in str(e):
                    print("Warning: Optimizer state structure mismatch detected.")
                    print(
                        "Restoring model parameters only, optimizer state will be reinitialized."
                    )
                    restore_args = orbax.checkpoint.PyTreeRestoreArgs(
                        item=state, partial_restore=True
                    )
                    restored = checkpoint_manager.restore(
                        latest_step, args=restore_args
                    )
                    state = state.replace(
                        step=restored.get("step", latest_step),
                        params=restored.get("params", state.params),
                    )
                else:
                    raise
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
                    global_step = int(step_str)
                except ValueError:
                    global_step = 0
            else:
                print("No checkpoint found to resume from. Starting from scratch.")
                global_step = 0
    else:
        global_step = 0

    # Data Loader Initialization
    if args.skip_batches > 0:
        loader_start_step = args.skip_batches
    elif args.resume_data:
        loader_start_step = global_step
    else:
        loader_start_step = 0
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
    elif args.hf_dataset:
        print(
            f"Loading HF streaming dataset: {args.hf_dataset} (subset: {args.hf_subset})"
        )
        dataset = HFStreamingDataset(
            args.hf_dataset,
            tokenizer,
            subset=args.hf_subset,
            seq_len=config.max_seq_len,
            batch_size=args.batch_size,
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

    from dpsn_r_jax.training.trainer import train_step

    # JIT the train step - XLA automatically handles the communication!
    # We just need to ensure inputs are sharded correctly before entering.
    distributed_train_step = jax.jit(train_step)

    flops_per_step = calculate_flops(config, args.batch_size)
    steps_per_epoch = max(1, args.dataset_size // args.batch_size)

    # Define test samples for generation
    test_samples = ["Sort: 5 2 8 1 ->", "Sort: 10 3 7 ->", "Sort: 1 1 1 ->"]

    dataset = BackgroundGenerator(dataset, args.batch_size, prefetch_size=5)

    for epoch in range(args.epochs):
        epoch_loss = 0

        for step in range(steps_per_epoch):
            if args.max_steps and global_step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}). Stopping training.")
                break

            batch = dataset.get_batch(args.batch_size)

            # Shard the batch input!
            # We must put the batch onto the mesh with the data sharding spec
            # (Batch, SeqLen) -> split Batch across 'shard' axis
            batch = jax.device_put(batch, batch_sharding)

            start_time = time.time()
            state, loss = distributed_train_step(state, batch, config.pad_token_id)
            loss.block_until_ready()
            step_time = time.time() - start_time

            tokens_per_sec = (args.batch_size * config.max_seq_len) / step_time
            tflops = flops_per_step / step_time / 1e12

            epoch_loss += loss
            global_step += 1

            # TensorBoard logging
            writer.add_scalar("Loss/train", float(loss), global_step)
            writer.add_scalar("Perf/TPS", tokens_per_sec, global_step)
            writer.add_scalar("Perf/TFLOPS", tflops, global_step)
            if hasattr(state, "learning_rate"):
                writer.add_scalar("LR", state.learning_rate, global_step)

            # Save Checkpoint
            if (
                checkpoint_manager
                and args.save_interval
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving checkpoint at step {global_step}...")
                checkpoint_manager.save(global_step, state)
                # Save data loader state if supported
                if hasattr(grain_loader, "get_state"):
                    import json

                    with open(args.resume_data_path, "w") as f:
                        json.dump(grain_loader.get_state(), f)

                if args.rclone_remote:
                    sync_checkpoints(args.checkpoint_dir, args.rclone_remote)

            if step % 10 == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | Global Step {global_step} | Loss: {loss:.4f} | TPS: {tokens_per_sec:.0f} | TFLOPS: {tflops:.4f}"
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
                print("---------------------------------------")

        if args.max_steps and global_step >= args.max_steps:
            break

        print(
            f"Epoch {epoch + 1} Complete | Avg Loss: {epoch_loss / steps_per_epoch:.4f}"
        )
        pool_util = log_pool_utilization(state)
        writer.add_scalar("Pool/Utilization", pool_util, global_step)

        # Save checkpoint at end of epoch
        if checkpoint_manager:
            print(
                f"Saving checkpoint at end of epoch {epoch + 1} (step {global_step})..."
            )
            checkpoint_manager.save(global_step, state)
            if hasattr(grain_loader, "get_state"):
                import json

                with open(args.resume_data_path, "w") as f:
                    json.dump(grain_loader.get_state(), f)

            if args.rclone_remote:
                sync_checkpoints(args.checkpoint_dir, args.rclone_remote)

    # Generation Test
    print("\nVerifying model generation...")

    if args.hf_dataset:
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
