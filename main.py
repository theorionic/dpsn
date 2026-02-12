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
        "--hf_dataset", type=str, default=None, help="HuggingFace dataset name"
    )
    parser.add_argument("--hf_subset", type=str, default=None, help="Dataset subset")
    parser.add_argument(
        "--hf_tokenizer", type=str, default=None, help="HuggingFace tokenizer"
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

    args = parser.parse_args()

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

    grain_loader = get_grain_loader(args)
    if grain_loader:
        print("Using Google Grain data loader.")

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

    tx = optax.adamw(config.learning_rate)
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
            # We must pass the target 'state' so Orbax knows the sharding layout
            state = checkpoint_manager.restore(latest_step, items=state)
            global_step = latest_step
        else:
            print("No checkpoint found to resume from. Starting from scratch.")
            global_step = 0
    else:
        global_step = 0

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
    steps_per_epoch = args.dataset_size // args.batch_size

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

            # Save Checkpoint
            if (
                checkpoint_manager
                and args.save_interval
                and global_step > 0
                and global_step % args.save_interval == 0
            ):
                print(f"Saving checkpoint at step {global_step}...")
                checkpoint_manager.save(global_step, state)

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

        # Save checkpoint at end of epoch
        # Save checkpoint at end of epoch
        if checkpoint_manager:
            print(
                f"Saving checkpoint at end of epoch {epoch + 1} (step {global_step})..."
            )
            checkpoint_manager.save(global_step, state)

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


if __name__ == "__main__":
    main()
