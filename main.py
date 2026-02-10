import argparse
import jax
import jax.numpy as jnp
import optax
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.data.dataset import SyntheticReasoningDataset, HFStreamingDataset
from dpsn_r_jax.data.tokenizer import get_tokenizer
from dpsn_r_jax.training.trainer import create_train_state
from dpsn_r_jax.config import DPSNRConfig, get_tiny_config
from dpsn_r_jax.utils.generation import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--hf_dataset", type=str, default=None, help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--hf_subset",
        type=str,
        default=None,
        help="HuggingFace dataset configuration/subset",
    )
    parser.add_argument(
        "--hf_tokenizer", type=str, default=None, help="HuggingFace tokenizer name"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--generation_steps",
        type=int,
        default=None,
        help="Generate output every N steps",
    )
    parser.add_argument(
        "--generation_max_tokens", type=int, default=20, help="Max tokens to generate"
    )
    args = parser.parse_args()

    print(f"Training on {jax.devices()[0].platform.upper()}")

    # Config
    if args.config:
        print(f"Loading config from {args.config}")
        config = DPSNRConfig.from_yaml(args.config)
    elif args.tiny:
        print("Using TINY config for testing...")
        config = get_tiny_config()
    else:
        config = DPSNRConfig()

    # --- DISTRIBUTED SETUP ---
    from jax.sharding import Mesh, PartitionSpec, NamedSharding
    from jax.experimental import mesh_utils

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

    if config.hf_tokenizer_name and config.hf_tokenizer_name.lower() != "numeric":
        if hasattr(tokenizer, "vocab_size"):
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
            print(
                f"Updated vocab size to {config.vocab_size}, pad_token_id to {config.pad_token_id} from pretrained tokenizer"
            )

    if args.hf_dataset:
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

    # Initialize state with sharding constraints
    # We first create the raw variables distributedly
    variables = jax.lax.with_sharding_constraint(
        init_model(rng, dummy_input), sharding_tree
    )

    # Create TrainState (using the sharded variables)
    from dpsn_r_jax.training.trainer import TrainState

    tx = optax.adamw(config.learning_rate)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
        rng=rng,
    )

    # Force the state params to respect the sharding (TrainState.create might lose it if not careful,
    # but since variables['params'] is already a sharded Array, it should persist).
    # Double check by re-imposing constraint if needed, but usually redundant if input is sharded.

    print(
        f"Model Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}"
    )

    from dpsn_r_jax.training.trainer import train_step

    # JIT the train step - XLA automatically handles the communication!
    # We just need to ensure inputs are sharded correctly before entering.
    distributed_train_step = jax.jit(train_step)

    steps_per_epoch = args.dataset_size // args.batch_size

    global_step = 0

    # Define test samples for generation
    test_samples = ["Sort: 5 2 8 1 ->", "Sort: 10 3 7 ->", "Sort: 1 1 1 ->"]

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

            state, loss = distributed_train_step(state, batch, config.pad_token_id)
            epoch_loss += loss
            global_step += 1

            if step % 10 == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | Global Step {global_step} | Loss: {loss:.4f}"
                )

            # Periodic Generation
            if (
                config.generation_steps
                and global_step > 0
                and global_step % config.generation_steps == 0
            ):
                print(f"\n--- Generation at step {global_step} ---")
                prompts_to_use = (
                    config.generation_prompts
                    if config.generation_prompts
                    else (["The quick brown fox"] if args.hf_dataset else test_samples)
                )

                # Limit to 1 sample to save time if not explicit list
                if not config.generation_prompts:
                    prompts_to_use = prompts_to_use[:1]

                for prompt in prompts_to_use:
                    print(f"Input: {prompt}")
                    output = generate(
                        state, prompt, tokenizer, max_len=config.generation_max_tokens
                    )
                    print(f"Output: {output}")
                print("---------------------------------------")

        if args.max_steps and global_step >= args.max_steps:
            break

        print(
            f"Epoch {epoch + 1} Complete | Avg Loss: {epoch_loss / steps_per_epoch:.4f}"
        )

    # Generation Test
    print("\nVerifying model generation...")

    if args.hf_dataset:
        prompt = "The quick brown fox"
        print(f"Input: {prompt}")
        output = generate(state, prompt, tokenizer)
        print(f"Output: {output}")
    else:
        test_samples = ["Sort: 5 2 8 1 ->", "Sort: 10 3 7 ->", "Sort: 1 1 1 ->"]

        for prompt in test_samples:
            print(f"Input: {prompt}")
            output = generate(state, prompt, tokenizer)
            print(f"Output: {output}")
            print("-" * 20)


if __name__ == "__main__":
    main()
