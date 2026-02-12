import argparse
import os
import jax
import orbax.checkpoint
from dpsn_r_jax.config import get_model_config
from dpsn_r_jax.training.trainer import create_train_state


def main():
    parser = argparse.ArgumentParser(
        description="Export DPSNR model to universal/unsharded format"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="large",
        choices=["tiny", "base", "large", "xl"],
        help="Model configuration size (default: large)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the source sharded checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where the unsharded model will be saved",
    )

    args = parser.parse_args()

    # Force CPU to merge shards
    cpu_device = jax.devices("cpu")[0]

    print(f"Loading configuration: {args.config}")
    config = get_model_config(args.config)

    abs_checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.exists(abs_checkpoint_dir):
        print(f"Error: Checkpoint directory {abs_checkpoint_dir} does not exist.")
        return

    print(f"Initializing dummy state on CPU...")
    rng = jax.random.PRNGKey(0)
    with jax.default_device(cpu_device):
        dummy_state = create_train_state(rng, config)

    # Ensure all arrays in dummy_state are on CPU and replicated
    dummy_state = jax.device_put(dummy_state, cpu_device)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        abs_checkpoint_dir, checkpointer
    )
    latest_step = checkpoint_manager.latest_step()

    # Construct restore_args to force resharding to CPU
    restore_args = orbax.checkpoint.checkpoint_utils.construct_restore_args(dummy_state)

    if latest_step is not None:
        print(
            f"Restoring checkpoint from {abs_checkpoint_dir} at step {latest_step}..."
        )
        state = checkpoint_manager.restore(
            latest_step,
            items=dummy_state,
            restore_kwargs={"restore_args": restore_args},
        )
    else:
        target_path = None
        if os.path.exists(os.path.join(abs_checkpoint_dir, "default")):
            target_path = os.path.join(abs_checkpoint_dir, "default")
        elif os.path.exists(os.path.join(abs_checkpoint_dir, "_METADATA")):
            target_path = abs_checkpoint_dir
        elif os.path.exists(os.path.join(abs_checkpoint_dir, "checkpoint")):
            target_path = os.path.join(abs_checkpoint_dir, "checkpoint")
        elif os.path.exists(os.path.join(abs_checkpoint_dir, "params")):
            target_path = os.path.join(abs_checkpoint_dir, "params")

        if target_path:
            print(f"Restoring checkpoint directly from {target_path}...")
            state = checkpointer.restore(
                target_path, item=dummy_state, restore_args=restore_args
            )
        else:
            print(f"Error: No checkpoint found in {abs_checkpoint_dir}.")
            return

    print(f"Saving universal/unsharded model to {args.output_dir}...")
    abs_output_dir = os.path.abspath(args.output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)

    output_manager = orbax.checkpoint.CheckpointManager(
        abs_output_dir, orbax.checkpoint.PyTreeCheckpointer()
    )

    output_manager.save(0, state)
    output_manager.wait_until_finished()

    print(f"\nSuccess! The exported model is now 'Universal / Unsharded'.")
    print(f"Location: {abs_output_dir}")


if __name__ == "__main__":
    main()
