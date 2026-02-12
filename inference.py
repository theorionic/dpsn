import argparse
import os

# Fix RuntimeError: Already borrowed
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import jax
import jax.numpy as jnp
import orbax.checkpoint
import optax
from flax.training import orbax_utils
from dpsn_r_jax.config import get_model_config
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.data.tokenizer import get_tokenizer
from dpsn_r_jax.utils.generation import generate
from dpsn_r_jax.training.trainer import TrainState, create_train_state


def load_model(config_name: str, checkpoint_dir: str, tokenizer_path: str):
    """
    Loads the model configuration, tokenizer, and restores the model state from a checkpoint.
    """
    config = get_model_config(config_name)
    tokenizer = get_tokenizer(tokenizer_path)

    # Update config vocab size if using a pretrained tokenizer
    if tokenizer_path and tokenizer_path.lower() != "numeric":
        if hasattr(tokenizer, "vocab_size"):
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
            print(
                f"Updated vocab size to {config.vocab_size} and pad_token_id to {config.pad_token_id}"
            )

    model = DPSNR(config)
    rng = jax.random.PRNGKey(0)

    print("Initializing model parameters...")
    state = create_train_state(rng, config)

    if checkpoint_dir:
        abs_checkpoint_dir = os.path.abspath(checkpoint_dir)
        if os.path.exists(abs_checkpoint_dir):
            checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            checkpoint_manager = orbax.checkpoint.CheckpointManager(
                abs_checkpoint_dir, checkpointer
            )
            latest_step = checkpoint_manager.latest_step()
            if latest_step is not None:
                print(
                    f"Restoring checkpoint from {abs_checkpoint_dir} at step {latest_step}..."
                )
                state = checkpoint_manager.restore(latest_step, items=state)
            else:
                print(
                    f"No checkpoint found in {abs_checkpoint_dir}. Using initialized parameters."
                )
        else:
            print(
                f"Checkpoint directory {abs_checkpoint_dir} does not exist. Using initialized parameters."
            )

    return state, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Inference with DPSNR Model")
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
        default="checkpoints_large",
        help="Path to checkpoints (default: checkpoints_large)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="HuggingFace tokenizer path (default: EleutherAI/gpt-neo-125M)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input text (if None, enter interactive mode)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Max tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temp", type=float, default=0.7, help="Temperature (default: 0.7)"
    )
    parser.add_argument(
        "--penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2)"
    )

    args = parser.parse_args()

    state, tokenizer, config = load_model(
        args.config, args.checkpoint_dir, args.tokenizer
    )

    rng = jax.random.PRNGKey(42)

    def run_inference(prompt: str):
        nonlocal rng
        rng, key = jax.random.split(rng)
        output = generate(
            state,
            prompt,
            tokenizer,
            rng=key,
            max_len=args.max_tokens,
            temperature=args.temp,
            repetition_penalty=args.penalty,
        )
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output}\n")

    if args.prompt:
        run_inference(args.prompt)
    else:
        print("\n--- DPSNR Interactive Inference ---")
        print("Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input(">>> ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                if not user_input.strip():
                    continue
                run_inference(user_input)
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break


if __name__ == "__main__":
    main()
