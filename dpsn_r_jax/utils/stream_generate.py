import sys
import os
import argparse
import jax
import jax.numpy as jnp
import orbax.checkpoint
from typing import Any, List, Optional

from dpsn_r_jax.config import get_model_config
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.data.tokenizer import get_tokenizer
from dpsn_r_jax.training.trainer import create_train_state


@jax.jit
def inference_step(
    state: Any, input_ids: jnp.ndarray, rng: jnp.ndarray, temperature: float = 1.0
) -> jnp.ndarray:
    """
    Performs a single forward pass and samples the next token.

    Args:
        state: TrainState containing model parameters and apply_fn.
        input_ids: Input sequence of token IDs [Batch, SeqLen].
        rng: PRNGKey for sampling.
        temperature: Scaling factor for logits.

    Returns:
        The sampled next token ID.
    """
    logits, _ = state.apply_fn({"params": state.params}, input_ids, deterministic=True)
    logits = logits[0, -1, :]

    return jax.lax.cond(
        temperature > 0,
        lambda l: jax.random.categorical(rng, l / temperature),
        lambda l: jnp.argmax(l, axis=-1),
        logits,
    )


def load_model(config_name: str, checkpoint_dir: str, tokenizer_path: str):
    """
    Loads model config, tokenizer, and restores state with CPU fix.
    """
    config = get_model_config(config_name)
    tokenizer = get_tokenizer(tokenizer_path)

    if tokenizer_path and tokenizer_path.lower() != "numeric":
        if hasattr(tokenizer, "vocab_size"):
            config.vocab_size = tokenizer.vocab_size
            config.pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
            print(
                f"Updated vocab size to {config.vocab_size} and pad_token_id to {config.pad_token_id}"
            )

    print("Initializing dummy state on CPU...")
    rng = jax.random.PRNGKey(0)
    devices = jax.devices("cpu")
    if not devices:
        dummy_state = create_train_state(rng, config)
    else:
        with jax.default_device(devices[0]):
            dummy_state = create_train_state(rng, config)

    state = dummy_state

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
                state = checkpoint_manager.restore(latest_step, items=dummy_state)
            else:
                print(
                    f"No checkpoint found in {abs_checkpoint_dir}. Using initialized parameters."
                )
        else:
            print(
                f"Checkpoint directory {abs_checkpoint_dir} does not exist. Using initialized parameters."
            )

    return state, tokenizer, config


def stream_generate(
    state: Any,
    prompt: str,
    tokenizer: Any,
    rng: jnp.ndarray,
    max_len: int = 50,
    temperature: float = 1.0,
):
    """
    Generates text autoregressively and streams tokens to stdout.

    Args:
        state: TrainState for the model.
        prompt: The input text to start generation from.
        tokenizer: Tokenizer object with encode/decode capability.
        rng: PRNGKey for sampling.
        max_len: Maximum number of tokens to generate.
        temperature: Scaling factor for logits (0.0 for greedy).
    """
    max_retries = 3
    input_ids_list = []
    for attempt in range(max_retries + 1):
        try:
            input_ids_list = tokenizer.encode(prompt)
            break
        except RuntimeError as e:
            if "Already borrowed" in str(e) and attempt < max_retries:
                continue
            raise e
    generated_ids = list(input_ids_list)

    print(f"Prompt: {prompt}\nResponse: ", end="", flush=True)

    for _ in range(max_len):
        input_tensor = jnp.array([generated_ids], dtype=jnp.int32)
        rng, step_rng = jax.random.split(rng)

        next_token_id = inference_step(state, input_tensor, step_rng, temperature)
        next_token_id = int(next_token_id)

        eos_id = getattr(tokenizer, "eos_token_id", 1)
        if next_token_id == eos_id:
            break

        generated_ids.append(next_token_id)

        token_str = tokenizer.decode([next_token_id])
        print(token_str, end="", flush=True)
        sys.stdout.flush()

    print("\n[Done]")


def main():
    parser = argparse.ArgumentParser(description="DPSNR Streaming Generation CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "base", "large", "xl"],
        help="Model configuration size (default: tiny)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="Tokenizer path or name (default: EleutherAI/gpt-neo-125M)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of AI is",
        help="Initial prompt for generation",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 for greedy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation",
    )

    args = parser.parse_args()

    state, tokenizer, config = load_model(
        args.config, args.checkpoint_dir, args.tokenizer
    )
    rng = jax.random.PRNGKey(args.seed)

    stream_generate(
        state=state,
        prompt=args.prompt,
        tokenizer=tokenizer,
        rng=rng,
        max_len=args.max_tokens,
        temperature=args.temp,
    )


if __name__ == "__main__":
    main()
