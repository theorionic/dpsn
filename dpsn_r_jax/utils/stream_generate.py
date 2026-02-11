import sys
import jax
import jax.numpy as jnp
from typing import Any, List


@jax.jit
def inference_step(state: Any, input_ids: jnp.ndarray) -> jnp.ndarray:
    """
    Performs a single forward pass to get logits for the next token.

    Args:
        state: TrainState containing model parameters and apply_fn.
        input_ids: Input sequence of token IDs [Batch, SeqLen].

    Returns:
        Logits for the last token in the sequence [VocabSize].
    """
    logits, _ = state.apply_fn({"params": state.params}, input_ids, deterministic=True)
    return logits[0, -1, :]


def stream_generate(
    state: Any,
    prompt: str,
    tokenizer: Any,
    max_len: int = 50,
    temperature: float = 1.0,
):
    """
    Generates text autoregressively and streams tokens to stdout.

    Args:
        state: TrainState for the model.
        prompt: The input text to start generation from.
        tokenizer: Tokenizer object with encode/decode capability.
        max_len: Maximum number of tokens to generate.
        temperature: Scaling factor for logits (defaults to greedy if 1.0).
    """
    # Requirement: Tokenize input prompt
    input_ids_list = tokenizer.encode(prompt)
    generated_ids = list(input_ids_list)

    print(f"Prompt: {prompt}\nResponse: ", end="", flush=True)

    # Requirement: Python loop for autoregressive generation
    for _ in range(max_len):
        # Requirement: Feed [prompt + generated] to model
        input_tensor = jnp.array([generated_ids], dtype=jnp.int32)

        # Requirement: Use jax.jit for single-step inference
        logits = inference_step(state, input_tensor)

        if temperature != 1.0 and temperature > 0:
            logits = logits / temperature

        # Greedy selection
        next_token_id = int(jnp.argmax(logits, axis=-1))

        # Requirement: Stop on <EOS> (ID 1)
        if next_token_id == 1:
            break

        generated_ids.append(next_token_id)

        # Requirement: Print each token immediately to stdout with flush=True
        token_str = tokenizer.decode([next_token_id])
        print(token_str, end="", flush=True)
        sys.stdout.flush()

    print("\n[Done]")


class SimpleASCIITokenizer:
    """Simple ASCII-based tokenizer for demonstration purposes."""

    def __init__(self, vocab_size: int = 256):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        # Leave IDs 0 and 1 for PAD and EOS
        return [min(ord(c) + 2, self.vocab_size - 1) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        res = []
        for tid in token_ids:
            if tid == self.eos_token_id:
                res.append("<EOS>")
            elif tid == self.pad_token_id:
                res.append("<PAD>")
            elif tid < 2:
                res.append(f"<S{tid}>")
            else:
                res.append(chr(min(tid - 2, 0x10FFFF)))
        return "".join(res)


if __name__ == "__main__":
    from dpsn_r_jax.config import get_tiny_config
    from dpsn_r_jax.training.trainer import create_train_state

    config = get_tiny_config()
    # Increase vocab size to accommodate ASCII tokens
    config.vocab_size = 258

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)

    tokenizer = SimpleASCIITokenizer(vocab_size=config.vocab_size)

    stream_generate(state, "Test prompt: ", tokenizer, max_len=20)
