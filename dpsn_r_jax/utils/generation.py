import jax
import jax.numpy as jnp
import time
from typing import Optional, Tuple, Any
from functools import partial


def _apply_repetition_penalty(
    logits: jnp.ndarray,
    generated_tokens: jnp.ndarray,
    penalty: float,
) -> jnp.ndarray:
    vocab_size = logits.shape[-1]
    token_mask = jnp.zeros((vocab_size,), dtype=jnp.bool_)
    token_mask = token_mask.at[generated_tokens].set(True)

    penalized = jnp.where(
        logits > 0,
        logits / penalty,
        logits * penalty,
    )
    return jnp.where(token_mask, penalized, logits)


def _apply_top_k(logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
    actual_k = min(top_k, logits.shape[-1])
    values, _ = jax.lax.top_k(logits, k=actual_k)
    min_value = values[-1]
    return jnp.where(
        logits < min_value,
        jnp.full_like(logits, -1e10),
        logits,
    )


def _sample_token(
    logits: jnp.ndarray,
    rng: jax.random.PRNGKey,
    temperature: float,
) -> jnp.ndarray:
    safe_temp = jnp.maximum(temperature, 1e-8)
    scaled_logits = logits / safe_temp
    return jax.random.categorical(rng, scaled_logits)


def generate_fast(
    state,
    prompt: str,
    tokenizer,
    rng: Optional[jax.random.PRNGKey] = None,
    max_len: int = 20,
    temperature: float = 1.0,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
    verbose: bool = False,
) -> str:
    """
    Generation without separate JIT compilation of step function.

    During training, this avoids expensive recompilation by using the
    model's apply_fn directly (which is already JIT-compatible through
    the training step). The forward pass is efficient; only the Python
    loop overhead is not JIT'd.

    Args:
        state: TrainState with model params and apply_fn
        prompt: Input text prompt
        tokenizer: Tokenizer with encode/decode methods
        rng: JAX PRNGKey for sampling
        max_len: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling cutoff
        repetition_penalty: Repetition penalty factor
        verbose: Print timing info

    Returns:
        Generated text string
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    input_ids = None
    for i in range(3):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            break
        except RuntimeError as e:
            if "Already borrowed" in str(e) and i < 2:
                time.sleep(0.1)
                continue
            raise e
    assert input_ids is not None

    generated = jnp.array(input_ids)
    eos_token_id = tokenizer.eos_token_id

    start_time = time.time()

    for _ in range(max_len):
        # Call model directly - no separate JIT, applies_fn handles efficiency
        logits, _ = state.apply_fn(
            {"params": state.params}, generated, deterministic=True
        )

        # Apply sampling logic
        next_logits = logits[0, -1, :]
        next_logits = _apply_repetition_penalty(
            next_logits, generated[0], repetition_penalty
        )
        next_logits = _apply_top_k(next_logits, top_k)

        rng, sample_rng = jax.random.split(rng)
        new_token = _sample_token(next_logits, sample_rng, temperature)

        generated = jnp.concatenate([generated, new_token[None, None]], axis=1)

        if new_token == eos_token_id:
            break

    if verbose:
        jax.effects_barrier()
        elapsed = time.time() - start_time
        print(f"Generation time: {elapsed:.3f}s")

    final_tokens = generated[0].tolist()
    return tokenizer.decode(final_tokens, skip_special_tokens=True)


def generate(
    state,
    prompt: str,
    tokenizer,
    rng: Optional[jax.random.PRNGKey] = None,
    max_len: int = 20,
    temperature: float = 1.0,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
) -> str:
    """Generate text from prompt using model state.

    Args:
        state: TrainState with model params and apply_fn
        prompt: Input text prompt
        tokenizer: Tokenizer with encode/decode methods
        rng: JAX PRNGKey for sampling
        max_len: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling cutoff
        repetition_penalty: Repetition penalty factor

    Returns:
        Generated text string
    """
    return generate_fast(
        state,
        prompt,
        tokenizer,
        rng=rng,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        verbose=False,
    )


def warmup_generation(
    state,
    tokenizer,
    max_len: int = 5,
    verbose: bool = True,
) -> float:
    """
    Warm up generation by running a short generation.

    This triggers any lazy initialization in the model's apply_fn
    so subsequent generations are faster.

    Args:
        state: TrainState with model params
        tokenizer: Tokenizer
        max_len: Short generation length for warmup
        verbose: Print timing info

    Returns:
        Time taken for warmup generation
    """
    dummy_prompt = "test"
    rng = jax.random.PRNGKey(42)

    start = time.time()
    _ = generate_fast(
        state,
        dummy_prompt,
        tokenizer,
        rng=rng,
        max_len=max_len,
        temperature=1.0,
        verbose=False,
    )
    jax.effects_barrier()
    elapsed = time.time() - start

    if verbose:
        print(f"Generation warmup: {elapsed:.2f}s")

    return elapsed
