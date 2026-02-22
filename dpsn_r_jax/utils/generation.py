"""Text generation with cached JIT compilation for TPU."""

import jax
import jax.numpy as jnp
import time
from typing import Optional
import os


# Enable JAX compilation logging if requested
# Set JAX_LOG_COMPILES=1 to see when compilation happens
def _log_compilation(msg: str):
    """Log compilation events if JAX_LOG_COMPILES is set."""
    if os.environ.get("JAX_LOG_COMPILES", "").lower() in ("1", "true"):
        print(f"[JAX COMPILE] {msg}")


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


# Cached forward function - JIT'd once per (batch_shape, seq_len) combination
_CACHED_FORWARD_FN = {}
_CACHE_KEY = None


def _get_forward_fn(state, batch_size: int, seq_len: int):
    """Get or create a JIT-compiled forward function for the given shape.

    JAX caches JIT'd functions by the shape of inputs. This caches the compiled
    forward pass so we don't recompile for the same shape.
    """
    global _CACHED_FORWARD_FN, _CACHE_KEY

    # Use state identity + shape as cache key
    state_id = id(state)
    cache_key = (state_id, batch_size, seq_len)

    if cache_key != _CACHE_KEY:
        _log_compilation(f"New cache key: batch_size={batch_size}, seq_len={seq_len}")
        _CACHE_KEY = cache_key

    if cache_key not in _CACHED_FORWARD_FN:
        _log_compilation(f"Compiling forward pass for shape ({batch_size}, {seq_len})")

        @jax.jit
        def forward_fn(params, input_ids):
            _log_compilation(f"Forward pass JIT executing for shape {input_ids.shape}")
            logits, aux = state.apply_fn(
                {"params": params}, input_ids, deterministic=True
            )
            return logits, aux

        _CACHED_FORWARD_FN[cache_key] = forward_fn
    else:
        _log_compilation(
            f"Using cached forward pass for shape ({batch_size}, {seq_len})"
        )

    return _CACHED_FORWARD_FN[cache_key]


def generate_fast(
    state,
    prompt: str,
    tokenizer,
    rng: Optional[jax.random.PRNGKey] = None,
    max_len: int = 20,
    temperature: float = 1.0,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
    max_seq_len: int = 512,
    verbose: bool = False,
) -> str:
    """Generate text with padded inputs for cached JIT compilation.

    CRITICAL for TPU: We pad inputs to fixed max_seq_len so that the JIT-compiled
    forward pass is reused (cached) instead of recompiling for each new sequence length.
    Without this, every token generation step would trigger a new XLA compilation.

    Args:
        state: TrainState with model params and apply_fn
        prompt: Input text prompt
        tokenizer: Tokenizer with encode/decode methods
        rng: JAX PRNGKey for sampling
        max_len: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling cutoff
        repetition_penalty: Repetition penalty factor
        max_seq_len: Fixed sequence length for padding (use model's max_seq_len)
        verbose: Print timing and compilation info

    Returns:
        Generated text string
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Encode prompt
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

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    prompt_len = input_ids.shape[1]
    if prompt_len > max_seq_len:
        if verbose:
            print(
                f"[Warning] Prompt ({prompt_len}) exceeds max_seq_len ({max_seq_len}), truncating"
            )
        input_ids = input_ids[:, :max_seq_len]
        prompt_len = max_seq_len

    start_time = time.time()
    generated_tokens = []

    # Get JIT'd forward function for fixed shape (1, max_seq_len)
    forward_fn = _get_forward_fn(state, batch_size=1, seq_len=max_seq_len)

    for step in range(max_len):
        current_len = prompt_len + step
        if current_len >= max_seq_len:
            break

        # Pad input to fixed max_seq_len for cache hit
        padded_input = jnp.zeros((1, max_seq_len), dtype=jnp.int32)
        padded_input = padded_input.at[:, : input_ids.shape[1]].set(input_ids)

        # Create attention mask for actual length
        # (some models may not need this, but DPSNR uses it)

        _log_compilation(
            f"Step {step}: calling forward_fn with shape {padded_input.shape}"
        )

        # Call cached JIT function
        logits, _ = forward_fn(state.params, padded_input)

        # Get logits for last actual token (not padded position)
        next_logits = logits[0, current_len - 1, :]

        # Apply sampling
        if generated_tokens:
            next_logits = _apply_repetition_penalty(
                next_logits,
                jnp.array(generated_tokens, dtype=jnp.int32),
                repetition_penalty,
            )
        next_logits = _apply_top_k(next_logits, top_k)

        rng, sample_rng = jax.random.split(rng)
        new_token = _sample_token(next_logits, sample_rng, temperature)
        generated_tokens.append(int(new_token))

        # Append to input for next iteration
        input_ids = jnp.concatenate([input_ids, new_token[None, None]], axis=1)

        if new_token == eos_token_id:
            break

    if verbose:
        jax.effects_barrier()
        elapsed = time.time() - start_time
        print(f"Generation time: {elapsed:.3f}s (max_len={max_len})")

    # Decode only the generated tokens (not the prompt)
    final_tokens = input_ids[0].tolist()
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
    max_seq_len: int = 512,
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
        max_seq_len: Fixed sequence length for padding (prevents recompilation)

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
        max_seq_len=max_seq_len,
        verbose=False,
    )


def warmup_generation(
    state,
    tokenizer,
    max_len: int = 5,
    max_seq_len: int = 512,
    verbose: bool = True,
) -> float:
    """Warm up generation to trigger initial JIT compilation.

    Call this once before training to pre-compile the forward pass.
    Subsequent generations will use the cached compilation.

    Args:
        state: TrainState with model params
        tokenizer: Tokenizer
        max_len: Short generation length for warmup
        max_seq_len: Fixed sequence length for padding
        verbose: Print timing and compilation info

    Returns:
        Time taken for warmup generation
    """
    dummy_prompt = "test"
    rng = jax.random.PRNGKey(42)

    _log_compilation("Warmup: Starting JIT compilation")
    start = time.time()

    _ = generate_fast(
        state,
        dummy_prompt,
        tokenizer,
        rng=rng,
        max_len=max_len,
        temperature=1.0,
        max_seq_len=max_seq_len,
        verbose=False,
    )

    jax.effects_barrier()
    elapsed = time.time() - start

    if verbose:
        print(
            f"Generation warmup: {elapsed:.2f}s (compiled forward pass for shape (1, {max_seq_len}))"
        )

    return elapsed
