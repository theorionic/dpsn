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


def _create_step_fn(
    state,
    eos_token_id: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
):
    """Create a JIT-compiled step function with captured static args."""

    @jax.jit
    def step_fn(carry):
        generated_ids, rng = carry

        logits, _ = state.apply_fn(
            {"params": state.params}, generated_ids, deterministic=True
        )
        next_logits = logits[0, -1, :]
        next_logits = _apply_repetition_penalty(
            next_logits, generated_ids[0], repetition_penalty
        )
        next_logits = _apply_top_k(next_logits, top_k)

        rng, sample_rng = jax.random.split(rng)
        new_token = _sample_token(next_logits, sample_rng, temperature)

        new_generated = jnp.concatenate([generated_ids, new_token[None, None]], axis=1)
        return (new_generated, rng), new_token, new_token == eos_token_id

    return step_fn


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
    Fast generation with JIT-compiled step function.

    Uses a Python loop with JIT-compiled individual steps for optimal
    performance while supporting early EOS termination.
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

    step_fn = _create_step_fn(
        state, eos_token_id, temperature, top_k, repetition_penalty
    )

    start_time = time.time()
    carry = (generated, rng)

    for _ in range(max_len):
        carry, new_token, hit_eos = step_fn(carry)
        if hit_eos:
            break

    if verbose:
        jax.effects_barrier()
        elapsed = time.time() - start_time
        print(f"Generation time: {elapsed:.3f}s")

    final_tokens = carry[0][0].tolist()
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
    max_len: int = 20,
    verbose: bool = True,
) -> float:
    """
    Warm up the generation function by running a dummy generation.
    Triggers JIT compilation so subsequent calls are fast.
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
        print(f"Generation warmup (JIT compile): {elapsed:.2f}s")

    return elapsed
