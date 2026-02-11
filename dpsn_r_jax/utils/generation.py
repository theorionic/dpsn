import jax
import jax.numpy as jnp


def generate(
    state,
    prompt,
    tokenizer,
    rng=None,
    max_len=20,
    temperature=1.0,
    top_k=40,
    repetition_penalty=1.2,
):
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    generated = jnp.array(input_ids)

    for _ in range(max_len):
        logits, _ = state.apply_fn(
            {"params": state.params}, generated, deterministic=True
        )
        next_token_logits = logits[0, -1, :]

        # Repetition Penalty
        if repetition_penalty != 1.0:
            mask = jnp.zeros_like(next_token_logits, dtype=jnp.bool_)
            mask = mask.at[generated[0]].set(True)

            penalized_logits = jnp.where(
                next_token_logits > 0,
                next_token_logits / repetition_penalty,
                next_token_logits * repetition_penalty,
            )
            next_token_logits = jnp.where(mask, penalized_logits, next_token_logits)

        # Temperature Scaling
        if temperature != 1.0 and temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Top-K filtering
        if top_k > 0:
            k = min(top_k, next_token_logits.shape[-1])
            values, _ = jax.lax.top_k(next_token_logits, k=k)
            min_value = values[-1]
            next_token_logits = jnp.where(
                next_token_logits < min_value,
                jnp.full_like(next_token_logits, -1e10),
                next_token_logits,
            )

        # Sampling
        if temperature > 0 and rng is not None:
            rng, sample_rng = jax.random.split(rng)
            next_token = jax.random.categorical(sample_rng, next_token_logits)
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)
        next_token = jnp.reshape(next_token, (1,))
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

        if next_token[0] == tokenizer.eos_token_id:
            break

    generated_list = generated[0].tolist()
    return tokenizer.decode(generated_list, skip_special_tokens=True)
