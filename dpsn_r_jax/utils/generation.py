import jax.numpy as jnp


def generate(state, prompt, tokenizer, max_len=20):
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    generated = input_ids

    for _ in range(max_len):
        logits, _ = state.apply_fn(
            {"params": state.params}, generated, deterministic=True
        )
        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

        if next_token[0] == tokenizer.eos_token_id:
            break

    generated_list = generated[0].tolist()
    return tokenizer.decode(generated_list, skip_special_tokens=True)
