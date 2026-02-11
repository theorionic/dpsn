import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
import optax
from typing import Any
from dpsn_r_jax.models.dpsnr import DPSNR


class TrainState(train_state.TrainState):
    rng: Any


def create_train_state(rng, config):
    model = DPSNR(config)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)

    tx = optax.adamw(learning_rate=config.learning_rate)

    return TrainState.create(
        apply_fn=model.apply, params=variables["params"], tx=tx, rng=rng
    )


@jax.jit
def train_step(state, batch, pad_token_id=0):
    dropout_rng, new_rng = random.split(state.rng)

    def loss_fn(params):
        logits, _ = state.apply_fn(
            {"params": params},
            batch,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )

        # Shift logits and labels
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]

        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )

        # Mask padding
        mask = (shift_labels != pad_token_id).astype(jnp.float32)
        loss = (loss * mask).sum() / (mask.sum() + 1e-9)

        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(rng=new_rng)
    return state, loss
