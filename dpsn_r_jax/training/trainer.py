import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax import struct, traverse_util
import optax
from typing import Any
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.training.sparse_adam import sparse_adam_update


class TrainState(train_state.TrainState):
    rng: Any
    pool_m: jnp.ndarray
    pool_v: jnp.ndarray
    window_size: int = struct.field(pytree_node=False)
    learning_rate: float = struct.field(pytree_node=False)


def create_train_state(rng, config):
    model = DPSNR(config)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables["params"]

    flat_params = traverse_util.flatten_dict(params)
    pool_key = ("pool", "params_storage")
    pool_params = flat_params[pool_key]

    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    tx = optax.adamw(learning_rate=config.learning_rate)
    opt_state = tx.init(dense_params)

    pool_m = jnp.zeros_like(pool_params)
    pool_v = jnp.zeros_like(pool_params)

    return TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        rng=rng,
        pool_m=pool_m,
        pool_v=pool_v,
        window_size=config.max_k,
        learning_rate=config.learning_rate,
    )


@jax.jit
def train_step(state, batch, pad_token_id=0):
    dropout_rng, new_rng = random.split(state.rng)

    def loss_fn(params):
        logits, (_, indices) = state.apply_fn(
            {"params": params},
            batch,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )

        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]

        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )

        mask = (shift_labels != pad_token_id).astype(jnp.float32)
        loss = (loss * mask).sum() / (mask.sum() + 1e-9)

        return loss, indices

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, indices), grads = grad_fn(state.params)

    pool_key = ("pool", "params_storage")
    flat_params = traverse_util.flatten_dict(state.params)
    flat_grads = traverse_util.flatten_dict(grads)

    pool_params = jnp.asarray(flat_params[pool_key])
    pool_grads = jnp.asarray(flat_grads[pool_key])

    dense_flat_grads = {k: v for k, v in flat_grads.items() if k != pool_key}
    dense_grads = traverse_util.unflatten_dict(dense_flat_grads)

    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    updates, new_opt_state = state.tx.update(dense_grads, state.opt_state, dense_params)
    new_dense_params = optax.apply_updates(dense_params, updates)

    B, L = indices.shape
    W = state.window_size
    flat_touched = (indices[:, :, None] + jnp.arange(W)).reshape(-1)
    unique_indices = jnp.unique(flat_touched, size=B * L * W, fill_value=-1)

    valid_mask = unique_indices != -1
    safe_indices = jnp.where(valid_mask, unique_indices, 0)

    p_slice = pool_params[safe_indices]
    g_slice = pool_grads[safe_indices]
    m_slice = state.pool_m[safe_indices]
    v_slice = state.pool_v[safe_indices]

    new_p_s, new_m_s, new_v_s = sparse_adam_update(
        p_slice,
        g_slice,
        m_slice,
        v_slice,
        state.step + 1,
        lr=state.learning_rate,
    )

    new_pool_params = pool_params.at[safe_indices].set(
        jnp.where(valid_mask[:, None], new_p_s, p_slice)
    )
    new_pool_m = state.pool_m.at[safe_indices].set(
        jnp.where(valid_mask[:, None], new_m_s, m_slice)
    )
    new_pool_v = state.pool_v.at[safe_indices].set(
        jnp.where(valid_mask[:, None], new_v_s, v_slice)
    )

    new_flat_params = traverse_util.flatten_dict(new_dense_params)
    new_flat_params[pool_key] = new_pool_params
    new_params = traverse_util.unflatten_dict(new_flat_params)

    state = state.replace(
        step=state.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        pool_m=new_pool_m,
        pool_v=new_pool_v,
        rng=new_rng,
    )

    return state, loss
