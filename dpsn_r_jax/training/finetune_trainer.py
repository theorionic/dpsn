import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax import struct, traverse_util
import optax
from typing import Any, Optional, Dict, Callable, Tuple
import os
import json
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.training.sparse_adam import sparse_adam_update
from dpsn_r_jax.training.lr_schedules import get_scheduler

IGNORE_INDEX = -100


class FineTuneState(train_state.TrainState):
    rng: Any
    pool_m: jnp.ndarray
    pool_v: jnp.ndarray
    window_size: int = struct.field(pytree_node=False)
    learning_rate_fn: Callable[[int], float] = struct.field(pytree_node=False)


def get_frozen_mask(
    params: Dict, freeze_controller: bool = False, freeze_pool: bool = True
) -> Dict:
    flat_params = traverse_util.flatten_dict(params)
    flat_mask = {}

    for key in flat_params:
        key_str = "/".join(str(k) for k in key)

        if "pool" in key_str and freeze_pool:
            flat_mask[key] = "freeze"
        elif (
            any(x in key_str for x in ["controller", "encoder", "decoder"])
            and freeze_controller
        ):
            flat_mask[key] = "freeze"
        else:
            flat_mask[key] = "train"

    return traverse_util.unflatten_dict(flat_mask)


def create_finetune_state(
    rng,
    config,
    learning_rate_fn: Callable[[int], float],
    freeze_controller: bool = False,
    freeze_pool: bool = True,
    pretrained_path: Optional[str] = None,
) -> FineTuneState:
    model = DPSNR(config)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    params = variables["params"]

    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        with open(os.path.join(pretrained_path, "params.json"), "r") as f:
            params_dict = json.load(f)
        params = jax.tree_util.tree_map(jnp.array, params_dict)

    flat_params = traverse_util.flatten_dict(params)
    pool_key = ("pool", "params_storage")
    pool_params = flat_params[pool_key]

    dense_flat_params = {k: v for k, v in flat_params.items() if k != pool_key}
    dense_params = traverse_util.unflatten_dict(dense_flat_params)

    if freeze_controller or freeze_pool:
        freeze_mask = get_frozen_mask(dense_params, freeze_controller, freeze_pool)
        tx = optax.multi_transform(
            {
                "train": optax.adamw(learning_rate=learning_rate_fn(0)),
                "freeze": optax.set_to_zero(),
            },
            freeze_mask,
        )
        opt_state = tx.init(dense_params)
    else:
        tx = optax.adamw(learning_rate=learning_rate_fn(0))
        opt_state = tx.init(dense_params)

    pool_m = jnp.zeros_like(pool_params)
    pool_v = jnp.zeros_like(pool_params)

    return FineTuneState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=opt_state,
        rng=rng,
        pool_m=pool_m,
        pool_v=pool_v,
        window_size=config.max_k,
        learning_rate_fn=learning_rate_fn,
    )


@jax.jit
def finetune_step(
    state: FineTuneState,
    batch: Dict[str, jnp.ndarray],
    pad_token_id: int = 0,
) -> Tuple[FineTuneState, jnp.ndarray]:
    dropout_rng, new_rng = random.split(state.rng)

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch.get("attention_mask", None)

    current_lr = state.learning_rate_fn(state.step)

    def loss_fn(params):
        logits, (_, indices) = state.apply_fn(
            {"params": params},
            input_ids,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )

        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        )

        valid_mask = (shift_labels != IGNORE_INDEX).astype(jnp.float32)
        pad_mask = (shift_labels != pad_token_id).astype(jnp.float32)
        combined_mask = valid_mask * pad_mask

        loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-9)

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
        lr=current_lr,
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


@jax.jit
def compute_loss_with_mask(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    pad_token_id: int = 0,
) -> jnp.ndarray:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    loss = optax.softmax_cross_entropy_with_integer_labels(shift_logits, shift_labels)

    valid_mask = (shift_labels != IGNORE_INDEX).astype(jnp.float32)
    pad_mask = (shift_labels != pad_token_id).astype(jnp.float32)
    combined_mask = valid_mask * pad_mask

    return (loss * combined_mask).sum() / (combined_mask.sum() + 1e-9)


@jax.jit
def validation_step(
    state: FineTuneState,
    batch: Dict[str, jnp.ndarray],
    pad_token_id: int = 0,
) -> jnp.ndarray:
    logits, _ = state.apply_fn(
        {"params": state.params},
        batch["input_ids"],
        deterministic=True,
    )

    return compute_loss_with_mask(logits, batch["labels"], pad_token_id)


def save_checkpoint(state: FineTuneState, path: str, step: int):
    os.makedirs(path, exist_ok=True)

    params_dict = jax.tree_util.tree_map(lambda x: x.tolist(), state.params)
    with open(os.path.join(path, f"params_step_{step}.json"), "w") as f:
        json.dump(params_dict, f)

    pool_m_dict = state.pool_m.tolist()
    pool_v_dict = state.pool_v.tolist()

    checkpoint = {
        "step": step,
        "pool_m": pool_m_dict,
        "pool_v": pool_v_dict,
    }
    with open(os.path.join(path, f"checkpoint_{step}.json"), "w") as f:
        json.dump(checkpoint, f)

    print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(state: FineTuneState, path: str, step: int) -> FineTuneState:
    with open(os.path.join(path, f"params_step_{step}.json"), "r") as f:
        params_dict = json.load(f)
    params = jax.tree_util.tree_map(jnp.array, params_dict)

    with open(os.path.join(path, f"checkpoint_{step}.json"), "r") as f:
        checkpoint = json.load(f)

    pool_m = jnp.array(checkpoint["pool_m"])
    pool_v = jnp.array(checkpoint["pool_v"])

    return state.replace(
        step=checkpoint["step"],
        params=params,
        pool_m=pool_m,
        pool_v=pool_v,
    )
