"""DWA training step — jax.jit + optax + nnx.split/merge.

Model architecture (dwt/dwa_model.py) is UNTOUCHED.
Only training mechanism: nnx.Optimizer → optax + jax.jit, same as DPSN-R.

Pattern:
    graphdef, nnx_state = nnx.split(model)      # once, at creation
    state = DWATrainState(graphdef, nnx_state, opt_state, alpha_ema)
    train_step = jax.jit(make_dwa_train_step(cfg, graphdef))
    ...
    # inside step:
    model = nnx.merge(graphdef, nnx_state)      # reconstruct
    logits, alpha, keys, w_norm = model(x, lambda_sharp, temp)
"""

from __future__ import annotations

import json
import os
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct


# ===========================================================================
# 1. TrainState (flax.struct pytree — registered with JAX)
# ===========================================================================

class DWATrainState(struct.PyTreeNode):
    """Pytree train state for DWA.

    graphdef is marked non-pytree (static) so jax.jit closes over the module
    structure while nnx_state / opt_state / alpha_ema are traced as leaves.
    """
    step: jax.Array
    nnx_state: Any
    opt_state: Any
    alpha_ema: jax.Array
    graphdef: Any = struct.field(pytree_node=False)

    @classmethod
    def create(cls, model, tx, N: int) -> "DWATrainState":
        import flax.nnx as nnx
        graphdef, nnx_state = nnx.split(model)
        opt_state = tx.init(nnx_state)
        alpha_ema = jnp.ones((N,), dtype=jnp.float32) / float(N)
        return cls(
            step=jnp.array(0, dtype=jnp.int32),
            nnx_state=nnx_state,
            opt_state=opt_state,
            alpha_ema=alpha_ema,
            graphdef=graphdef,
        )


# ===========================================================================
# 2. Phase helpers (host-side Python; pass results as jax scalars into step)
# ===========================================================================

def _get_lambda_sharp(step: int, cfg) -> float:
    """Sigmoid sharpness lambda for the current step (0 / ramp / ramp).

    Phase 1 [0, phase1_end): lambda=0  → pure softmax over top-k.
    Phase 2 [phase1_end, phase2_end): lambda ramps 0 → lambda_sharp_phase2_end.
    Phase 3 [phase2_end, ∞):  lambda ramps to lambda_sharp_final.
    """
    if step < cfg.phase1_end:
        return 0.0
    if step < cfg.phase2_end:
        t = (step - cfg.phase1_end) / max(1, (cfg.phase2_end - cfg.phase1_end))
        return float(t * cfg.lambda_sharp_phase2_end)
    t = min(1.0, (step - cfg.phase2_end) / max(1, cfg.phase2_end))
    return float(
        cfg.lambda_sharp_phase2_end
        + t * (cfg.lambda_sharp_final - cfg.lambda_sharp_phase2_end)
    )


def _get_aux_scale(step: int, cfg) -> float:
    """Auxiliary losses off during phase 1 warmup, on thereafter."""
    return 0.0 if step < cfg.phase1_end else 1.0


def _phase_name(step: int, cfg) -> str:
    if step < cfg.phase1_end:
        return "warmup"
    if step < cfg.phase2_end:
        return "gate_on"
    return "sharpen"


def _phase_idx(step: int, cfg) -> int:
    if step < cfg.phase1_end:
        return 0
    if step < cfg.phase2_end:
        return 1
    return 2


# ===========================================================================
# 3. Optimizer
# ===========================================================================

def make_lr_schedule(cfg):
    """Linear warmup + cosine decay — same pattern as DPSN-R."""
    warmup = min(int(cfg.warmup_steps), max(1, int(cfg.max_steps) - 1))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg.lr),
        warmup_steps=warmup,
        decay_steps=int(cfg.max_steps),
        end_value=float(cfg.lr) * 0.1,
    )


def make_tx(cfg):
    """AdamW with grad-clip, LR follows warmup_cosine_decay schedule.

    Betas (0.9, 0.95) match the DWA-paper recommendation for assembly models.
    """
    schedule = make_lr_schedule(cfg)
    return optax.chain(
        optax.clip_by_global_norm(float(cfg.grad_clip)),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=float(cfg.weight_decay),
            b1=0.9,
            b2=0.95,
        ),
    )


# ===========================================================================
# 4. Loss helpers (locally defined to avoid dwa_model import inside jit)
# ===========================================================================

def _cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Standard causal LM cross-entropy.

    logits  : [B, T, V]
    targets : [B, T]
    """
    B, T, V = logits.shape
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(B * T, V),
        targets.reshape(B * T),
    ).mean()


def _utilization_loss(alpha_ema: jax.Array, beta: float) -> jax.Array:
    """Prevent dead pool vectors via stable expm1 formulation.

    L_util = -mean_i log(1 - exp(-beta * EMA(alpha_i)))
    """
    ema = jnp.clip(alpha_ema, 1e-6, None)
    stable = -jnp.expm1(-beta * ema)
    return -jnp.mean(jnp.log(jnp.maximum(stable, 1e-8)))


def _diversity_loss(alpha: jax.Array, keys: jax.Array) -> jax.Array:
    """Penalise cosine similarity between pool key vectors (weighted by utilisation).

    alpha : [B*T, N]
    keys  : [N, S, d_k]
    """
    N, S, d_k = keys.shape
    alpha_mean = jnp.mean(alpha, axis=0)                              # [N]
    keys_flat = keys.reshape(N, S * d_k)
    k_norm = keys_flat / (jnp.linalg.norm(keys_flat, axis=-1, keepdims=True) + 1e-8)
    sim = jnp.einsum("id,jd->ij", k_norm, k_norm)
    outer = jnp.outer(alpha_mean, alpha_mean)
    off_diag = 1.0 - jnp.eye(N)
    return jnp.sum(outer * sim * off_diag) / (N * (N - 1) + 1e-8)


def _sparsity_loss(alpha: jax.Array) -> jax.Array:
    """Entropy regularisation — encourage sparse assembly weights.

    alpha : [B*T, N]
    """
    eps = 1e-8
    return jnp.mean(-jnp.sum(alpha * jnp.log(alpha + eps), axis=-1))


# ===========================================================================
# 5. Training / eval step factories
# ===========================================================================

def make_dwa_train_step(cfg, graphdef):
    """Return a jax.jit compiled training step closed over cfg + graphdef.

    Signature:
        (state, batch, lambda_sharp, aux_scale) -> (new_state, total_loss, breakdown)

    batch: [B, T+1]  — integer token ids; x = batch[:, :-1], y = batch[:, 1:].
    """
    import flax.nnx as nnx

    tx = make_tx(cfg)
    N = int(cfg.N)
    ema_decay = float(cfg.ema_decay)
    beta_util = float(cfg.beta_util)
    lambda_util = float(cfg.lambda_util)
    lambda_div = float(cfg.lambda_div)
    lambda_norm = float(cfg.lambda_norm)
    lambda_sparse = float(cfg.lambda_sparse)

    @jax.jit
    def train_step(
        state: DWATrainState,
        batch: jax.Array,
        lambda_sharp: jax.Array,
        aux_scale: jax.Array,
    ):
        x = batch[:, :-1]
        y = batch[:, 1:]

        def loss_fn(nnx_state):
            model = nnx.merge(graphdef, nnx_state)
            logits, alpha, keys, w_norm = model(x, lambda_sharp, jnp.array(1.0))

            ce = _cross_entropy(logits, y)
            l_u = _utilization_loss(state.alpha_ema, beta_util)
            l_d = _diversity_loss(alpha, keys)
            l_s = _sparsity_loss(alpha)

            aux = (
                lambda_util * l_u
                + lambda_div * l_d
                + lambda_norm * w_norm
                + lambda_sparse * l_s
            )
            total = ce + aux_scale * aux

            breakdown = {
                "ce":     ce,
                "util":   l_u,
                "div":    l_d,
                "sparse": l_s,
                "norm":   w_norm,
                "aux":    aux,
                "total":  total,
            }
            return total, (breakdown, alpha)

        (total, (breakdown, alpha)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(state.nnx_state)

        updates, new_opt_state = tx.update(grads, state.opt_state, state.nnx_state)
        new_nnx_state = optax.apply_updates(state.nnx_state, updates)

        # EMA update of per-vector utilisation.
        batch_mean = jnp.mean(alpha.reshape(-1, N), axis=0)
        new_alpha_ema = ema_decay * state.alpha_ema + (1.0 - ema_decay) * batch_mean

        new_state = state.replace(
            step=state.step + 1,
            nnx_state=new_nnx_state,
            opt_state=new_opt_state,
            alpha_ema=new_alpha_ema,
        )
        return new_state, total, breakdown

    return train_step


def make_dwa_eval_step(graphdef):
    """Return a jax.jit compiled eval step: (nnx_state, batch) → scalar CE.

    Uses lambda_sharp = 0 (pure softmax, no gating) and temperature = 1.
    """
    import flax.nnx as nnx

    @jax.jit
    def eval_step(nnx_state, batch: jax.Array) -> jax.Array:
        x = batch[:, :-1]
        y = batch[:, 1:]
        model = nnx.merge(graphdef, nnx_state)
        logits, *_ = model(x, jnp.array(0.0), jnp.array(1.0))
        return _cross_entropy(logits, y)

    return eval_step


# ===========================================================================
# 6. Checkpoint save / load
# ===========================================================================
#
# Stores nnx_state + opt_state leaves by flat index into a single .npz file.
# We rely on jax.tree_util.tree_flatten returning leaves in a deterministic
# order for a given treedef, which is guaranteed by JAX — so the save and
# the matching load (given the same treedef) agree on ordering.

def save_dwa_checkpoint(path: str, state: DWATrainState, metadata: dict | None = None) -> None:
    """Save DWATrainState to `{path}/state.npz` + `{path}/meta.json`."""
    os.makedirs(path, exist_ok=True)

    m_leaves, _m_def = jax.tree_util.tree_flatten(state.nnx_state)
    o_leaves, _o_def = jax.tree_util.tree_flatten(state.opt_state)

    arrays: dict[str, np.ndarray] = {}
    for i, leaf in enumerate(m_leaves):
        arrays[f"m_{i:05d}"] = np.asarray(leaf)
    for i, leaf in enumerate(o_leaves):
        arrays[f"o_{i:05d}"] = np.asarray(leaf)
    arrays["alpha_ema"] = np.asarray(state.alpha_ema)
    arrays["step"] = np.asarray(state.step)

    np.savez(os.path.join(path, "state.npz"), **arrays)

    meta = {"step": int(state.step), "n_m_leaves": len(m_leaves), "n_o_leaves": len(o_leaves)}
    if metadata:
        meta.update(metadata)
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_dwa_checkpoint(path: str, state: DWATrainState) -> Tuple[DWATrainState, int]:
    """Restore a DWATrainState from a directory saved by save_dwa_checkpoint.

    Uses the treedef of the *current* state to unflatten — so `state` must
    have been created with the same model architecture and optimizer as the
    one that was saved.
    """
    npz_path = os.path.join(path, "state.npz")
    meta_path = os.path.join(path, "meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    data = np.load(npz_path)

    m_leaves_old, m_def = jax.tree_util.tree_flatten(state.nnx_state)
    o_leaves_old, o_def = jax.tree_util.tree_flatten(state.opt_state)

    new_m_leaves = []
    for i, ref in enumerate(m_leaves_old):
        arr = jnp.asarray(data[f"m_{i:05d}"])
        new_m_leaves.append(arr.astype(ref.dtype) if hasattr(ref, "dtype") else arr)

    new_o_leaves = []
    for i, ref in enumerate(o_leaves_old):
        arr = jnp.asarray(data[f"o_{i:05d}"])
        new_o_leaves.append(arr.astype(ref.dtype) if hasattr(ref, "dtype") else arr)

    new_nnx_state = jax.tree_util.tree_unflatten(m_def, new_m_leaves)
    new_opt_state = jax.tree_util.tree_unflatten(o_def, new_o_leaves)
    alpha_ema = jnp.asarray(data["alpha_ema"])
    step = int(meta["step"])

    new_state = state.replace(
        step=jnp.array(step, dtype=jnp.int32),
        nnx_state=new_nnx_state,
        opt_state=new_opt_state,
        alpha_ema=alpha_ema,
    )
    return new_state, step


def latest_dwa_checkpoint(ckpt_dir: str) -> str | None:
    """Return the highest-step `step_XXXXXX` checkpoint directory, or None."""
    if not os.path.isdir(ckpt_dir):
        return None
    dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("step_")]
    if not dirs:
        return None
    dirs.sort(key=lambda d: int(d.split("_")[1]))
    return os.path.join(ckpt_dir, dirs[-1])
