"""Learning rate schedules for fine-tuning with JIT-compatible control flow."""

import jax
import jax.numpy as jnp
from typing import Callable
import optax


def create_linear_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        step = jnp.array(step, dtype=jnp.int32)
        warmup_steps_arr = jnp.array(warmup_steps, dtype=jnp.int32)

        def warmup_fn(_):
            return learning_rate * (step + 1.0) / warmup_steps

        def decay_fn(_):
            progress = (step - warmup_steps_arr) / max(1, total_steps - warmup_steps)
            progress = jnp.minimum(jnp.maximum(progress, 0.0), 1.0)
            return learning_rate * (1.0 - progress)

        return jax.lax.cond(
            step < warmup_steps_arr,
            warmup_fn,
            decay_fn,
            None,
        )

    return schedule


def create_cosine_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        step = jnp.array(step, dtype=jnp.int32)
        warmup_steps_arr = jnp.array(warmup_steps, dtype=jnp.int32)

        def warmup_fn(_):
            return learning_rate * (step + 1.0) / warmup_steps

        def decay_fn(_):
            progress = (step - warmup_steps_arr) / max(1, total_steps - warmup_steps)
            progress = jnp.minimum(jnp.maximum(progress, 0.0), 1.0)
            cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
            return min_lr + (learning_rate - min_lr) * cosine_factor

        return jax.lax.cond(
            step < warmup_steps_arr,
            warmup_fn,
            decay_fn,
            None,
        )

    return schedule


def create_cosine_with_hard_restarts_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    num_cycles: int = 1,
    min_lr: float = 0.0,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        step = jnp.array(step, dtype=jnp.int32)
        warmup_steps_arr = jnp.array(warmup_steps, dtype=jnp.int32)

        def warmup_fn(_):
            return learning_rate * (step + 1.0) / warmup_steps

        def decay_fn(_):
            cycle_length = (total_steps - warmup_steps) / num_cycles
            cycle_progress = (
                jnp.fmod(step - warmup_steps_arr, cycle_length) / cycle_length
            )
            cosine_factor = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
            return min_lr + (learning_rate - min_lr) * cosine_factor

        return jax.lax.cond(
            step < warmup_steps_arr,
            warmup_fn,
            decay_fn,
            None,
        )

    return schedule


def create_polynomial_schedule(
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    power: float = 1.0,
    min_lr: float = 0.0,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        step = jnp.array(step, dtype=jnp.int32)
        warmup_steps_arr = jnp.array(warmup_steps, dtype=jnp.int32)

        def warmup_fn(_):
            return learning_rate * (step + 1.0) / warmup_steps

        def decay_fn(_):
            progress = (step - warmup_steps_arr) / max(1, total_steps - warmup_steps)
            progress = jnp.minimum(jnp.maximum(progress, 0.0), 1.0)
            return min_lr + (learning_rate - min_lr) * (1.0 - progress) ** power

        return jax.lax.cond(
            step < warmup_steps_arr,
            warmup_fn,
            decay_fn,
            None,
        )

    return schedule


def create_constant_schedule(learning_rate: float) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        return learning_rate

    return schedule


def create_constant_with_warmup_schedule(
    learning_rate: float,
    warmup_steps: int,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        step = jnp.array(step, dtype=jnp.int32)
        warmup_steps_arr = jnp.array(warmup_steps, dtype=jnp.int32)

        def warmup_fn(_):
            return learning_rate * (step + 1.0) / warmup_steps

        def constant_fn(_):
            return learning_rate

        return jax.lax.cond(
            step < warmup_steps_arr,
            warmup_fn,
            constant_fn,
            None,
        )

    return schedule


def create_inverse_sqrt_schedule(
    learning_rate: float,
    warmup_steps: int,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        step = jnp.array(step, dtype=jnp.int32)
        warmup_steps_arr = jnp.array(warmup_steps, dtype=jnp.int32)

        def warmup_fn(_):
            return learning_rate * (step + 1.0) / warmup_steps

        def decay_fn(_):
            return learning_rate * (warmup_steps**0.5) / ((step + 1.0) ** 0.5)

        return jax.lax.cond(
            step < warmup_steps_arr,
            warmup_fn,
            decay_fn,
            None,
        )

    return schedule


def get_scheduler(
    scheduler_type: str,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    **kwargs,
) -> Callable[[int], float]:
    schedulers = {
        "linear": create_linear_schedule,
        "cosine": create_cosine_schedule,
        "cosine_with_restarts": create_cosine_with_hard_restarts_schedule,
        "polynomial": create_polynomial_schedule,
        "constant": lambda lr, ws, ts, **kw: create_constant_schedule(lr),
        "constant_with_warmup": create_constant_with_warmup_schedule,
        "inverse_sqrt": create_inverse_sqrt_schedule,
    }

    scheduler_type = scheduler_type.lower()
    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: {list(schedulers.keys())}"
        )

    return schedulers[scheduler_type](
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        **kwargs,
    )
