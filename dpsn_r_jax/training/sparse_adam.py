import jax
import jax.numpy as jnp
from typing import Tuple


def sparse_adam_update(
    params_slice: jnp.ndarray,
    grads_slice: jnp.ndarray,
    m_slice: jnp.ndarray,
    v_slice: jnp.ndarray,
    step: int,
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Standard Adam update applied to slices of parameters and optimizer states.

    Args:
        params_slice: Current parameter values (slice).
        grads_slice: Gradients for the parameters (slice).
        m_slice: First moment state (slice).
        v_slice: Second moment state (slice).
        step: Current training step (1-indexed).
        lr: Learning rate.
        b1: Exponential decay rate for the first moment.
        b2: Exponential decay rate for the second moment.
        eps: Small constant for numerical stability.

    Returns:
        A tuple of (updated_params_slice, updated_m_slice, updated_v_slice).
    """
    m_new = b1 * m_slice + (1 - b1) * grads_slice
    v_new = b2 * v_slice + (1 - b2) * (grads_slice**2)

    m_hat = m_new / (1 - b1**step)
    v_hat = v_new / (1 - b2**step)

    params_new = params_slice - lr * m_hat / (jnp.sqrt(v_hat) + eps)

    return params_new, m_new, v_new
