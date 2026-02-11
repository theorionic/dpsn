import jax
import jax.numpy as jnp
import optax
import numpy as np
from dpsn_r_jax.training.sparse_adam import sparse_adam_update


def test_sparse_adam():
    pool_shape = (1000, 64)
    key = jax.random.PRNGKey(0)

    k1, k2, k3, k4 = jax.random.split(key, 4)
    params = jax.random.normal(k1, pool_shape)
    grads = jax.random.normal(k2, pool_shape)
    m = jax.random.normal(k3, pool_shape) * 0.1
    v = jnp.abs(jax.random.normal(k4, pool_shape)) * 0.1

    indices = jnp.array([5, 100, 999])
    step = 1
    lr = 1e-3
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    optimizer = optax.adam(lr, b1, b2, eps)
    adam_state = optax.ScaleByAdamState(
        count=jnp.array(step - 1, dtype=jnp.int32), mu=m, nu=v
    )
    state = (adam_state, optax.EmptyState())

    updates, new_state_tuple = optimizer.update(grads, state, params)
    new_adam_state = new_state_tuple[0]
    params_new_full = optax.apply_updates(params, updates)

    params_slice = params[indices]
    grads_slice = grads[indices]
    m_slice = m[indices]
    v_slice = v[indices]

    params_new_slice, m_new_slice, v_new_slice = sparse_adam_update(
        params_slice, grads_slice, m_slice, v_slice, step, lr, b1, b2, eps
    )

    assert jnp.allclose(params_new_slice, params_new_full[indices]), (
        "Params slice mismatch"
    )
    assert jnp.allclose(m_new_slice, new_adam_state.mu[indices]), "M slice mismatch"
    assert jnp.allclose(v_new_slice, new_adam_state.nu[indices]), "V slice mismatch"

    print("SUCCESS: Sparse Adam update matches full Adam update for selected indices.")
    print(f"Indices processed: {indices.tolist()}")
    print(f"Shape of slices: {params_slice.shape}")


if __name__ == "__main__":
    test_sparse_adam()
