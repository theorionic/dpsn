import jax
import jax.numpy as jnp
import optax
import time
from dpsn_r_jax.training.sparse_adam import sparse_adam_update


def benchmark():
    pool_size = (1_000_000, 64)
    active_rows = 64
    step = 1
    lr = 1e-3

    print(f"Scenario: Pool Size {pool_size}, Active Rows {active_rows}")

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    params = jax.random.normal(k1, pool_size)
    grads_full = jax.random.normal(k2, pool_size)
    m_full = jax.random.normal(k3, pool_size) * 0.1
    v_full = jnp.abs(jax.random.normal(k4, pool_size)) * 0.1

    indices = jnp.arange(active_rows)
    params_slice = params[indices]
    grads_slice = grads_full[indices]
    m_slice = m_full[indices]
    v_slice = v_full[indices]

    optimizer = optax.adam(lr)
    adam_state = optax.ScaleByAdamState(
        count=jnp.array(step - 1, dtype=jnp.int32), mu=m_full, nu=v_full
    )
    state = (adam_state, optax.EmptyState())

    @jax.jit
    def full_update_fn(p, g, s):
        return optimizer.update(g, s, p)

    @jax.jit
    def sparse_update_fn(ps, gs, ms, vs, st):
        return sparse_adam_update(ps, gs, ms, vs, st, lr)

    print("Warming up JIT...")
    _, _ = full_update_fn(params, grads_full, state)
    _, _, _ = sparse_update_fn(params_slice, grads_slice, m_slice, v_slice, step)

    n_iters = 100
    print(f"Benchmarking Full Update ({n_iters} iterations)...")
    jax.block_until_ready(params)

    start_time_full = time.perf_counter()
    for _ in range(n_iters):
        updates, _ = full_update_fn(params, grads_full, state)
        jax.block_until_ready(updates)
    end_time_full = time.perf_counter()
    avg_full_time = (end_time_full - start_time_full) / n_iters
    print(f"Full Adam Update Time: {avg_full_time * 1000:.4f} ms")

    print(f"Benchmarking Sparse Update ({n_iters} iterations)...")
    start_time_sparse = time.perf_counter()
    for _ in range(n_iters):
        result = sparse_update_fn(params_slice, grads_slice, m_slice, v_slice, step)
        jax.block_until_ready(result)
    end_time_sparse = time.perf_counter()
    avg_sparse_time = (end_time_sparse - start_time_sparse) / n_iters
    print(f"Sparse Adam Update Time: {avg_sparse_time * 1000:.4f} ms")

    speedup_ratio = avg_full_time / avg_sparse_time
    print(f"\nSpeedup Factor: {speedup_ratio:.2f}x")


if __name__ == "__main__":
    benchmark()
