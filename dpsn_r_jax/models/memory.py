import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from dpsn_r_jax.config import PoolConfig


class LearnedIndexer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, query):
        x = nn.Dense(self.hidden_dim)(query)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.gelu(x)

        mu = nn.Dense(1)(x)
        mu = nn.sigmoid(mu)

        sigma = nn.Dense(1)(x)
        sigma = nn.softplus(sigma)

        return mu.squeeze(-1), sigma.squeeze(-1)


class CoordinateMassivePool(nn.Module):
    config: PoolConfig
    window_size: int

    def setup(self):
        self.params_storage = self.param(
            "params_storage",
            nn.initializers.normal(),
            (self.config.total_vectors, self.config.hidden_dim),
        )

    def __call__(self, mu, sigma):
        B = mu.shape[0]
        Total = self.config.total_vectors
        D = self.config.hidden_dim
        W = self.window_size

        center_idx = mu * (Total - 1)

        start_indices = jnp.clip(center_idx - W // 2, 0, Total - W).astype(jnp.int32)

        def slice_fn(start):
            return lax.dynamic_slice(self.params_storage, (start, 0), (W, D))

        selected = jax.vmap(slice_fn)(start_indices)

        relative_indices = jnp.arange(W)[None, :] + start_indices[:, None]

        distances = relative_indices - center_idx[:, None]

        weights = jnp.exp(-(distances**2) / (2 * (sigma[:, None] + 1e-6) ** 2))
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-6)

        aggregated = jnp.einsum("bw,bwd->bd", weights, selected)

        return aggregated, start_indices

    def organize_memory(self):
        mean_vec = jnp.mean(self.params_storage, axis=0)
        sim = jnp.dot(self.params_storage, mean_vec)
        indices = jnp.argsort(sim)
        return self.params_storage[indices]
