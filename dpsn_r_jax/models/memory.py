import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from dpsn_r_jax.config import PoolConfig


class RetrievalRouter(nn.Module):
    hidden_dim: int
    min_k: int = 128
    max_k: int = 1024

    @nn.compact
    def __call__(self, hidden):
        # hidden: (B, T, D)
        pooled = jnp.mean(hidden, axis=1)  # (B, D)

        # Complexity net
        c = nn.Dense(self.hidden_dim // 4)(pooled)
        c = nn.gelu(c)
        c = nn.Dense(1)(c)
        complexity = nn.sigmoid(c)  # (B, 1)

        # Dynamic K calculation (keep as tracer)
        # We take mean complexity for k calculation logic
        mean_complexity = jnp.mean(complexity)

        # Scaled k between min and max
        k = self.min_k + mean_complexity * (self.max_k - self.min_k)
        k = k.astype(jnp.int32)

        query = nn.Dense(self.hidden_dim)(pooled)

        return query, k, complexity


class HierarchicalMassivePool(nn.Module):
    config: PoolConfig
    # We remove num_clusters logic for this verification implementation
    # and do a direct global search to ensure correctness on CPU with static shapes.
    # In production, you would implement Hierarchical search with fixed max_candidates.

    def setup(self):
        # The pool of vectors
        self.params = self.param(
            "params",
            nn.initializers.normal(),
            (self.config.total_vectors, self.config.hidden_dim),
        )
        self.keys = self.param(
            "keys",
            nn.initializers.normal(),
            (self.config.total_vectors, self.config.hidden_dim),
        )

        self.router_proj = nn.Dense(self.config.hidden_dim)

    def __call__(self, hidden, k_dynamic, max_k):
        # hidden: (B, T, D)
        B, T, D = hidden.shape
        pooled = jnp.mean(hidden, axis=1)  # (B, D)

        query = self.router_proj(pooled)  # (B, D)

        # Global Similarity Score
        # (B, D) @ (D, Total) -> (B, Total)
        scores = jnp.matmul(query, self.keys.T) / jnp.sqrt(D)

        # XLA FRIENDLY TOP-K:
        # We always retrieve max_k items.
        top_scores, top_indices = lax.top_k(scores, max_k)

        # Masking Logic for Dynamic K:
        # Create a mask for valid items where index < k_dynamic
        # k_dynamic is (1,) or scalar.
        iota = jnp.arange(max_k)[None, :]  # (1, max_k)
        mask = iota < k_dynamic

        # Mask scores with -inf so they have 0 probability in softmax
        top_scores_masked = jnp.where(mask, top_scores, -1e9)

        weights = nn.softmax(top_scores_masked, axis=-1)  # (B, max_k)

        # Gather params
        # (B, max_k, D)
        selected_params = self.params[top_indices]

        # Aggregate
        # (B, max_k) * (B, max_k, D) -> (B, D)
        aggregated = jnp.einsum("bk,bkd->bd", weights, selected_params)

        return aggregated
