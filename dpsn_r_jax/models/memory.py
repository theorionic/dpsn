import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from dpsn_r_jax.config import PoolConfig


class LearnedIndexer(nn.Module):
    """Differentiable pool indexer with three improvements over the original:

    1. Attention-Pooled Query:  Instead of blindly using the last token's hidden
       state as the pool query, a lightweight attention mechanism learns *which*
       positions across the full sequence are most relevant for pool addressing.
       This is critical for long documents and code where the important "anchor"
       token may be anywhere (e.g., a function signature 200 lines back).

    2. Multi-Head Indexing:  `num_heads` independent (µ, σ) pairs are produced
       from the same pooled query.  Each head can specialise in a different pool
       region (e.g., syntax vs. API facts vs. algorithms).  Retrieved vectors are
       averaged, so the downstream interface is unchanged.

    3. Adaptive σ Clamping:  sigma is bounded to [sigma_min, sigma_max] via a
       sigmoid instead of the original unbounded softplus.  This lets the model
       learn *exactly how precise* a particular retrieval should be:
         σ ≈ sigma_min  →  sharp, almost nearest-neighbour retrieval (good for
                            exact API names, syntax tokens).
         σ ≈ sigma_max  →  broad, soft-average retrieval (good for general
                            thematic knowledge).

    Args:
        hidden_dim:  Hidden dimension D of the controller.
        num_heads:   Number of independent (µ, σ) pairs. Default 1 is fully
                     backward-compatible with the original single-head behaviour.
        sigma_min:   Hard lower bound on σ.  Default 0.01.
        sigma_max:   Hard upper bound on σ.  Default 5.0.
    """

    hidden_dim: int
    num_heads: int = 1
    sigma_min: float = 0.01
    sigma_max: float = 5.0

    @nn.compact
    def __call__(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, D)  – full encoded sequence from the controller.

        Returns:
            mu:    (B, num_heads)  – normalized pool coordinates in (0, 1).
            sigma: (B, num_heads)  – retrieval bandwidth in [sigma_min, sigma_max].
        """
        # ── 1. Attention-pooled query ────────────────────────────────────────
        # A single linear layer scores each position, then we take a soft
        # weighted sum.  This is parameter-cheap (just D weights) and fully
        # differentiable, so the model learns *where* to look by itself.
        attn_logits = nn.Dense(1, use_bias=False)(hidden_states)    # (B, T, 1)
        attn_weights = jax.nn.softmax(attn_logits, axis=1)           # (B, T, 1)
        pooled = jnp.sum(attn_weights * hidden_states, axis=1)       # (B, D)

        # ── 2. Shared feature extraction trunk ──────────────────────────────
        x = nn.Dense(self.hidden_dim)(pooled)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.gelu(x)

        # ── 3. Multi-head coordinate prediction ─────────────────────────────
        # One Dense produces ALL heads' raw µ values; same for σ.
        # This shares the trunk while keeping head-specific final projections.
        mu_raw    = nn.Dense(self.num_heads)(x)   # (B, num_heads)
        sigma_raw = nn.Dense(self.num_heads)(x)   # (B, num_heads)

        # µ: sigmoid → strictly in (0, 1) for valid pool addressing
        mu = jax.nn.sigmoid(mu_raw)               # (B, num_heads)

        # σ: sigmoid-rescaled into [sigma_min, sigma_max]
        #    (replaces unbounded softplus so the model can request sharp recall)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * jax.nn.sigmoid(sigma_raw)

        return mu, sigma  # (B, num_heads), (B, num_heads)


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
