import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from dpsn_r_jax.config import PoolConfig


class LearnedIndexer(nn.Module):
    """Differentiable pool indexer with four improvements over the original:

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

    4. Runtime σ Scale (Precision Routing):  `sigma_max_scale` multiplies the
       effective sigma_max at call time.  During σ-annealing training, trainer.py
       passes a schedule-derived scale (1.0 → small), so routing progressively
       tightens without any parameter changes.

    Args:
        hidden_dim:  Hidden dimension D of the controller.
        num_heads:   Number of independent (µ, σ) pairs. Default 1 is fully
                     backward-compatible with the original single-head behaviour.
        sigma_min:   Hard lower bound on σ.  Default 0.01.
        sigma_max:   Hard upper bound on σ before scaling.  Default 5.0.
    """

    hidden_dim: int
    num_heads: int = 1
    sigma_min: float = 0.01
    sigma_max: float = 5.0

    @nn.compact
    def __call__(self, hidden_states, sigma_max_scale: float = 1.0):
        """
        Args:
            hidden_states:    (B, T, D) – full encoded sequence from controller.
            sigma_max_scale:  Float in (0, 1].  1.0 = full range (broad).
                              Decrease during training to enforce precision.

        Returns:
            mu:    (B, num_heads)  – normalized pool coordinates in (0, 1).
            sigma: (B, num_heads)  – retrieval bandwidth in
                                    [sigma_min, sigma_max * sigma_max_scale].
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

        # ── 4. σ with dynamic scale for annealed precision routing ───────────
        # effective_sigma_max shrinks over training via sigma_max_scale.
        # At scale=1.0 (step 0): full broad range → easy coarse learning.
        # At scale→0 (late training): near-zero max → nearly exact retrieval.
        effective_sigma_max = self.sigma_max * sigma_max_scale
        sigma = (
            self.sigma_min
            + (effective_sigma_max - self.sigma_min) * jax.nn.sigmoid(sigma_raw)
        )                                          # (B, num_heads)

        return mu, sigma  # (B, num_heads), (B, num_heads)


class CoordinateMassivePool(nn.Module):
    """1D pool — original implementation. Still used when use_2d_pool=False."""
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

        weights = jnp.exp(-(distances**2) / (2 * (sigma[:, None] + 1e-6) ** 2)) + 1e-9
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

        aggregated = jnp.einsum("bw,bwd->bd", weights, selected)

        return aggregated, start_indices

    def organize_memory(self):
        mean_vec = jnp.mean(self.params_storage, axis=0)
        sim = jnp.dot(self.params_storage, mean_vec)
        indices = jnp.argsort(sim)
        return self.params_storage[indices]


class CoordinateMassivePool2D(nn.Module):
    """2D Grid Pool — precision routing improvement.

    Instead of a flat (N, D) array addressed by a single µ ∈ (0,1), the pool
    is organised as a (rows, cols, D) grid addressed by (µ_row, µ_col).

    Precision advantage:
        1D: µ must have precision 1/N     (e.g., 1/262144 ≈ 0.0004% for large)
        2D: each axis needs precision 1/√N (e.g., 1/512 ≈ 0.2% for large)
        → Each coordinate is 512× easier to learn precisely.

    Retrieval:
        row_weights = Gaussian(µ_row, rows, σ)         (B, rows)
        col_weights = Gaussian(µ_col, cols, σ)         (B, cols)
        w_2d        = outer_product(row_w, col_w)      (B, rows, cols)
        output      = einsum("brc, rcd -> bd", w_2d, pool)

    Sparse update: trainer uses the same ("pool", "params_storage") key.
    The 2D storage is reshaped to (rows, cols, D) transparently.

    Args:
        rows, cols:    Grid dimensions. rows × cols = effective pool size.
        hidden_dim:    Vector dimension D.
        window_size:   K — window radius per axis (actual window = K×K vectors).
    """
    rows: int
    cols: int
    hidden_dim: int
    window_size: int   # per-axis window; total retrieved = window_size^2

    def setup(self):
        # Stored as a true 2D grid so the addressing math is clean.
        # Flax saves it under the same "params_storage" key as 1D pool.
        self.params_storage = self.param(
            "params_storage",
            nn.initializers.normal(),
            (self.rows, self.cols, self.hidden_dim),
        )

    def __call__(self, mu_row, mu_col, sigma):
        """
        Args:
            mu_row: (B,) pool row coordinate in (0, 1)
            mu_col: (B,) pool col coordinate in (0, 1)
            sigma:  (B,) retrieval bandwidth (same σ for both axes)
        Returns:
            aggregated:   (B, D)  weighted knowledge vector
            flat_indices: (B,)    flattened window start index (for sparse Adam)
        """
        B   = mu_row.shape[0]
        R   = self.rows
        C   = self.cols
        D   = self.hidden_dim
        W   = self.window_size    # window half-size per axis

        # ── Row axis ──────────────────────────────────────────────────────────
        r_center = mu_row * (R - 1)                                  # (B,)
        r_start  = jnp.clip(r_center - W // 2, 0, R - W).astype(jnp.int32)

        # ── Col axis ──────────────────────────────────────────────────────────
        c_center = mu_col * (C - 1)                                  # (B,)
        c_start  = jnp.clip(c_center - W // 2, 0, C - W).astype(jnp.int32)

        # ── Fetch W×W sub-grid for each sample ────────────────────────────────
        def fetch_window(r_s, c_s):
            # Slice a (W, W, D) sub-grid starting at (r_s, c_s, 0)
            return lax.dynamic_slice(
                self.params_storage, (r_s, c_s, 0), (W, W, D)
            )                                                         # (W, W, D)

        windows = jax.vmap(fetch_window)(r_start, c_start)           # (B, W, W, D)

        # ── 2D Gaussian weights ────────────────────────────────────────────────
        r_idx    = jnp.arange(W)[None, :] + r_start[:, None]        # (B, W)
        c_idx    = jnp.arange(W)[None, :] + c_start[:, None]        # (B, W)

        r_dist   = r_idx - r_center[:, None]                         # (B, W)
        c_dist   = c_idx - c_center[:, None]                         # (B, W)

        sigma_sq = (sigma + 1e-6) ** 2
        r_w = jnp.exp(-r_dist ** 2 / (2 * sigma_sq[:, None]))       # (B, W)
        c_w = jnp.exp(-c_dist ** 2 / (2 * sigma_sq[:, None]))       # (B, W)

        # Outer product → 2D weight matrix
        w_2d = jnp.einsum("bi,bj->bij", r_w, c_w) + 1e-9            # (B, W, W)
        w_2d = w_2d / jnp.sum(w_2d, axis=(-2, -1), keepdims=True)

        # Weighted sum over the W×W window
        aggregated = jnp.einsum("bij,bijd->bd", w_2d, windows)       # (B, D)

        # ── Flat index for sparse Adam ────────────────────────────────────────
        # Represent window by its top-left flat index in the R×C grid
        flat_start = r_start * C + c_start                           # (B,)

        return aggregated, flat_start
