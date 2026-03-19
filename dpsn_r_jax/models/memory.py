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
    """1D pool — original implementation. Still used when use_2d_pool=False.

    Memory-bandwidth optimisation (Opt-1):
        ``params_storage`` is now kept natively in ``bfloat16`` so the HBM
        memory controller transfers 2 bytes per element instead of 4, giving
        a 2× effective bandwidth improvement for the scatter/gather dominated
        fetch path.

    SRAM super-window helpers (Opt-2):
        ``fetch_super_window`` gathers a wide contiguous slab once from slow
        HBM.  ``__call_from_super_window__`` then slices *within* that tensor,
        which XLA places in on-chip SRAM when the tensor is a ``lax.scan``
        carry, reducing per-iteration latency from ~100 ns to ~1 ns.
    """
    config: PoolConfig
    window_size: int

    def setup(self):
        # ── Optimisation 1: native bfloat16 storage ───────────────────────────
        # Storing in bfloat16 halves the bytes hauled over the HBM bus on every
        # dynamic_slice call.  The initializer is cast to bf16 so the param is
        # already at the right dtype in the checkpoint.
        self.params_storage = self.param(
            "params_storage",
            nn.initializers.normal(dtype=jnp.bfloat16),
            (self.config.total_vectors, self.config.hidden_dim),
            jnp.bfloat16,
        )

    # ── Standard forward (fetches from HBM each call) ────────────────────────
    def __call__(self, mu, sigma):
        return self._retrieve(self.params_storage, mu, sigma)

    # ── Super-window helpers (Opt-2 support) ─────────────────────────────────
    def fetch_super_window(
        self, mu: jnp.ndarray, super_window_factor: int
    ) -> tuple:
        """Grab a wide slab from HBM **once** before the reasoning loop.

        Args:
            mu:                  (B,) normalised pool coordinate in (0, 1).
            super_window_factor: Width multiplier; super-window = W * factor.

        Returns:
            super_window:  (B, SW, D) tensor — will be held in SRAM by the
                           XLA ``lax.scan`` carry mechanism.
            sw_start:      (B,) integer start index of the super-window.
        """
        Total = self.config.total_vectors
        D     = self.config.hidden_dim
        SW    = self.window_size * super_window_factor

        center_idx = mu * (Total - 1)
        sw_start   = jnp.clip(
            center_idx - SW // 2, 0, Total - SW
        ).astype(jnp.int32)

        def _fetch(start):
            return lax.dynamic_slice(
                self.params_storage, (start, 0), (SW, D)
            ).astype(jnp.bfloat16)   # keep bf16 dtype

        super_window = jax.vmap(_fetch)(sw_start)   # (B, SW, D)
        return super_window, sw_start

    def __call_from_super_window__(
        self,
        super_window: jnp.ndarray,
        sw_start: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
    ):
        """Retrieve from an already-fetched super_window (on-chip SRAM path).

        Args:
            super_window: (B, SW, D) carry tensor from ``lax.scan``.
            sw_start:     (B,) super-window start indices.
            mu:           (B,) pool coordinate in (0, 1).
            sigma:        (B,) retrieval bandwidth.

        Returns:
            aggregated:   (B, D)
            start_indices:(B,) local window start within super_window (+ sw_start
                          gives the global index, used for sparse Adam).
        """
        Total  = self.config.total_vectors
        SW     = super_window.shape[1]  # dynamic
        W      = self.window_size
        D      = self.config.hidden_dim

        center_idx = mu * (Total - 1)                        # (B,)

        # Local start index within the super-window
        local_center = center_idx - sw_start.astype(mu.dtype)  # (B,)
        local_start  = jnp.clip(
            local_center - W // 2, 0, SW - W
        ).astype(jnp.int32)                                  # (B,)

        def _local_slice(buf, ls):
            # buf: (SW, D), ls: scalar → (W, D)
            return lax.dynamic_slice(buf, (ls, 0), (W, D))

        selected = jax.vmap(_local_slice)(super_window, local_start)  # (B, W, D)

        # Global absolute indices for weight computation
        global_start = local_start + sw_start             # (B,)
        rel_idx      = jnp.arange(W)[None, :] + global_start[:, None]  # (B, W)
        distances    = rel_idx - center_idx[:, None]      # (B, W)

        weights = (
            jnp.exp(-(distances ** 2) / (2 * (sigma[:, None] + 1e-6) ** 2)) + 1e-6
        )
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

        aggregated = jnp.einsum(
            "bw,bwd->bd", weights, selected.astype(jnp.float32)
        )  # keep output in float32 for downstream precision

        return aggregated, global_start

    # ── Internal shared retrieval routine ────────────────────────────────────
    def _retrieve(self, storage, mu, sigma):
        """Gaussian-weighted fetch from ``storage`` (any slice of params_storage)."""
        Total = self.config.total_vectors
        D     = self.config.hidden_dim
        W     = self.window_size

        center_idx    = mu * (Total - 1)
        start_indices = jnp.clip(
            center_idx - W // 2, 0, Total - W
        ).astype(jnp.int32)

        def slice_fn(start):
            return lax.dynamic_slice(storage, (start, 0), (W, D))

        selected = jax.vmap(slice_fn)(start_indices)  # (B, W, D) bf16

        relative_indices = jnp.arange(W)[None, :] + start_indices[:, None]
        distances        = relative_indices - center_idx[:, None]

        weights = (
            jnp.exp(-(distances ** 2) / (2 * (sigma[:, None] + 1e-6) ** 2)) + 1e-6
        )
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

        aggregated = jnp.einsum(
            "bw,bwd->bd", weights, selected.astype(jnp.float32)
        )
        return aggregated, start_indices

    def organize_memory(self):
        mean_vec = jnp.mean(self.params_storage.astype(jnp.float32), axis=0)
        sim      = jnp.dot(self.params_storage.astype(jnp.float32), mean_vec)
        indices  = jnp.argsort(sim)
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
        # ── Optimisation 1: native bfloat16 storage ───────────────────────────
        self.params_storage = self.param(
            "params_storage",
            nn.initializers.normal(dtype=jnp.bfloat16),
            (self.rows, self.cols, self.hidden_dim),
            jnp.bfloat16,
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

        # ── Fetch W×W sub-grid for each sample (HBM path) ───────────────────
        def fetch_window(r_s, c_s):
            # Slice a (W, W, D) sub-grid starting at (r_s, c_s, 0)
            return lax.dynamic_slice(
                self.params_storage, (r_s, c_s, 0), (W, W, D)
            ).astype(jnp.bfloat16)                                    # (W, W, D)

        windows = jax.vmap(fetch_window)(r_start, c_start)           # (B, W, W, D)

        aggregated, flat_start = self._weight_and_aggregate(
            windows, r_start, c_start, r_center, c_center, sigma
        )
        return aggregated, flat_start

    # ── Super-window helpers for 2D pool (Opt-2 support) ─────────────────────
    def fetch_super_window_2d(
        self,
        mu_row: jnp.ndarray,
        mu_col: jnp.ndarray,
        super_window_factor: int,
    ) -> tuple:
        """Pre-fetch a wide (SW_r × SW_c × D) slab from HBM **once**.

        Returns:
            super_window: (B, SW_r, SW_c, D)
            sw_r_start:   (B,)
            sw_c_start:   (B,)
        """
        R   = self.rows
        C   = self.cols
        D   = self.hidden_dim
        W   = self.window_size
        SW  = W * super_window_factor

        r_center  = mu_row * (R - 1)
        c_center  = mu_col * (C - 1)
        sw_r_start = jnp.clip(r_center - SW // 2, 0, R - SW).astype(jnp.int32)
        sw_c_start = jnp.clip(c_center - SW // 2, 0, C - SW).astype(jnp.int32)

        def _fetch(r_s, c_s):
            return lax.dynamic_slice(
                self.params_storage, (r_s, c_s, 0), (SW, SW, D)
            ).astype(jnp.bfloat16)

        super_window = jax.vmap(_fetch)(sw_r_start, sw_c_start)  # (B, SW, SW, D)
        return super_window, sw_r_start, sw_c_start

    def __call_from_super_window_2d__(
        self,
        super_window: jnp.ndarray,
        sw_r_start: jnp.ndarray,
        sw_c_start: jnp.ndarray,
        mu_row: jnp.ndarray,
        mu_col: jnp.ndarray,
        sigma: jnp.ndarray,
    ):
        """2D retrieval from an already-in-SRAM super_window carry."""
        R   = self.rows
        C   = self.cols
        W   = self.window_size
        SW  = super_window.shape[1]  # dynamic

        r_center = mu_row * (R - 1)
        c_center = mu_col * (C - 1)

        local_r_center = r_center - sw_r_start.astype(mu_row.dtype)
        local_c_center = c_center - sw_c_start.astype(mu_col.dtype)

        local_r_start = jnp.clip(local_r_center - W // 2, 0, SW - W).astype(jnp.int32)
        local_c_start = jnp.clip(local_c_center - W // 2, 0, SW - W).astype(jnp.int32)

        def _local_slice(buf, lr, lc):
            # buf: (SW, SW, D)
            D_inner = buf.shape[-1]
            return lax.dynamic_slice(buf, (lr, lc, 0), (W, W, D_inner))

        windows = jax.vmap(_local_slice)(
            super_window, local_r_start, local_c_start
        )  # (B, W, W, D)

        # Global absolute indices for weight computation
        global_r_start = local_r_start + sw_r_start
        global_c_start = local_c_start + sw_c_start

        aggregated, flat_start = self._weight_and_aggregate(
            windows, global_r_start, global_c_start, r_center, c_center, sigma
        )
        return aggregated, flat_start

    # ── Shared weighting & aggregation ───────────────────────────────────────
    def _weight_and_aggregate(
        self, windows, r_start, c_start, r_center, c_center, sigma
    ):
        """Apply 2D Gaussian weights over a (B, W, W, D) window tensor."""
        W = self.window_size
        C = self.cols

        r_idx  = jnp.arange(W)[None, :] + r_start[:, None]         # (B, W)
        c_idx  = jnp.arange(W)[None, :] + c_start[:, None]         # (B, W)

        r_dist = r_idx - r_center[:, None]                           # (B, W)
        c_dist = c_idx - c_center[:, None]                           # (B, W)

        sigma_sq = (sigma + 1e-6) ** 2
        r_w = jnp.exp(-r_dist ** 2 / (2 * sigma_sq[:, None]))       # (B, W)
        c_w = jnp.exp(-c_dist ** 2 / (2 * sigma_sq[:, None]))       # (B, W)

        # Outer product → 2D weight matrix
        w_2d = jnp.einsum("bi,bj->bij", r_w, c_w) + 1e-6            # (B, W, W)
        w_2d = w_2d / jnp.sum(w_2d, axis=(-2, -1), keepdims=True)

        # Weighted sum — cast windows to float32 for numerical precision
        aggregated = jnp.einsum(
            "bij,bijd->bd", w_2d, windows.astype(jnp.float32)
        )                                                             # (B, D)

        # Flat index for sparse Adam
        flat_start = r_start * C + c_start                           # (B,)
        return aggregated, flat_start
