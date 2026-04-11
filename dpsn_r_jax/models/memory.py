import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from dpsn_r_jax.config import PoolConfig
from dpsn_r_jax.kernels import pool_retrieve_1d_pallas, pool_retrieve_2d_pallas


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
        hidden_dim:        Hidden dimension D of the controller (input size).
        indexer_hidden_dim: Width of the MLP trunk inside the indexer.  0 (default)
                           falls back to hidden_dim for backward compatibility.
                           Set to a large value (e.g. 10240) to give the indexer
                           its own large parameter budget independent of the controller.
        num_heads:         Number of independent (µ, σ) pairs. Default 1 is fully
                           backward-compatible with the original single-head behaviour.
        sigma_min:         Hard lower bound on σ.  Default 0.01.
        sigma_max:         Hard upper bound on σ before scaling.  Default 5.0.
    """

    hidden_dim: int
    indexer_hidden_dim: int = 0   # 0 = use hidden_dim (backward compat)
    num_heads: int = 1
    sigma_min: float = 0.01
    sigma_max: float = 5.0
    # Fraction of grid to exclude at each boundary.
    # With margin=0.05: mu maps to (0.05, 0.95) → rows [51, 972] on a 1024 grid.
    # Prevents sigmoid saturation from collapsing all routes to grid corners.
    coord_margin: float = 0.05

    @nn.compact
    def __call__(self, hidden_states, sigma_max_scale: float = 1.0, deterministic: bool = True):
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
        # mlp_dim may be set independently of the controller width so the
        # indexer can carry its own large parameter budget (e.g. 50M+).
        mlp_dim = self.indexer_hidden_dim if self.indexer_hidden_dim > 0 else self.hidden_dim
        x = nn.Dense(mlp_dim)(pooled)
        x = nn.gelu(x)
        x = nn.Dense(mlp_dim // 2)(x)
        x = nn.gelu(x)

        # ── 3. Multi-head coordinate prediction ─────────────────────────────
        # Softmax over discrete interior bin positions.
        #
        # WHY not tanh:
        #   tanh saturates at ±1. Once mu_raw grows large (driven by the LM
        #   loss rewarding a corner location), tanh'≈0 so gradients through
        #   the indexer vanish — the model can't escape even with a repulsion
        #   loss fighting it. This causes the observed 4-corner collapse where
        #   concentration monotonically increases despite strong diversity losses.
        #
        # WHY softmax-over-bins:
        #   1. Bin centres span [0.1, 0.9] → grid corners (row 0, row 1023)
        #      are physically unreachable, not just penalised.
        #   2. Softmax gradient is well-behaved everywhere (no saturation),
        #      so repulsion losses always have a non-zero signal path.
        #   3. The model can represent any interior position as a mixture of
        #      neighbouring bins — expressiveness is not sacrificed.
        _N_POS = 32  # 32 bins over [0.1, 0.9] → ~26-cell spacing on a 1024 grid
        pos_logits = nn.Dense(
            self.num_heads * _N_POS,
            kernel_init=nn.initializers.normal(stddev=0.02),
            bias_init=nn.initializers.zeros,
        )(x)  # (B, num_heads * _N_POS)
        B_dim = pos_logits.shape[0]
        pos_logits = pos_logits.reshape(B_dim, self.num_heads, _N_POS)
        pos_probs = jax.nn.softmax(pos_logits, axis=-1)          # (B, H, N_POS)
        bin_centers = jnp.linspace(
            jnp.float32(0.1), jnp.float32(0.9), _N_POS
        )                                                          # (N_POS,)
        mu_01 = jnp.sum(
            pos_probs * bin_centers[None, None, :], axis=-1
        )  # (B, num_heads) in [0.1, 0.9]

        sigma_raw = nn.Dense(self.num_heads)(x)   # (B, num_heads)

        # During training: add small coord-space noise for additional exploration.
        # Clipped to [0.1, 0.9] to stay within the bin range.
        if not deterministic:
            mu_01 = mu_01 + jax.random.normal(
                self.make_rng('dropout'), mu_01.shape, dtype=jnp.float32
            ) * jnp.float32(0.10)
            mu_01 = jnp.clip(mu_01, jnp.float32(0.1), jnp.float32(0.9))

        # coord_margin is 0.0 here; with mu_01 already in [0.1, 0.9] the mapping
        # is identity and corners are unreachable.
        mu = self.coord_margin + (1.0 - 2.0 * self.coord_margin) * mu_01

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


class DirectIndexPool(nn.Module):
    """Exact-recall memory for DPSN-R — one slot per unique fact type.

    The core property: each fact routes to one dedicated slot via
    content-based lookup, so gradients from different facts NEVER
    interfere with each other.  With temperature→0 (hard argmax), a
    slot trained on "requests.get() → Response" cannot be degraded by
    any other fact type.

    Design (mirrors the validated experiment_hybrid_pool.py):
      storage  : (n_slots, D)   — the actual stored vectors
      keys     : (n_slots, D)   — per-slot addressing keys
      proj     : Dense(D)       — small MLP for retrieved content

    Retrieval:
      scores  = query @ keys.T / sqrt(D)          # (B, ..., n_slots)
      weights = softmax(scores / temperature)
      out     = weights @ storage                  # (B, ..., D)
      return  gelu(proj(out))                      # (B, ..., D)

    Temperature:
      0.1  (default) → near-argmax, one slot per query → no bleed
      1.0            → soft mixture → generalisation at cost of sharpness

    Phase roles (phase_stop_gradient in trainer.py enforces this):
      Phase 1 (controller)    : params are stop_gradient'd
      Phase 2 (direct pool)   : ONLY this module trains; rest frozen
      Phase 3 (spatial pool)  : params are stop_gradient'd again
      Phase 0 (joint)         : trains alongside everything else

    Returns corrections in the same shape as the input hidden states so
    callers can do: ``state_hidden = state_hidden + direct_pool(state_hidden)``
    """

    n_slots:     int
    hidden_dim:  int
    temperature: float = 0.1

    def setup(self):
        init = nn.initializers.normal(stddev=0.02)
        # Per-slot addressing keys (matched against controller hidden states)
        self.keys    = self.param("keys",    init, (self.n_slots, self.hidden_dim))
        # Per-slot stored content vectors (retrieved and added to state_hidden)
        self.storage = self.param("storage", init, (self.n_slots, self.hidden_dim))
        # Small linear projection (same role as direct_proj in the experiment)
        self.proj    = nn.Dense(self.hidden_dim, use_bias=False, kernel_init=init)

    def __call__(self, hidden: jnp.ndarray) -> jnp.ndarray:
        """Retrieve from the direct pool and return additive corrections.

        Uses ``...`` in einsum so the same call works for both the
        full-sequence (B, T, D) case and the single-position (B, D) case.

        Args:
            hidden: (B, T, D) or (B, D) — controller / state hidden states.

        Returns:
            corrections: same shape as hidden — add to state_hidden.
        """
        scale   = jnp.float32(self.hidden_dim) ** -0.5
        scores  = jnp.einsum("...d,sd->...s", hidden.astype(jnp.float32),
                              self.keys.astype(jnp.float32)) * scale  # (..., n_slots)
        weights = jax.nn.softmax(scores / self.temperature, axis=-1)  # (..., n_slots)
        out     = jnp.einsum("...s,sd->...d", weights,
                              self.storage.astype(jnp.float32))        # (..., D)
        return nn.gelu(self.proj(out))                                 # (..., D)


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
        """Gaussian-weighted fetch from ``storage`` (any slice of params_storage).

        Uses the fused Pallas kernel on TPU (or interpret mode) to avoid
        materialising the intermediate (B, W, D) selected-vectors buffer and
        (B, W) Gaussian-weights buffer in HBM.  Falls back to pure JAX on GPU/CPU.
        """
        Total = self.config.total_vectors
        W     = self.window_size

        center_idx    = mu * (Total - 1)
        start_indices = jnp.clip(
            center_idx - W // 2, 0, Total - W
        ).astype(jnp.int32)

        aggregated = pool_retrieve_1d_pallas(
            storage, mu, sigma, start_indices, W=W, Total=Total
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
        R   = self.rows
        C   = self.cols
        W   = self.window_size    # window half-size per axis

        # ── Row axis ──────────────────────────────────────────────────────────
        r_center = mu_row * (R - 1)                                  # (B,)
        r_start  = jnp.clip(r_center - W // 2, 0, R - W).astype(jnp.int32)

        # ── Col axis ──────────────────────────────────────────────────────────
        c_center = mu_col * (C - 1)                                  # (B,)
        c_start  = jnp.clip(c_center - W // 2, 0, C - W).astype(jnp.int32)

        # Route through the fused Pallas kernel: avoids materialising the
        # (B, W, W, D) windows buffer and the (B, W, W) Gaussian weight matrix.
        # Falls back to pure JAX on non-TPU backends automatically.
        aggregated = pool_retrieve_2d_pallas(
            self.params_storage,
            r_start, c_start,
            r_center.astype(jnp.float32),
            c_center.astype(jnp.float32),
            sigma.astype(jnp.float32),
            W=W, R=R, C=C,
        )
        flat_start = r_start * C + c_start
        return aggregated, flat_start

    def bilinear_retrieve(self, mu_row: jnp.ndarray, mu_col: jnp.ndarray) -> jnp.ndarray:
        """Differentiable bilinear retrieval for Straight-Through Estimator.

        Used only in the backward pass:
            retrieved = retrieved_hard + (retrieved_soft - stop_gradient(retrieved_soft))
        Forward value is zero (STE identity). Backward gives d(loss)/d(mu) ≠ 0
        through the fractional interpolation weights (wr, wc).

        Pool storage is already stop_gradient'd by the trainer so this does NOT
        create the 805 MB pool gradient — only mu gets a gradient.

        Args:
            mu_row: (B,) normalised row coordinate in (0, 1)
            mu_col: (B,) normalised col coordinate in (0, 1)
        Returns:
            (B, D) bilinearly interpolated pool vectors in float32
        """
        # Do NOT cast the entire pool to fp32 first — that reads R*C*D values.
        # Instead gather only the 4 corner vectors (4*B*D values) and cast those.
        R, C = self.rows, self.cols

        r_f = mu_row.astype(jnp.float32) * (R - 1)  # (B,)
        c_f = mu_col.astype(jnp.float32) * (C - 1)  # (B,)

        r_lo = jnp.clip(jnp.floor(r_f).astype(jnp.int32), 0, R - 2)  # (B,)
        c_lo = jnp.clip(jnp.floor(c_f).astype(jnp.int32), 0, C - 2)  # (B,)

        # Fractional weights — gradient of loss flows through wr/wc → mu
        wr = (r_f - r_lo.astype(jnp.float32))[:, None]  # (B, 1)
        wc = (c_f - c_lo.astype(jnp.float32))[:, None]  # (B, 1)

        # Gather 4 corner vectors with advanced indexing, cast to fp32 after
        # (touches 4*B*D values, not R*C*D — avoids ~1.2 GB fp32 materialisation)
        v00 = self.params_storage[r_lo,     c_lo    ].astype(jnp.float32)
        v10 = self.params_storage[r_lo + 1, c_lo    ].astype(jnp.float32)
        v01 = self.params_storage[r_lo,     c_lo + 1].astype(jnp.float32)
        v11 = self.params_storage[r_lo + 1, c_lo + 1].astype(jnp.float32)

        return ((1 - wr) * (1 - wc) * v00 + wr * (1 - wc) * v10
                + (1 - wr) * wc * v01 + wr * wc * v11)  # (B, D)

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
        W   = self.window_size
        SW  = W * super_window_factor

        r_center  = mu_row * (R - 1)
        c_center  = mu_col * (C - 1)
        sw_r_start = jnp.clip(r_center - SW // 2, 0, R - SW).astype(jnp.int32)
        sw_c_start = jnp.clip(c_center - SW // 2, 0, C - SW).astype(jnp.int32)

        local_D = self.params_storage.shape[-1]  # hidden_dim / tp_size on feature-sharded pool

        def _fetch(r_s, c_s):
            return lax.dynamic_slice(
                self.params_storage, (r_s, c_s, 0), (SW, SW, local_D)
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

    # ── Prefetch-reasoning: one-shot full patch fetch ─────────────────────────
    def fetch_patch_2d(
        self,
        mu_row: jnp.ndarray,
        mu_col: jnp.ndarray,
        patch_size: int,
    ) -> tuple:
        """Pre-fetch a patch_size × patch_size region from the 2D pool grid.

        This is the **single** HBM access in the prefetch-reasoning design.
        The returned tensor is passed as a ``lax.scan`` carry so XLA keeps
        it in on-chip SRAM for the entire reasoning loop — no further HBM
        access occurs during the scan.

        SRAM cost (bf16):
            B_per_chip × patch_size² × hidden_dim × 2 bytes
            e.g. B/chip=4, patch=64, D=1024 → 33 MB  (fits in 128 MB VMEM)

        Args:
            mu_row:     (B,) normalised row coordinate in (0, 1).
            mu_col:     (B,) normalised col coordinate in (0, 1).
            patch_size: Number of vectors per grid axis.
                        Total candidates = patch_size × patch_size.

        Returns:
            patch:    (B, patch_size, patch_size, D) — candidate vectors
                      in bfloat16 (native pool storage dtype).
            r_start:  (B,) integer row start indices (for sparse-Adam tracking).
            c_start:  (B,) integer col start indices.
        """
        R    = self.rows
        C    = self.cols
        D    = self.params_storage.shape[-1]   # local dim (hidden_dim / tp_size)
        half = patch_size // 2

        r_center = (mu_row * (R - 1)).astype(jnp.int32)
        c_center = (mu_col * (C - 1)).astype(jnp.int32)

        # Clamp so the patch never exceeds pool boundaries
        r_start = jnp.clip(r_center - half, 0, R - patch_size)
        c_start = jnp.clip(c_center - half, 0, C - patch_size)

        def _fetch(rs, cs):
            return lax.dynamic_slice(
                self.params_storage, (rs, cs, 0), (patch_size, patch_size, D)
            ).astype(jnp.bfloat16)

        patch = jax.vmap(_fetch)(r_start, c_start)   # (B, patch_size, patch_size, D)
        return patch, r_start, c_start

    # ── Shared weighting & aggregation ───────────────────────────────────────
    def _weight_and_aggregate(
        self, windows, r_start, c_start, r_center, c_center, sigma
    ):
        """Apply 2D Gaussian weights over a (B, W, W, D) window tensor.

        NOTE: this path is only reached by the super-window variants
        (__call_from_super_window_2d__) where the window is already in SRAM.
        The standard __call__ path is routed through pool_retrieve_2d_pallas
        which skips materialising the (B, W, W, D) buffer entirely.
        """
        W = self.window_size
        C = self.cols

        r_idx  = jnp.arange(W)[None, :] + r_start[:, None]
        c_idx  = jnp.arange(W)[None, :] + c_start[:, None]

        r_dist = r_idx - r_center[:, None]
        c_dist = c_idx - c_center[:, None]

        sigma_sq = (sigma + 1e-6) ** 2
        r_w = jnp.exp(-r_dist ** 2 / (2 * sigma_sq[:, None]))
        c_w = jnp.exp(-c_dist ** 2 / (2 * sigma_sq[:, None]))

        w_2d = jnp.einsum("bi,bj->bij", r_w, c_w) + 1e-6
        w_2d = w_2d / jnp.sum(w_2d, axis=(-2, -1), keepdims=True)

        aggregated = jnp.einsum(
            "bij,bijd->bd", w_2d, windows.astype(jnp.float32)
        )

        flat_start = r_start * C + c_start
        return aggregated, flat_start

    def _retrieve_2d(self, mu_row, mu_col, sigma):
        """Fused 2-D retrieve via Pallas kernel (used by __call__)."""
        R, C = self.rows, self.cols
        W    = self.window_size

        r_center = mu_row * (R - 1)
        c_center = mu_col * (C - 1)
        r_start  = jnp.clip(r_center - W // 2, 0, R - W).astype(jnp.int32)
        c_start  = jnp.clip(c_center - W // 2, 0, C - W).astype(jnp.int32)

        aggregated = pool_retrieve_2d_pallas(
            self.params_storage,
            r_start, c_start,
            r_center.astype(jnp.float32),
            c_center.astype(jnp.float32),
            sigma.astype(jnp.float32),
            W=W, R=R, C=C,
        )
        flat_start = r_start * C + c_start
        return aggregated, flat_start
