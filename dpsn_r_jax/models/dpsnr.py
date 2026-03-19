import jax
import jax.numpy as jnp
import flax.linen as nn
from dpsn_r_jax.config import DPSNRConfig, PoolConfig
from dpsn_r_jax.models.controller import TinyController
from dpsn_r_jax.models.memory import (
    CoordinateMassivePool,
    CoordinateMassivePool2D,
    LearnedIndexer,
)
import jax.profiler
from dpsn_r_jax.models.reasoning import AdaptiveComputeController

# Set to True to print per-loop reasoning stats to stdout (via jax.debug.print).
# Works inside JIT/lax.scan. Disable for production to avoid print overhead.
DEBUG_REASONING_LOOPS: bool = False


class DPSNR(nn.Module):
    config: DPSNRConfig

    def setup(self):
        controller_cls = TinyController
        acc_cls = AdaptiveComputeController

        self.controller = controller_cls(self.config)
        self.indexer = LearnedIndexer(
            self.config.controller_hidden_dim,
            num_heads=self.config.num_indexer_heads,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
        )

        # ── Pool selection: 1D (flat) or 2D (grid) ────────────────────────────
        if self.config.use_2d_pool:
            # 2D Grid Pool: each coordinate only needs 1/sqrt(N) precision.
            # The window_size here is per-axis; total retrieved = max_k × max_k.
            axis_window = max(2, int(self.config.max_k ** 0.5))
            self.pool = CoordinateMassivePool2D(
                rows=self.config.pool_grid_rows,
                cols=self.config.pool_grid_cols,
                hidden_dim=self.config.controller_hidden_dim,
                window_size=axis_window,
            )
        else:
            self.pool = CoordinateMassivePool(
                PoolConfig(
                    self.config.pool_total_vectors,
                    self.config.controller_hidden_dim,
                ),
                window_size=self.config.max_k,
            )

        self.acc = AdaptiveComputeController(
            self.config.controller_hidden_dim,
            self.config.max_reasoning_loops,
            self.config.halt_threshold,
        )
        self.retrieval_integrator = nn.Sequential(
            [
                nn.Dense(self.config.controller_hidden_dim),
                nn.gelu,
                nn.Dense(self.config.controller_hidden_dim),
                nn.LayerNorm(),
            ]
        )

    def __call__(self, input_ids, deterministic=True, sigma_max_scale: float = 1.0):
        """
        Args:
            input_ids:        (B, T) integer token ids.
            deterministic:    Set False during training (enables dropout).
            sigma_max_scale:  Sigma annealing multiplier ∈ (0, 1].
                              Pass via trainer's sigma_anneal_fn; 1.0 at step 0,
                              decreasing to ~0.01 at sigma_anneal_steps.
                              Shrinks the effective sigma_max so routing becomes
                              progressively more precise without param changes.

        Returns:
            logits:       (B, T, vocab)
            aux:          (max_loops, all_indices, mean_sigma)
                          mean_sigma is logged and used for precision loss.
        """
        state_hidden, all_indices, mean_sigma = self._encode_hidden(
            input_ids, deterministic, sigma_max_scale
        )

        # 3. Decode — the expensive (B, T, V) step
        logits = self.controller.decode(state_hidden)

        return logits, (self.config.max_reasoning_loops, all_indices, mean_sigma)

    def encode_to_hidden(self, input_ids, deterministic=True, sigma_max_scale: float = 1.0):
        """Run controller + reasoning loop and return state_hidden WITHOUT the LM head.

        Used by chunked_lm_loss in trainer.py to avoid materialising the full
        (B, T, vocab) logits tensor.  Only the compact (B, T, D) hidden tensor
        is returned; the LM head is applied later in small batch chunks.

        Returns:
            state_hidden: (B, T, D)
            aux:          (max_loops, all_indices, mean_sigma)
        """
        state_hidden, all_indices, mean_sigma = self._encode_hidden(
            input_ids, deterministic, sigma_max_scale
        )
        return state_hidden, (self.config.max_reasoning_loops, all_indices, mean_sigma)

    def _encode_hidden(self, input_ids, deterministic=True, sigma_max_scale: float = 1.0):
        """Core shared encoder: controller + reasoning loop.

        Returns (state_hidden, all_indices, mean_sigma).
        Called by both __call__ and encode_to_hidden so they share code.
        """
        # 1. Encode
        # MUST pass deterministic as a positional argument so static_argnums=(1,) catches it!
        with jax.profiler.TraceAnnotation("TinyController_Forward"):
            hidden = self.controller(input_ids, deterministic)

        # 2. Reasoning Loop
        state_hidden = hidden
        B, T, D = hidden.shape

        halt_prob   = jnp.zeros((B, T, 1), dtype=hidden.dtype)
        halted_mask = jnp.zeros((B, T, 1), dtype=hidden.dtype)

        # ── Warm-up calls: required for Flax to trace sub-modules before scan ──
        # These dummy calls ensure Flax registers parameter bindings for all
        # sub-modules (indexer, pool, retrieval_integrator, acc) BEFORE
        # jax.lax.scan runs.  Without them, Dense layers inside the scan body
        # hold a reference to an intermediate tracer created outside the scan,
        # causing an UnexpectedTracerError.  The warm-up outputs are discarded;
        # only the side-effect of Flax module tracing matters.
        # NOTE: The Bug #5 fix in the diagnostic report (removing these calls)
        # does NOT apply to the installed Flax version — keep them.
        _mu, _sigma = self.indexer(
            jnp.zeros((B, T, D)), sigma_max_scale=sigma_max_scale
        )
        if self.config.use_2d_pool:
            _ = self.pool(
                jnp.zeros((B,)), jnp.zeros((B,)), jnp.zeros((B,))
            )
        else:
            _ = self.pool(jnp.zeros((B,)), jnp.zeros((B,)))
        _ = self.retrieval_integrator(
            jnp.zeros((B, T, D + self.config.controller_hidden_dim))
        )
        _ = self.acc(state_hidden, state_hidden, 0, halt_prob, halted_mask)

        use_2d    = self.config.use_2d_pool
        H         = self.config.num_indexer_heads
        SW_FACTOR = self.config.pool_super_window_factor  # Opt-2 knob
        USE_SW    = SW_FACTOR > 1

        # ── Opt-2: Pre-fetch a wide super-window from HBM ONCE ────────────────
        # Before the reasoning loop, run the indexer once to get initial mu
        # coordinates, then fetch SW_FACTOR × window_size vectors in a single
        # HBM pass.  Passing the result as a lax.scan carry instructs XLA to
        # hold it in on-chip SRAM, so per-iteration slices cost ~1 ns instead
        # of ~100 ns.  When SW_FACTOR == 1 this block is skipped entirely and
        # the original per-iteration HBM path is used.
        if USE_SW:
            # Warm-up calls for the new super-window helpers so Flax traces them
            # before the scan (same pattern as the existing warm-up block above).
            if use_2d:
                _sw_dim = self.pool.window_size * SW_FACTOR
                _ = self.pool.fetch_super_window_2d(
                    jnp.zeros((B,)), jnp.zeros((B,)), SW_FACTOR
                )
                _ = self.pool.__call_from_super_window_2d__(
                    jnp.zeros((B, _sw_dim, _sw_dim, D), dtype=jnp.bfloat16),
                    jnp.zeros((B,), dtype=jnp.int32),
                    jnp.zeros((B,), dtype=jnp.int32),
                    jnp.zeros((B,)), jnp.zeros((B,)), jnp.zeros((B,)),
                )
            else:
                _sw_dim = self.pool.window_size * SW_FACTOR
                _ = self.pool.fetch_super_window(jnp.zeros((B,)), SW_FACTOR)
                _ = self.pool.__call_from_super_window__(
                    jnp.zeros((B, _sw_dim, D), dtype=jnp.bfloat16),
                    jnp.zeros((B,), dtype=jnp.int32),
                    jnp.zeros((B,)), jnp.zeros((B,)),
                )

            # Run real initial indexer pass to anchor the super-window
            _mu_init, _ = self.indexer(state_hidden, sigma_max_scale=sigma_max_scale)
            if use_2d:
                heads_per_dim = max(1, H // 2)
                _mu_r0 = _mu_init[:, 0]
                _mu_c0 = _mu_init[:, min(heads_per_dim, H - 1)]
                init_sw, init_sw_r, init_sw_c = self.pool.fetch_super_window_2d(
                    _mu_r0, _mu_c0, SW_FACTOR
                )
                sw_carry = (init_sw, init_sw_r, init_sw_c)
            else:
                _mu_1d0 = _mu_init[:, 0]
                init_sw, init_sw_start = self.pool.fetch_super_window(
                    _mu_1d0, SW_FACTOR
                )
                sw_carry = (init_sw, init_sw_start)

        # ── reasoning_step ────────────────────────────────────────────────────
        def reasoning_step(carry, i):
            if USE_SW:
                if use_2d:
                    s_hidden, h_prob, h_mask, (sw, sw_r, sw_c) = carry
                else:
                    s_hidden, h_prob, h_mask, (sw, sw_s) = carry
            else:
                s_hidden, h_prob, h_mask = carry

            prev_s_hidden = s_hidden

            # ── 1. Multi-head indexing with runtime sigma scale ─────────────
            with jax.profiler.TraceAnnotation("LearnedIndexer_Forward"):
                mu, sigma = self.indexer(s_hidden, sigma_max_scale=sigma_max_scale)
            # mu: (B, H), sigma: (B, H)

            # ── 2. Per-head pool retrieval — vectorized with jax.vmap ─────────
            if use_2d:
                heads_per_dim = max(1, H // 2)
                h_row_ids = jnp.arange(heads_per_dim)
                h_col_ids = jnp.minimum(h_row_ids + heads_per_dim, H - 1)

                mu_r    = mu[:, h_row_ids]
                mu_c    = mu[:, h_col_ids]
                sigma_h = (sigma[:, h_row_ids] + sigma[:, h_col_ids]) / 2.0

                if USE_SW:
                    # ── SRAM path: slice from the in-scan super-window carry ──
                    def pool2d_head_sw(mu_r_h, mu_c_h, sig_h):
                        return self.pool.__call_from_super_window_2d__(
                            sw, sw_r, sw_c, mu_r_h, mu_c_h, sig_h
                        )
                    with jax.profiler.TraceAnnotation("CoordinateMassivePool2D_SW_vmap"):
                        retrieved_all, start_all = jax.vmap(
                            pool2d_head_sw, in_axes=(1, 1, 1), out_axes=(1, 1)
                        )(mu_r, mu_c, sigma_h)

                    # Refresh super-window for the next iteration (track new mu)
                    new_mu_r = jnp.mean(mu_r, axis=1)
                    new_mu_c = jnp.mean(mu_c, axis=1)
                    new_sw, new_sw_r, new_sw_c = self.pool.fetch_super_window_2d(
                        new_mu_r, new_mu_c, SW_FACTOR
                    )
                    new_sw_carry = (new_sw, new_sw_r, new_sw_c)
                else:
                    # ── Original HBM path ────────────────────────────────────
                    def pool2d_head(mu_r_h, mu_c_h, sig_h):
                        return self.pool(mu_r_h, mu_c_h, sig_h)
                    with jax.profiler.TraceAnnotation("CoordinateMassivePool2D_vmap"):
                        retrieved_all, start_all = jax.vmap(
                            pool2d_head, in_axes=(1, 1, 1), out_axes=(1, 1)
                        )(mu_r, mu_c, sigma_h)

                retrieved     = jnp.mean(retrieved_all, axis=1)  # (B, D)
                start_indices = start_all.reshape(-1)             # (B*P,)

            else:
                if USE_SW:
                    # ── SRAM path (1D) ───────────────────────────────────────
                    def pool1d_head_sw(mu_h, sig_h):
                        return self.pool.__call_from_super_window__(
                            sw, sw_s, mu_h, sig_h
                        )
                    with jax.profiler.TraceAnnotation("CoordinateMassivePool1D_SW_vmap"):
                        retrieved_all, start_all = jax.vmap(
                            pool1d_head_sw, in_axes=(1, 1), out_axes=(1, 1)
                        )(mu, sigma)

                    # Refresh super-window
                    new_mu_1d = jnp.mean(mu, axis=1)
                    new_sw, new_sw_s = self.pool.fetch_super_window(new_mu_1d, SW_FACTOR)
                    new_sw_carry = (new_sw, new_sw_s)
                else:
                    # ── Original HBM path (1D) ───────────────────────────────
                    def pool1d_head(mu_h, sig_h):
                        return self.pool(mu_h, sig_h)
                    with jax.profiler.TraceAnnotation("CoordinateMassivePool1D_vmap"):
                        retrieved_all, start_all = jax.vmap(
                            pool1d_head, in_axes=(1, 1), out_axes=(1, 1)
                        )(mu, sigma)

                retrieved     = jnp.mean(retrieved_all, axis=1)  # (B, D)
                start_indices = start_all.reshape(-1)             # (B*H,)

            # ── Mean sigma for logging and precision loss ─────────────────────
            mean_sigma_step = jnp.mean(sigma)

            if DEBUG_REASONING_LOOPS:
                halt_rate = jnp.mean(h_mask)
                retrieved_norm = jnp.sqrt(jnp.mean(retrieved ** 2))
                hidden_norm = jnp.sqrt(jnp.mean(s_hidden ** 2))
                jax.debug.print(
                    "[ReasoningLoop] loop={i} | mean_sigma={sigma:.4f} | "
                    "halt_rate={halt:.3f} | retrieved_l2={ret:.4f} | hidden_l2={hid:.4f}",
                    i=i,
                    sigma=mean_sigma_step,
                    halt=halt_rate,
                    ret=retrieved_norm,
                    hid=hidden_norm,
                )

            # ── 3. Integrate retrieved knowledge ───────────────────────────────
            with jax.profiler.TraceAnnotation("Retrieval_Integrator"):
                retrieved_expanded = jnp.expand_dims(retrieved, 1).repeat(T, axis=1)
                combined = jnp.concatenate([s_hidden, retrieved_expanded], axis=-1)
                integrated = self.retrieval_integrator(combined)

            # ── 4. ACC: accumulate state and decide whether to halt ────────────
            with jax.profiler.TraceAnnotation("AdaptiveComputeController"):
                new_s_hidden, h_prob, new_h_mask = self.acc(
                    s_hidden,
                    s_hidden + integrated,
                    i,
                    h_prob,
                    h_mask,
                )

            update_mask = 1.0 - h_mask
            s_hidden = update_mask * new_s_hidden + h_mask * prev_s_hidden

            # ── Preserve carry dtypes for jax.lax.scan ────────────────────────
            carry_dtype = prev_s_hidden.dtype
            s_hidden   = s_hidden.astype(carry_dtype)
            h_prob     = h_prob.astype(carry_dtype)
            new_h_mask = new_h_mask.astype(carry_dtype)

            if USE_SW:
                new_carry = (s_hidden, h_prob, new_h_mask, new_sw_carry)
            else:
                new_carry = (s_hidden, h_prob, new_h_mask)

            return new_carry, (start_indices, mean_sigma_step)

        # ── Optional gradient checkpointing on reasoning_step ─────────────────
        # The old tracer-leak (TracerBoolConversionError) was because the
        # previous reasoning_step closed over `deterministic` (a JAX bool
        # tracer).  In _encode_hidden, reasoning_step only closes over `self`,
        # `sigma_max_scale`, `use_2d`, `H`, `T` — concrete Python values —
        # so jax.checkpoint is now safe to use.
        #
        # Without checkpointing, the backward stores 18+ buffers of shape
        # f32[max_loops, B/chips, T, D] = ~450 MB each → 8+ GB total at BS=200.
        # With checkpointing, XLA recomputes reasoning_step during backward
        # keeping peak memory to one loop iteration at a time.
        _scan_fn = reasoning_step
        if self.config.gradient_checkpointing:
            _scan_fn = jax.checkpoint(reasoning_step)

        if USE_SW:
            init_carry = (state_hidden, halt_prob, halted_mask, sw_carry)
        else:
            init_carry = (state_hidden, halt_prob, halted_mask)

        final_carry, (all_indices, sigma_per_loop) = jax.lax.scan(
            _scan_fn,
            init_carry,
            jnp.arange(self.config.max_reasoning_loops),
        )

        if USE_SW:
            state_hidden, halt_prob, halted_mask, _ = final_carry
        else:
            state_hidden, halt_prob, halted_mask = final_carry

        # all_indices: (max_loops, heads*B) → transpose to (B*heads, max_loops)
        all_indices = jnp.transpose(all_indices, (1, 0))

        # mean_sigma averaged across all reasoning loops
        mean_sigma = jnp.mean(sigma_per_loop)

        return state_hidden, all_indices, mean_sigma
