import jax
import jax.numpy as jnp
import flax.linen as nn
from dpsn_r_jax.config import DPSNRConfig
from dpsn_r_jax.models.controller import TinyController
from dpsn_r_jax.models.memory import (
    CoordinateMassivePool2D,
    LearnedIndexer,
)
import jax.profiler
from dpsn_r_jax.models.reasoning import AdaptiveComputeController
from dpsn_r_jax.utils.component_timer import ctimer

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

        # 2D Grid Pool: each coordinate only needs 1/sqrt(N) precision.
        # The window_size here is per-axis; total retrieved = axis_window × axis_window.
        axis_window = max(2, int(self.config.max_k ** 0.5))
        self.pool = CoordinateMassivePool2D(
            rows=self.config.pool_grid_rows,
            cols=self.config.pool_grid_cols,
            hidden_dim=self.config.controller_hidden_dim,
            window_size=axis_window,
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

        # ── Prefetch reasoning query projections (only when enabled) ──────────
        # Two lightweight Dense layers used inside the prefetch scan body.
        # query_attn : scores each token position → attention-pool the sequence
        # query_proj : projects the pooled (B, D) vector into key space before
        #              the dot-product comparison against SRAM candidates.
        if self.config.prefetch_reasoning:
            self.prefetch_query_attn = nn.Dense(1, use_bias=False)
            self.prefetch_query_proj = nn.Dense(self.config.controller_hidden_dim)

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
        # ── Timing mark: encode (controller + reasoning loop) starting ────────
        ctimer.mark("00_encode_start", input_ids.astype(jnp.float32))

        state_hidden, all_indices, mean_sigma, _, _, _, _ = self._encode_hidden(
            input_ids, deterministic, sigma_max_scale
        )

        # 3. Decode — the expensive (B, T, V) step
        logits = self.controller.decode(state_hidden)
        # ── Timing mark: LM-head decode done (full forward complete) ─────────
        ctimer.mark("09_decode_done", logits)

        return logits, (self.config.max_reasoning_loops, all_indices, mean_sigma)

    def encode_to_hidden(self, input_ids, deterministic=True, sigma_max_scale: float = 1.0, retrieved_probes=None):
        """Run controller + reasoning loop and return state_hidden WITHOUT the LM head.

        Used by chunked_lm_loss in trainer.py to avoid materialising the full
        (B, T, vocab) logits tensor.  Only the compact (B, T, D) hidden tensor
        is returned; the LM head is applied later in small batch chunks.

        Returns:
            state_hidden: (B, T, D)
            aux:          (max_loops, all_indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d)
        """
        state_hidden, all_indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d = self._encode_hidden(
            input_ids, deterministic, sigma_max_scale, retrieved_probes=retrieved_probes
        )
        return state_hidden, (self.config.max_reasoning_loops, all_indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d)

    def _encode_hidden(self, input_ids, deterministic=True, sigma_max_scale: float = 1.0, retrieved_probes=None):
        """Core shared encoder: controller + reasoning loop.

        Returns (state_hidden, all_indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d).
        Called by both __call__ and encode_to_hidden so they share code.
        """
        # 1. Encode
        # MUST pass deterministic as a positional argument so static_argnums=(1,) catches it!
        with jax.profiler.TraceAnnotation("TinyController_Forward"):
            hidden = self.controller(input_ids, deterministic)
        # ── Timing mark: TinyController finished ─────────────────────────────
        # jax.debug.callback fires at actual XLA execution time (not trace time).
        # 'hidden' is the trigger array — sequences the mark AFTER the controller.
        ctimer.mark("01_controller_done", hidden)

        # 2. Reasoning Loop — branch on prefetch_reasoning flag
        B, T, D = hidden.shape

        if self.config.prefetch_reasoning:
            # ── Prefetch path: one HBM fetch, all loops read from SRAM ───────
            with jax.profiler.TraceAnnotation("PrefetchReasoning"):
                state_hidden, all_indices, mean_sigma = self._prefetch_encode(
                    hidden, sigma_max_scale
                )
            ctimer.mark("08_all_reasoning_loops_done", state_hidden)
            # Dummy pool info — sparse gradient not supported on prefetch path
            H_dim = max(1, self.config.num_indexer_heads // 2)
            R = self.config.max_reasoning_loops
            dummy_f = jnp.zeros((R, B, H_dim), dtype=jnp.float32)
            dummy_i = jnp.zeros((R, B, H_dim), dtype=jnp.int32)
            return state_hidden, all_indices, mean_sigma, dummy_f, dummy_f, dummy_f, dummy_i

        # ── Original path: per-iteration HBM fetching ─────────────────────────
        state_hidden = hidden
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
        _ = self.pool(jnp.zeros((B,)), jnp.zeros((B,)), jnp.zeros((B,)))
        _ = self.retrieval_integrator(
            jnp.zeros((B, T, D + self.config.controller_hidden_dim))
        )
        _ = self.acc(state_hidden, state_hidden, 0, halt_prob, halted_mask)
        # ── Timing mark: warm-up tracing finished, reasoning scan about to start ─
        ctimer.mark("02_warmup_done__scan_starting", state_hidden)

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

            # Run real initial indexer pass to anchor the super-window
            _mu_init, _ = self.indexer(state_hidden, sigma_max_scale=sigma_max_scale)
            heads_per_dim = max(1, H // 2)
            _mu_r0 = _mu_init[:, 0]
            _mu_c0 = _mu_init[:, min(heads_per_dim, H - 1)]
            init_sw, init_sw_r, init_sw_c = self.pool.fetch_super_window_2d(
                _mu_r0, _mu_c0, SW_FACTOR
            )
            sw_carry = (init_sw, init_sw_r, init_sw_c)

        # ── reasoning_step ────────────────────────────────────────────────────
        def reasoning_step(carry, i, retrieved_probes=retrieved_probes):
            if USE_SW:
                s_hidden, h_prob, h_mask, (sw, sw_r, sw_c) = carry
            else:
                s_hidden, h_prob, h_mask = carry

            prev_s_hidden = s_hidden

            # ── Timing mark: start of this reasoning iteration ────────────────
            # 'i' is a JAX traced scalar — use jax.debug.print to show it in console.
            # ctimer.mark fires once per scan iteration.
            ctimer.mark("03_iter_start", s_hidden)

            # ── 1. Multi-head indexing with runtime sigma scale ─────────────
            with jax.profiler.TraceAnnotation("LearnedIndexer_Forward"):
                mu, sigma = self.indexer(s_hidden, sigma_max_scale=sigma_max_scale)
            # mu: (B, H), sigma: (B, H)
            # ── Timing mark: indexer done ─────────────────────────────────────
            ctimer.mark("04_indexer_done", mu)

            # ── 2. Per-head pool retrieval — vectorized with jax.vmap ─────────
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

            # ── Sparse-gradient probe injection ──────────────────────────────────────
            # When training with sparse pool gradients, retrieved_probes is a (R, B, D)
            # zero tensor. By differentiating w.r.t. probe, we get ∂loss/∂retrieved
            # without computing a full (512,512,768) pool gradient tensor.
            if retrieved_probes is not None:
                from jax import lax
                retrieved = retrieved + lax.dynamic_index_in_dim(
                    retrieved_probes, i, axis=0, keepdims=False
                )

            # ── Timing mark: pool retrieval done ─────────────────────────────
            ctimer.mark("05_pool_retrieval_done", retrieved)

            # ── Mean sigma for logging and precision loss ─────────────────────
            mean_sigma_step = jnp.mean(sigma)

            # ── Per-iteration diagnostic print (always on when ctimer enabled,
            #    or when DEBUG_REASONING_LOOPS is True).
            #    jax.debug.print is the ONLY way to print tensor values inside
            #    jax.jit / jax.lax.scan — normal print() runs at trace time only.
            if DEBUG_REASONING_LOOPS or ctimer.enabled:
                halt_rate      = jnp.mean(h_mask)
                retrieved_norm = jnp.sqrt(jnp.mean(retrieved ** 2))
                hidden_norm    = jnp.sqrt(jnp.mean(s_hidden ** 2))
                jax.debug.print(
                    "[ReasoningLoop] iter={i} | sigma={sigma:.4f} | "
                    "halt_rate={halt:.3f} | retrieved_l2={ret:.4f} | hidden_l2={hid:.4f}",
                    i=i,
                    sigma=mean_sigma_step,
                    halt=halt_rate,
                    ret=retrieved_norm,
                    hid=hidden_norm,
                )

            # ── 3. Integrate retrieved knowledge ───────────────────────────────
            with jax.profiler.TraceAnnotation("Retrieval_Integrator"):
                retrieved_expanded = jnp.broadcast_to(retrieved[:, None, :], (B, T, D))
                combined = jnp.concatenate([s_hidden, retrieved_expanded], axis=-1)
                integrated = self.retrieval_integrator(combined)
            # ── Timing mark: retrieval integrator done ────────────────────────
            ctimer.mark("06_integrator_done", integrated)

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

            # ── Timing mark: full reasoning iteration done ────────────────────
            # This fires once per lax.scan iteration → per-iteration cost visible
            # in ctimer.print_summary() as repeated "07_acc_iter_done[N]" rows.
            ctimer.mark("07_acc_iter_done", s_hidden)

            if USE_SW:
                new_carry = (s_hidden, h_prob, new_h_mask, new_sw_carry)
            else:
                new_carry = (s_hidden, h_prob, new_h_mask)

            return new_carry, (start_indices, mean_sigma_step, mu_r, mu_c, sigma_h, start_all)

        # ── Optional gradient checkpointing on reasoning_step ─────────────────
        # The old tracer-leak (TracerBoolConversionError) was because the
        # previous reasoning_step closed over `deterministic` (a JAX bool
        # tracer).  In _encode_hidden, reasoning_step only closes over `self`,
        # `sigma_max_scale`, `H`, `T` — concrete Python values —
        # so jax.checkpoint is now safe to use.
        #
        # Without checkpointing, the backward stores 18+ buffers per loop
        # → 8+ GB total at BS=200.  With checkpointing, XLA recomputes
        # reasoning_step during backward, keeping peak memory to one iteration.
        _scan_fn = reasoning_step
        if self.config.gradient_checkpointing:
            _scan_fn = jax.checkpoint(reasoning_step)

        if USE_SW:
            init_carry = (state_hidden, halt_prob, halted_mask, sw_carry)
        else:
            init_carry = (state_hidden, halt_prob, halted_mask)

        final_carry, (all_indices, sigma_per_loop, all_mu_r, all_mu_c, all_sigma_h, all_start_2d) = jax.lax.scan(
            _scan_fn,
            init_carry,
            jnp.arange(self.config.max_reasoning_loops),
        )

        if USE_SW:
            state_hidden, halt_prob, halted_mask, _ = final_carry
        else:
            state_hidden, halt_prob, halted_mask = final_carry

        # ── Timing mark: all reasoning loop iterations complete ───────────────
        ctimer.mark("08_all_reasoning_loops_done", state_hidden)

        # all_indices: (max_loops, heads*B) → transpose to (B*heads, max_loops)
        all_indices = jnp.transpose(all_indices, (1, 0))

        # mean_sigma averaged across all reasoning loops
        mean_sigma = jnp.mean(sigma_per_loop)

        return state_hidden, all_indices, mean_sigma, all_mu_r, all_mu_c, all_sigma_h, all_start_2d

    # ─────────────────────────────────────────────────────────────────────────
    # Prefetch Reasoning path
    # ─────────────────────────────────────────────────────────────────────────

    def _prefetch_encode(self, hidden, sigma_max_scale: float = 1.0):
        """Prefetch-once, reason-in-SRAM encoding path.

        Design
        ──────
        Stage 1  One indexer forward pass → initial (mu_r, mu_c) coordinates.
        Stage 2  ONE lax.dynamic_slice call fetches a patch_size × patch_size
                 region from HBM into a (B, K, D) tensor where K = patch_size².
        Stage 3  lax.scan carries that tensor as part of its carry pytree.
                 XLA keeps carry tensors in on-chip SRAM across iterations, so
                 every reasoning step reads from SRAM (~1 ns) instead of HBM
                 (~100 ns per dynamic_slice).
        Stage 4  Each scan iteration does scaled dot-product attention over the
                 K SRAM-resident candidates instead of a Gaussian-weighted
                 dynamic_slice. The retrieval_integrator and ACC modules are
                 reused unchanged.

        Args:
            hidden:          (B, T, D) — controller output.
            sigma_max_scale: Sigma annealing multiplier (1.0 = full range).

        Returns:
            state_hidden: (B, T, D)
            all_indices:  (B, max_reasoning_loops) flat patch-start indices
                          for sparse-Adam pool update.
            mean_sigma:   scalar — mean sigma from the initial indexer call.
        """
        B, T, D = hidden.shape
        model_dtype = hidden.dtype

        # ── Stage 1: single indexer pass (one HBM probe, outside the scan) ───
        mu_init, sigma_init = self.indexer(hidden, sigma_max_scale=sigma_max_scale)
        ctimer.mark("04_indexer_done", mu_init)

        H             = self.config.num_indexer_heads
        heads_per_dim = max(1, H // 2)
        mu_r = mu_init[:, 0]
        mu_c = mu_init[:, min(heads_per_dim, H - 1)]

        # ── Stage 2: ONE HBM fetch — prefetch_size² candidates ────────────────
        patch_size = self.config.prefetch_size
        with jax.profiler.TraceAnnotation("PrefetchReasoning_HBM_Fetch"):
            candidates_2d, patch_r_start, patch_c_start = self.pool.fetch_patch_2d(
                mu_r, mu_c, patch_size
            )
        # candidates_2d : (B, patch_size, patch_size, D)  — bfloat16 from pool
        # Reshape spatial dims → (B, K, D)  K = patch_size²
        K          = patch_size * patch_size
        candidates = candidates_2d.reshape(B, K, D).astype(model_dtype)
        ctimer.mark("05_pool_retrieval_done", candidates)

        # ── Stage 3: Warm-up calls before lax.scan ────────────────────────────
        # Flax requires every module called inside lax.scan to be invoked at
        # least once outside it so parameter bindings are registered first.
        halt_prob   = jnp.zeros((B, T, 1), dtype=model_dtype)
        halted_mask = jnp.zeros((B, T, 1), dtype=model_dtype)

        _dummy_candidates = jnp.zeros((B, K, D), dtype=model_dtype)
        _dummy_attn  = self.prefetch_query_attn(hidden)                        # (B, T, 1)
        _dummy_q     = self.prefetch_query_proj(jnp.zeros((B, D), dtype=model_dtype))
        _dummy_integ = self.retrieval_integrator(
            jnp.zeros((B, T, 2 * D), dtype=model_dtype)
        )
        _dummy_acc   = self.acc(hidden, hidden, 0, halt_prob, halted_mask)
        ctimer.mark("02_warmup_done__scan_starting", hidden)

        # ── Stage 4: Reasoning scan — candidates live in SRAM as carry ────────
        def prefetch_step(carry, i):
            s_hidden, h_prob, h_mask, cands = carry
            prev = s_hidden

            ctimer.mark("03_iter_start", s_hidden)

            # ── Cross-attention over SRAM-resident candidates ─────────────────
            # Step a: attention-pool the full sequence → single query vector
            with jax.profiler.TraceAnnotation("PrefetchReasoning_CrossAttn"):
                attn_score = self.prefetch_query_attn(s_hidden)          # (B, T, 1)
                attn_w     = jax.nn.softmax(attn_score, axis=1)          # (B, T, 1)
                query      = jnp.sum(attn_w * s_hidden, axis=1)          # (B, D)

                # Step b: project query → key space
                query = nn.gelu(self.prefetch_query_proj(query))          # (B, D)

                # Step c: scaled dot-product over K SRAM candidates
                # Cast to float32 for softmax numerical stability; cands stays
                # in model_dtype (bf16) in the carry to save SRAM.
                scale     = D ** -0.5
                scores    = jnp.einsum(
                    'bd,bkd->bk',
                    query,
                    cands.astype(jnp.float32),
                ) * scale                                                  # (B, K)
                weights   = jax.nn.softmax(scores, axis=-1)               # (B, K)
                retrieved = jnp.einsum(
                    'bk,bkd->bd',
                    weights,
                    cands.astype(jnp.float32),
                )                                                          # (B, D) fp32

            ctimer.mark("05_pool_retrieval_done", retrieved)

            # ── Integrate retrieved knowledge (reuses existing module) ─────────
            with jax.profiler.TraceAnnotation("Retrieval_Integrator"):
                retrieved_exp = jnp.broadcast_to(
                    retrieved[:, None, :], (B, T, D)
                )
                combined   = jnp.concatenate(
                    [s_hidden, retrieved_exp.astype(model_dtype)], axis=-1
                )
                integrated = self.retrieval_integrator(combined)

            ctimer.mark("06_integrator_done", integrated)

            # ── Adaptive halting (reuses existing ACC module) ──────────────────
            with jax.profiler.TraceAnnotation("AdaptiveComputeController"):
                new_s, h_prob, new_h_mask = self.acc(
                    s_hidden, s_hidden + integrated, i, h_prob, h_mask
                )

            update_mask = 1.0 - h_mask
            s_hidden    = (update_mask * new_s + h_mask * prev).astype(model_dtype)
            h_prob      = h_prob.astype(model_dtype)
            new_h_mask  = new_h_mask.astype(model_dtype)

            ctimer.mark("07_acc_iter_done", s_hidden)

            # cands is passed through unchanged — XLA keeps it in SRAM
            return (s_hidden, h_prob, new_h_mask, cands), None

        _scan_fn = prefetch_step
        if self.config.gradient_checkpointing:
            _scan_fn = jax.checkpoint(prefetch_step)

        (state_hidden, _, _, _), _ = jax.lax.scan(
            _scan_fn,
            (hidden, halt_prob, halted_mask, candidates),
            jnp.arange(self.config.max_reasoning_loops),
        )

        # ── Build all_indices for sparse-Adam pool update ─────────────────────
        # The "touched" pool region is the pre-fetched patch.  We report its
        # flat start position so the trainer updates those pool vectors.
        flat_patch_start = (
            patch_r_start * self.config.pool_grid_cols + patch_c_start
        )                                                                  # (B,)
        # Tile to (B, max_reasoning_loops) to match the trainer's expected shape
        all_indices = jnp.tile(
            flat_patch_start[:, None],
            (1, self.config.max_reasoning_loops),
        )

        mean_sigma = jnp.mean(sigma_init)
        return state_hidden, all_indices, mean_sigma
