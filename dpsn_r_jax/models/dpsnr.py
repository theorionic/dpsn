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
from dpsn_r_jax.models.reasoning import AdaptiveComputeController


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
        hidden = self.controller(input_ids, deterministic)

        # 2. Reasoning Loop
        state_hidden = hidden
        B, T, D = hidden.shape

        halt_prob   = jnp.zeros((B, T, 1), dtype=hidden.dtype)
        halted_mask = jnp.zeros((B, T, 1), dtype=hidden.dtype)

        # ── Warm-up calls: force Flax to trace all sub-modules before scan ────
        _mu, _sigma = self.indexer(
            jnp.zeros((B, T, D)), sigma_max_scale=sigma_max_scale
        )
        if self.config.use_2d_pool:
            H = self.config.num_indexer_heads
            # For 2D pool, indexer outputs _mu (B, H) for row and col alternating.
            # Use half-heads for row, half for col.
            h_per_dim = max(1, H // 2)
            _ = self.pool(
                jnp.zeros((B,)), jnp.zeros((B,)), jnp.zeros((B,))
            )
        else:
            _ = self.pool(jnp.zeros((B,)), jnp.zeros((B,)))
        _ = self.retrieval_integrator(
            jnp.zeros((B, T, D + self.config.controller_hidden_dim))
        )
        _ = self.acc(state_hidden, state_hidden, 0, halt_prob, halted_mask)

        use_2d = self.config.use_2d_pool
        H = self.config.num_indexer_heads

        def reasoning_step(carry, i):
            s_hidden, h_prob, h_mask = carry
            prev_s_hidden = s_hidden

            # ── 1. Multi-head indexing with runtime sigma scale ─────────────
            mu, sigma = self.indexer(s_hidden, sigma_max_scale=sigma_max_scale)
            # mu: (B, H), sigma: (B, H)

            # ── 2. Per-head pool retrieval ────────────────────────────────────
            all_retrieved = []
            all_start_indices = []

            if use_2d:
                # 2D pool: each head supplies (mu_row, mu_col) from consecutive
                # pairs of heads. If H=1 reuse same coord for row and col.
                heads_per_dim = max(1, H // 2)
                for h in range(heads_per_dim):
                    h_row = h
                    h_col = min(h + heads_per_dim, H - 1)
                    # Use same sigma across both axes (mean of the two heads)
                    sigma_h = (sigma[:, h_row] + sigma[:, h_col]) / 2.0
                    retrieved_h, start_idx_h = self.pool(
                        mu[:, h_row], mu[:, h_col], sigma_h
                    )
                    all_retrieved.append(retrieved_h)
                    all_start_indices.append(start_idx_h)
            else:
                for h in range(H):
                    retrieved_h, start_idx_h = self.pool(mu[:, h], sigma[:, h])
                    all_retrieved.append(retrieved_h)
                    all_start_indices.append(start_idx_h)

            # Average retrieved vectors across heads → (B, D)
            retrieved = jnp.mean(jnp.stack(all_retrieved, axis=1), axis=1)

            # Concatenate start indices across heads → sparse Adam can update all
            start_indices = jnp.concatenate(all_start_indices, axis=0)   # (heads*B,)

            # ── Mean sigma for logging and precision loss ─────────────────────
            mean_sigma_step = jnp.mean(sigma)   # scalar

            # ── 3. Integrate retrieved knowledge ───────────────────────────────
            retrieved_expanded = jnp.expand_dims(retrieved, 1).repeat(T, axis=1)
            combined = jnp.concatenate([s_hidden, retrieved_expanded], axis=-1)
            integrated = self.retrieval_integrator(combined)

            # ── 4. ACC: accumulate state and decide whether to halt ────────────
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
            # Some ops inside ACC/LayerNorm/sigmoid can upcast to float32 even
            # when the inputs are bfloat16.  scan requires that carry input and
            # output have *identical* dtypes, so we cast back explicitly.
            carry_dtype = prev_s_hidden.dtype
            s_hidden   = s_hidden.astype(carry_dtype)
            h_prob     = h_prob.astype(carry_dtype)
            new_h_mask = new_h_mask.astype(carry_dtype)

            return (s_hidden, h_prob, new_h_mask), (start_indices, mean_sigma_step)

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

        init_carry = (state_hidden, halt_prob, halted_mask)
        (state_hidden, halt_prob, halted_mask), (all_indices, sigma_per_loop) = (
            jax.lax.scan(
                _scan_fn,
                init_carry,
                jnp.arange(self.config.max_reasoning_loops),
            )
        )


        # all_indices: (max_loops, heads*B) → transpose to (B*heads, max_loops)
        all_indices = jnp.transpose(all_indices, (1, 0))

        # mean_sigma averaged across all reasoning loops
        mean_sigma = jnp.mean(sigma_per_loop)

        return state_hidden, all_indices, mean_sigma
