import jax
import jax.numpy as jnp
import flax.linen as nn
from dpsn_r_jax.config import DPSNRConfig, PoolConfig
from dpsn_r_jax.models.controller import TinyController
from dpsn_r_jax.models.memory import CoordinateMassivePool, LearnedIndexer
from dpsn_r_jax.models.reasoning import AdaptiveComputeController


class DPSNR(nn.Module):
    config: DPSNRConfig

    def setup(self):
        # Conditionally apply gradient checkpointing (rematerialization)
        # to the heavy components of the model.
        if self.config.gradient_checkpointing:
            controller_cls = nn.remat(TinyController)
            acc_cls = nn.remat(AdaptiveComputeController)
        else:
            controller_cls = TinyController
            acc_cls = AdaptiveComputeController

        self.controller = controller_cls(self.config)
        self.indexer = LearnedIndexer(self.config.controller_hidden_dim)
        self.pool = CoordinateMassivePool(
            PoolConfig(
                self.config.pool_total_vectors, self.config.controller_hidden_dim
            ),
            window_size=self.config.max_k,
        )
        self.acc = acc_cls(
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

    def __call__(self, input_ids, deterministic=True):
        # 1. Encode
        # Use __call__ to support rematerialization if enabled
        hidden = self.controller(input_ids, deterministic=deterministic)

        # 2. Reasoning Loop
        state_hidden = hidden
        B, T, D = hidden.shape

        halt_prob = jnp.zeros((B, T, 1))
        halted_mask = jnp.zeros((B, T, 1))

        # Initialize sub-modules before scan to avoid UnexpectedTracerError
        # (JAX transformations like scan/jit do not allow parameter creation inside)
        _ = self.indexer(jnp.zeros((B, D)))
        _ = self.pool(jnp.zeros((B,)), jnp.zeros((B,)))
        _ = self.retrieval_integrator(
            jnp.zeros((B, T, D + self.config.controller_hidden_dim))
        )
        _ = self.acc(state_hidden, state_hidden, 0, halt_prob, halted_mask)

        # We use a functional scan for reasoning steps to support easy checkpointing
        def reasoning_step(carry, i):
            s_hidden, h_prob, h_mask = carry
            prev_s_hidden = s_hidden

            pooled_state = s_hidden[:, -1, :]
            mu, sigma = self.indexer(pooled_state)

            retrieved, start_indices = self.pool(mu, sigma)

            retrieved_expanded = jnp.expand_dims(retrieved, 1).repeat(T, axis=1)

            combined = jnp.concatenate([s_hidden, retrieved_expanded], axis=-1)
            integrated = self.retrieval_integrator(combined)

            # Step (ACC is already rematerialized if requested)
            new_s_hidden, h_prob, new_h_mask = self.acc(
                s_hidden,
                s_hidden + integrated,  # input to accumulation
                i,
                h_prob,
                h_mask,
            )

            # Mask format: (B, T, 1) -> Broadcast to (B, T, D)
            update_mask = 1.0 - h_mask
            s_hidden = update_mask * new_s_hidden + h_mask * prev_s_hidden

            return (s_hidden, h_prob, new_h_mask), start_indices

        # Checkpoint the scan body if requested
        if self.config.gradient_checkpointing:
            reasoning_step = jax.checkpoint(reasoning_step)

        init_carry = (state_hidden, halt_prob, halted_mask)
        (state_hidden, halt_prob, halted_mask), all_indices = jax.lax.scan(
            reasoning_step,
            init_carry,
            jnp.arange(self.config.max_reasoning_loops),
        )

        # 3. Decode
        logits = self.controller.decode(state_hidden)

        # Transpose all_indices from (max_loops, B) to (B, max_loops)
        all_indices = jnp.transpose(all_indices, (1, 0))
        return logits, (self.config.max_reasoning_loops, all_indices)
