import jax.numpy as jnp
import flax.linen as nn
from dpsn_r_jax.config import DPSNRConfig, PoolConfig
from dpsn_r_jax.models.controller import TinyController
from dpsn_r_jax.models.memory import HierarchicalMassivePool, RetrievalRouter
from dpsn_r_jax.models.reasoning import AdaptiveComputeController


class DPSNR(nn.Module):
    config: DPSNRConfig

    def setup(self):
        self.controller = TinyController(self.config)
        self.pool = HierarchicalMassivePool(
            PoolConfig(
                self.config.pool_total_vectors, self.config.controller_hidden_dim
            )
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
        self.router = RetrievalRouter(
            self.config.controller_hidden_dim, self.config.min_k, self.config.max_k
        )

    def __call__(self, input_ids, deterministic=True):
        # 1. Encode
        hidden = self.controller.encode(input_ids, deterministic=deterministic)

        # 2. Reasoning Loop
        state_hidden = hidden
        B, T, D = hidden.shape

        halt_prob = jnp.zeros((B, T, 1))
        halted_mask = jnp.zeros((B, T, 1))

        # Python loop unrolling (efficient for small max_loops on TPU)
        i = 0
        for i in range(self.config.max_reasoning_loops):
            # Save state before step to handle "already halted" samples
            prev_state_hidden = state_hidden

            # Router
            query, k_dynamic, _ = self.router(state_hidden)

            # Retrieve (using MAX_K for static shape)
            retrieved = self.pool(state_hidden, k_dynamic, self.config.max_k)

            # Integrate
            retrieved_expanded = jnp.expand_dims(retrieved, 1).repeat(T, axis=1)
            combined = jnp.concatenate([state_hidden, retrieved_expanded], axis=-1)
            integrated = self.retrieval_integrator(combined)

            # Step
            new_state_hidden, halt_prob, new_halted_mask = self.acc(
                state_hidden,
                state_hidden + integrated,  # input to accumulation
                i,
                halt_prob,
                halted_mask,
            )

            # FREEZE STATE Logic:
            # If a token was ALREADY halted (halted_mask == 1), we should keep its old state.
            # If it just halted this step, we keep the new state (which caused the halt).
            # The mask used here should be the 'halted_mask' from START of step.

            # Mask format: (B, T, 1) -> Broadcast to (B, T, D)
            update_mask = 1.0 - halted_mask
            state_hidden = (
                update_mask * new_state_hidden + halted_mask * prev_state_hidden
            )

            halted_mask = new_halted_mask

        # 3. Decode
        logits = self.controller.decode(state_hidden)

        return logits, i + 1
