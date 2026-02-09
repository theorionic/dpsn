import jax.numpy as jnp
import flax.linen as nn


class AdaptiveComputeController(nn.Module):
    hidden_dim: int
    max_loops: int = 8
    halt_threshold: float = 0.99

    def setup(self):
        self.halt_net = nn.Sequential(
            [nn.Dense(self.hidden_dim // 4), nn.gelu, nn.Dense(1), nn.sigmoid]
        )

        self.state_gate = nn.Sequential([nn.Dense(self.hidden_dim), nn.sigmoid])
        self.state_transform = nn.Dense(self.hidden_dim)
        self.state_norm = nn.LayerNorm()

        self.loop_embed = nn.Embed(32, self.hidden_dim)

    def __call__(
        self, state_hidden, step_output, loop_count, current_halt_prob, halted_mask
    ):
        # Add Loop Embedding
        loop_idx = jnp.array([loop_count], dtype=jnp.int32)
        emb = self.loop_embed(loop_idx)  # (1, D)
        step_output = step_output + emb

        # State Accumulation (Gated Residual)
        combined = jnp.concatenate([step_output, state_hidden], axis=-1)
        g = self.state_gate(combined)

        # Compute candidate new state
        candidate_state = g * self.state_transform(step_output) + (1 - g) * state_hidden
        candidate_state = self.state_norm(candidate_state)

        # Halt Prediction (on candidate state)
        hp = self.halt_net(candidate_state)  # (B, T, 1)

        # Update Halt Probabilities
        # only accumulate if not already halted
        still_running_mask = 1.0 - halted_mask
        new_halt_prob = current_halt_prob + hp * still_running_mask

        # Check if halted now
        # Cast to float for mask math
        is_halted_now = (new_halt_prob >= self.halt_threshold).astype(jnp.float32)
        final_halted_mask = jnp.maximum(halted_mask, is_halted_now)

        return candidate_state, new_halt_prob, final_halted_mask
