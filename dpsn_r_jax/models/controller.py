import jax.numpy as jnp
import flax.linen as nn
from dpsn_r_jax.config import DPSNRConfig
from dpsn_r_jax.models.layers import TinyTransformerLayer, _use_pallas


class TinyController(nn.Module):
    config: DPSNRConfig

    def setup(self):
        self.embedding = nn.Embed(
            self.config.vocab_size, self.config.controller_hidden_dim
        )
        # NOTE: No learned positional embedding — positions are encoded via
        # RoPE (Rotary Position Embeddings) inside FlashCausalSelfAttention.
        # RoPE encodes relative distances between tokens and generalises beyond
        # the training context length, unlike the old nn.Embed approach.

        ff_dim = int(
            self.config.controller_hidden_dim * self.config.controller_ff_multiplier
        )
        layer_cls = TinyTransformerLayer
        if self.config.gradient_checkpointing:
            layer_cls = nn.remat(TinyTransformerLayer, static_argnums=(1,))

        self.layers = [
            layer_cls(
                self.config.controller_hidden_dim,
                self.config.controller_num_heads,
                ff_dim,
                self.config.dropout,
                self.config.use_flash_attention,
                self.config.attn_window_size,
            )
            for _ in range(self.config.controller_num_layers)
        ]

        # Output head
        self.final_norm = nn.LayerNorm()
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(self, input_ids, deterministic=True):
        return self.encode(input_ids, deterministic)

    def encode(self, input_ids, deterministic=True):
        x = self.embedding(input_ids)   # (B, T, D) — no pos_embed added

        for layer in self.layers:
            # deterministic is the only extra arg; pass positionally so
            # static_argnums=(1,) in nn.remat catches it correctly.
            x = layer(x, deterministic)

        return x

    def decode(self, hidden):
        x = self.final_norm(hidden)
        logits = self.lm_head(x)
        return logits
