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
            # Flax remat counts the module instance as arg 0, so:
            #   arg 0 = module instance
            #   arg 1 = x  (JAX array — traced normally)
            #   arg 2 = deterministic  (Python bool — must be static)
            # Without static_argnums=(2,), deterministic becomes a JAX tracer
            # and `if not deterministic:` raises TracerBoolConversionError.
            #
            # No checkpoint policy: remat saves only the layer input
            # (f32[B,T,D] = 64MB per layer) and recomputes all intermediates.
            # dots_saveable was tried but saves matmul outputs instead (e.g.
            # f32[B,T,4096] FFN intermediate = 256MB) — 4× more HBM, not less.
            layer_cls = nn.remat(TinyTransformerLayer, static_argnums=(2,))

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

    def __call__(self, input_ids, deterministic=True, seq_pack_ids=None):
        return self.encode(input_ids, deterministic, seq_pack_ids=seq_pack_ids)

    def encode(self, input_ids, deterministic=True, seq_pack_ids=None):
        x = self.embedding(input_ids)   # (B, T, D) — no pos_embed added

        for layer in self.layers:
            # deterministic is the only extra arg; pass positionally so
            # static_argnums=(1,) in nn.remat catches it correctly.
            x = layer(x, deterministic, seq_pack_ids=seq_pack_ids)

        return x

    def decode(self, hidden):
        x = self.final_norm(hidden)
        logits = self.lm_head(x)
        return logits
