import jax.numpy as jnp
import flax.linen as nn
from dpsn_r_jax.config import DPSNRConfig
from dpsn_r_jax.models.layers import TinyTransformerLayer


class TinyController(nn.Module):
    config: DPSNRConfig

    def setup(self):
        self.embedding = nn.Embed(
            self.config.vocab_size, self.config.controller_hidden_dim
        )
        self.pos_encoding = nn.Embed(
            self.config.max_seq_len, self.config.controller_hidden_dim
        )

        ff_dim = int(
            self.config.controller_hidden_dim * self.config.controller_ff_multiplier
        )
        self.layers = [
            TinyTransformerLayer(
                self.config.controller_hidden_dim,
                self.config.controller_num_heads,
                ff_dim,
                self.config.dropout,
            )
            for _ in range(self.config.controller_num_layers)
        ]

        # Output head
        self.final_norm = nn.LayerNorm()
        self.lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

    def __call__(self, input_ids, deterministic=True):
        return self.encode(input_ids, deterministic)

    def encode(self, input_ids, deterministic=True):
        B, T = input_ids.shape

        embed = self.embedding(input_ids)

        pos_ids = jnp.arange(T)[None, :]
        pos_embed = self.pos_encoding(pos_ids)

        x = embed + pos_embed

        mask = nn.make_causal_mask(input_ids)
        # mask is (1, 1, T, T) bool usually? make_causal_mask returns (1, 1, T, T)
        mask = jnp.where(mask, 0, -1e9)

        for layer in self.layers:
            x = layer(x, mask=mask, deterministic=deterministic)

        return x

    def decode(self, hidden):
        x = self.final_norm(hidden)
        logits = self.lm_head(x)
        return logits
