import jax
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
        # Flax remat counts the module instance as arg 0, so:
        #   arg 0 = module instance
        #   arg 1 = x  (JAX array — traced normally)
        #   arg 2 = deterministic  (Python bool — must be static)
        # Without static_argnums=(2,), deterministic becomes a JAX tracer
        # and `if not deterministic:` raises TracerBoolConversionError.
        #
        # dots_saveable policy: during remat, save only the outputs of matmuls
        # (QKV projection, FFN dense, etc.) rather than all intermediate
        # activations.  This avoids saving the large (B, T, D) tensors at
        # every sub-operation, cutting HBM traffic by ~3–4× vs plain remat.
        #
        # controller_checkpoint_interval: checkpoint every N-th layer instead
        # of every layer.  With interval=4 on 24 layers we checkpoint layers
        # 0, 4, 8, 12, 16, 20 — the 6 boundary layers still save activations,
        # while the 18 inner layers recompute freely.  This is 6× fewer HBM
        # write/read round-trips per backward pass at the cost of ~10% more
        # peak activation memory (inner layers kept live until the boundary).
        if self.config.gradient_checkpointing:
            interval = getattr(self.config, "controller_checkpoint_interval", 1)
            remated_cls = nn.remat(
                TinyTransformerLayer,
                static_argnums=(2,),
                policy=jax.checkpoint_policies.dots_saveable,
            )
            self.layers = [
                remated_cls(
                    self.config.controller_hidden_dim,
                    self.config.controller_num_heads,
                    ff_dim,
                    self.config.dropout,
                    self.config.use_flash_attention,
                    self.config.attn_window_size,
                )
                if i % interval == 0
                else TinyTransformerLayer(
                    self.config.controller_hidden_dim,
                    self.config.controller_num_heads,
                    ff_dim,
                    self.config.dropout,
                    self.config.use_flash_attention,
                    self.config.attn_window_size,
                )
                for i in range(self.config.controller_num_layers)
            ]
        else:
            self.layers = [
                TinyTransformerLayer(
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
