import math

import jax
import jax.numpy as jnp
import flax.linen as nn

# ---------------------------------------------------------------------------
# Pallas TPU Splash Attention — imported once at module load.
#
# Splash attention replaces the older flash_attention because:
#   - flash_attention uses Mosaic/Pallas kernels that cannot be automatically
#     partitioned across multiple devices (raises NotImplementedError).
#   - splash_attention supports multi-device via jax.vmap over the batch axis:
#     each device runs the kernel independently on its local batch shard,
#     which is exactly how data-parallel XLA SPMD works.
#
# Falls back to standard nn.dot_product_attention on CPU/GPU.
# ---------------------------------------------------------------------------
try:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        make_splash_mha_single_device,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
        CausalMask,
        MultiHeadMask,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import (
        BlockSizes,
    )
    _SPLASH_FA_AVAILABLE = jax.devices()[0].platform == "tpu"
except Exception:
    _SPLASH_FA_AVAILABLE = False
    make_splash_mha_single_device = None
    CausalMask = None
    MultiHeadMask = None
    BlockSizes = None


def _use_pallas(flag: bool, seq_len: int) -> bool:
    """Return True only when all conditions for Pallas splash_attention are met.

    Splash attention requires:
      - the flag to be explicitly set
      - TPU backend at runtime
      - seq_len >= 128 (minimum block size)
      - seq_len divisible by 128 (block tiling constraint)

    Multi-device is supported: splash_attention runs per-sample via jax.vmap,
    so each device processes its local batch shard independently.
    """
    return (
        flag
        and _SPLASH_FA_AVAILABLE
        and seq_len >= 128
        and seq_len % 128 == 0
    )


class FlashCausalSelfAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    use_flash_attention: bool = False

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        # QKV projection
        qkv = nn.Dense(3 * self.hidden_dim, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # -> (B, T, H, D)
        q = q.reshape(B, T, self.num_heads, head_dim)
        k = k.reshape(B, T, self.num_heads, head_dim)
        v = v.reshape(B, T, self.num_heads, head_dim)

        if _use_pallas(self.use_flash_attention, T):
            # ----------------------------------------------------------------
            # Pallas TPU Splash Attention path — multi-device compatible.
            #
            # Splash attention has no batch dimension: inputs are (H, T, D).
            # We use jax.vmap to map over the batch axis so each sample in
            # the local device shard is processed independently.
            #
            # Mask: MultiHeadMask wrapping one CausalMask per head.
            #   Created at trace time (T and num_heads are concrete in JIT).
            #
            # Scaling: splash_attention has no sm_scale argument.
            #   Pre-scale q by 1/sqrt(head_dim) before calling the kernel.
            # ----------------------------------------------------------------
            q = jnp.transpose(q, (0, 2, 1, 3))   # (B, H, T, D)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            # Pre-scale q (equivalent to sm_scale in flash_attention)
            q = q * (1.0 / math.sqrt(head_dim))

            # Build kernel at trace time — T and num_heads are static shapes
            _causal_mask = MultiHeadMask(
                masks=[CausalMask((T, T)) for _ in range(self.num_heads)]
            )
            _block = min(128, T)
            _splash_mha = make_splash_mha_single_device(
                mask=_causal_mask,
                block_sizes=BlockSizes(block_q=_block, block_kv=_block),
            )

            # vmap over batch: (B, H, T, D) → runs (H, T, D) per sample
            y = jax.vmap(_splash_mha)(q, k, v)    # (B, H, T, D)
            y = jnp.transpose(y, (0, 2, 1, 3))    # (B, T, H, D)
        else:
            # ----------------------------------------------------------------
            # Standard Flax dot-product attention fallback.
            # Used on CPU/GPU, or when seq_len < 128 / not a multiple of 128.
            # The causal mask is passed as a bias from the controller.
            # ----------------------------------------------------------------
            dropout_rng = (
                self.make_rng("dropout")
                if not deterministic and self.dropout_rate > 0
                else None
            )
            y = nn.dot_product_attention(
                q,
                k,
                v,
                bias=mask,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
            )

        y = y.reshape(B, T, self.hidden_dim)
        y = nn.Dense(self.hidden_dim, use_bias=False)(y)

        if not deterministic:
            y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)

        return y


class TinyFFN(nn.Module):
    hidden_dim: int
    ff_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(self.ff_dim)(x)
        x = nn.gelu(x)
        if not deterministic:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim)(x)
        if not deterministic:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        return x


class TinyTransformerLayer(nn.Module):
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0
    use_flash_attention: bool = False

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        norm1 = nn.LayerNorm()(x)
        attn_out = FlashCausalSelfAttention(
            self.hidden_dim,
            self.num_heads,
            self.dropout_rate,
            self.use_flash_attention,
        )(norm1, mask=mask, deterministic=deterministic)
        x = x + attn_out

        norm2 = nn.LayerNorm()(x)
        ffn_out = TinyFFN(self.hidden_dim, self.ff_dim, self.dropout_rate)(
            norm2, deterministic=deterministic
        )
        x = x + ffn_out
        return x
