import math

import jax.numpy as jnp
import flax.linen as nn

# ---------------------------------------------------------------------------
# Pallas TPU Flash Attention — imported once at module load.
# Falls back gracefully if the Pallas kernel isn't available (CPU / GPU env).
# The module ships inside the standard jaxlib wheel but the kernel only
# executes on TPU; importing on GPU/CPU succeeds but calling it raises.
# We therefore gate on both the import AND the runtime platform.
# ---------------------------------------------------------------------------
import jax as _jax

try:
    from jax.experimental.pallas.ops.tpu.flash_attention import (
        flash_attention as pallas_flash_attention,
    )
    _PALLAS_FA_AVAILABLE = _jax.devices()[0].platform == "tpu"
except Exception:  # ImportError or missing XLA plugin
    _PALLAS_FA_AVAILABLE = False


def _use_pallas(flag: bool, seq_len: int) -> bool:
    """Return True only when all conditions for Pallas flash_attention are met.

    Pallas flash_attention requires:
      - the flag to be explicitly set
      - TPU backend at runtime (module is present in all wheels, but the
        kernel only lowers on TPU)
      - seq_len >= 128 (MIN_BLOCK_SIZE)
      - seq_len divisible by 128 (block tiling constraint)
    """
    return (
        flag
        and _PALLAS_FA_AVAILABLE
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
            # Pallas TPU Flash Attention path
            # Expected shape: (B, H, T, D)
            # causal=True handles the autoregressive mask internally — no
            # external bias needed.  sm_scale is 1/sqrt(head_dim) per the
            # standard scaled dot-product attention formula.
            # ----------------------------------------------------------------
            q = jnp.transpose(q, (0, 2, 1, 3))   # (B, H, T, D)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            sm_scale = 1.0 / math.sqrt(head_dim)
            y = pallas_flash_attention(q, k, v, causal=True, sm_scale=sm_scale)

            y = jnp.transpose(y, (0, 2, 1, 3))   # (B, T, H, D)
        else:
            # ----------------------------------------------------------------
            # Standard Flax dot-product attention fallback
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
