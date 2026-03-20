import math

import jax
import jax.numpy as jnp
import flax.linen as nn

# ---------------------------------------------------------------------------
# Pallas TPU Splash Attention
# ---------------------------------------------------------------------------
try:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        make_splash_mha_single_device,
    )
    from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
        CausalMask,
        LocalMask,
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
    LocalMask = None
    MultiHeadMask = None
    BlockSizes = None


def _use_pallas(flag: bool, seq_len: int) -> bool:
    """Return True only when all conditions for Pallas splash_attention are met."""
    return (
        flag
        and _SPLASH_FA_AVAILABLE
        and seq_len >= 128
        and seq_len % 128 == 0
    )


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
#
# Why RoPE instead of learned absolute embeddings:
#   - Encodes *relative* distances between tokens into the QK dot product.
#   - Generalises beyond the training context length (no hard cap).
#   - No extra parameters — positions are baked into Q and K via rotation.
#   - Now the universal standard: LLaMA, Gemma, Mistral, GPT-NeoX all use it.
#
# Implementation follows the original Su et al. (2021) paper.
# cos/sin tables are computed at trace time (T and head_dim are static in JIT)
# so there is zero runtime overhead beyond the two elementwise multiplies.
# ---------------------------------------------------------------------------

def _rotate_half(x):
    """Rotate the last dimension: [x1, x2] → [-x2, x1]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _precompute_rope(seq_len: int, head_dim: int, base: float = 10000.0):
    """Compute RoPE cos/sin tables.

    Returns:
        cos, sin — each shape (1, T, 1, head_dim), ready to broadcast with
                   q/k of shape (B, T, H, head_dim).
    """
    half = head_dim // 2
    # Inverse frequencies: θ_i = 1 / (base^(2i/d))
    inv_freq = 1.0 / (
        base ** (jnp.arange(0, half, dtype=jnp.float32) / half)
    )
    t = jnp.arange(seq_len, dtype=jnp.float32)
    # Outer product: (T, half)
    freqs = jnp.outer(t, inv_freq)
    # Duplicate along head_dim to match full head_dim via rotation trick
    cos = jnp.concatenate([jnp.cos(freqs), jnp.cos(freqs)], axis=-1)  # (T, D)
    sin = jnp.concatenate([jnp.sin(freqs), jnp.sin(freqs)], axis=-1)  # (T, D)
    # Add B and H broadcast dims: (1, T, 1, D)
    return cos[None, :, None, :], sin[None, :, None, :]


def _apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to q and k (shape: B, T, H, D)."""
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Sliding window causal mask (fallback path — standard attention)
# ---------------------------------------------------------------------------

def _make_sliding_window_causal_bias(seq_len: int, window_size: int):
    """Additive attention bias for causal sliding window attention.

    Each token i attends to: positions [max(0, i-window_size+1), i].
    Returns shape (T, T) with 0.0 where attention is allowed, -1e4 elsewhere.
    """
    i = jnp.arange(seq_len)[:, None]   # (T, 1)
    j = jnp.arange(seq_len)[None, :]   # (1, T)
    # causal: j <= i  |  window: j >= i - window_size + 1
    in_window = (j <= i) & (j >= i - window_size + 1)
    return jnp.where(in_window, 0.0, -1e4).astype(jnp.float32)  # (T, T)


# ---------------------------------------------------------------------------
# Attention layer
# ---------------------------------------------------------------------------

class FlashCausalSelfAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0
    use_flash_attention: bool = False
    # 0 = full causal attention; >0 = sliding window of this many tokens.
    # Set to ~512 for large context models — the pool handles long-range memory.
    window_size: int = 0

    @nn.compact
    def __call__(self, x, deterministic=True):
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads

        # QKV projection
        qkv = nn.Dense(3 * self.hidden_dim, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape: (B, T, H*D) → (B, T, H, D)
        q = q.reshape(B, T, self.num_heads, head_dim)
        k = k.reshape(B, T, self.num_heads, head_dim)
        v = v.reshape(B, T, self.num_heads, head_dim)

        # ── Apply RoPE to Q and K (B, T, H, D) ──────────────────────────────
        # Computed at trace time: T and head_dim are static shapes in JIT.
        cos, sin = _precompute_rope(T, head_dim)
        q, k = _apply_rope(q, k, cos, sin)

        if _use_pallas(self.use_flash_attention, T):
            # ----------------------------------------------------------------
            # Pallas TPU Splash Attention — multi-device compatible.
            #
            # Shape contract: splash expects (H, T, D) per sample.
            # We vmap over the batch axis so each device's local shard is
            # processed independently.
            #
            # Mask:
            #   window_size > 0 → LocalMask (causal sliding window)
            #     LocalMask(shape, window_size=(left, right)) where right=0
            #     means no future tokens → already causal, no need for
            #     LogicalAnd with CausalMask.
            #   window_size = 0 → full CausalMask
            #
            # Scale: pre-scale q by 1/√head_dim (splash has no sm_scale arg).
            # ----------------------------------------------------------------
            q = jnp.transpose(q, (0, 2, 1, 3))   # (B, H, T, D)
            k = jnp.transpose(k, (0, 2, 1, 3))
            v = jnp.transpose(v, (0, 2, 1, 3))

            q = q * (1.0 / math.sqrt(head_dim))

            if self.window_size > 0:
                _per_head_mask = LocalMask(
                    (T, T),
                    window_size=(self.window_size - 1, 0),
                    offset=0,
                )
            else:
                _per_head_mask = CausalMask((T, T))

            _splash_mask = MultiHeadMask(
                masks=[_per_head_mask] * self.num_heads
            )
            _block = min(128, T)
            _splash_mha = make_splash_mha_single_device(
                mask=_splash_mask,
                block_sizes=BlockSizes(
                    block_q=_block,
                    block_kv=_block,
                    block_q_dkv=_block,
                    block_kv_dkv=_block,
                    block_q_dq=_block,
                    block_kv_dq=_block,
                ),
            )

            y = jax.vmap(_splash_mha)(q, k, v)    # (B, H, T, D)
            y = jnp.transpose(y, (0, 2, 1, 3))    # (B, T, H, D)

        else:
            # ----------------------------------------------------------------
            # Standard Flax dot-product attention fallback.
            # Used on CPU/GPU or when seq_len constraints aren't met.
            # Mask is built here — causal or causal sliding window.
            # ----------------------------------------------------------------
            if self.window_size > 0:
                bias = _make_sliding_window_causal_bias(T, self.window_size)
                # Expand to (1, 1, T, T) for broadcasting with (B, H, T, T)
                bias = bias[None, None, :, :]
            else:
                # Full causal mask as additive bias
                causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
                bias = jnp.where(causal, 0.0, -1e4)[None, None, :, :]

            # Pass dropout_rate=0.0 to nn.dot_product_attention to avoid its
            # internal `if not deterministic:` Python branch, which raises
            # TracerBoolConversionError when deterministic is a traced JAX value.
            # Attention-weight dropout is omitted; the output nn.Dropout below
            # covers regularisation on the projected output instead.
            y = nn.dot_product_attention(
                q,
                k,
                v,
                bias=bias,
                dropout_rate=0.0,
                deterministic=True,
            )

        y = y.reshape(B, T, self.hidden_dim)
        y = nn.Dense(self.hidden_dim, use_bias=False)(y)
        # nn.Dropout handles deterministic internally — no Python if-branch needed.
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
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        return x


class TinyTransformerLayer(nn.Module):
    hidden_dim: int
    num_heads: int
    ff_dim: int
    dropout_rate: float = 0.0
    use_flash_attention: bool = False
    window_size: int = 0

    @nn.compact
    def __call__(self, x, deterministic=True):
        norm1 = nn.LayerNorm()(x)
        attn_out = FlashCausalSelfAttention(
            self.hidden_dim,
            self.num_heads,
            self.dropout_rate,
            self.use_flash_attention,
            self.window_size,
        )(norm1, deterministic=deterministic)
        x = x + attn_out

        norm2 = nn.LayerNorm()(x)
        ffn_out = TinyFFN(self.hidden_dim, self.ff_dim, self.dropout_rate)(
            norm2, deterministic=deterministic
        )
        x = x + ffn_out
        return x
