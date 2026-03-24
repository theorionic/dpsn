import math
import functools

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.sharding import PartitionSpec

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

try:
    from jax.experimental.shard_map import shard_map as _shard_map
    _SHARD_MAP_AVAILABLE = True
except Exception:
    _shard_map = None
    _SHARD_MAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Mesh registration
#
# Call set_mesh(mesh) once from main.py after the device mesh is created.
# FlashCausalSelfAttention reads _MESH at call time so that splash attention
# can be wrapped in shard_map on multi-device runs.
# ---------------------------------------------------------------------------
_MESH = None


def set_mesh(mesh):
    """Register the device mesh so splash attention can use shard_map."""
    global _MESH
    _MESH = mesh


def _use_pallas(flag: bool, seq_len: int) -> bool:
    """Return True only when all conditions for Pallas splash_attention are met.

    On multi-device runs we wrap the kernel in shard_map (see set_mesh / _MESH).
    shard_map requires the mesh to be registered via set_mesh() before the first
    compiled call, otherwise we fall back to JAX-native attention.
    """
    if not (flag and _SPLASH_FA_AVAILABLE and seq_len >= 128 and seq_len % 128 == 0):
        return False
    if jax.device_count() > 1 and (not _SHARD_MAP_AVAILABLE or _MESH is None):
        # Multi-device requires shard_map + a registered mesh.
        return False
    return True


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


def _make_packed_causal_bias(seq_len: int, seq_pack_ids: jnp.ndarray):
    """Additive attention bias for packed sequences (block-diagonal causal).

    Position j can be attended to by position i if:
      - same sub-sequence (seq_pack_ids[i] == seq_pack_ids[j])
      - causal order (j <= i)
      - not padding (seq_pack_ids[j] >= 0)

    Args:
        seq_len:      T — total sequence length.
        seq_pack_ids: (T,) int32 — sub-sequence id per position; -1 = padding.

    Returns:
        (T, T) float32 additive bias; 0.0 where attention is allowed, -1e4 elsewhere.
    """
    i = jnp.arange(seq_len)[:, None]   # (T, 1)
    j = jnp.arange(seq_len)[None, :]   # (1, T)
    same_seq = seq_pack_ids[i] == seq_pack_ids[j]          # (T, T) bool
    causal   = j <= i                                       # (T, T) bool
    not_pad  = seq_pack_ids[j] >= 0                        # (T, T) bool
    valid    = same_seq & causal & not_pad
    return jnp.where(valid, jnp.float32(0.0), jnp.float32(-1e4))


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
    # GQA: number of KV heads. 0 = full MHA (num_kv_heads == num_heads).
    # Must evenly divide num_heads. Typical: num_heads // 4.
    # KV projections shrink by (num_heads / num_kv_heads)x; Q projection unchanged.
    num_kv_heads: int = 0

    @nn.compact
    def __call__(self, x, deterministic=True, seq_pack_ids=None):
        B, T, _ = x.shape
        head_dim = self.hidden_dim // self.num_heads
        num_kv = self.num_kv_heads if self.num_kv_heads > 0 else self.num_heads

        # Q projection: full num_heads
        q = nn.Dense(self.num_heads * head_dim, use_bias=False)(x)
        # KV projection: num_kv heads only (smaller when GQA is active)
        kv = nn.Dense(2 * num_kv * head_dim, use_bias=False)(x)
        k, v = jnp.split(kv, 2, axis=-1)

        # Reshape: (B, T, H*D) → (B, T, H, D)
        q = q.reshape(B, T, self.num_heads, head_dim)
        k = k.reshape(B, T, num_kv, head_dim)
        v = v.reshape(B, T, num_kv, head_dim)

        # ── Apply RoPE to Q (all heads) and K (kv heads) ─────────────────────
        # Computed at trace time: T and head_dim are static shapes in JIT.
        cos, sin = _precompute_rope(T, head_dim)
        q, k = _apply_rope(q, k, cos, sin)

        # ── GQA head expansion ────────────────────────────────────────────────
        # Repeat each KV head (num_heads // num_kv) times so Q, K, V all share
        # shape (B, T, num_heads, head_dim) for the attention kernel.
        if num_kv != self.num_heads:
            groups = self.num_heads // num_kv
            k = jnp.repeat(k, groups, axis=2)   # (B, T, num_heads, head_dim)
            v = jnp.repeat(v, groups, axis=2)

        # Decide which attention path to take.
        # shard_map requires B to be divisible by device_count — if it isn't
        # (e.g. dummy batch=1 during jax.eval_shape) we fall back to standard
        # dot-product attention so the shape-inference pass doesn't crash.
        # Packed sequences always use the fallback path — splash attention does
        # not support dynamic per-position masks.
        _ndev = jax.device_count()
        _use_splash = (
            seq_pack_ids is None
            and _use_pallas(self.use_flash_attention, T)
            and (_ndev == 1 or (_MESH is not None and B % _ndev == 0))
        )

        if _use_splash:
            # ----------------------------------------------------------------
            # Pallas TPU Splash Attention
            #
            # Shape contract: splash expects (H, T, D) per sample.
            # Single-device: jax.vmap over B.
            # Multi-device:  shard_map splits B across devices, then jax.vmap
            #                over the local slice — GSPMD never sees the Mosaic
            #                kernel and cannot try to auto-partition it.
            #
            # Mask:
            #   window_size > 0 → LocalMask (causal sliding window)
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

            if _ndev > 1:
                # Attention shards along the DATA-PARALLEL axis (batch dim).
                # On a 1-D mesh axis_names = ("shard",) → _axis = "shard".
                # On a 2-D mesh axis_names = ("tp", "dp") → _axis = "dp".
                # We always want the last axis name as the DP axis because
                # main.py creates 2-D meshes as Mesh(devices, ("tp", "dp")).
                _axis = _MESH.axis_names[-1]
                _splash_fn = functools.partial(
                    _shard_map,
                    mesh=_MESH,
                    in_specs=(
                        PartitionSpec(_axis, None, None, None),
                        PartitionSpec(_axis, None, None, None),
                        PartitionSpec(_axis, None, None, None),
                    ),
                    out_specs=PartitionSpec(_axis, None, None, None),
                    check_rep=False,
                )(lambda q_, k_, v_: jax.vmap(_splash_mha)(q_, k_, v_))
                y = _splash_fn(q, k, v)
            else:
                y = jax.vmap(_splash_mha)(q, k, v)

            y = jnp.transpose(y, (0, 2, 1, 3))    # (B, T, H, D)

        else:
            # ----------------------------------------------------------------
            # Standard Flax dot-product attention fallback.
            # Used on CPU/GPU, when seq_len constraints aren't met, or when
            # batch size isn't divisible by device count (e.g. eval_shape),
            # or when seq_pack_ids is provided (packed sequences need a
            # block-diagonal causal mask that splash attention cannot express).
            # ----------------------------------------------------------------
            if seq_pack_ids is not None:
                bias = _make_packed_causal_bias(T, seq_pack_ids)
                bias = bias[None, None, :, :]  # (1, 1, T, T) broadcast over B and H
            elif self.window_size > 0:
                bias = _make_sliding_window_causal_bias(T, self.window_size)
                bias = bias[None, None, :, :]
            else:
                causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
                bias = jnp.where(causal, 0.0, -1e4)[None, None, :, :]

            # Pass dropout_rate=0.0 to avoid Flax's internal `if not deterministic:`
            # Python branch (TracerBoolConversionError with gradient checkpointing).
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
    num_kv_heads: int = 0  # 0 = full MHA; >0 = GQA with this many KV heads

    @nn.compact
    def __call__(self, x, deterministic=True, seq_pack_ids=None):
        norm1 = nn.LayerNorm()(x)
        attn_out = FlashCausalSelfAttention(
            self.hidden_dim,
            self.num_heads,
            self.dropout_rate,
            self.use_flash_attention,
            self.window_size,
            num_kv_heads=self.num_kv_heads,
        )(norm1, deterministic=deterministic, seq_pack_ids=seq_pack_ids)
        x = x + attn_out

        norm2 = nn.LayerNorm()(x)
        ffn_out = TinyFFN(self.hidden_dim, self.ff_dim, self.dropout_rate)(
            norm2, deterministic=deterministic
        )
        x = x + ffn_out
        return x
