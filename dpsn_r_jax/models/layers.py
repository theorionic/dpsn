import jax.numpy as jnp
import flax.linen as nn


class FlashCausalSelfAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        head_dim = self.hidden_dim // self.num_heads

        # QKV projection
        qkv = nn.Dense(3 * self.hidden_dim, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # Reshape for multi-head attention: (batch, seq, heads, dim)
        q = q.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        k = k.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)
        v = v.reshape(x.shape[0], x.shape[1], self.num_heads, head_dim)

        # Dot product attention: (batch, heads, seq_q, seq_k)
        attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(head_dim)

        if mask is not None:
            # mask shape (batch, 1, seq, seq)
            attn_weights = attn_weights + mask

        attn_weights = nn.softmax(attn_weights, axis=-1)

        if not deterministic:
            attn_weights = nn.Dropout(self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )

        # (batch, seq, heads, dim)
        y = jnp.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        y = y.reshape(x.shape[0], x.shape[1], self.hidden_dim)

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

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        norm1 = nn.LayerNorm()(x)
        attn_out = FlashCausalSelfAttention(
            self.hidden_dim, self.num_heads, self.dropout_rate
        )(norm1, mask=mask, deterministic=deterministic)
        x = x + attn_out

        norm2 = nn.LayerNorm()(x)
        ffn_out = TinyFFN(self.hidden_dim, self.ff_dim, self.dropout_rate)(
            norm2, deterministic=deterministic
        )
        x = x + ffn_out
        return x
