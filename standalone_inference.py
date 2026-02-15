import argparse
import jax
import jax.numpy as jnp
from jax import random, lax
import flax.linen as nn
from flax import struct
import numpy as np
import orbax.checkpoint
from transformers import AutoTokenizer
from typing import Optional, Any, Tuple, List
from dataclasses import dataclass
import time
import os

# ==========================================
# Config
# ==========================================


@dataclass
class PoolConfig:
    total_vectors: int
    hidden_dim: int


@dataclass
class DPSNRConfig:
    vocab_size: int = 50257
    controller_hidden_dim: int = 512
    controller_num_layers: int = 6
    controller_num_heads: int = 8
    controller_ff_multiplier: float = 2.0
    max_seq_len: int = 512
    dropout: float = 0.1
    pool_total_vectors: int = 65536
    pool_hidden_dim: int = 512
    librarian_hidden_dim: int = 32
    max_reasoning_loops: int = 4
    min_reasoning_loops: int = 1
    halt_threshold: float = 0.99
    min_k: int = 4
    max_k: int = 32
    num_clusters_to_search: int = 4
    hf_dataset_name: Optional[str] = None
    hf_tokenizer_name: Optional[str] = "EleutherAI/gpt-neo-125M"
    streaming: bool = True
    pad_token_id: int = 0
    max_steps: Optional[int] = None
    generation_steps: Optional[int] = None
    generation_max_tokens: int = 20
    generation_prompts: Optional[list[str]] = None
    learning_rate: float = 3e-4
    num_workers: int = 4
    gradient_checkpointing: bool = False


# ==========================================
# Model Components
# ==========================================


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


class LearnedIndexer(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, query):
        x = nn.Dense(self.hidden_dim)(query)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim // 2)(x)
        x = nn.gelu(x)

        mu = nn.Dense(1)(x)
        mu = nn.sigmoid(mu)

        sigma = nn.Dense(1)(x)
        sigma = nn.softplus(sigma)

        return mu.squeeze(-1), sigma.squeeze(-1)


class CoordinateMassivePool(nn.Module):
    config: PoolConfig
    window_size: int

    def setup(self):
        self.params_storage = self.param(
            "params_storage",
            nn.initializers.normal(),
            (self.config.total_vectors, self.config.hidden_dim),
        )

    def __call__(self, mu, sigma):
        B = mu.shape[0]
        Total = self.config.total_vectors
        D = self.config.hidden_dim
        W = self.window_size

        center_idx = mu * (Total - 1)

        start_indices = jnp.clip(center_idx - W // 2, 0, Total - W).astype(jnp.int32)

        def slice_fn(start):
            return lax.dynamic_slice(self.params_storage, (start, 0), (W, D))

        selected = jax.vmap(slice_fn)(start_indices)

        relative_indices = jnp.arange(W)[None, :] + start_indices[:, None]

        distances = relative_indices - center_idx[:, None]

        weights = jnp.exp(-(distances**2) / (2 * (sigma[:, None] + 1e-6) ** 2))
        weights = weights / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-6)

        aggregated = jnp.einsum("bw,bwd->bd", weights, selected)

        return aggregated, start_indices

    def organize_memory(self):
        mean_vec = jnp.mean(self.params_storage, axis=0)
        sim = jnp.dot(self.params_storage, mean_vec)
        indices = jnp.argsort(sim)
        return self.params_storage[indices]


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


class DPSNR(nn.Module):
    config: DPSNRConfig

    def setup(self):
        # Conditionally apply gradient checkpointing (rematerialization)
        if self.config.gradient_checkpointing:
            controller_cls = nn.remat(TinyController)
            acc_cls = nn.remat(AdaptiveComputeController)
        else:
            controller_cls = TinyController
            acc_cls = AdaptiveComputeController

        self.controller = controller_cls(self.config)
        self.indexer = LearnedIndexer(self.config.controller_hidden_dim)
        self.pool = CoordinateMassivePool(
            PoolConfig(
                self.config.pool_total_vectors, self.config.controller_hidden_dim
            ),
            window_size=self.config.max_k,
        )
        self.acc = acc_cls(
            self.config.controller_hidden_dim,
            self.config.max_reasoning_loops,
            self.config.halt_threshold,
        )
        self.retrieval_integrator = nn.Sequential(
            [
                nn.Dense(self.config.controller_hidden_dim),
                nn.gelu,
                nn.Dense(self.config.controller_hidden_dim),
                nn.LayerNorm(),
            ]
        )

    def __call__(self, input_ids, deterministic=True):
        # 1. Encode
        hidden = self.controller(input_ids, deterministic=deterministic)

        # 2. Reasoning Loop
        state_hidden = hidden
        B, T, D = hidden.shape

        halt_prob = jnp.zeros((B, T, 1))
        halted_mask = jnp.zeros((B, T, 1))

        # Initialize sub-modules before scan
        _ = self.indexer(jnp.zeros((B, D)))
        _ = self.pool(jnp.zeros((B,)), jnp.zeros((B,)))
        _ = self.retrieval_integrator(
            jnp.zeros((B, T, D + self.config.controller_hidden_dim))
        )
        _ = self.acc(state_hidden, state_hidden, 0, halt_prob, halted_mask)

        def reasoning_step(carry, i):
            s_hidden, h_prob, h_mask = carry
            prev_s_hidden = s_hidden

            pooled_state = s_hidden[:, -1, :]
            mu, sigma = self.indexer(pooled_state)

            retrieved, start_indices = self.pool(mu, sigma)

            retrieved_expanded = jnp.expand_dims(retrieved, 1).repeat(T, axis=1)

            combined = jnp.concatenate([s_hidden, retrieved_expanded], axis=-1)
            integrated = self.retrieval_integrator(combined)

            # Step
            new_s_hidden, h_prob, new_h_mask = self.acc(
                s_hidden,
                s_hidden + integrated,
                i,
                h_prob,
                h_mask,
            )

            # Mask
            update_mask = 1.0 - h_mask
            s_hidden = update_mask * new_s_hidden + h_mask * prev_s_hidden

            return (s_hidden, h_prob, new_h_mask), start_indices

        if self.config.gradient_checkpointing:
            reasoning_step = jax.checkpoint(reasoning_step)

        init_carry = (state_hidden, halt_prob, halted_mask)
        (state_hidden, halt_prob, halted_mask), all_indices = jax.lax.scan(
            reasoning_step,
            init_carry,
            jnp.arange(self.config.max_reasoning_loops),
        )

        # 3. Decode
        logits = self.controller.decode(state_hidden)

        all_indices = jnp.transpose(all_indices, (1, 0))
        return logits, (self.config.max_reasoning_loops, all_indices)


# ==========================================
# Inference Logic
# ==========================================


class InferenceState:
    def __init__(self, params, apply_fn):
        self.params = params
        self.apply_fn = apply_fn


def generate(
    state,
    prompt,
    tokenizer,
    rng=None,
    max_len=20,
    temperature=1.0,
    top_k=40,
    repetition_penalty=1.2,
):
    print(f"Generating for prompt: '{prompt}'")
    # Handle tokenizer padding if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in range(3):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="np")
            break
        except Exception as e:
            if i < 2:
                time.sleep(0.1)
                continue
            print(f"Tokenizer error: {e}")
            raise e

    # Ensure input_ids is 2D (batch, seq)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]

    generated = jnp.array(input_ids)

    eos_id = tokenizer.eos_token_id
    print("Starting generation loop...")
    for step in range(max_len):
        logits, _ = state.apply_fn(
            {"params": state.params}, generated, deterministic=True
        )
        next_token_logits = logits[0, -1, :]

        # Repetition Penalty
        if repetition_penalty != 1.0:
            mask = jnp.zeros_like(next_token_logits, dtype=jnp.bool_)
            # Mask out already generated tokens for penalty
            # Only unique tokens to avoid index issues
            unique_generated = jnp.unique(generated[0])
            mask = mask.at[unique_generated].set(True)

            penalized_logits = jnp.where(
                next_token_logits > 0,
                next_token_logits / repetition_penalty,
                next_token_logits * repetition_penalty,
            )
            next_token_logits = jnp.where(mask, penalized_logits, next_token_logits)

        # Temperature Scaling
        if temperature != 1.0 and temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Top-K filtering
        if top_k > 0:
            k = min(top_k, next_token_logits.shape[-1])
            values, _ = jax.lax.top_k(next_token_logits, k=k)
            min_value = values[-1]
            next_token_logits = jnp.where(
                next_token_logits < min_value,
                jnp.full_like(next_token_logits, -1e10),
                next_token_logits,
            )

        # Sampling
        if temperature > 0 and rng is not None:
            rng, sample_rng = random.split(rng)
            next_token = random.categorical(sample_rng, next_token_logits)
        else:
            next_token = jnp.argmax(next_token_logits, axis=-1)

        next_token_id = int(next_token.item())
        if next_token_id == eos_id:
            break

        next_token = jnp.reshape(next_token, (1,))
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

        # Stream print
        word = tokenizer.decode([next_token_id])
        print(word, end="", flush=True)

    print("\n")

    generated_list = generated[0].tolist()
    return tokenizer.decode(generated_list, skip_special_tokens=True)


def load_checkpoint_params(path):
    print(f"Loading checkpoint from {path}...")

    # Try restoring as a PyTree (dict)
    # If using Orbax standard checkpointer
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    # Check for 'default' subdirectory if direct path fails
    target_path = path
    if os.path.exists(os.path.join(path, "default")):
        target_path = os.path.join(path, "default")

    try:
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_state = checkpointer.restore(target_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Trying parent directory...")
        # Fallback logic if needed
        raise e

    if "params" in raw_state:
        return raw_state["params"]
    elif "model" in raw_state and "params" in raw_state["model"]:
        return raw_state["model"]["params"]

    # If the checkpoint is just the params
    return raw_state


def infer_config(params):
    print("Inferring config from params...")
    config = DPSNRConfig()

    # Pool
    if "pool" in params and "params_storage" in params["pool"]:
        pool_shape = params["pool"]["params_storage"].shape
        config.pool_total_vectors = pool_shape[0]
        config.pool_hidden_dim = pool_shape[1]
        print(f"Inferred Pool: {pool_shape}")

    # Controller Embedding
    if "controller" in params and "embedding" in params["controller"]:
        emb_shape = params["controller"]["embedding"]["embedding"].shape
        config.vocab_size = emb_shape[0]
        config.controller_hidden_dim = emb_shape[1]
        print(f"Inferred Embedding: {emb_shape}")

    # Controller Layers
    if "controller" in params and "layers" in params["controller"]:
        # layers is usually a list-like dict: '0', '1', ...
        keys = [k for k in params["controller"]["layers"].keys() if k.isdigit()]
        if keys:
            config.controller_num_layers = len(keys)
            print(f"Inferred Layers: {config.controller_num_layers}")

    # Controller Positional Encoding
    if "controller" in params and "pos_encoding" in params["controller"]:
        pos_shape = params["controller"]["pos_encoding"]["embedding"].shape
        config.max_seq_len = pos_shape[0]
        print(f"Inferred Max Seq Len: {config.max_seq_len}")

    return config


def main():
    parser = argparse.ArgumentParser(description="Standalone Inference for DPSNR")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/Users/dev/projects/orionic/dpsn-r-jax/result/dpsn/universal_56000/0",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neo-125M",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--prompt", type=str, default="The future of AI is", help="Initial prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    # 1. Load Params
    params = load_checkpoint_params(args.checkpoint)

    # 2. Infer Config
    config = infer_config(params)

    # 3. Initialize Model
    print("Initializing model...")
    model = DPSNR(config)

    # 4. Create Inference State
    inference_state = InferenceState(params, model.apply)

    # 5. Load Tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = random.PRNGKey(args.seed)

    # 6. Generate
    if args.interactive:
        print("Entering interactive mode. Type 'quit' to exit.")
        while True:
            prompt = input("\nPrompt: ")
            if prompt.lower() in ["quit", "exit"]:
                break

            rng, step_rng = random.split(rng)
            generate(
                inference_state,
                prompt,
                tokenizer,
                rng=step_rng,
                temperature=args.temperature,
            )
    else:
        generate(
            inference_state,
            args.prompt,
            tokenizer,
            rng=rng,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
