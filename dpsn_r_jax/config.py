from dataclasses import dataclass
from typing import Optional


@dataclass
class PoolConfig:
    total_vectors: int
    hidden_dim: int


@dataclass
class DPSNRConfig:
    vocab_size: int = 24
    controller_hidden_dim: int = 64
    controller_num_layers: int = 2
    controller_num_heads: int = 2
    controller_ff_multiplier: float = 2.0
    max_seq_len: int = 64
    dropout: float = 0.1
    pool_total_vectors: int = 1000
    pool_hidden_dim: int = 64
    librarian_hidden_dim: int = 32
    max_reasoning_loops: int = 4
    min_reasoning_loops: int = 1
    halt_threshold: float = 0.99
    min_k: int = 4
    max_k: int = 32
    num_clusters_to_search: int = 4
    hf_dataset_name: Optional[str] = None
    hf_tokenizer_name: Optional[str] = None
    streaming: bool = True
    pad_token_id: int = 0
    max_steps: Optional[int] = None
    generation_steps: Optional[int] = None
    generation_max_tokens: int = 20
    generation_prompts: Optional[list[str]] = None
    learning_rate: float = 3e-4
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str) -> "DPSNRConfig":
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_config)


def get_model_config(name: str) -> DPSNRConfig:
    """
    Returns a predefined configuration based on the name.

    Sizes:
    - tiny: ~6M params (Debug/CPU)
    - base: ~125M params (TPU v3-8 / GPU)
    - large: ~350M params (TPU Pod Slice)
    - xl: ~1B params (Large TPU Pod - Massive Pool)
    """
    if name == "tiny":
        return DPSNRConfig(
            vocab_size=24,
            controller_hidden_dim=32,
            controller_num_layers=2,
            controller_num_heads=2,
            controller_ff_multiplier=2.0,
            max_seq_len=64,
            dropout=0.0,
            pool_total_vectors=100,
            pool_hidden_dim=32,
            librarian_hidden_dim=16,
            max_reasoning_loops=2,
            min_reasoning_loops=1,
            halt_threshold=0.5,
            min_k=2,
            max_k=10,
            num_clusters_to_search=2,
            learning_rate=1e-3,
        )

    elif name == "base":
        return DPSNRConfig(
            vocab_size=50257,  # GPT-Neo / Standard
            controller_hidden_dim=512,
            controller_num_layers=6,
            controller_num_heads=8,
            max_seq_len=512,
            pool_total_vectors=65536,  # 2^16 vectors
            pool_hidden_dim=512,
            max_reasoning_loops=4,
            learning_rate=6e-4,
        )

    elif name == "large":
        return DPSNRConfig(
            vocab_size=50257,
            controller_hidden_dim=768,
            controller_num_layers=12,
            controller_num_heads=12,
            max_seq_len=1024,
            pool_total_vectors=262144,  # 2^18 vectors (~200M params in pool)
            pool_hidden_dim=768,
            max_reasoning_loops=6,
            learning_rate=3e-4,
        )

    elif name == "xl":
        return DPSNRConfig(
            vocab_size=50257,
            controller_hidden_dim=1024,
            controller_num_layers=16,
            controller_num_heads=16,
            max_seq_len=2048,
            pool_total_vectors=1048576,  # 2^20 vectors (~1.1B params in pool)
            pool_hidden_dim=1024,
            max_reasoning_loops=8,
            learning_rate=2e-4,
        )

    else:
        raise ValueError(f"Unknown config name: {name}")


def get_tiny_config():
    return get_model_config("tiny")
