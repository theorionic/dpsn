from dpsn_r_jax.config import DPSNRConfig


def calculate_flops(config: DPSNRConfig, batch_size: int) -> float:
    """
    Estimates FLOPs per training step for the DPSNR architecture.
    Formula: 6 * N * D * L for Transformers + 2 * N * D * PoolSize for Memory (Approximate).
    """
    n = config.max_seq_len
    d = config.controller_hidden_dim
    l = config.controller_num_layers
    p = config.pool_total_vectors
    r = config.max_reasoning_loops

    # Transformer part (Controller)
    transformer_flops = 6 * n * d * l

    # Memory Pool part (Retrieval occurs in each reasoning loop)
    # Formula: 2 * N * D * PoolSize
    memory_flops = r * (2 * n * d * p)

    total_flops = (transformer_flops + memory_flops) * batch_size
    return float(total_flops)
