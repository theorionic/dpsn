from dpsn_r_jax.config import DPSNRConfig


def calculate_flops(config: DPSNRConfig, batch_size: int) -> float:
    """
    Estimates FLOPs per training step for the DPSNR architecture.
    Accounts for Encoder, Decoder (LM Head), and the Reasoning Loop.
    Total FLOPs = 3 * (Forward Pass FLOPs) to include Backward pass.
    """
    n = config.max_seq_len
    d = config.controller_hidden_dim
    l = config.controller_num_layers
    v = config.vocab_size
    p = config.pool_total_vectors
    r = config.max_reasoning_loops
    k = config.max_k
    ff_mult = config.controller_ff_multiplier

    # 1. TinyController Encoder
    # - Attention projections: 8 * n * d^2
    # - Attention quadratic: 4 * n^2 * d
    # - FFN: 4 * ff_mult * n * d^2
    encoder_fwd = l * ((8 + 4 * ff_mult) * n * (d**2) + 4 * (n**2) * d)

    # 2. LM Head (Decoder)
    # 2 * n * d * v
    decoder_fwd = 2 * n * d * v

    # 3. Reasoning Loop (Retrieval + Integration)
    # - Pool Retrieval (pooled query): 2 * d * p
    # - Pool Aggregation: 2 * k * d
    # - Integration (2 Dense layers): 6 * n * d^2
    loop_fwd = r * (2 * d * p + 2 * k * d + 6 * n * (d**2))

    # Total FLOPs (Fwd + Bwd approx 3 * Fwd)
    total_fwd = encoder_fwd + decoder_fwd + loop_fwd
    total_flops = 3 * total_fwd * batch_size

    return float(total_flops)
