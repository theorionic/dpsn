import argparse
import jax
import jax.numpy as jnp
import optax
from dpsn_r_jax.models.dpsnr import DPSNR
from dpsn_r_jax.data.dataset import SyntheticReasoningDataset, HFStreamingDataset
from dpsn_r_jax.data.tokenizer import get_tokenizer
from dpsn_r_jax.training.trainer import create_train_state
from dpsn_r_jax.config import DPSNRConfig, get_tiny_config
from dpsn_r_jax.utils.generation import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--dataset_size", type=int, default=500)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--hf_dataset", type=str, default=None, help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--hf_subset",
        type=str,
        default=None,
        help="HuggingFace dataset configuration/subset",
    )
    parser.add_argument(
        "--hf_tokenizer", type=str, default=None, help="HuggingFace tokenizer name"
    )
    args = parser.parse_args()

    print(f"Training on {jax.devices()[0].platform.upper()}")

    # Config
    if args.tiny:
        print("Using TINY config for testing...")
        config = get_tiny_config()
    else:
        config = DPSNRConfig()

    config.hf_dataset_name = args.hf_dataset
    config.hf_tokenizer_name = args.hf_tokenizer

    tokenizer = get_tokenizer(args.hf_tokenizer)

    if hasattr(tokenizer, "vocab_size"):
        config.vocab_size = tokenizer.vocab_size
        config.pad_token_id = tokenizer.pad_token_id or 0
        print(
            f"Updated vocab size to {config.vocab_size}, pad_token_id to {config.pad_token_id}"
        )

    if args.hf_dataset:
        print(
            f"Loading HF streaming dataset: {args.hf_dataset} (subset: {args.hf_subset})"
        )
        dataset = HFStreamingDataset(
            args.hf_dataset,
            tokenizer,
            subset=args.hf_subset,
            seq_len=config.max_seq_len,
            batch_size=args.batch_size,
        )
    else:
        print("Generating synthetic sorting dataset...")
        dataset = SyntheticReasoningDataset(
            size=args.dataset_size, seq_len=config.max_seq_len
        )

    # Initialize Model
    model = DPSNR(config)

    # Initialize State
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, config)

    print(
        f"Model Parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}"
    )

    from dpsn_r_jax.training.trainer import train_step

    steps_per_epoch = args.dataset_size // args.batch_size

    for epoch in range(args.epochs):
        epoch_loss = 0

        for step in range(steps_per_epoch):
            batch = dataset.get_batch(args.batch_size)

            state, loss = train_step(state, batch, config.pad_token_id)
            epoch_loss += loss

            if step % 10 == 0:
                print(f"Epoch {epoch + 1} | Step {step} | Loss: {loss:.4f}")

        print(
            f"Epoch {epoch + 1} Complete | Avg Loss: {epoch_loss / steps_per_epoch:.4f}"
        )

    # Generation Test
    print("\nVerifying model generation...")
    from dpsn_r_jax.utils.generation import generate

    if args.hf_dataset:
        prompt = "The quick brown fox"
        print(f"Input: {prompt}")
        output = generate(state, prompt, tokenizer)
        print(f"Output: {output}")
    else:
        test_samples = ["Sort: 5 2 8 1 ->", "Sort: 10 3 7 ->", "Sort: 1 1 1 ->"]

        for prompt in test_samples:
            print(f"Input: {prompt}")
            output = generate(state, prompt, tokenizer)
            print(f"Output: {output}")
            print("-" * 20)


if __name__ == "__main__":
    main()
