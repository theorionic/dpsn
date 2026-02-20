#!/usr/bin/env python3
"""Fine-tuning CLI for DPSNR model."""

import argparse
import os
import sys
import time
import json
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:

        class SummaryWriter:
            def __init__(self, log_dir=None):
                pass

            def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
                pass

            def close(self):
                pass


from dpsn_r_jax.config import DPSNRConfig, FineTuningConfig, get_model_config
from dpsn_r_jax.data.finetune_dataset import (
    FineTuningDataset,
    StreamingFineTuningDataset,
)
from dpsn_r_jax.data.tokenizer import get_tokenizer
from dpsn_r_jax.training.finetune_trainer import (
    create_finetune_state,
    finetune_step,
    validation_step,
    save_checkpoint,
    load_checkpoint,
)
from dpsn_r_jax.training.lr_schedules import get_scheduler
from dpsn_r_jax.utils.generation import generate


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DPSNR Model")

    # Data arguments
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training data file (JSON or JSONL)",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="Path to validation data file (JSON or JSONL)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="alpaca",
        choices=[
            "alpaca",
            "alpaca_no_input",
            "chatml",
            "vicuna",
            "llama",
            "mistral",
            "sharegpt",
        ],
        help="Prompt template to use",
    )
    parser.add_argument(
        "--template_path",
        type=str,
        default=None,
        help="Path to custom template file (YAML or JSON)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Model arguments
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "base", "large", "xl"],
        help="Model configuration size",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default=None,
        help="Path to custom config YAML file",
    )
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps (overrides warmup_ratio)",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
        ],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides epochs)",
    )

    # Freezing arguments
    parser.add_argument(
        "--freeze_controller",
        action="store_true",
        help="Freeze controller parameters",
    )
    parser.add_argument(
        "--freeze_pool",
        action="store_true",
        default=True,
        help="Freeze pool parameters (default: True)",
    )
    parser.add_argument(
        "--no_freeze_pool",
        action="store_true",
        help="Do not freeze pool parameters",
    )

    # Evaluation arguments
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )

    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/finetuned",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="TensorBoard log directory (defaults to output_dir/logs)",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "none"],
        help="Reporting backend",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny config for testing (overrides --config)",
    )

    return parser.parse_args()


def compute_perplexity(loss: float) -> float:
    import math

    return math.exp(loss) if loss < 10 else float("inf")


def main():
    args = parse_args()

    # Handle flags
    if args.tiny:
        args.config = "tiny"

    if args.no_freeze_pool:
        args.freeze_pool = False

    # Print configuration
    print("=" * 60)
    print("DPSNR Fine-tuning Configuration")
    print("=" * 60)
    print(f"Training file: {args.train_file}")
    print(f"Validation file: {args.validation_file}")
    print(f"Template: {args.template}")
    print(f"Model config: {args.config}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Freeze controller: {args.freeze_controller}")
    print(f"Freeze pool: {args.freeze_pool}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save training args
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize RNG
    rng = random.PRNGKey(args.seed)
    rng, init_rng = random.split(rng)

    # Load model config
    if args.config_yaml:
        config = DPSNRConfig.from_yaml(args.config_yaml)
    else:
        config = get_model_config(args.config)

    config.max_seq_len = args.max_seq_length
    config.pad_token_id = 0

    # Get tokenizer
    tokenizer = get_tokenizer(
        name_or_path=config.hf_tokenizer_name,
        max_val=min(100, config.vocab_size - 4)
        if not config.hf_tokenizer_name
        else 100,
    )

    # Determine pad token ID
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = config.pad_token_id

    # Load dataset
    print("\nLoading training dataset...")
    train_dataset = FineTuningDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        template=args.template,
        template_path=args.template_path,
        max_seq_length=args.max_seq_length,
        pad_token_id=pad_token_id,
    )

    eval_dataset = None
    if args.validation_file:
        print("Loading validation dataset...")
        eval_dataset = FineTuningDataset(
            data_path=args.validation_file,
            tokenizer=tokenizer,
            template=args.template,
            template_path=args.template_path,
            max_seq_length=args.max_seq_length,
            pad_token_id=pad_token_id,
        )

    # Calculate training steps
    steps_per_epoch = len(train_dataset) // args.batch_size
    if args.max_steps:
        total_steps = args.max_steps
        num_epochs = (args.max_steps // steps_per_epoch) + 1
    else:
        num_epochs = args.epochs
        total_steps = steps_per_epoch * args.epochs

    # Calculate warmup steps
    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"\nTraining steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # Create learning rate schedule
    lr_schedule = get_scheduler(
        scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Create training state
    print("\nInitializing model...")
    state = create_finetune_state(
        rng=init_rng,
        config=config,
        learning_rate_fn=lr_schedule,
        freeze_controller=args.freeze_controller,
        freeze_pool=args.freeze_pool,
        pretrained_path=args.load_pretrained,
    )

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint_steps = [
            int(d.split("_")[-1])
            for d in os.listdir(args.resume_from_checkpoint)
            if d.startswith("checkpoint_")
        ]
        if checkpoint_steps:
            latest_step = max(checkpoint_steps)
            state = load_checkpoint(state, args.resume_from_checkpoint, latest_step)
            print(f"Resumed from step {state.step}")

    # Setup logging
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    writer = SummaryWriter(log_dir=log_dir) if args.report_to == "tensorboard" else None

    # Training loop
    print("\nStarting training...")
    global_step = state.step
    accumulated_loss = 0.0
    checkpoint_history = []

    for epoch in range(num_epochs):
        if args.max_steps and global_step >= args.max_steps:
            break

        epoch_loss = 0.0
        epoch_steps = 0

        for step in range(steps_per_epoch):
            if args.max_steps and global_step >= args.max_steps:
                break

            # Get batch
            batch = train_dataset.get_batch(args.batch_size)
            batch_jax = {
                "input_ids": jnp.array(batch["input_ids"]),
                "labels": jnp.array(batch["labels"]),
                "attention_mask": jnp.array(batch["attention_mask"]),
            }

            # Training step
            state, loss = finetune_step(state, batch_jax, pad_token_id)

            # Gradient accumulation
            accumulated_loss += float(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                avg_loss = accumulated_loss / args.gradient_accumulation_steps
                accumulated_loss = 0.0
                global_step += 1

                epoch_loss += avg_loss
                epoch_steps += 1

                # Logging
                if global_step % args.logging_steps == 0:
                    current_lr = lr_schedule(global_step)
                    perplexity = compute_perplexity(avg_loss)

                    print(
                        f"Epoch {epoch + 1}/{num_epochs} | Step {global_step}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | LR: {current_lr:.2e}"
                    )

                    if writer:
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/perplexity", perplexity, global_step)
                        writer.add_scalar(
                            "train/learning_rate", current_lr, global_step
                        )

                # Evaluation
                if (
                    eval_dataset
                    and args.evaluation_strategy == "steps"
                    and global_step % args.eval_steps == 0
                ):
                    eval_loss = 0.0
                    eval_steps_count = min(50, len(eval_dataset) // args.batch_size)

                    for _ in range(eval_steps_count):
                        eval_batch = eval_dataset.get_batch(args.batch_size)
                        eval_batch_jax = {
                            "input_ids": jnp.array(eval_batch["input_ids"]),
                            "labels": jnp.array(eval_batch["labels"]),
                            "attention_mask": jnp.array(eval_batch["attention_mask"]),
                        }
                        eval_loss += float(
                            validation_step(state, eval_batch_jax, pad_token_id)
                        )

                    avg_eval_loss = eval_loss / eval_steps_count
                    eval_perplexity = compute_perplexity(avg_eval_loss)

                    print(
                        f"  Evaluation | Loss: {avg_eval_loss:.4f} | PPL: {eval_perplexity:.2f}"
                    )

                    if writer:
                        writer.add_scalar("eval/loss", avg_eval_loss, global_step)
                        writer.add_scalar(
                            "eval/perplexity", eval_perplexity, global_step
                        )

                # Checkpoint saving
                if global_step % args.save_steps == 0:
                    checkpoint_dir = os.path.join(
                        args.output_dir, f"checkpoint_{global_step}"
                    )
                    save_checkpoint(state, checkpoint_dir, global_step)
                    checkpoint_history.append(global_step)

                    # Remove old checkpoints
                    while len(checkpoint_history) > args.save_total_limit:
                        old_step = checkpoint_history.pop(0)
                        old_dir = os.path.join(
                            args.output_dir, f"checkpoint_{old_step}"
                        )
                        if os.path.exists(old_dir):
                            import shutil

                            shutil.rmtree(old_dir)

        # Epoch-level metrics
        if epoch_steps > 0:
            avg_epoch_loss = epoch_loss / epoch_steps
            epoch_perplexity = compute_perplexity(avg_epoch_loss)
            print(
                f"Epoch {epoch + 1} completed | Avg Loss: {avg_epoch_loss:.4f} | PPL: {epoch_perplexity:.2f}"
            )

            if writer:
                writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch + 1)

    # Save final checkpoint
    final_dir = os.path.join(args.output_dir, "final")
    save_checkpoint(state, final_dir, global_step)
    print(f"\nTraining completed! Final checkpoint saved to {final_dir}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
