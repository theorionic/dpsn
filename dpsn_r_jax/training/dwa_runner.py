"""DWA (Dynamic Weight Assembly) training runner.

Uses DPSN-R's ChunkedHFDataset streaming pipeline instead of load_and_chunk,
so training starts in seconds (no dump-to-disk). Model code in dwt/dwa_model.py
is untouched — only the training mechanism mirrors DPSN-R's approach.

Architecture recap:
  tokens → embed → Transformer Part A
         → DWA Middle  (multi-aspect retrieval → factorised weight assembly)
         → Transformer Part B → LM head → logits

Three-phase sharpness schedule:
  Phase 1  (0 → phase1_end)    : lambda=0, softmax over top-k, no aux losses
  Phase 2  (phase1_end → phase2_end): lambda 0→5, aux losses on
  Phase 3  (phase2_end → ∞)    : lambda 5→10, cosine LR decay
"""

from __future__ import annotations

import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def _ensure_dwt_importable() -> None:
    """Insert dwt/ into sys.path so dwa_model / dwa_train are importable."""
    dwt_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dwt")
    )
    if dwt_dir not in sys.path:
        sys.path.insert(0, dwt_dir)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dwa_training(args, writer) -> None:
    """Train a DWA model using DPSN-R's streaming data pipeline.

    Args:
        args:   argparse.Namespace from main.py.
        writer: TensorBoard SummaryWriter (shared with main.py caller).
    """
    _ensure_dwt_importable()

    # ── dwt imports (flax.nnx, untouched) ───────────────────────────────────
    import flax.nnx as nnx
    from dwa_model import (
        LMConfig,
        DWALanguageModel,
        count_params,
        make_optimizer,
        make_dwa_step,
        _eval_dwa_batch,
        to_bf16,
        get_lambda_sharp,
        get_aux_scale,
        small_config,
        medium_config,
        large_config,
    )
    # Only checkpoint / device utilities from dwa_train — no data loading
    from dwa_train import (
        shard_batch,
        replicate,
        save_checkpoint,
        load_checkpoint,
        latest_checkpoint,
        N_DEVICES,
        _replicated,
        _data_sharding,
    )

    # ── DPSN-R streaming data pipeline ──────────────────────────────────────
    from dpsn_r_jax.data.dataset import ChunkedHFDataset
    from dpsn_r_jax.data.tokenizer import get_tokenizer

    print(f"\n{'=' * 64}")
    print("  DWA (Dynamic Weight Assembly) Training")
    print(f"{'=' * 64}")
    print(f"  JAX devices : {jax.devices()}")
    print(f"  Device count: {N_DEVICES}")

    # ── Config ───────────────────────────────────────────────────────────────
    dwa_config_name = getattr(args, "dwa_config", "small")
    _preset = {"small": small_config, "medium": medium_config, "large": large_config}
    cfg: LMConfig = _preset.get(dwa_config_name, small_config)()

    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.max_steps:
        cfg.max_steps = args.max_steps

    # ── Tokenizer — same 50257-vocab as tiktoken GPT-2 ──────────────────────
    tokenizer_name = getattr(args, "tokenizer", None) or "EleutherAI/gpt-neo-125m"
    tok = get_tokenizer(tokenizer_name)
    cfg.vocab_size = tok.vocab_size
    print(f"  Config      : {dwa_config_name}  "
          f"d={cfg.d_model} N={cfg.N} seq_len={cfg.seq_len} vocab={cfg.vocab_size}")
    print(f"  Steps       : {cfg.max_steps}  batch={cfg.batch_size}")

    # ── Data — streaming chunks, background prefetch (DPSN-R style) ─────────
    dataset_name = args.hf_dataset or "roneneldan/TinyStories"
    if hasattr(args, "hf_text_column"):
        text_field = (
            args.hf_text_column[0]
            if isinstance(args.hf_text_column, list)
            else args.hf_text_column
        )
    else:
        text_field = "text"

    chunk_size  = getattr(args, "chunk_size", None) or 10_000
    num_workers = getattr(args, "num_workers", None) or 4

    # seq_len+1 so we can split batch → x=[B,T], y=[B,T] (next-token targets)
    seq_len_fetch = cfg.seq_len + 1

    print(f"\n  Dataset     : {dataset_name}  field='{text_field}'")
    print(f"  chunk_size  : {chunk_size:,}  workers={num_workers}")

    train_dataset = ChunkedHFDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        chunk_size=chunk_size,
        split="train",
        seq_len=seq_len_fetch,
        batch_size=cfg.batch_size,
        num_tokenizer_workers=num_workers,
        text_columns=[text_field],
    )

    # Validation dataset — use the dataset's own val split when available
    val_dataset = ChunkedHFDataset(
        dataset_name=dataset_name,
        tokenizer_name=tokenizer_name,
        chunk_size=min(chunk_size, 2_000),
        split="validation",
        seq_len=seq_len_fetch,
        batch_size=cfg.batch_size,
        num_tokenizer_workers=num_workers,
        text_columns=[text_field],
    )

    # ── Model + Optimizer ────────────────────────────────────────────────────
    seed  = 0
    model = DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(seed)))

    use_bf16 = getattr(args, "bf16", False)
    if use_bf16:
        model = to_bf16(model)

    model = replicate(model)
    opt   = make_optimizer(model, cfg)
    opt   = replicate(opt)

    n_params = count_params(model)
    print(f"  Parameters  : {n_params:,}")

    step_fn   = make_dwa_step(cfg)
    alpha_ema = jax.device_put(jnp.ones(cfg.N) / cfg.N, _replicated)
    start     = 0

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    ckpt_dir: str | None = None
    if args.checkpoint_dir:
        ckpt_dir = os.path.join(os.path.abspath(args.checkpoint_dir), "dwa")
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume ───────────────────────────────────────────────────────────────
    if getattr(args, "resume", False) and ckpt_dir:
        ckpt_path = latest_checkpoint(ckpt_dir)
        if ckpt_path:
            start, loaded_ema = load_checkpoint(ckpt_path, model, opt, N=cfg.N)
            model = replicate(model)
            opt   = replicate(opt)
            if loaded_ema is not None:
                alpha_ema = jax.device_put(loaded_ema, _replicated)
            start += 1
            print(f"  Resumed from step {start}  (ckpt: {ckpt_path})")
        else:
            print(f"  --resume set but no checkpoint in {ckpt_dir} — starting fresh.")

    # ── Training loop ────────────────────────────────────────────────────────
    log_interval  = getattr(args, "log_interval", 50)
    save_interval = getattr(args, "save_interval", 1000)
    val_interval  = getattr(args, "val_interval", None) or save_interval
    val_steps     = getattr(args, "val_steps", 50)

    print(f"\n  Training {cfg.max_steps - start} steps  "
          f"(log={log_interval}, save={save_interval}, val={val_interval})\n")

    t0 = time.perf_counter()

    for s in range(start, cfg.max_steps):
        # ── Batch (streaming, no blocking disk reads) ──────────────────────
        batch  = train_dataset.get_batch(cfg.batch_size)   # numpy [B, T+1]
        x_s, y_s = shard_batch(batch[:, :-1], batch[:, 1:])

        # ── Phase scalars ─────────────────────────────────────────────────
        ls  = jnp.array(get_lambda_sharp(s, cfg))
        aux = jnp.array(get_aux_scale(s, cfg))

        # ── Train step ────────────────────────────────────────────────────
        total, bd, _alpha, alpha_ema = step_fn(
            model, opt, x_s, y_s,
            alpha_ema, ls, jnp.array(1.0), aux,
        )

        # ── Logging ───────────────────────────────────────────────────────
        if s % log_interval == 0 or s == cfg.max_steps - 1:
            phase   = _phase_name(s, cfg)
            elapsed = time.perf_counter() - t0
            ce_val  = float(bd["ce"])
            print(
                f"  step {s:6d} [{phase:8s}]  "
                f"ce={ce_val:.3f}  "
                f"util={float(bd.get('util', 0)):.3f}  "
                f"div={float(bd.get('div', 0)):.4f}  "
                f"lambda={float(ls):.2f}  "
                f"({elapsed:.0f}s)"
            )
            writer.add_scalar("dwa/train_ce",     ce_val,                   s)
            writer.add_scalar("dwa/lambda_sharp", float(ls),                s)
            writer.add_scalar("dwa/aux_util",     float(bd.get("util", 0)), s)
            writer.add_scalar("dwa/aux_div",      float(bd.get("div", 0)),  s)
            writer.add_scalar("dwa/aux_norm",     float(bd.get("norm", 0)), s)
            writer.add_scalar("dwa/aux_sparse",   float(bd.get("sparse", 0)), s)
            writer.add_scalar("dwa/phase_idx",    _phase_idx(s, cfg),       s)

        # ── Validation ────────────────────────────────────────────────────
        if s > 0 and s % val_interval == 0:
            ppl = _evaluate_ppl(model, val_dataset, cfg, val_steps)
            print(f"  step {s:6d}  val_ppl={ppl:.2f}")
            writer.add_scalar("dwa/val_ppl", ppl, s)

        # ── Checkpoint ────────────────────────────────────────────────────
        if ckpt_dir and s > 0 and (s % save_interval == 0 or s == cfg.max_steps - 1):
            step_ckpt = os.path.join(ckpt_dir, f"step_{s:06d}")
            save_checkpoint(
                step_ckpt, model, opt, s, alpha_ema,
                metadata={"phase": _phase_name(s, cfg), "dwa_config": dwa_config_name},
            )
            print(f"  Checkpoint → {step_ckpt}")

    # ── Final eval ───────────────────────────────────────────────────────────
    final_ppl = _evaluate_ppl(model, val_dataset, cfg, val_steps)
    elapsed   = time.perf_counter() - t0
    print(f"\n  Final val_ppl = {final_ppl:.2f}  total_time = {elapsed:.0f}s")
    writer.add_scalar("dwa/final_val_ppl", final_ppl, cfg.max_steps)
    print(f"{'=' * 64}\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _evaluate_ppl(model, val_dataset, cfg, n_batches: int = 50) -> float:
    """Compute validation perplexity over n_batches streaming batches."""
    from dwa_model import _eval_dwa_batch
    from dwa_train import shard_batch

    total_ce = 0.0
    counted  = 0
    for _ in range(n_batches):
        try:
            batch = val_dataset.get_batch(cfg.batch_size)
        except (StopIteration, Exception):
            break
        x_s, y_s = shard_batch(batch[:, :-1], batch[:, 1:])
        ce = _eval_dwa_batch(model, x_s, y_s)
        total_ce += float(ce)
        counted  += 1

    if counted == 0:
        return float("inf")
    return float(jnp.exp(total_ce / counted))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_name(step: int, cfg) -> str:
    if step < cfg.phase1_end:
        return "warmup"
    if step < cfg.phase2_end:
        return "gate_on"
    return "sharpen"


def _phase_idx(step: int, cfg) -> int:
    if step < cfg.phase1_end:
        return 0
    if step < cfg.phase2_end:
        return 1
    return 2
