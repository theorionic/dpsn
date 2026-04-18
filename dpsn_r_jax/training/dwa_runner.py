"""DWA training runner — mirrors DPSN-R training infrastructure.

Uses:
- ChunkedHFDataset (streaming, background prefetch) — same as DPSN-R
- DWATrainState (pytree, jax.jit step) — same pattern as DPSN-R TrainState
- Same logging, checkpointing, validation structure as DPSN-R

Model architecture (dwt/dwa_model.py) is completely untouched.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------

def _ensure_dwt_importable() -> None:
    """Insert dwt/ into sys.path so dwa_model is importable from dpsn_r_jax/."""
    dwt_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dwt")
    )
    if dwt_dir not in sys.path:
        sys.path.insert(0, dwt_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_tokenizer_name(args) -> str:
    """Tokenizer precedence: args.tokenizer > args.hf_tokenizer > default."""
    for key in ("tokenizer", "hf_tokenizer"):
        val = getattr(args, key, None)
        if val:
            return val
    return "EleutherAI/gpt-neo-125m"


def _resolve_text_field(args) -> str:
    if hasattr(args, "hf_text_column"):
        val = args.hf_text_column
        if isinstance(val, list):
            return val[0] if val else "text"
        if val:
            return val
    return "text"


def _evaluate_ppl(eval_step, nnx_state, val_dataset, cfg, data_sharding, n_batches: int) -> float:
    """Compute validation perplexity over n_batches streaming batches."""
    if val_dataset is None:
        return float("nan")

    total_ce = 0.0
    counted = 0
    for _ in range(n_batches):
        try:
            batch = val_dataset.get_batch(cfg.batch_size)
        except StopIteration:
            break
        except Exception:
            break
        batch = np.asarray(batch, dtype=np.int32)
        batch = jax.device_put(batch, data_sharding)
        ce = eval_step(nnx_state, batch)
        total_ce += float(ce)
        counted += 1

    if counted == 0:
        return float("nan")
    return float(np.exp(total_ce / counted))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dwa_training(args, writer) -> None:
    """Train a DWA model using DPSN-R's streaming data pipeline.

    Args:
        args:   argparse.Namespace from main.py (YAML values already merged).
        writer: TensorBoard SummaryWriter (shared with main.py caller).
    """
    _ensure_dwt_importable()

    # ── dwt imports (flax.nnx, untouched architecture) ─────────────────────
    import flax.nnx as nnx
    from dwa_model import (
        LMConfig,
        DWALanguageModel,
        count_params,
        small_config,
        medium_config,
        large_config,
    )

    # ── DPSN-R streaming data pipeline ─────────────────────────────────────
    from dpsn_r_jax.data.dataset import ChunkedHFDataset
    from dpsn_r_jax.data.tokenizer import get_tokenizer

    # ── DWA training utilities (new, mirrors DPSN-R trainer.py) ────────────
    from dpsn_r_jax.training.dwa_trainer import (
        DWATrainState,
        make_tx,
        make_lr_schedule,
        make_dwa_train_step,
        make_dwa_eval_step,
        save_dwa_checkpoint,
        load_dwa_checkpoint,
        latest_dwa_checkpoint,
        _get_lambda_sharp,
        _get_aux_scale,
        _phase_name,
        _phase_idx,
    )

    # ── Mesh + sharding (data-parallel over all devices) ───────────────────
    devices = jax.devices()
    mesh = js.Mesh(np.array(devices), ("data",))
    replicated = js.NamedSharding(mesh, js.PartitionSpec())
    data_sharding = js.NamedSharding(mesh, js.PartitionSpec("data", None))
    n_devices = len(devices)

    # ── Config ─────────────────────────────────────────────────────────────
    dwa_config_name = getattr(args, "dwa_config", None) or "small"
    _preset = {"small": small_config, "medium": medium_config, "large": large_config}
    cfg: LMConfig = _preset.get(dwa_config_name, small_config)()

    if getattr(args, "batch_size", None):
        cfg.batch_size = int(args.batch_size)
    if getattr(args, "max_steps", None):
        cfg.max_steps = int(args.max_steps)

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer_name = _resolve_tokenizer_name(args)
    tok = get_tokenizer(tokenizer_name)
    cfg.vocab_size = tok.vocab_size

    # ── Model ──────────────────────────────────────────────────────────────
    seed = int(getattr(args, "seed", 0) or 0)
    model = DWALanguageModel(cfg, nnx.Rngs(params=jax.random.key(seed)))

    use_bf16 = bool(getattr(args, "bf16", False))
    if use_bf16:
        from dwa_model import to_bf16
        model = to_bf16(model)

    n_params = count_params(model)

    # ── Optimizer + state ──────────────────────────────────────────────────
    tx = make_tx(cfg)
    state = DWATrainState.create(model, tx, cfg.N)

    # Replicate full state (params + opt_state + alpha_ema) across devices.
    state = jax.device_put(state, replicated)

    graphdef = state.graphdef
    train_step = make_dwa_train_step(cfg, graphdef)
    eval_step = make_dwa_eval_step(graphdef)
    lr_schedule = make_lr_schedule(cfg)

    # ── Startup banner ─────────────────────────────────────────────────────
    print(f"\n{'=' * 64}")
    print("  DWA (Dynamic Weight Assembly) Training")
    print(f"{'=' * 64}")
    print(f"  JAX devices : {devices}")
    print(f"  Device count: {n_devices}")
    print(f"  Config      : {dwa_config_name}  d={cfg.d_model} N={cfg.N} "
          f"seq_len={cfg.seq_len} vocab={cfg.vocab_size}")
    print(f"  Parameters  : {n_params:,}")
    print(f"  Optimizer   : AdamW  lr={cfg.lr:.1e}  warmup={cfg.warmup_steps}  "
          f"wd={cfg.weight_decay}")

    # ── Data pipeline ──────────────────────────────────────────────────────
    dataset_name = getattr(args, "hf_dataset", None) or "roneneldan/TinyStories"
    text_field = _resolve_text_field(args)

    chunk_size = int(getattr(args, "chunk_size", None) or 10_000)
    num_workers = int(getattr(args, "num_workers", None) or 4)

    # seq_len+1 so each batch yields x=[B,T] and y=[B,T] after the shift.
    seq_len_fetch = cfg.seq_len + 1

    log_interval = int(getattr(args, "log_interval", 100) or 100)
    save_interval = int(getattr(args, "save_interval", 2000) or 2000)
    val_interval = int(getattr(args, "val_interval", None) or save_interval)
    val_steps = int(getattr(args, "val_steps", None) or 50)

    print(f"  Steps       : {cfg.max_steps}  batch={cfg.batch_size}  "
          f"log={log_interval}  save={save_interval}  val={val_interval}")
    print(f"{'=' * 64}\n")

    print(f"  Dataset     : {dataset_name}  field='{text_field}'")
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

    # Validation dataset — not every HF dataset exposes a 'validation' split.
    val_dataset: Any = None
    try:
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
    except Exception as e:
        print(f"  [warn] Could not open validation split: {e}")
        val_dataset = None

    # ── Checkpoint setup ───────────────────────────────────────────────────
    ckpt_dir: str | None = None
    if getattr(args, "checkpoint_dir", None):
        ckpt_dir = os.path.join(os.path.abspath(args.checkpoint_dir), "dwa")
        os.makedirs(ckpt_dir, exist_ok=True)

    # ── Resume ─────────────────────────────────────────────────────────────
    start = 0
    if getattr(args, "resume", False) and ckpt_dir:
        ckpt_path = latest_dwa_checkpoint(ckpt_dir)
        if ckpt_path:
            state, loaded_step = load_dwa_checkpoint(ckpt_path, state)
            state = jax.device_put(state, replicated)
            start = loaded_step + 1
            print(f"  Resumed from step {start}  (ckpt: {ckpt_path})")
        else:
            print(f"  --resume set but no checkpoint in {ckpt_dir} — starting fresh.")

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n  Training {cfg.max_steps - start} steps  "
          f"(log={log_interval}, save={save_interval}, val={val_interval})\n")

    t0 = time.perf_counter()

    for s in range(start, cfg.max_steps):
        # ── Batch ──────────────────────────────────────────────────────────
        batch = np.asarray(train_dataset.get_batch(cfg.batch_size), dtype=np.int32)
        batch = jax.device_put(batch, data_sharding)

        # ── Phase scalars ──────────────────────────────────────────────────
        lambda_sharp = jnp.array(_get_lambda_sharp(s, cfg), dtype=jnp.float32)
        aux_scale = jnp.array(_get_aux_scale(s, cfg), dtype=jnp.float32)

        # ── Train step ─────────────────────────────────────────────────────
        state, loss, bd = train_step(state, batch, lambda_sharp, aux_scale)

        # ── Logging ────────────────────────────────────────────────────────
        if s % log_interval == 0 or s == cfg.max_steps - 1:
            phase = _phase_name(s, cfg)
            elapsed = time.perf_counter() - t0
            current_lr = float(lr_schedule(s))
            print(
                f"  step {s:6d} [{phase:8s}]  loss={float(loss):.3f}  "
                f"ce={float(bd['ce']):.3f}  util={float(bd['util']):.3f}  "
                f"div={float(bd['div']):.4f}  lr={current_lr:.2e}  "
                f"({elapsed:.0f}s)"
            )
            writer.add_scalar("dwa/train_loss",   float(loss),            s)
            writer.add_scalar("dwa/train_ce",     float(bd["ce"]),        s)
            writer.add_scalar("dwa/aux_util",     float(bd["util"]),      s)
            writer.add_scalar("dwa/aux_div",      float(bd["div"]),       s)
            writer.add_scalar("dwa/aux_norm",     float(bd["norm"]),      s)
            writer.add_scalar("dwa/aux_sparse",   float(bd["sparse"]),    s)
            writer.add_scalar("dwa/lambda_sharp", float(lambda_sharp),    s)
            writer.add_scalar("dwa/lr",           current_lr,             s)
            writer.add_scalar("dwa/phase_idx",    _phase_idx(s, cfg),     s)

        # ── Validation ─────────────────────────────────────────────────────
        if s > 0 and s % val_interval == 0:
            ppl = _evaluate_ppl(eval_step, state.nnx_state, val_dataset, cfg,
                                data_sharding, val_steps)
            print(f"  step {s:6d}  val_ppl={ppl:.2f}")
            writer.add_scalar("dwa/val_ppl", ppl, s)

        # ── Checkpoint ─────────────────────────────────────────────────────
        if ckpt_dir and s > 0 and (s % save_interval == 0 or s == cfg.max_steps - 1):
            step_ckpt = os.path.join(ckpt_dir, f"step_{s:06d}")
            save_dwa_checkpoint(
                step_ckpt, state,
                metadata={
                    "phase": _phase_name(s, cfg),
                    "config": dwa_config_name,
                },
            )
            print(f"  Checkpoint → {step_ckpt}")

    # ── Final eval ─────────────────────────────────────────────────────────
    final_ppl = _evaluate_ppl(eval_step, state.nnx_state, val_dataset, cfg,
                              data_sharding, val_steps)
    elapsed = time.perf_counter() - t0
    print(f"\n  Final val_ppl = {final_ppl:.2f}  total_time = {elapsed:.0f}s")
    writer.add_scalar("dwa/final_val_ppl", final_ppl, cfg.max_steps)
    print(f"{'=' * 64}\n")
