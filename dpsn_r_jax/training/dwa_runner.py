"""DWA (Dynamic Weight Assembly) training runner.

Bridges dwt/dwa_model.py + dwt/dwa_train.py into the main.py harness.
Invoked when --model_type dwa is passed. Uses the same --checkpoint_dir,
--resume, --max_steps, --batch_size, --hf_dataset, --log_interval,
--save_interval, and TensorBoard writer as the DPSN-R path.

Architecture recap (dwt/ARCHITECTURE.md):
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
    """Train a DWA model using settings from main.py's parsed args.

    Args:
        args:   argparse.Namespace from main.py (uses batch_size, max_steps,
                hf_dataset, hf_text_column, checkpoint_dir, resume,
                log_interval, save_interval, val_interval, bf16,
                dwa_config).
        writer: TensorBoard SummaryWriter (shared with main.py caller).
    """
    _ensure_dwt_importable()

    # ── dwt imports (flax.nnx based, independent of linen DPSN-R) ──────────
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
        update_ema,
        small_config,
        medium_config,
        large_config,
        cross_entropy,
    )
    from dwa_train import (
        load_and_chunk,
        get_batch,
        shard_batch,
        replicate,
        save_checkpoint,
        load_checkpoint,
        latest_checkpoint,
        N_DEVICES,
        _replicated,
        evaluate_ppl,
    )

    print(f"\n{'=' * 64}")
    print("  DWA (Dynamic Weight Assembly) Training")
    print(f"{'=' * 64}")
    print(f"  JAX devices : {jax.devices()}")
    print(f"  Device count: {N_DEVICES}")

    # ── Config ───────────────────────────────────────────────────────────────
    dwa_config_name = getattr(args, "dwa_config", "small")
    _preset = {"small": small_config, "medium": medium_config, "large": large_config}
    cfg: LMConfig = _preset.get(dwa_config_name, small_config)()

    # Apply overrides from main.py args
    if args.batch_size:
        cfg.batch_size = args.batch_size          # total across all devices
    if args.max_steps:
        cfg.max_steps = args.max_steps

    print(
        f"  Config      : {dwa_config_name}  "
        f"d={cfg.d_model} layers={cfg.n_layers_A}+{cfg.n_layers_B} "
        f"N={cfg.N} D={cfg.D} r={cfg.r} k_max={cfg.k_max}"
    )
    print(f"  Steps       : {cfg.max_steps}  batch={cfg.batch_size}")

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset_name = args.hf_dataset or "roneneldan/TinyStories"
    if hasattr(args, "hf_text_column"):
        text_field = (
            args.hf_text_column[0]
            if isinstance(args.hf_text_column, list)
            else args.hf_text_column
        )
    else:
        text_field = "text"

    print(f"\n  Loading data: {dataset_name}  field='{text_field}'")
    train_data, val_data, vocab_size = load_and_chunk(
        dataset_name=dataset_name,
        text_field=text_field,
        seq_len=cfg.seq_len,
    )
    cfg.vocab_size = vocab_size
    print(f"  Vocab size  : {vocab_size}")

    # ── Model + Optimizer ────────────────────────────────────────────────────
    seed = 0
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
            print(f"  --resume set but no checkpoint found in {ckpt_dir} — starting fresh.")

    # ── Training loop ────────────────────────────────────────────────────────
    np_rng       = np.random.default_rng(start)
    log_interval  = getattr(args, "log_interval", 50)
    save_interval = getattr(args, "save_interval", 1000)
    val_interval  = getattr(args, "val_interval", None) or save_interval

    print(f"\n  Training {cfg.max_steps - start} steps  "
          f"(log={log_interval}, save={save_interval}, val={val_interval})\n")

    t0 = time.perf_counter()

    for s in range(start, cfg.max_steps):
        # ── Batch ─────────────────────────────────────────────────────────
        x, y   = get_batch(train_data, cfg.seq_len, cfg.batch_size, np_rng)
        x_s, y_s = shard_batch(x, y)

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
            phase = _phase_name(s, cfg)
            elapsed = time.perf_counter() - t0
            ce_val = float(bd["ce"])
            print(
                f"  step {s:6d} [{phase:8s}]  "
                f"ce={ce_val:.3f}  "
                f"util={float(bd.get('util', 0)):.3f}  "
                f"div={float(bd.get('div', 0)):.4f}  "
                f"lambda={float(ls):.2f}  "
                f"({elapsed:.0f}s)"
            )
            writer.add_scalar("dwa/train_ce",      ce_val,                   s)
            writer.add_scalar("dwa/lambda_sharp",  float(ls),                s)
            writer.add_scalar("dwa/aux_util",      float(bd.get("util", 0)), s)
            writer.add_scalar("dwa/aux_div",       float(bd.get("div", 0)),  s)
            writer.add_scalar("dwa/aux_norm",      float(bd.get("norm", 0)), s)
            writer.add_scalar("dwa/aux_sparse",    float(bd.get("sparse", 0)), s)
            writer.add_scalar("dwa/phase_idx",     _phase_idx(s, cfg),       s)

        # ── Validation ────────────────────────────────────────────────────
        if s > 0 and s % val_interval == 0:
            ppl = evaluate_ppl(model, val_data, cfg, np_rng, is_dwa=True)
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
    final_ppl = evaluate_ppl(model, val_data, cfg, np_rng, is_dwa=True)
    elapsed   = time.perf_counter() - t0
    print(f"\n  Final val_ppl = {final_ppl:.2f}  total_time = {elapsed:.0f}s")
    writer.add_scalar("dwa/final_val_ppl", final_ppl, cfg.max_steps)
    print(f"{'=' * 64}\n")


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
