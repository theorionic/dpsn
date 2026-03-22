"""
dpsn_r_jax/utils/model_profiler.py

Fine-grained wall-clock profiler for every DPSNR component.

WHY this exists instead of ctimer
──────────────────────────────────
ctimer uses jax.debug.callback which is blocked on multi-device TPU
(jax.device_count() > 1) due to GSPMD auto-partition restrictions.
On v5e-8, ctimer silently does nothing.

This profiler works on ANY backend / device count by isolating each
component into its own jax.jit function and timing it with
jax.block_until_ready — the only universally reliable timing method
in JAX.

What is measured
─────────────────
  1.  TinyController       — full transformer encoder forward pass
  2.  LearnedIndexer       — attention pooling + mu/sigma projection
  3.  Pool retrieve (×1)   — one dynamic_slice + Gaussian aggregation
  4.  Retrieval integrator — 2-layer MLP on (B, T, 2D) → (B, T, D)
  5.  ACC (×1)             — one AdaptiveComputeController step
  6.  Reasoning iter (×1)  — full single iteration: indexer+pool+integ+acc
  7.  Reasoning loop (×R)  — full lax.scan over R iterations
  8.  LM head decoder      — chunked (B,T,D)×(D,V) + cross-entropy
  9.  Forward pass total   — controller + reasoning loop + decoder
  10. Train step total     — forward + backward + optimizer (wall clock)

  Derived:
    backward + optimizer  ≈ train_step_total − forward_total
    per-iteration overhead = reasoning_loop / R − reasoning_iter
    (overhead = scan bookkeeping + carry communication)

Usage
──────
    python main.py --config xxl --batch_size 32 \\
        --profile_model --profile_model_runs 5

    # or from code after state is initialised:
    from dpsn_r_jax.utils.model_profiler import run_model_profile
    run_model_profile(model, state, config, sample_batch,
                      batch_sharding, replicated_sharding,
                      warmup=3, runs=10)
"""

from __future__ import annotations

import time
import functools
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding, PartitionSpec

__all__ = ["run_model_profile"]

# ─────────────────────────────────────────────────────────────────────────────
# Low-level timing helper
# ─────────────────────────────────────────────────────────────────────────────

def _time_fn(fn, *args, warmup: int = 3, runs: int = 10) -> dict:
    """
    JIT-compile fn(*args), warm up, then time `runs` executions.

    Returns a dict:
        median_ms, mean_ms, min_ms, max_ms, std_ms
    All values are wall-clock milliseconds measured around
    jax.block_until_ready so device execution is fully included.
    """
    # First call compiles; block so compilation is not counted in warmup
    out = fn(*args)
    jax.block_until_ready(out)

    # Warmup (compiled, not timed)
    for _ in range(max(0, warmup - 1)):
        jax.block_until_ready(fn(*args))

    times_ms = []
    for _ in range(runs):
        t0  = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    a = np.array(times_ms)
    return {
        "median_ms": float(np.median(a)),
        "mean_ms":   float(np.mean(a)),
        "min_ms":    float(np.min(a)),
        "max_ms":    float(np.max(a)),
        "std_ms":    float(np.std(a)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Component extractors
# Each returns a JIT-compiled callable + args tuple for _time_fn.
# ─────────────────────────────────────────────────────────────────────────────

def _build_controller_fn(model, state, sample_batch, batch_sharding):
    """Time: controller-only forward (no reasoning, no LM head)."""
    params = state.params

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, x):
        return model.apply(
            {"params": p}, x,
            method=lambda m, ids: m.controller(ids, deterministic=True),
        )

    batch = jax.device_put(sample_batch, batch_sharding)
    return _fn, (params, batch)


def _build_indexer_fn(model, state, sample_batch, batch_sharding):
    """Time: indexer forward (attention pooling + mu/sigma projection)."""
    params = state.params

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, x):
        hidden = model.apply(
            {"params": p}, x,
            method=lambda m, ids: m.controller(ids, deterministic=True),
        )
        return model.apply(
            {"params": p}, hidden,
            method=lambda m, h: m.indexer(h, sigma_max_scale=1.0),
        )

    batch = jax.device_put(sample_batch, batch_sharding)
    return _fn, (params, batch)


def _build_pool_retrieve_fn(model, state, sample_batch, config, batch_sharding):
    """Time: a single pool retrieval call (dynamic_slice + Gaussian aggregation)."""
    params = state.params
    B      = sample_batch.shape[0]

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, mu_r, mu_c, sigma):
        return model.apply(
            {"params": p}, mu_r, mu_c, sigma,
            method=lambda m, r, c, s: m.pool(r, c, s),
        )

    # mu_r/mu_c/sigma are 1D (B,) — derive 1D sharding from batch_sharding
    # (batch_sharding has PartitionSpec(dp_axis, None) for 2D; we need 1D)
    try:
        _1d_sharding = NamedSharding(
            batch_sharding.mesh,
            PartitionSpec(batch_sharding.spec[0]),
        )
    except Exception:
        _1d_sharding = batch_sharding  # fallback: may error, caught by _bench
    mu_r  = jax.device_put(jnp.full((B,), 0.5), _1d_sharding)
    mu_c  = jax.device_put(jnp.full((B,), 0.5), _1d_sharding)
    sigma = jax.device_put(jnp.full((B,), 2.0), _1d_sharding)
    return _fn, (params, mu_r, mu_c, sigma)


def _build_integrator_fn(model, state, sample_batch, config, batch_sharding):
    """Time: retrieval integrator (2-layer MLP on concatenated hidden+retrieved)."""
    params = state.params
    B      = sample_batch.shape[0]
    T      = config.max_seq_len
    D      = config.controller_hidden_dim
    dtype  = jnp.bfloat16 if config.use_bf16 else jnp.float32

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, combined):
        return model.apply(
            {"params": p}, combined,
            method=lambda m, c: m.retrieval_integrator(c),
        )

    combined = jax.device_put(
        jnp.zeros((B, T, 2 * D), dtype=dtype), batch_sharding
    )
    return _fn, (params, combined)


def _build_acc_fn(model, state, sample_batch, config, batch_sharding):
    """Time: AdaptiveComputeController (one step: gate + norm + halt)."""
    params = state.params
    B      = sample_batch.shape[0]
    T      = config.max_seq_len
    D      = config.controller_hidden_dim
    dtype  = jnp.bfloat16 if config.use_bf16 else jnp.float32

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, h, step_out, halt_prob, halted_mask):
        return model.apply(
            {"params": p}, h, step_out, 0, halt_prob, halted_mask,
            method=lambda m, s, so, i, hp, hm: m.acc(s, so, i, hp, hm),
        )

    h           = jax.device_put(jnp.zeros((B, T, D), dtype=dtype), batch_sharding)
    step_out    = jax.device_put(jnp.zeros((B, T, D), dtype=dtype), batch_sharding)
    halt_prob   = jax.device_put(jnp.zeros((B, T, 1), dtype=dtype), batch_sharding)
    halted_mask = jax.device_put(jnp.zeros((B, T, 1), dtype=dtype), batch_sharding)
    return _fn, (params, h, step_out, halt_prob, halted_mask)


def _build_reasoning_iter_fn(model, state, sample_batch, config, batch_sharding):
    """
    Time: ONE complete reasoning iteration (indexer + pool + integrator + acc).

    This mirrors exactly one body of the lax.scan reasoning_step but compiled
    as a single JIT function — gives the minimum achievable per-iteration time.
    Multiply by max_reasoning_loops to compare against the actual scan cost.
    """
    params = state.params
    B      = sample_batch.shape[0]
    T      = config.max_seq_len
    D      = config.controller_hidden_dim
    dtype  = jnp.bfloat16 if config.use_bf16 else jnp.float32

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, hidden):
        def _step(m, h):
            mu, sigma = m.indexer(h, sigma_max_scale=1.0)
            H           = config.num_indexer_heads
            heads_per_dim = max(1, H // 2)
            mu_r  = mu[:, 0]
            mu_c  = mu[:, min(heads_per_dim, H - 1)]
            sigma_h = (sigma[:, 0] + sigma[:, min(heads_per_dim, H - 1)]) / 2.0

            retrieved, _ = m.pool(mu_r, mu_c, sigma_h)

            retrieved_exp = jnp.broadcast_to(retrieved[:, None, :], (h.shape[0], T, D))
            combined      = jnp.concatenate([h, retrieved_exp], axis=-1)
            integrated    = m.retrieval_integrator(combined)

            halt_prob   = jnp.zeros((h.shape[0], T, 1), dtype=h.dtype)
            halted_mask = jnp.zeros((h.shape[0], T, 1), dtype=h.dtype)
            new_h, _, _ = m.acc(h, h + integrated, 0, halt_prob, halted_mask)
            return new_h

        return model.apply({"params": p}, hidden, method=_step)

    hidden = jax.device_put(jnp.zeros((B, T, D), dtype=dtype), batch_sharding)
    return _fn, (params, hidden)


def _build_prefetch_encode_fn(model, state, sample_batch, config, batch_sharding):
    """
    Time: _prefetch_encode (used when config.prefetch_reasoning=True).

    This is the ACTUAL reasoning path when --prefetch_reasoning is passed.
    It fetches a patch_size×patch_size candidate block from the pool ONCE,
    then runs R iterations of dot-product attention over those SRAM candidates.

    Replaces _build_reasoning_loop_fn in the profiler when prefetch_reasoning
    is active — benchmarking the standard scan would measure the wrong path.
    """
    params = state.params
    B      = sample_batch.shape[0]
    T      = config.max_seq_len
    D      = config.controller_hidden_dim
    dtype  = jnp.bfloat16 if config.use_bf16 else jnp.float32

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, hidden):
        return model.apply(
            {"params": p}, hidden,
            method=lambda m, h: m._prefetch_encode(h, sigma_max_scale=1.0),
        )

    hidden = jax.device_put(jnp.zeros((B, T, D), dtype=dtype), batch_sharding)
    return _fn, (params, hidden)


def _build_reasoning_loop_fn(model, state, sample_batch, config, batch_sharding):
    """
    Time: full lax.scan reasoning loop (R iterations exactly as in training).
    This captures all scan overhead: carry communication, XLA loop bookkeeping.
    """
    params = state.params
    B      = sample_batch.shape[0]
    T      = config.max_seq_len
    D      = config.controller_hidden_dim
    R      = config.max_reasoning_loops
    dtype  = jnp.bfloat16 if config.use_bf16 else jnp.float32

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, hidden):
        def _loop(m, h):
            halt_prob   = jnp.zeros((B, T, 1), dtype=h.dtype)
            halted_mask = jnp.zeros((B, T, 1), dtype=h.dtype)

            def step(carry, i):
                s_h, hp, hm = carry
                mu, sigma = m.indexer(s_h, sigma_max_scale=1.0)
                H           = config.num_indexer_heads
                heads_per_dim = max(1, H // 2)
                mu_r  = mu[:, 0]
                mu_c  = mu[:, min(heads_per_dim, H - 1)]
                sigma_h = (sigma[:, 0] + sigma[:, min(heads_per_dim, H - 1)]) / 2.0

                retrieved, _ = m.pool(mu_r, mu_c, sigma_h)
                ret_exp   = jnp.broadcast_to(retrieved[:, None, :], (B, T, D))
                combined  = jnp.concatenate([s_h, ret_exp], axis=-1)
                integrated = m.retrieval_integrator(combined)
                new_s, hp, hm = m.acc(s_h, s_h + integrated, i, hp, hm)
                update_mask = 1.0 - hm
                s_h = (update_mask * new_s + hm * s_h).astype(dtype)
                return (s_h, hp.astype(dtype), hm.astype(dtype)), None

            (out, _, _), _ = jax.lax.scan(step, (h, halt_prob, halted_mask),
                                           jnp.arange(R))
            return out

        return model.apply({"params": p}, hidden, method=_loop)

    hidden = jax.device_put(jnp.zeros((B, T, D), dtype=dtype), batch_sharding)
    return _fn, (params, hidden)


def _build_decoder_fn(model, state, sample_batch, config, batch_sharding,
                      replicated_sharding, pad_token_id: int):
    """Time: LM head (chunked or unchunked cross-entropy)."""
    from dpsn_r_jax.training.trainer import chunked_lm_loss

    params    = state.params
    B         = sample_batch.shape[0]
    T         = config.max_seq_len
    D         = config.controller_hidden_dim
    chunk     = getattr(config, "loss_chunk_size", 0)
    dtype     = jnp.bfloat16 if config.use_bf16 else jnp.float32

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, hidden, labels):
        decode_fn = lambda h: model.apply(
            {"params": p}, h, method=lambda m, x: m.controller.decode(x)
        )
        if chunk > 0:
            return chunked_lm_loss(hidden, labels, decode_fn, pad_token_id, chunk)
        else:
            logits = decode_fn(hidden)
            loss   = optax.softmax_cross_entropy_with_integer_labels(
                logits[:, :-1].astype(jnp.float32),
                labels[:, 1:],
            )
            return jnp.mean(loss)

    hidden = jax.device_put(jnp.zeros((B, T, D), dtype=dtype), batch_sharding)
    labels = jax.device_put(jnp.zeros((B, T), dtype=jnp.int32), batch_sharding)
    return _fn, (params, hidden, labels)


def _build_forward_fn(model, state, sample_batch, config, batch_sharding):
    """Time: complete forward pass (controller + reasoning + decoder)."""
    params   = state.params
    use_bf16 = getattr(config, "use_bf16", False)

    @functools.partial(jax.jit, donate_argnums=())
    def _fn(p, x):
        logits, _ = model.apply({"params": p}, x, deterministic=True, sigma_max_scale=1.0)
        return logits

    batch = jax.device_put(sample_batch, batch_sharding)
    return _fn, (params, batch)


def _build_train_step_fn(train_step_fn, state, sample_batch,
                         config, batch_sharding, current_lr, sigma_scale):
    """Time: full train_step (forward + backward + optimizer)."""
    import jax

    batch = jax.device_put(sample_batch, batch_sharding)

    def _fn(s, b, lr, ss):
        return train_step_fn(
            s, b, lr, ss,
            pad_token_id=config.pad_token_id,
            precision_loss_weight=getattr(config, "precision_loss_weight", 0.0),
            sigma_anneal_steps=getattr(config, "sigma_anneal_steps", 0),
            use_bf16=getattr(config, "use_bf16", False),
            loss_chunk_size=getattr(config, "loss_chunk_size", 0),
        )

    return _fn, (state, batch, current_lr, sigma_scale)


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(results: dict, config, step: int, runs: int = 10) -> None:
    """
    Print a structured breakdown table.

    results keys  → timing dicts with median_ms etc.
    Special keys: 'reasoning_loop' used to derive per-iter overhead.
    """
    W = 78
    sep = "─" * W

    def _row(label, ms, pct_of=None, indent=0):
        prefix = "  " * indent
        pct_str = f"{100.0 * ms / pct_of:5.1f}%" if pct_of and pct_of > 0 else "      "
        bar_len = max(0, min(20, int(20 * ms / pct_of))) if pct_of else 0
        bar     = "█" * bar_len + "░" * (20 - bar_len) if pct_of else ""
        print(f"  {prefix}{label:<42} {ms:8.1f} ms  {pct_str}  {bar}", flush=True)

    R        = config.max_reasoning_loops
    loop_ms  = (results.get("reasoning_loop") or {}).get("median_ms")
    iter_ms  = (results.get("reasoning_iter") or {}).get("median_ms")
    ctrl_ms  = (results.get("controller")     or {}).get("median_ms")
    dec_ms   = (results.get("decoder")        or {}).get("median_ms")

    scan_overhead_ms = (loop_ms - R * iter_ms) if (loop_ms and iter_ms) else None

    # Use controller + loop + decoder as the denominator for % bars.
    # This is the forward-pass estimate (full forward OOMs on large configs).
    fwd_est_ms = sum(x for x in [ctrl_ms, loop_ms, dec_ms] if x)
    bar_base   = fwd_est_ms if fwd_est_ms > 0 else None

    print(f"\n{'═'*W}", flush=True)
    print(f"  DPSNR MODEL PROFILE  —  step={step}", flush=True)
    print(f"  config: T={config.max_seq_len}, D={config.controller_hidden_dim}, "
          f"L={config.controller_num_layers}, R={R}", flush=True)
    print(f"  Each cell = median over {runs} timed runs  "
          f"(% and bar relative to ctrl+loop+decoder sum)", flush=True)
    print(f"  NOTE: full forward/train_step not timed — "
          f"use --timing_interval for fwd/bwd split during training.", flush=True)
    print(f"  {sep}", flush=True)
    print(f"  {'Component':<44} {'median':>9}  {'% fwd':>6}  {'bar (20=100%)':>20}",
          flush=True)
    print(f"  {sep}", flush=True)

    if ctrl_ms:
        _row("TinyController (encoder)", ctrl_ms, bar_base)

    print(f"  {sep}", flush=True)

    _prefetch_active = getattr(config, "prefetch_reasoning", False)
    _loop_label = (f"Reasoning Loop ×{R} (prefetch_encode)"
                   if _prefetch_active else
                   f"Reasoning Loop ×{R} (lax.scan total)")
    if loop_ms:
        _row(_loop_label, loop_ms, bar_base)

        if results.get("indexer"):
            idx_total_ms = results["indexer"]["median_ms"]
            idx_ms       = max(idx_total_ms - (ctrl_ms or 0), 0.1)
            _row(f"  ├─ LearnedIndexer (×{R} est.)", idx_ms * R, bar_base, indent=1)

        if results.get("pool_retrieve"):
            pool_ms = results["pool_retrieve"]["median_ms"]
            _row(f"  ├─ Pool retrieve (×{R} est.)",  pool_ms * R, bar_base, indent=1)

        if results.get("integrator"):
            integ_ms = results["integrator"]["median_ms"]
            _row(f"  ├─ Retrieval integrator (×{R} est.)", integ_ms * R, bar_base, indent=1)

        if results.get("acc"):
            acc_ms = results["acc"]["median_ms"]
            _row(f"  ├─ ACC (×{R} est.)", acc_ms * R, bar_base, indent=1)

        if iter_ms:
            _row(f"  ├─ 1 full iteration (measured)", iter_ms, bar_base, indent=1)
            _row(f"  ├─ ×{R} iter extrapolated",      iter_ms * R, bar_base, indent=1)
            if scan_overhead_ms is not None and abs(scan_overhead_ms) > 0.5:
                _row(f"  └─ lax.scan overhead", scan_overhead_ms, bar_base, indent=1)

    print(f"  {sep}", flush=True)

    if dec_ms:
        _row("LM Head Decoder (chunked CE)", dec_ms, bar_base)

    print(f"  {sep}", flush=True)

    if bar_base:
        _row("Forward estimate (ctrl+loop+dec)", fwd_est_ms, bar_base)
        print(f"\n  Component timing notes:", flush=True)
        print(f"    Forward estimate (isolated benchmarks, no memory pressure): "
              f"{fwd_est_ms:.0f} ms", flush=True)
        print(f"    In full training, actual step time >> forward estimate because:", flush=True)
        print(f"      • Backward pass = ~2× forward compute", flush=True)
        print(f"      • gradient_checkpointing recompute = +1× forward", flush=True)
        print(f"      • Pool gradient scatter + sparse Adam = hidden cost", flush=True)
        print(f"      • HBM bandwidth contention across all components", flush=True)
        print(f"    Use --timing_interval to measure the actual fwd/bwd split.", flush=True)

    print(f"\n  Timing breakdown (min / median / max) over {runs} runs:", flush=True)
    print(f"  {sep}", flush=True)
    for name, r in results.items():
        if not r:
            continue
        print(f"  {name:<30}  min={r['min_ms']:7.1f}  med={r['median_ms']:7.1f}  "
              f"max={r['max_ms']:7.1f}  std={r['std_ms']:5.1f}  ms", flush=True)

    print(f"{'═'*W}\n", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_model_profile(
    model,
    state,
    config,
    sample_batch,
    batch_sharding,
    replicated_sharding,
    warmup: int = 3,
    runs:   int = 10,
    step:   int = 0,
) -> None:
    """
    Run all component benchmarks and print a detailed breakdown table.

    Each component is compiled into its own JIT function and timed with
    jax.block_until_ready — works on multi-device (v5e-8) unlike ctimer.

    The full forward pass and train_step are NOT timed here:
      - forward: OOMs on large configs (activations don't fit alongside params)
      - train_step: uses donate_argnums=(0,) which would invalidate state for
        subsequent training, and the first call includes JIT compilation time.
    Use --timing_interval to measure forward vs backward split during training.

    Args:
        model:               DPSNR model instance.
        state:               Current TrainState (contains params on device).
        config:              DPSNRConfig.
        sample_batch:        A representative (B, T) int32 batch (on device).
        batch_sharding:      NamedSharding for the batch dimension.
        replicated_sharding: NamedSharding for replicated tensors.
        warmup:              Warmup runs before timing (compiled, not counted).
        runs:                Timed runs per component (median reported).
        step:                Current training step (for the report header).
    """
    pad_token_id = getattr(config, "pad_token_id", 0)

    print(f"\n[MODEL PROFILER] Starting — warmup={warmup}, runs={runs}", flush=True)
    print(f"[MODEL PROFILER] Each component is JIT-compiled separately and timed", flush=True)
    print(f"[MODEL PROFILER] with jax.block_until_ready (works on multi-device).\n", flush=True)

    results = {}

    def _bench(name: str, builder_fn, *builder_args):
        print(f"  [{name}] compiling + timing...", end=" ", flush=True)
        try:
            fn, args = builder_fn(*builder_args)
            r = _time_fn(fn, *args, warmup=warmup, runs=runs)
            results[name] = r
            print(f"median {r['median_ms']:.1f} ms", flush=True)
        except Exception as exc:
            print(f"FAILED ({exc})", flush=True)
            traceback.print_exc()
            results[name] = None

    # ── 1. Controller ─────────────────────────────────────────────────────────
    _bench("controller", _build_controller_fn,
           model, state, sample_batch, batch_sharding)

    # ── 2. Indexer (includes controller in the fn; we subtract in report) ─────
    _bench("indexer", _build_indexer_fn,
           model, state, sample_batch, batch_sharding)

    # ── 3. Pool retrieve ──────────────────────────────────────────────────────
    _bench("pool_retrieve", _build_pool_retrieve_fn,
           model, state, sample_batch, config, batch_sharding)

    # ── 4. Retrieval integrator ───────────────────────────────────────────────
    _bench("integrator", _build_integrator_fn,
           model, state, sample_batch, config, batch_sharding)

    # ── 5. ACC ────────────────────────────────────────────────────────────────
    _bench("acc", _build_acc_fn,
           model, state, sample_batch, config, batch_sharding)

    # ── 6. One full reasoning iteration ──────────────────────────────────────
    _bench("reasoning_iter", _build_reasoning_iter_fn,
           model, state, sample_batch, config, batch_sharding)

    # ── 7. Full reasoning loop — standard OR prefetch path ────────────────────
    if getattr(config, "prefetch_reasoning", False):
        print(f"  [reasoning_loop] --prefetch_reasoning active: "
              f"benchmarking _prefetch_encode path...", end=" ", flush=True)
        try:
            fn, args = _build_prefetch_encode_fn(
                model, state, sample_batch, config, batch_sharding)
            r = _time_fn(fn, *args, warmup=warmup, runs=runs)
            results["reasoning_loop"] = r
            print(f"median {r['median_ms']:.1f} ms  (prefetch path)", flush=True)
        except Exception as exc:
            print(f"FAILED ({exc})", flush=True)
            traceback.print_exc()
            results["reasoning_loop"] = None
    else:
        _bench("reasoning_loop", _build_reasoning_loop_fn,
               model, state, sample_batch, config, batch_sharding)

    # ── 8. Decoder / LM head ──────────────────────────────────────────────────
    _bench("decoder", _build_decoder_fn,
           model, state, sample_batch, config, batch_sharding,
           replicated_sharding, pad_token_id)

    # ── Report ────────────────────────────────────────────────────────────────
    _print_report(results, config, step, runs=runs)
