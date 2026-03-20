"""
Component-level wall-clock timing for JAX JIT-compiled model internals.

WHY jax.debug.callback instead of print() / time.time():
  - Inside jax.jit, Python runs at TRACE time, not execution time.
    time.time() would always return the same compile-time value.
  - jax.debug.callback(fn, array, ordered=True) fires the Python fn
    at actual EXECUTION time, in order, once per call (or once per
    lax.scan iteration inside a scan body).
  - This records host-side dispatch timestamps as XLA enqueues each
    op group.  For the 1.3s step you are seeing, this will show you
    which component accounts for what fraction of that dispatch time.

NOTE:
  The callbacks are always compiled into the JIT trace (zero overhead
  at trace time).  At execution time, when disabled, the callback body
  returns immediately (~1 µs per mark).  Enable only for diagnosis.

Usage in JIT / lax.scan:
    from dpsn_r_jax.utils.component_timer import ctimer
    ctimer.mark("after_controller", hidden)   # hidden is any live JAX array
"""

import time
import threading
import jax

__all__ = ["ComponentTimer", "ctimer"]


class ComponentTimer:
    def __init__(self):
        self._enabled: bool = False
        self._lock = threading.Lock()
        self._marks: list[tuple[str, float]] = []

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def enable(self):
        """Call from the host before the training step you want to profile."""
        with self._lock:
            self._enabled = True
            self._marks = []

    def disable(self):
        with self._lock:
            self._enabled = False

    def reset(self):
        """Clear recorded marks (keep enabled state)."""
        with self._lock:
            self._marks = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Internal callback (runs on host at XLA execution time) ───────────────

    def _make_cb(self, tag: str):
        """Return a callback closure that records (tag, timestamp)."""
        def _cb(dummy_array):
            # Check enabled at execution time so we can toggle without
            # forcing a JIT re-trace.
            if not self._enabled:
                return
            with self._lock:
                self._marks.append((tag, time.perf_counter()))
        return _cb

    # ── Public API (call from inside JIT-compiled / lax.scan code) ───────────

    def mark(self, tag: str, trigger_array):
        """
        Insert a timing checkpoint labelled `tag`.

        `trigger_array` is any live JAX array produced by the operation
        you just finished — it ensures the callback is sequenced AFTER
        that operation in the XLA execution graph.

        Safe inside jax.lax.scan: fires once per scan iteration.
        Safe with jax.checkpoint: fires on the forward pass only.
        """
        # ordered=True is NOT supported on multi-device JIT (raises OrderedDebugEffect).
        # ordered=False works on any number of devices; timestamps are still accurate
        # because time.perf_counter() is called inside the callback at execution time.
        jax.debug.callback(self._make_cb(tag), trigger_array, ordered=False)

    # ── Host-side reporting (call AFTER jax.block_until_ready) ───────────────

    def print_summary(self, step: int = -1, total_step_ms: float = 0.0):
        """
        Print a table of all marks collected since the last enable()/reset().

        Call this from the HOST after block_until_ready() so that all
        callbacks have had a chance to fire.

        Args:
            step:           global training step (for the header line).
            total_step_ms:  full step wall time in ms (for % column).
        """
        with self._lock:
            marks = list(self._marks)

        if not marks:
            print("[ComponentTimer] No marks recorded — was ctimer.enable() called "
                  "before the JIT step?", flush=True)
            return

        dispatched_ms = (marks[-1][1] - marks[0][1]) * 1000.0
        ref_ms = total_step_ms if total_step_ms > 0 else dispatched_ms

        W = 72
        print(f"\n{'═'*W}", flush=True)
        hdr = f" COMPONENT TIMING  step={step}" if step >= 0 else " COMPONENT TIMING"
        print(f"{hdr}  (host-dispatch order; accelerator may overlap)", flush=True)
        print(f"  dispatch window captured: {dispatched_ms:.1f}ms of "
              f"{total_step_ms:.1f}ms total step", flush=True)
        print(f"  {'Stage':<42} {'Δms':>7}  {'cumul':>7}  {'%step':>6}  bar",
              flush=True)
        print(f"  {'─'*42}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*15}", flush=True)

        t0 = marks[0][1]
        prev_t = t0
        seen_tags: dict[str, int] = {}

        for tag, t in marks:
            # Deduplicate repeated tags from lax.scan — append [N] suffix
            count = seen_tags.get(tag, 0)
            seen_tags[tag] = count + 1
            display_tag = f"{tag}[{count}]" if count > 0 else tag

            delta_ms   = (t - prev_t)  * 1000.0
            cumul_ms   = (t - t0)      * 1000.0
            pct        = 100.0 * delta_ms / ref_ms if ref_ms > 0 else 0.0
            bar_len    = max(0, min(15, int(pct / 100.0 * 15)))
            bar        = "█" * bar_len + "░" * (15 - bar_len)

            print(f"  {display_tag:<42}  {delta_ms:>7.2f}  {cumul_ms:>7.2f}  "
                  f"{pct:>5.1f}%  {bar}", flush=True)
            prev_t = t

        print(f"  {'─'*42}  {'─'*7}  {'─'*7}  {'─'*6}", flush=True)
        print(f"  {'TOTAL DISPATCHED':<42}  {dispatched_ms:>7.2f}ms", flush=True)

        if total_step_ms > 0:
            uncaptured = total_step_ms - dispatched_ms
            print(f"  {'sync/overhead (not in marks)':<42}  {uncaptured:>7.2f}ms",
                  flush=True)

        print(f"{'═'*W}\n", flush=True)


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this in dpsnr.py and main.py — they share the same instance.
ctimer = ComponentTimer()
