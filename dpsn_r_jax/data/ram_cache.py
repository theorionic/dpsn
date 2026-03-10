"""RAM-resident tokenized data cache for eliminating CPU→TPU data starvation.

Solves the problem where HuggingFace streaming + tokenization can't keep up with
TPU compute speed.  Pre-tokenizes a large chunk of data into RAM so batches can
be served at memory bandwidth speed (~100 GB/s) instead of network+tokenization
speed (~1 MB/s per worker).

Pipeline with cache enabled:
    HF Stream → [bg fill thread] → RAM Cache (5 GB) → DevicePrefetchIterator → TPU
"""

import time
import threading
import numpy as np
from typing import Any, Optional


class TokenizedRAMCache:
    """Caches pre-tokenized sequences in a large RAM buffer.

    1. Synchronously pre-fills a configurable fraction of the buffer
    2. Training starts immediately after prefill completes
    3. Background thread continues filling the remainder during training
    4. ``get_batch()`` serves from RAM at memory speed with periodic shuffling

    Args:
        data_source:   Any object with a ``get_batch(batch_size)`` method
                       returning ``np.ndarray`` of shape ``(B, seq_len)``.
        batch_size:    Number of sequences per batch.
        seq_len:       Sequence length (tokens per sequence).
        cache_size_gb: Total RAM to allocate for the cache (in GB).
        prefill_pct:   Fraction of the cache to fill *before* training starts.
                       Range [0.0, 1.0].  Lower values start training sooner
                       but risk re-reading the same data early on.
    """

    def __init__(
        self,
        data_source: Any,
        batch_size: int,
        seq_len: int,
        cache_size_gb: float = 5.0,
        prefill_pct: float = 0.1,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len

        bytes_per_seq = seq_len * 4  # int32
        self.capacity = int(cache_size_gb * 1e9 / bytes_per_seq)

        total_gb = self.capacity * bytes_per_seq / 1e9
        print(
            f"RAM Cache: Allocating {total_gb:.2f} GB buffer "
            f"for {self.capacity:,} sequences (seq_len={seq_len})"
        )

        self.buffer = np.empty((self.capacity, seq_len), dtype=np.int32)

        self._write_pos = 0
        self._available = 0  # readable sequence count
        self._read_pos = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # ── Synchronous pre-fill (blocks until done) ───────────────────────
        prefill_target = max(
            batch_size * 10,  # absolute minimum
            int(self.capacity * min(prefill_pct, 1.0)),
        )
        prefill_target = min(prefill_target, self.capacity)
        self._prefill(data_source, prefill_target)

        # ── Background fill for the remainder ──────────────────────────────
        if self._write_pos < self.capacity:
            self._thread = threading.Thread(
                target=self._bg_fill, args=(data_source,), daemon=True
            )
            self._thread.start()
        else:
            print("RAM Cache: Buffer fully loaded.")

    # ── Internal fill methods ──────────────────────────────────────────────

    def _ingest_batch(self, batch: Any, limit: int) -> int:
        """Write one batch into the buffer. Returns number of sequences written."""
        if not isinstance(batch, np.ndarray):
            batch = np.array(batch, dtype=np.int32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        n = min(batch.shape[0], limit - self._write_pos)
        if n <= 0:
            return 0
        self.buffer[self._write_pos : self._write_pos + n] = batch[:n]
        self._write_pos += n
        return n

    def _prefill(self, source: Any, target: int) -> None:
        """Synchronously fill buffer before training starts."""
        start = time.time()
        last_print = start

        while self._write_pos < target:
            try:
                batch = source.get_batch(self.batch_size)
            except Exception as e:
                print(f"RAM Cache: Prefill stopped early — {e}")
                break

            self._ingest_batch(batch, target)

            now = time.time()
            if now - last_print >= 5.0:
                filled_gb = self._write_pos * self.seq_len * 4 / 1e9
                target_gb = target * self.seq_len * 4 / 1e9
                rate = self._write_pos / max(now - start, 0.01)
                eta = (target - self._write_pos) / max(rate, 1)
                pct = 100 * self._write_pos / target
                print(
                    f"  Prefill: {self._write_pos:,}/{target:,} seqs "
                    f"({filled_gb:.2f}/{target_gb:.2f} GB) [{pct:.0f}%] "
                    f"| {rate:.0f} seq/s | ETA: {eta:.0f}s"
                )
                last_print = now

        with self._lock:
            self._available = self._write_pos

        elapsed = time.time() - start
        gb = self._available * self.seq_len * 4 / 1e9
        print(
            f"RAM Cache: Pre-filled {self._available:,} sequences "
            f"({gb:.2f} GB) in {elapsed:.1f}s — training starts now!"
        )

        # Shuffle for randomness
        if self._available > 0:
            np.random.shuffle(self.buffer[: self._available])

    def _bg_fill(self, source: Any) -> None:
        """Continue filling cache in background during training."""
        last_print = time.time()
        while not self._stop.is_set() and self._write_pos < self.capacity:
            try:
                batch = source.get_batch(self.batch_size)
                written = self._ingest_batch(batch, self.capacity)
                if written > 0:
                    with self._lock:
                        self._available = self._write_pos

                    now = time.time()
                    if now - last_print >= 30.0:
                        gb = self._available * self.seq_len * 4 / 1e9
                        total_gb = self.capacity * self.seq_len * 4 / 1e9
                        pct = 100 * self._available / self.capacity
                        print(
                            f"  RAM Cache (bg): {self._available:,}/{self.capacity:,} "
                            f"seqs ({gb:.2f}/{total_gb:.2f} GB) [{pct:.0f}%]"
                        )
                        last_print = now
                else:
                    break
            except Exception:
                break

        gb = self._available * self.seq_len * 4 / 1e9
        print(
            f"RAM Cache: Background fill complete — "
            f"{self._available:,} sequences ({gb:.2f} GB)"
        )

        # Clean up data source workers if it supports stopping
        if hasattr(source, 'stop'):
            source.stop()

    # ── Public API ─────────────────────────────────────────────────────────

    def get_batch(self, batch_size: int = None) -> np.ndarray:
        """Return a batch from the cache at memory speed.

        When the read pointer wraps past the available data, the buffer is
        shuffled in-place and reading restarts from the beginning.
        """
        bs = batch_size or self.batch_size

        with self._lock:
            avail = self._available

        # Wrap around and shuffle when exhausted
        if self._read_pos + bs > avail:
            np.random.shuffle(self.buffer[:avail])
            self._read_pos = 0

        batch = self.buffer[self._read_pos : self._read_pos + bs].copy()
        self._read_pos += bs
        return batch

    def stop(self) -> None:
        """Gracefully shut down the background fill thread."""
        self._stop.set()
