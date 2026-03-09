"""Async double-buffered host-to-device prefetch for TPU training.

This module provides `DevicePrefetchIterator`, which transfers batches to TPU
in a background thread so that compute and data transfer overlap.  This is the
single most impactful optimisation for eliminating TPU idle time caused by
synchronous `jax.device_put` calls in the training loop.

Architecture:
    CPU DataSource → BackgroundGenerator (CPU queue, depth N)
                   → DevicePrefetchIterator (background thread doing device_put)
                   → training loop receives batches already on-device
"""

import threading
import queue
from typing import Any, Optional

import jax
import jax.numpy as jnp


class DevicePrefetchIterator:
    """Asynchronously transfers batches to TPU via a background thread.

    Maintains a double-buffer (configurable depth) of on-device arrays so that
    the next batch is already resident on TPU when `train_step` finishes the
    current one.

    Args:
        data_source:  Any object with a `get_batch(batch_size)` method that
                      returns a host-side NumPy array.
        batch_size:   Batch size passed to `data_source.get_batch`.
        sharding:     A `jax.sharding.NamedSharding` describing how the batch
                      should be distributed across devices.
        prefetch_depth: Number of on-device batches to keep ready (default 2).
                        2 is sufficient for full overlap; higher values trade
                        extra device memory for resilience to CPU jitter.
    """

    _SENTINEL = object()  # signals the background thread to stop

    def __init__(
        self,
        data_source: Any,
        batch_size: int,
        sharding: Any,
        prefetch_depth: int = 2,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.sharding = sharding

        self._queue: queue.Queue = queue.Queue(maxsize=prefetch_depth)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()

    # ── Background thread ──────────────────────────────────────────────────
    def _prefetch_loop(self) -> None:
        """Continuously fetches batches from CPU and pushes them to device."""
        while not self._stop_event.is_set():
            try:
                # 1. Get a host-side (NumPy) batch from the CPU data source
                host_batch = self.data_source.get_batch(self.batch_size)

                # 2. Transfer to device with the target sharding.
                #    `jax.device_put` is *non-blocking* on the Python side when
                #    given a sharding — it enqueues the DMA transfer and returns
                #    a future-like DeviceArray.  By the time the training loop
                #    actually consumes it, the transfer has completed.
                device_batch = jax.device_put(host_batch, self.sharding)

                # 3. Push the on-device batch into the consumer queue.
                #    Use a timeout so we can check the stop event periodically.
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(device_batch, timeout=1.0)
                        break
                    except queue.Full:
                        continue

            except Exception as exc:
                if not self._stop_event.is_set():
                    # Propagate exception to the consumer so it doesn't hang.
                    self._queue.put(exc)
                break

    # ── Public API ─────────────────────────────────────────────────────────
    def get_batch(self, batch_size: Optional[int] = None) -> jnp.ndarray:
        """Return the next batch, already on-device.

        This call blocks only if the background thread hasn't finished
        transferring the next batch yet — but with depth-2 buffering this
        almost never happens because H2D overlaps with compute.
        """
        item = self._queue.get(timeout=120.0)
        if isinstance(item, Exception):
            raise item
        return item

    def stop(self) -> None:
        """Gracefully shut down the prefetch thread."""
        self._stop_event.set()
        # Drain the queue so the producer thread isn't stuck on a full put()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._thread.join(timeout=5.0)
