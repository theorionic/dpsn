import random
import threading
import queue
import time
import gc
import multiprocessing as mp
import os
import logging
import warnings
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

if mp.current_process().name != "MainProcess":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from itertools import islice
import numpy as np
from typing import List, Optional
from datasets import load_dataset
from .tokenizer import SimpleNumberTokenizer


# ─── Top-level picklable worker for ChunkedHFDataset ──────────────────────────
# Must be at module level (not a nested/lambda fn) so multiprocessing can pickle it.
def _tokenize_texts_worker(args_tuple: tuple) -> np.ndarray:
    """
    Tokenizes a list of texts in a subprocess.
    args_tuple: (texts: List[str], tokenizer_name: str, seq_len: int)
    Returns np.ndarray of shape (N, seq_len) dtype int32.
    """
    texts, tokenizer_name, seq_len = args_tuple
    from dpsn_r_jax.data.tokenizer import get_tokenizer  # local import: child process
    tokenizer = get_tokenizer(tokenizer_name)
    is_hf = hasattr(tokenizer, "__call__") and not hasattr(tokenizer, "max_val")

    if is_hf:
        # GPT-2 and other decoder-only models ship without a pad token.
        # Set it to eos_token so padding="max_length" doesn't crash.
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id or 0

        encoded = tokenizer(
            texts,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        return encoded["input_ids"].astype(np.int32)
    else:
        result = []
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) > seq_len:
                ids = ids[:seq_len]
            else:
                ids = ids + [pad_id] * (seq_len - len(ids))
            result.append(ids)
        return np.array(result, dtype=np.int32)


class HFStreamingDataset:
    def __init__(
        self,
        dataset_name,
        tokenizer,
        subset=None,
        split="train",
        seq_len=64,
        batch_size=8,
    ):
        self.dataset = load_dataset(
            dataset_name, name=subset, split=split, streaming=True
        )
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.iterator = iter(self.dataset)

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        batch_texts = []
        try:
            for _ in range(batch_size):
                item = next(self.iterator)
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("sentence")
                    or str(item)
                )
                batch_texts.append(text)
        except StopIteration:
            self.iterator = iter(self.dataset)
            while len(batch_texts) < batch_size:
                item = next(self.iterator)
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("sentence")
                    or str(item)
                )
                batch_texts.append(text)

        batch_ids = []
        for text in batch_texts:
            if hasattr(self.tokenizer, "__call__"):
                if (
                    hasattr(self.tokenizer, "pad_token_id")
                    and self.tokenizer.pad_token_id is not None
                ):
                    pad_id = self.tokenizer.pad_token_id
                else:
                    pad_id = 0

                tokens = self.tokenizer(
                    text,
                    max_length=self.seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np",
                )
                ids = tokens["input_ids"][0]
            else:
                ids = self.tokenizer.encode(text)
                if len(ids) > self.seq_len:
                    ids = ids[: self.seq_len]

                pad_id = self.tokenizer.pad_token_id
                if len(ids) < self.seq_len:
                    ids = ids + [pad_id] * (self.seq_len - len(ids))
                ids = np.array(ids)

            batch_ids.append(ids)

        return np.array(batch_ids)

def _worker_tokenize(worker_id, dataset_name, subset, split, tokenizer_name, seq_len, batch_size, out_queue, stop_event):
    """
    Subprocess worker that streams HF dataset and tokenizes aggressively.
    """
    import numpy as np
    from datasets import load_dataset
    from dpsn_r_jax.data.tokenizer import get_tokenizer

    # Re-initialize tokenizer in the child process
    tokenizer = get_tokenizer(tokenizer_name)
    
    # Check if HF tokenizer
    is_hf = hasattr(tokenizer, "__call__") and not hasattr(tokenizer, "max_val")

    if is_hf:
        # GPT-2 and other decoder-only models ship without a pad token.
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

    # Each worker needs to skip a different number of items to avoid extreme overlap
    # Note: For True streaming we can't easily shard perfectly, but we skip to stagger them
    try:
        dataset = load_dataset(dataset_name, name=subset, split=split, streaming=True)
    except ValueError as e:
        if "Bad split" in str(e):
            import datasets
            builder = datasets.load_dataset_builder(dataset_name, name=subset)
            splits = list(builder.info.splits.keys())
            if splits:
                print(f"[Worker {worker_id}] Split '{split}' not found. Falling back to '{splits[0]}'")
                split = splits[0]
                dataset = load_dataset(dataset_name, name=subset, split=split, streaming=True)
            else:
                raise e
        else:
            raise e
            
    iterator = iter(dataset)
    
    # Stagger workers efficiently
    try:
        if worker_id > 0:
            # We skip a manageable number to stagger the streams without hammering HF servers
            dataset = dataset.skip(worker_id * 200)
    except AttributeError:
        # Fallback if skip isn't supported on this iterable
        for _ in range(worker_id * 200):
            try:
                next(iterator)
            except StopIteration:
                break

    batch_texts = []
    
    while not stop_event.is_set():
        try:
            while len(batch_texts) < batch_size:
                item = next(iterator)
                text = item.get("text") or item.get("content") or item.get("sentence") or ""
                if text:
                    batch_texts.append(text)
                    
            # Tokenize batch
            if is_hf:
                encoded = tokenizer(
                    batch_texts,
                    max_length=seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np"
                )
                batch_ids = encoded["input_ids"].astype(np.int32)
            else:
                batch_ids = []
                for text in batch_texts:
                    ids = tokenizer.encode(text)
                    if len(ids) > seq_len:
                        ids = ids[:seq_len]
                    else:
                        ids = ids + [pad_id] * (seq_len - len(ids))
                    batch_ids.append(ids)
                batch_ids = np.array(batch_ids, dtype=np.int32)

            # Try to push to queue (timeout allows checking stop_event)
            while not stop_event.is_set():
                try:
                    out_queue.put(batch_ids, timeout=1.0)
                    break
                except queue.Full:
                    continue

            batch_texts = []
            
        except StopIteration:
            # Loop dataset
            iterator = iter(dataset)
        except Exception as e:
            if not stop_event.is_set():
                import traceback
                traceback.print_exc()
            break

class MultiprocessingHFDataset:
    """
    A replacement for HFStreamingDataset that uses multiple Python processes 
    to fetch and tokenize huggingface data in parallel.
    """
    def __init__(
        self,
        dataset_name,
        tokenizer_name,
        subset=None,
        split="train",
        seq_len=64,
        batch_size=8,
        num_workers=8,
        prefetch_batches=100
    ):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.subset = subset
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Cross-process resources using SPAWN context to avoid JAX multithreading deadlocks
        # and OS fork throttling/limits
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        self.queue = manager.Queue(maxsize=prefetch_batches)
        self.stop_event = manager.Event()
        self.processes = []
        
        print(f"Starting {self.num_workers} parallel data workers for {dataset_name} using SPAWN context...")
        for i in range(self.num_workers):
            p = ctx.Process(
                target=_worker_tokenize, 
                args=(
                    i, 
                    self.dataset_name, 
                    self.subset, 
                    self.split, 
                    self.tokenizer_name, 
                    self.seq_len, 
                    self.batch_size, 
                    self.queue,
                    self.stop_event
                ),
                daemon=True
            )
            p.start()
            self.processes.append(p)

    def get_batch(self, batch_size=None):
        # We ignore requested batch_size since workers already batch it correctly
        try:
            # Block until a batch is ready
            return self.queue.get(timeout=60.0) 
        except queue.Empty:
            raise RuntimeError("Dataloader queue is empty! Workers might have crashed or network is down.")

    def stop(self):
        self.stop_event.set()
        for p in self.processes:
            p.terminate()
            p.join(timeout=1.0)


class SyntheticReasoningDataset:
    def __init__(self, size=1000, seq_len=64, max_val=20):
        self.size = size
        self.seq_len = seq_len
        self.max_val = max_val
        self.tokenizer = SimpleNumberTokenizer(max_val=max_val)
        self.data = self._generate_data()

    def _generate_data(self):
        print("Generating synthetic sorting dataset...")
        samples = []

        for _ in range(self.size):
            # Only Sort task
            length = random.randint(3, 8)
            # Generate numbers between 0 and max_val-1
            tokens = [random.randint(0, self.max_val - 1) for _ in range(length)]

            input_str = " ".join(map(str, tokens))
            sorted_str = " ".join(map(str, sorted(tokens)))

            text = f"Sort: {input_str} -> {sorted_str}"

            samples.append(text + " " + self.tokenizer.eos_token)
        return samples

    def get_batch(self, batch_size):
        batch_data = random.sample(self.data, batch_size)

        # Manually pad and batch
        batch_ids = []
        for text in batch_data:
            ids = self.tokenizer.encode(text)
            # Truncate
            if len(ids) > self.seq_len:
                ids = ids[: self.seq_len]
            # Pad
            if len(ids) < self.seq_len:
                ids = ids + [self.tokenizer.pad_token_id] * (self.seq_len - len(ids))
            batch_ids.append(ids)

        return np.array(batch_ids)


class BackgroundGenerator:
    def __init__(self, dataset, batch_size, prefetch_size=5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                batch = self.dataset.get_batch(self.batch_size)
                self.queue.put(batch)
            except StopIteration as e:
                self.queue.put(e)
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.queue.put(e)
                break

    def get_batch(self, batch_size=None):
        res = self.queue.get()
        if isinstance(res, Exception):
            raise res
        return res

    def stop(self):
        self.stop_event.set()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass


# ─── Chunk-based HuggingFace dataset ──────────────────────────────────────────

class ChunkedHFDataset:
    """
    Chunk-based HuggingFace streaming dataset.

    Downloads exactly `chunk_size` rows at a time using the HF streaming
    iterator (equivalent to .take(chunk_size)), tokenizes them in parallel
    with a multiprocessing.Pool, shuffles the result in-place for true
    within-chunk randomness, then serves batches from that RAM block.

    While training consumes chunk N a background thread is already
    downloading and tokenizing chunk N+1, so chunk boundaries cause zero
    training stall.

    Timeline:
        t=0  Download chunk-0 synchronously  →  training starts
        t=1  Train on chunk-0  |  BG thread downloads chunk-1
        t=2  Train on chunk-1  |  BG thread downloads chunk-2
        ...

    Args:
        dataset_name:          HuggingFace dataset identifier.
        tokenizer_name:        Tokenizer name passed to ``get_tokenizer()``.
        chunk_size:            Number of rows to download per chunk (e.g. 10_000).
        subset:                Optional dataset config/subset name.
        split:                 Dataset split (default ``"train"``).
        seq_len:               Token sequence length (sequences are truncated/padded).
        batch_size:            Default batch size for ``get_batch()``.
        num_tokenizer_workers: CPU cores used for parallel tokenization per chunk.
        text_columns:          Column names tried (in order) to extract text.
    """

    _SENTINEL = object()  # signals background thread has no more chunks

    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        chunk_size: int = 10_000,
        subset: Optional[str] = None,
        split: str = "train",
        seq_len: int = 512,
        batch_size: int = 8,
        num_tokenizer_workers: int = 4,
        text_columns: Optional[List[str]] = None,
        skip_rows: int = 0,
        hf_state: Optional[dict] = None,
        text_fn=None,
    ):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.chunk_size = chunk_size
        self.subset = subset
        self.split = split
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_tokenizer_workers = max(1, num_tokenizer_workers)
        self.text_columns = text_columns or ["text", "content", "sentence"]
        self.text_fn = text_fn  # optional type-aware formatter from preprocessor

        # Queue holds at most 1 pre-downloaded chunk so BG thread stays 1 chunk ahead.
        self._next_chunk_q: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

        # Current serving state
        self._current_chunk: Optional[np.ndarray] = None
        self._read_pos: int = 0

        # Total sequences served (used for grain_state.json resume)
        self._rows_consumed: int = 0

        # Reference to the HF IterableDataset (kept for state_dict / load_state_dict)
        self._hf_ds = None

        # Persistent HF iterator — advanced across all chunks so we never repeat rows
        self._hf_iter = self._make_iterator(hf_state=hf_state)

        # Restore rows_consumed counter so get_state() reports the correct absolute position.
        # When hf_state restores the iterator via O(1) seek, skip_rows still carries
        # the total rows consumed from the grain_state — use it to seed the counter.
        if hf_state is not None and skip_rows > 0:
            self._rows_consumed = skip_rows

        # ── Fast-forward via islice when no hf_state available (slower fallback) ──
        if hf_state is None and skip_rows > 0:
            print(
                f"[ChunkedHFDataset] Skipping {skip_rows:,} rows to resume data position "
                f"(this may take a while for large skip counts)..."
            )
            remaining = skip_rows
            skipped = 0
            while remaining > 0:
                n = min(remaining, chunk_size)
                consumed = sum(1 for _ in islice(self._hf_iter, n))
                skipped += consumed
                remaining -= consumed
                if consumed < n:
                    # Iterator exhausted mid-skip — restart
                    self._hf_iter = self._make_iterator()
                    remaining = 0  # stop; we've wrapped around
            self._rows_consumed = skipped
            print(f"[ChunkedHFDataset] Fast-forward complete — skipped {skipped:,} rows.")

        # ── Synchronous first chunk (training cannot start until it's ready) ──
        _resume_label = f"rows {self._rows_consumed:,}+" if self._rows_consumed > 0 else "beginning"
        print(
            f"[ChunkedHFDataset] Fetching chunk ({chunk_size:,} rows) "
            f"from '{dataset_name}' (resuming from {_resume_label})..."
        )
        first = self._fetch_and_tokenize_chunk()
        if first is None or len(first) == 0:
            raise RuntimeError(
                "[ChunkedHFDataset] Failed to fetch any data for the first chunk. "
                "Check dataset name, split, and network connectivity."
            )
        self._current_chunk = first
        self._read_pos = 0
        print(
            f"[ChunkedHFDataset] Chunk ready — {len(first):,} sequences "
            f"({len(first) * seq_len * 4 / 1e6:.1f} MB). Training starts now!"
        )

        # ── Background thread pre-fetches the next chunk immediately ──────────
        self._bg_thread = threading.Thread(
            target=self._background_prefetch, daemon=True
        )
        self._bg_thread.start()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_iterator(self, hf_state: Optional[dict] = None):
        """Create a fresh HF streaming iterator, with split fallback.

        If hf_state is provided, calls ds.load_state_dict() before iterating
        for O(1) seek to a previously saved position.
        """
        try:
            ds = load_dataset(
                self.dataset_name, name=self.subset, split=self.split, streaming=True
            )
        except ValueError as exc:
            if "Bad split" in str(exc):
                import datasets as _hf_datasets
                builder = _hf_datasets.load_dataset_builder(
                    self.dataset_name, name=self.subset
                )
                available = list(builder.info.splits.keys())
                if available:
                    print(
                        f"[ChunkedHFDataset] Split '{self.split}' not found; "
                        f"falling back to '{available[0]}'"
                    )
                    self.split = available[0]
                    ds = load_dataset(
                        self.dataset_name,
                        name=self.subset,
                        split=self.split,
                        streaming=True,
                    )
                else:
                    raise
            else:
                raise

        if hf_state is not None:
            try:
                ds.load_state_dict(hf_state)
                print("[ChunkedHFDataset] Restored HF iterator position from state_dict (O(1) seek).")
            except Exception as e:
                print(f"[ChunkedHFDataset] Warning: could not load hf_state ({e}); starting from beginning.")

        self._hf_ds = ds
        return iter(ds)

    def get_state(self) -> dict:
        """Return the current data loader state for grain_state.json.

        Compatible with the fine-tuning-v2 grain_state format:
        {"dataset_idx": 0, "sample_idx": N, "hf_state": {...}, "rows_consumed": N}
        """
        state = {
            "dataset_idx": 0,
            "sample_idx": self._rows_consumed,
            "rows_consumed": self._rows_consumed,
        }
        if self._hf_ds is not None:
            try:
                state["hf_state"] = self._hf_ds.state_dict()
            except Exception as e:
                print(f"[ChunkedHFDataset] Warning: could not capture hf_state ({e}).")
        return state

    def _extract_text(self, item: dict) -> str:
        if self.text_fn is not None:
            return self.text_fn(item)
        for col in self.text_columns:
            val = item.get(col)
            if val:
                return str(val)
        return ""

    def _fetch_and_tokenize_chunk(self) -> Optional[np.ndarray]:
        """
        Pull exactly `chunk_size` rows from the HF iterator using islice —
        the same primitive HF's .take(N) uses internally.  Applied to the
        *persistent* iterator so successive calls advance through the dataset
        instead of restarting each time.

        If the iterator is exhausted mid-chunk, restarts from the beginning
        and fills the remainder so training never stalls.
        """
        # ── One call, no Python-level loop — grabs exactly chunk_size rows ──
        raw: List[dict] = list(islice(self._hf_iter, self.chunk_size))

        while len(raw) < self.chunk_size:
            shortage = self.chunk_size - len(raw)
            if len(raw) == 0:
                print("[ChunkedHFDataset] End of dataset reached; restarting stream from the beginning.")
            else:
                print(f"[ChunkedHFDataset] Dataset exhausted after {len(raw):,} rows; restarting to collect {shortage:,} more.")
            
            self._hf_iter = self._make_iterator()
            new_rows = list(islice(self._hf_iter, shortage))
            if not new_rows:
                if not raw:
                    return None
                break
            raw.extend(new_rows)

        texts: List[str] = list(filter(None, (self._extract_text(item) for item in raw)))

        if not texts:
            return None

        # ── Parallel tokenization ──────────────────────────────────────────────
        # Split texts evenly across workers. Each worker gets a contiguous slice.
        n_workers = min(self.num_tokenizer_workers, len(texts))
        sub_size = max(1, len(texts) // n_workers)
        sub_batches = [
            (texts[i : i + sub_size], self.tokenizer_name, self.seq_len)
            for i in range(0, len(texts), sub_size)
        ]

        if n_workers > 1:
            ctx = mp.get_context("spawn")  # safe with JAX / existing threads
            pool = ctx.Pool(processes=n_workers)
            try:
                results = pool.map(_tokenize_texts_worker, sub_batches)
            finally:
                pool.close()
                pool.join()
        else:
            results = [_tokenize_texts_worker(b) for b in sub_batches]

        chunk = np.concatenate(results, axis=0)  # (N, seq_len)
        np.random.shuffle(chunk)                  # true in-chunk shuffle
        return chunk

    def _background_prefetch(self) -> None:
        """Continuously download + tokenize the next chunk and enqueue it."""
        while not self._stop_event.is_set():
            chunk = self._fetch_and_tokenize_chunk()
            if self._stop_event.is_set():
                break
            if chunk is None:
                self._next_chunk_q.put(self._SENTINEL)
                break
            # Block until the consumer slot is free (queue maxsize=1)
            while not self._stop_event.is_set():
                try:
                    self._next_chunk_q.put(chunk, timeout=1.0)
                    break
                except queue.Full:
                    continue

    def _swap_chunk(self) -> None:
        """
        Block until the background thread delivers the next chunk, then swap
        it in as the current serving buffer.

        The old chunk numpy array is explicitly deleted and gc.collect() is
        called BEFORE the new chunk is assigned so peak RAM stays at ~1 chunk
        rather than 2 chunks during the swap.
        """
        print("[ChunkedHFDataset] Current chunk exhausted — waiting for next chunk...")
        item = self._next_chunk_q.get(timeout=600.0)   # 10-min safety timeout
        if item is self._SENTINEL:
            # Background thread finished AND dataset has no more data;
            # restart by re-creating the stream.
            print("[ChunkedHFDataset] Sentinel received; restarting data stream.")
            self._hf_iter = self._make_iterator()
            # Re-launch background thread
            self._bg_thread = threading.Thread(
                target=self._background_prefetch, daemon=True
            )
            self._bg_thread.start()
            item = self._next_chunk_q.get(timeout=600.0)
            if item is self._SENTINEL:
                raise RuntimeError("[ChunkedHFDataset] No data available.")

        # ── Explicitly free old chunk BEFORE assigning new one ─────────────
        # Simply reassigning self._current_chunk drops the ref-count to 0 but
        # Python's allocator may not return the pages to the OS immediately.
        # del + gc.collect() forces an immediate release so RAM stays at ~1
        # chunk size instead of briefly peaking at 2× during the swap.
        if self._current_chunk is not None:
            old_mb = self._current_chunk.nbytes / 1e6
            del self._current_chunk
            self._current_chunk = None
            gc.collect()   # return numpy pages back to OS now
            print(f"[ChunkedHFDataset] Old chunk freed ({old_mb:.1f} MB released).")

        self._current_chunk = item
        self._read_pos = 0
        print(
            f"[ChunkedHFDataset] Swapped to new chunk: {len(self._current_chunk):,} "
            f"sequences ({self._current_chunk.nbytes / 1e6:.1f} MB in RAM)"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_batch(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Return the next batch of shape ``(batch_size, seq_len)``.

        Served directly from the RAM chunk at memory speed.  When the chunk
        is exhausted this call blocks only for the brief moment it takes the
        background thread to hand off the already-downloaded next chunk.
        """
        bs = batch_size or self.batch_size

        if self._current_chunk is None or self._read_pos + bs > len(self._current_chunk):
            self._swap_chunk()

        batch = self._current_chunk[self._read_pos : self._read_pos + bs].copy()
        self._read_pos += bs
        self._rows_consumed += bs
        return batch

    def stop(self) -> None:
        """Signal the background thread to stop and join it."""
        self._stop_event.set()
        # Drain so the background thread is not stuck on a full put()
        while not self._next_chunk_q.empty():
            try:
                self._next_chunk_q.get_nowait()
            except queue.Empty:
                break
        self._bg_thread.join(timeout=5.0)
