import random
import threading
import queue
import time
import multiprocessing as mp
import numpy as np
import jax.numpy as jnp
from datasets import load_dataset
from .tokenizer import SimpleNumberTokenizer


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
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0

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
