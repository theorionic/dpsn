import random
import threading
import queue
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
            except Exception:
                break

    def get_batch(self, batch_size=None):
        return self.queue.get()

    def stop(self):
        self.stop_event.set()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass
