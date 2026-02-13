try:
    import grain.python as grain

    GRAIN_AVAILABLE = True
except (ImportError, RuntimeError):
    GRAIN_AVAILABLE = False

import numpy as np
import sys
import bisect
from typing import Optional, Any


class DummySource:
    def __init__(self, path: str = "dummy", size: int = 1000):
        self.path = path
        self.size = size
        self.data = [f"[{path}] Sort: 3 1 2 -> 1 2 3 <eos>" for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"text": self.data[idx % self.size]}


class ConcatenatedSource:
    def __init__(self, sources: list[Any]):
        self.sources = sources
        self.sizes = [len(s) for s in sources]
        self.cum_sizes = np.cumsum(self.sizes).tolist()
        self.total_size = self.cum_sizes[-1] if self.cum_sizes else 0

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        if idx < 0:
            idx %= self.total_size
        if idx >= self.total_size:
            raise IndexError("Index out of range")

        source_idx = bisect.bisect_right(self.cum_sizes, idx)
        if source_idx == 0:
            inner_idx = idx
        else:
            inner_idx = idx - self.cum_sizes[source_idx - 1]
        return self.sources[source_idx][inner_idx]


def dummy_tokenize(text: str, max_length: int = 64):
    tokens = [ord(c) % 100 for c in text]
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))
    return np.array(tokens, dtype=np.int32)


class TokenizeTransform:
    def __init__(self, max_length: int = 64):
        self.max_length = max_length

    def map(self, element):
        element["input_ids"] = dummy_tokenize(element["text"], self.max_length)
        return element


def get_grain_loader(
    dataset_paths: Optional[list[str]], config: Any, start_step: int = 0
) -> Optional[Any]:
    if not GRAIN_AVAILABLE:
        return None

    try:
        dataset_size = getattr(config, "dataset_size", 1000)
        if not dataset_paths:
            source = DummySource(size=dataset_size)
        elif len(dataset_paths) == 1:
            source = DummySource(path=dataset_paths[0], size=dataset_size)
        else:
            # Multi-dataset support: concatenate sources
            sources = [DummySource(path=p, size=dataset_size) for p in dataset_paths]
            source = ConcatenatedSource(sources)

        batch_size = getattr(config, "batch_size", 8)
        start_index = start_step * batch_size

        operations = [
            TokenizeTransform(max_length=getattr(config, "seq_len", 64)),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ]

        worker_count = getattr(config, "num_workers", 4)
        if sys.platform == "darwin":
            worker_count = 0

        loader = grain.DataLoader(
            data_source=source,
            operations=operations,
            sampler=grain.IndexSampler(
                num_records=len(source),
                shard_options=grain.NoSharding(),
                shuffle=True,
                num_epochs=getattr(config, "epochs", 1),
            ),
            worker_count=worker_count,
            worker_buffer_size=500,
            read_options=grain.ReadOptions(start_index=start_index),
        )

        return loader
    except Exception:
        return None
