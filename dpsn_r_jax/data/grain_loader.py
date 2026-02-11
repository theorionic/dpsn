try:
    import grain.python as grain

    GRAIN_AVAILABLE = True
except (ImportError, RuntimeError):
    GRAIN_AVAILABLE = False

import numpy as np
import sys
from typing import Optional, Any


class DummySource:
    def __init__(self, size: int = 1000):
        self.size = size
        self.data = [f"Sort: 3 1 2 -> 1 2 3 <eos>" for _ in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"text": self.data[idx]}


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


def get_grain_loader(config: Any) -> Optional[Any]:
    if not GRAIN_AVAILABLE:
        return None

    try:
        source = DummySource(size=getattr(config, "dataset_size", 1000))

        operations = [
            TokenizeTransform(max_length=getattr(config, "seq_len", 64)),
            grain.Batch(
                batch_size=getattr(config, "batch_size", 8), drop_remainder=True
            ),
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
        )

        return loader
    except Exception:
        return None
