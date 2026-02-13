try:
    import grain.python as grain

    GRAIN_AVAILABLE = True
except (ImportError, RuntimeError):
    GRAIN_AVAILABLE = False

import numpy as np
import sys
import bisect
import os
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


class HFStreamSource:
    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        split: str = "train",
        text_column: Optional[str] = None,
    ):
        self.path = path
        self.name = name
        self.split = split
        self.text_column = text_column
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            import datasets

            self._dataset = datasets.load_dataset(
                self.path, name=self.name, split=self.split, streaming=True
            )
        return self._dataset

    def __iter__(self):
        for item in self.dataset:
            if self.text_column and self.text_column in item:
                item["text"] = item[self.text_column]
            yield item

    def skip(self, n: int):
        if n > 0:
            self._dataset = self.dataset.skip(n)
        return self


class SequentialSource:
    def __init__(
        self, sources: list[HFStreamSource], dataset_idx: int = 0, sample_idx: int = 0
    ):
        self.sources = sources
        self.dataset_idx = dataset_idx
        self.sample_idx = sample_idx
        self.current_sample_idx = sample_idx

    def __iter__(self):
        for i in range(self.dataset_idx, len(self.sources)):
            self.dataset_idx = i
            source = self.sources[i]

            if i == self.dataset_idx and self.sample_idx > 0:
                source.skip(self.sample_idx)
                self.current_sample_idx = self.sample_idx
            else:
                self.current_sample_idx = 0

            for item in source:
                yield item
                self.current_sample_idx += 1

            self.sample_idx = 0

        # After completing all sources, reset indices for potential re-iteration
        self.dataset_idx = 0
        self.sample_idx = 0
        self.current_sample_idx = 0

    def get_state(self):
        return {"dataset_idx": self.dataset_idx, "sample_idx": self.current_sample_idx}


class HFStreamLoader:
    def __init__(
        self,
        source: SequentialSource,
        transform: Any,
        batch_size: int,
    ):
        self.source = source
        self.transform = transform
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for item in self.source:
            # Map common HF text fields to "text" as fallback if not already set by Source
            if "text" not in item:
                for key in ["content", "body", "text_content"]:
                    if key in item:
                        item["text"] = item[key]
                        break

            if "text" not in item:
                # Use the first string field as a fallback
                for k, v in item.items():
                    if isinstance(v, str):
                        item["text"] = v
                        break

            if "text" in item:
                item = self.transform.map(item)
                batch.append(item)

            if len(batch) == self.batch_size:
                collated = {}
                for k in batch[0].keys():
                    if isinstance(batch[0][k], np.ndarray):
                        collated[k] = np.stack([b[k] for b in batch])
                yield collated
                batch = []

    def get_state(self):
        return self.source.get_state()


def dummy_tokenize(text: str, max_length: int = 64):
    tokens = [ord(c) % 100 for c in text]
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens = tokens + [0] * (max_length - len(tokens))
    return np.array(tokens, dtype=np.int32)


class TokenizeTransform:
    def __init__(self, tokenizer: Any, max_length: int = 64):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def map(self, element):
        text = element.get("text", "")

        if hasattr(self.tokenizer, "__call__") and not hasattr(
            self.tokenizer, "max_val"
        ):
            # HuggingFace Tokenizer
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            element["input_ids"] = encoded["input_ids"][0].astype(np.int32)
        elif hasattr(self.tokenizer, "encode"):
            # SimpleNumberTokenizer
            ids = self.tokenizer.encode(text)
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
            else:
                pad_id = getattr(self.tokenizer, "pad_token_id", 0)
                ids = ids + [pad_id] * (self.max_length - len(ids))
            element["input_ids"] = np.array(ids, dtype=np.int32)
        else:
            # Fallback to dummy_tokenize if no proper tokenizer is provided
            element["input_ids"] = dummy_tokenize(text, self.max_length)

        return element


def get_grain_loader(
    dataset_paths: Optional[list[str]], config: Any, start_step: int = 0
) -> Optional[Any]:
    from dpsn_r_jax.data.tokenizer import get_tokenizer

    tokenizer_name = getattr(config, "hf_tokenizer", None) or "numeric"
    tokenizer = get_tokenizer(tokenizer_name)
    seq_len = getattr(config, "seq_len", getattr(config, "max_seq_len", 64))

    # Check for HF datasets first
    hf_datasets = getattr(config, "hf_datasets", None)
    resume_data_path = getattr(config, "resume_data_path", "grain_state.json")

    if hf_datasets:
        dataset_idx = 0
        sample_idx = 0

        # Try to load resume state if resume_data flag is set
        if getattr(config, "resume_data", False) and os.path.exists(resume_data_path):
            try:
                import json

                with open(resume_data_path, "r") as f:
                    state = json.load(f)
                    dataset_idx = state.get("dataset_idx", 0)
                    sample_idx = state.get("sample_idx", 0)
                print(
                    f"Resuming HF stream from dataset {dataset_idx}, sample {sample_idx}"
                )
            except Exception as e:
                print(f"Failed to load resume state: {e}")

        # Prepare text columns
        text_columns = getattr(config, "hf_text_column", ["text"])
        if isinstance(text_columns, str):
            text_columns = [text_columns]

        # Broadcast text_columns if necessary
        if len(text_columns) == 1 and len(hf_datasets) > 1:
            text_columns = text_columns * len(hf_datasets)
        elif len(text_columns) != len(hf_datasets):
            print(
                f"Warning: Number of text columns ({len(text_columns)}) "
                f"does not match number of datasets ({len(hf_datasets)}). "
                "Using default 'text' for remaining datasets."
            )
            text_columns = text_columns + ["text"] * (
                len(hf_datasets) - len(text_columns)
            )

        sources = [
            HFStreamSource(path, text_column=col)
            for path, col in zip(hf_datasets, text_columns)
        ]
        source = SequentialSource(
            sources, dataset_idx=dataset_idx, sample_idx=sample_idx
        )

        transform = TokenizeTransform(tokenizer, max_length=seq_len)
        batch_size = getattr(config, "batch_size", 8)

        return HFStreamLoader(source, transform, batch_size)

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
            TokenizeTransform(tokenizer, max_length=seq_len),
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
