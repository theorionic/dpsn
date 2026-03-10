try:
    import grain.python as grain

    GRAIN_AVAILABLE = True
except (ImportError, RuntimeError):
    GRAIN_AVAILABLE = False

import gc
import glob
import numpy as np
import sys
import bisect
import os
from typing import Optional, Any


class NumpySource:
    def __init__(self, path: str, seq_len: int = 1024):
        self.path = path
        # Memory-map the numpy array to avoid loading it all into RAM at once
        raw = np.load(path, mmap_mode='r')

        # If the array is 1D (flat token stream), reshape into sequences
        if raw.ndim == 1:
            num_tokens = len(raw)
            num_sequences = num_tokens // seq_len
            # Trim any leftover tokens that don't fill a complete sequence
            self.data = raw[:num_sequences * seq_len].reshape(num_sequences, seq_len)
            print(f"Loaded {path}: {num_tokens:,} tokens → {num_sequences:,} sequences of length {seq_len}.")
        else:
            # Already 2D (num_sequences, seq_len)
            self.data = raw
            print(f"Loaded {path} with {len(self.data):,} pre-tokenized sequences.")

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Returns a 1D array of shape (seq_len,) — grain Batch stacks these into (B, seq_len)
        return {"input_ids": np.array(self.data[idx])}

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

            try:
                self._dataset = datasets.load_dataset(
                    self.path, name=self.name, split=self.split, streaming=True
                )
            except ValueError as e:
                if "Bad split" in str(e):
                    # Try to fall back to the first available split if the requested one doesn't exist
                    builder = datasets.load_dataset_builder(self.path, name=self.name)
                    splits = list(builder.info.splits.keys())
                    if splits:
                        print(f"Split '{self.split}' not found. Falling back to '{splits[0]}'")
                        self.split = splits[0]
                        self._dataset = datasets.load_dataset(
                            self.path, name=self.name, split=self.split, streaming=True
                        )
                    else:
                        raise e
                else:
                    raise e
        return self._dataset

    def __iter__(self):
        for item in self.dataset:
            if self.text_column and self.text_column in item:
                item["text"] = item[self.text_column]
            yield item

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)


class SequentialSource:
    def __init__(
        self, sources: list[HFStreamSource], dataset_idx: int = 0, hf_state: Optional[dict] = None
    ):
        self.sources = sources
        self.start_dataset_idx = dataset_idx
        self.hf_state = hf_state
        self.current_dataset_idx = 0
        self.current_sample_idx = 0

    def __iter__(self):
        for i, source in enumerate(self.sources):
            if i < self.start_dataset_idx:
                continue
            
            self.current_dataset_idx = i
            
            if i == self.start_dataset_idx and self.hf_state is not None:
                source.load_state_dict(self.hf_state)
                self.hf_state = None
                self.current_sample_idx = 0
            else:
                self.current_sample_idx = 0

            for item in source:
                yield item
                self.current_sample_idx += 1

        self.current_dataset_idx = 0
        self.current_sample_idx = 0
        self.start_dataset_idx = 0

    def get_state(self):
        state = {
            "dataset_idx": self.current_dataset_idx,
            "sample_idx": self.current_sample_idx
        }
        if self.current_dataset_idx < len(self.sources):
            state["hf_state"] = self.sources[self.current_dataset_idx].state_dict()
        return state


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
            elif "input_ids" in item:
                # If it's already tokenized, just append it but ensure it's a numpy array
                if not isinstance(item["input_ids"], np.ndarray):
                    item["input_ids"] = np.array(item["input_ids"], dtype=np.int32)
                batch.append(item)
            else:
                # If we cannot process it, we should probably warn, but for now we skip
                pass

            if len(batch) == self.batch_size:
                collated = {}
                for k in batch[0].keys():
                    if isinstance(batch[0][k], np.ndarray):
                        collated[k] = np.stack([b[k] for b in batch])
                yield collated
                batch = []

    def get_state(self):
        state = self.source.get_state()
        return state


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
        hf_state = None

        # Try to load resume state if resume_data flag is set
        if getattr(config, "resume_data", False) and os.path.exists(resume_data_path):
            try:
                import json

                with open(resume_data_path, "r") as f:
                    state = json.load(f)
                    dataset_idx = state.get("dataset_idx", 0)
                    sample_idx = state.get("sample_idx", 0)
                    hf_state = state.get("hf_state")
                print(
                    f"Resuming HF stream from dataset {dataset_idx}, sample {sample_idx} (using state_dict)"
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
            sources, dataset_idx=dataset_idx, hf_state=hf_state
        )

        transform = TokenizeTransform(tokenizer, max_length=seq_len)
        batch_size = getattr(config, "batch_size", 8)

        return HFStreamLoader(source, transform, batch_size)

    if not GRAIN_AVAILABLE:
        return None

    try:
        import glob
        
        # Expand wildcard dataset paths
        expanded_paths = []
        if dataset_paths:
            for p in dataset_paths:
                if '*' in p or '?' in p:
                    matches = glob.glob(p)
                    if matches:
                        expanded_paths.extend(matches)
                    else:
                        print(f"Warning: No files matched pattern {p}")
                        expanded_paths.append(p)
                else:
                    expanded_paths.append(p)
            dataset_paths = expanded_paths

        dataset_size = getattr(config, "dataset_size", 1000)
        if not dataset_paths:
            source = DummySource(size=dataset_size)
        elif len(dataset_paths) == 1:
            if dataset_paths[0].endswith('.npy'):
                source = NumpySource(path=dataset_paths[0], seq_len=seq_len)
            else:
                source = DummySource(path=dataset_paths[0], size=dataset_size)
        else:
            # Multi-dataset support: concatenate sources
            sources = []
            for p in dataset_paths:
                if p.endswith('.npy'):
                    sources.append(NumpySource(path=p, seq_len=seq_len))
                else:
                    sources.append(DummySource(path=p, size=dataset_size))
            source = ConcatenatedSource(sources)

        batch_size = getattr(config, "batch_size", 8)
        start_index = start_step * batch_size

        operations = []
        
        # Only tokenize if the source doesn't already provide input_ids
        # (e.g. if we loaded from .npy files, it's already tokenized!)
        if not any(p.endswith(".npy") for p in (dataset_paths or [])):
            operations.append(TokenizeTransform(tokenizer, max_length=seq_len))
            
        operations.append(grain.Batch(batch_size=batch_size, drop_remainder=True))

        worker_count = getattr(config, "num_workers", 4)
        if sys.platform == "darwin":
            worker_count = 0

        # Shuffling requires generating a random permutation of `num_records`. 
        # For huge datasets (e.g., 200M sequences), this takes several minutes on CPU
        # and hangs the DevicePrefetchIterator causing `_queue.Empty` timeouts.
        # We disable shuffling for massive datasets.
        total_records = len(source)
        should_shuffle = total_records < 10_000_000
        if not should_shuffle:
            print(f"Dataset is massive ({total_records:,} sequences). Disabling IndexSampler shuffling to prevent initialization timeouts.")

        loader = grain.DataLoader(
            data_source=source,
            operations=operations,
            sampler=grain.IndexSampler(
                num_records=total_records,
                shard_options=grain.NoSharding(),
                shuffle=should_shuffle,
                seed=0 if should_shuffle else None,
                num_epochs=getattr(config, "epochs", 1),
            ),
            worker_count=worker_count,
            worker_buffer_size=500,
        )

        return loader
    except Exception as e:
        import traceback
        print("Error initializing grain dataloader:")
        traceback.print_exc()
        return None


def expand_npy_paths(dataset_paths: Optional[list[str]]) -> list[str]:
    """Expand glob patterns and return a sorted list of .npy file paths."""
    if not dataset_paths:
        return []

    expanded = []
    for p in dataset_paths:
        if '*' in p or '?' in p:
            matches = sorted(glob.glob(p))
            if matches:
                expanded.extend(matches)
            else:
                print(f"Warning: No files matched pattern {p}")
        elif p.endswith('.npy'):
            expanded.append(p)

    return sorted(set(expanded))


def get_single_npy_grain_loader(
    npy_path: str, args: Any, start_step: int = 0, config: Any = None
) -> Optional[tuple[Any, int]]:
    """Create a grain DataLoader for a single .npy file.

    This loads only ONE file into memory at a time, avoiding the RAM blow-up
    caused by ConcatenatedSource loading all files simultaneously.

    Args:
        npy_path:    Path to a single .npy file.
        args:        Argparse namespace (has batch_size, num_workers).
        start_step:  Resume offset.
        config:      Model config (has max_seq_len). Falls back to args if None.

    Returns:
        (loader, total_records) tuple, or None on failure.
    """
    if not GRAIN_AVAILABLE:
        return None

    try:
        # Prefer config for seq_len, fall back to args
        cfg = config if config is not None else args
        seq_len = getattr(cfg, "max_seq_len", getattr(cfg, "seq_len", 1024))
        source = NumpySource(path=npy_path, seq_len=seq_len)
        batch_size = getattr(args, "batch_size", 8)
        start_index = start_step * batch_size

        operations = [
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ]

        worker_count = getattr(args, "num_workers", 4)
        if sys.platform == "darwin":
            worker_count = 0

        total_records = len(source)
        should_shuffle = total_records < 10_000_000
        if not should_shuffle:
            print(
                f"File has {total_records:,} sequences. "
                f"Disabling IndexSampler shuffling."
            )

        loader = grain.DataLoader(
            data_source=source,
            operations=operations,
            sampler=grain.IndexSampler(
                num_records=total_records,
                shard_options=grain.NoSharding(),
                shuffle=should_shuffle,
                seed=0 if should_shuffle else None,
                num_epochs=1,  # single pass per file
            ),
            worker_count=worker_count,
            worker_buffer_size=500,
        )

        return loader, total_records
    except Exception as e:
        import traceback
        print(f"Error initializing grain loader for {npy_path}:")
        traceback.print_exc()
        return None


def release_npy_loader(loader: Any) -> None:
    """Explicitly release a single-file grain loader and free memory."""
    if loader is None:
        return
    # Drop references to the data source so mmap can be released
    try:
        if hasattr(loader, '_data_source'):
            loader._data_source = None
        if hasattr(loader, 'data_source'):
            loader.data_source = None
    except Exception:
        pass
    del loader
    gc.collect()

