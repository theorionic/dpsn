import json
import os
from typing import Optional, Dict, Any, List, Union, Iterator
import numpy as np
from dataclasses import dataclass

from .templates import (
    get_template,
    load_custom_template,
    PromptTemplate,
    FlexibleTemplate,
    detect_dataset_format,
    build_column_mapping,
    format_with_columns,
    extract_placeholders,
    parse_template_spec,
)
from .tokenizer import SimpleNumberTokenizer

try:
    from datasets import load_dataset, Dataset as HFDataset

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

IGNORE_INDEX = -100


@dataclass
class FineTuneSample:
    instruction: str
    input_text: Optional[str]
    output: str
    source: Optional[str] = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON format in {path}")


def parse_alpaca_format(item: Dict[str, Any]) -> FineTuneSample:
    return FineTuneSample(
        instruction=item.get("instruction", ""),
        input_text=item.get("input") or None,
        output=item.get("output", ""),
    )


def parse_sharegpt_format(item: Dict[str, Any]) -> FineTuneSample:
    conversations = item.get("conversations", [])
    if len(conversations) < 2:
        raise ValueError("ShareGPT format requires at least 2 conversation turns")

    instruction_parts = []
    output = ""

    for conv in conversations:
        role = conv.get("from", conv.get("role", ""))
        content = conv.get("value", conv.get("content", ""))

        if role in ("human", "user"):
            instruction_parts.append(content)
        elif role in ("assistant", "gpt"):
            output = content
            break

    return FineTuneSample(
        instruction="\n".join(instruction_parts),
        input_text=None,
        output=output,
    )


def detect_and_parse_item(item: Dict[str, Any]) -> FineTuneSample:
    if "conversations" in item:
        return parse_sharegpt_format(item)
    elif "instruction" in item:
        return parse_alpaca_format(item)
    elif "text" in item:
        parts = item["text"].split("\n\n")
        if len(parts) >= 2:
            return FineTuneSample(
                instruction=parts[0],
                input_text=None,
                output=parts[-1],
            )
        return FineTuneSample(
            instruction=item["text"],
            input_text=None,
            output="",
        )
    else:
        raise ValueError(f"Unknown data format: {list(item.keys())}")


class FineTuningDataset:
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        template: Union[str, PromptTemplate] = "alpaca",
        template_path: Optional[str] = None,
        max_seq_length: int = 512,
        pad_token_id: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        if template_path:
            self.template = load_custom_template(template_path)
        elif isinstance(template, str):
            self.template = get_template(template)
        else:
            self.template = template

        if os.path.isdir(data_path):
            files = [
                f for f in os.listdir(data_path) if f.endswith((".json", ".jsonl"))
            ]
            if not files:
                raise ValueError(f"No JSON/JSONL files found in {data_path}")
            data_path = os.path.join(data_path, files[0])

        if data_path.endswith(".jsonl"):
            self.raw_data = load_jsonl(data_path)
        else:
            self.raw_data = load_json(data_path)

        self.samples = [detect_and_parse_item(item) for item in self.raw_data]
        self.processed_data = self._process_all()

        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def _process_all(self) -> List[Dict[str, np.ndarray]]:
        processed = []
        for sample in self.samples:
            item = self._process_sample(sample)
            processed.append(item)
        return processed

    def _process_sample(self, sample: FineTuneSample) -> Dict[str, np.ndarray]:
        formatted_text = self.template.format(
            instruction=sample.instruction,
            input_text=sample.input_text,
            output=sample.output,
        )

        response_start = self.template.get_response_start()
        prompt_text = self.template.format(
            instruction=sample.instruction,
            input_text=sample.input_text,
            output=None,
        )

        if hasattr(self.tokenizer, "__call__"):
            tokens = self.tokenizer(
                formatted_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            input_ids = tokens["input_ids"]
            if (
                isinstance(input_ids, list)
                and len(input_ids) > 0
                and isinstance(input_ids[0], list)
            ):
                input_ids = input_ids[0]

            prompt_tokens = self.tokenizer(
                prompt_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            prompt_ids = prompt_tokens["input_ids"]
            if (
                isinstance(prompt_ids, list)
                and len(prompt_ids) > 0
                and isinstance(prompt_ids[0], list)
            ):
                prompt_ids = prompt_ids[0]
        else:
            input_ids = self.tokenizer.encode(formatted_text)
            prompt_ids = self.tokenizer.encode(prompt_text)

        prompt_length = len(prompt_ids)

        labels = [IGNORE_INDEX] * len(input_ids)
        for i in range(prompt_length, len(input_ids)):
            labels[i] = input_ids[i]

        if len(input_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            labels = labels + [IGNORE_INDEX] * padding_length
        else:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.processed_data[idx]

    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        indices = np.random.choice(len(self), size=batch_size, replace=True)

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for idx in indices:
            item = self.processed_data[idx]
            batch_input_ids.append(item["input_ids"])
            batch_labels.append(item["labels"])
            batch_attention_mask.append(item["attention_mask"])

        return {
            "input_ids": np.stack(batch_input_ids),
            "labels": np.stack(batch_labels),
            "attention_mask": np.stack(batch_attention_mask),
        }


class StreamingFineTuningDataset:
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        template: Union[str, PromptTemplate] = "alpaca",
        template_path: Optional[str] = None,
        max_seq_length: int = 512,
        pad_token_id: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        if template_path:
            self.template = load_custom_template(template_path)
        elif isinstance(template, str):
            self.template = get_template(template)
        else:
            self.template = template

        self.data_path = data_path
        self._iterator = None

        if os.path.isdir(data_path):
            self.data_files = sorted(
                [
                    os.path.join(data_path, f)
                    for f in os.listdir(data_path)
                    if f.endswith((".json", ".jsonl"))
                ]
            )
        else:
            self.data_files = [data_path]

        if not self.data_files:
            raise ValueError(f"No data files found at {data_path}")

        print(f"Streaming from {len(self.data_files)} file(s)")

    def _iter_file(self, filepath: str):
        if filepath.endswith(".jsonl"):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            data = load_json(filepath)
            for item in data:
                yield item

    def _iter_all(self):
        while True:
            for filepath in self.data_files:
                for item in self._iter_file(filepath):
                    yield item

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, np.ndarray]:
        sample = detect_and_parse_item(item)

        formatted_text = self.template.format(
            instruction=sample.instruction,
            input_text=sample.input_text,
            output=sample.output,
        )

        prompt_text = self.template.format(
            instruction=sample.instruction,
            input_text=sample.input_text,
            output=None,
        )

        if hasattr(self.tokenizer, "__call__"):
            tokens = self.tokenizer(
                formatted_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            input_ids = tokens["input_ids"]
            if (
                isinstance(input_ids, list)
                and len(input_ids) > 0
                and isinstance(input_ids[0], list)
            ):
                input_ids = input_ids[0]

            prompt_tokens = self.tokenizer(
                prompt_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            prompt_ids = prompt_tokens["input_ids"]
            if (
                isinstance(prompt_ids, list)
                and len(prompt_ids) > 0
                and isinstance(prompt_ids[0], list)
            ):
                prompt_ids = prompt_ids[0]
        else:
            input_ids = self.tokenizer.encode(formatted_text)
            prompt_ids = self.tokenizer.encode(prompt_text)

        prompt_length = len(prompt_ids)
        labels = [IGNORE_INDEX] * len(input_ids)
        for i in range(prompt_length, len(input_ids)):
            labels[i] = input_ids[i]

        if len(input_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            labels = labels + [IGNORE_INDEX] * padding_length
        else:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }

    def get_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self._iterator is None:
            self._iterator = self._iter_all()

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for _ in range(batch_size):
            try:
                item = next(self._iterator)
                processed = self._process_item(item)
                batch_input_ids.append(processed["input_ids"])
                batch_labels.append(processed["labels"])
                batch_attention_mask.append(processed["attention_mask"])
            except StopIteration:
                self._iterator = self._iter_all()
                item = next(self._iterator)
                processed = self._process_item(item)
                batch_input_ids.append(processed["input_ids"])
                batch_labels.append(processed["labels"])
                batch_attention_mask.append(processed["attention_mask"])

        return {
            "input_ids": np.stack(batch_input_ids),
            "labels": np.stack(batch_labels),
            "attention_mask": np.stack(batch_attention_mask),
        }


class HFDatasetLoader:
    """Load datasets directly from HuggingFace Hub with streaming support.

    Supports flexible template parsing:
    - Built-in templates: "alpaca", "chatml", etc.
    - Custom template strings: "Q: {question}\\nA: {answer}"
    - Auto-detection from dataset columns
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: Any,
        template: Union[str, PromptTemplate, FlexibleTemplate] = "alpaca",
        template_path: Optional[str] = None,
        max_seq_length: int = 512,
        pad_token_id: int = 0,
        text_field: Optional[str] = None,
        train_split: str = "train",
        validation_split: Optional[str] = None,
        streaming: bool = True,
        dataset_config: Optional[str] = None,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        conversations_field: str = "conversations",
    ):
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "The 'datasets' library is required for HuggingFace dataset loading. "
                "Install it with: pip install datasets"
            )

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.text_field = text_field
        self.train_split = train_split
        self.validation_split = validation_split
        self.streaming = streaming
        self.dataset_config = dataset_config
        self.instruction_field = instruction_field
        self.input_field = input_field
        self.output_field = output_field
        self.conversations_field = conversations_field

        if template_path:
            self.template = load_custom_template(template_path)
        elif isinstance(template, str):
            template_type, template_content = parse_template_spec(template)
            if template_type == "builtin":
                self.template = get_template(template_content)
            elif template_type == "custom":
                self.template = FlexibleTemplate(template_content)
            else:
                self.template = get_template("alpaca")
        else:
            self.template = template

        self._train_dataset = None
        self._val_dataset = None
        self._train_iterator = None
        self._val_iterator = None
        self._column_mapping: Optional[Dict[str, str]] = None
        self._template_detected = False

        self._load_datasets()

    def _load_datasets(self):
        """Load train and validation datasets from HuggingFace Hub."""
        print(f"Loading dataset '{self.dataset_name}' from HuggingFace Hub...")

        load_kwargs = {"path": self.dataset_name}
        if self.dataset_config:
            load_kwargs["name"] = self.dataset_config

        if self.streaming:
            load_kwargs["streaming"] = True

        self._train_dataset = load_dataset(**load_kwargs, split=self.train_split)
        print(f"  Train split: {self.train_split}")

        if self.validation_split:
            try:
                self._val_dataset = load_dataset(
                    **load_kwargs, split=self.validation_split
                )
                print(f"  Validation split: {self.validation_split}")
            except Exception as e:
                print(f"  Warning: Could not load validation split: {e}")
                self._val_dataset = None

    def _detect_format(self, sample: Dict[str, Any]) -> str:
        """Detect the dataset format from a sample."""
        if self.conversations_field in sample:
            return "sharegpt"
        elif self.instruction_field in sample:
            return "alpaca"
        elif self.text_field and self.text_field in sample:
            return "text"
        elif "text" in sample:
            return "text"
        elif "prompt" in sample and "completion" in sample:
            return "prompt_completion"
        else:
            return "unknown"

    def _parse_sample(self, sample: Dict[str, Any]) -> FineTuneSample:
        """Parse a sample from HF dataset into FineTuneSample."""
        format_type = self._detect_format(sample)

        if format_type == "sharegpt":
            conversations = sample.get(self.conversations_field, [])
            instruction_parts = []
            output = ""

            for conv in conversations:
                role = conv.get("from", conv.get("role", ""))
                content = conv.get("value", conv.get("content", ""))

                if role in ("human", "user"):
                    instruction_parts.append(content)
                elif role in ("assistant", "gpt"):
                    output = content
                    break

            return FineTuneSample(
                instruction="\n".join(instruction_parts),
                input_text=None,
                output=output,
                source=f"hf:{self.dataset_name}",
            )

        elif format_type == "alpaca":
            return FineTuneSample(
                instruction=sample.get(self.instruction_field, ""),
                input_text=sample.get(self.input_field) or None,
                output=sample.get(self.output_field, ""),
                source=f"hf:{self.dataset_name}",
            )

        elif format_type == "text":
            text_field = self.text_field or "text"
            text = sample.get(text_field, "")
            parts = text.split("\n\n")

            if len(parts) >= 2:
                return FineTuneSample(
                    instruction=parts[0],
                    input_text=None,
                    output=parts[-1],
                    source=f"hf:{self.dataset_name}",
                )
            return FineTuneSample(
                instruction=text,
                input_text=None,
                output="",
                source=f"hf:{self.dataset_name}",
            )

        elif format_type == "prompt_completion":
            return FineTuneSample(
                instruction=sample.get("prompt", ""),
                input_text=None,
                output=sample.get("completion", ""),
                source=f"hf:{self.dataset_name}",
            )

        else:
            raise ValueError(
                f"Unknown dataset format. Keys: {list(sample.keys())}. "
                f"Specify --dataset_text_field or --instruction_field/--output_field"
            )

    def _tokenize_sample(self, sample: FineTuneSample) -> Dict[str, np.ndarray]:
        """Tokenize a sample with proper label masking."""
        formatted_text = self.template.format(
            instruction=sample.instruction,
            input_text=sample.input_text,
            output=sample.output,
        )

        prompt_text = self.template.format(
            instruction=sample.instruction,
            input_text=sample.input_text,
            output=None,
        )

        if hasattr(self.tokenizer, "__call__"):
            tokens = self.tokenizer(
                formatted_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            input_ids = tokens["input_ids"]
            if (
                isinstance(input_ids, list)
                and len(input_ids) > 0
                and isinstance(input_ids[0], list)
            ):
                input_ids = input_ids[0]

            prompt_tokens = self.tokenizer(
                prompt_text,
                max_length=self.max_seq_length,
                truncation=True,
                padding=False,
                return_tensors=None,
            )
            prompt_ids = prompt_tokens["input_ids"]
            if (
                isinstance(prompt_ids, list)
                and len(prompt_ids) > 0
                and isinstance(prompt_ids[0], list)
            ):
                prompt_ids = prompt_ids[0]
        else:
            input_ids = self.tokenizer.encode(formatted_text)
            prompt_ids = self.tokenizer.encode(prompt_text)

        prompt_length = len(prompt_ids)
        labels = [IGNORE_INDEX] * len(input_ids)
        for i in range(prompt_length, len(input_ids)):
            labels[i] = input_ids[i]

        if len(input_ids) < self.max_seq_length:
            padding_length = self.max_seq_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_length
            labels = labels + [IGNORE_INDEX] * padding_length
        else:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]

        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]

        return {
            "input_ids": np.array(input_ids, dtype=np.int32),
            "labels": np.array(labels, dtype=np.int32),
            "attention_mask": np.array(attention_mask, dtype=np.int32),
        }

    def get_batch(self, batch_size: int, split: str = "train") -> Dict[str, np.ndarray]:
        """Get a batch of processed samples."""
        if split == "train":
            if self._train_iterator is None:
                self._train_iterator = iter(self._train_dataset)
            iterator = self._train_iterator
        elif split == "validation" and self._val_dataset:
            if self._val_iterator is None:
                self._val_iterator = iter(self._val_dataset)
            iterator = self._val_iterator
        else:
            raise ValueError(f"Unknown split: {split}")

        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for _ in range(batch_size):
            try:
                sample = next(iterator)
            except StopIteration:
                if self.streaming:
                    iterator = iter(
                        self._train_dataset if split == "train" else self._val_dataset
                    )
                    if split == "train":
                        self._train_iterator = iterator
                    else:
                        self._val_iterator = iterator
                    sample = next(iterator)
                else:
                    iterator = iter(
                        self._train_dataset if split == "train" else self._val_dataset
                    )
                    sample = next(iterator)

            ft_sample = self._parse_sample(sample)
            processed = self._tokenize_sample(ft_sample)
            batch_input_ids.append(processed["input_ids"])
            batch_labels.append(processed["labels"])
            batch_attention_mask.append(processed["attention_mask"])

        return {
            "input_ids": np.stack(batch_input_ids),
            "labels": np.stack(batch_labels),
            "attention_mask": np.stack(batch_attention_mask),
        }

    def get_train_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Get a batch from training split."""
        return self.get_batch(batch_size, split="train")

    def get_validation_batch(self, batch_size: int) -> Optional[Dict[str, np.ndarray]]:
        """Get a batch from validation split."""
        if self._val_dataset is None:
            return None
        return self.get_batch(batch_size, split="validation")

    @property
    def has_validation(self) -> bool:
        """Check if validation split is available."""
        return self._val_dataset is not None
