import json
import os
from typing import Optional, Dict, Any, List, Union
import numpy as np
from dataclasses import dataclass

from .templates import get_template, load_custom_template, PromptTemplate
from .tokenizer import SimpleNumberTokenizer


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
