"""Multi-type data preprocessor for DPSN training.

Handles web text, code, chat/instruction, and math data with a unified
interface. Each dataset in the YAML config can declare a ``type`` field;
the corresponding formatter is applied during text extraction inside
ChunkedHFDataset, before tokenization.

Supported types
---------------
text  — raw text extraction (default column-scan behaviour)
code  — prepends a language tag: ``# python\\n{code}``
chat  — formats messages/instruction/output into a chat template
math  — formats problem/solution or question/answer pairs
auto  — sniffs column names and dispatches to the right formatter (default)

Typical YAML usage
------------------
datasets:
  - name: "openbmb/Ultra-FineWeb"
    text_column: "content"
    ratio: 0.7
    # type omitted → defaults to "auto" (works fine for plain web text)

  - name: "bigcode/the-stack-dedup"
    text_column: "content"
    ratio: 0.2
    type: code
    lang_column: "lang"      # prepends "# python\\n" etc.

  - name: "HuggingFaceH4/ultrachat_200k"
    split: "train_sft"
    ratio: 0.1
    type: chat               # formats messages[] or instruction/output

  - name: "lighteval/MATH"
    ratio: 0.05
    type: math               # formats problem/solution pairs
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional


# ── Per-type formatters ────────────────────────────────────────────────────────

def _format_text(item: dict, text_columns: List[str]) -> str:
    """Standard web-text extraction: try each column, fall back to first string."""
    for col in text_columns:
        val = item.get(col)
        if val and isinstance(val, str):
            return val
    for v in item.values():
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _format_code(
    item: dict,
    lang_column: Optional[str] = None,
    text_columns: Optional[List[str]] = None,
) -> str:
    """Code extraction with optional language prefix.

    Output: ``# {lang}\\n{code}`` when a language is available,
    plain ``{code}`` otherwise.
    """
    cols = text_columns or ["content", "code", "text", "source"]
    code = ""
    for col in cols:
        val = item.get(col)
        if val and isinstance(val, str):
            code = val
            break
    if not code:
        return ""

    lang = ""
    if lang_column:
        lang = item.get(lang_column, "") or ""
    if not lang:
        lang = (
            item.get("language")
            or item.get("lang")
            or item.get("ext")
            or item.get("programming_language")
            or ""
        )

    if lang:
        return f"# {str(lang).lower().strip()}\n{code}"
    return code


def _format_chat(item: dict) -> str:
    """Chat / instruction formatting.

    Supports three common schemas:

    1. OpenAI ``messages`` list — ``[{"role": "user", "content": "..."}]``
    2. ShareGPT ``conversations`` list — ``[{"from": "human", "value": "..."}]``
    3. Alpaca ``instruction`` + optional ``input`` + ``output``
    """
    # ── OpenAI / HF chat-template format ──────────────────────────────────────
    messages = item.get("messages") or item.get("conversation")
    if isinstance(messages, list) and messages:
        parts = []
        for msg in messages:
            role = str(msg.get("role", msg.get("from", "user"))).lower()
            content = msg.get("content", msg.get("value", ""))
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role in ("user", "human"):
                parts.append(f"<|user|>\n{content}")
            else:  # assistant / gpt / bot
                parts.append(f"<|assistant|>\n{content}")
        return "\n".join(parts)

    # ── ShareGPT conversations ─────────────────────────────────────────────────
    convs = item.get("conversations")
    if isinstance(convs, list) and convs:
        parts = []
        for turn in convs:
            speaker = str(turn.get("from", "human")).lower()
            value = turn.get("value", "")
            tag = "<|assistant|>" if speaker in ("gpt", "assistant", "bot") else "<|user|>"
            parts.append(f"{tag}\n{value}")
        return "\n".join(parts)

    # ── Alpaca instruction / output ────────────────────────────────────────────
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    output = item.get("output", "")
    if instruction:
        prompt = f"<|user|>\n{instruction}"
        if inp:
            prompt += f"\n{inp}"
        if output:
            prompt += f"\n<|assistant|>\n{output}"
        return prompt

    return _format_text(item, ["text", "content"])


def _format_math(item: dict) -> str:
    """Math problem/solution formatting.

    Tries ``problem``/``solution``, ``question``/``answer``, and
    ``query``/``response`` column pairs.
    """
    problem = (
        item.get("problem")
        or item.get("question")
        or item.get("query")
        or item.get("text")
        or ""
    )
    solution = (
        item.get("solution")
        or item.get("answer")
        or item.get("response")
        or ""
    )
    if problem and solution:
        return f"Problem: {problem}\nSolution: {solution}"
    if problem:
        return f"Problem: {problem}"
    if solution:
        return solution
    return _format_text(item, ["text", "content"])


def _auto_format(item: dict, text_columns: List[str]) -> str:
    """Sniff item keys and dispatch to the right formatter."""
    keys = set(item.keys())
    if "messages" in keys or "conversation" in keys or "conversations" in keys:
        return _format_chat(item)
    if "instruction" in keys and "output" in keys:
        return _format_chat(item)
    if ("problem" in keys or "question" in keys) and (
        "solution" in keys or "answer" in keys
    ):
        return _format_math(item)
    if "code" in keys or "source" in keys:
        # Use default code columns so "code"/"source" keys are found regardless
        # of what text_columns was set to (which is tuned for plain text).
        return _format_code(item, text_columns=["content", "code", "text", "source"])
    return _format_text(item, text_columns)


# ── Public factory ─────────────────────────────────────────────────────────────

def make_text_fn(
    data_type: str = "auto",
    text_columns: Optional[List[str]] = None,
    lang_column: Optional[str] = None,
) -> Callable[[dict], str]:
    """Return a text-extraction callable for the given data type.

    Args:
        data_type:    One of ``"text"``, ``"code"``, ``"chat"``, ``"math"``,
                      ``"auto"`` (default).
        text_columns: Column names to try for text/code/auto types.
        lang_column:  Column holding the programming language (code type only).

    Returns:
        A callable ``(item: dict) -> str`` for use as ``ChunkedHFDataset.text_fn``.
    """
    cols = text_columns or ["text", "content", "sentence"]
    dtype = (data_type or "auto").lower().strip()

    if dtype == "text":
        return lambda item: _format_text(item, cols)
    elif dtype == "code":
        return lambda item: _format_code(item, lang_column=lang_column, text_columns=cols)
    elif dtype == "chat":
        return _format_chat
    elif dtype == "math":
        return _format_math
    else:  # auto
        return lambda item: _auto_format(item, cols)


def build_mixed_dataset(
    yaml_datasets: List[Dict],
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    chunk_size: int = 10_000,
    num_workers: int = 4,
    hf_state: Optional[dict] = None,
    seed: int = 42,
):
    """Build a MixedDataset (or single ChunkedHFDataset) from YAML dataset configs.

    Each entry in ``yaml_datasets`` supports:

    =========== ============================================================
    Key         Description
    =========== ============================================================
    name        HuggingFace dataset path (required)
    subset      Dataset config / subset name
    split       Dataset split (default ``"train"``)
    text_column Column(s) for text extraction — str or list[str]
    ratio       Mixing ratio (default 1.0)
    type        Data type: text | code | chat | math | auto (default auto)
    lang_column Column holding programming language (code type only)
    =========== ============================================================

    Args:
        yaml_datasets:  List of dataset config dicts from the YAML ``data.datasets`` section.
        tokenizer_name: Tokenizer identifier (HF model name or ``"numeric"``).
        seq_len:        Token sequence length.
        batch_size:     Batch size per ``get_batch()`` call.
        chunk_size:     Rows per chunk for each ``ChunkedHFDataset``.
        num_workers:    Parallel tokenization workers per dataset.
        hf_state:       Optional HF iterator state for the first dataset (resume support).
        seed:           Random seed for ``MixedDataset`` sampling.

    Returns:
        A ``MixedDataset`` when multiple sources are configured, or a bare
        ``ChunkedHFDataset`` when only one source is present.
    """
    from dpsn_r_jax.data.dataset import ChunkedHFDataset
    from dpsn_r_jax.data.mixed_dataset import MixedDataset

    ds_list = []
    ratio_list = []

    for i, ds_cfg in enumerate(yaml_datasets):
        name = ds_cfg["name"]
        subset = ds_cfg.get("subset") or None
        split = ds_cfg.get("split", "train")
        ratio = float(ds_cfg.get("ratio", 1.0))
        data_type = ds_cfg.get("type", "auto")
        lang_column = ds_cfg.get("lang_column") or ds_cfg.get("lang_col") or None

        # Normalise text_columns: accept str or list
        tc = ds_cfg.get("text_column") or ds_cfg.get("text_columns") or ["text", "content"]
        if isinstance(tc, str):
            tc = [tc]

        text_fn = make_text_fn(data_type, text_columns=tc, lang_column=lang_column)

        print(f"[Preprocessor] Dataset {i+1}/{len(yaml_datasets)}: '{name}' | type={data_type} | ratio={ratio}")

        ds = ChunkedHFDataset(
            dataset_name=name,
            tokenizer_name=tokenizer_name,
            chunk_size=chunk_size,
            subset=subset,
            split=split,
            seq_len=seq_len,
            batch_size=batch_size,
            num_tokenizer_workers=num_workers,
            text_columns=tc,
            text_fn=text_fn,
            hf_state=hf_state if i == 0 else None,
        )
        ds_list.append(ds)
        ratio_list.append(ratio)

    if len(ds_list) == 1:
        return ds_list[0]

    return MixedDataset(ds_list, ratio_list, seed=seed)
