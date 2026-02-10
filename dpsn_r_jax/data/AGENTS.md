# PROJECT KNOWLEDGE BASE (dpsn_r_jax/data)

**Generated:** 2026-02-09
**Status:** Data & Tokenization Layer

## OVERVIEW
Handles synthetic reasoning data generation, HuggingFace dataset streaming, and specialized numeric tokenization.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Synthetic Data | `dataset.py` | `SyntheticReasoningDataset` for sorting tasks |
| HF Integration | `dataset.py` | `HFStreamingDataset` for generic text streams |
| Numeric Tokenizer | `tokenizer.py` | `SimpleNumberTokenizer` (integer-to-ID mapping) |
| Tokenizer Factory | `tokenizer.py` | `get_tokenizer` (Numeric vs. AutoTokenizer) |

## CONVENTIONS
- **Formats**: Synthetic tasks follow `Task: {Input} -> {Output}` string patterns.
- **Vocab**: `SimpleNumberTokenizer` vocab is `[0, max_val)` + 4 special tokens.
- **Special IDs**: PAD=`max_val`, EOS=`max_val+1`, SEP=`max_val+2`, UNK=`max_val+3`.
- **Tensors**: Dataset methods return `numpy.ndarray` (converted to JAX later).
