# PROJECT KNOWLEDGE BASE (dpsn-r-jax)

**Generated:** 2026-02-09
**Status:** Refactored Modular Structure

## OVERVIEW
JAX/Flax implementation of Dynamic Parameter Selection Network with Reasoning (DPSNR). Features a TinyController, HierarchicalMassivePool, and AdaptiveComputeController for dynamic reasoning loops.

## STRUCTURE
```
dpsn_r_jax/
├── models/      # Core neural components (layers, memory, reasoning)
├── data/        # Synthetic dataset generation & tokenization
├── training/    # JIT-compiled training loops & TrainState
└── utils/       # Text generation & inference helpers
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Model Architecture | `dpsn_r_jax/models/` | [AGENTS.md] Start with `dpsnr.py` |
| Training Logic | `dpsn_r_jax/training/` | `trainer.py` contains `train_step` |
| Data Loading | `dpsn_r_jax/data/` | [AGENTS.md] Synthetic dataset logic |
| Config | `dpsn_r_jax/config.py` | `@dataclass` definitions |
| Entry Point | `main.py` | CLI & orchestration |

## CODE MAP
| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `DPSNR` | class | `models/dpsnr.py` | Main integrated model |
| `TinyController` | class | `models/controller.py` | Transformer-based controller |
| `HierarchicalMassivePool` | class | `models/memory.py` | Knowledge retrieval system |
| `AdaptiveComputeController` | class | `models/reasoning.py` | Dynamic computation loop |
| `train_step` | function | `training/trainer.py` | Pure JIT-compiled optimization |

## CONVENTIONS
- **Types**: Mandatory `typing` hints for all functions.
- **Config**: Use `@dataclass` for all hyperparameters.
- **Pure Functions**: `train_step` must be pure and JIT-compilable.
- **RNG**: Explicitly pass `deterministic` and handle RNG splitting.

## ANTI-PATTERNS (THIS PROJECT)
- **Hyphenated Imports**: Do NOT use `dpsn-r_jax`. Use `dpsn_r_jax`.
- **Single File**: Do NOT revert to the legacy single-file implementation.
- **Mutable State**: Avoid non-Flax state management in Modules.

## COMMANDS
```bash
# Standard training
python main.py --epochs 3 --batch_size 8 --dataset_size 500

# Tiny verification (runs in <1m)
python main.py --tiny --epochs 1 --dataset_size 10 --max_steps 5
```

## NOTES
- The project was recently refactored; `dpsn-r_jax.py` is deprecated.
- **Testing**: No `/tests` directory. Use `--tiny` flag for ad-hoc verification.
- Uses `jax.lax.scan` patterns for internal reasoning loops.

