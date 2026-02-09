# AGENTS.md - Context & Guidelines for AI Agents

## 1. Project Overview
**Name**: dpsn-r-jax
**Description**: A JAX/Flax implementation of a Dynamic Parameter Selection Network with Reasoning (DPSNR). The model features a TinyController (Transformer), HierarchicalMassivePool (Memory/Knowledge), and AdaptiveComputeController (Reasoning Loop).

## 2. Environment & Commands

### Setup
- **Python Version**: >= 3.11
- **Key Dependencies**: `jax`, `jaxlib`, `flax`, `optax`, `transformers`, `numpy`
- **Installation**:
  ```bash
  pip install -e .
  # OR manually
  pip install jax flax optax transformers numpy
  ```

### Running
- **Main Training & Verification**:
  ```bash
  python dpsn-r_jax.py --epochs 3 --batch_size 8 --dataset_size 500
  ```
- **Entry Point**: `main.py` is a simple placeholder. The core logic resides in `dpsn-r_jax.py`.

### Testing
- **Current State**: No explicit `tests/` directory. `dpsn-r_jax.py` includes a `main()` function that runs a synthetic dataset training loop and verification prompts.
- **Agent Action**: If asked to test, run `python dpsn-r_jax.py` with small parameters to verify functionality quickly:
  ```bash
  python dpsn-r_jax.py --epochs 1 --batch_size 4 --dataset_size 20
  ```

## 3. Code Style & Conventions

### Imports
- **Grouping**: JAX/Flax ecosystem first, then standard libraries.
- **Aliases**:
  ```python
  import jax
  import jax.numpy as jnp
  from jax import random, lax
  import flax.linen as nn
  from flax.training import train_state
  import optax
  ```

### Typing & Configuration
- **Type Hints**: EXTENSIVE use of `typing` (List, Dict, Tuple, Optional, Any, Callable) is mandatory.
- **Config**: Use `@dataclass` for all configuration objects (e.g., `DPSNRConfig`, `PoolConfig`).
  ```python
  @dataclass
  class MyConfig:
      hidden_dim: int = 512
  ```

### JAX/Flax Patterns
- **Module Structure**: Use `flax.linen.Module`. Define submodules in `setup()`. Use `@nn.compact` only for simple layers where explicit `setup` adds boilerplate.
- **RNG Handling**: Explicitly pass `deterministic=True/False` flags. Handle RNG splitting for dropout manually in `train_step`.
- **Pure Functions**: Ensure `train_step` is pure and JIT-compilable (`@jax.jit`).
- **Looping**: Use Python loops for unrolling small, fixed numbers of iterations (like `max_reasoning_loops`). Use `jax.lax.scan` or `jax.lax.while_loop` for dynamic or large loops if necessary (though current codebase uses Python unrolling).

### Naming
- **Classes**: `CamelCase` (e.g., `FlashCausalSelfAttention`, `TinyFFN`).
- **Functions/Variables**: `snake_case` (e.g., `train_step`, `create_train_state`).
- **Tensors**: Often denote shape in comments, e.g., `# (B, T, D)`.

## 4. Architecture Specifics
- **Controller**: A standard Transformer-based LM.
- **Memory Pool**: `HierarchicalMassivePool` uses a dense retrieval mechanism.
- **Reasoning**: `AdaptiveComputeController` manages state updates and dynamic halting.
- **Integration**: The model allows for multiple "reasoning loops" before generating the final token.

## 5. Maintenance Guidelines
- **No Refactoring without Request**: Do not restructure the single-file architecture (`dpsn-r_jax.py`) into multiple files unless explicitly asked. The user likely prefers the self-contained nature of the research prototype.
- **Performance**: Keep operations vectorized (`jax.vmap` not explicitly used yet, but batching is handled). Use `jax.lax` ops for performance critical sections.
