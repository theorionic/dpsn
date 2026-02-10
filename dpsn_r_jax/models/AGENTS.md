# MODELS (dpsn_r_jax/models)

**OVERVIEW**
Core neural architecture defining the DPSNR model components and its integrated reasoning loop.

**WHERE TO LOOK**
| Component | File | Role |
| :--- | :--- | :--- |
| **Integrated Model** | `dpsnr.py` | Main entry point; orchestrates encoding, reasoning, and decoding. |
| **Reasoning Loop** | `dpsnr.py` | Implements the dynamic reasoning loop logic with halting. |
| **TinyController** | `controller.py` | Transformer encoder/decoder; handles token embedding and positional encoding. |
| **Memory Pool** | `memory.py` | `HierarchicalMassivePool` for knowledge retrieval and `RetrievalRouter` for dynamic K selection. |
| **Adaptive Compute** | `reasoning.py` | `AdaptiveComputeController` (ACT); manages halting and state accumulation. |
| **Neural Layers** | `layers.py` | Building blocks: `FlashCausalSelfAttention`, `TinyFFN`, and `TinyTransformerLayer`. |

**CONVENTIONS**
- **Architecture**: Modular design; each component is an independent `nn.Module`.
- **Initialization**: Prefers `setup()` for complex modules (e.g., `DPSNR`, `TinyController`) and `@nn.compact` for simpler layers.
- **Control Flow**: `deterministic` flag is mandatory for any module utilizing dropout.
- **Computation**: High performance `jnp.einsum` used for multi-head attention and retrieval math.
- **Static Shapes**: K-retrieval logic in `memory.py` and reasoning steps in `dpsnr.py` are designed for XLA/JIT compatibility.

**ANTI-PATTERNS**
- **Non-Functional State**: Never use global or class-level mutable state; stay within Flax `nn.Module` patterns.
- **Hardcoded Config**: Dimensions must be pulled from `DPSNRConfig` or `PoolConfig`, never hardcoded in methods.
- **Implicit Masks**: Avoid manual mask generation; use `nn.make_causal_mask` or standard Flax masking utilities.
- **Circular Imports**: Do not import from `training/` or `utils/` within `models/`.
