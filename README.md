# DPSN-R JAX Implementation

A JAX/Flax implementation of a Dynamic Parameter Selection Network with Reasoning (DPSNR).

## Project Structure

The project has been refactored into a modular structure:

```
dpsn_r_jax/
├── __init__.py
├── config.py           # Configuration classes (DPSNRConfig, PoolConfig)
├── models/             # Model components
│   ├── __init__.py
│   ├── layers.py       # Basic layers (Attention, FFN)
│   ├── controller.py   # TinyController
│   ├── memory.py       # HierarchicalMassivePool, RetrievalRouter
│   ├── reasoning.py    # AdaptiveComputeController
│   └── dpsnr.py        # Main DPSNR model
├── data/               # Data loading and generation
│   ├── __init__.py
│   └── dataset.py      # SyntheticReasoningDataset
├── training/           # Training logic
│   ├── __init__.py
│   └── trainer.py      # TrainState and training step
└── utils/              # Utilities
    ├── __init__.py
    └── generation.py   # Text generation
```

## Setup

1.  Install dependencies:
    ```bash
    pip install jax jaxlib flax optax transformers numpy
    ```

## Usage

Run the main training script:

```bash
python main.py --epochs 3 --batch_size 8 --dataset_size 500
```

## Legacy Code

The original single-file implementation is available in the `legacy/` directory for reference.
