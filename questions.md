# TPU Research Cloud (TRC) Proposal – Answers to Expected Review Questions

**Applicant**: [Your Full Name / Startup Name]  
**Date**: February 11, 2026  
**Project Title**: Scaling DPSNR: Disaggregating Knowledge from Logic via Dynamic Parameter Selection Networks  
**Goal**: Validate the DPSNR architecture by training a model with 100B+ effective parameters, demonstrating that disaggregating static knowledge from reasoning logic can democratize high-end LLM capabilities for consumer-grade hardware.

This document provides detailed answers to the key questions typically evaluated by the TRC review team for large-scale proposals.

## 1. What is the specific research problem or goal you are trying to solve with this model?

The primary research goal is to validate the paradigm of **Disaggregating Knowledge from Logic**. Current Large Language Models (LLMs) conflate reasoning capabilities (logic) with world knowledge (data) within the same monolithic parameter space. This results in extreme redundancy and prohibitive hardware requirements for 100B+ models.

This project introduces the **Dynamic Parameter Selection Network with Reasoning (DPSNR)**. DPSNR decouples these concerns into a compact, high-efficiency "Controller" (Logic) and a decoupled "Massive Pool" (Knowledge). By doing so, we aim to solve the scaling efficiency problem: achieving the performance of a 120B+ dense model while allowing the "Logic" component to run on consumer-grade hardware, dynamically retrieving parameters from the "Knowledge Pool" only as needed. This architecture specifically targets the bottlenecks in advanced mathematical reasoning and consistent instruction-following that currently plague even the largest open models.

## 2. Describe your model architecture in detail.

The **DPSNR** architecture is a modular, hierarchical system designed for high-compute efficiency and reasoning depth. It consists of three core components:

1.  **TinyController**: A lightweight, high-speed Transformer-based core (7B–14B range). It acts as the "CEO (Logic)" of the system, managing the sequence state and orchestrating reasoning trajectories.
2.  **Learned Indexer (Archivist)**: A new decoupled addressing model that replaces traditional routing with a learned projection into the parameter space. It treats the Massive Pool as a searchable vector space, achieving O(1) access to specialized knowledge segments and removing the "memorization pressure" from the TinyController.
3.  **Hierarchical Massive Pool (Library)**: A decoupled parameter repository (100B+ parameters). Unlike a standard MoE, this "Knowledge Library" is sharded across TPU HBM and retrieved dynamically via high-bandwidth interconnects based on the Indexer's addressing.

**Specifications:**
- **Total Effective Parameters**: ~120 billion.
- **Controller Size**: 7 billion.
- **Addressing Model**: Learned Indexer for O(1) sparse retrieval.
- **Reasoning Depth**: Dynamically adjustable per-token (up to 16 reasoning steps) via the Adaptive Compute Controller.
- **Efficiency**: Highlights O(1) access patterns and significantly reduced GPU/TPU memory pressure by offloading knowledge storage to a specialized pool.

Full architecture code in JAX/Flax is already developed and modularized, ready for pod-scale training.

## 3. What is your planned training setup and dataset?

- **Training Paradigm**: Pre-training from scratch using a specialized objective that jointly optimizes the **TinyController's** reasoning paths and the **RetrievalRouter's** selection efficiency within the **Massive Pool**. This involves a "Reasoning-Aware" pre-training phase where the model is rewarded for selecting the most relevant parameter shards for a given context.
- **Dataset**: ~15–20 trillion tokens total, with a heavy emphasis on high-signal reasoning data:
  - **Logic & Proofs**: arXiv papers, Lean/Isabelle formal proofs, and complex mathematical datasets.
  - **Code**: The Stack v2 and high-quality synthetic code execution traces.
  - **Knowledge Base**: FineWeb-Edu, Wikipedia, and curated educational corpora to populate the Massive Pool.
- **Data processing**: MinHash deduplication, exact matching, and semantic quality filtering using a JAX-accelerated pipeline.
- **Sequence length**: 4096 base, with late-stage curriculum learning extending to 32k via RoPE scaling.

## 4. Why do you need such large-scale compute (120B params), and why specifically Cloud TPUs?

The DPSNR architecture is specifically designed to leverage the unique hardware advantages of Google Cloud TPUs. While traditional architectures treat 120B parameters as a monolithic weight matrix, DPSNR requires high-bandwidth sharding of the **Massive Pool** to maintain real-time reasoning speeds.

**Cloud TPUs are essential for this project due to:**
- **High-Bandwidth Memory (HBM)**: The "Massive Pool" requires significant HBM capacity to store and quickly access 100B+ parameters. TPU v5p/Trillium's memory bandwidth is critical for the "Logic-Knowledge" swap mechanism.
- **Mesh Networking & Interconnect**: DPSNR relies on low-latency parameter sharding and retrieval across a TPU Pod. The ICI (Inter-Core Interconnect) enables the "TinyController" to query the "HierarchicalMassivePool" distributed across hundreds of chips as if it were local memory.
- **XLA & JAX Integration**: Our implementation uses advanced JAX sharding constraints (`jax.sharding.Mesh`) to optimize the data flow between the reasoning loop and the knowledge pool, a feat that is significantly more performant on TPU architectures than generic GPU clusters.
- **Dynamic Reasoning Scaling**: The `jax.lax.scan`-based adaptive compute loops are highly optimized by the XLA compiler, allowing for efficient JIT-compilation of the variable-length reasoning steps inherent to DPSNR.

## 5. What evidence do you have that this architecture works at smaller scales?

A smaller-scale proof-of-concept (~1.3B–7B parameter versions) has already been trained on Kaggle TPUs and publicly shared:

- Kaggle notebook link: [insert link here, e.g., https://www.kaggle.com/code/yourusername/small-model-poc]
- Public demo: [Hugging Face Space or Gradio link if available]
- Key results: Perplexity on C4 validation ~[value], downstream zero-shot accuracy on MMLU/ARC/Hellaswag improved by X% over baseline transformers of similar size.
- Loss curves and sample generations attached / linked.

These experiments confirm architectural stability and scaling potential.

## 6. What is your estimated compute requirement and timeline?

- Estimated FLOPs: ~3–5 × 10²⁵ (Chinchilla-optimal regime for 120B)
- Target tokens: 15–20 trillion
- Proposed config: v5p-1024 or v6 Trillium pod slice (or equivalent), ~8–12 weeks wall-clock time with efficient sharding.
- Alternative: Start with v5e/v5p-256 slice for intermediate checkpoints (30B–70B), then scale up via extension request.

## 7. How will you share the results and outputs openly with the community?

- Full model weights: Released on Hugging Face Hub under Apache 2.0 (safetensors format, sharded checkpoints)
- Code: Training/inference scripts, architecture definition on GitHub
- Documentation: Detailed README, training logs (Weights & Biases or public tensorboard), arXiv preprint or comprehensive blog post
- Demo: Public chat/inference interface via Hugging Face Spaces or self-hosted Gradio
- Timeline: Weights/code released within 1–2 months after final training completes

## 8. What framework(s) will you use, and do you have experience with large-scale distributed training on TPUs?

- **Primary framework**: JAX + Flax.
- **Hardware Abstraction**: We utilize `jax.sharding` and `jax.lax.with_sharding_constraint` for fine-grained control over how the Massive Pool is distributed across the TPU mesh.
- **Experience**: Our team has deep experience with XLA-optimized kernels and the JAX ecosystem. We have successfully implemented a modular DPSNR codebase that handles dynamic parameter retrieval and adaptive compute loops, verified on Kaggle TPU-v3 slices. We are proficient in managing TPU VMs, GCS-integrated checkpointing, and debugging distributed sharding issues (e.g., communication bottlenecks in `pjit` operations).

## 9. What potential risks or challenges do you foresee, and how will you mitigate them?

- **Retrieval Routing Collapse**: The risk of the Controller favoring a subset of shards in the Massive Pool. *Mitigation*: Implementation of load-balancing loss and "Routing Dropout" to ensure uniform exploration of the knowledge pool.
- **Inter-Chip Communication Overhead**: High-frequency retrieval could bottleneck on the interconnect. *Mitigation*: Strategic shard placement using JAX Meshes and caching frequently accessed parameters within the Controller's local memory.
- **Reasoning Loop Stability**: Deep reasoning loops can lead to vanishing/exploding activations. *Mitigation*: Adaptive compute stabilization via residual scaling and dynamic step-limiting within the `AdaptiveComputeController`.
- **Preemptions**: *Mitigation*: Frequent checkpointing to GCS every 4 hours, with an automated "Warm-Start" script that resumes training seamlessly from the latest shard state.

## 10. Who is involved in the project?

[Describe yourself/team: e.g., "Independent researcher / small startup team of 2–3 people with combined 5+ years in ML engineering and open-source contributions."]

## 11. How does this project align with Google's AI Principles?

The project prioritizes transparency (full open release), avoids harmful use cases, includes bias evaluation/mitigation steps during data curation, and contributes positively to open science.

## 12. What feedback or metrics will you provide to Google/TRC?

Detailed post-project report including:
- TPU utilization / performance metrics (MFU, throughput)
- Any bugs or feature requests for XLA/TPU software stack
- Suggestions for TRC process improvements

## 13. Any additional context or why this project has high open-research impact?

The DPSNR project aims to **democratize 100B+ parameter capabilities**. Currently, state-of-the-art reasoning models are inaccessible to most researchers because their memory requirements exceed consumer-grade hardware.

By validating the "Disaggregated Knowledge" paradigm, we prove that a user can run a 120B-class intelligence model by hosting the "TinyController" (7B) on a local GPU while streaming knowledge from a sharded "Massive Pool" (100B+) stored on cheaper, high-capacity storage or distributed across a local network. This shift from "Model as a Monolith" to "Model as a Dynamic System" is a critical step toward making frontier-level AI reproducible and accessible to the global academic community without requiring enterprise-grade server clusters for inference.

Thank you for considering this proposal. I am happy to provide additional details, code snippets, or intermediate results.

[Your Name]  
[Your Contact Email]  
[Links to GitHub / Kaggle / previous work]
