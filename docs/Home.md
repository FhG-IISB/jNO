# jNO Wiki

Welcome to the jNO documentation wiki. Use the links below to navigate.

---

## Contents

| Page | Description |
|------|-------------|
| [Getting Started](Getting-Started.md) | Installation, first example, core workflow |
| [Domain & Geometry](Domain-and-Geometry.md) | Built-in geometries, mesh loading, variable sampling, time-dependent problems |
| [Neural Network Architectures](Neural-Network-Architectures.md) | MLP, FNO, U-Net, DeepONet, Poseidon, custom models |
| [Differential Operators](Differential-Operators.md) | grad, laplacian, jacobian, hessian, divergence, curl, math functions |
| [Training](Training.md) | Solver setup, optimisers, LR schedules, weight schedules, multi-GPU, evaluation |
| [Adaptive Resampling](Adaptive-Resampling.md) | RAD, RARD, HA, CR3, PINNFluence strategies |
| [Hyperparameter Tuning](Hyperparameter-Tuning.md) | Architecture search, training sweeps, Nevergrad integration |
| [Save, Load & Configuration](Save-Load-and-Configuration.md) | Serialisation, RSA-signed saves, TOML config, logging |
| [Examples](Examples.md) | Annotated walkthrough of all example scripts |

---

## About

jNO (JAX Neural Operators) is a research-level library for solving partial differential equations (PDEs) with neural networks. It provides:

- A symbolic DSL for expressing PDE residuals, boundary conditions, and data constraints using lazy computation graphs
- A wide library of neural operator architectures (FNO, U-Net, DeepONet, Poseidon, …)
- Flexible, multi-phase training with per-model controls (freeze, LoRA, partial masks)
- Automatic and finite-difference differential operators via `jno.numpy`
- Adaptive collocation point resampling strategies
- Hyperparameter and architecture search via Nevergrad
- Multi-GPU parallelism (data, model, and hybrid)
- Encrypted (RSA-signed) serialisation

> **Warning:** This is a research-level repository. It may contain bugs and is subject to continuous change without notice.
