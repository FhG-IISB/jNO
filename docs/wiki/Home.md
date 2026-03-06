# jNO — JAX-Native Neural Operators

**jNO** is a Python library that provides JAX-native building blocks for training neural operators. It offers a high-level, NumPy-like API for defining PDEs, building neural network models, and solving forward and inverse problems using physics-informed or data-driven approaches.

## Key Features

- **Symbolic PDE definition** — write equations with `jno.numpy` using familiar math syntax (`sin`, `grad`, `laplacian`, …)
- **20+ neural operator architectures** — FNO, U-Net, DeepONet, GeoFNO, PCNO, PiT, GNOT, ScOT, Poseidon, and more
- **Automatic differentiation** — seamlessly compute gradients, Laplacians, Hessians, and Jacobians through neural networks
- **Multi-GPU support** — built-in data and model parallelism via JAX device meshes
- **Architecture search** — automated hyperparameter and architecture tuning with Nevergrad
- **LoRA fine-tuning** — parameter-efficient fine-tuning of trained models
- **Operator learning** — batched training across multiple input functions
- **Flexible meshing** — pygmsh-based geometry construction or external mesh import

## Quick Start

```python
import jno
import jno.numpy as jnn
import optax

# 1. Define the domain
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y = domain.variable("interior")

# 2. Define a neural network
u = jnn.nn.mlp(hidden_dims=64, num_layers=2)(x, y) * x * (1 - x) * y * (1 - y)

# 3. Write the PDE
pde = -jnn.laplacian(u, [x, y]) - 1.0

# 4. Solve
crux = jno.core([pde], domain)
crux.solve(10_000, optax.adam(1), jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5))
```

## Wiki Contents

| Page | Description |
|------|-------------|
| **[Installation](Installation.md)** | How to install jNO and its dependencies |
| **[Architecture Guide](Architecture-Guide.md)** | All supported neural operator architectures |
| **[Domain and Meshing](Domain-and-Meshing.md)** | Domain definition, geometry, and mesh construction |
| **[Training and Solving](Training-and-Solving.md)** | The core solver, optimizers, and learning rate schedules |
| **[jno.numpy API](jno.numpy-API.md)** | Math functions and differential operators |
| **[Operator Learning](Operator-Learning.md)** | Operator learning workflows with tensor data |
| **[Custom Models](Custom-Models.md)** | Wrapping Flax Linen/NNX modules and KAN networks |
| **[LoRA and Fine-Tuning](LoRA-and-Fine-Tuning.md)** | Parameter-efficient fine-tuning with LoRA |
| **[Architecture Search](Architecture-Search.md)** | Hyperparameter and architecture tuning |
| **[Logging and Callbacks](Logging-and-Callbacks.md)** | Logger setup, callbacks, and debugging utilities |
| **[Parallelism](Parallelism.md)** | Multi-GPU data and model parallelism |
| **[Examples](Examples.md)** | Walkthrough of included example scripts |
| **[FAQ](FAQ.md)** | Frequently asked questions and troubleshooting |

## Dependencies

jNO is built on top of these open-source libraries:

- [JAX](https://github.com/jax-ml/jax) — The backbone for array computing and automatic differentiation
- [Flax](https://github.com/google/flax) — Neural network library
- [Optax](https://github.com/google-deepmind/optax) — Gradient-based optimizers
- [SOAP (JAX)](https://github.com/haydn-jones/SOAP_JAX) — Second-order optimizer
- [lox](https://github.com/huterguier/lox) — Logging utility
- [jax-tqdm](https://github.com/jeremiecoullon/jax-tqdm) — Progress bar
- [pygmsh](https://github.com/nschloe/pygmsh) — Mesh generation
- [Nevergrad](https://github.com/facebookresearch/nevergrad) — Black-box optimization for architecture search
