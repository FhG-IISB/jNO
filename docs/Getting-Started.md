# Getting Started

This page walks you through installing jNO, running your first example, and understanding the core workflow.

---

## Prerequisites

- **Python 3.11 – 3.13**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended package manager)

---

## Installation

jNO uses `uv` for environment and dependency management. `uv` installs packages into a virtual environment under your user directory — no `sudo` required.

### CPU (default)

```bash
uv sync
```

### GPU / CUDA

```bash
uv sync --extra cuda
```

### Developer tools

```bash
uv sync --extra dev
```

### IREE ahead-of-time compiler

```bash
uv sync --extra iree
```

### Windows — allow script execution first

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Running Your First Example

```bash
cd examples
uv run python laplace1D.py
```

This solves the 1D Laplace equation `∂²u/∂x² = sin(πx)` with a Physics-Informed Neural Network (PINN) and saves results to `./runs/laplace1D/`.

---

## Core Workflow

Every jNO program follows this five-step pattern:

### 1. Define the Domain

```python
import jno

# 2D rectangular domain with mesh spacing 0.05
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
```

See [Domain & Geometry](Domain-and-Geometry.md) for all supported geometries.

### 2. Sample Variables

```python
x, y, t = domain.variable("interior")    # interior collocation points
xb, yb, tb = domain.variable("boundary") # boundary points
```

### 3. Define the Neural Network

```python
import jno.numpy as jnn
import jax

u_net = jnn.nn.mlp(in_features=2, hidden_dims=64, num_layers=3, key=jax.random.PRNGKey(0))
u = u_net(x, y)
```

### 4. Formulate Constraints

```python
import jno.numpy as jnn

π = jnn.pi
pde = -jnn.laplacian(u, [x, y]) - 1.0   # ∇²u + 1 = 0
boc = u(xb, yb) - 0.0                    # u=0 on boundary
```

Constraints are **symbolic expressions** — no computation happens at this stage.

### 5. Solve

```python
import optax
from jno import LearningRateSchedule as lrs

u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
crux = jno.core([pde.mse, boc.mse], domain)
stats = crux.solve(2000)
stats.plot("history.png")
```

---

## Project Setup Helper

`jno.setup()` initialises logging and returns the run directory in one call:

```python
dire = jno.setup(__file__)   # creates ./runs/<script_name>/
dire = jno.setup(__file__, name="experiment_v1")  # custom name
```

---

## Understanding the Output

During training jNO prints a progress line each epoch:

```
Epoch  1000/2000| L: 1.2345e-03 | C0: 1.1000e-03 | C1: 1.3500e-04
```

- `L` — total weighted loss
- `C0`, `C1` — individual constraint losses
- `T0`, `T1` — tracker values (if any)

After training, `stats.plot("history.png")` saves the loss curves.

---

## Next Steps

| Topic | Page |
|-------|------|
| All geometry types | [Domain & Geometry](Domain-and-Geometry.md) |
| Neural operator architectures | [Neural Network Architectures](Neural-Network-Architectures.md) |
| Differential operators | [Differential Operators](Differential-Operators.md) |
| Optimizers, LR schedules, constraint weights | [Training](Training.md) |
| Residual-adaptive point selection | [Adaptive Resampling](Adaptive-Resampling.md) |
| Architecture and hyperparameter search | [Hyperparameter Tuning](Hyperparameter-Tuning.md) |
| Saving and loading solvers | [Save, Load & Configuration](Save-Load-and-Configuration.md) |
