# Getting Started

This page is the fastest path from installation to a first PDE solve with jNO.

Before you begin, complete setup in [Installation](Installation.md).

## Recommended Path

1. Run the first example below.
2. Learn domain construction in [Domain and Geometry](Domain-and-Geometry.md).
3. Configure optimization in [Training](Training.md).
4. Control trainability in [Model Controls](Model-Controls.md).
5. Explore model families in [Foundax Core Architectures](foundation_models/Core-Architectures.md).

## Running Your First Example

```bash
cd examples
uv run python laplace1D.py
```

This solves a 1D Laplace problem with a Physics-Informed Neural Network (PINN) and writes outputs to `./runs/laplace1D/`.

## Core Workflow

Every jNO program follows this five-step pattern:

### 1. Define the Domain

```python
import jno

# 2D rectangular domain with mesh spacing 0.05
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
```

See [Domain and Geometry](Domain-and-Geometry.md) for supported geometries.

### 2. Sample Variables

```python
x, y, t = domain.variable("interior")     # interior collocation points
xb, yb, tb = domain.variable("boundary")  # boundary points
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

pde = -jnn.laplacian(u, [x, y]) - 1.0      # Laplace residual
boc = u_net(xb, yb) - 0.0                  # Dirichlet boundary condition
```

Constraints are symbolic expressions; no solver step runs yet.

### 5. Solve

```python
import optax
from jno import LearningRateSchedule as lrs

u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 2000, 1e-5))
crux = jno.core([pde.mse, boc.mse], domain)
stats = crux.solve(2000)
stats.plot("history.png")
```

## Project Setup Helper

`jno.setup()` initializes logging and returns a run directory in one call:

```python
dire = jno.setup(__file__)                    # creates ./runs/<script_name>/
dire = jno.setup(__file__, name="experiment")  # custom name
```

## Understanding Output

During training, jNO prints progress per epoch:

```text
Epoch  1000/2000| L: 1.2345e-03 | C0: 1.1000e-03 | C1: 1.3500e-04
```

- `L` is total weighted loss.
- `C0`, `C1` are per-constraint losses.
- `T0`, `T1` are tracker values when trackers are enabled.

## Documentation Map

| Topic | Page |
|------|------|
| Installation methods and Docker | [Installation](Installation.md) |
| Geometry and variable sampling | [Domain and Geometry](Domain-and-Geometry.md) |
| Symbolic operators and math helpers | [Differential Operators](Differential-Operators.md) |
| Training schedules and solver setup | [Training](Training.md) |
| Freeze, masks, LoRA, initialization | [Model Controls](Model-Controls.md) |
| Residual-adaptive sampling | [Adaptive Resampling](Adaptive-Resampling.md) |
| Hyperparameter search | [Hyperparameter Tuning](Hyperparameter-Tuning.md) |
| Save, load, and reproducibility | [Save, Load and Configuration](Save-Load-and-Configuration.md) |
| Architecture families | [Foundax Core Architectures](foundation_models/Core-Architectures.md) |
| Foundation-model families | [Foundation Models](foundation_models/Foundation-Models.md) |
