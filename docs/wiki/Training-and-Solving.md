# Training and Solving

The `jno.core` class is the central training engine. It compiles constraints into a JAX-optimized training loop, manages parameters, and provides checkpointing and visualization.

## Basic Workflow

```python
import jno
import jno.numpy as jnn
import optax

# 1. Define domain and variables
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y = domain.variable("interior")

# 2. Define the neural network
u = jnn.nn.mlp(hidden_dims=64, num_layers=2)(x, y) * x * (1 - x) * y * (1 - y)

# 3. Define constraints (PDE residuals, boundary conditions, etc.)
pde = -jnn.laplacian(u, [x, y]) - 1.0

# 4. Create the solver
crux = jno.core([pde], domain)

# 5. Train
crux.solve(10_000, optax.adam(1), jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5))
```

---

## The `core` Constructor

```python
crux = jno.core(
    constraints=[pde, bc, ...],  # List of constraint expressions
    domain=domain,                # Domain object
    rng_seed=42,                  # Random seed for reproducibility
    mesh=(4, 1),                  # Device mesh shape (batch, model)
)
```

| Parameter | Description |
|-----------|-------------|
| `constraints` | List of expressions to minimize (PDE residuals, BCs, data terms) |
| `domain` | Domain with sampled points and optional tensor data |
| `rng_seed` | Seed for parameter initialization and stochastic operations |
| `mesh` | Tuple `(batch, model)` for parallelism — see [Parallelism](Parallelism.md) |

---

## Constraint Types

### PDE Residuals (Soft Constraints)

Any expression involving differential operators becomes a soft constraint:

```python
pde = jnn.grad(u(x, y, t), t) - 0.1 * jnn.laplacian(u(x, y, t), [x, y])
```

### Hard Constraints

Enforce boundary conditions structurally through the network:

```python
# Homogeneous Dirichlet BCs via multiplication
u = network(x, y) * x * (1 - x) * y * (1 - y)
```

### Soft Boundary Conditions

```python
xb, yb = domain.variable("boundary")
bc = u(xb, yb) - 0.0  # Dirichlet BC = 0
```

### MSE Constraints

By default, constraints use mean residual. To explicitly use mean squared error:

```python
crux = jno.core([pde.mse, bc.mse], domain)
```

### Trackers (Non-Loss Monitoring)

Track quantities during training without contributing to the loss:

```python
val = jnn.tracker(jnn.mean(u(x, y) - u_exact(x, y)), interval=100)
crux = jno.core([pde, val], domain)
```

---

## The `solve` Method

```python
stats = crux.solve(
    epochs,              # Number of training epochs
    optimizer,           # Optax optimizer (or constructor)
    learning_rate=...,   # LearningRateSchedule
    weights=...,         # WeightSchedule for constraint balancing
    lora=...,            # Optional LoRA rank dict for fine-tuning
)
stats.plot("training_history.png")
```

### Chaining Training Phases

Each `solve()` call returns a statistics object. You can chain multiple phases:

```python
# Phase 1: Adam with warmup
crux.solve(4000, optax.adam(1),
    jno.schedule.learning_rate.warmup_cosine(4000, 500, 1e-3, 1e-4),
    jno.schedule.constraint([1.0, 1.0, 10.0])
).plot("phase1.png")

# Phase 2: SOAP optimizer
from soap_jax import soap
crux.solve(1000, soap(1, precondition_frequency=13),
    jno.schedule.learning_rate(lambda e, _: 1e-4 * (5e-5 / 1e-4) ** (e / 1000))
).plot("phase2.png")

# Phase 3: L-BFGS refinement
crux.solve(1000, optax.lbfgs,
    jno.schedule.learning_rate(5e-5)
).plot("phase3.png")
```

### Gradient Clipping

Combine optimizers with `optax.chain`:

```python
crux.solve(4000,
    optax.chain(optax.adam(1), optax.clip_by_global_norm(1e-3)),
    jno.schedule.learning_rate.warmup_cosine(4000, 500, 1e-3, 1e-4)
)
```

---

## Learning Rate Schedules

All schedules are accessed via `jno.schedule.learning_rate` (alias: `LearningRateSchedule`):

```python
from jno import LearningRateSchedule as lrs
```

| Schedule | Example |
|----------|---------|
| Constant | `lrs.constant(1e-3)` |
| Exponential decay | `lrs.exponential(lr0=1e-3, decay_rate=0.8, decay_steps=10_000, lr_end=1e-5)` |
| Cosine decay | `lrs.cosine(total_steps=10_000, lr0=1e-3, lr_end=1e-6)` |
| Warmup + cosine | `lrs.warmup_cosine(total_steps=10_000, warmup_steps=500, lr0=1e-3, lr_end=1e-6)` |
| Piecewise constant | `lrs.piecewise_constant(boundaries=[1000, 5000], values=[5e-4, 2e-4, 5e-5])` |
| Custom function | `lrs(lambda epoch, losses: 1e-3 * 0.99 ** epoch)` |

---

## Constraint Weight Schedules

Balance multiple constraints using `jno.schedule.constraint` (alias: `WeightSchedule`):

```python
from jno import WeightSchedule as ws
```

| Schedule | Example |
|----------|---------|
| Constant weights | `ws([1.0, 1.0, 10.0])` |
| Adaptive weights | `ws(lambda e, L: [1.0, 1.0, 10.0 * L[2]])` |

---

## Supported Optimizers

jNO works with any [Optax](https://optax.readthedocs.io/) optimizer:

- `optax.adam` — most common choice
- `optax.adamw` — Adam with weight decay
- `optax.lbfgs` — quasi-Newton for fine refinement
- `soap(...)` — second-order adaptive preconditioner (from [SOAP_JAX](https://github.com/haydn-jones/SOAP_JAX))

---

## Post-Training

### Plotting Predictions

```python
tst_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01), compute_mesh_connectivity=False)
crux.plot(operation=u, test_pts=tst_domain).savefig("u_pred.png", dpi=300)
crux.plot(operation=u - u_exact, test_pts=tst_domain).savefig("error.png", dpi=300)
```

### Error Metrics

Compute L1, L2, and L∞ errors:

```python
crux.errors.all(predictions=[u], references=[u_exact], test_pts=tst_domain, save_path="./results/")
```

### Inference

```python
import numpy as np
pred = crux.predict(points=test_points, operation=u, tensor_tags=test_tensor_tags)
```

### Visualizing the Computation Graph

```python
crux.visualize_trace(pde).save("trace_pde.dot")
```

The generated `.dot` files can be viewed at [edotor.net](https://edotor.net/).

---

## Checkpointing

### Save

```python
crux.save("checkpoint.pkl")
```

### Load and Continue

```python
crux = jno.core.load("checkpoint.pkl")
crux.solve(500, optax.adam(1), jno.schedule.learning_rate(1e-5))
```

Checkpoints are automatically saved after every `.solve()` call.

---

## See Also

- [Domain and Meshing](Domain-and-Meshing.md) — how to set up domains
- [Parallelism](Parallelism.md) — multi-GPU training configuration
- [LoRA and Fine-Tuning](LoRA-and-Fine-Tuning.md) — parameter-efficient fine-tuning
