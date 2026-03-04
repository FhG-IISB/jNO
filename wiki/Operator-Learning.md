# Operator Learning

Operator learning trains a neural network to map between function spaces — for example, learning a map from a spatially varying PDE coefficient κ(x, y) to the solution u(x, y; κ). jNO supports this through batched domains and tensor data.

## Overview

The typical operator learning workflow in jNO is:

1. Create a **batched domain** with `B_train * jno.domain(...)`
2. Attach **tensor data** (e.g., parameter vectors) as additional variables
3. Define a **branch-trunk model** (e.g., DeepONet) that takes both spatial coordinates and parameters
4. Define the **PDE** using these inputs
5. Train with `crux.solve()`

---

## Full Example

This example learns the solution operator for a variable-coefficient diffusion equation:

```python
import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
import optax
from flax import linen as nn
import jax.numpy as jnp

# Training configuration
B_train = 4   # Number of training samples (different κ fields)
B_test = 1

# Parameter vectors that define each κ field
theta_train = jnp.ones((B_train, 4))  # shape: (B_train, 4)
theta_test = jnp.ones((B_test, 4))

# Batched domain: tiles the mesh B_train times
domain = B_train * jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    compute_mesh_connectivity=True
)
x, y = domain.variable("interior", (None, None))
xb, yb = domain.variable("boundary", (None, None))

# Attach parameter tensor — shape (B_train, 4)
(θ,) = domain.variable("θ", theta_train)
```

### Define a DeepONet

```python
class DeepONet(nn.Module):
    width: int
    depth: int
    p: int  # number of basis functions

    @nn.compact
    def __call__(self, x, y, θ):
        # Trunk: takes spatial coordinates
        t = jnp.concatenate([x, y], axis=-1)
        for _ in range(self.depth):
            t = nn.tanh(nn.Dense(self.width)(t))
        t = nn.Dense(self.p)(t)

        # Branch: takes parameter vector
        for _ in range(self.depth):
            θ = nn.tanh(nn.Dense(self.width)(θ))
        θ = nn.Dense(self.p)(θ)

        return jnp.sum(t * θ, axis=-1)

net = jnn.nn.wrap(DeepONet(width=64, depth=4, p=32))
u = net(x, y, θ) * x * (1 - x) * y * (1 - y)
```

### Define the PDE

```python
sin = jnn.sin
exp = jnn.exp
π = jnn.pi

# Spatially varying diffusivity
κ = exp(θ[0] + θ[1] * sin(2 * π * x) + θ[2] * jnn.cos(2 * π * y))

# PDE: -div(κ grad u) = f
ux = jnn.grad(u(x, y, θ), x)
uy = jnn.grad(u(x, y, θ), y)
pde = -(jnn.grad(κ * ux, x) + jnn.grad(κ * uy, y)) - (2 * π**2 * sin(π * x) * sin(π * y))

# Solve
crux = jno.core([pde], domain)
crux.solve(1000, optax.adam(1), lrs.warmup_cosine(1000, 100, 1e-3, 1e-5))
```

### Inference on New Parameters

```python
tst_domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.01),
    compute_mesh_connectivity=False
)
tst_domain.variable("θ", theta_test)

crux.plot(operation=u, test_pts=tst_domain).savefig("u_test.png", dpi=300)
```

---

## Key Concepts

### Batched Domains

Multiplying a domain by an integer creates batch copies:

```python
domain = B_train * jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
```

All batch elements share the same mesh points but can have different tensor data. The training loop uses `vmap` to efficiently process all batch elements in parallel.

### Tensor Variables

Tensor data attached via `domain.variable("name", data)` can be:
- Scalars per batch: shape `(B, 1)`
- Vectors per batch: shape `(B, d)`
- Any tensor: shape `(B, ...)`

These are broadcast to each collocation point during evaluation.

### Indexing Tensor Components

Access individual components of tensor variables with bracket notation:

```python
(θ,) = domain.variable("θ", theta_train)  # shape (B, 4)
θ[0]  # first component
θ[1]  # second component
```

---

## Using Built-in DeepONet

Instead of defining your own DeepONet class, use the built-in one:

```python
model = jnn.nn.deeponet(n_sensors=100, coord_dim=2, basis_functions=64)
```

See [Architecture Guide](Architecture-Guide.md) for all available architectures.

---

## See Also

- [Domain and Meshing](Domain-and-Meshing.md) — creating domains and attaching data
- [Architecture Guide](Architecture-Guide.md) — DeepONet and other operator architectures
- [Examples](Examples.md) — the `operator_learning` example
