# Examples

jNO ships with several example scripts in the `examples/` directory that demonstrate different use cases. This page walks through each one.

---

## 1D Laplace Equation (`laplace1D.py`)

**Problem:** Solve -u''(x) = sin(πx) on [0, 1] with homogeneous Dirichlet BCs.

**Features demonstrated:**
- 1D domain creation
- Hard boundary conditions via multiplication
- Analytical solution comparison
- Tracker for monitoring error
- LoRA fine-tuning
- Error metrics

```python
import jno
import jno.numpy as jnn
import optax

domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
(x,) = domain.variable("interior")

# Analytical solution
_u = -(1 / (jnn.pi**2)) * jnn.sin(jnn.pi * x)

# Neural network with hard BCs
u = jnn.nn.mlp(hidden_dims=64, num_layers=3)(x) * x * (1 - x)

# PDE constraint
pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)

# Error tracker (does not contribute to loss)
con = jnn.tracker(jnn.mean(u - _u))

# Train
crux = jno.core([pde, con], domain)
crux.solve(10_000, optax.adam, jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5))

# Fine-tune with LoRA
crux.solve(
    10_000, optax.adam,
    jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5),
    lora=jno.create_rank_dict(crux.params, rank=1, alpha=1.0),
)
```

---

## 2D Heat Equation (`heat_equation.py`)

**Problem:** Solve the 2D heat equation u_t = 0.1∇²u with sinusoidal initial conditions.

**Features demonstrated:**
- Time-dependent domain
- Multiple constraint types (PDE + initial condition)
- Constants via `jnn.constant`
- Flax NNX module wrapping
- KAN network integration

```python
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, 1, 10),
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u = jnn.nn.mlp(hidden_dims=[64, 64])(x, y, t) * x * (1 - x) * y * (1 - y)

pde = jnn.grad(u(x, y, t), t) - 0.1 * jnn.laplacian(u(x, y, t), [x, y])
ini = u(x0, y0, t0) - jnn.sin(jnn.pi * x0) * jnn.sin(jnn.pi * y0)

crux = jno.core([pde, ini], domain)
crux.solve(10_000, optax.adam(1),
    jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5),
    jno.schedule.constraint([1.0, 3.0]),
)
```

---

## Coupled PDE System (`coupled_system.py`)

**Problem:** Solve a coupled elliptic system with manufactured solution:
- -Δu + v = f(x,y)
- -Δv + u = g(x,y)

**Features demonstrated:**
- Multiple neural networks (u and v)
- Custom geometry definition
- Sensor data as point constraints
- Mixed hard/soft boundary conditions
- Multi-phase training (Adam → SOAP → L-BFGS)
- Domain multiplication for batched geometry
- Boundary normals and view factors
- Computation graph visualization
- Checkpointing (save/load)

```python
# Multiple training phases
crux.solve(4000, optax.adam(1), lrs.warmup_cosine(4000, 500, 1e-3, 1e-4),
    ws([1.0, 1.0, 10.0, 1.0]))

crux.solve(1000, soap(1, precondition_frequency=13),
    lrs(lambda e, _: 1e-4 * (5e-5 / 1e-4) ** (e / 1000)))

crux.solve(1000, optax.lbfgs, lrs(5e-5),
    ws(lambda e, L: [1.0, 1.0, 10.0, 1.0 * L[3]]))
```

---

## Inverse Problem (`inverse_learning.py`)

**Problem:** Recover unknown scalar parameters a, b, c from observations.

**Features demonstrated:**
- Trainable parameters via `jnn.parameter`
- Inverse problem formulation
- Float64 precision

```python
import jax
jax.config.update("jax_enable_x64", True)

a = jnn.parameter((1), name="a")()
b = jnn.parameter((1), name="b")()
c = jnn.parameter((1), name="c")()

# Known target values
_a, _b, _c = 24.83, 1.83, 104.23

# Constraints that drive parameters toward targets
con1 = a * jnn.sin(jnn.pi * x) + _a * jnn.sin(jnn.pi * x)
con2 = b * jnn.sin(jnn.pi * x) - _b * jnn.sin(jnn.pi * x)
con3 = c * jnn.sin(jnn.pi * x) + _c * jnn.sin(jnn.pi * x)

crux = jno.core([con1, con2, con3], domain)
crux.solve(200_000, optax.adam(1.0),
    jno.schedule.learning_rate.exponential(1e-3, 0.9, 200_000, 1e-5))

print(crux.params)  # Recovered parameters
```

---

## Operator Learning (`operator_learning/`)

**Problem:** Learn the solution operator for -div(κ∇u) = f with varying diffusivity κ.

**Features demonstrated:**
- Batched domains for operator learning
- Custom DeepONet architecture
- Tensor variable attachment
- FEM reference comparison
- Inference on unseen parameters

```python
B_train = 2
domain = B_train * jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y = domain.variable("interior", (None, None))
(θ,) = domain.variable("θ", theta_train)

net = jnn.nn.wrap(DeepONet(width=64, depth=4, p=32))
u = net(x, y, θ) * x * (1 - x) * y * (1 - y)

κ = jnn.exp(θ[0] + θ[1] * jnn.sin(2 * π * x) + ...)

ux = jnn.grad(u(x, y, θ), x)
uy = jnn.grad(u(x, y, θ), y)
pde = -(jnn.grad(κ * ux, x) + jnn.grad(κ * uy, y)) - forcing

crux = jno.core([pde], domain)
crux.solve(1000, optax.adam(1), lrs.warmup_cosine(1000, 100, 1e-3, 1e-5))
```

---

## Architecture Search (`tuner.py`)

**Problem:** Find the best MLP architecture for a Poisson problem on a disk domain.

**Features demonstrated:**
- Architecture search space definition
- Architecture-aware Flax module with `jnn.tune.Arch`
- Training hyperparameter space
- Nevergrad-based sweep

```python
a_space = jnn.tune.space()
a_space.unique("act", [nn.tanh, nn.selu, nn.gelu, jnp.sin])
a_space.unique("hid", [32, 64, 128])
a_space.unique("dep", [2, 3, 4])

class MLP(nn.Module):
    arch: jnn.tune.Arch
    @nn.compact
    def __call__(self, x, y):
        h = jnp.concatenate([x, y], axis=-1)
        for _ in range(self.arch("dep")):
            h = self.arch("act")(nn.Dense(self.arch("hid"))(h))
        return nn.Dense(1)(h)

u = jnn.nn.wrap(MLP, space=a_space)(x, y)

crux = jno.core([pde], domain)
crux.sweep(space=t_space, optimizer=ng.optimizers.NGOpt, budget=10)
```

---

## Running an Example

```bash
cd examples
uv run python laplace1D.py
```

Output files (plots, checkpoints) are saved to the `./runs/<example_name>/` directory.

---

## See Also

- [Training and Solving](Training-and-Solving.md) — details on the training API
- [Architecture Guide](Architecture-Guide.md) — all available neural operator architectures
- [Domain and Meshing](Domain-and-Meshing.md) — domain construction details
