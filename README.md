<p align="center">
  <img src="assets/logo.png" alt="jNO logo" width="200"/>
</p>

# jNO — JAX Neural Operators

**JAX-native building blocks for training physics-informed neural operators.**

jNO is a research-level framework for solving partial differential equations (PDEs) with neural networks. It provides a symbolic DSL for expressing PDE residuals, a wide library of neural operator architectures, flexible training infrastructure, and tools for hyperparameter search — all built on JAX for JIT-compilation and hardware acceleration.

> **Warning:** This is a research-level repository. It may contain bugs and is subject to continuous change without notice.

---

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Key Concepts](#key-concepts)
- [Domain & Geometry](#domain--geometry)
- [Neural Network Architectures](#neural-network-architectures)
- [Differential Operators (`jno.numpy`)](#differential-operators-jnonumpy)
- [Training](#training)
- [Adaptive Resampling](#adaptive-resampling)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Save / Load](#save--load)
- [Configuration](#configuration)
- [Examples](#examples)
- [Dependencies](#dependencies)

---

## Installation

jNO uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for environment management. `uv` installs and manages environments in your user directory, so you can typically run everything locally **without sudo**.

**Windows:** Allow script execution first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Install (CPU):**
```bash
uv sync
```

**Install (GPU / CUDA):**
```bash
uv sync --extra cuda
```

**Install with IREE AOT compiler:**
```bash
uv sync --extra iree
```

**Development install:**
```bash
uv sync --extra dev
```

---

## Quick Start

### 1D Laplace Equation (PINN)

```python
import jax
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs

# Setup run directory and logging
dire = jno.setup(__file__)

# Define the computational domain (1D line, [0,1])
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, t = domain.variable("interior")

# Define the neural network
u_net = jnn.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adamw(1), lr=lrs.exponential(1e-3, 0.3, 1_000, 1e-4))

# Symbolic expression for u, enforcing u(0)=u(1)=0 via multiplication
u = u_net(x) * x * (1 - x)

# PDE residual: Δu = sin(πx)  →  ∂²u/∂x² - sin(πx) = 0
π = jnn.pi
pde = jnn.grad(jnn.grad(u, x), x) - jnn.sin(π * x)

# Solve: minimise the mean-squared PDE residual
crux = jno.core([pde.mse], domain)
crux.solve(1_000).plot(f"{dire}/training_history.png")
crux.save(f"{dire}/crux.pkl")
```

### 2D Heat Equation (DeepONet)

```python
import jax
import jno
import jno.numpy as jnn
import optax
from jno import LearningRateSchedule as lrs

dire = jno.setup(__file__)

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, 1, 10),
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_net = jnn.nn.deeponet(
    n_sensors=1, sensor_channels=1,
    coord_dim=2, basis_functions=64,
    hidden_dim=64, n_layers=2,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 10_000, 1e-5))

u  = u_net(t,  jnn.concat([x,  y ]))  * x  * (1-x)  * y  * (1-y)
u0 = u_net(t0, jnn.concat([x0, y0])) * x0 * (1-x0) * y0 * (1-y0)

pde = jnn.grad(u, t) - 0.1 * jnn.laplacian(u, [x, y])    # heat equation
ini = u0 - jnn.sin(π * x0) * jnn.sin(π * y0)              # initial condition

crux = jno.core([pde.mse, ini.mse], domain)
crux.solve(1000).plot(f"{dire}/history.png")
```

---

## Key Concepts

### Symbolic Tracing

jNO uses a **lazy computation graph**. Expressions like `u_net(x, y)` and `jnn.grad(u, x)` return symbolic `Placeholder` nodes — no computation happens until `core.solve()` is called. This allows jNO to:
- Perform common sub-expression elimination (CSE) before JIT compilation
- Inspect the computation tree via `crux.print_tree()`
- Evaluate individual sub-expressions via `crux.eval(expr)`

### Constraint-based Training

Training is formulated as minimising a weighted sum of constraint residuals:

```
Loss = Σ wᵢ · mean(constraintᵢ²)
```

Each constraint is an expression reduced to a scalar (e.g., `.mse`, `.mae`, `.mean`). Constraints can represent:
- **PDE residuals** — physics equations in the interior
- **Boundary conditions** — Dirichlet, Neumann, Robin
- **Initial conditions** — for time-dependent problems
- **Data fitting** — supervised terms with sensor observations
- **Trackers** — monitored metrics that *do not* affect the loss

---

## Domain & Geometry

The `jno.domain` class generates and manages meshes using [pygmsh](https://github.com/nschloe/pygmsh). Domains expose named physical groups (e.g., `"interior"`, `"boundary"`, `"left"`) that you sample variables from.

### Built-in Geometries

```python
# 1D line [x0, x1]
jno.domain.line(x_range=(0,1), mesh_size=0.1)

# 2D rectangle
jno.domain.rect(x_range=(0,1), y_range=(0,1), mesh_size=0.1)

# 2D rectangle with equidistant structured mesh
jno.domain.equi_distant_rect(x_range=(0,1), y_range=(0,1), nx=10, ny=10)

# 2D disk
jno.domain.disk(center=(0,0), radius=1.0, mesh_size=0.1)

# 2D L-shaped domain
jno.domain.l_shape(size=1.0, mesh_size=0.1)

# 2D rectangle with a rectangular hole
jno.domain.rectangle_with_hole(outer_size=1.0, hole_size=0.4, mesh_size=0.1)

# 2D rectangle with multiple holes
jno.domain.rectangle_with_holes(outer_size=(2.0,1.0), holes=[...], mesh_size=0.1)

# 2D rectangle with PML (Perfectly Matched Layer) regions
jno.domain.rect_pml(x_range=(0,1), y_range=(0,1), pml_thickness_top=0.2)

# 3D cube
jno.domain.cube(x_range=(0,1), y_range=(0,1), z_range=(0,1), mesh_size=0.1)

# 2D structured grid for foundation models (Poseidon etc.)
jno.domain.poseidon(nx=128, ny=128)
```

You can also load an existing mesh from a `.msh` file:
```python
domain = jno.domain('./mesh.msh')
```

### Sampling Variables

```python
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))

# Sample all coordinates from a region
x, y, t = domain.variable("interior")

# Sample with outward normals and view factors (for radiation BCs)
xb, yb, tb, nx, ny, VF = domain.variable("boundary", normals=True, view_factor=True)

# Inject point data (e.g., sensor locations)
xs, ys = domain.variable("sensor", 0.5 * jnp.ones((2, 1, 2)), point_data=True, split=True)

# Attach tensor data (e.g., PDE parameters per batch sample)
k = domain.variable("k", jnp.array([[1.0], [2.0]]))  # (B, 1)
```

### Time-dependent Problems

```python
# time=(start, end, n_steps)
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, 1, 10),
)
x, y, t = domain.variable("interior")   # t is the temporal variable
x0, y0, t0 = domain.variable("initial") # t0 = 0 (initial time slice)
```

### Operator Learning (Multiple Samples)

Multiply the domain by an integer to create B independent training samples:
```python
domain = 40 * jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
```

### Visualization

```python
domain.plot("domain.png")  # plots mesh, boundaries, and normals
```

---

## Neural Network Architectures

All architectures are accessible via `jno.numpy.nn` (or `jnn.nn`). They all return a `Model` wrapper that plugs into the jNO training pipeline and supports `.optimizer()`, `.freeze()`, `.lora()`, `.mask()`, and `.initialize()`.

### Multi-layer Perceptron (MLP)

```python
u_net = jnn.nn.mlp(
    in_features=2,
    output_dim=1,
    hidden_dims=64,        # int or list of ints
    num_layers=3,
    activation=jnp.tanh,   # or jax.nn.gelu, jax.nn.selu, jnp.sin, ...
    dropout_rate=0.0,
    layer_norm=False,
    key=jax.random.PRNGKey(0),
)
```

### Fourier Neural Operator (FNO)

```python
# 1D FNO — e.g., Burgers equation
model = jnn.nn.fno1d(in_features=1, hidden_channels=64, n_modes=16, key=key)

# 2D FNO — e.g., Darcy flow
model = jnn.nn.fno2d(in_features=1, hidden_channels=32, n_modes=12, d_vars=1, key=key)

# 3D FNO — spatiotemporal problems
model = jnn.nn.fno3d(in_features=1, hidden_channels=24, n_modes=8, d_vars=1, key=key)
```

### U-Net

```python
model = jnn.nn.unet1d(in_dim=1, out_dim=1, hidden_channels=32, n_levels=3, key=key)
model = jnn.nn.unet2d(in_dim=1, out_dim=1, hidden_channels=32, n_levels=3, key=key)
model = jnn.nn.unet3d(in_dim=1, out_dim=1, hidden_channels=16, n_levels=3, key=key)
```

### Continuous Neural Operator (CNO)

```python
model = jnn.nn.cno2d(in_dim=1, out_dim=1, size=64, N_layers=3, key=key)
```

### DeepONet (Branch-Trunk)

```python
model = jnn.nn.deeponet(
    n_sensors=10,        # number of sensor points for branch input
    sensor_channels=1,   # channels per sensor
    coord_dim=2,         # spatial coordinate dimension for trunk input
    basis_functions=64,  # latent dimension
    hidden_dim=64,
    n_layers=3,
    key=key,
)
```

### GeoFNO (Irregular Domains)

```python
model = jnn.nn.geofno2d(
    nks=(16, 16),   # number of Fourier modes per dimension
    Ls=(1.0, 1.0),  # domain extents
    in_dim=3,
    out_dim=1,
    key=key,
)
```

### Point Cloud Neural Operator (PCNO)

```python
model = jnn.nn.pcno(in_dim=3, out_dim=1, hidden_channels=64, key=key)
```

### Multigrid Neural Operator (MgNO)

```python
model = jnn.nn.mgno1d(in_dim=1, out_dim=1, key=key)
model = jnn.nn.mgno2d(in_dim=1, out_dim=1, key=key)
```

### Transformers / Attention

```python
model = jnn.nn.transformer(d_model=64, n_heads=4, n_layers=3, key=key)
model = jnn.nn.pit(in_dim=2, out_dim=1, key=key)       # Position-induced Transformer
model = jnn.nn.scot(in_dim=2, out_dim=1, key=key)      # Scalable Operator Transformer
```

### GNOT (General Neural Operator Transformer)

```python
model = jnn.nn.gnot(...)
model = jnn.nn.cgptno(...)   # Conditional GNOT
model = jnn.nn.moegnot(...)  # Mixture-of-Experts GNOT
```

### PointNet

```python
model = jnn.nn.pointnet(in_dim=3, out_dim=1, key=key)
```

### Poseidon Foundation Models

```python
model = jnn.nn.poseidonT(key=key)  # Tiny
model = jnn.nn.poseidonB(key=key)  # Base
model = jnn.nn.poseidonL(key=key)  # Large
```

### Custom Equinox / Flax Models

```python
import equinox as eqx

class MyNet(eqx.Module):
    linear: eqx.nn.Linear
    def __init__(self, *, key): self.linear = eqx.nn.Linear(2, 1, key=key)
    def __call__(self, x, y): return self.linear(jnp.concat([x, y], axis=-1))

my_net = jnn.nn.wrap(MyNet(key=jax.random.PRNGKey(0)))

# Flax Linen models
my_net = jnn.nn.flaxwrap(FlaxModule(), input=(dummy_x,), key=key)

# Flax NNX models (auto-detected)
my_net = jnn.nn.wrap(MyNNXModule(nnx.Rngs(0)))
```

### Trainable Parameters (Inverse Problems)

```python
# Learn a scalar coefficient
D = jnn.parameter((1,), key=jax.random.PRNGKey(0), name="D")
d = D()  # symbolic placeholder
```

---

## Differential Operators (`jno.numpy`)

`jno.numpy` (importable as `jnn`) mirrors NumPy's API but operates on symbolic `Placeholder` expressions and supports automatic/finite-difference differentiation.

### Calculus Operators

```python
import jno.numpy as jnn

# Gradient: ∂u/∂x
u_x = jnn.grad(u, x)                          # automatic differentiation
u_x = jnn.grad(u, x, scheme="finite_difference")  # mesh-based FD

# Laplacian: ∂²u/∂x² + ∂²u/∂y²
lap_u = jnn.laplacian(u, [x, y])

# Full Jacobian: [∂u/∂x, ∂u/∂y]
J = jnn.jacobian(u, [x, y])

# Full Hessian matrix
H = jnn.hessian(u, [x, y])

# Divergence: ∇·F
div_F = jnn.divergence([Fx, Fy], [x, y])

# 2D curl (scalar)
curl = jnn.curl_2d(Fx, Fy, x, y)

# 3D curl (vector)
curl_vec = jnn.curl_3d(Fx, Fy, Fz, x, y, z)
```

### Math Functions

```python
# Trigonometric
jnn.sin(x), jnn.cos(x), jnn.tan(x)
jnn.arcsin(x), jnn.arccos(x), jnn.arctan(x), jnn.arctan2(y, x)

# Hyperbolic
jnn.sinh(x), jnn.cosh(x), jnn.tanh(x)

# Exponential / logarithm
jnn.exp(x), jnn.log(x), jnn.log2(x), jnn.log10(x)

# Power / roots
jnn.sqrt(x), jnn.cbrt(x), jnn.square(x)

# Absolute value / rounding
jnn.abs(x), jnn.floor(x), jnn.ceil(x), jnn.round(x)
```

### Array Operations

```python
jnn.concat([x, y], axis=-1)   # concatenate along last axis
jnn.stack([x, y], axis=0)     # stack along new axis
jnn.reshape(x, (N, D))
jnn.squeeze(x)
jnn.expand_dims(x, axis=0)
```

### Reduction Properties

Every `Placeholder` expression supports reduction properties:

```python
expr.mse    # mean(expr²)   — use as a constraint
expr.mae    # mean(|expr|)
expr.mean   # mean(expr)
expr.sum    # sum(expr)
expr.max    # max(expr)
expr.min    # min(expr)
expr.std    # std(expr)
```

### Constants

```python
# From a dict
C = jnn.constant("C", {"k": 1.0, "rho": 2.7, "physics": {"g": 9.81}})
k = C.k          # Constant placeholder for use in expressions
g = C.physics.g

# From a file (.json, .yaml, .yml, .toml, .pkl, .npz, .npy)
C = jnn.constant("C", "params.json")
```

### Trackers

Monitored quantities that appear in the training log but **do not** contribute to the loss:

```python
val_err = jnn.tracker(jnn.mean(u(x, y) - u_exact(x, y)), interval=100)
crux = jno.core([pde.mse, boc.mse, val_err], domain)
```

### View Factor Operator (Radiation)

```python
VF_op = jnn.view_factor(VF)   # VF is the view-factor matrix from domain.variable(view_factor=True)
q_rad = VF_op @ q_boundary    # radiative heat flux
```

---

## Training

### Setting Up the Solver

```python
crux = jno.core(
    constraints=[pde.mse, boc.mse],   # list of scalar expressions to minimise
    domain=domain,
    rng_seed=42,                       # optional; also configurable via .jno.toml
    mesh=(1, 1),                       # device mesh (batch_devices, model_devices)
)
```

### Attaching Optimizers

Each model in the problem must have an optimizer set before calling `solve()`:

```python
u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.9, 1000, 1e-5))
v_net.optimizer(optax.adamw, lr=lrs.warmup_cosine(4000, 500, 1e-3, 1e-4))
```

### Learning Rate Schedules

```python
from jno import LearningRateSchedule as lrs

lrs.constant(1e-3)
lrs.exponential(lr0=1e-3, decay_rate=0.9, decay_steps=1000, lr_end=1e-5)
lrs.cosine(total_steps=5000, lr0=1e-3, lr_end=1e-6)
lrs.warmup_cosine(total_steps=5000, warmup_steps=500, lr0=1e-3, lr_end=1e-6)
lrs.piecewise_constant(boundaries=[1000, 3000], values=[1e-3, 5e-4, 1e-4])

# Custom schedule: (epoch, individual_losses) -> lr
lrs(lambda e, L: 1e-4 * (0.9 ** (e / 1000)))
```

### Weight Schedules

Control per-constraint weighting during training:

```python
from jno import WeightSchedule as ws

ws([1.0, 10.0, 1.0])  # fixed weights

# Adaptive: increase boundary weight when boundary loss is large
ws(lambda e, L: [1.0, 1.0, 10.0 * L[2]])
```

### Running Training

```python
stats = crux.solve(
    epochs=5000,
    constraint_weights=ws([1.0, 10.0]),
    batchsize=128,              # None = full batch
    checkpoint_gradients=True,  # gradient checkpointing (saves memory)
    offload_data=True,          # keep data on CPU, stream mini-batches to GPU
)
stats.plot("history.png")
```

### Multi-Phase Training

Call `.solve()` multiple times with different optimizers or schedules:

```python
# Phase 1: Adam warm-up
u_net.optimizer(optax.adam, lr=lrs.warmup_cosine(2000, 200, 1e-3, 1e-5))
crux.solve(2000).plot("phase1.png")

# Phase 2: L-BFGS refinement
u_net.optimizer(optax.lbfgs, lr=lrs(5e-5))
crux.solve(500).plot("phase2.png")

# Phase 3: SOAP
from soap_jax import soap
u_net.optimizer(soap(1), lr=lrs(1e-5))
crux.solve(500).plot("phase3.png")
```

### Per-Model Training Controls

```python
# Freeze all parameters of a model
v_net.freeze()

# Unfreeze by setting a new optimizer
v_net.optimizer(optax.adam, lr=lrs(1e-4))

# Partial freeze: train only specific layers
all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
mask = eqx.tree_at(lambda m: (m.dense3.weight, m.dense3.bias), all_false, (True, True))
u_net.mask(mask).optimizer(optax.adam, lr=lrs(5e-4))

# LoRA fine-tuning: freeze base weights, train low-rank adapters only
v_net.freeze().lora(rank=4, alpha=1.0).optimizer(optax.adam, lr=lrs(1e-4))

# Load pretrained weights
u_net.initialize(pretrained_module)         # equinox module or pytree
u_net.initialize("runs/pretrained_u.eqx")  # from file

# Load pretrained weights for a Flax model
v_net.initialize(flax_params)          # Flax params dict
v_net.initialize("runs/pretrained.msgpack")
```

### Multi-GPU / Multi-Device Parallelism

```python
import jax

# Data parallelism (split batches across 4 GPUs)
crux = jno.core(constraints, domain, mesh=(4, 1))

# Model parallelism (shard model across 2 GPUs)
crux = jno.core(constraints, domain, mesh=(1, 2))

# Hybrid (2 batch × 2 model = 4 GPUs total)
crux = jno.core(constraints, domain, mesh=(2, 2))

# Auto-detect all available GPUs for data parallelism
n = len(jax.devices())
crux = jno.core(constraints, domain, mesh=(n, 1))
```

### Evaluation

```python
# Evaluate any symbolic expression on the training domain
pred = crux.eval(u)

# Evaluate on a different domain (e.g., finer test grid)
test_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01))
pred_test = crux.eval(u, domain=test_domain)

# Prediction on arbitrary points
points = np.tile(test_domain.points[None, ...], (B, 1, 1))
pred = crux.predict(points=points, operation=u, context=test_domain.context)
```

### Debugging

```python
# Print the symbolic computation tree
crux.print_tree()
crux.print_tree("tree.txt")  # also write to file

# Print per-node shapes
crux.print_shapes()
```

---

## Adaptive Resampling

Adaptive resampling strategies replace collocation points during training to focus on high-error regions, reducing the chance of over-fitting in well-solved regions.

Pass a sampler to `domain.set_sampler()` (or specify via training arguments):

```python
from jno import sampler

# Random (baseline) — prevents overfitting by refreshing points
s = sampler.random(resample_every=100, resample_fraction=0.1, start_epoch=1000)

# RAD — Residual-Adaptive Distribution
s = sampler.rad(resample_every=100, resample_fraction=0.1, start_epoch=1000, k=10)

# RARD — Residual-Adaptive Refinement with Distribution
s = sampler.rard(resample_every=100, resample_fraction=0.1, start_epoch=1000, power=2.0)

# HA — Hybrid Adaptive (alternates random and adaptive phases)
s = sampler.ha(resample_every=100, resample_fraction=0.5, start_epoch=1000, alternate=True)

# CR3 — Causal Retain-Resample (for time-dependent PDEs)
s = sampler.cr3(resample_every=100, resample_fraction=0.5, start_epoch=1000)

# PINNFluence — Influence-function-based adaptive sampling
s = sampler.pinnfluence(resample_every=500, resample_fraction=0.2, start_epoch=2000)
```

---

## Hyperparameter Tuning

jNO integrates [Nevergrad](https://github.com/facebookresearch/nevergrad) for automatic architecture and training hyperparameter search.

### Defining a Search Space

```python
import jno.numpy as jnn
from jno.tuner import ArchSpace

space = jnn.tune.space()
space.unique("epochs", [500, 1000, 2000])
space.unique("optimizer", [optax.adam, optax.adamw])
space.unique("learning_rate", [lrs.exponential(1e-3, 0.8, 10000, 1e-5), lrs.constant(1e-3)])
space.unique("batchsize", [32, 64, 128])

# Continuous parameter
space.float_range("dropout", low=0.0, high=0.3)

# Integer parameter
space.int_range("hidden_dim", low=32, high=256)
```

### Architecture Search

```python
a_space = jnn.tune.space()
a_space.unique("act", [jnp.tanh, jax.nn.selu, jax.nn.gelu], category="architecture")
a_space.unique("hid", [32, 64, 128], category="architecture")
a_space.unique("dep", [2, 3, 4], category="architecture")

class MyMLP(eqx.Module):
    def __init__(self, arch: jnn.tune.Arch, *, key):
        depth = arch("dep")
        hidden = arch("hid")
        self.act = arch("act")
        # ...

u = jnn.nn.wrap(MyMLP, space=a_space)(x, y)
```

### Running a Sweep

```python
import nevergrad as ng

stats = crux.sweep(
    space=space,
    optimizer=ng.optimizers.NGOpt,  # or "NGOpt", "OnePlusOne", "CMA", etc.
    budget=20,      # number of configurations to try
)
stats.plot("best_history.png")
print(f"Best config: {crux.best_config}")
```

### Per-Model Tuning

```python
backbone.tune(
    freeze=[True, False],
    lora=[(4, 1.0), None],
    optimizer=[optax.adam(1), optax.lbfgs(1)],
    lr=[lrs.constant(1e-3), lrs.constant(1e-4)],
)
```

---

## Save / Load

jNO uses `cloudpickle` to serialise the full solver state (model weights, optimizer states, training logs, domain, expression tree).

```python
# Save
crux.save("runs/crux.pkl")

# Load and continue training
crux = jno.core.load("runs/crux.pkl")
crux.set_optimizer(optax.adam, lr=lrs(1e-5))
crux.solve(1000)

# Module-level convenience functions
jno.save(crux, "runs/crux.pkl")
crux2 = jno.load("runs/crux.pkl")
```

### Encrypted Save / Load (RSA-signed)

Configure RSA keys in `.jno.toml`:

```toml
[rsa]
public_key  = "~/.jno/public.pem"
private_key = "~/.jno/private.pem"
```

Then `save` / `load` automatically sign and verify the file:

```python
jno.save(crux, "runs/crux.pkl")                    # creates crux.pkl + crux.sig
jno.load("runs/crux.pkl", signature_path="runs/crux.sig")
```

---

## Configuration

jNO looks for a TOML config file in two locations (first match wins):

1. `.jno.toml` in the current working directory (project-level)
2. `~/.jno/config.toml` (user-level)

**Example `.jno.toml`:**
```toml
[jno]
seed = 42              # global RNG seed

[runs]
base_dir = "./runs"    # default run directory

[rsa]
public_key  = "~/.jno/public.pem"
private_key = "~/.jno/private.pem"
```

**API:**
```python
dire = jno.setup(__file__)          # init logging, returns run dir
dire = jno.setup(__file__, name="my_experiment")

jno.load_config()                   # force reload config
jno.get_config()                    # get cached config dict
jno.get_config_path()               # path of loaded config file
jno.get_runs_base_dir()             # value of runs.base_dir
jno.get_seed()                      # value of jno.seed
jno.get_rsa_public_key()            # RSA public key path
jno.get_rsa_private_key()           # RSA private key path
```

---

## Examples

The [`examples/`](examples/) directory contains self-contained scripts:

| Example | Description |
|---------|-------------|
| [`laplace1D.py`](examples/laplace1D.py) | 1D Laplace equation with MLP, tracker monitoring, save/load |
| [`heat_equation.py`](examples/heat_equation.py) | 2D heat equation with DeepONet, multi-phase training |
| [`coupled_system.py`](examples/coupled_system.py) | Coupled 2D PDE system, all training features showcased |
| [`inverse_learning.py`](examples/inverse_learning.py) | Identify unknown PDE coefficients |
| [`tuner.py`](examples/tuner.py) | Architecture and hyperparameter search |
| [`operator_learning/operator_learning.py`](examples/operator_learning/operator_learning.py) | Parametric PDE with DeepONet + FEM comparison |
| [`poseidon/poisson_run.py`](examples/poseidon/poisson_run.py) | Using the Poseidon foundation model |

---

## Dependencies (thank you!)

This project stands on the shoulders of some fantastic open-source JAX ecosystem libraries:

- Backbone → [JAX](https://github.com/jax-ml/jax)
- Optimizers → [Optax](https://github.com/google-deepmind/optax) and/or [SOAP](https://github.com/haydn-jones/SOAP_JAX)
- Neural Network → [Equinox](https://github.com/patrick-kidger/equinox) and/or [Flax](https://github.com/google/flax)
- Mesh generation → [pygmsh](https://github.com/nschloe/pygmsh) + [meshio](https://github.com/nschloe/meshio)
- Hyperparameter search → [Nevergrad](https://github.com/facebookresearch/nevergrad)
- Signed serialisation → [pylotte](https://github.com/FhG-IISB/pylotte)
- Einsum notation → [einops](https://github.com/arogozhnikov/einops)
