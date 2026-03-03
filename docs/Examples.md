# Examples

The `examples/` directory contains self-contained, annotated example scripts demonstrating all major features.

---

## Quick Reference

| Script | Problem | Key Features |
|--------|---------|--------------|
| [`laplace1D.py`](#laplace1d) | 1D Laplace equation | MLP, hard BCs, tracker, save/load, evaluation |
| [`heat_equation.py`](#heat-equation) | 2D heat equation | DeepONet, time-dependent domain, multi-phase training |
| [`coupled_system.py`](#coupled-system) | Coupled 2D PDE system | Two models, SOAP, LoRA, masks, operator freezing |
| [`inverse_learning.py`](#inverse-learning) | Identify unknown PDE coefficients | Trainable parameters, inverse problem |
| [`tuner.py`](#hyperparameter-tuner) | Disk domain — Poisson eq. | Architecture search, hyperparameter sweep, per-model tuning |
| [`operator_learning/operator_learning.py`](#operator-learning) | Parametric PDE | DeepONet, batch operator learning, FEM comparison |
| [`poseidon/poisson_run.py`](#poseidon-foundation-model) | Poisson equation | Poseidon foundation model, fine-tuning |

---

## Laplace 1D

**File:** `examples/laplace1D.py`

Solves `∂²u/∂x² = sin(πx)` on `[0,1]` with homogeneous Dirichlet BCs.

```
u(0) = u(1) = 0
∂²u/∂x² − sin(πx) = 0
```

Analytical solution: `u(x) = −sin(πx) / π²`

**What it shows:**
- 1D domain setup with `jno.domain.line`
- MLP neural network with hard boundary enforcement (`u_net(x) * x * (1-x)`)
- Exponential LR schedule
- Tracker monitoring the mean error against the analytical solution
- Multi-phase training (Adam → L-BFGS)
- Saving with `crux.save()` and evaluating with `crux.eval()`
- Plotting predictions with matplotlib

---

## Heat Equation

**File:** `examples/heat_equation.py`

Solves the 2D heat equation:
```
∂u/∂t − 0.1 Δu = 0   on [0,1]²×[0,1]
u(x,y,0) = sin(πx)sin(πy)
u = 0 on ∂Ω
```

**What it shows:**
- Time-dependent rectangular domain (`time=(0, 1, 10)`)
- DeepONet (`jnn.nn.deeponet`) with branch network for time and trunk for spatial coords
- Hard boundary enforcement via multiplication
- Simultaneous PDE residual + initial condition constraints
- Multi-phase training (Adam → L-BFGS → AdamW)

---

## Coupled System

**File:** `examples/coupled_system.py`

Solves a coupled elliptic PDE system:
```
−Δu + v = f(x,y)
−Δv + u = g(x,y)
```
with manufactured solution `u=sin(πx)sin(πy)`, `v=sin(2πx)sin(πy)`.

**What it shows:**
- Two independent neural networks (`u_net` as a built-in MLP, `v_net` as a custom Equinox module)
- Custom geometry constructor
- Outward normals and view factor (`domain.variable(..., normals=True, view_factor=True)`)
- Point sensor constraint
- Adaptive boundary weighting with `WeightSchedule`
- All optimiser and training control features:
  - Adam with warm-up cosine decay
  - Adam with gradient clipping
  - SOAP second-order optimiser
  - L-BFGS refinement
- Per-model training: freeze, partial mask, LoRA fine-tuning
- Loading pretrained weights with `model.initialize()`
- Checkpointing and restart with `crux.save()` / `core.load()`

---

## Inverse Learning

**File:** `examples/inverse_learning.py`

Identifies three unknown PDE coefficients `a`, `b`, `c` from synthetic data constraints:
```
a · sin(πx) + a_true · sin(πx) ≈ 0
b · sin(πx) − b_true · sin(πx) ≈ 0
c · sin(πx) + c_true · sin(πx) ≈ 0
```

**What it shows:**
- Trainable scalar parameters with `jnn.parameter()`
- Inverse problem formulation (identifying coefficients, not a field)
- Large-scale training (200,000 epochs)
- Printing the identified parameter values from `crux.models`

---

## Hyperparameter Tuner

**File:** `examples/tuner.py`

Searches for the best architecture and training configuration for the Poisson equation on a disk domain:
```
−Δu = 1   on unit disk
u = 0 on ∂Ω
```

**What it shows:**
- Architecture search space with `jnn.tune.space()`
- Custom tunable equinox module using `jnn.tune.Arch`
- Wrapping a class (not instance) with `jnn.nn.wrap(MyMLP, space=a_space)`
- Training hyperparameter space (epochs, optimizer, learning rate, batchsize)
- Running a sweep with `crux.sweep(space, ng.optimizers.NGOpt, budget=N)`
- Per-model `.tune()` with freeze, LoRA, optimizer choices
- Gradient checkpointing and data offloading demo

---

## Operator Learning

**File:** `examples/operator_learning/operator_learning.py`

Learns the solution operator for a parametric diffusion PDE:
```
−∇·(κ(x,y;θ) ∇u) = f    on [0,1]²
u = 0 on ∂Ω
```
where `κ` depends on a 4-parameter vector `θ`. Trains on 40 samples and evaluates against FEM reference solutions.

**What it shows:**
- Operator learning with multiple batch samples (`B * jno.domain(...)`)
- Attaching parameter tensor data to the domain (`domain.variable("θ", theta_train)`)
- Custom DeepONet with separate branch (for θ) and trunk (for x,y)
- Warmup cosine LR schedule
- `crux.predict()` for inference on new parameter vectors
- Comparison with FEM reference solutions
- Visualisation of FEM vs PINN predictions + error plots

---

## Poseidon Foundation Model

**File:** `examples/poseidon/poisson_run.py`

Fine-tunes the pretrained Poseidon foundation model on the Poisson equation.

**What it shows:**
- Using `jno.domain.poseidon(nx=128, ny=128)` for a structured pixel grid
- Loading and fine-tuning `jnn.nn.poseidonT()` / `poseidonB()` / `poseidonL()`
- Transfer learning with pretrained operator models

---

## Running Examples

```bash
# Navigate to the examples directory
cd examples

# Run with uv (manages the environment automatically)
uv run python laplace1D.py
uv run python heat_equation.py
uv run python coupled_system.py
uv run python inverse_learning.py
uv run python tuner.py
uv run python operator_learning/operator_learning.py
uv run python poseidon/poisson_run.py
```

Results are saved to `./runs/<script_name>/`.
