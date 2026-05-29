# Heat 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/heat_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to chapter</a>
</div>

This example solves the transient 1D heat equation and introduces time as an explicit input to the model.

## Problem Setup

The script solves a diffusion equation of the form `u_t = alpha u_xx` on a space-time domain with zero Dirichlet boundaries and a sinusoidal initial condition.

## Step 1: Build a Space-Time Domain

The domain includes both space and time, with separate sampling for interior and initial-condition points.

```python
α = 0.1   # thermal diffusivity
T_end = 0.5

domain = jno.domain.line(mesh_size=0.01, time=(0, T_end, 10))
x, t   = domain.variable("interior")
x0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-α * π**2 * t) * jno.np.sin(π * x)
```

## Step 2: Use a DeepONet-Style Model

The example uses a DeepONet architecture in PINN mode so the model can learn a time-dependent field over the full domain.

```python
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=1, n_outputs=1,
        n_layers=3, basis_functions=64, hidden_dim=32,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.9, 10000, 1e-5))
```

## Step 3: Hard-Enforce Spatial Boundary Conditions

A spatial envelope `x(1-x)` keeps the field zero at the two endpoints for every time. The initial condition `sin(πx)` is built directly into the ansatz via the additive term, so the IC is hard-enforced as well.

```python
u = jno.np.sin(π * x) + t * net(t, x) * x * (1 - x)
```

## Step 4: Add the Initial Condition as a Separate Constraint

The PDE residual governs the interior, while a second loss enforces the known initial profile at `t = 0`.

```python
pde = jno.np.grad(u, t) - α * jno.np.grad(jno.np.grad(u, x), x)

crux    = jno.core([pde.mse], domain)
history = crux.solve(10000)

_u, _u_exact = crux.eval([u, u_exact])
```

## What To Notice

- Time-dependent PDEs need both interior physics and initial data.
- The jNO workflow stays similar even though the field now depends on multiple coordinates.
- This is the cleanest parabolic starting point in the tutorial set.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/heat_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/heat_1d.py"
```
