# Convection-Reaction-Diffusion 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/convection_reaction_diffusion_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example mixes transport, diffusion, and reaction in a transient 2D PDE.

## Problem Setup

The script solves a convection-reaction-diffusion system with drift terms `(b_x, b_y)`, diffusion strength `nu`, and reaction strength `lambda`.

## Step 1: Build a 2D Space-Time Domain

The field depends on two spatial coordinates and time, so the sampled domain is larger and the residual has more moving parts.

```python
bx = 1.0
by = -0.5
nu = 0.05
lam = 0.25
T_end = 1.0
N_t = 4

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.06),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t   = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(pi * x) * jno.np.sin(pi * y)
source  = (
    (-1 + 2 * nu * pi**2 + lam) * u_exact
    + bx * pi * jno.np.exp(-t) * jno.np.cos(pi * x) * jno.np.sin(pi * y)
    + by * pi * jno.np.exp(-t) * jno.np.sin(pi * x) * jno.np.cos(pi * y)
)
```

## Step 2: Combine Multiple Physical Effects

The residual includes time evolution, first-order transport terms, second-order diffusion, and a linear reaction term.

```python
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=2, n_outputs=1,
        n_layers=4, basis_functions=64, hidden_dim=48,
        key=jax.random.PRNGKey(23),
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(...)))

xy  = jno.np.concat([x, y])
xy0 = jno.np.concat([x0, y0])
u   = net(t,  xy)  * x  * (1 - x)  * y  * (1 - y)
u0  = net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)

pde = (
    jno.np.grad(u, t)
    + bx * jno.np.grad(u, x)
    + by * jno.np.grad(u, y)
    - nu * jno.np.laplacian(u, [x, y])
    + lam * u
    - source
)
```

## Step 3: Train Against a Manufactured Solution

A manufactured forcing term keeps the problem verifiable while still exposing the full transient structure.

```python
ini = u0 - jno.np.sin(pi * x0) * jno.np.sin(pi * y0)

crux    = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(40000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
```

## What To Notice

- This is one of the richest transient tutorial examples in the set.
- The residual remains readable even with several physical terms.
- It is a good template for realistic advection-diffusion-reaction systems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/convection_reaction_diffusion_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/convection_reaction_diffusion_2d.py"
```
