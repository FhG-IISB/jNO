# Advection-Diffusion 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/advection_diffusion_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example combines transport and diffusion in a transient 1D problem.

## Problem Setup

The PDE has the form `u_t + c u_x = nu u_xx + f`, with a manufactured forcing term used for validation.

## Step 1: Build a Space-Time Domain

The script samples the interior and initial slice so both PDE dynamics and startup data can be enforced.

```python
c = 1.0    # advection speed
ν = 0.05   # diffusivity
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.1),
    time=(0, T_end, 4),
)
x, t   = domain.variable("interior")
x0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source  = jno.np.exp(-t) * ((ν * π**2 - 1) * jno.np.sin(π * x) + c * π * jno.np.cos(π * x))
```

## Step 2: Choose a Time-Dependent Model

A DeepONet-style architecture maps space-time inputs to the field value.

```python
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=1, n_outputs=1,
        n_layers=3, basis_functions=64, hidden_dim=32,
        key=jax.random.PRNGKey(1),
    )
)
net.optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.6, 10, 1e-5))

u = net(t, x) * x * (1 - x)
```

## Step 3: Encode Transport and Diffusion Together

The residual contains both a first derivative in space and a second derivative in space, so it mixes advective and diffusive behavior.

```python
u_t  = jno.np.grad(u, t)
u_x  = jno.np.grad(u, x)
u_xx = jno.np.grad(u_x, x)
pde  = u_t + c * u_x - ν * u_xx - source

u_0 = net(t0, x0) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

crux    = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(5000)
```

## What To Notice

- This is a good first transport example before moving to nonlinear convection.
- Advection and diffusion terms can differ strongly in scale.
- The overall workflow still matches the other PINN examples.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/advection_diffusion_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/advection_diffusion_1d.py"
```
