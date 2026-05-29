# Burgers Viscous 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/burgers_viscous_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example solves the viscous Burgers equation, one of the standard nonlinear PDE benchmarks.

## Problem Setup

The PDE has the form `u_t + u u_x = nu u_xx + f`, with a manufactured exact solution used to derive the forcing term.

## Step 1: Build the Space-Time Model

The script uses the same transient PINN pattern as advection-diffusion but with a stronger nonlinear term.

```python
ν = 0.05   # viscosity
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.1),
    time=(0, T_end, 4),
)
x, t   = domain.variable("interior")
x0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(π * x)
source  = (jno.np.exp(-t) * (ν * π**2 - 1) * jno.np.sin(π * x)
           + (π / 2) * jno.np.exp(-2 * t) * jno.np.sin(2 * π * x))

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=1, n_outputs=1,
        n_layers=4, basis_functions=64, hidden_dim=48,
        key=jax.random.PRNGKey(3),
    )
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

u = net(t, x) * x * (1 - x)
```

## Step 2: Add Nonlinear Convection

The product `u u_x` is what makes Burgers different from linear transport models.

```python
u_t  = jno.np.grad(u, t)
u_x  = jno.np.grad(u, x)
u_xx = jno.np.grad(u_x, x)
pde  = u_t + u * u_x - ν * u_xx - source
```

## Step 3: Train and Compare With the Exact Solution

The script includes diagnostics that report error and visualize the learned field.

```python
u_0 = net(t0, x0) * x0 * (1 - x0)
ini = u_0 - jno.np.sin(π * x0)

crux    = jno.core([pde.mse, ini.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
```

## What To Notice

- Burgers is often the first nonlinear transport PDE people try.
- The nonlinearity appears naturally once the field and its gradient are available.
- This example is a good benchmark for model capacity and training stability.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/burgers_viscous_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/burgers_viscous_1d.py"
```
