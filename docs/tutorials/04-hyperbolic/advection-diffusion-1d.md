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
xb, tb = domain.variable("boundary")

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
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.6, end_value=1e-5)))

u = net(t, x)
```

## Step 3: Three Soft Constraints — PDE + IC + BC

The residual mixes a first space derivative (advection) and a second space derivative (diffusion); the IC and BC are passed as separate loss terms.

```python
# PDE:  u_t + c u_x − ν u_xx − f = 0
pde = u.d(t) + c * u.d(x) - ν * u.d2(x) - source

# IC:   u(x, 0) = sin(πx)
ic = net(t0, x0) - jno.np.sin(π * x0)

# BC:   u(0, t) = u(1, t) = 0
bc = net(tb, xb)

crux    = jno.core([pde.mse, ic.mse, bc.mse], domain)
history = crux.solve(5000)
```

## What To Notice

- Same soft IC + BC pattern as [Heat 1D](../03-parabolic/heat-1d.md) and [Wave 1D](wave-1d.md) — three explicit constraints make the role of each condition obvious.
- Advection and diffusion residual terms can differ strongly in scale; small `ν` (here `0.05`) makes the problem convection-dominated and harder to converge — that's why this example uses a longer training run than pure diffusion.
- The manufactured forcing `source` is what makes the prescribed `u_exact = e^{-t} sin(πx)` actually satisfy the PDE; without it the equation would have a different solution.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/advection_diffusion_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/advection_diffusion_1d.py"
```
