# Wave 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/wave_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example solves a second-order-in-time wave equation and introduces both displacement and velocity initial data.

## Problem Setup

The PDE has the form `u_tt = c^2 u_xx`, together with initial displacement and initial velocity conditions.

## Step 1: Build a Space-Time Domain

The script samples interior and initial-time points, just as in the parabolic examples.

```python
c = 1.0    # wave speed
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.01),
    time=(0, T_end, 20),
)
x, t = domain.variable("interior")

u_exact = jno.np.cos(c * π * t) * jno.np.sin(π * x)

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=1, n_outputs=1,
        n_layers=6, basis_functions=128, hidden_dim=96,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(7),
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(...)))
```

## Step 2: Add Two Initial Constraints

Unlike diffusion equations, the wave equation needs both `u(x,0)` and `u_t(x,0)`. The ansatz hard-encodes both via the `t^2` factor: at `t=0` the additive `sin(πx)` fixes the displacement and `d/dt(t^2) = 2t = 0` enforces zero initial velocity.

```python
# u(x,0) = sin(πx), u_t(x,0) = 0, u(0,t) = u(1,t) = 0 — all hard
u = jno.np.sin(π * x) + t**2 * net(t, x) * x * (1 - x)
```

## Step 3: Build the Hyperbolic Residual

The residual uses a second derivative in time and a second derivative in space.

```python
u_t  = jno.np.grad(u, t)
u_tt = jno.np.grad(u_t, t)
u_xx = jno.np.grad(jno.np.grad(u, x), x)
pde  = u_tt - c**2 * u_xx

crux    = jno.core([pde.mse], domain)
history = crux.solve(50000)

_u, _u_exact = crux.eval([u, u_exact])
```

## What To Notice

- Hyperbolic equations often require more than one initial condition.
- This example is the cleanest starting point for second-order-in-time PDEs.
- It is a good reference before adding damping or source terms.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/wave_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/wave_1d.py"
```
