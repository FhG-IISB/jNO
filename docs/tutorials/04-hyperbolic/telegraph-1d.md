# Telegraph 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/telegraph_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example adds damping to a wave equation and produces the classical telegraph equation.

## Problem Setup

The PDE has the form `u_tt + beta u_t = c^2 u_xx + f`, with manufactured data used for validation.

## Step 1: Reuse the Wave-Equation Structure

The script keeps the second-order time derivative but introduces an additional first-order time term.

```python
beta = 0.5
c = 1.0
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.line(mesh_size=0.1),
    time=(0, T_end, 4),
)
x, t   = domain.variable("interior")
x0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(pi * x)
source  = (1 - beta + c**2 * pi**2) * u_exact

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=1, n_outputs=1,
        n_layers=4, basis_functions=64, hidden_dim=48,
        key=jax.random.PRNGKey(22),
    )
)
net.optimizer(optax.adam(1), lr=lrs.warmup_cosine(10, 1, 1e-3, 1e-5))

u = net(t, x) * x * (1 - x)
```

## Step 2: Handle Two Initial Conditions

As in the wave example, both displacement and velocity information at `t = 0` are required.

```python
dt_ic = 1e-2
u0   = net(t0, x0) * x0 * (1 - x0)
u_t0 = ((net(t0 + dt_ic, x0) - net(t0, x0)) / dt_ic) * x0 * (1 - x0)

ini_u  = u0   - jno.np.sin(pi * x0)
ini_ut = u_t0 + jno.np.sin(pi * x0)
```

## Step 3: Add Damping to the Residual

The `beta u_t` term changes the dynamics from undamped propagation to dissipative wave motion.

```python
pde = (jno.np.grad(jno.np.grad(u, t), t)
       + beta * jno.np.grad(u, t)
       - c**2 * jno.np.grad(jno.np.grad(u, x), x)
       - source)

crux    = jno.core([pde.mse, ini_u.mse, ini_ut.mse], domain)
history = crux.solve(5000)
```

## What To Notice

- This is a minimal extension of the wave equation that changes the qualitative behavior.
- Damping is straightforward to add once time derivatives are already available.
- The example is useful for understanding how multiple time-derivative orders coexist in one residual.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/telegraph_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/telegraph_1d.py"
```
