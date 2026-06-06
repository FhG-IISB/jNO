# Coupled Parabolic 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_parabolic_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This example takes the coupled-field idea into a transient setting.

## Problem Setup

The script solves two time-dependent PDEs with cross-coupling terms and manufactured transient reference solutions.

## Step 1: Build Two Time-Dependent Fields

Both unknowns depend on space and time, so the sampled domain and constraint set are larger than in the stationary case.

```python
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, 10),
)
x, y, t   = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-t) * jno.np.sin(pi * x) * jno.np.sin(pi * y)
v_exact = jno.np.exp(-t) * jno.np.sin(2 * pi * x) * jno.np.sin(pi * y)
f = (2 * pi**2 - 1) * u_exact + v_exact
g = (5 * pi**2 - 1) * v_exact + u_exact

u_net = jno.nn.wrap(foundax.deeponet(n_sensors=1, coord_dim=2, n_outputs=1,
                                      n_layers=5, basis_functions=96, hidden_dim=64,
                                      key=jax.random.PRNGKey(24)))
v_net = jno.nn.wrap(foundax.deeponet(n_sensors=1, coord_dim=2, n_outputs=1,
                                      n_layers=5, basis_functions=96, hidden_dim=64,
                                      key=jax.random.PRNGKey(25)))
for net in [u_net, v_net]:
    net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(...)))

xy  = jno.np.concat([x, y])
xy0 = jno.np.concat([x0, y0])
u   = u_net(t,  xy)  * x  * (1 - x)  * y  * (1 - y)
v   = v_net(t,  xy)  * x  * (1 - x)  * y  * (1 - y)
u0  = u_net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)
v0  = v_net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)
```

## Step 2: Add Initial Conditions for Both Fields

Each unknown needs its own initial condition in addition to the coupled PDE residuals.

```python
ini_u = u0 - jno.np.sin(pi * x0) * jno.np.sin(pi * y0)
ini_v = v0 - jno.np.sin(2 * pi * x0) * jno.np.sin(pi * y0)
```

## Step 3: Train the System Jointly

All losses are optimized together so the two models remain consistent with each other and with the data.

```python
pde_u = u.d(t) - jno.np.laplacian(u, [x, y]) + v - f
pde_v = v.d(t) - jno.np.laplacian(v, [x, y]) + u - g

crux    = jno.core([pde_u.mse, pde_v.mse, ini_u.mse, ini_v.mse], domain)
history = crux.solve(40_000)

_u, _u_exact, _v, _v_exact = crux.eval([u, u_exact, v, v_exact])
```

## What To Notice

- Coupling and time dependence can be combined cleanly in one workflow.
- The same hard-boundary ideas can be reused for both unknowns.
- This is a good reference for multi-field transient PDEs.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_parabolic_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/coupled_parabolic_2d.py"
```
