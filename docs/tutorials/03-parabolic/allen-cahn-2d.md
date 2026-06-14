# Allen-Cahn 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/03_parabolic/allen_cahn_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/03-parabolic/">Back to chapter</a>
</div>

This example solves a manufactured 2D Allen-Cahn problem and introduces a nonlinear cubic reaction term.

## Problem Setup

The PDE has the Allen-Cahn structure `u_t = epsilon^2 Delta u + u - u^3 + f`, with a known exact solution used to build the forcing term.

## Step 1: Build a Manufactured Nonlinear Problem

The exact solution is substituted into the PDE to derive a forcing term that makes validation straightforward.

```python
eps = 0.1
T_end = 1.0

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, 4),
)
x, y, t = domain.variable("interior")

S = sin(π * x) * sin(π * y)
u_exact = exp(-t) * S

coeff  = 2 * eps**2 * π**2 - 2
source = exp(-t) * S * coeff + exp(-3 * t) * S**3
```

## Step 2: Set Up the Space-Time Network

The model learns a field over space and time while respecting the chosen boundary handling.

```python
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=2, n_outputs=1,
        n_layers=3, basis_functions=64, hidden_dim=40,
        key=jax.random.PRNGKey(42),
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=1, decay_steps=500, end_value=1e-5)))

xy = jno.np.concat([x, y])
u  = net(t, xy) * x * (1 - x) * y * (1 - y)
```

## Step 3: Encode the Nonlinear Residual

The key change relative to the heat equation is the nonlinear reaction term `u - u^3`.

```python
pde = u.d(t) - eps**2 * jno.np.laplacian(u, [x, y]) - u + u**3 - source
```

## Step 4: Impose the Initial Condition

The script uses the same PDE infrastructure but anchors the solution at the initial time with an additional loss.

```python
u_at_0 = net(0 * t, xy) * x * (1 - x) * y * (1 - y)
ini     = u_at_0 - sin(π * x) * sin(π * y)

crux    = jno.core([pde.mse, ini.mse])
history = crux.solve(5000)
```

## What To Notice

- Nonlinear reaction terms are easy to express once the field is available symbolically.
- Manufactured solutions are especially valuable for nonlinear PDEs.
- This example is a good template for phase-field style problems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/03_parabolic/allen_cahn_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/allen_cahn_2d.py"
```
