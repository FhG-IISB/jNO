# Coupled Elliptic 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_elliptic_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This example solves a stationary system with two interacting unknown fields.

## Problem Setup

The script solves a coupled system of the form `-Delta u + v = f` and `-Delta v + u = g` on the unit square.

## Step 1: Build Two Unknown Fields

Instead of one neural network, the script defines one model for `u` and one model for `v`.

```python
domain = jno.domain.rect(mesh_size=0.2)
x, y, _ = domain.variable("interior")

u_exact = sin(π * x) * sin(π * y)
v_exact = sin(2 * π * x) * sin(π * y)
f = 2 * π**2 * sin(π * x) * sin(π * y) + sin(2 * π * x) * sin(π * y)
g = 5 * π**2 * sin(2 * π * x) * sin(π * y) + sin(π * x) * sin(π * y)

k1, k2 = jax.random.split(jax.random.PRNGKey(0))
u_net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=64, num_layers=4, key=k1))
v_net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=64, num_layers=4, key=k2))
for net in [u_net, v_net]:
    net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=1e-3, warmup_steps=1, decay_steps=10, end_value=1e-5)))

boundary_envelope = 16.0 * x * (1 - x) * y * (1 - y)
u = u_net(x, y) * boundary_envelope
v = v_net(x, y) * boundary_envelope
```

## Step 2: Assemble Coupled Residuals

Each PDE residual depends on both unknowns, so the optimization must update the two fields jointly.

```python
Δu = jno.np.laplacian(u, [x, y])
Δv = jno.np.laplacian(v, [x, y])

pde1 = -Δu + v - f
pde2 = -Δv + u - g

crux    = jno.core([pde1.mse, pde2.mse], domain)
history = crux.solve(10000)
```

## Step 3: Track Each Field Against Reference Data

The script uses manufactured solutions so both coupled fields can be validated during training.

```python
_u, _u_exact, _v, _v_exact = crux.eval([u, u_exact, v, v_exact])
rel_l2_u = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
rel_l2_v = float(jax.numpy.linalg.norm(_v - _v_exact) / (jax.numpy.linalg.norm(_v_exact) + 1e-8))
```

## What To Notice

- jNO can optimize multiple interacting models in one core.
- This is the simplest introduction to multi-physics style coupling.
- The pattern extends naturally to larger coupled systems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_elliptic_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/coupled_elliptic_2d.py"
```
