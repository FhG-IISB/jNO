# Helmholtz 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/helmholtz_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example adds an oscillatory Helmholtz term to the elliptic residual, which makes the solution behavior more wave-like than Poisson-like.

## Problem Setup

```text
Delta u + k^2 u = -f(x,y),   (x,y) in [0,1]^2,
u = 0 on the boundary
```

with exact solution `u(x,y) = sin(pi x) sin(pi y)`.

## Step 1: Choose a Wave Number

The parameter `k` controls the oscillatory regime. Try values near `k = pi*sqrt(2) ≈ 4.44` to approach the resonant regime.

```python
k = 2.0  # wave number — change to test different regimes
```

## Step 2: Build the Domain and Forcing

The manufactured forcing is derived by substituting the exact solution into the PDE.

```python
domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y, _ = domain.variable("interior")

u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
forcing  = (2 * π**2 - k**2) * jno.np.sin(π * x) * jno.np.sin(π * y)

net = jno.nn.wrap(
    foundax.mlp(in_features=2, hidden_dims=64, num_layers=5, key=jax.random.PRNGKey(0))
).optimizer(optax.adam(optax.exponential_decay(1e-3, 80, 0.5, end_value=1e-5)))

u = net(x, y) * x * (1 - x) * y * (1 - y)
```

## Step 3: Assemble the Helmholtz Residual

The PDE combines the Laplacian with the zeroth-order term `k^2 u`, which is the defining feature of Helmholtz problems.

```python
pde = u.laplacian(x, y, scheme="automatic_differentiation") + k**2 * u + forcing
```

## Step 4: Track Relative Error

After solving, the script computes a relative L2 error and plots exact, predicted, and absolute-error fields.

```python
crux    = jno.core([pde.mse], domain)
history = crux.solve(40_000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
```

## What To Notice

- Helmholtz problems become harder as `k` approaches the resonant value.
- Hard BCs keep the loss simpler even for oscillatory problems.
- This example is a good first step toward frequency-domain PDEs.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/helmholtz_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/helmholtz_2d.py"
```
