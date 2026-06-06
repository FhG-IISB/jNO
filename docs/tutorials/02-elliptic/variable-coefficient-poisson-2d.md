# Variable-Coefficient Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example keeps the same square geometry but replaces constant diffusion with a spatially varying conductivity field.

## Problem Setup

```text
-div(kappa(x,y) grad u(x,y)) = f(x,y),   (x,y) in [0,1]^2,
u = 0 on the boundary
```

with `kappa(x,y) = 1 + x + y` and exact solution `sin(pi x) sin(pi y)`.

## Step 1: Define the Coefficient Field

The script constructs `kappa` directly from the sampled coordinates, so the PDE coefficients vary pointwise across the domain.

```python
domain = jno.domain.rect(mesh_size=0.05)
x, y, _ = domain.variable("interior")

kappa    = 1 + x + y
u_exact  = jno.np.sin(pi * x) * jno.np.sin(pi * y)
forcing  = (
    2 * pi**2 * kappa * u_exact
    - pi * jno.np.cos(pi * x) * jno.np.sin(pi * y)
    - pi * jno.np.sin(pi * x) * jno.np.cos(pi * y)
)
```

## Step 2: Build the Flux Instead of Just the Laplacian

Rather than writing `Delta u`, the script forms flux components and then computes their divergence.

```python
net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=80, num_layers=5, key=jax.random.PRNGKey(13)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 80, 0.5, end_value=1e-5)))

u      = net(x, y) * x * (1 - x) * y * (1 - y)
flux_x = kappa * jno.np.grad(u, x)
flux_y = kappa * jno.np.grad(u, y)
```

## Step 3: Enforce Boundary Conditions Hard

The `x(1-x)y(1-y)` factor wraps the raw network output so the boundary condition is built into the ansatz (handled above in Step 2).

## Step 4: Train Against a Spatially Heterogeneous PDE

Once coefficients vary in space, the forcing term becomes more complex, but the overall jNO workflow remains: residual → core → solve.

```python
pde     = -jno.np.divergence([flux_x, flux_y], [x, y]) - forcing
crux    = jno.core([pde.mse], domain)
history = crux.solve(40_000)
```

## What To Notice

- Variable coefficients are a common bridge from toy PDEs to physically meaningful media.
- Forming fluxes explicitly keeps the PDE readable.
- The example shows how spatial heterogeneity enters both forcing and residual construction.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py"
```
