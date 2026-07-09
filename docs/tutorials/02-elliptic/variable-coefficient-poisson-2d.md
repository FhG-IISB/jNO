# Variable-Coefficient Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/02-elliptic/">Back to chapter</a>
</div>

Same square geometry as the constant-coefficient example, but with a spatially varying conductivity field.

## Problem Setup

```text
-div(kappa(x,y) grad u(x,y)) = f(x,y),   (x,y) in [0,1]^2,
u = 0 on the boundary
```

with `kappa(x,y) = 1 + x + y` and exact solution `sin(pi x) sin(pi y)`.

## Step 1 — Domain and coefficient field

`kappa` is built directly from the sampled coordinates, so the PDE coefficients vary pointwise across the domain.

```python
--8<-- "tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py:setup"
```

## Step 2 — Build the flux, then take its divergence

Rather than writing `Delta u`, the script forms the flux vector `kappa · grad u` directly and takes its divergence. The `x(1 - x) y(1 - y)` factor on `net(x, y)` enforces the homogeneous Dirichlet BC hard.

```python
--8<-- "tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py:residual"
```

## Step 3 — Train

The residual → core → solve workflow is unchanged from the constant-coefficient case; only the forcing term and the spatially varying `kappa` differ.

```python
--8<-- "tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py:solve"
```

## Result

![Three panels: the jNO field u, the exact sin(pi x) sin(pi y), and their signed pointwise error.](/jNO/assets/variable_coefficient_poisson_2d.png)

Evaluated on the mesh nodes, the trained network reproduces the manufactured solution to rel-$L^2 \approx 1.1\times10^{-3}$; the signed-error panel (right, centered at 0) shows the residual sits around $10^{-3}$ and is largest in the interior, well away from the hard-enforced boundary.

## What to notice

- `u.grad(x, y)` returns a `VectorView`; multiplying by the scalar `kappa` preserves the view type (`Placeholder × VectorView → VectorView`), so the chain `kappa * u.grad(x, y) → .div(x, y)` reads exactly like the math `∇·(κ∇u)`.
- Hard BC enforcement via the multiplicative ansatz keeps the loss focused on the PDE residual alone — no boundary-loss weighting to tune.
- Variable coefficients are a common bridge from toy PDEs to physically meaningful media.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Full script

```python
--8<-- "tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py:code"
```
