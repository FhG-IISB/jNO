# Integral Constraints and Flux Monitoring (2-D)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/06_integration/flux_conservation_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/06-integration/">Back to chapter</a>
</div>

This example solves the 2-D Poisson equation and shows two uses of `.integrate()` that are not possible with pointwise losses like `.mse`:

1. **Tracking a physical observable** — the volume mean `∫_Ω u dA` is logged throughout training without entering the gradient.
2. **Soft integral constraint** — the same quantity can be added to the loss to accelerate convergence when the PDE residual alone is slow to pin the solution's magnitude.

## Problem Setup

```text
−∇²u = 2π² sin(πx) sin(πy),   (x,y) ∈ [0,1]²
u = 0 on ∂Ω
```

Exact solution: `u(x,y) = sin(πx) sin(πy)`

The exact volume mean is `∫₀¹∫₀¹ sin(πx)sin(πy) dxdy = 4/π² ≈ 0.405`.

## Why integrals matter here

The Dirichlet condition forces `u = 0` on the entire boundary.  A network that memorises the boundary data without learning the interior would give `u ≈ 0` everywhere — satisfying the BC but not the PDE.  Tracking `∫_Ω u dA` during training immediately reveals this failure mode: if the integral stays near zero, the network has not learned the interior peak.

## Step 1: Define volume and boundary variables

```python
x,   y,   _ = domain.variable("interior")   # volume points
x_b, y_b, _ = domain.variable("boundary")   # boundary points
```

## Step 2: Hard-enforce the Dirichlet BC

The model output is multiplied by `x(1−x)y(1−y)`, which is zero on all four edges.  The network only needs to learn the interior shape.

```python
u = net(jno.np.concat([x, y], axis=-1)) * x * (1 - x) * y * (1 - y)
```

## Step 3: Add the PDE residual

```python
pde = -u.laplacian(x, y) - forcing
```

## Step 4: Track the volume integral

```python
from jno.numpy import tracker

TARGET = 4.0 / jnp.pi ** 2   # ≈ 0.405

vol_mean = tracker(u.integrate(), interval=200)
```

`tracker(...)` wraps any scalar expression so that it is evaluated and logged every `interval` epochs but **does not contribute to the gradient**.

## Step 5: Optionally add an integral constraint

To enforce the prescribed mean in the loss itself, add:

```python
integral_loss = (u.integrate() - TARGET).square()
losses = [pde.mse, integral_loss, vol_mean]
```

Without the constraint the PDE residual alone is usually sufficient.  With it, the optimizer receives a direct signal about the solution's global magnitude, which can be useful when the interior is poorly sampled or the PDE residual gradient is small.

## Step 6: Solve

```python
crux = jno.core(losses, domain)
crux.solve(30_000)
```

## What to notice

- `.integrate()` returns a scalar placeholder — it can appear anywhere a regular loss term can.
- The region (volume vs boundary) is inferred automatically from the variable's tag; no extra argument is needed.
- Integration weights are precomputed once at domain creation. They are embedded as JAX constants and reused across all training steps, so adding an integral term carries negligible runtime cost.
- Because `.integrate()` is differentiable, `jax.grad` and `eqx.filter_grad` work through it without modification.

## Flux integrals (extension)

If you also want to monitor the outward heat flux through the boundary, request normals and compute F·n by hand:

```python
x_b, y_b, _, nx, ny = domain.variable("boundary", normals=True)
u_b = net(jno.np.concat([x_b, y_b], axis=-1)) * x_b * (1 - x_b) * y_b * (1 - y_b)

# ∫_∂Ω ∂u/∂n ds  —  should equal ∫_Ω ∇²u dV = −∫_Ω f dV by Green's identity
outward_flux = (u_b.d(x_b) * nx + u_b.d(y_b) * ny).integrate()
flux_tracker  = tracker(outward_flux, interval=500)
```

jno does not dot F with n automatically — you specify the integrand explicitly.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/06_integration/flux_conservation_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/06-integration/">Back to 06 Integration</a>
</div>

## Script

```python
--8<-- "tutorial_examples/06_integration/flux_conservation_2d.py"
```
