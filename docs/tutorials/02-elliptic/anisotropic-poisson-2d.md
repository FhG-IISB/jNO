# Anisotropic Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/02_elliptic/anisotropic_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example modifies Poisson's equation so diffusion acts with different strength in the horizontal and vertical directions.

## Problem Setup

```text
-(a u_xx + b u_yy) = f(x,y),   (x,y) in [0,1]^2,
u = 0 on the boundary
```

with exact solution `u(x,y) = sin(pi x) sin(pi y)` and coefficients `a = 1`, `b = 3`.

## Step 1: Set Physical Coefficients

The script introduces separate constants `a` and `b` before building the residual. This is the simplest way to encode directional anisotropy.

```python
a = 1.0   # diffusion strength in x
b = 3.0   # diffusion strength in y
```

## Step 2: Create the Unit-Square Domain

Interior points are sampled on a rectangular domain and used to evaluate both the model and the manufactured forcing.

```python
domain = jno.domain.rect(mesh_size=0.1)
x, y, _ = domain.variable("interior")

u_exact = jno.np.sin(pi * x) * jno.np.sin(pi * y)
forcing  = (a + b) * pi**2 * u_exact
```

## Step 3: Impose Boundary Conditions Hard

The model output is multiplied by `x(1-x)y(1-y)`, so the field is zero on all four edges without an additional boundary loss.

```python
net = jno.nn.wrap(
    foundax.mlp(in_features=2, hidden_dims=64, num_layers=5,
                activation=jax.nn.tanh, key=jax.random.PRNGKey(12))
)
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 80, 0.5, end_value=1e-5)))

u = net(x, y) * x * (1 - x) * y * (1 - y)
```

## Step 4: Assemble an Anisotropic Residual

The residual uses weighted second derivatives in `x` and `y`, which is the main distinction from isotropic Poisson.

```python
pde = -(a * u.d2(x) + b * u.d2(y)) - forcing
```

## Step 5: Solve and Visualize

The script tracks error against the exact solution and plots exact, predicted, and absolute-error fields.

```python
crux    = jno.core([pde.mse])
history = crux.solve(40_000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
```

## What To Notice

- Anisotropy is often the first step beyond textbook Poisson problems.
- The only major PDE change is the weighted curvature in each coordinate direction.
- This pattern extends naturally to diffusion tensors and heterogeneous media.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/02_elliptic/anisotropic_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/anisotropic_poisson_2d.py"
```
