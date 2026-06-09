# Poisson 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/poisson_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to chapter</a>
</div>

The soft-BC + finite-difference companion to [Laplace 1D](laplace-1d.md). Same domain, harder PDE, two pedagogical patterns at once: **soft Dirichlet enforcement** via a boundary loss term, and **finite-difference derivatives** via `scheme="finite_difference"` instead of autodiff.

## Problem Setup

```text
-u''(x) = sin(π x),   x in [0, 1],   u(0) = u(1) = 0
```

Exact solution: `u(x) = sin(π x) / π²`.

## Step 1: Sample Interior and Boundary Points

Unlike the Laplace example, this script asks for **both** the interior and the spatial boundary points, since the BCs will be enforced through a separate loss term:

```python
domain = jno.domain.line(mesh_size=0.1)
x, _  = domain.variable("interior")
xb, _ = domain.variable("boundary")
```

## Step 2: Bare Network — No Multiplicative Ansatz

```python
u_net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=64, num_layers=4, key=jax.random.PRNGKey(0))
).optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

u = u_net(x)
```

## Step 3: PDE Residual with `scheme="finite_difference"`

```python
pde = -u.d2(x, scheme="finite_difference") - jno.np.sin(π * x)
bc  = u_net(xb)   # soft: u(0) = u(1) = 0
```

`.d2(x, scheme="finite_difference")` computes the second derivative via a precomputed FD stencil over the mesh, instead of using `jax.grad` twice. For dense interior batches the FD path is **much cheaper** per step than autodiff because there's no per-call gradient tape, but it costs accuracy that scales with the mesh size.

!!! tip "When to choose finite-difference over autodiff"
    Both `scheme="automatic_differentiation"` (the default) and `scheme="finite_difference"` produce the same residual up to mesh-resolution error. Pick:

    - **Autodiff** when you have irregular collocation, when accuracy matters more than throughput, or when the network architecture makes per-step gradient computation cheap.
    - **Finite-difference** when you have a dense regular mesh, when the batch size is large enough that the per-step gradient tape dominates training time, or when you want a sanity-check on a tricky autodiff residual. Requires the mesh connectivity to be known — for 2-D / 3-D meshes you also need `compute_mesh_connectivity=True` on the domain.

## Step 4: Solve With Two Constraints

```python
crux    = jno.core([pde.mse, bc.mse])
history = crux.solve(5000)
```

The optimiser balances the interior PDE residual against the boundary mismatch. Constraint weighting (`crux.solve(..., constraint_weights=...)`) is the lever if the BC loss converges slower than the PDE term — see [Schedules](../../adaptive/schedules.md).

## What To Notice

- Soft BCs add a loss term to manage but generalise to arbitrary geometries and non-zero BC values where multiplicative ansatzes are awkward.
- `scheme="finite_difference"` is a per-call switch — you can mix FD and autodiff in the same residual if you want.
- The exact solution `sin(π x)/π²` naturally satisfies `u(0)=u(1)=0`, so the soft BCs only need to break the **other** local minima where the network solves the PDE residual exactly but with a non-zero constant offset.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/poisson_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to 01 Basics</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/01_basics/poisson_1d.py"
```
