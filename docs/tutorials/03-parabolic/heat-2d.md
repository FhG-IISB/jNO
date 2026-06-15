# Heat 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/03_parabolic/heat_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/03-parabolic/">Back to chapter</a>
</div>

This example extends the heat equation to a square domain and shows how to inspect the learned solution at multiple time slices.

## Problem Setup

The PDE is `u_t = alpha Delta u` on the unit square with homogeneous Dirichlet boundaries and a sinusoidal initial state.

## Step 1: Build the 2D Space-Time Geometry

The script samples interior space-time points on a rectangular domain and uses a separate initial-time slice for the starting condition.

```python
α = 0.1
T_end = 0.5
N_t = 4

domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t   = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

u_exact = jno.np.exp(-2 * α * π**2 * t) * jno.np.sin(π * x) * jno.np.sin(π * y)
```

## Step 2: Use a DeepONet With a Hard Spatial Envelope

The model output is multiplied by `x(1-x)y(1-y)` so the boundary is satisfied on all four edges.

```python
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1, coord_dim=2, n_outputs=1,
        n_layers=4, basis_functions=96, hidden_dim=64,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(...)))

xy = jno.np.concat([x, y])
xy0 = jno.np.concat([x0, y0])

u  = net(t,  xy)  * x  * (1 - x)  * y  * (1 - y)
u0 = net(t0, xy0) * x0 * (1 - x0) * y0 * (1 - y0)
```

## Step 3: Combine PDE and Initial Losses

The transient residual enforces the heat equation, while a dedicated initial-condition residual anchors the solution at `t = 0`.

```python
pde = u.d(t) - α * jno.np.laplacian(u, [x, y])
ini = u0 - jno.np.sin(π * x0) * jno.np.sin(π * y0)

crux    = jno.core([pde.mse, ini.mse])
history = crux.solve(40000)
```

!!! tip "Autodiff vs finite-difference laplacian"
    On a 2-D mesh you can swap the autodiff Laplacian for the FD version by passing `scheme="finite_difference"` to `.laplacian(...)`:

    ```python
    pde = u.d(t) - α * jno.np.laplacian(u, [x, y], scheme="finite_difference")
    ```

    Two prerequisites: the domain must be created with `compute_mesh_connectivity=True` (so the FD stencils are precomputed at mesh build time), and the mesh must be regular enough that the stencil is well-defined. The FD path is **substantially cheaper per step** for dense interior meshes because it skips the autodiff tape, but it pays mesh-resolution error. A useful sanity check is to run the same PDE twice (once with each scheme) and confirm the two solutions agree to within `O(h²)`.

## Step 4: Plot Time Snapshots

One of the nice features of this script is explicit evaluation on selected time slices so you can inspect how the field evolves.

```python
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
```

## What To Notice

- This is the natural 2D extension of Heat 1D.
- Snapshot evaluation is a good debugging tool for time-dependent solves.
- The same ideas generalize to more complex transient PDEs.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/03_parabolic/heat_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/heat_2d.py"
```
