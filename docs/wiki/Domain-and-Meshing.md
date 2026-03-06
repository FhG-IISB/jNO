# Domain and Meshing

The `jno.domain` class manages computational domains, mesh generation, point sampling, and data attachment for training and inference.

## Creating a Domain

### Built-in Geometries

jNO provides built-in constructors for common geometries:

```python
import jno

# 1D line segment
domain = jno.domain(constructor=jno.domain.line(x_range=(0, 1), mesh_size=0.01))

# 2D rectangle
domain = jno.domain(constructor=jno.domain.rect(x_range=(0, 1), y_range=(0, 1), mesh_size=0.05))

# 2D disk
domain = jno.domain(constructor=jno.domain.disk(mesh_size=0.05))

# 2D equidistant rectangle (uniform grid)
domain = jno.domain(constructor=jno.domain.equi_distant_rect(
    x_range=(0, 1), y_range=(0, 1), nx=64, ny=64
))
```

### Custom Geometries (pygmsh)

You can define arbitrary geometries using pygmsh syntax:

```python
def my_geometry(mesh_size=0.1):
    def construct(geo):
        x0, x1 = 0, 1
        y0, y1 = 0, 1

        points = [
            geo.add_point([x0, y0], mesh_size=mesh_size),
            geo.add_point([x1, y0], mesh_size=mesh_size),
            geo.add_point([x1, y1], mesh_size=mesh_size),
            geo.add_point([x0, y1], mesh_size=mesh_size),
        ]

        lines = [
            geo.add_line(points[0], points[1]),
            geo.add_line(points[1], points[2]),
            geo.add_line(points[2], points[3]),
            geo.add_line(points[3], points[0]),
        ]

        curve_loop = geo.add_curve_loop(lines)
        surface = geo.add_plane_surface(curve_loop)

        # Physical groups for sampling
        geo.add_physical(surface, "interior")
        geo.add_physical(lines, "boundary")
        geo.add_physical([lines[0]], "bottom")
        geo.add_physical([lines[1]], "right")
        geo.add_physical([lines[2]], "top")
        geo.add_physical([lines[3]], "left")

        return geo, 2, mesh_size  # geometry, spatial_dim, mesh_size

    return construct

domain = jno.domain(constructor=my_geometry(mesh_size=0.05))
```

### Loading External Meshes

Import meshes from `.msh` files (Gmsh format):

```python
domain = jno.domain('./runs/mesh.msh')
```

---

## Time-Dependent Domains

Add a time dimension with the `time` parameter:

```python
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, 1, 10),  # (t_start, t_end, n_steps)
    compute_mesh_connectivity=False,
)
```

---

## Sampling Variables

Once a domain is created, sample variables from tagged mesh regions:

```python
# Interior points — returns one Variable per spatial dimension
x, y = domain.variable("interior")

# With explicit shape hints
x, y = domain.variable("interior", (None, None))

# Boundary points with normals and view factors
xb, yb, nx, ny, VF = domain.variable("boundary", (None, None), normals=True, view_factor=True)

# 1D domain
(x,) = domain.variable("interior")

# Time-dependent domain
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")
```

---

## Attaching Tensor Data

Attach additional tensor-valued data to the domain for operator learning or parameterized problems:

```python
import jax.numpy as jnp

# Scalar parameter per batch — shape (B, 1)
k = domain.variable("k", jnp.array([[1.0], [1.0]]))

# Parameter vectors per batch — shape (B, 4)
theta_train = jnp.ones((B_train, 4))
(θ,) = domain.variable("θ", theta_train)
```

Tensor data can be indexed with standard bracket notation:

```python
# Access individual components
κ = jnn.exp(θ[0] + θ[1] * jnn.sin(x))
```

---

## Point Data (Sensors)

Add specific point locations as sensor data:

```python
# Sensor points — shape (B, N, dim)
xs, ys = domain.variable(
    "sensor",
    0.5 * jnp.ones((2, 1, 2)),
    point_data=True,
    split=True
)
```

---

## Batched Domains (Operator Learning)

Multiply a domain by an integer to create batched copies for operator learning:

```python
B_train = 4
domain = B_train * jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
```

This tiles the mesh so that each batch element shares the same geometry but can have different tensor data (e.g., different PDE coefficients).

---

## Mesh Connectivity

For certain operations (view factors, normals), enable mesh connectivity computation:

```python
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    compute_mesh_connectivity=True,
)
```

Set `compute_mesh_connectivity=False` when these features are not needed (e.g., for inference domains).

---

## Plotting the Domain

Visualize the domain and its sampled points:

```python
domain.plot("domain.png")
```

---

## See Also

- [Training and Solving](Training-and-Solving.md) — using domains in the training pipeline
- [Operator Learning](Operator-Learning.md) — batched domains for operator learning
