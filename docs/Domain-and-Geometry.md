# Domain & Geometry

The `jno.domain` class manages mesh generation, physical group labelling, and sampling of collocation points. It wraps [pygmsh](https://github.com/nschloe/pygmsh) for mesh generation and [meshio](https://github.com/nschloe/meshio) for I/O.

---

## Constructing a domain

A `jno.domain` is built in one of three ways. **Reach for the meshed path first** — it is the only construction route that produces a triangulated/tetrahedral mesh, which is required for `.integrate()` (integration operators) and for any differential operator with `scheme="finite_difference"`. The polygon and point-cloud paths skip mesh generation and lose both capabilities; they exist for cases where you genuinely don't need a mesh.

### 1. Meshed domain — `jno.domain(constructor=...)` (recommended)

Pass any `pygmsh`-compatible callable to `jno.domain(constructor=...)`. The constructor builds an OpenCASCADE geometry, registers physical groups (the tags later accessed via `.variable(tag)`), and returns `(geo, spatial_dim, mesh_size)`.

```python
def my_geometry(mesh_size=0.1):
    def constructor(geo):
        p0 = geo.add_point([0, 0], mesh_size=mesh_size)
        # ... add lines, surfaces, physical groups ...
        geo.add_physical(surface, "interior")
        geo.add_physical(lines, "boundary")
        return geo, 2, mesh_size   # (geo, spatial_dim, mesh_size)
    return constructor

dom = jno.domain(constructor=my_geometry(mesh_size=0.05))
```

The [built-in geometries](#built-in-geometries) below (`jno.domain.rect`, `.disk`, `.cube`, …) are named constructors written against the same protocol — start there if your shape is in the table, switch to a custom constructor when it isn't.

You can also load a pre-meshed file (`.msh`, `.vtk`, `.med`, …) directly — physical groups in the file become `.variable(tag)` keys:

```python
dom = jno.domain('./mesh.msh')
```

### 2. Polygon-backed domain — `jno.PolygonDomain` (shapely)

For 2-D regions defined by a [shapely](https://shapely.readthedocs.io/) geometry where you only need points *inside* the region and not a mesh. CSG operations (union / difference / intersection) come for free via shapely, which makes irregular footprints, multi-region scenes, or "rectangle with a tilted hole" shapes easy to express.

```python
import jno
from shapely.geometry import box

# Square with a square hole
geo = box(0, 0, 1, 1).difference(box(0.4, 0.4, 0.6, 0.6))
dom = jno.PolygonDomain(geometry=geo)
x, y, _ = dom.variable("polygon", 256)   # 256 interior samples

# Or from a vertex list — simple polygons skip the shapely import
dom = jno.PolygonDomain(vertices=[[0, 0], [1, 0], [1, 0.5], [0.5, 1], [0, 1]])
```

**Trade-off:** no mesh is built — `.integrate()` and `scheme="finite_difference"` are unavailable on this domain. Use the meshed path above if you need either.

### 3. Point cloud — `jno.domain.from_array`

When the points are already known (sensor coordinates, observation grids, externally generated meshes), skip geometry entirely and pass arrays directly:

```python
import numpy as np

sensor_coords = np.array([[0.1, 0.2], [0.5, 0.5], ...])  # shape (N, 2)
dom = jno.domain.from_array({"obs": sensor_coords})

# Multiple tags can be packed into the same domain
dom = jno.domain.from_array({
    "interior_sensors": interior_coords,
    "boundary_sensors": boundary_coords,
})
```

**Trade-off:** same as `PolygonDomain` — no mesh, no integration, no FD.

---

## Built-in geometries

For common shapes, jno ships named constructors under `jno.domain.*`. Each is a small pygmsh script that follows the protocol of option 1 above and registers the listed physical groups.

| Constructor              | Dim | Signature                                                         | Physical groups                                                                  |
|--------------------------|-----|-------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `line`                   | 1D  | `line(x_range, mesh_size)`                                        | `interior`, `left`, `right`, `boundary`                                          |
| `rect`                   | 2D  | `rect(x_range, y_range, mesh_size)`                               | `interior`, `boundary`, `bottom`, `top`, `left`, `right`                         |
| `equi_distant_rect`      | 2D  | `equi_distant_rect(x_range, y_range, nx, ny)` — structured        | same as `rect`                                                                   |
| `disk`                   | 2D  | `disk(center, radius, mesh_size, num_points)`                     | `interior`, `boundary`                                                           |
| `l_shape`                | 2D  | `l_shape(size, mesh_size, separate_boundary=False)`               | `interior`, `boundary` (+ per-side groups when `separate_boundary=True`)         |
| `rectangle_with_hole`    | 2D  | `rectangle_with_hole(outer_size, hole_size, mesh_size, ...)`      | `interior`, `boundary`, hole sides                                               |
| `rectangle_with_holes`   | 2D  | `rectangle_with_holes(outer_size, holes=[{...}], mesh_size, ...)` | `interior`, `boundary`, `hole_boundary` (+ per-hole groups)                      |
| `rect_pml`               | 2D  | `rect_pml(..., pml_thickness_top, pml_thickness_bottom)`          | + `pml_top`, `pml_bottom` — wave-equation absorbing layers                       |
| `cube`                   | 3D  | `cube(x_range, y_range, z_range, mesh_size)`                      | `interior`, `boundary`, 6 face groups                                            |

Per-constructor quirks (PML thickness semantics, the `holes` list dict format, `separate_boundary` expansions) live in the constructor docstrings — reach them with `help(jno.domain.rect_pml)` and friends.

---

## Sampling Variables

```python
from jax import numpy as jnp
# Unpack spatial coordinates and (if time is set) a time variable
x, y, t = domain.variable("interior")

# Slice a specific coordinate range [0, None] means "all"
x, y = domain.variable("interior", (None, None))

# With boundary normals (outward unit normals at each boundary point)
xb, yb, tb, nx, ny = domain.variable("boundary", normals=True)

# With boundary normals AND view-factor matrix (for radiation problems)
xb, yb, tb, nx, ny, VF = domain.variable("boundary", normals=True, view_factor=True)

# Inject point data (sensor or observation locations)
xs, ys = domain.variable("sensor", 0.5 * jnp.ones((2, 1, 2)), point_data=True, split=True)

# Attach a tensor (e.g., spatially-varying PDE parameter, one row per batch sample)
k = domain.variable("k", jnp.array([[1.0], [2.0], [3.0]]))  # (B, 1)
```

---

## Time-Dependent Problems

```python
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(t_start, t_end, n_steps),
)

x, y, t   = domain.variable("interior")   # interior + time
x0, y0, t0 = domain.variable("initial")   # initial slice (t=0)
```

The solver automatically uses `jax.lax.scan` over time steps.

---

## Operator Learning (Multiple Batch Samples)

Multiply a domain by an integer `B` to replicate it across `B` independent batch samples. This is the standard setup for learning a PDE solution operator over a family of parameters.

```python
domain = 40 * jno.domain(constructor=jno.domain.rect(mesh_size=0.05))

# Attach B×4 parameter vectors
theta = ...  # shape (B, 4)
θ = domain.variable("θ", theta)
```

---

## Mesh Connectivity (for Finite Differences and Finite Elements)

Some schemes (e.g., `scheme="finite_difference"` in `jnn.grad` etc..) require the mesh topology to be pre-processed:

```python
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    compute_mesh_connectivity=True,   # pre-compute FD stencils
)
```

---

## Visualisation

```python
jno.domain.rect(x_range=(0,1), y_range=(0,1), mesh_size=0.1) # Create a domain (e.g., rectangle)
domain.variable("interior") # Set variables to trigger mesh generation
domain.plot("domain.png")   # saves a figure with mesh, boundaries, and normals
```

