# Domain & Geometry

`jno.domain` holds a meshed geometry, its named regions, and the collocation points sampled on
them. Build the geometry with the **`Shape`** DSL (gmsh-OpenCASCADE), load a mesh file, or pass a
point cloud.

```python
import jno
from jno import Shape

solid = (Shape.rect(0, 0, 4, 1) - Shape.disk(2, 1, 0.4)).extrude(0.6)   # a strip under a roll
d = solid.domain()                                                      # one-liner; == jno.domain(solid)

x, y, z, t = d.variable("interior")                                     # coords (+ a trailing time coord t)
xb, yb, zb, tb, nx, ny, nz = d.variable("top", normals=True, split=True)  # a named boundary + outward normals
```

---

## The `Shape` DSL

A `Shape` is an immutable build-plan: make **primitives**, combine them with **boolean operators**,
reshape with **transforms**, then hand the result to `jno.domain`. Every call returns a new shape.

### Primitives

Each takes an optional `size=` (target element size near it).

| Primitive | Auto-named boundaries |
|---|---|
| `Shape.rect(x0, y0, x1, y1)` | `left` `right` `top` `bottom` |
| `Shape.disk(cx, cy, r)` | `arc` |
| `Shape.polygon(points)` | `e0 e1 … eN` (one per edge) |
| `Shape.box(x0, y0, z0, x1, y1, z1)` | `left right top bottom front back` |
| `Shape.cylinder(x, y, z, dx, dy, dz, r)` | `side` `top` `bottom` (any axis) |
| `Shape.sphere(cx, cy, cz, r)` | `surface` |
| `Path(…).face()` | per-segment (see [Contours](#contours-with-arcs)) |

`interior` (the area/volume) and `boundary` (all of it) are always present.

### Combine

```python
a - b     # cut         a | b     # fuse         a & b     # intersect
```

### Transforms

| Transform | Meaning |
|---|---|
| `.extrude(h)` | sweep a 2-D shape straight up into 3-D |
| `.revolve(axis_point, axis_dir, angle=2π)` | rotate a 2-D profile about an axis (x- or y-axis through the origin); a partial angle gives a wedge/half-donut |
| `.sweep(path)` | sweep a 2-D profile along a smooth `Path` (bent pipe) |
| `.array(n, step=v)` / `.array(n, about=(pt, dir), angle=2π)` | `n` fused copies — linear or polar |
| `.translate(v)` · `.rotate(axis_point, axis_dir, angle)` | place/orient (names are preserved) |
| `.fillet(radius, where=None)` | round edges; `where=f(x,y,z)` selects them by midpoint |
| `.sized(size)` | set the target mesh size (scalar or `f(x,y,z)`) |

`extrude` is a special case of `sweep`; `revolve` is a distinct rotation. Mesh size lives on the
**shape** it describes, not on `jno.domain`.

### Contours with arcs

`Path` chains `line_to`/`arc_to` segments (points are 3-D, `z` defaults to 0); `.face()` closes it
into a 2-D shape. This is the general 2-D generator — a diameter plus a semicircular arc revolves
into a sphere:

```python
from jno import Path

sphere = (
    Path(0, -1)
    .line_to(0, 1, name="axis")            # the diameter, on the revolve axis
    .arc_to(0, -1, through=(1, 0), name="dome")   # a semicircular arc (3-point arc)
    .face()
    .revolve((0, 0, 0), (0, 1, 0))         # -> a sphere; the "dome" arc names its surface
)
```

An open `Path` is instead a **sweep trajectory**. Segments may be `name=`'d as you draw them; the name
flows onto the swept face. A sharp `line_to→line_to` corner is rejected (round it with `arc_to`).

### Naming & selection

Primitives auto-name their boundaries (table above); rename, merge, or carve regions afterwards with
`d.tag`, which accepts:

```python
d.tag("inlet",  lambda x, y, z: x < 1e-6)                       # a coordinate predicate
d.tag("outlet", lambda x, n, name: (n[:, 0] > 0.9) & (name != "left"))  # coords + outward normal + current name
```

The second form selects **boundary facets** by position, orientation, and existing name — inclusion
and exclusion in one predicate. `n` is the outward normal (boundary only), `name` each facet's
current region.

A coordinate predicate works in **1-D too**, where a boundary facet is a single vertex: `d.tag("right_end",
lambda x: x > 1 - 1e-9)` names an endpoint, carries its outward normal (`-1` at the left end, `+1` at the
right), and a `u(right_end) - g` term is an ordinary Dirichlet condition.

To name a region **and** grab its coordinates in one line, pass the predicate straight to
`variable` (it forwards to `tag`, then returns the split coords):

```python
xl, yl, zl, _ = d.variable("left", where=lambda x, y, z: x < 1e-6)   # tag AND bind in one call
```

### Mesh density

Control element size by attaching `size=` to a shape — gmsh's mesh-size fields do the work
(a Distance+Threshold field near a shape, a size callback for `f(x,y,z)`), combined by `min`:

```python
Shape.disk(2, 1, 0.4, size=0.02)               # fine in the band around this shape's boundary
(strip - roll).sized(0.05)                      # a global size cap for the whole shape
solid.sized(lambda x, y, z: 0.03 + 0.10 * y)    # graded: denser where the function is smaller
```

So "denser in a region" = give a small `size=` to the shape covering it, or a callable that is small
there.

### Multi-material regions

Name each material with `.name(...)` and combine with `+` to build a **multi-material** domain: the
sub-shapes are fragmented so element edges align exactly with every material interface (a *conforming*
mesh), and each cell is assigned to the first region — left-to-right order is priority — whose shape
contains its centroid. Each region becomes its own variable set; the outer boundary keeps its
auto-names and internal interface facets are not boundary.

```python
plate = Shape.rect(0, 0, 2, 1)
core  = Shape.disk(1, 0.5, 0.3)
d = (core.name("inclusion") + plate.name("matrix")).sized(0.05).domain()

d.variable("inclusion")     # the disk's cells/points (a distinct material)
d.variable("matrix")        # everything else
d.boundary_tags()           # {left, right, top, bottom, boundary} — the plate's outer edges only
```

`+` keeps the pieces as distinct materials with a conforming interface — unlike `|` (fuse), which
merges them into one. It composes n-ary (`a + b + c`), and `Shape.regions(inclusion=core,
matrix=plate)` is the equivalent keyword form. Regions may overlap: here the disk overlaps the plate,
and `inclusion` wins inside the disk because it is listed first. Use each region tag to restrict a
`jno.fem` term or coefficient to that material (per-region volume integration), or to sample it in a
PINN. A multi-material shape is a top-level construct — call `.domain()` on it directly; it does not
compose with boolean operators or transforms.

Region names that are not valid Python identifiers go through the dict form:
`Shape.regions({"Quartz.1": q1, "Quartz.2": q2})`. Dict order is priority order exactly as for
keywords, and the two forms combine (dict entries first).

**Interfaces** between materials are auto-named by the region pair, sorted — `d.variable("inclusion|matrix")`
gives *every* facet where those two materials meet (however many flat faces that spans — an E-core's
`air|core` boundary is one tag, not one per face). Impose a coupling/flux condition there, or sample it.
Interfaces are listed by `d.interface_tags()` and kept **out** of `d.boundary_tags()` (an interface is not
the outer boundary). When an interface is *topologically disjoint* — two separate inclusions, or the two
stacked layers of a winding — the connected pieces are additionally exposed as `"a|b.0"`, `"a|b.1"`, …
alongside the union `"a|b"`.

---

## Worked examples

=== "Bolt-circle plate"

    ```python
    holes = Shape.disk(3, 0, 0.3).array(6, about=((0, 0, 0), (0, 0, 1)))   # 6 holes in a ring
    d = jno.domain((Shape.rect(-5, -5, 5, 5) - holes).extrude(0.4))
    ```

=== "Filleted, drilled bracket"

    ```python
    part = (Shape.box(0, 0, 0, 8, 4, 1) - Shape.cylinder(2, 2, -1, 0, 0, 3, 0.5)) \
        .fillet(0.2, where=lambda x, y, z: z > 0.9)                          # round the top edges
    d = jno.domain(part)
    ```

=== "Bent pipe (sweep)"

    ```python
    pipe = Shape.disk(0, 0, 0.4).sweep(Path(0, 0, 0).arc_to(2, 0, 2, through=(0.6, 0, 1.4)))
    d = jno.domain(pipe)
    ```

=== "Graded roll-gap"

    ```python
    strip = Shape.rect(0, 0, 4, 1, size=0.1)          # coarse in the bulk
    roll  = Shape.disk(2, 1, 0.4, size=0.02)          # fine near the contact arc
    d = jno.domain((strip - roll).extrude(0.6))       # the carved arc is auto-named "arc"
    xc, yc, zc, tc, nx, ny, nz = d.variable("arc", normals=True, split=True)   # contact points + normals
    ```

---

## Other domain sources

**Pre-meshed file** — physical-group names become region tags:

```python
d = jno.domain("part.msh")     # .msh / .vtk / .med … built anywhere (gmsh, CAD, …)
```

**Point cloud** — coordinates you already have (no mesh, so no `.integrate()`/FD):

```python
d = jno.domain.from_array({"interior": interior_coords, "boundary": boundary_coords})
```

**1-D domains** — an open path, exactly like every other dimension's `Shape`:
`jno.Path(0, 0).line_to(1, 0).curve(size=0.01).domain()` (ends named `left`/`right`). This is the
form used throughout the docs; the `jno.domain.line(...)` shorthand still exists. `jno.domain`
also keeps the structured grids `equi_distant_rect` / `poseidon` and the point-cloud `from_array`.
For 2-D/3-D geometry build the shape with `Shape` — `Shape.rect(...).domain()`,
`Shape.box(...).domain()`, and so on. (Shapely geometries and vertex lists are also still accepted.)

---

## Sampling, time, and batching

```python
x, y, z, t = d.variable("interior")                        # all interior mesh nodes
x, y, z, t = d.variable("interior", sample=(500, None))    # 500 sampled points
xb, yb, zb, tb, nx, ny, nz = d.variable("top", normals=True, split=True)   # boundary + outward normals
```

`variable` always returns a trailing time coordinate `t` (a constant for steady domains), so a 3-D
domain unpacks as `(x, y, z, t)`. Time-dependent domains take `time=(t0, t1, n)`; `variable("initial")`
is the `t=0` slice. Multiply a
domain by an integer `B` (`B * jno.domain(...)`, or `B * shape.domain()`) to replicate it across `B`
operator-learning samples. `shape.domain(**kwargs)` forwards the *domain* arguments (`time=`, `sample=`,
`name=`, …) — but not a constructor or `mesh_size`, since size lives on the shape — so a full domain is
one line. `d.plot("domain.png")` renders the mesh, regions, and normals.
