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
| `.structured(n=None)` | mesh an axis-aligned rect/box as a regular lattice (see below) |

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

### Cell shape — `.quad()`

Meshes are simplicial by default (triangles, tetrahedra). `.quad()` asks gmsh to recombine them into
**quadrilaterals**, which works on any 2-D geometry:

```python
d = Shape.disk(0, 0, 1, size=0.1).quad().domain()      # pure quads, curved boundary and all
(plate.quad() - hole).sized(0.05)                       # survives the plan operators, like .sized()
```

Like `size=` and `.curved()`, this is a property of the shape rather than an argument to the solve.

**3-D needs `.structured()`.** gmsh cannot hexahedral-mesh general geometry — `Recombine3DAll` on a
plain box returns 944 tetrahedra and no hexahedra — so hexes come from a regular lattice:

```python
d = Shape.box(0, 0, 0, 1, 1, 1).structured(n=16).quad().domain()    # 4096 hexahedra
d = Shape.rect(0, 0, 1, 1).structured(n=40).quad().domain()          # 1600 quadrilaterals
```

`.quad()` and `.structured()` compose in either order.

### Regular lattices — `.structured()`

`.structured()` meshes an axis-aligned `rect`/`box` as a regular grid instead of calling gmsh. Three
things follow that a gmsh mesh cannot give:

* **hexahedra**, as above — the lattice is the one 3-D plan that can be hex-meshed;
* **a grid descriptor** on `domain.grid`, which is what lets `jno.fdm` take its assembly-free 5-/7-point
  stencils instead of the cotangent operator, and what a nodal field reshapes against;
* **exactly matched opposite faces**, so a whole-domain periodic tie collapses onto one DOF rather
  than holding to a tolerance.

```python
d = Shape.rect(0, 0, 1, 1, size=0.1).structured().domain()   # counts from size=: 10x10 cells
d = Shape.box(...).structured(n=(32, 16, 16)).domain()       # explicit, per axis

d.grid          # {"shape": (Nx, Ny[, Nz]), "spacing": (...), "origin": (...)}
u.reshape(d.grid["shape"])                                    # nodes are C-ordered
```

`n` counts **cells**, so a lattice has `n + 1` nodes per axis and `d.grid["shape"]` is `n + 1` —
consistent with the `nx`/`ny`/`nz` of every other grid in jNO. A 128×128 *pixel* grid for a
foundation model is therefore `.structured(n=127)`. Omit `n` to derive it from the shape's `size=`.

A plan that cannot be a lattice — a CSG cut, a disk, a graded `size=` callable, `.curved()` — is
**refused by name** rather than quietly meshed with gmsh, since the caller who then reads `d.grid` or
expects hexes would otherwise fail somewhere else. The named faces (`left`/`right`/`bottom`/`top`,
plus `front`/`back` in 3-D), `boundary`, `interior`, and any `.name()`/`.attach()` region come with
the lattice as usual.

See the FEM guide's tensor-product section for what these support: volume terms and Dirichlet
conditions work; surface integrals, `order > 1` and h-adaptivity refuse by name.

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

### Material properties — `.attach(...)`

A region can carry its own material properties, read back off the domain as a **per-region
coefficient** ready to drop into a weak form. `d.k` is exactly the `d.by_region({...})` assembled from
every region that declared a `k`:

```python
kri = Shape.polygon(v).name("Kristall").attach(k=220.0, eps=0.794)
gas = Shape.polygon(w).name("Gas").attach(k=0.186, eps=1.0)
d   = (kri + gas).domain()

heat = d.k * (T.x*s.x + T.y*s.y) - d.q * s      # one equation, both materials
```

A value may be anything jNO treats as a coefficient: a scalar, an array, a symbolic expression, a
typed view (`ScalarView` / `VectorView` / `MatrixView` — the view type survives, so `d.eps @ u` still
works), or a trainable `jno.np.parameter`, so an attached property can be fitted or differentiated
through. A plain **function** is also accepted and is called with the domain's spatial coordinates
when the property is read (`.attach(k=lambda r, z: 2.0 + 0.5*z)`); it has to be deferred that way
because a spatially varying coefficient is built from `d.variable(...)`, which does not exist while
the geometry plan is being written.

`d.attached("eps")` gives the raw `{region: value}` mapping instead of the coefficient, for consumers
that need a dict — `gap.emissivity(d.attached("eps"))`.

Rules worth knowing:

* Apply after `.name(...)`; like `name`, a later transform drops the attachment, and attaching to a
  shape that was never named raises.
* Repeated calls merge (last wins), so properties can be built up in stages.
* `d.<name>` raises if **any** region failed to declare that name, listing the ones that did not — a
  forgotten material surfaces at first use rather than as a region that silently conducts nothing. Use
  `d.by_region({...}, default=...)` explicitly when some regions genuinely have none.
* Values must agree on one view type across regions; a matrix on one region and a vector on another
  raises rather than silently taking whichever came last.
* Properties are **declared, not typed** for volume regions: whether `eps` is used as a volume or a
  surface quantity is decided by the term that consumes it. Boundary tags are the exception — see
  `d.attach(...)` below, where the kind is resolved when it is declared.

A property can also be attached **after** the domain exists, which is the only way to attach to a
`domain.tag` — or to a mesh-file domain, which has no `Shape` to declare on:

```python
d.tag("wall", lambda x, y: x < 1e-9)
d.tag("lid",  lambda x, y: y > 1 - 1e-9)
d.attach("wall", h=25.0).attach("lid", h=5.0)     # a SURFACE property, per boundary facet
```

Here the kind is decided once, from what the target owns on this mesh: a tag owning boundary facets
is a surface quantity (`d.h` becomes a per-facet `by_tag` coefficient, see the `jno.fem` docs), a tag
owning only cells is a volume quantity, and a tag owning **both** is ambiguous and raises rather than
guessing — split it in two, or build the coefficient explicitly.

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

`interior` and `boundary` are **derived automatically** when the file does not define them, so a
mesh exported from anywhere is immediately usable — no physical groups required. The boundary is
topological (facets belonging to exactly one cell), which is also the only option for the many
files that store no surface block at all. Names the file *does* define always win: a physical group
called `boundary` keeps its own meaning, and only the missing tag is derived.

When the boundary falls into more than one connected shell, each is additionally exposed as
`boundary_0`, `boundary_1`, … so an internal cavity or a second body can be addressed on its own; a
single-shell part gains no numbered tags.

Two kinds of file are refused rather than half-loaded, both by name:

- **Mixed cell types** (a mesh with both tetrahedra and hexahedra, say). jNO assembles on one
  element family, so it would otherwise take the first block it recognised and silently ignore the
  rest — measured on a real gmsh benchmark as 70 % of the domain. The error reports the blocks and
  how many cells would be lost.
- **A surface (shell) mesh** — triangles in 3-D with no volume behind them, a routine CAD/STL
  export. Solving on a manifold is a different discretisation, not a missing setting.

Reader limits are reported as such rather than as jNO errors: an unsupported `.msh` version (only
2.2 and 4.1 are read) and gmsh element types meshio does not implement both say so, and say what to
re-export.

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

## Mesh-free sampling — what a PINN actually needs

`shape.domain()` does **not** mesh. A `Shape` knows its own extent, its own membership test and its
own boundary in closed form, so collocation points are drawn from the geometry directly:

```python
d = jno.Shape.box(0, 0, 0, 1, 1, 1).domain()          # no gmsh, in 1-D, 2-D or 3-D
x, y, z, t = d.variable("interior", sample=(20_000, None), split=True)
```

Two things differ from sampling a mesh. `20_000` means twenty thousand — there is no node set to be
clipped to — and every draw is **fresh**, so an adaptive strategy explores the region instead of
reshuffling one fixed cloud. Tagging is unchanged, and works on the boundary as well as the interior:

```python
d.tag("hot",   lambda x, y, z: (x-.5)**2 + (y-.5)**2 + (z-.5)**2 < .04)   # a lump of interior
d.tag("inlet", lambda x, y, z: x < 1e-9)                                  # a face
bx, by, bz, t, nx, ny, nz = d.variable("inlet", sample=(500, None), normals=True, split=True)
```

Which of the two a predicate means is *measured* — jNO draws from both and sees which one it accepts
— because a boundary predicate applied to an interior draw matches nothing and would look like an
empty region. Boundary points land on the analytic surface, so a disk's samples lie on the circle to
machine precision with exactly radial normals, not on a chord with a per-facet normal. The
primitives' auto-names (`left`, `arc`, `surface`, …) are available before any mesh exists.

!!! note "When a mesh does get built"
    Some things are defined *on* a mesh, not on the geometry: `jno.fem` and `fem_symbols()`,
    `.integrate()`, the finite-difference schemes, `.points` / `normals_by_tag` / `tag_indices`, and
    a facet predicate `f(x, n, names)`. Reading any of them builds the mesh once, and says so:

    ```
    INFO: _fem.__init__ needs a mesh; this domain was mesh-free — building it now.
    ```

    Nothing you write has to change; a tag declared while mesh-free gains its mesh-derived half at
    that point.

!!! warning "Plans that stay eager"
    A plan is mesh-free only if its extent, its membership **and** its boundary are all closed-form.
    These are not, and mesh at construction as before rather than being half-served: `sweep` and
    `fillet` (no analytic membership — `fillet` removes material near edges, so recursing to the
    child would answer for the un-filleted solid); `revolve` (membership yes, analytic boundary
    sampler not yet); `.name(...)` / `Shape.regions(...)` (region and interface tags are the
    mesher's conforming sub-bodies); and `.structured()` (already a lattice).

## Sampling, time, and batching

```python
x, y, z, t = d.variable("interior")                        # all interior mesh nodes
x, y, z, t = d.variable("interior", sample=(500, None))    # 500 sampled points
xb, yb, zb, tb, nx, ny, nz = d.variable("top", normals=True, split=True)   # boundary + outward normals
```

On a **mesh-free** domain the first line means something different: with no count there is no node
set to hand back, so it is one point redrawn every step (the convention `jno.domain.poly` already
uses). Pass an explicit `sample=(n, None)` when you want a fixed number.

`variable` always returns a trailing time coordinate `t` (a constant for steady domains), so a 3-D
domain unpacks as `(x, y, z, t)`. Time-dependent domains take `time=(t0, t1, n)`; `variable("initial")`
is the `t=0` slice. Multiply a
domain by an integer `B` (`B * jno.domain(...)`, or `B * shape.domain()`) to replicate it across `B`
operator-learning samples. `shape.domain(**kwargs)` forwards the *domain* arguments (`time=`, `sample=`,
`name=`, …) — but not a constructor or `mesh_size`, since size lives on the shape — so a full domain is
one line. `d.plot("domain.png")` renders the mesh, regions, and normals.

### Attaching a tensor

Pass an array instead of a sampling spec to attach it as a **tensor tag** — the operator-learning
pattern, where each batch sample carries its own field:

```python
dom = 256 * jno.domain(constructor=jno.domain.poseidon(nx=64, ny=64))
dom.variable("_f", forcing)          # (256, 64, 64, 1) — the shape you actually have
```

Context tensors are laid out `(B, T, ...)`. On a **steady** domain the time axis is `1` by
definition, so it is inserted for you: attach `(B, H, W, C)` and it is stored as `(B, 1, H, W, C)`.
Writing the axis yourself still works and is left untouched.

!!! warning "This is load-bearing, not cosmetic"

    Without a time axis the compiler reads the first grid axis *as* the timestep count and hands
    the expression a single slice — an `(4, 8, 5, 1)` attach used to arrive as `(5, 1)`, with the
    rest of the field discarded and no error raised.

    On a **time-dependent** domain axis 1 is genuinely ambiguous (is it `T`, or a grid axis?), so a
    mismatch raises instead: pass `T` entries, or `1` to share one field across all steps.

    Only tensors of rank ≥ 4 whose leading dimension is `B` or `1` are touched — a parameter like
    `(B, 1, 1)` or a shared lookup table is left exactly as given. Multiply the domain by `B`
    **before** attaching, so the batch count is known.

### Datasets larger than memory

The same slot accepts a **lazy** source — anything exposing `.shape` and `__getitem__`. The handle is
stored unread and sliced one batch at a time, so the dataset never has to fit:

```python
dom.variable("_f", h5py.File("well.h5")["forcing"])     # or zarr, tensorstore, np.memmap
crux.solve(epochs=600, batchsize=32, offload_data=True)
```

No new dependency and no new argument — the contract is duck-typed, so jNO imports none of those
libraries. `offload_data=True` is required, because the on-device path holds the whole array; asking
for it with a lazy source raises and names the fix.

!!! note "What a lazy source gives up"

    Indices are **sorted** before each gather, since h5py and zarr only accept increasing fancy
    indices. That reorders samples within a batch — invisible to a mean-reduced loss, but a run is
    not bitwise identical to one recorded before this existed.

    The `(B, T, ...)` time axis is **validated, not inserted**: rewriting the layout would mean
    reading the source, which is the one thing it exists to avoid. Store it with the time axis, or
    pass an eager array.

    **Adaptive resampling cannot target a lazy tag** — it replaces the whole point set each round.
    It raises rather than silently skipping.
