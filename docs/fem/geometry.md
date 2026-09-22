# Geometry, regions, and moving meshes

## Mesh sizing — one number, or a function of position

`size=` takes a float for a uniform mesh, or a **callable** for a graded one. A callable becomes a
gmsh mesh-size callback and composes with every other size control by `min`, so it is the general
"denser here" knob:

```python
# THREE arguments, in 2-D as well as 3-D: gmsh calls a size function as f(x, y, z).
h_of = lambda x, y, z: H_FINE + (H_COARSE - H_FINE) * min(1.0, max(0.0, (LY - y) / BAND))
d = jno.shape.rect(0.0, 0.0, LX, LY, size=h_of).domain()
```

This is what makes a thin feature affordable. Measured on a 1.2 × 0.4 mm rectangle graded from 4 µm at
the top surface to 30 µm below — the melt pool of the
[laser melt-pool tutorial](../tutorials/08-fem-and-varpinns/melt-pool-laser.md), ~50 µm deep in a
400 µm domain:

| mesh | nodes | through the 50 µm feature |
|---|---|---|
| uniform 16 µm | 2,296 | 3 cells |
| uniform 4 µm | 35,226 | 12 cells |
| **graded 4 → 30 µm** | **2,035** | **12 cells** |

Same near-surface resolution as the uniform 4 µm mesh for **17× fewer nodes** — and fewer nodes than
the 16 µm mesh that resolved nothing.

!!! danger "`f(x, y)` is the natural thing to write in 2-D, and it is wrong"
    The callback passes three coordinates whatever the dimension. A two-argument function raises
    inside gmsh's C callback, where it surfaced as `Wrong mesh element size lc = 0 (lcmin = 0,
    lcmax = 1e+22)` — naming neither the callable, nor its signature, nor the shape it came from. And
    whether it surfaced *at all* depended on gmsh's global state: the same wrong function raised in a
    fresh process and meshed **silently** in one that had already built other meshes. jNO now probes
    the callable once before registering it and refuses by name.

!!! note "Meshing is lazy"
    `.domain()` does not build a mesh; the first thing that asks for one does. So a size callable is
    not called — and a wrong one not diagnosed — until then.

## Curved (isoparametric) geometry — `shape.curved()`

By default jNO meshes straight-sided and *synthesises* higher-order nodes at the straight-edge
midpoints, so the domain stays a polygon however high the element order goes. That approximation
carries an **O(h²) domain error at every basis order** — it is what caps P2/P3 at second order on a
round boundary, no matter how good the basis is. `curved()` asks the CAD kernel to place those nodes on
the true surface instead:

```python
d = jno.shape.disk(0, 0, 1, size=0.1).curved().domain()
u, v = d.fem_symbols(order=2)          # the basis order must MATCH the geometry order
```

!!! measured "Unit disk, `-Δu = 1`, exact `u = (1−r²)/4` — RMS nodal error"
    Measured on `-Δu = 1` on the unit disk (exact `u = (1−r²)/4`), RMS nodal error:

    | mesh size | straight-sided | curved |
    |---|---|---|
    | 0.4 | 6.73e-03 | 7.63e-05 |
    | 0.2 | 1.66e-03 | 7.67e-06 |
    | 0.1 | 4.23e-04 | 7.36e-07 |
    | **rate per halving** | **≈4× (O(h²))** | **≈10× (O(h³))** |

    Straight-sided is capped at second order by the geometry; curved recovers P2's own third order, and is
    **570× more accurate** at the finest resolution.

**Scope — what this does not cover.** Order 2 and simplices only. An **order mismatch is refused**:
isoparametric means geometry order == basis order, and a curved mesh under a P1 basis puts the midside
DOF coordinates (on the arc) and the geometric map (from the chord) in disagreement. **Non-nodal
families keep affine geometry** — Nédélec, RT, Argyris and Morley need Piola/curvature push-forwards
that are a separate change, so curved EM does *not* benefit yet — and a **4th-order form is refused** on
a curved cell, because the physical-Hessian transform is derived for an affine map and would gain a
curvature term it does not carry. **Facet normals are still straight-facet**, so the O(h) normal error
that affects radiation view factors, flux BCs and RCWA's field decomposition is unchanged by this.

Note also that a curved map makes the integrand rational, so **no quadrature rule is exact** any more.
The default degree is raised by 2 on curved cells and `jno.fem(quad_degree=...)` still overrides;
measured on the study above, refining the rule moves the answer by less than 0.01 %.

---

## Differentiable mesh geometry — trainable coordinates (`.trainable()`)

Any placeholder promotes to a `jno.np.parameter` seeded at its current values with **`.trainable()`** —
an existing coefficient / data tag becomes an inverse unknown in one call:

```python
k = domain.variable("kappa", sample=k0).trainable()   # trainable coefficient, seeded at k0
```

Called on a **spatial coordinate** (`domain.variable(region)`), `.trainable()` makes that region's **mesh
vertices** a design variable — the map from node positions to the solution is differentiable, so a solve
can be optimized *with respect to the mesh itself* (mesh relocation / r-adaptivity / shape optimization),
all in one JAX graph:

```python
xi, yi, _ = domain.variable("core", split=True)   # a where= / predicate sub-region
Xx = xi.trainable()                                # ONLY the x-positions of the core vertices move
#   differentiating a solve now yields the shape derivative ∂(solve)/∂X
```

!!! warning "`.trainable()` is per component — call it on every axis you want to move"
    The spelling is **literal, per component** (`x.trainable()` moves only x; call it per axis for full
    motion — which also gives constrained relocation for free, e.g. promote only the tangential component on a
    slip plane). Under the hood the coordinate parameter scatters into the assembly's P1 geometry *before* the
    element Jacobian is formed, so `J`, `JxW`, the physical gradients, the quadrature-point coordinates **and
    the boundary-facet normals** all become differentiable in the node positions. Scope: nodal-Lagrange volume
    + Neumann/Robin terms, 2D triangle / 3D tet, steady. The mesh **connectivity is fixed** — this is
    *relocation*, not remeshing (h-remeshing stays the non-differentiable outer AFEM loop); it is differentiable
    on valid meshes, with element inversion (tangling) the boundary of that regime.

**r-adaptivity in one call.** Tagging coordinates `.trainable()` and driving the relocation yourself is the
low-level path; the packaged form reuses the **same `adapt=` slot** as h-refinement:

```python
xm, ym, _ = domain.variable("core", where=interior, split=True)
xm.trainable(); ym.trainable()                              # BEFORE jno.fem(...)
u = fem.solve(adapt=jno.solve.relocate(max_iters=60))
```

??? note "How the descent works"
    `jno.solve.relocate()` descends the **equidistribution defect** of an arclength monitor through the
    differentiable solve, with a **backtracking `det J` line search** — so the fixed node set concentrates at
    solution features and the mesh never tangles (the validity constraint lives in the step control; a stock
    optimiser or an energy barrier alone cannot guarantee it on a stiff problem — see `run_adaptive_relocate`).
    It mutates the domain to the relocated mesh, returns the solution there, and **raises** if no coordinate was
    tagged. Works across **linear, nonlinear (Newton), transient (relocates for the whole trajectory via a
    time-averaged objective), periodic, and complex** problems, scalar or vector — the objective sums over every
    solution block, so a complex field's real and imaginary parts both contribute. Only complex-*transient* is
    not wired yet.

### A mesh objective that names the physics — `objective=<expression>`

The three built-in objectives (`"equidistribution"`, `"energy"`, `"huang"`) are mesh-*quality* measures:
they read the solution only through a monitor, so they can ask for resolution but cannot state a goal
about the physics. `objective=` also accepts a **weak-form expression**, assembled exactly as
`criterion=` is and summed to a scalar, over a **volume or a boundary** region:

```python
xs, ys, nx, ny = domain.variable("side", normals=True, split=True)
ys.trainable()                                   # the wall may move along y only
us, vs = u.bind(x=xs, y=ys), v.bind(x=xs, y=ys)
fem.solve(adapt=jno.solve.relocate(objective=(us[0]*nx + us[1]*ny)**2 * vs[0]))
```

That is a **free surface**: the wall moves until the flow through it vanishes. The facet normals are
rebuilt from the moving vertices, so `n` is the *current* mesh's normal. The gradient runs through the
solve — matched to central differences at `7.5e-09` — and the through-flow falls `11.4x` over 60 rounds
(`12.5x` at 120, so this is a descent, not a root-find).

??? measured "Why the obvious benchmark measures nothing"
    The benchmark deserves a word, because the obvious version of it measures nothing. In a channel with a
    **symmetry** bottom, uniform flow `u = (1,0)`, `p = 0` satisfies every equation and boundary condition
    for *any* shape of the traction-free top: measured, the solution stayed uniform to `2.3e-14` and moved
    by `9.8e-14` when the wall was displaced by `0.1`. The objective is then purely geometric — it exercises
    the normals and the facet measure but never the solve. A **no-slip** bottom couples them (`max|du| =
    7.2e-02` for the same displacement), and only then is `d(objective)/d(vertex)` a statement about the
    physics rather than about the mesh.

Three things to know:

* The objective is a **scalar**, so it needs a scalar test function; on a velocity/pressure saddle the
  pressure test is chosen automatically.
* When the expression reaches its region only through a **bound view** — `u.bind(x=xs, y=ys)` absorbs
  its coordinates — the test function cannot be auto-bound. Carry it yourself, as above
  (`* vs[0]`). That case raises with this instruction rather than a trace-level binding error.
* A surface objective needs the **form** to carry a surface term, because the facet quadrature tables
  are tabulated at build time only then. A traction-free wall (`0.0 * vs[0]`) in the term list is enough.

**A mesh condition, as an inequality.** `criterion=` also takes a `jno.le` / `jno.ge` constraint, and
then it is its own trigger — there is no cadence or threshold argument, because the condition already
says both *where* and *whether*:

```python
fem.solve(adapt=jno.solve.remesh(criterion=lambda d: jno.le(d.cell_aspect(), 2.0), max_iters=6))
```

Every cell whose margin is positive is marked — all of them, not a Dörfler fraction — and the march
stops when none is. Measured on a deliberately stretched mesh: worst aspect `2.87 → 1.57` in one
round, `0` marked on the next. `theta` is refused with a constraint (there is no bulk fraction to
choose), and a bare comparison (`q > 2.0`) is refused too: it records which cells are bad but not by
how much, so marking would take a fraction of them and quietly leave the rest.

**On a march** the same condition is a *trigger*. With `remesh(criterion=jno.le(...), every=k)` on a
transient problem it is checked on the current mesh every `k` steps, and the mesh is rebuilt only when
some cell breaks it; `fem.adapt_history` records `remeshed: False` for the rounds it held, so a
condition that never breaks never remeshes. A ranking criterion (`1 - phi**2`, `|grad u|`) is evaluated
on the live state at each remesh -- and at that remesh's time, so it may read `t` (a moving source) -- and
the vertex budget (`max_dofs`, else the starting count) is held on both the isotropic and the anisotropic
path.

Two things to know. **Set a threshold the mesher can actually reach** — an unstructured 2-D mesh
bottoms out around `1.2`–`1.5`, and a constraint below that never settles, so the march refines until
it runs out of rounds. And pass a **callable** for a geometry criterion: a geometry node captures the
cell table when it is constructed, so a single node keeps answering for the mesh it was born on and is
refused by name once the topology changes.

**When moving nodes is not enough: `relocate(...).remesh(...)`.** Relocation moves a *fixed* node set,
so once the mesh has to stretch further than its elements allow it can do nothing — `quality_floor` is
a line search that rejects the step, and rejecting a step never adds a node. Chain an h-step onto it
and say what "too far" means:

```python
fem.solve(adapt=jno.solve.relocate(objective=through, max_iters=200)
                         .remesh(criterion=lambda d: jno.le(d.cell_aspect(), 2.0), max_iters=4))
```

??? note "Where the quality floor is enforced"
    The condition is checked **inside the relocation line search**, on each candidate step, so an
    inadmissible mesh is never accepted — the march honours the bound exactly rather than reporting a
    breach after the fact. When no admissible step exists, relocation has run out of room and the cells
    *blocking* it are refined; the movable vertices are then re-derived from the **region** each was tagged
    on, since indices do not survive a remesh. Measured on a Poisson peak, bound `1.7`: `44 → 80` vertices
    over 3 remeshes, final worst aspect exactly `1.700`, objective still falling `8.4e-02 → 2.0e-02`.

The nested spec's `max_iters` caps how many remeshes the march may spend and `max_dofs` caps the size.
If the budget runs out while the mesh still breaks the condition, that is **raised**, not returned
quietly — refining does not repair every shape, and a bound below what the mesher can deliver would
otherwise refine without end (measured, before this: `44 → … → 15709` vertices and an out-of-memory
failure inside the solver).

Scope: the interleaved criterion must be a **mesh-geometry condition** (`cell_aspect`, `cell_volume`,
`cell_angles`), because it is evaluated on the moving vertices with no solve, and the bound must sit
directly on one node so it can be evaluated once per round rather than re-traced. A solution criterion
belongs on a standalone `jno.solve.remesh(...)`.

**Mesh quality as a term you can write.** `domain.cell_aspect()` is the longest edge over the inradius,
scaled so a **regular** simplex reads exactly `1.0` and a stretched one reads more — per cell, 2-D and
3-D, and differentiable in the vertex positions (checked against central differences at `2.7e-10`).
It is the companion to `domain.cell_size`, which is `|det J|^(1/dim)` — an isotropic *size* that cannot
see stretch at all, since a sliver and a regular element of the same area share it. `domain.cell_angles()`
also measures distortion but is 2-D only. Reference: Shewchuk, *What Is a Good Linear Finite Element?*
(2002), §2.

Tagging is **literal and per-axis**: `xm.trainable()` frees only the x column. On a boundary that is the
lever for sliding — free an edge's along-edge axis and its nodes redistribute *within* the wall, leave the
normal axis untagged and the domain shape is preserved exactly.

??? note "`method="monge_ampere"` — the alternative"
    `method="monge_ampere"` swaps the descent for a Monge–Ampère mesh solve, `m·det(I + H(φ)) = θ` with
    `x = ξ + ∇φ` (McRae, Cotter & Budd, SIAM J. Sci. Comput. **40**(2) 2018, arXiv:1612.08077 §3.1). The
    displacement is a gradient, so the *whole* map cannot fold and no line search is needed, and it converges in
    3–6 rounds against descent's 30. It is **not** the default, because on the Allen–Cahn front the suite
    measures (`h=0.06`, `eps=0.03`, error on a common fine grid so the metric does not depend on where each mesh
    puts its nodes) it loses on the answer:

| mesh | rel-L2 | vs uniform | min element quality |
|---|---|---|---|
| uniform | 1.096e-01 | 1.000 | 0.834 |
| `relocate()` (descent) | 3.951e-02 | **0.361** | 0.503 |
| `relocate(method="monge_ampere")` | 8.879e-02 | 0.811 | 0.160 |

The cause is structural rather than a tuning miss: with one global `θ`, concentrating elements at the front
forces the rest of the domain to stretch, which is why element quality collapses far from the feature. The
control is under-relaxation — `relax_step=0.02` recovers quality to 0.318 and the ratio to 0.633.

---

## A moving mesh is a term

A moving mesh is not a solve argument. Put `coord.d(t) - velocity` in the `jno.fem([...])` list — a residual
like any other equation — and the mesh moves as it says:

```python
jax.config.update("jax_enable_x64", True)          # required; see the scope list below

xb, yb, tb = domain.variable("boundary", split=True)
fem = jno.fem([ui.t * vi + kappa * (ui.x * vi.x + ui.y * vi.y),   # the physics
               u(xb, yb) - 0.0, u(*ci) - 1.0,
               yb.d(tb) - 0.5 * yb])                              # dy/dt = y/2 — the mesh
traj = fem.solve()                                                # one frame per moved mesh
```

??? note "How a moving-mesh term is recognised"
    It is recognised **structurally**, by containing `d(spatial coordinate)/d(temporal variable)`, so there is no
    new spelling: `Variable.d` and the term list already exist. Nothing about it is boundary-specific — an
    interior region, a boundary and a `where=` predicate all resolve the same way — and tagging is **per-axis**,
    so a term on `yb` alone moves the y column and holds x exactly. The velocity is ordinary traced math, so an
    interface law may read the solved field (a Stefan front `-(k/L)·∇T·n`), the coordinates, the outward normals,
    the time, or a `jno.np.parameter` that then becomes a design variable. The march is differentiable in all of
    them, and in where the mesh started.

Each step: evaluate every geometry term's velocity, scatter it into the vertices and axes those terms name,
extend harmonically over everything they do not, move, re-assemble on the moved vertices, and carry the state
across.

### The mesh velocity inside a weak form (ALE)

On a moving mesh a nodal value follows its vertex, so its rate is the ALE derivative

$$
\left.\frac{\partial u}{\partial t}\right|_{X} = \left.\frac{\partial u}{\partial t}\right|_{x} + w\cdot\nabla u ,
$$

with $w$ the mesh velocity. Transport written on the moving mesh therefore carries $w$, and you write it as
what it is — the rate of a coordinate:

```python
w0, w1 = xi.d(ti), yi.d(ti)                                        # the mesh velocity w
fem = jno.fem([ui.t * vi + ((c0 - w0) * ui.x + (c1 - w1) * ui.y) * vi   # u_t + (c - w)·∇u
               + nu * (ui.x * vi.x + ui.y * vi.y),
               xi.d(ti) - c0, yi.d(ti) - c1,                       # the mesh: it follows the flow
               u(*ci) - u0])
```

When a weak form reads `coord.d(t)`, the nodal values **ride with their vertices** and nothing is transferred
(transferring as well would count the mesh advection twice). Each step, `w` is the discrete motion of every
vertex, $(X^{n+1} - X^n)/\Delta t$, harmonically extended vertices included, in volume and surface terms
alike. This is the non-conservative ALE form, backward Euler, with mass and operator on the end-of-step
configuration. Measured (`tests/test_fem_ale_mesh_velocity.py`):

* a mesh translating rigidly with the material reproduces the fixed-mesh diffusion march to **1.1e-15**;
* a surface term `(w·n) u v` equals `(c·n) u v` on that mesh to 3.6e-16;
* a Gaussian advected at `c` past a mesh moving at `c/2` meets its closed form to **2.0 %** (5.2 % on a fixed
  mesh with the same `h` and `dt`, since less relative motion means less numerical diffusion).

`w` lives on the mesh vertices and borrows a **P1 Lagrange** field's basis, so the problem needs one (a
Taylor–Hood pressure qualifies). Refused by name: `coord.d(t)` with no geometry term (it would be identically
zero), the rate of a normal or of `cell_size`, and the mesh acceleration `xi.d(ti).d(ti)`.

!!! measured "The stabilised-flow `tau` works for a capillary drop as written"
    The SUPG/PSPG `tau` of the stabilised-flow tutorial needs no rescaling for a surface-tension driven
    drop. Measured on an ellipse released from rest (`tests/test_fem_drop_oscillation.py`), the recipe as
    written rings at **ω within 0.4 %** of Lamb's `ω² = n(n²−1)σ/(ρR³)`, decays at **0.89×** the viscous
    rate `2n(n−1)ν/R²`, and leaves the `n = 4` mode at its initial amplitude. Scaling `tau` by `1e-4`
    changes none of these.

    Earlier versions of this page warned the opposite: that this `tau` over-damps the drop tenfold
    (ω = 0.23) and feeds a spurious `n = 4` mode (×9.7). That was an **assembly defect**, not physics. The
    stabilisation products contain `u_t` inside the strong residual, and the transient assembler routed each
    such product whole into the mass matrix. They are now split by temporal order. A spelling the assembler
    cannot split, such as `tau * inner(a, u.t + (u.grad)u)`, is refused with instructions to write the `u.t`
    part as its own term.

    Two further practicalities. Measure the shape as a **Fourier mode of `r(θ)`**, not as a bounding-box aspect
    ratio, and fit a damped sinusoid — extremum counting returned `+7.4/s` and `−1.6/s` for the same run over
    different spans. And the capillary time step is tighter than `√(ρh³/2πσ)` suggests: at `h = 0.03` that
    estimate gives `8e-4` while `5e-4` already tangles, so shrink `dt` as you refine.

### Bodies that merge — `remesh(alpha=...)`

Moving a mesh cannot change its topology: two droplets meshed as separate bodies in a void can approach
forever and stay two meshes. `alpha=` re-decides which nodes form elements, instead of meshing the geometry
afresh — a Delaunay triangulation of the nodes where the motion left them, keeping only the triangles whose
circumradius is below `alpha` × the local mean edge length (the **alpha shape**, the remeshing step of the
Particle Finite Element Method):

```python
traj = fem.solve(adapt=jno.solve.remesh(alpha=1.2, every=1))   # re-triangulate the nodes every step
```

Every node stays put, so a P1 state carries across **by identity** — nothing is interpolated and nothing is
lost. What changes is which nodes are neighbours, so a gap that closes gets bridged and the two bodies become
one mesh.

!!! measured "Two disks (R = 0.5, h = 0.075, 357 nodes) driven together"
    | gap | 0.35 → 0.150 | 0.136 |
    |---|---|---|
    | bodies | 2 | **1** |

    They merge at about `1.8 h`, a little later than the flat-wall threshold `2·alpha·h = 0.18`: two facing
    arcs put fewer nodes near the closest point than two straight walls do. The node count is unchanged
    (357), the cell count goes 630 → 638, and the field — two flat plateaus at ∓1 — starts to diffuse across
    the new neck (minimum |u| there 0.479, against 0.972 when the bodies never merge). Only 2 of 12 steps
    actually rebuilt anything: the reconnection is skipped when it reproduces the elements already there.
    The march cost 5.0 s against 3.6 s without reconnection.

Merging is therefore a **mesh-length contact model** — bodies join when their gap is comparable to the
element size — and not film drainage. Scope: 2-D, P1 fields (a new edge bridging two bodies has its midpoint
in the void, where a P2 value cannot be read), a geometry term must be present (nothing else moves the
nodes), and `alpha=` does not combine with `criterion=` — they are different operations. A node left in no
triangle (a free particle) is refused rather than dropped, because dropping it would renumber the rest and
silently permute the state.

**Scope** — the rest raises rather than guessing:

* **Operator-split ALE, explicit in the velocity**, hence first order in the step — *measured*, against a
  manufactured solution on a translating domain: observed rates 1.14 / 1.12 / 1.12 / 1.10 with the mesh
  moving, 0.99 / 1.01 / 1.04 / 1.10 with it still. The motion multiplies the error *constant* by ~3× (that
  is the state transfer) and leaves the *order* intact. Refining `h` converges too — 1.51 → 1.76 toward the
  expected 2 for P1 — and P2 is ~18× more accurate than P1 on the same moving mesh, so higher order still
  pays here.

  If you repeat that measurement, compare against a fine-`dt` reference **on the same mesh**, not against
  the exact solution: the temporal and spatial errors have opposite signs, so the direct comparison shows
  rates of +1.4 then −0.4 as they cancel and separate again. That reads as a scheme that stops converging,
  and is only a contaminated measurement.

  ??? note "Why the mesh motion is explicit, not an implicit coupled equation"
    The term list *reads* like a coupled equation and this is not one: an implicit mesh would need the
      coordinates as unknowns in the monolithic system and the ALE convective term.
    * **The state transfer is a conservative L2 projection** onto the moved mesh, and it is still diffusive. On a
      rigid translation carrying a marginally-resolved bump the peak falls ~9 % (the pointwise re-interpolation
      this replaced fell ~33 %, and got *worse* as `dt` shrank). Conservation is algebraic — `Σφ = 1` — so the
      residual is quadrature error on an integrand with kinks: ~2e-4 relative against the pointwise route's
      3e-3 to 9e-3. Removing the diffusion entirely means not transferring at all (Lagrangian DOFs plus an ALE
      `-w·∇u` term), which is a different semidiscretisation — and what happens when a weak form reads the
      mesh velocity (above).
    * **Requires `jax_enable_x64`.** The transfer locates quadrature points in the previous mesh, and in float32
      that carries ~4e-4 — enough for a mesh that never moves to drift 1.5e-3 over a march (2.6e-10 with x64).
    * **Backward Euler only**: `θ` comes from the block, and `time=jno.solve.theta(...)` is a solver slot, which
      a geometry term does not compose with.
    * **Connectivity-preserving, unless told to remesh**: a move that would invert an element raises. With
      `fem.solve(adapt=jno.solve.remesh(criterion=lambda d: jno.le(d.cell_aspect(), 3.0), every=1))` the march
      checks the condition on the moved mesh every `every` steps and, where it breaks, rebuilds the mesh
      (mmg, at the starting vertex budget), re-assembles and carries the state across. Measured on a top edge
      bulging as `y' = 2y sin(πx)`: worst cell aspect 6.27 → 3.19 with one remesh, the surface where the plain
      march puts it, a constant field exact to 4e-16. The laws may read only `boundary` and `interior` (any
      other region is found by a position that does not follow a moved surface), a remesh is not
      differentiable (refused under `jax.grad`), and each remesh drifts a curved boundary's enclosed area by
      ~6e-3 (mmg, measured on a disk).
    * A Dirichlet BC on the moving surface must be tied to a whole-boundary or held tag, not to a spatial
      sub-predicate — a predicate does not follow the motion.
    * **Any nodal-Lagrange field(s), real, non-periodic**, 2D or 3D — scalar or vector, P1 or higher, and
      *mixed orders* across a coupled system (a Taylor–Hood pair moves as one). **Nonlinear** problems work
      too. Complex, periodic, a non-nodal family (RT / Nédélec / Hermite / Argyris / Morley), a custom
      `solve_fn` and `save_ts=` each raise.

  ??? note "Why higher order costs almost nothing here"
    Higher order costs almost nothing structurally, for two reasons worth knowing: the mesh geometry is
      **P1 whatever the field order** — a moved simplex stays straight-sided — so the quadrature map and the
      point location are shared by every field; and a topology-preserving move leaves the P{k} **connectivity
      unchanged**, so the seed assembly's tables stay valid for the whole march and the moved DOF
      *coordinates* are never needed at all. The quadrature degree follows the order (`2k`), because the mass
      `∫φᵢφⱼ` must be integrated exactly or the solve is not a projection.

### Reconnecting without recompiling — runtime connectivity

A Delaunay flip changes which nodes form a cell, but not how many cells there are, so no array changes
shape. A march with a geometry term therefore assembles against a **runtime** connectivity: the cell
array rides the same argument channel the moved vertices already use, and a reconnection that keeps
every shape hands the compiled program a new triangulation instead of rebuilding it.

It is inferred: `coord.d(t) - velocity` already states that the mesh moves, and that is the only case
in which the connectivity can change under a fixed node set. `Domain.dynamic_topology()` is the explicit
spelling, and `dynamic_topology(False)` opts out.

!!! measured "Runtime connectivity, on a geometry-term march"
    | | baked connectivity | runtime connectivity |
    |---|---|---|
    | one reconnection | one XLA compilation, ~31 s | 0.13 s |
    | two-drop coalescence, whole march | 1587 s | **69.8 s** (22.7×), trajectory identical to 1.1e-16 |
    | `relocate`, no reconnection at all (41 frames) | 1.6 s | 1.0 s |
    | `remesh(alpha=1.2, every=1)` (41 frames) | 3.5 s | 0.9 s |

    It is faster even when nothing reconnects, which is why it is the default: the gather goes through
    the index array the moved vertices travel on anyway.

Only same-shape changes become free. A reconnection that changes the **cell count** (two bodies merging,
node management inserting a node) still rebuilds. Affine simplices only: on a curved or tensor-product
mesh an explicit `dynamic_topology()` raises, and the inferred one falls back to the baked connectivity.

When the node set does change, the state is carried by a **conservative L2 projection** onto the new
mesh rather than by pointwise interpolation. On a laser-heated drop, whose hot layer is a few cells
thick, one remesh used to cost 540 K of peak temperature and 6 % of the thermal energy; it now costs
6 K and +0.5 %.

### Adapting a moving mesh — `relocate`, `escalate=`, anisotropic `remesh`

All three adaptivity kinds that move or add nodes compose with a geometry-term march.

**r-adaptivity is nearly free here.** Vertex positions ride the scan carry, so a relocation hands a new
`X` to the same compiled program: no new nodes, no new connectivity, no rebuild.

```python
traj = fem.solve(adapt=jno.solve.relocate(method="monge_ampere", every=20))
```

!!! measured "600 W melt ball (R = 150 µm, h = 40 µm), 800 steps"
    | adaptivity | nodes | wall | p90 interpolation error | dent |
    |---|---|---|---|---|
    | `remesh(alpha=1.2, every=1)` | 78 | 129 s | 0.1720 | −22.78 µm |
    | `relocate(every=20)` | 74 | **34 s** | 0.1722 | −22.23 µm |
    | `relocate(every=20, escalate=0.5)` | 112 | 61 s | **0.1124** | −26.18 µm |

    How often it relocates is free once the Monge–Ampère round is compiled: `every=100/50/20/10` all
    cost 34–35 s.

- **Area is conserved; perimeter is not.** A boundary vertex slides only along the chord joining its
  two boundary neighbours, which leaves the enclosed area unchanged to first order, and an iterated
  normal correction removes the rest (4.7e-3 → 4.2e-11 relative). A free surface stretches, so its
  perimeter is left free.
- **A sliver is a connectivity defect.** Moving nodes cannot repair it, so `relocate().remesh(alpha=...)`
  reconnects first, at a fixed node set, and then relocates.
- **A relocation that would invert a cell is rejected**, not applied. Relocation optimises the
  discretisation, never the physics, so skipping a round is harmless and visible in the trajectory's
  records (`relocate_skipped`, `relocate_would_invert`). `JNO_RELOCATE_STRICT=1` raises instead. Before
  this guard, a 600 W run reported 10,201 plausible frames while 98 % of its cells had inverted, and
  its temperatures stayed in range throughout. Only the area and the orientation revealed it.
- **`escalate=tol` adds nodes only when moving them was not enough.** After each relocation the march
  measures the interpolation error, and above `tol` (relative to the field's range) it adds vertices
  along the anisotropic `mmg` path. The error indicator pairs each curvature with the element's extent in
  its own direction (Alauzet & Loseille, *J. Comput. Phys.* **229** (2010) §2.2). An isotropic shape
  measure is deliberately not the trigger: on an anisotropically adapted mesh it flags the stretched
  cells that are doing their job. The loop is self-limiting. Above, it fired once, at the error peak.

**Anisotropic h-adaptivity** is `remesh(anisotropic=True)` with a criterion that says *when* to remesh,
and a Hessian metric that says *how*. On a squeezed disk carrying a sharp front, it produced elements with
aspect ratio 9.04 lying along the level sets (`|long axis · ∇u|` = 0.046, against 0.330 for an isotropic
remesh).

**Not wired, and the reasons are substantive:**

- **p-adaptivity (`enrich()`)** needs a `space="cover"` field, whose degrees of freedom are not nodal
  values. Plain P2 does march on a moving mesh.
- **r-adaptivity on two separate bodies** still fails. On a two-ball problem, 399 of 400 relocation
  rounds would invert cells and are rejected. Use `alpha`, which is also the only adaptivity that can
  merge bodies.

### A graded mesh reconnects on its own length scale

The `alpha` filter keeps a triangle when its circumradius is below `alpha` × a length scale, and that
scale is now each node's own mean edge length, not one number for the whole mesh. On a graded mesh a
single value fails at both ends: coarse cells all exceed it, and their nodes become free particles; in
the fine region, the coarse value fuses surfaces that are really apart. Two rod tips graded 50 → 500 µm:
the scalar mean (98 µm) stranded 50 of 363 nodes; the per-node scale triangulates all of them and keeps
the rods apart. An inserted midpoint inherits the mean of the two nodes it splits.

### A march that survives being killed — `checkpoint=`

A march holds every frame and returns them only when `solve()` returns, so a run killed by an OOM, a
signal or a reboot used to return nothing, however far it had got. `checkpoint=` writes frames to disk as
they are produced:

```python
traj = fem.solve(
    nonlinear=jno.solve.newton(direct=True),
    adapt=jno.solve.remesh(alpha=1.2, every=1),
    checkpoint=jno.solve.checkpoint("runs/ball", every=500),
)
```

- **`every`** is the number of steps between writes. A write also happens at every rebuild, because
  that is where a restart has to begin anyway.
- **`keep="last"`** (the default) drops each written chunk from memory, and the returned trajectory
  loads its frames from disk on demand, so a march no longer has to fit in RAM. `keep="all"` keeps
  everything resident and checkpoints only for crash safety.
- **`resume=True`** continues an unfinished run in `path`. A run marked complete is never resumed;
  delete the directory to redo it.

The restart state is the one a rebuild already re-enters the march with, plus the domain the next
segment starts on. Trajectories are bit-identical with and without checkpointing, including across
rebuilds (600 and 1,400 steps on a melt ball; `max |Δstate| = max |Δpoints| = 0`).

`checkpoint=` needs `adapt=`, and is refused otherwise. A plain march is one compiled `lax.scan` with no
host-side boundary to write at; extending it would cost end-to-end reverse-mode differentiation (#141).

Two memory leaks in the rebuild path were fixed with it:

- a closure kept each segment's sampled point pools alive;
- re-sampling a region wrote to a fresh alias instead of the region's own name, which also left the
  constraints reading the old points.

Per-rebuild memory growth fell from 0.361 GB to 0.027 GB, and device memory now stays flat.
`JNO_MARCH_MEMDEBUG=1` prints the per-rebuild breakdown (RSS, live JAX buffers by shape, Python heap,
`domain.context`) that found both leaks.

---

## Per-region (sub-domain) integration

A weak term integrates over the **region of the coordinates it is written on** — exactly the rule that
already routes boundary terms. Bind the trial/test to a **sub-region's** coordinates and the term
integrates over that sub-domain's cells only. Name a region with `domain.tag(name, predicate)` (or use
a multi-part mesh's geometry parts) and ask for its coordinates with `domain.variable(name, split=True)`:

```python
d.tag("core", lambda x, y: (x - 0.5)**2 + (y - 0.5)**2 < 0.2**2)   # an interior sub-region
xc, yc, _ = d.variable("core", split=True)
uc, vc = u.bind(x=xc, y=yc), phi.bind(x=xc, y=yc)

fem = jno.fem([
    ui.x * vi.x + ui.y * vi.y,            # ∫_Ω   ∇u·∇v        (whole domain)
    9.0 * (uc.x * vc.x + uc.y * vc.y),    # ∫_core 9 ∇u·∇v     (k = 1 outside, 10 inside `core`)
    q * vc,                               # ∫_core q·v          (a localized source)
    u(xb, yb) - 0.0,
])
```

Multi-material conduction is then *one term per material*; a data-fit / QoI confined to a region is
`(uc - u_data) * vc`. A cell belongs to a region iff its **centroid** does (classified once at assembly
— exact when the mesh respects the region boundaries, e.g. gmsh meshing each part separately; O(h) at
an arbitrary-predicate interface). Region integration is a scalar mask on the integrand, so it
**composes with everything**: constant / `jno.fn` / `.freeze()` / trainable coefficients, and the
steady-linear, nonlinear, transient, coupled, and 3-D forms. A `jno.np.parameter` that multiplies a
sub-region term is recovered **per sub-domain** through `crux`.

!!! warning "Every field still spans the whole mesh"
    `fem_symbols` has no region argument, so a field restricted *in its terms* still carries DOFs
    everywhere. If **every** term touching a field is region-restricted, that field's DOFs on the rest of
    the mesh appear in no equation at all and the system is structurally singular — the case that shows
    up the moment you write different physics per region (Navier–Stokes on `fluid`, elasticity on
    `solid`).

    `jno.fem` refuses that at build time, naming the field, the count and the region:

    ```
    jno.fem: these DOFs appear in no term and carry no prescribed value, so the system is
    structurally singular:
      field 'u' (block 0): 87 DOFs, every term reaching it is restricted to 'lower'
    ```

    The fix is to give the field a term over the region it is missing from; where it carries no physics
    there, a cheap `1e-8 * u * phi` on that region is enough. A **Dirichlet pin on the region is not a
    reliable substitute**: an essential condition resolves its nodes through `domain.tag_node_mask`,
    which for a *volume* region is a proximity test against sampled points rather than a containment
    test, and can miss interior nodes (measured: it reached 32 of the 33 ungoverned DOFs on a two-block
    domain). Reach itself is measured by cell centroid — the assembler's own resolution — so the check
    cannot disagree with what was assembled.

    Scope: this finds a DOF that appears in **no term**. It is not a rank test, and does not detect a
    missing pressure gauge or an unrestrained rigid-body mode.

### `domain.attach` — many materials as one equation

For *many* regions, writing one term per region is noisy. Declare the property on each region with
`attach`, then read it back as `d.<name>`: a single coefficient whose value is chosen, per cell, by the
region the cell's centroid lies in — so the whole multi-material weak form is **one equation** over the
whole `interior`:

```python
xi, yi, _ = d.variable("interior", split=True)        # whole domain, bound once
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

d.attach("steel", k=16.0).attach("air", k=0.026).attach("core", k=25.0)
d.attach("core", Q=heat_source).attach(Q=0.0)         # a bare attach = the default for the rest
fem = jno.fem([d.k * (ui.x*vi.x + ui.y*vi.y) - d.Q * vi, u(xb, yb) - 0.0])
```

**Omit the target for a default.** `attach(Q=0.0)` gives the value for every region that names none of
its own — the source that is zero outside one body. An explicit target always wins over a bare one,
*whichever order they are written in*: without that rule the two orderings would differ silently.
A default applies to volume regions only; boundary tags are not a partition of the boundary, so a
surface default has nothing to mean and is refused.

A value can be a scalar, a `jno.fn` field, or a trainable `jno.np.parameter`, so the same primitive
expresses conductivity, a source, a density, a reaction rate, an elastic modulus — and trainable
per-region values compose for free, calibrated through `crux`. Each target must be a geometry part
(`from_regions`), a `domain.tag`, or a mesh-file region. It desugars to `sum_r RegionMask(r)·value_r`,
inheriting the centroid classification.

> Not yet wired: second-order-in-time (`u_tt`) sub-region terms (fail loud). 3-D sub-regions are
> defined by a predicate `where(x, y, z)`.

---

## Axisymmetric (bodies of revolution)

**jNO applies no ring measure for you — you write it in the weak form.** A meridional `(r, z)` mesh is
an ordinary 2-D mesh; what makes a form axisymmetric is the measure you put in it.

For a **scalar** field that is exactly the Cartesian integrand times `2πr`:

```
∫ (∂ᵣu ∂ᵣv + ∂zu ∂zv) 2πr dr dz     is the weak form of     (1/r)(r u_ᵣ)_ᵣ + u_zz
```

so nothing is missing — write the factor and you have it:

```python
d = jno.domain(box(a, 0, b, h), mesh_size=0.05)      # just a 2-D mesh
d.tag("inner", lambda x, y: jnp.abs(x - a) < 1e-9)
d.tag("outer", lambda x, y: jnp.abs(x - b) < 1e-9)

u, v = d.fem_symbols()
r, z, _ = d.variable("interior", split=True)
ui, vi = u.bind(x=r, y=z), v.bind(x=r, y=z)
ra, za, _ = d.variable("inner", split=True)
rb, zb, _ = d.variable("outer", split=True)

dV = 2 * jnp.pi * r                                   # the ring measure, written once
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) * dV,
               -g * v.bind(x=rb, y=zb) * (2 * jnp.pi * rb),   # boundary terms carry it too
               u(ra, za) - T_a])
T = fem.solve()      # logarithmic in r, and Q = 2πk ΔT/ln(b/a) comes out right
```

Every term needs it — volume terms, source terms and Neumann/Robin terms alike. Miss one and the
answer is wrong by exactly that factor with no error raised, so bind `dV` once at the top and reuse it.

!!! danger "Vector fields need more than the measure"
    Axisymmetric elasticity carries a hoop strain
    `ε_θθ = u_r/r`, and divergence picks up `u_r/r`. Neither can be produced by weighting the Cartesian
    form by anything — they are extra terms you must write out. This is precisely why jNO does not offer
    to apply the weighting automatically: it would be exact for scalars and quietly wrong for vectors.

### The vector recipe, written out

A displacement `u = (u_r, u_z)` on the meridian strains four components, not three:

$$\varepsilon_{rr}=\partial_r u_r,\qquad \varepsilon_{zz}=\partial_z u_z,\qquad \varepsilon_{rz}=\tfrac12(\partial_z u_r+\partial_r u_z),\qquad \boxed{\varepsilon_{\theta\theta}=u_r/r}$$

The last one is the whole difference, and it is ordinary arithmetic in the term list:

```python
u, v = d.fem_symbols(value_shape=(2,), order=2)          # (u_r, u_z)
r, z, _ = d.variable("interior", split=True)
ub, vb = u.bind(x=r, y=z), v.bind(x=r, y=z)

def strains(w):
    return w.x[0], w.y[1], w[0] / r, 0.5 * (w.y[0] + w.x[1])   # rr, zz, THETA-THETA, rz

e_rr, e_zz, e_qq, e_rz = strains(ub)
f_rr, f_zz, f_qq, f_rz = strains(vb)
tr_u, tr_v = e_rr + e_zz + e_qq, f_rr + f_zz + f_qq            # the trace carries it too
energy = lam * tr_u * tr_v + 2 * mu * (e_rr*f_rr + e_zz*f_zz + e_qq*f_qq + 2*e_rz*f_rz)
fem = jno.fem([energy * (2 * np.pi * r), ...])                 # and then the ring measure
```

For flow the same four components appear, and the **divergence** is
$\nabla\!\cdot\!\mathbf u=\partial_r u_r+u_r/r+\partial_z u_z$ — so `div_u = e_rr + e_qq + e_zz`, which is
what the continuity equation and the pressure term must both use.

!!! measured "Both halves are pinned by `tests/test_fem_axisymmetric_vector.py`"
    **Lamé thick-walled cylinder** under internal pressure (plane strain), exact
    $u_r = \frac{(1+\nu)pa^2}{E(b^2-a^2)}\left[(1-2\nu)r + b^2/r\right]$: the form above reproduces it to
    **under 1 %**, with $u_z$ vanishing. Drop `ε_θθ` from that same form and the error is **over 10 %**
    and the cylinder measurably softer — with no error raised, which is what makes it worth a test.

    **Poiseuille pipe** ($u_z = G(R^2-r^2)/4\eta$) on a meridian that *includes the axis*, Taylor–Hood
    P2/P1: **machine precision**, since P2 represents the parabola exactly. Two details matter — a
    traction-free pipe end is not what fully developed flow satisfies (the parabola carries a shear
    traction $\eta u_z'$ on a z-face), and on the axis `u_r/r` is harmless because quadrature points sit
    strictly inside cells and $u_r=0$ is imposed there.

!!! danger "This applies to vector EM too, and there is no guard"
    An axisymmetric `(r, z)` **vector
    Maxwell / eddy-current** form — the natural geometry for a coil, a solenoid, a tokamak vessel — is
    *not* the Cartesian curl-curl weighted by `2πr`. In cylindrical coordinates the curl of a vector
    field picks up its own `1/r` terms — `(∇×E)_z = (1/r)∂(rE_θ)/∂r` — and for an axisymmetric (`m=0`)
    problem the meridional (`E_r, E_z`) and azimuthal (`E_θ`) components **decouple into two different
    operators** that must each be written out. The azimuthal one reduces to a *scalar* equation (in
    `E_θ`, or in `rA_θ` for the vector-potential/eddy-current form), not a component of the Cartesian
    form, and it needs care on the axis where `1/r` is singular. Weighting an
    `"N1E"` form by `2πr` therefore produces a **silently wrong** answer, not an approximate one.
    Nothing raises: multiplying by `r` is ordinary arithmetic, and the assembler cannot tell it apart
    from a legitimate radial coefficient — so this limit is stated here, where you choose the geometry,
    rather than enforced at assembly. Use a **full 3-D** mesh for vector Maxwell (3-D N1E is wired and
    validated), or derive and write the cylindrical operator yourself as an ordinary scalar/coupled
    form. jNO ships no axisymmetric H(curl)/H(div) element and no meridional/azimuthal split.

!!! warning "Enclosure radiation carries the same factor"
    `domain.enclosure(tags, axisymmetric=True)` gives ring areas `2πr̄·L` and a
    `gap.load(q)` that is **per full revolution** (W, not W/m). The weak form you add it to must carry
    the same `2πr`, or the two sides differ by exactly that factor. jNO cannot check this for you.

---

## Enclosure radiation (nonlocal boundary flux)

Grey-body radiation between surfaces is **nonlinear** (`T⁴`) and **nonlocal** (every surface element
exchanges with every other via the view-factor matrix `F`), so it cannot be a local weak term. jNO
provides the *geometric* building block — the view matrix — and you write the radiosity **as math** in
`jno.np`; there is no `jno.radiation()` helper.

`domain.enclosure(tags)` discretises the radiating boundary surfaces into **elements** aligned to the
FEM mesh nodes and returns a handle:

```python
gap = d.enclosure(["inner_gap", "outer_gap"])   # name the surfaces once
gap.check()                          # F-quality gate: closure (Σ_j F_ij→1) + reciprocity (A_i F_ij=A_j F_ji)
F   = gap.view_factor                # (m, m) element view factor — fully geometry-determined
eps = gap.emissivity({"inner_gap": 0.8, "outer_gap": 0.6})         # per-element ε from a {tag: ε} map
rho = 1.0 - eps
```

??? note "How the view factors are computed"
    `F` is computed purely from geometry (occlusion + orientation) by **double-area Gauss quadrature** of
    the diffuse kernel — so a *concave* surface keeps its self-view. Tags only group elements (for
    per-surface emissivity); they never block exchange. The enclosure **inherits** `axisymmetric` from the
    domain (see below), so its ring areas and the FEM measure cannot disagree; passing a contradicting value
    raises. By default the boundary normals point *out of* the mesh (radiation across an un-meshed gap); for
    an **oven/furnace cavity** where the meshed fluid is inside, pass `inward=True` so the facing walls see
    one another (see the *Oven* tutorial); for a meshed *medium* between solids use `medium_tags`.

??? note "What blocks a ray — interface mode"
    Every meshed region that is *not* listed in `medium_tags`
    is opaque — including a solid that carries no radiating surface of its own. The occluder set is
    resolved once from the region list, not inferred from element tags, and is shared by the visibility
    test and the near-field refinement: a solid with no radiating surface still blocks, and a chord
    through it is never counted as visible.

??? note "Axisymmetric near field, and when the graded rule runs"
    The ring kernel's azimuthal integrand peaks at `φ = 0` with width `d/r`,
    so a uniform `n_phi` rule overshoots every near-touching pair (two surfaces meeting in a wedge, and
    every element's own ring self-view) by roughly `dφ/(d/r)`. A graded azimuthal rule fixes it and
    restores closure to ~1e-3, but it needs an occluder model to test its refined chords against, so it
    runs when one is available: **`medium_tags=...`** (interface mode — the solid polygons), or
    **`occlude=False`** (you asserting nothing blocks any ray, e.g. a convex cavity). Plain boundary mode
    with occlusion on keeps the uniform rule plus the `r_min` floor and a closure error around 1e-1, and
    logs a warning saying so. Its occlusion is also a *meridian-only* test reused at every azimuth, which
    is wrong for a general solid of revolution — interface mode checks the true 3-D chord per azimuth.
    Always call `gap.check()`.

Write the **full grey-body radiosity** (reflections included) and couple it to the conduction FEM by
adding the net flux as a consistent surface load to the residual:

```python
SIGMA, KELVIN = 5.670374419e-8, 273.15

def q_rad(u):                        # net radiative flux per element:  q = σ·G·T⁴
    Ts = gap.field(u)                # nonlocal gather: per-element temperature from the solution
    J  = jno.np.linalg.solve(jno.np.eye(gap.size) - rho[:, None] * F, eps * SIGMA * (Ts + KELVIN)**4)
    return J - F @ J                 # (I − F)(I − diag(ρ)F)⁻¹ diag(ε) σ T⁴

# −k ∂T/∂n = q_rad  enters the residual as a consistent load:  A u = b − gap.load(q_rad(u))
```

`gap.field(u)` gathers the per-element temperature; `gap.load(q)` scatters a per-element flux back to
the FEM nodes as `∫_Γ q·v ds`. The radiosity `(I − ρF)⁻¹` solve is `jno.np` — it is **traced**, so a
trainable `jno.np.parameter` emissivity flows through it for inverse problems.

**Solver note (BYO, jax-native).** The Dirichlet conditions are penalty-enforced, so the conduction `A`
is ill-conditioned — use a **direct** linear solve (a *matrix-free iterative* solver such as the
built-in `newton_krylov` may stall). The whole coupled solve stays jax-native and **differentiable**
(so `jax.grad`/`crux` recover an emissivity *through* the radiation) with a short direct-solve Newton
wrapped in `jax.lax.custom_root`. Validated on two concentric cylinders against the closed-form
two-surface series to <1%, including `jax.grad` w.r.t. emissivity vs finite differences
(`tests/test_fem_enclosure_radiation.py`).

### In-residual coupling (`jno.Coupling`) — implicit, trainable, transient

To instead solve conduction **and** radiation as one implicit system, pass the nonlocal residual **in
the `jno.fem([...])` list**. A plain function `f(u) -> (n_dofs,)` there is taken as a nonlocal
*coupling*: `jno.fem` adds it to the assembled residual `R(u) = R_local(u) + Σ_k coupling_k(u)`,
promoting a linear form to a nonlinear one, and `fem.solve()` drives the whole thing with the
matrix-free, `custom_root`-differentiable `newton_krylov`:

```python
def radiation(u):                         # the same radiosity, now a residual contribution
    Ts = gap.field(u)
    J  = jnp.linalg.solve(jnp.eye(gap.size) - rho[:, None] * F, eps * SIGMA * (Ts + KELVIN)**4)
    return gap.load(J - F @ J)             # net flux scattered to nodes

fem  = jno.fem([conduction, radiation, u(xc, yc) - T_COOL])    # radiation is the bare function
Tsol = fem.solve(u0=T_guess)                                   # conduction + radiation, one implicit solve
```

A jitted residual / callable *object* isn't a plain function — wrap it as `jno.Coupling(fn)`, which is
also how you reach the options below:

- **Trainable coupling parameters.** A coupling is opaque to the trace walk — a `jno.np.parameter`
  inside it is never found, so declare it. The residual then takes a second argument, the
  `{name: value}` dict:

    ```python
    eps = jno.np.parameter((1,), name="eps")           # emissivity, calibrated from data
    eps.initialize(jax.nn.initializers.constant(0.8))

    def radiation(w, p):                                # p -> {"eps": value}
        e = p["eps"].reshape(())
        J = jnp.linalg.solve(eye - (1.0 - e) * F, e * SIGMA * gap.field(w) ** 4)
        return gap.load(s_row * J - F @ J)

    fem = jno.fem([conduction, jno.Coupling(radiation, params=[eps]), u(xc, yc) - T_COOL])
    ```

    The gradient reaches `eps` *through* the radiosity solve, so `jno.core` recovers it from
    temperature observations like any other parameter (checked against finite differences to 3% in
    `tests/test_fem_enclosure_radiation.py`).
- **Multifield.** `jno.Coupling(fn, field_key=T_key)` acts on one field's DOF block (e.g. radiation on
  `T` in a heat+flow / thermo-mechanical solve).
- **Transient.** The coupling enters each implicit step, so enclosure radiation over a heating cycle
  solves in-residual. (Not combined with periodic ties.)

All four (bare function, `params`, `field_key`, transient) are covered in
`tests/test_fem_enclosure_radiation.py`. Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed.,
Ch. 4–5 (view factors; the net-radiation / radiosity method for diffuse-grey enclosures).

---

## Nonlocal coupling, in general — `jno.Coupling` and `jno.derived`

Enclosure radiation is one instance of a shape that recurs: **gather → operate → scatter**, where the
"operate" is not a local integrand. jNO has two mechanisms for it, and which one you want is decided by
*what the nonlocal quantity is*, not by the physics.

| | `jno.Coupling(fn)` | `jno.derived(fn, inputs=[...], on=u)` |
|---|---|---|
| produces | a **residual vector** `R(u) += c(u)` | a **nodal field** `d = f(u)` |
| used as | a term in the `jno.fem([...])` list | a value **inside** any term |
| tangent | carries the coupling **exactly** | **lagged** (Picard); stays local and sparse |
| solver | matrix-free Newton–Krylov only | assembled tangent, so `newton(direct=True)` works |
| converges | quadratically | linearly, and can stall on a strong coupling |

**Reach for `derived` when the nonlocal quantity is needed inside a term**, which is the case a residual
vector simply cannot express. A laser's Beer–Lambert attenuation is the clean example: the depth
`τ(x) = ∫α ds` depends on the whole chord behind each point, and `exp(−τ)` *multiplies* the source:

```python
from jno.utils.optics import beam_paths, optical_depth      # the beam GEOMETRY (private; host-side, once)

nodes, w = beam_paths(pts, cells, direction, pts, n_samples=600)
alpha    = lambda T: a0 * (1.0 + beta * T)                  # a hotter body absorbs more
tau      = jno.derived(lambda T: optical_depth(alpha(T), nodes, w), inputs=[u], on=u)
Q        = a0 * I0 * jno.np.exp(-tau)                       # tau reads as an ordinary field

fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - Q * vi, u(xb, yb) - 0.0])
```

The rule is **pure JAX** on the input fields' nodal values. Everything host-side — ray tables, view
factors, neighbour lists — is built once *outside* and closed over; a numpy/scipy rule is refused at
`jno.fem` build, with the fix named. Non-traceability, a wrong output length and an unknown `on=` field
all fail there rather than inside a Newton step.

**What lagging does and does not change.** The values are computed inside the residual from
`stop_gradient(u)`, so the linearization sees them as data and the nonlinear solver *is* the fixed-point
loop — no extra solver slot, no outer driver.

- The **converged root is exact**: `R(u) = 0` does not depend on gradient markers, so lagging changed the
  path, not the answer (the same argument `jno.lag` makes). Convergence is checked on the *true* coupled
  residual, so a fixed point that does not converge **raises** rather than returning a plausible field.
- Convergence is **linear**, at a rate set by the coupling strength. Past a critical strength it does not
  converge at all; `fem.solve(nonlinear=jno.solve.picard(damping=...))` or `newton(line_search=True)` is
  the remedy, and `jno.Coupling` is the alternative when you want the coupling in the tangent.
- Gradients are the **Picard adjoint** — descent-worthy and trainable through `jno.core`, but not exact;
  a finite-difference check will not agree to machine precision. A `params=[...]` value inside the rule is
  *not* lagged, so the direct sensitivity through it is exact.

**Cadence.** `every="residual"` (default) re-evaluates the rule at every residual evaluation, so a
transient step is fully implicit in the coupling. `every="step"` evaluates it once per step of a march
from the previous step's state — cheaper, but an operator splitting with its own `O(dt)` error that no
residual check can see, because the step converges, just to a slightly different problem.

!!! warning "Host geometry is frozen at build"
    Ray tables and view factors closed over by the rule are fixed for the life of the `jno.fem`. On a
    **moving mesh they go stale silently** — a rebuild means a new `jno.fem`. This is the same caveat the
    enclosure carries, and it is the open edge of droplet-plus-laser work.

### The same enclosure, written as a derived field

`gap.load(q)` returns an *integrated* load, so it becomes a nodal field by dividing out the consistent
weights `W_j = ∫φ_j ds` — which is `gap.load` of a unit flux, no new API:

```python
W  = gap.load(jnp.ones(gap.size), size=nd)               # the consistent nodal weight
qn = jno.derived(lambda T: gap.load(q_elem(T), size=nd) / jnp.where(W > 0, W, 1.0), inputs=[u], on=u)
fem = jno.fem([conduction, qn * v.bind(x=xg1, y=yg1), qn * v.bind(x=xg2, y=yg2), *bcs])
```

Because `Σ_i (M_∂)_ij = W_j`, the **total radiative power is conserved exactly**; the difference is one P1
boundary-mass smoothing of its distribution, consistent at `O(h²)` — measured at a thousandth of a kelvin
on the concentric-cylinder case in `tests/test_fem_enclosure_radiation.py`. What it buys is the tangent:
the derived form has an assembled sparse Jacobian, so `newton(direct=True)` solves it, where the
`Coupling` path is matrix-free only because its tangent couples every enclosure element to every other.
What it costs is robustness on a radiation-dominated enclosure, per above. **Both spellings are supported;
the `Coupling` one remains the default in the examples.**

### What stays on its own path — contact

Contact is *not* written with either mechanism, and deliberately. The gap value is a weighted gather
**inside** the differentiated residual, so the secondary–main coupling is exact in the tangent; only the
*pairing* is lagged, by a host-side search between rounds. Routing it through `derived` would
`stop_gradient` the gap value itself, turning a quadratic Newton contact into a Picard iteration on a
penalty interface — the regime where Picard is worst. The gap also lives at face quadrature points rather
than on nodes, so there is nothing for `on=` to name.

What contact *does* share is the delivery contract: **host-frozen shapes, per-round values threaded on
`args`**. `__gap_tables__`, `__loadpath__` (load-path fields, previous states, the mesh velocity) and
derived fields are all that one idea. Anything that writes to those channels must **merge**, never assign
— a march can carry a mesh velocity and a derived field at once.

---
