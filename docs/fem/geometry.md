# Geometry, regions, and moving meshes

## Curved (isoparametric) geometry — `Shape.curved()`

By default jNO meshes straight-sided and *synthesises* higher-order nodes at the straight-edge
midpoints, so the domain stays a polygon however high the element order goes. That approximation
carries an **O(h²) domain error at every basis order** — it is what caps P2/P3 at second order on a
round boundary, no matter how good the basis is. `curved()` asks the CAD kernel to place those nodes on
the true surface instead:

```python
d = jno.Shape.disk(0, 0, 1, size=0.1).curved().domain()
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
      `-w·∇u` term), which is a different semidiscretisation.
    * **Requires `jax_enable_x64`.** The transfer locates quadrature points in the previous mesh, and in float32
      that carries ~4e-4 — enough for a mesh that never moves to drift 1.5e-3 over a march (2.6e-10 with x64).
    * **Backward Euler only**: `θ` comes from the block, and `time=jno.solve.theta(...)` is a solver slot, which
      a geometry term does not compose with.
    * **Connectivity-preserving**: a move that would invert an element raises. Remesh-on-tangle is the next
      extension.
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

### `domain.by_region` — many materials as one equation

For *many* regions, writing one term per region is noisy. `domain.by_region({region: value})` returns a
single coefficient whose value is chosen, per cell, by the region the cell's centroid lies in — so the
whole multi-material weak form is **one equation** over the whole `interior`:

```python
xi, yi, _ = d.variable("interior", split=True)        # whole domain, bound once
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

k = d.by_region({"steel": 16.0, "air": 0.026, "core": 25.0})   # per-region conductivity
Q = d.by_region(heat_source, default=0.0)                      # 0 in any unlisted region
fem = jno.fem([k * (ui.x*vi.x + ui.y*vi.y) - Q * vi, u(xb, yb) - 0.0])
```

A value can be a scalar, a `jno.fn` field, or a trainable `jno.np.parameter`, so the same primitive
expresses conductivity, a source, a density, a reaction rate, an elastic modulus — and trainable
per-region values compose for free (`d.by_region({**k, "air": nu*0.026})`, calibrated through `crux`).
Each key must be a geometry part (`from_regions`) or a `domain.tag` predicate; `default` fills cells in
no listed region. It desugars to `sum_r RegionMask(r)·value_r`, inheriting the centroid classification.

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

- **Trainable coupling parameters.** A coupling is opaque to the trace walk — declare its parameters so
  they thread through the solve and `crux` recovers them: `jno.Coupling(fn, params=[eps])`, with the
  residual taking the `{name: value}` dict, `fn(u, p)`.
- **Multifield.** `jno.Coupling(fn, field_key=T_key)` acts on one field's DOF block (e.g. radiation on
  `T` in a heat+flow / thermo-mechanical solve).
- **Transient.** The coupling enters each implicit step, so enclosure radiation over a heating cycle
  solves in-residual. (Not combined with periodic ties.)

All four (bare function, `params`, `field_key`, transient) are covered in
`tests/test_fem_enclosure_radiation.py`. Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed.,
Ch. 4–5 (view factors; the net-radiation / radiosity method for diffuse-grey enclosures).

---
