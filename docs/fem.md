# Finite Element Method

jNO assembles and solves finite-element problems through a **single entry point**,
`jno.fem([...])`. You write the weak form as a plain Python list of **residual terms** —
volume physics, natural boundary terms, and essential boundary conditions, all in the same
list — and `jno.fem` returns a `FEM` object carrying the assembled operators.

The same traced weak-form language powers the steady solve, the transient time-stepper, and
the **differentiable** `fem.solve()` used for inverse problems. For a catalog of the symbolic
primitives you can write into a term — fields, derivatives, conditionals, non-local integrals,
geometry symbols (normals, `cell_size`), and the `jno.fn` escape hatch — see the
[weak-form vocabulary](weak_form_vocabulary.md).

```python
import jax.numpy as jnp
import jno

d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1))
u, phi = d.fem_symbols()                                   # trial / test functions
xi, yi, _ = d.variable("interior", split=True)            # volume quadrature coords
xb, yb, _ = d.variable("boundary", split=True)            # boundary coords
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)         # views with .x / .y derivatives

f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])   # weak form + u = 0 on the boundary
u_h = fem.solve()          # matrix-free default; slots pick anything else (see "Choosing the solver")
```

> The flat accessors — `fem.A`, `fem.b`, `fem.M`, `fem.state0`, and `fem.residual(u[, t])` /
> `fem.jacobian(u[, t])` — return ready-to-use **dense** matrices and **flat** vectors for
> hand-rolled experiments (e.g. `u_h = jnp.linalg.solve(fem.A, fem.b)`). `fem.operator` still
> exposes the raw sparse (`BCOO`) operator that `fem.solve()` and the solver slots work on directly.

---

## Domain, symbols, and derivatives

* **Domain** — any jNO domain works (a `jno.Shape`, `jno.domain.cube`, a CSG/`gmsh` constructor).
  Add `time=(t0, t1, n_steps)` to make it transient.
* **Symbols** — `u, phi = d.fem_symbols(value_shape=(), names=("u", "phi"), order=1)`.
  Use `value_shape=(2,)` for a vector unknown (elasticity, flow velocity), `order=k` for degree-`k`
  Lagrange (`order=2` quadratic P2, `order=3` cubic P3, … — any `k ≥ 1`), `space="RT"`/`"N1E"`/`"P0"`
  for the non-nodal families (see below), and call `fem_symbols` once per field for coupled systems.
  On a **curved boundary**, `order ≥ 2` is capped by the straight-edge geometry — see *Known
  limitations* for the measured rates before paying for P3 there.
  A **1D line domain** takes a vector unknown too (`value_shape=(n,)`), so a 1D *system* — a
  two-species model, a Timoshenko pair, a bar with several dofs per node — is one field with
  node-major dofs and per-component essential conditions (`u(region)[i] - g`).
  On a 1D line domain **any order `k ≥ 1`** is available: degree `k` adds `k-1` interior dofs per
  element, laid out after all vertices, so read `fem.points` for the coordinates the solution lives
  on. Orders above 2 are tabulated by basix on the reference interval through the same builder the
  2D/3D path uses, so a P{k} line and a P{k} triangle agree on what degree `k` means. Measured on
  `-u'' + u = f`: at the **vertices** 1D Lagrange is superconvergent at O(h^2k) — P1 gives O(h²), P2
  O(h⁴), P3 O(h⁶) — and each P{k} reproduces a degree-`k` solution exactly.
  A **coupled** 1D system carries a per-field order, so a mixed-order pair (the 1D Taylor-Hood shape)
  assembles — the blocks are then unequal and the coupling blocks rectangular, so read
  `fem.field_points` for each field's dof coordinates. In a coupled *transient* system a field may be
  **algebraic** (no `u_t`): its mass rows are zero, so the block is a DAE and the implicit step solves
  `A p = c` on those rows — which is how a constraint/closure field (a pressure, a saturation, an
  equilibrium concentration) is written.
  A `jno.np.parameter` coefficient (scalar or nodal field) also works on a **steady linear** 1D form, so
  a 1D differentiable inverse problem runs through `crux.solve` — as does a **neural** (`jno.nn(...)`)
  coefficient, so a learned `k(x)` can be trained from 1D data. Transient too — recovering a diffusivity
  from a 1D time series works — except that the transient **mass** must be parameter-free (it is
  assembled once, so a parameter there would be silently frozen; it fails loud). *Nonlinear* forms are
  parametric too — Newton runs on `R(·, θ)` and implicit differentiation gives `∂u/∂θ`. A **coupled**
  1D system is parametric too (steady, linear and nonlinear) — the block element kernels publish the
  same `volume_vars` / neural-table keys the single-field ones do, so the shared evaluator reads them
  regardless of field layout. Not wired in 1D: a parameter on a coupled *transient* block, which is
  assembled once and would freeze the parameter at its placeholder (it fails loud). The vector and
  triangle-only non-nodal families (`"RT"`, `"N1curl"`, `"Argyris"`, `"Morley"`) have no 1D counterpart
  and raise a clear error on a line.
* **VPINN in 1D** — a network trial (`u = net(x)` inside the weak form, test-projected onto the FE
  test space) works on a line, so the cheapest dimension for prototyping a variational PINN is
  available. The native `fem_context` it projects onto now builds on an interval: `lagrange_interval`
  is the 1D sibling of `lagrange_triangle`/`lagrange_tet` from the same basix builder, and an
  interval's facets are its two endpoints (outward normals ∓1). Still single-field only.
* **Adaptivity in 1D** — `fem.solve(adapt=jno.solve.remesh(...))` works on a line, steady and
  transient. mmg has no 1-D mode and needs none: an interval mesh is a sorted vertex list, so
  honouring a size field is subdivision rather than remeshing (exact where mmg is approximate, and no
  optional dependency), with mmg's `hgrad` gradation rule imposed by two monotone sweeps. Solution
  transfer is the same code as 2D/3D — an interval is a 1-simplex, so its barycentric weights are the
  two linear hat values. Measured on a boundary layer `-eps u'' + u' = 0`: adaptive refinement from 11
  dofs reaches 5.7e-4 where uniform refinement at 81 dofs is at 1.2e-2. Not wired in 1D: mesh motion
  via a geometry term (a 1-D "moving boundary" is a single endpoint).
* **The beam element (`space="Hermite"` in 1D)** — the C¹ cubic Hermite, i.e. the classical
  Euler-Bernoulli beam element and the 1D counterpart of Argyris/Morley on triangles. Two dofs per
  vertex, `(w, dw/dx)`, laid out `2*node` / `2*node + 1`; sharing the slope dof across elements is what
  makes the space C¹, so the fourth-order operator has a well-defined weak form:

  ```python
  u, v = d.fem_symbols(space="Hermite")
  ui, vi = u.bind(x=xi), v.bind(x=xi)
  lap = jno.np.laplacian
  fem = jno.fem([EI * lap(ui, [xi]) * lap(vi, [xi]) - q * vi,   # EI w'''' = q
                 u(xl) - 0.0, u.dn(xl) - 0.0])                  # clamped at the left end
  ```

  The classical supports are just *which* of a node's two dofs are pinned — `u(region)` alone is
  **simply supported**, adding `u.dn(region)` makes it **clamped**, `u.dn` alone is **guided**, and
  neither is **free**. The slope condition rides the same `u.dn` essential-rotation channel the 2D C¹
  plate families use, so a beam and a plate are clamped by the same notation. The element is *nodally
  exact* for a uniform load: a cantilever gives `qL⁴/8` at the tip and `qL³/6` for the tip slope to
  machine precision, simply supported gives `5qL⁴/384` at mid-span and clamped-clamped `qL⁴/384`.
* **Complex forms in 1D** — a `1j` anywhere in a 1D weak form routes through the same real-equivalent
  Re/Im split the 2D/3D and non-nodal paths use, so 1D Helmholtz-type problems (complex coefficient,
  complex source, or both) solve and return a complex `u`. A runtime parameter inside the complex
  coefficient keeps both legs parametric, so the complex **inverse** works in 1D too. Scope: steady and
  linear, single field. Complex *transient*, complex *nonlinear*, a complex *coupled* 1D system, and a
  **complex essential value** each raise — the last because the two legs share one Dirichlet row set,
  which can impose `Re u = g` with `Im u = 0` but not a prescribed `Im u`.
* **Quadrature coordinates** — `d.variable("interior", split=True)` returns the volume
  coordinates; `d.variable("<edge>", split=True)` returns a boundary edge's coordinates. A
  `Shape.rect` auto-tags `"left"`, `"right"`, `"bottom"`, `"top"` (and `"front"`/`"back"` for a box);
  `"boundary"` is the whole boundary and `"initial"` the `t = t0` slice. To define a custom region
  and fetch its coordinates in one call, pass a predicate: `d.variable("port", where=lambda x, y: x < 1e-6)`
  tags `"port"` (exactly as `d.tag` would) and returns its split coordinates.
* **Bound views** — `ui = u.bind(x=xi, y=yi, t=ti)` ties a symbol to a set of coordinates.
  The value is `ui`; spatial derivatives are `ui.x`, `ui.y`, `ui.z`; the time derivative is `ui.t`.
* **Second derivatives (4th-order weak forms).** `jno.np.laplacian(ui, [xi, yi])` (the Laplacian `Δu`)
  and `jno.np.hessian(ui, [xi, yi])` (the full `D²u`) assemble against the element's second shape-function
  derivatives, so a biharmonic / plate / Cahn–Hilliard form is written directly, e.g.
  `jno.np.laplacian(ui, [xi, yi]) * jno.np.laplacian(vi, [xi, yi])` for `∫Δu·Δv`. Needs **`order ≥ 2`**
  (a P1 Hessian is identically zero), scalar Lagrange fields only.
  > **Conformity caveat.** Standard Lagrange is **C⁰**, so `∫Δu·Δv` over P2 is *non-conforming* and does
  > **not** give a convergent biharmonic discretisation. For a convergent solve use a purpose-built
  > biharmonic element — the **C¹ Argyris** element (`space="Argyris"`) or the cheaper **non-conforming
  > Morley** element (`space="Morley"`, full-Hessian form `∫D²u:D²v`) — or the mixed (Ciarlet–Raviart)
  > method (two coupled C⁰ fields with `w = Δu`; see `tests/test_fem_hessian.py`).

---

## Boundary conditions are residual terms

There is no separate `jno.dirichlet(...)`/`neumann(...)` call — every condition is just a term
in the `jno.fem([...])` list, and `jno.fem` classifies each by the region it is bound to (see
`fem.classification`).

| Condition | Term |
|-----------|------|
| Dirichlet `u = g` | `u(xb, yb) - g` |
| Per-component (roller) `u_i = g` | `u(xb, yb)[i] - g` |
| Neumann flux `du/dn = g` | `-g * phi.bind(x=xb, y=yb)` |
| Robin `du/dn + a u = g` | `(a * u.bind(x=xb, y=yb) - g) * phi.bind(x=xb, y=yb)` |
| Vector traction `t` | `-jno.np.inner(t, phi.bind(x=xb, y=yb), n_contract=1)` |

`g` may be a constant or a coordinate expression (e.g. `u(xb, yb) - jno.np.sin(jno.np.pi * xb)`
for a spatially varying Dirichlet value). A zero Neumann flux is the natural default and needs
no term.

#### Per-tag surface coefficients — `d.by_tag({...})`

A boundary term is normally written per tag, on that tag's coordinates. When the *same* condition
applies over the whole boundary with only its coefficient changing, `by_tag` collapses it into one
term — the surface mirror of `by_region`:

```python
d.tag("wall", lambda x, y: x < 1e-9)
d.tag("lid",  lambda x, y: y > 1 - 1e-9)

h = d.by_tag({"wall": 25.0, "lid": 5.0})          # per-tag film coefficient
xb, yb, _ = d.variable("boundary", split=True)
ub, vb = u.bind(x=xb, y=yb), v.bind(x=xb, y=yb)
robin = h * (ub - T_inf) * vb                      # ONE term, both tags
```

It desugars to `sum_t TagMask(t) * values[t]`, and assembles the identical operator and load vector as
the per-tag term loop. A facet belongs to a tag by the assembler's own facet selection — the same rule
that decides which facets a Dirichlet condition on that tag pins — so the two can never disagree.
Values may be anything a coefficient can be (scalars, expressions, trainable parameters, typed views),
`default=` covers the facets no listed tag claims, and `d.attach("wall", h=25.0)` declares the value on
the tag itself so the term reads `d.h * (ub - T_inf) * vb`.

**Limits, all loud:** surface terms only — a `TagMask` in a *volume* term raises, as does `by_tag` on
a non-nodal space (N1E / RT / Morley / Argyris) or in 1-D, where the per-facet mask is not threaded. A
tag owning no boundary facet on the mesh raises rather than integrating over nothing. Facets that no
listed tag claims contribute nothing unless `default=` is given — untagged boundary is deliberately
natural (do-nothing) in jNO, so tags are not required to partition the boundary.

### Inequalities — `u.bounds(lo, hi)`

A **box constraint** is the inequality sibling of a Dirichlet condition, so it is a term too, and
`fem.solve()` still takes nothing:

```python
jno.fem([
    inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi,   # -Δu = -1
    u(*ends) - 0.0,
    u.bounds(-c, None),                               # an obstacle from below (one side; None = free)
])
```

This turns the solve into a **variational inequality**. Instead of `R(u) = 0` everywhere, the solution
satisfies the KKT conditions

| where | condition |
|---|---|
| `lo < u < hi` | `R = 0` (equilibrium) |
| `u = lo` | `R ≥ 0` (the constraint pushes back) |
| `u = hi` | `R ≤ 0` |

which are exactly the zeros of the **min-map** `min(max(R, u - hi), u - lo)` — the natural residual of
a box-constrained VI (Facchinei & Pang, *Finite-Dimensional Variational Inequalities and
Complementarity Problems*, Springer 2003, §1.5). That function is semismooth rather than smooth, and
Newton on it converges locally superlinearly (Qi & Sun, *A nonsmooth version of Newton's method*,
Math. Programming **58**, 1993). No new solver is involved: `jax.linearize` differentiates through
`min`/`max` by selecting the active branch, which *is* the semismooth Jacobian, so the existing
Newton–Krylov and sparse-direct drivers apply unchanged — and `lax.custom_root` differentiates the
result on the same operator.

**A bound is solved, not clipped.** Clipping an unconstrained solution satisfies the bound just as
exactly and gives the wrong answer, because it puts the *free boundary* in the wrong place. On the
classic obstacle problem `-u'' = -1`, `u(0)=u(1)=0`, `u ≥ -c`, the membrane leaves the obstacle where
it meets it **tangentially**, at `x = √(2c)`; a clip detaches where the unconstrained parabola
**crosses** `-c`, at `x = (1-√(1-8c))/2`. At `c = 1/18` that is 0.333 versus 0.127.

`lo`/`hi` accept a number, a coordinate expression (evaluated at that field's DOF points, like a
Dirichlet value), or `u.i(-1)` — the previous load step on a `domain(tau=...)` march, which gives
**bound-constrained irreversibility**: `u.bounds(u.i(-1), None)` lets a field ratchet up and never
come back down. They may not depend on the live unknown; that is a general complementarity problem
rather than a box, and is rejected.

!!! warning "Sign convention"
    The min-map takes the multiplier's sign from the residual's, so the weak form must be written in
    the standard variational orientation `a(u,v) - L(v)` — the gradient of an energy, which is how
    every form in these docs is written. Written with the opposite sign it states the *other*
    inequality. This cannot be detected from the residual alone, so it is a convention, not a check.

**Scope.** Bounds are wired on the steady residual path (real, 2D/3D native Lagrange, single-field or
coupled), including inside a `tau=` load-path march; a transient or complex assembly is rejected with
a clear error. One box per field. Note that a bound is not a cure for an ill-posed operator: in a
phase-field form `dm.bounds(0, 1)` keeps the damage in range but does **not** remove the need for a
floor on `(1-dm)²`, which at `dm = 1` would otherwise make the displacement block singular. And on a
non-convex energy a monolithic Newton is not expected to converge whether or not a bound is present —
see the staggered driver. Non-convergence raises on an eager solve; inside a march it cannot (the
step runs under `lax.scan`), which is the same pre-existing limitation every marched Newton has.

### Components and derivatives — `u[i]` vs `u.x`

For a vector field the two are distinct spellings and each means one thing:

| spelling | meaning |
|---|---|
| `u[i]`, `u[..., i]`, `u.vector[i]`, `u(region)[i]` | the **i-th component** |
| `u.d(x)`, `u.x` on a bound view, `u.t` | the **derivative** |

`u[i]` is the component *everywhere* — on a raw `fem_symbols` field exactly as on a typed view — and
all four component spellings assemble the identical term. Indexing a **scalar** field raises: it has no
components, and the message points at `u.d(x)`.

(Historically a raw `u[0]` indexed the leading array axis, which at assembly is quadrature points, so
it died inside the assembler with a broadcast error naming nothing — while `u(region)[0]` and
`u.vector[0]`, built by the views as `u[..., 0]`, selected the component correctly.)

### Reading the reaction off a constrained region — `fem.eval`

The quantity conjugate to an essential condition is the **reaction**: force in mechanics, total heat
flux through a Dirichlet wall, current in electrostatics, flow rate in Darcy. It is one operation, and
it is arithmetic on a residual — but not on the residual any solve path keeps:

```python
fem = jno.fem([mech, u(*left)[0] - 0.0, u(*left)[1] - 0.0])
u_h = fem.solve()
R   = fem.eval(mech, u_h)                                  # free residual, one value per DOF
Fx  = R[fem.region_dofs("left", component=0)].sum()        # reaction on the pinned face
```

`fem.eval(term, u)` assembles a weak term at a solution with **no essential elimination applied**, and
`fem.region_dofs(region, field=…, component=…)` gives that region's global DOF indices.

**Why this needs its own entry point.** Every solve path elimination-mutates the system it keeps: the
linear route applies symmetric elimination (`fem.A`/`fem.b` have the constrained rows zeroed and a unit
diagonal set), and Newton replaces those rows with `u[d] - g`. Both are right for solving, and both are
**exactly zero** at the DOFs a reaction asks about — so reading it off `fem.A`, `fem.b` or
`fem.residual` returns a plausible, silent zero rather than an error. (`fem.residual` also refuses
outright on a linear problem, which is the commonest reaction case.)

`term` is any weak term built from this domain's symbols and does not have to be one the FEM was built
from, so a diagnostic form can be assembled against an existing solution. **Scope:** volume terms on
the native Lagrange assembler. A term with no test function is a field readout rather than an assembly,
and a surface term needs the front-end's per-region facet bucketing — both are refused by name.

Verified by global balance, not by restating the assembly: the wall flux equals the integrated source,
and the reaction equals the applied load.

### Tying two boundaries — `u(A) - u(B)`

A term that names two boundary regions and carries no test function is a **tie**: it identifies the
DOFs on region `A` with those on region `B`. It is enforced by algebraic reduction (a prolongation
`P` that eliminates the `A` DOFs), not by assembly, so it composes with everything downstream —
complex, transient, Bloch (`u(A) - c*u(B)`), and `basis=` all reuse the same `P`.

```python
d.tag("left",  lambda x, y: x < 1e-9)      # a tag predicate includes the corner nodes,
d.tag("right", lambda x, y: x > 1 - 1e-9)  # which matters — see below

fem = jno.fem([weak_form, u("left") - u("right")])
```

How the two faces are identified depends on their meshes:

* **Conforming** (the node layouts match) — an exact node-to-node 0/1 map. This is the cheap path and
  it keeps the fast selection-based reduction.
* **Non-matching**, when both faces carry facet connectivity and the main face covers the secondary —
  a **dual-mortar** coupling (Bernardi/Maday/Patera 1994; dual multiplier spaces from Wohlmuth 2000):
  the tie is imposed in the integral sense `∫ ψ (u_A − u_B∘Φ) = 0` over the secondary face, segmented
  against the main facets. Interval clipping in 2-D, polygon clipping in 3-D.
* **Non-matching, otherwise** (native 1-D chains, a tag that selects nodes but no whole facet, or two
  faces that do not cover each other) — node-to-segment **collocation**: each secondary node takes the
  main facet value at its own location.

Worth being precise about what the mortar coupling buys, because it is less than the usual framing
suggests. jNO enforces a tie by main–secondary **elimination** through a prolongation `P`, and such a
scheme passes the linear patch test whenever `P` reproduces linear fields — which node-to-segment
interpolation does, in 2-D *and* 3-D. So **the patch test does not separate the two couplings here**;
the textbook "node-to-segment fails the patch test" result is about contact formulations that
distribute nodal forces, not about a linearly-complete MPC elimination.

What does differ: for a field the main space cannot represent, mortar returns the integral (L²)
projection and collocation the pointwise value. Measured on a non-matching 3-D interface, mortar's RMS
error is 4–40 % lower across a range of mesh ratios. The two also coincide exactly when the main
nodes are a subset of the secondary nodes, since the main basis then lies inside the secondary space.

**P2 triangular interfaces (3-D quadratic) stay collocated**, and this is a theorem rather than a gap
in the implementation. The dual basis is built from the facet mass matrix as `A = diag(∫N)·Mass⁻¹`, and
the P2 triangle's vertex functions integrate to *exactly zero* (`∫L(2L−1) = 2/12 − 1/6`), so the
scaling is singular. Rescaling does produce a biorthogonal basis, but not one whose span contains the
linear functions — and Lemma 3.4 of Lamichhane's thesis proves no locally supported dual space of that
dimension can, which is precisely what the optimal error estimate requires. The published remedy uses
*fewer* multipliers than secondary DOFs, making the tie a constrained solve rather than an elimination,
which a prolongation cannot express. P2 **edges** (2-D) are unaffected — there `∫N = 1/6` — and the
same source confirms the 2-D quadratic dual space does contain the linear hats.

Two practical consequences: tag periodic faces with a **predicate** (`d.tag(name, lambda ...)`) so each
face includes its corner nodes — a face tagged from geometry may drop them, leaving the two sides with
different extents, which both disqualifies mortar and leaves the corner DOFs untied. And multidirectional
periodicity *requires* shared corners; `jno.fem` raises rather than silently mis-solving if they are absent.

### Gluing two independently meshed bodies

`Shape.regions` fragments its pieces, so a shared interface meshes **conforming** — one node set, no tie
needed. `conforming=False` skips the fragment: each piece is meshed on its own, and two touching regions
end up with two coincident but non-matching surfaces and duplicated nodes. That is how you join bodies
meshed at different resolutions, or couple subdomains you would rather mesh separately.

The two sides are spatially *identical*, so no `d.tag` predicate can separate them — the emitter names
them, extending the `"a|b"` convention it already uses for material interfaces:

```python
d = jno.Shape.regions(
        lower=jno.Shape.box(0, 0, 0, 1, 1, 1),
        upper=jno.Shape.box(0, 0, 1, 1, 1, 2.5),
        conforming=False,
    ).sized(0.18).domain()

a = d.variable("lower|upper.lower", split=True)     # one tag per side
b = d.variable("lower|upper.upper", split=True)

fem = jno.fem([
    weak_form,
    u(*a) - u(*b),                                  # glue them
    u(xb, yb, zb) - 0.0,
])
```

The tie then resolves as above — conforming node-to-node where the layouts happen to match, mortar where
they do not.

One subtlety worth knowing, because it is invisible: each interface face **is** a facet of exactly one
cell, so it is topologically part of the boundary. The catch-all `"boundary"` region therefore excludes
nodes that lie *only* on an interface — otherwise `u(boundary) - g` would pin the interface and silently
solve two disconnected bodies. Nodes where the interface meets the outer wall lie on a genuine outer
facet too and stay pinned.

Because the two sides are spatially identical, **name them by the mesh's own tags** (above) or by
`d.tag(pred, region=...)`, which says which body owns the facets:

```python
d.tag("cap_face",  lambda x, y: jnp.abs(y - 1.0) < 1e-9, region="cap")
d.tag("base_face", lambda x, y: jnp.abs(y - 1.0) < 1e-9, region="base")
```

A bare coordinate predicate cannot tell them apart, and a surface term written on such a tag is applied
to **both** bodies.

### Contact — `u.gap(secondary, main)`

A tie makes two surfaces move together. Contact lets them separate and push, and the difference is one
term. `u.gap` is the signed normal gap at the secondary face's quadrature points:

$$g = g_0 - n \cdot (u_s - u_m \circ \Phi)$$

`g0` is the initial along-normal separation and `Φ` the mortar projection onto the main surface — the
same projection a tie uses, so a gap and a tie are the same machinery read two ways. The gap is a
symbol, so the contact traction is an ordinary boundary term and **nothing is passed to `fem.solve()`**:

```python
n = d.variable(secondary, normals=True)          # the secondary's OUTWARD normal
g = u.gap(secondary, main, domain=d)
p = jno.np.maximum(0.0, -c * g)              # pressure: positive only when penetrating
fem = jno.fem([..., p * jno.np.inner(n, phi.bind(x=sv[0], y=sv[1]), n_contract=1)])
```

**The sign convention, spelled out**, because every downstream sign follows it:

| quantity | meaning |
|---|---|
| `n` | the **secondary's outward** normal — on a contacting pair it points *at* the main |
| `g > 0` | separated (open) |
| `g < 0` | interpenetrating |
| `p = max(0, -c*g)` | contact pressure, `≥ 0`, active only in penetration |
| `+p * inner(n, phi)` | the traction term — **not** `-p * ...` |

The `+` is not a convention you may flip: since `∂g/∂u_s = -n`, it is the sign that adds a
positive-definite `+c (n·δu)(n·φ)` to the tangent. The opposite sign is anti-stabilising and, measured
on a weakly penalised interface, leaves the jump *larger* than not penalising at all.

You write **one** term, on the secondary face. The equal-and-opposite traction on the main body is
supplied by the pairing — it is the same integrand tested against the main's projected trace — so
Newton's third law holds without you restating it. Ablating that reaction leaves the main body
identically zero.

For a bonded (tied) interface use the two-sided penalty `p = -c*g` instead of the `max`; it is smooth,
and stiffening `c` converges to the tie: two bonded unit blocks squeezed by 0.02 reproduce the single-bar
answer `uy(y=1) = -0.01` to 1.5e-05 at `c = 1e6`.

> **Scope.** Small sliding — the pairing is frozen at build time, so a configuration that slides must be
> rebuilt per load step. Differentiable in the DOF values but **not** in the mesh coordinates (the
> projection weights are host-computed). Frictionless: no tangential traction, so a body held *only* by
> contact is free to slide and its system is singular — constrain the tangential direction independently.
> The gap is non-local (it reads DOFs on the other body's cells), so the tangent is matrix-free;
> `newton(direct=True)` is refused. **One-sided contact does not yet converge to tight tolerances**: the
> `max(0, ·)` kink is non-smooth and the residual stalls around 1e-2 with a line search, against a 1e-8
> target. The bonded two-sided path is smooth and converges. Signorini contact is the same kind of
> object as [`u.bounds(lo, hi)`](#inequalities--uboundslo-hi) — a complementarity condition rather than
> a penalty — so that machinery is the natural route for it, but the reformulation has **not** been
> done and the penalty path above is what exists today.

---

## Non-nodal element families: H(div) and H(curl)

> **⚠️ Experimental.** Validated on 2-D triangular meshes at lowest order (RT₀ / N1E₀); the API may
> still change. Steady-linear, steady-nonlinear (Newton), and transient `M u̇ + A u = c` (including the
> mixed/saddle DAE, e.g. transient Darcy) all work, as does the differentiable `fem.solve()` inverse
> (a **scalar** or spatially-varying **P1 field** `k(x)` in a *volume* term, steady and transient). Every
> edge/cell (RT/N1E/P0) operator — the steady `A`, and the transient **mass `M` and spatial `A`** (also when
> re-assembled per step for a parametric march) — is assembled **sparsely, one element at a time** (a `BCOO`,
> mirroring the native Lagrange assembler), never a dense global `jacfwd`; the mixed-DAE initial state is
> projected by a matrix-free CG on the (SPD) field mass block. So a 3-D N1E **eddy-current / time-domain
> Maxwell** transient scales instead of hitting the `O(n_dof²)` dense-assembly wall past ~10⁴ edges.
>
> **3-D Nédélec (H(curl)) is supported** on a **tetrahedral** mesh: the first-kind `"N1E"` element
> assembles the H(curl) **mass and curl-curl** forms (`inner(u, v) + inner(u.vector.curl(x,y,z),
> v.vector.curl(x,y,z))`) — the correct edge discretisation for **Maxwell / eddy currents** (nodal
> Lagrange gives spurious modes). The covariant push-forward `Φ_phys = J^{-T} Φ_ref` is dimension-agnostic
> and the curl is taken from the physical gradient. The essential **PEC wall** `n×E = 0` is supported,
> written `u.vector.cross(d.variable(region, normals=True))` (facet-based; it pins every boundary-face edge
> DOF of the region). On a tet mesh **only N1E is wired** — RT / P0 / Hermite / Argyris / Morley remain
> 2-D-triangle only and raise. On a **line** mesh `"Hermite"` selects the 1D cubic beam element
> (see above); the other families raise.
>
> **Each of these families has ONE intrinsic order** — RT₀ / N1E₀ lowest, P0 constant, Morley quadratic,
> Hermite cubic, Argyris quintic — set by the element definition, not chosen. `order=` is a nodal-Lagrange
> knob and **is refused** here rather than ignored: `space="N1E", order=2` used to hand back the same
> lowest-order space silently (an identical operator), which is the worst failure shape for a wave problem
> — you pay for accuracy, get first-order convergence, and only find out from a convergence study that
> stalls. Refine the mesh instead (see *Mesh resolution for wave problems* below), or use a nodal Lagrange
> field, where `order=` does apply.
>
> **Not yet:** the rest of the zoo in 3-D (RT / C¹ / plate are 2-D only); the *inhomogeneous* `n×E = g` on
> N1E; higher order; other families (BDM, second-kind Nédélec, Bell); quad / non-triangular meshes; a
> runtime parameter **or trainable neural coefficient** in a **host-assembled** RT-pressure / plate
> boundary term (the N1E tangential-trace impedance / incident **surface** BC *is* differentiable in a
> boundary-term parameter *and* a learned `net(x)` coefficient); the
> constraint-consistent algebraic initial state at `t0` in the saddle-DAE transient (only the reported `t0`
> algebraic value is affected).

Beyond nodal Lagrange (P1/P2), `jno.fem` assembles **edge-DOF** families on 2-D triangles — for
problems whose natural space is *not* H¹. Pick one with the `space=` knob on `fem_symbols`:

| `space` | Space | DOF | Use |
|---------|-------|-----|-----|
| `"Lagrange"` (default) | H¹ | nodal value | standard PDEs |
| `"RT"` | **H(div)** Raviart–Thomas | edge normal flux `∫ₑ u·n` | mixed Poisson, Darcy, conservation |
| `"N1E"` | **H(curl)** Nédélec (1st kind) | edge tangential `∫ₑ u·t` | Maxwell, eddy currents |
| `"P0"` | L² (piecewise constant) | one per cell | the pressure / multiplier of a mixed pair |
| `"Hermite"` | C⁰ cubic, **vertex value + ∇ DOFs** | `u`, `∂u/∂x`, `∂u/∂y` at vertices (+ centroid) | smooth/gradient-aware fields; the foundation for C¹ elements |
| `"Argyris"` | **C¹** quintic (TUBA-6) | value + `∇u` + `D²u` at vertices, `∂u/∂n` at edge midpoints | conforming **biharmonic** / plate / Cahn–Hilliard |
| `"Morley"` | **non-conforming** quadratic (6 DOF) | value at vertices, `∂u/∂n` at edge midpoints | **cheap biharmonic** / plate — scales to fine meshes |

> **Hermite** is the first element with a per-cell **DOF-mixing** transform `M(cell)` (global derivative
> DOFs are the physical gradient `∇u` at the vertices). It is **C⁰** (not C¹), so it is *not* a conforming
> biharmonic element — it de-risks the `M(cell)` machinery the **C¹ Argyris** element reuses. A
> value-Dirichlet `u(region) - g` pins boundary-vertex value DOFs (derivatives free); composes with the
> steady / transient / nonlinear paths (`tests/test_fem_hermite.py`).

> **These families assemble sparsely.** Their *linear* operator is built element-by-element like every
> other family, so peak memory grows as `n^1.0` and is set by the operator rather than by an
> intermediate. Argyris reaches **22,511 DOFs** on an 8 GB card; Morley and Hermite go further still.
>
> The solve stays **sparse-direct** for these families. Going sparse would otherwise have handed 4th-order
> biharmonic operators to the Jacobi-preconditioned BiCGStab that serves real elliptic systems, where it
> does not converge — a solver change disguised as a storage change. Their **second-order-in-time** path
> is still dense, as it is for every non-nodal family.

> **Argyris** is the **C¹-conforming** quintic triangle (21 DOF) — the element for **4th-order PDEs**.
> Across a shared edge both `u` and `∂u/∂n` are continuous, so `∫Δu·Δv` is a *convergent* biharmonic
> discretisation (the conformity caveat above does not apply). The reference dual basis is mapped to each
> physical cell by the affine-equivalence DOF-transform `M(cell)` (Kirby, *SMAI J. Comput. Math.* 4, 2018;
> element: Argyris–Fried–Scharpf 1968); its **globally-oriented edge-normal DOF** is what makes it C¹ on an
> *unstructured* mesh. Essential BCs are the two plate traces — `u(region) - g` (deflection) and
> `u.dn(region) - h` (rotation `∂u/∂n`); the boundary curvature `∂²u/∂n²` is always left **free**. These are
> wired for **axis-aligned boundary edges** only; a non-axis-aligned edge is **rejected with a clear error**
> (use Morley there). Composes with steady / transient / nonlinear `fem.solve()` and **inverse** (scalar or
> P1-field `k(x)`, steady and transient — `tests/test_fem_argyris.py`, `tests/test_fem_inverse.py`).

> **Morley** is the **cheapest** biharmonic element (6 DOF, quadratic). It is **non-conforming** — yet
> passes the patch test and converges (energy `O(h)`, L² `O(h²)`). It reuses Argyris's `M(cell)` transform
> and globally-oriented edge-normal DOF but with ~3.5× fewer DOF, so it is **much cheaper per node**
> and scales to finer meshes. **Modelling subtlety:** because it is non-conforming,
> the biharmonic form must be the **full-Hessian inner product** `inner(hessian(u), hessian(v))` (`∫D²u:D²v`),
> *not* `∫Δu·Δv` — the Laplacian form is singular for Morley (`xy` has `Δu = 0` but `D²u ≠ 0`, a spurious
> kernel). Its two plate traces `u(region) - g` and `u.dn(region) - h` work on **any** boundary orientation.
> **Periodic ties** compose too — `u(top) - u(bottom)` ties the vertex-value *and* the edge-normal-derivative
> DOFs across a matched (conforming) boundary pair, so a y-periodic biharmonic solve recovers a manufactured
> `sin(πx)sin(2πy)` at the optimal L² rate (`tests/test_fem_morley.py::test_morley_periodic_biharmonic_convergence`);
> the reduction requires **conforming** periodic boundaries (matching vertices/edges) and is Morley-only for now —
> the other C¹ families (Argyris, Hermite) raise a clear `NotImplementedError`
> (Morley, *Aeronautical Quarterly* **19**, 1968; `tests/test_fem_morley.py`).

> **Plate boundary conditions.** For a 4th-order (plate/biharmonic) field the boundary trace has two
> essential parts to pin independently — the **deflection** `u(region) - g` and the **rotation**
> `u.dn(region) - h` — plus two conjugate **natural** parts (bending moment `M_n`, effective shear `V_n`)
> that emerge on any trace you *don't* pin. The classical BCs compose from these:
>
> | BC | Physics | How to write it (on a region) |
> |----|---------|-------------------------------|
> | **Clamped** | `w=0`, `∂w/∂n=0` | `u(reg)-g`, `u.dn(reg)-0` |
> | **Simply-supported** | `w=0`, `M_n=0` | `u(reg)-g` |
> | **Guided / sliding** | `∂w/∂n=0`, `V_n=0` | `u.dn(reg)-h` |
> | **Free** | `M_n=0`, `V_n=0` | *(write neither — natural)* |
>
> A free edge gets the natural `M_n=V_n=0` from the ν-weighted plate energy
> `(1-ν)·inner(hessian(u),hessian(v)) + ν·laplacian(u)·laplacian(v)` (validated against Timoshenko's
> square-plate coefficients). A *nonzero* prescribed edge moment is the boundary load `M_n * phi.dn(region)`
> (`∮_region M_n ∂φ/∂n ds`), built from each cell's geometry so it applies on **any** edge orientation
> (though on Argyris the *essential* pin is still axis-aligned-only). The conjugate **shear** load `V_n` is
> not yet wired.

```python
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")   # H(div) flux
p, q = d.fem_symbols(names=("p", "q"), space="P0")                     # piecewise-constant scalar
```

**Vector operators** (on a bound vector view): `u.div(x, y)` is the divergence and `u.curl(x, y)` the
2-D scalar curl `∂uy/∂x − ∂ux/∂y`; in 3-D `u.curl(x, y, z)` returns the **curl vector** (used for the
N1E tet curl-curl form). After binding, the no-arg `u.div()` / `u.curl()` reuse the bound coordinates.

**Essential (edge-trace) BCs** — the outward normal is `d.variable(region, normals=True, split=True)`:

| Family | Trace | Term |
|--------|-------|------|
| RT  | normal flux `u·n = g` | `u(b)[0]*nx + u(b)[1]*ny - g` |
| N1E | tangential `u×n = g`  | `u(b)[0]*ny - u(b)[1]*nx - g` |

For the RT mixed-Poisson saddle, a Dirichlet condition on the scalar `p` is *natural* — add the weak
term `p_D * (v[0]*nx + v[1]*ny)`, no essential constraint on the flux. A BC may target a sub-region
(a `Shape.rect` edge tag or any `d.tag(...)` boundary subset; sub-region normals are computed from the
geometry).

Tutorials: `mixed_poisson_rt_2d.py` (H(div)), `maxwell_nedelec_2d.py` (H(curl): magnetostatics + eddy
current), and `maxwell_nedelec_3d.py` (the **3-D PEC cube cavity resonator** — recovers `k²=π²(l²+m²+n²)`,
spurious-free, converging to `2π²` from below).

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

### Mesh resolution for wave problems

Because N1E is **lowest order only** (`order=` is refused, above), the mesh is your *only* accuracy
knob on a wave problem — so it is worth knowing what a given resolution buys. Measured on the PEC cube
cavity, whose lowest mode `k² = 2π²` is analytic, with `h` expressed as **points per wavelength**
(`λ = 2π/k = √2` on the unit cube, `ppw = λ/h`):

| `h` | ppw | DOFs | mean lowest triplet | rel. error |
|-----|-----|------|--------------------|------------|
| 0.50 | 2.8 | 53 | 14.52 | 26% |
| 0.40 | 3.5 | 121 | 18.67 | 5.4% |
| 0.30 | 4.7 | 217 | 19.48 | 1.3% |
| 0.25 | 5.7 | 276 | 19.07 | 3.4% |
| 0.20 | 7.1 | 532 | 19.27 | 2.4% |
| 0.16 | 8.8 | 1083 | 19.46 | 1.4% |

The fitted rate is **2.1 in `h`**, consistent with the theoretical `O(h²)` for *eigenvalues* with
lowest-order edge elements (the *field* itself converges at `O(h)`). Extrapolating the fit: **~9 ppw for
1%** and **~28 ppw for 0.1%** eigenvalue error.

Read those numbers with two caveats, both real:

- **The sequence is not monotone** (0.25 is worse than 0.30). gmsh meshes at different `h` are
  unstructured and non-nested, and the mode is 3-fold degenerate, so the triplet mean wobbles by a
  percent or so between meshes. Treat the table as a trend, not a per-`h` guarantee — and do your own
  convergence check on your own geometry rather than reading a single number off a single mesh.
- **A cavity eigenvalue is the friendly case.** A *driven* problem at high frequency additionally
  suffers the **pollution effect**: holding `ppw` fixed as the domain grows in wavelengths does *not*
  hold the error fixed — it degrades with `k(kh)^{2p}` (Babuška & Sauter, *Is the pollution effect of
  the FEM avoidable for the Helmholtz equation considering high wave numbers?*, SIAM J. Numer. Anal.
  **34**(6), 2392–2423, 1997). So an electrically large problem needs *more* points per wavelength
  than a small one, and lowest order is where that bites hardest. If you need many wavelengths across
  the domain, budget accordingly.

---

## What `jno.fem` returns

`jno.fem` picks the operator type from the form:

| Form | `fem.is_linear` / `is_transient` | Use |
|------|----------------------------------|-----|
| steady, linear in `u` | `True` / `False` | `fem.A`, `fem.b` → `jnp.linalg.solve` |
| steady, nonlinear in `u` | `False` / `False` | `fem.residual(u)`, `fem.jacobian(u)`, `fem.dofs` (Newton) |
| has a `u.t` term | — / `True` | `fem.M`, `fem.operator.A`, `fem.state0`, `fem.dt`, `fem.t0`, `fem.t1` |

Always-available: `fem.dofs`, `fem.points` (the coordinates the DOFs live on — use these for P2,
where they differ from the mesh vertices), `fem.operator`, and `fem.classification`.

> **Operator storage.** The assembled BCOO stores each `(row, col)` pair **once**. The assemblers
> emit one triplet block per additive weak-form term and every interior DOF pair receives a
> contribution from each element sharing it (~20 tets for P1), so the raw triplets are ~10–20×
> redundant; they are summed once at assembly instead of lazily on every matvec. Measured on a 3-D
> transient heat block: mass 18240 → 753 stored triplets, operator 54992 → 1025, step operator
> `M + dt·A` 73232 → 1025 (1.12 → 0.02 MiB); on a 3-D nonlinear Jacobian, 36752 → 3841 with a 6.6×
> faster matvec. Two consequences if you read the triplets yourself: `nse` is the true nonzero count,
> and **numerically-zero entries are dropped** — symmetric Dirichlet elimination leaves whole rows of
> them behind, and on a Dirichlet-heavy operator that removal is the larger of the two effects. Code
> that needs the *structural* pattern (an entry that is zero at this parameter value but nonzero at
> another) must not read it off the assembled operator.

> **Element-loop chunking — `jno.fem(chunk=)`.** A single `vmap` over every cell materialises the
> whole batched intermediate at once, and on a 3-D mesh that intermediate — not the assembled
> operator — is what sets the memory ceiling. The element loop is therefore chunked: measured on a 3-D
> nonlinear solve, peak memory fell from 2324 MiB to 509 MiB at 52k DOFs (and 1378 → 381 at 31k), for
> roughly 15% more assembly time. Reverse mode keeps the win — `scan` stores per-iteration residuals,
> but the chunked total is still below the unchunked one (183.5 vs 194.4 MiB).
>
> The default needs no tuning: a chunk may use **~0.15% of device memory**, taken from
> `memory_stats()["bytes_limit"]`, so the same problem that must be split on an 8 GB card runs
> unsplit — and therefore at full speed — on an 80 GB one. A cell floor (8192) keeps large per-cell
> blocks (P2 tets at 8 KB/cell, vector P1 at 13.8 KB/cell) from collapsing below GPU saturation, where
> the measured penalty is a ~2× slowdown; that floor is the one number JAX gives no way to derive, as
> it exposes device memory but not the SM count.
>
> `chunk=False` restores the single `vmap`, and a positive int pins cells per chunk — as an upper
> bound: whatever size is chosen (default or explicit) is shrunk to the smallest one giving the *same
> number of chunks*, so the chunks come out even where they can. That is a compile-time concern, not a
> memory one: `lax.map` emits the element kernel **twice** when the chunk does not divide the cell
> count — once as the scan body, once unrolled for the tail — and on a problem the size of Taylor–Hood
> Stokes that duplicate doubles the build. Evening the chunks costs nothing in either direction: the
> chunk *count* is unchanged (no extra scan step) and each chunk is no larger than requested (peak
> memory only falls). It cannot always divide — 100 items in 3 chunks is 34+34+32 — and then it is
> simply a no-op.
>
> `chunk=` lives on `jno.fem` rather than `fem.solve` because the steady-linear operator is assembled
> *here*, before any solve.
>
> Coverage, since it is not uniform. The **native** assembler chunks its residual and jacobian, volume
> and surface loops. The **non-nodal** assembler chunks both, for every family — including the C⁰/C¹
> vertex families (see below). The **nonlinear tangent** is per-element too: the assembler takes the
> current iterate, so `J(u_k)` assembles element-by-element like the linear operator. One path still
> takes a global dense `jacfwd` and is unchunked, for **every** non-nodal family rather than any
> particular one: the **second-order-in-time** block, which builds a dense `2n × 2n` system by
> construction. The
> **1-D** assembler is not chunked at all, and an explicit `chunk=` there raises rather than being
> silently ignored.

> **Term introspection (provisional).** `fem.term_kinds` returns a `list[TermKind]` — each
> additively-split PDE (volume) term classified by `support`, `time_order`, `trial_channel` /
> `test_channel`, and `linear`, with `is_local` flagging a spatially pointwise term (reaction/mass) vs. a
> neighbour-coupling global one (diffusion/advection). This is the basis for operator-splitting routing;
> the API is provisional until that routing lands.

### Steady linear

```python
u_h = jnp.linalg.solve(dense(fem.A), jnp.asarray(fem.b).reshape(-1))
```

### Steady nonlinear (Newton)

A cubic reaction `+ (u**3 - u) * vi` makes the form nonlinear, so `jno.fem` returns a residual
operator — solve it with any Newton/root-finder using `fem.residual` and `fem.jacobian`:

```python
import scipy.optimize as spo
sol = spo.root(lambda v: np.asarray(fem.residual(v)),
               np.zeros(fem.dofs),
               jac=lambda v: np.asarray(fem.jacobian(v)), method="hybr")
```

### Transient (semidiscrete `M u̇ + A u = c`)

```python
M, A, dt = fem.M, dense(fem.operator.A), float(fem.dt)  # fem.M is dense; operator.A is raw sparse
w = fem.state0
for _ in range(round((fem.t1 - fem.t0) / dt)):          # backward Euler
    w = jnp.linalg.solve(M + dt * A, M @ w)
```

### Second order in time (`u_tt`) — wave / elastodynamics

A weak form carrying a **second** time derivative (`ui.tt`) is auto-reduced to the equivalent
first-order system in `y = [u, v]` (velocity `v = u_t`) and integrated by the energy-conserving
**trapezoidal rule** (θ=½, equivalent to Newmark average-acceleration — Newmark 1959, β=¼, γ=½) —
backward Euler would spuriously damp an undamped wave. A second-order problem needs **two** initial
conditions: displacement `u(initial) - u0` and velocity `u.t(initial) - v0` (bind the velocity IC with
the `"initial"`-slice coordinates *and time*, `u.bind(x=xi0, y=yi0, t=ti0).t`; a missing velocity IC
defaults to zero).

```python
d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1), time=(0.0, 2.0, 200))
u, phi = d.fem_symbols()
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
xi0, yi0, ti0 = d.variable("initial", split=True)          # initial slice: coords + time
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
ui0 = u.bind(x=xi0, y=yi0, t=ti0)

# u_tt = Δu  ->  ∫ u_tt φ + ∫ ∇u·∇φ = 0
fem = jno.fem([ui.tt * vi + (ui.x * vi.x + ui.y * vi.y),
               u(xb, yb) - 0.0,                            # fixed boundary
               u(xi0, yi0) - jno.fn(u0_fn, [xi0, yi0]),    # displacement IC
               ui0.t - 0.0])                               # velocity IC (here: at rest)
```

The assembled block is a standard transient block, so `fem.M` / `fem.state0` and the differentiable
`fem.solve()` work unchanged. The state is `y = [u; v]` of size `2N`; use `fem.offsets` (`[0, N, 2N]`)
to split it — displacement is `y[:N]`, velocity `y[N:]`. Add a damping term `c * ui.t * vi` for a
damped wave.

> **Integrate with `fem.solve()` — or step with θ=½ yourself.** The energy-conserving trapezoidal
> rule lives inside `fem.solve()`. **Do not** hand-roll backward Euler `(M + dt·A) w = M·w` off a
> second-order block — it spuriously **damps** the wave. If you integrate manually, use the trapezoidal
> step `(M + ½·dt·A) w_next = (M − ½·dt·A) w + dt·c`.

A **vector** field works too (`value_shape=(2,)`/`(3,)`) — that is elastodynamics,
`ρ u_tt = ∇·σ(u)` (see the vibrating-cantilever tutorial). A **coupled system** where every field
carries `u_tt` (spring-coupled membranes, coupled waves) rides the same augmented formula with the
coupled blocks — damping `u_t` terms, a nonlinear spatial operator (Newton on the augmented
residual) and a driven boundary `g(x, t)` all apply to the coupled case exactly as to a single
field. *Scope: nodal Lagrange, 2D/3D (1D has its own narrower path). Fail-loud: a coupled field
with no `u_tt` term (write a first-order field as an explicit first-order system), runtime
parameters on a coupled form, a trainable Dirichlet value, and `g(x, t)` on a nonlinear form.*

---

## Coefficient fields — known (`.freeze()`) vs trainable

A coefficient in the weak form (a conductivity `k`, an emissivity, a source weight) can be a plain
constant, a **coordinate function** `jno.fn(lambda x, y: ...)`, or a `jno.np.parameter` — written
straight into the math like any other value:

```python
k = jno.np.parameter(phi).initialize(lambda x, y: 1.0 + 4.0 * x).freeze()   # KNOWN coefficient
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
u_h = jnp.linalg.solve(fem.A, fem.b)                                        # non-parametric forward solve
```

A `jno.np.parameter` is a **trainable** unknown by default — it makes the system runtime-parametric
(resolved through `crux`, below). Marking it **`.freeze()`** declares it a *known* coefficient:
`jno.fem` evaluates its `.initialize` value at the quadrature points — exactly like `jno.fn` — so the
system assembles non-parametrically (`fem.A` / `fem.b`, no `crux`) **and works in every form**
(steady-linear, nonlinear, transient, coupled). The frozen value is a **scalar** (`.initialize(3.0)`)
or a **coordinate function** (`.initialize(lambda x, y: ...)`, scalar- or vector-valued); a raw
per-node array, a JAX initializer, or no value all fail loud. Leave the parameter **un-frozen** to
make it an inverse unknown — the next section. A vector-valued coefficient is best written **per
component** with scalar functions (a single function returning a tuple hits a kernel limit shared with
`jno.fn`).

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

`jno.solve.relocate()` descends the **equidistribution defect** of an arclength monitor through the
differentiable solve, with a **backtracking `det J` line search** — so the fixed node set concentrates at
solution features and the mesh never tangles (the validity constraint lives in the step control; a stock
optimiser or an energy barrier alone cannot guarantee it on a stiff problem — see `run_adaptive_relocate`).
It mutates the domain to the relocated mesh, returns the solution there, and **raises** if no coordinate was
tagged. Works across **linear, nonlinear (Newton), transient (relocates for the whole trajectory via a
time-averaged objective), periodic, and complex** problems, scalar or vector — the objective sums over every
solution block, so a complex field's real and imaginary parts both contribute. Only complex-*transient* is
not wired yet.

Tagging is **literal and per-axis**: `xm.trainable()` frees only the x column. On a boundary that is the
lever for sliding — free an edge's along-edge axis and its nodes redistribute *within* the wall, leave the
normal axis untagged and the domain shape is preserved exactly.

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

> **Vector fields need more than the measure.** Axisymmetric elasticity carries a hoop strain
> `ε_θθ = u_r/r`, and divergence picks up `u_r/r`. Neither can be produced by weighting the Cartesian
> form by anything — they are extra terms you must write out. This is precisely why jNO does not offer
> to apply the weighting automatically: it would be exact for scalars and quietly wrong for vectors.

> **⚠️ This applies to vector EM too, and there is no guard.** An axisymmetric `(r, z)` **vector
> Maxwell / eddy-current** form — the natural geometry for a coil, a solenoid, a tokamak vessel — is
> *not* the Cartesian curl-curl weighted by `2πr`. In cylindrical coordinates the curl of a vector
> field picks up its own `1/r` terms — `(∇×E)_z = (1/r)∂(rE_θ)/∂r` — and for an axisymmetric (`m=0`)
> problem the meridional (`E_r, E_z`) and azimuthal (`E_θ`) components **decouple into two different
> operators** that must each be written out. The azimuthal one reduces to a *scalar* equation (in
> `E_θ`, or in `rA_θ` for the vector-potential/eddy-current form), not a component of the Cartesian
> form, and it needs care on the axis where `1/r` is singular. Weighting an
> `"N1E"` form by `2πr` therefore produces a **silently wrong** answer, not an approximate one.
> Nothing raises: multiplying by `r` is ordinary arithmetic, and the assembler cannot tell it apart
> from a legitimate radial coefficient — so this limit is stated here, where you choose the geometry,
> rather than enforced at assembly. Use a **full 3-D** mesh for vector Maxwell (3-D N1E is wired and
> validated), or derive and write the cylindrical operator yourself as an ordinary scalar/coupled
> form. jNO ships no axisymmetric H(curl)/H(div) element and no meridional/azimuthal split.

> **Enclosure radiation.** `domain.enclosure(tags, axisymmetric=True)` gives ring areas `2πr̄·L` and a
> `gap.load(q)` that is **per full revolution** (W, not W/m). The weak form you add it to must carry
> the same `2πr`, or the two sides differ by exactly that factor. jNO cannot check this for you.

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

`F` is computed purely from geometry (occlusion + orientation) by **double-area Gauss quadrature** of
the diffuse kernel — so a *concave* surface keeps its self-view. Tags only group elements (for
per-surface emissivity); they never block exchange. The enclosure **inherits** `axisymmetric` from the
domain (see below), so its ring areas and the FEM measure cannot disagree; passing a contradicting value
raises. By default the boundary normals point *out of* the mesh (radiation across an un-meshed gap); for
an **oven/furnace cavity** where the meshed fluid is inside, pass `inward=True` so the facing walls see
one another (see the *Oven* tutorial); for a meshed *medium* between solids use `medium_tags`.

> **What blocks a ray (interface mode).** Every meshed region that is *not* listed in `medium_tags`
> is opaque — including a solid that carries no radiating surface of its own. The occluder set is
> resolved once from the region list, not inferred from element tags, and is shared by the visibility
> test and the near-field refinement: a solid with no radiating surface still blocks, and a chord
> through it is never counted as visible.

> **Axisymmetric near field.** The ring kernel's azimuthal integrand peaks at `φ = 0` with width `d/r`,
> so a uniform `n_phi` rule overshoots every near-touching pair (two surfaces meeting in a wedge, and
> every element's own ring self-view) by roughly `dφ/(d/r)`. A graded azimuthal rule fixes it and
> restores closure to ~1e-3, but it needs an occluder model to test its refined chords against, so it
> runs when one is available: **`medium_tags=...`** (interface mode — the solid polygons), or
> **`occlude=False`** (you asserting nothing blocks any ray, e.g. a convex cavity). Plain boundary mode
> with occlusion on keeps the uniform rule plus the `r_min` floor and a closure error around 1e-1, and
> logs a warning saying so. Its occlusion is also a *meridian-only* test reused at every azimuth, which
> is wrong for a general solid of revolution — interface mode checks the true 3-D chord per azimuth.
> Always call `gap.check()`.

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

## Differentiable solve & inverse problems

`fem.solve()` is the **differentiable forward solve as a trace node** — the entry point for
inverse problems. Put a `jno.np.parameter` in the weak form, compare `fem.solve()` to data, and
train the parameter through `crux.solve`. The gradient flows through the solve back to the
parameter (see also [Inverse problems](inverse-problems.md)).

```python
import jax, optax
k = jno.np.parameter((1,), name="k")                      # unknown scalar
k.dtype(jnp.float64); k.initialize(jax.nn.initializers.constant(2.0)); k.optimizer(optax.adam(5e-2))
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs_domain)
crux.solve(200)                                           # recovers k
recovered = crux.eval([k])                                # the array (do not index [0])
```

`fem.solve(solve_fn)` lets you choose the solver, but every problem ships with a differentiable
default (no external dependency): the linear default is a sparse-direct factorisation
(`sparse_lu_solve`, JAX `spsolve`), with a Jacobi-preconditioned matrix-free BiCGStab as the iterative
alternative; the nonlinear default is a matrix-free Newton-Krylov, and the transient default
backward-Euler over those. All are implicit-diff, so `crux.solve` recovers parameters through them.

### Choosing the solver — the slot API (`jno.solve` / `jno.precond`)

Between "accept the default" and "write a full `solve_fn`" sits the **slot API**: the solver
factorises into four orthogonal slots, each a configured **callable** (never a string) from the
`jno.solve` / `jno.precond` namespaces — or your own with the same contract. Every `None` keeps
today's default; `solve_fn=` stays the total override (passing both is an error).

```python
u = fem.solve(
    x0        = u_guess,                 # warm start (previous solve, coarse solve, a surrogate…)
    nonlinear = jno.solve.newton(),      # linearization driver (nonlinear problems)
    linear    = jno.solve.gmres(),       # inner linear solve: lu / dense / cg / bicgstab / gmres
    precond   = jno.precond.jacobi(),    # v -> M⁻¹v spec, materialized against the assembled A
)
```

Everything shipped is **pure JAX** — `jit`/`vmap`-native, differentiable (Krylov on
`lax.custom_linear_solve`, Newton on `lax.custom_root`), receives the assembler's **BCOO** operator
directly (no densification), and composes with the periodic reduction and every parametric/inverse path.
Pick by structure:

| structure | solver | notes |
|---|---|---|
| SPD (Poisson, elasticity, mass) | `cg` | cheapest per iteration |
| non-symmetric (advection, SUPG) | `bicgstab` / `gmres` | `bicgstab` with `jacobi()` is the default |
| **iterative preconditioner** (inner Krylov, block/Schur) | `fgmres` | flexible right preconditioning — Saad, *SISSC* 14(2), 1993 |
| symmetric **indefinite** (Stokes/Biot saddle, biharmonic) | `minres` | monotone residual, `O(1)` memory — Paige & Saunders, *SINUM* 12(4), 1975 |
| SPD, batched/GPU-heavy | `chebyshev` | inner-product free (no reductions) — Golub & Varga 1961 |
| indefinite, single solve | `lu` | sparse-direct; **no vmap rule** — use a Krylov solver inside batched solves |
| cuSolver refuses it, or is slow | `lu(backend="host")` | factors on the HOST (SuperLU) and drives it from the device; same answer and same gradients (wrapped in `custom_linear_solve`, transpose via SuperLU `trans="T"`). Measured **faster** where cuSolver also works — Stokes 21,839 DOFs 0.27 s vs 1.67 s, H(curl) 17,072 complex DOFs 13.3 s vs 36.4 s — and it runs meshes cuSolver rejects (Stokes 26,908, H(curl) 26,154, both of which fail on GPU). Affordable because a direct solve factorises **once**: the operator crosses PCIe once, not per iteration. Read the win as *cuSolver's sparse LU is weak*, not *GPUs lose* — see the row below |
| **shift-invert eigs, or a constant-operator transient** | `lu(backend="cudss")` | NVIDIA cuDSS — the **fastest** direct backend wherever it runs, because it separates the symbolic *plan* from the numeric *factorization* and jNO caches on the **sparsity**, so the plan survives a change of values. Against `backend="host"` on an RTX 3070 (fp64 at 1/64 rate — the *unfavourable* card): Stokes saddle factorization **3.4 ms vs 79.9 ms**, lap3d 50³ **576 ms vs 64,856 ms**, and **64.7× per Newton step** at n=64,000. Also factors the Stokes saddle cuSolver calls *singular*, with smaller residuals. Needs the optional stack (`nvmath-python`, `cudss`, `cupy`); raises a clear `ImportError` otherwise. Fill-in still governs 3-D (69×→218× nnz growth at lap3d 20³–40³), so it moves the ceiling and makes **device memory** the binding constraint — it is not a substitute for a preconditioner  Two things happen automatically: a **block right-hand side** `(n, k)` is solved in one call (measured **2.7× at k=4 to 5.4× at k=16** over the same factorization solved column by column — this is what the shift-invert eigensolver's subspace iteration needs every sweep), and an exactly symmetric operator is factored as **LDLᵀ instead of general LU** (1.41× faster, **1.38× less peak device memory** on lap3d 40³). Symmetry is tested to within a few ulps rather than bitwise — an assembled FEM tangent for a symmetric form is symmetric only up to *assembly round-off* (a vector element block contracts components in a different order for `(a,i),(b,j)` than for `(b,j),(a,i)`, measured **0.25 ulps**), so a bitwise rule sent every vector and coupled problem down the general-LU branch. Within the gate the two triangles are **averaged**, a correction bounded by the asymmetry it removes and therefore no larger than the round-off already in the matrix; outside it nothing is touched. The margin is not a knife-edge: the weakest *genuine* asymmetry that can be constructed — an advection coefficient of 1e-12, meaningless beside the Laplacian — is already **191 ulps**. Measured end-to-end on a 3-D vector tangent, general → symmetric is **1.07–1.13×** at 3.4k–11.8k DOFs (growing with size) and **1.10–1.62×** on lap3d 20³–50³. The rest of the rule stands: symmetry is still tested and SPD is never inferred — a matrix symmetric only to ~1e-15 would otherwise be silently factored as `(A+Aᵀ)/2`, and a wrong SPD guess returns NaN. A singular operator **raises**: cuDSS signals it through neither an exception nor a NaN, returning a finite plausible vector instead |
| **a Newton loop** (or no GPU / a factorization too big for device memory) | `lu(backend="pardiso")` | Intel MKL PARDISO, multithreaded CPU. **The fastest factorization of the four**, and like cuDSS it splits symbolic analysis from numeric factorization, so a Newton step reuses the analysis. On lap3d 50³ (n=125,000) against single-threaded SuperLU's 65,212 ms: factorization **298 ms**, Newton re-factorization **296 ms — 220×**, where cuDSS reaches 115×. Its adjoint is cheaper too — `Aᵀx = b` comes from the *same* factorization rather than a second one. An exactly symmetric operator uses LDLᵀ (1.9× on lap3d 50³, 13× on a saddle), which needs the upper triangle with an **explicit diagonal** — without that a saddle's constraint rows come back empty and PARDISO rejects the matrix. Like cuDSS it returns finite garbage on a singular operator, so jNO checks the perturbed-pivot count and **raises**. `pip install jax-numerical-operators[pardiso]`, x86-64 |
| small systems / coarse blocks | `dense` | LAPACK, vmap-native |

> **Choosing between `cudss` and `pardiso`: pick by the phase your problem repeats.** A Newton loop re-*factorizes* every iteration, so PARDISO wins (220× vs 115× over SuperLU). A shift-invert eigensolve or a constant-operator transient re-*solves* against one factorization, and there cuDSS is ~11× faster per solve (3.5 ms vs 40 ms at lap3d 50³) and takes a whole block of right-hand sides at once. There is deliberately no `auto`: which wins depends on hardware jNO cannot inspect. Install both with `pip install jax-numerical-operators[fem]`.

**Preconditioner specs** (declarative — materialized against the assembled operator at solve time; a
preconditioner never changes the converged solution, only the speed, so specs need no gradient path):

* `jno.precond.jacobi()` — diagonal.
* `jno.precond.chebyshev(degree=…)` — fixed-degree Chebyshev **polynomial** preconditioner: matvecs and
  AXPYs only, the GPU-era substitute for Gauss-Seidel/ILU smoothing, and a fixed *linear* map so it
  legally preconditions `cg`/`minres`. Spectrum bounds come from `lmin`/`lmax` when you pass them, else
  **both** ends are measured by Lanczos (Lanczos 1950, §II — the extreme Ritz values), at the same cost
  as a power iteration on the top end alone. This matters because the polynomial is a contraction only
  *inside* the interval it is fitted to: the `lmin = lmax/30` guess, whenever the true ratio is smaller,
  leaves the lowest modes outside that interval where the polynomial amplifies them. Without the
  optional `matfree` package that guess is the fallback.
* `jno.precond.nystrom(rank=…)` — **randomized Nyström** low-rank preconditioner (Frangella, Tropp &
  Udell, *SIAM J. Matrix Anal. Appl.* 44(2), 2023, Alg. 2.1 and §3), the rung between `jacobi` and a
  multilevel method. Sketches `A` against a random `n × rank` matrix — exactly `rank` matvecs, no
  factorization, no triangular sweep — and deflates the captured top of the spectrum. That is what
  `jacobi` cannot do: a diagonal rescales, it cannot *separate* a few large outlying eigenvalues, and on
  such a spectrum Jacobi can be worse than no preconditioner at all (measured on a rank-15-outlier SPD
  system: 124 CG iterations for `jacobi`, 98 unpreconditioned, **46** for `nystrom(rank=20)`). **SPD
  only** — the sketch takes a Cholesky, so an indefinite operator gives NaN rather than a quiet wrong
  answer.
* `jno.precond.inner(solver)` — any `jno.solve` solver as the `M⁻¹` application (an inexact block/system
  solve). Iterative inner ⇒ flexible outer (`fgmres`).
* `jno.precond.form([...terms], inner=…)` — **preconditioners as weak forms**: assemble an auxiliary
  operator from ordinary traced terms and invert it as `M⁻¹` (weighted mass matrices, shifted-Laplacian
  Helmholtz twins, low-order proxies — written in the PDE's language).
* `jno.precond.block_diag((field, spec), …)` / `jno.precond.triangular((field, spec), …)` — per-field
  composition over `fem.blocks`. `triangular` is the standard saddle-point shape: last block solved
  first, substituted back through the assembled off-diagonal matvecs.
* `jno.precond.amg(cycles=…)` — **hybrid algebraic multigrid**: setup once on the host via the *optional*
  `pyamg` (Vaněk, Mandel & Brezina, *Computing* 56, 1996; PyAMG — Bell et al., *JOSS* 8(87), 2023),
  applied as a pure-JAX V-cycle with Chebyshev smoothing (Adams et al., *JCP* 188, 2003). The apply is
  `jit`/`vmap`-native and exactly linear, so it preconditions `cg`/`minres` too. Mesh-independent
  convergence ⇒ *the* choice for large elliptic blocks. Inside traced/parametric solves, pre-build
  eagerly: `spec = jno.precond.amg(); spec.build(fem.A)`. Without pyamg, `amg` raises a clear install hint.

The flagship pattern — Taylor–Hood **Stokes** by FGMRES with an inexact velocity block solve and the
viscosity-weighted pressure-mass Schur approximation (Elman, Silvester & Wathen, 2014, §9.2):

```python
sol = fem.solve(
    linear  = jno.solve.fgmres(tol=1e-10, restart=40),
    precond = jno.precond.triangular(
        (u, jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=60))),   # Â⁻¹: inexact CG
        (p, jno.precond.form([(1.0/mu) * pp * qq], inner=jno.solve.dense())),  # Ŝ ≈ μ⁻¹ M_p
    ),
)
```

**Picard / lagged coefficients — `jno.lag`.** When a solution-dependent coefficient's Newton tangent
destroys the linearized system's structure (the classic case: a shear-thinning viscosity `μ_eff(u)` in
non-Newtonian Stokes flow, whose full-Newton velocity block defeats AMG/block preconditioners), freeze
it with `jno.lag(...)` and drive with `jno.solve.picard()`:

```python
fem = jno.fem([2 * jno.lag(mu_eff) * inner(eps(ui), eps(vi)) - pi * div(vi), ...])
sol = fem.solve(nonlinear=jno.solve.picard(damping=0.7), linear=jno.solve.fgmres(),
                precond=jno.precond.triangular(...))
```

`lag` is `stop_gradient` on the traced expression, so the residual's linearization *is* the Picard
operator — each outer step re-solves the lagged system (linear convergence, but every inner system keeps
its symmetry/definiteness); the converged solution is identical to full Newton's. Without any `lag`
marker, `picard(damping=…)` is exactly damped Newton (`jno.solve.newton(damping=…)`). Caveat for inverse
problems: implicit differentiation then also uses the lagged Jacobian — drop `lag` when exact parameter
gradients matter more than per-step solvability.

**Alternate minimization — `jno.solve.staggered([u, d])`.** Some coupled energies are **non-convex in
the fields jointly but convex in each separately**. A monolithic Newton then has no descent guarantee
and simply diverges; solving one field at a time turns the problem into a sequence of convex solves:

```python
sol = fem.solve(nonlinear=jno.solve.staggered([u, dm]))   # sweep u, then dm, until both converge
```

Variational phase-field fracture is the canonical case — `(1-d)²|∇u|²` is quartic in the pair, while
`u` alone is a linear elasticity problem and `d` alone a linear elliptic one. Measured on the coupled
damage form in `tests/test_fem_staggered.py`: `newton()` leaves with a residual around **1e+25** (it
raises), and the staggered sweep converges to a genuine root of the *coupled* system. Fixed-stress Biot
poroelasticity and thermo-mechanical staggering have the same shape.

Algorithm: Bourdin, Francfort & Marigo, *Numerical experiments in revisited brittle fracture*, JMPS
**48** (2000), §3 — as the staggered operator split with a history field, Miehe, Welschinger & Hofacker,
IJNME **83** (2010).

**`direct=True` factorizes each field's diagonal block** rather than solving it matrix-free, and pairs
with a `linear=` slot:

```python
fem.solve(nonlinear=jno.solve.staggered([u, dm], direct=True), linear=jno.solve.lu(backend="pardiso"))
```

It exists because the matrix-free sub-solve **cannot be preconditioned**. A `precond=` spec materializes
against an assembled operator, and a staggered sub-problem is a restriction *closure*
(`x -> R(u with block set to x)[block]`) with no matrix, so an ill-conditioned block is solved by
**unpreconditioned BiCGStab** — near-incompressible elasticity (ν → 0.5) being the usual victim. That
was also why a direct `linear=` slot used to be refused against `staggered` outright: there was nothing
to factorize.

The extraction avoids a data-dependent `nnz` (which would break the static shapes a traced Newton
needs): instead of slicing `J[b][:, b]` out, the *complement's* rows and columns are zeroed and given a
unit diagonal, so the block is solved as `[[J_bb, 0], [0, I]]` against `[-r_b, 0]`. The padding is pure
diagonal — no fill-in — so the factorization cost stays the block's. With `bounds`, the matrix handed to
the factorization is the **min-map's** semismooth Jacobian (identity rows on the active set), which
`jax.linearize` derives for free on the matrix-free path but an assembled tangent does not.

Measured on a 3-D Yeoh phase-field march (576 DOFs, 8 load steps, CPU): **42.80 s → 10.78 s, 4.0×**, to
the same answer. Two honest caveats: the **full** tangent is assembled to use one block of it (a
sparsity-caching backend — `pardiso`/`cudss` — then pays only the numeric re-factorization per sweep),
and on a well-conditioned problem the matrix-free default is cheaper. This is not a free upgrade and it
is not the default. The load-path march threads the tangent too, which it previously never did: before
this, a direct driver inside a `tau=` path silently fell back to the matrix-free inner solve.

**`line_search=` globalizes each sub-solve's Newton steps**, and it is now on by default:

| value | what it does |
|---|---|
| `"backtrack"` *(default)* | residual-norm Armijo, halving from `damping`. What this used to do under `line_search=True`. |
| `True` | **exact** line search — bisect for the root of the directional derivative `R(x+λd)·d`, i.e. the minimizer of the energy along the Newton direction |
| `False` | no line search; take `damping` |

The default moved from `False` to `"backtrack"` because no line search is a genuine footgun on
finite-strain forms: measured here, a 3-D Yeoh P2 march produced **NaN on load step 1** without one
(the undamped step inverts an element, `det F ≤ 0`, and `J^(-2/3)` is NaN) and solved cleanly with one.

`True` implements Heinzmann, Vicentini, Carrara et al., *Iterative convergence in phase-field brittle
fracture computations: exact line search is all you need*, Computational Mechanics (2026),
[arXiv:2511.23064](https://arxiv.org/abs/2511.23064), §3 Algorithm 2 — the same algorithm they
contributed to PETSc as `SNESLineSearchBisection`. Their Props. 1–2 and Remark 4 chain into a
convergence guarantee for the whole alternate-minimization scheme, provided each sub-problem is
strictly convex and coercive (jNO's `bounds` min-map is semismooth, so that proof does not cover it —
the same gap they note for reduced-space active sets).

**It is not the default, because we have not measured it beating backtracking.** On the problems
tested here — a 2-D SENT plate with and without a volumetric-deviatoric split, at several load levels —
both need the same 21–22 staggered sweeps, and the exact search costs ~15% more wall time from the
extra residual evaluations. Note the paper's own failure cases arise only where the *mechanical*
sub-problem is non-linear (their §: the residual "reduces to an affine form in the absence of an energy
decomposition"), at critical load steps reached along a path — a regime not reproduced here. Reach for
`True` when a sub-solve stalls; the theory is on its side even where our measurements are neutral.

**`over_relax=ω` accelerates the sweep itself.** Alternate minimization *is* a nonlinear block
Gauss–Seidel iteration, so over-relaxation accelerates it exactly as it does the linear one — go `ω`
times as far along each sub-step's own update direction, per block:

```python
fem.solve(nonlinear=jno.solve.staggered([u, dm], over_relax=1.4))
```

Algorithm: Farrell & Maurini, *Linear and nonlinear solvers for variational phase-field models of
brittle fracture*, IJNME **109** (2017) 648–667, **Algorithm 2 (ORAM), §2.1**. `ω = 1` is plain
alternate minimization. Kahan's classical bound gives `ω ∈ (0, 2)` as necessary for SOR to converge, and
anything outside raises.

**Whether it pays is problem-dependent, and there is no way to know in advance.** The paper is explicit —
they "rely on the naïve strategy of numerical experimentation on coarser problems" and defer automatic
selection to future work. Their own results split cleanly: Table I (a propagating crack, where AM
converges slowly) gains **58–73% fewer iterations**; Table II (where AM already converges fast) gets
**0% — over-relaxation hinders it**, going 37 → 111 → 185 → 326 → 747 iterations as ω runs 1.0 → 1.8.
On a 2-D small-strain phase-field problem that drives damage to 1, ω = 1.4 was **1.95× faster**
(3096 → 1587 ms, warm median of 3). So it defaults to 1, and a short ω sweep on a coarse version of the
problem is the only way to know — the paper's own two tables disagree with each other.

**ω cannot diverge.** The extrapolation is not taken on trust: the step retreats by bisection on
`[1, ω]` until the trial point has a **finite** residual and is **feasible**, and `ω = 1` is the
sub-solve's own converged answer, already evaluated and therefore admissible by construction. So the
worst case is that the sweep degrades to plain alternate minimization. This matters most on
finite-strain forms, where an unguarded step past a converged answer inverts an element (`det F ≤ 0`,
so `J^(-2/3)` is NaN) — measured on the 3-D Yeoh SENT march, which NaN'd on load step 1 at *every* ω
down to 1.1 before the guard and runs to completion with it.

The test is finiteness and feasibility, **not** descent. Over-relaxation is deliberately not a descent
method, so demanding a residual decrease would reject almost every ω > 1 and silently collapse the
feature back to ω = 1 while appearing to keep it. Feasibility is the paper's own rule (§2.1 backs ω off
by bisection on `[1, ω]` for the bound constraint); finiteness is the generalization.

Over-relaxation also acts on the **free** DOFs only. Farrell & Maurini's `ũ` lives in the constrained
space `C_ū`, where a prescribed DOF has `δ = 0`; jNO imposes essential conditions as residual rows, so
without the mask the sub-solve's exact hit on the prescribed value gets extrapolated past — measured on
one row with `g = 2`, ω = 1.7 gave 3.40 → 1.02 → 2.69, an oscillation decaying only as `|1−ω|ᵏ`, worst
on a *ramped* condition where `g` moves every load step.

Cost when ω ≠ 1: one extra full residual evaluation per block per sweep.

Under `bounds`, over-relaxation steps *past* the sub-solve's answer — which is feasible by construction,
while the extrapolation need not be — so the driver takes the box projector from the `bounds` wrapper and
clips. That deviates from the paper, which backs the scalar `ω` off to the largest feasible value;
clipping is componentwise, also feasible, and keeps more of the step.

**The trade is the convergence rate, and it is not small.** Alternate minimization converges *linearly*
where Newton is quadratic, so it can need hundreds of sweeps near a propagating crack — hence the
`max_sweeps=200` default. It buys robustness, not speed: where Newton converges, Newton is the better
choice (Farrell & Maurini, CMAME **312**, 2017, compare the two directly). Sweeping is Gauss-Seidel, so
the **order matters**, and every field block must be listed — an unlisted field's equations would never
be solved, which is rejected rather than skipped. Each field is solved alone; sweeping a *group* of
fields together (a Stokes velocity/pressure pair inside one sweep) is not wired.

Differentiable in the ordinary way: at convergence the full residual is zero, so the sweep is just a way
of *finding* that root and `lax.custom_root` supplies the gradient from the full Jacobian — the
alternating structure is absent from the derivative by construction.

**Sparse-direct Newton — `jno.solve.newton(direct=True)`.** The default Newton solves each linear step
**matrix-free** (BiCGStab on the JVP), which stalls on an **indefinite / ill-conditioned** tangent with no
good preconditioner — a Taylor–Hood velocity/pressure saddle, a stiff Carman–Kozeny phase-change drag in a
melt pool. `direct=True` instead **assembles and factorizes** the tangent each step with a sparse LU (the
transient stepper factorizes the backward-Euler step tangent `M/dt + ∂R/∂u`; the steady path `∂R/∂u`).
It composes wherever the assembler provides that tangent — `fem.solve(nonlinear=jno.solve.newton(direct=True))`
on a native nonlinear problem, **steady or the transient march** — and stays differentiable: implicit
differentiation uses a *direct, transposable* tangent solve on the tangent assembled at the root (the adjoint
solves `Jᵀ` directly too, not with a stalling Krylov). `damping` / `line_search` apply unchanged. It needs the
assembled tangent, so it does **not** apply to the matrix-free-only paths (a coupled-residual wrapper,
complex) — those fail loud.

The tangent it factorizes is always the **element-scattered sparse** one, in 1-D as in 2-D/3-D. 1-D used
to build its nonlinear tangent with a global `jax.jacfwd` instead, on the assumption that only the
matrix-free default would ever ask for it; `direct=True` does ask, from inside the Newton loop, where a
dense tangent cannot be sparsified at all (`BCOO.fromdense` needs a concrete `nse`). Besides working, the
scattered tangent is `O(nnz)` rather than `O(N²)` — which is what 1-D node counts want.

**A direct `linear=` slot selects it.** `lu`, `dense` and `amg` all need an assembled matrix, so pairing one
with the *matrix-free* Newton has nothing to factorize. `fem.solve(linear=jno.solve.lu(backend="host"))` on a
nonlinear or transient problem therefore routes to the direct Newton, and that slot is the solver that runs on
the assembled tangent (and on `Jᵀ` in the adjoint); `precond=` materializes against the same assembled
operator. Which factorization you pick is not cosmetic here: on a 26-step Rayleigh–Bénard march (three fields,
saddle, nonlinear) the default matrix-free Jacobi-BiCGStab takes 20.1 s, `linear=jno.solve.lu()` 7.6 s and
`linear=jno.solve.lu(backend="host")` **3.1 s**, all to the same 2.8e-07 per-step Newton residual. An *explicit*
matrix-free `nonlinear=` alongside a direct `linear=` is contradictory and raises rather than picking one
silently.

> Limits, since they are not obvious. This routing only fires when the *linear* slot is direct — a
> `precond=` that needs an assembled matrix (`jacobi`, an unbuilt `amg`) raises where it is
> materialized. `picard()` has no assembled-tangent form: its linearization is the lagged
> *matrix-free* JVP, so `nonlinear=picard(), linear=lu()` raises — moving to `newton(direct=True)`
> there changes the algorithm, not just the solver. And the direct Newton needs the assembler to
> supply a tangent, so the matrix-free-only routes (a coupled-residual wrapper) fail loud either way.

**User extension** is duck-typed — a linear solver is any `fn(A, b, *, M=None, x0=None) -> x` with `A` a
`jno.solve.LinearOperator` (`.mv`, `.T`, `.diag()`, `.bcoo`, `.dense()`); a preconditioner is any
`ctx -> (v -> M⁻¹v)`:

```python
def my_precond(ctx):                      # ctx.A, ctx.diag(), ctx.fem
    inv = 1.0 / ctx.diag()
    return lambda v: inv * v
u = fem.solve(linear=jno.solve.cg(), precond=my_precond)
```

Calling a solver **directly** (outside `fem.solve`) takes the preconditioner as `M=`, which is the
*application* `v -> M⁻¹v`. A `jno.precond.*` spec is accepted there too and materialized against `A` on
the way in — `jno.solve.cg()(A, b, M=jno.precond.jacobi())`. A **bare callable** is always the applier,
never a `ctx -> applier` factory: as a `precond=` slot it would be the factory, nothing about a callable
tells the two apart, and guessing wrong would apply a preconditioner you did not ask for. Specs that
need eager preparation (`form`, which assembles an auxiliary operator) require `fem.solve(precond=...)`
— a direct call has no owning FEM.

If your callable is pure JAX it inherits `jit`/`vmap`/AD automatically. On the matrix-free **nonlinear**
path the `precond` spec is materialized *per Newton/Picard linearization* against the JVP operator — so
`form`, `inner(...)`, `chebyshev`, a pre-built `amg`, and their `block_diag`/`triangular` compositions
all work; only specs that need the assembled matrix (`jacobi`, an unbuilt `amg`) raise.

**Transient problems.** The slots configure the *per-step* solves of the default theta-method integrator:
`linear`/`precond` see the step operator `M + θ·dt·A` — when it is time-independent the step matrix is
formed **once** and the preconditioner materialized **once before the time loop** — and `nonlinear` drives
each implicit step of a nonlinear block. Second-order-in-time (`u_tt`) flows through the same augmented
block. Each step warm-starts from the previous state (so `x0=` is rejected).

> **`lu(backend="host")` factorizes a constant step operator once, not once per step.** The step matrix is
> formed once (above), and the host factorization is cached on the operator's *content*, so a march
> that solves against the same matrix every step pays one factorization for the whole trajectory — and
> the transpose solve reuses it, so the adjoint pass adds none. On a 51-step heat march that is worth
> 1.55× of whole-solve wall clock at 8,355 DOFs and **2.9× at 23,934**. The gain grows with mesh size,
> because factorization cost grows faster than the per-step solve; at 513 DOFs it is 1.05×, where the
> factorization is sub-millisecond.
>
> What it does **not** help: a *nonlinear* march, or any Newton loop. There the tangent's values
> change every iteration, so every call legitimately misses and pays a content hash (~1–2% of a
> factorization) for nothing. Reusing only the *symbolic* analysis — which genuinely is constant when
> the sparsity pattern is fixed — is not something SuperLU exposes through scipy. It is also
> host-path-only: `lu()` on GPU goes through cuSolver's `spsolve`, a single fused call with no
> factorization object to keep.

> **Adjoint memory: the march is gradient-checkpointed.** Reverse-mode through the default integrator
> rematerializes each step in the backward pass instead of storing every step's solver internals for
> the whole trajectory. Measured at 8,355 DOFs × 399 steps: peak memory **968 → 112 MB (8.6×)** for a
> gradient cost of **+60%** (2.98 → 4.76 s), gradient identical to 10 digits. The trade is
> deliberate: a differentiable march OOMs long before it is time-walled on consumer cards. A pure
> forward solve is unaffected — checkpointing is the identity outside differentiation.

**Complex problems** are assembled as one real `2n` system over the stacked `[Re; Im]` state — the
real-equivalent block `[[A_r, -A_i], [A_i, A_r]]` — at assembly rather than at solve time. A complex
transient is therefore an ordinary transient block (the slots configure its per-step solve as above, and
`theta` / `adaptive` / `exponential` all apply), and a complex steady problem is an ordinary linear
system: the `linear` / `precond` slots and `x0=` work on it, with a **complex** `x0` mapped into the
block layout for you. Its default solver stays **sparse-direct**, not the matrix-free BiCGStab real
elliptic systems get — the real-equivalent block is indefinite for Helmholtz/PML, where Jacobi-BiCGStab
does not converge.

One exception keeps a dedicated path: a **complex-native** preconditioner (`ams`) solves `A_r + i·A_i`
directly rather than the block, so the Re/Im legs are retained for it.

A **Bloch** (quasi-periodic) tie fuses like everything else. Its complex prolongation `P` cannot
reduce the Re/Im legs independently, but on the fused `[Re; Im]` state the same tie is the *real*
prolongation `B(P) = [[P_r, -P_i], [P_i, P_r]]`, and the ordinary real congruence `B(P)ᵀ A B(P)`
equals the Hermitian reduction `P^H A_c P` the Bloch space requires. Consequences: `solve_fn=`, the
`linear`/`precond` slots and `x0=` all apply to a Bloch problem (each used to be silently discarded by
a dedicated block routine); a Bloch tie composes with a **complex transient** (the quasi-periodic
plane-wave march, previously a dtype crash); and a **real** weak form with a Bloch tie is promoted to
the complex path automatically — the phase makes the field complex anyway, and the real path's
bilinear `Pᵀ A P` is not a Galerkin projection for a complex `P` (measured 8.1 rel-L2 off the
Hermitian answer on a manufactured mode, with the tie itself satisfied exactly).

A **coupled (multi-field) complex steady system** — coupled Helmholtz-type equations — takes the same
Re/Im split through the coupled assembler: one fused real `2n` block over `[Re_all; Im_all]`, with
`fem.offsets` still listing the per-field blocks of the recombined complex solution. Scope: steady and
linear (a complex *nonlinear* coupled form and a complex coupled *transient* refuse, as everywhere).

**Essential values on a complex form must be real.** The two legs share one Dirichlet row set, which
imposes `Re u = g` with `Im u = 0` — right for a real `g`, and the usual case (the complexity lives in
the operator and the source). A *complex* `g` is not expressible there: pinning `Im u = g_i` would need
the imaginary leg's rows zeroed rather than set to identity, and the symmetric elimination's
known-column lift is cross-leg (the real equation needs `A_r[:,j] g_r - A_i[:,j] g_i`, which no per-leg
elimination produces). It raises a clear error. Carry the complex part in the operator or the source.

`adapt=` composes with a complex **transient** too: the stacked `[Re; Im]` halves transfer across
each remesh as a doubled field layout, the **modulus** `|u|` drives the remesh metric (refining on
`Re` alone would miss a rotating phase), and the saved frames come back complex.

Not yet supported (clear errors): a Bloch tie on a **real** transient march (the phase forces a
complex field — make the problem complex, or use a plain tie) or on a **nonlinear** form (complex
nonlinear forms are not wired).

### Multiple devices — `fem.solve(shard=...)`

Sharding is **automatic**: on a machine with more than one visible device, the assembled operator's
nonzero axis is partitioned across all of them, each device scatter-adds its slice, and one
`all-reduce` combines the partials. The operator shards; the vectors stay replicated.

```python
u = fem.solve()                 # automatic: every visible device
u = fem.solve(shard=False)      # opt out — single device (1 means the same)
u = fem.solve(shard=2)          # pin a device count; over-requesting fails loud
u = fem.solve(shard=jax.devices()[:4])   # pin exactly these
```

It is on by default because the change is **answer-preserving** — same operator, same solvers, only
the reduction order moves (~1e-14) — and because the realistic alternative on a multi-GPU box is not
a tuned single-device run, it is idle silicon. On a single-device host it resolves to the untouched
single-device path, so the default carries no risk there.

The reason no solver needed changing is that the operator is ~100× the vector, so replicating the
vectors costs nothing and removes the entire distributed-FEM apparatus: no mesh partitioning, no halo
exchange, no ghost DOFs, no DOF renumbering. Every Krylov step is either a matvec (sharded,
`all-reduce` inside) or a vector operation on replicated data (identical on every device, no
communication).

**What shards:** the default steady-linear solve, and the slot-composed solve
(`linear=` any Krylov solver, with `precond=None` or `jno.precond.jacobi()`).

**Parametric / differentiate-through solves shard too, but only on an explicit `shard=`:**

```python
u = fem.solve(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi(), shard=4)
```

`device_put` cannot place a tracer, so this route uses `lax.with_sharding_constraint` from inside the
trace. Gradients flow through it unchanged. It is opt-in rather than automatic for a safety reason,
not out of caution: inside a trace jNO is a guest in someone else's computation, and a sharding
constraint has to agree with the device commitments of every other value in that `jit`. Under `crux`
it does not — the optimiser's parameters arrive committed to a single device while the constraint
spans all of them, and JAX rejects the mix. That conflict cannot be detected in advance
(`get_abstract_mesh()` is empty there) and cannot be caught locally, because it surfaces when the
*outer* `jit` compiles, long after the solve was traced. There is no fallback to write, so automatic
placement leaves traced operators alone; an explicit `shard=` is a request you can diagnose.

> Two traps on this route were invisible in the answers and only showed up in the compiled HLO, which
> is why the tests assert on collectives rather than on values. Padding the triplet axis to a multiple
> of the device count makes XLA `all-gather` the **whole operator** onto every device to feed the
> concatenate; constraining an *uneven* axis makes it gather the index array instead (the same 8 bytes
> per triplet as the data). Both produced correct answers and correct gradients with the memory saving
> entirely gone. jNO therefore shards the divisible prefix and leaves the sub-device-count tail
> replicated — at most 3 triplets on a 4-device run.

**What does not** — each falls back silently to the single-device path rather than raising:

| | why |
|---|---|
| sparse-direct branches (periodic, 1-D, fused-complex) | route to `spsolve` — single-device, no batching rule |
| `linear=jno.solve.lu()` | `spsolve` is single-device with no batching rule — a genuine wall, not a wiring gap. Distributing it means a distributed sparse-direct solver (SuperLU_DIST class), which is not a placement change |
| `linear=jno.solve.dense()` | not wired. Dense LU with partial pivoting shards poorly, but the `N²` matrix itself would split — the win here would be capacity, not speed |
| `precond=amg()` / `ams()` | the hierarchy is built host-side through scipy/pyamg; distributing the V-cycle is a distributed-AMG project, not a placement change |
| `precond=chebyshev()` / `form()` | not wired yet, and **not** a hard limit — Chebyshev is matvec-only by construction (spectral bounds by power iteration), so it composes with the sharded matvec directly; `form`'s auxiliary operator is just another assembled BCOO |
| other `precond=` | the applier closes over the assembled operator, so a full copy would be replicated anyway. Jacobi is the exception: it needs only the diagonal, computed from the *sharded* triplets |
| parametric / differentiate-through solves | **opt-in only** — needs an explicit `shard=`, see below |
| transient | not wired yet. A sharding constraint inside the `lax.scan` body already produces the right collectives with the operator still closed over; threading it in as a jit argument additionally makes the per-device footprint provable (measured: exactly `nnz/N` per device) |

No speedup figure is quoted here because none has been measured — the development machine has one
GPU. What *is* verified, on simulated devices, is correctness, the even split, that XLA emits
`all-reduce` and **zero** `all-gather` (no device ever reconstitutes the matrix), and that the
fallbacks decline rather than silently gathering.

### Reduced-order solves — `fem.solve(basis=U)`

A periodic prolongation and a **reduced-order basis** are the same object: a tall `(n_dofs, k)` map `U`
defining `UᵀAU`, `Uᵀb`, and the lift `u = U x`. So `basis=` reuses the reduction the periodic ties
already drive, and the answer comes back in the **full** space — nothing downstream changes.

Solve the family a few times, keep the recurring shapes, then every later solve costs `k` unknowns:

```python
snapshots = jnp.stack([build(p).solve() for p in sweep])    # (n_snapshots, n_dofs)
U, s, Vt  = jno.solve.svd(snapshots.T, k=10)                # columns of U are the spatial modes
u = build(p_new).solve(basis=U)                             # 10 unknowns; full field returned
```

Mind the orientation: for a `(n_snapshots, n_dofs)` snapshot matrix the spatial modes are `Vt.T`, and
for its transpose they are `U`. Passing the wrong one is refused by shape, with the fix in the message.

This is the **only** path here that returns an approximation, so it is measured rather than trusted: the
relative residual of the full system at the lifted solution is computed each call (one matvec), kept on
`fem.basis_residual`, and a basis that does not span the solution **raises** instead of returning a
plausible wrong field. Deliberately coarse work (a rank sweep) can raise `fem.BASIS_RESIDUAL_LIMIT`.

The basis is per-call, must be orthonormal (a non-orthonormal one would need `(UᵀU)⁻¹` when restricting
a state, and would be silently wrong), and composes with `linear=` / `precond=`, which see the reduced
operator. `∂u/∂U` flows under `jax.grad`, so the subspace itself can be **learned** — note an orthonormal
basis lives on the Stiefel manifold, so put the orthonormalisation inside the differentiated function
(`net -> QR -> basis`) rather than projecting the step afterwards.

**Transient too** — and that is what a ROM is really for, since the cost avoided is a whole time
integration rather than one solve. The block is reduced once at solve time (`PᵀMP`, `PᵀAP`, restricted
`state0`) and the marcher steps in the reduced space, returning the trajectory at full width:

```python
snaps = np.concatenate([np.asarray(build(p).solve().fn()) for p in sweep])   # (n_sweep*n_t, n_dofs)
U = np.linalg.svd(snaps.T, full_matrices=False)[0][:, :8]
traj = build(p_new).solve(basis=U).fn()                                     # 8 unknowns per step
```

A transient solve is certified differently: the steady residual has no analogue, so what is measured is
the **projection error of the initial state**, `‖u0 − U Uᵀ u0‖/‖u0‖`. If the span cannot represent where
the trajectory starts, the march is wrong from step 0. It is a floor, not a bound — it says nothing
about whether the span keeps up *later* (measured on a nonlinear case it came in below the true
trajectory error), and the docstring is explicit about that.

Scope, each refused with its own reason: **second-order-in-time** (`u_tt` marches the augmented `[u; v]`
state, so a field basis needs `blkdiag(U, U)` and the row convention is unsettled), **complex** (solves
through an internal real-equivalent 2n layout), a **periodic tie** (composing two prolongations has no
decided convention yet), and a `jno.np.parameter` basis (a trace node, not an array — `jax.grad` over a
concrete basis is the supported differentiable path). A reduced **nonlinear** solve works, but is a
memory win, not a speed one: the full-order residual is still evaluated per Newton step (no
hyper-reduction).

### Field parameters `k(x)` + regularization

`jno.np.parameter(phi)` is a **nodal field** on the trial space — a trainable value per node. Field
inversion is ill-posed, so add a smoothness/structure prior with `k.regularize(...)` (`"h1seminorm"`,
`"l2"`/`"tikhonov"`, `"tv"`, `"nonneg"`, `"bounded"`):

```python
k = jno.np.parameter(phi, name="k")                       # P1 field, one DOF per node
crux = jno.core([(fem.solve() - u_obs).mse, 1e-3 * k.regularize("h1seminorm").mean], domain=obs)
```

### Neural coefficients — `jno.nn(net)` inside the weak form

A network called inside a weak form is a trainable **coefficient** on an assembled FE system —
mesh-independent (remeshing never touches the weights), smooth by architecture, and trained through the
same differentiable `fem.solve()` as any parameter:

```python
net = jno.nn(foundax.mlp(2, hidden_dims=16, num_layers=2,
                              activation=jax.nn.tanh, key=key))
net.dtype(jnp.float64)                                       # match the f64 assembly
net.optimizer(optax.adam(1e-2))

# k(x) = 1 + net(x, y): the offset keeps A(θ) nonsingular at the (near-zero) net init
fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs).solve(600)   # trains the weights
```

The kernel re-evaluates the network at the quadrature points during every re-assembly, so the
coefficient is *not* interpolated on the mesh — it composes with scalar/nodal parameters, per-region
masks, vector trials, and surface (Robin/Neumann) terms. `net.freeze()` makes it a **known** network
coefficient. This is the unsupervised coefficient-recovery setting of NN-EUCLID (Flaschel, Kumar &
De Lorenzis, *J. Mech. Phys. Solids* 165, 2022) and Tartakovsky et al. (*Water Resour. Res.* 56, 2020).

**Learned constitutive laws — `net(u)`, `net(∇u)`.** A network may also take the *solution* (or its
derivatives) as input — then it is a material law, not a spatial map, and the form becomes nonlinear in
`u` (routed to the matrix-free Newton path automatically). Observe `u`, learn the hidden law
unsupervised through the residual:

```python
net = jno.nn(foundax.mlp(1, hidden_dims=16, num_layers=2,
                              activation=jax.nn.tanh, key=key)).dtype(jnp.float64)
# hidden truth k(u) = 1 + 0.5 u²; learn it from a single observed field
fem = jno.fem([(1.0 + net(ui)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs).solve(600)
```

`net(ui.x, ui.y)` (a `k(∇u)` law) and mixed inputs (`net(xi, yi, ui)`) work the same way; a net whose
arguments carry the unknown makes the form nonlinear. **Transient forms** recover a diffusivity or
constitutive law from a `u(t)` trajectory; a coordinate `net(x)` on the **mass** (`u_t`) term is
supported (an unknown density `ρ(x)·u_t`), but a *solution-dependent* `net(u)` on the mass is rejected (a
nonlinear mass the semidiscrete form cannot express). Net coefficients also compose with **complex**
steady forms and **coupled (multi-field)** forms, and with the scalar C¹ families
(`"Argyris"`/`"Morley"`/`"Hermite"`) and a *scalar coordinate* `net(x)` on the vector edge families
(`"RT"`/`"N1E"`); the non-nodal path assembles a *dense* operator, so wants an explicit dense `solve_fn`.

**Unknown boundary / initial conditions.** A network as an *essential value* is a trainable *profile*
(it enters the lift, not the operator): a *Dirichlet* value `u(∂Ω) - net(xb, yb)` (an unknown boundary
profile) or an *initial condition* `u(initial) - net(xi, yi)` (an unknown starting state, recovered from
a trajectory). Both supported for a **bare** `net(x)`, native Lagrange single-field — the Dirichlet on
steady / nonlinear / linear-transient / nonlinear-transient forms, the IC on a linear-transient form; a
compound value, or a net IC on a nonlinear transient, fails loud.

*Current scope:* steady/transient/steady-complex on the native 2D/3D Lagrange assembler (single or
coupled multi-field), steady scalar C¹ non-nodal, a bare `net(x)` steady/nonlinear/linear-transient/
nonlinear-transient Dirichlet value, and a bare `net(x)` linear-transient initial condition. Not yet
(each fails loud): a compound net essential value, a net IC on a *nonlinear* transient, a net Dirichlet
with a state-dependent mass, `net(u)` on the mass term, k(u) in complex forms, the complex transient, a
net Dirichlet combined with a time-varying `g(x,t)` Dirichlet, `net(u)` on the vector edge families, and
1D domains.

### Transient inverse

For a transient form, `fem.solve()` returns the **trajectory** `u(save_ts)` (default: backward
Euler over the assembled `dt`, sampled at the domain time grid), differentiable in the
parameters — so a rate constant is recovered from a time series:

```python
fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(*ci) - u0])
crux = jno.core([(fem.solve() - u_traj).mse], domain=obs).solve(200)   # recovers alpha
```

`fem.solve(my_integrator, save_ts=...)` swaps the integrator: `my_integrator(block, args, save_ts) ->
trajectory`. Build your own (e.g. diffrax) from the block's `block.M` / `block.A` / `block.state0` — form
`u_dot = M⁻¹(c − A u)`; the implicit backward-Euler default is preferred for Dirichlet problems.

---

## Vector, coupled, and higher-order problems

* **Vector / elasticity** — `u, phi = d.fem_symbols(value_shape=(2,))`; write the elasticity bilinear
  form `λ (∇·u)(∇·φ) + 2μ ε(u):ε(φ)` with `jno.np.symgrad` and `jno.np.inner(..., n_contract=2)` —
  **or component-wise**: `u[i]` and `u[i].x` / `u[i].d(var)` are first-class on a vector Lagrange field,
  and the two spellings mix freely in one term (the shape conventions are shared). A boundary traction on
  one component is `-t * phi_b[i]`.
* **Finite-strain (hyperelastic) mechanics** — the component spelling is what makes it expressible:
  build `F = I + ∇u` from `u[i].d(x_j)`, then `det F`, `F⁻ᵀ` and `log` are ordinary term algebra, and a
  form nonlinear in `∇u` routes to the matrix-free Newton automatically (use
  `nonlinear=jno.solve.newton(line_search=True)` for large steps). Compressible Neo-Hookean
  `P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ` is verified in `tests/test_fem_vector_components.py`: it matches the
  coupled-scalar spelling to machine precision and linear elasticity in the small-load limit. Ramp a hard
  load with a warm-started **`sequence` axis**: `space.sequence("load", ramp, keep="last")` then
  `crux.sweep(space)` — measured on the cantilever: cold *default* Newton fails at `load = 0.1` while
  the four-step warm-started ramp reaches it.
* **Coupled / mixed (Stokes)** — call `fem_symbols(...)` once per field and add one momentum and
  one continuity term; an inf-sup-stable Taylor–Hood pair is `order=2` velocity + `order=1`
  pressure. Pure-Dirichlet velocity leaves the pressure defined only up to a constant; gauge-fix
  that null space by adding `p.pin()` to the constraint list (`p.pin(value)` sets the gauge).
* **1D and 3D** — a 1D interval or a 3D `cube`/extruded `gmsh` volume use the identical API with
  one fewer / one more coordinate (`ui.z`, `u(xb, yb, zb) - g`).
* **Higher-order Lagrange** — `order=k` gives degree-`k` elements (P2, P3, P4, … on triangles and tets);
  read the solution at `fem.points`. The geometry stays affine-P1 (straight-sided), so on a *curved*
  boundary the geometric error caps the observed order regardless of `k` — measure high-order convergence
  on straight-sided/polygonal domains.

---

## Elasto-plasticity — a trace formula, not a module

Plasticity is not a module in jNO; it is a **formula** in the term list (the FEM contract). The J2 (von
Mises) radial return contracts against the test strain to a *scalar* per Gauss point — the same trick the
elastic form uses (`lam*trace*trace + 2*mu*inner`, never an identity), via `dev(A):B = A:B - tr(A)tr(B)/3`
and `||dev(A)||^2 = A:A - tr(A)^2/3`. So the whole return map is six lines of `jno.np`, behind your aliases:

```python
sym, grad, trace, inner, sqrt, maximum = jno.np.sym, jno.np.grad, jno.np.trace, jno.np.inner, jno.np.sqrt, jno.np.maximum
lam, mu = lame(E, nu); K = lam + 2*mu/3; rt = 1.5**0.5
eps = lambda w: sym(grad(w, [x, y, z]))
eu, ev = eps(u), eps(phi)
tru, trv = trace(eu), trace(ev)
ddev = sqrt(maximum(inner(eu, eu, 2) - tru*tru/3, 0) + 1e-30)   # ||dev eps(u)||, safe von-Mises norm at 0
dg   = maximum(rt*2*mu*ddev - sy, 0) / (3*mu + H)               # plastic multiplier
dev_ev = inner(eu, ev, 2) - tru*trv/3                          # dev eps(u) : eps(phi)
mech = K*tru*trv + 2*mu*dev_ev - 2*mu*rt*dg*dev_ev/ddev        # = the integrand of  sigma(eps(u)) : eps(phi)
sol  = jno.fem([mech, u(*bc) - 0.0]).solve(nonlinear=jno.solve.newton())
```

`jno.fem` sees the nonlinear form and routes to Newton; the element Jacobian is the consistent
elastoplastic tangent for free (AD of the formula). The solve is differentiable — thread `sy` (or `H`,
`E`) as a `jno.np.parameter` to recover it from an observed deformation (material-identification inverse
problem). This is Hencky deformation theory (virgin every solve): exact for monotonic proportional loading.

**Flow theory** (path-dependent; unloading leaves a permanent set) is the *identical* formula reading the
previous step's per-quadrature-point state with the step-history index `.i(k)`: `ee = eps(u) - ep.i(-1)`
and `sy -> sy + H*al.i(-1)`, with `ep, al` declared like any field via `fem_symbols`. How each state
*advances* is a **named update term** in the same list — `state.evolves(<formula>)`, an update, not an
equation (and not an operator: `==` is reserved for identity, `<` for comparison). The load is written as
a function of the pseudo-time coordinate `tau`, the domain carries a `tau=` load grid, and `fem.solve()`
**marches** the path with **nothing passed** — triggered by `.i(k)` exactly as `u.t` triggers transient:

```python
d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.1).domain(tau=(0.0, 1.0, 40))   # pseudo-time load path
x, y, z, tau = d.variable("interior", split=True)            # τ is a coordinate, like t
dev = lambda A: A - trace(A) / 3 * I3                         # I3 = jno.np.identity(3)
nrm = lambda A: sqrt(maximum(inner(A, A, 2), 0) + 1e-30)      # safe Frobenius norm
ee  = eps(u) - ep.i(-1); D = dev(ee); dd = nrm(D)             # elastic predictor about the previous state
dg  = maximum(rt * 2 * mu * dd - (sy + H * al.i(-1)), 0) / (3 * mu + H)
n   = D / dd
sig = K * trace(ee) * I3 + 2 * mu * D - 2 * mu * rt * dg * n  # returned stress
P   = peak * (1 - jno.np.abs(2 * tau - 1))                    # load ramps 0 → peak → 0 with τ
traj = jno.fem([
    inner(sig, eps(phi), 2) - P * inner(zhat, phi, 1),       # equilibrium              (test phi)
    ep.evolves(ep.i(-1) + rt * dg * n),                      # plastic strain advances  (a named update)
    al.evolves(al.i(-1) + dg),                               # hardening advances
    u(*bc) - 0.0,                                            # clamp
]).solve()                                                   # (n_steps, n_dofs) load-path trajectory
```

`.i(-k)` **reads** history, `.evolves` **writes** it. The build infers the keep-depth from the
most-negative index and threads a zeroed per-quadrature-point buffer through the march's `lax.scan` carry
(one compiled residual, reused every step; frozen-constant in the tangent → the consistent return-map
tangent). The whole march rides `custom_root`, so it stays differentiable end-to-end: thread `sy` as a
`jno.np.parameter` and `∂(unloaded state)/∂sy` flows through the entire load path (a material-
identification inverse). A **primary-unknown** history (`u.i(-1)`/`u.i(-2)`, e.g. a BDF2 time scheme) is
auto-buffered from the solved `u` — no `.evolves`; an **internal** state read at `.i(-1)` with no
`.evolves` on a `tau=` domain is a build error (never a silently frozen buffer = deformation theory).

**Scope:** small-strain, isotropic, linear-hardening; 3-D (2-D is plane strain). Kinematic / nonlinear
hardening and contact are separate (not built).

**A state can be shared by a coupled system.** The march is not single-field: history buffers are indexed
by *cell*, never by field, and the readout gathers every field's cell DOFs at once — so a state written
by one field and read by another is the same march. That is the phase-field / gradient-damage shape,
where an irreversible history `H = max_τ ψ⁺(u)` couples a displacement field to a damage field:

```python
psi = 0.5 * inner(grad(u, X), grad(u, X), 1)                 # driving force, from the u field
deg = (1 - dm)**2 + eta                                      # degradation, from the dm field
jno.fem([
    deg * inner(grad(u, X), grad(phi, X), 1) - load(tau)*phi,        # equilibrium      (test phi)
    (gc/l)*dm*q + gc*l*inner(grad(dm, X), grad(q, X), 1)
        - 2*(1 - dm)*Hs.i(-1)*q,                                     # damage evolution (test q)
    Hs.evolves(maximum(Hs.i(-1), psi)),                              # irreversibility  (a named update)
    u(*bc) - 0.0,
]).solve()                                                           # nothing passed
```

The running `maximum` is what makes it irreversible: at zero load the damage is retained rather than
healing. Note the block order is *first appearance in the term walk* — here `dm` precedes `u`, because the
degradation factor is written first — so resolve a block with `fem.block_index(dm)`, never a hardcoded
index.

A form that is **linear in every unknown** but reads `.i(k)` marches too — the AT1 damage equation with a
fully lagged driving force is exactly that shape. Each load step is a different linear system whose
coefficients the buffers set, so it routes through the same residual operator as the nonlinear march
(Newton converges in one step on a linear residual). You pay roughly one extra linear solve per step
versus a pure linear assembly, in exchange for one march path rather than two.

The coupled march is differentiable in a material parameter, exactly as the single-field one is: thread a
`jno.np.parameter` into the form and `∂trajectory/∂θ` flows through the whole scan. The same holds for a
coupled *steady* **nonlinear** form. What still refuses a runtime parameter is a coupled form that is
**linear and carries no history**, because that one assembles as a matrix/rhs pair and the coupled linear
assembly has no parametric route; anything on the residual path re-evaluates at the runtime args and is
field-agnostic.

Not carried, each rejected with a clear error: a real `u.t` transient (drive time through `tau` instead),
a complex form, 1D, non-nodal (Argyris/Morley/edge) elements, VPINN, and periodic ties.

**A step that did not converge is refused, not carried forward.** The march runs its per-step Newton
inside a single `lax.scan`, and the driver's own convergence check needs a *concrete* residual — so
inside the scan it disables itself, exactly where the signal matters most. A load path compounds the
loss: a non-converged step becomes the next step's initial state *and* its history buffers, so one
silent failure contaminates everything after it, and the trajectory still comes back finite and
entirely plausible. Measured on a 3-D Yeoh phase-field march whose undamped Newton overshot into an
inverted element (`J = det F ≤ 0`, so `J**(-2/3)` is NaN, which is absorbing): with the grip *pinned*
to 0.4 the returned displacement read 0.70, with no error raised.

So the per-step residual is carried out of the scan and tested where it is concrete, against the
driver's **own** `rtol`/`atol` — the net can only catch what the driver would have caught eagerly, and
never second-guesses a solve configured loosely. Under `bounds` it scores the **min-map**, not the bare
residual: on an active constraint that residual is non-zero by construction (it *is* the multiplier),
and scoring against it would read a correct answer as a divergence. The check costs two residual
evaluations per step — measured at **2.4%** (30.30 s → 31.03 s) of an 8-step, 576-DOF Yeoh march. It is
a no-op under `jax.grad` of a runtime-parametric march, where the norms are themselves traced; there,
as everywhere else in jNO, the solver's iteration cap is all there is.

The error names the fixes in order of what usually works: globalize the per-step solve
(`jno.solve.newton(line_search=True)` / `staggered(line_search=True)`, or `damping<1`), take smaller
steps (a finer `domain(tau=(...))` grid, or the adaptive path below), or raise `max_steps`. In the Yeoh
case above, `line_search=True` alone recovers the exact answer — and note P1 solves the same form
undamped: a higher-order element's full Newton step produces larger gradients at its extra quadrature
points, so P2 is the more exposed one.

**Adaptive load stepping — `fem.solve(tau=jno.solve.adaptive(limit=...))`.** A uniform load grid is
wrong in both directions at once: it wastes steps while nothing happens and takes too-large ones through
the event. On a path-dependent march that second failure is not merely coarse — a step can converge
perfectly and skip the entire transition, leaving a valid sequence of equilibria with no resolved event
between them, which is a *different* answer, not a coarser one.

```python
sol = fem.solve(tau=jno.solve.adaptive(limit=0.05))            # bound every DOF's per-step change
sol = fem.solve(tau=jno.solve.adaptive(limit=[(dm, 0.05)]))    # per field — the usual case
```

The criterion is deliberately not the transient's. A rate-independent load path has **no local
truncation error to estimate** — each step is an equilibrium, not an approximation to a trajectory — so
the `rtol`/`atol` step-doubling estimate that sizes `time=` measures nothing here. `limit` bounds how
much the solution may change in one step; a step is rejected (and cut by `shrink`) when the solve fails
to converge *or* the change exceeds `limit`, and a comfortable step grows by `grow`.

Mechanism: **pilot → freeze → replay**. March eagerly with rejection to discover the schedule, freeze
it, replay it as a fixed-length differentiable scan. Rejection is exactly why the pilot must be
separate: the transient marcher accepts every attempt on purpose, because a discarded state makes the
per-step adjoint run at zero cotangent and returns a NaN gradient. The replay has nothing to reject.
The schedule is piecewise constant in the parameters, so the gradient over a frozen one is the true
derivative almost everywhere — the same contract `adapt=` makes for a frozen mesh sequence.

The trajectory is resampled back onto the domain's declared `tau=` grid (as the transient resamples onto
`save_ts`), so the returned shape does not depend on the steps taken and the resampling error is bounded
by `limit` itself. `fem.tau_schedule` reports what the pilot chose.

**A parametric form refuses to pilot, by design.** The pilot needs concrete values to accept or reject a
step and a differentiable solve hands it tracers; piloting at the parameters' *stored* values would
silently adapt to whatever they happen to be — 0.0 for a fresh `jno.np.parameter`, i.e. a load path that
never happened. Discover the schedule forward, then replay it:

```python
fem.solve(tau=jno.solve.adaptive(limit=0.05))   # forward, at the values you want
fem.solve(tau=fem.tau_schedule)                 # differentiable replay of that schedule
```

`tau=<array>` accepts any strictly increasing grid spanning the declared path, so it doubles as the
"non-uniform grid I chose myself" spelling. Not composable with a per-load-step field
(`freeze_path(frames)`), whose frames are indexed by the declared step count.

Note what adaptivity does **not** fix. If the step is cut to the floor and the change still exceeds
`limit`, that is an **unstable branch**, not a step that is merely too big: under load control a
snap-back has no nearby equilibrium, so no refinement finds one. The error says so and points at
displacement/arc-length control, which is a different instrument and is not built.

Keep `dm` in range with [`dm.bounds(0, 1)`](#inequalities--uboundslo-hi), which composes with the march.
Note that the bound does **not** replace the floor `eta` on the degradation: at `dm = 1` exactly,
`(1-dm)²` makes the displacement block singular, so the floor is a well-posedness requirement in its own
right. And a monolithic Newton is not expected to converge on this energy at all — drive it with
[`jno.solve.staggered([u, dm])`](#choosing-the-solver--the-slot-api-jnosolve--jnoprecond).

**Finite strain is also just a formula.** Tensor constants broadcast correctly (`jno.np.identity(n)` carries
a leading batch axis), so `F = I + ∇u`, `E = ½(FᵀF − I)`, `S = λ tr(E) I + 2μ E` and the internal virtual
work `∫ (F S):∇δu` are written directly — St. Venant-Kirchhoff in five lines, no module:

```python
grad, trace, inner, einsum, I = jno.np.grad, jno.np.trace, jno.np.inner, jno.np.einsum, jno.np.identity(3)
H = lambda w: grad(w, [x, y, z])
F = I + H(u);  E = 0.5*(einsum("...ki,...kj->...ij", F, F) - I);  S = lam*trace(E)*I + 2*mu*E
mech = inner(einsum("...ij,...jk->...ik", F, S), H(phi), 2)      # ∫ (F S):∇δu
```

`jno.fem` routes the nonlinear form to Newton (exact 20%-stretch patch test; reduces to linear elasticity
as strain → 0). Combine with the plastic return map for finite-strain plasticity — both are formulas.

**A hyperelastic material IS its stored energy — `jno.np.diff(psi, F)`.** For anything past St. Venant-
Kirchhoff, hand-deriving the stress is where the algebra errors live. `diff` differentiates a **scalar
expression with respect to another expression** (the constitutive counterpart of `grad`, which
differentiates w.r.t. a coordinate), so you write the energy from the paper and get the 1st
Piola-Kirchhoff stress:

```python
det, trace, einsum, jac, I = jno.np.det, jno.np.trace, jno.np.einsum, jno.np.jacobian, jno.np.identity(3)
F   = I + jac(u, X)                                     # bind it ONCE, then reuse
I1b = det(F)**(-2/3) * trace(einsum("...ki,...kj->...ij", F, F))    # first isochoric invariant
psi = C10*(I1b - 3) + C20*(I1b - 3)**2 + C30*(I1b - 3)**3           # Yeoh, 3rd order
mech = inner(jno.np.diff(psi, F), jac(phi, X), 2)       # P = ∂psi/∂F, then ∫ P:∇δu
```

Measured: on a Yeoh solid this reproduces the hand-derived `S = 2 ∂psi/∂C`, `P = F S` residual
**bit-for-bit**, and the Neo-Hookean `P = μ(F − F⁻ᵀ) + λ ln(J) F⁻ᵀ` to 1e-11. The consistent tangent
`∂P/∂F` comes out of the assembler's own element differentiation — you never write it.

Two scope limits, both fail-loud rather than silent:

* **It is pointwise.** The derivative is taken independently at each quadrature point, which is what a
  constitutive law is; an `Integral` inside the target is refused (differentiate the integrand, then
  integrate).
* **`wrt` is matched by identity.** Bind `F` to a variable and pass that same object. A rebuilt copy
  (`diff(psi, I + jac(u, X))` written inline) is a different node, and rather than differentiate to a
  silent zero it raises.

Any energy-derived law works the same way — Mooney-Rivlin, Ogden, Gent, anisotropic tissue models — as
does a chemical potential `mu = diff(f, c)` or an electro/magnetostrictive coupling.

---

## Worked examples

The [FEM tutorials](tutorials/08-fem-and-varpinns/poisson-2d-fem.md) cover every pattern above:
Poisson, mixed Dirichlet/Robin reaction–diffusion, a nonlinear Allen–Cahn interface, mixed-BC
Helmholtz, a linear-elastic cantilever, Stokes channel flow, transient heat, two inverse problems (a
hidden diffusivity field and a transient rate), two **second-order-in-time** wave examples (a vibrating
membrane and a vector-elastodynamics cantilever), the non-nodal **H(div) mixed Poisson** and **H(curl)
Maxwell / eddy-current** examples, and a **variational PINN** (a neural trial in the same weak form).

---

## Build time: what to expect, and the one knob

`jno.fem([...])` **fully assembles** the operator — it returns concrete matrix values, which is why
`fem.solve()` is then only the linear solve (measured **7 ms** on 3-D Poisson at 27,833 nodes). Some
libraries instead defer assembly into their solve; that makes their "build" look faster and their
solve slower, so compare **build + solve**, and remember jNO assembles *once* while a per-solve
assembler pays again on every Newton iteration.

Most of a cold build is **XLA compilation**, and the cost is fixed per problem *structure* rather than
per DOF — a 15x larger mesh still compiles about the same number of programs. That makes it worth
caching across processes, which is **opt-in**:

```python
dire = jno.setup(__file__, compile_cache=True)     # or, per project, in .jno.toml:
                                                   #   [jno]
                                                   #   compile_cache = true
```

| 3-D Poisson, 27,833 nodes | first build | repeat build |
|---|---|---|
| default (no cache) | 4.75 s | 2.48 s |
| `compile_cache=True` | **2.22 s** | **1.51 s** |

**Off by default on purpose**, for two reasons: a library should not write to your disk uninvited, and
the run that *populates* the cache is **slower** than having none at all — so a single cold run is a
straight loss. Turn it on for anything you run more than once: a sweep, an optimisation loop, a test
suite, or simply re-running a script after an edit.

---

## Known limitations

Almost every boundary below is an explicit, fail-loud `NotImplementedError`. **Two are not, and say so
in place**: affine (straight-edge) geometry on a curved boundary, and the `2πr` measure on an
axisymmetric *vector* form — both are cases the assembler cannot distinguish from a legitimate choice,
so they are stated where you make that choice rather than enforced. These apply when you **assemble a
weak form** or solve a **transient problem through the time route** — the residual-PINN path is
unaffected. Full detail is inline in the sections above.

- **Transient mass terms must be parameter-free** — put affine trainable parameters on the stiffness /
  residual, not on `u_t * phi`.
- **Second-order in time (`u_tt`) is scoped** to nodal Lagrange, 1D/2D/3D, scalar or vector. A
  **nonlinear spatial** operator (sine-Gordon, cubic Klein–Gordon, large-deformation elastodynamics)
  *is* supported — Newton on the augmented `[u; v]` block — but the **temporal** side must stay linear:
  a state-dependent mass or damping `c(u)·u_tt` is refused, since `M2`/`C` are extracted by
  differentiating at `u=0` and would otherwise be frozen there. A **coupled** 2D/3D system flows
  through the *same* assembler as a single field, so damping, the nonlinear path and a driven
  boundary `g(x,t)` compose with coupling; what a coupled form still refuses: a field with **no**
  `u_tt` term (its velocity rows would be singular), **runtime parameters** (the parametric coupled
  steady assembly underneath is not wired), and periodic ties. Time-varying Dirichlet is refused on
  nonlinear forms. A **complex coefficient** on any `u_tt` form is refused by name — it used to be
  silently cast to real (write the problem first-order in time; the complex transient is supported).
  A coupled 1D system carries `u_tt` on narrower terms (linear, undamped): the augmented state is
  `[u_all; v_all]`, so `fem.offsets` lists the displacement blocks then the velocity blocks.
- **Reduced-order (`basis=`) solves** cover steady and **first-order transient** (linear and
  nonlinear). Second-order-in-time (`u_tt`), complex, and periodic-tied problems refuse, each with its
  own reason. Nonlinear reduces, but without hyper-reduction that is a memory win, not a speed one.
- **No runtime Dirichlet parameters** — a trainable parameter may sit in the operator (stiffness) but
  not in an essential/Dirichlet boundary *value*.
- **Affine parameter lowering expects a single, direct factor** — one trainable scalar per additive
  term (`nu * grad(u)·grad(phi)`), not nested or buried in a nonlinear expression.
- **Enclosure radiation is a composition, not an auto-detected term** — it is 2D / axisymmetric and
  needs a direct linear solve; you write the radiosity and couple it yourself.
- **Plasticity is small-strain, isotropic, linear-hardening, whole-domain.** Deformation theory
  (monotonic / proportional) and the path-dependent flow-theory **`tau=` load-path march** both run
  today; the march assembles on the real, steady native-Lagrange path — **single-field or coupled**
  (not transient / complex / 1D / non-nodal / periodic — each rejected with a clear error). The
  internal-state readout runs on every cell (sub-region-restricted plasticity is not wired). Kinematic /
  nonlinear (Voce) hardening and contact are separate formulas / machinery, not built.

- **Geometry is affine (straight-edge) at every order** — there is no isoparametric mapping, so on a
  **curved boundary** the domain itself is approximated to O(h²) and that error caps every element
  order above it. This one is *not* fail-loud: the solve is silently suboptimal. Measured on the unit
  disk (`-Δu = 4`, `u|∂Ω = 0`, exact `u* = 1 − r²`), L2 rates under refinement:

  | order | expected | measured | error at `h = 0.05` |
  |---|---|---|---|
  | P1 | 2 | **2.00** | 1.14e-03 (1 536 dofs) |
  | P2 | 3 | **2.02** | 7.46e-04 (6 015 dofs) |
  | P3 | 4 | **2.01** | 7.40e-04 (13 438 dofs) |

  P3 buys *nothing* over P2 on a curved domain — 13 438 dofs for the same answer — and both are barely
  ahead of P1. The cap comes from imposing the boundary condition at straight-edge nodes that sit on the
  chord, O(h²) inside the true arc. On a **polygonal** domain the advertised rates hold exactly (the
  suite measures P2/P3 there). Until an isoparametric mapping exists, prefer `h`-refinement (or the
  adaptive loop) over `order ≥ 2` near curved boundaries.
- **Element order on a non-nodal family is refused, not applied** — RT/N1E/P0/Hermite/Argyris/Morley
  each have one intrinsic order. `space="N1E", order=2` used to return the same lowest-order space
  silently; it now raises. The mesh is the only accuracy knob on an H(curl)/H(div) problem — see
  *Mesh resolution for wave problems* for what a given points-per-wavelength buys, measured.
- **`jno.solve.eigs` / `FEM.eigs` route on the operator's actual symmetry.** A symmetric pencil takes
  the symmetric reductions (real spectrum, differentiable). A genuinely non-self-adjoint operator is
  routed to **ARPACK/Arnoldi** (Lehoucq & Sorensen 1996) and returns the **complex** spectrum it
  actually has — the case that matters for stability problems (resistive MHD growth rates, drift
  waves, anything with a mean flow), where the sign of the growth rate *is* the physics. Neither path
  ever returns the spectrum of `½(K+Kᵀ)` as though it were the answer. The routing probe is a
  randomized bilinear test and is concrete-only, so **under `jit` the symmetric path is assumed**.
  Limits of the non-symmetric path, which are real: **the eigenvalues are differentiable in reverse
  mode** — `dλ = wᴴ(dA − λ dB)v / (wᴴBv)` for a simple eigenvalue (Wilkinson 1965 ch. 2), with the left
  eigenvector obtained by inverse iteration on `(A − λᵢB)ᴴ`, verified against finite differences to
  1e-09 — but the **eigenvectors are not**, and differentiating through them yields **NaN** rather than
  the silent zero a plain callback would give. A **defective** eigenvalue has no derivative at all (its
  perturbation series runs in `√ε`) and is detected via the eigenvalue condition number, giving NaN
  instead of the enormous finite number the formula would otherwise produce. Because that guard is a
  `custom_vjp`, **forward mode (`jax.jvp`/`jacfwd`) is unavailable here** — use `jax.grad`/`jacrev`. It
  accepts `linear=jno.solve.lu(backend="pardiso"/"cudss"/"host")` to drive ARPACK's shift-invert
  factorization — those kernels are plain numpy functions, so ARPACK can call them from host code —
  but refuses `backend="device"` (a JAX primitive), the Krylov solvers, and `precond=` rather than
  ignoring them. **`linear=` is opt-in because it is not always a win:** ARPACK applies the inverse
  ~50–70 times, so it trades one fast factorization against per-application overhead — measured with
  PARDISO, **0.72× at n=3,000** (slower) but **10.05× at n=20,000** (21.6 s vs 217 s). Finally it needs
  `k < n-1`, smaller pencils taking an exact dense `scipy.linalg.eig`. Order with `which="LR"`/`"SR"` (real part — the growth rate) or
  target an interior region with `sigma=`.
- **Axisymmetric `(r, z)` VECTOR forms are your responsibility, and this one is *not* fail-loud** —
  the `2πr` measure is exact for scalars and wrong for vectors (elasticity hoop strain; and for vector
  Maxwell the cylindrical curl's own `1/r` terms plus the meridional/azimuthal decoupling). jNO ships
  no axisymmetric H(curl)/H(div) element, and multiplying by `r` is arithmetic the assembler cannot
  distinguish from a legitimate radial coefficient, so nothing raises. Use a full 3-D mesh for vector
  Maxwell, or write the cylindrical operator out yourself. See *Axisymmetric (bodies of revolution)*.
