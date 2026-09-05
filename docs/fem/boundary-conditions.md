# Boundary conditions are residual terms

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

A tie works on a **scalar or a vector** field. On a vector field the mortar rows are unchanged — they
are node-pair weights — and the prolongation is expanded componentwise, `kron(P_node, I_vec)`. That is
what lets one interface be **tied** while another is in **contact** on the same displacement field, which
is the ordinary two-body setup:

```python
u, phi = d.fem_symbols(value_shape=(3,))
fem = jno.fem([mech,
               u(*d.variable(seam1_a, split=True)) - u(*d.variable(seam1_b, split=True)),   # bonded
               maximum(0.0, -c * u.gap(seam2_a, seam2_b, domain=d)) * inner(n, phi_s, 1)])  # contact
```

!!! warning "Scope"
    A **transient** tie is still scalar-only (that route pre-builds its own reduction), and refuses by
    name. A tie combined with `u.gap` assembles but solves to a **deferred trace node** rather than an
    array, because the gap marks the form structurally nonlinear and a reduced nonlinear system stays
    lazy so its node can flow into `jno.core` for an inverse problem. Evaluate it the way
    `tests/test_fem_periodic_unstructured.py::test_periodic_nonlinear_reaction_diffusion` does, via a
    throwaway `jno.core([...]).eval([node])` — note that `np.asarray` on it yields a 0-d **object**
    array, not the solution.

!!! measured "The mortar tie passes the patch test — and what it took"
    A dual-mortar coupling reproduces a linear field exactly across a non-matching interface: measured
    **1.0e-17** on the linear-field patch test, against a field of order 3e-02, and
    `P^T (A u* - b)` = 3.0e-18 on every free reduced row. It did not always, and the two defects behind
    that are worth knowing because they are easy to reintroduce.

    **One formula per interface.** The builder used to take an exact node-to-node shortcut whenever a
    secondary node happened to coincide with a main node, and fall back to a mortar row otherwise — 8 of
    12 secondaries on a typical stacked-block interface. Mixing weight-1 collocation rows into a dual
    operator destroys the biorthogonality the method's consistency rests on. The interface is now
    classified **once** (`conforming` / `mortar` / `collocated`, reported as `tie_counts`) and every
    secondary takes that formula.

    **The multipliers at the interface rim.** With one formula throughout, the normal mode became exact
    but the two tangential ones broke (1.1e-02), because perimeter secondaries drag outer-boundary flux
    into interior interface equations. jNO uses Wohlmuth's boundary-modified multiplier space (SIAM J.
    Numer. Anal. 38(3):989-1012, 2000, §3): rim multipliers are dropped and redistributed onto their
    interior neighbours, `psi~_i = psi_i + sum_p c_ip psi_p`, with the column sum `sum_i c_ip = 1` that
    keeps `sum psi~ == 1`. Rim secondaries then carry no multiplier and stay **free DOFs** — they are
    kept, not eliminated, which is visible in `kept_nodes`.

    Two consequences worth expecting: mortar weights are no longer non-negative (rim entries are
    negative), and a one-element-wide secondary patch has no interior node at all, so it falls back to
    collocation. `P.sum(axis=1) == 1` and linear reproduction both still hold exactly.

    Pinned in `tests/test_fem_tie_dirichlet_conflict.py` and `tests/test_fem_mortar.py`.

!!! measured "A prescribed value on a tied face — imposed after the reduction, on every path"
    A tie is imposed as `u = P x` and the system reduced as `P^T A P`. `P^T` **sums** each eliminated
    DOF's equation into the rows it ties to — so a prescribed DOF that is a tie target loses the unit row
    holding its value, and `P^T b` loses `b[d] = g` with it. Nothing raised; the boundary condition
    simply stopped being imposed.

    Two things fix it, and both are needed. A prescribed DOF is excluded from the elimination (per **DOF**,
    not per node, so a roller keeps the tie on its free components), and the rows the congruence pollutes
    are re-imposed in the reduced space — symmetrically, because restoring the row alone leaves the
    reduced column populated and silently downgrades LDL^T to general LU.

    Every reduced-space path goes through one helper for this (`impose_reduced_dirichlet`, or
    `wrap_reduced_dirichlet` for the residual-form paths). That matters: the steady real path was fixed
    first and the others stayed wrong for exactly as long as they had their own copy of the logic —
    5.8e-04 on the fused-complex path, whose `blkdiag(P, P)` transform dropped the record entirely, and
    4.2e-04 on the transient, which built its reduction through a second construction site that never
    annotated it. Both are now at round-off and are measured against their conforming controls.

    A **time-varying** essential value on a polluted interface row is refused by name: its held value is
    written into the full row every step and there is no constant to put back.

### The tangential companion — `u.slide`

`u.gap(secondary, main, domain=d)` gives the **normal** separation of a contact pair as a scalar,
`g = g0 + n·(u_s - u_m∘Phi)`. `u.slide(secondary, main, domain=d)` is its sibling: the **tangential**
part of the same relative displacement, as a vector,

```
s = (u_s - u_m∘Phi) - n (n·(u_s - u_m∘Phi))
```

— the jump with its normal component projected out. Same pair, same frozen mortar weights, same frame;
the two together decompose the relative displacement completely, which is why they are separate symbols
rather than one bundled return.

It exists because the normal traction alone is frictionless: a body held only by `u.gap` can slide freely
along the interface and its system is **singular**. A tangential term built from `u.slide` is what closes
that null space — a bonded (stick) interface at penalty stiffness `ct`:

```python
sv  = d.variable(sec, split=True)
vs  = phi.bind(x=sv[0], y=sv[1], z=sv[2])
nrm = d.variable(sec, normals=True)
g, s = u.gap(sec, main, domain=d), u.slide(sec, main, domain=d)

terms = [mech,
         (-cn * g) * inner(nrm, vs, 1)      # normal:     no interpenetration
         + (-ct) * inner(s, vs, 1)]         # tangential: no sliding
```

Coulomb friction is the same term with `ct` replaced by a formula in the trace — a radial return on the
surface state, written out as a `state.evolves(...)` update, not a material object.

!!! warning "Scope — `u.slide`"
    `s` is a **displacement**, not a velocity or a plastic slip: it measures the tangential offset from
    the frozen pairing built at assembly, so it is meaningful for small sliding only, on the same terms as
    `u.gap`. Reading it through `fem.eval` returns zeros — interface symbols are not evaluable that way —
    so measure it through the mechanics (the displacement it produces), which is what
    `tests/test_fem_contact_slide.py` does.

    Two independently meshed blocks pressed together **do** register a non-zero slide even under a purely
    normal load: they bulge laterally by slightly different amounts, which is a real ~1% tangential
    mismatch, not a numerical artefact. Add roller conditions on the side faces if you want a genuinely
    uniaxial press.

### Naming one body's surface — `domain.tag(..., region=...)`

A contact pair needs two surfaces that genuinely differ, and a coordinate predicate often cannot supply
them: two gears' rims span the same radii about their own centres, and the two sides of a non-conforming
interface are coincident. `region=` says which **body** owns the face, and ownership is decided from cell
topology — the same thing that decides which cells a region's terms integrate over:

```python
d.tag("sA", lambda x, y: x**2 + y**2 > r_hub**2, region="rimA")   # gear A's outer flank
d.tag("sB", lambda x, y: (x - c)**2 + y**2 > r_hub**2, region="rimB")
g = u.gap("sA", "sB", domain=d)
```

Without `region=` a predicate true on both bodies hands **both** tags the whole boundary, and the
resulting "pair" is a body against itself. The tag's normals follow the same restriction, so
`d.variable("sA", normals=True)` gives gear A's outward normals and nothing else.

### Letting the pairing follow the solution — `fem.solve(contact=...)`

The pairing above is built **once**, from the reference configuration, and is correct only while
displacements stay far below the element size. Past that a secondary point is still tied to the facet
it faced before anything moved. Nothing reports it: the solve converges perfectly well, just for a
contact configuration that is not the one being solved — and **refining makes it worse**. Measured on
a 12:20 involute gear pair against the kinematic oracle `|T_B/T_A| = z_B/z_A`, as the rim mesh went
0.050 → 0.018:

| h_rim | 0.050 | 0.035 | 0.025 | 0.018 |
|---|---|---|---|---|
| frozen pairing | 1.97% | 2.33% | 2.66% | **2.83%** |
| `contact=` | 2.24% | 2.27% | 2.30% | 2.30% |

```python
u = fem.solve(contact=jno.solve.contact())      # re-pair from x + u until it settles
```

Each round solves the ordinary system with the current pairing, then re-runs the search at `x + u`,
warm-started from the last round. It stops when the pairing is unchanged **and** the solution has
stopped moving; exhausting `rounds` raises and says which of the two failed, rather than returning a
half-converged answer.

`main` may also be a **list of candidate surfaces** — each point pairs with the nearest across them —
and listing the secondary itself is **self-contact**, `u.gap("strip", ["strip"])`. A facet is then
barred from pairing with its own neighbours *and* from pairing with a facet whose normal does not face
it, which is what stops a flat stretch from reading as touching itself everywhere. A candidate list
requires `contact=` and is refused without it.

**Either tangent works**, and the choice is a speed/memory trade rather than a restriction. The
matrix-free default re-pairs for free; `nonlinear=jno.solve.newton(direct=True)` assembles the tangent
and rebuilds the contact block's sparsity pattern each round. Measured on a 12:20 gear pair, 11k DOF,
5 rounds:

| tangent | time | peak RSS |
|---|---|---|
| matrix-free (default) | 41.8 s | 1830 MB |
| `newton(direct=True)` | **13.4 s** | 2611 MB |

On a form that **marches** — step history (`.i(k)`) plus a `domain(tau=...)` grid — the march owns the
loop and re-runs the search at *every load step*, because the pairing that is right at the end of the
path is not the one that was right in the middle of it:

```python
u = fem.solve(contact=jno.solve.contact())     # the march is triggered by the form, not by a slot
```

!!! warning "Scope — surface ROTATION, not sliding"
    `u.gap` handles a surface that slides arbitrarily far; it does **not** handle one that ROTATES far.
    The traction is written on `d.variable(sec, normals=True)`, the *reference* normal, so once the
    contacting surface turns by `theta` the force is misdirected by `sin(theta)` — 0.62 at 38°.
    `follow_normals=True` gives the deformed normal instead, but use it **with finite-strain
    kinematics**: `sym(grad u)` is not rotation invariant, and at 38° it manufactures 0.300 of strain
    where Green–Lagrange gives 1.9e-17. Following the normal in a small-strain form buys a better force
    direction on a materially wrong stress. Write `F = I + grad u`, `E = (FᵀF − I)/2` and a PK stress,
    then follow the normal.

!!! warning "Scope — `contact=`"
    A contact march is a **host loop, not a `lax.scan`**, so it gives up the load path's reverse-mode
    differentiability — the scanned march keeps that, at a frozen pairing. `tau=jno.solve.adaptive(...)`
    and `tau=jno.solve.arclength(...)` are refused by name: both replay or root-find under a scan, where
    a host-side search cannot run. Each round is its own solve, so there is no gradient through the
    round loop either, and the search itself is host-side and not differentiable in the mesh coordinates.

!!! warning "Scope — the frozen pairing (without `contact=`)"
    Small sliding — the pairing is frozen at build time, so a configuration that slides must be
    rebuilt per load step, or driven with `contact=` above. Differentiable in the DOF values but
    **not** in the mesh coordinates (the projection weights are host-computed). The gap alone is
    **frictionless** — a body held *only* by a normal traction is free to slide and its system is
    singular; give it a tangential term built from
    [`u.slide`](#the-tangential-companion-uslide) or constrain that direction independently.
    The **assembled tangent carries the gap's nonlocal blocks** — `(s,m)` from `jacfwd` w.r.t. the
    gathered main values chained through the frozen mortar weights, plus the reaction rows' `(m,s)` and
    `(m,m)` — verified against the matrix-free JVP on random probes in both the active and separated
    branches, so `newton(direct=True)` + `lu`/cuDSS works here (inactive contact contributes zeros in
    the data, which keeps the sparsity-keyed factorization caches valid). Under `contact=` the same
    blocks are rebuilt per round, and the equivalence is asserted against a **re-paired** table set.

---
