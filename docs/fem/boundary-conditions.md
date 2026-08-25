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

**One-sided (separating, Signorini) contact works** with the `max(0, -c*g)` spelling above, and the
earlier "the kink stalls Newton" note was re-measured and re-classified: the stall was a **float32
residual floor** (~2e-5 on the reference stack), not active-set cycling — `jax.linearize` through
`max` selects the active branch, which *is* the semismooth Jacobian, and under x64 the identical
iteration meets `rtol=1e-8` at residual 4e-10, superlinearly. Practical guidance: set tolerances the
precision can reach (`newton(line_search=True, rtol=1e-6, atol=1e-6)` in float32; anything tighter
wants x64), press converges to the bonded answer like `1/c`, and release separates exactly — no
adhesion, measured `max|u| < 1e-9` on the far body.

To remove the penalty's `O(1/c)` penetration error, add the **augmented-Lagrangian** multiplier — a
scalar *surface state* on the secondary face riding the existing `evolves` + `tau=` march machinery,
no new API:

```python
lam, _ = d.fem_symbols(value_shape=())            # the multiplier, per face quadrature point
p = jno.np.maximum(0.0, lam.i(-1) + c*(-g))       # AL pressure
fem = jno.fem([..., p * inner(n, phi_s, 1), lam.evolves(p), *bcs])   # tau march = Uzawa updates
```

Measured on the reference stack at the *same* `c = 1e3`: penalty error 3.4e-3, AL error **1.1e-5**
after 8 updates, falling monotonically. Differentiable through *closed* contact: `jax.grad` of a
response w.r.t. a load or (parametric-Dirichlet) grip displacement runs through the driver's
`custom_root` on the branch-selected operator and FD-checks; at contact *onset* the derivative is a
subgradient (the `max` kink).

!!! warning "Scope — small sliding"
    Small sliding — the pairing is frozen at build time, so a configuration that slides must be
    rebuilt per load step. Differentiable in the DOF values but **not** in the mesh coordinates (the
    projection weights are host-computed). Frictionless: no tangential traction, so a body held *only* by
    contact is free to slide and its system is singular — constrain the tangential direction independently.
    The **assembled tangent now carries the gap's nonlocal blocks** — `(s,m)` from `jacfwd` w.r.t. the
    gathered main values chained through the frozen mortar weights, plus the reaction rows' `(m,s)` and
    `(m,m)` — verified against the matrix-free JVP on random probes in both the active and separated
    branches, so `newton(direct=True)` + `lu`/cuDSS works with contact (the pattern is static; inactive
    contact contributes zeros in the data, which keeps the sparsity-keyed factorization caches valid).

---
