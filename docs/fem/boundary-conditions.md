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

#### Time-varying Dirichlet data — `g(x, t)`

On a transient form `g` may depend on time too: write it with the boundary variable's `t`. Nothing is
passed to `fem.solve()`:

```python
xin, yin, tin = d.variable("inlet", split=True)
ramp = 1.0 - jno.np.exp(-tin / 0.1)                              # smooth start-up
fem = jno.fem([
    momentum, continuity,
    u(xin, yin)[0] - ramp * 4 * U * yin * (H - yin) / H**2,      # ramped (or pulsatile) inflow
    u(xin, yin)[1] - 0.0,
    ...                                                          # walls, initial condition
])
```

The constrained row reads `u = g(x, t)` at the time the step lands on, in every time scheme: `t_{n+1}`
for `theta` (a Dirichlet row carries no time derivative, so every θ imposes it at the new time) and
`bdf2`, each stage time for `sdirk`, and each stage time plus the `∂g/∂t` term for `rosenbrock`. The
interior equations see the boundary's rate through the mass matrix, whose columns on those rows are
kept. This holds on linear and **nonlinear** forms, single-field or **coupled** — so a ramped or
pulsatile inflow, a moving lid, or a manufactured solution with exact time-dependent boundary values on
Navier–Stokes is written the same way — and the march stays differentiable in the form's runtime
parameters, a `jno.np.parameter` or a trainable net coefficient, single-field or coupled (a P2
manufactured solution with `k` set at runtime is reproduced to 1e-8, identical to `k` written as a
number). Measured on the Taylor–Green vortex with the exact velocity imposed on all four walls
(`tests/test_fem_coupled_time_dirichlet.py`): BDF2 stays second order in time (2.3; with the data one
step late it drops to 1.05, and at 16 steps is ~900× less accurate).

**Boundary data you want to identify.** A trainable parameter may sit inside the value — an inflow
amplitude, a ramp rate. The held value is re-evaluated at each step's time with the runtime value of the
parameter, so the march is differentiable in it (reverse mode through every scheme):

```python
a = jno.np.parameter((1,), name="a")                                  # unknown amplitude
a.dtype(jnp.float64); a.initialize(jax.nn.initializers.constant(0.5)); a.optimizer(optax.adam(5e-2))
fem = jno.fem([form, u(xb, yb) - a * xb * jno.np.sin(3 * tb), u(ci[0], ci[1]) - 0.0])
crux = jno.core([(fem.solve(time=jno.solve.bdf2()) - u_traj).mse], domain=obs)
crux.solve(300)                                                       # a: 0.5 -> 1.7
```

Measured (`tests/test_fem_coupled_time_dirichlet.py`): the recovery above lands on 1.70000007. With `a`
set at runtime the march is bit-identical to the one with `a` written as a number — linear and nonlinear,
coupled and single-field, in `theta`, `bdf2`, `sdirk` and `ros2` — and `jax.grad` w.r.t. `a` of an
interior functional matches central differences to 1e-6.

*Scope, all loud:* a trainable **net** inside a time-varying value, or a net/parameter-valued Dirichlet
**beside** one on a transient form; a parameter inside the value on a second-order (`u_tt`) form or on
the τ load path; a nonlinear second-order (`u_tt`) form; a time-varying value on a non-matching tied interface (see
below). **Accuracy:** a time-varying boundary value costs Runge–Kutta-type schemes order — the classical
*order reduction* from their low stage order (Ostermann & Roche, *Math. Comp.* 59 (1992) 403–420). See
[the measured orders](limitations.md#the-detail).

#### Per-tag surface coefficients — `d.attach(tag, h=...)`

A boundary term is normally written per tag, on that tag's coordinates. When the *same* condition
applies over the whole boundary with only its coefficient changing, attaching per tag collapses it
into one term — the surface mirror of a per-region coefficient:

```python
d.tag("wall", lambda x, y: x < 1e-9)
d.tag("lid",  lambda x, y: y > 1 - 1e-9)

d.attach("wall", h=25.0).attach("lid", h=5.0)     # per-tag film coefficient, read back as d.h
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

**Limits, all loud:** surface terms only — a `TagMask` in a *volume* term raises, as does a surface coefficient on
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
a clear error. One box per field. A box composes with a **periodic tie**, steady or on a `tau=` march:
the tied solve runs on the kept unknowns `u = P ũ`, and the box is imposed there — the full box restricted
to the kept DOFs, which is exact because an eliminated DOF *is* the DOF it is tied to. So the bound must be
the same on both tied faces (a periodic `lo`/`hi`, or `u.i(-1)`, which satisfies the tie by construction);
a bound that differs across the tie is refused by name, and so is a box beside a *weighted* elimination
(a non-matching mortar interface, hanging nodes, a slip condition), whose eliminated values a box on the
kept ones would not bound. Measured: the periodic obstacle problem matches its analytic free boundary, and
a tied ratchet `u.bounds(u.i(-1), None)` holds the unit-load linear solve to 1e-8
(`tests/test_fem_bounds_periodic.py`, `tests/test_fem_history_march_periodic.py`). Note that a bound is
not a cure for an ill-posed operator: in a
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

A **vector wall value** clamps every component at once: `u(xb, yb) - (1.0, -0.5)`, or a varying one
`u(xb, yb) - jno.np.stack([gx, gy], axis=-1)` (`gx`, `gy` may read `t` for a driven wall). A scalar value
on a vector field is the same number on every component, and `u(xb, yb)[i] - g` clamps one component
and wants a scalar `g`; a vector there raises. Pinned in `tests/test_fem_vector_dirichlet_values.py`.

A **matrix field** (`value_shape=(n, m)`) takes the same two forms: `S(xb, yb) - G` pins every entry, with
`G` a matrix — a constant, `jno.np.identity(2)`, or one built from coordinates with
`jno.np.stack([jno.np.stack([a, b], axis=-1), jno.np.stack([c, d], axis=-1)], axis=-2)` — and
`S.bind(x=xb, y=yb)[i, j] - g` pins one entry. A row `S(...)[i]` is not one entry and raises. A vector
field with more than three components pins any of them, `u(xb, yb)[3] - g`; `fem.classification` labels
those entries, and every matrix entry, by their flat index (`dirichlet@left[2]` is entry `(1, 0)` of a
2×2 field), not by an axis name.
On a `symmetric=True` field the whole-tensor value must be symmetric — an asymmetric one raises rather than
keeping half of it — and `S[0, 1]` and `S[1, 0]` name the same stored value.

!!! warning "Fixed: a vector wall value used to keep only its first component"
    Until this was fixed, `u(xb, yb) - (1.0, -0.5)` imposed `(1.0, 1.0)` — silently, steady and
    transient alike — and a vector `g(x, t)` wrote the wrong values. Examples that only used `(0, 0)`
    could not show it.

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
from, so a diagnostic form can be assembled against an existing solution. **Scope:** the native Lagrange
assembler, volume terms and terms on a tagged boundary region. A term with **no** test function is not
refused: it integrates to a scalar instead, as described next.

Verified by global balance, not by restating the assembly: the wall flux equals the integrated source,
and the reaction equals the applied load.

### Integrating a quantity — `fem.eval` on a test-free expression

The same entry point, given an expression with **no** test function, returns the scalar
`∫ F dΩ` (or `∮ F ds`) instead of a per-DOF vector. The expression decides which: a weak term
assembles, an integrand integrates.

```python
C    = fem.eval(sig(u, rho) @ eps(u), sol, args=design)   # compliance, an energy integral
V    = fem.eval(rho, sol, args=design)                    # the material volume
out  = fem.eval(inner(u.bind(x=xo, y=yo), e), sol)        # a mechanism's output displacement
```

This is the *objective* half of a design problem, and it is why it exists: every one of compliance, a
volume fraction, a stress p-norm and a mechanism output is one integral, where each previously needed
its own hand-written reduction over the DOF vector.

**It integrates on the quadrature the operator was assembled with**, not a rule of its own. That is
what makes the identity below exact rather than exact-to-within-a-quadrature-error that nothing
reports:

```python
fem.eval(a(u, u), sol) == sol @ fem.eval(a(u, phi), sol)     # ∫ σ(u):ε(u) dΩ  ==  uᵀKu
```

The region comes from the coordinate Variables in the expression, exactly as it does for a weak term,
so a boundary integrand is spelled by binding the field to that region's coordinates. A term naming
two regions is refused — an integral has one measure.

Differentiable in all three of its arguments: the solution, the runtime parameters (a P0 density
included), and the **mesh coordinates** — `∂/∂X` flows through `|det J|` and through the facet area
element, which is what a deformable-mesh design problem needs.

Scope, and it is narrower than the weak path: the whole volume and tagged boundary regions only (a
sub-region-restricted functional raises), and steady native-Lagrange problems only — transient,
complex, 1-D, non-nodal elements and the VPINN path never publish the assembler this rides on, and
say so. Where it is unavailable the equivalent is `sum(fem.eval(F * phi, sol))`: a Lagrange basis is a
partition of unity, so summing the weak term `F·φ` over every DOF is the same integral.

### Tying a region to a constant — `u(region) - U`

With `U = d.unknown.scalar(constant=True)`, the term `u(xr, yr) - U` makes every DOF of `u` on the region
**the same unknown** as `U` — a uniform value whose magnitude is solved for: a floating conductor, a rigid
plug leaving a channel, an electrode at an unknown potential. `u(xr, yr)[i] - U` ties one component, and
`u(xr, yr) - U` with a vector constant ties a vector field component by component.

It is exact, not a penalty: the tied DOFs are eliminated by a prolongation `u = P ũ`, the way a periodic tie
is, and their equations are **summed** into U's row. That sum is the virtual work of the constraint, so the
equation for `U` is "the net flux (force, current) through the region equals what the form applies there":
a Neumann term on the region becomes U's equation without being written twice. On `-Δu = 1` with `u = U`
on the right edge and `-(Q/H) * v` there, `U = Q/H + 1/2` comes out exactly.

`U - g` pins the constant (a one-DOF Dirichlet row), and then the tie reproduces the hand-built
`u(xr, yr) - g`. `U.bounds(lo, hi)` bounds it, and through the tie every tied DOF (a tied DOF without a
bound of its own takes U's). A DOF on the region that also carries a Dirichlet value keeps it — the
prescribed value wins over the tie, as on a periodic face.

A tie composes with an exact slip condition `n·u = 0`: the slip is eliminated first, and the tie is a
selection on what the slip leaves (`u = P_slip P_tie ũ`). A node may carry both when they constrain
different components, e.g. an axis-aligned wall's normal component by the slip and the other component by the
tie. A node whose tied component is also the one the slip eliminates (a whole-vector tie on a slip node,
for instance at a corner) is refused: it would be eliminated twice. Tie the region without those nodes.
`U.bounds(lo, hi)` together with a slip raises, as a bound with a slip alone does (the slip rows are
weighted).

Scope: steady linear and nonlinear forms on the native 2-D/3-D Lagrange assembler; a tie together with a
periodic tie, a slip surface that moves with trainable coordinates, or a hanging-node mesh raises (two
prolongations would have to be composed), and so does a transient form.

### Tying two boundaries — `u(A) - u(B)`

A term that names two boundary regions and carries no test function is a **tie**: it identifies the
DOFs on region `A` with those on region `B`. It is enforced by algebraic reduction (a prolongation
`P` that eliminates the `A` DOFs), not by assembly, so it composes with everything downstream —
complex, transient, Bloch (`u(A) - c*u(B)`), `basis=`, and the `domain(tau=...)` load-path march all
reuse the same `P`.

```python
d.tag("left",  lambda x, y: x < 1e-9)      # a tag predicate includes the corner nodes,
d.tag("right", lambda x, y: x > 1 - 1e-9)  # which matters — see below
xl, yl = d.variable("left", split=True)[:2]
xr, yr = d.variable("right", split=True)[:2]

fem = jno.fem([weak_form, u(xl, yl) - u(xr, yr)])
```

The faces enter through their coordinates, like every other boundary term: `u("left")` with a bare tag
name is not a tie and raises.

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

**A value on one node of a tied face, and `p.pin()`.** The tie says a node and its periodic images are
**one** unknown, so a value written at one of them holds at all of them: `u(corner) - 0` on the corner of a
doubly periodic box fixes all four corners. `p.pin()` needs no such care. It pins the vertex nearest the
min-corner that lies on **no** tied face (with no ties, the min-corner itself). A gauge only needs some DOF
of the field's constant null space, and a node off every tied face is never eliminated and never has
another row summed into it. So the pin is the same one-node condition with or without periodicity, steady
or transient, single-field or coupled. `p.pin(mean=True)` re-levels afterwards to `∫p dx = 0` exactly as
without ties. Two different values on nodes a tie identifies are refused by name. So is a value on a
single node of a *weighted* (mortar / collocated / Bloch) tie, whose image is a weighted sum of other
nodes: prescribe it on the kept side of the tie (the `B` in `u(A) - u(B)`) instead.

```python
at = lambda tag: d.variable(tag, split=True)[:2]                          # (x, y) on a face
fem = jno.fem([momentum, continuity,
               u(*at("left")) - u(*at("right")), u(*at("bottom")) - u(*at("top")),   # fully periodic
               p(*at("left")) - p(*at("right")), p(*at("bottom")) - p(*at("top")),
               p.pin()])                                                  # a vertex off the tied faces
```

!!! warning "Scope"
    A tie works on a **transient** form as on a steady one: a single field, scalar or vector, first or
    second order in time (`u.t`, `u.tt`), linear or nonlinear, with constant or time-varying wall data
    `g(x, t)`; and a coupled first-order system. Measured: a vector march equals two scalar marches of
    the same equation to 1e-10 with the seam equal node for node, and the mortar patch test marched in
    time stays on the linear field to 1e-17 (`tests/test_fem_periodic_transient_vector.py`,
    `tests/test_fem_vector_tie.py`). Refused by name: a tie on a **coupled** `u_tt` form; a **complex**
    transient with a time-varying Dirichlet value (not wired with or without a tie); a Bloch tie on a
    real transient (see [solvers](../solvers.md)).

    A tie combined with `u.gap` assembles but solves to a **deferred trace node** rather than an
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

    The exclusion applies where the tie partner is prescribed as well, e.g. the corner of a periodic
    channel with no-slip walls. A prescribed DOF whose exact partner is **free** is left to the tie instead,
    and its value is imposed on the reduced DOF it resolves to. Excluding it tore the tie at that node,
    silently: a pinned corner of a doubly periodic Poisson problem held 0 while its three images held
    -0.037, for a solution of order 1. The coupled (multi-field) reduction did not apply the exclusion at
    all, so every coupled problem with a value on a tied face failed to build, a pressure pin on a
    periodic box among them. A nonlinear **transient** with a restored row failed on its first step, and
    its reduced mass row is now emptied as the linear march empties it. All of this is pinned in
    `tests/test_fem_pin_periodic.py` against Poisson, Poiseuille and Taylor–Green solutions.

    Every reduced-space path goes through one helper for this (`impose_reduced_dirichlet`, or
    `wrap_reduced_dirichlet` for the residual-form paths). That matters: the steady real path was fixed
    first and the others stayed wrong for exactly as long as they had their own copy of the logic —
    5.8e-04 on the fused-complex path, whose `blkdiag(P, P)` transform dropped the record entirely, and
    4.2e-04 on the transient, which built its reduction through a second construction site that never
    annotated it. Both are now at round-off and are measured against their conforming controls. The
    second-order (`u_tt`) route was a third construction site, with neither the exclusion nor the
    record: on a membrane periodic in x with held walls, a non-periodic wall value (`u = x` on y = 0)
    moved a tied corner by the whole value, 1.0, and a value held on one side of the tie only was 0.5 off
    at its image. It now goes through the same helpers, and since a `u_tt` block starts from a
    wall-consistent state (`u = g`, `u_t = 0`), the image starts at the held value too. Pinned in
    `tests/test_fem_second_order_time.py` against the analytic standing wave and the same data written
    without the corner conflict.

    A **time-varying** essential value on a polluted interface row is refused by name: its held value is
    written into the full row every step and there is no constant to put back.

!!! measured "A tie on a load-path march — applied at every step"
    A form that reads step history (`u.i(-1)`, or a `state.evolves(...)` update) marches over a
    `domain(tau=...)` grid, and that march used to hand Newton the **full** residual: the tie was dropped
    without a word. A backward-Euler heat step written by hand, periodic in x, came back bit-identical to
    the same form with *no* condition on the tied faces (natural Neumann) — 30% of max|u| off the `u.t`
    reference. The exact slip condition `n·u = 0` and the hanging-node constraint ride the same `P` and
    were dropped the same way (a wall velocity of full magnitude; 29% of max|u| on a hanging node).

    Each step now solves `Pᵀ r(P ũ) = 0` and the march carries `u = P ũ`, so the history buffers and every
    `.evolves` update are computed from a field that satisfies the tie. The same hand-written step matches
    the `u.t` march with `time=jno.solve.theta(1.0)` to ~1e-10 relative, single-field and coupled, with the
    seam values equal exactly. Pinned in `tests/test_fem_history_march_periodic.py`.
    `tau=jno.solve.arclength(...)` does not solve in the reduced space and refuses such a form by name. A
    `.bounds(...)` box is imposed on the reduced unknowns (see [inequalities](#inequalities-uboundslo-hi)).

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
it faced before anything moved, and **nothing reports it** — the solve converges perfectly well, just
for a contact configuration that is not the one being solved.

How wrong that gets: a block resting on a disk's crown, slid right by 0.9 — ten element widths. Its
nearest point is now over `x = 0.6`, where the disk has fallen away and the true separation is
**0.182**. The frozen pairing still reports **0.05**, the value it measured at the crown, because the
slide is tangential and nothing in `g0 − n·D` moves it. Nearly four times wrong, and silent
(`tests/test_fem_contact_search.py::test_the_search_follows_a_slide_of_many_facet_widths`).

Where the surfaces do *not* slide far, the two agree closely — the other half of the picture, and the
reason this is a slot rather than the default. A 12:20 involute gear pair against the kinematic oracle
`|T_B/T_A| = z_B/z_A`, rim mesh 0.050 → 0.018 (6856 → 16418 DOF):

| h_rim | 0.050 | 0.035 | 0.025 | 0.018 |
|---|---|---|---|---|
| frozen pairing | 0.84% | 0.85% | 0.85% | 0.85% |
| `contact=` | 0.64% | 0.64% | 0.65% | 0.64% |

Both are flat under refinement and the search buys about 0.2 points, so on a rolling contact that
barely slides it is an accuracy refinement, not a rescue. Reach for it when the surfaces genuinely
move against each other.

```python
u = fem.solve(contact=jno.solve.contact())      # re-pair from x + u until it settles
```

All four knobs are optional. `capture=` is the search radius: `None` (the default) derives one per
pair from the local facet size — 3× the mean secondary facet diameter — which is right unless the
bodies must close several elements' worth of distance before they touch. Beyond it a point is
**inactive**: it keeps its slot with zero weight, so the tables never change shape. `rounds=` caps the
loop (12); `tol=` is the relative movement `|u_k − u_{k−1}|∞ / |u_k|∞` that counts as settled (1e-4);
`relax=` damps the round-to-round update when the search oscillates, which a follower normal can
cause. Damping does not move the fixed point, but converging under it is **not** the same as
converging to the right branch — check a damped result against something independent, such as the
same solve with the pairing frozen.

Each round solves the ordinary system with the current pairing, then re-runs the search at `x + u`,
warm-started from the last round. It stops when the pairing is unchanged **and** the solution has
stopped moving; exhausting `rounds` raises and says which of the two failed, rather than returning a
half-converged answer.

`main` may also be a **list of candidate surfaces** — each point pairs with the nearest across them —
and listing the secondary itself is **self-contact**, `u.gap("strip", ["strip"])`. A facet is then
barred from pairing with its own neighbours *and* from pairing with a facet whose normal does not face
it, which is what stops a flat stretch from reading as touching itself everywhere. A candidate list
requires `contact=` and is refused without it.

**Either tangent works**, and on this problem the assembled one is simply faster. The matrix-free
Newton (`newton(direct=False)`) re-pairs for free; `nonlinear=jno.solve.newton(direct=True)` assembles
the tangent and rebuilds the contact block's sparsity pattern each round. Measured on a 12:20 gear pair,
11 682 DOF, 5 rounds, median of three runs, when the matrix-free Newton was still the default:

| tangent | time | peak RSS |
|---|---|---|
| matrix-free (`direct=False`; the default then) | 44.2 s | 1848 MB |
| `newton(direct=True)` | **14.7 s** | 1894 MB |

Peak RSS is *comparable* here, not a trade: the assembled contact block is small next to the ~1.8 GB
the JAX/XLA runtime already holds at this size, so its cost does not surface. Expect that to change as
the interface grows — the time difference is the robust part of this measurement (matrix-free spanned
42.7–48.6 s across runs, direct 14.5–15.1 s).

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
