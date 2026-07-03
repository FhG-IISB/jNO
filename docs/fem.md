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
from shapely.geometry import box
import jno

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
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
> hand-rolled experiments (e.g. `u_h = jnp.linalg.solve(fem.A, fem.b)`), so no
> `.todense()`/`reshape` is needed. `fem.operator` still exposes the raw sparse (`BCOO`) operator
> for large problems — `fem.solve()` and the solver slots work on that sparse operator directly.

---

## Domain, symbols, and derivatives

* **Domain** — any jNO domain works (`box`, `jno.domain.cube`, a CSG/`gmsh` constructor).
  Add `time=(t0, t1, n_steps)` to make it transient.
* **Symbols** — `u, phi = d.fem_symbols(value_shape=(), names=("u", "phi"), order=1)`.
  Use `value_shape=(2,)` for a vector unknown (elasticity, flow velocity), `order=k` for degree-`k`
  Lagrange (`order=2` quadratic P2, `order=3` cubic P3, … — any `k ≥ 1`), `space="RT"`/`"N1E"`/`"P0"`
  for the non-nodal families (see below), and call `fem_symbols` once per field for coupled systems.
* **Quadrature coordinates** — `d.variable("interior", split=True)` returns the volume
  coordinates; `d.variable("<edge>", split=True)` returns a boundary edge's coordinates. A
  `box` auto-tags `"left"`, `"right"`, `"bottom"`, `"top"` (and `"front"`/`"back"` for a cube);
  `"boundary"` is the whole boundary and `"initial"` the `t = t0` slice.
* **Bound views** — `ui = u.bind(x=xi, y=yi, t=ti)` ties a symbol to a set of coordinates.
  The value is `ui`; spatial derivatives are `ui.x`, `ui.y`, `ui.z`; the time derivative is
  `ui.t`. (This replaces the old `jno.np.grad(u, xg)` / `u.d(xg)` spelling.)
* **Second derivatives (4th-order weak forms).** `jno.np.laplacian(ui, [xi, yi])` (the Laplacian `Δu`)
  and `jno.np.hessian(ui, [xi, yi])` (the full `D²u`) assemble against the element's second shape-function
  derivatives, so a biharmonic / plate / Cahn–Hilliard form is written directly, e.g.
  `jno.np.laplacian(ui, [xi, yi]) * jno.np.laplacian(vi, [xi, yi])` for `∫Δu·Δv`. Needs **`order ≥ 2`**
  (a P1 Hessian is identically zero). Scalar Lagrange fields only; the physical Hessian is the exact
  affine map `∂²φ/∂x_a∂x_b = K_ia K_jb ∂²φ/∂ξ_i∂ξ_j` (`K = J⁻¹`, no curvature term on the P1 geometry).
  > **Conformity caveat.** Standard Lagrange is **C⁰**, so `∫Δu·Δv` over P2 is *non-conforming* and does
  > **not** give a convergent biharmonic discretisation. For a convergent solve use a purpose-built
  > biharmonic element — the **C¹ Argyris** element (`space="Argyris"`, below — the conforming quintic,
  > accurate) or the cheaper **non-conforming Morley** element (`space="Morley"`, below — 6 DOF; note it uses
  > the full-Hessian form `∫D²u:D²v`) — or the mixed (Ciarlet–Raviart) method (two coupled C⁰ fields with
  > `w = Δu`, first derivatives only; see `tests/test_fem_hessian.py`). The shape-Hessian assembly here is the
  > prerequisite they build on.

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

---

## Non-nodal element families: H(div) and H(curl)

> **⚠️ Experimental.** The non-nodal element zoo is new — validated on 2-D triangular meshes at lowest
> order — and its API may still change.
>
> **Supported:** Raviart–Thomas `"RT"` (H(div)) and first-kind Nédélec `"N1E"` (H(curl)) edge elements
> + `"P0"`; the `.div` / `.curl` view operators; essential edge-trace BCs — normal flux `u·n = g` (RT)
> and tangential trace `u×n = g` (N1E) — and the natural pressure BC, on the whole boundary or any
> sub-region tag (geometry-computed normals); **all solver modes** — steady-linear, steady-nonlinear
> (Newton), and transient `M u̇ + A u = c` (including nonlinear-transient and the mixed/saddle **DAE**,
> e.g. transient Darcy); and the differentiable `fem.solve()` for inverse problems.
>
> **Supported inverse:** a runtime parameter in a *volume* term is wired for **steady and transient**
> problems — both a **scalar** and a spatially-varying **P1 field** `k(x)` (`fem.solve()` returns a
> differentiable `FemLinearSystem` / `FemResidualOperator` / parametric time block that re-assembles at each
> parameter value — so `crux.solve` recovers e.g. a plate stiffness from a deflection, a diffusivity from a
> trajectory, or a full `k(x)` field). A field parameter is authored as its own P1 symbol,
> `k = jno.np.parameter(kf)` for `kf, _ = d.fem_symbols()`, and interpolated with P1 shape functions at the
> mesh vertices — independent of the non-nodal trial's basis. **Second-order-in-time (`u_tt`, a vibrating
> plate) is supported** on the vertex families too (the augmented `[w, v]` block; a direct θ-solver is
> recommended for the stiff biharmonic).
>
> **3-D Nédélec (H(curl)) is supported.** On a 3-D **tetrahedral** mesh the first-kind Nédélec `"N1E"`
> element assembles the H(curl) **mass and curl-curl** forms (`inner(u, v) + inner(u.vector.curl(x,y,z),
> v.vector.curl(x,y,z))`) — the correct edge discretisation for **Maxwell / eddy currents** (nodal Lagrange
> gives spurious modes). The covariant push-forward `Φ_phys = J^{-T} Φ_ref` is dimension-agnostic and the
> (vector) curl is taken from the physical gradient, so `.curl(x, y, z)` assembles directly; the tet-edge
> orientation is validated by the exact bilinear form `∫|curl u|²` on a multi-tet mesh. On a tet mesh **only
> N1E is wired** — RT / P0 / Hermite / Argyris / Morley remain 2-D-triangle only and raise.
>
> The essential **PEC wall** `n×E = 0` (homogeneous tangential trace) is supported, written
> `u.vector.cross(d.variable(region, normals=True))` — `variable(region, normals=True)` returns the boundary
> normal as a single vector (its default `split=False`; pass `split=True` for the flat component tuple). It
> pins every boundary-face edge DOF of the region (facet-based; the 2-D "edge used once" rule is wrong on a
> tet, where most boundary edges are shared by several tets). Validated by a driven manufactured Maxwell
> problem that converges under refinement. An *inhomogeneous* `n×E = g` and the cavity-eigenvalue benchmark
> (needs a generalized eigensolve) are follow-ons.
>
> **Not yet / excluded:** the rest of the zoo in 3-D (RT / C¹ / plate are 2-D only — 3-D otherwise uses
> nodal Lagrange); the *inhomogeneous* tangential BC `n×E = g` on N1E (the homogeneous PEC case IS wired);
> higher order (lowest
> RT₀ / N1E₀ only); other families (BDM, second-kind Nédélec, Bell); quad / non-triangular
> meshes; a parameter in a **boundary** term through the non-nodal path (rejected — the natural-BC load is
> assembled non-differentiably); and a constraint-consistent *algebraic* initial state at `t0` in the
> saddle-DAE transient (the differential field and all `t > 0` values are correct; only the reported `t0`
> algebraic value is). The **C¹ Argyris** element IS supported (below).

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

> **Hermite** is the first element with a per-cell **DOF-mixing** transform `M(cell)` (its global
> derivative DOFs are the physical gradient `∇u` at the vertices). It is **C⁰** (not C¹), so it is *not*
> a conforming biharmonic element — it de-risks the `M(cell)` / vertex-derivative-DOF machinery that the
> **C¹ Argyris** element (below) reuses. A **value-Dirichlet** `u(region) - g` pins boundary-vertex
> value DOFs (derivatives free); it composes with the steady, transient, and nonlinear `fem.solve()` paths
> (see `tests/test_fem_hermite.py` for Poisson / heat / reaction-diffusion).

> **Argyris** is the **C¹-conforming** quintic triangle (21 DOF: value, gradient and Hessian at each
> vertex; the normal derivative at each edge midpoint) — the element for **4th-order PDEs**. Across a shared
> edge both `u` and `∂u/∂n` are continuous, so `∫Δu·Δv` is now a *convergent* biharmonic discretisation
> (the conformity caveat above is lifted for this space). basix has no Argyris family, so the reference dual
> basis is built from the monomials and mapped to each physical cell by the affine-equivalence DOF-transform
> `M(cell)` (R.C. Kirby, *A general approach to transforming finite elements*, SMAI J. Comput. Math. 4,
> 2018; the original element is Argyris–Fried–Scharpf 1968). The **globally-oriented edge-normal DOF** is
> what makes it C¹ on an *unstructured* mesh — the reference-normal reduced-quintic (Bell) is not affine
> equivalent and fails there. Its essential BCs are the two plate traces — `u(region) - g` pins the
> **deflection** and `u.dn(region) - h` pins the **rotation** `∂u/∂n` (see *Plate boundary conditions* below);
> the boundary curvature `∂²u/∂n²` is always left **free** (a natural BC, as a physical plate requires). These
> are wired for **axis-aligned boundary edges** (where the `(n,t)` frame is the `(x,y)` frame, so each trace is
> a single DOF); a non-axis-aligned edge needs the general `(n,t)` rotation and is **rejected with a clear
> error** (use the Morley element there — any orientation). Composes with the steady,
> transient and nonlinear `fem.solve()` paths (see
> `tests/test_fem_argyris.py`: exact biharmonic recovery on an unstructured mesh, convergence, nonlinear
> `Δ²u + u³ = f`, the dissipative biharmonic heat flow, and a **vibrating clamped plate** `w_tt + Δ²w = 0`
> (the augmented `[w, v]` block, energy-conserving trapezoidal integration — a direct θ-solver for the stiff
> biharmonic). **Inverse** problems work too — a scalar coefficient (plate stiffness in `α·Δ²u = f`, a
> diffusivity in a transient flow) *or* a spatially-varying **P1 field** `k(x)` in a volume term is recovered
> by `crux.solve` (`tests/test_fem_inverse.py`), for both steady and transient forms.

> **Morley** is the **cheapest** biharmonic element (6 DOF: the value at the 3 vertices + the normal
> derivative at the 3 edge midpoints, quadratic). It is **non-conforming** — neither C⁰ nor C¹ — yet passes
> the patch test and converges (energy `O(h)`, L² `O(h²)`). It reuses the same `M(cell)` transform and
> globally-oriented edge-normal DOF as Argyris, but with a quadratic basis and ~3.5× fewer DOF it is far
> cheaper, so it **clears the Argyris construction memory ceiling** and scales to much finer meshes (e.g. a
> sharper phase-field crack). **Modelling subtlety:** because it is non-conforming, the biharmonic form must
> be the **full-Hessian inner product** `inner(hessian(u), hessian(v))` (`∫D²u:D²v`), *not* `∫Δu·Δv` — the
> Laplacian form is singular for Morley (functions like `xy` have `Δu = 0` but `D²u ≠ 0`, a spurious kernel).
Its essential BCs are the same two plate traces as Argyris — `u(region) - g` (deflection) and
> `u.dn(region) - h` (rotation) — but on **any** boundary orientation (Morley's DOFs are already the vertex
> value and the edge-normal derivative, so no `(n,t)` rotation is needed). See `tests/test_fem_morley.py`.
> Reference: L.S.D. Morley, *The triangular equilibrium element in the solution of plate bending problems*,
> Aeronautical Quarterly **19** (1968) 149–169.

> **Plate boundary conditions.** For a 4th-order (plate/biharmonic) field on the Argyris or Morley element the
> boundary trace has two essential parts you can pin independently — the **deflection** `u(region) - g` and the
> **rotation** `u.dn(region) - h` (`∂u/∂n = h`) — plus two conjugate **natural** parts (bending moment `M_n`,
> effective shear `V_n`) that emerge on any trace you *don't* pin. The classical BCs compose from these:
>
> | BC | Physics | How to write it (on a region) |
> |----|---------|-------------------------------|
> | **Clamped** | `w=0`, `∂w/∂n=0` | `u(reg)-g`, `u.dn(reg)-0` |
> | **Simply-supported** | `w=0`, `M_n=0` | `u(reg)-g` |
> | **Guided / sliding** | `∂w/∂n=0`, `V_n=0` | `u.dn(reg)-h` |
> | **Free** | `M_n=0`, `V_n=0` | *(write neither — natural)* |
>
> A free edge (no essential BC) gets the natural `M_n=V_n=0` from the physically-correct **ν-weighted plate
> energy** `(1-ν)·inner(hessian(u),hessian(v)) + ν·laplacian(u)·laplacian(v)`. Validated against Timoshenko's
> square-plate coefficients — clamped `w_max = 0.00126 qa⁴/D`, simply-supported `w_max = 0.00406 qa⁴/D`.
>
> **Prescribed edge moment (inhomogeneous natural BC).** A *nonzero* bending moment `M_n` on an edge is applied
> as the boundary load `M_n * phi.dn(region)` (the test function's normal derivative) — assembled as
> `∮_region M_n ∂φ/∂n ds` on the Argyris/Morley elements. It composes with the essential traces: a
> *moment-loaded simply-supported* edge is `u(reg)-0` **and** `M_n * phi.dn(reg)`. Validated by a manufactured
> simply-supported plate (`u* = x(1-x)y(1-y)`, `M_n = Δu*`): Argyris recovers `u*` to machine precision (it is a
> quartic in the P5 space), Morley converges `O(h²)`. The moment integral is built from each cell's own geometry
> (Jacobian, push-forward, outward normal), so it applies on **any edge orientation** — but note the *essential*
> Argyris pin is still axis-aligned-only, so on Argyris a moment is reachable only where that pin is; **Morley**
> carries a prescribed moment on any boundary (verified on a slanted diamond). *(The conjugate **shear** load —
> the effective Kirchhoff shear `V_n = Q_n + ∂M_{nt}/∂t`, which carries corner forces — is not yet wired.)*

```python
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")   # H(div) flux
p, q = d.fem_symbols(names=("p", "q"), space="P0")                     # piecewise-constant scalar
```

jNO assembles these with its own push-forward engine, but the weak form reads like
any other coupled problem.

**Vector operators** (on a bound vector view): `u.div(x, y)` is the divergence and `u.curl(x, y)` the
2-D scalar curl `∂uy/∂x − ∂ux/∂y`; in 3-D `u.curl(x, y, z)` returns the **curl vector** (used for the
N1E tet curl-curl form). After binding, the no-arg `u.div()` / `u.curl()` reuse the bound coordinates.
(`div` is equivalently `trace(grad(u, [x, y]))`.)

**Essential (edge-trace) BCs** — the outward normal is `d.variable(region, normals=True, split=True)`:

| Family | Trace | Term |
|--------|-------|------|
| RT  | normal flux `u·n = g` | `u(b)[0]*nx + u(b)[1]*ny - g` |
| N1E | tangential `u×n = g`  | `u(b)[0]*ny - u(b)[1]*nx - g` |

For the RT mixed-Poisson saddle, a Dirichlet condition on the scalar `p` is *natural* — add the weak
term `p_D * (v[0]*nx + v[1]*ny)`, no essential constraint on the flux. A BC may target a sub-region
(a `box` edge tag or any `d.tag(...)` boundary subset; sub-region normals are computed from the
geometry). All solver modes work — **steady-linear**, **steady-nonlinear** (Newton), and **transient**
(`M u̇ + A u = c`), including a mixed/saddle transient (a DAE with singular mass, e.g. transient Darcy).

Tutorials: `mixed_poisson_rt_2d.py` (H(div)), `maxwell_nedelec_2d.py` (H(curl): magnetostatics + eddy
current), and `maxwell_nedelec_3d.py` (the **3-D PEC cube cavity resonator** — recovers the analytic modes
`k²=π²(l²+m²+n²)`, spurious-free, converging to `2π²` from below). *Scope: lowest-order RT₀ / N1E₀ on 2-D
triangular meshes; N1E₀ also on 3-D **tetrahedral** meshes (H(curl) mass + curl-curl + PEC, for 3-D Maxwell
/ curl-curl and cavity eigenproblems).*

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

> **Term introspection (provisional).** `fem.term_kinds` returns a `list[TermKind]` — each
> additively-split PDE (volume) term classified by `support`, `time_order`, `trial_channel` /
> `test_channel` (spatial-gradient presence), and `linear`, with `is_local` flagging a spatially
> pointwise term (reaction/mass: no spatial gradient on trial or test) vs. a neighbour-coupling
> global one (diffusion/advection). This is the basis for operator-splitting routing; the API is
> provisional until that routing lands. A mass term `u.t·v` is `is_local` (its derivative is
> temporal, not spatial).

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
**trapezoidal rule** (θ=½, equivalent to Newmark average-acceleration — Newmark 1959, *"A Method of
Computation for Structural Dynamics"*, J. Eng. Mech. Div. ASCE 85(3), the constant-average-acceleration
case β=¼, γ=½) — backward Euler would spuriously damp an undamped wave. A second-order
problem needs **two** initial conditions: displacement `u(initial) - u0` and velocity
`u.t(initial) - v0` (bind the velocity IC with the `"initial"`-slice coordinates *and time*,
`u.bind(x=xi0, y=yi0, t=ti0).t`; a missing velocity IC defaults to zero).

```python
d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1, time=(0.0, 2.0, 200))
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
> rule lives inside `fem.solve()`. Unlike the first-order (parabolic) block above, **do not** hand-roll
> backward Euler `(M + dt·A) w = M·w` off `fem.M` / `fem.operator.A` on a second-order block: backward
> Euler spuriously **damps** the wave. If you integrate manually, use the trapezoidal step
> `(M + ½·dt·A) w_next = (M − ½·dt·A) w + dt·c`.

A **vector** field works too (`value_shape=(2,)`/`(3,)`) — that is elastodynamics,
`ρ u_tt = ∇·σ(u)` (see the vibrating-cantilever tutorial). *Scope: linear, single field (scalar or
vector), nodal Lagrange, 2D/3D, constant Dirichlet; nonlinear / multi-field / runtime-parameter /
time-varying-Dirichlet second-order forms are rejected (fail-loud) — write those as a first-order
system.*

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
(resolved through `crux`, see below). Marking it **`.freeze()`** declares it a *known* coefficient:
`jno.fem` evaluates its `.initialize` value at the quadrature points — exactly like `jno.fn` — so the
system assembles non-parametrically (`fem.A` / `fem.b`, no `crux`) **and works in every form**
(steady-linear, nonlinear, transient, coupled). The frozen value is a **scalar** (`.initialize(3.0)`)
or a **coordinate function** (`.initialize(lambda x, y: ...)`, scalar- or vector-valued); a raw
per-node array, a JAX initializer, or no value all fail loud (a known coefficient is a function/const,
not nodal data — for nodal data interpolate it into a function). Leave the parameter **un-frozen** to
make it an inverse unknown — the next section.

> `.freeze()` is equivalent to writing `jno.fn(...)` / the constant directly; it exists so one
> `jno.np.parameter` can be *trained* (un-frozen) or *fixed* (frozen) without rewriting the form. A
> vector-valued coefficient is best written **per component** with scalar functions (a single function
> returning a tuple hits a kernel limit shared with `jno.fn`).

---

## Per-region (sub-domain) integration

A weak term integrates over the **region of the coordinates it is written on** — exactly the rule that
already routes boundary terms. Bind the trial/test to `domain.variable("interior")` and the term covers
the whole domain; bind them to a **sub-region's** coordinates and the term integrates over that
sub-domain's cells only. No new function — name a region with `domain.tag(name, predicate)` (or use a
multi-part mesh's geometry parts) and ask for its coordinates with `domain.variable(name, split=True)`:

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

Multi-material conduction is then *one term per material* (each on its region's coordinates); a
data-fit / QoI confined to a region is `(uc - u_data) * vc`. A cell belongs to a region iff its
**centroid** does (classified once at assembly — exact when the mesh respects the region boundaries,
e.g. gmsh meshing each part separately; for an arbitrary predicate on a non-conforming mesh it is
centroid-accurate, O(h) at the interface). Region integration is a scalar mask on the integrand, so it
**composes with everything**: constant / `jno.fn` / `.freeze()` / trainable coefficients, and the
steady-linear, nonlinear, transient, coupled (multi-field), and 3-D forms. In particular a
`jno.np.parameter` that multiplies a sub-region term is recovered **per sub-domain** through `crux` —
fit a per-material property on its own region (see *Inverse problems*).

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

It is *general* — a value can be a scalar, a `jno.fn` field, or a trainable `jno.np.parameter`, so the
same primitive expresses conductivity, a source, a density, a reaction rate, an elastic modulus — and
trainable per-region values compose for free (`d.by_region({**k, "air": nu*0.026})`, calibrated through
`crux`). Each key must be a geometry part (`from_regions`) or a `domain.tag` predicate; `default` fills
cells in no listed region. It desugars to `sum_r RegionMask(r)·value_r` — exactly the per-region terms
above, so it inherits the centroid classification and composes with every solve form.

> Not yet wired: second-order-in-time (`u_tt`) sub-region terms — they fail loud rather than silently
> integrate over the whole domain. 3-D sub-regions are defined by a predicate `where(x, y, z)`
> (shapely polygons are planar).

---

## Enclosure radiation (nonlocal boundary flux)

Grey-body radiation between surfaces is **nonlinear** (`T⁴`) and **nonlocal** (every surface element
exchanges with every other via the view-factor matrix `F`), so it cannot be a local weak term. jNO
provides the *geometric* building block — the view matrix — and you write the radiosity **as math** in
`jno.np`; there is no `jno.radiation()` helper.

`domain.enclosure(tags)` discretises the radiating boundary surfaces into **elements** aligned to the
FEM mesh nodes and returns a handle:

```python
gap = d.enclosure(["inner_gap", "outer_gap"], axisymmetric=False)   # name the surfaces once
gap.check()                          # F-quality gate: closure (Σ_j F_ij→1) + reciprocity (A_i F_ij=A_j F_ji)
F   = gap.view_factor                # (m, m) element view factor — fully geometry-determined
eps = gap.emissivity({"inner_gap": 0.8, "outer_gap": 0.6})         # per-element ε from a {tag: ε} map
rho = 1.0 - eps
```

`F` is computed purely from geometry (occlusion + orientation; only the `i==i` self-pair is removed) by
**double-area Gauss quadrature** of the diffuse kernel — so a *concave* surface keeps its self-view (the
outer cylinder's `F₂₂ = 1 − r₁/r₂`). Tags only group elements (for per-surface emissivity); they never
block exchange. Use `axisymmetric=True` for a body of revolution (the `(r, z)` meridional mesh); its ring kernel applies
a near-field floor `r_min` (default: half the median element length) so near-coincident / on-axis pairs
stay physical (`F ≤ 1`) — override via `r_min=` if needed. By
default the boundary normals point *out of* the mesh — radiation across an un-meshed gap (a vacuum
between solid parts). For an **oven/furnace cavity** where the fluid inside is meshed and radiation
crosses that meshed interior, pass `inward=True` so the wall normals point into the cavity and the facing
walls see one another (see the *Oven* tutorial). For a meshed *medium* between solids, use `medium_tags`.

Write the **full grey-body radiosity** (reflections included) and couple it to the conduction FEM by
adding the net flux as a consistent surface load to the residual:

```python
SIGMA, KELVIN = 5.670374419e-8, 273.15

def q_rad(u):                        # net radiative flux per element:  q = σ·G·T⁴
    Ts = gap.field(u)                # nonlocal gather: per-element temperature from the solution
    J  = jno.np.linalg.solve(jno.np.eye(gap.size) - rho[:, None] * F, eps * SIGMA * (Ts + KELVIN)**4)
    return J - F @ J                 # (I − F)(I − diag(ρ)F)⁻¹ diag(ε) σ T⁴

# −k ∂T/∂n = q_rad  enters the residual as a consistent load:  A u = b − gap.load(q_rad(u))
A = fem.operator[0].todense()        # BCOO → dense via the jax path (.todense() is fast; np.asarray is NOT)
b = fem.operator[1]
u = newton(lambda u: A @ u - b + gap.load(q_rad(u)), jnp.linalg.solve(A, b))   # direct-solve Newton, below
```

`gap.field(u)` gathers the per-element temperature; `gap.load(q)` scatters a per-element flux back to the
FEM nodes as `∫_Γ q·v ds`. The radiosity `(I − ρF)⁻¹` solve is `jno.np` — it is **traced**, so a trainable
`jno.np.parameter` emissivity flows through it for inverse problems.

**Solver note (BYO, jax-native).** jNO imposes no solver. The Dirichlet conditions are penalty-enforced,
so the conduction `A` is ill-conditioned — a **direct** linear solve handles it (a *matrix-free iterative*
solver such as the built-in `newton_krylov` may stall). The whole coupled solve stays jax-native and
**differentiable** (so `jax.grad`/`crux` recover an emissivity *through* the radiation) with a short
direct-solve Newton wrapped in `jax.lax.custom_root`:

```python
def newton(residual, u0, steps=50, tol=1e-9):       # ~10 lines; no external solver
    f = lambda u: jnp.asarray(residual(u)).reshape(-1)
    def step(fn, x0):
        def body(s):  # Newton step with a DIRECT linear solve (dense Jacobian via autodiff)
            du = jnp.linalg.solve(jax.jacfwd(fn)(s[0]), -fn(s[0]))
            return s[0] + du, jnp.linalg.norm(du), s[2] + 1
        return jax.lax.while_loop(lambda s: (s[1] > tol) & (s[2] < steps), body, (x0, 1.0, 0))[0]
    tangent = lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(jnp.zeros_like(y)), y)
    return jax.lax.custom_root(f, jnp.asarray(u0).reshape(-1), step, tangent)   # implicit-diff
```

Validated on two concentric cylinders against the closed-form two-surface series
(`q = σ(T₁⁴−T₂⁴)/(1/ε₁ + (r₁/r₂)(1/ε₂−1))`) to <1%, including `jax.grad` of the surface temperature w.r.t.
emissivity matching finite differences (`tests/test_fem_enclosure_radiation.py`). The dense Jacobian is
fine for moderate meshes; for large problems, precondition a matrix-free Newton with the conduction solve.

### In-residual coupling (`jno.Coupling`) — implicit, trainable, transient

The bring-your-own-loop above is operator-splitting: you reach into `fem.operator` and march the radiation
yourself. To instead solve conduction **and** radiation as one implicit system, pass the nonlocal residual
**in the `jno.fem([...])` list**. A plain function `f(u) -> (n_dofs,)` there is taken as a nonlocal
*coupling* (weak/Dirichlet terms are trace nodes, never plain callables): `jno.fem` adds it to the assembled
residual `R(u) = R_local(u) + Σ_k coupling_k(u)`, promoting a linear form to a nonlinear one, and
`fem.solve()` drives the whole thing with the matrix-free, `custom_root`-differentiable `newton_krylov`:

```python
def radiation(u):                         # the same radiosity, now a residual contribution
    Ts = gap.field(u)
    J  = jnp.linalg.solve(jnp.eye(gap.size) - rho[:, None] * F, eps * SIGMA * (Ts + KELVIN)**4)
    return gap.load(J - F @ J)             # net flux scattered to nodes

fem  = jno.fem([conduction, radiation, u(xc, yc) - T_COOL])    # radiation is the bare function
Tsol = fem.solve(u0=T_guess)                                   # conduction + radiation, one implicit solve
```

(A jitted residual / callable *object* isn't a plain function — wrap it as `jno.Coupling(fn)`, which is also
how you reach the options below. A stiff/dense coupling may still need a tailored `fem.solve(solve_fn=…)`.)

- **Trainable coupling parameters.** A `jno.np.parameter` in a *weak* term is found by the trace walk, but a
  coupling is opaque — declare its parameters so they thread through the solve and `crux` recovers them:
  `jno.Coupling(fn, params=[eps])`, with the residual taking the `{name: value}` dict, `fn(u, p)`.
- **Multifield.** `jno.Coupling(fn, field_key=T_key)` acts on one field's DOF block (e.g. radiation on `T`
  in a heat+flow / thermo-mechanical solve); the residual sees and returns that field's sub-vector.
- **Transient.** The coupling enters each implicit step — a nonlinear time block gains the term, a linear one
  is promoted to a nonlinear (backward-Euler) block — so enclosure radiation over a heating cycle solves
  in-residual. (Not combined with periodic ties.)

All four (bare function, `params`, `field_key`, transient) are covered in `tests/test_fem_enclosure_radiation.py`.

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4–5 (view factors; the net-radiation /
radiosity method for diffuse-grey enclosures).

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
(`sparse_lu_solve`, JAX `spsolve` — robust on saddle-point systems), with a Jacobi-preconditioned
matrix-free BiCGStab as the iterative alternative; the nonlinear default is a matrix-free
Newton-Krylov, and the transient default backward-Euler over those. All are implicit-diff, so
`crux.solve` recovers parameters through them. Bring your own `solve_fn` for anything else.

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

Everything shipped is **pure JAX** — `jit`- and `vmap`-native, differentiable (the Krylov
wrappers sit on `lax.custom_linear_solve`, Newton on `lax.custom_root`) — and *reuses* existing
implementations (`jax.scipy.sparse.linalg`, `sparse_lu_solve`) rather than duplicating them.
Slot solvers receive the assembler's **BCOO** operator directly (no densification), and compose
with the periodic reduction and every parametric/inverse path unchanged. Pick by structure:

| structure | solver | notes |
|---|---|---|
| SPD (Poisson, elasticity, mass) | `cg` | cheapest per iteration |
| non-symmetric (advection, SUPG) | `bicgstab` / `gmres` | `bicgstab` == the historic default (with `jacobi()`) |
| **iterative preconditioner** (inner Krylov, block/Schur with inexact inner solves) | `fgmres` | flexible right preconditioning — Saad, *SIAM J. Sci. Stat. Comput.* 14(2), 1993, Alg. 2.2 |
| symmetric **indefinite** (Stokes/Biot saddle, biharmonic) | `minres` | monotone residual, `O(1)` memory — Paige & Saunders, *SIAM J. Numer. Anal.* 12(4), 1975 |
| SPD, batched/GPU-heavy | `chebyshev` | inner-product free (no reductions) — Golub & Varga 1961; Saad, *Iterative Methods*, 2003, §12.3 |
| indefinite, single solve | `lu` | sparse-direct; **no vmap rule** — use a Krylov solver inside batched solves |
| small systems / coarse blocks | `dense` | LAPACK, vmap-native |

**Preconditioner specs** (declarative — materialized against the assembled operator at solve
time; a preconditioner never changes the converged solution, only the speed, so specs need no
gradient path):

* `jno.precond.jacobi()` — diagonal.
* `jno.precond.chebyshev(degree=…)` — fixed-degree Chebyshev **polynomial** preconditioner
  (same references as the solver): matvecs and AXPYs only, the GPU-era substitute for
  Gauss-Seidel/ILU smoothing, and a fixed *linear* map so it legally preconditions `cg`/`minres`.
* `jno.precond.inner(solver)` — any `jno.solve` solver as the `M⁻¹` application (an inexact
  block/system solve). Iterative inner ⇒ flexible outer (`fgmres`).
* `jno.precond.form([...terms], inner=…)` — **preconditioners as weak forms**: assemble an
  auxiliary operator from ordinary traced terms and invert it as `M⁻¹`. Weighted mass matrices,
  local proxies of nonlocal (radiation) operators, shifted-Laplacian Helmholtz twins, low-order
  proxies — written in the same language as the PDE.
* `jno.precond.block_diag((field, spec), …)` / `jno.precond.triangular((field, spec), …)` —
  per-field composition over `fem.blocks` (fields are the trial symbols; `fem.block_index`
  resolves them, offsets-ordered). `triangular` is the standard saddle-point shape: last block
  solved first, substituted back through the assembled off-diagonal matvecs.
* `jno.precond.amg(cycles=…)` — **hybrid algebraic multigrid**: setup once on the host via the
  *optional* `pyamg` (smoothed aggregation — Vaněk, Mandel & Brezina, *Computing* 56, 1996;
  PyAMG — Bell et al., *JOSS* 8(87):5495, 2023), applied as a pure-JAX V-cycle with Chebyshev
  smoothing (Adams et al., *JCP* 188, 2003). The apply is `jit`/`vmap`-native (one frozen
  hierarchy legitimately serves a whole batch) and exactly linear, so it preconditions
  `cg`/`minres` too. Mesh-independent convergence ⇒ *the* choice for large elliptic blocks.
  Inside traced/parametric solves, pre-build eagerly: `spec = jno.precond.amg(); spec.build(fem.A)`.
  Without pyamg installed everything else works; using `amg` raises a clear install hint.

The flagship pattern — Taylor–Hood **Stokes** by FGMRES with an inexact velocity block solve and
the viscosity-weighted pressure-mass Schur approximation (Elman, Silvester & Wathen, *Finite
Elements and Fast Iterative Solvers*, 2nd ed., 2014, §9.2) — no densification anywhere:

```python
sol = fem.solve(
    linear  = jno.solve.fgmres(tol=1e-10, restart=40),
    precond = jno.precond.triangular(
        (u, jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=60))),   # Â⁻¹: inexact CG
        (p, jno.precond.form([(1.0/mu) * pp * qq], inner=jno.solve.dense())),  # Ŝ ≈ μ⁻¹ M_p
    ),
)
```

**Picard / lagged coefficients — `jno.lag`.** When a solution-dependent coefficient's Newton
tangent destroys the linearized system's structure (the classic case: a shear-thinning viscosity
`μ_eff(u)` in non-Newtonian/rigid-plastic Stokes flow, whose full-Newton velocity block is
strongly nonsymmetric and defeats AMG/block preconditioners), freeze it with `jno.lag(...)` and
drive with `jno.solve.picard()`:

```python
fem = jno.fem([2 * jno.lag(mu_eff) * inner(eps(ui), eps(vi)) - pi * div(vi), ...])
sol = fem.solve(nonlinear=jno.solve.picard(damping=0.7), linear=jno.solve.fgmres(),
                precond=jno.precond.triangular(...))
```

`lag` is `stop_gradient` on the traced expression, so the residual's linearization *is* the
Picard operator — each outer step re-solves the lagged system (linear convergence, but every
inner system keeps its symmetry/definiteness); the converged solution is identical to full
Newton's. Without any `lag` marker, `picard(damping=…)` is exactly damped Newton
(`jno.solve.newton(damping=…)` spells the same thing). Caveat for inverse problems: implicit
differentiation then also uses the lagged Jacobian (the standard "Picard adjoint" approximation)
— drop `lag` when exact parameter gradients matter more than per-step solvability.

**User extension** is duck-typed — a linear solver is any
`fn(A, b, *, M=None, x0=None) -> x` with `A` a `jno.solve.LinearOperator` (`.mv`, `.T`,
`.diag()`, `.bcoo`, `.dense()`); a preconditioner is any `ctx -> (v -> M⁻¹v)`:

```python
def my_precond(ctx):                      # ctx.A, ctx.diag(), ctx.fem
    inv = 1.0 / ctx.diag()
    return lambda v: inv * v
u = fem.solve(linear=jno.solve.cg(), precond=my_precond)
```

If your callable is pure JAX it inherits `jit`/`vmap`/AD automatically. On the matrix-free
**nonlinear** path the `precond` spec is materialized *per Newton/Picard linearization* against
the JVP operator — so `form`, `inner(...)`, `chebyshev`, a pre-built `amg`, and their
`block_diag`/`triangular` compositions all work (this is the nonlinear-saddle production
pattern: `nonlinear=picard() + linear=fgmres() + precond=triangular(...)`); only specs that
need the assembled matrix (`jacobi`, an unbuilt `amg`) raise.

**Transient problems.** The slots configure the *per-step* solves of the default theta-method
integrator (the bring-your-own `(block, args, save_ts)` contract via `solve_fn=` is unchanged):
`linear`/`precond` see the step operator `M + θ·dt·A` — when the operator is time-independent
the step matrix is formed **once** and the preconditioner materialized **once before the time
loop** (an AMG hierarchy or auxiliary `form` operator is then reused by every step; `jacobi`
uses the exact step diagonal either way) — and `nonlinear` drives each implicit step of a
nonlinear block (`picard` + `jno.lag` per step included). Second-order-in-time (`u_tt`) flows
through the same augmented block unchanged. Each step warm-starts from the previous state
(so `x0=` is rejected — the initial state is the ICs' job); `lu()` inside the time loop
re-factorizes per step (JAX's `spsolve` has no factorization cache).

Not yet supported (clear errors, see `plans/fem-solver-api.md`): slots on **complex** /
complex-transient problems (`x0` on complex included), and slots combined with `adapt=`.

### Field parameters `k(x)` + regularization

`jno.np.parameter(phi)` is a **nodal field** on the trial space — a trainable value per node.
Field inversion is ill-posed, so add a smoothness/structure prior with `k.regularize(...)`
(`"h1seminorm"`, `"l2"`/`"tikhonov"`, `"tv"`, `"nonneg"`, `"bounded"`):

```python
k = jno.np.parameter(phi, name="k")                       # P1 field, one DOF per node
crux = jno.core([(fem.solve() - u_obs).mse, 1e-3 * k.regularize("h1seminorm").mean], domain=obs)
```

### Neural coefficients — `jno.nn.wrap(net)` inside the weak form

A network called inside a weak form is a trainable **coefficient** on an assembled FE system —
mesh-independent (remeshing never touches the weights), smooth by architecture, and trained
through the same differentiable `fem.solve()` as any parameter:

```python
net = jno.nn.wrap(foundax.mlp(2, hidden_dims=16, num_layers=2,
                              activation=jax.nn.tanh, key=key))
net.dtype(jnp.float64)                                       # match the f64 assembly
net.optimizer(optax.adam(1e-2))

# k(x) = 1 + net(x, y): the offset keeps A(θ) nonsingular at the (near-zero) net init
fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs).solve(600)   # trains the weights
```

The kernel re-evaluates the network at the quadrature points during every re-assembly, so the
coefficient is *not* interpolated on the mesh (unlike the P1 nodal field above) — it composes
with scalar/nodal parameters in one weak form, with per-region masks, vector trials, and surface
(Robin/Neumann) terms. `net.freeze()` makes it a **known** network coefficient (evaluated from
its stored weights; the system stays non-parametric). The role is decided by the constraints: a
weak form whose trial is a *real* FE symbol makes the network a coefficient; a network written
*in place of* the trial is a VPINN (see the VPINN section).

This is the unsupervised coefficient-recovery setting of NN-EUCLID (M. Flaschel, S. Kumar,
L. De Lorenzis, *NN-EUCLID: Deep-learning hyperelasticity without stress data*, J. Mech. Phys.
Solids 165 (2022) 105076, §2.2–2.3) and Tartakovsky et al., *Learning Parameters and Constitutive
Relationships with Physics-Informed Deep Neural Networks* (Water Resour. Res. 56, 2020, §2).

**Learned constitutive laws — `net(u)`, `net(∇u)`.** A network may also take the *solution* (or
its derivatives) as input — then it is a material law, not a spatial map, and the form becomes
nonlinear in `u` (routed to the matrix-free Newton path automatically; the net's `u`-dependence
enters the element Jacobians through per-element forward AD). This is the NN-EUCLID setting:
observe `u`, learn the hidden law unsupervised through the residual:

```python
net = jno.nn.wrap(foundax.mlp(1, hidden_dims=16, num_layers=2,
                              activation=jax.nn.tanh, key=key)).dtype(jnp.float64)
# hidden truth k(u) = 1 + 0.5 u²; learn it from a single observed field
fem = jno.fem([(1.0 + net(ui)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
crux = jno.core([(fem.solve() - u_obs).mse], domain=obs).solve(600)
```

`net(ui.x, ui.y)` (a `k(∇u)` p-Laplacian-type law) works the same way, and solution- and
coordinate-inputs can be mixed (`net(xi, yi, ui)`). Classification is automatic: a net whose
arguments carry the unknown makes the form nonlinear — including a bare reaction term
`net(u)*v` — while `net(x, y)` keeps the system linear(-parametric).

**Transient forms** work the same way — the per-step operator (or the per-step Newton residual,
for `net(u)`) re-evaluates the network, so a diffusivity or constitutive law is recovered from a
`u(t)` trajectory exactly like the scalar/nodal transient inverse below. One exclusion: a
*trainable* net on the mass (`u_t`) term is rejected (the mass matrix assembles once — the net
would silently freeze; `.freeze()` it or keep it on spatial terms).

A real coordinate-input net also composes with **complex** steady forms (a Helmholtz coefficient
recovered from complex full-field data): the Re/Im legs assemble as parametric systems and the
real-equivalent block solve stays differentiable in the weights. And a net coefficient works in a
**coupled (multi-field)** form — it is evaluated at the shared quadrature points and a trial-input
`net(u_i)` resolves its own field, so no per-field bookkeeping is needed.

**Non-nodal elements.** A net coefficient — `k(x)` or a constitutive `k(u)` — also works on the
scalar C¹ families (`space="Argyris"`/`"Morley"`/`"Hermite"`), e.g. a spatially varying or
solution-dependent stiffness on a biharmonic plate: the network is evaluated at the quadrature
points independently of the C¹ trial's DOF layout (the same property the P1 field parameter uses).
On the vector edge families (`"RT"`/`"N1E"`) a *scalar coordinate* `net(x)` coefficient works too (a
spatially-varying permeability/permittivity multiplying a vector term); only a solution-dependent
`net(u)` there is unsupported, since a vector-valued trial input to the network is undefined. The
non-nodal path assembles a *dense* operator, so a parametric solve wants an explicit dense
`solve_fn` — `fem.solve(lambda A, b: jnp.linalg.solve(A, b))`.

Current scope: steady/transient/steady-complex on the native 2D/3D Lagrange assembler (single or
coupled multi-field), and steady scalar C¹ non-nodal (Argyris/Morley/Hermite). Not yet supported
(each fails loud): networks inside Dirichlet/IC values, a *trainable* net on the mass (`u_t`) term
(a frozen/known one is fine — the mass is assembled once), k(u) in complex forms, the complex
transient, time-varying Dirichlet `g(x,t)` with a trainable net, transient non-nodal,
solution-dependent `net(u)` on the vector edge families (RT/Nédélec), and 1D domains (the 1D
assembler has no runtime-parameter path at all).

### Transient inverse

For a transient form, `fem.solve()` returns the **trajectory** `u(save_ts)` (default: backward
Euler over the assembled `dt`, sampled at the domain time grid), differentiable in the
parameters — so a rate constant is recovered from a time series:

```python
fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
crux = jno.core([(fem.solve() - u_traj).mse], domain=obs).solve(200)   # recovers alpha
```

`fem.solve(my_integrator, save_ts=...)` swaps the integrator: `my_integrator(block, args,
save_ts) -> trajectory`. Build your own (e.g. diffrax) from the block's `block.M` / `block.A` /
`block.state0` — form `u_dot = M⁻¹(c − A u)`; the implicit backward-Euler default is preferred for
Dirichlet problems.

---

## Vector, coupled, and higher-order problems

* **Vector / elasticity** — `u, phi = d.fem_symbols(value_shape=(2,))`; use `vi.component(i)`,
  `jno.np.symgrad`, `jno.np.trace`, and `jno.np.inner(..., n_contract=2)` to write the
  elasticity bilinear form `λ (∇·u)(∇·φ) + 2μ ε(u):ε(φ)`.
* **Coupled / mixed (Stokes)** — call `fem_symbols(...)` once per field and add one momentum and
  one continuity term; an inf-sup-stable Taylor–Hood pair is `order=2` velocity + `order=1`
  pressure. Pure-Dirichlet velocity leaves the pressure defined only up to a constant; gauge-fix
  that null space by adding `p.pin()` to the constraint list (it pins one arbitrary DOF — no
  coordinates needed; pass `p.pin(value)` to set the gauge).
* **1D and 3D** — a 1D interval or a 3D `cube`/extruded `gmsh` volume use the identical API with
  one fewer / one more coordinate (`ui.z`, `u(xb, yb, zb) - g`, `element_type="TET4"`).
* **Higher-order Lagrange** — `order=k` gives degree-`k` elements (P2 quadratic, P3 cubic, P4, … on
  triangles and tets); the assembly mesh places the element's basix interpolation points on each cell
  (deduplicated by coordinate, so shared edges/faces stay conforming). Read the solution at `fem.points`.
  The geometry stays affine-P1 (straight-sided), so on a *curved* boundary the geometric error caps the
  observed order regardless of `k` — measure high-order convergence on straight-sided/polygonal domains.

---

## Worked examples

The [FEM tutorials](tutorials/08-fem-and-varpinns/poisson-2d-fem.md) cover every pattern above:
Poisson, mixed Dirichlet/Robin reaction–diffusion, a nonlinear Allen–Cahn interface, a 3-D
Helmholtz solve on an extruded domain, mixed-BC Helmholtz, a linear-elastic cantilever beam,
Poiseuille channel flow (Stokes), transient heat, and two inverse problems (a hidden
diffusivity field and a transient rate). Two **second-order-in-time** examples show the wave path:
a **vibrating membrane** (`wave_membrane_2d.py`, verified against the analytic standing wave) and a
**vibrating cantilever** (`elastodynamics_cantilever_2d.py`, vector elastodynamics verified by energy
conservation). The non-nodal families add an **H(div) mixed Poisson** (Raviart–Thomas + P0) and an
**H(curl) Maxwell / eddy-current** example (Nédélec edge elements, `maxwell_nedelec_2d.py`); a
**variational PINN** writes a neural-network trial straight into the same `jno.fem` weak form.

---

## Known limitations

The FEM / weak-form path is stable for the cases the tutorials cover, but the
lowering has a few boundaries worth knowing. They apply only when you
**assemble a weak form** (`target="fem_system"` / `"fem_residual"`) or solve a
**transient problem through the time route** — the residual-PINN path is
unaffected. Each boundary is an explicit, fail-loud `NotImplementedError`, never a
silently wrong result.

- **Transient mass terms must be parameter-free.** In a time-dependent solve the
  mass term (`u_t * phi`) may not carry a trainable/runtime parameter. Keep it
  constant and place affine trainable parameters in the operator/residual instead
  — e.g. a diffusivity `nu` on the stiffness term, not on the time derivative.

- **Second-order in time is scoped.** A second-order-in-time weak form (`u_tt`, e.g.
  the wave equation `u_tt = c² Δu`, or elastodynamics `ρ u_tt = ∇·σ`) **is** assembled —
  `jno.fem` auto-reduces it to a first-order augmented `(u, v=u_t)` block, integrated by the
  energy-conserving trapezoidal rule (see *Second order in time* above). It is scoped to
  **linear, single field (scalar or vector), nodal Lagrange, 2D/3D, constant Dirichlet**;
  a nonlinear, multi-field, runtime-parameter, or time-varying-Dirichlet second-order form
  is rejected (fail-loud) — rewrite those as a first-order system. The Diffrax /
  residual-PINN strong-form adapters remain first-order (manual reduction).

- **No runtime Dirichlet parameters.** A trainable parameter may sit in the
  operator (stiffness) but not in an essential/Dirichlet boundary *value*: a
  runtime contribution that lifts Dirichlet data (a non-zero right-hand side) is
  rejected. Operator-coefficient inverse problems (e.g. recovering `nu`) are fine.

- **Affine parameter lowering expects a single, direct factor.** For trainable FEM
  coefficients, the affine fast-path recovers a parameter that is a *direct* scalar
  factor of a weak-form term (`nu * grad(u) · grad(phi)`). One trainable scalar per
  additive term — not nested inside another parameter or buried in a nonlinear
  expression — is the well-supported shape.

- **Enclosure radiation is a composition, not an auto-detected term.** `domain.enclosure`
  supplies the view matrix + gather/scatter; you write the radiosity in `jno.np` and couple it
  with your own solver (`A u = b − gap.load(q_rad(u))`). It is **2D / axisymmetric** (3-D view
  factors are future work), and because Dirichlet is penalty-enforced it needs a **direct** linear
  solve (the matrix-free `newton_krylov` may stall) — a short jax-native direct-solve Newton does it,
  differentiably (see *Enclosure radiation* above). Auto-detecting a radiosity term inside the
  `jno.fem([...])` list (so `fem.solve()` handles it) is not wired yet.

Hitting one of these is a signal to reformulate (move the parameter, reduce the
time order) rather than a bug — the error message names the offending term.

