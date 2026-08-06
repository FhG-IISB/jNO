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
  A **1D line domain** takes a vector unknown too (`value_shape=(n,)`), so a 1D *system* — a
  two-species model, a Timoshenko pair, a bar with several dofs per node — is one field with
  node-major dofs and per-component essential conditions (`u(region)[i] - g`).
  On a 1D line domain orders 1 (LINE2) and 2 (LINE3) are available — P2 adds a dof per element
  midpoint, so read `fem.points` for the coordinates the solution lives on. Measured on
  `-u'' + u = f`: P1 converges at O(h²) nodally and P2 at O(h⁴), which at an equal 41 dofs is
  4.7e-5 against 2.4e-7. A **coupled** 1D system carries a per-field order, so a mixed-order pair
  (the 1D Taylor-Hood shape) assembles — the blocks are then unequal and the coupling blocks
  rectangular, so read `fem.field_points` for each field's dof coordinates. Orders above 2 are not
  wired (clear error). In a coupled *transient* system a field may be **algebraic** (no `u_t`): its
  mass rows are zero, so the block is a DAE and the implicit step solves `A p = c` on those rows —
  which is how a constraint/closure field (a pressure, a saturation, an equilibrium concentration)
  is written.
  A `jno.np.parameter` coefficient (scalar or nodal field) also works on a **steady linear** 1D form, so
  a 1D differentiable inverse problem runs through `crux.solve` — as does a **neural** (`jno.nn.wrap`)
  coefficient, so a learned `k(x)` can be trained from 1D data. Transient too — recovering a diffusivity
  from a 1D time series works — except that the transient **mass** must be parameter-free (it is
  assembled once, so a parameter there would be silently frozen; it fails loud). *Nonlinear* forms are
  parametric too — Newton runs on `R(·, θ)` and implicit differentiation gives `∂u/∂θ`. A **coupled**
  1D system is parametric too (steady, linear and nonlinear) — the block element kernels publish the
  same `volume_vars` / neural-table keys the single-field ones do, so the shared evaluator reads them
  regardless of field layout. Not wired in 1D: a parameter on a coupled *transient* block, which is
  assembled once and would freeze the parameter at its placeholder (it fails loud). A **non-nodal** family (`"RT"`, `"N1curl"`, `"Argyris"`,
  `"Morley"`, `"Hermite"`) is defined on triangles/tets and has no 1D counterpart — asking for one on a
  line raises a clear error.
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
> 2-D-triangle only and raise.
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

`jno.solve.relocate()` descends the FE energy through the differentiable solve with a **backtracking
`det J` line search** — so the fixed node set concentrates at solution features and the mesh never tangles
(the validity constraint lives in the step control; a stock optimiser or an energy barrier alone cannot
guarantee it on a stiff problem — see `run_adaptive_relocate`). It mutates the domain to the relocated mesh,
returns the solution there, and **raises** if no coordinate was tagged. Works across **linear, nonlinear
(Newton), transient (relocates for the whole trajectory via a time-averaged objective), periodic, and
complex** problems, scalar or vector — the energy sums over every solution block, so a complex field's real
and imaginary parts both contribute. Only complex-*transient* is not wired yet.

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
| cuSolver refuses it, or is slow | `lu(host=True)` | factors on the HOST (SuperLU) and drives it from the device; same answer and same gradients (wrapped in `custom_linear_solve`, transpose via SuperLU `trans="T"`). Measured **faster** where cuSolver also works — Stokes 21,839 DOFs 0.27 s vs 1.67 s, H(curl) 17,072 complex DOFs 13.3 s vs 36.4 s — and it runs meshes cuSolver rejects (Stokes 26,908, H(curl) 26,154, both of which fail on GPU). Affordable because a direct solve factorises **once**: the operator crosses PCIe once, not per iteration. Faster in all 12 points measured here (0.15–0.81× of cuSolver), but that is **hardware-specific** — this card's FP64 is 1/64-rate (~0.3 TFLOPS) against ~1.1 TFLOPS on 20 CPU cores, and a direct factorisation is FLOP-heavy; on a full-rate-FP64 GPU the ranking may invert |
| small systems / coarse blocks | `dense` | LAPACK, vmap-native |

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

**A direct `linear=` slot selects it.** `lu`, `dense` and `amg` all need an assembled matrix, so pairing one
with the *matrix-free* Newton has nothing to factorize. `fem.solve(linear=jno.solve.lu(host=True))` on a
nonlinear or transient problem therefore routes to the direct Newton, and that slot is the solver that runs on
the assembled tangent (and on `Jᵀ` in the adjoint); `precond=` materializes against the same assembled
operator. Which factorization you pick is not cosmetic here: on a 26-step Rayleigh–Bénard march (three fields,
saddle, nonlinear) the default matrix-free Jacobi-BiCGStab takes 20.1 s, `linear=jno.solve.lu()` 7.6 s and
`linear=jno.solve.lu(host=True)` **3.1 s**, all to the same 2.8e-07 per-step Newton residual. An *explicit*
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

> **`lu(host=True)` factorizes a constant step operator once, not once per step.** The step matrix is
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

**Complex problems** are assembled as one real `2n` system over the stacked `[Re; Im]` state — the
real-equivalent block `[[A_r, -A_i], [A_i, A_r]]` — at assembly rather than at solve time. A complex
transient is therefore an ordinary transient block (the slots configure its per-step solve as above, and
`theta` / `adaptive` / `exponential` all apply), and a complex steady problem is an ordinary linear
system: the `linear` / `precond` slots and `x0=` work on it, with a **complex** `x0` mapped into the
block layout for you. Its default solver stays **sparse-direct**, not the matrix-free BiCGStab real
elliptic systems get — the real-equivalent block is indefinite for Helmholtz/PML, where Jacobi-BiCGStab
does not converge.

Two exceptions keep a dedicated path. A **complex-native** preconditioner (`ams`) solves `A_r + i·A_i`
directly rather than the block, so the Re/Im legs are retained for it. And a **Bloch** (quasi-periodic)
tie has a *complex* prolongation `P`, which does not split into two real legs — that case still solves
through its own complex block routine, and `x0=` on it is rejected.

Not yet supported (clear errors): `adapt=` on a complex transient (the cross-remesh state transfer is
not complex-aware yet).

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
fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
crux = jno.core([(fem.solve() - u_traj).mse], domain=obs).solve(200)   # recovers alpha
```

`fem.solve(my_integrator, save_ts=...)` swaps the integrator: `my_integrator(block, args, save_ts) ->
trajectory`. Build your own (e.g. diffrax) from the block's `block.M` / `block.A` / `block.state0` — form
`u_dot = M⁻¹(c − A u)`; the implicit backward-Euler default is preferred for Dirichlet problems.

---

## Vector, coupled, and higher-order problems

* **Vector / elasticity** — `u, phi = d.fem_symbols(value_shape=(2,))`; use `vi.component(i)`,
  `jno.np.symgrad`, `jno.np.trace`, and `jno.np.inner(..., n_contract=2)` to write the
  elasticity bilinear form `λ (∇·u)(∇·φ) + 2μ ε(u):ε(φ)`.
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

---

## Worked examples

The [FEM tutorials](tutorials/08-fem-and-varpinns/poisson-2d-fem.md) cover every pattern above:
Poisson, mixed Dirichlet/Robin reaction–diffusion, a nonlinear Allen–Cahn interface, mixed-BC
Helmholtz, a linear-elastic cantilever, Stokes channel flow, transient heat, two inverse problems (a
hidden diffusivity field and a transient rate), two **second-order-in-time** wave examples (a vibrating
membrane and a vector-elastodynamics cantilever), the non-nodal **H(div) mixed Poisson** and **H(curl)
Maxwell / eddy-current** examples, and a **variational PINN** (a neural trial in the same weak form).

---

## Known limitations

Each boundary below is an explicit, fail-loud `NotImplementedError` (never a silently wrong result),
and applies only when you **assemble a weak form** or solve a **transient problem through the time
route** — the residual-PINN path is unaffected. Full detail is inline in the sections above.

- **Transient mass terms must be parameter-free** — put affine trainable parameters on the stiffness /
  residual, not on `u_t * phi`.
- **Second-order in time (`u_tt`) is scoped** to nodal Lagrange, 1D/2D/3D, scalar or vector. A
  **nonlinear spatial** operator (sine-Gordon, cubic Klein–Gordon, large-deformation elastodynamics)
  *is* supported — Newton on the augmented `[u; v]` block — but the **temporal** side must stay linear:
  a state-dependent mass or damping `c(u)·u_tt` is refused, since `M2`/`C` are extracted by
  differentiating at `u=0` and would otherwise be frozen there. Coupled multi-field is 2D/3D and
  **all**-second-order only; time-varying Dirichlet is refused on nonlinear forms. A coupled 1D
  system carries `u_tt` on the same terms as 2D/3D: the augmented state is `[u_all; v_all]`, so
  `fem.offsets` lists the displacement blocks then the velocity blocks.
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
  today; the march assembles on the real, steady, single-field native-Lagrange path only (not
  transient / complex / multifield / non-nodal / periodic — each rejected with a clear error). The
  internal-state readout runs on every cell (sub-region-restricted plasticity is not wired). Kinematic /
  nonlinear (Voce) hardening and contact are separate formulas / machinery, not built.

Hitting one of these is a signal to reformulate (move the parameter, reduce the time order) rather than
a bug — the error message names the offending term.
