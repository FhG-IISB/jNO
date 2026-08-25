# Finite Element Method

jNO assembles and solves finite-element problems through a **single entry point**, `jno.fem([...])`.
You write the weak form as a plain Python list of **residual terms** — volume physics, natural
boundary terms, and essential boundary conditions, all in the same list — and `jno.fem` returns a
`FEM` object carrying the assembled operators.

The same traced language powers the steady solve, the transient time-stepper, and the
**differentiable** `fem.solve()` used for inverse problems. For the symbolic primitives you can write
into a term — fields, derivatives, conditionals, non-local integrals, geometry symbols, the `jno.fn`
escape hatch — see the [weak-form vocabulary](../weak_form_vocabulary.md).

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

??? note "Reading the assembled operators directly"
    The flat accessors — `fem.A`, `fem.b`, `fem.M`, `fem.state0`, and `fem.residual(u[, t])` /
    `fem.jacobian(u[, t])` — return ready-to-use **dense** matrices and **flat** vectors for
    hand-rolled experiments (e.g. `u_h = jnp.linalg.solve(fem.A, fem.b)`). `fem.operator` still
    exposes the raw sparse (`BCOO`) operator that `fem.solve()` and the solver slots work on directly.

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
* **Quadrature coordinates** — `d.variable("interior", split=True)` returns the volume
  coordinates; `d.variable("<edge>", split=True)` returns a boundary edge's coordinates. A
  `Shape.rect` auto-tags `"left"`, `"right"`, `"bottom"`, `"top"` (and `"front"`/`"back"` for a box);
  `"boundary"` is the whole boundary and `"initial"` the `t = t0` slice. To define a custom region
  and fetch its coordinates in one call, pass a predicate: `d.variable("port", where=lambda x, y: x < 1e-6)`
  tags `"port"` (exactly as `d.tag` would) and returns its split coordinates.
* **Bound views** — `ui = u.bind(x=xi, y=yi, t=ti)` ties a symbol to a set of coordinates.
  The value is `ui`; spatial derivatives are `ui.x`, `ui.y`, `ui.z`; the time derivative is `ui.t`.

!!! tip "One-dimensional problems have their own page"
    Systems on a line, arbitrary order and vertex superconvergence, coupled and mixed-order fields,
    the C¹ beam element, complex forms, variational PINNs and adaptivity — all of it is on
    **[One-dimensional problems](one-dimensional.md)**.

### Second derivatives — 4th-order weak forms

`jno.np.laplacian(ui, [xi, yi])` (the Laplacian `Δu`)
and `jno.np.hessian(ui, [xi, yi])` (the full `D²u`) assemble against the element's second shape-function
derivatives, so a biharmonic / plate / Cahn–Hilliard form is written directly, e.g.
`jno.np.laplacian(ui, [xi, yi]) * jno.np.laplacian(vi, [xi, yi])` for `∫Δu·Δv`. Needs **`order ≥ 2`**
(a P1 Hessian is identically zero), scalar Lagrange fields only.
!!! warning "Conformity caveat"
    Standard Lagrange is **C⁰**, so `∫Δu·Δv` over P2 is *non-conforming* and does **not** give a
    convergent biharmonic discretisation. For a convergent solve use a purpose-built biharmonic
    element — the **C¹ Argyris** element (`space="Argyris"`) or the cheaper **non-conforming Morley**
    element (`space="Morley"`, full-Hessian form `∫D²u:D²v`) — or the mixed (Ciarlet–Raviart) method
    (two coupled C⁰ fields with `w = Δu`; see `tests/test_fem_hessian.py`).

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

??? note "Operator storage — why `nse` is smaller than you expect"
    **Operator storage.** The assembled BCOO stores each `(row, col)` pair **once**. The assemblers
    emit one triplet block per additive weak-form term and every interior DOF pair receives a
    contribution from each element sharing it (~20 tets for P1), so the raw triplets are ~10–20×
    redundant; they are summed once at assembly instead of lazily on every matvec. Measured on a 3-D
    transient heat block: mass 18240 → 753 stored triplets, operator 54992 → 1025, step operator
    `M + dt·A` 73232 → 1025 (1.12 → 0.02 MiB); on a 3-D nonlinear Jacobian, 36752 → 3841 with a 6.6×
    faster matvec. Two consequences if you read the triplets yourself: `nse` is the true nonzero count,
    and **numerically-zero entries are dropped** — symmetric Dirichlet elimination leaves whole rows of
    them behind, and on a Dirichlet-heavy operator that removal is the larger of the two effects. Code
    that needs the *structural* pattern (an entry that is zero at this parameter value but nonzero at
    another) must not read it off the assembled operator.

    **Element-loop chunking — `jno.fem(chunk=)`.** A single `vmap` over every cell materialises the
    whole batched intermediate at once, and on a 3-D mesh that intermediate — not the assembled
    operator — is what sets the memory ceiling. The element loop is therefore chunked: measured on a 3-D
    nonlinear solve, peak memory fell from 2324 MiB to 509 MiB at 52k DOFs (and 1378 → 381 at 31k), for
    roughly 15% more assembly time. Reverse mode keeps the win — `scan` stores per-iteration residuals,
    but the chunked total is still below the unchunked one (183.5 vs 194.4 MiB).

    The default needs no tuning: a chunk may use **~0.15% of device memory**, taken from
    `memory_stats()["bytes_limit"]`, so the same problem that must be split on an 8 GB card runs
    unsplit — and therefore at full speed — on an 80 GB one. A cell floor (8192) keeps large per-cell
    blocks (P2 tets at 8 KB/cell, vector P1 at 13.8 KB/cell) from collapsing below GPU saturation, where
    the measured penalty is a ~2× slowdown; that floor is the one number JAX gives no way to derive, as
    it exposes device memory but not the SM count.

    `chunk=False` restores the single `vmap`, and a positive int pins cells per chunk — as an upper
    bound: whatever size is chosen (default or explicit) is shrunk to the smallest one giving the *same
    number of chunks*, so the chunks come out even where they can. That is a compile-time concern, not a
    memory one: `lax.map` emits the element kernel **twice** when the chunk does not divide the cell
    count — once as the scan body, once unrolled for the tail — and on a problem the size of Taylor–Hood
    Stokes that duplicate doubles the build. Evening the chunks costs nothing in either direction: the
    chunk *count* is unchanged (no extra scan step) and each chunk is no larger than requested (peak
    memory only falls). It cannot always divide — 100 items in 3 chunks is 34+34+32 — and then it is
    simply a no-op.

    `chunk=` lives on `jno.fem` rather than `fem.solve` because the steady-linear operator is assembled
    *here*, before any solve.

    Coverage, since it is not uniform. The **native** assembler chunks its residual and jacobian, volume
    and surface loops. The **non-nodal** assembler chunks both, for every family — including the C⁰/C¹
    vertex families (see below). The **nonlinear tangent** is per-element too: the assembler takes the
    current iterate, so `J(u_k)` assembles element-by-element like the linear operator. One path still
    takes a global dense `jacfwd` and is unchunked, for **every** non-nodal family rather than any
    particular one: the **second-order-in-time** block, which builds a dense `2n × 2n` system by
    construction. The
    **1-D** assembler is not chunked at all, and an explicit `chunk=` there raises rather than being
    silently ignored.

    **Term introspection (provisional).** `fem.term_kinds` returns a `list[TermKind]` — each
    additively-split PDE (volume) term classified by `support`, `time_order`, `trial_channel` /
    `test_channel`, and `linear`, with `is_local` flagging a spatially pointwise term (reaction/mass) vs. a
    neighbour-coupling global one (diffusion/advection). This is the basis for operator-splitting routing;
    the API is provisional until that routing lands.

??? note "Element-loop chunking — `jno.fem(chunk=)`"
    **Element-loop chunking — `jno.fem(chunk=)`.** A single `vmap` over every cell materialises the
    whole batched intermediate at once, and on a 3-D mesh that intermediate — not the assembled
    operator — is what sets the memory ceiling. The element loop is therefore chunked: measured on a 3-D
    nonlinear solve, peak memory fell from 2324 MiB to 509 MiB at 52k DOFs (and 1378 → 381 at 31k), for
    roughly 15% more assembly time. Reverse mode keeps the win — `scan` stores per-iteration residuals,
    but the chunked total is still below the unchunked one (183.5 vs 194.4 MiB).

    The default needs no tuning: a chunk may use **~0.15% of device memory**, taken from
    `memory_stats()["bytes_limit"]`, so the same problem that must be split on an 8 GB card runs
    unsplit — and therefore at full speed — on an 80 GB one. A cell floor (8192) keeps large per-cell
    blocks (P2 tets at 8 KB/cell, vector P1 at 13.8 KB/cell) from collapsing below GPU saturation, where
    the measured penalty is a ~2× slowdown; that floor is the one number JAX gives no way to derive, as
    it exposes device memory but not the SM count.

    `chunk=False` restores the single `vmap`, and a positive int pins cells per chunk — as an upper
    bound: whatever size is chosen (default or explicit) is shrunk to the smallest one giving the *same
    number of chunks*, so the chunks come out even where they can. That is a compile-time concern, not a
    memory one: `lax.map` emits the element kernel **twice** when the chunk does not divide the cell
    count — once as the scan body, once unrolled for the tail — and on a problem the size of Taylor–Hood
    Stokes that duplicate doubles the build. Evening the chunks costs nothing in either direction: the
    chunk *count* is unchanged (no extra scan step) and each chunk is no larger than requested (peak
    memory only falls). It cannot always divide — 100 items in 3 chunks is 34+34+32 — and then it is
    simply a no-op.

    `chunk=` lives on `jno.fem` rather than `fem.solve` because the steady-linear operator is assembled
    *here*, before any solve.

    Coverage, since it is not uniform. The **native** assembler chunks its residual and jacobian, volume
    and surface loops. The **non-nodal** assembler chunks both, for every family — including the C⁰/C¹
    vertex families (see below). The **nonlinear tangent** is per-element too: the assembler takes the
    current iterate, so `J(u_k)` assembles element-by-element like the linear operator. One path still
    takes a global dense `jacfwd` and is unchunked, for **every** non-nodal family rather than any
    particular one: the **second-order-in-time** block, which builds a dense `2n × 2n` system by
    construction. The
    **1-D** assembler is not chunked at all, and an explicit `chunk=` there raises rather than being
    silently ignored.

    **Term introspection (provisional).** `fem.term_kinds` returns a `list[TermKind]` — each
    additively-split PDE (volume) term classified by `support`, `time_order`, `trial_channel` /
    `test_channel`, and `linear`, with `is_local` flagging a spatially pointwise term (reaction/mass) vs. a
    neighbour-coupling global one (diffusion/advection). This is the basis for operator-splitting routing;
    the API is provisional until that routing lands.

??? note "Term introspection (provisional)"
    **Term introspection (provisional).** `fem.term_kinds` returns a `list[TermKind]` — each
    additively-split PDE (volume) term classified by `support`, `time_order`, `trial_channel` /
    `test_channel`, and `linear`, with `is_local` flagging a spatially pointwise term (reaction/mass) vs. a
    neighbour-coupling global one (diffusion/advection). This is the basis for operator-splitting routing;
    the API is provisional until that routing lands.

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

!!! danger "Do not hand-roll backward Euler on a second-order block"
    The energy-conserving trapezoidal
    rule lives inside `fem.solve()`. **Do not** hand-roll backward Euler `(M + dt·A) w = M·w` off a
    second-order block — it spuriously **damps** the wave. If you integrate manually, use the trapezoidal
    step `(M + ½·dt·A) w_next = (M − ½·dt·A) w + dt·c`.

A **vector** field works too (`value_shape=(2,)`/`(3,)`) — that is elastodynamics,
`ρ u_tt = ∇·σ(u)` (see the vibrating-cantilever tutorial). A **coupled system** where every field
carries `u_tt` (spring-coupled membranes, coupled waves) rides the same augmented formula with the
coupled blocks — damping `u_t` terms, a nonlinear spatial operator (Newton on the augmented
residual) and a driven boundary `g(x, t)` all apply to the coupled case exactly as to a single
field. *Scope: nodal Lagrange, 2D/3D (1D has its own narrower path). Fail-loud: a coupled field
with no `u_tt` term (write a first-order field as an explicit first-order system), runtime
parameters on a coupled form, a trainable Dirichlet value, and `g(x, t)` on a nonlinear form.*
---

## The rest of this guide

| Page | What it covers |
|------|----------------|
| [Boundary conditions](boundary-conditions.md) | Dirichlet, Neumann, Robin, periodic, inequalities — all as residual terms |
| [Element families](elements.md) | H(div) / H(curl) / C¹ non-nodal elements; quadrilateral and hexahedral cells |
| [Geometry, regions & moving meshes](geometry.md) | Curved cells, trainable coordinates, r-adaptivity, per-region integration, axisymmetry, enclosure radiation |
| [Differentiable solve & inverse](inverse.md) | Solving at a value, sharding, reduced-order bases, field parameters, neural coefficients |
| [Vector, coupled & nonlinear](formulations.md) | Multifield problems and elasto-plasticity as a trace formula |
| [Limits & build time](limitations.md) | Known limitations and what compilation costs |

Choosing a linear or nonlinear solver is shared with `jno.fdm` and lives on its own page:
**[Solvers & preconditioners](../solvers.md)**.
