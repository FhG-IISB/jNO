# Finite Element Method

jNO assembles and solves finite-element problems through a **single entry point**,
`jno.fem([...])`. You write the weak form as a plain Python list of **residual terms** —
volume physics, natural boundary terms, and essential boundary conditions, all in the same
list — and `jno.fem` returns a `FEM` object carrying the assembled operators.

The same traced weak-form language powers the steady solve, the transient time-stepper, and
the **differentiable** `fem.solve()` used for inverse problems.

```python
import jax.numpy as jnp
from shapely.geometry import box
import jno

dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # A may be sparse or dense

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
u, phi = d.fem_symbols()                                   # trial / test functions
xi, yi, _ = d.variable("interior", split=True)            # volume quadrature coords
xb, yb, _ = d.variable("boundary", split=True)            # boundary coords
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)         # views with .x / .y derivatives

f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])   # weak form + u = 0 on the boundary
u_h = jnp.linalg.solve(fem.A, fem.b)
```

> The flat accessors — `fem.A`, `fem.b`, `fem.M`, `fem.state0`, and `fem.residual(u[, t])` /
> `fem.jacobian(u[, t])` — return ready-to-use **dense** matrices and **flat** vectors, so no
> `.todense()`/`reshape` is needed. `fem.operator` still exposes the raw sparse (`BCOO`) operator
> for large problems; the `dense(...)` helper above densifies those (e.g. `fem.operator.A`).

---

## Domain, symbols, and derivatives

* **Domain** — any jNO domain works (`box`, `jno.domain.cube`, a CSG/`gmsh` constructor).
  Add `time=(t0, t1, n_steps)` to make it transient.
* **Symbols** — `u, phi = d.fem_symbols(value_shape=(), names=("u", "phi"), order=1)`.
  Use `value_shape=(2,)` for a vector unknown (elasticity, flow velocity), `order=2` for P2
  (quadratic) elements, `space="RT"`/`"N1E"`/`"P0"` for the non-nodal families (see below), and
  call `fem_symbols` once per field for coupled systems.
* **Quadrature coordinates** — `d.variable("interior", split=True)` returns the volume
  coordinates; `d.variable("<edge>", split=True)` returns a boundary edge's coordinates. A
  `box` auto-tags `"left"`, `"right"`, `"bottom"`, `"top"` (and `"front"`/`"back"` for a cube);
  `"boundary"` is the whole boundary and `"initial"` the `t = t0` slice.
* **Bound views** — `ui = u.bind(x=xi, y=yi, t=ti)` ties a symbol to a set of coordinates.
  The value is `ui`; spatial derivatives are `ui.x`, `ui.y`, `ui.z`; the time derivative is
  `ui.t`. (This replaces the old `jno.np.grad(u, xg)` / `u.d(xg)` spelling.)

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
> **Not yet / excluded:** 3-D (the zoo is 2-D only — 3-D uses nodal Lagrange); higher order (lowest
> RT₀ / N1E₀ only); other families (BDM, second-kind Nédélec, **Argyris**/C¹); quad / non-triangular
> meshes; and a constraint-consistent *algebraic* initial state at `t0` in the saddle-DAE transient
> (the differential field and all `t > 0` values are correct; only the reported `t0` algebraic value is).

Beyond nodal Lagrange (P1/P2), `jno.fem` assembles **edge-DOF** families on 2-D triangles — for
problems whose natural space is *not* H¹. Pick one with the `space=` knob on `fem_symbols`:

| `space` | Space | DOF | Use |
|---------|-------|-----|-----|
| `"Lagrange"` (default) | H¹ | nodal value | standard PDEs |
| `"RT"` | **H(div)** Raviart–Thomas | edge normal flux `∫ₑ u·n` | mixed Poisson, Darcy, conservation |
| `"N1E"` | **H(curl)** Nédélec (1st kind) | edge tangential `∫ₑ u·t` | Maxwell, eddy currents |
| `"P0"` | L² (piecewise constant) | one per cell | the pressure / multiplier of a mixed pair |

```python
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")   # H(div) flux
p, q = d.fem_symbols(names=("p", "q"), space="P0")                     # piecewise-constant scalar
```

jNO assembles these with its own push-forward engine, but the weak form reads like
any other coupled problem.

**Vector operators** (on a bound vector view): `u.div(x, y)` is the divergence and `u.curl(x, y)` the
2-D scalar curl `∂uy/∂x − ∂ux/∂y`; after binding, the no-arg `u.div()` / `u.curl()` reuse the bound
coordinates. (`div` is equivalently `trace(grad(u, [x, y]))`.)

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

Tutorials: `mixed_poisson_rt_2d.py` (H(div)) and `maxwell_nedelec_2d.py` (H(curl): magnetostatics +
eddy current). *Scope: lowest-order RT₀ / N1E₀ on 2-D triangular meshes.*

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
block exchange. Use `axisymmetric=True` for a body of revolution (the `(r, z)` meridional mesh). By
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

### Field parameters `k(x)` + regularization

`jno.np.parameter(phi)` is a **nodal field** on the trial space — a trainable value per node.
Field inversion is ill-posed, so add a smoothness/structure prior with `k.regularize(...)`
(`"h1seminorm"`, `"l2"`/`"tikhonov"`, `"tv"`, `"nonneg"`, `"bounded"`):

```python
k = jno.np.parameter(phi, name="k")                       # P1 field, one DOF per node
crux = jno.core([(fem.solve() - u_obs).mse, 1e-3 * k.regularize("h1seminorm").mean], domain=obs)
```

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
* **P2 elements** — `order=2` gives quadratic elements; read the solution at `fem.points`.

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

