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
> `.todense()`/`reshape` is needed. `fem.operator` still exposes the raw sparse (`BCOO`) feax form
> for large problems; the `dense(...)` helper above densifies those (e.g. `fem.operator.A`).

---

## Domain, symbols, and derivatives

* **Domain** — any jNO domain works (`box`, `jno.domain.cube`, a CSG/`gmsh` constructor).
  Add `time=(t0, t1, n_steps)` to make it transient.
* **Symbols** — `u, phi = d.fem_symbols(value_shape=(), names=("u", "phi"), order=1)`.
  Use `value_shape=(2,)` for a vector unknown (elasticity, flow velocity), `order=2` for P2
  (quadratic) elements, and call `fem_symbols` once per field for coupled systems.
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

## What `jno.fem` returns

`jno.fem` picks the operator type from the form:

| Form | `fem.is_linear` / `is_transient` | Use |
|------|----------------------------------|-----|
| steady, linear in `u` | `True` / `False` | `fem.A`, `fem.b` → `jnp.linalg.solve` |
| steady, nonlinear in `u` | `False` / `False` | `fem.residual(u)`, `fem.jacobian(u)`, `fem.dofs` (Newton) |
| has a `u.t` term | — / `True` | `fem.M`, `fem.operator.A`, `fem.state0`, `fem.dt`, `fem.t0`, `fem.t1` |

Always-available: `fem.dofs`, `fem.points` (the coordinates the DOFs live on — use these for P2,
where they differ from the mesh vertices), `fem.operator`, and `fem.classification`.

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

`fem.solve(solve_fn)` lets you choose the solver (jNO writes none): the linear default is
`jnp.linalg.solve`, the nonlinear default an `optimistix` Newton `root_find` (implicit-diff).

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
save_ts) -> trajectory`. The block exposes `block.as_diffrax(args=...)` (a diffrax ODE term)
and `block.as_feax_pipeline(args=...)` (a feax backward-Euler pipeline) as ready-made options;
the implicit default is preferred for Dirichlet problems.

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
diffusivity field and a transient rate).

---

## Known limitations

The FEM / weak-form path is stable for the cases the tutorials cover, but the
FEAX-backed lowering has a few boundaries worth knowing. They apply only when you
**assemble a weak form** (`target="fem_system"` / `"fem_residual"`) or solve a
**transient problem through the FEAX time route** — the residual-PINN path is
unaffected. Each boundary is an explicit, fail-loud `NotImplementedError`, never a
silently wrong result.

- **Transient mass terms must be parameter-free.** In a time-dependent solve the
  mass term (`u_t * phi`) may not carry a trainable/runtime parameter. Keep it
  constant and place affine trainable parameters in the operator/residual instead
  — e.g. a diffusivity `nu` on the stiffness term, not on the time derivative.

- **First order in time only.** The Diffrax and FEAX time-stepping adapters handle
  first-order semidiscrete systems. A second-order-in-time PDE (such as the
  undamped wave equation, `u_tt = c² Δu`) must be rewritten as a first-order
  system; it cannot be assembled as a single second-order block.

- **No runtime Dirichlet parameters.** A trainable parameter may sit in the
  operator (stiffness) but not in an essential/Dirichlet boundary *value*: a
  runtime contribution that lifts Dirichlet data (a non-zero right-hand side) is
  rejected. Operator-coefficient inverse problems (e.g. recovering `nu`) are fine.

- **Affine parameter lowering expects a single, direct factor.** For trainable FEM
  coefficients, the affine fast-path recovers a parameter that is a *direct* scalar
  factor of a weak-form term (`nu * grad(u) · grad(phi)`). One trainable scalar per
  additive term — not nested inside another parameter or buried in a nonlinear
  expression — is the well-supported shape.

Hitting one of these is a signal to reformulate (move the parameter, reduce the
time order) rather than a bug — the error message names the offending term.

