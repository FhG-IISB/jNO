# Finite Difference Method

`jno.fdm` is the **strong-form sibling** of [`jno.fem`](fem.md). You write the PDE and its boundary
and initial conditions as the *same* kind of constraint list — but instead of a weak form with test
functions and quadrature, the **strong residual** is collocated at the mesh nodes with
finite-difference stencils. There is no test function, no mass matrix, and no quadrature, so it is
leaner than the assembler; and because the residual is a plain differentiable function of the nodal
DOFs, the solve is **differentiable** (through `custom_root`) and composes into inverse problems just
like `fem.solve()`.

```python
import jax
jax.config.update("jax_enable_x64", True)     # the strong-form solve accumulates in float64
import numpy as np
from shapely.geometry import box
import jno
import jno.jnp_ops as jnn

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
x, y, _  = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
u  = d.unknown()               # a valued P1 nodal field — the strong-form counterpart of fem_symbols()
ui = u.bind(x=x, y=y)          # bound view with .d / .d2 (finite differences by default)

f = 2.0 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
sol = jno.fdm([
    -ui.d2(x) - ui.d2(y) - f,  # -Delta u = f   (collocated at the mesh nodes)
    u(xb, yb) - 0.0,           # Dirichlet u = 0
]).solve()                     # -> the nodal solution vector
```

The API deliberately mirrors `jno.fem`: `u = domain.unknown()` plays the role of the trial symbol,
`u.bind(x=…, y=…)` gives the derivative views, and each boundary/initial condition is a term in the
same list — no separate BC objects.

---

## The nodal unknown and its derivatives

`domain.unknown()` returns a **valued** nodal field: one degree of freedom per mesh vertex (a P1
field). It is discrete, so autodiff with respect to a coordinate is meaningless — the derivative
views therefore default to **finite differences**:

| you write        | meaning                                                        |
| ---------------- | ------------------------------------------------------------- |
| `ui.d(x)`        | `∂u/∂x` by finite differences (no `scheme=` needed)           |
| `ui.d2(x)`       | `∂²u/∂x²` by finite differences                               |
| `ui.d2(x) + ui.d2(y)` | the FD Laplacian, one direction at a time               |

!!! warning "Do not split the whole-Laplacian stencils"
    `"finite_difference:cotangent"` and `"finite_difference:lsq"` compute the **whole** Laplacian
    `Δu` in one shot. Writing `ui.d2(x, scheme="finite_difference:cotangent") + ui.d2(y, …)` therefore
    **doubles** it. Use the default per-direction stencil for `.d2(x) + .d2(y)`.

### Choosing the stencil

Every derivative view takes an optional `scheme=` — the *config stays on the operator it describes*.
The built-in stencils (parsed from the scheme string) are:

| `scheme=`                              | gradient stencil     | Laplacian stencil        |
| -------------------------------------- | -------------------- | ------------------------ |
| `"finite_difference"` (default)        | area-weighted        | gradient-of-gradient     |
| `"finite_difference:lsq"`              | least-squares        | lsq-of-gradient          |
| `"finite_difference:cotangent"`        | area-weighted        | cotangent (whole-Δ)      |
| `"finite_difference:uniform"`          | uniform              | gradient-of-gradient     |
| `"finite_difference:inverse_distance"` | inverse-distance     | gradient-of-gradient     |

The `cotangent` Laplacian is the most accurate and is symmetric; the gradient methods trade accuracy
for locality. The scheme stays on the operator it describes — `ui.d2(x, scheme=…)` — so different
terms in the same residual can use different stencils.

---

## Boundary conditions

### Dirichlet

An essential condition is the term `u(region) - g`, with `g` a constant or a coordinate expression —
identical to `jno.fem`:

```python
u(xb, yb) - 0.0                     # homogeneous
u(xb, yb) - (xb**2 + yb**2)         # inhomogeneous g(x, y)
```

### Flux conditions — Neumann, Robin, and beyond

A flux boundary condition carries the **normal derivative** `∂u/∂n`. Get the edge's outward normal
from `domain.variable(region, normals=True)` and write the condition with *that edge's own tags* —
bind the field to the edge and take its normal derivative:

```python
xr, yr, _ = d.variable("right", split=True)
nr        = d.variable("right", normals=True)     # outward-normal Variable for the right edge
ur        = u.bind(x=xr, y=yr)                     # field bound to the edge (flux + value terms)

ur.d(nr) - h                                       # Neumann:  ∂u/∂n = h
ur.d(nr) + alpha * (ur - u_inf)                    # Robin:    ∂u/∂n + α(u - u∞) = 0
```

`jno.fdm` handles **any condition affine in `∂u/∂n`** — Neumann, Robin, a coordinate-coefficient
`κ(x)·ur.d(n)`, either sign — by reading the coefficient of `∂u/∂n` directly (it evaluates the term
with the normal derivative pinned to `0` and to `1`, giving the row `a·(∇u·n) + b`). There are no
special BC objects, and **any mix** of Dirichlet, Neumann, and Robin on different edges composes:

```python
jno.fdm([
    -ui.d2(x) - ui.d2(y) + 2.0,     # -Delta u = -2
    u(xbo, ybo) - 0.0,              # Dirichlet (bottom)
    ul.d(nl) - 0.0,                 # Neumann   (left, insulated)
    ur.d(nr) - 0.0,                 # Neumann   (right, insulated)
    ut.d(nt) + 1.0 * (ut - 3.0),    # Robin     (top)
]).solve()
```

!!! note "How flux BCs differ from `jno.fem`"
    In `jno.fem` a Neumann condition is a *natural* weak term `h·v` carrying the test function. The
    strong form has no test function, so the flux is imposed **directly** on the boundary node's
    equation. The normal is computed from the mesh boundary segments (exact on axis-aligned edges).
    A **corner** node shared by two flux edges has no single outward normal, so it falls back to the
    interior PDE residual — give such a corner an explicit Dirichlet value if it needs anchoring. A
    condition that is *not* affine in `∂u/∂n` raises rather than returning a wrong answer.

---

## Transient problems

A problem is **transient** exactly when it carries an initial condition — and, as in `jno.fem`, the
IC is *found from the constraints* (`u(xi, yi) - u0`, with `xi, yi` the `"initial"` region), never a
config flag. The time window and step count come from `domain.time = (t0, t1, n)`; the `u.t` term
marks the time derivative, and `jno.fdm` marches by the **method of lines**, reusing the very same
semidiscrete time-stepper `jno.fem` uses. `.solve()` returns the trajectory `(n_steps, N)`:

```python
d = jno.domain(box(0, 0, 1, 1), mesh_size=0.06, time=(0.0, 0.5, 200))
x, y, t   = d.variable("interior", split=True)     # note the temporal Variable t
xb, yb, _ = d.variable("boundary", split=True)
xi, yi, _ = d.variable("initial",  split=True)     # the t = t0 slice
ui = u.bind(x=x, y=y, t=t)

traj = jno.fdm([
    ui.t - nu * (ui.d2(x) + ui.d2(y)),                 # u_t = nu * Delta u
    u(xb, yb) - 0.0,                                   # Dirichlet
    u(xi, yi) - jnn.sin(np.pi*xi) * jnn.sin(np.pi*yi), # initial condition
]).solve()
```

The `u.t` term must appear with unit coefficient (the standard `u.t - 𝒩(u)` form). Nonlinear
transient residuals are handled the same way (the march reuses the Newton driver).

---

## Differentiable inverse problems

When the constraint list carries a **trainable** `jno.np.parameter` — a source amplitude, a
diffusivity, a `jno.nn.wrap` network — `jno.fdm([...]).solve()` returns a differentiable **trace node**
(not an array), exactly as `fem.solve()` does. It therefore composes straight into `jno.core`: put the
solve inside a data-misfit loss and let the parameter's attached optimizer recover it.

```python
s = jno.np.parameter((1,), name="s")            # the unknown to recover
s.optimizer(optax.adam(1e-1))
u = d.unknown(); ui = u.bind(x=x, y=y)

solve = jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()   # a trace node
crux  = jno.core([(solve - u_obs).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
crux.solve(150)                                  # recovers s from the observation
```

At each `crux` step the parameter node resolves to its current value, the solve re-runs
(differentiably, through the `jno.solve` Newton–Krylov `custom_root`), and the gradient flows back to
the optimizer — no adjoint code. With **no** trainable parameter, `.solve()` returns the solution
array eagerly, as in every section above.

---

## Scope and limitations

**Supported (v1):** scalar fields on a 2-D triangular mesh; any mix of steady Dirichlet and
flux (Neumann / Robin / coordinate-coefficient, affine in `∂u/∂n`) boundary conditions; transient
problems by the method of lines (`M = I`, unit-coefficient `u.t`); linear and nonlinear residuals;
differentiable inverse problems.

**Planned:** transient flux boundary conditions (a flux node keeps `M = 1`, unlike a pinned Dirichlet
node); periodic boundaries; a general `u.t` mass coefficient; a structured-grid stencil backend; 1-D
and 3-D meshes. A pure-Neumann problem (no Dirichlet node anywhere) is singular — the solution is
defined only up to an additive constant — and is solved as-is.
