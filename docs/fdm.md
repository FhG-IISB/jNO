# Finite Difference Method

`jno.fdm` is the **strong-form sibling** of [`jno.fem`](fem/index.md). You write the PDE and its boundary
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

!!! danger "`:cotangent` is a whole-Laplacian stencil — and `.d2` now refuses it"
    `"finite_difference:cotangent"` computes the **whole** Laplacian `Δu` for any dimension you ask
    for, so `ui.d2(x, scheme=…) + ui.d2(y, scheme=…)` used to **double** it — silently. The solve
    converged to half the true answer (relative error 0.494 → 0.498 → 0.499 under refinement, never
    converging, never raising).

    `.d2` and `.dd` now **raise** on that sub-scheme. Write the Laplacian as one term instead, which
    takes every coordinate at once and so cannot be double-counted:

    ```python
    lap = ui.laplacian(x, y, scheme="finite_difference:cotangent")   # ✅ one term, cannot double
    lap = ui.d2(x, scheme="finite_difference:cotangent") + ui.d2(y, …)   # ❌ raises
    ```

    `"finite_difference:lsq"` is **not** affected — it is a genuine per-direction stencil, so
    `.d2(x, scheme=":lsq") + .d2(y, scheme=":lsq")` is correct and equals
    `.laplacian(x, y, scheme=":lsq")` (both 1.978e-02 on the study below; the single `.d2(x, ":lsq")`
    alone is 1.046, as a per-axis derivative should be).

!!! measured "Every stencil's adjoint is exact"
    A strong-form solve is differentiable through whichever stencil you author. Measured on
    `−Δu = s·f` over the unit square (mesh 0.08, x64), where `u` is linear in `s` so `d(Σu)/ds`
    has a closed form:

    | stencil | `d(Σu)/ds` (AD) | closed form |
    |---|---|---|
    | `.laplacian(x, y, ":cotangent")` | +8.171969e+01 | +8.171969e+01 |
    | `.d2(x) + .d2(y)` (default) | +8.336520e+01 | +8.336520e+01 |
    | `.laplacian(x, y, ":lsq")` | +8.334671e+01 | +8.334671e+01 |

    This needed a fix: `jno.np.parameter` hardcoded `float32`, and `jno.fdm` casts the DOF vector to
    the unknown's dtype on every residual evaluation — so under x64 the operator silently rounded to
    single precision. That made it *non-linear* at the 6e-08 level, which capped the forward solve
    near 1e-05 and broke the adjoint Krylov solve outright (a gradient wrong by twenty orders). The
    dtype now follows `jax_enable_x64`, and every stencil above is linear to 2e-16.

### Variable coefficients — `(κ * ui.x).x`

The partials `ui.x`, `ui.y`, `ui.z` of the bound field compose, so a divergence-form operator
−∇·(κ∇u) is written exactly as on paper. κ can depend on the coordinates or on `u` itself:

```python
κ = 1.0 + x                      # or 1.0 + ui for a nonlinear diffusivity
jno.fdm([-(κ * ui.x).x - (κ * ui.y).y - f, u(xb, yb) - 0.0]).solve()
```

!!! measured "Manufactured u = sin(πx)sin(πy), unstructured mesh, h = 0.1 → 0.05 → 0.025"
    κ = 1 + x: 4.9e-2 → 1.2e-2 → 3.1e-3. κ = 1 + u: 5.1e-2 → 1.2e-2 → 3.0e-3. Both second order.

Use `.x`, not `.d(x)`, on an expression like `κ * ui.x`: `.d(x)` asks for an autodiff derivative, which a
nodal field cannot provide, so it raises.

### Convection — upwinding is a formula

Central differences for `b·∇u` oscillate once the cell Péclet number `|b|h/2ε` exceeds 1. First-order
upwinding is the central difference plus the numerical diffusion `|b|h/2`, an exact identity on a grid,
so it is written as that math. `domain.cell_size` is the node spacing `h`:

```python
h = d.cell_size
jno.fdm([-ε*Δu + b*ui.x - jnn.abs(b)*h/2*ui.xx - f, u(xb, yb) - 0.0]).solve()
```

!!! measured "−εΔu + u_x = 1, structured grid h = 0.05, exact solution ≤ 1"
    ε = 1e-2 (Péclet 2.5): central peaks at 1.38; upwind peaks at 0.87 and equals the hand-assembled
    upwind system to 1e-8. ε = 1e-3: central 2.36, upwind 0.93. Upwinding is first order and smears
    boundary layers; where Péclet < 1 the central form is more accurate.

In `jno.fdm`, `cell_size` at a node is the mean of `(d!·|K|)^(1/d)` over its cells: exactly the grid
spacing on a structured grid in 2-D and 3-D. This differs from `jno.fem`, where it is `|K|^(1/d)`, which is
`h/√2` on the same right triangles.

### Choosing the stencil

Every derivative view takes an optional `scheme=` — the *config stays on the operator it describes*.
The built-in stencils (parsed from the scheme string) are:

| `scheme=`                              | gradient stencil     | Laplacian stencil        |
| -------------------------------------- | -------------------- | ------------------------ |
| `"finite_difference"` (default)        | area-weighted        | gradient-of-gradient (a full Laplacian fuses to cotangent, below) |
| `"finite_difference:lsq"`              | least-squares        | lsq-of-gradient          |
| `"finite_difference:cotangent"`        | area-weighted        | cotangent (whole-Δ)      |
| `"finite_difference:uniform"`          | uniform              | gradient-of-gradient     |
| `"finite_difference:inverse_distance"` | inverse-distance     | gradient-of-gradient     |

!!! measured "How much the cotangent stencil buys — unit square, −Δu = f, float64"
    | mesh `h` | `.laplacian(x, y, scheme=":cotangent")` | `.d2(x) + .d2(y)` (default) |
    |---|---|---|
    | 0.10 | **1.164e-02** | 4.942e-02 |
    | 0.06 | **4.058e-03** | 1.674e-02 |
    | 0.035 | **1.417e-03** | 5.780e-03 |

    A stable ~4× at every resolution — the same convergence *rate*, a better constant.

!!! note "The plain Laplacian is the cotangent one on an unstructured mesh"
    The per-axis default above (a gradient of the area-weighted gradient) has a spurious oscillating
    mode on an unstructured mesh. Its lowest Dirichlet eigenvalue on the unit square is about 5.4, not
    2π², and refinement does not remove it. Measured: an advection–diffusion solve came out 2.09 off, and
    Helmholtz at c = 5.41 was 19.6 off. So `ui.xx + ui.yy`, `ui.d2(x) + ui.d2(y)`, `-ui.d2(x) - ui.d2(y)`
    and `ui.laplacian(x, y)` with the default scheme all become the cotangent Laplacian (0.047 and 0.004 on
    the same two problems). The fusion needs every spatial axis once, with one shared coefficient.
    `a·ui.xx + b·ui.yy`, a partial sum in 3-D, and an explicit sub-scheme are left as written, and
    `scheme="finite_difference:area_weighted"` still gives the gradient-of-gradient stencil. One exception:
    an FDM problem posed on a named sub-region (a domain-decomposition subdomain) keeps the per-axis
    stencil for now, because its interface flux is only consistent with that stencil.

An unknown sub-scheme (a typo, or one jNO does not have, such as `":upwind"`) raises. It used to fall
through silently to the default area-weighted stencil.

The `cotangent` Laplacian is the most accurate and is symmetric; the gradient methods trade accuracy
for locality. The scheme stays on the operator it describes — `ui.d2(x, scheme=…)` — so different
terms in the same residual can use different stencils. (`cotangent` is the accurate default in 2-D
**and** 3-D — the cotangent-weight operator on triangles, and its exact analogue the P1 finite-element
Laplace–Beltrami operator on tetrahedra; see [3-D tetrahedral meshes](#3-d-tetrahedral-meshes).)

## Structured grid (fast stencils)

For an axis-aligned **rectangle** or **box**, build a **regular grid** instead of an unstructured mesh by
asking the shape for a regular lattice with `.structured()`:

```python
d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.02).structured().domain()           # 2-D
d = jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.05).structured().domain()   # 3-D
```

This meshes the rectangle as a uniform right-triangulation — or, in 3-D, the box as a Kuhn
6-tets-per-voxel grid — (spacing from the shape's `size=`) and records a grid descriptor on
`d.mesh_connectivity["grid"]`. The interior operators (`jno.fdm.laplacian` / `gradient`, and the
constraint-list `ui.d2(x)` authoring) then detect the grid and apply the **direct finite-difference
stencils** — the 5-point (2-D) / 7-point (3-D) Laplacian `Σ (u₊ − 2u + u₋)/hₖ²` and central-difference
gradients — by array reshaping, with **no per-element assembly**. On a uniform 2-D right-triangulation the
5-point stencil coincides *exactly* with the cotangent P1 finite-element Laplacian (a classical result —
see Strang & Fix, *An Analysis of the Finite Element Method*, 1973), so the structured path is the *same
answer* as the unstructured `cotangent` operator, only cheaper.

!!! measured "Solve time, −Δu = f on the unit square, CPU (machine shared with other jobs)"
    | nodes | structured, first / repeat solve | unstructured `cotangent`, first / repeat |
    |---|---|---|
    | ~17k | 2.8 s / 0.03 s | 2.9 s / 0.17 s |
    | ~66k–75k | 3.1 s / 0.14 s | 7.8 s / 1.1 s |

    The first solve is mostly compilation. It is compiled once per problem and reused, so a repeat solve
    is fast. Before 2026-09 both paths re-traced every solve, and a compiled residual built an N×N distance
    table to map mesh nodes onto themselves. That made the structured path *slower* (7.9 s / 5.0 s at 17k),
    and both paths ran out of memory at 66k.

!!! measured "Parallelism: XLA does it, and the stencil runs at memory bandwidth"
    You write no parallel code. XLA spreads the fused stencil over the CPU cores and compiles it to one
    GPU kernel. i5-13600K (20 threads) and RTX 3070, float64:

    | 5-point Laplacian, 4096² grid | CPU, 1 core | CPU, all cores | GPU |
    |---|---|---|---|
    | jNO structured stencil | 18.5 ms | 6.8 ms (40 GB/s) | 1.6 ms (172 GB/s) |
    | plain copy `2*u` (the bandwidth ceiling) | 13.1 ms | 5.9 ms | 0.7 ms |

    On CPU the stencil sits at DRAM bandwidth. On GPU it is within 8% of a hand-written single-pass
    stencil. Whole steady solve (GMRES + multigrid), repeat call:

    | nodes | CPU | GPU |
    |---|---|---|
    | 1M | 1.6 s | 0.12 s |
    | 4M | 6.3 s | 0.57 s |

    The GPU loses below about 100k nodes, where kernel launches (about 20 µs each) dominate. GMRES keeps
    its restart vectors, so an 8 GB card runs out of memory near 16M nodes in float64.

The full `jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(bnd) - g]).solve()` works unchanged and stays
differentiable — no authoring change from the unstructured case. **Transient** composes too:
`.structured()` together with `time=(t0, t1, n)` and a `ui.t` term marches by method of lines as usual
(its backward-Euler operator is diagonally dominant, so it stays on the default inner solve). **Periodic**
boundaries and **complex** fields are *not* supported — they are absent from `jno.fdm` in general (see
[Scope](#scope-and-limitations)), not just on a structured grid; a regular grid is the natural home for
periodic wrap-around stencils, so that is a planned extension. (The grid operator does preserve a complex
field rather than silently dropping the imaginary part, matching the unstructured cotangent path.)

!!! note "Inner solver on a structured grid"
    The strong-form `−u.d2(x) − u.d2(y)` with row-replaced Dirichlet gives a **nonsymmetric**
    reduced operator, on which the default matrix-free BiCGStab can break down. A structured solve
    therefore defaults its inner Krylov to **GMRES** (robust for nonsymmetric systems, still matrix-free
    and differentiable via `custom_linear_solve`), **preconditioned by a geometric-multigrid V-cycle**
    (`jno.precond.gmg()`) — O(N), grid-independent convergence (~0.1 residual reduction per cycle). All
    automatic, with no change to how you write the problem; it falls back to plain GMRES when the grid is
    too small to coarsen (an odd cell count on any axis — pick a size giving an even, ideally power-of-two,
    cell count for the full multigrid speedup). `jno.precond.gmg()` is also a reusable slot for
    `fem.solve(linear=jno.solve.gmres(), precond=jno.precond.gmg())` on a structured domain. Override the
    inner solver with `.solve(nonlinear=…)` as usual.

    Supported: **2-D axis-aligned rectangles** (`shape.rect`) and **3-D boxes** (`shape.box`). A
    composite/CSG shape or a spatially varying `size=` raises; composite / cut-cell geometry is planned.

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
    equation. In **2-D** the normal is computed from the mesh boundary segments (exact on axis-aligned
    edges), and a **corner** node shared by two flux edges has no single outward normal, so it falls
    back to the interior PDE residual — give such a corner an explicit Dirichlet value if it needs
    anchoring. In **3-D** the normal comes from the region's boundary **faces**, each oriented outward
    exactly via its owning tetrahedron's apex (a flat face gives an exact axis normal), so face-edge
    nodes keep their flux row; where a flux face meets a Dirichlet face, the Dirichlet value wins (its
    row is applied last). A condition that is *not* affine in `∂u/∂n` raises rather than returning a
    wrong answer.

!!! measured "The flux closure follows the interior stencil — both stay second order"
    The gradient used for `∂u/∂n` must match the stencil the PDE uses, and `jno.fdm` picks it for you.
    The default `.d2` is a gradient of the area-weighted gradient, so imposing the flux on that same
    gradient is its consistent closure. The `cotangent` Laplacian, and any second derivative on a
    structured grid, never read it; they get a quadratic least-squares fit over the node's two-ring
    instead. Unit square, `sin(πx/2) sin(πy)`, Neumann on the right edge, rel. error at h = 0.1 / 0.05 / 0.025:

    | interior | area-weighted closure | quadratic closure |
    |---|---|---|
    | `.d2` (default) | **2.9e-2 / 7.5e-3 / 1.8e-3** | 3.6e-2 / 1.5e-2 / 5.7e-3 |
    | `cotangent` | 8.2e-3 / 3.6e-3 / 2.2e-3 | **5.1e-3 / 2.3e-3 / 7.4e-4** |

    The bold column is what you get. Before this, `cotangent` was paired with the first-order column and
    stalled. A structured grid is 6.7e-3 / 1.7e-3 / 4.3e-4; there a flux condition used to be dropped
    altogether (∂u/∂n = 2 and ∂u/∂n = 5 gave identical answers). A box face is axis-aligned, so on a grid
    `∂u/∂n` is the three-point one-sided difference `(−3u₀ + 4u₁ − u₂)/2h`, not the quadratic fit. Both
    are second order, but the fit's constant is large. Where the PDE fixes the mean of `u` only
    through a small reaction term, a flux error ε shifts the whole solution by `∮ε`. On `−Δu + u = f` with
    `∂u/∂n = 0` on all four sides, `u = cos πx cos πy + ½`, the fit gave 0.43 at h = 0.1 (the mean off by
    0.31); the one-sided difference gives 2.4e-3 / 5.5e-4 / 2.4e-4 at h = 0.1 / 0.05 / 0.025, rate → 2
    under further refinement (1.6, 1.8). On the single-edge problem above the two agree to within 7%. Naming a sub-scheme on the flux term,
    `ur.d(nr, scheme="finite_difference:lsq")`, overrides the choice.

### Periodic

Tie two opposite faces with a `u(A) - u(B)` constraint — exactly as `jno.fem`:

```python
jno.fdm([
    -ui.d2(x) - ui.d2(y) - f,
    u(xl, yl) - u(xr, yr),          # periodic in x  (left/right; 2-D bottom/top → y; a box: front/back → y, bottom/top → z)
    u(xb, yb) - 0.0, u(xt, yt) - 0.0,  # Dirichlet in y
]).solve()
```

This is **structured-only**: the tie wraps that grid axis so the `jnp.roll` stencil gives the true
periodic 5-/7-point Laplacian (a strong-form stencil must *wrap* — a mere boundary tie on an unstructured
mesh would keep a one-sided edge and solve the wrong problem, so it raises). The redundant `x=L ≡ x=0`
face is pinned to its main by the tie. Note: a periodic structured solve is currently un-preconditioned
(the geometric-multigrid V-cycle assumes Dirichlet boundaries), so it is slow on fine grids — periodic GMG
is a planned extension.

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

The `u.t` term carries a **unit or a general `c(x)·u.t` mass coefficient** (variable material, e.g.
`ρcₚ(x)·u.t`): it is extracted by a two-probe `c = F(u.t=1) − F(u.t=0)` and carried as `M = diag(c)`, so
no structural parsing is needed; a nonlinear `c(u)·u.t` fails loud. Nonlinear transient residuals are
handled the same way (the march reuses the Newton driver).

### Time-dependent data

Write a source, a boundary value or a coefficient with the time variable, and the march evaluates it at
each step's own time:

```python
x, y, t    = d.variable("interior", split=True)
xb, yb, tb = d.variable("boundary", split=True)

jno.fdm([
    ui.t - (1 + t) * Δu - f(x, y, t),       # coefficient κ(t) and source f(x, t)
    u(xb, yb) - jnn.exp(-tb) * jnn.cos(xb), # boundary value g(x, t)
    u(xi, yi) - u0,
]).solve()
```

!!! measured "Heat on the unit square with exact solutions, Crank–Nicolson, h = 0.05"
    A source f(x, t), a boundary value g(x, t), and a coefficient κ(t) = 1 + t each reach the spatial
    error floor (about 2e-3) and agree between the default path and `linear=` / `precond=` slots.
    Before this, a source written with t raised `KeyError('__time__')`, and a boundary value written
    with t was accepted but held at its start value (error 3.5 at T). With a slot set, data that
    depends on time rides the assembled time block's forcing; an operator that changes in time (κ(t))
    is detected and stepped by Newton on the assembled tangent instead.

### Time schemes

The march is **backward Euler** by default. Pass a `jno.solve` time scheme to `.solve(time=…)` — exactly
as `fem.solve(time=…)`, the *same* slot object — to change it:

```python
traj = jno.fdm([...]).solve(time=jno.solve.theta(0.5))   # Crank–Nicolson (2nd-order in time)
traj = jno.fdm([...]).solve(time=jno.solve.adaptive())   # step-doubling adaptive step size
```

`save_ts=` picks the times the march returns, as in `fem.solve(save_ts=…)`. The step Δt stays the one
`domain.time` sets; the trajectory is sampled at those times, interpolating linearly between steps:

```python
ts = np.linspace(t0, t1, n_steps)
traj = jno.fdm([...]).solve(save_ts=ts[::10])            # every 10th step
```

`jno.solve.theta(θ)` (θ = 1 backward Euler, 0.5 Crank–Nicolson, 0 forward Euler) and
`jno.solve.adaptive(…)` compose onto the method-of-lines march. The Dirichlet and flux rows have zero
mass, so they are constraints, and every θ imposes them at the new time. Forward Euler is explicit only in
the interior: it is stable for `Δt·λ_max ≤ 2`, i.e. `Δt ≤ h²/4` for the 2-D five-point Laplacian (the
largest eigenvalue is `8/h²`). The **exponential** integrator is *not*
available for `jno.fdm`, and it raises. It forms `exp(−Δt M⁻¹A)`, and a strong-form march is a DAE: the
Dirichlet and flux rows are algebraic constraints with zero mass. Measured with an assembled operator, it
ran but came back 4.8e-3 off a converged reference, where Crank–Nicolson at the same step was 2.0e-5.

## Solver slots — `linear=` and `precond=`

`jno.fdm(...).solve()` takes the same solver slots as `fem.solve()`: any `jno.solve` linear solver
(`cg`, `bicgstab`, `gmres`, `lu`, …) and any `jno.precond` spec (`jacobi`, `amg`, `gmg`, …).

```python
sol = jno.fdm([...]).solve(linear=jno.solve.cg(), precond=jno.precond.amg())       # steady or transient
sol = jno.fdm([...]).solve(linear=jno.solve.gmres(), precond=jno.precond.gmg())    # structured grid
```

Setting either assembles the strong-form operator as a sparse matrix once: one JVP per colour of the
stencil pattern, checked against the matrix-free action before use. What happens next depends on the
problem:

- **Linear, steady:** one `(A, b)` solve, composed exactly as `jno.fem` composes it.
- **Linear, transient:** a linear time block. Every step is one preconditioned solve, and an AMG setup is
  built once before the march.
- **Nonlinear:** Newton with the assembled tangent. A preconditioner that needs a matrix (`amg`, `gmg`) is
  set up once on the tangent at the initial guess.

Left unset, the matrix-free default is unchanged. The assembled operator stays differentiable, so a
crux-driven inverse runs through the slots too.

!!! measured "100 heat steps, unstructured `cotangent`, CPU, repeat solve (machine under load ≈ 5)"
    | nodes | default | `bicgstab` + `jacobi` | `cg` + `amg` | `lu` |
    |---|---|---|---|---|
    | 19k | 5.8 s | 4.4 s | 3.9 s | 6.1 s |
    | 76k | 40 s | 28 s | **15 s** | 38 s |

    The default's inner Krylov is unpreconditioned, so its iteration count grows with Δt and with mesh
    refinement. AMG holds it at 6–8 per step. Every column gives the same answer to every printed digit.

Scope: `gmg` preconditions one scalar field on a structured grid, so it refuses a coupled system; use
`amg` there. A time step's matrix is `α(−Δ) + σI` (`I + θΔt(−Δ)` for a heat step, `(4/Δt²)I − Δ` for the
Newmark wave step), and `gmg` reads `α` and `σ` from the operator and builds its V-cycle for exactly that.
A cycle for `−Δ` alone had cost the 201² wave march 3.0 s at Δt = 1e-3 (1.0 s now). When the shift
dominates (small Δt), `jacobi` is already nearly exact and is cheapest: 0.46 s on that march. `gmg` wins
at large Δt, where diffusion dominates: 2.6 s against 11.8 s for `jacobi` on a 401² heat march at Δt = 0.1.
For an operator of another form (a variable coefficient, advection), the cycle is built from its value at
the centre node.

---

### Second order in time — `u.tt`

A `ui.tt` term makes the problem second order in time: waves, vibrating membranes, elastodynamics.
Give the initial displacement as usual and, optionally, the initial velocity as `ui0.t - v0` on the
`initial` region (it defaults to zero), exactly as in `jno.fem`:

```python
xi, yi, ti = d.variable("initial", split=True)
ui, ui0 = u.bind(x=x, y=y, t=t), u.bind(x=xi, y=yi, t=ti)
Δu = ui.d2(x) + ui.d2(y)

traj = jno.fdm([
    ui.tt + c * ui.t - Δu,     # damped wave; drop c * ui.t for the undamped one
    u(xb, yb) - 0.0,
    u(xi, yi) - u0,            # initial displacement
    ui0.t - v0,                # initial velocity (optional, default 0)
]).solve()
```

By default it takes the trapezoidal step (Newmark average acceleration; Newmark 1959, *J. Eng. Mech.
Div.* 85), which does not damp an undamped wave, and solves it for the new displacement alone:
`(2m/Δt² + c/Δt)(u⁺ − u) − (2m/Δt)·v + ½(R(u⁺) + R(u)) = 0`, then `v⁺ = 2(u⁺ − u)/Δt − v`. The inertia
`m(x)` and damping `c(x)` may vary in space; if either depends on `u`, the solve raises. For a linear
problem the step matrix is assembled once. Each step then solves for the correction to the predictor
`u + Δt·v`, with CG where the step matrix is verified symmetric (a structured grid, with the Dirichlet
columns moved to the right-hand side) and GMRES otherwise. The Krylov iterations apply the stencil
matrix-free. An explicit `time=` scheme marches the augmented `[u; v]` state instead: backward Euler
visibly damps the wave, and BDF2 refuses. `.solve()` returns the `u` trajectory.

!!! measured "Gaussian pulse in a closed box, 300 steps to t = 1.5, RTX 3070, float64"
    | nodes | augmented `[u; v]`, `gmres` | Newmark default |
    |---|---|---|
    | 263k | 22 s first / 16 s repeat | 3.9 s / 1.7 s |
    | 1M | out of memory | 12.5 s / 7.1 s |

    The whole trajectory is kept (`(n_steps, N)`), which is what limits the grid on an 8 GB card.

!!! measured "Standing wave on the structured grid, h = 0.1, T = 0.5"
    Against the exact semidiscrete mode `cos(ω_h t)·sin(πx)sin(πy)`, the error at T is 1.2e-3 → 2.9e-4 →
    7.3e-5 for 25 → 50 → 100 steps (h = 0.05), so it is second order in Δt. The damped and
    initial-velocity cases match their closed forms to below 1e-3.

    Before this was wired, `ui.tt` was silently read as `ui.t`: the wave equation was solved as a heat
    equation.

## Differentiable inverse problems

When the constraint list carries a **trainable** `jno.np.parameter` — a source amplitude, a
diffusivity, a `jno.nn(...)` network — `jno.fdm([...]).solve()` returns a differentiable **trace node**
(not an array), exactly as `fem.solve()` does. It therefore composes straight into `jno.core`: put the
solve inside a data-misfit loss and let the parameter's attached optimizer recover it.

```python
s = jno.np.parameter((1,), name="s")            # the unknown to recover
s.optimizer(optax.adam(1e-1))
u = d.unknown(); ui = u.bind(x=x, y=y)

solve = jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()   # a trace node
crux  = jno.core([(solve - u_obs).mse])          # domain inferred from the graph
crux.solve(150)                                  # recovers s from the observation
```

At each `crux` step the parameter node resolves to its current value, the solve re-runs
(differentiably, through the `jno.solve` Newton–Krylov `custom_root`), and the gradient flows back to
the optimizer — no adjoint code. With **no** trainable parameter, `.solve()` returns the solution
array eagerly, as in every section above.

A trainable parameter can appear **anywhere** in the constraint list, and it can be a **time-dependent**
problem:
- in the PDE (a coefficient, a source);
- in a Dirichlet value, a Neumann value or a Robin coefficient;
- in a time coefficient (`ρ·u.t`, a wave speed);
- in an initial value.

```python
alpha = jno.np.parameter((1,), name="alpha")
node  = jno.fdm([-Δu - f, u(xl, yl) - 0.0, ut.d(nt) + alpha * (ut - 0.5)]).solve()        # Robin α
nu    = jno.np.parameter((1,), name="nu")
traj  = jno.fdm([ui.t - nu * Δu, u(xb, yb) - 0.0, u(xi, yi) - u0]).solve()                 # heat: ν from a trajectory
```

!!! measured "Recovered through jno.core, 300 Adam steps"
    Robin α (steady, structured and unstructured), and, in time-dependent problems, a diffusivity, a source
    amplitude, a time-dependent boundary amplitude, a Robin coefficient, a wave speed, and a diffusivity
    through `linear=` / `precond=` slots. Each comes back to within 1e-3 of the value that produced the
    observations.

    Before this, a Robin parameter crashed, a time-dependent inverse raised "No model for Model N", and a
    parameter in a Dirichlet value was read from its stored value, so its gradient was zero and the
    inverse silently never moved.

    A time-dependent inverse runs one forward solve at the parameters' current values when it is built.
    That warm-up makes the solver's structural decisions (linearity, sparsity, symmetry, what varies in
    time) on concrete values; the traced solve reuses them.

---

## 3-D tetrahedral meshes

Everything above dispatches on `domain.dimension`: give `jno.fdm` a **3-D tetrahedral** domain and the
same constraint list solves in 3-D. Build the mesh with [`jno.shape`](Domain-and-Geometry.md) — a box, sphere,
cylinder, or any boolean combination — and add the third coordinate:

```python
d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.1).domain()
x, y, z, _   = d.variable("interior", split=True)          # note the z coordinate
xb, yb, zb, _ = d.variable("boundary", split=True)
u  = d.unknown()
ui = u.bind(x=x, y=y, z=z)

f = 3.0 * np.pi**2 * jnn.sin(np.pi*x) * jnn.sin(np.pi*y) * jnn.sin(np.pi*z)
sol = jno.fdm([
    -ui.d2(x) - ui.d2(y) - ui.d2(z) - f,                   # -Delta u = f on the cube
    u(xb, yb, zb) - 0.0,                                   # Dirichlet u = 0
]).solve()
```

A cube from `jno.shape.box` auto-names its six faces `left/right/front/back/bottom/top`, so **flux**
conditions work per face exactly as in 2-D — bind to the face and take the normal derivative
(`nr = d.variable("right", normals=True)`, then `ui.d(nr) - h` or `ur.d(nr) + alpha*(ur - u_inf)`).

!!! note "3-D Laplacian stencil"
    The default `cotangent` Laplacian is the **P1 tetrahedral finite-element** (Laplace–Beltrami)
    operator — the exact 3-D analogue of the 2-D cotangent weights — so it is symmetric and second-order
    for the Galerkin solve. `gradient_of_gradient` (first-order, local) is the alternative;
    `lsq_of_gradient` is unstable for a *second* derivative on tets (the nested least-squares amplifies)
    and is not recommended in 3-D. As in 2-D, the whole-Laplacian `cotangent` stencil **cannot be split**
    across directions — write it as the single term `ui.d2(x, scheme="finite_difference:cotangent")`,
    not summed; the plain `−d2(x) − d2(y) − d2(z)` uses the per-direction `gradient_of_gradient`.

---

## Scope and limitations

**Supported:** scalar fields — or a **coupled system** of several `domain.unknown()` fields (Dirichlet
conditions, one PDE equation per unknown, steady or first order in time; `.solve()` returns `(nf, N)`, a
march `(n_steps, nf, N)`) — on a **2-D triangular or 3-D
tetrahedral** mesh; any mix of Dirichlet and flux (Neumann / Robin / coordinate-coefficient, affine in
`∂u/∂n`) boundary conditions, in 2-D and 3-D, **steady or transient** (a transient flux node is an
algebraic zero-mass-row constraint); transient problems by the method of lines, first order (a `u.t` term with a unit or a general `c(x)·u.t` mass
coefficient, `M = diag(c)`) or second order ([`u.tt`](#second-order-in-time-utt), with optional damping and
initial velocity), with a selectable [time scheme](#time-schemes); linear and nonlinear
residuals; differentiable inverse problems.

Author a coupled system as one PDE equation per unknown, in declaration order (equation *k* drives
unknown *k*), plus each field's BCs:

```python
u = d.unknown(); v = d.unknown()
ui = u.bind(x=x, y=y); vi = v.bind(x=x, y=y)
uh, vh = jno.fdm([
    -ui.d2(x) - ui.d2(y) + vi - f_u,   # equation for u
    -vi.d2(x) - vi.d2(y) + ui - f_v,   # equation for v
    u(xb, yb) - 0.0, v(xb, yb) - 0.0,  # Dirichlet per field
]).solve()                              # returns (2, N): uh = row 0, vh = row 1
```

A coupled system marches like a scalar one: give each field its `u(xi, yi) - u0` (a field without one
starts at 0) and write equation *k* with its own unknown's `u.t`. An equation with no time derivative makes
its field **algebraic**: a constraint solved at every step (a DAE), such as an elliptic potential driven by
a diffusing field. The time scheme, solver slots and trainable parameters work as for one field:

```python
traj = jno.fdm([
    ui.t - (ui.xx + ui.yy) + ui * vi**2 - F * (1 - ui),   # Gray–Scott reaction–diffusion
    vi.t - 0.5 * (vi.xx + vi.yy) - ui * vi**2 + (F + k) * vi,
    u(xb, yb) - 1.0, v(xb, yb) - 0.0, u(xi, yi) - u0, v(xi, yi) - v0,
]).solve(time=jno.solve.theta(0.5))      # (n_steps, 2, N)
```

Measured on `u_t = Δu − v`, `v_t = Δv + u` (exact `e^{−2π²t} S (cos t, sin t)`), Crank–Nicolson with
Δt ∝ h: 1.0e-2 / 2.5e-3 / 6.2e-4 structured, 1.7e-2 / 4.4e-3 / 1.1e-3 unstructured at h = 0.1 / 0.05 / 0.025.
A flux condition on a coupled system belongs to the one field whose normal derivative it carries, and
replaces that field's equation at the boundary nodes. Its value may read the other fields and their
derivatives. The normal's components `nx, ny` are available as values, so a wall pressure from the
momentum balance, ∂p/∂n = n·(νΔ**u** − (**u**·∇)**u**), is one expression for every wall:

```python
xw, yw, _, nx, ny = d.variable(wall, normals=True, split=True)
n = d.variable(wall, normals=True)
mx, my = nu*Δ(uw) - (uw*uw.x + vw*uw.y), nu*Δ(vw) - (uw*vw.x + vw*vw.y)
pw.d(n) - (nx*mx + ny*my)
```

`nx, ny` are the same per-node normals the flux row uses (exact on a box face). The flux itself must be
written as `ub.d(n)`. Spelled in components, `nx*ub.x + ny*ub.y`, it raises: it has no normal
derivative to be recognised by, and it used to be read as a second PDE. A condition that differentiates
two fields raises too.

Not supported on a coupled system, and each raises: `u.tt` (write it as a first-order system in
`(u, v = u.t)`), and the time derivative of another field inside equation *k* (a non-diagonal mass).

#### Incompressible Navier–Stokes

Velocity and pressure are three coupled fields, and the equations are written as they are:

```python
h = d.cell_size
jno.fdm([
    ui.t + ui*ui.x + vi*ui.y + pi.x - nu*(ui.xx + ui.yy),
    vi.t + ui*vi.x + vi*vi.y + pi.y - nu*(vi.xx + vi.yy),
    ui.x + vi.y - 0.05*h**2*(pi.xx + pi.yy),        # continuity, pressure-stabilised
    u(xb, yb) - U, v(xb, yb) - V, p(xb, yb) - P, u(xi, yi) - U0, v(xi, yi) - V0,
]).solve(time=jno.solve.bdf2())
```

Two things decide whether the pressure is right. Both were measured against exact solutions.

- **The `− 0.05·h²·Δp` term.** On one collocated grid, central differences split the pressure into four
  decoupled sub-lattices (a checkerboard). The velocity still converges, but the pressure does not.
  On Kovasznay flow (Re = 40) the pressure error stalls at 1e-1. The O(h²) pressure Laplacian
  (pressure stabilisation, Brezzi & Pitkäranta 1984) couples the sub-lattices and vanishes as h → 0:
  the pressure then converges (3.4e-2 / 1.0e-2 / 3.6e-3 at h = 0.1 / 0.05 / 0.025), and the velocity is
  unchanged (second order).
- **The time scheme.** The pressure has no time derivative, so it is an algebraic field. Crank–Nicolson
  leaves its value at each step oscillating: on the Taylor–Green vortex it does not converge (0.11 / 0.18 /
  0.21). Use `jno.solve.bdf2()` (5.7e-2 / 1.5e-2 / 4.0e-3, velocity second order) or backward Euler.

Where the boundary pressure is not known, impose it from the momentum balance, as a flux condition on p:
`pb.d(n) - n·(νΔu − u·∇u)`. With p given on one edge and that condition on the other three, Kovasznay
converges the same way. Where every wall has that condition, p is defined only up to a constant. Fix it at
one interior node:

```python
d.point_region("gauge", (0.5, 0.5))            # the mesh node nearest (0.5, 0.5)
xg, yg, _ = d.variable("gauge", split=True)
terms.append(p(xg, yg) - 0.0)                   # replaces continuity there, the one dependent equation
```

A boundary node would not do: the pressure's flux rows are independent of each other, and pinning one of
them leaves the system singular. Measured on the lid-driven cavity at Re = 100 (no slip, lid u = 1, wall
pressure from the momentum balance, default solver), `u(0.5, y)` against Ghia, Ghia & Shin (1982,
Table I): max deviation 0.014 on 33² and 0.0019 on 65² (7 s). From a zero initial guess, Newton at
higher Re diverges and raises. Continuing in Re with `x0=` from the previous solution
(`nonlinear=jno.solve.newton(line_search=True)`) reached Re = 300 on 65² and failed at 350, where the cell
Reynolds number Re·h is 5.5. That is past the classical limit of 2 for central differencing of
convection, where the discrete steady solution can stop existing. Higher Re is not verified here.
`jno.fdm` problem can also be **one subdomain of a larger solve** — coupled to a FEM or PINN region by
overlapping Schwarz or Dirichlet–Neumann, and differentiable through the converged fixed point. See
[Domain decomposition](domain-decomposition.md).

An axis-aligned 2-D rectangle or 3-D box can use a fast [structured grid](#structured-grid-fast-stencils)
(`.structured()`) with direct finite-difference stencils in place of the unstructured mesh.

A periodic tie `u(left) - u(right)` (opposite faces) wraps that axis on a
[structured grid](#structured-grid-fast-stencils).

**Planned:** periodic on unstructured meshes and periodic geometric multigrid (a periodic structured solve
is currently un-preconditioned, so it is slow on fine grids); composite / cut-cell structured geometry
(axis-aligned rectangles and boxes are supported, above); 1-D meshes. Authoring a `jno.shape` sub-region
through `domain.region(name, shape)` + `d.variable`, and 3-D coupled solves, additionally need
region-tag support on the base 3-D domain (a separate 3-D domain-decomposition feature). A pure-Neumann
problem (no Dirichlet node anywhere) is singular — the solution is defined only up to an additive
constant — and is solved as-is.
