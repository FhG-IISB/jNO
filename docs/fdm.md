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
| `"finite_difference"` (default)        | area-weighted        | gradient-of-gradient     |
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

    A stable ~4× at every resolution — the same convergence *rate*, a better constant. Worth the one
    extra word on the term whenever the mesh is unstructured.

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
    stalled. A structured grid is 6.9e-3 / 1.6e-3 / 4.1e-4; there a flux condition used to be dropped
    altogether (∂u/∂n = 2 and ∂u/∂n = 5 gave identical answers). Naming a sub-scheme on the flux term,
    `ur.d(nr, scheme="finite_difference:lsq")`, overrides the choice.

### Periodic

Tie two opposite faces with a `u(A) - u(B)` constraint — exactly as `jno.fem`:

```python
jno.fdm([
    -ui.d2(x) - ui.d2(y) - f,
    u(xl, yl) - u(xr, yr),          # periodic in x  (left/right; bottom/top → y, front/back → z)
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

### Time schemes

The march is **backward Euler** by default. Pass a `jno.solve` time scheme to `.solve(time=…)` — exactly
as `fem.solve(time=…)`, the *same* slot object — to change it:

```python
traj = jno.fdm([...]).solve(time=jno.solve.theta(0.5))   # Crank–Nicolson (2nd-order in time)
traj = jno.fdm([...]).solve(time=jno.solve.adaptive())   # step-doubling adaptive step size
```

`jno.solve.theta(θ)` (θ = 1 backward Euler, 0.5 Crank–Nicolson, 0 forward Euler) and
`jno.solve.adaptive(…)` compose onto the method-of-lines march. The **exponential** integrator is *not*
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

Scope: `gmg` preconditions one scalar field on a structured grid, so it refuses a coupled system and the
`[u; v]` state of a `u.tt` problem; use `amg` there. That augmented system is not symmetric, so `cg` does
not apply to it; use `gmres` or `bicgstab`.

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

It marches the augmented state `[u; v]` with `v = u_t`: `u̇ = v`, `m·v̇ + c·v + R(u) = 0`. The inertia `m(x)`
and damping `c(x)` may vary in space; if either depends on `u`, the solve raises. The default scheme is
θ = ½ (trapezoidal, Newmark average acceleration), which does not damp an undamped wave. `time=` swaps in
another θ; backward Euler visibly damps it, and BDF2 refuses. `.solve()` returns the `u` trajectory.

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

**Supported:** scalar fields — or a **coupled system** of several `domain.unknown()` fields (steady +
Dirichlet, one PDE equation per unknown; `.solve()` returns `(nf, N)`) — on a **2-D triangular or 3-D
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

Coupled fields are v1-limited to steady + Dirichlet (transient / flux on coupled fields are planned). A
`jno.fdm` problem can also be **one subdomain of a larger solve** — coupled to a FEM or PINN region by
overlapping Schwarz or Dirichlet–Neumann, and differentiable through the converged fixed point. See
[Domain decomposition](domain-decomposition.md).

An axis-aligned 2-D rectangle or 3-D box can use a fast [structured grid](#structured-grid-fast-stencils)
(`.structured()`) with direct finite-difference stencils in place of the unstructured mesh.

A periodic tie `u(left) - u(right)` (opposite faces) wraps that axis on a
[structured grid](#structured-grid-fast-stencils).

**Planned:** periodic on unstructured meshes and periodic geometric multigrid (a periodic structured solve
is currently un-preconditioned, so it is slow on fine grids); composite / cut-cell structured geometry
(axis-aligned rectangles and boxes are supported, above); 1-D meshes; transient / flux BCs on coupled
multi-field systems. Authoring a `jno.shape` sub-region
through `domain.region(name, shape)` + `d.variable`, and 3-D coupled solves, additionally need
region-tag support on the base 3-D domain (a separate 3-D domain-decomposition feature). A pure-Neumann
problem (no Dirichlet node anywhere) is singular — the solution is defined only up to an additive
constant — and is solved as-is.
