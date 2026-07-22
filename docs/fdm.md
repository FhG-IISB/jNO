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
terms in the same residual can use different stencils. (`cotangent` is the accurate default in 2-D
**and** 3-D — the cotangent-weight operator on triangles, and its exact analogue the P1 finite-element
Laplace–Beltrami operator on tetrahedra; see [3-D tetrahedral meshes](#3-d-tetrahedral-meshes).)

## Structured grid (fast stencils)

For an axis-aligned **rectangle** or **box**, build a **regular grid** instead of an unstructured mesh by
passing `structured=True`:

```python
d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.02), structured=True)           # 2-D
d = jno.domain(jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.05), structured=True)   # 3-D
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

The full `jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(bnd) - g]).solve()` works unchanged and stays
differentiable — no authoring change from the unstructured case. **Transient** composes too:
`structured=True` together with `time=(t0, t1, n)` and a `ui.t` term marches by method of lines as usual
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

    Supported: **2-D axis-aligned rectangles** (`Shape.rect`) and **3-D boxes** (`Shape.box`). A
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
available for `jno.fdm`: it needs a linear operator `A` (for `e^{AΔt}`), which the strong-form
matrix-free residual does not assemble, so it fails loud pointing you to a θ-scheme.

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
same constraint list solves in 3-D. Build the mesh with [`jno.Shape`](Domain-and-Geometry.md) — a box, sphere,
cylinder, or any boolean combination — and add the third coordinate:

```python
d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.1).domain()
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

A cube from `jno.Shape.box` auto-names its six faces `left/right/front/back/bottom/top`, so **flux**
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
algebraic zero-mass-row constraint); transient problems by the method of lines (a `u.t` term with a unit or a general `c(x)·u.t` mass
coefficient, `M = diag(c)`) with a selectable [time scheme](#time-schemes); linear and nonlinear
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

Coupled fields are v1-limited to steady + Dirichlet (transient / flux on coupled fields are planned). A geometric sub-region for a subdomain /
domain-decomposition solve (`jno.dd.couple([(problem, region)])`) resolves to a mesh-node subset via
the analytic, shapely-free [`Shape.contains`](Domain-and-Geometry.md) — in 2-D **and** 3-D.

An axis-aligned 2-D rectangle or 3-D box can use a fast [structured grid](#structured-grid-fast-stencils)
(`structured=True`) with direct finite-difference stencils in place of the unstructured mesh.

**Planned:** periodic boundaries; composite / cut-cell structured geometry (axis-aligned rectangles and
boxes are supported, above); 1-D meshes; transient / flux BCs on coupled multi-field systems. Authoring a `jno.Shape` sub-region
through `domain.region(name, shape)` + `d.variable`, and 3-D coupled solves, additionally need
region-tag support on the base 3-D domain (a separate 3-D domain-decomposition feature). A pure-Neumann
problem (no Dirichlet node anywhere) is singular — the solution is defined only up to an additive
constant — and is solved as-is.
