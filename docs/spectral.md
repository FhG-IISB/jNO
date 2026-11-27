# Spectral methods

`jno.spectral([...])` solves a strong-form constraint list, written exactly as for [`jno.fdm`](fdm.md), whose unknowns
are **spectral**: their nodal values live on Fourier, Chebyshev or Legendre nodes, and their derivatives are taken
from the basis. On smooth solutions the error falls geometrically with the number of nodes, not as a power of h.

```python
import jno, jno.numpy as jnn
from math import pi

d = jno.shape.rect(0, -1, 2 * pi, 1).domain()                  # geometry only: no mesh is built
U = d.unknown(space="spectral", order=(32, 25))                 # nodes per axis
x, y, _ = d.variable("interior", split=True)                    # the unknown's own nodes
xl, yl, _ = d.variable("left", split=True); xr, yr, _ = d.variable("right", split=True)
xb, yb, _ = d.variable("bottom", split=True); xt, yt, _ = d.variable("top", split=True)
u = U.bind(x=x, y=y)
sol = jno.spectral([
    -(u.xx + u.yy) - jnn.sin(x) * jnn.cos(pi * y / 2),
    U(xb, yb) - 0.0, U(xt, yt) - 0.0,                          # walls: Dirichlet rows
    U(xl, yl) - U(xr, yr),                                      # a written tie: x is Fourier
]).solve()
coeffs = U.basis.coeffs(sol)                                    # Fourier x Chebyshev coefficients
fine = U.basis(sol, xq, yq)                                     # spectral interpolation anywhere
```

`jno.fdm([...])` given a spectral unknown hands the problem to `jno.spectral`.

## The pieces

| piece | what it is |
|---|---|
| `d.unknown(space="spectral", order=(nx, ny))` | the trial space; `order` is nodes per axis (an int: every axis) |
| a periodic tie `U(xl, yl) - U(xr, yr)` | makes that axis **Fourier**; every other axis is **Chebyshev** (Gauss–Lobatto) |
| `space=("fourier", "legendre")` | names the families instead; checked against the ties |
| `space="spectral:gauss"`, `"legendre:gauss"` | the Gauss rule (interior nodes) instead of Gauss–Lobatto |
| `d.variable(tag)` | the points of a region of the first spectral unknown's node set: `interior`, `initial`, `boundary`, a face (`left`, `right`, `bottom`, `top`, `front`, `back`) |
| `d.variable(tag, nodes=P)` / `nodes=jno.sp.gauss(y=31)` | the same region on another unknown's nodes / a derived node set |
| `U.bind(x=x, y=y, scheme=jno.sp(dealias=3/2))` | an equation built from this binding is evaluated on a 3/2-padded grid and projected back (Orszag's rule) |
| `psi.mean` (or a pin `P(xp, yp) - 0`) | a gauge row: fixes the constant of a periodic or pure-Neumann unknown |
| `U.basis` | `nodes`, `coeffs(values)`, `values(coeffs)`, `basis(values, x, y)`, `refine(values, order)`, `integrate`, `mean`, `wavenumbers(axis)` |

Rows: every equation holds at every node of its unknown; a boundary condition replaces the rows of its nodes
(Dirichlet wins at a corner, as in `jno.fdm`); a gauge replaces one more. An equation written on another node set
(continuity on the pressure nodes) gives that set's rows. Rows must equal the unknown's nodes.

## Solvers

| call | what runs |
|---|---|
| `.solve()` | dense LU up to 4096 DOFs; above, GMRES (tolerance 1e-12) with `jno.precond.spectral()` |
| `.solve(linear=jno.solve.gmres(), precond=jno.precond.spectral())` | GMRES preconditioned by the **Fourier-mode block** inverse: exact for an operator constant along the Fourier axes (walls included), so 1–3 iterations. Without a Fourier axis (or blocks above `max_block`) it is **fast diagonalisation** of the separable constant-coefficient part, exact for Dirichlet rows: a rough-data 72² Chebyshev problem in 0.4 s, where unpreconditioned GMRES stalled after 130 s |
| nonlinear residual | Newton (`jno.solve.newton`) with the same inner solves |
| `domain(time=(t0, t1, n))` + `u.t` + an initial condition | the method of lines on any `jno.solve` time scheme (`theta`, `bdf2`, `sdirk`, ...); `precond=jno.precond.spectral()` makes each step Newton-GMRES on the mode-block inverse of `M + s ∂R` (opt-in: the default step solve was measured faster at 16x17 .. 64x65) |
| `.solve(time=jno.solve.etdrk4())` | fourth-order exponential time differencing on a periodic box: the constant-coefficient linear terms are integrated exactly from their symbol, algebraic unknowns (a streamfunction) are solved mode by mode at every stage, the state stays in coefficient space |

## Coupling

A spectral problem is a participant of `jno.core([...])`'s partitioned coupling like `jno.fem`, `jno.fdm` and
`jno.fvm` (see [Coupling](coupling.md)). Its interface is a face of its box. Between two spectral sides the data
cross by spectral interpolation along the face, so the coupled solution keeps spectral accuracy whatever the two
node counts; to or from a mesh side they cross by the usual consistent and conservative maps.

## Spectral elements in `jno.fem`

A Lagrange element of degree > 2 on a quadrilateral or hexahedron (`fem_symbols(order=k)` on a `.quad()` mesh) uses
the Gauss–Lobatto–Legendre nodes: the spectral-element basis, well conditioned at high order. This is the route to
geometry `jno.spectral` does not take: any quadrilateral mesh, structured or not.

```python
d = jno.shape.polygon([(0, 0), (2, 0.3), (2.4, 1.6), (0.9, 2.1), (-0.3, 1.2)]).sized(0.6).quad().domain()
u, v = d.fem_symbols(order=8)                                    # Q8 on 33 unstructured quadrilaterals
x, y, _ = d.variable("interior", split=True); xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - g],
              quadrature="gll",                                  # GLL rule at the nodes: diagonal mass
              matrix_free=True)                                  # the operator is applied, never stored
sol = fem.solve(linear=jno.solve.cg(), precond=jno.precond.jacobi())
```

| piece | what it does |
|---|---|
| `order=k`, k > 2, on quads / hexes | GLL nodes (`gll_warped`), no option needed |
| `quadrature="gll"` | the GLL rule at those nodes (`k + 1` points per axis, exact to degree `2k - 1`): the mass matrix is diagonal (exact lumping, total mass to round-off) and an affine cell's stiffness stays exact |
| sum factorisation (automatic, k > 2) | a field's values and gradients at the quadrature points are 1-D contractions along each axis of the cell's DOF tensor, `O(k^(d+1))` per cell where the dense basis tables cost `O(k^(2d))`. Every residual (linear, nonlinear, transient) and the element tangent use it |
| `matrix_free=True` | a linear form's operator is linearised once and applied through the cells, never stored. Steady: `fem.operator` is `(A, b)` with `A` a matvec (`A @ v`, `A.T`, `A.diag()`), solved by Jacobi-BiCGStab or the Krylov slot you pass. Transient (`u.t` or `u.tt`, constant boundary values): every `jno.solve` time scheme marches it; the mass is stored as its diagonal when it is diagonal (`quadrature="gll"`), so `jno.solve.erk()` costs one stiffness apply per stage and no solve. A direct solver or a matrix-based preconditioner has nothing to factor and raises. Nonlinear: Newton-Krylov, the default of `fem.solve()` and of `jno.solve.newton()`. Each step applies the tangent as a JVP of the residual and preconditions it with the tangent's diagonal, sum-factorised as well, so no tangent is ever assembled (`jno.solve.newton(direct=True)` assembles it anyway on a steady form). A nonlinear transient form takes each implicit step the same way, with the diagonal mass |

An explicit wave march, the standard use of spectral elements:

```python
d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=16).quad().domain(time=(0.0, 0.05, 51))
u, v = d.fem_symbols(order=8)                                    # 2.1M DOFs
...
fem = jno.fem([ui.tt * vi + ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(xb, yb, zb) - 0.0,
               u(x0, y0, z0) - u0], quadrature="gll", matrix_free=True)
traj = fem.solve(time=jno.solve.erk("rk4"), save_ts=np.linspace(0, 0.05, 6))
```

The step is the explicit (CFL) limit: about `h / k^2`.

A curved boundary needs `shape.quad().curved()`. Without it the cells are straight-sided, the domain is a polygon, and
its `O(h²)` geometry error is not reduced by the degree. With it, the nodes of each boundary cell are put on the true
boundary at the field's degree: edge nodes by the shape's signed distance, the cell's interior by transfinite
blending (Gordon-Hall), so the map stays smooth. Holes and boolean shapes work; interfaces between regions and
hexahedra stay straight. A cell with two boundary edges meeting at a smooth point of the boundary would open to 180°
when curved and is refused with a hint to mesh finer there.

The default `fem.solve()` is an iterative solve to a relative residual of 1e-8, which is coarser than a
high-degree discretisation: a Q8 solution good to 6e-11 came back at 1.2e-6. Pass `linear=jno.solve.lu()` or a
Krylov tolerance (`jno.solve.cg(tol=1e-13)`) to see the spectral error.

`matrix_free=True` pays from about Q5 on a GPU and at every degree on a CPU; at Q3 on a GPU the assembled matvec is
the faster one and the gain is memory only. In float32 the matrix products run at full precision (a GPU's default
for float32 is TensorFloat-32, which made the operator wrong by 5.6e-2 at 2M DOFs), but the residual a float32 CG
can reach, `eps x condition`, is 6e-4 to 1e-3 at 2M DOFs: above jNO's 1e-4 solve gate, so solve such sizes in
float64.

!!! measured "Measured (2026-11-27)"
    | case | result |
    |---|---|
    | Chebyshev² Poisson, 20 x 20 | max error < 1e-11 |
    | periodic Helmholtz, Fourier x Chebyshev channel Poisson | < 1e-12 |
    | Neumann + Robin rows, 18 x 18 | < 1e-10 |
    | Kovasznay flow, P_N–P_{N-2} (pressure on Gauss nodes), N = 12, 16, 20 | velocity error falls > 20x per +4 nodes, < 1e-7 at N = 20 |
    | Taylor–Green vortex, ETDRK4, 16², T = 1 | 1e-12 against the exact decay |
    | ETDRK4 order in dt (perturbed, nonlinear) | slopes > 3.6 |
    | two spectral slabs coupled, 16 x 14 and 14 x 19 nodes, sin(πy) interface | < 1e-9, 3 sweeps |
    | 2-D NS, ETDRK4, 3/2 dealiasing, RTX 3070, against a hand-written JAX solver | 256²: 1.9 ms f64 (0.99x), 0.29 ms f32 (0.89x); 512²: 8.9 ms (1.05x), 1.6 ms (0.93–1.00x); 1024²: 29.6 ms (0.97x), 8.8 ms (0.81x); 0.5 GB peak at 1024² f64 |
    | same problem, 128², against Dedalus 3 (RK443) | relative L2 difference 1.1e-4 at dt 1e-3, 2.6e-5 at Dedalus dt/4: the gap is Dedalus' own time error |
    | spectral elements, 3-D Poisson, 2³ hexahedra, Q2 / Q3 / Q4 / Q5 / Q6 | max error 2.3e-2 / 4.5e-4 / 3.4e-5 / 8.9e-7 / 7.9e-8 |
    | `-Δu = 1` on the unit disk, 52 quadrilaterals, Q2 / Q4 / Q8, straight against `shape.quad().curved()` | straight 5.1e-3 / 5.1e-3 / 5.1e-3 (the polygon's error); curved 8.8e-5 / 2.3e-7 / 2.1e-12 |
    | spectral elements off a box, Q3 / Q5 / Q8, `matrix_free=True`, GLL rule | L-shape (12 quads) 9.3e-5 / 6.4e-8 / 1.5e-12; unstructured polygon (33 quads) 1.8e-4 / 3.0e-7 / 8.7e-12 |
    | one Q_k Poisson apply, hand-written sum-factorised kernel against the assembled matvec (227k / 69k / 36k DOFs; Q3 / Q5 / Q8) | i5-13600K f64: 4.6x / 16x / 30x faster; RTX 3070 f64: 0.53x / 1.7x / 3.5x; f32: 1.1x / 3.5x / 6.0x; operator memory 205 / 128 / 140 MB against 25 / 5.5 / 2.3 MB |
    | `jno.fem(matrix_free=True)`, 3-D Poisson, 2.0-2.1M DOFs, RTX 3070, f64, Q3 / Q5 / Q8 | build 6.8 / 6.0 / 6.4 s; apply 19 / 11 / 11 ms (1.03x / 1.34x / 1.94x the hand-written kernel); Jacobi-CG solve 0.8 / 0.9 / 1.2 s (10 / 29 / 61 iterations, a single-mode right-hand side); max error 1.3e-10 / 2.0e-12 / 5.6e-12; 1.1 / 0.9 / 0.7 GB peak on the card, of which 0.56 / 0.40 / 0.36 GB kept coefficients (14 numbers per quadrature point) |
    | `jno.fem(matrix_free=True)`, 3-D `-div((1 + u²) grad u) = f`, Newton-Krylov (5 steps), RTX 3070, f64, Q3 / Q5 / Q8 | 530k DOFs: solve 3.3 / 2.1 / 2.8 s, 270-370 inner Jacobi-BiCGStab iterations, one JVP 4.9 / 2.9 / 3.3 ms (the linear apply: 4.9 / 2.7 / 3.1 ms), the diagonal 58 / 36 / 32 ms, 0.4-0.55 GB peak; 2.1-2.2M DOFs: 36 / 14 / 14 s, 1.2-1.4 GB; 130k DOFs against the assembled Newton: 0.55 / 0.41 / 0.37 s vs 0.60 / 1.31 s / out of memory, peak 0.2 / 0.14 / 0.15 GB vs 1.0 / 2.4 GB / > 2.8 GB |
    | 3-D wave `u.tt = Δu`, `matrix_free=True`, GLL mass, RK4, 2.0-2.1M DOFs, RTX 3070, Q3 / Q5 / Q8 | float64: 114 / 54 / 52 ms a step (29 / 13 / 13 ms a stage), build 15 / 7 / 10 s, 1.1-1.3 GB peak, max error after 50 steps 3.0e-10 / 5.9e-12 / 5.3e-13 against the exact standing wave; float32: 31 / 40 / 43 ms a step, 0.7 GB, error 1-2e-6 |
    | same, f32 | build 8.5 / 7.4 / 7.8 s; apply 6.0 / 4.6 / 4.1 ms (2.6x / 1.9x / 2.7x the hand-written kernel); CG stalls at a true residual of 6e-4 / 7e-4 / 1e-3 and is refused |

## Limits

`jno.spectral`: box domains only (rect, box, a 1-D line); other geometry goes through the spectral elements of
`jno.fem` above. `etdrk4` needs every axis Fourier. A transient equation must live on its
unknown's own nodes. Second order in time (`u.tt`) is not supported yet.

`jno.fem(matrix_free=True)`: Jacobi is the only preconditioner, so a Krylov solve takes more iterations as the degree
and the mesh grow (a Newton step of the nonlinear Poisson above: 55-73 inner iterations at 530k DOFs, 77-104 at 2M).
A parametric nonlinear form takes the exact diagonal pass (`O(k^(2d+1))` per cell, not the sum-factorised one); a
nonlinear transient form with a state-dependent mass, a driven boundary `g(x, t)` or a parameter-valued boundary
value assembles its step tangent as before. Periodic ties and complex fields are refused.
