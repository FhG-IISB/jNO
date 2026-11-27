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
| `matrix_free=True` | a steady linear operator is linearised once and applied through the cells: `fem.operator` is `(A, b)` with `A` a matvec (`A @ v`, `A.T`, `A.diag()`); solved by Jacobi-BiCGStab or the Krylov slot you pass. A direct solver or a matrix-based preconditioner has nothing to factor and raises |

A curved boundary is still meshed with straight-sided cells: its `O(h²)` geometry error is not reduced by the degree.

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
    | spectral elements off a box, Q3 / Q5 / Q8, `matrix_free=True`, GLL rule | L-shape (12 quads) 9.3e-5 / 6.4e-8 / 1.5e-12; unstructured polygon (33 quads) 1.8e-4 / 3.0e-7 / 8.7e-12 |
    | one Q_k Poisson apply, hand-written sum-factorised kernel against the assembled matvec (227k / 69k / 36k DOFs; Q3 / Q5 / Q8) | i5-13600K f64: 4.6x / 16x / 30x faster; RTX 3070 f64: 0.53x / 1.7x / 3.5x; f32: 1.1x / 3.5x / 6.0x; operator memory 205 / 128 / 140 MB against 25 / 5.5 / 2.3 MB |
    | `jno.fem(matrix_free=True)`, 3-D Poisson, 2.0-2.1M DOFs, RTX 3070, f64, Q3 / Q5 / Q8 | build 11 / 15 / 33 s; apply 31 / 20 / 17 ms (1.7x / 2.5x / 3.0x the hand-written kernel); Jacobi-CG solve 1.1 / 1.3 / 1.6 s (10 / 29 / 61 iterations, a single-mode right-hand side); max error 1.3e-10 / 2.0e-12 / 5.7e-12; 0.63 GB peak on the card |
    | same, f32 | apply 9.3 / 7.1 / 5.9 ms (4.1x / 3.0x / 3.9x the hand-written kernel); CG stalls at a true residual of 6e-4 / 7e-4 / 1e-3 and is refused |

## Limits

`jno.spectral`: box domains only (rect, box, a 1-D line); other geometry goes through the spectral elements of
`jno.fem` above. `etdrk4` needs every axis Fourier. A transient equation must live on its
unknown's own nodes. Second order in time (`u.tt`) is not supported yet.
