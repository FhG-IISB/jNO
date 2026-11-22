# Finite Volume Method

`jno.fvm` solves the **same constraint list as [`jno.fdm`](fdm.md)** by finite volumes. Each PDE term is
integrated over a control volume: an outer derivative becomes a flux through the faces, everything else is
evaluated at the centroid. Fluxes telescope, so what leaves one volume enters its neighbour and the scheme is
conservative to round-off. That is what convection-dominated flow, shocks and incompressible flow need.

```python
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jno
import jno.jnp_ops as jnn

d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).domain()
x, y, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
u = d.unknown()
ui = u.bind(x=x, y=y)

f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
prob = jno.fvm([-(ui.xx + ui.yy) - f, u(xb, yb) - 0.0])
sol = prob.solve()            # one value per cell
prob.centroids                # where they live, (n_cells, 2)
prob.to_nodes(sol)            # volume-weighted values at the mesh nodes
```

`jno.fvm([...], centering="vertex")` uses median-dual control volumes around the mesh nodes instead, and
returns nodal values, as `jno.fdm` and `jno.fem` do.

!!! measured "Poisson, −Δu = f from sin(πx)sin(πy), relative L² error"
    | mesh | h = 0.1 | 0.05 | 0.025 |
    |---|---|---|---|
    | triangles, cell-centred | 4.9e-3 | 1.0e-3 | 2.6e-4 |
    | quadrilaterals, cell-centred | 8.3e-3 | 2.1e-3 | 5.1e-4 |
    | triangles, vertex-centred | 1.2e-2 | 3.1e-3 | |
    | fdm cotangent (for comparison) | 1.2e-2 | 4.1e-3 (h = 0.06) | |

    Second order on lines, triangles, quadrilaterals, tetrahedra and hexahedra, and for
    −∇·((1 + u)∇u) = f. A linear field is exact (1e-9) with Dirichlet, Neumann and a u⁴ radiation condition
    on a skewed mesh, and on −Δu = 1 the boundary outflow equals the total source to 1e-12 at any resolution.

## How a term is read

| you write | finite-volume meaning |
|---|---|
| `(F).x`, `F.d(x)` | the flux `Σ_f F_f S_f,x` through the faces |
| `ui.xx + ui.yy` | `Σ_f ∇u_f · S_f`: two fluxes of the face gradient |
| `(k*ui.x).x + (k*ui.y).y` | `Σ_f k_f ∇u_f · S_f`, with `k` taken at the face |
| `c(x) * (F).x` | the coefficient at the volume, times the face sum |
| `ui.t`, a source, a reaction | evaluated at the centroid (the midpoint rule) and multiplied by the volume |
| `b*ui.x`, `ui*ui.x` | **not** a divergence: the volume gradient is used, and a warning says the term is not conservative |

Inside a flux, a sub-term that holds a derivative of an unknown is **diffusive**. It gets the face gradient:
least-squares gradients interpolated to the face, with the component along the centroid line replaced by the
two-point difference. That is compact on an orthogonal mesh and corrected for skewness otherwise. A sub-term
in the unknowns' values alone is **convective**. All of them, from every equation, are gathered into one
normal flux `F·n` and passed to a numerical flux. Gathering first counts the dissipation of a Rusanov flux once
rather than once per axis.

A product distributes over a sum, so `nu * (ui.xx + ui.yy)` is two diffusive fluxes, and a number traced by
`jax.grad` stays a coefficient.

## Boundary conditions

They are written as for `jno.fdm` and hold on the boundary faces of their region:

```python
u(xl, yl) - g                 # Dirichlet
ur.d(nr) - h                  # Neumann,  nr = d.variable("right", normals=True)
ur.d(nr) + a * (ur - u_inf)   # Robin
ur.d(nr) - 1.0 + (ur**4 - 16) # any nonlinear flux law
```

Every condition goes through one rule. A boundary face carries a value `q_f`. With the owner value carried to
the foot of the face normal, `q_P'`, its normal derivative is `(q_f − q_P')/d⊥`. The condition is then one
equation in `q_f` per face, solved by a per-face Newton step (one step when it is affine). A face without a
condition has zero normal derivative: an insulated wall, and plain outflow for convection.

A condition with fewer equations than its unknown has components, such as the slip wall `U·n = 0` on a vector
unknown, projects the interior state onto the condition set. The convective flux there sees the mirror state,
so no mass crosses the wall. The normal components are `n[..., 0]` and `n[..., 1]`. Note that
`d.variable(region, normals=True, split=True)` returns the coordinates, not the normal.

```python
nb = d.variable("boundary", normals=True)
qb = q.bind(x=xb, y=yb)
qb[1] * nb[..., 0] + qb[2] * nb[..., 1]     # slip: (hu, hv)·n = 0
```

A value condition on a region without boundary faces, such as `d.point_region("ppin", (0, 0))`, pins the
nearest cell. That is how a pressure gauge is set. A periodic tie `u(left) - u(right)` joins the two faces.
The mesh must be periodic-conforming, and ties for several unknowns on the same faces join them once.

## Convection: `jno.fv(...)`

The scheme of a flux rides on its outer derivative, as a `jno.fd(...)` stencil does:

```python
s = jno.fv(flux="hll", limiter="minmod")
(bx * ui).d(x, scheme=s) + (by * ui).d(y, scheme=s)
```

| option | values |
|---|---|
| `flux=` | `"rusanov"` (default; exact upwinding for a linear flux), `"hll"`, `"roe"`, or a callable `flux(qL, qR, n, Fn)` |
| `limiter=` | `"vanalbada"` (default, smooth), `"minmod"`, `"vanleer"`, `"superbee"`, `"barth"`, `"venkat"`, `None`, or a callable ψ(r) |
| `recon=` | `"lsq"` (default), `"green_gauss"`, `"none"` (first order) |
| `average=` | a diffusive coefficient at the face as the `"arithmetic"` or `"harmonic"` mean of its cells |
| `speed=` | the wave speed as an expression, in place of the spectral radius |
| `equilibrium=` | a steady state to preserve exactly (well-balancing) |

Nothing in the scheme knows about a particular PDE:
- **The flux Jacobian** `∂(F·n)/∂q` comes from JAX.
- **Rusanov** uses its spectral radius.
- **HLL** uses its extreme eigenvalues: the extreme real roots of its characteristic polynomial, found by
  Newton from ±‖A‖∞. This is exact for a hyperbolic system and runs the same on every device.
- **Roe** damps the jump with `|A| = sqrt(A² + δ²I)` (Denman–Beavers), which carries Harten's entropy fix and
  needs no eigendecomposition.
- **The ψ(r) limiters** are carried to unstructured faces by the Darwish–Moukalled upwind difference, written
  so that they stay smooth where the jump vanishes.
- **Barth–Jespersen and Venkatakrishnan** bound the reconstruction by the neighbours' values. Venkatakrishnan's
  `K` is `venkat_k=0.3`: his 5 let a step overshoot by 13 % on a triangulation, while 0.3 keeps it within
  0.1 % and smooth enough for Newton.

!!! measured "−0.01 u″ + u′ = 0, u(0) = 0, u(1) = 1 (cell Péclet 2.5 at h = 0.05)"
    | scheme | range at h = 0.05 | max error at h = 1/320 |
    |---|---|---|
    | first order (`recon="none"`) | [0, 0.29] | 4.8e-2 |
    | van Albada (default) | [0, 0.21] | 7.5e-3 |
    | unlimited (`limiter=None`) | [−0.09, 0.00] | 8.4e-3 |
    | HLL / Roe + van Leer | [0, 0.18] | 8.0e-3 / 7.9e-3 |

    Rusanov with `recon="none"` is the hand-assembled upwind matrix to 1e-12. A harmonic face mean is exact
    across a layered coefficient (κ = 1 | 10). Steady viscous Burgers, `(u²/2)′ = ν u″`, converges to its tanh
    profile at second order.

Newton is the default for nonlinear problems. It factorises the assembled tangent, with a line search. Use a
smooth limiter (van Albada, Venkatakrishnan) for steady solves. Barth–Jespersen's `min()` and minmod are for
explicit time marching.

## Transient problems and `jno.solve.erk`

A problem with an initial condition marches `M u̇ + R(u, t) = 0` with `M = diag(c·V)`, where `c` is the
coefficient of `u.t`. Every `jno.solve` time scheme applies, and `u.tt` problems march `[u; v]`. Explicit
Runge–Kutta is new:

```python
prob.solve(time=jno.solve.erk("ssprk3", cfl=0.5))
```

`erk(method)` offers `"ssprk3"` (Shu–Osher), `"ssprk2"`, `"ssprk54"` (Spiteri–Ruuth), `"rk4"`, `"euler"`, or any
explicit Butcher tableau. It also works for `jno.fdm` and `jno.fem`: rows with zero mass are constraints and
are solved at every stage. With `cfl=`, every step is split into substeps from
`Δt = 2·cfl/ρ(M⁻¹ ∂R/∂u)` of the first-order operator. For upwinded convection that is the classical Courant
number. Eagerly it is re-read at every step, and under a gradient trace the count is fixed from the initial
state.

!!! measured "Transient"
    - **Heat equation:** rel-L² 8.2e-4 at t = 0.5 with Crank–Nicolson and with SSPRK3.
    - **Square pulse** under minmod + SSPRK3: its total variation never grows, and it stays within [0, 1].
    - **Inviscid Burgers shock:** at x₀ + t/2 (Rankine–Hugoniot) to one cell.
    - **SSPRK(5,4):** the error falls 16-fold per halved step.
    - **Taylor–Green vortex** on a doubly periodic square: kinetic energy decays as exp(−4νt) to 2 %.

## Systems

A system is a vector unknown with a vector flux, or several scalar unknowns. Their convective fluxes go
through one Riemann solve, so the two layouts give the same answer. The Euler equations:

```python
q = d.unknown.vector(3); qi = q.bind(x=x, t=t)
rho, mom, E = qi[0], qi[1], qi[2]
u = mom / rho; p = (gamma - 1) * (E - 0.5 * mom * u)
F = jnn.stack([mom, mom * u + p, u * (E + p)], axis=-1)    # axis=-1: one row per point
jno.fvm([qi.t + F.d(x, scheme=jno.fv(flux="roe", limiter="minmod")), q(xi) - q0])
```

`equilibrium=q_eq` subtracts the discrete residual of a steady state (Berberich, Chandrashekar & Klingenberg
2021), so that state is preserved to round-off for any balance law. That covers a lake at rest over a bed, or a
hydrostatic atmosphere.

!!! measured "Systems"
    - **Sod shock tube** at h = 0.01, density L1 error: Rusanov 5.7e-3, HLL 4.6e-3, Roe 4.5e-3; mass constant
      to 1e-12.
    - **Lake at rest** over a bump with slip walls: |hu| stays at 1e-15 with `equilibrium=` and drifts to
      2.6e-2 without it.
    - **Sloshing hump** in a closed box: its volume is kept to 1e-12.

## Saddle pairs: incompressible flow, Darcy

An equation without a term in its own unknown is a **constraint** on the unknown whose flux it carries. The
pair is found by structure, not by name. Its flux is central with the Rhie–Chow momentum-weighted correction,
which couples the multiplier (a pressure) to its neighbours. A gradient of the multiplier in a volume term is
read in Green–Gauss form, which is conservative. The weight `D_f = β V/a_P` is probed from the momentum
equation itself:
- **Convection:** the spectral radius of its convective flux.
- **Diffusion:** the sensitivity of its central flux to the gradient.
- **Drag:** the diagonal of its volume terms (Darcy's `u/K`).
- **β:** its coefficient of `∇p`.

```python
U, P = d.unknown.vector(2), d.unknown()
Ui, pi = U.bind(x=x, y=y), P.bind(x=x, y=y)
u, v = Ui[0], Ui[1]; s = jno.fv()
momentum = (u*Ui).d(x, scheme=s) + (v*Ui).d(y, scheme=s) - nu*(Ui.xx + Ui.yy) + jnn.stack([pi.x, pi.y], axis=-1)
jno.fvm([momentum, Ui[0].x + Ui[1].y, U(xl, yl) - lid, U(xw, yw) - 0.0, P(xp, yp) - 0.0]).solve()
```

Newton on the assembled tangent is the default. Three alternatives work on any detected pair:
- `nonlinear=jno.solve.simple(alpha_u=0.7, alpha_p=0.3)` (steady),
- `nonlinear=jno.solve.piso(n_corr=2)` (transient),
- `precond=jno.precond.saddle(schur="simple")` with `gmres` or `fgmres`.

!!! measured "Saddle pairs"
    - **Lid-driven cavity**, Re = 100, 32 × 32 cells, constant lid (a cell-centred scheme has no corner node):
      within 3.2e-3 of Ghia, Ghia & Shin (1982) on the centre line; continuity to 3e-13.
    - **Kovasznay flow** (Re = 40): 7.8e-2 → 1.2e-2 from 16 to 32 cells across.
    - **Darcy flow** on triangles: second order (1.9e-3 → 4.8e-4).
    - **SIMPLE** reaches the Newton cavity solution to 4e-8, and **PISO** the implicit march to 5e-9.

## Solvers, preconditioners, inverse problems

- **Linear problems** are assembled by colouring. The default is `cg` + `amg` when the operator is symmetric,
  `gmres` + `ilu` otherwise, and `linear=` / `precond=` take any `jno.solve` / `jno.precond` spec.
- **`jno.precond.gmg()`:** on a structured quadrilateral or hexahedral mesh the cells come in lattice order,
  so `gmg` works on the cell grid. `cg` + `gmg` equals the default to 2e-14 from 16² to 64² cells.
- **Inverse problems:** a trainable `jno.np.parameter` or network in the constraints makes `.solve()` a node
  that `jno.core` trains, and `jax.grad` differentiates a solve directly.

```python
s = jno.np.parameter((1,), name="s"); s.optimizer(optax.adam(1e-1))
node = jno.fvm([-(ui.xx + ui.yy) - s * f, u(xb, yb) - 0.0]).solve()
jno.core([(node - observed).mse]).solve(120)     # s -> 1.0 within 2e-3
```

`jno.info(prob)` reports the control volumes and how each equation was read.

## Limitations

- A PDE on a sub-region (`domain.region`) is not supported.
- A partial boundary condition (`U·n = 0`) needs a vector unknown, and with `centering="vertex"` it is not
  supported.
- `jno.solve.simple` iterates eagerly. Inside `jno.core`, keep the default Newton.
- The automatic wave speed costs `O(n³)` per face for an `n`-component system. Give `speed=` for large ones.
- A steady problem whose solution is fixed only by exponentially small terms (a thin stationary shock between
  two Dirichlet values) has a numerically singular Jacobian. March it in time instead.

## References

- Moukalled, Mangani & Darwish (2016), *The Finite Volume Method in Computational Fluid Dynamics*, Springer.
- LeVeque (2002), *Finite Volume Methods for Hyperbolic Problems*, CUP.
- Toro (2009), *Riemann Solvers and Numerical Methods for Fluid Dynamics*, Springer.
- Darwish & Moukalled (2003), *Int. J. Heat Mass Transfer* 46, 599–611.
- Barth & Jespersen (1989), AIAA 89-0366.
- Venkatakrishnan (1995), *J. Comput. Phys.* 118, 120–130.
- Gottlieb, Shu & Tadmor (2001), *SIAM Review* 43, 89–112.
- Spiteri & Ruuth (2002), *SIAM J. Numer. Anal.* 40, 469–491.
- Berberich, Chandrashekar & Klingenberg (2021), *Comput. Fluids* 219.
- Rhie & Chow (1983), *AIAA J.* 21, 1525–1532.
- Patankar & Spalding (1972), *Int. J. Heat Mass Transfer* 15, 1787–1806.
- Issa (1986), *J. Comput. Phys.* 62, 40–65.
- Ghia, Ghia & Shin (1982), *J. Comput. Phys.* 48, 387–411.
