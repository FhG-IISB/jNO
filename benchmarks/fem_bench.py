"""A fixed set of representative jNO FEM problems, timed end to end.

Run ONE case per process (``python fem_bench.py <case> <size_index>``) and it prints a single JSON
line. That isolation is not fussiness: running several cases in one process lets later ones inherit
earlier XLA compilations, which silently flatters whatever runs last -- a bias that produced wrong
numbers more than once while this suite was being written. :mod:`run_bench` drives the matrix.

Each case reports the split that matters for jNO, where assembly compiles a program per problem:

* ``build_ms``  -- ``jno.domain(...)`` + ``jno.fem(...)``, i.e. meshing, assembly and XLA compilation
* ``solve_ms``  -- the solve alone, on an already-built problem
* ``peak_mb``   -- peak device memory over the whole case

and two things that decide whether those numbers mean anything:

* ``rel_residual`` / ``converged`` -- the correctness gate. A timing taken on a solve that never
  converged measures nothing, and without this the suite had no way to notice. It is not
  hypothetical: the first case to get a gate immediately caught an unpreconditioned solve burning
  ``maxiter``, which had been timing as a gradient FASTER than its own forward. ``None`` means the
  residual is not defined for that case (a transient has a per-step system, not one system).
* provenance -- ``gpu``, ``sm_clock``, ``jax``, ``git``, ``dirty``, ``when``. Stamped per RECORD,
  not per file, because ``run_bench`` merges across runs: one ``results.json`` routinely holds
  points measured at different commits, and a file-level header would misdescribe most rows.

Iteration counts are NOT recorded and would be the most useful thing to add next -- they would have
made the non-convergence above obvious on sight. jNO's Krylov solvers do not return them, so it
needs a library API change rather than a benchmark change.

The set spans the axes that change jNO's behaviour rather than just the physics: scalar vs vector,
linear vs Newton, steady vs transient, single-field vs saddle-point, 2-D vs 3-D, real vs complex, and
nodal Lagrange vs H(curl) edge elements.

WHY THE COEFFICIENTS ARE HETEROGENEOUS
--------------------------------------
Several cases carry a log-uniform coefficient ``k = exp(c sin.. sin..)``, spanning ``exp(-c)`` to
``exp(+c)``. That is not decoration: on a CONSTANT-coefficient Laplacian the default
Jacobi-BiCGStab converges so fast that the solve is unmeasurable next to its own compilation --
3-D Poisson at 31k DOFs solves in 10 ms while the first call reports 250 ms, i.e. 96% XLA
compilation. A heterogeneous coefficient raises the iteration count until the solve is genuinely
what is being timed, which is the only honest way to make these cases solver-bound.

The contrast is capped by breakdown, not by taste. At 1.6e5 contrast the 2-D Poisson case stops
converging at 805k DOFs (relative residual 1.4e+00 -- BiCGStab breaking down, so raising ``maxiter``
does not help), which is why that one case runs at 1.1e3 instead.

``stokes2d`` and ``eddy3d`` USE ``jno.solve.lu(host=True)`` -- SuperLU factoring in host memory,
driven from the device. Their previous ceilings were cuSolver's, not the problems': cuSolver
returned "Singular matrix" on Stokes at mesh_size 0.04 and failed outright on H(curl) at 0.06, both
of which the host factors to a ~2e-15 relative residual. On this box the host path is also faster
wherever both run (0.15-0.81x of cuSolver over 12 measured points), though that is hardware-specific
-- see :func:`jno.solve.lu`. With it, Stokes reaches 146k DOFs and H(curl) 17k complex DOFs, and
both clear the 10 s target that neither could before.

TWO CASES STILL DO NOT REACH A 10 s SOLVE, and are included at their ceiling:

* ``poisson3d`` -- never becomes SOLVE-BOUND. Its solve is flat at 0.28-0.35 s from 6.8k to 87k
  DOFs, i.e. essentially independent of size, while its build grows to 8.4 s: 96% of the wall clock
  is build. It is NOT memory-walled -- an earlier claim to that effect was an artefact of a bug in
  this suite's own convergence gate, which densified the operator and OOM'd every size above 20k.
  With that fixed the case reaches 87k DOFs in 494 MB.
* ``elastic2d`` -- walled by CONVERGENCE, and instructively so. At 1.6e5 contrast it does reach 5-12 s,
  but only sometimes: the same mesh that converged in an isolated run failed inside the matrix. That
  is BiCGStab sitting on the edge of breakdown, where GPU reduction order is enough to tip it. A
  benchmark point that reports a time only when it feels like it is worse than a slow one, so this
  case runs at the mild contrast and reports ~3 s at 1.03M DOFs -- the largest size the card holds.
"""

from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402

PI = np.pi
inner, vec, grad, trace = jno.np.inner, jno.np.vector, jno.np.grad, jno.np.trace


def _sq(ms):
    from shapely.geometry import box

    return jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=ms)


def _rough(c, *coords, freq=6):
    """Log-uniform coefficient over the unit cell, spanning exp(-c) to exp(+c).

    Smooth and strictly positive, so the problem stays well posed while the condition number --
    and with it the Krylov iteration count -- rises with ``c``. See the module docstring for why
    the cases need it and where it breaks down.
    """
    field = 1.0
    for x in coords:
        field = field * jno.np.sin(freq * PI * x)
    return jno.np.exp(c * field)


#: log-contrast per case. exp(2*C) is the coefficient ratio: C_MILD -> 1.1e3, C_HARD -> 1.6e5.
C_MILD, C_HARD = 3.5, 6.0

#: transient steps. The trajectory is what the solve returns, so this also sets its memory.
HEAT_STEPS = 1500

#: Rayleigh-Benard steps. Far fewer than the heat case: every one is a Newton solve on the full
#: three-field coupled block, not a single linear solve.
BOUSSINESQ_STEPS = 26


# --------------------------------------------------------------------------------------------
# problems: each returns (fem, solve_kwargs)
# --------------------------------------------------------------------------------------------
def poisson2d(ms):
    """2-D scalar Poisson, P1 — the reference point every other case is read against.

    Runs at the MILD contrast: 1.6e5 breaks BiCGStab down at this case's largest size.
    """
    d = _sq(ms)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = _rough(C_MILD, xi, yi)
    return jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3), {}


def poisson2d_adj(ms):
    """The GRADIENT of a scalar loss through the same 2-D Poisson solve — an inverse-problem step.

    Uses the raw-AD entry: the form carries a ``jno.np.parameter``, the operator is assembled once
    and the parameter supplied per call via ``fem.operator.evaluate({name: value})``. Gradients flow
    by implicit differentiation through ``lax.custom_linear_solve``; there is no adjoint flag.

    NOT the other differentiable entry point. ``fem.solve()`` on a parametric form returns a lazy
    trace node rather than an array, so plain ``jax.grad`` over it does not apply -- that route is
    ``crux.solve(n)``. And a gradient that rebuilds ``jno.fem(...)`` INSIDE the differentiated
    function cannot be jitted at all: ``_build_dirichlet_pairs`` concretizes a traced coefficient.

    The solver here is an explicit BiCGStab on the assembled operator, not the slot-composed default
    the forward cases use, so the gap to the solid ``poisson2d`` curve is not purely adjoint cost.
    The honest overhead is against this case's OWN forward, recorded as ``forward_ms``.
    """
    d = _sq(ms)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    k = _rough(C_MILD, xi, yi)
    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="alpha")
    fem = jno.fem([alpha * k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)
    system = fem.operator

    def _solve(theta):
        A, b = system.evaluate({"alpha": theta})
        rhs = jnp.asarray(b).reshape(-1)
        # Jacobi-preconditioned, to match what the forward cases run. An UNPRECONDITIONED BiCGStab
        # does not converge on this contrast past ~500k DOFs -- it silently burns maxiter instead,
        # which made the gradient time look SMALLER than its own forward. That is impossible for
        # reverse mode (which computes the primal AND an adjoint solve), and was the tell.
        rows, cols = A.indices[:, 0], A.indices[:, 1]
        diag = jnp.zeros(rhs.size, rhs.dtype).at[rows].add(jnp.where(rows == cols, A.data, 0.0))
        inv = jnp.where(diag == 0, 1.0, 1.0 / jnp.where(diag == 0, 1.0, diag))
        x, _ = jax.scipy.sparse.linalg.bicgstab(lambda z: A @ z, rhs, M=lambda z: z * inv, tol=1e-8, maxiter=4000)
        return x, A, rhs

    def loss(theta):
        return jnp.sum(_solve(theta)[0] ** 2)

    def residual(theta):
        """Relative residual of the forward solve — the gate that catches a non-converged timing."""
        x, A, rhs = _solve(theta)
        return jnp.linalg.norm(A @ x - rhs) / jnp.linalg.norm(rhs)

    return fem, {"grad": loss, "check": residual}


def poisson2d_amg(ms):
    """The same 2-D Poisson, solved with GPU ALGEBRAIC MULTIGRID instead of Jacobi-BiCGStab.

    The one case here that varies the SOLVER rather than the problem, because that is where the
    scaling exponent lives. Jacobi-preconditioned Krylov needs iterations growing like sqrt(n);
    multigrid needs O(1) of them, so the two curves differ in slope, not just offset -- measured
    n^1.28 against n^0.74, which is 4.8x at 73k DOFs and 17.6x at 805k.

    Run at tol=1e-10 deliberately: at its 1e-6 default AMG returns a residual ~35x looser than the
    Jacobi path's, and a speed comparison against a slacker convergence criterion is not a
    comparison. At 1e-10 it lands ~100x TIGHTER than the default solver and is still 17.6x faster.

    READ THIS CURVE AS SETUP, NOT ITERATION. Like every case here it reports the FIRST solve, and
    AMG's is dominated by building the multigrid hierarchy: it is flat at ~1.1 s from 73k to 959k
    DOFs, where the warm solve underneath is 0.03-0.16 s. So the curve is a setup cost with the
    algorithm hidden inside it, and it crosses the Jacobi line near ~400k DOFs -- below that the
    hierarchy costs more than it saves on a single cold solve, above it AMG wins and keeps winning
    (3.4x at 959k). For repeated solves on one operator the setup amortises and the warm numbers
    apply instead, which is where the 17.6x lives.

    Needs the optional ``jaxamg`` extra plus a built AmgX (``AMGX_ROOT``, ``CUDA_HOME`` and AmgX on
    ``LD_LIBRARY_PATH``); without them this case records a failure and the rest of the matrix runs.
    """
    d = _sq(ms)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = _rough(C_MILD, xi, yi)
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)
    return fem, {"linear": jno.solve.amg(tol=1e-10)}


def elastic2d(ms):
    """2-D VECTOR field — the element block grows with local DOFs, and so does its AD tangent."""
    d = _sq(ms)
    u, v = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = _rough(C_MILD, xi, yi)  # see the module docstring: C_HARD is where BiCGStab starts breaking down
    body = k * inner(grad(ui, [xi, yi]), grad(vi, [xi, yi]), n_contract=2) - 1.0 * vi[1]
    return jno.fem([body, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0], quad_degree=3), {}


def reaction2d(ms):
    """2-D NONLINEAR reaction-diffusion — Newton re-enters the element loop every iteration."""
    d = _sq(ms)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    ss = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    f = 2.0 * PI**2 * ss + ss**3
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui**3 * vi - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    return fem, {"nonlinear": jno.solve.newton()}


def heat2d(ms, steps=HEAT_STEPS):
    """2-D TRANSIENT heat — the whole trajectory compiles into one scan, so the cost is steps x
    per-step solve while the compilation is paid once.

    Both knobs are needed and both are capped. Steps alone flatten out (the scan amortises the fixed
    cost: 201 steps 1.3 s, 1500 steps only 3.3 s) and the stored trajectory is what runs out of
    memory -- 6000 steps at this mesh OOMs an 8 GB card. Heterogeneity carries the rest.
    """
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=ms, time=(0.0, 0.2, steps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    k = _rough(C_HARD, xi, yi)
    return jno.fem([ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic]), {}


def stokes2d(ms):
    """2-D MULTIPHYSICS: Taylor-Hood Stokes — two fields at mixed order, a saddle-point system."""
    from shapely.geometry import box

    G, mu, H, Lx = 1.0, 1.0, 1.0, 4.0
    d = jno.domain(box(0.0, 0.0, Lx, H), mesh_size=ms)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            mu * inner(gu, gv, n_contract=2) - pp * trace(gv),
            -qq * trace(gu),
            u(xb, yb)[0] - (G / (2 * mu)) * yb * (H - yb),
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )
    return fem, {"linear": jno.solve.lu(host=True)}


def boussinesq2d(ms, steps=BOUSSINESQ_STEPS):
    """2-D Rayleigh-Benard convection — the hardest system here: THREE fields, two-way NONLINEAR
    coupling, a saddle point, and transient, all at once.

    Incompressible Navier-Stokes with a buoyancy body force, two-way coupled to advection-diffusion
    for heat (Boussinesq)::

        du/dt + (u.grad)u = -grad p + Pr lap u + Pr*Ra*T e_y
        div u = 0
        dT/dt + u.grad T  = lap T

    Velocity P2 (vector), pressure P1, temperature P1 — mixed order AND mixed rank in one system.
    What makes it the stiffest case in this suite is that no single difficulty dominates: the
    incompressibility constraint makes it indefinite (as Stokes is), the buoyancy term ``Pr*Ra*T``
    couples heat into momentum, and ``u.grad T`` is a product of two DIFFERENT unknowns, so the
    system is genuinely nonlinear rather than merely parametric. Every step is a Newton solve on the
    full coupled block, and it marches in time on top of that.

    Physics and formulation taken from the ``rayleigh_benard_2d`` tutorial (Ra = 1e4, well above the
    critical ~1708, so the rolls actually form) rather than rewritten, so the case is one a reader
    can look up in validated form.

    It is the most SOLVE-bound case in the suite by a wide margin -- 96% of the wall clock at its
    largest size, against 4-8% for the steady linear cases -- and it gets there at only 6,574 DOFs,
    two to three orders of magnitude fewer than the Poisson ladders need. That is the honest cost of
    coupling: 26 Newton solves on a three-field indefinite block cost 61 s where a scalar Poisson
    solve of 1.8M DOFs costs 13 s. Solve time measures n^0.76, so the expense is per-step Newton
    work rather than anything that scales badly.
    """
    Pr, Ra, Lx, Ly = 1.0, 1.0e4, 2.0, 1.0
    dt = 0.009
    d = jno.Shape.rect(0, 0, Lx, Ly, size=ms).domain(time=(0.0, steps * dt, steps + 1))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    T, sT = d.fem_symbols(names=("T", "sT"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pb, qb = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    Tb, sb = T.bind(x=xi, y=yi, t=ti), sT.bind(x=xi, y=yi, t=ti)

    ux, uy, vx, vy = ub[0], ub[1], vb[0], vb[1]
    uxx, uxy, uyx, uyy = ub.x[0], ub.y[0], ub.x[1], ub.y[1]
    vxx, vxy, vyx, vyy = vb.x[0], vb.y[0], vb.x[1], vb.y[1]
    momentum = (
        (ub.t[0] * vx + ub.t[1] * vy)
        + ((ux * uxx + uy * uxy) * vx + (ux * uyx + uy * uyy) * vy)  # (u.grad)u -- nonlinear
        + Pr * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)
        - pb * (vxx + vyy)
        - Pr * Ra * Tb * vy  # buoyancy: temperature -> momentum
    )
    continuity = qb * (uxx + uyy)
    energy = Tb.t * sb + (ux * Tb.x + uy * Tb.y) * sb + (Tb.x * sb.x + Tb.y * sb.y)
    Tcond = 1.0 - ci[1] / Ly
    T0 = Tcond + 0.05 * jno.np.sin(2 * PI * ci[0] / Lx) * jno.np.sin(PI * ci[1] / Ly)
    fem = jno.fem(
        [
            momentum,
            continuity,
            energy,
            u(xb, yb) - 0.0,
            T(xb, yb) - (1.0 - yb / Ly),
            p.pin(),
            u(ci[0], ci[1]) - 0.0,
            T(ci[0], ci[1]) - T0,
        ]
    )
    return fem, {}


def poisson3d(ms):
    """3-D scalar Poisson on tets — ~2x the per-DOF cost of 2-D, from bigger element programs."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=ms).domain()
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    u, v = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    g = 1.0 + 2.0 * cb[0] + 3.0 * cb[1] + 4.0 * cb[2]
    k = _rough(C_HARD, xi, yi, zi, freq=4)
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z), u(cb[0], cb[1], cb[2]) - g], element_type="TET4")
    return fem, {}


def eddy3d(ms):
    """3-D COMPLEX H(curl) eddy current, N1E edge elements — the hardest shipped case.

    Solved directly: the default Jacobi-BiCGStab does not converge on curl-curl at all (the gradient
    null-space), so a direct solve is the honest baseline. It is also the ceiling — cuSolver fails
    above ~20k complex DOFs, which is why AMS exists.
    """
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=ms).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    ui, vi = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cu, cv = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    sig = jno.np.where(x > 0.5, 1.0, 0.0)
    fem = jno.fem(
        [inner(cu, cv) + 1j * 2 * PI * 1e3 * (sig + 1e-3) * inner(ui, vi) - inner(vec(0.0 * x, 0.0 * x, sig), vi)]
    )
    return fem, {"linear": jno.solve.lu(host=True)}


def eddy3d_ams(ms):
    """The same complex H(curl) eddy problem, solved ITERATIVELY with an AMS preconditioner.

    The sharpest algorithmic contrast in this suite, and jNO's own capability: ``jno.precond.ams``
    is the only JAX-native auxiliary-space Maxwell preconditioner there is. Plain AMG cannot
    precondition a curl-curl operator at all -- the gradient null-space defeats it -- and
    Jacobi-BiCGStab diverges outright, which is why the forward case resorts to a direct solve.

    Direct factorisation of a 3-D curl-curl system scales at a measured n^2.19 (fill-in); AMS with a
    flexible outer Krylov scales at n^0.71. They cross near ~6k DOFs: below it the direct solve wins,
    above it AMS pulls away -- 4.3x at 10.8k DOFs, 7.2x at 17.1k, 17.7x at 32.7k, where the direct
    solve needs 86 s and AMS needs 4.85 s.

    ``.build(fem)`` freezes the auxiliary operators against this problem, which is what puts the
    preconditioner on the compiled path; FGMRES because an AMS apply is not a fixed linear operator.
    Converges to ~8e-09 rather than the direct solve's ~2e-15 -- an iterative tolerance, not a
    factorisation, so the two are not equally exact and the residual is recorded per point.
    """
    fem, _ = eddy3d(ms)
    return fem, {"linear": jno.solve.fgmres(), "precond": jno.precond.ams().build(fem)}


#: case -> (builder, mesh sizes, human label).
#:
#: Sized so the SOLVE carries the wall clock rather than the build. That is the regime worth
#: measuring -- a problem small enough for a per-problem XLA compilation to dominate says more about
#: the compiler than about the solver. Roughly a decade of DOFs per case, chosen to land each largest
#: size in tens of seconds.
#:
#: Six sizes per case, spaced GEOMETRICALLY in mesh size. Element count goes as h^-2 in 2-D and h^-3
#: in 3-D, so a geometric ladder in h is an even one in DOFs -- which is what the log-log scaling
#: panel is read against. An arithmetic ladder would bunch every point at the coarse end.
#:
#: Two cases are heavy for a reason other than mesh size. The transient runs 201 steps, because its
#: cost is steps x per-step solve and the trajectory compiles into a single scan either way. The
#: complex H(curl) case is solved directly and gets expensive fast: LU fill-in in 3-D was measured at
#: 81x, and cuSolver fails outright somewhere above ~20k complex DOFs -- if the largest size here
#: reports a failure, that ceiling is the finding, not a broken benchmark.
#:
#: Each ladder's LARGEST size was measured to put the solve at or above 10 s, except the two named
#: in the module docstring, which are at their hardware/solver ceiling instead.
CASES = {
    "poisson2d": (poisson2d, (0.004, 0.0029, 0.0021, 0.0015, 0.0011, 0.0008), "2-D Poisson (1e3 contrast)"),
    "poisson2d_adj": (
        poisson2d_adj,
        (0.004, 0.0029, 0.0021, 0.0015, 0.0011),
        "2-D Poisson adjoint ($\\partial/\\partial\\alpha$)",
    ),
    "poisson2d_amg": (
        poisson2d_amg,
        (0.004, 0.0029, 0.0021, 0.0015, 0.0011, 0.0008),
        "2-D Poisson, GPU AMG",
    ),
    "elastic2d": (elastic2d, (0.006, 0.0045, 0.0034, 0.0026, 0.002, 0.0015), "2-D vector (convergence-walled)"),
    "reaction2d": (reaction2d, (0.008, 0.0062, 0.0048, 0.0037, 0.0028, 0.0022), "2-D nonlinear (Newton)"),
    "heat2d": (heat2d, (0.02, 0.0167, 0.0139, 0.0115, 0.0096, 0.008), f"2-D transient ({HEAT_STEPS} steps)"),
    "boussinesq2d": (
        boussinesq2d,
        (0.16, 0.13, 0.11, 0.09, 0.075, 0.062),
        "2-D Rayleigh-Benard (3 fields, nonlinear)",
    ),
    "stokes2d": (stokes2d, (0.09, 0.063, 0.044, 0.031, 0.021, 0.015), "2-D Stokes (Taylor-Hood)"),
    "poisson3d": (poisson3d, (0.05, 0.042, 0.035, 0.029, 0.024, 0.02), "3-D Poisson (build-bound)"),
    "eddy3d": (eddy3d, (0.16, 0.136, 0.115, 0.097, 0.083, 0.07), "3-D complex H(curl)"),
    "eddy3d_ams": (
        eddy3d_ams,
        (0.16, 0.124, 0.096, 0.075, 0.058, 0.045),
        "3-D H(curl), AMS iterative",
    ),
}

#: Cases whose solution is a TRAJECTORY. ``np.size`` would report steps x nodes and put the case
#: ~3 decades right of everything else on a DOF axis; the system actually solved is one step's.
TRAJECTORY_STEPS = {"heat2d": HEAT_STEPS, "boussinesq2d": BOUSSINESQ_STEPS + 1}


#: a solve whose residual is above this was not a solve, and its timing measures nothing. Recorded
#: rather than raised, so the matrix completes and the bad point is visible instead of missing.
RESIDUAL_TOL = 1e-6

#: Separate, looser threshold for the TRANSIENT step gate, because it measures a different thing.
#: A direct or Krylov solve of one system lands at 1e-14; a per-step Newton lands at its own
#: convergence tolerance, measured 3e-07 to 6e-07 across the Rayleigh-Benard ladder. Judging that
#: against 1e-6 would flag a perfectly converged solve the moment it drifted to 2e-6 -- a gate that
#: cries wolf is worse than none. The two regimes are far apart (converged ~1e-7, a trajectory
#: perturbed 5% gives 6e-01), so this sits between them with three orders of margin either side.
TRANSIENT_RESIDUAL_TOL = 1e-4


def _relative_residual(fem, sol, case):
    """Convergence gate: how far the reported answer actually is from solving the system.

    Returns ``(value, how)``; a ``None`` value carries the reason it is not defined. This exists
    because the suite had no correctness check at all, and the first case that got one immediately
    caught a solve that was silently burning ``maxiter`` and being timed as though it had converged.
    """
    if case in TRAJECTORY_STEPS:
        return _transient_step_residual(fem, np.asarray(sol))
    u = jnp.asarray(sol).reshape(-1)
    if not fem.is_linear:
        # relative to the residual at zero, so the number is comparable across sizes
        r0 = float(jnp.linalg.norm(jnp.asarray(fem.residual(jnp.zeros_like(u))).reshape(-1)))
        r = float(jnp.linalg.norm(jnp.asarray(fem.residual(u)).reshape(-1)))
        return r / max(r0, 1e-300), "||R(u)|| / ||R(0)||"
    # ``fem.operator``, NOT ``fem.A``: the latter is a densifying convenience property, and calling
    # it here allocated n^2 -- 71.9 GiB at 98k DOFs. That OOM'd 18 of 47 points and inflated peak_mb
    # on every survivor, i.e. the correctness gate silently wrecked the measurement it guards.
    A, b = fem.operator
    b = jnp.asarray(b).reshape(-1)
    if jnp.iscomplexobj(u) and b.size == 2 * u.size:
        u = jnp.concatenate([u.real, u.imag])  # complex assembles as the real 2n block
    return float(jnp.linalg.norm(A @ u - b) / jnp.linalg.norm(b)), "||Au-b|| / ||b||"


def _transient_step_residual(fem, traj):
    """Worst per-step residual over a transient trajectory — the gate for a time-marched solve.

    A transient has no single system to residual, which left the nonlinear coupled case (the one
    where an under-converged Newton is MOST likely and least visible) with no correctness check at
    all. This checks the equation each step is supposed to have solved: for backward Euler,

        G(u_n) = M(t_n) (u_n - u_(n-1)) / dt + R(u_n, t_n) = 0

    evaluated on the rows NOT pinned by a Dirichlet condition -- the pinned rows are row-replaced by
    the constraint, so their residual is meaningless here. Reported relative to ||R(u_n)|| and
    maximised over steps, so one bad step cannot hide behind many good ones.

    Validated to discriminate: a converged Rayleigh-Benard solve gives 2.8e-07, and the same
    trajectory perturbed by 5% gives 6.3e-01.
    """
    blk = fem.operator
    theta = float((getattr(blk, "metadata", None) or {}).get("theta", 1.0))
    if theta != 1.0:
        return None, f"theta={theta}: this gate implements backward Euler only"
    if getattr(blk, "mass", None) is None or getattr(blk, "residual", None) is None:
        return None, "linear transient: block carries M/A rather than mass+residual callables"

    dt = float(blk.dt)
    pairs = getattr(getattr(fem, "domain", None), "_fem_native_dirichlet_pairs", None) or []
    free = np.ones(traj.shape[1], dtype=bool)
    if pairs:
        free[np.asarray([int(dof) for dof, _v in pairs])] = False

    worst = 0.0
    for n in range(1, traj.shape[0]):
        t_n = blk.t0 + n * dt
        u_n, u_prev = jnp.asarray(traj[n]), jnp.asarray(traj[n - 1])
        residual = jnp.asarray(blk.residual(u_n, t_n, {})).reshape(-1)
        g = np.asarray(jnp.asarray(blk.mass(t_n, {}) @ ((u_n - u_prev) / dt)).reshape(-1) + residual)
        scale = max(float(np.linalg.norm(np.asarray(residual)[free])), 1e-30)
        worst = max(worst, float(np.linalg.norm(g[free]) / scale))
    return worst, f"max_n ||M(u_n-u_(n-1))/dt + R(u_n)|| / ||R(u_n)|| over {int(free.sum())} free rows"


def _provenance() -> dict:
    """Machine and code identity, stamped on EVERY record.

    Per-record rather than once per file on purpose: ``run_bench`` merges results across runs, so
    one ``results.json`` routinely holds points measured at different commits on different days.
    A single file-level header would be a lie about most of the rows.
    """
    import subprocess

    def sh(cmd):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, shell=True)
            return out.stdout.strip() or None
        except Exception:
            return None

    return {
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
        # sampled at process start: this card idles at 285 MHz against a 2100 MHz max, so a cold
        # start and a thermally-loaded one are not the same machine
        "sm_clock": sh("nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader"),
        "jax": jax.__version__,
        "git": sh("git rev-parse --short HEAD"),
        "dirty": bool(sh("git status --porcelain")),
        "x64": bool(jax.config.jax_enable_x64),
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _peak_mb():
    try:
        return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        return float("nan")


def _value(out):
    # a TRANSIENT solve returns a lazy thunk; anything else is already an array
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


def _timed_gradient(loss) -> dict:
    """Time one gradient evaluation, and the forward it is the derivative of.

    ``solve_ms`` is the FIRST call, compile included, exactly as every forward case reports it --
    consistency inside one figure matters more than the cleaner warm number. The warm pair is
    recorded alongside, and it is the one to quote for adjoint overhead.
    """
    theta = jnp.asarray(1.0)
    fwd, grad = jax.jit(loss), jax.jit(jax.grad(loss))

    t0 = time.perf_counter()
    float(fwd(theta))
    forward_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    float(grad(theta))
    solve_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    float(fwd(theta))
    forward_warm_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    float(grad(theta))
    grad_warm_ms = (time.perf_counter() - t0) * 1e3
    return {
        "solve_ms": round(solve_ms, 1),
        "forward_ms": round(forward_ms, 1),
        "forward_warm_ms": round(forward_warm_ms, 1),
        "grad_warm_ms": round(grad_warm_ms, 1),
        "adjoint_ratio": round(grad_warm_ms / max(forward_warm_ms, 1e-9), 3),
    }


def main(case: str, idx: int) -> dict:
    builder, sizes, label = CASES[case]
    ms = sizes[idx]

    t0 = time.perf_counter()
    fem, kw = builder(ms)
    build_ms = (time.perf_counter() - t0) * 1e3

    extra = {}
    loss, check = kw.pop("grad", None), kw.pop("check", None)
    if loss is not None:
        extra = _timed_gradient(loss)
        if check is not None:
            # a timing on a non-converged solve is not a measurement of anything
            rel = float(jax.jit(check)(jnp.asarray(1.0)))
            extra["rel_residual"] = float(f"{rel:.3g}")
            extra["residual_of"] = "||Au-b|| / ||b||"
        solve_ms = extra.pop("solve_ms")
        dofs = int(np.asarray(fem.operator.evaluate({"alpha": 1.0})[1]).size)
    else:
        t0 = time.perf_counter()
        sol = _value(fem.solve(**kw))
        solve_ms = (time.perf_counter() - t0) * 1e3
        dofs = int(np.size(sol)) // TRAJECTORY_STEPS.get(case, 1)
        rel, how = _relative_residual(fem, sol, case)
        extra = {"rel_residual": None if rel is None else float(f"{rel:.3g}"), "residual_of": how}

    rel = extra.get("rel_residual")
    tol = TRANSIENT_RESIDUAL_TOL if case in TRAJECTORY_STEPS else RESIDUAL_TOL
    return {
        **extra,
        "residual_tol": tol,
        "converged": None if rel is None else bool(rel < tol),
        **_provenance(),
        "case": case,
        "label": label,
        "mesh_size": ms,
        "dofs": dofs,
        "build_ms": round(build_ms, 1),
        "solve_ms": round(solve_ms, 1),
        "peak_mb": round(_peak_mb(), 1),
        "backend": jax.default_backend(),
    }


if __name__ == "__main__":
    print("RESULT " + json.dumps(main(sys.argv[1], int(sys.argv[2]))))
