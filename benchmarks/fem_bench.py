"""A fixed set of representative jNO FEM problems, timed end to end.

Run ONE case per process (``python fem_bench.py <case> <size_index>``) and it prints a single JSON
line. That isolation is not fussiness: running several cases in one process lets later ones inherit
earlier XLA compilations, which silently flatters whatever runs last -- a bias that produced wrong
numbers more than once while this suite was being written. :mod:`run_bench` drives the matrix.

Each case reports the split that matters for jNO, where assembly compiles a program per problem:

* ``build_ms``  -- ``jno.domain(...)`` + ``jno.fem(...)``, i.e. meshing, assembly and XLA compilation
* ``solve_ms``  -- the solve alone, on an already-built problem
* ``peak_mb``   -- peak device memory over the whole case

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

THREE CASES CANNOT REACH A 10 s SOLVE ON AN 8 GB CARD, and are included at their ceiling:

* ``stokes2d``  -- walled by the SOLVER. At mesh_size 0.04 the saddle-point system comes back
  "Singular matrix in linear solve" from cuSolver's sparse LU, and an iterative outer does not
  converge on it either (fgmres stalls at 2.8e-3). 0.045 works and is the largest size here: ~1.8 s.
* ``poisson3d`` -- walled by MEMORY. Even at 8.9e6 contrast the solve is 0.20 s at 98k DOFs; a 10 s
  solve needs ~1.3M tets, which is ~10 GB of peak device memory and an ~8 minute build.
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
    return fem, {"linear": jno.solve.lu()}


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
    return fem, {"linear": jno.solve.lu()}


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
    "elastic2d": (elastic2d, (0.006, 0.0045, 0.0034, 0.0026, 0.002, 0.0015), "2-D vector (convergence-walled)"),
    "reaction2d": (reaction2d, (0.008, 0.0062, 0.0048, 0.0037, 0.0028, 0.0022), "2-D nonlinear (Newton)"),
    "heat2d": (heat2d, (0.02, 0.0167, 0.0139, 0.0115, 0.0096, 0.008), f"2-D transient ({HEAT_STEPS} steps)"),
    "stokes2d": (stokes2d, (0.09, 0.078, 0.068, 0.059, 0.052, 0.045), "2-D Stokes (solver-walled)"),
    "poisson3d": (poisson3d, (0.05, 0.042, 0.035, 0.029, 0.024, 0.02), "3-D Poisson (memory-walled)"),
    "eddy3d": (eddy3d, (0.16, 0.141, 0.124, 0.109, 0.096, 0.085), "3-D complex H(curl)"),
}

#: Cases whose solution is a TRAJECTORY. ``np.size`` would report steps x nodes and put the case
#: ~3 decades right of everything else on a DOF axis; the system actually solved is one step's.
TRAJECTORY_STEPS = {"heat2d": HEAT_STEPS}


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
            if not rel < 1e-6:
                raise RuntimeError(f"forward solve did not converge (relative residual {rel:.3g})")
        solve_ms = extra.pop("solve_ms")
        dofs = int(np.asarray(fem.operator.evaluate({"alpha": 1.0})[1]).size)
    else:
        t0 = time.perf_counter()
        sol = _value(fem.solve(**kw))
        solve_ms = (time.perf_counter() - t0) * 1e3
        dofs = int(np.size(sol)) // TRAJECTORY_STEPS.get(case, 1)

    return {
        **extra,
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
