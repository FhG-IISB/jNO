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
"""

from __future__ import annotations

import json
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402

PI = np.pi
inner, vec, grad, trace = jno.np.inner, jno.np.vector, jno.np.grad, jno.np.trace


def _sq(ms):
    from shapely.geometry import box

    return jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=ms)


# --------------------------------------------------------------------------------------------
# problems: each returns (fem, solve_kwargs)
# --------------------------------------------------------------------------------------------
def poisson2d(ms):
    """2-D scalar Poisson, P1 — the reference point every other case is read against."""
    d = _sq(ms)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3), {}


def elastic2d(ms):
    """2-D VECTOR field — the element block grows with local DOFs, and so does its AD tangent."""
    d = _sq(ms)
    u, v = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    body = inner(grad(ui, [xi, yi]), grad(vi, [xi, yi]), n_contract=2) - 1.0 * vi[1]
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


def heat2d(ms):
    """2-D TRANSIENT heat, 41 steps — the whole trajectory compiles into one scan."""
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=ms, time=(0.0, 0.2, 41))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic]), {}


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
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(cb[0], cb[1], cb[2]) - g], element_type="TET4")
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


#: case -> (builder, mesh sizes, human label). Sizes chosen to span roughly a decade of DOFs.
CASES = {
    "poisson2d": (poisson2d, (0.05, 0.02, 0.012), "2-D Poisson (P1)"),
    "elastic2d": (elastic2d, (0.06, 0.03, 0.018), "2-D vector"),
    "reaction2d": (reaction2d, (0.06, 0.03, 0.018), "2-D nonlinear (Newton)"),
    "heat2d": (heat2d, (0.06, 0.03, 0.018), "2-D transient (41 steps)"),
    "stokes2d": (stokes2d, (0.4, 0.25, 0.18), "2-D Stokes (Taylor-Hood)"),
    "poisson3d": (poisson3d, (0.15, 0.11, 0.085), "3-D Poisson (tets)"),
    "eddy3d": (eddy3d, (0.3, 0.22, 0.17), "3-D complex H(curl)"),
}


def _peak_mb():
    try:
        return (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e6
    except Exception:
        return float("nan")


def _value(out):
    # a TRANSIENT solve returns a lazy thunk; anything else is already an array
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


def main(case: str, idx: int) -> dict:
    builder, sizes, label = CASES[case]
    ms = sizes[idx]

    t0 = time.perf_counter()
    fem, kw = builder(ms)
    build_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    sol = _value(fem.solve(**kw))
    solve_ms = (time.perf_counter() - t0) * 1e3

    return {
        "case": case,
        "label": label,
        "mesh_size": ms,
        "dofs": int(np.size(sol)),
        "build_ms": round(build_ms, 1),
        "solve_ms": round(solve_ms, 1),
        "peak_mb": round(_peak_mb(), 1),
        "backend": jax.default_backend(),
    }


if __name__ == "__main__":
    print("RESULT " + json.dumps(main(sys.argv[1], int(sys.argv[2]))))
