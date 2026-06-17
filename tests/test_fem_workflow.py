"""End-to-end ``jno.fem`` workflows and edge cases — beyond "assembles + symmetric".

These solve the assembled system and check correctness against known solutions:
Neumann and Robin natural BCs, transient time-stepping vs the analytic decay, a
nonlinear steady Newton solve, and a non-box (CSG) domain. Plus edge cases that
must error cleanly: a region-ambiguous residual, a residual with neither the
trial nor the test, and a non-affine essential condition.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import box  # noqa: E402


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _solve(fem):
    return np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))


# --------------------------------------------------------------------------
# natural BCs recovered against a known solution (u = x)
# --------------------------------------------------------------------------
def test_neumann_recovers_linear_solution():
    # -lap u = 0 ; u=0 on left ; du/dn=1 on right ; natural (zero-flux) top/bottom => u = x.
    # The residual boundary term for a flux g_N is -g_N * phi.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, -1.0 * phi.bind(x=xr, y=yr), u(xl, yl) - 0.0])
    sol = _solve(fem)
    c = np.asarray(d.mesh.points)[:, :2]
    assert np.linalg.norm(c[:, 0] - sol) / np.linalg.norm(c[:, 0]) < 1e-8


def test_robin_recovers_linear_solution():
    # du/dn + a u = g on right, with g = 1 + a so that u = x is exact; u=0 on left.
    a = 2.0
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    robin = (a * u.bind(x=xr, y=yr) - (1.0 + a)) * phi.bind(x=xr, y=yr)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, robin, u(xl, yl) - 0.0])
    sol = _solve(fem)
    c = np.asarray(d.mesh.points)[:, :2]
    assert np.linalg.norm(c[:, 0] - sol) / np.linalg.norm(c[:, 0]) < 1e-8


# --------------------------------------------------------------------------
# transient: full time-stepping vs the analytic heat decay
# --------------------------------------------------------------------------
def test_transient_heat_time_stepping_matches_analytic():
    nu = 1.0
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1, time=(0.0, 0.05, 11))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ic = u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [xi0, yi0])
    fem = jno.fem([ui.t * vi + nu * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, ic])

    M, A = _dense(fem.M), _dense(fem.operator.A)
    w, dt = np.asarray(fem.state0), float(fem.dt)
    nsteps = round((fem.t1 - fem.t0) / dt)
    for _ in range(nsteps):  # backward Euler: (M + dt A) w_next = M w
        w = np.linalg.solve(M + dt * A, M @ w)
    c = np.asarray(d.mesh.points)[:, :2]
    analytic = np.exp(-2 * nu * np.pi**2 * fem.t1) * np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
    # first-order backward Euler over 10 steps -> a few percent
    assert np.linalg.norm(analytic - w) / np.linalg.norm(analytic) < 6e-2


# --------------------------------------------------------------------------
# nonlinear steady: Newton solve recovers a manufactured solution
# --------------------------------------------------------------------------
def test_nonlinear_reaction_newton_recovers_manufactured():
    spo = pytest.importorskip("scipy.optimize")
    # -lap u + u^3 = f, u_exact = x(1-x)y(1-y), f = -lap(u_exact) + u_exact^3, u=0 on boundary.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    g = xi * (1 - xi) * yi * (1 - yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi)) + g**3
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + (u * u * u) * vi - f * vi, u(xb, yb) - 0.0])
    assert not fem.is_linear
    sol = spo.root(
        lambda v: np.asarray(fem.residual(v)),
        np.zeros(fem.dofs),
        jac=lambda v: _dense(fem.jacobian(v)),
        method="hybr",
    )
    assert sol.success
    c = np.asarray(d.mesh.points)[:, :2]
    exact = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    assert np.linalg.norm(exact - sol.x) / np.linalg.norm(exact) < 1e-2


# --------------------------------------------------------------------------
# non-box (CSG) domain
# --------------------------------------------------------------------------
def test_csg_domain_assembles_through_jno_fem():
    pytest.importorskip("pygmsh", reason="pygmsh required for build_mesh")
    dom = jno.domain.csg([(0, 0), (2, 0), (2, 1), (0, 1)], name="chamber") - jno.domain.csg(
        [(0.8, 0.35), (1.2, 0.35), (1.2, 0.65), (0.8, 0.65)], name="hole"
    )
    dom.build_mesh(0.15)
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    A = _dense(fem.A)
    assert A.shape[0] == A.shape[1] == fem.dofs
    assert np.allclose(A, A.T, atol=1e-7)


# --------------------------------------------------------------------------
# edge cases that must error cleanly
# --------------------------------------------------------------------------
def _scalar_volume_weak(d):
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    return u, ui.x * vi.x + ui.y * vi.y - vi


def test_region_ambiguous_residual_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, weak = _scalar_volume_weak(d)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    with pytest.raises(ValueError):
        jno.fem([weak, u(xl, yl) + u(xr, yr) - 0.0])  # spans two boundary regions


def test_residual_without_trial_or_test_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    xi, yi, _ = d.variable("interior", split=True)
    with pytest.raises(ValueError):
        jno.fem([xi - 0.0])  # neither trial nor test


def test_non_affine_dirichlet_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, weak = _scalar_volume_weak(d)
    xb, yb, _ = d.variable("boundary", split=True)
    with pytest.raises(ValueError):
        jno.fem([weak, u(xb, yb) ** 2 - 1.0])  # nonlinear in u -> not an essential BC
