"""Lagged-Jacobian Newton: ``jno.solve.newton(direct=True, reuse=True)``.

Oracles: an algebraic system built around a known root, and a manufactured nonlinear FEM problem whose
exact solution is known. Reuse must reach the same root as a fresh-tangent Newton, with FEWER
factorizations, and a stale tangent that makes things worse must be rejected rather than followed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.experimental.sparse as jsp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.newton_krylov import LAST_NEWTON_STATS, newton_direct  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _cubic_system(n=40, seed=0):
    """F(u) = A u + u^3 - b with a chosen root u*: A a 1-D Laplacian plus a shift, b built from u*."""
    rng = np.random.default_rng(seed)
    main = np.full(n, 2.5)
    off = np.full(n - 1, -1.0)
    A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
    u_star = rng.uniform(-1.5, 1.5, n)
    b = A @ u_star + u_star**3
    Aj, bj = jnp.asarray(A), jnp.asarray(b)

    def F(u):
        return Aj @ u + u**3 - bj

    def Jac(u):
        return jsp.BCOO.fromdense(Aj + jnp.diag(3.0 * u**2), nse=3 * n - 2)

    return F, Jac, u_star


def test_reuse_reaches_the_known_root_with_fewer_factorizations():
    F, Jac, u_star = _cubic_system()
    u0 = jnp.zeros_like(jnp.asarray(u_star))
    fresh = np.asarray(newton_direct(F, Jac, u0, rtol=1e-12, atol=1e-12))
    s_fresh = dict(LAST_NEWTON_STATS)
    lagged = np.asarray(newton_direct(F, Jac, u0, rtol=1e-12, atol=1e-12, reuse=True))
    s_lag = dict(LAST_NEWTON_STATS)
    assert np.max(np.abs(fresh - u_star)) < 1e-10
    assert np.max(np.abs(lagged - u_star)) < 1e-10
    assert s_fresh["factorizations"] == s_fresh["steps"]  # one tangent per step without reuse
    assert s_lag["factorizations"] < s_fresh["factorizations"], (s_lag, s_fresh)
    assert s_lag["steps"] >= s_fresh["steps"]  # the trade: more steps, fewer factorizations


def test_a_stale_tangent_that_hurts_is_rejected_not_followed():
    """u^3 = 8 from u0 = 10: the tangent at 10 (300) is far too stiff for later iterates, so steps on it
    contract weakly and must trigger refreshes; the answer is still the root 2."""

    def F(u):
        return u**3 - 8.0

    def Jac(u):
        return jsp.BCOO.fromdense(jnp.diag(3.0 * u**2), nse=1)

    root = np.asarray(newton_direct(F, Jac, jnp.array([10.0]), rtol=1e-13, atol=1e-13, reuse=True))
    assert abs(float(root[0]) - 2.0) < 1e-10  # |r| <= 1e-13 * (1 + |r0| = 992) -> |du| ~ 1e-11
    assert LAST_NEWTON_STATS["factorizations"] >= 2  # it had to refresh at least once


def test_nonfinite_stale_step_is_rejected():
    """A stale step that lands where the residual is NaN must not be taken (NaN-safe rejection)."""

    def F(u):
        return jnp.sqrt(u) - 1.0  # NaN for u < 0

    def Jac(u):
        return jsp.BCOO.fromdense(jnp.diag(0.5 / jnp.sqrt(u)), nse=1)

    root = np.asarray(newton_direct(F, Jac, jnp.array([0.05]), rtol=1e-12, atol=1e-12, reuse=True, line_search=True))
    assert abs(float(root[0]) - 1.0) < 1e-10


def _manufactured(mesh_size=0.08):
    """-div((1 + u^2) grad u) = f on the unit square, u = 0 on the boundary, exact u = sin(pi x) sin(pi y)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    sin, cos = jno.np.sin, jno.np.cos
    ue = sin(PI * x) * sin(PI * y)
    uex, uey = PI * cos(PI * x) * sin(PI * y), PI * sin(PI * x) * cos(PI * y)
    # f = -div((1+ue^2) grad ue) = (1+ue^2)(2 pi^2 ue) - 2 ue |grad ue|^2
    f = (1.0 + ue * ue) * (2.0 * PI**2 * ue) - 2.0 * ue * (uex * uex + uey * uey)
    k = 1.0 + ui * ui
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=4)
    pts = np.asarray(fem.points)[:, :2]
    return fem, np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])


def test_fem_reuse_matches_fresh_newton_and_the_exact_solution():
    fem, exact = _manufactured()
    lin = jno.solve.lu(backend="host")  # content-keyed factor cache: a reused tangent is a solve
    u_fresh = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=1e-12), linear=lin))
    n_fresh = fem.stats["nonlinear"]["factorizations"]
    u_lag = np.asarray(
        fem.solve(nonlinear=jno.solve.newton(direct=True, reuse=True, rtol=1e-12, atol=1e-12), linear=lin)
    )
    n_lag = fem.stats["nonlinear"]["factorizations"]
    assert np.max(np.abs(u_lag - u_fresh)) < 1e-9  # same discrete root
    rel = np.linalg.norm(u_lag.ravel() - exact) / np.linalg.norm(exact)
    assert rel < 2e-2, rel  # and it is the PDE's solution, to discretisation error
    assert n_lag < n_fresh, (n_lag, n_fresh)


def test_reuse_without_direct_raises():
    with pytest.raises(ValueError, match="direct=True"):
        jno.solve.newton(reuse=True)
