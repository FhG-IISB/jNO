"""``jno.lag`` (lagged-coefficient marker) + the ``jno.solve.picard`` / damped-Newton drivers.

Pins: on a manufactured nonlinear diffusion problem ``-div((1+u^2) grad u) = f``, Picard on the
``lag``-frozen form converges to the *same* solution as full Newton on the unlagged form (the
root is independent of gradient markers); damping converges too; the drivers compose with the
``linear=`` slot; ``lag`` freezes differentiation (stop-gradient semantics, array fallback
included) while leaving values untouched; and ``picard`` without any ``lag`` marker is exactly
damped Newton.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _nonlinear_diffusion(lagged: bool, mesh_size=0.2):
    """-div((1+u^2) grad u) = f manufactured from u* = x(1-x)y(1-y)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    us = xi * (1 - xi) * yi * (1 - yi)
    ux, uy = (1 - 2 * xi) * yi * (1 - yi), xi * (1 - xi) * (1 - 2 * yi)
    lap = -2 * yi * (1 - yi) - 2 * xi * (1 - xi)
    f = -((2 * us) * (ux * ux + uy * uy) + (1 + us**2) * lap)
    k = 1.0 + ui**2
    kk = jno.lag(k) if lagged else k
    return jno.fem([kk * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=4)


def test_picard_on_lagged_form_matches_full_newton():
    u_newton = np.asarray(_nonlinear_diffusion(lagged=False).solve())
    fem_lag = _nonlinear_diffusion(lagged=True)
    u_picard = np.asarray(fem_lag.solve(nonlinear=jno.solve.picard()))
    assert np.abs(u_picard - u_newton).max() < 1e-8
    u_damped = np.asarray(fem_lag.solve(nonlinear=jno.solve.picard(damping=0.7)))
    assert np.abs(u_damped - u_newton).max() < 1e-7


def test_picard_composes_with_linear_slot():
    u_newton = np.asarray(_nonlinear_diffusion(lagged=False).solve())
    fem_lag = _nonlinear_diffusion(lagged=True)
    u = np.asarray(fem_lag.solve(nonlinear=jno.solve.picard(), linear=jno.solve.bicgstab(tol=1e-12)))
    assert np.abs(u - u_newton).max() < 1e-8


def test_damped_newton_converges():
    fem = _nonlinear_diffusion(lagged=False)
    u_ref = np.asarray(fem.solve())
    u = np.asarray(fem.solve(nonlinear=jno.solve.newton(damping=0.5)))
    assert np.abs(u - u_ref).max() < 1e-8


def test_picard_without_lag_is_damped_newton():
    fem = _nonlinear_diffusion(lagged=False)
    u_ref = np.asarray(fem.solve())
    u = np.asarray(fem.solve(nonlinear=jno.solve.picard(damping=1.0)))
    assert np.abs(u - u_ref).max() < 1e-8


def test_nonlinear_precond_form_full_system():
    """precond= on the matrix-free nonlinear path: a form-based (assembled auxiliary) operator
    is materialized per Newton linearization against the JVP operator."""
    fem = _nonlinear_diffusion(lagged=False)
    u_ref = np.asarray(fem.solve())
    d = fem.domain
    # the linear part of the operator as the auxiliary preconditioner (fresh symbols, same space)
    w, s = d.fem_symbols(names=("w_aux", "s_aux"))
    xi, yi, _ = d.variable("interior", split=True)
    wi, si = w.bind(x=xi, y=yi), s.bind(x=xi, y=yi)
    spec = jno.precond.form([wi.x * si.x + wi.y * si.y + wi * si], inner=jno.solve.lu(), quad_degree=4)
    u = np.asarray(fem.solve(nonlinear=jno.solve.newton(), linear=jno.solve.fgmres(tol=1e-10), precond=spec))
    assert np.abs(u - u_ref).max() < 1e-7


def test_nonlinear_stokes_picard_fgmres_triangular():
    """The production nonlinear-saddle architecture (the cold-rolling / rigid-plastic pattern):
    velocity-dependent viscosity frozen with jno.lag, driven by Picard, each lagged Stokes system
    solved by FGMRES with a block upper-triangular preconditioner (inexact CG velocity block +
    weighted pressure-mass Schur approximation)."""
    inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    G, H, Lx = 1.0, 1.0, 2.0
    u_profile = lambda y: (G / 2.0) * y * (H - y)
    d = jno.domain(box(0.0, 0.0, Lx, H), mesh_size=0.3)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    mu_eff = 1.0 + 0.5 * inner_(gu, gu, n_contract=2)  # shear-dependent viscosity
    fem = jno.fem(
        [
            jno.lag(mu_eff) * inner_(gu, gv, n_contract=2) - pp * trace(gv),
            -qq * trace(gu),
            u(xb, yb)[0] - u_profile(yb),
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )

    # reference: dense Picard on the same lagged residual (same root, brute-force linear algebra)
    def dense_picard(rf, u0):
        w = u0
        for _ in range(60):
            J = jax.jacfwd(rf)(w)
            w = w + jnp.linalg.solve(J, -rf(w))
            if float(jnp.linalg.norm(rf(w))) < 1e-11:
                break
        return w

    u_ref = np.asarray(fem.solve(solve_fn=dense_picard))

    sol = fem.solve(
        nonlinear=jno.solve.picard(),
        linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=4000),
        precond=jno.precond.triangular(
            (u, jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=60))),
            (p, jno.precond.form([pp * qq], inner=jno.solve.dense())),
        ),
    )
    assert np.abs(np.asarray(sol) - u_ref).max() < 1e-6


def test_line_search_rescues_divergent_fixed_step():
    """A stiff residual where a full fixed step overshoots to a non-finite iterate, but
    residual-norm Armijo backtracking (``line_search=True``) finds a safe step and converges.

    ``f(u) = exp(u) - 1`` (root ``u = 0``). From ``u0 = -8`` the Jacobian ``exp(u)`` is tiny, so the
    Newton step is ~ +3000: the fixed full step lands at ``exp(3000) = inf`` and never recovers,
    while the line search halves the step until the residual actually decreases."""
    from jno.utils.solver.newton_krylov import newton_krylov

    f = lambda u: jnp.exp(u) - 1.0
    diag_solve = lambda mv, rhs: rhs / mv(jnp.ones_like(rhs))  # exact 1x1 (diagonal) inner solve
    u0 = jnp.array([-8.0])

    u_fixed = newton_krylov(f, u0, linear_solve=diag_solve, damping=1.0, line_search=False, max_steps=50)
    assert not np.isfinite(np.asarray(u_fixed)).all()  # fixed full step diverges

    u_ls = newton_krylov(f, u0, linear_solve=diag_solve, damping=1.0, line_search=True, max_steps=200)
    assert np.isfinite(np.asarray(u_ls)).all()
    assert np.abs(np.asarray(u_ls)).max() < 1e-6  # converged to the root


def test_picard_line_search_matches_reference():
    """``line_search=True`` reaches the same root as full Newton (globalization changes the path to
    the solution, never the solution) through the public ``jno.solve.picard`` slot."""
    u_ref = np.asarray(_nonlinear_diffusion(lagged=False).solve())
    u_ls = np.asarray(_nonlinear_diffusion(lagged=True).solve(nonlinear=jno.solve.picard(line_search=True)))
    assert np.abs(u_ls - u_ref).max() < 1e-7


def test_lag_freezes_gradients_but_not_values():
    # array fallback: values pass through, differentiation sees a constant
    x = jnp.asarray([1.5, -2.0])
    assert np.allclose(np.asarray(jno.lag(x)), np.asarray(x))
    g = jax.grad(lambda z: jnp.sum(jno.lag(z) * z))(x)
    assert np.allclose(np.asarray(g), np.asarray(x))  # d/dz [c*z] = c (not 2z)
    # traced views return a view (the .stop_gradient property), not a callable
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.5)
    u, _ = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui = u.bind(x=xi, y=yi)
    lagged = jno.lag(1.0 + ui**2)
    assert hasattr(lagged, "_expr") or hasattr(lagged, "op_id")
