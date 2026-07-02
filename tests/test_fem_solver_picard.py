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
