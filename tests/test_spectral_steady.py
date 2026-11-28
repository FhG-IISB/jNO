"""``jno.spectral([...]).solve()`` on steady problems: the error falls to round-off on analytic solutions.

Oracles are manufactured solutions; the claim is spectral accuracy, so each tolerance is near machine
precision at a modest number of nodes -- something no fixed-order stencil reaches.
"""

import jax
import jno.numpy as jnn
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _faces(d, *names):
    return [d.variable(n, split=True) for n in names]


def test_chebyshev_poisson_dirichlet_to_round_off():
    pi = np.pi
    d = jno.shape.rect(-1.0, -1.0, 1.0, 1.0).domain()
    U = d.unknown(space="spectral", order=(20, 20))
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = U.bind(x=x, y=y)
    exact = lambda X, Y: np.sin(pi * X) * np.sin(pi * Y) + X**3 * Y  # noqa: E731
    f = -(-2 * pi**2 * jnn.sin(pi * x) * jnn.sin(pi * y) + 6 * x * y)
    g = xb**3 * yb
    sol = jno.spectral([-(u.xx + u.yy) - f, U(xb, yb) - g]).solve()
    P = U.basis.nodes
    err = np.abs(np.asarray(sol) - exact(P[:, 0], P[:, 1])).max()
    assert err < 1e-11, err


def test_periodic_helmholtz_is_exact():
    d = jno.shape.rect(0.0, 0.0, 2 * np.pi, 2 * np.pi).domain()
    U = d.unknown(space="spectral", order=(16, 16))
    x, y, _ = d.variable("interior", split=True)
    (xl, yl, _), (xr, yr, _), (xb, yb, _), (xt, yt, _) = _faces(d, "left", "right", "bottom", "top")
    u = U.bind(x=x, y=y)
    exact = lambda X, Y: np.cos(2 * X) * np.sin(3 * Y) + 0.5  # noqa: E731
    f = (1 + 13) * jnn.cos(2 * x) * jnn.sin(3 * y) + 0.5
    sol = jno.spectral([-(u.xx + u.yy) + u - f, U(xl, yl) - U(xr, yr), U(xb, yb) - U(xt, yt)]).solve()
    assert U.basis.families == ("fourier", "fourier")
    P = U.basis.nodes
    assert np.abs(np.asarray(sol) - exact(P[:, 0], P[:, 1])).max() < 1e-12


def test_fourier_by_chebyshev_channel_poisson():
    d = jno.shape.rect(0.0, -1.0, 2 * np.pi, 1.0).domain()
    U = d.unknown(space="spectral", order=(16, 21))
    x, y, _ = d.variable("interior", split=True)
    (xl, yl, _), (xr, yr, _), (xb, yb, _), (xt, yt, _) = _faces(d, "left", "right", "bottom", "top")
    u = U.bind(x=x, y=y)
    # u = sin(x) (1 - y^2) e^y : zero on the walls
    exact = lambda X, Y: np.sin(X) * (1 - Y**2) * np.exp(Y)  # noqa: E731
    lap = lambda X, Y: jnn.sin(X) * jnn.exp(Y) * (-(1 - Y**2) + (-2 - 4 * Y + 1 - Y**2))  # noqa: E731
    sol = jno.spectral([(u.xx + u.yy) - lap(x, y), U(xb, yb) - 0.0, U(xt, yt) - 0.0, U(xl, yl) - U(xr, yr)]).solve()
    assert U.basis.families == ("fourier", "chebyshev")
    P = U.basis.nodes
    assert np.abs(np.asarray(sol) - exact(P[:, 0], P[:, 1])).max() < 1e-12


def test_neumann_and_robin_rows():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    U = d.unknown(space="spectral", order=(18, 18))
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _, nxr, nyr = d.variable("right", normals=True, split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    u = U.bind(x=x, y=y)
    ur = U.bind(x=xr, y=yr)
    ut = U.bind(x=xt, y=yt)
    exact = lambda X, Y: np.exp(X) * np.cos(Y)  # harmonic  # noqa: E731
    sol = jno.spectral(
        [
            u.xx + u.yy,
            U(xl, yl) - jnn.cos(yl),
            U(xb, yb) - jnn.exp(xb),
            ur.x - jnn.exp(xr) * jnn.cos(yr),  # Neumann on the right (a boundary derivative row)
            ut.y + ut - jnn.exp(xt) * (np.cos(1.0) - np.sin(1.0)),  # Robin on the top
        ]
    ).solve()
    P = U.basis.nodes
    assert np.abs(np.asarray(sol) - exact(P[:, 0], P[:, 1])).max() < 1e-10


def test_dense_and_gmres_agree():
    d = jno.shape.rect(-1.0, -1.0, 1.0, 1.0).domain()
    U = d.unknown(space="spectral", order=(10, 10))
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = U.bind(x=x, y=y)
    prob = jno.spectral([-(u.xx + u.yy) - 1.0, U(xb, yb) - 0.0])
    a = np.asarray(prob.solve())
    b = np.asarray(prob.solve(linear=jno.solve.gmres(tol=1e-13, maxiter=400, restart=100)))
    assert np.abs(a - b).max() < 1e-8


def test_nonlinear_bratu_by_newton():
    d = jno.shape.rect(-1.0, -1.0, 1.0, 1.0).domain()
    U = d.unknown(space="spectral", order=(14, 14))
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = U.bind(x=x, y=y)
    exact = lambda X, Y: (1 - X**2) * (1 - Y**2)  # noqa: E731
    ue = (1 - x**2) * (1 - y**2)
    f = 2 * (1 - y**2) + 2 * (1 - x**2) - jnn.exp(ue)  # -Δu - e^u = f
    sol = jno.spectral([-(u.xx + u.yy) - jnn.exp(u) - f, U(xb, yb) - 0.0]).solve()
    P = U.basis.nodes
    assert np.abs(np.asarray(sol) - exact(P[:, 0], P[:, 1])).max() < 1e-10


def test_fdm_forwards_a_spectral_unknown():
    d = jno.shape.rect(-1.0, -1.0, 1.0, 1.0).domain()
    U = d.unknown(space="spectral", order=(8, 8))
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = U.bind(x=x, y=y)
    assert type(jno.fdm([-(u.xx + u.yy) - 1.0, U(xb, yb)])).__name__ == "_TraceSpectral"


def test_refusals():
    d = jno.shape.rect(-1.0, -1.0, 1.0, 1.0).domain()
    U = d.unknown(space=("fourier", "chebyshev"), order=(8, 8))
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    u = U.bind(x=x, y=y)
    with pytest.raises(ValueError, match="no periodic tie"):
        jno.spectral([-(u.xx + u.yy) - 1.0, U(xb, yb)])


class TestPrecond:
    def _channel(self, nx, ny, kappa=None):
        d = jno.shape.rect(0.0, -1.0, 2 * np.pi, 1.0).domain()
        U = d.unknown(space="spectral", order=(nx, ny))
        x, y, _ = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        xb, yb, _ = d.variable("bottom", split=True)
        xt, yt, _ = d.variable("top", split=True)
        u = U.bind(x=x, y=y)
        k = 1.0 if kappa is None else kappa(x, y)
        eq = -(k * u.xx + u.yy) + u - jnn.sin(x) * jnn.cos(np.pi * y / 2)
        return U, jno.spectral([eq, U(xb, yb) - 0.0, U(xt, yt) - 0.0, U(xl, yl) - U(xr, yr)])

    def test_mode_blocks_invert_a_constant_coefficient_channel_exactly(self):
        U, prob = self._channel(32, 25)
        ref = np.asarray(prob.solve(linear=jno.solve.dense()))
        out = np.asarray(
            prob.solve(linear=jno.solve.gmres(tol=1e-13, maxiter=3, restart=3), precond=jno.precond.spectral())
        )
        assert np.abs(out - ref).max() < 1e-11

    def test_variable_coefficient_converges_with_the_preconditioner(self):
        U, prob = self._channel(32, 25, kappa=lambda x, y: 1.0 + 0.0 * x + 0.3 * y**2)
        ref = np.asarray(prob.solve(linear=jno.solve.dense()))
        out = np.asarray(
            prob.solve(linear=jno.solve.gmres(tol=1e-12, maxiter=20, restart=20), precond=jno.precond.spectral())
        )
        assert np.abs(out - ref).max() < 1e-9


def test_large_chebyshev_problem_with_rough_data_uses_fast_diagonalisation():
    """Above the dense limit (4096 DOFs) an all-Chebyshev problem has no Fourier mode to decouple; unpreconditioned
    GMRES stalled here (relative residual 8e-2 after 130 s). Fast diagonalisation of the separable part makes it
    one Krylov solve of a few iterations, variable coefficient and all."""
    n = 72
    d = jno.shape.rect(-1.0, -1.0, 1.0, 1.0).domain()
    U = d.unknown(space="spectral", order=(n, n))
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = U.bind(x=x, y=y)
    f = jnn.cos(23 * x + 0.3) * jnn.sin(17 * y) + jnn.exp(3 * x) * jnn.cos(31 * x * y)
    prob = jno.spectral([-(u.xx + u.yy) + (1 + x * x) * u - f, U(xb, yb) - jnn.sin(5 * xb) * yb])
    ref = np.asarray(prob.solve(linear=jno.solve.dense()))
    out = np.asarray(prob.solve())
    # both solves sit at the conditioning floor, cond ~ N^4 ~ 3e7 times machine epsilon
    assert np.abs(out - ref).max() < 1e-8 * max(1.0, np.abs(ref).max())
