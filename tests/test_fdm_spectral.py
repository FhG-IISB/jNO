"""`jno.fdm` reaches the spectral backend without any wiring — and is usually worse for it.

The cross-module claim holds: `fdm.py` rebuilds `Jacobian`/`Hessian` nodes carrying `node.scheme`,
so a scheme added to the evaluator arrives in the strong-form solver for free. No FDM code changed.

The *useful* result is the second one. Spectral is not a free upgrade: FDM's bread-and-butter
problem is Dirichlet, whose solution has `u' != 0` at the boundary, so neither the periodic nor the
even extension holds and the 5-point stencil wins by an order of magnitude. This is the docs'
periodicity warning showing up in a real solve, pinned so it is a number rather than a caveat.
"""

import jax
import numpy as np
import pytest

import jno
import jno.numpy as jnn


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dirichlet_poisson(scheme=None, size=0.05):
    """-Δu = f with u = sin(πx)sin(πy), zero on the boundary."""
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain(structured=True)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])

    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    kw = {} if scheme is None else {"scheme": scheme}
    sol = np.asarray(jno.fdm([-ui.d2(x, **kw) - ui.d2(y, **kw) - f, u(xb, yb) - 0.0]).solve())
    return float(np.linalg.norm(sol.reshape(-1) - exact) / np.linalg.norm(exact))


def test_spectral_reaches_jno_fdm_and_solves():
    """The cross-module claim: no FDM code was changed for this to work."""
    assert _dirichlet_poisson(scheme="spectral") < 1.0, "the solve must at least converge"


def test_spectral_is_worse_than_the_stencil_on_a_dirichlet_problem():
    """Not a free upgrade. u' != 0 at the boundary, so the periodic extension has a kink and the
    second derivative rings — the 5-point stencil wins comfortably.

    The natural basis here is a SINE transform (odd extension), which JAX does not provide: it
    implements only DCT-2 and has no DST at all.
    """
    fd = _dirichlet_poisson(scheme=None)  # nodal unknown defaults to finite_difference
    spec = _dirichlet_poisson(scheme="spectral")
    assert fd < spec / 5, f"expected FD to win clearly: fd={fd:.2e} spectral={spec:.2e}"
    assert fd < 1e-2


def test_cosine_does_not_rescue_a_dirichlet_problem():
    """`:cosine` assumes u' = 0 at the ends, which a Dirichlet sine profile also violates."""
    assert _dirichlet_poisson(scheme="spectral:cosine") > 5 * _dirichlet_poisson(scheme=None)


def test_an_explicit_ad_scheme_on_a_nodal_unknown_still_fails_loudly():
    """A nodal unknown is stored values, not a function of x, so AD has nothing to differentiate.
    It raises rather than returning zeros — unchanged by the scheme registry work."""
    with pytest.raises(Exception):
        _dirichlet_poisson(scheme="automatic_differentiation")


class TestPeriodicProblem:
    """Where spectral actually belongs — and the practical recipe: one scheme per DIRECTION.

    MMS `-Δu = 5π² sin(2πx) sin(πy)`, periodic in x via the tie `u(left) - u(right)`, Dirichlet in y.
    The x-direction is exactly the band-limited periodic case the FFT is exact on; the y-direction is
    Dirichlet, where it rings. Choosing per term beats choosing one scheme for the whole residual.
    """

    def _run(self, sx=None, sy=None, size=0.08):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(structured=True)
        p = np.asarray(d.mesh_connectivity["points"])[:, :2]
        x, y, _ = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        xb, yb, _ = d.variable("bottom", split=True)
        xt, yt, _ = d.variable("top", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        f = 5 * np.pi**2 * jnn.sin(2 * np.pi * x) * jnn.sin(np.pi * y)
        kx = {} if sx is None else {"scheme": sx}
        ky = {} if sy is None else {"scheme": sy}
        sol = np.asarray(
            jno.fdm(
                [
                    -ui.d2(x, **kx) - ui.d2(y, **ky) - f,
                    u(xl, yl) - u(xr, yr),
                    u(xb, yb) - 0.0,
                    u(xt, yt) - 0.0,
                ]
            ).solve()
        ).reshape(-1)
        exact = np.sin(2 * np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
        return float(np.linalg.norm(sol - exact) / np.linalg.norm(exact))

    def test_spectral_in_the_periodic_direction_wins(self):
        """The headline: spectral on x (periodic) + FD on y (Dirichlet) beats FD everywhere."""
        all_fd = self._run()
        mixed = self._run(sx="spectral", sy=None)
        assert mixed < all_fd / 5, f"expected a clear win: all-FD {all_fd:.2e} vs mixed {mixed:.2e}"

    def test_spectral_everywhere_is_worse_than_mixing(self):
        """Applying it to the Dirichlet direction too gives back more than it gains."""
        assert self._run(sx="spectral", sy="spectral") > self._run(sx="spectral", sy=None)

    def test_the_periodic_tie_still_holds(self):
        """Spectral must not break the wrap-around constraint the tie imposes."""
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).domain(structured=True)
        x, y, _ = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        xb, yb, _ = d.variable("bottom", split=True)
        xt, yt, _ = d.variable("top", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        f = 5 * np.pi**2 * jnn.sin(2 * np.pi * x) * jnn.sin(np.pi * y)
        sol = np.asarray(
            jno.fdm(
                [
                    -ui.d2(x, scheme="spectral") - ui.d2(y) - f,
                    u(xl, yl) - u(xr, yr),
                    u(xb, yb) - 0.0,
                    u(xt, yt) - 0.0,
                ]
            ).solve()
        ).reshape(-1)
        sx_n, sy_n = d.mesh_connectivity["grid"]["shape"]
        grid = sol.reshape(sx_n, sy_n)
        assert float(np.max(np.abs(grid[0, :] - grid[-1, :]))) < 1e-9
