"""``jno.fvm`` convection: numerical fluxes, reconstruction and limiters (:func:`jno.fv`)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402

EPS = 0.01


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _line(h):
    d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=h).domain()
    x = d.variable("interior", split=True)[0]
    xl, xr = d.variable("left", split=True)[0], d.variable("right", split=True)[0]
    u = d.unknown()
    return d, x, xl, xr, u, u.bind(x=x)


def _advdiff(h, scheme):
    """u' - 0.01 u'' = 0, u(0) = 0, u(1) = 1: a boundary layer at x = 1 (cell Péclet 2.5 at h = 0.05)."""
    _, x, xl, xr, u, ui = _line(h)
    p = jno.fvm([ui.d(x, scheme=scheme) - EPS * ui.xx, u(xl) - 0.0, u(xr) - 1.0])
    s, c = np.asarray(p.solve()), p.centroids[:, 0]
    exact = (np.exp((c - 1) / EPS) - np.exp(-1 / EPS)) / (1 - np.exp(-1 / EPS))
    return s, np.abs(s - exact).max()


def test_first_order_rusanov_is_upwinding():
    """For a linear flux the Rusanov flux is the upwind flux: the assembled operator equals the hand-built
    upwind + central-diffusion matrix."""
    _, x, xl, xr, u, ui = _line(0.1)
    p = jno.fvm([ui.d(x, scheme=jno.fv(recon="none")) - EPS * ui.xx, u(xl) - 0.0, u(xr) - 1.0])
    n = p._N
    A = np.asarray(p._operator("steady", jax.jit(p._residual_fn()), jnp.zeros(n)).todense())
    h, D = 0.1, EPS / 0.1
    ref = np.zeros((n, n))
    for i in range(n):
        ref[i, i] += 1.0 + 2 * D  # upwind outflow + diffusion to both neighbours
        if i > 0:
            ref[i, i - 1] += -1.0 - D
        if i < n - 1:
            ref[i, i + 1] += -D
    ref[0, 0] += D  # left face: the inflow carries only the boundary value; diffusion over h/2 (2D, one D counted)
    ref[n - 1, n - 1] += D  # right face: upwind outflow (counted) and diffusion over h/2 (2D, one D counted)
    assert h == pytest.approx(p.volumes[0])
    np.testing.assert_allclose(A, ref, atol=1e-12)


def test_limiters_stay_bounded_where_the_unlimited_reconstruction_oscillates():
    unlimited, _ = _advdiff(0.05, jno.fv(limiter=None))
    assert unlimited.min() < -0.05
    for limiter in ("vanalbada", "minmod", "vanleer", "superbee", "barth", "venkat"):
        s, _ = _advdiff(0.05, jno.fv(limiter=limiter))
        assert s.min() > -1e-3 and s.max() < 1 + 1e-3, limiter


def test_second_order_beats_first_order_in_the_layer():
    _, first = _advdiff(0.003125, jno.fv(recon="none"))
    _, second = _advdiff(0.003125, jno.fv())
    assert second < first / 4


@pytest.mark.parametrize("flux", ["rusanov", "hll", "roe"])
def test_every_flux_resolves_the_layer(flux):
    _, err = _advdiff(0.0125, jno.fv(flux=flux, limiter="vanleer"))
    assert err < 6e-2


def test_a_custom_flux_callable_equals_the_built_in_rusanov():
    def rusanov(qL, qR, n, Fn):
        return 0.5 * (Fn(qL) + Fn(qR)) - 0.5 * jnp.abs(n[:, :1]) * (qR - qL)  # wind 1 along x

    a, _ = _advdiff(0.05, jno.fv(flux=rusanov))
    b, _ = _advdiff(0.05, jno.fv())
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_burgers_shock_sits_where_conservation_puts_it():
    """Steady viscous Burgers ``(u²/2)' = ν u''`` with u(0) = 1, u(1) = -1: by symmetry the shock is at 0.5,
    and only the conservative form keeps it there with the right jump."""
    _, x, xl, xr, u, ui = _line(0.01)
    p = jno.fvm([(ui * ui / 2).d(x, scheme=jno.fv()) - 0.005 * ui.xx, u(xl) - 1.0, u(xr) + 1.0])
    s, c = np.asarray(p.solve(x0=np.asarray(1.0 - 2.0 * (np.arange(p._N) > p._N // 2)))), p.centroids[:, 0]
    crossing = c[np.argmin(np.abs(s))]
    assert abs(crossing - 0.5) < 0.011
    assert s[c < 0.4].min() > 0.99 and s[c > 0.6].max() < -0.99


def test_a_step_advected_across_triangles_stays_bounded():
    """Pure convection b·∇u = 0 with b = (1, 0.5) on a triangulation: inflow u = 1 on the left, 0 on the
    bottom. Venkatakrishnan's limiter keeps the values within [0, 1] to 0.1% and stays smooth enough for Newton
    (Barth-Jespersen's min() does not: it is for explicit time marching)."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.05).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xbo, ybo, _ = d.variable("bottom", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    s = jno.fv(limiter="venkat")
    p = jno.fvm([(1.0 * ui).d(x, scheme=s) + (0.5 * ui).d(y, scheme=s), u(xl, yl) - 1.0, u(xbo, ybo) - 0.0])
    sol = np.asarray(p.solve())
    assert sol.min() > -2e-3 and sol.max() < 1 + 2e-3
    assert sol[(p.centroids[:, 1] > 0.9) & (p.centroids[:, 0] < 0.3)].min() > 0.9  # above the line y = x/2


def test_axes_with_different_schemes_raise():
    d = jno.shape.rect(0, 0, 1, 1, size=0.2).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.raises(ValueError, match="same"):
        jno.fvm([ui.d(x, scheme=jno.fv(flux="hll")) + ui.d(y, scheme=jno.fv()), u(xb, yb) - 0.0])


def test_unknown_scheme_names_raise():
    with pytest.raises(ValueError, match="flux"):
        jno.fv(flux="hllc")
    with pytest.raises(ValueError, match="limiter"):
        jno.fv(limiter="koren")


def test_a_harmonic_face_mean_is_exact_across_a_layered_coefficient():
    """-(κ u')' = 0 with κ = 1 | 10 jumping at a face: the flux is constant and u piecewise linear. The
    harmonic mean of the two cell values is the exact series conductance; the arithmetic one is not."""
    _, x, xl, xr, u, ui = _line(0.1)
    import jno.jnp_ops as jnn

    k = 1.0 + 9.0 * (jnn.tanh(1e3 * (x - 0.5)) + 1.0) / 2.0  # 1 left of 0.5, 10 right of it, at the cells
    flux = 1.0 / (0.5 / 1.0 + 0.5 / 10.0)  # the exact flux for u(0) = 0, u(1) = 1
    out = {}
    for mode in ("harmonic", "arithmetic"):
        p = jno.fvm([-(k * ui.x).d(x, scheme=jno.fv(average=mode)), u(xl) - 0.0, u(xr) - 1.0])
        s, c = np.asarray(p.solve()), p.centroids[:, 0]
        exact = np.where(c < 0.5, flux * c, 1.0 - flux * (1.0 - c) / 10.0)
        out[mode] = np.abs(s - exact).max()
    assert out["harmonic"] < 1e-10 < out["arithmetic"]
