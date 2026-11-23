"""``jno.fvm`` systems of conservation laws: Euler and shallow water, written as fluxes of a vector unknown."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

GAMMA, G = 1.4, 9.81


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _sod_exact(x, t, x0=0.5, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1)):
    """Density of the exact Riemann solution (Toro 2009, ch. 4) for an ideal gas."""
    g = GAMMA
    rl, ul, pl = left
    rr, ur, pr = right
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def f(p, rk, pk, ck):
        if p > pk:
            a, b = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(a / (p + b))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

    p = 0.5 * (pl + pr)
    for _ in range(100):
        F = f(p, rl, pl, cl) + f(p, rr, pr, cr) + ur - ul
        dp = 1e-7 * p
        dF = (f(p + dp, rl, pl, cl) + f(p + dp, rr, pr, cr) - f(p - dp, rl, pl, cl) - f(p - dp, rr, pr, cr)) / (2 * dp)
        p -= F / dF
    us = 0.5 * (ul + ur) + 0.5 * (f(p, rr, pr, cr) - f(p, rl, pl, cl))
    rsl, csl = rl * (p / pl) ** (1 / g), cl * (p / pl) ** ((g - 1) / (2 * g))
    rsr = rr * (p / pr + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pr + 1)
    shock = ur + cr * np.sqrt((g + 1) / (2 * g) * p / pr + (g - 1) / (2 * g))
    xi = (x - x0) / t
    fan = rl * (2 / (g + 1) + (g - 1) / ((g + 1) * cl) * (ul - xi)) ** (2 / (g - 1))
    return np.where(xi < ul - cl, rl, np.where(xi < us - csl, fan, np.where(xi < us, rsl, np.where(xi < shock, rsr, rr))))


def _sod(scheme, h=0.01, T=0.2):
    d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=h).domain(time=(0.0, T, 21))
    x, t = d.variable("interior", split=True)[:2]
    xi = d.variable("initial", split=True)[0]
    q = d.unknown.vector(3)
    qi = q.bind(x=x, t=t)
    rho, mom, E = qi[0], qi[1], qi[2]
    u = mom / rho
    p = (GAMMA - 1) * (E - 0.5 * mom * u)
    F = jnn.stack([mom, mom * u + p, u * (E + p)], axis=-1)
    left = (1.0 - jnn.tanh(1e4 * (xi - 0.5))) / 2
    q0 = jnn.stack([0.125 + 0.875 * left, 0.0 * xi, (0.1 + 0.9 * left) / (GAMMA - 1)], axis=-1)
    prob = jno.fvm([qi.t + F.d(x, scheme=scheme), q(xi) - q0])
    traj = np.asarray(prob.solve(time=jno.solve.erk("ssprk3", cfl=0.5)))
    c = prob.centroids[:, 0]
    return traj, float(np.abs(traj[-1, 0] - _sod_exact(c, T)).mean())


@pytest.mark.parametrize(
    "flux", ["rusanov", pytest.param("hll", marks=pytest.mark.slow), pytest.param("roe", marks=pytest.mark.slow)]
)
def test_sod_shock_tube(flux):
    traj, l1 = _sod(jno.fv(flux=flux, limiter="minmod"))
    assert traj.shape[1:] == (3, 100)
    assert l1 < 1.2e-2
    mass = traj[:, 0].sum(axis=1) * 0.01  # no inflow or outflow reaches the ends by T: mass is constant
    assert np.abs(mass - mass[0]).max() < 1e-12


@pytest.mark.slow
def test_roe_resolves_the_contact_better_than_rusanov():
    _, roe = _sod(jno.fv(flux="roe", limiter="minmod"))
    _, rus = _sod(jno.fv(flux="rusanov", limiter="minmod"))
    assert roe < 0.9 * rus


def test_a_flux_callable_and_a_speed_expression_agree():
    """A local Lax-Friedrichs flux with the fixed speed 2, once as a callable and once as ``speed=``."""

    def llf(qL, qR, n, Fn):
        return 0.5 * (Fn(qL) + Fn(qR)) - 0.5 * 2.0 * (qR - qL)

    a, _ = _sod(jno.fv(flux=llf, limiter="minmod"), h=0.02, T=0.1)
    b, _ = _sod(jno.fv(speed=2.0 + 0.0 * jnn.sin(0.0), limiter="minmod"), h=0.02, T=0.1)
    np.testing.assert_allclose(a, b, atol=1e-12)


def _shallow_water(d, scheme, bed, q0):
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    nb = d.variable("boundary", normals=True)
    nx, ny = nb[..., 0], nb[..., 1]  # the outward normal's components
    xi, yi, _ = d.variable("initial", split=True)
    q = d.unknown.vector(3)
    qi, qb = q.bind(x=x, y=y, t=t), q.bind(x=xb, y=yb)
    h, hu, hv = qi[0], qi[1], qi[2]
    b = bed(x, y)
    Fx = jnn.stack([hu, hu * hu / h + G * h * h / 2, hu * hv / h], axis=-1)
    Fy = jnn.stack([hv, hu * hv / h, hv * hv / h + G * h * h / 2], axis=-1)
    src = jnn.stack([0.0 * h, G * h * b.d(x), G * h * b.d(y)], axis=-1)
    prob = jno.fvm(
        [qi.t + Fx.d(x, scheme=scheme) + Fy.d(y, scheme=scheme) + src, qb[1] * nx + qb[2] * ny, q(xi, yi) - q0(xi, yi)]
    )
    return prob, np.asarray(prob.solve(time=jno.solve.erk("ssprk3", cfl=0.5)))


def _bump(X, Y):
    return 0.3 * jnn.exp(-((X - 0.5) ** 2 + (Y - 0.4) ** 2) / 0.03)


@pytest.mark.slow
def test_equilibrium_keeps_a_lake_at_rest_over_a_bumpy_bed():
    """Slip walls all round (U·n = 0 on a vector unknown) and a bump in the bed: h + b = 1 at rest stays at
    rest to round-off with ``equilibrium=``; without it the bed source and the flux do not balance."""
    out = {}
    for well_balanced in (False, True):
        d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain(time=(0.0, 0.05, 11))
        x = d.variable("interior", split=True)[0]
        y = d.variable("interior", split=True)[1]
        eq = jnn.stack([1.0 - _bump(x, y), 0.0 * x, 0.0 * x], axis=-1) if well_balanced else None
        _, traj = _shallow_water(
            d,
            jno.fv(equilibrium=eq),
            _bump,
            lambda X, Y: jnn.stack([1.0 - _bump(X, Y), 0.0 * X, 0.0 * X], axis=-1),
        )
        out[well_balanced] = np.abs(traj[-1, 1:]).max()
    assert out[True] < 1e-12 < 1e-4 < out[False]


def test_slip_walls_keep_the_water_in():
    """A hump of water sloshing in a closed box: the volume of water is conserved to round-off."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain(time=(0.0, 0.2, 11))
    prob, traj = _shallow_water(
        d,
        jno.fv(),
        lambda X, Y: 0.0 * X,
        lambda X, Y: jnn.stack([1.0 + 0.1 * jnn.exp(-((X - 0.3) ** 2 + (Y - 0.6) ** 2) / 0.02), 0.0 * X, 0.0 * X], axis=-1),
    )
    volume = (traj[:, 0] * prob.volumes).sum(axis=1)
    assert np.abs(volume - volume[0]).max() < 1e-12
    assert np.abs(traj[-1, 1:]).max() > 1e-3  # it does move


def test_a_system_of_scalar_unknowns_matches_the_vector_unknown():
    """1-D shallow water as two scalar unknowns (h, hu) gives the vector unknown's answer: the convective
    fluxes of all equations go through one Riemann solve."""

    def run(vector):
        d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.02).domain(time=(0.0, 0.05, 6))
        x, t = d.variable("interior", split=True)[:2]
        xi = d.variable("initial", split=True)[0]
        h0 = 1.0 + 0.5 * (1.0 - jnn.tanh(1e3 * (xi - 0.5))) / 2
        s = jno.fv(limiter="minmod")
        if vector:
            q = d.unknown.vector(2)
            qi = q.bind(x=x, t=t)
            h, hu = qi[0], qi[1]
            F = jnn.stack([hu, hu * hu / h + G * h * h / 2], axis=-1)
            out = jno.fvm([qi.t + F.d(x, scheme=s), q(xi) - jnn.stack([h0, 0.0 * xi], axis=-1)])
        else:
            hf, mf = d.unknown(), d.unknown()
            h, hu = hf.bind(x=x, t=t), mf.bind(x=x, t=t)
            out = jno.fvm(
                [h.t + hu.d(x, scheme=s), hu.t + (hu * hu / h + G * h * h / 2).d(x, scheme=s), hf(xi) - h0, mf(xi) - 0.0]
            )
        return np.asarray(out.solve(time=jno.solve.erk("ssprk3")))

    np.testing.assert_allclose(run(True), run(False), atol=1e-12)


def test_power_iteration_extremes_match_eigenvalues():
    """HLL and Roe read the extreme eigenvalues and |A| of the flux Jacobian without an eigendecomposition."""
    from jno._fvm.trace import _abs_matrix, _eig_extremes, _spectral_radius

    rng = np.random.default_rng(0)
    R = rng.standard_normal((4, 3, 3))
    lam = np.sort(rng.uniform(-3, 2, (4, 3)), axis=1)
    A = jnp.asarray(R @ (lam[:, :, None] * np.linalg.inv(R)))
    lo, hi = _eig_extremes(A)
    np.testing.assert_allclose(lo, lam[:, 0], rtol=1e-6)
    np.testing.assert_allclose(hi, lam[:, -1], rtol=1e-6)
    np.testing.assert_allclose(_spectral_radius(A), np.abs(lam).max(axis=1), rtol=1e-6)
    absA = np.asarray(_abs_matrix(A, jnp.zeros(4)))
    ref = R @ (np.abs(lam)[:, :, None] * np.linalg.inv(R))
    np.testing.assert_allclose(absA, ref, atol=1e-8)


def _two_diffusions(stacked):
    """Two decoupled diffusion problems, ``c*(u.xx + u.yy) + f = 0`` with ``u = g`` on the boundary: one vector
    unknown whose equation is a stack of Laplacians, or two scalar unknowns."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(12, 12)).quad().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    c = 1.0 + x  # a variable coefficient outside the Laplacian: the scalar meaning is c times div(grad u)
    f0, f1 = jnn.sin(3.0 * x) * y, 1.0 + 0.0 * x
    g0, g1 = xb * yb, 0.0 * xb
    if stacked:
        q = d.unknown.vector(2)
        qi = q.bind(x=x, y=y)
        lap = jnn.stack([c * (qi[0].xx + qi[0].yy) + f0, 2.0 * (qi[1].xx + qi[1].yy) + f1], axis=-1)
        cons = [lap, q(xb, yb) - jnn.stack([g0, g1], axis=-1)]
    else:
        a, b = d.unknown(), d.unknown()
        ai, bi = a.bind(x=x, y=y), b.bind(x=x, y=y)
        cons = [c * (ai.xx + ai.yy) + f0, 2.0 * (bi.xx + bi.yy) + f1, a(xb, yb) - g0, b(xb, yb) - g1]
    return np.asarray(jno.fvm(cons).solve())


def test_a_laplacian_inside_a_stacked_equation_is_a_face_flux():
    # Name the break: a Laplacian inside jnn.stack evaluated cell by cell (the volume gradient of the volume
    # gradient: a stencil twice as wide, and not conservative) gives another answer than the scalar unknowns'
    # face fluxes. Both are direct linear solves on 144 cells: equal to round-off.
    np.testing.assert_allclose(_two_diffusions(True), _two_diffusions(False), atol=1e-10)


@pytest.mark.parametrize("source", [0, 1.0])
def test_a_stacked_laplacian_beside_a_bare_number(source):
    """``q.t - stack([source, K*(b.xx + b.yy)])``: what is left of the stack once its Laplacian is a flux is a bare
    number, which must still be one value per cell (it was the constant vector (2,), refused as a cell term)."""
    K = 0.3

    def run(stacked):
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(10, 10)).quad().domain(time=(0.0, 0.1, 6))
        x, y, t = d.variable("interior", split=True)
        xi, yi, _ = d.variable("initial", split=True)
        g = jnn.sin(np.pi * xi) * jnn.cos(2.0 * yi)
        if stacked:
            q = d.unknown.vector(2)
            qi = q.bind(x=x, y=y, t=t)
            rhs = jnn.stack([source, K * (qi[1].xx + qi[1].yy)], axis=-1)
            cons = [qi.t - rhs, q(xi, yi) - jnn.stack([1.0 + g, g], axis=-1)]
        else:
            a, b = d.unknown(), d.unknown()
            ai, bi = a.bind(x=x, y=y, t=t), b.bind(x=x, y=y, t=t)
            cons = [ai.t - source + 0.0 * ai, bi.t - K * (bi.xx + bi.yy), a(xi, yi) - (1.0 + g), b(xi, yi) - g]
        return np.asarray(jno.fvm(cons).solve(time=jno.solve.erk("ssprk3", cfl=0.4)))

    # the same explicit steps on the same fluxes: equal to round-off
    np.testing.assert_allclose(run(True), run(False), atol=1e-12)
