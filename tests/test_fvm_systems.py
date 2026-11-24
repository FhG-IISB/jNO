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


@pytest.mark.parametrize("n", [1, 2, 5])
def test_small_block_solves_recover_a_known_solution_where_rows_must_be_swapped(n):
    # Name the break: the per-cell and per-face blocks are solved by an elimination of their own (batched LAPACK
    # deadlocked XLA:CPU on few threads); a wrong pivot or elimination step gives a wrong X. Each A is a row
    # permutation of a well-conditioned matrix, so its leading entry is zero for n > 1 and a step without the row swap
    # divides by zero. X is chosen, B = A X; the solve recovers X to round-off.
    from jno._fvm.trace import _small_solve

    rng = np.random.default_rng(n)
    base = rng.standard_normal((6, n, n)) + 3.0 * np.eye(n)
    A = base[:, np.roll(np.arange(n), 1)]  # row i of A is row i-1 of base
    if n > 1:
        A[:, 0, 0] = 0.0
    X = rng.standard_normal((6, n, 3))
    got = np.asarray(_small_solve(jnp.asarray(A), jnp.asarray(A @ X)))
    np.testing.assert_allclose(got, X, atol=1e-11)


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


def _steady_conducting_gas_error(n):
    """A gas at rest in a tube between a reservoir (p = 1, T = 1) and a wall held at T = 2: the steady Euler
    equations with conduction of T = p/rho, written as stacked fluxes only. The pressure is the reservoir's and the
    temperature linear; returns the largest error of T and of p."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 0.1).structured(n=(n, 1)).quad().domain()
    x, y, _ = d.variable("interior", split=True)
    q, T = d.unknown.vector(3), d.unknown()
    qi, Ti = q.bind(x=x, y=y), T.bind(x=x, y=y)
    rho, m, En = qi[0], qi[1], qi[2]
    u = m / rho
    p = (GAMMA - 1.0) * (En - 0.5 * m * u)
    F = jnn.stack([m, m * u + p, u * (En + p)], axis=-1)
    cons = [F.d(x, scheme=jno.fv(flux="roe")) - jnn.stack([0, 0, 0.1 * Ti.xx], axis=-1), Ti - p / rho]
    xl, yl, _ = d.variable("left", split=True)
    cons += [q(xl, yl) - jnn.stack([1.0 + 0.0 * xl, 0.0 * xl, 1.0 / (GAMMA - 1.0) + 0.0 * xl], axis=-1), T(xl, yl) - 1.0]
    xr, yr, _ = d.variable("right", split=True)
    cons += [q.bind(x=xr, y=yr)[1] - 0.0, T(xr, yr) - 2.0]
    gas = jno.fvm(cons)
    N = gas.centroids.shape[0]
    at_rest = np.concatenate([np.ones(N), np.zeros(N), np.full(N, 1.0 / (GAMMA - 1.0)), np.ones(N)])
    s = np.asarray(gas.solve(x0=at_rest)).reshape(4, N)
    xc = np.asarray(gas.centroids)[:, 0]
    pressure = (GAMMA - 1.0) * (s[2] - 0.5 * s[1] ** 2 / s[0])
    return np.abs(s[3] - (1.0 + xc)).max(), np.abs(pressure - 1.0).max()


def test_a_steady_conducting_gas_with_an_algebraic_temperature():
    # Name the break: with no bound view on top of the vector equation (no qi.t, as in a march), it had no region,
    # so the algebraic row T - p/rho beside it was taken for a pin and the system refused ("one PDE equation per
    # unknown"). Solved, the flux's dissipation on the density jump (a stationary contact) drives a small flow, an
    # error of second order: measured 5.2e-3 and 1.5e-3 on 10 and 20 cells (order 1.76), 4.2e-4 on 40 (1.88).
    (e1, p1), (e2, p2) = _steady_conducting_gas_error(10), _steady_conducting_gas_error(20)
    assert e1 < 1e-2 and max(p1, p2) < 1e-2
    assert 1.6 < np.log2(e1 / e2) < 2.2


def _euler1d_flux(Q):
    rho, m, E = Q[:, 0], Q[:, 1], Q[:, 2]
    p = (GAMMA - 1.0) * (E - 0.5 * m * m / rho)
    return jnp.stack([m, m * m / rho + p, m * (E + p) / rho], axis=1)


def _state(rho, u, p):
    return jnp.asarray([[rho, rho * u, p / (GAMMA - 1.0) + 0.5 * rho * u * u]])


@pytest.mark.parametrize("smooth", [False, True])
def test_a_smooth_hll_flux_has_a_derivative_at_a_sonic_face(smooth):
    # A sonic left state (u = c, so its slowest wave speed is 0) next to a supersonic right one: the HLL wave bound
    # s_L = min(min(lo_L, lo_R), 0) sits exactly on its kinks, and the flux, which depends on s_L through fL - fR,
    # has no derivative there: the jvp picks one side, a central difference averages them. Smoothed (smooth=True),
    # the jvp is the derivative: central differences agree to O(h^2), and the flux moves by O(delta) = 1e-3 of the
    # wave speeds.
    from jno._fvm.trace import _eig_extremes, _flux_jacobian, _hll

    QL, QR = _state(1.0, np.sqrt(GAMMA * (1.0 / GAMMA) / 1.0), 1.0 / GAMMA), _state(0.8, 1.5, 0.5)

    def flux(ql, qr, sm=smooth):
        ext = [_eig_extremes(_flux_jacobian(_euler1d_flux, q)) for q in (ql, qr)]
        return _hll(_euler1d_flux(ql), _euler1d_flux(qr), ql, qr, ext[0], ext[1], smooth=sm)

    lo_L, _ = _eig_extremes(_flux_jacobian(_euler1d_flux, QL))
    assert abs(float(lo_L[0])) < 1e-12  # sonic indeed
    v = jnp.asarray([[0.3, -0.2, 0.5]])
    jvp = jax.jvp(lambda q: flux(q, QR), (QL,), (v,))[1]
    h = 1e-6
    fd = (flux(QL + h * v, QR) - flux(QL - h * v, QR)) / (2 * h)
    mismatch = float(jnp.abs(jvp - fd).max() / jnp.abs(fd).max())
    if smooth:
        assert mismatch < 1e-6
        rel = float(jnp.abs(flux(QL, QR) - flux(QL, QR, False)).max() / jnp.abs(flux(QL, QR, False)).max())
        assert rel < 1e-2
    else:
        assert mismatch > 1e-3  # the kink the smooth form removes


@pytest.mark.parametrize("smooth", [False, True])
def test_a_smooth_spectral_radius_is_smooth_for_a_gas_at_rest(smooth):
    # At rest |u - c| = |u + c|: the spectral radius max(|lo|, |hi|) (Rusanov's speed, Roe's entropy fix) has a kink
    # in u there, its slope jumping from -1 to +1 (JAX splits a tie, so the jump shows on either side, not at it).
    # Smoothed within delta = 1e-3 c, the slope a micro-step either side differs by O(1e-3), not by 2.
    from jno._fvm.trace import _flux_jacobian, _spectral_radius

    def slope(u):
        q = _state(1.0, u, 1.0)
        return float(jax.jvp(lambda w: _spectral_radius(_flux_jacobian(_euler1d_flux, w), smooth=smooth),
                             (q,), (jnp.asarray([[0.0, 1.0, 0.0]]),))[1][0])  # fmt: skip

    jump = abs(slope(1e-6) - slope(-1e-6))
    if smooth:
        assert jump < 1e-2
    else:
        assert jump > 1.0
