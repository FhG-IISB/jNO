"""A load from one problem on another: gas pressure on a thick wall that also carries a radial temperature.

A long thick cylinder, a < r < b, in plane strain, held at T_a inside and T_b outside (steady conduction,
ΔT = T - T0 logarithmic in r) and loaded on its bore by the pressure of an fvm gas at rest. Its radial
displacement is u = K1 I(r)/r + C1 r + C2/r with I(r) = ∫_a^r ΔT s ds, K1 = α(1+ν)/(1-ν), and C1, C2 from
σ_r(a) = -p, σ_r(b) = 0 (Timoshenko & Goodier, Theory of Elasticity, §150): the reference, computed here by quadrature
from the closed form. A quarter of the section with symmetry rollers.

The traction condition σ·n_s + p_g n_s = 0 differentiates only the wall, so it is a load on the wall and the gas
never hears back: one sweep per window, nothing to iterate.
"""

import numpy as np
import pytest
from scipy.integrate import quad

import jax

import jno
import jno.jnp_ops as jnn
from jno._coupling.driver import CouplingSpec, couple

A, B = 0.5, 1.0
E_MOD, NU, ALPHA, K_TH = 1.0, 0.3, 0.5, 1.0
TA, TB, T0, P0, GAMMA = 1.0, 0.0, 0.0, 1.0, 1.4
LAM, G = E_MOD * NU / ((1 + NU) * (1 - 2 * NU)), E_MOD / (2 * (1 + NU))


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def exact_ur(r):
    dT = lambda s: TA - T0 + (TB - TA) * np.log(s / A) / np.log(B / A)  # noqa: E731
    integral = lambda s: quad(lambda z: dT(z) * z, A, s)[0]  # noqa: E731
    k1 = ALPHA * (1 + NU) / (1 - NU)
    c = E_MOD / ((1 + NU) * (1 - 2 * NU))

    def sigma_r(rr, c1, c2):
        u = k1 * integral(rr) / rr + c1 * rr + c2 / rr
        du = k1 * (dT(rr) - integral(rr) / rr**2) + c1 - c2 / rr**2
        return c * ((1 - NU) * du + NU * u / rr) - E_MOD * ALPHA * dT(rr) / (1 - 2 * NU)

    # σ_r is affine in (C1, C2): two columns and the offset give the 2x2 system
    rows = []
    for rr, target in ((A, -P0), (B, 0.0)):
        s0 = sigma_r(rr, 0.0, 0.0)
        rows.append(([sigma_r(rr, 1.0, 0.0) - s0, sigma_r(rr, 0.0, 1.0) - s0], target - s0))
    c1, c2 = np.linalg.solve(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    return np.array([k1 * integral(rr) / rr + c1 * rr + c2 / rr for rr in np.atleast_1d(r)])


def _wall(h):
    inner, trace, symgrad, ident, grad = jno.np.inner, jno.np.trace, jno.np.symgrad, jno.np.identity, jno.np.grad
    ring = (jno.shape.disk(0.0, 0.0, B) - jno.shape.disk(0.0, 0.0, A)) & jno.shape.rect(0.0, 0.0, 2.0, 2.0)
    d = ring.sized(h).domain()
    # the two arcs share the geometry's `arc` tag: split it by radius (the facet centroids lie inside a circle)
    d.tag("bore", lambda x, n, name: (np.asarray(name) == "arc") & (np.hypot(x[..., 0], x[..., 1]) < 0.5 * (A + B)))
    d.tag("outer", lambda x, n, name: (np.asarray(name) == "arc") & (np.hypot(x[..., 0], x[..., 1]) > 0.5 * (A + B)))
    T, S = d.fem_symbols(names=("T", "S"), order=1)
    w, phi = d.fem_symbols(value_shape=(2,), names=("w", "phi"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    ax = [xi, yi]
    I2 = ident(2)

    def sigma(wv, Tv, axes):
        e = symgrad(wv, axes)
        return 2.0 * G * e + LAM * trace(e) * I2 - (3.0 * LAM + 2.0 * G) * ALPHA * (Tv - T0) * I2

    xs, ys, _ = d.variable("bore", split=True)
    xo, yo, _ = d.variable("outer", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    fem = jno.fem(
        [
            K_TH * inner(grad(T, ax), grad(S, ax), n_contract=1),
            inner(sigma(w, T.bind(x=xi, y=yi), ax), symgrad(phi, ax), n_contract=2),
            T(xs, ys) - TA,
            T(xo, yo) - TB,
            w(xl, yl)[0] - 0.0,  # symmetry rollers
            w(xb, yb)[1] - 0.0,
        ]
    )
    ns = d.variable("bore", normals=True)
    traction = inner(sigma(w.bind(x=xs, y=ys), T.bind(x=xs, y=ys), [xs, ys]), ns, n_contract=1)
    return fem, traction, ns, d, w


def _gas(h):
    d = jno.shape.disk(0.0, 0.0, A) & jno.shape.rect(0.0, 0.0, 2.0, 2.0)
    d = d.sized(h).domain(time=(0.0, 0.02, 3))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    q = d.unknown.vector(4)
    qi = q.bind(x=x, y=y, t=t)
    rho, mx, my, E = qi[0], qi[1], qi[2], qi[3]
    u, v = mx / rho, my / rho
    p = (GAMMA - 1.0) * (E - 0.5 * (mx * u + my * v))
    s = jno.fv(flux="hll", limiter="minmod")
    Fx = jnn.stack([mx, mx * u + p, my * u, u * (E + p)], axis=-1)
    Fy = jnn.stack([my, mx * v, my * v + p, v * (E + p)], axis=-1)
    cons = [qi.t + Fx.d(x, scheme=s) + Fy.d(y, scheme=s)]
    for tag in ("left", "bottom", "arc"):  # slip walls: the symmetry lines and the bore
        xe, ye, _ = d.variable(tag, split=True)
        ne = d.variable(tag, normals=True)
        qe = q.bind(x=xe, y=ye)
        cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
    rest = jnn.stack([1.0 + 0.0 * xi, 0.0 * xi, 0.0 * xi, P0 / (GAMMA - 1.0) + 0.0 * xi], axis=-1)
    cons.append(q(xi, yi) - rest)
    xa, ya, _ = d.variable("arc", split=True)
    qa = q.bind(x=xa, y=ya)
    pa = (GAMMA - 1.0) * (qa[3] - 0.5 * (qa[1] * qa[1] + qa[2] * qa[2]) / qa[0])
    return jno.fvm(cons), pa


def test_gas_pressure_and_a_radial_temperature_load_a_thick_cylinder():
    wall, traction, ns, dw, w = _wall(0.03)
    gas, pg = _gas(0.05)
    c = couple([gas, wall], [traction + pg * ns], CouplingSpec(time=jno.solve.erk("ssprk3", cfl=0.5)))
    _, sw = c.solve()
    # a load: the wall takes it, the gas hears nothing back, so one sweep settles every window
    assert c.assign[0] == 1 and c.window_iterations == [1, 1]
    state = sw[-1]
    sl = wall.blocks[wall.block_index(w)]
    W = np.asarray(state[sl.start : sl.stop]).reshape(-1, 2)  # node-major (x, y) per node
    pts = np.asarray(dw.mesh_connectivity["points"])[:, :2]
    r = np.hypot(pts[:, 0], pts[:, 1])
    ur = (W[:, 0] * pts[:, 0] + W[:, 1] * pts[:, 1]) / r
    ref = exact_ur(r)
    # P1 elements, h ~ 0.03 on a polygonal bore: 0.2 % of the displacement (the wall alone, loaded by hand, is the same)
    assert np.abs(ur - ref).max() <= 0.005 * np.abs(ref).max()
    # the hoop part of the displacement vanishes (axisymmetric load on a symmetric section)
    ut = (-W[:, 0] * pts[:, 1] + W[:, 1] * pts[:, 0]) / r
    assert np.abs(ut).max() <= 0.01 * np.abs(ref).max()


def test_an_axisymmetric_gas_presses_on_an_axisymmetric_wall():
    # the meridian (r, z) of a long cylinder, a < r < b, plane-strain ends, loaded on its bore by an axisymmetric
    # gas at rest in r < a: Lamé, u_r = (1+ν) p a² / (E (b²-a²)) [(1-2ν) r + b²/r]. Both forms carry 2π r, and
    # so does the load the coupling writes on the wall (measure "axisymmetric:0", r the first coordinate).
    a, b, h, p0, E, nu = 0.5, 1.0, 0.3, 1.0, 50.0, 0.3
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    dw = jno.shape.rect(a, 0.0, b, h, size=0.04).domain()
    u, v = dw.fem_symbols(value_shape=(2,), order=2)
    r, z, _ = dw.variable("interior", split=True)
    rt, zt, _ = dw.variable("top", split=True)
    rb, zb, _ = dw.variable("bottom", split=True)

    def strains(w, rr):
        return w.x[0], w.y[1], w[0] / rr, 0.5 * (w.y[0] + w.x[1])  # e_rr, e_zz, e_qq (the hoop), e_rz

    eu, ev = strains(u.bind(x=r, y=z), r), strains(v.bind(x=r, y=z), r)
    tr_u, tr_v = eu[0] + eu[1] + eu[2], ev[0] + ev[1] + ev[2]
    energy = lam * tr_u * tr_v + 2.0 * mu * (eu[0] * ev[0] + eu[1] * ev[1] + eu[2] * ev[2] + 2.0 * eu[3] * ev[3])
    wall = jno.fem([energy * (2.0 * np.pi * r), u(rt, zt)[1] - 0.0, u(rb, zb)[1] - 0.0])
    ra, za, _ = dw.variable("left", split=True)
    na = dw.variable("left", normals=True)
    ua = u.bind(x=ra, y=za)
    e_rr, e_zz, e_qq, e_rz = strains(ua, ra)
    trace_e = e_rr + e_zz + e_qq
    sig_rr, sig_zz, sig_rz = lam * trace_e + 2.0 * mu * e_rr, lam * trace_e + 2.0 * mu * e_zz, 2.0 * mu * e_rz
    traction = jnn.stack([sig_rr * na[..., 0] + sig_rz * na[..., 1], sig_rz * na[..., 0] + sig_zz * na[..., 1]], axis=-1)

    dg = jno.shape.rect(0.0, 0.0, a, h, size=0.05).domain(time=(0.0, 0.02, 3))
    x, y, t = dg.variable("interior", split=True)
    xi, yi, _ = dg.variable("initial", split=True)
    q = dg.unknown.vector(4)  # (rho, m_r, m_z, E), r = x
    qi = q.bind(x=x, y=y, t=t)
    rho, mr, mz, En = qi[0], qi[1], qi[2], qi[3]
    ur_, uz_ = mr / rho, mz / rho
    pr = (GAMMA - 1.0) * (En - 0.5 * (mr * ur_ + mz * uz_))
    s = jno.fv(flux="hll", limiter="minmod")
    Fr = jnn.stack([mr, mr * ur_ + pr, mz * ur_, ur_ * (En + pr)], axis=-1)
    Fz = jnn.stack([mz, mr * uz_, mz * uz_ + pr, uz_ * (En + pr)], axis=-1)
    hoop = jnn.stack([0.0 * pr, pr, 0.0 * pr, 0.0 * pr], axis=-1)
    cons = [x * qi.t + (x * Fr).d(x, scheme=s) + (x * Fz).d(y, scheme=s) - hoop]
    for tag in ("left", "right", "top", "bottom"):
        xe, ye, _ = dg.variable(tag, split=True)
        ne = dg.variable(tag, normals=True)
        qe = q.bind(x=xe, y=ye)
        cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
    rest = jnn.stack([1.0 + 0.0 * xi, 0.0 * xi, 0.0 * xi, p0 / (GAMMA - 1.0) + 0.0 * xi], axis=-1)
    cons.append(q(xi, yi) - rest)
    gas = jno.fvm(cons)
    xg, yg, _ = dg.variable("right", split=True)
    qg = q.bind(x=xg, y=yg)
    pg = (GAMMA - 1.0) * (qg[3] - 0.5 * (qg[1] * qg[1] + qg[2] * qg[2]) / qg[0])

    c = couple(
        [gas, wall], [traction + pg * na], CouplingSpec(measure="axisymmetric:0", time=jno.solve.erk("ssprk3", cfl=0.5))
    )
    _, sw = c.solve()
    sol = np.asarray(sw[-1]).reshape(-1, 2)
    pts = np.asarray(wall.points)
    exact = (1 + nu) * p0 * a**2 / (E * (b**2 - a**2)) * ((1 - 2 * nu) * pts[:, 0] + b**2 / pts[:, 0])
    np.testing.assert_allclose(sol[:, 0], exact, atol=0.01 * np.abs(exact).max())
