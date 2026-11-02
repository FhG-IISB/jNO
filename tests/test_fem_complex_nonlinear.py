"""Complex fields written as math in linear, nonlinear and coupled forms; integral equations; fem.split.

* A complex constant that meets a regional coefficient first (``1j * w * d.sigma * A * W``) is split into its real
  and imaginary parts (it used to fail to assemble, or drop the imaginary part).
* A complex field in a NONLINEAR coupled form (sigma(T), |A|^2 Joule heating) gives exactly the real-pair
  formulation written by hand.
* ``(expr * dV).integrate() - P`` is the equation of the constant unknown that has none.
Oracles: the Joule power of a conducting cylinder in a long coil (Bessel functions); the hand-written real pair
(identical discretisation); a linear problem whose constant follows from one extra solve.
"""

import numpy as np
import pytest
from jax import config
from scipy.integrate import quad
from scipy.special import jv

import jno

config.update("jax_enable_x64", True)
MU0 = 4e-7 * np.pi
F, SIG, RAD, K, L, T_COIL = 5e4, 1e5, 0.02, 1000.0, 0.01, 0.002
W_ = 2 * np.pi * F


def _rod():
    R = jno.shape.rect
    return (
        (
            R(0, 0, RAD, L).name("rod").attach(sigma=SIG, J=0.0).size(0.0005)
            + R(0.03, 0, 0.03 + T_COIL, L).name("coil").attach(sigma=0.0, J=K / T_COIL)
            + R(0, 0, 0.05, L).name("air").attach(sigma=0.0, J=0.0)
        )
        .sized(0.001)
        .domain()
        .attach(nu=1 / MU0)
    )


def _curl_dot(a, b, r, z):
    return a.d(z) * b.d(z) + (a.d(r) + a / r) * (b.d(r) + b / r)


def test_complex_constant_meeting_a_regional_coefficient_matches_bessel():
    d = _rod()
    A, W = d.fem_symbols(names=("A", "W"), complex=True)
    r, z, _ = d.variable("interior", split=True)
    Ai, Wi = A.bind(r=r, z=z), W.bind(r=r, z=z)
    dV = 2 * np.pi * r
    fem = jno.fem(
        [
            d.nu * _curl_dot(Ai, Wi, r, z) * dV + 1j * W_ * d.sigma * Ai * Wi * dV - d.J * Wi * dV,
            A(*d.variable("left", split=True)[:2]) - 0.0,
        ]
    )
    a = fem.split(fem.solve(linear=jno.solve.lu()))["A"]
    assert np.iscomplexobj(a)
    pts, tri = np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])
    blocks = [c.type == "triangle" for c in d.mesh.cells]
    rod = np.concatenate([np.asarray(i) for i, b in zip(d.mesh.cell_sets["rod"], blocks) if b and i is not None and len(i)])
    p = pts[tri[rod]]
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    P = (
        sum(
            (
                area
                / 3
                * 2
                * np.pi
                * 0.5
                * (p[:, i, 0] + p[:, j, 0])
                * np.abs(0.5 * (a[tri[rod, i]] + a[tri[rod, j]])) ** 2
            ).sum()
            for i, j in ((0, 1), (1, 2), (2, 0))
        )
        * 0.5
        * SIG
        * W_**2
    )
    k = (1 - 1j) / np.sqrt(2 / (W_ * MU0 * SIG))
    exact = (
        L
        * quad(lambda s: abs(K * k * jv(1, k * s) / jv(0, k * RAD)) ** 2 / (2 * SIG) * 2 * np.pi * s, 0, RAD, limit=200)[0]
    )
    assert P == pytest.approx(exact, rel=2e-3)


def test_complex_field_in_a_nonlinear_coupled_form_equals_the_real_pair():
    """Induction heating of the rod: sigma(T) in the field equation, sigma w^2 |A|^2 / 2 in the heat equation, one
    form. Written with a complex field it must give the hand-written (Re A, Im A) system's answer."""
    sigma_T = lambda T: SIG / (1.0 + 2e-3 * (T - 300.0))  # noqa: E731

    def build(as_complex):
        d = _rod()
        d.attach("rod", sigma=sigma_T)
        d.attach("rod", k=20.0).attach("coil", k=400.0).attach("air", k=0.05)
        T, v = d.fem_symbols(names=("T", "v"))
        r, z, _ = d.variable("interior", split=True)
        Ti, vi = T.bind(r=r, z=z), v.bind(r=r, z=z)
        dV = 2 * np.pi * r
        sig = d.sigma(T)
        heat_bc = T(*d.variable("right", split=True)[:2]) - 300.0
        if as_complex:
            A, W = d.fem_symbols(names=("A", "W"), complex=True)
            Ai, Wi = A.bind(r=r, z=z), W.bind(r=r, z=z)
            field = [
                d.nu * _curl_dot(Ai, Wi, r, z) * dV + 1j * W_ * sig * Ai * Wi * dV - d.J * Wi * dV,
                A(*d.variable("left", split=True)[:2]) - 0.0,
            ]
            joule = 0.5 * W_**2 * sig * abs(Ai) ** 2
        else:
            A, W = d.fem_symbols(value_shape=(2,), names=("A", "W"))
            Ai, Wi = A.bind(r=r, z=z), W.bind(r=r, z=z)
            field = [
                d.nu * _curl_dot(Ai[0], Wi[0], r, z) * dV - W_ * sig * Ai[1] * Wi[0] * dV - d.J * Wi[0] * dV,
                d.nu * _curl_dot(Ai[1], Wi[1], r, z) * dV + W_ * sig * Ai[0] * Wi[1] * dV,
                A(*d.variable("left", split=True)[:2]) - 0.0,
            ]
            joule = 0.5 * W_**2 * sig * (Ai[0] ** 2 + Ai[1] ** 2)
        fem = jno.fem([*field, d.k * (Ti.d(r) * vi.d(r) + Ti.d(z) * vi.d(z)) * dV - joule * vi * dV, heat_bc])
        n = int(np.asarray(fem.offsets)[-1])
        x0 = np.zeros(n)
        sol = fem.split(
            fem.solve(x0=x0 + 300.0 * 0, nonlinear=jno.solve.newton(direct=True, rtol=1e-13, atol=0.0, max_steps=40))
        )
        return sol

    c, p = build(True), build(False)
    a_pair = p["A"][:, 0] + 1j * p["A"][:, 1]
    np.testing.assert_allclose(c["T"], p["T"], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(c["A"], a_pair, rtol=1e-9, atol=1e-14)
    assert c["T"].max() > 301.0  # it was heated


def test_integral_equation_sets_the_constant_unknown():
    """-lap u = I on the unit square, u = 0 on its boundary, and int u dA = P: I = P / c, c = int u for I = 1."""
    d = jno.shape.rect(0, 0, 1, 1).sized(0.05).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols(names=("u", "v"))
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    I = d.unknown.scalar(constant=True, name="I")  # noqa: E741  (the current, as written in the equations)
    P = 0.37
    fem = jno.fem([ui.d(x) * vi.d(x) + ui.d(y) * vi.d(y) - I * vi, u(xb, yb) - 0.0, ui.integrate() - P])
    sol = fem.split(fem.solve(linear=jno.solve.lu()))
    ref = jno.fem([ui.d(x) * vi.d(x) + ui.d(y) * vi.d(y) - 1.0 * vi, u(xb, yb) - 0.0])
    u1 = np.asarray(ref.solve(linear=jno.solve.lu()))
    c = float(np.asarray(sol["u"]).sum() * 0 + _p1_integral(d, u1))
    assert float(sol["I"]) == pytest.approx(P / c, rel=1e-8)


def _p1_integral(d, u):
    pts, tri = np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])
    p = pts[tri]
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    return float((area * u[tri].mean(axis=1)).sum())


def test_an_integral_equation_needs_one_free_constant():
    d = jno.shape.rect(0, 0, 1, 1).sized(0.2).domain()
    x, y, _ = d.variable("interior", split=True)
    u, v = d.fem_symbols(names=("u", "v"))
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    a = d.unknown.scalar(constant=True, name="a")
    b = d.unknown.scalar(constant=True, name="b")
    with pytest.raises(ValueError, match="found 2 such constants"):
        jno.fem([ui.d(x) * vi.d(x) + ui.d(y) * vi.d(y) - (a + b) * vi, ui.integrate() - 1.0])


def test_complex_field_with_a_constant_set_by_its_power():
    """The coil current I of the rod is the constant unknown of int sigma w^2 |A|^2 / 2 dV = P. The field is linear
    in I, so I = sqrt(P / P1) with P1 the power at I = 1 (one linear solve)."""
    d = _rod()
    A, W = d.fem_symbols(names=("A", "W"), complex=True)
    r, z, _ = d.variable("interior", split=True)
    Ai, Wi = A.bind(r=r, z=z), W.bind(r=r, z=z)
    dV = 2 * np.pi * r
    axis = A(*d.variable("left", split=True)[:2]) - 0.0
    joule = 0.5 * W_**2 * d.sigma * abs(Ai) ** 2

    def field(current):
        return d.nu * _curl_dot(Ai, Wi, r, z) * dV + 1j * W_ * d.sigma * Ai * Wi * dV - current * d.J * Wi * dV

    one = jno.fem([field(1.0), axis])
    a1 = one.split(one.solve(linear=jno.solve.lu()))["A"]
    P1 = float((0.5 * W_**2 * SIG) * _rod_power(d, a1))
    P = 4.0 * P1
    I = d.unknown.scalar(constant=True, name="I")  # noqa: E741
    fem = jno.fem([field(I), axis, (joule * dV).integrate() - P])
    A.initialize(a1)
    I.initialize(1.0)
    sol = fem.split(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=0.0, max_steps=30)))
    assert float(sol["I"]) == pytest.approx(np.sqrt(P / P1), rel=1e-6)
    np.testing.assert_allclose(sol["A"], float(sol["I"]) * a1, rtol=1e-6, atol=1e-12)


def _rod_power(d, a):
    """int_rod |A|^2 2 pi r dA for a P1 field (edge-midpoint rule, exact for the quadratic |A|^2)."""
    pts, tri = np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])
    blocks = [c.type == "triangle" for c in d.mesh.cells]
    rod = np.concatenate([np.asarray(i) for i, b in zip(d.mesh.cell_sets["rod"], blocks) if b and i is not None and len(i)])
    p = pts[tri[rod]]
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    return sum(
        (
            area / 3 * 2 * np.pi * 0.5 * (p[:, i, 0] + p[:, j, 0]) * np.abs(0.5 * (a[tri[rod, i]] + a[tri[rod, j]])) ** 2
        ).sum()
        for i, j in ((0, 1), (1, 2), (2, 0))
    )
