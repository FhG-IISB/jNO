"""Radiation written as the equation: d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v * dS, lowered by jno.fem.

Oracles: two long concentric grey cylinders exchange Q = 2 pi r1 sigma (T1^4 - T2^4) / (1/eps1 + (r1/r2)(1/eps2 - 1))
per unit length (Modest, Radiative Heat Transfer, Ch. 5); the same gap in a coupled solve against its series
solution (conduction in the core, the gas and the shell, radiation across the gas); and the analytic tangent
against forward-mode autodiff of the residual."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import config

import jno
from jno.domain.radiation import STEFAN_BOLTZMANN as SIGMA
from jno.domain.radiation import lower_radiation_terms

config.update("jax_enable_x64", True)
R1, R2, R3 = 0.3, 0.6, 0.8


def _cylinders(eps1=0.6, eps2=0.8, h=0.04, **kw):
    disk = jno.shape.disk
    return (
        (
            disk(0, 0, R1).name("core").attach(eps=eps1, k=20.0, q=0.0)
            + (disk(0, 0, R3) - disk(0, 0, R2)).name("shell").attach(eps=eps2, k=20.0, q=0.0)
            + disk(0, 0, R2).name("gas").attach(kappa=0.0, eps=1.0, k=0.05, q=0.0)
        )
        .sized(h)
        .domain(**kw)
    )


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("JNO_CACHE", str(tmp_path))
    yield


def test_the_term_reproduces_the_two_cylinder_exchange():
    eps1, eps2, T1, T2 = 0.6, 0.8, 1500.0, 900.0
    d = _cylinders(eps1, eps2)
    T, v = d.fem_symbols(names=("T", "v"))
    term = d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v
    c = lower_radiation_terms([term])[0]
    pts = np.asarray(d.mesh.points)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    q = np.asarray(c.residual_fn(np.where(rho < 0.5 * (R1 + R2), T1, T2)))
    exact = 2 * np.pi * R1 * SIGMA * (T1**4 - T2**4) / (1 / eps1 + (R1 / R2) * (1 / eps2 - 1))
    assert q[np.isclose(rho, R1, atol=1e-9)].sum() == pytest.approx(exact, rel=1e-2)
    assert q.sum() == pytest.approx(0.0, abs=1e-6 * exact)


def test_conduction_and_radiation_across_the_gap_match_the_series_solution():
    """A heated core, a conducting gas gap and a shell held at 300 K outside: the power P = q pi r1^2 (per unit
    length) leaves the core by conduction through the gas plus radiation across it, then conducts through the
    shell. With T2 = 300 + P ln(r3/r2) / (2 pi k), T1 solves
        P = 2 pi r1 sigma (T1^4 - T2^4) / (1/eps1 + (r1/r2)(1/eps2 - 1)) + 2 pi k_gas (T1 - T2) / ln(r2/r1)."""
    from scipy.optimize import brentq

    eps1, eps2, k, k_gas, q = 0.6, 0.8, 20.0, 0.05, 5e5
    d = _cylinders(eps1, eps2, h=0.02)
    d.attach("core", q=q)
    T, v = d.fem_symbols(names=("T", "v"))
    x, y, _ = d.variable("interior", split=True)
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    outer = d.variable("boundary", split=True)[:2]
    T.initialize(800.0)
    fem = jno.fem(
        [
            d.k * (Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y)) - d.q * vi,
            d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v,
            T(*outer) - 300.0,
        ]
    )
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, atol=1e-8, rtol=0.0, max_steps=60)))

    P = q * np.pi * R1**2
    T2 = 300.0 + P * np.log(R3 / R2) / (2 * np.pi * k)
    rad = lambda T1: 2 * np.pi * R1 * SIGMA * (T1**4 - T2**4) / (1 / eps1 + (R1 / R2) * (1 / eps2 - 1))  # noqa: E731
    T1 = brentq(lambda T1: rad(T1) + 2 * np.pi * k_gas * (T1 - T2) / np.log(R2 / R1) - P, T2, 5000.0)
    pts = np.asarray(d.mesh.points)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    assert sol[np.isclose(rho, R2, atol=1e-9)].mean() == pytest.approx(T2, rel=1e-3)
    assert sol[np.isclose(rho, R1, atol=1e-9)].mean() == pytest.approx(T1, rel=1e-3)
    assert sol.max() == pytest.approx(T1 + q * R1**2 / (4 * k), rel=1e-3)  # the core's parabola on top


def test_axisymmetric_term_reproduces_two_concentric_spheres():
    """The same geometry read as a meridian half-plane: two concentric spheres, Q = 4 pi r1^2 sigma (T1^4 - T2^4)
    / (1/eps1 + (r1/r2)^2 (1/eps2 - 1)). Axisymmetry comes from the domain, the ring measure from dS = 2 pi r."""
    eps1, eps2, T1, T2 = 0.6, 0.8, 1500.0, 900.0
    disk, rect = jno.shape.disk, jno.shape.rect
    half = lambda rad: disk(0, 0, rad) & rect(0, -rad, rad, rad)  # noqa: E731
    d = (
        (
            half(R1).name("core").attach(eps=eps1)
            + (half(R3) - half(R2)).name("shell").attach(eps=eps2)
            + half(R2).name("gas").attach(kappa=0.0, eps=1.0)
        )
        .sized(0.03)
        .domain(axisymmetric=True)
    )
    T, v = d.fem_symbols(names=("T", "v"))
    r, z, _ = d.variable("interior", split=True)
    c = lower_radiation_terms([d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v * (2 * np.pi * r)])[0]
    pts = np.asarray(d.mesh.points)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    q = np.asarray(c.residual_fn(np.where(rho < 0.5 * (R1 + R2), T1, T2)))
    exact = 4 * np.pi * R1**2 * SIGMA * (T1**4 - T2**4) / (1 / eps1 + (R1 / R2) ** 2 * (1 / eps2 - 1))
    assert q[np.isclose(rho, R1, atol=1e-9)].sum() == pytest.approx(exact, rel=2e-2)
    assert q.sum() == pytest.approx(0.0, abs=1e-6 * exact)


def test_guards():
    d = _cylinders()
    T, v = d.fem_symbols(names=("T", "v"))
    x, y, _ = d.variable("interior", split=True)
    Ti = T.bind(x=x, y=y)
    with pytest.raises(NotImplementedError, match="not with their coordinate-bound views"):
        lower_radiation_terms([d.eps(T) * (SIGMA * Ti**4 - d.irradiation(T)) * v])
    with pytest.raises(ValueError, match="exactly one test function"):
        lower_radiation_terms([d.eps(T) * (SIGMA * T**4 - d.irradiation(T))])
    e = jno.np.parameter((1,), name="e")
    with pytest.raises(NotImplementedError, match="trainable parameter"):
        jno.fem([e * (SIGMA * T**4 - d.irradiation(T)) * v])
    with pytest.raises(ValueError, match="linear in the test function"):
        lower_radiation_terms([d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v + 1.0 + 0.0 * v])
    disk = jno.shape.disk
    d2 = (disk(0, 0, R1).name("core").attach(eps=0.5) + disk(0, 0, R2).name("gas").attach(eps=1.0)).sized(0.05).domain()
    T2, v2 = d2.fem_symbols(names=("T", "v"))
    with pytest.raises(ValueError, match="kappa"):
        lower_radiation_terms([d2.eps(T2) * (SIGMA * T2**4 - d2.irradiation(T2)) * v2])


def test_the_term_takes_a_field_from_d_unknown():
    """``T = d.unknown()`` (a valued field for jno.fdm) stands for its trial symbol in the radiation term too:
    the full solve equals the fem_symbols spelling."""

    def solve(by_unknown):
        d = _cylinders(h=0.08)
        if by_unknown:
            T = d.unknown.scalar("T")
            v = T.test()
        else:
            T, v = d.fem_symbols(names=("T", "v"))
        x, y, _ = d.variable("interior", split=True)
        Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
        q = 1e4 * jno.np.exp(-((x**2 + y**2) / 0.05))
        xo, yo, _ = d.variable("boundary", split=True)
        T.initialize(800.0)
        fem = jno.fem(
            [
                d.k * (Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y)) - q * vi,
                d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v,
                T(xo, yo) - 500.0,
            ]
        )
        return np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=0.0)))

    np.testing.assert_allclose(solve(True), solve(False), rtol=1e-12)


def _lowered(law=True, h=0.08):
    d = _cylinders(h=h)
    if law:
        d.attach("core", eps=lambda T: 0.3 + 2e-4 * T)
    T, v = d.fem_symbols(names=("T", "v"))
    c = lower_radiation_terms([d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v])[0]
    pts = np.asarray(d.mesh.points)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    u = 600.0 + 900.0 * np.exp(-((rho / 0.4) ** 2)) + 40.0 * np.sin(7 * pts[:, 0] + 3 * pts[:, 1])
    return c, d, jnp.asarray(u)


def test_the_analytic_tangent_equals_forward_mode_autodiff():
    """The block `jno.fem` assembles (dG/dT = F M^-1 diag(h), no JVPs) is the Jacobian of the residual on the
    radiating nodes -- with a temperature law for the core's emissivity, so the d eps/dT term is exercised."""
    c, _, u = _lowered()
    S = jnp.asarray(c._support)
    ref = jax.jacfwd(lambda us: c.residual_fn(u.at[S].set(us))[S])(u[S])
    np.testing.assert_allclose(
        np.asarray(c._tangent_block(u)), np.asarray(ref), rtol=1e-8, atol=1e-8 * float(jnp.abs(ref).max())
    )
