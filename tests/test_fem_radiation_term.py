"""Radiation written as the equation: d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v * dS, lowered by jno.fem.

Oracles: two long concentric grey cylinders exchange Q = 2 pi r1 sigma (T1^4 - T2^4) / (1/eps1 + (r1/r2)(1/eps2 - 1))
per unit length (Modest, Radiative Heat Transfer, Ch. 5); and in the plane the term must reproduce the explicit
enclosure spelling (rad.load of the same per-face flux) to round-off, since both scatter half a face to each node."""

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
    from jno.domain.enclosure import set_radiation_storage

    set_radiation_storage("auto")


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


def test_the_term_solves_like_the_explicit_enclosure():
    """Conduction in all three regions, a heat source in the core, the outer surface held at 300 K: the solve with
    the radiation written as the equation equals the solve with rad.load(...) of the same flux."""
    d = _cylinders()
    d.attach("core", q=5e5)
    T, v = d.fem_symbols(names=("T", "v"))
    x, y, _ = d.variable("interior", split=True)
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    outer = d.variable("boundary", split=True)[:2]
    conduction = d.k * (Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y)) - d.q * vi
    newton = jno.solve.newton(direct=True, atol=1e-8, rtol=0.0, max_steps=60)
    x0 = np.full(len(d.mesh.points), 800.0)

    as_math = jno.fem([conduction, d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v, T(*outer) - 300.0])
    T_math = np.asarray(as_math.solve(x0=x0, nonlinear=newton))

    rad = d.enclosure(medium="gas", enforce_closure=True)
    Te = rad.field(T)
    e = d.eps(Te)
    explicit = jno.fem([conduction, rad.load(e * (SIGMA * Te**4 - rad.irradiation(Te, eps=e))), T(*outer) - 300.0])
    T_expl = np.asarray(explicit.solve(x0=x0, nonlinear=newton))
    assert T_math.max() > 400.0  # the core is heated
    np.testing.assert_allclose(T_math, T_expl, atol=1e-6)


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


def _lowered(storage="dense", law=True, h=0.08):
    from jno.domain.enclosure import set_radiation_storage

    set_radiation_storage(storage)
    d = _cylinders(h=h)
    if law:
        d.attach("core", eps=lambda T: 0.3 + 2e-4 * T)
    T, v = d.fem_symbols(names=("T", "v"))
    c = lower_radiation_terms([d.eps(T) * (SIGMA * T**4 - d.irradiation(T)) * v])[0]
    pts = np.asarray(d.mesh.points)
    rho = np.hypot(pts[:, 0], pts[:, 1])
    u = 600.0 + 900.0 * np.exp(-((rho / 0.4) ** 2)) + 40.0 * np.sin(7 * pts[:, 0] + 3 * pts[:, 1])
    return c, d, jnp.asarray(u)


@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_the_analytic_tangent_equals_forward_mode_autodiff(storage):
    """The block `jno.fem` assembles (dG/dT = F M^-1 diag(h), no JVPs) is the Jacobian of the residual on the
    radiating nodes -- with a temperature law for the core's emissivity, so the d eps/dT term is exercised."""
    c, _, u = _lowered(storage)
    S = jnp.asarray(c._support)
    ref = jax.jacfwd(lambda us: c.residual_fn(u.at[S].set(us))[S])(u[S])
    np.testing.assert_allclose(
        np.asarray(c._tangent_block(u)), np.asarray(ref), rtol=1e-8, atol=1e-8 * float(jnp.abs(ref).max())
    )


def test_sparse_storage_gives_the_same_irradiation():
    cd, _, u = _lowered("dense")
    cs, d, _ = _lowered("sparse")
    assert all(p.storage == "sparse" for p in d.__dict__["_radiation_faces"].parts)
    np.testing.assert_allclose(np.asarray(cs.residual_fn(u)), np.asarray(cd.residual_fn(u)), rtol=1e-9, atol=1e-6)


def test_auto_storage_is_dense_for_a_small_enclosure():
    _, d, _ = _lowered("auto")
    assert all(p.storage == "dense" for p in d.__dict__["_radiation_faces"].parts)
    with pytest.raises(ValueError, match="storage must be"):
        d.__dict__["_radiation_faces"].parts[0].set_storage("coo")


def test_jno_setup_sets_the_radiation_storage(tmp_path):
    from jno.domain.enclosure import get_radiation_storage

    jno.setup(str(tmp_path / "run.py"), radiation_storage="sparse")
    assert get_radiation_storage() == "sparse"
    jno.setup(str(tmp_path / "run.py"))
    assert get_radiation_storage() == "auto"  # the default is back unless set again
    with pytest.raises(ValueError, match="radiation_storage"):
        jno.setup(str(tmp_path / "run.py"), radiation_storage="coo")
