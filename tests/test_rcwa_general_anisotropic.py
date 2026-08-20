"""jno.rcwa engine with a GENERAL anisotropic layer — both ε̂ and μ̂ tensors, via fmmax's
``eigensolve_general_anisotropic_media`` (10 component grids: ε_xx..ε_zz, μ_xx..μ_zz).

This is the engine foundation for an in-plane uniaxial PML (a coordinate stretch is a diagonal ε̂ AND
μ̂) and for magnetic / magneto-optic media. A general layer with μ̂ = I must reduce exactly to the
ε-only anisotropic path; a genuine μ̂ ≠ I must change the result (magnetic response actually enters).
"""

import importlib.util
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the fmmax solve OOMs on a small GPU at these orders

import jax  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


from jno.rcwa import Rcwa  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")

PERIOD = (0.6, 0.6)
WL, ORDERS, D = 1.0, 60, 0.35
EPS = 6.0


def _T(layers):
    sol = Rcwa(layers, period=PERIOD, orders=ORDERS, wavelength=WL, assume_periodic=True).solve()
    return float(sol.efficiency("T")), float(sol.efficiency("R"))


def _iso(eps):
    return [(float("inf"), 1.0), (D, eps), (float("inf"), 1.0)]


def _aniso5(eps):  # (ε_xx, ε_xy, ε_yx, ε_yy, ε_zz)
    return [(float("inf"), 1.0), (D, (eps, 0.0, 0.0, eps, eps)), (float("inf"), 1.0)]


def _general10(eps, mu):  # (ε_xx..ε_zz, μ_xx..μ_zz) with diagonal μ̂ = μ·I
    slab = (eps, 0.0, 0.0, eps, eps, mu, 0.0, 0.0, mu, mu)
    return [(float("inf"), 1.0), (D, slab), (float("inf"), 1.0)]


@needs_fmmax
def test_general_mu_one_reduces_to_anisotropic():
    """μ̂ = I in the 10-tuple general layer must give exactly the ε-only anisotropic (5-tuple) result —
    the general eigensolve reduces to the anisotropic one when the medium is non-magnetic."""
    Tg, Rg = _T(_general10(EPS, 1.0))
    Ta, Ra = _T(_aniso5(EPS))
    assert Tg == pytest.approx(Ta, abs=1e-6)
    assert Rg == pytest.approx(Ra, abs=1e-6)


@needs_fmmax
def test_general_mu_one_matches_isotropic():
    """And with ε̂ = ε·I, μ̂ = I it matches the plain isotropic slab — full reduction to the scalar path."""
    Tg, Rg = _T(_general10(EPS, 1.0))
    Ti, Ri = _T(_iso(EPS))
    assert Tg == pytest.approx(Ti, abs=1e-6)
    assert Rg == pytest.approx(Ri, abs=1e-6)


@needs_fmmax
def test_magnetic_mu_changes_result():
    """A genuine μ̂ ≠ I changes the response: μ alters the wave impedance √(μ/ε), so reflection differs from
    the non-magnetic slab. Proves μ actually enters the solve (not silently dropped)."""
    Tn, Rn = _T(_general10(EPS, 1.0))
    Tm, Rm = _T(_general10(EPS, 2.0))
    assert abs(Rm - Rn) > 1e-2


@needs_fmmax
def test_general_lossless_conserves_energy():
    """A lossless general-anisotropic slab (real ε̂, μ̂) conserves energy: T + R ≈ 1."""
    T, R = _T(_general10(EPS, 2.0))
    assert T + R == pytest.approx(1.0, abs=5e-3)
