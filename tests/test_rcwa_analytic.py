"""Analytic anchors for ``jno.rcwa``: results that no other RCWA code is needed to check.

A stack without lateral structure is a thin-film problem, whatever lattice it is expanded on. Its T and R are the
Airy sums of the Fresnel coefficients, for every period, angle and polarization. That pins the ambient split, the
incident order, the oblique wavevector convention and the power normalization, including once the period lets
diffraction orders propagate (where an engine that picks the incident wave among the propagating ambient modes
goes wrong while still conserving energy).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402
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

WL = 1.0
K0 = 2 * np.pi / WL


def airy_slab(eps, d, kx, pol):
    """Power reflectance and transmittance of one film between two half-spaces (Born & Wolf, sec. 1.6.4).

    ``eps = (eps_top, eps_film, eps_bottom)`` may be complex, ``kx`` is the conserved in-plane wavenumber and
    ``pol`` is ``"s"`` (E out of the plane of incidence) or ``"p"`` (H out of the plane of incidence). The
    amplitude coefficients are those of the out-of-plane field, so one formula serves both polarizations with
    the weight ``w = 1`` (s) or ``w = 1/eps`` (p). T is zero when the bottom half-space carries no power.
    """
    eps = [complex(e) for e in eps]
    kz = [np.sqrt(e * K0**2 - kx**2 + 0j) for e in eps]
    kz = [k if k.imag >= 0 else -k for k in kz]
    q = [k if pol == "s" else k / e for k, e in zip(kz, eps)]  # the wave admittance of the out-of-plane field
    r01, r12 = (q[0] - q[1]) / (q[0] + q[1]), (q[1] - q[2]) / (q[1] + q[2])
    t01, t12 = 2 * q[0] / (q[0] + q[1]), 2 * q[1] / (q[1] + q[2])
    ph = np.exp(1j * kz[1] * d)
    den = 1 + r01 * r12 * ph**2
    r, t = (r01 + r12 * ph**2) / den, t01 * t12 * ph / den
    return abs(r) ** 2, abs(t) ** 2 * q[2].real / q[0].real


def _slab(eps, d, period, kx, pol, orders=9):
    rc = Rcwa(
        [(np.inf, eps[0]), (d, eps[1]), (np.inf, eps[2])],
        period=(period, period),
        orders=orders,
        wavelength=WL,
        k_in=(kx, 0.0),
        assume_periodic=True,
        polarization={"s": "y", "p": "x"}[pol],  # the plane of incidence is x-z
    )
    sol = rc.solve()
    return float(sol.efficiency("R")), float(sol.efficiency("T"))


# Periods on both sides of the first Rayleigh point: at 0.3 only the zeroth order propagates in every medium;
# at 0.6, 1.3 and 1.7 one or more diffraction orders propagate at the oblique angles.
PERIODS = (0.3, 0.6, 1.3, 1.7)


@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("theta_deg", [0.0, 20.0, 45.0, 70.0])
def test_uniform_slab_matches_airy_for_every_period(theta_deg, pol):
    """A lossless film (n = 1.5, d = 0.37 wavelengths) between air and n = 1.2: R and T equal the Airy values to
    round-off at each period, so they do not depend on the period at all."""
    eps, d = (1.0, 1.5**2, 1.2**2), 0.37
    kx = K0 * np.sin(np.radians(theta_deg))
    r_ref, t_ref = airy_slab(eps, d, kx, pol)
    assert r_ref + t_ref == pytest.approx(1.0, abs=1e-13)  # the reference itself conserves energy
    for period in PERIODS:
        r, t = _slab(eps, d, period, kx, pol)
        assert r == pytest.approx(r_ref, abs=1e-11), f"R at period {period}"
        assert t == pytest.approx(t_ref, abs=1e-11), f"T at period {period}"


@pytest.mark.parametrize("pol", ["s", "p"])
def test_absorbing_slab_matches_airy(pol):
    """A lossy film (eps = 4 + 0.5i): R and T match Airy and the absorbed part 1 - R - T is positive."""
    eps, d = (1.0, 4.0 + 0.5j, 2.25), 0.5
    kx = K0 * np.sin(np.radians(35.0))
    r_ref, t_ref = airy_slab(eps, d, kx, pol)
    assert 0.05 < 1 - r_ref - t_ref < 0.95
    for period in (0.4, 1.3):
        r, t = _slab(eps, d, period, kx, pol)
        assert r == pytest.approx(r_ref, abs=1e-11)
        assert t == pytest.approx(t_ref, abs=1e-11)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_frustrated_total_internal_reflection_matches_airy(pol):
    """Glass / air gap / glass beyond the critical angle: the wave in the gap is evanescent and tunnels. The
    transmitted fraction through a 0.2-wavelength gap is the Airy value (here 0.4-0.6), not 0 or 1."""
    eps, d = (2.25, 1.0, 2.25), 0.2
    kx = K0 * 1.5 * np.sin(np.radians(50.0))  # critical angle: 41.8 degrees
    r_ref, t_ref = airy_slab(eps, d, kx, pol)
    assert 0.05 < t_ref < 0.95 and r_ref + t_ref == pytest.approx(1.0, abs=1e-13)
    for period in (0.3, 1.3):
        r, t = _slab(eps, d, period, kx, pol)
        assert r == pytest.approx(r_ref, abs=1e-11)
        assert t == pytest.approx(t_ref, abs=1e-11)


@pytest.mark.parametrize("pol", ["s", "p"])
def test_total_internal_reflection_reflects_everything(pol):
    """Glass onto an air half-space beyond the critical angle: R = 1, T = 0."""
    eps, d = (2.25, 2.25, 1.0), 0.3
    kx = K0 * 1.5 * np.sin(np.radians(60.0))
    r, t = _slab(eps, d, 0.5, kx, pol)
    assert r == pytest.approx(1.0, abs=1e-11)
    assert t == pytest.approx(0.0, abs=1e-11)
