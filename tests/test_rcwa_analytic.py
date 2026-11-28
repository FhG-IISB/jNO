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


def _film_airy(eps_top, eps_bottom, kz_film, q_film, d, kx, pol):
    """Airy R, T of a film whose wavenumber ``kz_film`` and admittance ``q_film`` are given (an anisotropic film),
    between isotropic half-spaces, for the out-of-plane field of ``pol``."""
    kz = [np.sqrt(complex(e) * K0**2 - kx**2 + 0j) for e in (eps_top, eps_bottom)]
    kz = [k if k.imag >= 0 else -k for k in kz]
    q0, q2 = [k if pol == "s" else k / complex(e) for k, e in zip(kz, (eps_top, eps_bottom))]
    q1 = q_film
    r01, r12 = (q0 - q1) / (q0 + q1), (q1 - q2) / (q1 + q2)
    t01, t12 = 2 * q0 / (q0 + q1), 2 * q1 / (q1 + q2)
    ph = np.exp(1j * kz_film * d)
    den = 1 + r01 * r12 * ph**2
    r, t = (r01 + r12 * ph**2) / den, t01 * t12 * ph / den
    return abs(r) ** 2, abs(t) ** 2 * q2.real / q0.real


def _tensor_slab(tensor, d, kx, pol, eps_bottom=1.2**2, period=0.45, orders=9):
    rc = Rcwa(
        [(np.inf, 1.0), (d, tensor), (np.inf, eps_bottom)],
        period=(period, period),
        orders=orders,
        wavelength=WL,
        k_in=(kx, 0.0),
        assume_periodic=True,
        polarization={"s": "y", "p": "x"}[pol],
    )
    sol = rc.solve()
    return float(sol.efficiency("R")), float(sol.efficiency("T"))


@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("theta_deg", [0.0, 25.0, 60.0])
def test_biaxial_slab_at_oblique_incidence_matches_airy(theta_deg, pol):
    """A biaxial film ``diag(eps_xx, eps_yy, eps_zz)`` in the x-z plane of incidence: the s wave (E_y) sees only
    ``eps_yy``, ``kz^2 = eps_yy k0^2 - kx^2``; the p wave (H_y) sees ``kz^2 = eps_xx (k0^2 - kx^2 / eps_zz)`` with
    the admittance ``kz / eps_xx`` (Berreman's 4 x 4 problem decouples for a diagonal tensor)."""
    exx, eyy, ezz = 2.6, 3.4, 1.9 + 0.05j
    kx = K0 * np.sin(np.deg2rad(theta_deg))
    if pol == "s":
        kz = np.sqrt(eyy * K0**2 - kx**2 + 0j)
        q = kz
    else:
        kz = np.sqrt(exx * (K0**2 - kx**2 / ezz) + 0j)
        q = kz / exx
    r_ref, t_ref = _film_airy(1.0, 1.2**2, kz, q, 0.41, kx, pol)
    r, t = _tensor_slab((exx, 0.0, 0.0, eyy, ezz), 0.41, kx, pol)
    assert r == pytest.approx(r_ref, abs=1e-12)
    assert t == pytest.approx(t_ref, abs=1e-12)


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 72.0])
def test_rotated_in_plane_tensor_splits_into_its_eigenpolarizations(angle_deg):
    """At normal incidence a film whose in-plane tensor has eigenvalues ``a``, ``b`` along axes rotated by
    ``angle`` carries the two eigenpolarizations independently: for x-polarized light
    ``T = cos^2 T_a + sin^2 T_b`` and the same for R (the two outputs are orthogonal, so their powers add)."""
    a, b, c = 2.2, 4.1, 3.0
    th = np.deg2rad(angle_deg)
    cs, sn = np.cos(th), np.sin(th)
    tensor = (a * cs**2 + b * sn**2, (a - b) * cs * sn, (a - b) * cs * sn, a * sn**2 + b * cs**2, c)
    ra, ta = airy_slab((1.0, a, 1.2**2), 0.33, 0.0, "s")
    rb, tb = airy_slab((1.0, b, 1.2**2), 0.33, 0.0, "s")
    r, t = _tensor_slab(tensor, 0.33, 0.0, "p")
    assert t == pytest.approx(cs**2 * ta + sn**2 * tb, abs=1e-12)
    assert r == pytest.approx(cs**2 * ra + sn**2 * rb, abs=1e-12)
