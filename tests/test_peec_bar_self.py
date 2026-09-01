"""The self partial inductance of a rectangular bar, at ANY aspect ratio.

``bar_self`` used Ruehli's asymptotic form, whose own docstring says ``valid for l >> w, t``. A PEEC
lattice violates that routinely: one cell through a 0.5 mm conductor with a 0.06 mm lateral pitch is
an element eight times shorter than it is thick, and there ``ln(2l/(w+t))`` goes NEGATIVE.

It failed the way that is hardest to catch -- the answer stayed plausible and moved the wrong way
under refinement. On the copper bar of Romano et al. (IEEE TEMC 65(2), 2023, sec. V-A), where Q3D,
volume PEEC, surface PEEC and full-wave all agree at 2.85 nH, refining the lattice laterally with the
thickness held at one cell gave 3.01 -> 3.23 -> 3.41 nH: diverging, not converging.

The replacement evaluates the defining double-volume integral instead:

    Lp = (mu0 / 4 pi A^2) int_V int_V dV dV' / |r - r'|,   A = w t

which reduces, by the difference variable, to ``8 W(w, t, l)`` with
``W = int (w-x)(t-y)(l-z) / r``. A Duffy split by largest coordinate removes the 1/r singularity
exactly -- the s^2 Jacobian cancels it -- leaving a smooth integrand that Gauss-Legendre nails.

The oracles here are independent of that machinery: Ruehli's asymptotic form in the regime where it
IS valid, and the published benchmark.
"""

import numpy as np
import pytest

from jno.utils.solver.kernel import bar_self

MU0 = 4e-7 * np.pi


def ruehli_asymptotic(length, w, t):
    """The old formula. Correct only for l >> w, t -- which is exactly where it is used as an oracle."""
    s = w + t
    return 2.0 * (np.log(2.0 * length / s) + 0.5 + 0.2235 * s / length) / length


def L_of(g, length):
    return (MU0 / (4 * np.pi)) * length**2 * g


@pytest.mark.parametrize("ratio", [50.0, 200.0, 1000.0])
def test_it_matches_the_asymptotic_form_where_that_form_is_valid(ratio):
    """l >> w, t is where Ruehli's expression is right, so the two must agree there."""
    w = t = 1e-3
    length = ratio * w
    got = float(bar_self(np.array([length]), np.array([w]), np.array([t]))[0])
    want = ruehli_asymptotic(length, w, t)
    assert abs(got / want - 1) < 5e-3, f"l/w={ratio}: {got} vs {want}"


def test_the_published_copper_bar():
    """Romano et al. sec. V-A: 0.5 x 0.5 x 5 mm copper. Q3D and three PEEC variants give ~2.85 nH."""
    w = t = 0.5e-3
    length = 5.0e-3
    L = L_of(float(bar_self(np.array([length]), np.array([w]), np.array([t]))[0]), length)
    assert abs(L * 1e9 - 2.85) < 0.05, f"{L * 1e9:.4f} nH"


def test_a_SHORT_fat_element_is_finite_and_positive():
    """The regime that broke: an element far shorter than its own cross-section.

    The old form returns a negative logarithm here and the resulting inductance is nonsense; the
    integral is simply the integral, and it must stay positive and finite.
    """
    g = float(bar_self(np.array([0.0625e-3]), np.array([0.0625e-3]), np.array([0.5e-3]))[0])
    assert np.isfinite(g) and g > 0
    L = L_of(g, 0.0625e-3)
    assert 0 < L < 1e-9  # a 60 um element cannot have nanohenries of self inductance


def test_it_is_symmetric_in_the_two_cross_section_axes():
    """w and t enter the same way; swapping them cannot change the answer."""
    a = float(bar_self(np.array([1e-3]), np.array([0.3e-3]), np.array([0.7e-3]))[0])
    b = float(bar_self(np.array([1e-3]), np.array([0.7e-3]), np.array([0.3e-3]))[0])
    assert abs(a / b - 1) < 1e-12


def test_it_scales_correctly_with_size():
    """Inductance has dimensions of length, so scaling every dimension by k scales L by k."""
    k = 3.7
    l0, w0, t0 = 2e-3, 0.4e-3, 0.9e-3
    L1 = L_of(float(bar_self(np.array([l0]), np.array([w0]), np.array([t0]))[0]), l0)
    L2 = L_of(float(bar_self(np.array([k * l0]), np.array([k * w0]), np.array([k * t0]))[0]), k * l0)
    assert abs(L2 / (k * L1) - 1) < 1e-9


def test_it_agrees_with_a_brute_force_integral():
    """An oracle that shares no code with the implementation: plain Monte Carlo on the definition.

    Crude, so the tolerance is loose -- but it is wrong in a completely different way, which is the
    point of having it.
    """
    rng = np.random.default_rng(0)
    w, t, length = 0.4e-3, 0.9e-3, 0.6e-3
    n = 2_000_000
    p = rng.random((n, 3)) * np.array([w, t, length])
    q = rng.random((n, 3)) * np.array([w, t, length])
    r = np.linalg.norm(p - q, axis=1)
    vol = w * t * length
    integral = vol**2 * np.mean(1.0 / r)          # <1/r> over V x V, times |V|^2
    L_mc = MU0 / (4 * np.pi) * integral / (w * t) ** 2
    L = L_of(float(bar_self(np.array([length]), np.array([w]), np.array([t]))[0]), length)
    assert abs(L / L_mc - 1) < 0.02, f"exact {L * 1e12:.4f} pH vs Monte Carlo {L_mc * 1e12:.4f} pH"
