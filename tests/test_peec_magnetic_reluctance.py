"""The magnetic element's constitutive law: chi, not mu_r.

A magnetic material enters PEEC as a MAGNETISATION, so what circulates in the magnetic mesh is the
flux the material adds rather than the whole of it. The constitutive quantity is the susceptibility
`chi = mu_r - 1`, and the element reluctance is `l / (mu0 chi A)` -- pypeec's
`rho = 1 / (cst.mu_0 * chi)`, from Torchio et al., IEEE TMTT 66(5), 2018.

Getting this wrong -- using mu_r where chi belongs -- would leave a mu_r = 1 region behaving like a
weak core instead of like the air it is, which no absolute test of a strong core would catch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.kernel import magnetic_reluctance

jax.config.update("jax_enable_x64", True)

MU0 = 4e-7 * np.pi
ELL, AREA = 0.020, 48e-6  # a limb 20 mm long, 6 x 8 mm in section


def test_it_is_the_reluctance_of_the_susceptibility():
    """R_m = l / (mu0 chi A), with chi = mu_r - 1."""
    for mu_r in (2.0, 100.0, 2400.0):
        want = ELL / (MU0 * (mu_r - 1.0) * AREA)
        assert abs(complex(magnetic_reluctance(ELL, AREA, mu_r)).real / want - 1) < 1e-12


def test_chi_is_what_diverges_as_the_material_becomes_air():
    """Reluctance -> infinity as mu_r -> 1, which is how a core stops being one.

    If mu_r stood where chi does, this would tend to `l / (mu0 A)` -- a perfectly ordinary finite
    reluctance -- and air would behave like a core.
    """
    r = [complex(magnetic_reluctance(ELL, AREA, 1.0 + e)).real for e in (1e-2, 1e-4, 1e-6)]
    assert r[0] < r[1] < r[2]
    assert r[2] / r[1] > 50  # genuinely diverging, not merely increasing
    wrong = ELL / (MU0 * 1.0 * AREA)  # what a mu_r-instead-of-chi slip would give at mu_r = 1
    assert r[2] > 100 * wrong


def test_a_high_permeability_core_carries_flux_easily():
    """The dual of a good conductor: large chi, small reluctance, inversely."""
    a = complex(magnetic_reluctance(ELL, AREA, 1000.0)).real
    b = complex(magnetic_reluctance(ELL, AREA, 2000.0)).real
    assert abs((a / b) / (1999.0 / 999.0) - 1) < 1e-12


def test_a_complex_permeability_is_a_lossy_core():
    """Core loss enters as the imaginary part of chi, exactly as a complex permittivity carries
    dielectric loss. A real mu_r must stay lossless."""
    assert abs(complex(magnetic_reluctance(ELL, AREA, 2400.0)).imag) < 1e-30
    lossy = complex(magnetic_reluctance(ELL, AREA, 2400.0 - 120.0j))
    assert lossy.imag != 0.0


def test_geometry_enters_as_length_over_area():
    """Twice as long is twice the reluctance; twice the section is half."""
    base = complex(magnetic_reluctance(ELL, AREA, 500.0)).real
    assert abs(complex(magnetic_reluctance(2 * ELL, AREA, 500.0)).real / base - 2.0) < 1e-12
    assert abs(complex(magnetic_reluctance(ELL, 2 * AREA, 500.0)).real / base - 0.5) < 1e-12


def test_the_permeability_is_differentiable():
    """A core is only a design variable if a gradient reaches its material."""
    f = lambda m: jnp.real(magnetic_reluctance(ELL, AREA, m))
    g = float(jax.grad(f)(500.0))
    fd = float((f(500.5) - f(499.5)) / 1.0)
    # R_m goes as 1/chi, so a central difference at h = 0.5 carries a truncation error of about
    # h^2 / chi^2 ~ 1e-6 of its own -- the bound is on the difference, not on the gradient.
    assert np.isfinite(g) and abs(g / fd - 1) < 1e-4
