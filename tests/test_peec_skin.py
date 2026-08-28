"""The element impedance is a shape-aware SURFACE one, not rho*l/A.

That is what lets a conductor be ONE element through its thickness and still show the skin effect,
which is how the PEEC literature keeps a package tractable (Romano, Kovacevic-Badstuebner, Antonini
& Grossner, IEEE Trans. Electromagn. Compat. 65(2), 2023, sec. II-A). It also strengthens the
preconditioner, whose diagonal is Z_s + j w Lp_aa.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import internal_impedance
from jno.utils.solver.peec import bar_filaments, line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG, MU0 = 5.8e7, 4e-7 * np.pi


def test_it_is_the_dc_resistance_below_the_skin_depth():
    """A conductor thin against the skin depth conducts through its whole section, as it must."""
    a, ell = 1e-3, 1.0
    dc = ell / (SIG * np.pi * a**2)
    for hz in (0.0, 1.0, 100.0):
        z = complex(internal_impedance(ell, np.pi * a**2, a, True, 2 * np.pi * hz, SIG))
        assert abs(z.real / dc - 1) < 2e-3


def test_a_round_wire_approaches_its_thin_skin_asymptote():
    """Far above the skin depth the current runs in a shell of depth delta: R -> rho / (2 pi a delta)."""
    a, ell = 1e-3, 1.0
    hz = 1e7
    delta = np.sqrt(1.0 / (np.pi * hz * MU0 * SIG))
    asympt = ell / (SIG * 2 * np.pi * a * delta)
    z = complex(internal_impedance(ell, np.pi * a**2, a, True, 2 * np.pi * hz, SIG))
    assert abs(z.real / asympt - 1) < 0.05  # the asymptote drops the curvature correction
    assert z.imag > 0  # and it carries an internal inductance


def test_one_cell_through_the_thickness_still_shows_the_skin_effect():
    """The point of the surface impedance: no splitting across the section, and sqrt(f) still appears."""
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=(0.002, 0.004, 0.002))
    assert f.lattice["n"][2] == 1  # ONE cell through the 2 mm thickness
    p = np.asarray(f.nodes)
    a = terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)

    rs = []
    for hz in (0.0, 1e5, 1e6, 1e7):
        _c, _phi, inj = solve_network(
            f, SIG, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * hz, matrix_free=False
        )
        rs.append(complex(1.0 / inj["A"]).real)

    assert rs[0] > 0 and all(x > y for x, y in zip(rs[1:], rs[:-1]))  # rises with frequency
    assert rs[3] / rs[0] > 20  # and by a lot: measured 47.9x at 10 MHz
    # deep in the skin regime R grows as sqrt(f), so a decade multiplies it by about sqrt(10)
    assert 2.7 < rs[3] / rs[2] < 3.6


def test_a_wire_and_a_bar_take_different_coefficients():
    """A round section is not a slab: the cylindrical form is 0.02 % where a flat one is 12.5 % out."""
    area, ell, hz = 1e-6, 0.01, 1e7
    z_round = complex(internal_impedance(ell, area, np.sqrt(area / np.pi), True, 2 * np.pi * hz, SIG))
    z_slab = complex(internal_impedance(ell, area, np.sqrt(area), False, 2 * np.pi * hz, SIG))
    assert abs(z_round.real / z_slab.real - 1) > 0.05  # genuinely different, not a shared formula


def test_the_impedance_is_differentiable_through_the_surface_form():
    """The sqrt in gamma is zero at DC, and differentiating a masked branch through it gives NaN."""
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, 0.03)], r=3e-4, size=0.005))
    for hz in (0.0, 1e6):
        loss = lambda s, hz=hz: jax.numpy.real(
            jax.numpy.sum(internal_impedance(f.length, f.area, f.skin, f.round_, 2 * np.pi * hz, s))
        )
        g = float(jax.grad(loss)(SIG))
        fd = float((loss(SIG * 1.001) - loss(SIG * 0.999)) / (0.002 * SIG))
        assert np.isfinite(g)
        assert abs(g / fd - 1) < 1e-5
        assert g < 0  # more conductive, less resistive
