"""A conductor's INDUCTANCE must see the same current its impedance does.

A one-cell conductor takes a shape-aware surface impedance, so R knows the current is confined to a
skin layer at the faces. Its partial inductance did not: the current was spread through the whole
cell, so a return plane's THICKNESS changed L at a frequency where the copper below the skin layer
is electromagnetically invisible.

Validated against pypeec 5.8.0, which is voxel-based and resolves the skin depth with volume cells,
so R and L see the same distribution by construction. On a rectangular microstrip with the plane's
top face, the trace, the gap and the port all fixed, taking the plane from 0.4 to 1.6 mm:

    skin-confined (100 kHz, 1.9 -> 7.7 delta)    pypeec  -0.05 %     jNO  +21.25 %
    uniform current (1 kHz, 0.2 -> 0.8 delta)    pypeec +20.90 %     jNO   +5.05 %

The two are INVERTED -- jNO's sensitivity is 4x stronger where it must vanish. These tests pin the
physical behaviour: thickness is invisible once the current is skin-confined, and matters when it is
not.
"""

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

CU, MU0 = 5.8e7, 4e-7 * np.pi
LEN, RAD, ZW, ZTOP = 0.040, 3.0e-4, 2.25e-3, 1.63e-3


def _skin(freq):
    return 1.0 / np.sqrt(np.pi * freq * MU0 * CU)


def _microstrip(thick, freq, pitch=5.0e-4):
    """Wire out, via down, return through a plane whose TOP face is fixed at ``ZTOP``."""
    wire = (
        jno.Shape.line([(0.0, 0.02, ZW), (LEN, 0.02, ZW), (LEN, 0.02, ZTOP)], r=RAD, size=5.0e-4)
        .attach(sigma=CU)
        .name("w")
    )
    plane = (
        jno.Shape.box(-0.004, 0.014, ZTOP - thick, LEN + 0.004, 0.026, ZTOP, size=(pitch, pitch, thick))
        .attach(sigma=CU)
        .name("plane")
    )
    d = (wire + plane).domain()
    d.tag("A", lambda x, y, z: (x < 1e-9) & (z > ZTOP + 1e-9))
    d.tag("B", lambda x, y, z: (np.abs(x) < pitch * 0.6) & (np.abs(y - 0.02) < pitch * 0.6) & (z < ZTOP + 1e-9))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    s = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build().solve()
    return float(np.real(s.R)), float(np.real(s.L))


def test_a_return_planes_thickness_is_invisible_once_the_current_is_skin_confined():
    """Copper 8 skin depths below the conducting face carries nothing, so it cannot change L.

    pypeec measures -0.05 % over this range; jNO measured +21.25 %.
    """
    freq = 1e5
    delta = _skin(freq)
    thin, thick = 0.4e-3, 1.6e-3
    assert thin / delta > 1.5 and thick / delta > 5.0  # both genuinely skin-confined

    r_thin, l_thin = _microstrip(thin, freq)
    r_thick, l_thick = _microstrip(thick, freq)

    assert abs(r_thick / r_thin - 1) < 0.10  # the surface impedance already gets this right
    assert abs(l_thick / l_thin - 1) < 0.03  # and the inductance must agree with it


def test_thickness_still_matters_where_the_current_really_is_uniform():
    """The other half of the claim: this must not become 'thickness never matters'.

    Below the skin depth the current fills the section, the centroid genuinely moves down with a
    thicker plane, and L genuinely rises -- pypeec measures +20.9 % over the same range.
    """
    freq = 1e3
    delta = _skin(freq)
    thin, thick = 0.4e-3, 1.6e-3
    assert thick / delta < 1.0  # no skin confinement anywhere in this range

    _r_thin, l_thin = _microstrip(thin, freq)
    _r_thick, l_thick = _microstrip(thick, freq)
    assert l_thick / l_thin - 1 > 0.05  # rises, as a uniformly-filled section must
