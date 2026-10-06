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

The two were INVERTED -- jNO's sensitivity was 4x stronger where it must vanish. A conductor thick
against the skin depth now carries a current sheet per face, and these tests pin the physical
behaviour: thickness is invisible once the current is skin-confined, and matters when it is not.
"""

import jax
import numpy as np

import jno

jax.config.update("jax_enable_x64", True)

CU, MU0 = 5.8e7, 4e-7 * np.pi
LEN, RAD, ZW, ZTOP = 0.040, 3.0e-4, 2.25e-3, 1.63e-3


def _skin(freq):
    return 1.0 / np.sqrt(np.pi * freq * MU0 * CU)


def _microstrip(thick, freq, pitch=5.0e-4):
    """Wire out, via down, return through a plane whose TOP face is fixed at ``ZTOP``."""
    wire = (
        jno.shape.line([(0.0, 0.02, ZW), (LEN, 0.02, ZW), (LEN, 0.02, ZTOP)], r=RAD, size=5.0e-4).attach(sigma=CU).name("w")
    )
    plane = (
        jno.shape.box(-0.004, 0.014, ZTOP - thick, LEN + 0.004, 0.026, ZTOP, size=(pitch, pitch, thick))
        .attach(sigma=CU)
        .name("plane")
    )
    d = (wire + plane).domain()
    d.tag("A", lambda x, y, z: (x < 1e-9) & (z > ZTOP + 1e-9))
    d.tag("B", lambda x, y, z: (np.abs(x) < pitch * 0.6) & (np.abs(y - 0.02) < pitch * 0.6) & (z < ZTOP + 1e-9))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    s = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build().solve()
    # pypeec's inductance is Im(Z) / omega (utils/matrix.py), which includes the INTERNAL inductance
    # of the skin layer; `s.L` is the external, field-energy one, so it is not the quantity compared
    z = complex(np.asarray(s.Z))
    return z.real, z.imag / (2.0 * np.pi * freq)


def test_a_return_planes_thickness_is_invisible_once_the_current_is_skin_confined():
    """Copper 8 skin depths below the conducting face carries nothing, so it cannot change L.

    pypeec measures -0.05 % over this range; the one-current model measured +21.25 %.
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


def test_the_loop_inductance_does_not_jump_where_a_sheet_pair_would_start():
    """The guard that was missing when the sheet-pair model was written.

    Emitting a current sheet per face is triggered by the conductor being thick against the SKIN
    DEPTH, so the discretisation changes at a particular frequency. A discretisation change must not
    move the answer: an inductance is continuous in frequency, and nothing physical happens to a
    0.5 mm trace between 50 and 80 kHz.

    The first sheet model failed exactly here while passing the plane-thickness test above. Measured
    on a real power module:

        50 kHz, unpaired   60.5 nH        80 kHz, paired   20.1 nH

    a 3x collapse. The cause was not the physics: the sheets duplicate and reorder the bars, and the
    incidence was built in the original order, so each sheet was wired to another family's nodes.
    """
    sig, thick = 5.8e7, 1.0e-3
    # the pairing threshold is thickness = delta, which for 1 mm of copper is about 4.4 kHz
    lo, hi = 2e3, 1e4
    d_lo = 1.0 / np.sqrt(np.pi * lo * MU0 * sig)
    d_hi = 1.0 / np.sqrt(np.pi * hi * MU0 * sig)
    assert thick < d_lo and thick > d_hi  # the threshold really is crossed

    def bar(freq):
        sh = jno.shape.box(0, 0, 0, 0.040, 0.006, thick, size=(0.004, 0.006, thick)).attach(sigma=sig).name("b")
        w = (
            jno.shape.line(
                [(0.004, 0.003, thick), (0.004, 0.003, 0.004), (0.036, 0.003, 0.004), (0.036, 0.003, thick)],
                r=2e-4,
                size=0.004,
            )
            .attach(sigma=sig)
            .name("w")
        )
        d = (sh + w).domain()
        d.tag("A", lambda x, y, z: (x < 0.0041) & (z < thick + 1e-9))
        d.tag("B", lambda x, y, z: (x > 0.0359) & (z < thick + 1e-9))
        i, v = d.peec_symbols()
        at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
        s = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build().solve()
        return float(np.real(s.L))

    a, b = bar(lo), bar(hi)
    assert abs(b / a - 1) < 0.25, f"L jumped {a * 1e9:.2f} -> {b * 1e9:.2f} nH across the pairing threshold"


def test_a_paired_bar_carries_the_same_current_as_an_unpaired_one():
    """An isolated bar has no proximity to redistribute its current, so pairing must not move R or L.

    The incidence regression: with the sheets wired to the wrong nodes this bar read R 5401 against
    1509 uOhm and L 15.5 against 25.9 nH at 1 MHz.
    """
    length, width, thick = 0.040, 0.004, 0.51e-3

    def solve(freq):
        bar = jno.shape.box(0, 0, 0, length, width, thick, size=(1e-3, 1e-3, thick)).attach(sigma=CU).name("bar")
        d = bar.domain()
        d.tag("A", lambda x, y, z: x < 1.1e-3)
        d.tag("B", lambda x, y, z: x > length - 1.1e-3)
        i, v = d.peec_symbols()
        at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
        b = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).build()
        s = b.solve()
        return int(np.asarray(b.fil.length).size), float(np.real(s.R)), float(np.real(s.L))

    # either side of the pairing threshold, thick = delta at about 16.8 kHz
    n0, r0, l0 = solve(1.6e4)
    n1, r1, l1 = solve(1.75e4)
    assert n1 == 2 * n0  # the second really is paired
    assert abs(r1 / r0 - 1) < 0.01
    assert abs(l1 / l0 - 1) < 0.01
    # and deep in the skin regime it stays the conductor it was
    _n, r_hi, l_hi = solve(1e6)
    assert 1.4e-3 < r_hi < 1.6e-3  # the one-current surface-impedance model gives 1.509 mOhm
    assert abs(l_hi / l0 - 1) < 0.03


def test_a_strip_over_a_plane_matches_the_closed_form_microstrip():
    """The loop inductance of a trace over its return plane, where the current is on the FACING faces.

    One current per conductor puts it at mid-thickness, which widens the loop; measured here +52 %
    over the closed form (Hammerstad-Jensen with Wheeler's thickness correction, accurate to about
    1 % in this range). A sheet per face, drawn ON the face, gives +7.0 % at a 1 mm pitch and
    +5.9 % at 0.5 mm. The per-length inductance is the difference of two lengths, so the ends and
    the short cancel.
    """
    mm = 1e-3
    pitch, gap, thick, width, plane_w = 1.0 * mm, 0.37 * mm, 0.37 * mm, 4.0 * mm, 20.0 * mm

    def loop(length):
        size = (pitch, pitch, thick)
        y0 = -width / 2
        geo = jno.shape.box(0, y0, gap, length, y0 + width, gap + thick, size=size).attach(sigma=CU).name("s")
        geo = geo + jno.shape.box(0, -plane_w / 2, -thick, length, plane_w / 2, 0, size=size).attach(sigma=CU).name("p")
        # the far-end short as round vias, so every bar of the strip and plane stays one cell thick
        for k, yv in enumerate(np.arange(y0 + pitch / 2, y0 + width, pitch)):
            pts = [(length - pitch / 2, yv, gap + thick / 2), (length - pitch / 2, yv, -thick / 2)]
            geo = geo + jno.shape.line(pts, r=0.2 * pitch, size=pitch).attach(sigma=CU).name(f"v{k}")
        d = geo.domain()
        d.tag("A", lambda x, y, z: (x < pitch * 1.01) & (z > gap - 1e-9) & (np.abs(y) < width / 2))
        d.tag("B", lambda x, y, z: (x < pitch * 1.01) & (z < 1e-9) & (np.abs(y) < width / 2))
        i, v = d.peec_symbols()
        at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
        return float(np.real(jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).build().solve().L))

    per_m = (loop(40 * mm) - loop(20 * mm)) / (20 * mm)
    we = width + thick / np.pi * (1 + np.log(2 * gap / thick))
    ue = we / gap
    ref = 120 * np.pi / (ue + 1.393 + 0.667 * np.log(ue + 1.444)) / 299_792_458.0
    assert 1.0 < per_m / ref < 1.10, f"{per_m * 1e9:.2f} nH/m against the closed form {ref * 1e9:.2f}"
