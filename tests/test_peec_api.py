"""``jno.peec`` end to end: geometry and ports in, impedance out, nothing meshed.

The oracles are the layer below — `network_impedance` and `pair_matrix`, both validated against
closed forms in their own tests — so these cases pin the FRONT DOOR: that the constraint list is read
correctly, that conductivity and filament length are taken from the geometry, and that the readouts
mean what they say.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import pair_matrix
from jno.utils.solver.peec import line_filaments, network_impedance

jax.config.update("jax_enable_x64", True)

MU0 = 4e-7 * np.pi
SIG = 5.8e7
ELL, RAD = 0.050, 5e-4


def one_wire():
    wire = jno.Shape.line([(0, 0, 0), (0, 0, ELL)], r=RAD, size=ELL / 10).attach(sigma=SIG).name("wire")
    pads = jno.Shape.sphere(0, 0, 0.0, 2 * RAD).name("A") + jno.Shape.sphere(0, 0, ELL, 2 * RAD).name("B")
    return wire, (wire + pads).domain()


def ports(d):
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    return i, v, at


@pytest.mark.parametrize("freq", [0.0, 1e6])
def test_the_port_impedance_matches_the_layer_below(freq):
    wire, d = one_wire()
    _i, v, at = ports(d)
    z = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).solve().Z
    ref, _ = network_impedance(line_filaments(wire), SIG, ((0, 0, 0), (0, 0, ELL)), omega=2 * np.pi * freq)
    assert abs(complex(z) - complex(ref)) / abs(complex(ref)) < 1e-12


def test_nothing_is_meshed():
    """The whole point of a partial-element method."""
    _wire, d = one_wire()
    _i, v, at = ports(d)
    jno.peec([v(*at("A")) - v(*at("B")) - 1.0]).solve()
    assert d.__dict__.get("_mesh") is None


def test_inductance_is_the_magnetic_energy_and_is_positive():
    wire, d = one_wire()
    _i, v, at = ports(d)
    f = line_filaments(wire)
    ref = float(np.asarray(pair_matrix(f.pos, f.mom, lambda r: 1 / r, f.self_g, group=f.group)).sum()) * MU0 / (4 * np.pi)
    for freq in (0.0, 1e6):
        sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq).solve()
        assert float(sol.L) > 0  # the Hermitian form; the transpose one goes negative once phases spread
        assert abs(float(sol.L) / ref - 1) < 1e-12  # one series path, so no redistribution with frequency


def test_joule_is_the_ohmic_loss_of_the_solved_currents():
    wire, d = one_wire()
    _i, v, at = ports(d)
    sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0]).solve()
    f = line_filaments(wire)
    r = np.asarray(f.length) / (SIG * np.pi * np.asarray(f.radius) ** 2)
    assert abs(float(sol.joule) / float((r * np.abs(np.asarray(sol.i)) ** 2).sum()) - 1) < 1e-12
    # and at DC one series path dissipates exactly V*I: the port sees the same watts the filaments do
    assert abs(float(sol.joule) / abs(complex(sol.current("A"))) - 1) < 1e-9
    assert abs(complex(sol.current("A")) + complex(sol.current("B"))) < 1e-9  # what goes in comes out


def test_a_frequency_array_sweeps():
    _wire, d = one_wire()
    _i, v, at = ports(d)
    fs = np.array([0.0, 1e5, 1e7])
    sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=fs).solve()
    assert np.asarray(sol.Z).shape == (3,)
    assert np.asarray(sol.L).shape == (3,)
    assert np.allclose(np.asarray(sol.R), np.asarray(sol.R)[0])  # one path: R is frequency-flat
    assert np.asarray(sol.Z).imag[2] > np.asarray(sol.Z).imag[1] > 0  # wL grows with f


def test_current_leaves_the_low_resistance_path_at_high_frequency():
    """The reason to run PEEC: with two paths the DC answer is the wrong answer at switching speeds."""
    direct = jno.Shape.line([(0, 0, 0), (0.05, 0, 0)], r=1.5e-4, size=0.004).attach(sigma=SIG).name("direct")
    detour = (
        jno.Shape.line([(0, 0, 0), (0, -0.04, 0), (0.05, -0.04, 0), (0.05, 0, 0)], r=6e-4, size=0.004)
        .attach(sigma=SIG)
        .name("detour")
    )
    # pads FIRST: a terminal sits ON a conductor, and regions resolve by declaration order, so a pad
    # declared after the conductor it marks is subtracted away to nothing.
    pads = jno.Shape.sphere(0, 0, 0, 3e-4).name("P") + jno.Shape.sphere(0.05, 0, 0, 3e-4).name("N")
    d = (pads + direct + detour).domain()
    _i, v, at = ports(d)
    sol = jno.peec([v(*at("P")) - v(*at("N")) - 1.0], freq=np.array([0.0, 1e8])).solve()
    ldc, lhf = float(np.asarray(sol.L)[0]), float(np.asarray(sol.L)[1])
    rdc, rhf = float(np.asarray(sol.R)[0]), float(np.asarray(sol.R)[1])
    # measured on this geometry: L 78.13 -> 39.14 nH (2.00x down), R 1.705 -> 5.445 mOhm (3.19x up),
    # saturating by 1 MHz. The bounds sit inside those with room, so they pin the effect, not the digits.
    assert ldc > 1.9 * lhf > 0  # current moves to the lower-INDUCTANCE path, so the loop inductance falls
    assert rhf > 3.0 * rdc  # and that path is the thin, resistive one


def test_a_solid_conductor_says_so_rather_than_guessing():
    solid = jno.Shape.box(0, 0, 0, 0.01, 0.01, 0.05, size=0.005).attach(sigma=SIG).name("bar")
    pads = jno.Shape.sphere(0, 0, 0, 1e-3).name("A") + jno.Shape.sphere(0, 0, 0.05, 1e-3).name("B")
    d = (solid + pads).domain()
    _i, v, at = ports(d)
    with pytest.raises(NotImplementedError, match="only line conductors"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0]).solve()


def test_a_conductor_without_a_conductivity_is_named():
    wire = jno.Shape.line([(0, 0, 0), (0, 0, ELL)], r=RAD, size=ELL / 5).name("wire")  # no .attach
    d = (wire + jno.Shape.sphere(0, 0, 0.0, 2 * RAD).name("A") + jno.Shape.sphere(0, 0, ELL, 2 * RAD).name("B")).domain()
    _i, v, at = ports(d)
    with pytest.raises(ValueError, match=r"conductor 'wire' has no conductivity"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0]).solve()


def test_the_readouts_describe_one_port():
    _wire, d = one_wire()
    _i, v, at = ports(d)
    with pytest.raises(ValueError, match="the impedance readouts describe ONE port"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0, v(*at("B")) - v(*at("A")) - 2.0])


def test_an_empty_constraint_list_is_refused():
    with pytest.raises(ValueError, match="carries no current"):
        jno.peec([])


def _two_paths():
    direct = jno.Shape.line([(0, 0, 0), (0.05, 0, 0)], r=1.5e-4, size=0.004).attach(sigma=SIG).name("direct")
    detour = (
        jno.Shape.line([(0, 0, 0), (0, -0.04, 0), (0.05, -0.04, 0), (0.05, 0, 0)], r=6e-4, size=0.004)
        .attach(sigma=SIG)
        .name("detour")
    )
    return direct, detour


def test_a_terminal_can_be_a_tag_and_then_declaration_order_stops_mattering():
    """A terminal is a named SUBSET of a conductor, not a material, so it belongs in a tag.

    Regions resolve by declaration order because a cell belongs to ONE material. A pad does not: it
    sits wholly inside the conductor it marks. Written as a region and declared second, the priority
    rule subtracts it to nothing — this same geometry raised "the region the tag names appears to be
    empty". As a tag there is no priority and no ordering rule to remember.
    """
    direct, detour = _two_paths()
    d = (direct + detour).domain()  # conductors FIRST — the order that broke the region spelling
    d.tag("P", lambda x, y, z: (x**2 + y**2 + z**2) < 3e-4**2)
    d.tag("N", lambda x, y, z: ((x - 0.05) ** 2 + y**2 + z**2) < 3e-4**2)
    _i, v, at = ports(d)
    sol = jno.peec([v(*at("P")) - v(*at("N")) - 1.0], freq=np.array([0.0, 1e8])).solve()
    assert float(np.asarray(sol.L)[0]) > 1.9 * float(np.asarray(sol.L)[1]) > 0
    assert d.__dict__.get("_mesh") is None


def test_the_two_terminal_spellings_agree():
    """A pad written as a region (declared first) and as a tag must give the same circuit."""
    direct, detour = _two_paths()
    pads = jno.Shape.sphere(0, 0, 0, 3e-4).name("P") + jno.Shape.sphere(0.05, 0, 0, 3e-4).name("N")

    as_region = (pads + direct + detour).domain()
    _i, v, at = ports(as_region)
    z_region = jno.peec([v(*at("P")) - v(*at("N")) - 1.0], freq=1e6).solve().Z

    as_tag = (direct + detour).domain()
    as_tag.tag("P", lambda x, y, z: (x**2 + y**2 + z**2) < 3e-4**2)
    as_tag.tag("N", lambda x, y, z: ((x - 0.05) ** 2 + y**2 + z**2) < 3e-4**2)
    _i, v, at = ports(as_tag)
    z_tag = jno.peec([v(*at("P")) - v(*at("N")) - 1.0], freq=1e6).solve().Z

    assert abs(complex(z_region) - complex(z_tag)) / abs(complex(z_tag)) < 1e-12


def test_a_terminal_that_names_no_region_or_tag_says_how_to_declare_one():
    """`side` is an auto boundary name: it binds, but it marks no part of any conductor."""
    _wire, d = one_wire()
    _i, v, at = ports(d)
    with pytest.raises(ValueError, match="neither a region nor a tag"):
        jno.peec([v(*at("A")) - v(*at("side")) - 1.0]).solve()
