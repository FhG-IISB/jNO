"""A two-terminal DEVICE: ``v(A) - v(B) - Z*i(A)``, which is Ohm's law and reads as one.

This is how a component enters the network without being meshed as metal. It matters because a
device parameter does not belong to the geometry: voxelise a MOSFET die and its on-resistance
follows the grid instead of the datasheet -- a 0.18 mm die on a 0.285 mm grid is 1.58x too
resistive before any current-crowding error on top.

The oracle throughout is the series circuit, which is exact: one current path means the port
impedance is the sum of what is in it.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import port_spec

jax.config.update("jax_enable_x64", True)

SIG, ELL, RAD = 5.8e7, 0.050, 5e-4
R_WIRE = ELL / (SIG * np.pi * RAD**2)


def two_wires(gap=0.004):
    """Two collinear wires with a gap: nothing conducts across it but a device."""
    lo = jno.Shape.line([(0, 0, 0), (0, 0, ELL)], r=RAD, size=ELL / 10).attach(sigma=SIG).name("lo")
    hi = jno.Shape.line([(0, 0, ELL + gap), (0, 0, 2 * ELL + gap)], r=RAD, size=ELL / 10).attach(sigma=SIG).name("hi")
    pads = (
        jno.Shape.sphere(0, 0, 0.0, 2 * RAD).name("A")
        + jno.Shape.sphere(0, 0, ELL, 2 * RAD).name("M")
        + jno.Shape.sphere(0, 0, ELL + gap, 2 * RAD).name("N")
        + jno.Shape.sphere(0, 0, 2 * ELL + gap, 2 * RAD).name("B")
    )
    d = (lo + hi + pads).domain()
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    return i, v, at


def test_a_device_carries_its_own_impedance_and_nothing_else():
    """Series: two wires and a resistor between them. R_port = 2 R_wire + R_dev, exactly."""
    i, v, at = two_wires()
    a, m, n, b = at("A"), at("M"), at("N"), at("B")
    rdev = 5e-3
    sol = jno.peec([v(*a) - v(*b) - 1.0, v(*m) - v(*n) - rdev * i(*m)], freq=0.0).solve()
    assert abs(float(sol.R) - (2 * R_WIRE + rdev)) / (2 * R_WIRE + rdev) < 1e-9


@pytest.mark.parametrize("rdev", [0.0, 1e-6, 1.0, 1e6])
def test_the_device_value_passes_straight_through(rdev):
    """Extremes: a dead short, a milliohm, an ohm, and a megohm all land on 2 R_wire + R_dev."""
    i, v, at = two_wires()
    a, m, n, b = at("A"), at("M"), at("N"), at("B")
    sol = jno.peec([v(*a) - v(*b) - 1.0, v(*m) - v(*n) - rdev * i(*m)], freq=0.0).solve()
    assert abs(float(sol.R) - (2 * R_WIRE + rdev)) / (2 * R_WIRE + rdev) < 1e-9


def test_a_device_may_be_complex_a_lumped_R_L():
    """`(R + 1j w L) * i(A)` is a lumped R-L, and its reactance adds to the port like a series one."""
    i, v, at = two_wires()
    a, m, n, b = at("A"), at("M"), at("N"), at("B")
    freq, rdev, ldev = 1e6, 5e-3, 3e-9
    z = rdev + 1j * 2 * np.pi * freq * ldev
    sol = jno.peec([v(*a) - v(*b) - 1.0, v(*m) - v(*n) - z * i(*m)], freq=freq).solve()
    bare = jno.peec([v(*a) - v(*b) - 1.0, v(*m) - v(*n) - rdev * i(*m)], freq=freq).solve()
    assert abs(complex(sol.Z) - (complex(bare.Z) + 1j * 2 * np.pi * freq * ldev)) < 1e-12


def test_a_device_carries_no_partial_inductance():
    """It is a circuit element, not geometry: it adds no field energy and couples to nothing."""
    i, v, at = two_wires()
    a, m, n, b = at("A"), at("M"), at("N"), at("B")
    lo = float(jno.peec([v(*a) - v(*b) - 1.0, v(*m) - v(*n) - 1e-6 * i(*m)], freq=1e6).solve().L)
    hi = float(jno.peec([v(*a) - v(*b) - 1.0, v(*m) - v(*n) - 1.0 * i(*m)], freq=1e6).solve().L)
    assert abs(hi / lo - 1) < 1e-6  # a millionfold change in R moves the loop inductance not at all


def test_no_device_means_no_current_at_all():
    """The gap is the point: without the device the two wires are separate metal, and it says so."""
    lo = jno.Shape.line([(0, 0, 0), (0, 0, ELL)], r=RAD, size=ELL / 10).attach(sigma=SIG).name("lo")
    hi = jno.Shape.line([(0, 0, ELL + 0.004), (0, 0, 2 * ELL + 0.004)], r=RAD, size=ELL / 10)
    hi = hi.attach(sigma=SIG).name("hi")
    pads = jno.Shape.sphere(0, 0, 0.0, 2 * RAD).name("A") + jno.Shape.sphere(0, 0, 2 * ELL + 0.004, 2 * RAD).name("B")
    d = (lo + hi + pads).domain()
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    with pytest.raises(ValueError, match="no conducting path"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).solve()


def test_the_current_must_be_taken_at_the_positive_terminal():
    """Writing it at the far end is a sign the reader would have to guess, so it is refused."""
    i, v, at = two_wires()
    m, n = at("M"), at("N")
    with pytest.raises(ValueError, match="takes its current at 'N'"):
        port_spec([v(*m) - v(*n) - 5e-3 * i(*n)])


def test_a_third_terminal_is_named_as_a_controlled_source():
    i, v, at = two_wires()
    a, m, n = at("A"), at("M"), at("N")
    with pytest.raises(ValueError, match="controlled source"):
        port_spec([v(*m) - v(*n) - 5e-3 * i(*a)])


def test_a_varying_impedance_is_a_conductor():
    """Only a constant times the current is a device; anything else has geometry and should have it."""
    i, v, at = two_wires()
    m, n = at("M"), at("N")
    with pytest.raises(ValueError, match="must be a CONSTANT"):
        port_spec([v(*m) - v(*n) - i(*m) * i(*m)])


# --------------------------------------------------------------------------------------------
# Why a device rather than a weakly-conducting solid: the answer must not follow the grid.
# --------------------------------------------------------------------------------------------

MM = 1e-3
T_BAR, T_BRIDGE, W = 1.8 * MM, 0.6 * MM, 4 * MM


def _bridged(hz, rdev, as_metal):
    """Two coplanar bars with a gap, bridged either by a DEVICE or by a thin poor conductor.

    The bridge is 0.6 mm thick against bars of 1.8 mm -- a die under a bond wire, in miniature.
    """
    from jno.utils.solver.peec import bar_filaments, solve_network, terminal_nodes

    bars = [jno.Shape.box(0, 0, 0, 10 * MM, W, T_BAR), jno.Shape.box(12 * MM, 0, 0, 22 * MM, W, T_BAR)]
    sig = [SIG, SIG]
    if as_metal:  # R = rho L / A, and A is the CELL's, so the grid sets the on-resistance
        bars.append(jno.Shape.box(10 * MM, 0, 0, 12 * MM, W, T_BRIDGE))
        sig.append((2 * MM) / (rdev * W * T_BRIDGE))
    f = bar_filaments(bars, size=(1 * MM, W, hz), sigma=sig)
    p = np.asarray(f.nodes)
    term = {
        "A": terminal_nodes(f, lambda q: q[:, 0] < 1 * MM),
        "B": terminal_nodes(f, lambda q: q[:, 0] > 21 * MM),
    }
    dev = []
    if not as_metal:
        term["D"] = np.flatnonzero((p[:, 0] > 9 * MM) & (p[:, 0] < 10 * MM))
        term["S"] = np.flatnonzero((p[:, 0] > 12 * MM) & (p[:, 0] < 13 * MM))
        dev = [("D", "S", rdev + 0j)]
    _c, _phi, inj = solve_network(
        f, f.lattice["sigma"], term, [("A", "B", 1.0 + 0j)], (), (), dev, omega=0.0, matrix_free=False
    )
    return complex(1.0 / inj["A"]).real


def test_a_device_does_not_follow_the_grid_but_a_meshed_one_does():
    """The whole reason a component is a device: its value is the datasheet's, not the mesh's.

    Both z-pitches tile the bars exactly (1.8 = 3 x 0.6 = 2 x 0.9), so the only thing that changes
    between them is how thick a cell the 0.6 mm bridge is given. Measured on the example power
    module, refining z by 2x moved the port resistance by 0.08 % with the dies as devices and by a
    factor of 30 with them as metal.
    """
    rdev = 5e-3
    dev = [_bridged(hz, rdev, as_metal=False) for hz in (0.6 * MM, 0.9 * MM)]
    assert max(dev) / min(dev) - 1 < 1e-9  # the device is the same element at either pitch

    met = [_bridged(hz, rdev, as_metal=True) for hz in (0.6 * MM, 0.9 * MM)]
    assert abs(met[0] / dev[0] - 1) < 0.02  # resolved exactly, metal agrees with the device
    assert met[0] / met[1] > 1.3  # a 1.5x thicker cell, a 1.5x more conductive "die"
