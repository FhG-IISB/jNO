"""``sol.voltage(...)`` -- the readout an OPEN terminal needs.

A terminal that carries no current is invisible to every other readout: `current` is zero by
construction and the port impedance describes the driven pair only. The induced voltage on an
unloaded secondary lives in the nodal potentials alone, and those were solved for all along and
then discarded. These cases pin what `voltage` means, with oracles that need no reference value:
the driven pair must reproduce the source that drives it, and a pair of coupled conductors must be
RECIPROCAL -- the same mutual inductance whichever one is driven.
"""

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIG, RAD, ELL = 5.8e7, 5e-4, 0.050
FREQ = 1e6


def _wire(x, name):
    """A straight wire along z at offset `x`, with a pad at each end."""
    w = jno.Shape.line([(x, 0, 0), (x, 0, ELL)], r=RAD, size=ELL / 10).attach(sigma=SIG).name(name)
    pads = jno.Shape.sphere(x, 0, 0.0, 2 * RAD).name(f"{name}0") + jno.Shape.sphere(x, 0, ELL, 2 * RAD).name(f"{name}1")
    return w + pads


def _ports(d):
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    return i, v, at


def test_the_driven_pair_reproduces_the_source_that_drives_it():
    """The tightest oracle available: `v(A) - v(B) - 1.0` IS the constraint the solve enforced, so
    `voltage('A', 'B')` must return exactly 1 V. It pins the sign, the pad definition and the scale
    at once -- a readout off by a sign, or reading the wrong node of the pad, fails here."""
    d = (_wire(0.0, "w")).domain()
    _i, v, at = _ports(d)
    sol = jno.peec([v(*at("w0")) - v(*at("w1")) - 1.0], freq=FREQ).solve()
    assert abs(complex(sol.voltage("w0", "w1")) - 1.0) < 1e-9


def test_an_open_terminal_carries_no_current_and_still_has_a_voltage():
    """The feature. A conductor with an open terminal is solved for but carries nothing, so every
    other readout says zero about it."""
    g = _wire(0.0, "a") + _wire(0.004, "b")
    d = g.domain()
    _i, v, at = _ports(d)
    sol = jno.peec([v(*at("a0")) - v(*at("a1")) - 1.0, _i(*at("b0")) - 0.0, v(*at("b1")) - 0.0], freq=FREQ).solve()
    assert abs(complex(sol.current("b0"))) < 1e-12, "an open terminal must carry no current"
    assert abs(complex(sol.voltage("b0", "b1"))) > 1e-9, "and must still show the induced voltage"


def test_the_induced_voltage_is_reciprocal():
    """Mutual inductance is symmetric, so the transfer impedance is the same whichever conductor is
    driven: `V_open / I_driven` must match under a swap. That is an exact claim about the physics
    with no reference number in it, and it is what a wrong pad or a wrong gauge would break."""

    def transfer(drive, open_):
        d = (_wire(0.0, "a") + _wire(0.004, "b")).domain()
        _i, v, at = _ports(d)
        sol = jno.peec(
            [
                v(*at(f"{drive}0")) - v(*at(f"{drive}1")) - 1.0,
                _i(*at(f"{open_}0")) - 0.0,
                v(*at(f"{open_}1")) - 0.0,
            ],
            freq=FREQ,
        ).solve()
        return complex(sol.voltage(f"{open_}0", f"{open_}1")) / complex(sol.current(f"{drive}0"))

    zab, zba = transfer("a", "b"), transfer("b", "a")
    assert abs(zab - zba) / abs(zab) < 1e-8, (zab, zba)


def test_a_two_terminal_difference_is_free_of_the_gauge():
    """A single potential is defined only against whatever the solve pinned; a difference is not.

    So move the reference: ground the open conductor at one end, then at the other. That is a
    different gauge for the same physics, and the induced voltage ACROSS it must not notice --
    which is why the docstring points at the two-terminal form rather than the absolute one.
    """

    def solve(open_end, gnd_end):
        d = (_wire(0.0, "a") + _wire(0.004, "b")).domain()
        _i, v, at = _ports(d)
        return jno.peec(
            [v(*at("a0")) - v(*at("a1")) - 1.0, _i(*at(open_end)) - 0.0, v(*at(gnd_end)) - 0.0],
            freq=FREQ,
        ).solve()

    lo = complex(solve("b0", "b1").voltage("b0", "b1"))
    hi = complex(solve("b1", "b0").voltage("b0", "b1"))
    assert abs(lo - hi) / abs(lo) < 1e-8, (lo, hi)


def test_a_weighted_pad_reads_the_weighted_average_the_circuit_used():
    """A weighted terminal is deliberately NOT shorted, so its nodes differ and the pad's potential
    is the weighted sum the constraint rows use -- unnormalised, exactly as `_pot` builds it. A
    readout that took the first node, or that normalised, would disagree with its own circuit."""
    d = _wire(0.0, "w").domain()
    _i, v, at = _ports(d)
    built = jno.peec([v(*at("w0")) - v(*at("w1")) - 1.0], freq=FREQ).build()
    n = len(np.asarray(built.nodes["w0"]))
    w = np.full(n, 1.0 / n)
    sol = built.solve(weights={"w0": w})
    phi = np.asarray(sol._pot)
    ids = np.asarray(built.nodes["w0"], dtype=int)
    assert abs(complex(sol.voltage("w0")) - complex(phi[ids] @ w)) < 1e-12


def test_an_unknown_terminal_is_named_not_guessed():
    d = _wire(0.0, "w").domain()
    _i, v, at = _ports(d)
    sol = jno.peec([v(*at("w0")) - v(*at("w1")) - 1.0], freq=FREQ).solve()
    with pytest.raises(ValueError, match="no terminal"):
        sol.voltage("nope")
