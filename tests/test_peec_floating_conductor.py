"""A conductor with no terminal floats, and jNO pins it rather than handing back a scipy traceback.

A body that no port touches has an undetermined potential -- add any constant and every current is
unchanged -- so its block of the system is singular by exactly one direction per floating piece.
Whether that is fatal depended on which preconditioner the network happened to take: a plain
lattice's ``diag(Z)`` Schur complement tolerated it, while a WELDED network's whole-system LU raised
``Factor is exactly singular`` from inside a callback, with nothing in the message about conductors.

Pinning one node of each floating piece removes the null direction and changes nothing physical: the
KCL row it replaces is IMPLIED by the others, because ``1' A = 0`` on an isolated component means
the summed balance is identically zero. So this is housekeeping, not a modelling choice -- which is
why it happens automatically and is only reported.

A ground plane under a DBC trace layer is exactly this case, and it is the reason the tests use one.
"""

import logging

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIG, mm = 5.8e7, 1e-3
T = 0.5 * mm


class _Catch(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _module(wire, ground, freq=1e6):
    """A bar over an isolated plane. ``wire`` makes it a WELDED network, which is the hard case."""
    bar = jno.Shape.box(0, 0, 2 * T, 20 * mm, 4 * mm, 3 * T, size=(2 * mm, 2 * mm, T)).attach(sigma=SIG).name("bar")
    plane = jno.Shape.box(0, 0, 0, 20 * mm, 4 * mm, T, size=(2 * mm, 2 * mm, T)).attach(sigma=SIG).name("plane")
    sh = bar + plane
    if wire:
        p = [(6 * mm, 2 * mm, 3 * T), (10 * mm, 2 * mm, 3 * T + 1.5 * mm), (14 * mm, 2 * mm, 3 * T)]
        sh = sh + jno.Shape.line(p, r=1.9e-4, size=1 * mm).attach(sigma=SIG).name("w")
    d = sh.domain()
    d.tag("A", lambda x, y, z: (x < 2.1 * mm) & (z > 1.9 * T))
    d.tag("B", lambda x, y, z: (x > 17.9 * mm) & (z > 1.9 * T))
    cons_i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    cons = [v(*at("A")) - v(*at("B")) - 1.0]
    if ground:
        # exactly ONE node: a multi-node tag would tie those nodes equipotential, which is a real
        # short across part of the plane and not the pure reference the auto-pin applies
        d.tag("GND", lambda x, y, z: (x < 1.1 * mm) & (y < 1.1 * mm) & (z < T))
        cons.append(v(*at("GND")) - 0.0)
    return jno.peec(cons, freq=freq).build()


def _solve(wire, ground):
    h = _Catch()
    log = logging.getLogger("jno")
    lvl = log.level
    log.setLevel(logging.INFO)
    log.addHandler(h)
    try:
        s = _module(wire, ground).solve()
        return float(np.real(s.R)), float(np.real(s.L)), "\n".join(h.lines)
    finally:
        log.removeHandler(h)
        log.setLevel(lvl)


@pytest.mark.parametrize("wire", [False, True])
def test_a_floating_conductor_solves_without_being_grounded_by_hand(wire):
    """The regression: with a bond wire this used to raise `Factor is exactly singular`."""
    R, L, _txt = _solve(wire, ground=False)
    assert np.isfinite(R) and np.isfinite(L) and L > 0


@pytest.mark.parametrize("wire", [False, True])
def test_pinning_changes_nothing_that_can_be_measured(wire):
    """The claim that makes this housekeeping rather than modelling: same R, same L."""
    Ra, La, _ = _solve(wire, ground=False)
    Rb, Lb, _ = _solve(wire, ground=True)
    assert abs(Ra / Rb - 1) < 1e-9, (Ra, Rb)
    assert abs(La / Lb - 1) < 1e-9, (La, Lb)


def test_it_says_which_conductor_it_pinned():
    _R, _L, txt = _solve(wire=True, ground=False)
    assert "float" in txt.lower()
    assert "plane" in txt or "pinned" in txt.lower()


def test_a_fully_connected_network_is_left_alone():
    """No floating piece -> nothing to pin, and nothing said."""
    bar = jno.Shape.box(0, 0, 0, 20 * mm, 4 * mm, T, size=(2 * mm, 2 * mm, T)).attach(sigma=SIG).name("bar")
    d = bar.domain()
    d.tag("A", lambda x, y, z: x < 2.1 * mm)
    d.tag("B", lambda x, y, z: x > 17.9 * mm)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    h = _Catch()
    log = logging.getLogger("jno")
    log.setLevel(logging.INFO)
    log.addHandler(h)
    try:
        s = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).build().solve()
    finally:
        log.removeHandler(h)
    assert np.isfinite(float(np.real(s.R)))
    assert "float" not in "\n".join(h.lines).lower()
