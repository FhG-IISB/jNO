"""A terminal that spans two disconnected pieces of metal is almost always a coordinate predicate
catching more than it meant to.

A tag is a function of position, so ``x < 3*mm`` selects that column of the WHOLE model -- every
layer of a stack, not just the one the pad is on. Wiring a port into a ground plane that way is
silent and produces perfectly plausible wrong answers: the port draws current through the plane, the
resistance collapses because the return path widened, and the loop inductance stops depending on how
far away the plane is. Every one of those reads as a solver problem and none of them is.

Measured on a real DBC module: a DC+ tag with no z filter gave 13.6 nH and 621 uOhm; the same model
with the tag restricted to the trace layer gave 20.8 nH and 2180 uOhm, against a reference of
21.7 nH. So this is worth a word.
"""

import logging

import jax
import numpy as np

import jno
from jno.utils.solver.peec import bar_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG, mm = 5.8e7, 1e-3
T = 0.5 * mm


def _stack():
    """A bar with an ISOLATED plane one cell below it -- the shape of a DBC stack."""
    bar = jno.Shape.box(0, 0, 2 * T, 20 * mm, 4 * mm, 3 * T, size=(2 * mm, 2 * mm, T)).attach(sigma=SIG).name("bar")
    plane = jno.Shape.box(0, 0, 0, 20 * mm, 4 * mm, T, size=(2 * mm, 2 * mm, T)).attach(sigma=SIG).name("plane")
    return bar + plane


class _Catch(logging.Handler):
    """A handler on the `jno` logger itself -- caplog attaches to the root, which never sees it."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _solve(term_where, _caplog=None):
    from jno.utils.solver import peec as P

    P._PITCH_WARNED.clear()  # the warning is deduped per process, so each case starts clean
    f = bar_filaments(
        [
            jno.Shape.box(0, 0, 2 * T, 20 * mm, 4 * mm, 3 * T, size=(2 * mm, 2 * mm, T)),
            jno.Shape.box(0, 0, 0, 20 * mm, 4 * mm, T, size=(2 * mm, 2 * mm, T)),
        ],
        sigma=[SIG, SIG],
    )
    term = {k: terminal_nodes(f, w) for k, w in term_where.items()}
    h = _Catch()
    log = logging.getLogger("jno")
    log.addHandler(h)
    try:
        solve_network(f, f.lattice["sigma"], term, [("A", "B", 1.0 + 0j)], (), (), (), omega=2 * np.pi * 1e6)
    finally:
        log.removeHandler(h)
    return "\n".join(h.lines)


def test_a_tag_that_spans_two_layers_is_flagged(caplog):
    """The failure this exists for: no z filter, so the pad grabs the plane underneath it too."""
    txt = _solve(
        {
            "A": lambda q: q[:, 0] < 2.1 * mm,  # NO z filter -- both layers
            "B": lambda q: (q[:, 0] > 17.9 * mm) & (q[:, 2] > 1.9 * T),
        },
        caplog,
    )
    assert "'A'" in txt and "not connected" in txt
    assert "SHORTS" in txt
    assert "restrict the tag in z" in txt  # it must point at the fix, not just the symptom


def test_a_correctly_filtered_tag_says_nothing(caplog):
    """The other half of the contract: a tag on one layer must not be nagged about."""
    txt = _solve(
        {
            "A": lambda q: (q[:, 0] < 2.1 * mm) & (q[:, 2] > 1.9 * T),
            "B": lambda q: (q[:, 0] > 17.9 * mm) & (q[:, 2] > 1.9 * T),
        },
        caplog,
    )
    assert txt == "", txt


def test_a_multi_node_pad_on_ONE_conductor_says_nothing(caplog):
    """A pad legitimately owns many nodes; only spanning SEPARATE metal is suspicious."""
    txt = _solve(
        {
            "A": lambda q: (q[:, 0] < 6.1 * mm) & (q[:, 2] > 1.9 * T),  # a wide pad, one layer
            "B": lambda q: (q[:, 0] > 17.9 * mm) & (q[:, 2] > 1.9 * T),
        },
        caplog,
    )
    assert txt == "", txt


def test_it_warns_rather_than_raising(caplog):
    """Two pieces of metal joined only by a bond wire added LATER is a legitimate model, so this
    cannot be fatal -- it says so and carries on."""
    txt = _solve({"A": lambda q: q[:, 0] < 2.1 * mm, "B": lambda q: (q[:, 0] > 17.9 * mm) & (q[:, 2] > 1.9 * T)}, caplog)
    assert txt  # it warned...
    # ...and the solve still returned, which the fixture already proves by not raising
