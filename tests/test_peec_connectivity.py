"""A port pair with no metal between it is refused -- and the refusal is a graph question.

The check used to be a python union-find over every filament and every node it touches: 685,194
calls to its `find` on a 113,800-bar network, 0.186 s a solve and 14 % of a GPU solve, to answer a
question about geometry `build()` had already frozen. It is now `connected_components`.

The rewrite has to preserve three things that are easy to lose, and each is a case here: a terminal
is ONE electrical point however many nodes it owns, a DEVICE conducts and so bridges what it sits
between, and only a SOURCE pair is an error (a network may legitimately carry islands that no port
spans).
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import bar_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7


def _two_islands(gap=True):
    """Two boxes, touching or not, each with a pad at its far end."""
    a = jno.Shape.box(0, 0, 0, 0.010, 0.004, 0.001).name("a")
    x0 = 0.010 if not gap else 0.020
    b = jno.Shape.box(x0, 0, 0, x0 + 0.010, 0.004, 0.001).name("b")
    f = bar_filaments(a + b, size=0.001)
    p = np.asarray(f.nodes)
    term = {
        "A": terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9),
        "B": terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9),
    }
    return f, term


def test_a_gap_between_the_port_pair_is_refused():
    f, term = _two_islands(gap=True)
    with pytest.raises(ValueError, match="no conducting path between terminals"):
        solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0)


def test_metal_that_touches_is_accepted():
    """The other half: the same two boxes, in contact, must solve."""
    f, term = _two_islands(gap=False)
    cur, _phi, inj = solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0)
    assert np.isfinite(complex(inj["A"])) and abs(complex(inj["A"])) > 0


def test_a_DEVICE_bridges_the_gap_it_sits_across():
    """A device conducts, so a pair joined only through one is connected -- not an error.

    This is the case a components-based rewrite loses first: the graph must carry an edge for the
    device, which is not in the incidence matrix at all.
    """
    f, term = _two_islands(gap=True)
    # two more pads, on the facing ends of the two islands, with a device between them
    term = dict(term)
    term["M"] = terminal_nodes(f, lambda q: (q[:, 0] > 0.0089) & (q[:, 0] < 0.0101))
    term["N"] = terminal_nodes(f, lambda q: (q[:, 0] > 0.0199) & (q[:, 0] < 0.0211))
    assert len(term["M"]) and len(term["N"])  # the pads exist, or the test proves nothing
    cur, _phi, inj = solve_network(
        f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (("M", "N", 1e-3),), omega=0.0
    )
    assert np.isfinite(complex(inj["A"])) and abs(complex(inj["A"])) > 0


def test_an_island_no_PORT_spans_is_not_an_error():
    """Only a source pair has to be connected. Stray metal is a modelling choice, not a fault."""
    bar = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.001).name("bar")
    island = jno.Shape.box(0, 0.010, 0, 0.020, 0.014, 0.001).name("island")  # parallel, not touching
    f = bar_filaments(bar + island, size=0.001)
    p = np.asarray(f.nodes)
    on_bar = p[:, 1] < 0.005
    term = {
        "A": terminal_nodes(f, lambda q: (q[:, 0] < 0.0011) & (q[:, 1] < 0.005)),
        "B": terminal_nodes(f, lambda q: (q[:, 0] > 0.0189) & (q[:, 1] < 0.005)),
    }
    assert on_bar.sum() < len(p)  # the island really is there
    cur, _phi, inj = solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0)
    assert np.isfinite(complex(inj["A"])) and abs(complex(inj["A"])) > 0


def test_a_multi_node_terminal_is_one_electrical_point():
    """A pad owns many nodes; the graph must treat them as shorted, as the solve does."""
    f, term = _two_islands(gap=False)
    assert len(term["A"]) > 1 and len(term["B"]) > 1  # the pads really are multi-node
    cur, _phi, inj = solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0)
    assert np.isfinite(complex(inj["A"]))
