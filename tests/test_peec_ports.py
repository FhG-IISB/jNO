"""Reading a PEEC constraint list into ports.

``port_spec`` is pure — it reads field names, region tags and constants off the trace and touches no
geometry — so these cases pin the vocabulary itself, independent of what a terminal name resolves to.
"""

import numpy as np
import pytest

import jno
from jno.utils.solver.peec import port_spec


@pytest.fixture
def sym():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    for name, f in [
        ("P", lambda x, y: np.isclose(x, 0)),
        ("N", lambda x, y: np.isclose(x, 1)),
        ("AC", lambda x, y: np.isclose(y, 1)),
    ]:
        d.tag(name, f)
    i, v = d.trial_function(name="i"), d.trial_function(name="v")
    at = lambda t: d.variable(t, split=True)[:2]
    return i, v, at


def test_the_three_port_forms(sym):
    i, v, at = sym
    src, cur, gnd, _dev = port_spec(
        [
            v(*at("P")) - v(*at("N")) - 12.0,  # a 12 V source across the pair
            i(*at("AC")) - 0.0,  # open terminal
            v(*at("N")) - 0.0,  # ground
        ]
    )
    assert src == [("P", "N", 12.0 + 0j)]
    assert cur == [("AC", 0.0 + 0j)]
    assert gnd == [("N", 0.0 + 0j)]


def test_a_current_source_is_a_nonzero_terminal_current(sym):
    i, v, at = sym
    _, cur, _, _ = port_spec([i(*at("P")) - 250.0])
    assert cur == [("P", 250.0 + 0j)]


def test_a_bare_terminal_potential_reads_as_ground(sym):
    """`v(A)` with no offset is `v(A) - 0`: the reference node."""
    i, v, at = sym
    _, _, gnd, _ = port_spec([v(*at("N"))])
    assert gnd == [("N", 0.0 + 0j)]


def test_a_relation_between_two_terminals_is_refused_on_the_current(sym):
    """`i(A) - i(B)` has no reading: a two-terminal relation is a voltage source."""
    i, v, at = sym
    with pytest.raises(ValueError, match="is not a port"):
        port_spec([i(*at("P")) - i(*at("N"))])


def test_a_constraint_mixing_both_fields_must_be_a_whole_device(sym):
    """Naming both fields reads as a device, so it needs the two terminals a device sits between."""
    i, v, at = sym
    with pytest.raises(ValueError, match="to itself is a short"):
        port_spec([v(*at("P")) - i(*at("P"))])  # one terminal only: not a device, not anything


def test_an_unknown_field_is_named_in_the_error(sym):
    i, v, at = sym
    d = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    d.tag("P", lambda x, y: np.isclose(x, 0))
    w = d.trial_function(name="w")
    with pytest.raises(ValueError, match=r"unknown field 'w'"):
        port_spec([w(*d.variable("P", split=True)[:2]) - 1.0])


def test_an_unbound_constraint_says_to_bind_the_region(sym):
    """A constraint that names no region cannot be a port, and the message says what to write."""
    i, v, at = sym
    with pytest.raises(ValueError, match="bind the region first"):
        port_spec([v - 1.0])
