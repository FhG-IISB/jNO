"""The region tags of a tie survive the operations wrapped around it.

``u(A) - u(B)`` is the only point where each side's region is still visible — the ``BinaryOp``
discards the per-side bound views — so the trace stashes the two tags on ``_periodic_tie``. Anything
built ON TOP of that tie (an offset, a scale) discards the tie's own view in turn, and the tags have
nowhere else to survive.

Propagating the stamp does not widen what any solver accepts: the stamp says "these two regions were
combined", the recogniser decides whether the relation means anything. jno.fem still refuses an
offset tie, and this pins that it refuses it LOUDLY rather than reading it as a plain tie.
"""

import numpy as np
import pytest

import jno


@pytest.fixture
def dom():
    d = jno.Shape.rect(0, 0, 1, 1).domain(size=0.4)
    d.tag("A", lambda x, y: np.isclose(x, 0))
    d.tag("B", lambda x, y: np.isclose(x, 1))
    d.tag("C", lambda x, y: np.isclose(y, 1))
    return d


def _sides(d):
    u, _ = d.fem_symbols()
    return u, d.variable("A", split=True)[:2], d.variable("B", split=True)[:2]


def test_a_plain_tie_is_stamped_with_both_regions(dom):
    u, a, b = _sides(dom)
    assert getattr(u(*a) - u(*b), "_periodic_tie", None) == ("A", "B")


@pytest.mark.parametrize(
    "build",
    [
        lambda u, a, b: u(*a) - u(*b) - 1.0,  # an inhomogeneous tie: a source across the pair
        lambda u, a, b: 1.0 - (u(*a) - u(*b)),  # the same, written the other way round
        lambda u, a, b: 2.0 * (u(*a) - u(*b)),  # a scaled tie
    ],
    ids=["offset", "reversed", "scaled"],
)
def test_the_stamp_survives_an_operation_on_the_tie(dom, build):
    u, a, b = _sides(dom)
    assert getattr(build(u, a, b), "_periodic_tie", None) == ("A", "B")


def test_a_one_sided_constraint_is_never_stamped(dom):
    """A plain Dirichlet must stay a plain Dirichlet — there is no second region to tie to."""
    u, _ = dom.fem_symbols()
    c = dom.variable("C", split=True)[:2]
    assert getattr(u(*c) - 0.0, "_periodic_tie", None) is None


def test_fem_still_refuses_an_offset_tie_loudly(dom):
    """The stamp is not permission: jno.fem has no reading for `u(A) - u(B) - g` and must say so.

    Without the refusal the offset would be dropped and the constraint read as a plain tie -- the
    quietly-wrong answer this repo exists to avoid.
    """
    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xa, ya, _ = dom.variable("A", split=True)
    xb, yb, _ = dom.variable("B", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    laplace = ui.x * vi.x + ui.y * vi.y - vi
    with pytest.raises(ValueError, match=r"must be `u\(A\) - u\(B\)`"):
        jno.fem([laplace, u(xa, ya) - u(xb, yb) - 1.0])
