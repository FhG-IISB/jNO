"""The cross-check harness's geometry rule, pinned.

`benchmarks/pypeec_layout.py` runs pypeec on the same power-module layouts jNO and Ansys Q3D were
run on, and it is worth having because an independent code is the only thing that can separate "our
model of this layout is wrong" from "both codes read the geometry the same wrong way".

It rotted once, silently. Its cell-selection had become an OVERLAP test while its own comment
documented a CENTRE test, and it then reproduced the exact failure value the comment records --
12.674 nH against a 20.641 reference. Nothing caught it: the harness ran, converged, and returned a
plausible number.

So the rule is pinned here rather than trusted. It needs neither pypeec nor the collaborator's
layouts -- it is arithmetic on a grid, which is precisely why it is testable and why it should have
been all along.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, "benchmarks")
from pypeec_layout import cells_in  # noqa: E402

PITCH = 1.0
NX = NY = 12
LIN = lambda ix, iy, iz: int(ix + NX * iy + NX * NY * iz)
KS = range(1)


def _rect(x0, x1, y0, y1):
    return {"x": [x0, x1], "y": [y0, y1]}


def test_a_cell_belongs_to_the_rectangle_holding_its_CENTRE():
    """The rule itself. A rectangle spanning 2.0-5.0 mm at a 1 mm pitch owns the cells centred at
    2.5, 3.5 and 4.5 -- three of them, not the five an overlap test would claim."""
    got = cells_in(_rect(2.0, 5.0, 0.0, 1.0), NX, NY, PITCH, KS, LIN)
    assert [c % NX for c in got] == [2, 3, 4]


def test_two_traces_separated_by_a_SUB_PITCH_gap_do_not_merge():
    """The failure that cost 40 % of the loop inductance.

    This module has 40+ traces with sub-millimetre gaps. Rounding outward grows each by up to a
    voxel per side, so neighbours touch, the port finds a shortcut that is not in the layout, and
    the inductance collapses. Two rectangles either side of a 0.2 mm gap must stay disjoint at a
    1 mm pitch -- the case an overlap test gets wrong and a centre test gets right.
    """
    a = set(cells_in(_rect(0.0, 3.9, 0.0, 1.0), NX, NY, PITCH, KS, LIN))
    b = set(cells_in(_rect(4.1, 8.0, 0.0, 1.0), NX, NY, PITCH, KS, LIN))
    assert a and b, "both traces must exist at all"
    assert not (a & b), f"traces merged across a 0.2 mm gap: {sorted(a & b)}"


def test_it_is_not_the_overlap_test_it_once_silently_became():
    """Stated as the difference, so the regression cannot come back wearing the right comment.

    The overlap rule is `(i+1)p > x0 and i*p < x1`. The two agree wherever every edge cell also has
    its centre covered, so the case has to be chosen: edges at 2.7 and 5.3 fall INSIDE cells 2 and 5
    without reaching their centres, which is exactly the situation a real trace boundary is in and
    exactly where the two rules part company.
    """
    r = _rect(2.7, 5.3, 0.0, 1.0)
    centre = set(cells_in(r, NX, NY, PITCH, KS, LIN))
    lo, hi = np.arange(NX) * PITCH, (np.arange(NX) + 1) * PITCH
    overlap = {LIN(i, 0, 0) for i in np.flatnonzero((hi > r["x"][0]) & (lo < r["x"][1])).tolist()}
    assert centre < overlap, (sorted(centre), sorted(overlap))
    assert len(overlap - centre) == 2, "one cell per side is exactly what rounding outward adds"


def test_a_rectangle_narrower_than_one_cell_owns_at_most_one():
    """The extreme: a trace thinner than the pitch cannot become two cells wide. It may round to
    nothing -- which is a resolution limit, and honest -- but never to more than it covers."""
    for x0 in np.linspace(0.0, 1.0, 11):
        got = cells_in(_rect(x0, x0 + 0.4, 0.0, 1.0), NX, NY, PITCH, KS, LIN)
        assert len(got) <= 1, (x0, got)


def test_the_layout_harness_needs_its_paths_named():
    """Neither input is in this repo -- the layouts are the collaborator's. Asking for them by name
    is the difference between a clear stop and a run against whatever happens to be on disk."""
    import os

    import pypeec_layout

    keep = {k: os.environ.pop(k, None) for k in ("JNO_LAYOUT_DIR", "PYPEEC_EXAMPLES")}
    try:
        with pytest.raises(SystemExit, match="JNO_LAYOUT_DIR"):
            pypeec_layout._paths()
    finally:
        os.environ.update({k: v for k, v in keep.items() if v is not None})
