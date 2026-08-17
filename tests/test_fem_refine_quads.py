"""Local refinement of a quadrilateral mesh, and the hanging nodes it creates.

The mesh half of hanging-node adaptivity: split marked cells into four, keep neighbours within one
refinement level, and report which nodes ended up constrained. The constraint *weights* are proven
separately (``tests/test_fem_hex_tie.py``); this file is about the mesh.

**Not yet wired into `fem.solve`.** Assembling on the refined mesh needs one thing this does not
provide: a boundary. jNO derives the boundary topologically -- a facet belonging to one cell -- and
that rule is false on a non-conforming mesh, where a coarse cell's full edge and each of its
neighbour's two half-edges all belong to exactly one cell. Measured below: 12 of 32 "boundary" edges
are actually the 2:1 interface. So the pieces here are tested on their own until that is resolved.
"""

from __future__ import annotations

import numpy as np
import pytest

from jno.domain.geometries import Geometries
from jno.utils.solver.fem_facets import _boundary_faces, _face_table
from jno.utils.solver.fem_refine import balance_marks, hanging_nodes, refine_quads


def _grid(n=4):
    m, _, _ = Geometries.equi_distant_rect(nx=n, ny=n, cell="quad")(None)
    return np.asarray(m.points)[:, :2], np.asarray({c.type: np.asarray(c.data) for c in m.cells}["quad"])


def _areas(pts, quads):
    """Signed area of each quad by its diagonals — positive iff the winding survived the split."""
    v = pts[quads]
    return (
        (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 3, 1] - v[:, 1, 1]) - (v[:, 3, 0] - v[:, 1, 0]) * (v[:, 2, 1] - v[:, 0, 1])
    ) / 2


def _mark_near(pts, quads, centre, radius):
    return np.where(np.linalg.norm(pts[quads].mean(axis=1) - np.asarray(centre), axis=1) < radius)[0]


# ------------------------------------------------------------------------------- the split itself


def test_a_split_conserves_area_and_winding():
    """Four children exactly tile the parent. Area is the cheap global check; the SIGN is the one that
    catches a child whose corners were listed in the wrong order, which would still tile correctly and
    assemble to a negative Jacobian."""
    pts, quads = _grid()
    a0 = _areas(pts, quads).sum()
    p1, q1 = refine_quads(pts, quads, _mark_near(pts, quads, (0.5, 0.5), 0.3))
    a1 = _areas(p1, q1)
    assert a1.sum() == pytest.approx(a0, rel=1e-12)
    assert (a1 > 0).all(), "a child quad came out with reversed winding"


def test_one_marked_cell_becomes_four():
    pts, quads = _grid()
    p1, q1 = refine_quads(pts, quads, [0])
    assert len(q1) == len(quads) + 3  # one replaced by four
    assert len(p1) == len(pts) + 5  # four edge midpoints + one centre


def test_midpoints_are_shared_between_neighbours():
    """A midpoint is keyed by global EDGE, so two cells splitting the same edge get the same node.
    Creating one per cell would duplicate it and silently disconnect the two halves while every count
    still looked right."""
    pts, quads = _grid()
    p1, q1 = refine_quads(pts, quads, [0, 1])  # two cells; on this grid they share an edge
    assert len(p1) == len(np.unique(np.round(p1, 12), axis=0)), "a node was duplicated"


# ---------------------------------------------------------------------------------- hanging nodes


def test_hanging_nodes_sit_exactly_at_their_parents_midpoint():
    pts, quads = _grid()
    p1, q1 = refine_quads(pts, quads, _mark_near(pts, quads, (0.15, 0.15), 0.3))
    hang = hanging_nodes(p1, q1)
    assert hang, "a partial refinement must leave hanging nodes"
    for node, parents in hang.items():
        (a, wa), (b, wb) = parents
        assert wa == pytest.approx(0.5) and wb == pytest.approx(0.5)
        np.testing.assert_allclose(p1[node], 0.5 * (p1[a] + p1[b]), atol=1e-12)


def test_a_fully_refined_mesh_has_no_hanging_nodes():
    """Refining everything is conforming again — the degenerate case, and a guard that the detector
    is not simply reporting every midpoint."""
    pts, quads = _grid()
    p1, q1 = refine_quads(pts, quads, np.arange(len(quads)))
    assert hanging_nodes(p1, q1) == {}
    assert len(q1) == 4 * len(quads)


def test_hanging_detection_is_history_free():
    """The set is derived from the MESH, not from the split that produced it. A node left hanging by
    an earlier round stays hanging, and a history-based set drops it — so the same mesh must give the
    same answer however it was reached."""
    pts, quads = _grid()
    p1, q1 = refine_quads(pts, quads, _mark_near(pts, quads, (0.15, 0.15), 0.3))
    first = hanging_nodes(p1, q1)
    # re-detected on the identical mesh, with no reference to the refinement
    assert hanging_nodes(np.array(p1), np.array(q1)) == first


# -------------------------------------------------------------------------------------- 2:1 balance


def test_repeated_refinement_stays_two_to_one_balanced_and_unchained():
    """The property the whole scheme rests on: no hanging node may have a hanging PARENT.

    Two ways it broke while this was written, both measured. Edge-topology adjacency goes blind once a
    hanging node exists — a coarse cell's edge and its neighbour's two half-edges are different edges,
    so the cells stop sharing an edge id, and three rounds produced cells at levels 0 and 3 side by
    side. And checking only the CURRENT mesh misses a 2-level jump that this round is about to create,
    which let a parent hang by round 4.
    """
    pts, quads = _grid()
    for rnd in range(4):
        pts, quads = refine_quads(pts, quads, _mark_near(pts, quads, (0.12, 0.12), 0.22))
        hang = hanging_nodes(pts, quads)
        chained = [n for w in hang.values() for n, _ in w if n in hang]
        assert not chained, f"round {rnd + 1}: {len(chained)} hanging nodes have a hanging parent"
        assert _areas(pts, quads).sum() == pytest.approx(1.0, rel=1e-12)


def test_balance_marks_grows_the_marked_set_rather_than_the_caller_s():
    """Marking one cell beside an already-refined patch must pull in its neighbours."""
    pts, quads = _grid()
    p1, q1 = refine_quads(pts, quads, _mark_near(pts, quads, (0.15, 0.15), 0.3))
    p2, q2 = refine_quads(p1, q1, _mark_near(p1, q1, (0.12, 0.12), 0.12))
    asked = _mark_near(p2, q2, (0.1, 0.1), 0.06)
    closed = balance_marks(p2, q2, asked)
    assert closed.sum() >= len(asked)
    assert not any(n in hanging_nodes(p2, q2) for w in hanging_nodes(p2, q2).values() for n, _ in w)


# ------------------------------------------------------- why this is not wired into the solve yet


def test_the_topological_boundary_rule_fails_on_a_non_conforming_mesh():
    """The blocker, pinned so it is not rediscovered.

    jNO derives the boundary as "facets belonging to exactly one cell". On a 2:1 interface the coarse
    cell's full edge belongs to one cell, and so does each of the neighbour's two half-edges — so the
    interface is indistinguishable from the true perimeter by that rule alone. Assembling on such a
    mesh therefore pins the interface as a Dirichlet boundary, which is silent and wrong.
    """
    pts, quads = _grid()
    lf, nfn = _face_table("quad")

    flat, sel, _ = _boundary_faces(quads, lf, nfn)
    assert len(flat[sel]) == 16, "a conforming 4x4 grid's perimeter is 16 edges"

    p1, q1 = refine_quads(pts, quads, _mark_near(pts, quads, (0.15, 0.15), 0.3))
    flat, sel, _ = _boundary_faces(q1, lf, nfn)
    bf = flat[sel]
    on_perimeter = np.array(
        [
            np.all(
                np.isclose(p1[e][:, 0], 0)
                | np.isclose(p1[e][:, 0], 1)
                | np.isclose(p1[e][:, 1], 0)
                | np.isclose(p1[e][:, 1], 1)
            )
            for e in bf
        ]
    )
    assert (~on_perimeter).sum() > 0, "the interface must be what makes this rule fail"
    assert (~on_perimeter).sum() == len(bf) - on_perimeter.sum() == 12
