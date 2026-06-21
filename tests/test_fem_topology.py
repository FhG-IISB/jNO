"""Global edge numbering + orientation for edge-DOF elements.

``jno/utils/solver/fem_topology.py`` is the persisted, orientation-carrying edge
layer the non-nodal element zoo (Raviart–Thomas / Nédélec / Argyris edge DOFs)
assembles against. These tests pin the two properties the assembler relies on:
one global id per edge (shared edges deduped) and a per-cell orientation sign
that makes the two cells incident to an edge agree on its global direction.
"""

from __future__ import annotations

import numpy as np

from jno.utils.solver.fem_topology import (
    BASIX_TRIANGLE_EDGES,
    build_edge_topology,
)


def _structured_grid(n: int):
    """``n x n`` unit-square grid split into ``2 n^2`` triangles."""
    idx = lambda r, c: r * (n + 1) + c  # noqa: E731
    tris = []
    for r in range(n):
        for c in range(n):
            a, b, cc, d = idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)
            tris += [[a, b, cc], [b, d, cc]]
    return np.asarray(tris)


def test_two_triangle_square_dedups_shared_edge():
    # pts 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1); the diagonal (1,2) is shared.
    cells = np.array([[0, 1, 2], [1, 3, 2]])
    top = build_edge_topology(cells)
    assert top.n_edges == 5  # 4 boundary + 1 diagonal
    # basix edges [(1,2),(0,2),(0,1)]: T0 local-edge 0 == T1 local-edge 1 == diagonal (1,2)
    assert top.cell_edges[0, 0] == top.cell_edges[1, 1]
    # edge-use multiplicity: interior edge twice, boundary once
    counts = np.bincount(top.cell_edges.reshape(-1))
    assert (counts == 2).sum() == 1 and (counts == 1).sum() == 4


def test_orientation_sign_matches_canonical_global_direction():
    cells = _structured_grid(3)
    top = build_edge_topology(cells)
    for c in range(cells.shape[0]):
        for k, (i, j) in enumerate(BASIX_TRIANGLE_EDGES):
            a, b = int(cells[c, i]), int(cells[c, j])
            # sign is +1 iff the local traversal already runs low->high global vertex
            assert top.cell_edge_signs[c, k] == (1 if a < b else -1)
            # the stored edge is the canonical (min, max) pair
            np.testing.assert_array_equal(top.edge_vertices[top.cell_edges[c, k]], sorted((a, b)))


def test_shared_edge_has_consistent_global_direction():
    """Both cells incident to an interior edge must reference the same (min,max) dir."""
    cells = _structured_grid(4)
    top = build_edge_topology(cells)
    # for each global edge, every cell-use must carry the canonical vertex pair
    for c in range(cells.shape[0]):
        for k, (i, j) in enumerate(BASIX_TRIANGLE_EDGES):
            eid = top.cell_edges[c, k]
            a, b = int(cells[c, i]), int(cells[c, j])
            np.testing.assert_array_equal(top.edge_vertices[eid], sorted((a, b)))


def test_edge_count_matches_euler_for_structured_grids():
    # closed-ish triangulation of the square: V - E + F = 1 (F = triangles) => E = V + F - 1
    for n in (1, 2, 3, 5):
        cells = _structured_grid(n)
        top = build_edge_topology(cells)
        v = (n + 1) ** 2
        f = 2 * n * n
        assert top.n_edges == v + f - 1
