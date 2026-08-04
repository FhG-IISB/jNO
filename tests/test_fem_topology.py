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


# --------------------------------------------------------------------------------------------
# the numbering itself: edge ids ARE the global DOF numbering of an edge element
# --------------------------------------------------------------------------------------------
import pytest  # noqa: E402

from jno.utils.solver.fem_topology import BASIX_TET_EDGES  # noqa: E402


def _first_encounter_reference(cells, local_edges):
    """The nested Python loop ``build_edge_topology`` was, kept as the oracle."""
    cells = np.asarray(cells)
    n_cells, n_local = cells.shape[0], len(local_edges)
    cell_edges = np.empty((n_cells, n_local), dtype=np.int64)
    signs = np.empty((n_cells, n_local), dtype=np.int8)
    edge_map, edge_vertices = {}, []
    for c in range(n_cells):
        for k, (i, j) in enumerate(local_edges):
            a, b = int(cells[c, i]), int(cells[c, j])
            key = (a, b) if a < b else (b, a)
            eid = edge_map.get(key)
            if eid is None:
                eid = len(edge_vertices)
                edge_map[key] = eid
                edge_vertices.append(key)
            cell_edges[c, k] = eid
            signs[c, k] = 1 if a < b else -1
    return cell_edges, signs, np.asarray(edge_vertices, dtype=np.int64).reshape(-1, 2)


@pytest.mark.parametrize(
    "cells,edges",
    [
        (np.array([[0, 1, 2], [1, 3, 2]]), BASIX_TRIANGLE_EDGES),
        (_structured_grid(6), BASIX_TRIANGLE_EDGES),
        (np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), BASIX_TET_EDGES),
        # vertex ids deliberately NOT ascending within a cell, so encounter order != sorted order
        (np.array([[9, 4, 7], [4, 1, 7], [7, 1, 0]]), BASIX_TRIANGLE_EDGES),
        (np.zeros((0, 3), dtype=int), BASIX_TRIANGLE_EDGES),
    ],
)
def test_edge_numbering_is_first_encounter_order(cells, edges):
    """Edge ids are the DOF numbering, so a permutation would silently reorder every N1E/RT/Morley
    solution vector. The vectorised build must reproduce the loop's order EXACTLY, not merely the
    same set of edges."""
    top = build_edge_topology(cells, edges)
    ref_ids, ref_signs, ref_verts = _first_encounter_reference(cells, edges)
    np.testing.assert_array_equal(top.cell_edges, ref_ids)
    np.testing.assert_array_equal(top.cell_edge_signs, ref_signs)
    np.testing.assert_array_equal(top.edge_vertices, ref_verts)
    assert top.n_edges == len(ref_verts)


def test_edge_numbering_matches_the_loop_on_a_real_tet_mesh():
    import jno

    cells = np.asarray(jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain().built_mesh.cells_dict["tetra"])
    top = build_edge_topology(cells, BASIX_TET_EDGES)
    ref_ids, ref_signs, ref_verts = _first_encounter_reference(cells, BASIX_TET_EDGES)
    np.testing.assert_array_equal(top.cell_edges, ref_ids)
    np.testing.assert_array_equal(top.cell_edge_signs, ref_signs)
    np.testing.assert_array_equal(top.edge_vertices, ref_verts)
