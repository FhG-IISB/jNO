"""Mesh edge topology for non-nodal (edge-DOF) finite elements.

The nodal-Lagrange path indexes DOFs as ``node*vec + comp`` and never needs to
know which edge is shared by which two cells. Edge-DOF families — Raviart–Thomas
(H(div)), Nédélec (H(curl)), and the edge-normal DOFs of Argyris — do: each
interior edge carries DOF(s) shared by its two incident triangles, so the
assembler needs (a) one global id per edge and (b) a per-cell orientation sign so
the two cells agree on the edge's reference direction (else the normal flux /
tangential trace is discontinuous and the method does not converge).

This module builds both from the linear (P1) triangle connectivity. It is the
persisted, orientation-carrying generalisation of the throwaway ``edge_map`` in
:func:`fem_utils._promote_to_quadratic` (which dedups edges only to place P2
midpoints, then discards the map).

Orientation convention (standard, e.g. DOLFINx): the *global* direction of an
edge runs from its lower- to its higher-numbered vertex. A cell's local edge
``(i, j)`` (global vertices ``a = cells[c, i]``, ``b = cells[c, j]``) gets sign
``+1`` if ``a < b`` (its local traversal matches the global direction) else
``-1``. Two cells sharing an edge reference the same global direction, so their
edge DOFs are sign-consistent.

The ``local_edges`` argument is the element's reference edge ordering as
vertex-index pairs; it MUST match the element library's convention so DOF ``k``
maps to the intended edge. basix's triangle edges are ``[(1, 2), (0, 2), (0, 1)]``
(note: different from meshio's ``[(0, 1), (1, 2), (2, 0)]``).
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np

# basix reference edge ordering for a triangle (DOF k of RT/Nédélec lives on
# local edge k). Kept here so callers don't hard-code it at each use site.
BASIX_TRIANGLE_EDGES: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))


class EdgeTopology(NamedTuple):
    """Global edge numbering + per-cell orientation for an edge-DOF element.

    ``cell_edges[c, k]``      global edge id of cell ``c``'s local edge ``k``.
    ``cell_edge_signs[c, k]`` orientation sign (``+1``/``-1``) of that local edge.
    ``edge_vertices[e]``      canonical ``(min, max)`` global vertex pair of edge ``e``.
    ``n_edges``               total number of unique edges.
    """

    cell_edges: np.ndarray  # (n_cells, n_local_edges) int
    cell_edge_signs: np.ndarray  # (n_cells, n_local_edges) int8 (+1/-1)
    edge_vertices: np.ndarray  # (n_edges, 2) int
    n_edges: int


def build_edge_topology(cells: np.ndarray, local_edges: Sequence[Tuple[int, int]] = BASIX_TRIANGLE_EDGES) -> EdgeTopology:
    """Global edge ids + per-cell orientation signs for triangle ``cells``.

    ``cells`` is ``(n_cells, n_vertices)`` of global vertex indices (the linear/P1
    connectivity; only the first ``max(local_edges)+1`` columns are read, so a P2
    ``triangle6`` array works unchanged). See the module docstring for the
    orientation convention.
    """
    cells = np.asarray(cells)
    n_cells = cells.shape[0]
    n_local = len(local_edges)
    cell_edges = np.empty((n_cells, n_local), dtype=np.int64)
    cell_edge_signs = np.empty((n_cells, n_local), dtype=np.int8)
    edge_map: Dict[Tuple[int, int], int] = {}
    edge_vertices: List[Tuple[int, int]] = []

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
            cell_edge_signs[c, k] = 1 if a < b else -1

    return EdgeTopology(
        cell_edges=cell_edges,
        cell_edge_signs=cell_edge_signs,
        edge_vertices=np.asarray(edge_vertices, dtype=np.int64).reshape(-1, 2),
        n_edges=len(edge_vertices),
    )
