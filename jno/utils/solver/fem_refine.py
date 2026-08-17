"""Local refinement of a quadrilateral mesh, with hanging-node constraints.

This is the *other* h-adaptivity, and the one the tensor-product world actually uses. mmg adapts
simplices by edge split/collapse/swap and has no quad analogue; rebuilding a ``Shape`` plan at a finer
size field works but is a global remesh that needs a geometry to rebuild from and has no 3-D
counterpart. Splitting a quadrilateral into 4 needs none of that -- it is exact, local, and works on a
mesh loaded from a file.

The price is **conformity**: a split cell's edge midpoint is not a vertex of its unrefined neighbour,
so the mesh is no longer conforming and that node's value is not free. It is a *hanging node*, and it
is constrained to the coarse edge it lies on::

    u_hanging = sum_i w_i u_parent_i

which is the same ``u_constrained = sum w u_free`` relation as a periodic tie and a mortar coupling.
That is why this reuses their prolongation rather than introducing a second constraint mechanism:
deal.II, MFEM and p4est all take this route for the same reason.

**2:1 balance.** Neighbouring cells are kept within one refinement level of each other, by adding to
the marked set until that holds. Without it a coarse edge can carry two hanging nodes at 1/4 and 3/4,
whose parents are themselves hanging -- a chained constraint this does not build and would silently
get wrong.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# A quadrilateral's edges in VTK order: the perimeter walk 0-1-2-3, which is how meshio stores a quad
# and how `fem_facets._LOCAL_FACES_QUAD` reads it.
QUAD_EDGES: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))


def _edge_topology(quads: np.ndarray):
    from .fem_topology import build_edge_topology

    return build_edge_topology(np.asarray(quads), QUAD_EDGES)


def _node_lookup(points: np.ndarray, tol: float | None = None):
    """A coordinate -> node-id map, for asking "is there a node HERE?"."""
    pts = np.asarray(points, dtype=float)
    span = float(np.ptp(pts)) if pts.size else 1.0
    q = max(span, 1.0) * (1e-9 if tol is None else tol)
    keys = np.round(pts / q).astype(np.int64)
    return {tuple(k): i for i, k in enumerate(keys)}, q


def hanging_nodes(points: np.ndarray, quads: np.ndarray) -> Dict[int, List[Tuple[int, float]]]:
    """Every hanging node of a (possibly non-conforming) quad mesh, found GEOMETRICALLY.

    Returns ``{node: [(parent, weight), ...]}``. A node is hanging when it lies at the midpoint of some
    cell's edge without being a vertex of that cell: the cell still spans the whole edge, so its basis
    carries no degree of freedom there and the node's value is not free.

    Detected from the mesh alone rather than from the refinement that produced it. Deriving it from the
    split instead only finds the nodes created in THAT round -- a node left hanging by an earlier round
    stays hanging, and a history-based set silently drops it. It also survives a mesh that arrives from
    anywhere else, which a history cannot.

    Relies on the 2:1 balance below: with it, a hanging node is always an edge MIDPOINT. Without it a
    coarse edge can carry nodes at 1/4 and 3/4 whose own parents hang, and this returns weights that
    refer to constrained nodes -- a chained constraint that is not built here.
    """
    pts = np.asarray(points, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    lut, q = _node_lookup(pts)
    out: Dict[int, List[Tuple[int, float]]] = {}
    for cell in quads:
        own = set(int(x) for x in cell)
        for a_loc, b_loc in QUAD_EDGES:
            a, b = int(cell[a_loc]), int(cell[b_loc])
            mid = tuple(np.round(0.5 * (pts[a] + pts[b]) / q).astype(np.int64))
            n = lut.get(mid)
            if n is not None and n not in own:
                out[int(n)] = [(a, 0.5), (b, 0.5)]  # Q1 on the coarse edge: linear interpolation
    return out


def balance_marks(points: np.ndarray, quads: np.ndarray, marked) -> np.ndarray:
    """Grow ``marked`` until refining it leaves no cell edge spanning more than one finer neighbour.

    A cell must be refined if one of its edges would end up carrying two hanging nodes -- which is
    what a 2-level jump means, and what makes a hanging node's parents themselves hang.

    Geometric, for the same reason :func:`hanging_nodes` is: once a hanging node exists, a coarse
    cell's edge and its neighbour's two half-edges are DIFFERENT edges, so the two cells no longer
    share an edge id and any adjacency built from edge topology goes blind to them. Measured before
    that was fixed: three rounds of corner refinement produced cells at levels 0 and 3 side by side
    with zero hanging nodes reported.
    """
    pts = np.asarray(points, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    marked = np.isin(np.arange(len(quads)), np.asarray(marked, dtype=np.int64))
    lut, q = _node_lookup(pts)

    def _node_at(pa, pb, t):
        return lut.get(tuple(np.round(((1 - t) * pa + t * pb) / q).astype(np.int64)))

    cells_of_node: Dict[int, List[int]] = {}
    for c, cell in enumerate(quads):
        for v in cell:
            cells_of_node.setdefault(int(v), []).append(c)

    for _ in range(64):  # each sweep only adds cells; 64 levels is far past any real use
        grew = False
        for c in range(len(quads)):
            if marked[c]:
                continue
            cell = quads[c]
            own = set(int(x) for x in cell)
            for a_loc, b_loc in QUAD_EDGES:
                a, b = int(cell[a_loc]), int(cell[b_loc])
                pa, pb = pts[a], pts[b]
                # (1) already two levels finer: nodes sit at the QUARTER points of this edge
                hit = _node_at(pa, pb, 0.25) is not None or _node_at(pa, pb, 0.75) is not None
                # (2) about to become two levels finer: this edge already carries a hanging node, and
                #     the finer cell holding it is being refined THIS round. Checking only the current
                #     mesh misses this -- measured, it let a hanging node's own parent hang by round 4.
                if not hit:
                    m = _node_at(pa, pb, 0.5)
                    if m is not None and m not in own:
                        hit = any(marked[k] for k in cells_of_node.get(int(m), ()))
                if hit:
                    marked[c] = True
                    grew = True
                    break
        if not grew:
            return marked
    raise RuntimeError("2:1 balance did not converge; the mesh is inconsistent.")


def refine_quads(points: np.ndarray, quads: np.ndarray, marked):
    """Split each marked quadrilateral into four, sharing edge midpoints with its neighbours.

    Returns ``(points, quads)``. The mesh's hanging nodes are read back with :func:`hanging_nodes`,
    which derives them from the mesh rather than from this split -- see there for why.

    Midpoints are keyed by **global edge id**, so the node a split cell creates on a shared edge is
    the same node its neighbour sees -- creating them per cell instead would duplicate the node and
    silently disconnect the two cells while every count still looked right.
    """
    points = np.asarray(points, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    marked = balance_marks(points, quads, marked)
    if not marked.any():
        return points, quads

    topo = _edge_topology(quads)
    ce, ev = np.asarray(topo.cell_edges), np.asarray(topo.edge_vertices)

    # one midpoint per edge touched by a marked cell, created once and shared
    need = np.zeros(topo.n_edges, dtype=bool)
    need[ce[marked].ravel()] = True
    eids = np.where(need)[0]
    mid_of_edge = np.full(topo.n_edges, -1, dtype=np.int64)
    mid_of_edge[eids] = len(points) + np.arange(len(eids))
    new_pts = [points, points[ev[eids]].mean(axis=1)]

    # one centre per marked cell
    m_cells = np.where(marked)[0]
    centre_of = np.full(len(quads), -1, dtype=np.int64)
    centre_of[m_cells] = len(points) + len(eids) + np.arange(len(m_cells))
    new_pts.append(points[quads[m_cells]].mean(axis=1))
    pts_out = np.vstack(new_pts)

    out_cells: List[np.ndarray] = []
    for c in range(len(quads)):
        if not marked[c]:
            out_cells.append(quads[c])
            continue
        v = quads[c]
        m = mid_of_edge[ce[c]]  # midpoints of edges (v0,v1), (v1,v2), (v2,v3), (v3,v0)
        k = centre_of[c]
        # four children, each keeping the parent's counter-clockwise winding
        out_cells += [
            np.array([v[0], m[0], k, m[3]]),
            np.array([m[0], v[1], m[1], k]),
            np.array([k, m[1], v[2], m[2]]),
            np.array([m[3], k, m[2], v[3]]),
        ]

    return pts_out, np.asarray(out_cells, dtype=np.int64)
