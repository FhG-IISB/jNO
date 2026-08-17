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


def refine_domain(domain, marked, *, copy: bool = False):
    """Refine a quadrilateral domain's marked cells in place, keeping its geometry exactly.

    The counterpart to :func:`~jno.utils.solver.fem_adapt._rebuild_to_size`, and the reason this exists:
    that path re-runs gmsh on the ``Shape`` plan, so it needs a geometry to rebuild from and produces a
    mesh that does not nest inside the old one. Splitting cells needs neither -- it works on a mesh
    loaded from a file, and the old mesh's nodes all survive with their values.

    The boundary is supplied explicitly, from :func:`boundary_edges` rather than from the topological
    "belongs to one cell" rule, which is false on the non-conforming mesh this produces; see there.
    Named boundary sub-regions re-derive from their predicates exactly as they do after a remesh
    (``_capture_geometric_boundary_tags``), and the refinement preserves the boundary geometry, so the
    new perimeter nodes lie on the same curves.

    The hanging nodes are stashed on the domain, where ``jno.fem`` picks them up and turns them into the
    constraint prolongation.
    """
    import meshio

    from .fem_adapt import _apply_new_mesh
    from .fem_native import mesh_cell_type

    dim = int(domain.dimension)
    if mesh_cell_type(domain, dim) != "quad":
        raise NotImplementedError(
            "local (hanging-node) refinement is written for quadrilateral meshes. A simplex mesh adapts "
            "through mmg (`jno.solve.remesh()`), which is local already; hexahedra need the octree "
            "bookkeeping and the face-centre constraints, which are not built yet."
        )
    pts = np.asarray(domain.mesh.points)[:, :dim]
    quads = np.asarray(domain.mesh.cells_dict["quad"])

    new_pts, new_quads = refine_quads(pts, quads, marked)
    hang = hanging_nodes(new_pts, new_quads)
    bedges = boundary_edges(new_pts, new_quads, hang)

    pts3 = np.zeros((len(new_pts), 3), dtype=np.float64)
    pts3[:, :dim] = new_pts
    empty = np.asarray([], dtype=np.int64)
    mesh = meshio.Mesh(
        pts3,
        [("quad", new_quads), ("line", bedges)],
        cell_sets={
            "interior": [np.arange(len(new_quads), dtype=np.int64), empty],
            "boundary": [empty, np.arange(len(bedges), dtype=np.int64)],
        },
    )
    target = _apply_new_mesh(domain, mesh, copy=copy)
    target._fem_hanging_nodes = hang
    target._fem_hanging_cells = new_quads
    return target


def hanging_prolongation(
    points: np.ndarray,
    quads: np.ndarray,
    *,
    vec: int = 1,
    hang: Dict | None = None,
    tied_nodes=None,
) -> Dict[str, object]:
    """The hanging-node constraints as a prolongation ``P``, in the form the solve already consumes.

    A hanging node is constrained to the coarse edge it lies on, ``u_h = sum_i w_i u_parent_i`` -- the
    same relation a periodic tie and a mortar coupling impose. So this hands the pairs to
    :func:`~jno.utils.solver.fem_utils.prolongation_from_ties`, the shared elimination, and inherits
    ``reduce_matrix_periodic`` / ``prolong_periodic`` and the ``B(P)`` block fusion unchanged. Building
    a second constraint mechanism beside that one is how the two drift apart.

    ``tied_nodes`` are nodes already constrained by a periodic or tied interface. A hanging node among
    them would compose two prolongations, and the ORDER of that composition is a decision this does not
    make -- so it is refused by name rather than left to half-work.
    """
    from .fem_utils import prolongation_from_ties

    pts = np.asarray(points, dtype=float)
    hang = hanging_nodes(pts, quads) if hang is None else hang

    if tied_nodes is not None:
        clash = sorted(set(int(n) for n in np.asarray(tied_nodes).reshape(-1)) & set(hang))
        if clash:
            raise NotImplementedError(
                f"{len(clash)} hanging node(s) lie on a tied or periodic interface (e.g. node {clash[0]} at "
                f"{tuple(np.round(pts[clash[0]], 6))}). That composes two prolongations -- the hanging "
                "constraint eliminates the node onto its coarse edge, the tie eliminates it onto the other "
                "interface -- and which comes first changes the answer, so jNO refuses rather than picking "
                "one. Keep the refined region off the tied interface, or refine both sides to match."
            )

    return prolongation_from_ties(
        len(pts),
        {},
        {int(n): [(int(p), float(w)) for p, w in ws] for n, ws in hang.items()},
        vec=vec,
        coupling="hanging",
    )


def boundary_edges(points: np.ndarray, quads: np.ndarray, hang: Dict | None = None) -> np.ndarray:
    """The true perimeter of a possibly **non-conforming** quad mesh, as ``(n_edges, 2)`` node pairs.

    jNO derives the boundary topologically -- a facet belonging to exactly one cell -- and that rule is
    *false* here. Across a 2:1 interface the coarse cell's full edge ``(a, b)`` belongs to one cell, and
    so does each of the neighbour's half-edges ``(a, m)`` and ``(m, b)``: three edges, each occurring
    once, none of them on the perimeter. Measured on a 4x4 grid with one refined corner, that rule
    returns 32 edges of which only 20 are real -- and the other 12, being handed to the solve as
    ``boundary``, get pinned as Dirichlet. That is silent: the first end-to-end hanging solve came back
    with the interior pinned to zero, with nothing naming the interface.

    The correct rule is *coverage*, not multiplicity: a once-occurring edge is interior when the rest of
    the mesh covers it. With a 2:1 balance in force the hanging nodes say exactly where that happens, so
    the three edges above are recognised by their relationship to a hanging node ``m`` and dropped --
    both the coarse edge that ``m`` splits and the two half-edges that split it.

    Quadrilaterals only. A hexahedral face additionally hangs on a face CENTRE and covers four
    sub-faces, which is the same argument with more bookkeeping and is not written yet.
    """
    pts = np.asarray(points, dtype=float)
    quads = np.asarray(quads, dtype=np.int64)
    hang = hanging_nodes(pts, quads) if hang is None else hang

    counts: Dict[Tuple[int, int], int] = {}
    for cell in quads:
        for a_loc, b_loc in QUAD_EDGES:
            a, b = int(cell[a_loc]), int(cell[b_loc])
            k = (a, b) if a < b else (b, a)
            counts[k] = counts.get(k, 0) + 1

    once = np.array([list(k) for k, n in counts.items() if n == 1], dtype=np.int64).reshape(-1, 2)
    return drop_covered_facets(once, hang)


def covered_facet_keys(hang: Dict) -> set:
    """The facets a 2:1 interface makes interior, as sorted node-pair keys.

    Three per hanging node ``m``: the coarse edge ``(a, b)`` that ``m`` splits, and the two half-edges
    ``(a, m)`` and ``(m, b)`` that split it. Each belongs to exactly one cell, so each looks like a
    boundary facet, and none of them is one.
    """
    out = set()
    for m, parents in hang.items():
        (a, _), (b, _) = parents
        for i, j in ((a, b), (a, int(m)), (int(m), b)):
            out.add((i, j) if i < j else (j, i))
    return out


def drop_covered_facets(facets: np.ndarray, hang: Dict) -> np.ndarray:
    """Remove the 2:1-interface facets from a topologically-derived boundary facet array.

    The topological rule -- "a facet belonging to exactly one cell" -- is what every boundary consumer
    in jNO is built on, and it is false on a non-conforming mesh. Rather than change that rule
    everywhere, this subtracts the facets a hanging node proves are covered, so a caller that already
    has ``bf`` gets the true boundary by filtering rather than by re-deriving it.

    Reads the first two columns, which are the vertices of an edge facet under both P1 and P2 (the
    higher-order nodes follow), and returns the rows unchanged.
    """
    facets = np.asarray(facets, dtype=np.int64)
    if not hang or facets.size == 0:
        return facets
    covered = covered_facet_keys(hang)
    keep = np.array(
        [(min(int(r[0]), int(r[1])), max(int(r[0]), int(r[1]))) not in covered for r in facets],
        dtype=bool,
    )
    return facets[keep]


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

    # An edge's midpoint may ALREADY be a node: it is the hanging node a previous round left there, and
    # refining this cell is exactly what makes it a regular one. Edge topology cannot see that -- once
    # the mesh is non-conforming the coarse edge and its neighbour's half-edges are different edges, so
    # the shared-by-edge-id rule stops sharing -- and creating a second node at the same coordinate
    # splits the mesh in two along the interface. Silent: the area, the winding and the 2:1 balance all
    # still check out. Measured before this lookup, refining the same region four times drove the
    # -Lap u = 1 centre value from 0.0751 (right) to 0.0180 against a reference 0.0737.
    lut, q = _node_lookup(points)
    mids = points[ev[eids]].mean(axis=1)
    existing = np.array([lut.get(tuple(k), -1) for k in np.round(mids / q).astype(np.int64)], dtype=np.int64)
    fresh = existing < 0
    mid_of_edge[eids[~fresh]] = existing[~fresh]
    mid_of_edge[eids[fresh]] = len(points) + np.arange(int(fresh.sum()))
    new_pts = [points, mids[fresh]]

    # one centre per marked cell
    m_cells = np.where(marked)[0]
    centre_of = np.full(len(quads), -1, dtype=np.int64)
    centre_of[m_cells] = len(points) + int(fresh.sum()) + np.arange(len(m_cells))
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
