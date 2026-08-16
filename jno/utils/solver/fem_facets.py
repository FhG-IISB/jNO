"""Boundary facet connectivity for the native 2D/3D Lagrange assembler.

Identifies boundary faces (faces belonging to exactly one cell) and builds the
``(parent_cell, local_face_index, face_nodes)`` table needed for surface integration.
Also computes outward unit normals and maps reference quadrature to physical faces.

This module is mesh-topology-only (pure NumPy); it is vmappable only once the caller
converts the results to JAX arrays.

Local face conventions
----------------------
*Triangle* (3-node, 2-D): 3 faces, each an edge.

    Face k = local nodes ``(k, (k+1)%3)``, opposite local node ``(k+2)%3``.

    ===================== ====================== ====================
    Face 0                Face 1                 Face 2
    ===================== ====================== ====================
    nodes (0, 1)          nodes (1, 2)           nodes (2, 0)
    opposite vertex 2     opposite vertex 0      opposite vertex 1
    ===================== ====================== ====================

*Tetrahedron* (4-node, 3-D): 4 faces, each a triangle.  Face ``k`` is opposite
local vertex ``k`` (the remaining 3 local vertices in ascending order).
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np

# (face_node, opposite_vertex) for each of the 2 facets of an interval. A facet of a `dim`-simplex has
# `dim` vertices, so in 1D it is a single endpoint, and the "opposite vertex" is the other end.
_LOCAL_FACES_INT: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0))

# (face_node_a, face_node_b, opposite_vertex) for each of 3 triangle faces
_LOCAL_FACES_TRI: Tuple[Tuple[int, int, int], ...] = ((0, 1, 2), (1, 2, 0), (2, 0, 1))

# (face_node_a, face_node_b, face_node_c, opposite_vertex) for each of 4 tet faces.
# Face k is the face opposite local vertex k: nodes are the remaining 3 in sorted order.
_LOCAL_FACES_TET: Tuple[Tuple[int, int, int, int], ...] = (
    (1, 2, 3, 0),  # face 0: opposite vertex 0
    (0, 2, 3, 1),  # face 1: opposite vertex 1
    (0, 1, 3, 2),  # face 2: opposite vertex 2
    (0, 1, 2, 3),  # face 3: opposite vertex 3
)

# ---------------------------------------------------------------------------------------------
# Tensor-product cells. These carry NO trailing apex entry: a simplex facet has exactly one
# opposite vertex, and that is what orients it outward, but a quad edge or a hex face has none.
# Facets of these cells are oriented away from the owning cell's CENTROID instead -- exact for a
# convex cell, which every cell of a structured or recombined mesh is.
#
# Vertex numbering is meshio/VTK, which is what a mesh's `cells` array holds: quad is
# 0(0,0) 1(1,0) 2(1,1) 3(0,1); hex is that bottom face followed by the matching top face. basix
# numbers its reference cells lexicographically instead, so anything that tabulates a BASIS must
# permute first -- these tables are topology, not basis.

# (edge_node_a, edge_node_b) for each of the 4 quadrilateral edges, counterclockwise.
_LOCAL_FACES_QUAD: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))

# The 4 nodes of each of the 6 hexahedron faces, each traversed consistently around its own plane.
_LOCAL_FACES_HEX: Tuple[Tuple[int, int, int, int], ...] = (
    (0, 3, 2, 1),  # z = 0
    (4, 5, 6, 7),  # z = 1
    (0, 1, 5, 4),  # y = 0
    (1, 2, 6, 5),  # x = 1
    (2, 3, 7, 6),  # y = 1
    (3, 0, 4, 7),  # x = 0
)


class FacetConnectivity(NamedTuple):
    """Boundary facet connectivity for a P1 simplex mesh.

    Attributes
    ----------
    parent_cell : ``(n_bfaces,)`` int64 — global cell index of each boundary face.
    local_face  : ``(n_bfaces,)`` int64 — local face index within the parent cell.
    face_nodes  : ``(n_bfaces, n_nodes_per_face)`` int64 — global node ids
                  (P1 corner nodes only; excludes P2 edge-midpoints).
    n_bfaces    : total number of boundary faces.
    """

    parent_cell: np.ndarray  # (n_bfaces,) int64
    local_face: np.ndarray  # (n_bfaces,) int64
    face_nodes: np.ndarray  # (n_bfaces, n_nodes_per_face) int64
    n_bfaces: int


#: Cache of the sort+unique over a mesh's faces, so the domain build and assembly do it ONCE
#: between them. They ask different questions of the same computation -- the domain wants the set
#: of boundary faces, assembly wants each one's parent cell and local index -- and were each
#: paying ~1 s for it on a 424k-tet mesh.
#:
#: Keyed on the CONTENT of the ``cells`` array, not its identity: the domain build and the
#: assembler reach this with equal-but-distinct arrays, so an id-based key measured a 0% hit rate.
#: Hashing 13.5 MB of connectivity costs ~10 ms against the ~1 s it saves, and cannot collide the
#: way a shape/dtype key would.
_FACET_CACHE: dict = {}
_FACET_CACHE_MAX = 8


def pack_face_keys(canonical: np.ndarray):
    """One ``int64`` per already-sorted face row, or ``None`` if the id range would overflow it.

    ``np.unique(..., axis=0)`` sorts a VOID VIEW of the rows, and its argsort is the single most
    expensive thing anyone does with a face table here: 2.32 s (a quarter of a 424k-tet domain
    build) against a fraction of that for a 1-D sort of the same count, and 5.8x on a 3.1M-face 2-D
    mesh. Packing ``row -> ((r0 * n) + r1) * n + r2`` is exact for ids in ``[0, n)``, and sorting the
    packed key is *exactly* lexicographic on the rows -- so a caller's row ORDER, first-occurrence
    indices and inverse are all unchanged, not merely equivalent.

    Every consumer of "which faces occur once" goes through this rather than writing its own
    ``unique``: the version in :mod:`jno._fem` was left behind when this one was packed, and paid
    1.57 s of a 26.7 s build for it.
    """
    if canonical.size == 0:
        return np.zeros(len(canonical), dtype=np.int64)
    n_pts = int(canonical.max()) + 1
    if n_pts ** canonical.shape[1] >= 2**62:
        return None  # caller falls back to the row-wise unique
    keys = np.zeros(len(canonical), dtype=np.int64)
    for j in range(canonical.shape[1]):
        keys = keys * n_pts + canonical[:, j]
    return keys


def _boundary_faces(cells: np.ndarray, local_faces, n_face_nodes: int):
    """``(flat, sel, n_local)``: every (cell, local face) row, and which rows are on the boundary.

    ``flat`` is CELL-MAJOR, so a row index ``r`` decodes as cell ``r // n_local``, local face
    ``r % n_local``. ``sel`` indexes the rows whose canonical (sorted) key occurs exactly once.
    """
    import hashlib

    contiguous = np.ascontiguousarray(cells)
    key = (hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest(), cells.shape, n_face_nodes)
    hit = _FACET_CACHE.get(key)
    if hit is not None:
        return hit[1]

    idx = np.asarray(local_faces, dtype=np.int64)[:, :n_face_nodes]
    n_local = idx.shape[0]
    flat = cells[:, idx].reshape(-1, n_face_nodes).astype(np.int64, copy=False)
    canonical = np.sort(flat, axis=1)

    keys = pack_face_keys(canonical)
    if keys is not None:
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    else:
        _, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    sel = np.flatnonzero(counts[np.asarray(inverse).ravel()] == 1)

    value = (flat, sel, n_local)
    if len(_FACET_CACHE) >= _FACET_CACHE_MAX:
        _FACET_CACHE.pop(next(iter(_FACET_CACHE)))
    _FACET_CACHE[key] = (None, value)
    return value


def boundary_face_set(cells: np.ndarray, cell_type: str = "triangle") -> np.ndarray:
    """The mesh's boundary faces as canonical rows, sorted within a row and lexicographically.

    Same answer as an independent ``np.unique(..., return_counts=True)`` over the faces, but it
    reuses :func:`_boundary_faces`, so a caller that also builds facet connectivity pays once.
    """
    local_faces, n_face_nodes = _face_table(cell_type)
    flat, sel, _ = _boundary_faces(np.asarray(cells), local_faces, n_face_nodes)
    faces = np.sort(flat[sel], axis=1)
    return faces[np.lexsort(faces.T[::-1])] if faces.size else faces


def _face_table(cell_type: str):
    """``(local_faces, n_face_nodes)`` for a cell type, under either naming.

    Simplex tables carry the facet's opposite vertex as a trailing entry; tensor-product tables do
    not (there is no single opposite vertex) -- ask :func:`has_facet_apex` before reading one.
    """
    if cell_type in ("interval", "line"):
        return _LOCAL_FACES_INT, 1
    if cell_type in ("triangle", "tri"):
        return _LOCAL_FACES_TRI, 2
    if cell_type in ("tetrahedron", "tetra", "tet"):
        return _LOCAL_FACES_TET, 3
    if cell_type in ("quad", "quadrilateral"):
        return _LOCAL_FACES_QUAD, 2
    if cell_type in ("hexahedron", "hex"):
        return _LOCAL_FACES_HEX, 4
    raise NotImplementedError(
        f"build_facet_connectivity: cell_type {cell_type!r} not supported "
        "(interval / triangle / tetrahedron / quadrilateral / hexahedron only)."
    )


def local_faces_in_basix_order(cell_type: str):
    """``(local_faces, n_face_nodes)`` for the SAME facets as :func:`_face_table`, but with node ids
    renumbered into basix's vertex order (and the trailing apex entry dropped).

    Two numberings meet here. Mesh topology -- which facets exist, which cell owns each -- is done in
    the mesh's own meshio/VTK order, because that is what a ``cells`` array holds. Anything that
    tabulates a BASIS works in basix's reference cell instead. Facet ``k`` has to mean the same facet
    on both sides, because ``build_facet_connectivity`` hands out a ``local_face`` index that is then
    used to pick a row of the tabulated facet tables; deriving one table from the other by permuting
    node ids keeps that correspondence by construction, where writing a second table by hand would
    only keep it by luck.
    """
    import numpy as _np

    from .fem_lagrange import vtk_to_basix_vertex_perm

    local_faces, n_face_nodes = _face_table(cell_type)
    inv = _np.argsort(vtk_to_basix_vertex_perm(cell_type))  # VTK index -> basix index
    return tuple(tuple(int(inv[v]) for v in face[:n_face_nodes]) for face in local_faces), n_face_nodes


def has_facet_apex(cell_type: str) -> bool:
    """Whether ``cell_type``'s facet table carries an opposite vertex to orient the facet outward.

    True for simplices, False for tensor-product cells. A caller that needs an outward direction
    uses the apex when this is True and the owning cell's centroid when it is False; the centroid
    is exact for any convex cell and agrees with the apex on simplices, so it is a fallback in
    coverage only, not in accuracy.
    """
    return cell_type not in ("quad", "quadrilateral", "hexahedron", "hex")


def build_facet_connectivity(cells: np.ndarray, cell_type: str = "triangle") -> FacetConnectivity:
    """Build boundary facet connectivity for a P1 simplex mesh.

    A face is on the boundary if it belongs to exactly one cell.

    Parameters
    ----------
    cells     : ``(n_cells, n_verts_per_cell)``  global vertex indices (P1 connectivity;
                P2 ``triangle6`` / ``tetra10`` arrays work — only the first 3/4 columns
                are read).
    cell_type : ``"triangle"`` (2-D, 3-node) or ``"tetrahedron"`` (3-D, 4-node).
    """
    cells = np.asarray(cells)
    local_faces, n_face_nodes = _face_table(cell_type)

    # The dict-of-tuples version of this walked cells x local_faces in Python with an ``int()`` per
    # vertex -- 5.1M of them on a 424k-tet mesh, and over 3 s of a 14 s assembly. Shared with the
    # domain build via the cache, so the sort+unique happens once per mesh rather than once here
    # and again in ``boundary_face_set``.
    flat, sel, n_local = _boundary_faces(cells, local_faces, n_face_nodes)

    n_bfaces = int(sel.size)
    empty_face = np.empty((0, n_face_nodes), dtype=np.int64)
    return FacetConnectivity(
        parent_cell=(sel // n_local).astype(np.int64),
        local_face=(sel % n_local).astype(np.int64),
        # the ORIGINAL vertex order of the face, not the sorted key -- orientation is load-bearing
        # for the outward normals computed from it
        face_nodes=flat[sel] if n_bfaces > 0 else empty_face,
        n_bfaces=n_bfaces,
    )


def compute_face_normals(
    points: np.ndarray,
    conn: FacetConnectivity,
    cells: np.ndarray,
    cell_type: str = "triangle",
) -> np.ndarray:
    """Outward unit normals for every boundary face.

    For a 2-D triangle mesh the normal is the 90°-rotated edge tangent, oriented
    away from the opposite cell vertex.  For a 3-D tet mesh it is the cross product
    of two edge vectors, oriented away from the opposite vertex.

    Returns ``(n_bfaces, dim)``.

    One array expression over all faces, not a Python loop over them. The loop form called
    ``np.cross`` / ``np.mean`` / ``np.linalg.norm`` on 3-element vectors once per face, where numpy's
    per-call dispatch dwarfs the arithmetic: 16,730 boundary faces on a 3-D unit cube cost 0.63 s of
    a 5.6 s build, and the cost is per-face, so it grows with every refinement.
    """
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells)
    local_faces, n_face_nodes = _face_table(cell_type)
    dim = {"interval": 1, "line": 1, "triangle": 2, "quad": 2, "quadrilateral": 2}.get(cell_type, 3)

    if conn.n_bfaces == 0:
        return np.zeros((0, dim), dtype=float)

    entry = np.asarray(local_faces, dtype=np.int64)[np.asarray(conn.local_face)]
    parent = cells[np.asarray(conn.parent_cell)]  # (n_bfaces, n_verts_per_cell)
    face_ids = np.take_along_axis(parent, entry[:, :n_face_nodes], axis=1)
    verts = points[face_ids, :dim]  # (n_bfaces, n_face_nodes, dim)

    # The reference point the facet is oriented AWAY from. A simplex facet has an opposite vertex,
    # which is exact for any geometry including a concave boundary; a tensor-product facet has none,
    # so the owning cell's centroid stands in -- exact for a convex cell, and identical to the apex
    # wherever both exist.
    if has_facet_apex(cell_type):
        opp = points[np.take_along_axis(parent, entry[:, n_face_nodes : n_face_nodes + 1], axis=1)[:, 0], :dim]
    else:
        opp = points[parent, :dim].mean(axis=1)

    if dim == 1:  # 1-D: a facet is a point, so there is no tangent to rotate — the unit candidate is
        # +1 and the shared away-from-the-apex flip below picks the outward sign
        n = np.ones((conn.n_bfaces, 1), dtype=float)
    elif dim == 2:  # 2-D: edge → rotate tangent 90° clockwise
        t = verts[:, 1] - verts[:, 0]
        n = np.stack([t[:, 1], -t[:, 0]], axis=1)
    elif n_face_nodes == 4:  # 3-D quadrilateral face: the cross product of its DIAGONALS
        n = np.cross(verts[:, 2] - verts[:, 0], verts[:, 3] - verts[:, 1])
    else:  # 3-D triangle: cross product of two edges
        n = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])

    mid = verts.mean(axis=1)
    flip = np.einsum("ij,ij->i", n, mid - opp) < 0
    n = np.where(flip[:, None], -n, n)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def facet_quad_data(
    points: np.ndarray,
    face_nodes: np.ndarray,
    gp: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Physical quadrature points and Jacobian scale for one boundary facet.

    Maps the reference quadrature ``gp`` to physical coordinates via the affine
    map from the reference face (interval [0,1] for a 2-D edge, reference triangle
    for a 3-D face) to the physical face.

    Parameters
    ----------
    points     : ``(n_nodes, dim)``  global mesh coordinates.
    face_nodes : ``(n_face_nodes,)`` global node indices of the face.
    gp         : ``(n_quad,)`` for a 2-D edge, ``(n_quad, 2)`` for a 3-D face —
                 reference quadrature points.

    Returns
    -------
    xq        : ``(n_quad, dim)``  physical quadrature coordinates.
    jac_scale : scalar — the Jacobian scaling (edge length or face area).
    """
    verts = np.asarray(points)[np.asarray(face_nodes)]  # (n_face_nodes, dim)
    gp = np.asarray(gp)
    n_face_nodes = verts.shape[0]

    if n_face_nodes == 2:  # 2-D edge: x = (1-t) v0 + t v1
        t = gp.reshape(-1, 1)
        xq = (1.0 - t) * verts[0] + t * verts[1]
        jac_scale = float(np.linalg.norm(verts[1] - verts[0]))
    elif n_face_nodes == 3:  # 3-D face: affine map from ref-triangle
        s, t_ref = gp[:, 0:1], gp[:, 1:2]
        xq = (1.0 - s - t_ref) * verts[0] + s * verts[1] + t_ref * verts[2]
        e1, e2 = verts[1] - verts[0], verts[2] - verts[0]
        jac_scale = 0.5 * float(np.linalg.norm(np.cross(e1, e2)))  # triangle area
    else:
        raise ValueError(f"Expected 2 or 3 face nodes, got {n_face_nodes}.")

    return xq, jac_scale
