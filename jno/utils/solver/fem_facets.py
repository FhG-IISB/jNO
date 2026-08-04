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
    if cell_type == "triangle":
        local_faces = _LOCAL_FACES_TRI
        n_face_nodes = 2
    elif cell_type == "tetrahedron":
        local_faces = _LOCAL_FACES_TET
        n_face_nodes = 3
    else:
        raise NotImplementedError(
            f"build_facet_connectivity: cell_type {cell_type!r} not supported (triangle / tetrahedron only)."
        )

    # Every (cell, local face) in one array, then keep the faces whose canonical (sorted) key occurs
    # exactly once. The dict-of-tuples version of this walked cells x local_faces in Python with an
    # ``int()`` per vertex -- 5.1M of them on a 424k-tet mesh, and over 3 s of a 14 s assembly.
    idx = np.asarray(local_faces, dtype=np.int64)[:, :n_face_nodes]  # (n_local_faces, n_face_nodes)
    n_local = idx.shape[0]
    flat = cells[:, idx].reshape(-1, n_face_nodes).astype(np.int64, copy=False)

    _, inverse, counts = np.unique(np.sort(flat, axis=1), axis=0, return_inverse=True, return_counts=True)
    # flat is cell-major, so a boundary face's row index recovers both its cell and its local face.
    # Selecting in row order also reproduces the old dict-insertion order, since a boundary face
    # occurs exactly once and was therefore inserted at this position.
    sel = np.flatnonzero(counts[np.asarray(inverse).ravel()] == 1)

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
    """
    points = np.asarray(points)
    cells = np.asarray(cells)
    if cell_type == "triangle":
        local_faces = _LOCAL_FACES_TRI
        n_face_nodes, dim = 2, 2
    elif cell_type == "tetrahedron":
        local_faces = _LOCAL_FACES_TET
        n_face_nodes, dim = 3, 3
    else:
        raise NotImplementedError(f"compute_face_normals: cell_type {cell_type!r} not supported.")

    normals = np.empty((conn.n_bfaces, dim))
    for i in range(conn.n_bfaces):
        c = int(conn.parent_cell[i])
        k = int(conn.local_face[i])
        entry = local_faces[k]
        face_ids = [int(cells[c, j]) for j in entry[:n_face_nodes]]
        opp_id = int(cells[c, entry[n_face_nodes]])
        verts = points[face_ids, :dim]  # (n_face_nodes, dim)
        opp = points[opp_id, :dim]

        if dim == 2:  # 2-D: edge → rotate tangent 90° clockwise
            t = verts[1] - verts[0]
            n = np.array([t[1], -t[0]])
        else:  # 3-D: face → cross product of two edges
            n = np.cross(verts[1] - verts[0], verts[2] - verts[0])

        mid = np.mean(verts, axis=0)
        if np.dot(n, mid - opp) < 0:
            n = -n
        normals[i] = n / np.linalg.norm(n)

    return normals


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
