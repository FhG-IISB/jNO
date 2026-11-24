"""Control volumes and faces of a finite-volume discretisation.

Based on:
  Moukalled, F., Mangani, L., & Darwish, M. (2016). The Finite Volume Method in Computational Fluid
  Dynamics. Springer, Fluid Mechanics and Its Applications 113 -- ch. 6 (geometric quantities: face
  area vectors and centroids by triangulation, cell volumes and centroids by pyramid decomposition).

One :class:`FVGraph` describes any centering. A control volume (a mesh cell, or the median-dual cell of
a node) has a centroid and a volume. An interior face joins an owner and a neighbour volume and carries
its area vector (pointing owner -> neighbour), its centroid and the centroid-to-centroid vector; a
boundary face has an owner and an outward area vector. Everything numerical in ``jno.fvm`` reads this
structure only, so the flux, reconstruction and solver code does not depend on the cell type or on
whether the unknowns sit in cells or at nodes.

The geometry is computed for any polytope from its faces, in any dimension: a face is split into
triangles about its vertex mean (exact for a planar face), and a cell into pyramids with apex at its
vertex mean (exact for a cell with planar faces). Lines, triangles, quadrilaterals, tetrahedra and
hexahedra all go through the same code.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..utils.sets import unique_values

__all__ = ["FVGraph", "cell_graph", "dual_graph", "face_geometry"]


@dataclasses.dataclass(frozen=True)
class FVGraph:
    """Control volumes and faces, as host ``numpy`` arrays.

    Attributes:
        dim: spatial dimension.
        centroid: ``(n, dim)`` control-volume centroids.
        volume: ``(n,)`` control-volume measures (length, area or volume).
        owner, neigh: ``(n_if,)`` the two volumes of each interior face.
        area: ``(n_if, dim)`` interior area vectors, pointing from ``owner`` to ``neigh``; their norm is
            the face measure (1 for a point face in 1-D).
        fcentroid: ``(n_if, dim)`` interior face centroids, on the owner's side of a periodic pair.
        f_nodes: ``(n_if, k)`` mesh node ids of each interior face (owner's side of a periodic pair).
        delta: ``(n_if, dim)`` ``centroid[neigh] - centroid[owner]``; across a periodic pair the
            difference with the neighbour translated by one period.
        b_owner: ``(n_bf,)`` the volume of each boundary face.
        b_area: ``(n_bf, dim)`` outward boundary area vectors.
        b_centroid: ``(n_bf, dim)`` boundary face centroids.
        b_nodes: ``(n_bf, k)`` mesh node ids of each boundary face (how a region tag is resolved).
        points: ``(n_nodes, dim)`` mesh nodes; ``cells``, ``cell_type``: the mesh cells the graph was
            built from.
    """

    dim: int
    centroid: np.ndarray
    volume: np.ndarray
    owner: np.ndarray
    neigh: np.ndarray
    area: np.ndarray
    fcentroid: np.ndarray
    f_nodes: np.ndarray
    delta: np.ndarray
    b_owner: np.ndarray
    b_area: np.ndarray
    b_centroid: np.ndarray
    b_nodes: np.ndarray
    points: np.ndarray
    cells: np.ndarray
    cell_type: str
    centering: str = "cell"  # "vertex": the control volumes are the median-dual cells of the mesh nodes

    @property
    def n(self) -> int:
        """Number of control volumes."""
        return int(self.volume.shape[0])

    def boundary_faces_on(self, nodes) -> np.ndarray:
        """Indices of the boundary faces whose nodes all lie in the node set ``nodes``."""
        inside = np.zeros(self.points.shape[0], dtype=bool)
        inside[np.asarray(nodes, dtype=np.int64)] = True
        return np.flatnonzero(inside[self.b_nodes].all(axis=1))

    def with_periodic(self, faces_a, faces_b, shift) -> "FVGraph":
        """The graph with boundary faces ``faces_a`` and ``faces_b`` joined as interior faces.

        ``shift`` is the translation taking face set ``b`` onto face set ``a``
        (``b_centroid[b] + shift == b_centroid[a]``). Faces are paired by nearest translated centroid,
        and a pairing that does not match to round-off raises: the mesh must be periodic-conforming.
        """
        from scipy.spatial import cKDTree

        fa, fb = np.asarray(faces_a, dtype=np.int64), np.asarray(faces_b, dtype=np.int64)
        if fa.size != fb.size or fa.size == 0:
            raise ValueError(
                f"jno.fvm periodic: the two faces have {fa.size} and {fb.size} mesh faces -- a periodic pair "
                "needs a conforming mesh with matching faces on both sides."
            )
        shift = np.asarray(shift, dtype=float).reshape(1, self.dim)
        dist, match = cKDTree(self.b_centroid[fb] + shift).query(self.b_centroid[fa])
        extent = float(np.ptp(self.points, axis=0).max())
        if unique_values(match).size != fa.size or dist.max() > 1e-8 * extent:
            raise ValueError(
                f"jno.fvm periodic: the faces do not match under the translation {shift.ravel().tolist()} "
                f"(largest mismatch {dist.max():.3g}). Build the mesh periodic-conforming."
            )
        fb = fb[match]
        own, nb = self.b_owner[fa], self.b_owner[fb]
        # the neighbour volume seen from `a` sits one period away: translate its centroid by `shift`
        delta = (self.centroid[nb] + shift) - self.centroid[own]
        keep = np.setdiff1d(np.arange(self.b_owner.shape[0]), np.concatenate([fa, fb]))
        return dataclasses.replace(
            self,
            owner=np.concatenate([self.owner, own]),
            neigh=np.concatenate([self.neigh, nb]),
            area=np.concatenate([self.area, self.b_area[fa]]),
            fcentroid=np.concatenate([self.fcentroid, self.b_centroid[fa]]),
            f_nodes=np.concatenate([self.f_nodes, self.b_nodes[fa]]),
            delta=np.concatenate([self.delta, delta]),
            b_owner=self.b_owner[keep],
            b_area=self.b_area[keep],
            b_centroid=self.b_centroid[keep],
            b_nodes=self.b_nodes[keep],
        )


def face_geometry(verts: np.ndarray):
    """``(area_vector, centroid)`` of faces given their vertices, ``verts`` of shape ``(n, k, dim)``.

    The vertices are in cyclic order around the face, and the orientation of the area vector follows
    that order (callers orient it). In 1-D a face is a point and its area vector is ``+1``; in 2-D a face
    is a segment and the area vector its tangent rotated clockwise; in 3-D a polygon is split into
    triangles about its vertex mean, which is exact for any planar polygon and the usual definition
    otherwise.
    """
    verts = np.asarray(verts, dtype=float)
    n, _k, dim = verts.shape
    if dim == 1:
        return np.ones((n, 1)), verts[:, 0]
    if dim == 2:
        t = verts[:, 1] - verts[:, 0]
        return np.stack([t[:, 1], -t[:, 0]], axis=1), verts.mean(axis=1)
    m = verts.mean(axis=1, keepdims=True)
    nxt = np.roll(verts, -1, axis=1)
    s = 0.5 * np.cross(verts - m, nxt - m)  # (n, k, 3): one sub-triangle per edge
    tri_c = (m + verts + nxt) / 3.0
    w = np.linalg.norm(s, axis=-1)
    centroid = np.einsum("nk,nkd->nd", w, tri_c) / np.maximum(w.sum(axis=1), np.finfo(float).tiny)[:, None]
    return s.sum(axis=1), centroid


def cell_graph(points, cells, cell_type: str) -> FVGraph:
    """The cell-centred :class:`FVGraph` of a mesh: one control volume per cell.

    ``cells`` is ``(n_cells, n_verts)`` in meshio/VTK order and ``cell_type`` one of ``line``,
    ``triangle``, ``quad``, ``tetra``, ``hexahedron``. Vectorised over all faces: one sort and one
    ``np.unique`` find the face pairs.
    """
    from ..utils.solver.fem_facets import _face_table

    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    local, k = _face_table(cell_type)
    dim = points.shape[1]
    local = np.asarray(local, dtype=np.int64)[:, :k]
    n_cells, n_local = cells.shape[0], local.shape[0]

    nodes = cells[:, local]  # (n_cells, n_local, k), cyclic order per face
    s, c = face_geometry(points[nodes.reshape(-1, k)])
    s, c = s.reshape(n_cells, n_local, dim), c.reshape(n_cells, n_local, dim)
    apex = points[cells].mean(axis=1)  # (n_cells, dim): inside every convex cell
    out = np.einsum("cfd,cfd->cf", s, c - apex[:, None, :])
    if np.any(np.abs(out) <= 1e-14 * np.abs(out).max()):
        raise ValueError(f"jno.fvm: a degenerate {cell_type} cell (a face through its vertex mean).")
    s = np.where((out < 0)[..., None], -s, s)  # outward from each cell

    # pyramids from the vertex mean: exact volume and centroid for any cell with planar faces
    vol_f = np.abs(out) / dim
    volume = vol_f.sum(axis=1)
    pyr_c = apex[:, None, :] + dim / (dim + 1.0) * (c - apex[:, None, :])
    centroid = np.einsum("cf,cfd->cd", vol_f, pyr_c) / volume[:, None]

    key = np.sort(nodes.reshape(-1, k), axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    if counts.max() > 2:
        raise ValueError("jno.fvm: a non-manifold mesh (a face shared by more than two cells).")
    order = np.argsort(inv, kind="stable")  # the entries of one face are adjacent, lower cell first
    grouped = inv[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = grouped[1:] != grouped[:-1]
    head = np.flatnonzero(first)
    pair = counts[grouped[head]] == 2
    a, b, bd = order[head[pair]], order[head[pair] + 1], order[head[~pair]]

    flat_s, flat_c, flat_n = s.reshape(-1, dim), c.reshape(-1, dim), nodes.reshape(-1, k)
    owner, neigh = a // n_local, b // n_local
    return FVGraph(
        dim=dim,
        centroid=centroid,
        volume=volume,
        owner=owner,
        neigh=neigh,
        area=flat_s[a],
        fcentroid=flat_c[a],
        f_nodes=flat_n[a],
        delta=centroid[neigh] - centroid[owner],
        b_owner=bd // n_local,
        b_area=flat_s[bd],
        b_centroid=flat_c[bd],
        b_nodes=flat_n[bd],
        points=points,
        cells=cells,
        cell_type=cell_type,
    )


def _cell_edges(cell_type):
    """``[(a, b, faces)]``: the local edges of a cell type and the local faces that contain each."""
    from ..utils.solver.fem_facets import _face_table

    local, k = _face_table(cell_type)
    faces = [tuple(f[:k]) for f in local]
    if k <= 2:  # 1-D and 2-D: the faces are the edges
        return [(f[0], f[-1], (i,)) for i, f in enumerate(faces)] if k == 2 else []
    edges = {}
    for i, f in enumerate(faces):
        for a, b in zip(f, f[1:] + f[:1]):
            edges.setdefault(tuple(sorted((a, b))), []).append(i)
    return [(a, b, tuple(fs)) for (a, b), fs in edges.items()]


def dual_graph(points, cells, cell_type: str) -> FVGraph:
    """The vertex-centred :class:`FVGraph` of a mesh: one control volume per mesh node, the median dual.

    Inside every cell, the dual face between the nodes of an edge runs from the edge midpoint to the cell
    centroid (in 3-D through the centroids of the two cell faces at the edge); the pieces of all cells sharing
    the edge add up to one dual face. A boundary facet is shared equally by its nodes. Volumes: a simplex
    gives each vertex ``1/(d+1)`` of itself (exact for the median dual), a quadrilateral or hexahedron
    ``1/2^d`` (exact for a parallelogram or parallelepiped). The control volume's point is the node itself.
    In 1-D a node's volume is half of each adjacent cell and the dual face is the cell midpoint.

    Based on: Barth, T. J. (1994). Aspects of unstructured grids and finite-volume solvers for the Euler and
    Navier-Stokes equations. VKI Lecture Series 1994-05 -- the median-dual control volume.
    """
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    dim = points.shape[1]
    n_nodes = points.shape[0]
    cg = cell_graph(points, cells, cell_type)  # the boundary facets and cell volumes
    nv = cells.shape[1]
    volume = np.zeros(n_nodes)
    np.add.at(volume, cells.ravel(), np.repeat(cg.volume / nv, nv))
    cent = points[cells].mean(axis=1)

    if dim == 1:
        a, b = cells[:, 0], cells[:, 1]
        S = np.sign(points[b, 0] - points[a, 0])[:, None]
        pieces = [(a, b, S, cent)]
    else:
        from ..utils.solver.fem_facets import _face_table

        local, k = _face_table(cell_type)
        local = np.asarray(local, dtype=np.int64)[:, :k]
        pieces = []
        for la, lb, fs in _cell_edges(cell_type):
            a, b = cells[:, la], cells[:, lb]
            m = 0.5 * (points[a] + points[b])
            if dim == 2:
                t = cent - m
                S = np.stack([t[:, 1], -t[:, 0]], axis=1)
                c = 0.5 * (m + cent)
            else:
                f1 = points[cells[:, local[fs[0]]]].mean(axis=1)
                f2 = points[cells[:, local[fs[1]]]].mean(axis=1)
                S, c = face_geometry(np.stack([m, f1, cent, f2], axis=1))
            flip = np.einsum("fd,fd->f", S, points[b] - points[a]) < 0
            S = np.where(flip[:, None], -S, S)
            pieces.append((a, b, S, c))
    a = np.concatenate([p[0] for p in pieces])
    b = np.concatenate([p[1] for p in pieces])
    S = np.concatenate([p[2] for p in pieces])
    C = np.concatenate([p[3] for p in pieces])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    S = np.where((a > b)[:, None], -S, S)  # every piece from the lower node to the higher one
    key, inv = np.unique(np.stack([lo, hi], axis=1), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    area = np.zeros((key.shape[0], dim))
    np.add.at(area, inv, S)
    w = np.linalg.norm(S, axis=1)
    wsum = np.zeros(key.shape[0])
    np.add.at(wsum, inv, w)
    fc = np.zeros((key.shape[0], dim))
    np.add.at(fc, inv, w[:, None] * C)
    fc = fc / np.maximum(wsum, np.finfo(float).tiny)[:, None]

    # boundary: each facet shared equally by its nodes
    kb = cg.b_nodes.shape[1]
    b_owner = cg.b_nodes.ravel()
    b_area = np.repeat(cg.b_area / kb, kb, axis=0)
    b_centroid = 0.5 * (points[b_owner] + np.repeat(cg.b_centroid, kb, axis=0))
    b_nodes = np.repeat(cg.b_nodes, kb, axis=0)
    return FVGraph(
        dim=dim,
        centroid=points.copy(),
        volume=volume,
        owner=key[:, 0],
        neigh=key[:, 1],
        area=area,
        fcentroid=fc,
        f_nodes=key.copy(),
        delta=points[key[:, 1]] - points[key[:, 0]],
        b_owner=b_owner,
        b_area=b_area,
        b_centroid=b_centroid,
        b_nodes=b_nodes,
        points=points,
        cells=cells,
        cell_type=cell_type,
        centering="vertex",
    )
