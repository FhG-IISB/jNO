"""Global DOF maps for basix elements of ANY degree on simplices -- entity numbering + orientation.

The lowest-order edge elements need only one thing beyond a nodal map: a ``±1`` sign per (cell, edge)
so the two cells sharing an edge agree on its direction (:mod:`fem_topology`). From degree 2 on that is
no longer enough. An edge carries ``k`` DOFs, and reversing the edge does not just flip their sign: it
mixes them. A tetrahedron face carries DOFs too, and its six possible orientations (three rotations,
each optionally reflected) act on them by matrices that are generally not permutations, and not
orthogonal. basix defines those matrices as the element's *base transformations*
(Scroggs, Dokken, Richardson & Wells, "Construction of arbitrary order finite element degree-of-freedom
maps on polygonal and polyhedral cell meshes", ACM Trans. Math. Softw. 48(2), 2022, §3-§4). This module
applies them, unchanged, per cell:

* every mesh entity (vertex, edge, face, cell) gets a global id; DOFs are numbered entity by entity,
  ``[vertex DOFs | edge DOFs | face DOFs | interior DOFs]``, each entity's DOFs contiguous;
* every (cell, local entity) gets an *orientation*: edge reflected iff its first local vertex has the
  larger global index; face rotations/reflection from the position of its lowest global vertex and the
  order of its two neighbours (the DOLFINx ``cell_info`` rule);
* the physical basis on a cell is ``φ = B ψ̂`` with ``ψ̂`` the reference basis and
  ``B = T(cell_info)⁻¹``, ``T`` what ``basix.FiniteElement.T_apply`` applies. ``B`` is block-diagonal,
  one block per entity; the blocks are read off basix once per (entity, orientation), so there is no
  hand-derived transformation anywhere in jNO.

The choice ``B = T⁻¹`` (not ``T`` or ``Tᵀ``) is the one that makes the global space conforming with the
``cell_info`` rule above. It was established by direct measurement, not by reading: two cells sharing a
facet, random global vertex labels, the tangential (N1E/N2E), normal (RT) or full (Lagrange) trace
compared from both sides. ``T⁻¹`` matched to 1e-11 for N1E/RT/N2E k = 1..4 and Lagrange P3/P4 on
triangles and tetrahedra; ``T`` and ``Tᵀ`` failed by O(1)-O(100) on tetrahedra.
``tests/test_fem_nedelec_general.py`` keeps that measurement as a regression test.

Interpolation uses the same blocks: a function ``u = Σ c_i φ_i`` has reference DOF values
``ℓ̂(u) = Bᵀ c``, so ``c = B⁻ᵀ ℓ̂(u)``, entity by entity. Essential traces, periodic ties and the AMS
transfer operators are all built that way (:meth:`DofMap.entity_interpolate`).

When every block is ``1 × 1`` (degree-1 N1E/RT, P1/P2 Lagrange), ``B`` is diagonal with entries ``±1``
and the map exposes it as ``signs``: callers keep the sign path they always had, which is what keeps the
degree-1 operators bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# family name in jNO -> (basix family, map type tag). The map tag decides the pull-back used by
# interpolation: covariant Piola (H(curl)), contravariant Piola (H(div)) or identity (H1 / L2).
_FAMILIES = {
    "N1E": ("N1E", "covariant"),
    "N2E": ("N2E", "covariant"),
    "RT": ("RT", "contravariant"),
    "BDM": ("BDM", "contravariant"),
    "Lagrange": ("P", "identity"),
    "DG": ("P", "identity"),  # discontinuous Lagrange: every DOF interior to its cell
}


def basix_element(family: str, cell: str, degree: int):
    """The basix element jNO uses for ``family`` (jNO name) on ``cell`` (``"triangle"``/``"tetrahedron"``).

    Degree 1 calls basix with its defaults, exactly as the lowest-order factories always did, so the
    tabulation is bit-identical. Higher degrees need a variant: Lagrange takes GLL-warped points (well
    conditioned at high order, and identical to equispaced up to degree 2); the vector families take the
    Legendre variant (orthonormal integral moments -- the variant DOLFINx defaults to)."""
    import basix
    from basix import CellType, ElementFamily, LagrangeVariant

    if family not in _FAMILIES:
        raise ValueError(f"basix_element: unknown family {family!r}; expected one of {sorted(_FAMILIES)}.")
    fam = getattr(ElementFamily, _FAMILIES[family][0])
    ct = getattr(CellType, cell)
    degree = int(degree)
    if family == "DG":
        if degree < 0:
            raise ValueError(f"basix_element: DG degree must be >= 0, got {degree}.")
        if degree <= 2:
            return basix.create_element(fam, ct, degree, discontinuous=True)
        return basix.create_element(fam, ct, degree, LagrangeVariant.gll_warped, discontinuous=True)
    if degree < 1:
        raise ValueError(f"basix_element: degree must be >= 1, got {degree} for {family}.")
    if degree == 1:
        return basix.create_element(fam, ct, 1)
    if family == "Lagrange":
        return basix.create_element(fam, ct, degree, LagrangeVariant.gll_warped)
    return basix.create_element(fam, ct, degree, LagrangeVariant.legendre)


def map_type(family: str) -> str:
    return _FAMILIES[family][1]


# --------------------------------------------------------------------------------------------------
# orientation (DOLFINx cell_info) and the per-entity transformation blocks
# --------------------------------------------------------------------------------------------------


def _reference_topology(cell: str):
    import basix
    from basix import CellType

    return basix.topology(getattr(CellType, cell))


def cell_orientations(cells: np.ndarray, cell: str) -> Dict[int, np.ndarray]:
    """Per-(cell, local entity) orientation index, vectorised over cells.

    ``{1: (n_cells, n_edges) int8 in {0,1}}`` (1 = reflected: local first vertex has the larger global
    index) and, on a tetrahedron, ``{2: (n_cells, 4) int8 in 0..5}`` with ``index = 2*rotations +
    reflection`` -- rotations = the local position of the face's lowest global vertex, reflection =
    whether the vertex after it (cyclically) is larger than the one before it. Exactly DOLFINx's rule
    (``mesh/permutationcomputation.cpp``)."""
    topo = _reference_topology(cell)
    cells = np.asarray(cells, dtype=np.int64)
    out: Dict[int, np.ndarray] = {}
    le = np.asarray(topo[1], dtype=np.int64)
    out[1] = (cells[:, le[:, 0]] > cells[:, le[:, 1]]).astype(np.int8)
    if len(topo) == 4:  # tetrahedron: triangular faces
        lf = np.asarray(topo[2], dtype=np.int64)  # (4, 3)
        ev = cells[:, lf]  # (n_cells, 4, 3) global vertices in the face's local order
        rots = np.argmin(ev, axis=2)  # (n_cells, 4)
        idx = np.arange(3)
        pre = np.take_along_axis(ev, ((rots - 1) % 3)[..., None], axis=2)[..., 0]
        post = np.take_along_axis(ev, ((rots + 1) % 3)[..., None], axis=2)[..., 0]
        refl = (post > pre).astype(np.int64)
        del idx
        out[2] = (2 * rots + refl).astype(np.int8)
    return out


def _cell_info_bits(cell: str, dim: int, k: int, orient: int) -> int:
    """The basix ``cell_info`` integer that orients ONLY local entity ``(dim, k)`` by ``orient``."""
    topo = _reference_topology(cell)
    tdim = len(topo) - 1
    if tdim == 2:
        if dim != 1:
            raise ValueError("triangle: only edges carry an orientation")
        return int(orient) << k
    n_faces = len(topo[2])
    if dim == 1:
        return int(orient) << (3 * n_faces + k)
    rots, refl = divmod(int(orient), 2)
    return (refl << (3 * k)) | (rots << (3 * k + 1))


def _apply_T(elem, info: int) -> np.ndarray:
    n = int(elem.dim)
    data = np.eye(n).reshape(-1).copy()
    elem.T_apply(data, n, info)
    return data.reshape(n, n)


def entity_blocks(elem, cell: str) -> Dict[int, np.ndarray]:
    """``{dim: (n_local_entities, n_orient, nd, nd)}`` blocks of ``B = T⁻¹`` for every oriented entity
    dimension with DOFs, read off ``T_apply`` one entity and orientation at a time.

    The off-block part of each ``T`` is checked to be the identity (the block-diagonal structure the
    per-cell assembly relies on) -- a basix that ever coupled two entities would fail here, loudly."""
    topo = _reference_topology(cell)
    tdim = len(topo) - 1
    n_orient = {1: 2, 2: 6}
    out: Dict[int, np.ndarray] = {}
    for dim in range(1, tdim):
        ed = elem.entity_dofs[dim]
        nd = len(ed[0])
        if nd == 0:
            continue
        blocks = np.zeros((len(ed), n_orient[dim], nd, nd))
        for k, idx in enumerate(ed):
            idx = np.asarray(idx, dtype=np.int64)
            for o in range(n_orient[dim]):
                T = _apply_T(elem, _cell_info_bits(cell, dim, k, o))
                off = T.copy()
                off[np.ix_(idx, idx)] = np.eye(nd)
                if np.abs(off - np.eye(elem.dim)).max() > 1e-12:
                    raise RuntimeError(
                        f"basix base transformation for entity ({dim}, {k}) orientation {o} is not "
                        "block-diagonal; jNO's per-entity DOF map cannot represent it."
                    )
                blocks[k, o] = np.linalg.inv(T[np.ix_(idx, idx)])
        out[dim] = blocks
    return out


# --------------------------------------------------------------------------------------------------
# global entity numbering
# --------------------------------------------------------------------------------------------------


def _number_entities(cells: np.ndarray, local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Global ids for the sub-entities ``local`` (``(n_local, n_v)`` local vertex lists) of ``cells``.

    Ids are assigned in FIRST-ENCOUNTER order (cells, then local entities) -- the order
    :func:`fem_topology.build_edge_topology` uses, so the degree-1 edge numbering is unchanged.
    Returns ``(cell_entities (n_cells, n_local), entity_vertices (n_entities, n_v) sorted)``."""
    cells = np.asarray(cells, dtype=np.int64)
    verts = np.sort(cells[:, local], axis=2)  # (n_cells, n_local, n_v)
    flat = verts.reshape(-1, verts.shape[-1])
    nv = flat.shape[1]
    n = int(cells.max()) + 1 if cells.size else 1
    if float(n) ** nv < 2.0**62:
        # pack the sorted vertex tuple into one int64 (mixed radix n): a 1-D unique is ~10x faster than
        # np.unique(axis=0), which sorts rows lexicographically through a structured view
        key = np.zeros(flat.shape[0], dtype=np.int64)
        for j in range(nv):
            key = key * np.int64(n) + flat[:, j]
        _u, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        uniq = flat[first]
    else:  # too many vertices to pack the tuple: the row-wise unique
        uniq, first, inverse = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    relabel = np.empty(len(uniq), dtype=np.int64)
    relabel[order] = np.arange(len(uniq), dtype=np.int64)
    return relabel[np.asarray(inverse).reshape(-1)].reshape(verts.shape[:2]), uniq[order]


@dataclass
class DofMap:
    """Global DOF map of one basix element (any degree) on a simplex mesh.

    ``cell_dofs[c, i]`` is the block-local global DOF of reference DOF ``i`` on cell ``c``. Global DOFs
    are laid out entity-major, ``[vertices | edges | faces | cells]``; inside each dimension, entity ``e``
    owns ``entity_offset[dim] + e*nd + j`` for its ``nd`` DOFs. Vertex ids are the MESH vertex indices
    (``n_verts`` = all mesh points, used or not), which is what keeps a P1 field's DOF = its vertex.

    ``blocks[dim]`` and ``orient[dim]`` give the per-cell basis transform ``B`` (see the module doc);
    ``signs`` is set instead when ``B`` is diagonal ``±1`` for every cell."""

    family: str
    degree: int
    cell: str  # "triangle" | "tetrahedron"
    tdim: int
    n_dofs: int
    ndof_local: int
    cell_dofs: np.ndarray
    entity_dofs: list  # basix entity_dofs (local DOF ids per (dim, local entity))
    n_entity_dofs: Tuple[int, ...]  # DOFs per entity, by dim
    entity_offset: Tuple[int, ...]  # first global DOF of each dim
    n_entities: Tuple[int, ...]
    cell_entities: List[np.ndarray]  # per dim: (n_cells, n_local) global entity ids (dim 0: vertices, dim tdim: cell)
    entity_vertices: List[np.ndarray]  # per dim: (n_entities, dim+1) sorted global vertices
    orient: Dict[int, np.ndarray]
    blocks: Dict[int, np.ndarray]
    signs: Optional[np.ndarray] = None  # (n_cells, ndof_local) when B is diagonal ±1
    element: object = field(default=None, repr=False)

    # ---- pickling ------------------------------------------------------------------------------
    # A non-nodal build leaves its maps on the domain (`_fem_nonnodal_topology`), and `element` is a
    # basix C++ object that does not pickle -- so `jno.save` of any domain a degree-k form had been
    # built on failed with "cannot pickle 'basix._basixcpp.FiniteElement_float64'". The element is a
    # pure function of (family, cell, degree): drop it on the way out, rebuild it on the way in. The
    # cached device tables `_jt` are rebuilt on first use as well.
    def __getstate__(self):
        state = dict(self.__dict__)
        state["element"] = None
        state.pop("_jt", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.element is None:
            self.element = basix_element(self.family, self.cell, self.degree)

    # ---- per-cell transform -------------------------------------------------------------------
    @property
    def is_diagonal(self) -> bool:
        return self.signs is not None

    def jax_tables(self):
        """Device tables for :meth:`cell_transform` (built once; cached on the map)."""
        if getattr(self, "_jt", None) is None:
            import jax.numpy as jnp

            tabs = []
            for dim, blk in sorted(self.blocks.items()):
                ed = np.asarray(self.entity_dofs[dim], dtype=np.int64)  # (n_local, nd)
                rows = np.broadcast_to(ed[:, :, None], blk.shape[:1] + blk.shape[2:])
                cols = np.broadcast_to(ed[:, None, :], blk.shape[:1] + blk.shape[2:])
                tabs.append(
                    (
                        jnp.asarray(blk),
                        jnp.asarray(self.orient[dim].astype(np.int32)),
                        jnp.asarray(rows.astype(np.int32)),
                        jnp.asarray(cols.astype(np.int32)),
                    )
                )
            self._jt = tuple(tabs)
        return self._jt

    def cell_transform(self, c):
        """``B`` for cell ``c`` (traced-index friendly), ``(ndof_local, ndof_local)``. Call
        :meth:`jax_tables` once OUTSIDE any trace first (the assembler does); inside a kernel prefer
        :func:`transform_from_tables` on tables held in the closure, which ``elem_map`` lifts."""
        return transform_from_tables(self.jax_tables(), self.ndof_local, c)

    def cell_transform_np(self, c: int) -> np.ndarray:
        B = np.eye(self.ndof_local)
        for dim, blk in self.blocks.items():
            for k, idx in enumerate(self.entity_dofs[dim]):
                B[np.ix_(idx, idx)] = blk[k, int(self.orient[dim][c, k])]
        return B

    def entity_block(self, c: int, dim: int, k: int) -> np.ndarray:
        """``B`` restricted to local entity ``(dim, k)`` of cell ``c`` (identity for unoriented ones)."""
        nd = len(self.entity_dofs[dim][k])
        if dim in self.blocks:
            return self.blocks[dim][k, int(self.orient[dim][c, k])]
        return np.eye(nd)

    def dof_entity_centroids(self, points: np.ndarray) -> np.ndarray:
        """``(n_dofs, gdim)``: the centroid of the mesh entity each global DOF belongs to (a vertex, edge
        midpoint, face or cell centroid) -- where a DOF "lives", e.g. to select the DOFs on a plane."""
        pts = np.asarray(points)
        out = np.zeros((self.n_dofs, pts.shape[1]))
        for d in range(self.tdim + 1):
            nd = self.n_entity_dofs[d]
            if nd == 0 or self.n_entities[d] == 0:
                continue
            cen = pts[self.entity_vertices[d]].mean(axis=1)  # (n_entities, gdim)
            out[self.entity_offset[d] : self.entity_offset[d] + self.n_entities[d] * nd] = np.repeat(cen, nd, axis=0)
        return out

    def entity_global_dofs(self, dim: int, e: int) -> np.ndarray:
        nd = self.n_entity_dofs[dim]
        return self.entity_offset[dim] + e * nd + np.arange(nd)

    # ---- interpolation ------------------------------------------------------------------------
    def entity_interpolate(self, c: int, dim: int, k: int, J: np.ndarray, x0: np.ndarray, fn) -> np.ndarray:
        """Global DOF values of local entity ``(dim, k)`` of cell ``c`` for the physical field ``fn``.

        ``fn(X) -> (n_pts, value_size)`` at physical points ``X``. Applies the entity's reference
        functionals (basix ``x``/``M``) to the pulled-back field and then ``B⁻ᵀ`` for the orientation.
        ``J`` is the cell's affine Jacobian, ``x0`` its first vertex."""
        elem = self.element
        Xr = np.asarray(elem.x[dim][k])  # (npts, tdim) reference points on the entity
        Mk = np.asarray(elem.M[dim][k])  # (nd, vs, npts, nderivs)
        if Xr.shape[0] == 0:
            return np.zeros((0,))
        X = x0[None, :] + Xr @ J.T
        vals = np.asarray(fn(X)).reshape(Xr.shape[0], -1)
        mt = map_type(self.family)
        if mt == "covariant":
            ref = vals @ J  # û = Jᵀ u  (row form)
        elif mt == "contravariant":
            ref = np.linalg.det(J) * vals @ np.linalg.inv(J).T  # û = det J · J⁻¹ u
        else:
            ref = vals
        lref = np.einsum("dvp,pv->d", Mk[..., 0], ref)
        Bk = self.entity_block(c, dim, k)
        return np.linalg.solve(Bk.T, lref)


def transform_from_tables(tables, ndof_local: int, c):
    """``B`` of cell ``c`` from :meth:`DofMap.jax_tables` -- a plain tuple of device arrays, so a kernel
    that closes over it has its mesh-sized ``orient`` arrays LIFTED by ``elem_map`` instead of baked."""
    import jax.numpy as jnp

    B = jnp.eye(ndof_local)
    for blk, orient, rows, cols in tables:
        nl = blk.shape[0]
        sel = blk[jnp.arange(nl), orient[c]]  # (n_local, nd, nd)
        B = B.at[rows, cols].set(sel)
    return B


def build_dofmap(cells: np.ndarray, family: str, degree: int, n_verts: Optional[int] = None, edges=None) -> DofMap:
    """Build the :class:`DofMap` of ``family``/``degree`` on the simplex ``cells`` (``(n_cells, 3|4)``).

    ``edges`` optionally passes an already-built ``(cell_edges, edge_vertices)`` numbering in basix edge
    order (:func:`fem_topology.build_edge_topology` -- the same first-encounter numbering), so an
    assembler that has one does not number the edges twice."""
    cells = np.asarray(cells, dtype=np.int64)
    tdim = cells.shape[1] - 1
    cell = "tetrahedron" if tdim == 3 else "triangle"
    if tdim not in (2, 3):
        raise ValueError(f"build_dofmap: simplex cells of 3 or 4 vertices expected, got {cells.shape[1]}.")
    elem = basix_element(family, cell, degree)
    topo = _reference_topology(cell)
    n_cells = cells.shape[0]
    n_verts = int(cells.max()) + 1 if n_verts is None else int(n_verts)

    cell_entities: List[np.ndarray] = [cells]
    entity_vertices: List[np.ndarray] = [np.arange(n_verts, dtype=np.int64)[:, None]]
    for dim in range(1, tdim):
        if dim == 1 and edges is not None:
            ce, ev = np.asarray(edges[0], dtype=np.int64), np.asarray(edges[1], dtype=np.int64)
            if ce.shape != (n_cells, len(topo[1])):
                raise ValueError(f"build_dofmap: edges= has shape {ce.shape}, expected {(n_cells, len(topo[1]))}.")
            cell_entities.append(ce)
            entity_vertices.append(ev)
            continue
        ce, ev = _number_entities(cells, np.asarray(topo[dim], dtype=np.int64))
        cell_entities.append(ce)
        entity_vertices.append(ev)
    cell_entities.append(np.arange(n_cells, dtype=np.int64)[:, None])
    entity_vertices.append(np.sort(cells, axis=1))
    n_entities = tuple(int(ev.shape[0]) for ev in entity_vertices)

    ned = tuple(int(len(elem.entity_dofs[d][0])) for d in range(tdim + 1))
    offs, acc = [], 0
    for d in range(tdim + 1):
        offs.append(acc)
        acc += n_entities[d] * ned[d]
    cell_dofs = np.zeros((n_cells, elem.dim), dtype=np.int64)
    for d in range(tdim + 1):
        for k, idx in enumerate(elem.entity_dofs[d]):
            for j, ldof in enumerate(idx):
                cell_dofs[:, ldof] = offs[d] + cell_entities[d][:, k] * ned[d] + j

    orient = cell_orientations(cells, cell)
    blocks = entity_blocks(elem, cell)
    orient = {d: o for d, o in orient.items() if d in blocks}
    signs = None
    if all(b.shape[2] == 1 for b in blocks.values()) and all(
        np.allclose(np.abs(b[..., 0, 0]), 1.0) for b in blocks.values()
    ):
        signs = np.ones((n_cells, elem.dim))
        for d, blk in blocks.items():
            for k, idx in enumerate(elem.entity_dofs[d]):
                # np.sign: basix's RT reflection is -1 only to rounding (measured -1 - 2e-16), and the
                # sign path promises EXACT ±1 -- that is what keeps degree 1 bit-identical.
                signs[:, idx[0]] = np.sign(blk[k, :, 0, 0])[orient[d][:, k].astype(np.int64)]
            blocks[d] = np.sign(blk)
    return DofMap(
        family=family,
        degree=int(degree),
        cell=cell,
        tdim=tdim,
        n_dofs=int(acc),
        ndof_local=int(elem.dim),
        cell_dofs=cell_dofs,
        entity_dofs=[[list(map(int, i)) for i in lvl] for lvl in elem.entity_dofs],
        n_entity_dofs=ned,
        entity_offset=tuple(offs),
        n_entities=n_entities,
        cell_entities=cell_entities,
        entity_vertices=entity_vertices,
        orient=orient,
        blocks=blocks,
        signs=signs,
        element=elem,
    )


def cell_jacobians(points: np.ndarray, cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Affine Jacobians ``J[c] = [v1-v0, v2-v0(, v3-v0)]`` and first vertices, host numpy."""
    V = np.asarray(points)[np.asarray(cells)]
    J = np.stack([V[:, k] - V[:, 0] for k in range(1, V.shape[1])], axis=2)
    return J, V[:, 0]


def local_entity_index(cell_entities_dim: np.ndarray) -> Dict[int, Tuple[int, int]]:
    """``{global entity: (cell, local index)}`` for the FIRST cell that contains each entity."""
    out: Dict[int, Tuple[int, int]] = {}
    n_cells, n_local = cell_entities_dim.shape
    flat = cell_entities_dim.reshape(-1)
    uniq, first = np.unique(flat, return_index=True)
    for e, f in zip(uniq.tolist(), first.tolist()):
        out[int(e)] = divmod(int(f), n_local)
    return out


def closure_entities(dofmap: DofMap, c: int, dim: int, k: int) -> List[Tuple[int, int]]:
    """Local sub-entities ``(d, j)`` (``1 <= d <= dim``) in the closure of local entity ``(dim, k)`` of a
    cell -- the entities whose DOFs a trace on ``(dim, k)`` depends on (vertices excluded for H(curl)/H(div);
    included for Lagrange via :func:`closure_entities_with_vertices`)."""
    topo = _reference_topology(dofmap.cell)
    verts = set(topo[dim][k])
    out = []
    for d in range(1, dim + 1):
        for j, vs in enumerate(topo[d]):
            if set(vs) <= verts:
                out.append((d, j))
    return out


def closure_entities_with_vertices(dofmap: DofMap, dim: int, k: int) -> List[Tuple[int, int]]:
    topo = _reference_topology(dofmap.cell)
    verts = set(topo[dim][k])
    out = [(0, v) for v in sorted(verts)]
    for d in range(1, dim + 1):
        for j, vs in enumerate(topo[d]):
            if set(vs) <= verts:
                out.append((d, j))
    return out


# --------------------------------------------------------------------------------------------------
# facets, regions and trace interpolation (essential BCs, periodic ties)
# --------------------------------------------------------------------------------------------------


def facet_owners(dm: DofMap) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(count, cell, local)`` per global facet (entity of dim ``tdim-1``): how many cells share it and
    the first cell / local facet index that contains it. A facet with ``count == 1`` is a boundary facet."""
    fd = dm.tdim - 1
    ce = dm.cell_entities[fd]
    n_f = dm.n_entities[fd]
    flat = ce.reshape(-1)
    count = np.bincount(flat, minlength=n_f)
    first = np.full(n_f, -1, dtype=np.int64)
    order = np.arange(flat.size - 1, -1, -1)
    first[flat[order]] = order  # the smallest position wins (written last)
    cell, local = np.divmod(first, ce.shape[1])
    return count, cell, local


def facet_outward_normals(dm: DofMap, points: np.ndarray, cells: np.ndarray, cell: np.ndarray, local: np.ndarray):
    """Unit normals of local facets ``local`` of ``cell``, pointing OUT of that cell. Host numpy."""
    topo = _reference_topology(dm.cell)
    fd = dm.tdim - 1
    pts = np.asarray(points)
    lf = np.asarray(topo[fd], dtype=np.int64)  # (n_local_facets, tdim)
    cv = np.asarray(cells)[cell]  # (n, tdim+1)
    fv = pts[np.take_along_axis(cv, lf[local], axis=1)]  # (n, tdim, gdim)
    if dm.tdim == 3:
        n = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    else:
        t = fv[:, 1] - fv[:, 0]
        n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    opp = (6 if dm.tdim == 3 else 3) - lf[local].sum(axis=1)  # the local vertex NOT on the facet
    apex = pts[np.take_along_axis(cv, opp[:, None], axis=1)[:, 0]]
    flip = np.einsum("nd,nd->n", n, fv[:, 0] - apex) < 0
    n[flip] *= -1.0
    return n


def region_trace_entities(dm: DofMap, region_mask: np.ndarray, *, boundary_only: bool, cells_too: bool = False):
    """The (dim, global entity) -> (cell, local entity, facet) tasks whose DOFs an essential trace on a
    region constrains.

    ``region_mask[v]`` marks the region's mesh vertices. Facets qualify when all their vertices are in
    the region and, with ``boundary_only``, they are boundary facets (one cell) -- the 3-D-correct
    criterion: an interior edge can join two boundary vertices through the volume, so "both endpoints in
    the region" alone is wrong. Every entity in the closure of a qualifying facet is constrained (vertex
    entities only when the element has vertex DOFs). ``cells_too`` adds the whole closure of every cell
    lying entirely in the region (a VOLUME region, e.g. a conductor held at a potential). Returns a dict
    ``{(dim, e): (cell, local_index, facet_or_-1)}``, first cell wins."""
    topo = _reference_topology(dm.cell)
    fd = dm.tdim - 1
    count, fcell, flocal = facet_owners(dm)
    fin = region_mask[dm.entity_vertices[fd]].all(axis=1)
    cand = fin & (count == 1) if boundary_only else fin
    tasks: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    has_vertex_dofs = dm.n_entity_dofs[0] > 0
    lo_dim = 0 if has_vertex_dofs else 1
    sub = {}
    for kf in range(len(topo[fd])):
        vs = set(topo[fd][kf])
        sub[kf] = [(d, j) for d in range(lo_dim, fd + 1) for j, ev in enumerate(topo[d]) if set(ev) <= vs]
    for f in np.flatnonzero(cand).tolist():
        c, kf = int(fcell[f]), int(flocal[f])
        for d, j in sub[kf]:
            key = (d, int(dm.cell_entities[d][c, j]))
            if key not in tasks:
                tasks[key] = (c, j, f)
    if cells_too:
        full = np.flatnonzero(region_mask[dm.cell_entities[0]].all(axis=1))
        for c in full.tolist():
            for d in range(lo_dim, dm.tdim + 1):
                for j in range(len(topo[d])):
                    key = (d, int(dm.cell_entities[d][c, j]))
                    if key not in tasks:
                        tasks[key] = (c, j, -1)
    if has_vertex_dofs:
        for v in np.flatnonzero(region_mask).tolist():
            if v < dm.n_entities[0] and (0, v) not in tasks:
                # a region vertex outside every qualifying facet (a point / curve region): pin its value
                rows = np.argwhere(dm.cell_entities[0] == v)
                if rows.size:
                    tasks[(0, v)] = (int(rows[0, 0]), int(rows[0, 1]), -1)
    return tasks


def interpolate_tasks(dm: DofMap, points: np.ndarray, cells: np.ndarray, tasks: Dict, value_fn, normals=None):
    """Global DOF values for entity ``tasks`` (from :func:`region_trace_entities`) of the field ``value_fn``.

    ``value_fn(X, N) -> (n, value_size)`` at physical points ``X`` (``(n, gdim)``) with ``N`` the outward
    normal of the facet the task came from (zeros for a cell-closure task). The entity's reference
    functionals (basix ``x``/``M``) act on the pulled-back field and ``B⁻ᵀ`` orients the result --
    vectorised per (dim, local entity) group. Returns ``(dofs, values)`` in block-local numbering."""
    if not tasks:
        return np.zeros((0,), np.int64), np.zeros((0,))
    elem = dm.element
    J, x0 = cell_jacobians(points, cells)
    mt = map_type(dm.family)
    groups: Dict[Tuple[int, int], list] = {}
    for (d, e), (c, j, f) in tasks.items():
        groups.setdefault((d, j), []).append((e, c, f))
    dofs_l, vals_l = [], []
    for (d, j), items in groups.items():
        es = np.asarray([t[0] for t in items], dtype=np.int64)
        cs = np.asarray([t[1] for t in items], dtype=np.int64)
        fs = np.asarray([t[2] for t in items], dtype=np.int64)
        Xr = np.asarray(elem.x[d][j])  # (npts, tdim)
        Mk = np.asarray(elem.M[d][j])[..., 0]  # (nd, vs, npts)
        nd = Mk.shape[0]
        if nd == 0:
            continue
        Jc = J[cs]  # (n, tdim, tdim)
        X = x0[cs][:, None, :] + np.einsum("pa,nda->npd", Xr, Jc)  # (n, npts, gdim)
        npts = Xr.shape[0]
        N = (
            np.zeros((len(cs), X.shape[-1]))
            if normals is None
            else np.where(fs[:, None] >= 0, normals[np.maximum(fs, 0)], 0.0)
        )
        Nq = np.repeat(N, npts, axis=0)
        vals = np.asarray(value_fn(X.reshape(-1, X.shape[-1]), Nq))
        vals = vals.reshape(len(cs), npts, -1)
        if mt == "covariant":
            ref = np.einsum("npd,nda->npa", vals, Jc)  # û = Jᵀ u
        elif mt == "contravariant":
            Kc = np.linalg.inv(Jc)
            ref = np.linalg.det(Jc)[:, None, None] * np.einsum("nad,npd->npa", Kc, vals)  # det J · J⁻¹ u
        else:
            ref = vals
        lref = np.einsum("dvp,npv->nd", Mk, ref)  # (n, nd) reference DOF values
        if d in dm.blocks:
            Bk = dm.blocks[d][j][dm.orient[d][cs, j].astype(np.int64)]  # (n, nd, nd)
            glob = np.linalg.solve(np.swapaxes(Bk, 1, 2), lref[..., None])[..., 0]
        else:
            glob = lref
        base = dm.entity_offset[d] + es * dm.n_entity_dofs[d]
        dofs_l.append((base[:, None] + np.arange(nd)[None, :]).reshape(-1))
        vals_l.append(glob.reshape(-1))
    if not dofs_l:
        return np.zeros((0,), np.int64), np.zeros((0,))
    return np.concatenate(dofs_l), np.concatenate(vals_l)


def entity_dofs_of_tasks(dm: DofMap, tasks: Dict) -> np.ndarray:
    """Block-local global DOFs of every entity in ``tasks`` (for a homogeneous trace: all pinned to 0)."""
    out = [dm.entity_global_dofs(d, e) for (d, e) in tasks]
    return np.concatenate(out) if out else np.zeros((0,), np.int64)


def basis_at_points(dm: DofMap, points: np.ndarray, cells: np.ndarray, c: int, X: np.ndarray) -> np.ndarray:
    """Physical basis of cell ``c`` at physical points ``X`` (``(n, gdim)``) -> ``(n, n_dof, vs)``."""
    J, x0 = cell_jacobians(points, cells[c : c + 1])
    J, x0 = J[0], x0[0]
    xi = np.linalg.solve(J, (np.asarray(X) - x0).T).T
    tab = np.asarray(dm.element.tabulate(0, xi)[0])
    B = dm.cell_transform_np(c) if not dm.is_diagonal else np.diag(dm.signs[c])
    t = np.einsum("ij,qjv->qiv", B, tab)
    mt = map_type(dm.family)
    if mt == "covariant":
        return np.einsum("ji,qnj->qni", np.linalg.inv(J), t)
    if mt == "contravariant":
        return np.einsum("ij,qnj->qni", J, t) / np.linalg.det(J)
    return t


def basis_at_points_batch(dm: DofMap, points: np.ndarray, cells: np.ndarray, cs: np.ndarray, X: np.ndarray) -> np.ndarray:
    """:func:`basis_at_points` for many (cell, point) pairs at once: point ``X[i]`` in cell ``cs[i]``
    -> ``(n, n_dof, vs)``. One tabulation for all points, the orientation transforms gathered per pair."""
    cs = np.asarray(cs, dtype=np.int64)
    J, x0 = cell_jacobians(points, np.asarray(cells)[cs])
    xi = np.linalg.solve(J, (np.asarray(X) - x0)[..., None])[..., 0]
    tab = np.asarray(dm.element.tabulate(0, xi)[0])  # (n, n_dof, vs)
    if dm.is_diagonal:
        t = np.asarray(dm.signs)[cs][:, :, None] * tab
    else:
        t = np.array(tab)
        for dim, blk in dm.blocks.items():
            for k, idx in enumerate(dm.entity_dofs[dim]):
                Bk = blk[k][np.asarray(dm.orient[dim])[cs, k].astype(np.int64)]  # (n, nd, nd)
                t[:, idx] = np.einsum("nij,njv->niv", Bk, tab[:, idx])
    mt = map_type(dm.family)
    if mt == "covariant":
        return np.einsum("nji,nqj->nqi", np.linalg.inv(J), t)
    if mt == "contravariant":
        return np.einsum("nij,nqj->nqi", J, t) / np.linalg.det(J)[:, None, None]
    return t


def periodic_prolongation(dm: DofMap, points: np.ndarray, cells: np.ndarray, ties, *, tol: Optional[float] = None):
    """DOF-level periodic / Floquet-Bloch prolongation ``u = P ũ`` for a field of ANY family and degree.

    ``ties`` is a list of ``(main_vertex_mask, secondary_vertex_mask, phase)``. Every entity in the
    closure of a secondary boundary facet is matched to the main entity it lands on under the
    translation that maps the secondary face onto the main one (by entity centroid). Its DOFs are then
    expressed through the main side by INTERPOLATION: the secondary entity's functionals (basix ``x``/``M``,
    pulled back, oriented by ``B⁻ᵀ``) applied to ``phase · u_main(x - shift)``, with ``u_main`` the basis of
    a main-facet cell. That one rule absorbs edge reversals, face rotations/reflections, the relative
    orientation of the two faces and the Bloch phase -- no per-family sign logic. Ties that chain (an edge
    on two periodic faces) are resolved by substitution until only retained DOFs remain.

    Returns ``(P, kept, is_bloch)``: ``P`` a BCOO ``(n_dofs, n_kept)``, ``kept`` the retained DOF ids.
    Raises if a secondary entity has no main partner (a non-conforming periodic mesh)."""
    import jax.experimental.sparse as jsparse
    import jax.numpy as jnp
    from scipy.spatial import cKDTree

    pts = np.asarray(points)[:, : dm.tdim]
    cells = np.asarray(cells, dtype=np.int64)
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    tol = max(span, 1.0) * 1e-6 if tol is None else tol
    is_bloch = any(abs(complex(ph) - 1.0) > 1e-12 for (_m, _s, ph) in ties)
    J, x0 = cell_jacobians(pts, cells)
    elem = dm.element
    mt = map_type(dm.family)
    rel: Dict[int, Dict[int, complex]] = {}  # secondary dof -> {dof: weight}
    for mmask, smask, ph in ties:
        mmask = np.asarray(mmask, bool)
        smask = np.asarray(smask, bool)
        dvec = pts[smask].mean(axis=0) - pts[mmask].mean(axis=0)
        axis = int(np.argmax(np.abs(dvec)))
        shift = np.zeros(dm.tdim)
        shift[axis] = dvec[axis]
        t_main = region_trace_entities(dm, mmask, boundary_only=True)
        t_sec = region_trace_entities(dm, smask, boundary_only=True)
        if not t_main or not t_sec:
            raise ValueError("periodic tie: a tied boundary has no boundary facet (check the tags).")
        by_dim: Dict[int, list] = {}
        for (d, e), (c, j, f) in t_main.items():
            by_dim.setdefault(d, []).append((e, c, j))
        trees = {
            d: (cKDTree(np.stack([pts[dm.entity_vertices[d][e]].mean(axis=0) for e, _c, _j in lst])), lst)
            for d, lst in by_dim.items()
        }
        for (d, e_s), (c_s, j_s, _f) in t_sec.items():
            gd_s = dm.entity_global_dofs(d, e_s)
            if gd_s.size == 0 or int(gd_s[0]) in rel:
                continue
            cen = pts[dm.entity_vertices[d][e_s]].mean(axis=0) - shift
            tree, lst = trees[d]
            dist, k = tree.query(cen)
            if dist > tol:
                raise ValueError(
                    f"periodic tie: a secondary dim-{d} entity at {np.round(cen + shift, 6)} has no main partner "
                    f"(nearest {dist:.2e} > tol {tol:.2e}); a conforming periodic mesh is required."
                )
            e_m, c_m, _j_m = lst[int(k)]
            if e_m == e_s:
                continue  # an entity on both faces of its own tie (degenerate): nothing to tie
            Xr = np.asarray(elem.x[d][j_s])
            Mk = np.asarray(elem.M[d][j_s])[..., 0]  # (nd, vs, npts)
            Xs = x0[c_s] + Xr @ J[c_s].T
            Phi = basis_at_points(dm, pts, cells, c_m, Xs - shift)  # (npts, n_dof, vs) of the main cell
            Jc = J[c_s]
            if mt == "covariant":
                ref = np.einsum("pnd,da->pna", Phi, Jc)
            elif mt == "contravariant":
                ref = np.linalg.det(Jc) * np.einsum("ad,pnd->pna", np.linalg.inv(Jc), Phi)
            else:
                ref = Phi
            lref = np.einsum("dvp,pnv->dn", Mk, ref)  # (nd, n_dof of the main cell)
            Bk = dm.entity_block(c_s, d, j_s)
            R = np.linalg.solve(Bk.T, lref) * complex(ph)  # (nd, n_dof)
            cols = dm.cell_dofs[c_m]
            scale = max(float(np.abs(R).max()), 1e-300)
            for a, gs in enumerate(gd_s.tolist()):
                row = {}
                for b, gm in enumerate(cols.tolist()):
                    w = R[a, b]
                    if abs(w) > 1e-10 * scale:
                        row[int(gm)] = row.get(int(gm), 0.0) + w
                rel[int(gs)] = row
    # resolve chains: substitute secondary dofs until every weight sits on a retained dof
    resolved: Dict[int, Dict[int, complex]] = {}

    def _resolve(sdof, stack=()):
        if sdof in resolved:
            return resolved[sdof]
        if sdof in stack:
            raise ValueError("periodic tie: the ties form a cycle (a face tied to itself through others).")
        out: Dict[int, complex] = {}
        for m, w in rel[sdof].items():
            if m in rel:
                for mm, ww in _resolve(m, stack + (sdof,)).items():
                    out[mm] = out.get(mm, 0.0) + w * ww
            else:
                out[m] = out.get(m, 0.0) + w
        resolved[sdof] = out
        return out

    for sd in list(rel):
        _resolve(sd)
    secondary = np.zeros(dm.n_dofs, dtype=bool)
    secondary[list(rel)] = True
    kept = np.flatnonzero(~secondary)
    col = np.full(dm.n_dofs, -1, dtype=np.int64)
    col[kept] = np.arange(kept.size)
    rows, cols, vals = [kept], [np.arange(kept.size)], [np.ones(kept.size, dtype=complex)]
    for sd, row in resolved.items():
        if row:
            rows.append(np.full(len(row), sd))
            cols.append(col[np.asarray(list(row), dtype=np.int64)])
            vals.append(np.asarray(list(row.values()), dtype=complex))
    rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    if not is_bloch:
        vals = vals.real
    P = jsparse.BCOO((jnp.asarray(vals), jnp.asarray(np.stack([rows, cols], axis=1))), shape=(dm.n_dofs, kept.size))
    return P, kept, is_bloch


__all__: Sequence[str] = (
    "DofMap",
    "basix_element",
    "build_dofmap",
    "cell_jacobians",
    "cell_orientations",
    "entity_blocks",
    "map_type",
)
