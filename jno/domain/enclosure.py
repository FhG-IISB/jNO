"""Enclosure radiation handle: ``domain.enclosure(tags)``.

An :class:`Enclosure` turns a set of boundary tags into the geometric ingredients an enclosure-radiation
boundary condition needs — built directly from the **FEM mesh boundary edges**, so the view-factor rows
align with global mesh nodes (and therefore FEM DOFs for a P1 field). The radiating surface is
discretised into **boundary elements** (mesh edges in 2D / meridional segments in the axisymmetric
``(r, z)`` plane); the element-to-element view-factor matrix ``F`` is *fully geometry-determined* —
occlusion (visibility) and orientation (cosine) decide every entry, only the ``i == i`` self-pair is
removed. Tags are merely element groups (for per-surface emissivity); they never block exchange, so a
concave surface keeps its self-view.

The radiosity physics (``q = (I-F)(I-diag(rho)F)^-1 diag(eps) sigma T^4``) is written by the user in
``jno.np`` on top of ``enclosure.view_factor``; this module only supplies the quality-gated ``F``, the
element/node bookkeeping, and the per-element measure used to scatter a flux back onto FEM nodes.

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4-5 (view factors; the net-radiation /
radiosity method for diffuse-grey enclosures).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from .mesh_utils import MeshUtils


def _boundary_edge_third_vertex(triangles: np.ndarray):
    """Map each boundary edge (a tuple of two global node indices, sorted) to the third vertex of its
    one adjacent triangle. Boundary edges occur in exactly one triangle; the third vertex fixes the
    outward (out-of-solid) normal direction."""
    third: dict[tuple[int, int], int] = {}
    seen: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for i0, i1, opp in ((a, b, c), (b, c, a), (c, a, b)):
            key = (i0, i1) if i0 < i1 else (i1, i0)
            seen[key] = seen.get(key, 0) + 1
            third[key] = opp
    return third, seen


def _element_visibility(mids: np.ndarray, own_edge: np.ndarray, occ0: np.ndarray, occ1: np.ndarray):
    """Element-to-element visibility (0/1): element *i* sees *j* iff the segment between their midpoints
    crosses no occluder edge (excluding each element's own edge). Segment-edge intersection, vectorised
    over occluders per source (mirrors :meth:`MeshUtils.get_visibility_matrix_raytrace`)."""
    m = mids.shape[0]
    edge_dir = occ1 - occ0  # (E, 2)

    def cross2d(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    vm = np.ones((m, m), dtype=np.float64)
    idx = np.arange(m)
    for i in range(m):
        ab = mids - mids[i]  # (m, 2)
        denom = cross2d(ab[:, None, :], edge_dir[None, :, :])  # (m, E)
        parallel = np.abs(denom) < 1e-12
        denom_safe = np.where(parallel, 1.0, denom)
        diff = (occ0 - mids[i])[None, :, :]  # (1, E, 2)
        t_seg = cross2d(diff, edge_dir[None, :, :]) / denom_safe
        t_edge = cross2d(diff, ab[:, None, :]) / denom_safe
        eps = 1e-9
        crossing = (~parallel) & (t_seg > eps) & (t_seg < 1 - eps) & (t_edge > eps) & (t_edge < 1 - eps)
        crossing[:, own_edge[i]] = False  # source element's own edge never blocks
        crossing[idx, own_edge] = False  # each target element's own edge never blocks
        vm[i, :] = (~np.any(crossing, axis=1)).astype(np.float64)
    vm[idx, idx] = 0.0
    return vm


class Enclosure:
    """Geometric handle for an enclosure-radiation surface set (see module docstring).

    Attributes
    ----------
    view_factor : (m, m) jnp.ndarray
        Element-to-element diffuse view factor ``F`` (rows = receiving elements). Quality-gated on build.
    elements : (m, 2) np.ndarray
        Global mesh node-index pairs for each boundary element (endpoints are FEM nodes).
    element_tags : (m,) np.ndarray[object]
        The tag each element belongs to (for per-surface emissivity).
    areas : (m,) jnp.ndarray
        Per-element measure: edge length (2D) or ring area ``2*pi*r*length`` (axisymmetric).
    normals : (m, 2) jnp.ndarray
        Unit element normals pointing into the enclosure.
    nodes : (k,) np.ndarray
        Unique global node indices used by the enclosure (FEM DOFs for a scalar P1 field).
    """

    def __init__(self, domain, tags, F, elements, element_tags, areas, normals, midpoints, axisymmetric):
        self.domain = domain
        self.tags = list(tags)
        self._F = jnp.asarray(F)
        self.elements = np.asarray(elements)
        self.element_tags = np.asarray(element_tags, dtype=object)
        self.areas = jnp.asarray(areas)
        self.normals = jnp.asarray(normals)
        self.midpoints = np.asarray(midpoints)
        self.axisymmetric = bool(axisymmetric)

    @property
    def size(self) -> int:
        return int(self._F.shape[0])

    @property
    def view_factor(self):
        return self._F

    @property
    def nodes(self) -> np.ndarray:
        return np.unique(self.elements)

    def tag_mask(self, tag: str) -> np.ndarray:
        """Boolean (m,) mask of elements belonging to ``tag`` — e.g. to build a per-element emissivity."""
        return self.element_tags == tag

    def emissivity(self, values) -> jnp.ndarray:
        """Per-element emissivity ``(m,)`` from a ``{tag: eps}`` mapping (or a scalar for all surfaces)."""
        if np.isscalar(values):
            return jnp.full(self.size, float(values))
        eps = np.zeros(self.size)
        for tag, val in dict(values).items():
            eps[self.tag_mask(tag)] = float(val)
        return jnp.asarray(eps)

    def field(self, u) -> jnp.ndarray:
        """Per-element temperature ``(m,)`` from the global solution ``u`` — the **nonlocal gather**.

        Radiosity is piecewise-constant per boundary element, so each element's temperature is the mean
        of its two endpoint (FEM node) values. Differentiable in ``u`` (used inside the Newton residual)."""
        u = jnp.asarray(u).reshape(-1)
        return 0.5 * (u[self.elements[:, 0]] + u[self.elements[:, 1]])

    def load(self, q, *, size: Optional[int] = None) -> jnp.ndarray:
        """Consistent global surface load ``(n_dofs,)`` from a per-element flux ``q`` ``(m,)``.

        Scatters ``∫_Γ q v ds`` onto the FEM nodes: for piecewise-constant ``q`` and P1 test functions
        ``∫_elem q N_i ds = q · (measure/2)`` per endpoint, so each element contributes ``q·area/2`` to
        each of its two nodes. ``size`` defaults to the mesh node count (scalar P1 DOF layout)."""
        q = jnp.asarray(q).reshape(-1)
        n = int(size) if size is not None else int(np.asarray(self.domain.mesh.points).shape[0])
        half = q * self.areas * 0.5
        load = jnp.zeros(n, dtype=half.dtype)
        load = load.at[jnp.asarray(self.elements[:, 0])].add(half)
        load = load.at[jnp.asarray(self.elements[:, 1])].add(half)
        return load

    def quality(self):
        """Return ``(closure_error, reciprocity_error)`` for the assembled ``F`` (raw, un-normalized).

        ``closure_error = max|sum_j F_ij - 1|`` (should -> 0 for a closed enclosure) and
        ``reciprocity_error = max|A_i F_ij - A_j F_ji|`` (should -> 0 by reciprocity)."""
        F = np.asarray(self._F)
        A = np.asarray(self.areas)
        closure = float(np.abs(F.sum(axis=1) - 1.0).max())
        reciprocity = float(np.abs(A[:, None] * F - (A[:, None] * F).T).max())
        return closure, reciprocity

    def check(self, *, closure_atol: float = 5e-2, reciprocity_atol: float = 1e-7):
        """F-quality gate: raise if closure or reciprocity exceeds tolerance (mesh too coarse / normals
        mis-oriented / surfaces not enclosing). Returns ``self`` for chaining."""
        closure, reciprocity = self.quality()
        if reciprocity > reciprocity_atol:
            raise ValueError(
                f"Enclosure view factor fails reciprocity (max|A_i F_ij - A_j F_ji| = {reciprocity:.2e} "
                f"> {reciprocity_atol:.0e}); normals or geometry are inconsistent."
            )
        if closure > closure_atol:
            raise ValueError(
                f"Enclosure view factor fails closure (max|row_sum - 1| = {closure:.2e} > {closure_atol:.0e}); "
                "the surfaces may not form a closed enclosure or the mesh is too coarse."
            )
        return self

    def __repr__(self):
        c, r = self.quality()
        kind = "axisymmetric" if self.axisymmetric else "2D"
        return f"Enclosure({self.tags}, {kind}, {self.size} elements, closure={c:.1e}, reciprocity={r:.1e})"


def build_enclosure(
    domain,
    tags: Sequence[str],
    *,
    axisymmetric: bool = False,
    n_quad: int = 3,
    opaque_tags: Optional[Sequence[str]] = None,
) -> Enclosure:
    """Assemble an :class:`Enclosure` from boundary ``tags`` on a meshed 2D / axisymmetric domain."""
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(t) for t in tags]
    if not tags:
        raise ValueError("domain.enclosure requires at least one boundary tag.")
    if int(getattr(domain, "dimension", 0)) != 2:
        raise NotImplementedError(
            "domain.enclosure is implemented for 2D and axisymmetric (r, z) meshes; 3D enclosure radiation is future work."
        )

    mesh = domain.mesh
    pts = np.asarray(mesh.points)[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])
    boundary_edges = np.asarray(MeshUtils.extract_boundary_edges(jnp.asarray(triangles), pts.shape[0]))
    third, _ = _boundary_edge_third_vertex(triangles)

    # Per-tag node masks (global node indices on each radiating surface). Prefer the spatial predicate
    # registered by ``domain.tag`` (it takes coordinate arrays); fall back to a geometry-part node set.
    predicates = getattr(domain, "_tag_predicates", {}) or {}
    tag_indices = getattr(domain, "tag_indices", {}) or {}

    def _tag_mask(tag: str) -> np.ndarray:
        if tag in predicates:
            cols = [pts[:, i] for i in range(pts.shape[1])]
            return np.asarray(predicates[tag](*cols)).astype(bool)
        if tag in tag_indices:
            mask = np.zeros(pts.shape[0], dtype=bool)
            mask[np.asarray(tag_indices[tag], dtype=np.int64)] = True
            return mask
        raise ValueError(
            f"domain.enclosure: tag '{tag}' has neither a spatial predicate (domain.tag) nor mesh node "
            "indices; define it with domain.tag(name, predicate) or as a geometry part."
        )

    tag_masks = {tag: _tag_mask(tag) for tag in tags}

    # Elements = boundary edges whose BOTH endpoints lie on one radiating tag.
    elem_nodes: List[np.ndarray] = []
    elem_tag: List[str] = []
    for a, b in boundary_edges:
        a, b = int(a), int(b)
        for tag in tags:
            mk = tag_masks[tag]
            if mk[a] and mk[b]:
                elem_nodes.append((a, b))
                elem_tag.append(tag)
                break
    if not elem_nodes:
        raise ValueError(f"domain.enclosure: no boundary elements found on tags {tags}.")
    elem_nodes = np.asarray(elem_nodes, dtype=np.int64)  # (m, 2)
    elem_tag_arr = np.asarray(elem_tag, dtype=object)

    e0 = pts[elem_nodes[:, 0]]
    e1 = pts[elem_nodes[:, 1]]
    mids = 0.5 * (e0 + e1)
    edge_vec = e1 - e0
    length = np.linalg.norm(edge_vec, axis=1)

    # Into-enclosure normals: perpendicular to the edge, pointing away from the adjacent triangle's
    # third vertex (i.e. out of the solid -> into the gap).
    normals = np.zeros_like(mids)
    for k in range(elem_nodes.shape[0]):
        a, b = int(elem_nodes[k, 0]), int(elem_nodes[k, 1])
        key = (a, b) if a < b else (b, a)
        c = third.get(key)
        t = edge_vec[k]
        n = np.array([-t[1], t[0]])
        n = n / (np.linalg.norm(n) + 1e-30)
        if c is not None and np.dot(mids[k] - pts[c], n) < 0:
            n = -n  # ensure it points away from the solid interior
        normals[k] = n

    # Occluders for visibility: all participating element edges (+ optional opaque-tag boundary edges).
    occ0_list, occ1_list = [e0], [e1]
    if opaque_tags:
        for otag in opaque_tags:
            om = _tag_mask(otag)
            sel = np.array([om[int(a)] and om[int(b)] for a, b in boundary_edges], dtype=bool)
            occ0_list.append(pts[boundary_edges[sel, 0]])
            occ1_list.append(pts[boundary_edges[sel, 1]])
    occ0 = np.concatenate(occ0_list, axis=0)
    occ1 = np.concatenate(occ1_list, axis=0)
    own_edge = np.arange(elem_nodes.shape[0])  # each element's own edge is the first m occluders
    vm = _element_visibility(mids, own_edge, occ0, occ1)

    if axisymmetric:
        F = MeshUtils.get_view_factor_axisymmetric_element(
            jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(normals), jnp.asarray(vm), n_quad=n_quad
        )
        areas = 2.0 * np.pi * mids[:, 0] * length  # ring areas
    else:
        F = MeshUtils.get_view_factor_2d_element(
            jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(normals), jnp.asarray(vm), n_quad=n_quad
        )
        areas = length

    return Enclosure(domain, tags, F, elem_nodes, elem_tag_arr, areas, normals, mids, axisymmetric)
