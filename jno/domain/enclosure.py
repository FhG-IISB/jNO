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


def _solid_polygon_visibility(domain, elem_tag, mids, normals, length):
    """Element-to-element visibility using the clean solid geometry: element *i* sees *j* iff the segment
    between their midpoints (nudged into the medium along the element normals) does not pass through any
    opaque solid region's interior. Uses ``domain._source_regions`` (the exact polygons), so it is immune
    to mesh-classification slivers; correct for axisymmetric since a solid ring blocks every azimuth."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from shapely.prepared import prep

    m = mids.shape[0]
    regions = getattr(domain, "_source_regions", {}) or {}
    geoms = [regions[s] for s in sorted(set(map(str, elem_tag))) if s in regions]
    if not geoms:
        return 1.0 - np.eye(m)
    union = unary_union(geoms)
    prepared = prep(union)
    eps = 0.5 * float(np.median(length))  # nudge endpoints off the solid boundary into the medium
    tol = 0.25 * eps
    P = mids + eps * np.asarray(normals)
    vm = np.eye(m) * 0.0
    for i in range(m):
        for j in range(i + 1, m):
            seg = LineString((P[i], P[j]))
            vis = 1.0
            if prepared.intersects(seg):  # cheap filter; only then measure the crossing length
                if float(getattr(seg.intersection(union), "length", 0.0)) > tol:
                    vis = 0.0
            vm[i, j] = vm[j, i] = vis
    return vm


class _RadiationFlux:
    """Result of :meth:`Enclosure.flux` — a net grey-body surface flux that becomes a radiation
    :class:`jno._fem.Coupling` when multiplied by a test function (``gap.flux(T, eps) * sT``), so radiation
    reads as ``flux * test`` like any other weak-form term. The test function only marks the equation; the
    flux already knows its surface DOFs from the enclosure."""

    def __init__(self, gap, field, kw):
        self._gap, self._field, self._kw = gap, field, kw

    def _as_coupling(self):
        return self._gap.radiation(self._field, **self._kw)

    def __mul__(self, test):
        return self._as_coupling()

    __rmul__ = __mul__


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

    def radiation(self, field, *, emissivity, sigma: float = 5.670374419e-8, scale=1.0, offset=0.0, size=None):
        """A grey-body enclosure-radiation **coupling term** for ``jno.fem([...])``.

        Returns a :class:`jno._fem.Coupling` whose residual is the net radiative surface load this
        enclosure exerts on the temperature ``field``: with absolute per-element temperatures
        ``Tk = field(u) + offset``, the radiosity is ``J = (I - rho F)^{-1} eps sigma Tk^4`` (``rho=1-eps``),
        the net flux per element is ``s_row*J - F@J``, and the consistent nodal load is scattered back.
        ``jno.fem`` adds ``scale * load`` to the assembled residual and ``fem.solve()`` solves the coupled
        conduction+radiation system **implicitly** (Newton-Krylov, ``custom_root``), so it is differentiable
        in any ``jno.np.parameter`` in the form and trains through ``jno.core`` -- no bring-your-own loop::

            gap  = d.enclosure(solids, axisymmetric=True, medium_tags=["Gas","Air"])
            fem  = jno.fem([conduction, gap.radiation(T, emissivity=eps_map), u(xc,yc)-T_COOL])
            Tsol = fem.solve(u0=T_guess)        # conduction + radiation, one implicit solve

        ``emissivity`` is a ``{tag: eps}`` map or a scalar (see :meth:`emissivity`); ``sigma`` the
        Stefan-Boltzmann constant (use a non-dimensional value with ``offset`` = absolute-temperature
        offset, since ``T^4`` is not offset-invariant); ``scale`` an optional conduction-radiation number.
        The temperature must be a **scalar P1** field on the mesh nodes (this enclosure's global node
        indices address its DOFs); ``size`` defaults to the node count. Multifield / transient coupling is
        not yet wired (``jno.fem`` raises). Pure-JAX, so it composes with autodiff and ``jno.core``."""
        from .._fem import Coupling

        F = jnp.asarray(self.view_factor)
        eps = self.emissivity(emissivity)
        rho = 1.0 - eps
        eye = jnp.eye(self.size)
        s_row = F.sum(axis=1)

        def residual_fn(u):
            Tk = self.field(u) + offset
            J = jnp.linalg.solve(eye - rho[:, None] * F, eps * sigma * Tk**4)
            return scale * self.load(s_row * J - F @ J, size=size)

        return Coupling(residual_fn, name="radiation", field_key=getattr(field, "field_key", None))

    def flux(self, field, emissivity, *, sigma: float = 5.670374419e-8, offset=0.0, scale=1.0):
        """The net grey-body radiative flux this enclosure exerts on ``field`` -- written like a weak-form
        **surface flux**: multiply it by the test function to add it to the temperature equation, exactly
        parallel to a Robin term ``Bi*(T - T_ext)*sT``::

            radiation = gap.flux(T, eps, offset=273.15) * sT      # ∫_Γ q_rad · v   (grey-body net flux)

        Same physics as :meth:`radiation` (which returns the ready-made coupling directly); this form just
        reads as ``flux * test`` so the radiation line matches the other equations in ``jno.fem([...])``."""
        return _RadiationFlux(self, field, dict(emissivity=emissivity, sigma=sigma, offset=offset, scale=scale))

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


def _classify_triangles(domain, triangles: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Region name (from ``domain._source_regions``) containing each triangle's centroid, else ``None``.

    Used by the interface (``medium_tags``) path to find solid|medium radiating faces. The geometry
    regions tile the domain, so on a connected mesh every centroid lands in exactly one region."""
    regions = getattr(domain, "_source_regions", {}) or {}
    cent = pts[triangles].mean(axis=1)
    region_of = np.full(triangles.shape[0], None, dtype=object)
    try:
        from shapely.vectorized import contains as _vcontains

        for name, geom in regions.items():
            hit = np.asarray(_vcontains(geom, cent[:, 0], cent[:, 1]))
            region_of[hit & (region_of == None)] = name  # noqa: E711
    except Exception:
        from shapely.geometry import Point

        for i in range(cent.shape[0]):
            for name, geom in regions.items():
                if geom.contains(Point(cent[i])):
                    region_of[i] = name
                    break
    return region_of


def _enforce_closure(F, areas, n_iter: int = 200):
    """Correct a raw view-factor matrix to satisfy closure (``sum_j F_ij = 1``) and reciprocity
    (``A_i F_ij = A_j F_ji``) simultaneously — the standard consistency fix for view factors from an
    approximate / discretised kernel (where near-field rings are over- or under-resolved).

    Symmetric matrix scaling (Sinkhorn) on the exchange matrix ``G = A_i F_ij``: repeatedly rescale ``G``
    by ``sqrt(A_i / sum_j G_ij)`` on both sides (keeping ``G`` symmetric) until its row sums approach the
    areas; then ``F = G / A_i`` has unit row sums and exact reciprocity. Done once at build time (NumPy).

    Reference: A. M. van Leersum, "A method for determining a consistent set of radiation view factors
    from a set generated by a non-exact method", Int. Comm. Heat Mass Transfer 16 (1989) 83-94.
    """
    F = np.asarray(F, dtype=np.float64)
    A = np.asarray(areas, dtype=np.float64)
    G = A[:, None] * F
    G = 0.5 * (G + G.T)  # symmetrize -> exact reciprocity
    for _ in range(int(n_iter)):
        s = G.sum(axis=1)
        s = np.where(s < 1e-30, 1.0, s)
        sc = np.sqrt(A / s)
        G = G * sc[:, None] * sc[None, :]
    return jnp.asarray(G / A[:, None])


def build_enclosure(
    domain,
    tags: Sequence[str],
    *,
    axisymmetric: bool = False,
    n_quad: int = 3,
    n_phi: int = 16,
    opaque_tags: Optional[Sequence[str]] = None,
    medium_tags: Optional[Sequence[str]] = None,
    enforce_closure: bool = False,
    closure_iters: int = 200,
    occlude: bool = True,
    inward: bool = False,
    r_min: Optional[float] = None,
) -> Enclosure:
    """Assemble an :class:`Enclosure` from radiating ``tags`` on a meshed 2D / axisymmetric domain.

    Two surface-discretisation modes:

    * **Boundary mode** (``medium_tags is None``, default): radiating elements are domain **boundary
      edges** (one adjacent triangle) whose endpoints lie on a ``tags`` surface. By default the normals
      point *out of the mesh* — the radiating surface faces an *un-meshed* gap (e.g. a vacuum between
      solid parts). Set ``inward=True`` when the radiating surfaces are instead the **outer walls of a
      meshed cavity** and radiation crosses the *meshed* interior (an oven/furnace filled with a
      transparent fluid): the normals then point *into the mesh* so the facing walls see one another.
    * **Interface mode** (``medium_tags`` given): the radiating gap is itself **meshed** (a participating
      but transparent medium such as a furnace gas/air). Radiating elements are the internal
      **solid|medium interface edges** between a ``tags`` region (a solid, by geometry-part name) and a
      ``medium_tags`` region; normals point out of the solid into the medium. This is the common furnace
      case where every region is meshed.

    ``r_min`` (axisymmetric only) softens the ring kernel's near-field ``1/R^2`` singularity; it defaults
    to half the median element length, which keeps near-coincident and on-axis (``r -> 0``) view factors
    physical (<= 1). Pass an explicit value to override.
    """
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

    elem_nodes_list: List[tuple] = []
    elem_tag: List[str] = []
    if medium_tags is None:
        # Boundary mode: radiating elements are domain boundary edges whose endpoints lie on a tag.
        third, _ = _boundary_edge_third_vertex(triangles)
        tag_masks = {tag: _tag_mask(tag) for tag in tags}
        for a, b in boundary_edges:
            a, b = int(a), int(b)
            for tag in tags:
                mk = tag_masks[tag]
                if mk[a] and mk[b]:
                    elem_nodes_list.append((a, b))
                    elem_tag.append(tag)
                    break
        if not elem_nodes_list:
            raise ValueError(f"domain.enclosure: no boundary elements found on tags {tags}.")
    else:
        # Interface mode: radiating elements are internal solid|medium interface edges. ``tags`` are
        # solid geometry-part names; ``medium_tags`` are the transparent (meshed) media. The normal of
        # each element points out of the solid into the medium (its solid-side third vertex sets it).
        medium_set = {str(m) for m in medium_tags}
        tag_set = set(tags)
        tri_region = _classify_triangles(domain, triangles, pts)
        edge_adj: dict = {}
        for ti in range(triangles.shape[0]):
            a, b, c = int(triangles[ti, 0]), int(triangles[ti, 1]), int(triangles[ti, 2])
            rg = tri_region[ti]
            for i0, i1, opp in ((a, b, c), (b, c, a), (c, a, b)):
                key = (i0, i1) if i0 < i1 else (i1, i0)
                edge_adj.setdefault(key, []).append((rg, opp))
        third = {}
        for key, adj in edge_adj.items():
            if len(adj) != 2:
                continue
            (r0, t0), (r1, t1) = adj
            if r0 in tag_set and r1 in medium_set:
                elem_nodes_list.append(key)
                elem_tag.append(r0)
                third[key] = t0
            elif r1 in tag_set and r0 in medium_set:
                elem_nodes_list.append(key)
                elem_tag.append(r1)
                third[key] = t1
        if not elem_nodes_list:
            raise ValueError(
                f"domain.enclosure: no solid|medium interface elements found between tags {tags} and "
                f"medium_tags {list(medium_tags)}."
            )
    elem_nodes = np.asarray(elem_nodes_list, dtype=np.int64)  # (m, 2)
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
    if inward and medium_tags is None:
        normals = -normals  # enclosure is the meshed cavity (e.g. an oven): normals point into the mesh

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
    if not occlude:
        vm = 1.0 - np.eye(elem_nodes.shape[0])  # diagnostic: no occlusion (all mutually visible)
    elif medium_tags is not None:
        # Interface mode: occlude with the CLEAN solid polygons (a ray is blocked iff it passes through
        # a solid interior in the (r,z) meridian). Correct for axisymmetric (a solid ring blocks all
        # azimuths) and immune to the mesh-sliver artefacts that the element-edge occluder suffers.
        vm = _solid_polygon_visibility(domain, elem_tag_arr, mids, normals, length)
    else:
        vm = _element_visibility(mids, own_edge, occ0, occ1)

    if axisymmetric:
        # Near-field 1/R^2 floor: without it, near-coincident ring pairs (and the on-axis r->0 elements)
        # produce spuriously large (>1) view factors. Default to half the median element length, matching
        # the element size — callers can override via ``r_min``.
        rmin = 0.5 * float(np.median(length)) if r_min is None else float(r_min)
        F = MeshUtils.get_view_factor_axisymmetric_element(
            jnp.asarray(e0),
            jnp.asarray(e1),
            jnp.asarray(normals),
            jnp.asarray(vm),
            n_quad=n_quad,
            n_phi=n_phi,
            r_min=rmin,
        )
        areas = 2.0 * np.pi * mids[:, 0] * length  # ring areas
    else:
        F = MeshUtils.get_view_factor_2d_element(
            jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(normals), jnp.asarray(vm), n_quad=n_quad
        )
        areas = length

    if enforce_closure:
        F = _enforce_closure(F, areas, n_iter=closure_iters)

    return Enclosure(domain, tags, F, elem_nodes, elem_tag_arr, areas, normals, mids, axisymmetric)
