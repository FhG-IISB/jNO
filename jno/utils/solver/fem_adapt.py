"""Adaptive FEM (AFEM): metric-driven local remeshing for ``jno.fem``.

The classical adaptive loop is ``solve -> estimate -> mark -> refine`` repeated
until an error target or DOF budget is met.  jNO's FEM assembler ingests the mesh
as a *static* numpy array (see :func:`jno.utils.solver.fem_native.build_native_fem_context`),
so mesh adaptation cannot live inside the differentiable trace.  Instead this module
implements the loop as an *outer* Python driver that rebuilds the ``domain`` between
solves; every fixed-mesh solve remains fully differentiable, and the final adapted
mesh feeds the ordinary (differentiable) ``jno.fem`` path unchanged.

The refinement step is delegated to **Mmg** (via the ``mmgpy`` binding) -- a
metric-based *local* remesher that inserts/deletes/swaps only where the requested
size field demands, rather than re-triangulating from scratch.

Reference: Dörfler, "A convergent adaptive algorithm for Poisson's equation",
SIAM J. Numer. Anal. 33 (1996), 1106-1124 (bulk marking).  Zienkiewicz & Zhu,
"A simple error estimator and adaptive procedure for practical engineering
analysis", Int. J. Numer. Methods Eng. 24 (1987), 337-357 (recovery estimator).

Public surface: ``FEM.solve(adapt=AdaptSpec(...))`` drives the loop; ``domain.refine``
applies a hand-built size field.  The building blocks (``remesh_with_mmg``,
``transfer_solution`` (mesh-to-mesh nodal-field interpolation — the keystone for carrying state
across a remesh in a transient / moving-mesh loop), ``zz_error_indicators``, ``dorfler_mark``,
``size_field_from_marks``) are reusable on their own.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np


def _boundary_edges_from_triangles(tris: np.ndarray) -> np.ndarray:
    """Return the ``(n_boundary, 2)`` boundary edges of a triangle mesh.

    A facet is on the boundary iff it belongs to exactly one triangle.  This is
    purely topological (no coordinates), so it is robust for any simplicial mesh
    regardless of how its boundary was originally tagged.
    """
    tris = np.asarray(tris, dtype=np.int64)
    # the three edges of every triangle, each sorted so (a,b)==(b,a)
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    e_sorted = np.sort(e, axis=1)
    uniq, counts = np.unique(e_sorted, axis=0, return_counts=True)
    return uniq[counts == 1].astype(np.int32)


def _boundary_faces_from_tets(tets: np.ndarray) -> np.ndarray:
    """Return the ``(n_boundary, 3)`` triangular boundary faces of a tetrahedral mesh.

    The 3D analogue of :func:`_boundary_edges_from_triangles`: a triangular facet is on the
    boundary iff it belongs to exactly one tetrahedron.  Purely topological, so robust for
    any tet mesh regardless of how its surface was tagged.
    """
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]], axis=0)
    f_sorted = np.sort(faces, axis=1)
    uniq, counts = np.unique(f_sorted, axis=0, return_counts=True)
    return uniq[counts == 1].astype(np.int32)


def _corner_vertices(points: np.ndarray, bedges: np.ndarray, angle_tol: float = 0.35) -> np.ndarray:
    """Boundary vertices where the boundary turns by more than ``angle_tol`` (radians).

    These are the geometric corners of a polygonal domain (including the reentrant
    corner of an L-shape).  They are pinned as *required* during remeshing so Mmg
    never smooths or collapses them -- if the reentrant corner drifted, the solution
    singularity would move and every convergence measurement would be against the
    wrong problem.
    """
    points = np.asarray(points)[:, :2]
    bedges = np.asarray(bedges, dtype=np.int64)
    # boundary-vertex adjacency along the boundary polygon
    neigh: dict[int, list[int]] = {}
    for a, b in bedges:
        neigh.setdefault(int(a), []).append(int(b))
        neigh.setdefault(int(b), []).append(int(a))

    corners: list[int] = []
    for v, nbrs in neigh.items():
        nbrs = list(dict.fromkeys(nbrs))  # dedupe
        if len(nbrs) != 2:
            corners.append(v)  # endpoint / non-manifold -> always a corner
            continue
        t1 = points[nbrs[0]] - points[v]
        t2 = points[nbrs[1]] - points[v]
        n1 = np.linalg.norm(t1)
        n2 = np.linalg.norm(t2)
        if n1 == 0.0 or n2 == 0.0:
            corners.append(v)
            continue
        cos = float(np.clip(np.dot(t1, t2) / (n1 * n2), -1.0, 1.0))
        # straight-through => the two tangents are anti-parallel (angle ~ pi).
        if abs(np.pi - np.arccos(cos)) > angle_tol:
            corners.append(v)
    return np.asarray(sorted(corners), dtype=np.int32)


# Empirical Mmg vertices produced per unit metric-complexity (calibrated on layer problems);
# used by hessian_metric so its `target_complexity` tracks the actual vertex count per dimension.
_VERTS_PER_COMPLEXITY = {2: 1.5, 3: 2.2}


def _sym_tensor_indices(dim: int) -> list[tuple[int, int]]:
    """Upper-triangle ``(i, j)`` index pairs for a symmetric ``dim x dim`` tensor, row-major:
    2D -> [(0,0),(0,1),(1,1)] (3 comps); 3D -> [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)] (6 comps).
    This is Mmg's Medit tensor-solution component ordering."""
    return [(i, j) for i in range(dim) for j in range(i, dim)]


def _unpack_sym_tensor(field: np.ndarray, dim: int) -> np.ndarray:
    """Expand packed upper-triangle metric components ``(n, 3|6)`` to full ``(n, dim, dim)``."""
    n = field.shape[0]
    M = np.zeros((n, dim, dim))
    for k, (i, j) in enumerate(_sym_tensor_indices(dim)):
        M[:, i, j] = field[:, k]
        M[:, j, i] = field[:, k]
    return M


def remesh_with_mmg(
    domain: Any,
    vertex_size: np.ndarray,
    *,
    copy: bool = True,
    hmin: float | None = None,
    hmax: float | None = None,
    hgrad: float = 1.3,
    hausd: float | None = None,
    verbose: int = -1,
):
    """Locally remesh ``domain`` to a per-vertex target edge size.

    Parameters
    ----------
    domain
        A meshed 2D (triangle) or 3D (tetrahedron) ``jno`` domain.
    vertex_size
        Either ``(n_vertices,)`` isotropic target edge lengths (smaller = finer), or an
        ``(n_vertices, 3)`` **anisotropic** metric tensor ``(m11, m12, m22)`` per vertex
        (e.g. from :func:`hessian_metric`), which additionally sets the *direction* of
        refinement -- stretched triangles aligned to a directional feature.
    copy
        If ``True`` (default) apply the new mesh to a shallow copy and return it,
        leaving ``domain`` untouched.  If ``False`` remesh ``domain`` **in place**
        (used by the adaptive driver, so constraints bound to ``domain`` re-assemble
        on the refined mesh).
    hmin, hmax
        Hard clamps on the produced edge length.  Default to half the smallest and
        twice the largest requested size, so the requested field is never itself
        clamped away (a too-large ``hmin`` would silently cap the refinement).
    hgrad
        Maximum ratio between adjacent edge lengths (size gradation).
    hausd
        Hausdorff boundary-approximation tolerance (only relevant for curved
        boundaries; polygonal boundaries stay exact regardless).

    Returns
    -------
    domain
        The remeshed domain (a shallow copy if ``copy``, else ``domain`` itself).
        Spatial tag predicates (``domain.tag(...)``) are preserved, so ``jno.fem``
        boundary conditions bound to those tags resolve geometrically on the new nodes.

    Notes
    -----
    Polygonal corners (including a reentrant corner) are pinned as *required* so the
    domain geometry is preserved exactly; boundary edges are left splittable so the
    boundary can still be refined along its straight segments.
    """
    dim = int(domain.dimension)
    if dim == 1:
        # mmg has no 1-D mode, and needs none: an interval mesh is a sorted list of vertices, so
        # honouring a size field is subdivision rather than remeshing. Same signature and the same
        # returned domain, so every caller (the steady AFEM loop, the transient re-mesher) is
        # dimension-agnostic above this line.
        return _remesh_line_1d(domain, vertex_size, hmin=hmin, hmax=hmax, hgrad=hgrad, copy=copy)
    if dim not in (2, 3):
        raise NotImplementedError(f"remesh_with_mmg supports 1D line and 2D/3D simplicial meshes; got dimension {dim}.")

    import mmgpy  # lazy: optional dependency, only needed for adaptive refinement

    pts = np.asarray(domain.mesh.points)[:, :dim].astype(np.float64)
    field = np.asarray(vertex_size, dtype=np.float64)

    # (n,) / (n,1) -> isotropic size; (n, 3 in 2D / 6 in 3D) -> anisotropic metric tensor
    n_tensor = 3 if dim == 2 else 6
    anisotropic = field.ndim == 2 and field.shape[1] == n_tensor
    if not anisotropic:
        field = field.reshape(-1)
        if not np.all(field > 0):
            raise ValueError("vertex_size must be strictly positive.")
    if field.shape[0] != pts.shape[0]:
        raise ValueError(f"metric field has {field.shape[0]} entries but the mesh has {pts.shape[0]} vertices.")

    if dim == 2:
        elems = np.asarray(domain.mesh.cells_dict["triangle"]).astype(np.int32)
        bfacets = _boundary_edges_from_triangles(elems)
        corners = _corner_vertices(pts, bfacets)
        m = mmgpy.MmgMesh2D()
        m.set_mesh_size(len(pts), len(elems), 0, len(bfacets))
        m.set_vertices(pts)
        m.set_triangles(elems)
        m.set_edges(bfacets, np.ones(len(bfacets), dtype=np.int32))
        if len(corners):  # pin polygonal corners so the geometry is preserved exactly
            m.set_corners(corners)
            m.set_required_vertices(corners)
    else:
        elems = np.asarray(domain.mesh.cells_dict["tetra"]).astype(np.int32)
        bfacets = _boundary_faces_from_tets(elems)
        m = mmgpy.MmgMesh3D()
        m.set_mesh_size(len(pts), len(elems), 0, len(bfacets), 0, 0)
        m.set_vertices(pts)
        m.set_tetrahedra(elems, np.ones(len(elems), dtype=np.int32))
        m.set_triangles(bfacets, np.ones(len(bfacets), dtype=np.int32))
        # mmg3d auto-detects the boundary ridges/corners of a polyhedral surface and keeps
        # boundary vertices on it, so a polyhedral geometry is preserved without extra marking

    if anisotropic:
        # eigen-sizes of the tensor set the hmin/hmax window; the mmg tensor channel is aniso
        eig = np.linalg.eigvalsh(_unpack_sym_tensor(field, dim))
        sizes = 1.0 / np.sqrt(np.clip(eig, 1e-300, None))
        m.set_field("tensor", field)
    else:
        sizes = field
        m.set_field("metric", field.reshape(-1, 1))

    if hmin is None:
        hmin = float(sizes.min()) * 0.5
    if hmax is None:
        hmax = float(sizes.max()) * 2.0
    opts: dict[str, float] = {"hmin": hmin, "hmax": hmax, "hgrad": hgrad, "verbose": verbose}
    if hausd is not None:
        opts["hausd"] = hausd
    m.remesh(**opts)

    v_out = np.asarray(m.get_vertices())[:, :dim]
    if dim == 2:
        t_out, _ = m.get_triangles_with_refs()
        f_out, _ = m.get_edges_with_refs()
    else:
        t_out, _ = m.get_tetrahedra_with_refs()
        f_out, _ = m.get_triangles_with_refs()
    return _domain_from_arrays(domain, v_out, np.asarray(t_out), np.asarray(f_out), copy=copy)


def _remesh_line_1d(domain: Any, vertex_size: Any, *, hmin, hmax, hgrad: float, copy: bool):
    """Rebuild a 1-D line mesh to honour a per-vertex target size — the 1-D face of
    :func:`remesh_with_mmg`.

    An interval mesh *is* a sorted vertex list, so there is nothing to remesh: each element is
    subdivided into as many equal pieces as its requested size demands. That makes the 1-D path exact
    where mmg is approximate, and it needs no optional dependency.

    ``hgrad`` is the size gradation mmg applies in 2-D/3-D — the cap on the ratio between neighbouring
    edge lengths. Without it a sharply peaked estimator produces a 100x jump between adjacent elements,
    which is both wasteful and badly conditioned; it is imposed here by two monotone sweeps over the
    sorted vertices (forward then backward), the 1-D form of mmg's gradation.

    The endpoints are never moved, so the domain's geometry is preserved exactly.
    """
    verts = np.asarray(domain.mesh.points)[:, 0].astype(np.float64)
    cells = np.asarray(domain.mesh.cells_dict["line"], dtype=np.int64)
    sizes = np.asarray(vertex_size, dtype=np.float64).reshape(-1)
    if sizes.shape[0] != verts.shape[0]:
        raise ValueError(f"metric field has {sizes.shape[0]} entries but the mesh has {verts.shape[0]} vertices.")
    if not np.all(sizes > 0):
        raise ValueError("vertex_size must be strictly positive.")

    lengths = np.abs(verts[cells[:, 1]] - verts[cells[:, 0]])
    hmin = float(sizes.min()) * 0.5 if hmin is None else float(hmin)
    hmax = float(sizes.max()) * 2.0 if hmax is None else float(hmax)
    sizes = np.clip(sizes, hmin, hmax)

    order = np.argsort(verts)
    xs, s = verts[order], sizes[order].copy()
    # Gradation, as mmg means it: the ratio between ADJACENT edge sizes is capped at `hgrad`. One
    # forward and one backward sweep is enough — after them every neighbouring pair satisfies the cap,
    # because each sweep is monotone in the direction it travels.
    for i in range(1, len(xs)):
        s[i] = min(s[i], s[i - 1] * hgrad)
    for i in range(len(xs) - 2, -1, -1):
        s[i] = min(s[i], s[i + 1] * hgrad)

    out = [xs[0]]
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        n_sub = max(1, int(np.ceil((b - a) / min(s[i], s[i + 1]))))
        out.extend(np.linspace(a, b, n_sub + 1)[1:])
    new_x = np.asarray(out, dtype=np.float64)
    del lengths

    n = new_x.shape[0] - 1
    new_cells = np.column_stack([np.arange(n), np.arange(1, n + 1)]).astype(np.int64)
    # the boundary of an interval is its two endpoint VERTICES (the block `jno.domain.line` builds)
    bfacets = np.array([[0], [n]], dtype=np.int64)
    return _domain_from_arrays(domain, new_x.reshape(-1, 1), new_cells, bfacets, copy=copy)


def _domain_from_arrays(template: Any, points: np.ndarray, elems: np.ndarray, bfacets: np.ndarray, *, copy: bool):
    """Apply remeshed ``points / elements / boundary-facets`` to a domain.

    Builds a ``meshio.Mesh`` carrying ``interior`` (triangles in 2D / tetrahedra in 3D) and
    ``boundary`` (line / triangle) cell-sets, then runs it through ``_apply_mesh`` on either
    a shallow copy of ``template`` (``copy=True``) or ``template`` itself (``copy=False``, in
    place).  Named boundary sub-tags are *not* stored in the mesh -- they re-derive
    geometrically from ``template``'s spatial predicates.
    """
    import meshio

    points = np.asarray(points, dtype=np.float64)
    dim = points.shape[1]
    n_pts = points.shape[0]
    pts3 = np.zeros((n_pts, 3), dtype=np.float64)
    pts3[:, :dim] = points
    elems = np.asarray(elems, dtype=np.int64)
    bfacets = np.asarray(bfacets, dtype=np.int64)

    elem_type, facet_type = {1: ("line", "vertex"), 2: ("triangle", "line")}.get(dim, ("tetra", "triangle"))
    cells = [(elem_type, elems), (facet_type, bfacets)]
    n_e, n_f = len(elems), len(bfacets)
    empty = np.asarray([], dtype=np.int64)
    cell_sets = {
        "interior": [np.arange(n_e, dtype=np.int64), empty],
        "boundary": [empty, np.arange(n_f, dtype=np.int64)],
    }
    if dim == 1:
        # `jno.domain.line` names the two endpoint vertices; re-declare them here so `left` / `right`
        # survive the remesh as cell sets, exactly as the mesh generator's named edge regions are
        # re-derived geometrically in 2D/3D by `_capture_geometric_boundary_tags`.
        lo = int(np.argmin(points[:, 0][np.asarray(bfacets).reshape(-1)]))
        cell_sets["left"] = [empty.copy(), np.array([lo], dtype=np.int64)]
        cell_sets["right"] = [empty.copy(), np.array([1 - lo], dtype=np.int64)]
    new_mesh = meshio.Mesh(pts3, cells, cell_sets=cell_sets)

    target = _shallow_copy(template) if copy else template
    # Drop native-FEM caches keyed to the old mesh so they rebuild for the new one.
    for attr in list(vars(target)):
        if attr.startswith("_fem_native") or attr in ("_fem_assembly_cache", "_integral_weight_cache"):
            delattr(target, attr)
    # Give the mesh-generator's named boundary regions a predicate BEFORE the reset, so they are
    # re-derived on the new mesh exactly like a user `.tag()` (see `_capture_geometric_boundary_tags`).
    _capture_geometric_boundary_tags(target)
    # Drop the OLD mesh's predicate-tag state (boundary regions / indices / normals / pools /
    # context) so a re-tag re-derives it cleanly on the new mesh; stale surface-tag state otherwise
    # corrupts re-assembled Neumann/Robin/absorbing terms (predicates in `_tag_predicates` are kept).
    if hasattr(target, "_reset_custom_tag_state"):
        target._reset_custom_tag_state()
    target._apply_mesh(new_mesh)
    # `_apply_mesh` rebuilds the spatial (N, D) mesh pool but does not re-broadcast it over time; on a
    # transient domain re-tile it (idempotent), so `domain.variable(...)` on the remeshed / moved domain
    # samples spatiotemporally again — otherwise a state-dependent velocity reading the field via the DSL
    # on the moved mesh hits a 2-D pool where it expects (T, N, D). Mirrors `_remesh_periodic`.
    if getattr(target, "_is_time_dependent", False) and getattr(target, "time", None) is not None:
        target._add_time_dimension(*target.time)
    return target


def _capture_geometric_boundary_tags(domain: Any) -> None:
    """Give every named boundary region a **mesh-independent predicate**, so it survives a remesh.

    The mesh generator's named edge regions (``left`` / ``right`` / ``top`` / ``bottom`` / any
    ``Shape`` sub-boundary) are baked into the ORIGINAL mesh as *cell sets*.  A remeshed mesh carries
    only ``interior`` and ``boundary`` (:func:`_domain_from_arrays` builds exactly those), so those
    names vanish at the first remesh — and a Dirichlet condition bound to one of them then reaches
    ``jno.fem`` as a whole-domain residual with a trial but no test function, failing with an error
    that names neither the mesh nor the region that disappeared.  Only ``.tag()`` regions survived,
    because a *predicate* is mesh-independent and the drivers re-apply it.

    The domain does keep a mesh-independent description of each named boundary: the shapely curve it
    registered in ``_polygon_tags``.  Turn that into the same kind of spatial predicate a user
    ``.tag()`` supplies — "within tol of this curve" — and every existing re-tag path re-derives the
    region on whatever mesh comes next.  A **distance** test, not ``contains``: these are curves, and
    a point-in-curve test is false for essentially every floating-point node.  Remeshing preserves the
    boundary *geometry* (only its discretisation changes), so the new nodes lie on the same curve.

    The description used is the region's own ``BoundaryRegion`` — the segments (2-D) or triangles (3-D)
    it was registered with — and its ``contains`` test, so no new geometry code is introduced and both
    dimensions are covered by construction.  Remeshing preserves the boundary *geometry* (only its
    discretisation changes), so the new nodes lie on those same facets.

    A **distance-to-facet** test, not a point-in-curve one: these regions are curves/surfaces, and an
    exact containment test is false for essentially every floating-point node.

    Idempotent, and it never overrides a predicate that is already there — a user ``.tag()`` of the
    same name keeps its own definition.  The aggregate ``boundary`` region is skipped: it is rebuilt
    from the new mesh's cell sets directly.
    """
    regions = getattr(domain, "_boundary_regions", None)
    if not isinstance(regions, dict):
        return
    preds = getattr(domain, "_tag_predicates", None)
    if preds is None:
        preds = domain._tag_predicates = {}

    for name, region in list(regions.items()):
        if name in preds or name == "boundary":
            continue  # already predicate-backed (a user tag), or the aggregate (rebuilt from the mesh)
        if region is None or not hasattr(region, "contains"):
            continue
        has_facets = any(
            getattr(region, attr, None) is not None and len(getattr(region, attr)) > 0 for attr in ("edges", "triangles")
        )
        if not has_facets:
            continue

        def _on_region(*coords, _r=region):
            import jax
            import jax.numpy as jnp

            pts = np.stack([np.asarray(c, dtype=float).reshape(-1) for c in coords], axis=-1)
            return np.asarray(jax.vmap(_r.contains)(jnp.asarray(pts)))

        preds[name] = _on_region


def _shallow_copy(domain: Any):
    return copy.copy(domain)


# ---------------------------------------------------------------------------
# Solution transfer -- piecewise-linear (barycentric) mesh-to-mesh interpolation
# ---------------------------------------------------------------------------
def _locate_in_cells(src_pts: np.ndarray, src_cells: np.ndarray, qpts: np.ndarray, *, tol: float, k: int):
    """Locate each query point in a simplicial mesh; return the containing **cell index** and its
    barycentric weights.

    Point location is a KD-tree candidate search over cell centroids (the containing cell is almost
    always among the nearest centroids; raise ``k`` for strongly anisotropic meshes) followed by an
    exact barycentric inside-test. Returns ``(cell_idx, weights, inside)``: ``cell_idx`` ``(Q,)`` the
    chosen simplex's index into ``src_cells``; ``weights`` ``(Q, D+1)`` its barycentric coordinates
    ``[λ0, λ1, …]`` (``weights[:, 1:]`` are the point's **basix reference coordinates** in that cell,
    since the reference simplex is ``v0=0, v_i=e_i``); ``inside`` ``(Q,)`` bool (``True`` = strictly
    contained; ``False`` = fell outside every candidate and was projected onto the nearest simplex by
    clamping + renormalising the weights, a bounded convex combination of that cell's vertices). Pure
    host/NumPy — this is the shared point-location core behind :func:`_locate_barycentric` (P1
    interpolation) and :func:`_eval_fe_fields_at_points` (general P{k}/vector transfer)."""
    from scipy.spatial import cKDTree

    dim = src_pts.shape[1]
    cell_verts = src_pts[src_cells]  # (C, D+1, D)
    kk = int(min(max(k, 1), len(src_cells)))
    _, cand = cKDTree(cell_verts.mean(axis=1)).query(qpts, k=kk)
    cand = np.asarray(cand).reshape(len(qpts), kk)  # (Q, kk) candidate cell indices

    V = cell_verts[cand]  # (Q, kk, nv, D)
    v0 = V[:, :, 0, :]  # (Q, kk, D)
    T = np.transpose(V[:, :, 1:, :] - v0[:, :, None, :], (0, 1, 3, 2))  # (Q, kk, D, D): columns v_i - v0
    rhs = qpts[:, None, :] - v0  # (Q, kk, D)
    detT = np.linalg.det(T)  # (Q, kk); ~0 marks a degenerate candidate
    safe = np.abs(detT) > 1e-300
    Tsafe = np.where(safe[..., None, None], T, np.eye(dim))  # avoid a singular batch member raising
    lam_rest = np.linalg.solve(Tsafe, rhs[..., None])[..., 0]  # (Q, kk, D)
    lam0 = 1.0 - lam_rest.sum(axis=-1)
    lam = np.concatenate([lam0[..., None], lam_rest], axis=-1)  # (Q, kk, nv)

    inside_cand = safe & np.all(lam >= -tol, axis=-1)  # (Q, kk)
    any_inside = inside_cand.any(axis=1)
    choice = np.where(any_inside, np.argmax(inside_cand, axis=1), np.argmax(lam.min(axis=-1), axis=1))
    q = np.arange(len(qpts))
    chosen, chosen_lam, inside = cand[q, choice], lam[q, choice], inside_cand[q, choice]

    proj = np.clip(chosen_lam, 0.0, None)  # nearest-simplex projection for the outside points
    proj = proj / np.clip(proj.sum(axis=1, keepdims=True), 1e-300, None)
    weights = np.where(inside[:, None], chosen_lam, proj)
    return chosen.astype(np.int64), weights, inside


def _locate_barycentric(src_pts: np.ndarray, src_cells: np.ndarray, qpts: np.ndarray, *, tol: float, k: int):
    """P1 stencil form of :func:`_locate_in_cells`: return ``(idx, weights, inside)`` where ``idx``
    ``(Q, D+1)`` are the chosen simplex's **source-vertex indices** (the P1 interpolation stencil).
    Unchanged public behaviour — the state-transfer / resample / moving-boundary callers use this."""
    cell_idx, weights, inside = _locate_in_cells(src_pts, src_cells, qpts, tol=tol, k=k)
    return src_cells[cell_idx].astype(np.int64), weights, inside


def _one_ring_cells(cells: np.ndarray, n_vert: int) -> tuple[np.ndarray, np.ndarray]:
    """``(cand, mask)`` -- for each vertex, the cells incident to it, padded to a fixed width.

    Connectivity only, so a connectivity-preserving march hoists this ONCE and it is a compile-time
    constant. ``cand`` is ``(n_vert, R)`` cell indices (padding repeats the vertex's first cell, so a
    padded slot is a *valid* simplex and never produces a singular solve) and ``mask`` ``(n_vert, R)``
    marks the real entries."""
    cells = np.asarray(cells, dtype=np.int64)
    n_local = cells.shape[1]
    vid = cells.reshape(-1)
    cid = np.repeat(np.arange(cells.shape[0], dtype=np.int64), n_local)
    order = np.argsort(vid, kind="stable")
    vid, cid = vid[order], cid[order]
    counts = np.bincount(vid, minlength=n_vert)
    width = int(counts.max()) if len(counts) else 1
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    cand = np.zeros((n_vert, width), dtype=np.int64)
    mask = np.zeros((n_vert, width), dtype=bool)
    for j in range(width):
        take = counts > j
        cand[take, j] = cid[starts[take] + j]
        mask[take, j] = True
    # a vertex in no cell would leave an all-invalid row; point it at cell 0 so the solve stays regular
    cand[~mask.any(axis=1), :] = 0
    cand = np.where(mask, cand, cand[:, :1])
    return cand, mask


def _locate_in_one_ring_jax(src_pts, cells, cand, mask, qpts, *, tol: float = 1e-9):
    """P1 stencil + weights for each query point, searching only its own vertex's incident cells.

    The moving-mesh transfer asks a special question: query ``i`` is vertex ``i`` displaced by less than
    an element, on a mesh with the SAME connectivity. Its containing simplex is therefore in vertex
    ``i``'s one-ring, so the candidate set is fixed by connectivity instead of found by a KD-tree --
    static shapes, and differentiable in the positions. Same barycentric algebra as
    :func:`_locate_in_cells`, batched over a static candidate axis.

    Returns ``(idx, weights, escaped, cell)``: ``idx`` ``(Q, D+1)`` source-vertex indices, ``weights``
    ``(Q, D+1)`` barycentric coordinates, ``escaped`` ``(Q,)`` True where the point fell outside every
    candidate (the caller decides; the host route clamps such points silently), and ``cell`` ``(Q,)`` the
    chosen source cell. A P1 caller wants ``idx``, since a P1 field's DOFs *are* the vertices; a
    higher-order one wants ``cell``, to index that cell's row of the P{k} connectivity. Both are returned
    rather than derived by the caller, because re-deriving the cell from its vertex set is a search where
    here it is already known."""
    import jax.numpy as jnp

    dim = qpts.shape[1]
    cell_v = jnp.asarray(cells)[jnp.asarray(cand)]  # (Q, R, D+1)
    V = src_pts[cell_v]  # (Q, R, D+1, D)
    v0 = V[:, :, 0, :]
    T = jnp.swapaxes(V[:, :, 1:, :] - v0[:, :, None, :], 2, 3)  # columns v_i - v0
    rhs = qpts[:, None, :] - v0
    det = jnp.linalg.det(T)
    safe = jnp.abs(det) > 1e-300
    Tsafe = jnp.where(safe[..., None, None], T, jnp.eye(dim, dtype=T.dtype))
    lam_rest = jnp.linalg.solve(Tsafe, rhs[..., None])[..., 0]
    lam = jnp.concatenate([(1.0 - lam_rest.sum(axis=-1))[..., None], lam_rest], axis=-1)  # (Q, R, D+1)

    valid = jnp.asarray(mask) & safe
    inside = valid & jnp.all(lam >= -tol, axis=-1)
    worst = jnp.where(valid, jnp.min(lam, axis=-1), -jnp.inf)  # best-effort pick when nothing contains it
    choice = jnp.where(jnp.any(inside, axis=1), jnp.argmax(inside.astype(jnp.int32), axis=1), jnp.argmax(worst, axis=1))
    q = jnp.arange(qpts.shape[0])
    chosen_lam, chosen_idx = lam[q, choice], cell_v[q, choice]
    hit = jnp.any(inside, axis=1)

    proj = jnp.clip(chosen_lam, 0.0, None)  # nearest-simplex projection, as the host route does
    proj = proj / jnp.clip(proj.sum(axis=1, keepdims=True), 1e-300, None)
    return chosen_idx, jnp.where(hit[:, None], chosen_lam, proj), ~hit, jnp.asarray(cand)[q, choice]


def _cell_patch_cells(cells: np.ndarray, n_vert: int) -> tuple[np.ndarray, np.ndarray]:
    """``(cand, mask)`` -- for each CELL, the cells sharing at least one vertex with it, padded to a fixed
    width. The cell-level twin of :func:`_one_ring_cells`.

    :func:`_l2_transfer_jax` asks a different question from the vertex transfer: its query points are a
    cell's *quadrature* points on the moved mesh, not a moved vertex, so the containing old cell is not in
    any one vertex's ring -- it is in the union of the rings of the cell's own vertices. Connectivity only,
    so a connectivity-preserving march hoists this once and it rides the scan as a constant."""
    cells = np.asarray(cells, dtype=np.int64)
    ring, rmask = _one_ring_cells(cells, n_vert)
    # union the incident-cell lists of a cell's own vertices, per cell
    per = ring[cells]  # (n_cells, n_local, R)
    flat = np.where(rmask[cells], per, -1).reshape(cells.shape[0], -1)  # (n_cells, n_local*R)

    # Row-wise unique WITHOUT a Python loop. A per-row `np.unique` cost 87.7 ms at 23k cells and grows
    # linearly (~0.4 s at 100k), which is real on this library's dominant cost, the build. Sorting puts
    # duplicates adjacent and the -1 padding first, so "keep" is a shifted comparison, and a stable
    # argsort on `~keep` compacts the survivors left in one gather.
    s = np.sort(flat, axis=1)
    keep = s >= 0
    keep[:, 1:] &= s[:, 1:] != s[:, :-1]
    order = np.argsort(~keep, axis=1, kind="stable")
    s, keep = np.take_along_axis(s, order, axis=1), np.take_along_axis(keep, order, axis=1)
    width = max(int(keep.sum(axis=1).max()), 1)
    cand, mask = s[:, :width].copy(), keep[:, :width]
    # padding repeats the row's first candidate, so a padded slot is a VALID simplex and never
    # produces a singular solve -- the same convention `_one_ring_cells` uses
    cand = np.where(mask, cand, cand[:, :1])
    return cand, mask


def _l2_transfer_jax(
    X_old, X_new, cells, dim, u, off, *, orders=None, vecs=None, cells_f=None, qdeg=None, tol=None, maxiter: int = 200
):
    r"""Carry a finite-element state from the old mesh onto the moved one by **conservative L2
    projection** -- the Galerkin transfer ``M(X_new) u_new = b``, ``b_i = ∫_{Ω_new} u_old φ_i^new``. Pure
    JAX, static shapes, differentiable in both meshes and in ``u``.

    Handles **any nodal-Lagrange order and value shape**, per field: pass ``orders`` / ``vecs`` /
    ``cells_f`` from :func:`_field_layout` (defaults describe scalar P1). Two things make the
    higher-order case cheap rather than a rewrite. The mesh geometry is **P1 whatever the field order**
    -- a moved simplex stays straight-sided -- so the quadrature map and the point location are shared by
    every field and computed once. And the P{k} connectivity is **unchanged by the move**, because the
    move preserves topology; only the vertex positions differ, so the seed assembly's ``cells_f`` stays
    valid for the whole march and the new DOF *coordinates* are never needed at all.

    Replaces a pointwise re-interpolation (``u_new[i] = u_old(x_new[i])``, still available as
    :func:`_locate_in_one_ring_jax`, which remains this function's point-location core and its test
    oracle). Pointwise sampling is neither conservative nor optimal, and it is applied once per step so
    its error **accumulates**: on a rigid translation carrying a Gaussian bump with ``κ ≈ 0`` -- where the
    exact answer is "the field never changes at a fixed spatial point" -- the peak fell 27.6 % over 2
    steps and 33.0 % over 16. Refining ``dt`` made the answer *worse*, which is the opposite of what a
    first-order scheme should do.

    Measured against the route it replaces (same problem, same mesh):

    ============================  ==================  ==================
    rigid translation, peak       pointwise           L2
    ============================  ==================  ==================
    2 steps                       -27.6 %             **-11.5 %**
    16 steps                      -33.0 %             **-9.1 %**
    ============================  ==================  ==================

    -- three times less loss, and the sign of the *trend* flips: refining ``dt`` now helps, where before
    every extra step cost more. A zero-velocity march is untouched either way (0.00 %), and transferring
    between two identical meshes returns the field bit-for-bit, which is the sharp self-test: the system
    is then ``M u_new = M u_old``.

    **Conservation is algebraic but not exact.** ``Σ_i φ_i = 1`` gives ``Σ_i b_i = ∫_{Ω_new} u_old`` and
    ``Σ_i (M u_new)_i = ∫_{Ω_new} u_new`` by construction, so the projection conserves whatever the load
    vector integrated. What it integrated is only accurate to the quadrature, and the integrand
    ``u_old·φ_i^new`` is piecewise-quadratic **with kinks** wherever an old cell edge crosses a new cell --
    which no smooth rule integrates exactly. Measured on a fixed domain (interior vertices jittered, so
    ``Ω`` cannot change), relative drift in ``∫u``:

    ============  ================  ================
    jitter        pointwise         L2
    ============  ================  ================
    0.004         3.2e-03           **1.5-2.5e-04**
    0.012         8.6e-03           **2.4-5.8e-04**
    ============  ================  ================

    The spread is across ``qdeg`` 2/3/4 and is **not monotone in it** -- the residual is set by the kinks,
    not by the rule, so raising the degree buys nothing. Exact conservation would need the old/new
    supermesh (Farrell & Maddison, *Conservative interpolation between volume meshes*, CMAME **200**
    (2011) 89-100), whose intersections are variable-size polygons and therefore not traceable. ``qdeg=3``
    is the default because it is exact for the *unbroken* quadratic and measures no worse than 4.

    **What this does not fix.** An L2 projection is still a contraction, so it reduces the peak loss
    rather than removing it (and, not being monotone, it overshoots slightly where the pointwise route
    undershoots). Removing it entirely means not transferring at all -- holding the DOFs on the moving
    vertices (Lagrangian) and carrying the motion in an ALE ``-w·∇u`` term instead, which is a different
    semidiscretisation, not a better transfer.

    **Cost.** More location work than the pointwise route (a cell's quadrature points against a wider
    patch, rather than one moved vertex against its own ring) plus a mass solve, but the march is
    dominated by the θ-step: end to end, 513 vertices over 10 steps measured 2.03 s -> 2.24 s, i.e. ~10 %,
    for an identical mesh trajectory.

    Steps, all of them traceable: the reference rule comes from ``basix.make_quadrature`` (the same
    builder the assembler uses); the physical points are affine in ``X_new``; each is located in the OLD
    mesh over its cell's fixed-width patch (:func:`_cell_patch_cells`); the load vector is a
    ``segment_sum`` scatter; and the mass is applied matrix-free under Jacobi-CG, the vector case being
    ``M ⊗ I`` so every component rides one solve.

    The test functions are tabulated on the host, at the fixed reference quadrature points. The old field
    has to be read at the **located** points, which are tracers, so those go through
    :func:`_lagrange_monomial_coeffs` instead. One basis, two evaluation routes, only because one input is
    static and the other is not -- and both are built from the single ``_lagrange_basix`` element, so they
    cannot drift apart.

    ``Ω_new ⊄ Ω_old`` where a boundary moves outward: such quadrature points clamp to the nearest simplex
    exactly as the pointwise route does, and conservation then holds against that clamped extension rather
    than the true integral. The returned ``escaped`` flag reports it.

    Returns ``(u_new, escaped)``.
    """
    import jax
    import jax.numpy as jnp
    from basix import CellType, make_quadrature

    cells_j = jnp.asarray(cells)
    n_cells = cells.shape[0]
    n_fields = len(off) - 1
    orders = [1] * n_fields if orders is None else [int(o) for o in orders]
    vecs = [1] * n_fields if vecs is None else [int(v) for v in vecs]
    cells_f = [cells] * n_fields if cells_f is None else [np.asarray(c) for c in cells_f]

    # The mass ∫φᵢφⱼ is degree 2k and MUST be integrated exactly -- under-integrate it and ``M`` is not
    # the mass matrix, so the solve is not a projection at all. One rule serves every field (a
    # higher-degree rule is merely wasted on a lower-order one). The load vector cannot be exact at any
    # degree, because its integrand has kinks; 2k is what makes the operator right, not the data.
    if qdeg is None:
        qdeg = max(3, 2 * max(orders))
    qp, qw = make_quadrature(CellType.triangle if dim == 2 else CellType.tetrahedron, int(qdeg))
    qp, qw = np.asarray(qp, dtype=np.float64), np.asarray(qw, dtype=np.float64)
    qp_j, qw_j = jnp.asarray(qp, dtype=X_new.dtype), jnp.asarray(qw, dtype=X_new.dtype)

    # physical quadrature points on the NEW mesh: affine in the moved vertices. The GEOMETRY is P1
    # whatever the field order -- a moved simplex stays straight-sided -- so this is the same map for
    # every field, and only the basis tabulated on it changes.
    v_new = X_new[cells_j]  # (n_cells, n_loc, dim)
    J_new = jnp.stack([v_new[:, i + 1] - v_new[:, 0] for i in range(dim)], axis=2)  # columns are edges
    detJ = jnp.abs(jnp.linalg.det(J_new))  # (n_cells,)
    # x(q) = v0 + sum_d xi_d * edge_d. `J_new[c, :, d]` IS edge d (columns are edges), so the contraction
    # runs over J's LAST axis -- the batched form of the assembler's `verts[0] + qp @ J.T`.
    xq = v_new[:, 0][:, None, :] + jnp.einsum("qd,ced->cqe", qp_j, J_new)  # (n_cells, nq, dim)

    # Locate every new quadrature point in the OLD mesh. Done ONCE and shared by every field: the point
    # set is the field-independent geometry, and only what is evaluated there differs.
    cand, cmask = _cell_patch_cells(np.asarray(cells), X_new.shape[0])
    _idx, w, esc, src_cell = _locate_in_one_ring_jax(
        X_old,
        cells,
        np.repeat(cand, qp.shape[0], axis=0),
        np.repeat(cmask, qp.shape[0], axis=0),
        xq.reshape(-1, dim),
    )
    w = jnp.asarray(w, dtype=X_new.dtype)
    # the located point's REFERENCE coordinates in its old cell: barycentric lambda_1..lambda_d, which is
    # exactly the chart `_tabulate_lagrange_at` / `_lagrange_monomial_coeffs` are written on
    xi_src = w[:, 1:]

    if tol is None:
        tol = max(1e-13, 100.0 * float(jnp.finfo(X_new.dtype).eps))

    outs = []
    for f in range(n_fields):
        k, vec, cf = orders[f], vecs[f], jnp.asarray(cells_f[f])
        n_nodes = (off[f + 1] - off[f]) // vec
        phi_q = jnp.asarray(_tabulate_lagrange_at(dim, k, qp), dtype=X_new.dtype)  # (nq, n_dof) -- host

        # The TEST functions sit at fixed reference quadrature points, so basix tabulates them on the
        # host. The OLD field must be read at the LOCATED points, which are traced, so those go through
        # the polynomial form. Same basis, two evaluation routes only because one input is static and the
        # other is not -- both come from the one `_lagrange_basix` builder, so they cannot disagree.
        exps, coef = _lagrange_monomial_coeffs(dim, k)
        phi_src = _eval_lagrange_traced(xi_src, exps, coef)  # (n_cells*nq, n_dof)

        blk = u[off[f] : off[f + 1]].reshape(n_nodes, vec)  # node-major: node*vec + comp
        # u_old at every new quadrature point, all components at once
        uq = jnp.einsum("pn,pnv->pv", phi_src, blk[cf[src_cell]]).reshape(n_cells, -1, vec)

        # Mass on the NEW mesh from the SAME rule as the load vector -- not a closed form. Two
        # computations of one integral disagree at round-off, and that difference IS the answer when the
        # meshes coincide: `b - M u_old` must be exactly zero there. With two routes it was ~1e-7 in
        # float32 and a still mesh drifted instead of transferring untouched.
        m_loc = jnp.einsum("q,c,qi,qj->cij", qw_j, detJ, phi_q, phi_q)
        m_diag = jax.ops.segment_sum(jnp.einsum("cii->ci", m_loc).reshape(-1), cf.reshape(-1), num_segments=n_nodes)
        jac = 1.0 / jnp.where(jnp.abs(m_diag) > 1e-30, m_diag, 1.0)

        def mass_apply(z, _m=m_loc, _cf=cf, _n=n_nodes):
            """The scalar mass applied to every component at once -- the vector mass is M (x) I."""
            out_c = jnp.einsum("cij,cjv->civ", _m, z[_cf])
            return jax.ops.segment_sum(out_c.reshape(-1, z.shape[-1]), _cf.reshape(-1), num_segments=_n)

        # b_i = sum_c sum_q w_q |det J_c| u_old(xq) phi_i(qp_q)
        b_c = jnp.einsum("q,c,cqv,qi->civ", qw_j, detJ, uq, phi_q)
        b = jax.ops.segment_sum(b_c.reshape(-1, vec), cf.reshape(-1), num_segments=n_nodes)
        sol = jax.scipy.sparse.linalg.cg(
            mass_apply, b, x0=blk, tol=tol, atol=0.0, maxiter=maxiter, M=lambda z, _j=jac: _j[:, None] * z
        )[0]
        outs.append(sol.reshape(-1))
    return jnp.concatenate(outs), esc.reshape(n_cells, -1)


def transfer_solution(
    source_domain: Any, values: Any, target_domain: Any, *, fill: Any = "nearest", tol: float = 1e-9, k: int = 32
):
    r"""Transfer a nodal field from one simplicial mesh to another by piecewise-linear (barycentric)
    interpolation -- the mesh-to-mesh **solution transfer** an adaptive / moving-mesh time loop needs
    to carry state across a remesh (:func:`remesh_with_mmg`).

    For each vertex :math:`x` of ``target_domain`` the containing simplex of ``source_domain`` is
    located and the field evaluated from its barycentric coordinates
    :math:`u_\text{new}(x) = \sum_i \lambda_i(x)\,\text{values}[c_i]`. This is **exact for a P1 field**
    (and for any affine field, to machine precision); for a higher-order source it is the P1 (linear)
    interpolant sampled at the target vertices -- first-order accurate, *not* the full basis.

    Scope: 2-D triangle / 3-D tet meshes. ``values`` may be scalar ``(n_src,)``, vector ``(n_src, c)``,
    or any ``(n_src, ...)`` -- stack several fields on the trailing axis to transfer them in one
    location pass. The point location runs on the **host** (NumPy + a SciPy ``cKDTree`` over cell
    centroids -- a structural step with no gradient); the interpolation **apply is pure JAX**, so
    gradients flow through ``values`` (a field parameter carried across a remesh in an inverse loop
    stays differentiable).

    Parameters
    ----------
    source_domain, target_domain
        Meshed 2-D/3-D ``jno`` domains (``.mesh.points`` + triangle/tetra ``cells_dict``).
    values
        Nodal field on ``source_domain``'s **vertices**: leading axis ``n_src == n_source_vertices``.
    fill
        How to treat target vertices that fall outside the source mesh (numerically, or genuinely
        non-overlapping domains): ``"nearest"`` (default) projects onto the nearest source simplex;
        ``"error"`` raises, naming the count; a float substitutes that constant there. A remesh of the
        same domain preserves the boundary, so outside points are rare and ``"nearest"`` is safe;
        ``"error"`` is the strict check for a genuinely mismatched target.
    tol
        Barycentric inside-tolerance (a point on a shared face/edge belongs to every incident cell,
        which all agree on the value).
    k
        Candidate cells tested per query point (nearest centroids). Raise for strongly anisotropic
        meshes where the containing cell's centroid may not be among the ``k`` nearest.

    Returns
    -------
    The field on ``target_domain``'s vertices, shape ``(n_target_vertices, ...)`` and the dtype of
    ``values`` (complex is preserved).

    Notes
    -----
    A shape/nodes mismatch (``values`` not aligned with the source vertices) or a dimension mismatch
    raises rather than silently mis-transferring. Reference: barycentric interpolation over a
    simplicial complex; KD-tree candidate location is the standard scattered-mesh interpolation route.
    """
    import jax.numpy as jnp

    dim = int(source_domain.dimension)
    if dim not in (1, 2, 3):
        raise NotImplementedError(f"transfer_solution supports 1D line and 2D/3D simplicial meshes; got {dim}.")
    if int(target_domain.dimension) != dim:
        raise ValueError(f"source ({dim}D) and target ({int(target_domain.dimension)}D) mesh dimensions differ.")
    if fill not in ("nearest", "error") and not isinstance(fill, (int, float)):
        raise ValueError(f"fill must be 'nearest', 'error', or a numeric constant; got {fill!r}.")

    # An interval IS a 1-simplex, so the barycentric location core below is dimension-agnostic: in 1D
    # the "barycentric weights" are just the two linear hat values, which is exactly P1 interpolation.
    key = {1: "line", 2: "triangle"}.get(dim, "tetra")
    src_pts = np.asarray(source_domain.mesh.points)[:, :dim].astype(np.float64)
    src_cells = np.asarray(source_domain.mesh.cells_dict[key]).astype(np.int64)
    qpts = np.asarray(target_domain.mesh.points)[:, :dim].astype(np.float64)

    vals = jnp.asarray(values)
    if vals.shape[0] != src_pts.shape[0]:
        raise ValueError(
            f"transfer_solution: values has {vals.shape[0]} rows but the source mesh has "
            f"{src_pts.shape[0]} vertices — it takes a P1 (vertex) nodal field aligned with source_domain."
        )

    idx, weights, inside = _locate_barycentric(src_pts, src_cells, qpts, tol=tol, k=k)  # host
    n_out = int((~inside).sum())
    if n_out and fill == "error":
        raise ValueError(
            f"transfer_solution: {n_out}/{len(qpts)} target vertices fall outside the source mesh. "
            "Pass fill='nearest' to project them onto the nearest source simplex, or fill=<float>."
        )

    rdtype = vals.real.dtype  # keep float32 float32 / complex complex; barycentric weights are real
    out = jnp.einsum("qk,qk...->q...", jnp.asarray(weights, dtype=rdtype), vals[jnp.asarray(idx)])
    if isinstance(fill, (int, float)) and n_out:
        keep = jnp.asarray(inside).reshape((-1,) + (1,) * (out.ndim - 1))
        out = jnp.where(keep, out, jnp.asarray(fill, dtype=vals.dtype))
    return out


def _tabulate_lagrange_at(dim: int, order: int, xi: np.ndarray) -> np.ndarray:
    """The Lagrange P{order} simplex basis tabulated at reference coordinates ``xi`` ``(Q, dim)`` ->
    ``(Q, n_dof)``. Uses the SAME :func:`fem_lagrange._lagrange_basix` builder as the assembler, so the
    returned columns are in the element's DOF order and line up with the recorded P{order} cell
    connectivity. ``order == 1`` is ordinary linear (barycentric) interpolation. Host/basix."""
    from basix import CellType

    from .fem_lagrange import _lagrange_basix

    cell = {1: CellType.interval, 2: CellType.triangle}.get(int(dim), CellType.tetrahedron)
    tab = _lagrange_basix(cell, int(order)).tabulate(0, np.asarray(xi, dtype=np.float64))
    return np.asarray(tab[0, :, :, 0])  # (Q, n_dof): basis values (0th-derivative block, scalar value)


def _lagrange_monomial_coeffs(dim: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    r"""``(exponents, C)`` for evaluating the P{order} simplex basis at **traced** reference coordinates.

    :func:`_tabulate_lagrange_at` goes through basix, which is host-only, so it cannot be called on a
    reference coordinate that is itself a tracer -- which is exactly what the moving-mesh transfer has,
    since the point it must evaluate the old field at is located inside the scan. The basis is a
    polynomial, so its coefficients can be found once on the host and the polynomial evaluated in-trace:

    .. math:: \varphi_j(\xi) = \sum_m \xi^{e_m}\, C_{mj}

    ``C`` is the inverse of the Vandermonde of the monomials at the element's own nodal points, which is
    where this is well conditioned rather than merely correct: Lagrange nodes satisfy
    :math:`\varphi_j(\text{node}_s) = \delta_{sj}`, so the tabulated right-hand side is the identity and
    ``C = A^{-1}`` exactly. The monomial count :math:`\binom{k+d}{d}` equals the Lagrange DOF count, so
    ``A`` is square.

    Nodes and DOF order come from the same :func:`fem_lagrange._lagrange_basix` builder the assembler
    uses, so the columns line up with the recorded P{order} connectivity -- the property that lets the
    result be contracted against a cell's DOFs directly.
    """
    from basix import CellType

    from .fem_lagrange import _lagrange_basix

    cell = {1: CellType.interval, 2: CellType.triangle}.get(int(dim), CellType.tetrahedron)
    nodes = np.asarray(_lagrange_basix(cell, int(order)).points, dtype=np.float64)  # (n_dof, dim)

    exps = np.array(
        [e for e in itertools.product(range(int(order) + 1), repeat=int(dim)) if sum(e) <= int(order)],
        dtype=np.int64,
    )
    if exps.shape[0] != nodes.shape[0]:  # a Lagrange simplex is unisolvent on exactly this monomial set
        raise AssertionError(f"P{order} on a {dim}-simplex: {nodes.shape[0]} nodes vs {exps.shape[0]} monomials.")
    A = np.prod(nodes[:, None, :] ** exps[None, :, :], axis=2)  # (n_dof, n_mon)
    return exps, np.linalg.inv(A)


def _eval_lagrange_traced(xi, exps, coeffs):
    """P{order} basis values at traced reference coordinates ``xi`` ``(..., dim)`` -> ``(..., n_dof)``.

    The in-trace half of :func:`_lagrange_monomial_coeffs`.

    The monomials are built from **cumulative products**, never from ``xi ** e``. ``xi ** 0`` has the
    right *value* (1) everywhere, but its derivative is taken by the general power rule as
    ``0 · xi**(-1)``, which is ``0 · inf = NaN`` at ``xi == 0`` -- and a reference coordinate is exactly
    zero whenever a quadrature point lands on a cell edge, which some rules do. The value is unaffected,
    so this shows up only under differentiation: it put NaNs in ``d(march)/dX₀`` while every forward test
    stayed green. Repeated multiplication has the same value and is differentiable everywhere."""
    import jax.numpy as jnp

    exps = np.asarray(exps, dtype=np.int64)
    dim, max_e = exps.shape[1], int(exps.max(initial=0))
    pows = [jnp.ones_like(xi)]  # pows[k][..., a] == xi[..., a] ** k
    for _ in range(max_e):
        pows.append(pows[-1] * xi)
    stack = jnp.stack(pows, axis=0)  # (max_e + 1, ..., dim)

    mon = jnp.ones(xi.shape[:-1] + (exps.shape[0],), dtype=xi.dtype)
    for a in range(dim):
        mon = mon * jnp.moveaxis(stack[..., a], 0, -1)[..., exps[:, a]]
    return mon @ jnp.asarray(coeffs, dtype=xi.dtype)


def _eval_fe_fields_at_points(
    src_pts_p1: np.ndarray,
    src_cells_p1: np.ndarray,
    src_state: Any,
    src_offsets: Any,
    src_orders: Any,
    src_cells_f: Any,
    field_vecs: Any,
    query_pts: Any,
    *,
    dim: int,
    tol: float = 1e-9,
    k: int = 32,
    fill: Any = "nearest",
) -> list:
    """Evaluate an OLD finite-element solution at NEW per-field DOF coordinates -- the basis-aware,
    value-shape-aware generalisation of :func:`transfer_solution`, used to carry state across a remesh
    for vector / higher-order (P2) / mixed (Taylor-Hood) fields.

    For each field ``i``: locate its NEW sample points ``query_pts[i]`` ``(Q_i, dim)`` in the OLD **P1
    base** mesh (:func:`_locate_in_cells` -> containing cell + reference coords), tabulate the OLD
    element's P{``src_orders[i]``} basis there, and contract with that cell's OLD DOFs -- for all
    ``field_vecs[i]`` components at once (node-major ``node*vec + comp`` layout, matching the assembler).
    Returns a list of ``(Q_i, vec_i)`` arrays; ``reshape(-1)`` restores each field's flat block. Point
    location is host; the interpolation apply stays differentiable in ``src_state`` (like
    :func:`transfer_solution`). Points outside the old mesh use the nearest-simplex projection (``fill``
    default 'nearest', first-order); a numeric ``fill`` substitutes a constant there instead."""
    import jax.numpy as jnp

    state = jnp.asarray(src_state).reshape(-1)
    rdtype = state.real.dtype
    off = [int(x) for x in src_offsets]
    out = []
    for i in range(len(off) - 1):
        vec_i = int(field_vecs[i])
        blk = state[off[i] : off[i + 1]].reshape(-1, vec_i)  # (n_nodes_i, vec_i)
        cell_idx, weights, inside = _locate_in_cells(src_pts_p1, src_cells_p1, np.asarray(query_pts[i]), tol=tol, k=k)
        phi = _tabulate_lagrange_at(dim, int(src_orders[i]), weights[:, 1:])  # (Q_i, n_dof_i) at ref coords
        cells_i = np.asarray(src_cells_f[i])[cell_idx]  # (Q_i, n_dof_i): the old DOF ids of the containing cell
        vals = jnp.einsum("qn,qnc->qc", jnp.asarray(phi, dtype=rdtype), blk[jnp.asarray(cells_i)])  # (Q_i, vec_i)
        if isinstance(fill, (int, float)) and bool((~inside).any()):
            vals = jnp.where(jnp.asarray(inside)[:, None], vals, jnp.asarray(fill, dtype=state.dtype))
        out.append(vals)
    return out


# ---------------------------------------------------------------------------
# Error estimation -- Zienkiewicz-Zhu recovery indicator (scalar P1)
# ---------------------------------------------------------------------------
def zz_error_indicators(domain: Any, u_vertex: np.ndarray) -> tuple[np.ndarray, float]:
    """Zienkiewicz-Zhu recovery-based per-element error indicators for a P1 field.

    The raw FEM gradient is elementwise-constant and discontinuous across element
    boundaries.  ZZ recovers a smoother (continuous, superconvergent) gradient
    ``g*`` by area-weighted averaging of the incident element gradients at each
    vertex; the elementwise gap ``|g* - grad u_h|`` is the error indicator.

    Parameters
    ----------
    domain
        A meshed domain whose P1 gradient geometry is read via
        :func:`jno._fem._fe_element_gradient_data`.
    u_vertex
        ``(n_vertices,)`` nodal values of a SCALAR field, or ``(n_vertices, vec)`` for a VECTOR field
        (one column per component). Values may be complex (Helmholtz).

    Returns
    -------
    (eta, global_estimate)
        ``eta`` is ``(n_cells,)`` non-negative indicators; ``global_estimate`` is
        ``sqrt(sum eta**2)``, an estimate of the global energy-norm error.

    For a vector field the ZZ energy-norm error generalises to the **sum over components** of the
    per-component recovered-gradient gaps (the Frobenius norm of the recovered gradient-*tensor* gap), so
    one scalar indicator per cell still drives refinement.

    Reference: Zienkiewicz & Zhu (1987), IJNME 24, 337-357.
    """
    u = np.asarray(u_vertex)
    comps = [u[:, c] for c in range(u.shape[1])] if (u.ndim == 2 and u.shape[1] > 1) else [u.reshape(-1)]

    # elementwise gap, integrated over the cell (centroid rule: P1 g_cell is exact there and the centroid
    # value of the P1-recovered field is the vertex mean), summed over the field's components. For a COMPLEX
    # component the gap is complex and the energy-norm uses its modulus ``|g* - grad u_h|^2`` (exact for
    # reals since ``|x|^2 == x^2``) -- so one indicator drives real, complex and vector fields alike.
    eta2 = None
    for uc in comps:
        g_star, g_cell, area, cells = _recover_nodal_gradient(domain, uc)
        g_star_centroid = g_star[cells].mean(axis=1)  # (n_cells, dim)
        e = area * np.sum(np.abs(g_star_centroid - g_cell) ** 2, axis=1)  # (n_cells,), real
        eta2 = e if eta2 is None else eta2 + e
    eta = np.sqrt(np.maximum(eta2, 0.0))
    return eta, float(np.sqrt(eta2.sum()))


def _p1_element_gradients(domain: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant P1 shape-function gradients per simplex, computed geometrically.

    Returns ``(grad, measure, cells)``: ``grad`` is ``(n_cells, dim+1, dim)`` (the gradient of
    each local barycentric shape function, constant over the element), ``measure`` the
    ``(n_cells,)`` area (2D) / volume (3D), and ``cells`` the connectivity.  Exact for P1 on
    triangles and tetrahedra and dimension-general, so the recovery estimator works in both 2D
    and 3D without the native FEM context.
    """
    dim = int(domain.dimension)
    pts = np.asarray(domain.mesh.points)[:, :dim].astype(np.float64)
    cells = np.asarray(domain.mesh.cells_dict[_simplex_cell_key(dim)])
    v = pts[cells]  # (n_cells, dim+1, dim)
    edge = v[:, 1:, :] - v[:, :1, :]  # (n_cells, dim, dim): rows are (v_i - v_0), i=1..dim
    einv = np.linalg.inv(edge)  # column j = grad of barycentric lambda_{j+1}
    measure = np.abs(np.linalg.det(edge)) / _simplex_measure_divisor(dim)  # simplex volume = |det|/d!

    grad = np.zeros((cells.shape[0], dim + 1, dim))
    grad[:, 1:, :] = np.transpose(einv, (0, 2, 1))  # grad lambda_i = column (i-1) of E^{-1}
    grad[:, 0, :] = -grad[:, 1:, :].sum(axis=1)  # lambda_0 = 1 - sum(lambda_i)
    return grad, measure, cells


def _recover_nodal_gradient(domain: Any, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Patch-recovered (superconvergent) nodal gradient of a P1 ``field``.

    Returns ``(g_star, g_cell, measure, cells)``: ``g_star`` is the ``(n_vert, dim)`` recovered
    nodal gradient (measure-weighted average of incident element gradients), ``g_cell`` the raw
    ``(n_cells, dim)`` elementwise-constant gradient, ``measure`` the ``(n_cells,)`` cell
    areas/volumes, and ``cells`` the ``(n_cells, n_local)`` connectivity.  Works for 2D triangle
    and 3D tetrahedron P1 meshes.
    """
    sg, measure, cells = _p1_element_gradients(domain)  # geometric, exact for P1 simplices
    sg = np.asarray(sg)  # (n_cells, n_local, dim)
    cells = np.asarray(cells)
    n_cells, _, dim = sg.shape
    area = measure  # (n_cells,)

    f = np.asarray(field).reshape(-1)
    if f.shape[0] < int(cells.max()) + 1:
        raise ValueError("field is shorter than the mesh vertex count; expected one value per P1 vertex.")

    g_cell = np.einsum("cld,cl->cd", sg, f[cells])  # (n_cells, dim): constant P1 gradient (complex if f is)
    n_vert = f.shape[0]
    g_star = np.zeros((n_vert, dim), dtype=g_cell.dtype)  # keep complex gradients intact (don't drop Im)
    wsum = np.zeros(n_vert)
    for lv in range(cells.shape[1]):
        idx = cells[:, lv]
        np.add.at(g_star, idx, area[:, None] * g_cell)
        np.add.at(wsum, idx, area)
    g_star /= np.maximum(wsum[:, None], 1e-300)
    return g_star, g_cell, area, cells


def recover_hessian(domain: Any, u_vertex: np.ndarray) -> np.ndarray:
    """Recovered nodal Hessian ``(n_vert, dim, dim)`` of a P1 field by *double* gradient
    recovery: recover the nodal gradient, then recover the gradient of each of its
    components; symmetrise. The Hessian controls the P1 interpolation error and is the basis
    of the anisotropic metric (:func:`hessian_metric`)."""
    g_star, _, _, _ = _recover_nodal_gradient(domain, u_vertex)
    dim = g_star.shape[1]
    cols = [_recover_nodal_gradient(domain, g_star[:, k])[0] for k in range(dim)]  # each (n_vert, dim)
    H = np.stack(cols, axis=1)  # (n_vert, dim, dim): H[:, k, :] = grad of g_star[:, k]
    return 0.5 * (H + np.transpose(H, (0, 2, 1)))


def hessian_metric(
    domain: Any,
    u_vertex: np.ndarray,
    *,
    target_complexity: float,
    hmin: float,
    hmax: float,
) -> np.ndarray:
    """Anisotropic metric tensor from the recovered Hessian, for :func:`remesh_with_mmg`.

    Builds, per vertex, the metric ``M = |H|`` (Hessian eigenvectors, absolute eigenvalues),
    globally scaled so the mesh *complexity* ``∫ sqrt(det M)`` equals ``target_complexity``
    (≈ the target vertex count), then clamps the edge sizes to ``[hmin, hmax]``. The metric's
    eigenvectors align triangles with the solution's curvature and its eigenvalues set the
    size along each -- thin, stretched elements across a directional feature, which resolve
    it at a fraction of the isotropic cost.

    Returns packed symmetric tensors: ``(n_vert, 3)`` ``(m11, m12, m22)`` in 2D, or
    ``(n_vert, 6)`` ``(m11, m12, m13, m22, m23, m33)`` in 3D.

    Reference: Alauzet & Loseille, *Metric-based anisotropic mesh adaptation* (2010).
    """
    dim = int(domain.dimension)
    H = recover_hessian(domain, u_vertex)  # (n_vert, dim, dim)
    evals, evecs = np.linalg.eigh(H)  # ascending eigenvalues, orthonormal eigenvectors
    lam = np.abs(evals)  # |H|: interpolation error ~ |curvature|
    # floor so det > 0 (a flat direction otherwise gives an infinite size)
    lam = np.maximum(lam, 1e-12 * lam.max(axis=1, keepdims=True).clip(min=1e-300))

    # normalize complexity: with M = s*|H|, complexity = s^(dim/2) * sum sqrt(det|H|)*area_v.
    _, _, area_cell, cells = _recover_nodal_gradient(domain, u_vertex)
    n_vert = H.shape[0]
    area_v = np.zeros(n_vert)
    for lv in range(cells.shape[1]):
        np.add.at(area_v, cells[:, lv], area_cell / cells.shape[1])
    det_raw = np.sqrt(np.prod(lam, axis=1))  # sqrt(det|H|) per vertex
    complexity_raw = float(np.sum(det_raw * area_v))
    # Mmg produces ~`_VERTS_PER_COMPLEXITY[dim]` vertices per unit metric-complexity
    # (empirically ~1.5 in 2D, ~2.2 in 3D), so aim for a complexity that yields
    # `target_complexity` *vertices* -- keeps the DOF budget meaningful in both dimensions.
    metric_complexity = target_complexity / _VERTS_PER_COMPLEXITY[dim]
    s = (metric_complexity / max(complexity_raw, 1e-300)) ** (2.0 / dim)
    lam = s * lam

    # clamp eigenvalues to the size window [hmin, hmax] (size = 1/sqrt(lambda))
    lam = np.clip(lam, 1.0 / hmax**2, 1.0 / hmin**2)

    # reassemble M = V diag(lam) V^T and pack its upper triangle (Mmg tensor ordering)
    M = np.einsum("vij,vj,vkj->vik", evecs, lam, evecs)  # (n_vert, dim, dim)
    return np.stack([M[:, i, j] for i, j in _sym_tensor_indices(dim)], axis=1)


# ---------------------------------------------------------------------------
# Marking -- Dörfler bulk (equilibration) strategy
# ---------------------------------------------------------------------------
def dorfler_mark(eta: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Return indices of the smallest cell set carrying a ``theta`` fraction of the error.

    Bulk (Dörfler) marking: pick the fewest elements whose summed squared indicator
    is at least ``theta * total``.  ``theta`` near 0 marks aggressively few (fast,
    more iterations); near 1 marks almost everything (close to uniform).

    Reference: Dörfler (1996), SIAM J. Numer. Anal. 33, 1106-1124.
    """
    eta = np.asarray(eta).reshape(-1)
    eta2 = eta**2
    total = eta2.sum()
    if total <= 0.0:
        return np.asarray([], dtype=np.int64)
    order = np.argsort(eta2)[::-1]
    cum = np.cumsum(eta2[order])
    k = int(np.searchsorted(cum, theta * total) + 1)
    return np.sort(order[:k]).astype(np.int64)


def size_field_from_marks(domain: Any, marked_cells: np.ndarray, *, refine_factor: float = 2.0) -> np.ndarray:
    """Per-vertex target edge size that halves (``/refine_factor``) at marked cells.

    The current local size at a vertex is its mean incident edge length; vertices
    touched by a marked cell get that divided by ``refine_factor``, so Mmg refines
    exactly the flagged region while leaving the rest near its current resolution.
    """
    dim = int(domain.dimension)
    pts = np.asarray(domain.mesh.points)[:, :dim]
    tris = np.asarray(domain.mesh.cells_dict[_simplex_cell_key(dim)])
    n_vert = pts.shape[0]

    # mean incident edge length per vertex
    acc = np.zeros(n_vert)
    cnt = np.zeros(n_vert)
    n_local = tris.shape[1]
    for a in range(n_local):
        for b in range(a + 1, n_local):
            va, vb = tris[:, a], tris[:, b]
            length = np.linalg.norm(pts[va] - pts[vb], axis=1)
            np.add.at(acc, va, length)
            np.add.at(acc, vb, length)
            np.add.at(cnt, va, 1.0)
            np.add.at(cnt, vb, 1.0)
    h0 = acc / np.maximum(cnt, 1.0)
    h0[cnt == 0] = h0[cnt > 0].mean() if np.any(cnt > 0) else 1.0

    size = h0.copy()
    marked_cells = np.asarray(marked_cells, dtype=np.int64)
    if marked_cells.size:
        marked_vertices = np.unique(tris[marked_cells].reshape(-1))
        size[marked_vertices] = h0[marked_vertices] / float(refine_factor)
    return size


def _mean_edge_length(domain: Any) -> float:
    """Mean triangle-edge length of the current mesh (a size scale for metric clamps)."""
    dim = int(domain.dimension)
    pts = np.asarray(domain.mesh.points)[:, :dim]
    tris = np.asarray(domain.mesh.cells_dict[_simplex_cell_key(dim)])
    n_local = tris.shape[1]
    lengths = [
        np.linalg.norm(pts[tris[:, a]] - pts[tris[:, b]], axis=1) for a in range(n_local) for b in range(a + 1, n_local)
    ]
    return float(np.mean(np.concatenate(lengths)))


# ---------------------------------------------------------------------------
# Moving mesh -- ALE vertex motion + harmonic (Laplacian) mesh smoothing
#
# When the *boundary* of the domain moves (a free surface / melt front), the mesh must deform to
# follow it.  The cheap, connectivity-preserving way is an ALE move: displace the vertices but keep
# the topology, so a nodal field simply rides along on its (now-moved) vertices -- no re-interpolation
# (:func:`transfer_solution`) is needed, unlike after a genuine :func:`remesh_with_mmg`.  The interior
# vertices are moved by *harmonically extending* the prescribed boundary motion (solve ``∇²d = 0`` with
# ``d`` fixed on ``∂Ω``), which keeps the elements well-shaped far longer than a naive rigid follow.
# These are the free-boundary companions of ``transfer_solution``; the outer driver (large deformation)
# combines a move with an occasional ``remesh_with_mmg`` + ``transfer_solution`` when the mesh distorts.
# ---------------------------------------------------------------------------
def _simplex_cell_key(dim: int) -> str:
    """meshio's cell-block name for the ``dim``-simplex: interval / triangle / tetrahedron.

    One place, because every geometric helper below reads the same block and a per-helper conditional
    is exactly how a dimension gets silently forgotten."""
    return {1: "line", 2: "triangle"}.get(int(dim), "tetra")


def _simplex_measure_divisor(dim: int) -> float:
    """``d!`` — the simplex volume is ``|det(edge matrix)| / d!`` (length in 1D, area in 2D, volume in 3D)."""
    return {1: 1.0, 2: 2.0}.get(int(dim), 6.0)


def _mesh_cells(domain: Any) -> tuple[np.ndarray, int]:
    dim = int(domain.dimension)
    return np.asarray(domain.mesh.cells_dict[_simplex_cell_key(dim)]).astype(np.int64), dim


def _signed_simplex_measures(points: np.ndarray, cells: np.ndarray, dim: int) -> np.ndarray:
    """Signed area (2D) / volume (3D) of every simplex; the *sign* flips iff a cell inverts (tangles)."""
    v = np.asarray(points)[cells]  # (n_cells, dim+1, dim)
    edge = v[:, 1:, :] - v[:, :1, :]  # (n_cells, dim, dim): rows v_i - v_0
    return np.linalg.det(edge) / _simplex_measure_divisor(dim)


def _signed_simplex_measures_jax(points, cells, dim: int):
    """:func:`_signed_simplex_measures` for traced points -- the tangle test inside a scanned march.

    Same quantity by the same formula; only the sign is read, so a cell that inverts flips it. Separate
    from the host version because that one is a plain NumPy path used by eager callers, and one
    implementation cannot be both."""
    import jax.numpy as jnp

    v = points[jnp.asarray(cells)]
    return jnp.linalg.det(v[:, 1:, :] - v[:, :1, :]) / _simplex_measure_divisor(dim)


def _mesh_boundary_facets(domain: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(cells, boundary_facets, dim)`` -- the interior simplices, their topological boundary
    facets (edges in 2D / triangles in 3D), and the dimension."""
    cells, dim = _mesh_cells(domain)
    if dim == 1:
        # the boundary of an interval mesh is its two endpoint vertices: the nodes referenced by
        # exactly one element (an interior node is shared by two)
        ids, counts = np.unique(cells.reshape(-1), return_counts=True)
        bfacets = ids[counts == 1].reshape(-1, 1).astype(np.int64)
    else:
        bfacets = _boundary_edges_from_triangles(cells) if dim == 2 else _boundary_faces_from_tets(cells)
    return cells, bfacets, dim


def _p1_stiffness_jax(pts, cells, dim):
    """``(matvec, diagonal)`` for the P1 stiffness ``K_ij = ∫_Ω ∇φ_i·∇φ_j`` on ``pts`` — matrix-free, pure
    JAX, and **differentiable in the vertex positions**.

    The element block is the standard ``|K|·(∇φᵢ·∇φⱼ)`` with barycentric gradients from the inverse edge
    matrix. It replaces a numpy+scipy assembly that existed only to be factorized by
    :func:`harmonic_extension`; there is deliberately **one** implementation, since two of the same
    operator drift apart silently.

    :func:`_dirichlet_energy_jax` carries the same quantity (it is ``Σ|∇u|²·vol = uᵀKu``, so ``½·∂E/∂u``
    *is* this matvec). That equivalence is the **test oracle**, not the implementation: the gradient route
    would pay a reverse-mode sweep over the element loop on every CG iteration.

    Nothing here depends on the connectivity's values, only on positions, so a caller marching a
    connectivity-preserving motion hoists ``cells`` once and re-calls this as the mesh moves.
    """
    import jax.numpy as jnp

    cells_j = jnp.asarray(cells)
    v = pts[cells_j]
    e = jnp.stack([v[:, i + 1] - v[:, 0] for i in range(dim)], axis=1)  # (n_cell, dim, dim)
    measure = jnp.abs(jnp.linalg.det(e)) / (2.0 if dim == 2 else 6.0)
    g = jnp.swapaxes(jnp.linalg.inv(e), 1, 2)  # ∇φ of vertices 1..d
    sg = jnp.concatenate([-g.sum(axis=1, keepdims=True), g], axis=1)  # φ₀ closes the partition of unity
    loc = jnp.einsum("cid,cjd->cij", sg, sg) * measure[:, None, None]

    n_local, n_vert = cells.shape[1], pts.shape[0]

    def matvec(u):
        out_c = jnp.einsum("cij,cj->ci", loc, u[cells_j])
        out = jnp.zeros((n_vert,), dtype=u.dtype)
        for a in range(n_local):
            out = out.at[cells_j[:, a]].add(out_c[:, a])
        return out

    diag = jnp.zeros((n_vert,), dtype=measure.dtype)
    for a in range(n_local):
        diag = diag.at[cells_j[:, a]].add(loc[:, a, a])
    return matvec, diag


def _harmonic_extension_jax(pts, cells, dim, given, prescribed, *, tol=None, maxiter=2000):
    r"""Harmonically extend ``given`` off the ``prescribed`` vertices — pure JAX, differentiable in both.

    Solves ``(K d)ᵢ = 0`` on the free rows with ``d`` held at ``given`` on the prescribed rows, written as
    a masked SPD system so the shapes stay static under trace. The free/prescribed split is a **multiply,
    never a gather**, which is what lets this run inside a ``lax.scan``::

        A(v) = mask·K(mask·v) + (1 − mask)·v          rhs = −mask·K((1 − mask)·given)

    ``A`` is the identity on the prescribed block and ``K`` on the free block, hence SPD, so CG applies.
    Jacobi-preconditioned :func:`jax.scipy.sparse.linalg.cg` — the same masked-CG idiom as
    ``_consistent_m_solve`` in :mod:`~jno.utils.solver.timeschemes` — which is built on
    ``lax.custom_linear_solve``, so the gradient is an exact adjoint solve rather than backpropagation
    through the iterations.

    ``tol`` defaults tight because the bar is **absolute**: an affine field is harmonic and so must come
    back exactly (1e-9 in 2D, 1e-7 in 3D). Tight, but **reachable in the working precision** — it used to
    be a hardcoded 1e-13, which is below float32 eps (1.2e-7), the default here since x64 is opt-in. The
    relative termination test could then never fire and CG ground on to the float32 noise floor every
    call. Measured at 1932 vertices: **8.26 ms -> 3.99 ms**, same answer. Same defect and the same
    ``max(floor, 100*eps)`` repair as ``SemidiscreteTimeBlock.step``'s GMRES; in float64 the 1e-13 floor
    keeps the previous behaviour exactly. Pass ``tol=`` to override.
    """
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    if tol is None:
        tol = max(1e-13, 100.0 * float(jnp.finfo(jnp.asarray(pts).dtype).eps))
    matvec, diag = _p1_stiffness_jax(pts, cells, dim)
    mask = 1.0 - jnp.asarray(prescribed, dtype=pts.dtype)  # 1 free / 0 prescribed
    held = (1.0 - mask)[:, None] * given
    jac = 1.0 / (mask * diag + (1.0 - mask))  # Jacobi on the free block, identity on the held one

    def op(v):
        return mask * matvec(mask * v) + (1.0 - mask) * v

    def solve_col(held_col):
        return cg(op, -mask * matvec(held_col), tol=tol, atol=0.0, maxiter=maxiter, M=lambda z: jac * z)[0]

    free = jnp.stack([solve_col(held[:, c]) for c in range(dim)], axis=1)
    return mask[:, None] * free + held


_HARMONIC_JIT: dict = {}


def _harmonic_extension_compiled(pts, cells, dim, given, prescribed):
    """:func:`_harmonic_extension_jax` under ``jax.jit``, compiled once per ``(dim, cell-array shape)``.

    Eager, this solve costs ~160 ms *flat* regardless of mesh size (74 and 377 vertices measure the same),
    because it is per-call JAX tracing overhead rather than arithmetic -- the CG itself is negligible.
    Compiled it is 0.35 ms / 0.66 ms at those sizes: **466x / 242x**. ``cells`` rides as a runtime argument
    rather than a static one, so meshes sharing a cell-array shape share the compiled kernel and a moving
    mesh never recompiles; only ``dim`` and the shapes are static, which they must be to fix the loops.
    """
    import jax
    import jax.numpy as jnp

    key = (int(dim), tuple(cells.shape), tuple(pts.shape))
    fn = _HARMONIC_JIT.get(key)
    if fn is None:

        def _run(p, c, g, m, _d=int(dim)):
            return _harmonic_extension_jax(p, c, _d, g, m)

        fn = _HARMONIC_JIT[key] = jax.jit(_run)
    return fn(pts, jnp.asarray(cells), given, prescribed)


def harmonic_extension(
    domain: Any, boundary_displacement: np.ndarray, *, prescribed: np.ndarray | None = None
) -> np.ndarray:
    r"""Harmonically extend a **boundary** displacement into the mesh interior (ALE mesh motion).

    Solves the vector Laplace problem :math:`\nabla^2 d = 0` in the interior with :math:`d` fixed to
    ``boundary_displacement`` on :math:`\partial\Omega` (one scalar solve per coordinate, sharing the
    factorization).  This is the standard way to move an FE mesh when its boundary moves: interior nodes
    follow the boundary *as smoothly as possible*, which keeps elements well-shaped far longer than
    rigidly dragging them.  A boundary field that is already **affine** (:math:`d = A x + b` -- a uniform
    expansion, translation, or shear) extends to exactly that affine field in the interior (an affine
    field is harmonic), so those motions are reproduced to machine precision.

    Parameters
    ----------
    domain
        A meshed 2-D/3-D simplicial ``jno`` domain.
    boundary_displacement
        ``(n_vert, dim)`` array; **only its prescribed rows are read** (as Dirichlet data).  The remaining
        rows are ignored and overwritten by the harmonic solve -- build a full-length array and set the
        prescribed rows to the desired motion (e.g. an interface velocity times ``dt``).
    prescribed
        Boolean ``(n_vert,)`` mask of the vertices whose displacement is *given*.  Defaults to the
        geometric boundary, which is the ALE case.  Pass it when the motion is stated on an arbitrary
        region -- a geometry term ``coord.d(t) - v`` may name an interior region or a ``where=`` predicate
        just as well as a boundary, and then *those* vertices are the Dirichlet data and everything else
        (including the outer boundary) relaxes around them.

    Returns
    -------
    ``(n_vert, dim)`` full displacement (given boundary rows, harmonically-extended interior); pass it
    straight to :func:`move_mesh`.  Host/NumPy + a SciPy sparse solve -- a structural mesh step, outside
    the differentiable trace.

    Reference: harmonic / Laplacian mesh motion, the simplest ALE mesh-update operator; e.g. Johnson &
    Tezduyar, *Mesh update strategies in parallel FE computations of flows with moving boundaries*,
    Comput. Methods Appl. Mech. Engrg. 119 (1994) 73-94 (§3).
    """
    import jax.numpy as jnp

    cells, dim = _mesh_cells(domain)
    pts = np.asarray(domain.mesh.points)[:, :dim]
    n = pts.shape[0]
    bd = np.asarray(boundary_displacement, dtype=np.float64)
    if bd.shape != (n, dim):
        raise ValueError(
            f"harmonic_extension: boundary_displacement must be (n_vert, dim) = ({n}, {dim}); got {bd.shape}. "
            "Build a full-length array and set its boundary-vertex rows to the desired motion."
        )

    if prescribed is None:
        bfacets = _boundary_edges_from_triangles(cells) if dim == 2 else _boundary_faces_from_tets(cells)
        is_b = np.zeros(n, dtype=bool)
        is_b[np.unique(bfacets.reshape(-1))] = True
    else:
        is_b = np.asarray(prescribed, dtype=bool).reshape(-1)
        if is_b.shape != (n,):
            raise ValueError(f"harmonic_extension: prescribed must be a ({n},) boolean mask; got {is_b.shape}.")
        if not is_b.any():
            raise ValueError("harmonic_extension: `prescribed` selects no vertices — there is no data to extend.")
    if not (~is_b).any():
        return np.where(is_b[:, None], bd, 0.0)  # everything prescribed: nothing to extend

    # A thin host wrapper over the JAX core, rather than a second numpy+scipy implementation of the same
    # operator: two implementations of one solve is exactly the shape that drifts apart silently. The
    # existing affine-reproduction and maximum-principle tests are therefore the acceptance bar for the
    # traced path too.
    return np.asarray(
        _harmonic_extension_compiled(jnp.asarray(pts), cells, dim, jnp.asarray(bd), jnp.asarray(is_b)),
        dtype=np.float64,
    )


def move_mesh(domain: Any, displacement: np.ndarray, *, copy: bool = True, check: bool = True) -> Any:
    r"""Move the mesh vertices by a per-vertex ``displacement`` (an ALE mesh motion), **keeping the
    connectivity**.  Returns the deformed domain (ready for ``jno.fem``); its boundary changes shape.

    Because the topology is unchanged, a nodal solution field stays attached to its (now-moved) vertices
    -- **no** :func:`transfer_solution` is needed, unlike after a :func:`remesh_with_mmg`.  Use this for
    the *smooth* part of a free-boundary march; once the accumulated motion distorts the mesh too much
    (``check`` trips, or element quality drops), ``remesh_with_mmg`` + ``transfer_solution`` instead.

    Parameters
    ----------
    domain
        A meshed 2-D/3-D simplicial ``jno`` domain.
    displacement
        ``(n_vert, dim)`` per-vertex displacement -- typically :func:`harmonic_extension` of a boundary
        motion, so the interior follows smoothly.
    copy
        Return a moved copy (``True``, default) or mutate ``domain`` in place (``False``).
    check
        If ``True`` (default), raise if the motion **inverts or collapses** any element (a tangled mesh
        would silently give a wrong solve -- house rule: fail loud).  Pass ``False`` only if you validate
        element quality yourself.

    Returns
    -------
    The deformed domain.

    Scope / limitations (fail-loud, each a later extension):
    - **Connectivity-preserving** only -- large boundary motion eventually tangles; the recovery is an
      outer remesh + transfer, not this call.
    - **Boundary sub-tags re-derive from the domain's spatial predicates on the moved coordinates.** A
      predicate pinned to a fixed location (e.g. ``x == 1``) will *not* follow an edge that moved past it;
      the moving surface should be tagged by a predicate that tracks it, or driven through the outer loop.
    """
    cells, bfacets, dim = _mesh_boundary_facets(domain)
    pts = np.asarray(domain.mesh.points)[:, :dim].astype(np.float64)
    disp = np.asarray(displacement, dtype=np.float64)
    n = pts.shape[0]
    if disp.shape != (n, dim):
        raise ValueError(f"move_mesh: displacement must be (n_vert, dim) = ({n}, {dim}); got {disp.shape}.")
    new_pts = pts + disp

    if check:
        old_m = _signed_simplex_measures(pts, cells, dim)
        new_m = _signed_simplex_measures(new_pts, cells, dim)
        floor = 1e-12 * float(np.median(np.abs(old_m)) or 1.0)
        tangled = int(np.sum((np.sign(new_m) != np.sign(old_m)) | (np.abs(new_m) <= floor)))
        if tangled:
            raise ValueError(
                f"move_mesh: the displacement inverts or collapses {tangled}/{cells.shape[0]} elements "
                "(the mesh would tangle). Take a smaller step, harmonic_extension the boundary motion into "
                "the interior, or remesh_with_mmg + transfer_solution instead of a connectivity-preserving move."
            )
    return _domain_from_arrays(domain, new_pts, cells, bfacets, copy=copy)


# ---------------------------------------------------------------------------
# Driver -- the outer adaptive loop
# ---------------------------------------------------------------------------
@dataclass
class AdaptSpec:
    """Controls for the adaptive-FEM loop (``FEM.solve(adapt=...)``).

    Attributes
    ----------
    theta
        Dörfler bulk-marking fraction (0..1).
    max_iters
        Maximum number of refine-solve rounds.
    refine_factor
        Local edge-size reduction applied to marked cells each round.
    tol
        Stop early once the global error estimate falls below this (optional).
    max_dofs
        Stop once the mesh vertex count reaches this budget (optional).
    eps
        Relative-change convergence tolerance: stop once the round's figure of merit
        stops changing by more than ``eps`` between successive rounds (optional). The
        figure of merit is the ZZ estimate for the forward driver
        (:func:`run_adaptive_solve`) and the recovered parameter for the inverse driver
        (:func:`run_adaptive_inverse`). A **plateau detector** ("the answer has stopped
        moving as I refine"), not a certified error bound -- the lever for more accuracy
        is the ``max_dofs`` / ``max_iters`` budget. Requires two consecutive rounds under
        ``eps`` (patience) so a single flat step does not stop the loop prematurely.
    anisotropic
        If ``True``, refine the forward loop with an :func:`hessian_metric` (stretched
        elements aligned to the solution's curvature) grown by ``refine_factor``× vertices
        per round, instead of isotropic ZZ + Dörfler marking. Far fewer DOFs for directional
        features (layers, fronts); 2D and 3D scalar. ``hmin`` / ``hmax`` bound the edge sizes.
        Metric-based DOF control is approximate, so ``max_dofs`` is honored only loosely here
        (a round may overshoot it by up to ~1.5x, especially in 3D).
    hmin, hmax
        Edge-size window for the anisotropic metric (defaults derive from the mesh).
    every
        **Transient only** (``FEM.solve(adapt=...)`` on a ``u.t`` problem): remesh every ``every``
        time steps, carrying the state across each remesh (:func:`transfer_solution`). Between remeshes
        the march is the ordinary differentiable stepper; smaller = the mesh tracks a fast feature more
        tightly, larger = cheaper. Ignored by the steady loop.
    metric_field
        **Transient multifield only**: the index of the coupled field whose curvature drives the
        anisotropic metric (which feature the mesh tracks) — first-appearance order in the ``jno.fem``
        constraints. Default 0. A vector and/or higher-order (P2) metric field is reduced to a scalar
        **per-vertex magnitude** for the (scalar) ZZ / Hessian estimator. (Refining on *all* fields at
        once — metric intersection — is a later refinement.)
    """

    theta: float = 0.5
    max_iters: int = 8
    refine_factor: float = 2.0
    tol: float | None = None
    max_dofs: int | None = None
    eps: float | None = None
    anisotropic: bool = False
    hmin: float | None = None
    hmax: float | None = None
    every: int = 5
    metric_field: int = 0
    # --- r-adaptivity (mesh relocation) --------------------------------------------------------------
    relocate: bool = False
    """Switch ``FEM.solve(adapt=...)`` from h-refinement (add elements) to **r-adaptivity**: relocate the
    mesh vertices tagged with :meth:`Variable.trainable` so the mesh equidistributes the solution's
    features, at fixed connectivity and no new DOFs. Requires at least one coordinate tagged
    ``domain.variable(region)[i].trainable()`` before ``jno.fem`` (else it raises). See
    :func:`run_adaptive_relocate`; :attr:`relocate_method` picks *how* the vertices move."""
    quality_floor: float = 0.1
    """``relocate_method="descent"`` only: a step is backtracked (halved) until no element's ``|det J|``
    falls below this fraction of the initial worst element — the mesh-validity line search that keeps the
    descent from tangling (a stock optimiser / a barrier alone cannot guarantee this on stiff problems).
    Unused by ``"monge_ampere"``, which needs no line search."""
    lr: float = 3e-3
    """``relocate_method="descent"`` only: base step size for the RMS-normalised gradient descent."""
    ma_relax: int = 60
    """``relocate_method="monge_ampere"`` only: relaxation iterations per outer round (McRae et al. §3.1).
    Each is one Poisson solve against a pre-factorized constant matrix, so these are cheap. The relaxation
    is explicit in :attr:`ma_dt`, so more is not monotonically better — see :func:`jno.solve.relocate`."""
    ma_dt: float = 0.1
    """``relocate_method="monge_ampere"`` only: the relaxation pseudo-step ``Δt`` of eq. (3.7). Larger
    converges faster and can overshoot; the nonlinearity is carried entirely by this outer relaxation."""
    relocate_method: str = "descent"
    """Relocation only: *how* the mesh is moved. ``"descent"`` — the default here and the default
    :func:`jno.solve.relocate` builds — walks the vertices down :attr:`objective` with a mesh-validity
    line search. ``"monge_ampere"`` instead solves a Monge-Ampère equation for a mesh potential ``φ`` and
    takes the mesh as ``x = ξ + ∇φ`` (:func:`_monge_ampere_displacement`): no descent, no line search, and
    non-folding by construction because the displacement is a gradient (for the *whole* map — see that
    function on what holding a subset costs). With it, ``objective`` is only a *diagnostic* and ``lr`` /
    ``quality_floor`` are unused.

    Descent is the default **on the measured answer**, not on convergence rate. Monge-Ampère does converge
    in far fewer rounds (3-6 against 30) and reaches a comparable equidistribution defect, but on the
    Allen-Cahn front the suite measures it loses on accuracy and wrecks element quality: rel-L2 8.879e-02
    against descent's 3.951e-02 (uniform 1.096e-01), min element quality 0.160 against 0.503. See
    :func:`jno.solve.relocate` for the full table and the ``relax_step`` control that recovers part of it.
    (This docstring previously claimed the opposite of all of that.)"""
    objective: str = "equidistribution"
    """Relocation only, **internal / not yet exposed** by :func:`jno.solve.relocate`: which mesh functional
    to descend. ``"equidistribution"`` (default) is :func:`_equidistribution_jax`, the scale-free
    equidistribution defect of an arclength monitor; ``"huang"`` is :func:`_huang_ea_jax`, Huang's
    equidistribution+alignment functional with the same (isotropic) monitor. Present so the two can be
    measured against each other on the same problem before either is given a public spelling."""


# Consecutive rounds that must satisfy ``eps`` before the loop stops on convergence.
# Two, not one: a single flat step can be a false plateau (e.g. two coarse rounds that
# barely move the answer before it jumps again), which would stop the loop far too early.
_EPS_PATIENCE = 2


def _rel_change(cur: Any, prev: Any) -> float:
    """Relative L2 change ``||cur - prev|| / ||cur||`` between two (array-like) values."""
    c = np.atleast_1d(np.asarray(cur, dtype=float)).reshape(-1)
    p = np.atleast_1d(np.asarray(prev, dtype=float)).reshape(-1)
    denom = float(np.linalg.norm(c))
    diff = float(np.linalg.norm(c - p))
    return diff / denom if denom > 0.0 else diff


def _solve_vertex_values(fem: Any, solve_fn: Any = None, *, allow_vector: bool = False, **kwargs: Any) -> np.ndarray:
    """Solve ``fem`` and return the nodal solution at the mesh vertices -- complex-valued if the form is
    complex.

    Scalar P1 returns all DOFs as ``(n_vertices,)``; a higher-order scalar field (e.g. P2/TET10) returns
    its **vertex** DOFs (the first ``n_vertices`` entries, which are nodal). With ``allow_vector=True`` a
    P1 **vector** field returns its node-major nodal values reshaped to ``(n_vertices, vec)`` (which the ZZ
    estimator sums over components); without it -- and for a higher-order vector field, whose vertex DOFs
    are not a simple prefix -- a vector problem is rejected (refine on a scalar readout instead)."""
    from jno._fem import _infer_vec  # local import: fem_adapt is loaded lazily by the domain

    sol = np.asarray(fem.solve(solve_fn, **kwargs)).reshape(-1)
    n_vert = int(np.asarray(fem.domain.mesh.points).shape[0])
    vec = _infer_vec(fem._constraints) if getattr(fem, "_constraints", None) else 1
    if vec != 1:
        if allow_vector and sol.shape[0] == n_vert * vec:
            return sol.reshape(n_vert, vec)  # node-major P1 vector: (n_vert, vec), one column per component
        raise NotImplementedError(
            f"The ZZ estimator supports a scalar field or a P1 vector field; got a vector field (vec={vec})"
            f"{' with a non-P1 DOF layout' if allow_vector else ''}. Refine per component or on a scalar readout."
        )
    if sol.shape[0] == n_vert:
        return sol
    if sol.shape[0] > n_vert:
        return sol[:n_vert]  # higher-order scalar: the first n_vert DOFs are the vertex (nodal) values
    raise NotImplementedError(f"got {sol.shape[0]} DOFs for {n_vert} vertices (fewer than one per vertex).")


def run_adaptive_solve(fem: Any, spec: AdaptSpec, *, solve_fn: Any = None, **kwargs: Any) -> np.ndarray:
    """Drive ``FEM.solve(adapt=spec)``: ``solve -> estimate -> mark -> refine`` in a loop.

    The domain (``fem.domain``) is remeshed **in place** each round and the constraints
    re-assembled on the refined mesh (they reference the domain, so re-tracing picks up
    the new nodes).  Returns the solution on the final adapted mesh.  On return, ``fem``
    and its domain refer to that final mesh and ``fem.adapt_history`` holds the per-round
    trace ``{n_dofs, estimate, n_marked}``.

    Requires ``fem`` to have been built by :func:`jno.fem` (so its constraint recipe was
    retained); a hand-constructed ``FEM`` has no recipe to re-assemble.
    """
    import jno

    if fem._constraints is None:
        raise ValueError("FEM.solve(adapt=...) requires a FEM built by jno.fem(...) (its constraint list is retained).")

    d = fem.domain
    cons = fem._constraints
    kw = fem._fem_kwargs
    history: list[dict] = []
    cur = fem
    u = None
    prev_est = None
    n_converged = 0
    for it in range(spec.max_iters):
        u = _solve_vertex_values(cur, solve_fn, allow_vector=True, **kwargs)
        if u.ndim == 2 and spec.anisotropic:  # vector field: the ZZ estimator sums components, but the
            raise NotImplementedError(  # anisotropic Hessian metric is scalar-only (a single Hessian field)
                "anisotropic (Hessian-metric) adaptation is scalar-only; use isotropic ZZ "
                "(AdaptSpec(anisotropic=False)) to refine a vector field."
            )
        eta, est = zz_error_indicators(d, u)
        n_dofs = int(np.asarray(d.mesh.points).shape[0])
        marked = None if spec.anisotropic else dorfler_mark(eta, spec.theta)
        _rec_cells, _ = _mesh_cells(d)  # points/cells: for a refinement animation (connectivity changes each round)
        history.append(
            {
                "n_dofs": n_dofs,
                "estimate": est,
                "n_marked": None if marked is None else int(marked.size),
                "points": np.asarray(d.mesh.points)[:, : int(d.dimension)].copy(),
                "cells": np.asarray(_rec_cells).copy(),
            }
        )

        # eps: the estimate stopped improving (plateau) for _EPS_PATIENCE rounds
        if spec.eps is not None and prev_est is not None:
            n_converged = n_converged + 1 if _rel_change(est, prev_est) < spec.eps else 0
        prev_est = est

        last = it == spec.max_iters - 1
        below_tol = spec.tol is not None and est < spec.tol
        over_budget = spec.max_dofs is not None and n_dofs >= spec.max_dofs
        plateaued = spec.eps is not None and n_converged >= _EPS_PATIENCE
        nothing_marked = marked is not None and marked.size == 0
        if last or below_tol or over_budget or plateaued or nothing_marked:
            break

        if spec.anisotropic:
            if np.iscomplexobj(u):
                raise NotImplementedError(
                    "anisotropic (Hessian-metric) adaptation is real-only; use isotropic ZZ "
                    "(AdaptSpec(anisotropic=False)) for a complex field."
                )
            h_typ = _mean_edge_length(d)
            hmin = spec.hmin if spec.hmin is not None else h_typ / 50.0
            hmax = spec.hmax if spec.hmax is not None else h_typ * 2.0
            # target the vertex count directly (hessian_metric is calibrated per dimension) and
            # cap it at the DOF budget so a single round cannot blow far past max_dofs
            target = n_dofs * spec.refine_factor
            if spec.max_dofs is not None:
                target = min(target, float(spec.max_dofs))
            metric = hessian_metric(d, u, target_complexity=target, hmin=hmin, hmax=hmax)
            # a loose size gradation lets adjacent elements change size fast, which is what
            # permits the high aspect ratios that make anisotropic adaptation pay off
            remesh_with_mmg(d, metric, copy=False, hmin=hmin, hmax=hmax, hgrad=3.0)
        else:
            size = size_field_from_marks(d, marked, refine_factor=spec.refine_factor)
            remesh_with_mmg(d, size, copy=False)  # mutate the domain in place
        # Re-materialize custom coordinate-predicate tags on the refreshed mesh so that
        # surface-integral terms -- Neumann / Robin / absorbing boundary conditions -- re-derive on
        # the new boundary facets. (Dirichlet already re-resolves geometrically via its location
        # function; without this the flux terms reference stale nodes and silently vanish, leaving a
        # homogeneous problem after the first remesh.)
        for _name, _pred in list(getattr(d, "_tag_predicates", {}).items()):
            d.tag(_name, _pred)
        cur = jno.fem(cons, **kw)  # re-assemble the same problem on the refined mesh

    # rebind the caller's FEM to the final adapted state so fem.points / A / b match ``u``
    fem.__dict__.update(cur.__dict__)
    fem.adapt_history = history
    return u


def _dirichlet_energy_jax(pts, u_nodal, cells, dim):
    """``Σ_cells |∇u|² · vol`` for a P1 field — differentiable in the vertices ``pts`` and the nodal values
    ``u_nodal`` (shape ``(n_verts,)`` or ``(n_verts, n_comp)``). Field-agnostic (scalar / vector / a
    primary field), 2-D triangle and 3-D tet: the relocation objective (concentrate nodes where the
    solution's gradient is large). ``pts`` are the *moved* vertices, so this is the shape derivative's
    integrand -- differentiating it w.r.t. the coordinate parameters is the r-adaptivity gradient."""
    import jax.numpy as jnp

    v = pts[cells]  # (n_cell, dim+1, dim)
    u_nodal = u_nodal[:, None] if u_nodal.ndim == 1 else u_nodal
    s = u_nodal[cells]  # (n_cell, dim+1, n_comp)
    e = jnp.stack([v[:, i + 1] - v[:, 0] for i in range(dim)], axis=1)  # (n_cell, dim, dim): rows = edge vectors
    du = jnp.stack([s[:, i + 1] - s[:, 0] for i in range(dim)], axis=1)  # (n_cell, dim, n_comp): du along each edge
    grad = jnp.linalg.solve(e, du)  # (n_cell, dim, n_comp): ∇u, since du = E·∇u with E the edge matrix
    vol = jnp.abs(jnp.linalg.det(e)) / (2.0 if dim == 2 else 6.0)
    return jnp.sum(jnp.sum(grad**2, axis=(1, 2)) * vol)


def _equidistribution_jax(pts, u_nodal, cells, dim):
    """Equidistribution defect of a **monitor function** over a P1 mesh — the r-adaptivity objective.

    Classic moving-mesh r-adaptivity equidistributes a monitor ``ρ``: every element should carry the same
    share of ``∫ρ``, which puts small elements where ``ρ`` is large (Huang & Russell, *Adaptive Moving Mesh
    Methods*, Springer 2011, §2.1 — the equidistribution principle; Winslow / MMPDE build the same idea into
    a mesh PDE).  Here ``ρ_T = sqrt(1 + |∇u_T|² / ⟨|∇u|²⟩)`` — the **arclength** monitor, normalised by the
    mesh average so the functional is dimensionless and one ``lr`` works across problems.  The ``1 +`` floor
    keeps a flat region from being emptied entirely.

    Returns ``Σ_T (w_T − w̄)² / (n_T · w̄²)`` with ``w_T = ρ_T · |T|``: zero exactly when every element carries
    an equal share, and scale-free.  Differentiable in ``pts`` (the moved vertices) and in ``u_nodal``.

    **Why not the Dirichlet energy.**  Descending ``Σ|∇u|²·vol`` w.r.t. the vertices looks like the same idea
    and is not: for a non-convex functional (Allen–Cahn's double well) the mesh can *lower* it by
    under-resolving the layer.  Measured on the Allen–Cahn interface at 377 nodes, the energy objective cut
    the energy 4.949 → 4.422 while making the final-time error 10.7× worse (1.68e-2 → 1.78e-1) and visibly
    coarsening the mesh across the front.  Equidistribution targets resolution directly.

    **Why arclength and not a Hessian monitor.**  P1 interpolation error is governed by curvature, so a
    Hessian monitor is the textbook choice — but ``∇u`` is constant per P1 element, so the Hessian has to be
    recovered (ZZ patch averaging), and that recovery is meaningless exactly where adaptivity matters: a
    front spanning ~1 element gives a noisy Hessian, and equidistributing noise scatters nodes.  Measured
    against a uniform mesh at equal node count (Allen–Cahn, ``h = 0.06``), relocated/uniform final error:

    ==================  ==========  =================  ================
    front width         arclength   recovered Hessian  verdict
    ==================  ==========  =================  ================
    0.47 (≈8 cells)     4.98        2.61               both worse
    0.19 (≈3 cells)     **0.86**    1.36               arclength wins
    0.09 (≈1.5 cells)   **0.55**    1.22               arclength wins
    ==================  ==========  =================  ================

    Arclength wins wherever relocation is worth doing.  Both lose on an already over-resolved front — there
    is nothing to redistribute there, and moving nodes only costs; see the scope note on
    :func:`run_adaptive_relocate`.
    """
    import jax.numpy as jnp

    v = pts[cells]  # (n_cell, dim+1, dim)
    u_nodal = u_nodal[:, None] if u_nodal.ndim == 1 else u_nodal
    s = u_nodal[cells]
    e = jnp.stack([v[:, i + 1] - v[:, 0] for i in range(dim)], axis=1)
    du = jnp.stack([s[:, i + 1] - s[:, 0] for i in range(dim)], axis=1)
    grad = jnp.linalg.solve(e, du)  # (n_cell, dim, n_comp)
    vol = jnp.abs(jnp.linalg.det(e)) / (2.0 if dim == 2 else 6.0)

    g2 = jnp.sum(grad**2, axis=(1, 2))  # |∇u|² per element, summed over components/blocks
    scale = jnp.mean(g2) + 1e-30  # normalise so the monitor (and the step size) is problem-independent
    rho = jnp.sqrt(1.0 + g2 / scale)  # the "1 +" floor keeps flat regions from being emptied entirely
    w = rho * vol  # each element's share of ∫ρ
    wbar = jnp.mean(w) + 1e-30
    return jnp.mean((w - wbar) ** 2) / wbar**2


def _huang_ea_jax(pts, pts0, u_nodal, cells, dim, *, theta=1.0 / 3.0, p=1.5):
    r"""Huang's **equidistribution–alignment** meshing functional — the variational r-adaptivity objective.

    Huang, *Variational mesh adaptation: isotropy and equidistribution* (2001); stated as Example 2.2 and
    discretized as eq. (6) of Huang & Kamenski, *A geometric discretization and a simple implementation for
    variational mesh generation and adaptation*, J. Comput. Phys. **301** (2015) 322–337
    (arXiv:1410.7872)::

        I[ξ] = θ ∫_Ω √det(M) · tr(J M⁻¹ Jᵀ)^(dp/2) dx
             + (1−2θ)·d^(dp/2) ∫_Ω √det(M) · (det(J)/√det(M))^p dx

    **Conventions, because both are easy to invert and the wrong one equidistributes backwards.**
    ``J = ∂ξ/∂x`` is the Jacobian of the *inverse* map (computational-from-physical) — meshing functionals
    are written in terms of it because the transformation so determined is less likely to be singular
    (Dvinsky 1991). The integral is over the **physical** domain, so the discrete sum is weighted by the
    **physical** element volume ``|K|``, and ``J`` is discretized as ``(F'_K)⁻¹`` with ``F_K: K_c → K`` the
    affine element map — not by differentiating anything on the mesh, which is what preserves the
    functional's geometric structure.

    Here ``pts`` is the physical (moving) mesh and ``pts0`` the computational (fixed) one, so with ``E``
    the physical and ``Ê`` the computational edge matrix, ``J_K = Ê E⁻¹`` and ``|K| = |det E|/d!``.

    The two terms are the **alignment** condition (elements shaped and oriented by ``M``) and the
    **equidistribution** condition (equal share of ``∫√det M``). :func:`_equidistribution_jax`, the
    objective this one is measured against, is a scale-free proxy for the *second* term alone with an
    isotropic monitor; the first term has no counterpart there.

    ``M`` is isotropic here — ``M = ρ^(2/d) I`` for the same arclength density ``ρ`` that
    :func:`_equidistribution_jax` uses, so a comparison between the two isolates the functional rather than
    confounding it with a change of monitor. A genuine metric-tensor ``M`` (the anisotropic case, where the
    alignment term earns its keep) needs a differentiable recovered Hessian and is deferred.

    Defaults ``θ = 1/3``, ``p = 3/2`` are MMPDElab's (Huang, arXiv:1904.05535). They sit inside the
    coercivity + polyconvexity conditions ``0 < θ ≤ 1/2``, ``dp ≥ 2``, ``p ≥ 1``, which is what underwrites
    the mesh-nonsingularity result; ``1 − 2θ = θ`` there, so the two conditions carry equal weight.

    **The barrier is the point.** ``det J → 0`` drives the second term to ``+∞``: an element cannot flatten
    without paying infinite energy. That is the coercivity that replaces a ``det J`` step-control heuristic
    — so ``det J`` is deliberately *not* clamped here. The gradient is only ever taken at a valid mesh.
    """
    import jax.numpy as jnp

    v, v0 = pts[cells], pts0[cells]  # (n_cell, dim+1, dim)
    u_nodal = u_nodal[:, None] if u_nodal.ndim == 1 else u_nodal
    s = u_nodal[cells]
    e = jnp.stack([v[:, i + 1] - v[:, 0] for i in range(dim)], axis=1)  # Eᵀ, physical
    e0 = jnp.stack([v0[:, i + 1] - v0[:, 0] for i in range(dim)], axis=1)  # Êᵀ, computational
    du = jnp.stack([s[:, i + 1] - s[:, 0] for i in range(dim)], axis=1)
    grad = jnp.linalg.solve(e, du)  # (n_cell, dim, n_comp)

    g2 = jnp.sum(grad**2, axis=(1, 2))  # |∇u|² per element
    rho = jnp.sqrt(1.0 + g2 / (jnp.mean(g2) + 1e-30))  # arclength density, as in _equidistribution_jax

    # J = Ê E⁻¹, so Jᵀ = (E⁻¹)ᵀ Êᵀ = solve(Eᵀ, Êᵀ) = solve(e, e0). Frobenius norm and det are
    # transpose-invariant, so the transpose never has to be formed.
    a = jnp.linalg.solve(e, e0)  # = Jᵀ
    fro2 = jnp.sum(a**2, axis=(1, 2))  # tr(J Jᵀ)
    det_j = jnp.linalg.det(a)
    vol = jnp.abs(jnp.linalg.det(e)) / (2.0 if dim == 2 else 6.0)  # |K|, PHYSICAL

    m_inv = rho ** (-2.0 / dim)  # M = ρ^(2/d)·I  ⇒  M⁻¹ = ρ^(-2/d)·I
    align = rho * (m_inv * fro2) ** (dim * p / 2.0)
    equi = rho * (det_j / rho) ** p
    return jnp.sum(vol * (theta * align + (1.0 - 2.0 * theta) * dim ** (dim * p / 2.0) * equi))


def _p1_operators(pts, cells, dim):
    """Constant P1 operators on a **fixed** mesh: ``(sg, measure, wsum, stiffness+null-space shift)``.

    ``sg`` ``(n_cell, dim+1, dim)`` barycentric gradients, ``measure`` the cell volumes, ``wsum`` the
    per-vertex incident volume (the denominator of patch recovery), and ``K`` the P1 stiffness matrix
    with a rank-one shift ``(1/n)·11ᵀ`` that removes its constant null space *without* perturbing the
    solution: ``K·1 = 0``, so on the mean-zero subspace — which is where a compatible right-hand side
    puts the answer — the shifted operator acts identically to ``K``.

    Everything here depends only on the mesh geometry, and in :func:`_monge_ampere_displacement` that
    mesh is the *computational* one, which never moves. So this is assembled and factorized **once** and
    reused across every relaxation step and every outer round — the reason the Monge-Ampère route costs
    no re-factorization, unlike a mesh flow whose operator changes as the mesh does.
    """
    v = pts[cells]
    edge = v[:, 1:, :] - v[:, :1, :]
    einv = np.linalg.inv(edge)
    measure = np.abs(np.linalg.det(edge)) / (2.0 if dim == 2 else 6.0)
    sg = np.zeros((cells.shape[0], dim + 1, dim))
    sg[:, 1:, :] = np.transpose(einv, (0, 2, 1))
    sg[:, 0, :] = -sg[:, 1:, :].sum(axis=1)

    n = pts.shape[0]
    wsum = np.zeros(n)
    for lv in range(cells.shape[1]):
        np.add.at(wsum, cells[:, lv], measure)

    k = np.zeros((n, n))
    loc = np.einsum("cid,cjd->cij", sg, sg) * measure[:, None, None]  # (n_cell, d+1, d+1)
    for a in range(dim + 1):
        for b in range(dim + 1):
            np.add.at(k, (cells[:, a], cells[:, b]), loc[:, a, b])
    return sg, measure, np.maximum(wsum, 1e-300), k + 1.0 / n


def _nodal_grad_jax(f, sg_j, meas_j, wsum_j, cells_j, n_local, dim):
    """Patch-recovered nodal gradient of a P1 field, in JAX — the volume-weighted average of the incident
    elementwise-constant gradients. The JAX mirror of :func:`_recover_nodal_gradient` (which is numpy and
    so cannot sit inside a trace), and the ``Π_[P1]^d`` projection of McRae et al. eq. (3.5)."""
    import jax.numpy as jnp

    g_cell = jnp.einsum("cld,cl->cd", sg_j, f[cells_j])  # (n_cell, dim), constant per element
    num = jnp.zeros((wsum_j.shape[0], dim), dtype=g_cell.dtype)
    for lv in range(n_local):
        num = num.at[cells_j[:, lv]].add(meas_j[:, None] * g_cell)
    return num / wsum_j[:, None]


def _arclength_monitor_jax(u_nodal, sg_j, meas_j, wsum_j, cells_j, n_local, dim):
    """Per-**vertex** arclength density ``ρ = sqrt(1 + |∇u|²/⟨|∇u|²⟩)`` — the same monitor
    :func:`_equidistribution_jax` and :func:`_huang_ea_jax` use, but recovered at vertices (a P1 field on
    the computational mesh) because that is what the Monge-Ampère solve integrates against. Summed over
    components, so a vector field contributes all of them."""
    import jax.numpy as jnp

    u_nodal = u_nodal[:, None] if u_nodal.ndim == 1 else u_nodal
    g2 = sum(
        jnp.sum(_nodal_grad_jax(u_nodal[:, c], sg_j, meas_j, wsum_j, cells_j, n_local, dim) ** 2, axis=1)
        for c in range(u_nodal.shape[1])
    )
    return jnp.sqrt(1.0 + g2 / (jnp.mean(g2) + 1e-30))


def _monge_ampere_displacement(monitor, ops, cells, dim, *, n_relax=60, dt=0.1):
    r"""Vertex displacement ``Π_[P1]^d ∇φ`` from one **Monge–Ampère** mesh solve — pure JAX.

    Solves ``m(x)·det(I + H(φ)) = θ`` for a scalar mesh potential ``φ`` on the fixed computational mesh,
    from which the adapted mesh is ``x(ξ) = ξ + ∇φ(ξ)``. Because the displacement is a *gradient*, the
    map cannot fold — non-tangling is structural here, not a step-control heuristic, which is what makes
    this route differentiable without a validity line search.

    Reference: McRae, Cotter & Budd, *Optimal-transport-based mesh adaptivity on the plane and sphere
    using finite elements*, SIAM J. Sci. Comput. **40**(2) (2018) A1121–A1148 (arXiv:1612.08077), §3.1 —
    the relaxation method, their eqs. (3.5)–(3.8). Each iteration is

    1. ``σᵏ = H(φᵏ)`` — recovered by patch averaging (their eq. (3.8) recovers it by a mass solve;
       the P1 patch recovery already in this module plays the same role and needs no extra matrix),
    2. ``θᵏ = ∫ m·det(I + σᵏ) / ∫ 1``  (eq. (3.6)) — the normalisation that makes the next system
       *solvable*: it is exactly what makes the right-hand side orthogonal to the constant null space,
    3. ``⟨∇v, ∇φᵏ⁺¹⟩ = ⟨∇v, ∇φᵏ⟩ + Δt⟨v, m·det(I + σᵏ) − θᵏ⟩``  (eq. (3.7)) — one Poisson solve.

    So a step is a Poisson solve against a **pre-factorized constant** matrix plus a mass apply; the
    nonlinearity is carried entirely by the outer relaxation in ``dt``.

    **Scope — stated, not hidden.**

    - ``m`` is taken as a function of the *computational* coordinate ``ξ``, not of ``x``. That is the
      paper's "very straightforward" case (§3, above eq. (3.5)); the genuine problem has ``m = m(x)``,
      an extra nonlinearity, and dropping it means the monitor lags the mesh it produces. The outer
      loop in :func:`run_adaptive_relocate` recovers it approximately by re-solving and re-sampling.
    - ``φ`` is P1 here, so ``∇φ`` is elementwise-constant and both it and ``σ`` are patch-recovered.
      The paper takes ``φ ∈ P_n, n ≥ 2`` and L²-projects ``∇φ`` into ``[P1]^d`` (eq. (3.5)); patch
      recovery is the cheaper stand-in and is the same order.
    - Natural (Neumann) boundary treatment only. The caller decides which vertices may move and the
      driver holds every untagged vertex exactly — but note that **holding vertices is what can tangle
      the mesh**, not this solve. Non-folding is a property of the *whole* map ``x = ξ + ∇φ``; apply it
      to a subset and the truncated map carries no such guarantee. Measured on a 21² unit square with a
      diagonal front: applied to every vertex ``min det J`` stays positive across ``dt ∈ [0.01, 0.2]`` and
      ``n_relax ∈ [40, 300]``, while freezing the boundary reaches ``-1.2e-03``. Freeing each boundary
      edge's *tangential* axis only (``x`` on a horizontal edge, ``y`` on a vertical one — which per-axis
      :meth:`~jno.trace.Placeholder.trainable` tagging expresses directly) recovers almost all of it and
      equidistributes better, because the held component is then only the one normal to the wall.
      Whatever the tagging, :func:`run_adaptive_relocate` checks ``det J`` and keeps the last valid mesh.
    - The relaxation is **explicit in** ``dt`` and has a stability limit, so more iterations is not
      monotonically better: at ``dt = 0.2`` the equidistribution spread on that same mesh *degrades*
      from 0.111 at ``n_relax = 40`` to 0.292 at 300. Lower ``dt`` before raising ``n_relax``.

    **The map is global, and that is intrinsic.** ``det(I + H(φ)) = θ/m`` carries a single *global* ``θ``, so
    concentrating elements where ``m`` is large forces every region where ``m`` is small to stretch — area has
    to come from somewhere. Elements far from the feature are therefore distorted whether or not that helps:
    measured on the Allen-Cahn front, largest/smallest element area ratio 8.3 and worst radius-ratio quality
    0.160, against 2.4 and 0.834 for the uniform mesh it started from (and 3.3 / 0.503 for ``"descent"``,
    which reaches partial equidistribution in damped steps and so stays gentler).

    The lever is **under-relaxation**, not a boundary condition: lowering ``dt`` stops short of exact
    equidistribution and keeps the map closer to the identity away from the feature (``dt`` 0.10 → 0.02 moved
    quality 0.160 → 0.318 and the error ratio 0.811 → 0.633). Imposing ``φ = 0`` on the held vertices does
    *not* localize it — ``φ = 0`` does not make ``∇φ = 0``, so the recovered displacement at a held vertex
    beside a free one is if anything *larger* (0.107 against 0.075 measured), and the truncation gets worse
    rather than better. Genuine confinement needs the problem restricted to the tagged sub-mesh, which is not
    implemented.

    **Graded computational meshes.** What this solves equidistributes ``m`` against the *ratio*
    ``|K_phys|/|K_comp|``, not against ``|K_phys|`` — so on a graded ξ (an already h-adapted mesh, i.e. the
    ``remesh`` → ``relocate`` composition) the grading and the monitor would compound and the adaptation
    would be applied twice. ``monitor`` is therefore pre-multiplied here by the local computational cell
    size, after which ``m·|K_phys| = θ`` — true equidistribution — holds on any ξ. The size is the *mean*
    incident element volume, not ``wsum`` (the incident volume *sum*, which halves at an edge and quarters
    at a corner purely from patch arity and would push nodes off the boundary of a perfectly uniform mesh).
    On a uniform ξ the factor is constant and this is a no-op.
    """
    import jax
    import jax.numpy as jnp

    sg, measure, wsum, k_shift = ops
    sg_j, meas_j, wsum_j = jnp.asarray(sg), jnp.asarray(measure), jnp.asarray(wsum)
    cells_j = jnp.asarray(cells)
    lu = jax.scipy.linalg.lu_factor(jnp.asarray(k_shift))  # constant mesh ⇒ factorize ONCE
    n_vert = wsum.shape[0]
    total = float(measure.sum())

    counts = np.zeros(n_vert)  # incident element count per vertex (patch arity)
    np.add.at(counts, cells.ravel(), 1.0)
    k_comp = wsum / np.maximum(counts, 1.0)  # mean incident element volume = local computational cell size
    monitor = monitor * jnp.asarray(k_comp / k_comp.mean())  # a global scale cancels in θ; kept O(1) for dt
    fac = 1.0 / ((dim + 1) * (dim + 2))  # P1 consistent mass: ∫φᵢφⱼ = |K|·(1+δᵢⱼ)/((d+1)(d+2))

    n_local = cells.shape[1]

    def nodal_grad(f):
        return _nodal_grad_jax(f, sg_j, meas_j, wsum_j, cells_j, n_local, dim)

    def mass_apply(g):
        """``(M g)ᵢ`` for the P1 consistent mass matrix, assembled on the fly (no matrix stored)."""
        gc = g[cells_j]  # (n_cell, d+1)
        s = gc.sum(axis=1, keepdims=True)
        loc = fac * meas_j[:, None] * (s + gc)
        out = jnp.zeros((n_vert,), dtype=g.dtype)
        for lv in range(cells.shape[1]):
            out = out.at[cells_j[:, lv]].add(loc[:, lv])
        return out

    eye = jnp.eye(dim)

    def step(phi, _):
        g = nodal_grad(phi)
        sigma = jnp.stack([nodal_grad(g[:, j]) for j in range(dim)], axis=1)  # (n_vert, dim, dim)
        sigma = 0.5 * (sigma + jnp.swapaxes(sigma, 1, 2))  # H(φ) is symmetric; recovery need not be
        f = monitor * jnp.linalg.det(eye + sigma)  # m·det(I + σ)
        theta = jnp.sum(meas_j * f[cells_j].mean(axis=1)) / total  # eq. (3.6)
        rhs = jnp.asarray(k_shift) @ phi + dt * mass_apply(f - theta)  # eq. (3.7)
        return jax.scipy.linalg.lu_solve(lu, rhs), None

    phi, _ = jax.lax.scan(step, jnp.zeros((n_vert,), dtype=monitor.dtype), None, length=n_relax)
    return nodal_grad(phi)


def _transient_march_fn(block):
    """Build a differentiable ``args -> time-averaged nodal state`` for a :class:`SemidiscreteTimeBlock`.

    The time grid / dt / theta are constants (independent of the runtime parameters), so they are captured
    once here; the returned closure is a plain ``lax.scan`` over the block's one-step primitive, reverse-mode
    differentiable in ``args`` (the coordinate gradient flows). The relocation driver averages the trajectory
    in time -- it optimises the mesh for the whole run, not one snapshot."""
    import jax
    import jax.numpy as jnp

    from .backend_blocks import _block_time_grid

    state0 = jnp.asarray(block.state0).reshape(-1)
    ts = jnp.asarray(np.asarray(_block_time_grid(block)))  # constant grid, materialised outside the trace
    dt = float(block.dt)
    theta = float(block.metadata.get("theta", 1.0)) if block.metadata else 1.0

    def march(args):
        def _scan_step(u, t):
            un = block.step(u, t, dt, args=args, theta=theta)
            return un, un

        _, traj = jax.lax.scan(_scan_step, state0, ts[:-1])
        return jnp.mean(jnp.concatenate([state0[None, :], traj], axis=0), axis=0)

    return march


def run_adaptive_relocate(fem: Any, spec: AdaptSpec, *, solve_fn: Any = None, **kwargs: Any) -> np.ndarray:
    """Drive ``FEM.solve(adapt=AdaptSpec(relocate=True, ...))`` -- **r-adaptivity** (mesh relocation).

    Relocate the mesh vertices tagged with :meth:`Variable.trainable` at **fixed connectivity** and no new
    DOFs, so a *fixed* node set concentrates at the solution's features; the relocation companion of the
    h-refinement :func:`run_adaptive_solve`. ``spec.relocate_method`` picks how:

    - ``"monge_ampere"`` (what :func:`jno.solve.relocate` builds) re-solves a Monge-Ampère problem each
      round from the **original** mesh with a freshly sampled monitor, so rounds never compound. Needs no
      line search -- the map is a gradient -- but each round is still validity-checked, because holding the
      untagged vertices truncates that map (:func:`_monge_ampere_displacement`).
    - ``"descent"`` walks the vertices down ``spec.objective`` -- evaluated *through the differentiable
      solve* (``∂(solve)/∂X``) -- with a **backtracking line search on ``det J``**, since on a stiff problem
      neither a stock optimiser nor an energy barrier can guarantee validity from outside the step control.

    Either way the loop keeps the **last valid mesh**: a round that would invert or diverge is dropped and
    the loop stops there rather than returning a broken mesh.

    Requires ≥1 coordinate tagged ``domain.variable(region)[i].trainable()`` before ``jno.fem`` (else raises).
    Mutates ``fem`` / its domain to the relocated mesh (like the refinement loop) and returns the solution there;
    ``fem.adapt_history`` traces the per-round objective and energy.

    **Scope.** 2D and 3D; scalar or vector; **any nodal-Lagrange order** (P1/P2/P3 measured); linear,
    nonlinear, transient, periodic and complex — everything but complex-*transient*, which ``jno.fem``
    rejects at build time. It does **not** compose with a geometry term (``coord.d(t) - v``): that driver
    owns the march.

    **The monitor only ever sees VERTEX values** (:func:`_vertex_values`), whatever the element order. At
    P2 and above the mid-edge DOFs are invisible to it, so the mesh adapts to the P1 sub-sampling of the
    field rather than to everything the field resolves. That is a real ceiling on what r-adaptivity buys
    at higher order, not merely an implementation note — a monitor built on the full basis would see more.
    """
    import jax
    import jax.numpy as jnp

    dom = fem.domain
    coord_specs = list(getattr(dom, "_trainable_coords", None) or [])
    if not coord_specs:
        raise ValueError(
            "FEM.solve(adapt=AdaptSpec(relocate=True)) found no trainable mesh coordinates to move. Tag the "
            "vertices to relocate with `x, y[, z] = domain.variable(region, split=True); x.trainable()` "
            "(per component) BEFORE `jno.fem(...)` -- otherwise there is nothing to relocate."
        )
    mode = getattr(fem, "_mode", "linear")
    if mode not in ("linear", "nonlinear", "transient"):
        # A complex *linear* form is mode "linear" (a real 2N block system) and relocates fine. A complex
        # *transient* problem is caught earlier at jno.fem build time (its assembly builds static real blocks
        # and does not thread runtime parameters, so it cannot carry a trainable coordinate yet).
        raise NotImplementedError(f"AdaptSpec(relocate=True): unsupported problem mode {mode!r}.")

    dim = int(dom.dimension)
    cells, _ = _mesh_cells(dom)
    cells_j = jnp.asarray(cells)
    pts0 = np.asarray(dom.mesh.points)[:, :dim].copy()
    pts0_j = jnp.asarray(pts0)
    n_verts = pts0.shape[0]
    names = [sp["name"] for sp in coord_specs]

    def _scatter(vals):
        p = pts0_j
        for sp in coord_specs:
            p = p.at[jnp.asarray(sp["ids"]), sp["axis"]].set(vals[sp["name"]])
        return p

    # Per-block value shape, needed to read VERTEX values out of a solution block. DOFs are node-major
    # (``node*vec + comp``) and a nodal Lagrange element numbers the mesh vertices FIRST, so every
    # component's vertex values are ``blk[: n_verts*vec].reshape(n_verts, vec)``.
    #
    # ``vec`` cannot be guessed from the block LENGTH, which is what this used to do. For a P2 vector
    # field ``nb % n_verts != 0`` (634 against 88 vertices), so the guess fell through to ``blk[:n_verts]``
    # -- the first ``n_verts/vec`` NODES times their components, read as though it were one value per
    # vertex. It did not raise. It relocated against a misread array (measured 1.4e-02 away from the true
    # component-0 vertex values, the scale of the solution itself), and it was the one configuration in a
    # sweep where element quality got WORSE: min |det J| 8.7e-03 -> 4.7e-03. The sibling h-adaptive path,
    # `_solve_vertex_values`, refuses this case outright -- so one of the two silently mangled what the
    # other declined to touch.
    #
    # ``field_points`` and ``offsets`` are per-FEM snapshots (unlike the domain's ``_fem_native_*``
    # records, which the next assembly clobbers), so reading them here is safe.
    _fpts = getattr(fem, "field_points", None)
    _off0 = list(fem.offsets) if getattr(fem, "offsets", None) is not None else None
    _vecs = None
    if _fpts is not None and _off0 is not None and len(_fpts) == len(_off0) - 1:
        _vecs = []
        for _i in range(len(_fpts)):
            _nn = int(np.asarray(_fpts[_i]).shape[0])
            _nb = int(_off0[_i + 1] - _off0[_i])
            _vecs.append(_nb // _nn if _nn and _nb % _nn == 0 else 1)

    def _vertex_values(blk, i):
        """Block ``i``'s values at the mesh VERTICES: ``(n_verts,)`` scalar or ``(n_verts, vec)`` vector."""
        nb = int(blk.shape[0])
        vec = (
            _vecs[i] if (_vecs is not None and i < len(_vecs)) else (nb // n_verts if n_verts and nb % n_verts == 0 else 1)
        )
        if vec > 1:
            if nb < n_verts * vec:
                raise NotImplementedError(
                    f"AdaptSpec(relocate=True): solution block {i} has {nb} DOFs for {n_verts} vertices at "
                    f"vec={vec}, so its vertex values are not a prefix and the monitor cannot be built."
                )
            return blk[: n_verts * vec].reshape(n_verts, vec)
        return blk[:n_verts]

    def _block_energy(u, pts, bounds):
        """Total Dirichlet energy summed over EVERY solution block (``bounds``) — so a scalar, a vector (per
        component), a **complex** field (its real + imaginary blocks) and a coupled multifield all contribute.
        A higher-order block reads its vertex DOFs (see :func:`_vertex_values`)."""
        e = 0.0
        for i in range(len(bounds) - 1):
            e = e + _dirichlet_energy_jax(pts, _vertex_values(u[bounds[i] : bounds[i + 1]], i), cells_j, dim)
        return e

    _march = _transient_march_fn(fem.operator) if mode == "transient" else None

    def _solve_at(vals):
        """The solution at the coordinate values ``vals`` -- differentiable in them (the keystone).
        Linear: ``A(X)⁻¹ b(X)``. Nonlinear: Newton (``custom_root`` keeps ``∂u/∂X`` exact). Transient:
        the *time-averaged* nodal state over the marched trajectory (relocate the mesh for the whole run)."""
        if mode == "linear":
            a_mat, b_vec = fem.operator.evaluate(vals)
            b = jnp.asarray(b_vec).reshape(-1)
            if hasattr(a_mat, "indices"):  # BCOO: differentiable sparse-direct — no O(n²) densify / O(n³) solve
                from .linear import sparse_lu_solve

                return sparse_lu_solve(a_mat, b)  # reverse-mode diff in A(X)'s values and b, so ∂u/∂X still flows
            return jnp.linalg.solve(jnp.asarray(a_mat), b)  # already dense (vertex C¹ / 1D): small, keep dense
        if mode == "nonlinear":
            from .newton_krylov import newton_krylov

            op = fem.operator
            u0 = jnp.zeros((int(op.size),), dtype=jnp.result_type(float))
            return newton_krylov(lambda uu: op.residual(uu, vals), u0)
        return _march(vals)  # transient: time-averaged nodal state over the marched trajectory

    if spec.objective not in ("equidistribution", "huang"):
        raise ValueError(f"AdaptSpec.objective must be 'equidistribution' or 'huang'; got {spec.objective!r}.")
    if spec.relocate_method not in ("descent", "monge_ampere"):
        raise ValueError(f"AdaptSpec.relocate_method must be 'descent' or 'monge_ampere'; got {spec.relocate_method!r}.")

    def _block_defect(u, pts, bounds):
        """Mesh functional summed over every solution block — the mirror of :func:`_block_energy`, so a
        scalar / vector / complex / coupled multifield all contribute their own monitor. Which functional
        is :attr:`AdaptSpec.objective`; both take the same arclength monitor, so switching between them
        changes the functional alone."""
        d_ = 0.0
        for i in range(len(bounds) - 1):
            bf = _vertex_values(u[bounds[i] : bounds[i + 1]], i)
            if spec.objective == "huang":
                d_ = d_ + _huang_ea_jax(pts, pts0_j, bf, cells_j, dim)
            else:
                d_ = d_ + _equidistribution_jax(pts, bf, cells_j, dim)
        return d_

    def _objective(vals):
        """What relocation descends: the monitor's **equidistribution defect**, not the FE energy.

        The energy is still computed and reported in ``adapt_history`` as a diagnostic — it is a useful
        thing to watch, it just makes a bad objective (see :func:`_equidistribution_jax`)."""
        u = _solve_at(vals)
        pts = _scatter(vals)
        bounds = list(fem.offsets) if fem.offsets is not None else [0, int(u.shape[0])]
        return _block_defect(u, pts, bounds), _block_energy(u, pts, bounds)

    val_grad = jax.jit(jax.value_and_grad(lambda arrs: _objective(dict(zip(names, arrs))), has_aux=True))

    def _min_detj(p):
        vv = p[cells]
        ee = np.stack([vv[:, i + 1] - vv[:, 0] for i in range(dim)], axis=1)
        return float(np.min(np.linalg.det(ee.transpose(0, 2, 1))))

    def _moved(arrs):
        p = pts0.copy()
        for i, sp in enumerate(coord_specs):
            p[sp["ids"], sp["axis"]] = np.asarray(arrs[i])
        return p

    floor = spec.quality_floor * _min_detj(pts0)
    arrs = [jnp.asarray(pts0[sp["ids"], sp["axis"]]) for sp in coord_specs]
    history: list[dict] = []

    if spec.relocate_method == "monge_ampere":
        # x = ξ + ∇φ is ABSOLUTE, not incremental: every round re-solves from the same computational
        # mesh with a freshly sampled monitor, so a round never compounds the previous round's error.
        ops = _p1_operators(pts0, cells, dim)
        sg_j, meas_j, wsum_j = (jnp.asarray(o) for o in ops[:3])
        cells_j2, n_local = jnp.asarray(cells), cells.shape[1]

        @jax.jit
        def _ma_round(arrs_in):
            vals = dict(zip(names, arrs_in))
            u = _solve_at(vals)
            pts = _scatter(vals)
            bounds = list(fem.offsets) if fem.offsets is not None else [0, int(u.shape[0])]
            blk = u[bounds[0] : bounds[1]]  # the monitor rides the FIRST field (cf. spec.metric_field)
            nb = int(blk.shape[0])
            veci = nb // n_verts if (n_verts and nb % n_verts == 0) else 1
            bf = blk.reshape(n_verts, veci) if veci > 1 else blk[:n_verts]
            m = _arclength_monitor_jax(bf, sg_j, meas_j, wsum_j, cells_j2, n_local, dim)
            disp = _monge_ampere_displacement(m, ops, cells, dim, n_relax=spec.ma_relax, dt=spec.ma_dt)
            return disp, _block_defect(u, pts, bounds), _block_energy(u, pts, bounds)

        # `max_iters` MOVES need `max_iters + 1` evaluations, so each history entry's ``objective`` is the
        # objective OF its own ``points`` rather than of the mesh one round earlier.
        best = -1
        for it in range(spec.max_iters + 1):
            disp, obj, e = _ma_round(arrs)
            history.append({"step": it, "objective": float(obj), "energy": float(e), "points": _moved(arrs)})
            # Keep the BEST mesh, not the last. Monge-Ampère does not descend this objective -- it solves a
            # different problem -- and holding the untagged vertices truncates its map, so a round genuinely
            # can make the mesh worse (badly, when few vertices are free: 13 of 57 raised the defect ~10%).
            # Descent cannot do this, its line search forbids it; the outer loop has to supply the guarantee.
            if best < 0 or float(obj) < history[best]["objective"]:
                best = it
            if it == spec.max_iters:
                break
            cand = [pts0_j[sp["ids"], sp["axis"]] + disp[jnp.asarray(sp["ids"]), sp["axis"]] for sp in coord_specs]
            # `not (x > 0)` rather than `x <= 0`: the relaxation is explicit in `ma_dt` and diverges past
            # its stability limit, and `nan <= 0` is False -- a diverged solve would sail through the
            # naive test and move the mesh to NaN. Folding should not happen for the whole map (it is a
            # gradient); a truncated map and a diverged relaxation both can.
            if not (_min_detj(_moved(cand)) > 0.0):
                break
            arrs = cand
        # Truncate to the accepted trajectory so the last entry IS the mesh handed back (and an animation
        # of `history` ends on it) rather than a rejected excursion past it.
        history[:] = history[: best + 1]
        arrs = [jnp.asarray(history[best]["points"][sp["ids"], sp["axis"]]) for sp in coord_specs]
        return _finish_relocate(fem, dom, coord_specs, arrs, pts0, n_verts, dim, cells, history, solve_fn, kwargs)

    msq = [jnp.zeros_like(a) for a in arrs]  # RMSProp running average (near-feature gradients dwarf the rest)
    for it in range(spec.max_iters):
        (obj, e), g = val_grad(arrs)
        msq = [0.9 * m + 0.1 * gi**2 for m, gi in zip(msq, g)]
        stepdir = [gi / jnp.sqrt(m + 1e-8) for gi, m in zip(g, msq)]
        a = spec.lr
        for _ in range(25):  # backtracking line search: shrink the step until no element inverts
            cand = [ai - a * si for ai, si in zip(arrs, stepdir)]
            if _min_detj(_moved(cand)) > floor:
                break
            a *= 0.5
        else:
            break  # at the mesh-quality limit -- no admissible step
        arrs = cand
        # ``objective`` is what is descended (the equidistribution defect); ``energy`` is kept as a
        # diagnostic. ``points``: the moved vertices, so a relocation run can be animated.
        history.append({"step": it, "objective": float(obj), "energy": float(e), "points": _moved(arrs)})

    return _finish_relocate(fem, dom, coord_specs, arrs, pts0, n_verts, dim, cells, history, solve_fn, kwargs)


def _finish_relocate(fem, dom, coord_specs, arrs, pts0, n_verts, dim, cells, history, solve_fn, kwargs):
    """Land a finished relocation: check validity, move the domain, drop the trainable tags, re-solve.

    Shared by both :attr:`AdaptSpec.relocate_method` branches -- the mesh they hand over is just a set of
    vertex positions, so everything downstream of "where do the nodes go" is identical."""
    import jno

    final = pts0.copy()
    for i, sp in enumerate(coord_specs):
        final[sp["ids"], sp["axis"]] = np.asarray(arrs[i])
    vv = final[cells]
    ee = np.stack([vv[:, i + 1] - vv[:, 0] for i in range(dim)], axis=1)
    # `not (x > 0)`, not `x <= 0`: NaN must fail this test too (a diverged Monge-Ampere relaxation
    # returns a non-finite mesh, and `nan <= 0` is False). Fail loud rather than re-solve on it.
    if not (float(np.min(np.linalg.det(ee.transpose(0, 2, 1)))) > 0.0):
        raise RuntimeError(
            "FEM.solve(adapt=relocate): the mesh is invalid (min det J <= 0, or non-finite). For "
            "relocate_method='descent' lower AdaptSpec.lr or raise AdaptSpec.quality_floor; for "
            "'monge_ampere' lower AdaptSpec.ma_dt (the relaxation is explicit and has a stability limit)."
        )

    # apply the relocation to the domain (connectivity-preserving), drop the trainable tags, re-solve plainly
    disp = np.zeros((n_verts, dim))
    for i, sp in enumerate(coord_specs):
        disp[sp["ids"], sp["axis"]] = np.asarray(arrs[i]) - pts0[sp["ids"], sp["axis"]]
    move_mesh(dom, disp, copy=False, check=True)
    dom._trainable_coords = []  # relocation consumed the tags; the moved vertices are now the geometry
    for _name, _pred in list(getattr(dom, "_tag_predicates", {}).items()):
        dom.tag(_name, _pred)
    cur = jno.fem(fem._constraints, **fem._fem_kwargs)
    u_final = np.asarray(cur.solve(solve_fn, **kwargs)).reshape(-1)
    fem.__dict__.update(cur.__dict__)
    fem.adapt_history = history
    return u_final


@dataclass
class AdaptiveTrajectory:
    """Output of a **transient adaptive** solve (``FEM.solve(adapt=...)`` on a ``u.t`` problem).

    The mesh changes with time, so this is *not* a single ``(n_save, n_dofs)`` array — each saved
    frame lives on its own adapted mesh: ``times[i]`` ↔ ``states[i]`` (the nodal field) on
    ``meshes[i]`` (a ``(points, cells)`` pair; ``points`` ``(n_i, dim)``, ``cells`` ``(m_i, dim+1)`` —
    the same mesh object is shared between remeshes). Call :meth:`resample` to project every frame onto
    one reference mesh and get a uniform array for post-processing / plotting.
    """

    times: np.ndarray
    states: list
    meshes: list
    layouts: Any = None  # per-frame field layout (offsets/orders/vecs/cells_f/field_points); None = scalar-P1 legacy

    def __len__(self):
        return len(self.times)

    def final(self):
        """``(state, (points, cells))`` at the last time — the solution on the final adapted mesh."""
        return self.states[-1], self.meshes[-1]

    def resample(self, domain: Any, *, field: Any = None, fill: Any = "nearest", tol: float = 1e-9, k: int = 32):
        """Project every saved state onto ``domain``'s vertices. Returns ``(n_save, n_target_vertices)``
        for a single scalar field, ``(n_save, n_fields, n_ref)`` for a coupled all-scalar system, or —
        with ``field=i`` on a vector/mixed (e.g. Taylor-Hood) system — that field's ``(n_save, n_ref)``
        (scalar) / ``(n_save, n_ref, vec_i)`` (vector). The transfer is basis-aware (P1/P2, per component)
        via each frame's recorded per-field layout; a legacy scalar-P1 trajectory (no layouts — e.g. a
        moving-boundary run) uses the plain barycentric transfer of :func:`transfer_solution`. Frames are
        loss-free; this adds one interpolation per frame on demand."""
        import jax.numpy as jnp

        dim = int(domain.dimension)
        tgt = np.asarray(domain.mesh.points)[:, :dim].astype(np.float64)

        if self.layouts is not None:  # basis-aware / value-shape-aware projection from per-frame layouts
            n_fields = len(self.layouts[0]["offsets"]) - 1
            if field is None and any(v != 1 for v in self.layouts[0]["vecs"]):
                raise ValueError(
                    f"resample: this system has a vector field — pass field=i (0..{n_fields - 1}); a vector "
                    "field returns (n_save, n_ref, vec_i)."
                )
            if field is not None and not 0 <= int(field) < n_fields:
                raise ValueError(f"resample: field={field} out of range for {n_fields} field(s).")
            frames = []
            for u, (pts, cells), lay in zip(self.states, self.meshes, self.layouts):
                pts, cells = np.asarray(pts, np.float64), np.asarray(cells, np.int64)
                vals = _eval_fe_fields_at_points(
                    pts,
                    cells,
                    u,
                    lay["offsets"],
                    lay["orders"],
                    lay["cells_f"],
                    lay["vecs"],
                    [tgt] * n_fields,
                    dim=dim,
                    tol=tol,
                    k=k,
                    fill=fill,
                )
                if field is not None:
                    v = vals[int(field)]
                    frames.append(v[:, 0] if v.shape[1] == 1 else v)
                else:
                    blocks = [v[:, 0] for v in vals]  # all scalar (guarded above)
                    frames.append(blocks[0] if n_fields == 1 else jnp.stack(blocks, axis=0))
            return jnp.stack(frames, axis=0)

        # --- legacy scalar-P1 path (no per-frame layouts: a moving-boundary trajectory) ---
        if field not in (None, 0):
            raise ValueError("resample(field=...) needs a layout-carrying trajectory; this is a legacy scalar-P1 result.")
        frames = []
        for u, (pts, cells) in zip(self.states, self.meshes):
            pts, cells = np.asarray(pts, np.float64), np.asarray(cells, np.int64)
            nv = pts.shape[0]
            uu = jnp.asarray(u)
            if uu.shape[0] % nv:
                raise ValueError(
                    f"resample: a frame's state ({int(uu.shape[0])}) is not a whole number of scalar-P1 fields on {nv} vertices."
                )
            nf = uu.shape[0] // nv
            idx, w, inside = _locate_barycentric(pts, cells, tgt, tol=tol, k=k)
            wj, ij = jnp.asarray(w, dtype=uu.real.dtype), jnp.asarray(idx)
            blocks = [jnp.einsum("qk,qk->q", wj, uu[f * nv : (f + 1) * nv][ij]) for f in range(nf)]
            n_out = int((~inside).sum())
            if n_out and fill == "error":
                raise ValueError(
                    f"resample: {n_out} reference vertices fall outside a frame's mesh; use fill='nearest' or a constant."
                )
            if isinstance(fill, (int, float)) and n_out:
                keep = jnp.asarray(inside)
                blocks = [jnp.where(keep, b, jnp.asarray(fill, dtype=uu.dtype)) for b in blocks]
            frames.append(blocks[0] if nf == 1 else jnp.stack(blocks, axis=0))
        return jnp.stack(frames, axis=0)


def _field_layout(fem: Any) -> dict:
    """Per-field layout of an assembled native-Lagrange FEM, for the transient adaptive transfer:
    ``offsets`` (flat block boundaries), ``field_points`` (per-field DOF coords), ``orders`` (per-field
    element order), ``cells_f`` (per-field P{order} connectivity, cell-aligned with the P1 base mesh),
    ``vecs`` (per-field component count). Reads the per-FEM ``offsets``/``field_points`` snapshots plus
    the domain's ``_fem_native_*`` records -- which the NEXT assembly clobbers, so call this right after
    ``jno.fem(...)``. Fail-loud if the problem is not a native nodal-Lagrange assembly (the basis-aware
    transfer only tabulates nodal Lagrange bases)."""
    import jax.numpy as jnp

    d = fem.domain
    off = fem.offsets
    if off is None:  # single scalar field with no block structure
        n_state = int(jnp.asarray(fem._op.state0).reshape(-1).shape[0])
        _meta = getattr(fem._op, "metadata", None) or {}
        # A fused complex transient marches the stacked [Re; Im] state (2n) but its FIELD layout is the
        # n complex DOFs — halve, or the layout would misread [Re; Im] as an interleaved vec-2 field.
        off = [0, n_state // 2 if _meta.get("complex") else n_state]
    off = [int(x) for x in off]
    n_fields = len(off) - 1
    fpts = fem.field_points
    orders = getattr(d, "_fem_native_field_orders", None)
    cells_f = getattr(d, "_fem_native_assembly_cells_all", None)
    if fpts is None or orders is None or cells_f is None or len(fpts) != n_fields or len(orders) != n_fields:
        raise NotImplementedError(
            "transient adaptive remeshing requires a native nodal-Lagrange assembly (its per-field DOF "
            "coordinates / element orders / connectivity drive the basis-aware transfer); this problem "
            "assembled through another route (e.g. non-nodal RT/Nedelec/P0, or a non-native path)."
        )
    fpts = [np.asarray(p) for p in fpts]
    vecs = [(off[i + 1] - off[i]) // max(1, fpts[i].shape[0]) for i in range(n_fields)]
    return {
        "offsets": off,
        "field_points": fpts,
        "orders": [int(o) for o in orders],
        "cells_f": [np.asarray(c) for c in cells_f],
        "vecs": vecs,
    }


def _double_layout(layout: dict) -> dict:
    """The transfer/metric layout for a fused complex transient: the block state is the stacked
    ``[Re; Im]`` pair, so the imaginary half rides as a SECOND COPY of every field — same nodes, same
    order, same vec, offset by the total. The basis-aware transfer then carries both halves with no
    complex branch, exactly as :func:`jno._fem._duplicate_periodic` doubles the periodic blocks."""
    off = layout["offsets"]
    n_half = off[-1]
    return {
        "offsets": off + [n_half + o for o in off[1:]],
        "field_points": layout["field_points"] + layout["field_points"],
        "orders": layout["orders"] + layout["orders"],
        "cells_f": layout["cells_f"] + layout["cells_f"],
        "vecs": layout["vecs"] + layout["vecs"],
    }


def _scalar_vertex_metric(state: Any, layout: dict, mf: int, n_verts: int) -> np.ndarray:
    """Reduce the metric-driving field (possibly vector and/or P2) to a scalar field on the mesh
    VERTICES, to feed the scalar ZZ / Hessian estimator: per-node magnitude across components,
    restricted to the leading ``n_verts`` DOFs (``_promote_to_degree`` keeps the P1 vertices as ids
    ``0..n_verts-1``, and a P1 field's nodes ARE the vertices)."""
    off, vec = layout["offsets"], int(layout["vecs"][mf])
    blk = np.asarray(state[off[mf] : off[mf + 1]]).reshape(-1, vec)  # (n_nodes, vec)
    mag = blk[:, 0] if vec == 1 else np.sqrt((blk**2).sum(axis=1))
    return np.asarray(mag[:n_verts])


def run_adaptive_transient(
    fem: Any,
    spec: AdaptSpec,
    *,
    solve_fn: Any = None,
    save_ts: Any = None,
    nonlinear: Any = None,
    linear: Any = None,
    precond: Any = None,
    **kwargs: Any,
) -> "AdaptiveTrajectory":
    """Drive ``FEM.solve(adapt=spec)`` for a **transient** problem: march the semidiscrete block and,
    every ``spec.every`` steps, remesh from the current field and carry the state onto the new mesh
    (:func:`transfer_solution`), so the mesh **tracks a moving feature**. Returns an
    :class:`AdaptiveTrajectory` (each frame on its own adapted mesh).

    **Fields**: one or several coupled native-Lagrange fields — scalar or **vector**, **P1 or higher
    order (P2)**, and **mixed spaces** (e.g. Taylor-Hood P2 velocity + P1 pressure). State is carried
    across each remesh by a basis-aware, value-shape-aware transfer (:func:`_eval_fe_fields_at_points`);
    ``spec.metric_field`` picks the field driving the (scalar) metric, reduced to a per-vertex magnitude.
    The between-remesh march is :meth:`SemidiscreteTimeBlock.step` over a ``lax.scan`` — a **linear**
    θ-step, or for a **nonlinear** block (e.g. Navier–Stokes) a per-step Newton solve; the
    ``nonlinear=``/``linear=``/``precond=`` slots configure it. The remesh is the non-differentiable outer
    Python loop; the time grid is the block's fixed ``t0..t1`` at ``dt`` (remeshing never drifts time).

    **Scope — fail-loud on the rest** (a mis-transferred solve is silently wrong): **real**,
    **non-periodic**, native nodal-Lagrange fields only (non-nodal RT/Nédélec/P0 raise via
    :func:`_field_layout`); the driver owns the march (no whole-march ``solve_fn``, and ``x0=``/``time=``
    do not compose with a remesh — the DOF layout changes). The forward march is differentiable within
    each fixed-mesh chunk and through the transfers, but the *remesh decisions* are not (the AFEM-inverse
    pattern — freeze the mesh sequence, differentiate the chunks — applies)."""
    import jax
    import jax.numpy as jnp

    import jno

    from .backend_blocks import _block_time_grid

    if fem._constraints is None:
        raise ValueError("FEM.solve(adapt=...) requires a FEM built by jno.fem(...) (its constraint list is retained).")
    if solve_fn is not None:
        raise NotImplementedError(
            "fem.solve(adapt=..., solve_fn=...) is not supported on a transient problem: the adaptive driver "
            "owns the time march (it steps between remeshes). Drop solve_fn (use the default θ-stepper), or drop adapt."
        )
    if getattr(fem, "_periodic", None) is not None:
        raise NotImplementedError(
            "fem.solve(adapt=...) with periodic ties on a transient problem is not supported yet "
            "(the remesh would need a periodic-aware state transfer)."
        )

    d = fem.domain
    dim = int(d.dimension)
    if dim not in (1, 2, 3):
        raise NotImplementedError(f"transient adaptive remeshing supports 1D/2D/3D meshes; got dimension {dim}.")

    cons, kw = fem._constraints, fem._fem_kwargs
    block = fem._op
    n_verts = int(np.asarray(d.mesh.points).shape[0])
    state = jnp.asarray(block.state0).reshape(-1)
    if jnp.iscomplexobj(state):
        raise NotImplementedError(
            "transient adaptive remeshing found a complex-dtype block state — a supported complex "
            "transient marches the REAL stacked [Re; Im] block; this one assembled through another route."
        )
    # A fused COMPLEX transient marches the real stacked [Re; Im] state: the transfer/metric see the
    # DOUBLED layout (the Im half rides as a second copy of every field), the remesh metric is the
    # MODULUS |u| = sqrt(Re² + Im²), and each saved frame is recombined to the complex field the user
    # authored against.
    is_cx = bool(block.metadata and block.metadata.get("complex"))
    # Per-field block layout (offsets / orders / vecs / P{order} connectivity / DOF coords), generalised
    # beyond scalar-P1: vector, higher-order (P2), and mixed (Taylor-Hood) fields are carried across a
    # remesh by the basis-aware transfer below. Read right after assembly — the domain's per-field
    # metadata is clobbered by the next one.
    layout = _field_layout(fem)  # the (complex) FIELD layout — what saved frames are described by
    tlayout = _double_layout(layout) if is_cx else layout  # what the stacked block state is laid out as
    off = layout["offsets"]
    n_fields = len(off) - 1
    cur_nverts = n_verts  # current mesh's vertex count (updates each remesh; drives the metric slice)
    mf = int(spec.metric_field)
    if not 0 <= mf < n_fields:
        raise ValueError(f"AdaptSpec.metric_field={spec.metric_field} is out of range for {n_fields} field(s).")

    def _frame(s):  # a saved frame: the complex field for a fused complex transient, the state itself else
        if not is_cx:
            return s
        half = s.shape[0] // 2
        return s[:half] + 1j * s[half:]

    key = _simplex_cell_key(dim)
    ts = np.asarray(_block_time_grid(block))  # fixed t0..t1 at dt -- unchanged by remeshing
    dt = float(block.dt)
    theta = float(block.metadata.get("theta", 1.0)) if block.metadata else 1.0
    n_steps = len(ts) - 1
    every = max(1, int(spec.every))

    # Transient budget: a CONSTANT target complexity + a FIXED edge-size window (from the initial mesh), so
    # each remesh REDISTRIBUTES ~the same number of DOFs to follow the moving feature — the mesh tracks it
    # and coarsens the wake, instead of ratcheting up by refine_factor every remesh like the steady loop
    # (which grows the mesh toward convergence). Budget = max_dofs if given, else the initial vertex count.
    h_typ0 = _mean_edge_length(d)
    hmin = spec.hmin if spec.hmin is not None else h_typ0 / 50.0
    hmax = spec.hmax if spec.hmax is not None else h_typ0 * 2.0
    target = float(spec.max_dofs) if spec.max_dofs is not None else float(n_verts)

    def _snapshot():
        return (np.asarray(d.mesh.points)[:, :dim].astype(np.float64), np.asarray(d.mesh.cells_dict[key]).astype(np.int64))

    cur_mesh = _snapshot()
    times, states, meshes, layouts = [float(ts[0])], [_frame(state)], [cur_mesh], [layout]
    history: list[dict] = []
    cur = fem

    i = 0
    while i < n_steps:
        chunk = int(min(every, n_steps - i))
        blk = block  # capture the current-mesh block for the scan closure
        # Per-step solve config (nonlinear=/linear=/precond= slots) composed on the CURRENT-mesh block, so
        # a Navier-Stokes chunk gets its Newton (picard/damping) driver — re-composed after each remesh.
        # No slots -> (None, None) -> block.step's defaults (theta linear solve / newton_krylov), unchanged.
        if nonlinear is not None or linear is not None or precond is not None:
            from .solver_api import compose_transient_step_solvers

            lin_s, nonlin_s = compose_transient_step_solvers(nonlinear, linear, precond, cur, blk)
        else:
            lin_s, nonlin_s = None, None

        def _body(u, t, _blk=blk, _l=lin_s, _n=nonlin_s):
            un = _blk.step(u, t, dt, theta=theta, linear_solve=_l, nonlinear_solve=_n)
            return un, un

        state, traj = jax.lax.scan(_body, state, jnp.asarray(ts[i : i + chunk], dtype=state.dtype))
        for j in range(chunk):
            i += 1
            times.append(float(ts[i]))
            states.append(_frame(traj[j]))
            meshes.append(cur_mesh)
            layouts.append(layout)
        if i >= n_steps:
            break

        # remesh from the metric-driving field (fixed budget above), then carry every field's block over
        if is_cx:  # the modulus drives the metric — refining on Re alone would miss a rotating phase
            _re = _scalar_vertex_metric(state, tlayout, mf, cur_nverts)
            _im = _scalar_vertex_metric(state, tlayout, mf + n_fields, cur_nverts)
            u_v = np.sqrt(_re**2 + _im**2)
        else:
            u_v = _scalar_vertex_metric(state, tlayout, mf, cur_nverts)  # scalar VERTEX field (vector/P2 reduced)
        old_pts, old_cells = cur_mesh
        if spec.anisotropic:
            metric = hessian_metric(d, u_v, target_complexity=target, hmin=hmin, hmax=hmax)
            remesh_with_mmg(d, metric, copy=False, hmin=hmin, hmax=hmax, hgrad=3.0)
        else:
            eta, _est = zz_error_indicators(d, u_v)
            remesh_with_mmg(
                d, size_field_from_marks(d, dorfler_mark(eta, spec.theta), refine_factor=spec.refine_factor), copy=False
            )
        for _name, _pred in list(getattr(d, "_tag_predicates", {}).items()):  # flux tags re-derive on the new facets
            d.tag(_name, _pred)
        # Drop the cached p.pin() gauge nodes so `_lower_gauge_pin` re-creates the single-vertex pin region
        # on the NEW mesh (its point-region does not survive a remesh) — needed for a Taylor-Hood pressure
        # gauge under adapt=. Harmless when there is no pin.
        d.__dict__.pop("_gauge_pin_coords", None)
        cur = jno.fem(cons, **kw)  # re-assemble the same transient problem on the refined mesh
        block = cur._op
        cur_mesh = _snapshot()
        new_n = int(cur_mesh[0].shape[0])
        new_layout = _field_layout(cur)  # per-field layout on the refined mesh (read now, before any re-assembly)
        new_tlayout = _double_layout(new_layout) if is_cx else new_layout
        # Basis-aware, value-shape-aware state transfer: evaluate each OLD field at its NEW per-field DOF
        # coordinates using the OLD element's shape functions (P1 vertices / P2 midpoints / vector comps).
        # For a fused complex transient the doubled layout carries the Re and Im halves separately.
        vals = _eval_fe_fields_at_points(
            old_pts,
            old_cells,
            state,
            tlayout["offsets"],
            tlayout["orders"],
            tlayout["cells_f"],
            tlayout["vecs"],
            new_tlayout["field_points"],
            dim=dim,
        )
        state = jnp.concatenate([v.reshape(-1) for v in vals])
        off, layout, tlayout, cur_nverts = new_layout["offsets"], new_layout, new_tlayout, new_n
        history.append({"t": float(ts[i]), "n_dofs": int(tlayout["offsets"][-1]), "fields": n_fields})

    fem.__dict__.update(cur.__dict__)  # rebind to the final adapted mesh (matches the steady driver)
    fem.adapt_history = history
    return AdaptiveTrajectory(np.asarray(times), states, meshes, layouts=layouts)


def _geometry_motion_specs(fem: Any, dom: Any) -> list[dict]:
    """One record per geometry term: which mesh vertices it moves, along which axis, and the residual to
    evaluate. See :func:`jno.trace.mesh_velocity` for what makes a term a geometry term."""
    from ...trace import Variable, frozen_fields_in, mesh_velocity, substitute

    pts = np.asarray(dom.mesh.points)
    specs = []
    for term in fem._geometry:
        coord, _tvar, jac = mesh_velocity(term)
        ids = np.asarray(Variable._region_vertex_ids(dom, coord.tag, pts), dtype=int)
        if ids.size == 0:
            raise ValueError(f"jno.fem: the geometry term on region {coord.tag!r} matches no mesh vertices.")
        # Store the BARE Placeholder, not a typed view. A velocity like `0.05*(Tf.x*nx + Tf.y*ny)*ny` comes
        # back wrapped, and both `substitute` and `frozen_fields_in` walk the graph -- handed the wrapper
        # they find nothing, so the frozen field is never re-pinned and the derivative is never replaced.
        # The term then evaluates as `d(static coordinate)/dt`, which is identically zero: no motion, no error.
        bare = term._expr if hasattr(term, "_expr") else term
        # The two probe expressions are built ONCE here, not per step. They are what recovers the velocity
        # from the residual (see :func:`_geometry_velocity`), and their structure does not depend on the
        # mesh -- only their *values* do. Building them once is what lets `Crux.eval`'s compiled-function
        # cache, which is keyed on the op object, hit for the whole march instead of compiling every step.
        zero = 0.0 * coord
        specs.append(
            {
                "ids": ids,
                "axis": int(coord.dim[0]),
                "term": bare,
                "jac": jac,
                "coord": coord,
                "expr0": substitute(bare, {jac: zero}),
                "expr1": substitute(bare, {jac: zero + 1.0}),
            }
        )
    for s in specs:
        s["frozen"] = frozen_fields_in(s["term"])
    return specs


def _tags_read(expr) -> list:
    """Every region tag the expression reads, in first-seen order.

    The mesh-motion driver rewrites a context entry per tag it hands to the compiled velocity, and it has
    to rewrite **all** of them: everything left at its seed value is frozen for the whole march. A law
    reading a second region's coordinates was silently stale -- with two tags over the identical vertex
    set and ``dy/dt = y``, reading its own tag compounded to 1.46410 while reading the twin returned
    1.40000, exactly the frozen-seed answer, with no error.

    Walks with :func:`jno.trace._iter_placeholder_children`, the single traversal shape the other trace
    walks use (:func:`mesh_velocity`, :func:`frozen_fields_in`)."""
    from ...trace import Placeholder, Variable, _iter_placeholder_children

    node = expr._expr if hasattr(expr, "_expr") else expr
    seen: set = set()
    tags: list = []

    def visit(n):
        if not isinstance(n, Placeholder) or id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, Variable):
            t = getattr(n, "tag", None)
            if isinstance(t, str) and t not in tags:
                tags.append(t)
        for kind, _attr, val in _iter_placeholder_children(n):
            for c in val if kind == "list" else (val,):
                visit(c)

    visit(node)
    return tags


def _tag_facet_vertex_ids(dom: Any, tag: str, dim: int) -> np.ndarray:
    """The tag's boundary facets as MESH-VERTEX INDICES ``(E, dim)``.

    ``BoundaryRegion`` stores facets as vertex *coordinates*; under a connectivity-preserving move the
    incidence never changes, so resolving it to indices once lets the normals be recomputed from moved
    positions without touching the domain object."""
    from scipy.spatial import cKDTree

    region = (getattr(dom, "_boundary_regions", None) or {}).get(tag)
    ents = None if region is None else (region.edges if dim == 2 else region.triangles)
    if ents is None or len(ents) == 0:
        return np.zeros((0, dim), dtype=np.int64)
    ents = np.asarray(ents)[:, :, :dim]
    tree = cKDTree(np.asarray(dom.mesh.points)[:, :dim])
    return np.asarray(tree.query(ents.reshape(-1, dim))[1]).reshape(ents.shape[0], ents.shape[1]).astype(np.int64)


def _facet_outward_sign(dom: Any, facet_ids: np.ndarray, dim: int) -> np.ndarray:
    """``(E,)`` of ±1 fixing the outward orientation of each tag facet, from the mesh **topology**.

    Delegates to :func:`jno.utils.solver.fem_facets.compute_face_normals`, which orients each boundary
    facet *away from its parent cell's opposite vertex*. That is exact for any topology; the mesh-centroid
    rule this replaces is only exact for a star-shaped domain, and on an annulus it gave the inner hole a
    normal pointing into the solid (``n·r̂`` = +1 where outward-from-material is -1). The old test did not
    see it because it asserted *agreement* with ``domain.normals_by_tag`` -- and both carried the same
    wrong convention, which is exactly how a shared representation hides a defect.

    The sign is resolved once on the seed mesh and then frozen, exactly as ``assemble_fem_native`` freezes
    it for :func:`~jno.utils.solver.fem_native._face_normals_jax`: it is locally constant, flipping only
    at element inversion, which the march's tangle check already rejects."""
    from .fem_facets import build_facet_connectivity, compute_face_normals

    facet_ids = np.asarray(facet_ids, dtype=np.int64)
    if facet_ids.size == 0:
        return np.zeros((0,), dtype=float)
    cell_type = "triangle" if dim == 2 else "tetrahedron"
    cells, _ = _mesh_cells(dom)
    conn = build_facet_connectivity(cells, cell_type)
    good = compute_face_normals(np.asarray(dom.mesh.points), conn, cells, cell_type)

    # Match each tag facet to its global boundary facet by its (order-independent) vertex set. Sorted
    # rows viewed as single records give a lexicographic key, so this is one searchsorted rather than a
    # dict lookup per facet -- 29 ms at 2340 facets and growing, on a library whose cost is its build.
    def _rowkey(a):
        a = np.ascontiguousarray(np.sort(np.asarray(a, dtype=np.int64), axis=1))
        return a.view([("", a.dtype)] * a.shape[1]).ravel()

    gkey, tkey = _rowkey(conn.face_nodes), _rowkey(facet_ids)
    order = np.argsort(gkey, kind="stable")
    pos = np.searchsorted(gkey[order], tkey)
    ok = pos < gkey.size
    j = order[np.clip(pos, 0, max(gkey.size - 1, 0))]
    ok &= gkey[j] == tkey
    if not ok.all():
        raise ValueError(
            f"jno.fem: a moving region names {int((~ok).sum())} facet(s) that are not boundary facets of "
            "the mesh. An interior facet has no outward normal -- tag the region on the boundary."
        )

    v = np.asarray(dom.mesh.points)[facet_ids, :dim]  # (E, k, dim)
    if dim == 2:
        t = v[:, 1] - v[:, 0]
        raw = np.stack([t[:, 1], -t[:, 0]], axis=1)
    else:
        raw = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    return np.where(np.einsum("ij,ij->i", raw, good[j]) >= 0.0, 1.0, -1.0)


def _vertex_normals_jax(pts, facet_ids, dim: int, sign):
    """Per-vertex outward unit normals from moved vertex positions -- pure JAX, differentiable in ``pts``.

    The facet normals come from :func:`~jno.utils.solver.fem_native._face_normals_jax` -- the one traced
    normal builder in the library, already used by the native surface assembly -- with the frozen outward
    ``sign`` from :func:`_facet_outward_sign`. Only the per-vertex averaging is here: accumulate each
    facet's normal at its vertices and renormalise, mirroring ``domain_class.py:1450-1458``. With the
    incidence resolved to indices that averaging is a scatter-add, which is what makes it traceable.

    This used to carry its own facet-normal geometry *and* its own orientation rule, i.e. a second copy of
    a quantity the library already computes. That is the shape this codebase's normal bugs keep taking, so
    there is now one producer and this is a consumer of it."""
    import jax.numpy as jnp

    from .fem_native import _face_normals_jax

    n = _face_normals_jax(pts, facet_ids, jnp.asarray(sign, dtype=pts.dtype))
    acc = jnp.zeros_like(pts)
    for a in range(facet_ids.shape[1]):
        acc = acc.at[facet_ids[:, a]].add(n)
    return acc / (jnp.linalg.norm(acc, axis=1, keepdims=True) + 1e-30)


def _geometry_velocity_fn(spec: dict, dom: Any):
    """Build ``velocity(pts, state, params, t) -> (n_ids,)`` for one geometry term: a **traced,
    differentiable** function of the vertex positions, evaluated in the spec's own ``ids`` order.

    This is what lets the march be scanned. :func:`Crux.eval` compiles each expression to
    ``raw_fn(models, ctx, ...)`` where ``ctx`` is a pytree of **arrays** (``jno/core.py:5016-5029``), so
    handing it a ctx whose tag entries are built from the *carried* positions makes the whole evaluation
    traceable -- rather than going through ``prepare_domain_data``, which mutates host state and cannot run
    inside ``lax.scan``. The compiler is reused rather than a second velocity evaluator written: two
    evaluators for one expression drift apart silently.

    Everything position-independent is resolved ONCE here -- the compiled expressions, the base context,
    the sample<->vertex permutations, the facet incidence. Only values are rebuilt per call.

    The two-point recovery is unchanged (see :func:`_geometry_velocity`): the term is a residual
    ``a·d(coord)/dt - v``, so evaluating it with the derivative replaced by 0 and by 1 gives ``-v`` and
    ``a - v``, hence ``v = -r0/(r1 - r0)``."""
    import functools

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from scipy.spatial import cKDTree

    import jno

    from ...core import cse, fuse_laplacian
    from ...trace_compiler import TraceCompiler

    dim = int(dom.dimension)
    tag = spec["coord"].tag
    crux = jno.core([spec["expr0"], spec["expr1"]], domain=dom)
    base_ctx = dict(crux.prepare_domain_data(dom).context)
    models = eqx.tree_inference(crux._unwrapped_models)

    def _compile(op):
        raw = TraceCompiler.compile_traced_expression(cse(fuse_laplacian(op)), crux.all_ops)
        return eqx.filter_jit(functools.partial(raw, min_consecutive=1))

    fn0, fn1 = _compile(spec["expr0"]), _compile(spec["expr1"])

    # How EVERY tag the law reads relates to mesh vertices -- not just the driven one. A tag is sampled as
    # (B, T, N, D) on a transient domain; the samples are a spatial snapshot, so one (N, D) slice describes
    # them all. Resolved on the SEED mesh and then held fixed, which is the material-set (Lagrangian)
    # convention a moving driven set needs -- re-deriving it from moved points is what let a `where=`
    # region drift out of its own tag.
    verts = np.asarray(dom.mesh.points)[:, :dim]
    vtree = cKDTree(verts)
    live_tags: dict = {}  # tag -> {"sample_vid", "nkey", "facet_ids"}
    for _t in _tags_read(spec["term"]):
        # `n_<tag>` is that tag's NORMALS, not a point set -- `domain.variable(tag, normals=True)` hands
        # back normal components carrying it as their own tag. It is refreshed together with its owning
        # tag below; matching it against vertices would (and did) fail with a gap of exactly 1.0, a unit
        # normal compared against a coordinate.
        if _t.startswith("n_") and _t[2:] in base_ctx:
            continue
        _e = base_ctx.get(_t)
        if _e is None:
            continue  # not a context-backed region (a bare symbol / a parameter tag): nothing to refresh
        _a = np.asarray(_e)
        if _a.ndim < 2 or _a.shape[-1] < dim:
            continue
        _n = int(_a.shape[-2])
        _pts = _a.reshape(-1, _a.shape[-1])[:_n, :dim]
        _gap, _vid = vtree.query(_pts)
        # A tag whose samples are NOT mesh vertices (a `gauss_*` quadrature pool, a mesh-free resampled
        # region) cannot be moved with the mesh, so its values would go stale exactly as the second-tag
        # defect did. Refuse, rather than march on frozen coordinates.
        _tol = 1e-6 * max(float(np.ptp(_pts)), 1.0) if _pts.size else 1.0
        if _pts.size and float(np.max(_gap)) > _tol:
            raise NotImplementedError(
                f"jno.fem: the geometry term on region {tag!r} reads region {_t!r}, whose sample points are "
                f"not mesh vertices (worst gap {float(np.max(_gap)):.2e} > {_tol:.1e}) -- a quadrature pool "
                "or a mesh-free region. The driver cannot move those with the mesh, so the law would read "
                "the seed positions for the whole march. Write the law in terms of a vertex-backed region."
            )
        _nk = f"n_{_t}"
        _fids = _tag_facet_vertex_ids(dom, _t, dim) if _nk in base_ctx else None
        live_tags[_t] = {
            "sample_vid": np.asarray(_vid, dtype=np.int64),
            "nkey": _nk if _nk in base_ctx else None,
            "facet_ids": _fids,
            "sign": None if _fids is None else _facet_outward_sign(dom, _fids, dim),
        }

    # vertex -> sample, for the spec's DRIVEN ids: which entry of the evaluated residual belongs to which
    # driven vertex. Position-based, as the boundary readout is.
    ctx_tag = np.asarray(base_ctx[tag])
    n_samp = int(ctx_tag.shape[-2])
    tag_pts = ctx_tag.reshape(-1, ctx_tag.shape[-1])[:n_samp, :dim]
    gap, perm = cKDTree(tag_pts).query(verts[np.asarray(spec["ids"], dtype=int)])
    tol = 1e-6 * max(float(np.ptp(tag_pts)), 1.0)
    if float(np.max(gap)) > tol:
        raise ValueError(
            f"jno.fem: could not match region {tag!r}'s sample points to its mesh vertices "
            f"(worst gap {float(np.max(gap)):.2e} > {tol:.1e}). The velocity cannot be attributed to vertices."
        )
    perm = np.asarray(perm, dtype=np.int64)
    # A state-reading interface law (a Stefan front reading the solved field) carries FrozenField nodes.
    # Their nodal values and the mesh they sit on are delivered through the context -- see
    # `trace_evaluator._eval_frozen_field`. Baking them into the graph with `refreeze` is what forced a
    # host sync and pinned the law to the SEED state on the SEED mesh.
    frozen_ids = [getattr(f, "frozen_id", None) for f in (spec.get("frozen") or [])]

    # A parameter written INTO the law (`yb.d(tb) - v0*yb`) is what makes the interface law itself a
    # design variable. It reaches the compiled expression through `models`, which is an ARGUMENT of
    # `raw_fn`, so substituting a traced value there keeps `d/d(v0)` connected. Without this the
    # parameter is not merely non-differentiable but INERT: a geometry term is pulled out before the
    # weak-form assembly, so it never reaches the block's `runtime_parameter_exprs`, and the march ran
    # with the parameter's seed (measured: no motion at all, and no error).
    from .parametric_helpers import _collect_runtime_parameter_exprs

    _law_nodes: dict = {}
    for _e in (spec["expr0"], spec["expr1"]):
        _collect_runtime_parameter_exprs(_e, _law_nodes)
    law_param_lids = {nm: int(node.model.layer_id) for nm, node in _law_nodes.items()}

    def _with_value(model, value):
        """Replace a parameter model's single array leaf with ``value`` (traced or concrete)."""
        arrs, static = eqx.partition(model, eqx.is_inexact_array)
        leaves, treedef = jax.tree_util.tree_flatten(arrs)
        if len(leaves) != 1:
            raise NotImplementedError(
                f"jno.fem: a geometry-term parameter must be a plain `jno.np.parameter`; this one carries "
                f"{len(leaves)} array leaves, so its value cannot be substituted unambiguously."
            )
        new = jax.tree_util.tree_unflatten(treedef, [jnp.asarray(value, dtype=leaves[0].dtype).reshape(leaves[0].shape)])
        return eqx.combine(new, static)

    def _write(entry, values):
        """Broadcast an (N, D) spatial array into a context entry's (..., N, D) layout.

        Every leading (batch / time) slice gets the same positions: the mesh is one geometry per step, not
        one per sampled time. Writing only the first slice happens to work for a purely spatial expression
        and silently would not for anything that reads across the time axis."""
        e = jnp.asarray(entry)
        return jnp.broadcast_to(values.reshape((1,) * (e.ndim - 2) + values.shape), e.shape)

    def velocity(pts, state=None, params=None, t=None):
        mdl = models
        for _nm, _lid in law_param_lids.items():
            if params is not None and _nm in params:
                mdl = {**mdl, _lid: _with_value(mdl[_lid], params[_nm])}
        ctx = dict(base_ctx)
        for _t, _lt in live_tags.items():
            ctx[_t] = _write(base_ctx[_t], pts[_lt["sample_vid"]])
            if _lt["facet_ids"] is not None:
                ctx[_lt["nkey"]] = _write(
                    base_ctx[_lt["nkey"]],
                    _vertex_normals_jax(pts, _lt["facet_ids"], dim, _lt["sign"])[_lt["sample_vid"]],
                )
        if t is not None and "__time__" in base_ctx:
            # The TIME the velocity is evaluated at. A temporal variable does not live in the tag's own
            # pool -- the tag entry is (B, T, N, D) with D = dim, purely spatial -- it resolves from the
            # separate `__time__` context key (``trace_evaluator._eval_variable``, axis == "temporal").
            # Left at the seed grid, `r.reshape(-1)[perm]` reads the FIRST time slice, so a law with an
            # explicit `t` marched at t = ts[0] forever: `yb.d(tb) - tb` gave no motion at all, and
            # `- (1 + tb)` gave 1.40000 where forward Euler is 1.46000. No error either way.
            #
            # Every slice is filled with the SAME value rather than the step's sub-grid, which is what
            # makes the existing `[perm]` gather correct by construction: the mesh has one geometry (and
            # therefore one velocity) per step, not one per sampled time.
            ctx["__time__"] = jnp.full_like(jnp.asarray(base_ctx["__time__"]), t)
        if frozen_ids:
            if state is None:
                raise ValueError(
                    f"jno.fem: the geometry term on region {tag!r} reads the solved field, so its velocity "
                    "needs the current state. This is a driver bug -- report it."
                )
            ctx["__mesh_points__"] = pts
            ctx["__frozen_values__"] = {fid: jnp.asarray(state).reshape(-1) for fid in frozen_ids}
        r0 = jnp.asarray(fn0(mdl, ctx, batchsize=None, key=None)).reshape(-1)
        r1 = jnp.asarray(fn1(mdl, ctx, batchsize=None, key=None)).reshape(-1)
        # `[perm]` gathers the driven vertices out of the driven tag's FIRST (B, T) slice. With `__time__`
        # held constant above every slice carries the same value, so which one is read does not matter.
        return (-r0 / (r1 - r0))[perm]

    velocity.law_params = frozenset(law_param_lids)  # names the driver must accept from `fem.solve(...)`
    return velocity


def _geometry_velocity(spec: dict, dom: Any, state: Any) -> np.ndarray:
    """Evaluate one geometry term's velocity on the current mesh and state.

    The term is a *residual* ``a·d(coord)/dt - v = 0``, not a velocity, so the velocity is recovered by
    evaluating it twice with the derivative node replaced by a **constant field**: at 0 it reads ``-v``, at 1
    it reads ``a - v``, hence ``a`` by difference and ``v = -r0/a``. Two evaluations, no symbolic
    rearrangement, and it handles any term linear in the derivative (``2*yb.d(tb) - 1`` gives ``v = 0.5``).
    The stand-in is ``0*coord`` rather than a bare float so the expression keeps its shape *and* its domain
    reference -- a constant velocity would otherwise leave nothing for the evaluator to infer a domain from.

    A frozen field in the term is re-pinned to the live ``state`` first, which is what lets an interface law
    read the solution (a Stefan front ``-(k/L)·∇T·n``). ``substitute`` matches by identity and leaves nodes
    off the replacement path shared, so the derivative node survives that first pass and can still be found.

    **Not used by the march** — :func:`_geometry_velocity_fn` is, because this one goes through
    ``Crux.eval``, which mutates host state and so cannot run inside a ``lax.scan``. This is kept as the
    independent *oracle* the traced route is checked against (two evaluators for one expression drift
    apart silently, so the parity test is what keeps them honest).

    **Oracle scope: time-independent laws only.** It evaluates against the domain's own ``__time__`` grid
    and takes no ``t``, where the traced route is given the step time. Giving it one would mean a third
    substitution on top of the frozen-field/derivative pair below, whose ordering is already load-bearing
    (see :func:`_at`). A law with an explicit ``t`` is checked against its closed-form forward-Euler answer
    instead, which is a stronger oracle than this one anyway.
    """
    from ...trace import refreeze, substitute

    live = {f: refreeze(f, np.asarray(state)) for f in spec["frozen"]}

    def _ev(node):
        # One `Crux` per term, built once and reused. `Crux.eval` keys its compiled-function cache on the
        # op OBJECT, so a *static* expression hits that cache and is compiled once for the whole march;
        # `domain=` is what makes it re-read the moved mesh (without it the call returns the mesh the crux
        # was built on). A term reading a frozen field still misses, because `refreeze` mints fresh nodes
        # each step by design -- that is the 3b half, and it needs the state delivered through the context
        # rather than baked into the graph.
        crux = spec.get("_crux")
        if crux is None:
            import jno

            crux = spec["_crux"] = jno.core([spec["expr0"], spec["expr1"]], domain=dom)
        return np.asarray(crux.eval([node], domain=dom)).reshape(-1)

    def _at(value):
        """Replace the derivative node with ``value``, THEN re-pin the frozen fields — not the other way
        round. ``substitute`` rebuilds every ancestor of what it replaces, so re-pinning first would rebuild
        the derivative node's parents and lose the identity this second substitution matches on; the
        replacement would silently do nothing and the recovered coefficient would read zero everywhere."""
        return _ev(substitute(value, live) if live else value)

    r0, r1 = _at(spec["expr0"]), _at(spec["expr1"])
    a = r1 - r0
    if not np.all(np.abs(a) > 1e-30):
        raise ValueError(
            f"jno.fem: the geometry term on region {spec['coord'].tag!r} has a vanishing coefficient on "
            "d(coord)/dt somewhere, so it states no velocity there. Write it as `coord.d(t) - velocity`."
        )
    v = -r0 / a

    # The velocity comes out in the region's SAMPLE order, which is not the mesh-vertex order and need not
    # even have the same length (a tag may sample a point that is not a vertex). Align by position, as the
    # boundary readout does -- never by index, which would silently permute the motion.
    #
    # The sample coordinates are read straight out of `domain.context` rather than evaluated through the
    # trace: it is the very array the trace would gather from, in the same order (checked at 2 mesh sizes),
    # and evaluating it instead cost two evaluations per term per step -- half of this driver's trace work.
    #
    # Reusing one compiled `jno.core` across steps IS safe, as long as `domain=` is passed on every call:
    # `Crux.eval` re-prepares the domain data when given one and reuses the cached data when not, so a bare
    # `core.eval([expr])` returns the mesh it was built on (measured 0.5 where the moved mesh gives 0.75)
    # while `core.eval([expr], domain=d)` returns 0.75 correctly. It is not done here because it is not
    # worth much: measured 1.10x over 12 steps, since the per-call cost is the domain re-preparation rather
    # than compilation. Cache the core when the re-preparation itself gets cheaper, not before.
    dim = int(dom.dimension)
    ctx = np.asarray(dom.context[spec["coord"].tag])
    tag_pts = ctx.reshape(-1, ctx.shape[-1])[: ctx.shape[-2], :dim] if ctx.ndim > 2 else ctx[:, :dim]
    if v.shape[0] != tag_pts.shape[0]:
        raise ValueError(
            f"jno.fem: the geometry term on region {spec['coord'].tag!r} evaluated to {v.shape[0]} values but the "
            f"region samples {tag_pts.shape[0]} points. A velocity must be one value per sample point; an explicit "
            "dependence on the time variable is not supported here (a region samples space x time) -- write the "
            "law in terms of the coordinates and the solved field instead."
        )
    from scipy.spatial import cKDTree

    verts = np.asarray(dom.mesh.points)[spec["ids"], :dim]
    dist, perm = cKDTree(tag_pts).query(verts)
    # 1e-6, not 1e-8: the sampled context is float32, so an exact match still lands ~1e-7 out. A genuine
    # mismatch is a point that is not a vertex at all, i.e. ~one element away -- five orders clear of this.
    tol = 1e-6 * max(float(np.ptp(tag_pts)), 1.0)
    if float(np.max(dist)) > tol:
        raise ValueError(
            f"jno.fem: could not match region {spec['coord'].tag!r}'s sample points to its mesh vertices "
            f"(worst gap {float(np.max(dist)):.2e} > {tol:.1e}). The velocity cannot be attributed to vertices."
        )
    return v[perm]


def run_mesh_motion(fem: Any, *, solve_fn: Any = None, **kwargs: Any) -> "AdaptiveTrajectory":
    """March a transient problem whose ``jno.fem([...])`` list contains **geometry terms** — the mesh moves
    as those terms say, the physics marches on the moved mesh, and the state is carried across each move.

    Each step: evaluate every geometry term's velocity on the current mesh and state, scatter it into the
    vertices and axes those terms name, harmonically extend over everything they do *not* name
    (:func:`harmonic_extension`), move, re-assemble, and carry the state onto the moved mesh by
    conservative L2 projection (:func:`_l2_transfer_jax`). Returns an :class:`AdaptiveTrajectory`, one
    frame per moved mesh.

    **Method / scope** (house rule: fail loud on the rest).

    - **Operator-split ALE, explicit in the velocity.** The velocity is evaluated from the state at the
      *start* of each move, not solved implicitly with the interface position, so the scheme is first-order
      in the step — **measured**, not asserted: against a manufactured solution on a translating domain
      the observed rates are 1.14 / 1.12 / 1.12 / 1.10 moving, and 0.99 / 1.01 / 1.04 / 1.10 still. The
      motion multiplies the error *constant* by ~3x (that is the state transfer) and leaves the *order*
      intact. Refining ``h`` converges too, at 1.51 → 1.76 toward the expected 2 for P1, and P2 is ~18x
      more accurate than P1 on the same moving mesh, so higher order still pays here.

      A caveat for anyone repeating that measurement: the temporal and spatial errors have **opposite
      signs**, so comparing directly against ``u*`` shows rates of +1.4 then −0.4 as they cancel and
      separate again. That looks like a scheme that stops converging and is only a contaminated
      measurement — compare against a fine-``dt`` reference on the *same* mesh instead, which cancels the
      spatial error exactly.

      The term-list spelling reads like a coupled equation and this is not one — moving the mesh
      implicitly would need the coordinates as unknowns in the monolithic system *and* the ALE
      convective ``(c-w)·∇u`` term. The state is re-projected onto the moved mesh, which transports the
      field under the mesh motion; that is right for a field whose material is quasi-stationary while the
      domain deforms, and wrong when a represented material velocity differs from the mesh velocity (it
      would double-count advection).
    - **The transfer is diffusive**, though far less so than the pointwise re-interpolation it replaced
      (a marginally-resolved peak loses ~9 % rather than ~33 %, and refining ``dt`` now helps instead of
      hurting). See :func:`_l2_transfer_jax` for the measurements and for what would remove it entirely.
    - **Backward Euler only** in practice: ``θ`` is read from the block's metadata, and the only way to
      set it is ``fem.solve(time=jno.solve.theta(...))``, which the slot guard below rejects. Crank–
      Nicolson on a moving mesh is not reachable today.
    - **Connectivity-preserving only.** A move that would invert an element raises (:func:`move_mesh`
      ``check``); remesh-on-tangle for large deformation is the next extension. Reduce the step or the
      motion.
    - Boundary conditions on a moving surface must be **natural** or tied to a whole-boundary / held tag —
      those re-derive on the moved mesh. A Dirichlet BC pinned to the moving surface by a spatial
      sub-predicate would not follow the motion.
    - **Any nodal-Lagrange field(s), real, non-periodic** — scalar or vector, P1 or higher, mixed orders
      across a coupled system (a Taylor–Hood pair moves as one), in 2D or 3D, linear or **nonlinear** (the
      block's Newton branch). All of those are covered by tests. Complex, periodic, a non-nodal family
      (RT / Nédélec / Hermite / Argyris / Morley) and a custom ``solve_fn`` each raise. Note the nonlinear
      branch does not take ``_step_solve``: ``SemidiscreteTimeBlock.step`` threads ``linear_solve`` only
      through the linear path, so a nonlinear march runs ``newton_krylov`` at its own defaults.
    - **Needs ``jax_enable_x64``** and raises without it — see the guard below for the measurement.
    - **Every frame on the block's own time grid.** There is no ``save_ts=``: it used to be accepted here
      and silently ignored (a request for 3 frames returned the grid's 4), and it evaded the
      unknown-keyword check below precisely by being a named parameter. It now raises with everything
      else. Sampling *between* frames would mean interpolating a state across two different meshes — a
      second transfer error on top of the march's own, not a formatting choice.
    """
    import jax
    import jax.numpy as jnp

    import jno

    from .backend_blocks import _block_time_grid

    if fem._constraints is None:
        raise ValueError("A geometry term requires a FEM built by jno.fem(...) (its constraint list is retained).")
    if solve_fn is not None:
        raise NotImplementedError(
            "fem.solve(solve_fn=...) with a geometry term is not supported: the mesh-motion driver owns the "
            "time march. Drop solve_fn (use the default θ-stepper), or drop the geometry term."
        )
    if getattr(fem, "_periodic", None) is not None:
        raise NotImplementedError("A geometry term with periodic ties is not supported yet.")
    if not jax.config.jax_enable_x64:
        # The state transfer projects onto the moved mesh, which means locating each cell's QUADRATURE
        # points in the old mesh -- interior points, where the barycentric solve is far less accurate than
        # it is at a vertex. Measured in float32: the located weights carry 3.9e-04 against the reference
        # basis, and a mesh that does not move at all then drifts 1.5e-03 from the fixed-mesh march. With
        # x64 the same case matches to 2.6e-10. Refusing beats returning that quietly; the assembly is
        # float64 regardless (see `SemidiscreteTimeBlock.solve`).
        raise NotImplementedError(
            "jno.fem: a geometry term (`coord.d(t) - velocity`) needs jax_enable_x64. The state transfer "
            "locates quadrature points in the previous mesh, and in float32 that carries ~4e-4, enough for "
            "a stationary mesh to drift 1.5e-3 over a march. Enable x64:\n"
            "    jax.config.update('jax_enable_x64', True)"
        )

    d = fem.domain
    dim = int(d.dimension)
    if dim not in (2, 3):
        raise NotImplementedError(f"mesh motion supports 2D/3D simplicial meshes; got dimension {dim}.")

    cons, kw = fem._constraints, fem._fem_kwargs
    block = fem._op
    n_verts = int(np.asarray(d.mesh.points).shape[0])
    state = jnp.asarray(block.state0).reshape(-1)
    if jnp.iscomplexobj(state):
        raise NotImplementedError("mesh motion is real-only (the state transfer / mesh motion are); complex is future.")
    # The layout the transfer needs is read from the REBUILT problem below, since that is the one the
    # march actually steps; this early `off` only sizes the complex check above.

    # Make the assembly a FUNCTION of the vertex positions, once, instead of rebuilding it every step.
    # `Variable.trainable()` on a spatial coordinate registers it as a runtime parameter that
    # `_apply_coord_params` scatters into the P1 geometry before the element Jacobian is formed; the
    # transient block then re-forms A and M from `args` inside `step` (see `SemidiscreteTimeBlock`). One
    # spec per axis, covering EVERY vertex, since any of them may move.
    #
    # ONE route, for prescribed and state-reading velocities alike. An earlier version split them, on the
    # belief that a state-reading law needed the rebuild to refresh a host array behind the frozen-field
    # readout. That was wrong twice over: `move_mesh` does refresh `domain.mesh_connectivity` (measured
    # 1.0 -> 1.5 across a move), and the actual cause of the NaN was the BiCGStab breakdown that
    # `_step_solve` now avoids. With that fixed, forcing the state-reading case down this route reproduces
    # the rebuild route exactly (final ymax 0.9902 either way), so the split bought nothing and is gone.
    _axis_names = [f"__meshmotion_x{a}__" for a in range(dim)]
    # A coordinate the USER tagged `.trainable()` is where the mesh STARTS; the driver's own registration
    # is where the mesh IS at each step. Different roles, so they compose -- this used to raise, on the
    # reading that "two specs writing the same axis would silently let one win". They cannot: the user's
    # tag seeds the scan's initial carry and is never put in `args`, while the driver's axis params carry
    # the evolving positions and are supplied every step from that carry. Together they complete the
    # coordinate table -- free AND determined = the march moves it, from a design-variable start.
    #
    # The driver's own registration from an earlier `solve()` on the same domain is neither: it is dropped
    # and re-made below, so a second solve behaves like the first (without that distinction, solving twice
    # raised).
    _init_coords = [sp for sp in (getattr(d, "_trainable_coords", None) or []) if sp["name"] not in _axis_names]
    _init_specs = [
        {"ids": np.asarray(sp["ids"], dtype=int), "axis": int(sp["axis"]), "name": str(sp["name"])} for sp in _init_coords
    ]
    d._trainable_coords = []  # re-registered from the CURRENT mesh just below
    _all_parts = d.variable("__meshmotion_all__", where=lambda *c: np.ones_like(np.asarray(c[0]), dtype=bool), split=True)
    for _a, _nm in enumerate(_axis_names):
        _all_parts[_a].trainable(name=_nm)
    _coord_ids = np.asarray(d._trainable_coords[0]["ids"], dtype=int)
    cur = jno.fem(cons, **kw)  # rebuilt ONCE, now parametric in the vertex positions
    block = cur._op
    state = jnp.asarray(block.state0).reshape(-1)

    # Per-field element order / value shape / P{k} connectivity, for the state transfer. Read HERE,
    # immediately after the rebuild, because the next assembly on this domain overwrites the
    # `_fem_native_*` records it comes from. It also raises for a non-nodal family (RT / Nedelec /
    # Hermite / Argyris / Morley), which is the fail-loud this driver wants: the transfer tabulates a
    # nodal Lagrange basis, and there is nothing sensible it could do with an edge or a normal-moment DOF.
    # The layout is CONSTANT over the march -- the move preserves topology, so every field's connectivity
    # and DOF count are the same at every frame; only the vertex positions differ.
    try:
        layout = _field_layout(cur)
    except NotImplementedError as _e:
        raise NotImplementedError(f"jno.fem: a geometry term needs a nodal-Lagrange assembly. {_e}") from _e
    off = [int(x) for x in layout["offsets"]]

    # Runtime parameter VALUES (a coefficient, a neural field) travel with the coordinates. Without this
    # they were accepted by `**kwargs` and silently discarded: the block exposes them in
    # `runtime_parameter_exprs` and the assembly reads them from `args`, but `args` only ever held the
    # mesh-motion axes -- so a moving-mesh solve used each parameter's SEED value and reported no error,
    # and any gradient with respect to one was identically zero. An unknown name raises rather than being
    # swallowed, matching what the ordinary (non-motion) transient path does.
    # A parameter can live in the WEAK FORM (reaches the assembly through `args`) or in a GEOMETRY TERM
    # (reaches the velocity through its compiled expression's `models`), and the two take different
    # routes. Both are validated together and split below, once the velocity functions exist; the names
    # are disjoint sets, so a parameter used in both places is delivered to both.
    _step_params = set(getattr(block, "runtime_parameter_exprs", None) or {}) - set(_axis_names)
    _user_args: dict = {}
    _law_args: dict = {}

    def _coord_args(pts_now):
        """The moved vertices, in the layout `_apply_coord_params` scatters from (one per axis), plus any
        weak-form parameter values the caller supplied."""
        out = {nm: jnp.asarray(pts_now[_coord_ids, ax]) for ax, nm in enumerate(_axis_names)}
        out.update(_user_args)
        return out

    def _step_solve(step_op, rhs, u0, diag_fn):
        """The θ-step solve for this march, GMRES rather than the block's default BiCGStab.

        BiCGStab **breaks down** on the parametric branch's step operator and returns NaN. Measured on a
        29-dof heat problem whose step matrix has condition number 2.5: a direct solve gives a clean
        answer, CG with the very same Jacobi preconditioner converges to 2.4e-10, BiCGStab *without* the
        preconditioner converges to 1.4e-10 -- and BiCGStab *with* it returns NaN. The parametric branch
        imposes Dirichlet by ROW REPLACEMENT, which leaves the operator non-symmetric, and JAX's BiCGStab
        carries no breakdown handling. GMRES does not break down there (the same reason the complex
        real-equivalent block already asks for it).

        This is a **general** defect of that branch, not of mesh motion; it is repaired here rather than in
        `SemidiscreteTimeBlock.step` because changing the default Krylov method for every parametric
        transient is a wider blast radius than this change should carry.

        **Tolerance and budget -- this is what the march's wall time was.** Two defects compounded:

        1. ``tol=1e-10`` is below float32 eps (1.2e-7), the default working precision, so the termination
           test could never fire and every solve ran to its cap.
        2. ``maxiter=None`` means ``10*n`` OUTER iterations, each of ``restart`` inner ones -- 10140 outer
           at 1014 dofs. That is a scipy convention for un-restarted GMRES and is wildly wrong here.

        Jacobi-preconditioned GMRES on this row-replaced (non-symmetric) operator stagnates rather than
        converging quickly, so the cap, not the tolerance, sets the price. Measured at 1014 dofs over 5
        steps, all giving an IDENTICAL answer (final ymax 1.02525 against the forward-Euler reference
        1.02525, field finite, max 1.0000):

        ===========================  ==============
        ``tol=1e-10``, no cap          159 s/step
        ``tol=1e-6``, no cap          30.8 s/step
        ``tol``=eps-scaled, cap 20    **0.80 s/step**
        ===========================  ==============

        i.e. ~200x, for a solve that was already converged after two outer iterations -- the previous
        state is an excellent initial guess and the step correction is small.

        **Limitation:** the budget is fixed, so a stiffer step (much larger ``dt*kappa``, or a mesh fine
        enough that Jacobi is too weak) can exhaust it and return an under-solved step *silently* -- a
        residual check here would be traced away, since this runs inside the jitted step. The march's
        accuracy is covered instead by the first-order convergence test against the analytic domain.
        """
        import jax

        dd = diag_fn()
        inv = 1.0 / jnp.where(jnp.abs(dd) > 1e-30, dd, 1.0)
        n = int(jnp.asarray(rhs).shape[0])
        eps = float(jnp.finfo(jnp.asarray(rhs).dtype).eps)
        out, _ = jax.scipy.sparse.linalg.gmres(
            step_op,
            rhs,
            x0=u0,
            tol=max(1e-10, 100.0 * eps),  # a tolerance the working precision can actually reach
            atol=0.0,
            restart=min(n, 40),
            maxiter=20,  # bounded: this operator stagnates, so the cap is the price
            M=lambda x: inv * x,
        )
        return out

    key = _simplex_cell_key(dim)
    ts = np.asarray(_block_time_grid(block))  # fixed t0..t1 grid; moving the mesh never changes the grid
    dt = float(block.dt)
    theta = float(block.metadata.get("theta", 1.0)) if block.metadata else 1.0
    n_steps = len(ts) - 1

    # The step is compiled ONCE and reused, so the march re-traces nothing per step. `dt`, `theta` and the
    # linear solve are closed over as constants; only the state, the time and the moved vertices are
    # traced. Worth **2.9x** end-to-end (21.1 s -> 7.3 s, 1014 vertices, 20 steps, identical answer).
    #
    # That win is only visible once `_step_solve` has a reachable tolerance and a bounded budget: while
    # every step was burning its whole GMRES budget, the same jit measured 1.0x and looked worthless. A
    # microbenchmark here once claimed 574x for it, which was wrong in the other direction -- it timed the
    # jitted step without `block_until_ready`, measuring how long it took to QUEUE the step, not to run it.
    _jstep = jax.jit(lambda u_, t_, a_: block.step(u_, t_, dt, args=a_, theta=theta, linear_solve=_step_solve))

    # Connectivity is PRESERVED by construction here (move_mesh never retriangulates), so every frame can
    # share one cell array instead of copying it. A frame-per-copy costs n_steps x n_cells x (dim+1) x 8 B
    # of identical data -- ~240 MB for 100k cells over 100 steps, for nothing. Points genuinely differ per
    # frame and are still copied.
    shared_cells = np.asarray(d.mesh.cells_dict[key]).astype(np.int64)
    _pts0 = np.asarray(d.mesh.points)[:, :dim].astype(np.float64)

    # ── everything below here is hoisted: it depends on CONNECTIVITY and tag membership, never on where
    # the vertices currently are, so it is resolved once and rides the scan as a constant.
    #
    # Region membership is FROZEN at t0. That is required for static shapes, and it is also the fix for a
    # defect the per-step re-resolution carried: a `where=` region re-resolved on the MOVED points leaves
    # its own predicate as soon as it succeeds in moving, so its vertices silently drop into the harmonic
    # extension and the harder it is driven the less it moves. The driven set is a MATERIAL set -- the
    # Lagrangian convention every ALE code uses. Freezing alone used not to be enough (the velocity was
    # evaluated against the tag's re-sampled points, a different point set from the frozen vertices, so
    # the alignment then failed); that is resolved because the velocity now reads the positions it is
    # handed rather than the tag's context. The two had to land together, and do.
    specs = _geometry_motion_specs(cur, d)
    vel_fns = [_geometry_velocity_fn(sp, d) for sp in specs]

    _law_params = set().union(*(getattr(vf, "law_params", frozenset()) for vf in vel_fns)) if vel_fns else set()
    _init_params = {sp["name"] for sp in _init_specs}
    _accepted = _step_params | _law_params | _init_params
    _unknown = set(kwargs) - _accepted
    if _unknown:
        raise TypeError(
            f"jno.fem: fem.solve() got unexpected keyword argument(s) {sorted(_unknown)!r} for a moving-mesh "
            f"problem. Runtime parameters on this problem: {sorted(_accepted)!r}."
        )
    _user_args.update({k: jnp.asarray(v) for k, v in kwargs.items() if k in _step_params})
    _law_args.update({k: jnp.asarray(v) for k, v in kwargs.items() if k in _law_params})

    # The starting geometry. A `.trainable()` coordinate the caller supplied a value for is scattered into
    # the seed positions, so `d(anything)/d(X0)` flows through the whole march; unsupplied, it is simply
    # the mesh as meshed. This is the ONLY place a user coordinate tag enters -- deliberately not `args`,
    # where it would race the driver's per-step axis parameters.
    X0 = jnp.asarray(_pts0)
    for sp in _init_specs:
        if sp["name"] in kwargs:
            X0 = X0.at[jnp.asarray(sp["ids"]), sp["axis"]].set(jnp.asarray(kwargs[sp["name"]]).reshape(-1))

    disp_rows, disp_cols, named_np = [], [], np.zeros(n_verts, dtype=bool)
    written = np.zeros((n_verts, dim), dtype=bool)
    for sp in specs:
        # Two terms naming the same vertex AND the same axis state two velocities for one degree of
        # freedom. Regions may overlap (a corner belongs to both edges), so this is easy to write by
        # accident -- and scattering in list order would just let the last one win. Index-only, so it is
        # decided here rather than per step.
        clash = written[sp["ids"], sp["axis"]]
        if clash.any():
            raise ValueError(
                f"jno.fem: two geometry terms both prescribe axis {sp['axis']} of "
                f"{int(clash.sum())} vertex/vertices (region {sp['coord'].tag!r} overlaps an earlier "
                "term). One coordinate can have only one velocity — narrow the regions so they do not "
                "overlap on that axis."
            )
        written[sp["ids"], sp["axis"]] = True
        disp_rows.append(jnp.asarray(np.asarray(sp["ids"], dtype=np.int64)))
        disp_cols.append(int(sp["axis"]))
        # A vertex named on ANY axis is Dirichlet data for the extension, with its untagged columns held
        # at zero -- per-axis tagging is literal, so an untagged column does not drift.
        named_np[sp["ids"]] = True
    named_j = jnp.asarray(named_np)

    cells_j = jnp.asarray(shared_cells)
    # Only an INTERIOR cell whose quadrature leaves the old mesh is a fault: a cell touching the boundary
    # genuinely leaves it when that boundary moves outward, and the transfer clamps those to the nearest
    # simplex as the pointwise route did. Boundary-ness is connectivity, so it is hoisted.
    _bfac = _mesh_boundary_facets(d)[1]
    _on_bnd = np.zeros(n_verts, dtype=bool)
    _on_bnd[np.unique(np.asarray(_bfac).reshape(-1))] = True
    interior_cell_j = jnp.asarray(~_on_bnd[shared_cells].any(axis=1))
    sgn0 = jnp.sign(_signed_simplex_measures_jax(X0, cells_j, dim))  # orientation baseline is the START, not the seed mesh

    def _march_step(carry, t0c):
        u_c, X_c, bad = carry
        # 1) MOVE: each geometry term drives its own vertices along its own axis; everything the terms do
        #    not name relaxes harmonically around them (so a moving boundary drags the interior smoothly,
        #    and a moving interior region lets the mesh around it accommodate).
        disp = jnp.zeros((n_verts, dim), dtype=X_c.dtype)
        for rows, col, vf in zip(disp_rows, disp_cols, vel_fns):
            # `t0c`, not `t0c + dt`: the velocity is EXPLICIT, read at the start of the step. That is the
            # documented scheme and what `test_prescribed_motion_converges_first_order_to_the_analytic_domain`
            # pins -- it asserts the march reproduces forward Euler exactly.
            disp = disp.at[rows, col].set(jnp.asarray(vf(X_c, u_c, _law_args, t0c), dtype=X_c.dtype) * dt)
        X_n = X_c + _harmonic_extension_jax(X_c, shared_cells, dim, disp, named_j)

        # A tangled step cannot `raise` from inside a trace, so it is carried out and raised after the
        # march -- the failure stays loud, the trace stays valid.
        tangled = jnp.any(jnp.sign(_signed_simplex_measures_jax(X_n, cells_j, dim)) != sgn0)

        # 2) carry the state onto the moved mesh -- a CONSERVATIVE L2 projection, which transports the
        #    field under the motion; this IS the ALE convective term, treated semi-Lagrangian. The
        #    pointwise re-interpolation this replaces lost 27.6 % of a marginally-resolved peak over 2
        #    steps and 33.0 % over 16, i.e. it got worse as `dt` shrank. See `_l2_transfer_jax`.
        u_t, q_esc = _l2_transfer_jax(
            X_c,
            X_n,
            shared_cells,
            dim,
            u_c,
            off,
            orders=layout["orders"],
            vecs=layout["vecs"],
            cells_f=layout["cells_f"],
        )
        # An escaping quadrature point of an INTERIOR cell means the projection integrated against a
        # clamped extension rather than the field: the same fault the vertex route reports, at the same
        # place. A cell touching the boundary genuinely leaves the old mesh when that boundary moves
        # outward, so it is not a fault there.
        esc_cell = jnp.any(q_esc, axis=1) & interior_cell_j

        # 3) step on the moved mesh. The operator and the mass are re-formed from the moved vertices HERE,
        #    inside the step, through `args` -- `_apply_coord_params` scatters them into the P1 geometry
        #    before the element Jacobian, so J, detJ, JxW, physical gradients and the facet normals follow.
        u_n = block.step(u_t, t0c.astype(u_c.dtype), dt, args=_coord_args(X_n), theta=theta, linear_solve=_step_solve)
        bad = (bad[0] | tangled, bad[1] | jnp.any(esc_cell))
        return (u_n, X_n, bad), (u_n, X_n)

    (_u_f, _X_f, (tangled_any, escaped_any)), (u_hist, X_hist) = jax.lax.scan(
        _march_step, (state, X0, (jnp.array(False), jnp.array(False))), jnp.asarray(ts[:-1])
    )

    if bool(tangled_any):
        raise ValueError(
            "jno.fem: the mesh motion inverts or collapses an element (the mesh would tangle). Take a "
            "smaller time step, or drive the motion through a region whose harmonic extension can "
            "accommodate it."
        )
    if bool(escaped_any):
        raise ValueError(
            "jno.fem: a step moved an interior vertex further than its own element, so the state transfer "
            "could not locate it and would have silently clamped it to the nearest simplex. Take a smaller "
            "time step, or refine the mesh where the motion is fastest."
        )

    times = [float(t) for t in ts]
    states = [state] + [u_hist[i] for i in range(n_steps)]
    meshes = [(X0, shared_cells)] + [(X_hist[i], shared_cells) for i in range(n_steps)]
    history = [{"t": float(ts[i + 1]), "n_dofs": int(n_verts)} for i in range(n_steps)]

    # Leave the domain on the final moved mesh, as the eager driver did -- callers inspect `fem.points`
    # after a solve. Host state, so it can only take a concrete value: inside `jax.grad` the final
    # positions are a tracer and the domain simply stays where it was, which is right (a differentiated
    # solve should not mutate the caller's mesh as a side effect).
    try:
        _final_pts = np.asarray(_X_f)
    except Exception:  # noqa: BLE001 -- a tracer: differentiating through the march, nothing to write back
        _final_pts = None
    if _final_pts is not None:
        move_mesh(d, _final_pts - _pts0, copy=False, check=False)
        # Re-tag and re-sample so the domain is left SELF-CONSISTENT: `move_mesh` moves
        # `domain.mesh.points` without touching the cached tag pools / contexts, and a second `solve()`
        # on the same domain resolves its sample<->vertex alignment against both. Stale pools made that
        # second solve fail the alignment check by exactly the distance the mesh had moved. The eager
        # driver paid this every step to keep the velocity current; the march no longer needs it (the
        # velocity reads the positions it is handed), so it is paid ONCE, at the end.
        _preds = getattr(d, "_tag_predicates", None) or {}
        for _name, _pred in _preds.items() if hasattr(_preds, "items") else []:
            d.tag(_name, _pred)
        for _tag in {sp["coord"].tag for sp in specs}:
            try:
                d.variable(_tag, normals=True, split=True)  # keep the normals a Stefan-type law reads
            except Exception:  # noqa: BLE001 -- an interior region has no normals; its coordinates suffice
                d.variable(_tag, split=True)
    fem.__dict__.update(cur.__dict__)
    fem.adapt_history = history
    # One layout, repeated: the move preserves topology, so every frame has the same per-field orders,
    # value shapes and connectivity. Carrying it is what lets `AdaptiveTrajectory.resample` take its
    # basis-aware branch -- without it a P2 or vector trajectory falls to the legacy scalar-P1 path, which
    # would mis-slice the state blocks rather than fail.
    return AdaptiveTrajectory(np.asarray(times), states, meshes, layouts=[layout] * len(times))


def run_adaptive_inverse(
    domain: Any,
    build_inverse: Any,
    spec: AdaptSpec,
    *,
    n_opt: int,
    readout: Any = None,
) -> list[dict]:
    """Adaptive mesh refinement wrapped around a differentiable inverse solve.

    Alternates, on ``domain`` (refined **in place** each round)::

        optimize parameters  ->  recovered state  ->  ZZ-estimate  ->  Dörfler-mark  ->  refine

    so the final mesh is the minimal one that resolves the *recovered design* -- not merely
    the forward solution at some fixed guess.  Refinement is a non-differentiable outer
    Python loop; each inner ``crux.solve`` is a fully differentiable inverse solve on the
    (currently frozen) mesh, so gradients reach the parameters unchanged.

    Parameters
    ----------
    domain
        The mesh to adapt.  Mutated in place (``domain.refine``) between rounds.
    build_inverse
        ``build_inverse(domain) -> (crux, state_op)``.  Rebuilds the inverse problem on the
        current mesh: ``crux`` is a :func:`jno.core` whose ``.solve(n)`` optimizes the shared
        ``jno.np`` parameters, and ``state_op`` is the FEM solve op whose value at the
        optimized parameters is the recovered nodal state (this drives the ZZ estimator).
        The parameters must be **shared** ``jno.np`` objects created once by the caller and
        closed over here, so their optimized value carries (warm-starts) into the next round.
        Scalar / low-dimensional parameters only: a *field* parameter tied to mesh vertices
        changes shape on remesh and would need solution transfer (not supported here).
    spec
        :class:`AdaptSpec` controlling marking (``theta``), the round budget (``max_iters``),
        the local refinement (``refine_factor``), and optional ``tol`` / ``max_dofs`` stops.
    n_opt
        Optimizer steps (``crux.solve(n_opt)``) per round.
    readout
        Optional ``readout(crux) -> Any`` called with the just-solved ``crux`` to snapshot
        the recovered parameter value(s) (e.g. ``crux.eval([kappa])``); stored under
        ``"params"`` in the returned history.  Optimized values live in the ``crux``
        instance (they are *not* written back to the ``jno.np`` object), so read them here.
        To warm-start the next round, reseed inside ``build_inverse`` via
        ``param.initialize(...)`` from the value snapshotted here.

    The loop stops on whichever of ``spec``'s criteria fires first, including ``eps`` --
    convergence of the recovered parameter: once ``readout`` returns numeric value(s) whose
    relative change ``||Δθ|| / ||θ||`` stays below ``eps`` for two consecutive rounds, the
    design is mesh-converged and the loop ends. Setting ``eps`` requires ``readout`` to
    return the parameter value(s) (array-like), not just a label.

    Returns
    -------
    list of dict
        One record per round: ``{n_dofs, estimate, n_marked[, params]}``.
    """
    if spec.eps is not None and readout is None:
        raise ValueError("AdaptSpec.eps needs `readout` to return the parameter value(s) to measure convergence.")
    history: list[dict] = []
    prev_param = None
    n_converged = 0
    for it in range(spec.max_iters):
        crux, state_op = build_inverse(domain)
        crux.solve(n_opt)  # optimize parameters on the current (frozen) mesh

        # recovered nodal state at the optimized parameters -> drives the estimator
        u = np.asarray(crux.eval([state_op])).reshape(-1)
        n_vert = int(np.asarray(domain.mesh.points).shape[0])
        if u.shape[0] != n_vert:
            raise NotImplementedError(
                "The ZZ estimator currently assumes a scalar P1 state (one DOF per vertex); "
                f"got {u.shape[0]} values for {n_vert} vertices."
            )

        eta, est = zz_error_indicators(domain, u)
        marked = dorfler_mark(eta, spec.theta)
        rec: dict = {"n_dofs": n_vert, "estimate": est, "n_marked": int(marked.size)}
        if readout is not None:
            rec["params"] = readout(crux)
        history.append(rec)

        # eps: the recovered parameter stopped moving (plateau) for _EPS_PATIENCE rounds
        if spec.eps is not None and prev_param is not None:
            n_converged = n_converged + 1 if _rel_change(rec["params"], prev_param) < spec.eps else 0
        if readout is not None:
            prev_param = rec["params"]

        last = it == spec.max_iters - 1
        below_tol = spec.tol is not None and est < spec.tol
        over_budget = spec.max_dofs is not None and n_vert >= spec.max_dofs
        converged = spec.eps is not None and n_converged >= _EPS_PATIENCE
        if last or below_tol or over_budget or converged or marked.size == 0:
            break

        size = size_field_from_marks(domain, marked, refine_factor=spec.refine_factor)
        remesh_with_mmg(domain, size, copy=False)  # mutate the domain in place
    return history
