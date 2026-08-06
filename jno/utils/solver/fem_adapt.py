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
    import mmgpy  # lazy: optional dependency, only needed for adaptive refinement

    dim = int(domain.dimension)
    if dim not in (2, 3):
        raise NotImplementedError(f"remesh_with_mmg supports 2D/3D simplicial meshes; got dimension {dim}.")

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

    elem_type, facet_type = ("triangle", "line") if dim == 2 else ("tetra", "triangle")
    cells = [(elem_type, elems), (facet_type, bfacets)]
    n_e, n_f = len(elems), len(bfacets)
    empty = np.asarray([], dtype=np.int64)
    cell_sets = {
        "interior": [np.arange(n_e, dtype=np.int64), empty],
        "boundary": [empty, np.arange(n_f, dtype=np.int64)],
    }
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
    if dim not in (2, 3):
        raise NotImplementedError(f"transfer_solution supports 2D/3D simplicial meshes; got dimension {dim}.")
    if int(target_domain.dimension) != dim:
        raise ValueError(f"source ({dim}D) and target ({int(target_domain.dimension)}D) mesh dimensions differ.")
    if fill not in ("nearest", "error") and not isinstance(fill, (int, float)):
        raise ValueError(f"fill must be 'nearest', 'error', or a numeric constant; got {fill!r}.")

    key = "triangle" if dim == 2 else "tetra"
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

    cell = CellType.triangle if dim == 2 else CellType.tetrahedron
    tab = _lagrange_basix(cell, int(order)).tabulate(0, np.asarray(xi, dtype=np.float64))
    return np.asarray(tab[0, :, :, 0])  # (Q, n_dof): basis values (0th-derivative block, scalar value)


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
    cells = np.asarray(domain.mesh.cells_dict["triangle" if dim == 2 else "tetra"])
    v = pts[cells]  # (n_cells, dim+1, dim)
    edge = v[:, 1:, :] - v[:, :1, :]  # (n_cells, dim, dim): rows are (v_i - v_0), i=1..dim
    einv = np.linalg.inv(edge)  # column j = grad of barycentric lambda_{j+1}
    measure = np.abs(np.linalg.det(edge)) / (2.0 if dim == 2 else 6.0)  # simplex volume = |det|/d!

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
    tris = np.asarray(domain.mesh.cells_dict["triangle" if dim == 2 else "tetra"])
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
    tris = np.asarray(domain.mesh.cells_dict["triangle" if dim == 2 else "tetra"])
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
def _mesh_cells(domain: Any) -> tuple[np.ndarray, int]:
    dim = int(domain.dimension)
    return np.asarray(domain.mesh.cells_dict["triangle" if dim == 2 else "tetra"]).astype(np.int64), dim


def _signed_simplex_measures(points: np.ndarray, cells: np.ndarray, dim: int) -> np.ndarray:
    """Signed area (2D) / volume (3D) of every simplex; the *sign* flips iff a cell inverts (tangles)."""
    v = np.asarray(points)[cells]  # (n_cells, dim+1, dim)
    edge = v[:, 1:, :] - v[:, :1, :]  # (n_cells, dim, dim): rows v_i - v_0
    return np.linalg.det(edge) / (2.0 if dim == 2 else 6.0)


def _mesh_boundary_facets(domain: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(cells, boundary_facets, dim)`` -- the interior simplices, their topological boundary
    facets (edges in 2D / triangles in 3D), and the dimension."""
    cells, dim = _mesh_cells(domain)
    bfacets = _boundary_edges_from_triangles(cells) if dim == 2 else _boundary_faces_from_tets(cells)
    return cells, bfacets, dim


def _p1_stiffness(domain: Any):
    """Assemble the P1 (linear-simplex) stiffness / discrete Laplacian ``K_ij = ∫_Ω ∇φ_i·∇φ_j``.

    Returns a symmetric sparse ``(n_vert, n_vert)`` SciPy CSR matrix built from the constant per-element
    barycentric gradients (:func:`_p1_element_gradients`).  This is the operator whose harmonic solve
    (:func:`harmonic_extension`) propagates a boundary motion smoothly into the interior.
    """
    from scipy.sparse import coo_matrix

    grad, measure, cells = _p1_element_gradients(domain)  # (n_cells, nv, dim), (n_cells,), (n_cells, nv)
    nv = cells.shape[1]
    ke = np.einsum("c,cad,cbd->cab", measure, grad, grad)  # (n_cells, nv, nv): measure * (∇φ_a·∇φ_b)
    ii = np.broadcast_to(cells[:, :, None], (cells.shape[0], nv, nv)).reshape(-1)
    jj = np.broadcast_to(cells[:, None, :], (cells.shape[0], nv, nv)).reshape(-1)
    n = int(np.asarray(domain.mesh.points).shape[0])
    return coo_matrix((ke.reshape(-1), (ii, jj)), shape=(n, n)).tocsr()


def harmonic_extension(domain: Any, boundary_displacement: np.ndarray) -> np.ndarray:
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
        ``(n_vert, dim)`` array; **only its boundary-vertex rows are read** (as Dirichlet data).  Interior
        rows are ignored and overwritten by the harmonic solve -- build a full-length array and set the
        boundary rows to the desired boundary motion (e.g. an interface velocity times ``dt``).

    Returns
    -------
    ``(n_vert, dim)`` full displacement (given boundary rows, harmonically-extended interior); pass it
    straight to :func:`move_mesh`.  Host/NumPy + a SciPy sparse solve -- a structural mesh step, outside
    the differentiable trace.

    Reference: harmonic / Laplacian mesh motion, the simplest ALE mesh-update operator; e.g. Johnson &
    Tezduyar, *Mesh update strategies in parallel FE computations of flows with moving boundaries*,
    Comput. Methods Appl. Mech. Engrg. 119 (1994) 73-94 (§3).
    """
    from scipy.sparse.linalg import splu

    cells, dim = _mesh_cells(domain)
    pts = np.asarray(domain.mesh.points)[:, :dim]
    n = pts.shape[0]
    bd = np.asarray(boundary_displacement, dtype=np.float64)
    if bd.shape != (n, dim):
        raise ValueError(
            f"harmonic_extension: boundary_displacement must be (n_vert, dim) = ({n}, {dim}); got {bd.shape}. "
            "Build a full-length array and set its boundary-vertex rows to the desired motion."
        )

    bfacets = _boundary_edges_from_triangles(cells) if dim == 2 else _boundary_faces_from_tets(cells)
    bverts = np.unique(bfacets.reshape(-1))
    is_b = np.zeros(n, dtype=bool)
    is_b[bverts] = True
    iverts = np.where(~is_b)[0]

    disp = np.zeros((n, dim), dtype=np.float64)
    disp[bverts] = bd[bverts]
    if iverts.size:
        k = _p1_stiffness(domain)
        kii = k[iverts][:, iverts].tocsc()
        kib = k[iverts][:, bverts]
        rhs = -(kib @ bd[bverts])  # (n_i, dim): move the boundary term to the RHS
        disp[iverts] = splu(kii).solve(np.asarray(rhs))  # one factorization, all `dim` columns at once
    return disp


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
    mesh vertices tagged with :meth:`Variable.trainable` down the FE-energy gradient — *through the
    differentiable solve* — at fixed connectivity and no new DOFs. Requires at least one coordinate tagged
    ``domain.variable(region)[i].trainable()`` before ``jno.fem`` (else it raises). See
    :func:`run_adaptive_relocate`. ``max_iters`` sets the number of relocation steps; ``lr`` the step size."""
    quality_floor: float = 0.1
    """Relocation only: a step is backtracked (halved) until no element's ``|det J|`` falls below this
    fraction of the initial worst element — the mesh-validity line search that keeps the relocation from
    tangling (a stock optimiser / a barrier alone cannot guarantee this on stiff problems)."""
    lr: float = 3e-3
    """Relocation only: base step size for the RMS-normalised energy-gradient descent."""


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

    Relocate the mesh vertices tagged with :meth:`Variable.trainable` down the FE Dirichlet-energy gradient
    -- evaluated *through the differentiable solve* (``∂(solve)/∂X``) -- at **fixed connectivity** and no new
    DOFs, with a **backtracking line search on ``det J``** so the mesh never tangles (a stock optimiser or a
    barrier alone cannot guarantee validity on stiff problems -- the constraint must live in the step control).
    Concentrates a *fixed* node set at solution features; the relocation companion of the h-refinement
    :func:`run_adaptive_solve`.

    ``spec.max_iters`` relocation steps, ``spec.lr`` step size, ``spec.quality_floor`` the validity bound.
    Requires ≥1 coordinate tagged ``domain.variable(region)[i].trainable()`` before ``jno.fem`` (else raises).
    Mutates ``fem`` / its domain to the relocated mesh (like the refinement loop) and returns the solution there;
    ``fem.adapt_history`` traces the per-step energy.
    """
    import jax
    import jax.numpy as jnp

    import jno

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

    def _block_energy(u, pts, bounds):
        """Total Dirichlet energy summed over EVERY solution block (``bounds``) — so a scalar, a vector (per
        component), a **complex** field (its real + imaginary blocks) and a coupled multifield all contribute.
        A higher-order block (P2) falls back to its vertex DOFs."""
        e = 0.0
        for i in range(len(bounds) - 1):
            blk = u[bounds[i] : bounds[i + 1]]
            nb = int(blk.shape[0])
            veci = nb // n_verts if (n_verts and nb % n_verts == 0) else 1
            bf = blk.reshape(n_verts, veci) if veci > 1 else blk[:n_verts]
            e = e + _dirichlet_energy_jax(pts, bf, cells_j, dim)
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

    def _block_defect(u, pts, bounds):
        """Equidistribution defect summed over every solution block — the mirror of :func:`_block_energy`,
        so a scalar / vector / complex / coupled multifield all contribute their own monitor."""
        d_ = 0.0
        for i in range(len(bounds) - 1):
            blk = u[bounds[i] : bounds[i + 1]]
            nb = int(blk.shape[0])
            veci = nb // n_verts if (n_verts and nb % n_verts == 0) else 1
            bf = blk.reshape(n_verts, veci) if veci > 1 else blk[:n_verts]
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
    msq = [jnp.zeros_like(a) for a in arrs]  # RMSProp running average (near-feature gradients dwarf the rest)
    history: list[dict] = []
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

    final = _moved(arrs)
    if _min_detj(final) <= 0.0:  # fail loud rather than hand back / re-solve on a tangled mesh
        raise RuntimeError(
            "FEM.solve(adapt=relocate): the mesh tangled (min det J <= 0). Lower AdaptSpec.lr or raise "
            "AdaptSpec.quality_floor."
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
        off = [0, int(jnp.asarray(fem._op.state0).reshape(-1).shape[0])]
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
    if dim not in (2, 3):
        raise NotImplementedError(f"transient adaptive remeshing supports 2D/3D simplicial meshes; got dimension {dim}.")

    cons, kw = fem._constraints, fem._fem_kwargs
    block = fem._op
    n_verts = int(np.asarray(d.mesh.points).shape[0])
    state = jnp.asarray(block.state0).reshape(-1)
    if jnp.iscomplexobj(state):
        raise NotImplementedError(
            "transient adaptive remeshing is real-only (the Hessian metric and the ZZ estimator are); a complex "
            "transient is not supported yet."
        )
    # Per-field block layout (offsets / orders / vecs / P{order} connectivity / DOF coords), generalised
    # beyond scalar-P1: vector, higher-order (P2), and mixed (Taylor-Hood) fields are carried across a
    # remesh by the basis-aware transfer below. Read right after assembly — the domain's per-field
    # metadata is clobbered by the next one.
    layout = _field_layout(fem)
    off = layout["offsets"]
    n_fields = len(off) - 1
    cur_nverts = n_verts  # current mesh's vertex count (updates each remesh; drives the metric slice)
    mf = int(spec.metric_field)
    if not 0 <= mf < n_fields:
        raise ValueError(f"AdaptSpec.metric_field={spec.metric_field} is out of range for {n_fields} field(s).")

    key = "triangle" if dim == 2 else "tetra"
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
    times, states, meshes, layouts = [float(ts[0])], [state], [cur_mesh], [layout]
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
            states.append(traj[j])
            meshes.append(cur_mesh)
            layouts.append(layout)
        if i >= n_steps:
            break

        # remesh from the metric-driving field (fixed budget above), then carry every field's block over
        u_v = _scalar_vertex_metric(state, layout, mf, cur_nverts)  # scalar VERTEX field (vector/P2 reduced)
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
        # Basis-aware, value-shape-aware state transfer: evaluate each OLD field at its NEW per-field DOF
        # coordinates using the OLD element's shape functions (P1 vertices / P2 midpoints / vector comps).
        vals = _eval_fe_fields_at_points(
            old_pts,
            old_cells,
            state,
            layout["offsets"],
            layout["orders"],
            layout["cells_f"],
            layout["vecs"],
            new_layout["field_points"],
            dim=dim,
        )
        state = jnp.concatenate([v.reshape(-1) for v in vals])
        off, layout, cur_nverts = new_layout["offsets"], new_layout, new_n
        history.append({"t": float(ts[i]), "n_dofs": int(off[-1]), "fields": n_fields})

    fem.__dict__.update(cur.__dict__)  # rebind to the final adapted mesh (matches the steady driver)
    fem.adapt_history = history
    return AdaptiveTrajectory(np.asarray(times), states, meshes, layouts=layouts)


@dataclass
class MovingBoundary:
    """Free-surface / moving-boundary spec for ``FEM.solve(move=...)`` on a **transient** problem.

    The domain boundary moves by a **prescribed velocity**; the mesh deforms to follow it (harmonic
    interior extension -- :func:`harmonic_extension` + :func:`move_mesh`), the physics marches on the
    moving mesh, and the state is carried across each move. Returns an :class:`AdaptiveTrajectory` (each
    frame on its own moved mesh -- use ``.resample(ref)`` to project onto a fixed grid).

    Attributes
    ----------
    velocity
        The boundary velocity, in one of two forms (arity-detected):

        - **prescribed** -- ``velocity(t, x) -> (n_boundary, dim)``: the velocity at time ``t`` evaluated
          at the current boundary-vertex positions ``x`` (``(n_boundary, dim)``; position-based, so no
          vertex indices needed);
        - **state-dependent** (physics-driven) -- ``velocity(t, x, state, domain) -> (n_boundary, dim)``:
          additionally receives the **current nodal field** ``state`` (the solution on the current mesh)
          and the **current** ``domain``. Use this to read a functional of the solution and drive the
          boundary with it -- e.g. a **Stefan** front ``v_n = -k/L · ∇T·n``. The boundary-functional
          readout (``u.bind(x=xb, y=yb).freeze(state)`` then ``(Tf.x*nx + Tf.y*ny).eval()``; see
          :class:`jno.trace.FrozenField`) expresses such a functional as traced math. Caveat: ``domain``
          here carries the **transient time grid**, so evaluate a purely-spatial readout on a steady view
          of the current mesh (or read ``∇field`` via nodal recovery); the standalone readout is
          validated on a steady domain, and a transient-domain spatial ``.eval()`` is a follow-up.

        In both callback forms, **return zero rows for the held (fixed) part of the boundary** and the
        interface velocity for the moving surface -- the returned field *is* the specification of which
        boundary moves.

        - **trace expression** (velocity *as math*) -- a jNO trace node giving the **scalar normal
          speed** at the boundary, referencing a frozen field for the current state::

              xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
              Tf = u.bind(x=xb, y=yb).freeze(state0)          # state0 = any placeholder (e.g. zeros/IC)
              velocity = -(k / L) * (Tf.x * nx + Tf.y * ny)   # v_n = -k/L · ∇T·n, a Stefan front

          The driver swaps the **live state** into ``Tf`` each step (via :func:`jno.trace.substitute` /
          :func:`jno.trace.refreeze`), evaluates the expression, and moves each boundary vertex **along
          its outward normal** at that speed. This is the fully-declarative form -- the interface law
          *is* the input, and it is differentiable in the field. (Interpreted as a *normal* speed; a
          callback returns the full velocity vector instead.)

          **Only part of the boundary moves** -- because the boundary coordinates ``xb, yb`` are trace
          nodes too, a coordinate-masked speed frees just the surface(s) you want and holds the rest
          exactly (the mask *follows* the moving surface, since the coordinates re-sample each step). No
          separate movable-tag is needed::

              v = (-(k / L) * (Tf.x * nx + Tf.y * ny)) * (yb > y_base)   # top free, base held

          (several parts: a compound mask; the held nodes get zero speed and are pinned by the harmonic
          interior extension.)
    every
        Move + re-assemble every ``every`` steps (default 1 = every step, most accurate). Larger is
        cheaper (fewer re-assemblies) but lags the domain shape within a chunk.

    Method / scope (house rule: fail loud on the rest).
    - **Operator-split ALE.** The physics marches on the current mesh; between steps the mesh is moved
      and the field **re-interpolated** onto the moved vertices (:func:`transfer_solution`), which
      transports the field under the mesh motion. This is first-order in ``every`` and is correct for a
      field whose material is (quasi-)stationary while the *boundary* moves (a melt/free surface with a
      quasi-static bulk); it does **not** add an ALE convective ``(c-w)·∇u`` term, so it is *not* the
      right discretization when a represented material velocity ``c`` differs from the mesh velocity
      ``w`` (coupled flow -- that would double-count advection). A **state-dependent** ``velocity`` (a
      Stefan / kinematic law reading the current field) is supported via the 4-argument form above; the
      velocity is still applied **explicitly** (evaluated from the state at the start of each move), not
      solved implicitly with the interface position.
    - **Connectivity-preserving move only.** A move that would invert an element raises
      (:func:`move_mesh` ``check``); the remesh-on-tangle fallback (``remesh_with_mmg`` + transfer, for
      large deformation) is the next extension. Reduce ``dt`` / the motion, or await it.
    - **Boundary conditions on the moving surface must be *natural* (unconstrained) or a whole-boundary
      / held-boundary tag** -- those re-derive correctly on the moved mesh. A Dirichlet/Robin BC pinned
      to the moving surface by a spatial sub-predicate would not follow the motion; an index-carried
      moving tag is the next extension.
    - **scalar-P1 field(s), real, non-periodic**, default θ-stepper (mirrors the transient adaptive
      driver): vector / higher-order / complex / periodic / a custom ``solve_fn`` each raise.
    """

    velocity: Any
    every: int = 1


def _velocity_wants_state(vel: Any) -> bool:
    """True if ``vel`` takes the state-dependent form ``velocity(t, x, state, domain)`` (>=4 positional
    parameters) rather than the prescribed ``velocity(t, x)``. Falls back to prescribed if unintrospectable."""
    import inspect

    try:
        params = [
            p for p in inspect.signature(vel).parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return len(params) >= 4
    except (ValueError, TypeError):
        return False


def _is_trace_velocity(vel: Any) -> bool:
    """True if ``vel`` is a **trace expression** (a Placeholder / typed view) rather than a Python
    callback — i.e. the user passed the velocity *as math* (`v = -(k/L)*(Tf.x*nx + Tf.y*ny)`)."""
    from ...trace import Placeholder

    return isinstance(vel, Placeholder) or isinstance(getattr(vel, "expr", None), Placeholder)


def _trace_velocity_vb(vexpr: Any, frozen_nodes: list, state: Any, dom: Any, bverts: np.ndarray, old_pts, dim: int):
    """Turn a **trace-expression** velocity (a *scalar normal speed*) into a per-boundary-vertex velocity
    VECTOR, moved along the outward normal. Swaps the live ``state`` into the static expression
    (:func:`jno.trace.substitute` / :func:`jno.trace.refreeze`), evaluates it on the current domain (in
    boundary-tag order), aligns to the driver's boundary vertices, and multiplies by the boundary normal.
    The interface thus moves along its normal at the speed the expression gives (the Stefan/kinematic case)."""
    from scipy.spatial import cKDTree

    from ...trace import refreeze, substitute

    v_now = substitute(vexpr, {f: refreeze(f, np.asarray(state)) for f in frozen_nodes})
    v_n = np.asarray(v_now.eval()).reshape(-1)  # scalar normal speed at the "boundary" tag points

    parts = dom.variable("boundary", normals=True, split=True)  # (x, y, [z], t, nx, ny, [nz])
    coords = np.column_stack([np.asarray(parts[i].eval()).reshape(-1) for i in range(dim)])
    normals = np.column_stack([np.asarray(parts[len(parts) - dim + i].eval()).reshape(-1) for i in range(dim)])
    if v_n.shape[0] != coords.shape[0]:
        raise ValueError(
            f"MovingBoundary trace velocity evaluated to {v_n.shape[0]} values but the boundary tag has "
            f"{coords.shape[0]} points — a trace velocity must be a SCALAR per boundary point (the normal speed)."
        )
    _, perm = cKDTree(coords).query(np.asarray(old_pts)[bverts])  # align tag order → driver's boundary vertices
    return v_n[perm][:, None] * normals[perm]  # v_n · n̂ : each vertex moves along its outward normal


def run_moving_boundary(
    fem: Any, spec: MovingBoundary, *, solve_fn: Any = None, save_ts: Any = None, **kwargs: Any
) -> "AdaptiveTrajectory":
    """Drive ``FEM.solve(move=spec)``: march a transient problem while the **boundary moves** by the
    prescribed ``spec.velocity``. Every ``spec.every`` steps the mesh is deformed to follow the boundary
    (:func:`harmonic_extension` + :func:`move_mesh`), the problem re-assembled on the moved mesh, and the
    state carried across (:func:`transfer_solution`, connectivity-preserving). Returns an
    :class:`AdaptiveTrajectory`. See :class:`MovingBoundary` for the method and its scope."""
    import jax
    import jax.numpy as jnp

    import jno

    from .backend_blocks import _block_time_grid

    if fem._constraints is None:
        raise ValueError("FEM.solve(move=...) requires a FEM built by jno.fem(...) (its constraint list is retained).")
    if solve_fn is not None:
        raise NotImplementedError(
            "fem.solve(move=..., solve_fn=...) is not supported: the moving-boundary driver owns the time march. "
            "Drop solve_fn (use the default θ-stepper), or drop move=."
        )
    if getattr(fem, "_periodic", None) is not None:
        raise NotImplementedError("fem.solve(move=...) with periodic ties is not supported yet.")
    _vel = getattr(spec, "velocity", None)
    if not (callable(_vel) or _is_trace_velocity(_vel)):
        raise TypeError(
            "MovingBoundary.velocity must be a callable (velocity(t, x[, state, domain]) -> (n_boundary, dim)) "
            "or a trace expression (a scalar normal speed, e.g. -(k/L)*(Tf.x*nx + Tf.y*ny))."
        )

    d = fem.domain
    dim = int(d.dimension)
    if dim not in (2, 3):
        raise NotImplementedError(f"moving-boundary supports 2D/3D simplicial meshes; got dimension {dim}.")

    cons, kw = fem._constraints, fem._fem_kwargs
    block = fem._op
    n_verts = int(np.asarray(d.mesh.points).shape[0])
    state = jnp.asarray(block.state0).reshape(-1)
    if jnp.iscomplexobj(state):
        raise NotImplementedError("moving-boundary is real-only (the state transfer / mesh motion are); complex is future.")
    off = [int(x) for x in (fem.offsets or [0, int(state.shape[0])])]
    n_fields = len(off) - 1
    for _f in range(n_fields):
        if off[_f + 1] - off[_f] != n_verts:
            raise NotImplementedError(
                f"moving-boundary supports scalar-P1 field(s) only for now: field {_f} has {off[_f + 1] - off[_f]} "
                f"DOFs vs {n_verts} mesh vertices (vector / higher-order). Express it as scalar-P1 fields, or await it."
            )

    key = "triangle" if dim == 2 else "tetra"
    ts = np.asarray(_block_time_grid(block))  # fixed t0..t1 grid; moving the mesh never changes the grid
    dt = float(block.dt)
    theta = float(block.metadata.get("theta", 1.0)) if block.metadata else 1.0
    n_steps = len(ts) - 1
    every = max(1, int(spec.every))
    vel = spec.velocity
    vel_is_trace = _is_trace_velocity(vel)
    wants_state = (not vel_is_trace) and _velocity_wants_state(vel)
    frozen_nodes = vexpr = None
    if vel_is_trace:
        from ...trace import Placeholder, frozen_fields_in

        vexpr = vel if isinstance(vel, Placeholder) else vel.expr  # unwrap a typed view to its Placeholder
        frozen_nodes = frozen_fields_in(vexpr)
        if not frozen_nodes:
            raise ValueError(
                "MovingBoundary.velocity given as a trace expression must reference a frozen field — build it "
                "as `Tf = u.bind(x=xb, y=yb).freeze(state0)` and write the speed in terms of `Tf` (e.g. "
                "`-(k/L)*(Tf.x*nx + Tf.y*ny)`); the driver swaps the live state into `Tf` each step."
            )

    def _snapshot():
        return (np.asarray(d.mesh.points)[:, :dim].astype(np.float64), np.asarray(d.mesh.cells_dict[key]).astype(np.int64))

    cur_mesh = _snapshot()
    times, states, meshes = [float(ts[0])], [state], [cur_mesh]
    history: list[dict] = []
    cur = fem

    i = 0
    while i < n_steps:
        chunk = int(min(every, n_steps - i))
        t0c, t1c = float(ts[i]), float(ts[i + chunk])
        old_pts, old_cells = cur_mesh

        # 1) MOVE the boundary from shape(t0c) toward shape(t1c) by the prescribed velocity, hold the rest.
        bfacets = _boundary_edges_from_triangles(old_cells) if dim == 2 else _boundary_faces_from_tets(old_cells)
        bverts = np.unique(bfacets.reshape(-1))
        # Velocity in one of three forms: a trace expression (a scalar normal speed the driver moves along
        # the normal), a state-dependent callback velocity(t, x, state, domain), or a prescribed
        # velocity(t, x). The state-reading forms see the CURRENT field on the CURRENT mesh (before this
        # move), e.g. a Stefan v_n = -k/L·∇T·n.
        if vel_is_trace:
            vb = np.asarray(_trace_velocity_vb(vexpr, frozen_nodes, state, d, bverts, old_pts, dim), dtype=np.float64)
        elif wants_state:
            vb = np.asarray(vel(t0c, old_pts[bverts], np.asarray(state), d), dtype=np.float64)
        else:
            vb = np.asarray(vel(t0c, old_pts[bverts]), dtype=np.float64)
        if vb.shape != (bverts.shape[0], dim):
            raise ValueError(
                f"MovingBoundary.velocity must return (n_boundary, dim) = ({bverts.shape[0]}, {dim}); got {vb.shape}."
            )
        bdisp = np.zeros((old_pts.shape[0], dim), dtype=np.float64)
        bdisp[bverts] = vb * (t1c - t0c)
        move_mesh(d, harmonic_extension(d, bdisp), copy=False)  # fail loud on tangle (house rule 1)

        # 2) re-tag (held boundaries re-derive on the moved facets) + re-assemble on the moved mesh
        for _name, _pred in list(getattr(d, "_tag_predicates", {}).items()):
            d.tag(_name, _pred)
        cur = jno.fem(cons, **kw)
        block = cur._op
        new_mesh = _snapshot()
        new_pts, _new_cells = new_mesh

        # 3) carry the state onto the moved vertices (re-interpolate -> transports the field under the motion)
        idx, w, _inside = _locate_barycentric(old_pts, old_cells, new_pts, tol=1e-9, k=32)
        wj, ij = jnp.asarray(w, dtype=state.real.dtype), jnp.asarray(idx)
        state = jnp.concatenate([jnp.einsum("qk,qk->q", wj, state[off[_f] : off[_f + 1]][ij]) for _f in range(n_fields)])

        # 4) march the chunk on the moved mesh (the ordinary differentiable θ-stepper)
        blk = block

        def _body(u, t, _blk=blk):
            un = _blk.step(u, t, dt, theta=theta)
            return un, un

        state, traj = jax.lax.scan(_body, state, jnp.asarray(ts[i : i + chunk], dtype=state.dtype))
        cur_mesh = new_mesh
        for j in range(chunk):
            i += 1
            times.append(float(ts[i]))
            states.append(traj[j])
            meshes.append(cur_mesh)
        history.append({"t": float(ts[i]), "n_dofs": int(new_pts.shape[0])})

    fem.__dict__.update(cur.__dict__)  # rebind to the final moved mesh
    fem.adapt_history = history
    return AdaptiveTrajectory(np.asarray(times), states, meshes)


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
