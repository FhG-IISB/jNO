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
    # Drop the OLD mesh's predicate-tag state (boundary regions / indices / normals / pools /
    # context) so a re-tag re-derives it cleanly on the new mesh; stale surface-tag state otherwise
    # corrupts re-assembled Neumann/Robin/absorbing terms (predicates in `_tag_predicates` are kept).
    if hasattr(target, "_reset_custom_tag_state"):
        target._reset_custom_tag_state()
    target._apply_mesh(new_mesh)
    return target


def _shallow_copy(domain: Any):
    return copy.copy(domain)


# ---------------------------------------------------------------------------
# Solution transfer -- piecewise-linear (barycentric) mesh-to-mesh interpolation
# ---------------------------------------------------------------------------
def _locate_barycentric(src_pts: np.ndarray, src_cells: np.ndarray, qpts: np.ndarray, *, tol: float, k: int):
    """Locate each query point in a simplicial mesh and return its barycentric interpolation stencil.

    Point location is a KD-tree candidate search over cell centroids (the containing cell is almost
    always among the nearest centroids; raise ``k`` for strongly anisotropic meshes) followed by an
    exact barycentric inside-test. Returns ``(idx, weights, inside)``: ``idx`` ``(Q, D+1)`` the chosen
    simplex's source-vertex indices, ``weights`` ``(Q, D+1)`` its barycentric coordinates, ``inside``
    ``(Q,)`` bool (``True`` = strictly contained; ``False`` = the point fell outside every candidate
    and was projected onto the nearest simplex by clamping + renormalising the barycentric weights,
    which keeps the result a bounded convex combination of that cell's vertices). Pure host/NumPy."""
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
    return src_cells[chosen].astype(np.int64), weights, inside


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
        ``(n_vertices,)`` nodal solution values at the mesh vertices.

    Returns
    -------
    (eta, global_estimate)
        ``eta`` is ``(n_cells,)`` non-negative indicators; ``global_estimate`` is
        ``sqrt(sum eta**2)``, an estimate of the global energy-norm error.

    Reference: Zienkiewicz & Zhu (1987), IJNME 24, 337-357.
    """
    g_star, g_cell, area, cells = _recover_nodal_gradient(domain, u_vertex)

    # elementwise gap, integrated over the cell (centroid rule: P1 g_cell is exact
    # there and the centroid value of the P1-recovered field is the vertex mean).
    # For a COMPLEX field ``u_vertex`` the gradient gap is complex and the energy-norm error uses
    # its modulus: ``|g* - grad u_h|^2``. Since ``|x|^2 == x^2`` for reals, this stays exact in the
    # real case -- so one indicator drives refinement for real and complex (Helmholtz) fields alike.
    g_star_centroid = g_star[cells].mean(axis=1)  # (n_cells, dim)
    eta2 = area * np.sum(np.abs(g_star_centroid - g_cell) ** 2, axis=1)  # (n_cells,), real
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
        **Transient multifield only**: for a coupled scalar-P1 system, the index of the field whose
        curvature drives the anisotropic metric (which feature the mesh tracks) — first appearance order
        in the ``jno.fem`` constraints. Default 0. (Refining on *all* fields at once — metric
        intersection — is a later refinement.)
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


def _solve_vertex_values(fem: Any, solve_fn: Any = None, **kwargs: Any) -> np.ndarray:
    """Solve ``fem`` and return the nodal solution at the mesh vertices as a plain
    ``(n_vertices,)`` array -- complex-valued if the form is complex.

    Scalar P1 returns all DOFs; a higher-order scalar field (e.g. P2/TET10) returns its **vertex**
    DOFs (the first ``n_vertices`` entries, which are nodal), which drive the ZZ estimator. Vector /
    multifield problems are rejected -- refine on a scalar readout instead."""
    from jno._fem import _infer_vec  # local import: fem_adapt is loaded lazily by the domain

    sol = np.asarray(fem.solve(solve_fn, **kwargs)).reshape(-1)
    n_vert = int(np.asarray(fem.domain.mesh.points).shape[0])
    vec = _infer_vec(fem._constraints) if getattr(fem, "_constraints", None) else 1
    if vec != 1:
        raise NotImplementedError(
            f"The ZZ estimator supports a scalar field; got a vector field (vec={vec}). "
            "Refine per component or on a scalar readout instead."
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
        u = _solve_vertex_values(cur, solve_fn, **kwargs)
        eta, est = zz_error_indicators(d, u)
        n_dofs = int(np.asarray(d.mesh.points).shape[0])
        marked = None if spec.anisotropic else dorfler_mark(eta, spec.theta)
        history.append({"n_dofs": n_dofs, "estimate": est, "n_marked": None if marked is None else int(marked.size)})

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

    def __len__(self):
        return len(self.times)

    def final(self):
        """``(state, (points, cells))`` at the last time — the solution on the final adapted mesh."""
        return self.states[-1], self.meshes[-1]

    def resample(self, domain: Any, *, fill: Any = "nearest", tol: float = 1e-9, k: int = 32):
        """Project every saved state onto ``domain``'s vertices (the same barycentric transfer as
        :func:`transfer_solution`). Returns ``(n_save, n_target_vertices)`` for a single field, or
        ``(n_save, n_fields, n_target_vertices)`` for a coupled scalar-P1 system (the field count is
        read from each frame's DOFs ÷ vertices). The per-frame data in ``states`` is loss-free; this
        adds one interpolation per frame on demand."""
        import jax.numpy as jnp

        dim = int(domain.dimension)
        tgt = np.asarray(domain.mesh.points)[:, :dim].astype(np.float64)
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


def run_adaptive_transient(
    fem: Any, spec: AdaptSpec, *, solve_fn: Any = None, save_ts: Any = None, **kwargs: Any
) -> "AdaptiveTrajectory":
    """Drive ``FEM.solve(adapt=spec)`` for a **transient** problem: march the semidiscrete block and,
    every ``spec.every`` steps, remesh from the current field and carry the state onto the new mesh
    (:func:`transfer_solution`), so the mesh **tracks a moving feature**. Returns an
    :class:`AdaptiveTrajectory` (each frame on its own adapted mesh).

    The between-remesh march is the ordinary differentiable ``θ``-stepper (:meth:`SemidiscreteTimeBlock.step`
    over a ``lax.scan``); the remesh is the non-differentiable outer Python loop, exactly like the steady
    driver. The time grid is the block's fixed ``t0..t1`` at ``dt`` — remeshing changes only the mesh, not
    the grid, so time never drifts across chunks.

    **Scope — fail-loud on the rest** (a mis-transferred transient solve is silently wrong): one or
    several coupled **scalar-P1** fields (``spec.metric_field`` picks which drives the metric), **real**,
    **non-periodic**, and the driver owns the march (no custom ``solve_fn``). A **vector / higher-order**
    field (P1-vector velocity, Taylor-Hood P2 — needs component-wise / P2-basis transfer), a complex
    problem, periodic ties, and a bring-your-own integrator each raise a clear error and are later
    extensions. The forward march is differentiable within each fixed-mesh chunk and through the
    transfers, but the *remesh decisions* are not (the AFEM-inverse pattern -- freeze the mesh sequence,
    differentiate the chunks -- applies)."""
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
    # Per-field block layout: one or several coupled **scalar P1** fields, each on the mesh vertices.
    off = [int(x) for x in (fem.offsets or [0, int(state.shape[0])])]
    n_fields = len(off) - 1
    for _f in range(n_fields):
        if off[_f + 1] - off[_f] != n_verts:
            raise NotImplementedError(
                f"transient adaptive remeshing supports scalar-P1 field(s) only for now: field {_f} has "
                f"{off[_f + 1] - off[_f]} DOFs vs {n_verts} mesh vertices (it is vector / higher-order — a P1 vector "
                "velocity or Taylor-Hood P2). A vector / P2 field needs component-wise / P2-basis transfer, the next "
                "extension. Express the coupled system as scalar-P1 fields, or await it."
            )
    mf = int(spec.metric_field)
    if not 0 <= mf < n_fields:
        raise ValueError(f"AdaptSpec.metric_field={spec.metric_field} is out of range for {n_fields} field(s).")

    key = "triangle" if dim == 2 else "tetra"
    ts = np.asarray(_block_time_grid(block))  # fixed t0..t1 at dt -- unchanged by remeshing
    dt = float(block.dt)
    theta = float(block.metadata.get("theta", 1.0)) if block.metadata else 1.0
    n_steps = len(ts) - 1
    every = max(1, int(spec.every))

    def _snapshot():
        return (np.asarray(d.mesh.points)[:, :dim].astype(np.float64), np.asarray(d.mesh.cells_dict[key]).astype(np.int64))

    cur_mesh = _snapshot()
    times, states, meshes = [float(ts[0])], [state], [cur_mesh]
    history: list[dict] = []
    cur = fem

    i = 0
    while i < n_steps:
        chunk = int(min(every, n_steps - i))
        blk = block  # capture the current-mesh block for the scan closure

        def _body(u, t, _blk=blk):
            un = _blk.step(u, t, dt, theta=theta)
            return un, un

        state, traj = jax.lax.scan(_body, state, jnp.asarray(ts[i : i + chunk], dtype=state.dtype))
        for j in range(chunk):
            i += 1
            times.append(float(ts[i]))
            states.append(traj[j])
            meshes.append(cur_mesh)
        if i >= n_steps:
            break

        # remesh from the metric-driving field, then carry every field's block onto the new mesh
        u_v = np.asarray(state[off[mf] : off[mf + 1]])  # the metric_field's scalar-P1 vertex values
        h_typ = _mean_edge_length(d)
        hmin = spec.hmin if spec.hmin is not None else h_typ / 50.0
        hmax = spec.hmax if spec.hmax is not None else h_typ * 2.0
        n_dofs = int(np.asarray(d.mesh.points).shape[0])
        target = n_dofs * spec.refine_factor
        if spec.max_dofs is not None:
            target = min(target, float(spec.max_dofs))
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
        cur = jno.fem(cons, **kw)  # re-assemble the same transient problem on the refined mesh
        block = cur._op
        cur_mesh = _snapshot()
        new_n = int(cur_mesh[0].shape[0])
        idx, w, _inside = _locate_barycentric(old_pts, old_cells, cur_mesh[0], tol=1e-9, k=32)  # carry each field
        wj, ij = jnp.asarray(w, dtype=state.real.dtype), jnp.asarray(idx)
        state = jnp.concatenate([jnp.einsum("qk,qk->q", wj, state[off[_f] : off[_f + 1]][ij]) for _f in range(n_fields)])
        off = [_f * new_n for _f in range(n_fields + 1)]  # all scalar-P1 -> uniform n_verts blocks on the new mesh
        history.append({"t": float(ts[i]), "n_dofs": new_n, "fields": n_fields})

    fem.__dict__.update(cur.__dict__)  # rebind to the final adapted mesh (matches the steady driver)
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
