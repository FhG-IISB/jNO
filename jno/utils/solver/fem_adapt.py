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
``zz_error_indicators``, ``dorfler_mark``, ``size_field_from_marks``) are reusable on
their own.
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
    target._apply_mesh(new_mesh)
    return target


def _shallow_copy(domain: Any):
    return copy.copy(domain)


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
    g_star_centroid = g_star[cells].mean(axis=1)  # (n_cells, dim)
    eta2 = area * np.sum((g_star_centroid - g_cell) ** 2, axis=1)  # (n_cells,)
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

    g_cell = np.einsum("cld,cl->cd", sg, f[cells])  # (n_cells, dim): constant P1 gradient
    n_vert = f.shape[0]
    g_star = np.zeros((n_vert, dim))
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
    """Solve ``fem`` and return the P1 nodal solution as a plain ``(n_vertices,)`` array."""
    sol = np.asarray(fem.solve(solve_fn, **kwargs)).reshape(-1)
    n_vert = int(np.asarray(fem.domain.mesh.points).shape[0])
    if sol.shape[0] != n_vert:
        raise NotImplementedError(
            "The ZZ estimator currently assumes a scalar P1 field (one DOF per vertex); "
            f"got {sol.shape[0]} DOFs for {n_vert} vertices."
        )
    return sol


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
        cur = jno.fem(cons, **kw)  # re-assemble the same problem on the refined mesh

    # rebind the caller's FEM to the final adapted state so fem.points / A / b match ``u``
    fem.__dict__.update(cur.__dict__)
    fem.adapt_history = history
    return u


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
