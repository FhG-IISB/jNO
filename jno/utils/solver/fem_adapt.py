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
        A meshed 2D ``jno`` domain (``domain.mesh`` a ``meshio.Mesh`` of triangles).
    vertex_size
        ``(n_vertices,)`` isotropic target edge length at each *current* mesh vertex.
        Smaller values request local refinement.
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
    if dim != 2:
        raise NotImplementedError("remesh_with_mmg currently supports 2D (mmg2d) meshes; 3D (mmg3d) is future work.")

    pts = np.asarray(domain.mesh.points)[:, :2].astype(np.float64)
    tris = np.asarray(domain.mesh.cells_dict["triangle"]).astype(np.int32)
    size = np.asarray(vertex_size, dtype=np.float64).reshape(-1)
    if size.shape[0] != pts.shape[0]:
        raise ValueError(f"vertex_size has {size.shape[0]} entries but the mesh has {pts.shape[0]} vertices.")
    if not np.all(size > 0):
        raise ValueError("vertex_size must be strictly positive.")

    bedges = _boundary_edges_from_triangles(tris)
    corners = _corner_vertices(pts, bedges)

    m = mmgpy.MmgMesh2D()
    m.set_mesh_size(len(pts), len(tris), 0, len(bedges))
    m.set_vertices(pts)
    m.set_triangles(tris)
    m.set_edges(bedges, np.ones(len(bedges), dtype=np.int32))
    if len(corners):
        m.set_corners(corners)
        m.set_required_vertices(corners)
    m.set_field("metric", size.reshape(-1, 1))

    if hmin is None:
        hmin = float(size.min()) * 0.5
    if hmax is None:
        hmax = float(size.max()) * 2.0
    opts: dict[str, float] = {"hmin": hmin, "hmax": hmax, "hgrad": hgrad, "verbose": verbose}
    if hausd is not None:
        opts["hausd"] = hausd
    m.remesh(**opts)

    v_out = np.asarray(m.get_vertices())[:, :2]
    t_out, _ = m.get_triangles_with_refs()
    e_out, _ = m.get_edges_with_refs()
    return _domain_from_arrays(domain, v_out, np.asarray(t_out), np.asarray(e_out), copy=copy)


def _domain_from_arrays(template: Any, points2d: np.ndarray, tris: np.ndarray, bedges: np.ndarray, *, copy: bool):
    """Apply remeshed ``points/triangles/boundary-edges`` to a domain.

    Builds a ``meshio.Mesh`` carrying ``interior`` (triangles) and ``boundary``
    (line) cell-sets, then runs it through ``_apply_mesh`` on either a shallow copy
    of ``template`` (``copy=True``) or ``template`` itself (``copy=False``, in place).
    Named boundary sub-tags are *not* stored in the mesh -- they re-derive
    geometrically from ``template``'s spatial predicates.
    """
    import meshio

    n_pts = points2d.shape[0]
    pts3 = np.zeros((n_pts, 3), dtype=np.float64)
    pts3[:, :2] = points2d
    tris = np.asarray(tris, dtype=np.int64)
    bedges = np.asarray(bedges, dtype=np.int64)

    cells = [("triangle", tris), ("line", bedges)]
    n_tri, n_edge = len(tris), len(bedges)
    empty = np.asarray([], dtype=np.int64)
    cell_sets = {
        "interior": [np.arange(n_tri, dtype=np.int64), empty],
        "boundary": [empty, np.arange(n_edge, dtype=np.int64)],
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
    from ..._fem import _fe_element_gradient_data

    sg, jxw, cells = _fe_element_gradient_data(domain)
    sg = np.asarray(sg)  # (n_cells, n_quad, n_local, dim)
    cells = np.asarray(cells)  # (n_cells, n_local)
    n_cells, _, _, dim = sg.shape
    area = np.asarray(jxw).reshape(n_cells, -1).sum(axis=1)  # (n_cells,)

    u = np.asarray(u_vertex).reshape(-1)
    if u.shape[0] < int(cells.max()) + 1:
        raise ValueError("u_vertex is shorter than the mesh vertex count; expected one value per P1 vertex.")

    # raw (elementwise-constant) gradient of u_h
    g_cell = np.einsum("cqld,cl->cqd", sg, u[cells]).mean(axis=1)  # (n_cells, dim)

    # recover a nodal gradient g* by area-weighted averaging of incident cells
    n_vert = u.shape[0]
    g_star = np.zeros((n_vert, dim))
    w = np.zeros(n_vert)
    for lv in range(cells.shape[1]):
        idx = cells[:, lv]
        np.add.at(g_star, idx, area[:, None] * g_cell)
        np.add.at(w, idx, area)
    g_star /= np.maximum(w[:, None], 1e-300)

    # elementwise gap, integrated over the cell (centroid rule: P1 g_cell is exact
    # there and the centroid value of the P1-recovered field is the vertex mean).
    g_star_centroid = g_star[cells].mean(axis=1)  # (n_cells, dim)
    eta2 = area * np.sum((g_star_centroid - g_cell) ** 2, axis=1)  # (n_cells,)
    eta = np.sqrt(np.maximum(eta2, 0.0))
    return eta, float(np.sqrt(eta2.sum()))


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
    """

    theta: float = 0.5
    max_iters: int = 8
    refine_factor: float = 2.0
    tol: float | None = None
    max_dofs: int | None = None


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
    for it in range(spec.max_iters):
        u = _solve_vertex_values(cur, solve_fn, **kwargs)
        eta, est = zz_error_indicators(d, u)
        n_dofs = int(np.asarray(d.mesh.points).shape[0])
        marked = dorfler_mark(eta, spec.theta)
        history.append({"n_dofs": n_dofs, "estimate": est, "n_marked": int(marked.size)})

        last = it == spec.max_iters - 1
        below_tol = spec.tol is not None and est < spec.tol
        over_budget = spec.max_dofs is not None and n_dofs >= spec.max_dofs
        if last or below_tol or over_budget or marked.size == 0:
            break

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

    Returns
    -------
    list of dict
        One record per round: ``{n_dofs, estimate, n_marked[, params]}``.
    """
    history: list[dict] = []
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

        last = it == spec.max_iters - 1
        below_tol = spec.tol is not None and est < spec.tol
        over_budget = spec.max_dofs is not None and n_vert >= spec.max_dofs
        if last or below_tol or over_budget or marked.size == 0:
            break

        size = size_field_from_marks(domain, marked, refine_factor=spec.refine_factor)
        remesh_with_mmg(domain, size, copy=False)  # mutate the domain in place
    return history
