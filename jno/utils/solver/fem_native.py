"""Native Lagrange assembler for ``jno.fem``.

Implements the full assembly pipeline for scalar/vector Lagrange P1/P2 fields on 2D
triangle and 3D tetrahedral meshes (single- and multi-field, linear/nonlinear/transient),
mirroring the contract of :func:`fem_1d.assemble_fem_1d` and
:func:`fem_nonnodal.assemble_fem_nonnodal`. The assembler is dimension-generic: the cell
Jacobian, element factory and facet machinery all key off ``dim``.

Key components re-used without change:

* :func:`fem_utils._eval_integrand` — the DSL integrand evaluator.
* :func:`fem_1d._integrate_term` — weighted sum over quad points.
* :func:`fem_1d._apply_dirichlet_*` — Dirichlet enforcement (symmetric/row/transient).
* :func:`fem_utils._promote_to_quadratic` — P1→P2 mesh promotion.
* :func:`fem_utils._cell_region_mask` — per-cell sub-region indicator.

New components (this module only):

* :func:`fem_lagrange.lagrange_triangle` / :func:`fem_lagrange.lagrange_tet` /
  :func:`fem_lagrange.identity_pushforward` — basix-backed Lagrange reference tabulation +
  isoparametric gradient map.
* :func:`fem_facets.build_facet_connectivity` / :func:`fem_facets.compute_face_normals`
  — boundary face connectivity + outward normals for surface integration.

References
----------
Matrix extraction via ``jax.jacfwd(residual)(zeros)`` follows Griewank & Walther,
*Evaluating Derivatives*, SIAM (2008), §3.5 — the same pattern as :mod:`fem_1d`
and :mod:`fem_nonnodal`.

Scope
-----
Lagrange P1/P2 fields on 2D triangle and 3D tetrahedral meshes (single- and multi-field,
linear, nonlinear, and transient), with Dirichlet and Neumann/Robin boundary conditions
(2D edge / 3D tet-face surface quadrature).  Niches outside this scope raise a clear
``NotImplementedError`` from ``jno.fem`` rather than assembling silently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

from .fem_1d import (
    _apply_dirichlet_rows,
    _apply_dirichlet_symmetric,
    _apply_dirichlet_transient,
    _integrate_term,
    _line_quadrature,
    _region_node_ids,
)
from .fem_facets import _LOCAL_FACES_TET, build_facet_connectivity, compute_face_normals
from .fem_lagrange import (
    _lagrange_basix,
    identity_pushforward,
    identity_pushforward_hess,
    lagrange_interp_points,
    lagrange_tet,
    lagrange_triangle,
)
from .fem_utils import (
    _cell_region_mask,
    _collect_region_mask_names,
    _eval_integrand,
    _gather_temporal_tags,
    _infer_fields,
    _lower_statefield_to_trial,
    _promote_to_degree,
    _test_field_index,
    apply_compress_plan,
    bcoo_set_dirichlet_rows,
    bcoo_zero_rows,
    bcoo_zero_rows_cols,
    compress_eager,
    compress_plan,
)
from .parametric_helpers import _collect_runtime_parameter_exprs
from .weak_form import (
    _apply_sign,
    _contains_temporal_derivative,
    _is_obviously_nonlinear_in_unknown,
    _split_additive_terms,
)

# Reference simplex vertex coordinates (basix convention).
_REF_TRI_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # v0=(0,0), v1=(1,0), v2=(0,1)
_REF_TET_VERTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# Local face ordering for a triangle: entry k = (local_node_a, local_node_b, opp_node).
_LOCAL_FACES_TRI = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def _get_mesh(domain, dim: int, order: int):
    """P1 base mesh + optionally promoted P{order} mesh, both as NumPy arrays.

    Returns ``(pts_p1, cells_p1, pts_f, cells_f)`` where:

    * ``pts_p1, cells_p1`` — the original P1 mesh (used for region masks and facets).
    * ``pts_f, cells_f`` — same as P1 when ``order=1``; promoted P2 when ``order=2``.
    """
    # meshio names the simplex cell block "triangle" (2D) / "tetra" (3D) -- distinct from the basix
    # CellType name "tetrahedron" the facet machinery uses.
    meshio_key = "triangle" if dim == 2 else "tetra"
    pts_p1 = np.asarray(domain.mesh.points)[:, :dim]
    cells_p1 = np.asarray(domain.mesh.cells_dict[meshio_key], dtype=np.int64)
    if order == 1:
        return pts_p1, cells_p1, pts_p1, cells_p1
    if dim not in (2, 3):
        raise NotImplementedError(f"Dimension {dim} not supported by native assembler.")
    # P{order} node mesh: place the element's reference interpolation points (basix DOF order) on each
    # cell and dedup by coordinate. One code path for P2 and P3+ (the P2 midpoints are the k=2 case).
    pts_f, cells_f = _promote_to_degree(pts_p1, cells_p1, lagrange_interp_points(dim, order))
    return pts_p1, cells_p1, pts_f, cells_f


def _region_node_ids_from_pts(domain, region: str, pts_all: np.ndarray) -> List[int]:
    """Node ids in ``pts_all`` for ``region`` — a **geometric interior sub-region**
    (``domain.region(name, polygon)``) by point-in-polygon, else the region's location function.

    A shapely polygon is not jax-traceable, so an interior sub-region cannot go through the jax
    ``_make_tag_location_fn`` path below; it is resolved here in numpy. This is what lets a subdomain
    solve (domain decomposition) restrict/pin on a named sub-region."""
    ptags = getattr(domain, "_polygon_tags", {})
    if region in ptags and ptags[region][0] == "interior":
        pts = np.asarray(pts_all)
        try:
            from shapely import contains_xy  # vectorized (shapely >= 2.0.2)

            hits = np.asarray(contains_xy(ptags[region][1].buffer(1e-9), pts[:, 0], pts[:, 1]))
        except (ImportError, AttributeError):
            from shapely.geometry import Point

            g = ptags[region][1].buffer(1e-9)
            hits = np.array([g.contains(Point(float(q[0]), float(q[1]))) for q in pts])
        return list(np.where(hits.reshape(-1))[0])

    loc = domain._make_tag_location_fn(region)
    if loc is None:
        raise ValueError(f"jno.fem (native): region {region!r} has no location function.")
    pts_j = jnp.asarray(pts_all)
    n = int(pts_j.shape[0])
    num_args = loc.__code__.co_argcount if hasattr(loc, "__code__") else 1
    hits = jax.vmap(loc)(pts_j) if num_args == 1 else jax.vmap(loc)(pts_j, jnp.arange(n))
    return list(np.where(np.asarray(hits).reshape(-1))[0])


# ---------------------------------------------------------------------------
# Face (edge) pre-tabulation for surface integration
# ---------------------------------------------------------------------------


def _build_face_tables(elem_degree: int, quad_degree: int, dim: int = 2):
    """Pre-tabulate the parent-cell Lagrange basis at the quad points of each local facet.

    Dimension-generic: a 2D triangle's facets are its 3 edges (1-D Gauss quadrature); a 3D tet's
    facets are its 4 triangular faces (2-D triangle quadrature). The facet ordering matches
    ``build_facet_connectivity`` (``_LOCAL_FACES_TRI`` / ``_LOCAL_FACES_TET``) so a connectivity
    ``local_face`` index ``k`` selects the right table.

    Returns ``(face_phi, face_dphi_ref, face_ref_qp, face_ref_tangs, face_w)``:

    * ``face_phi``       ``(n_faces, n_q, n_dof)``       parent basis values at facet qp.
    * ``face_dphi_ref``  ``(n_faces, n_q, n_dof, dim)``  reference-domain gradients.
    * ``face_ref_qp``    ``(n_faces, n_q, dim)``         parent-reference coords of facet qp.
    * ``face_ref_tangs`` ``(n_faces, dim-1, dim)``       the ``dim-1`` reference tangent vectors that
      span each facet (one edge tangent in 2D; two face tangents in 3D). The physical area element is
      ``|J·t|`` (2D edge length) or ``|（J·t0) × (J·t1)|`` (3D face area), formed in ``_surf_elem_res``.
    * ``face_w``         ``(n_q,)``                      reference-facet quadrature weights (1-D Gauss
      on [0, 1] summing to 1 in 2D; triangle weights summing to 1/2 in 3D).
    """
    import basix
    from basix import CellType

    if dim == 2:
        cell, ref_verts, local_faces = CellType.triangle, _REF_TRI_VERTS, _LOCAL_FACES_TRI
        gp_1d, face_w = (np.asarray(x) for x in _line_quadrature(quad_degree))

        def _facet_qp_tangs(nodes):  # an edge between two vertices
            va, vb = ref_verts[nodes[0]], ref_verts[nodes[1]]
            ref_qp = va[None, :] * (1.0 - gp_1d[:, None]) + vb[None, :] * gp_1d[:, None]  # (n_q, 2)
            return ref_qp, np.stack([vb - va])  # tangs (1, 2)
    else:
        cell, ref_verts, local_faces = CellType.tetrahedron, _REF_TET_VERTS, _LOCAL_FACES_TET
        qp_tri, face_w = (np.asarray(x) for x in basix.make_quadrature(CellType.triangle, quad_degree))

        def _facet_qp_tangs(nodes):  # a triangular face spanned by three vertices
            va, vb, vc = ref_verts[nodes[0]], ref_verts[nodes[1]], ref_verts[nodes[2]]
            xi, eta = qp_tri[:, 0], qp_tri[:, 1]
            ref_qp = va[None] * (1 - xi - eta)[:, None] + vb[None] * xi[:, None] + vc[None] * eta[:, None]
            return ref_qp, np.stack([vb - va, vc - va])  # tangs (2, 3)

    elem = _lagrange_basix(cell, elem_degree)
    phi_list, dphi_list, qp_list, tang_list = [], [], [], []
    for entry in local_faces:
        ref_qp, tangs = _facet_qp_tangs(entry[:dim])  # entry[:dim] = the facet's vertex local ids
        tab = elem.tabulate(1, ref_qp)  # (1 + dim, n_q, n_dof, 1)
        phi_list.append(tab[0, :, :, 0])  # (n_q, n_dof)
        dphi_list.append(np.stack([tab[1 + d, :, :, 0] for d in range(dim)], axis=-1))  # (n_q, n_dof, dim)
        qp_list.append(ref_qp)
        tang_list.append(tangs)

    return (
        jnp.asarray(np.stack(phi_list)),  # (n_faces, n_q, n_dof)
        jnp.asarray(np.stack(dphi_list)),  # (n_faces, n_q, n_dof, dim)
        jnp.asarray(np.stack(qp_list)),  # (n_faces, n_q, dim)
        jnp.asarray(np.stack(tang_list)),  # (n_faces, dim-1, dim)
        jnp.asarray(face_w),  # (n_q,)
    )


def _facet_area_element(J, tangs):
    """Physical facet measure element from the reference tangents ``tangs`` (``dim-1, dim``) pushed
    forward by the cell Jacobian ``J`` (``dim, dim``): the edge length ``|J·t|`` in 2D, the face area
    ``|(J·t0) × (J·t1)|`` in 3D. Multiplying it by the reference-facet weights gives ``dS``."""
    T = tangs @ J.T  # (dim-1, dim) physical tangents
    return jnp.linalg.norm(T[0]) if T.shape[0] == 1 else jnp.linalg.norm(jnp.cross(T[0], T[1]))


def _face_normals_jax(points, facet_verts, sign):
    """Differentiable outward unit boundary-facet normals ``(n_bfaces, dim)`` from (traced) vertex
    positions -- the JAX companion of the host-numpy :func:`fem_facets.compute_face_normals`, so a facet's
    normal re-evaluates (and stays differentiable) when its vertices move (trainable coordinates / ALE).

    ``facet_verts`` is ``(n_bfaces, dim)`` P1 vertex ids per facet (``conn.face_nodes``); ``sign`` is a
    precomputed ``±1`` per facet fixing the outward orientation. The raw normal is the 90°-rotated edge
    tangent (2D) or the edge cross product (3D); the orientation sign is **frozen** because it is locally
    constant -- it only flips at element inversion (tangling), the same validity envelope as ``detJ``. See
    plans/differentiable-r-adaptivity.md (Feature 3)."""
    v = points[facet_verts]  # (n_bfaces, n_face_nodes, dim)
    dim = v.shape[-1]
    if dim == 2:  # edge -> rotate the tangent 90°
        t = v[:, 1] - v[:, 0]
        n_raw = jnp.stack([t[:, 1], -t[:, 0]], axis=1)
    else:  # triangular face -> cross product of two edges
        n_raw = jnp.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n = sign[:, None] * n_raw
    return n / jnp.linalg.norm(n, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Native fem_context for the VPINN / grouped-weak-form path
# ---------------------------------------------------------------------------


def build_native_fem_context(domain, *, element_type, quad_degree, vec=1, neumann_tags=(), dirichlet_node_ids=None):
    """Build ``domain.fem_context`` for the VPINN / grouped-weak-form evaluator
    (``trace_evaluator._eval_grouped_assembly``).

    Returns ``(fem_context, vol_quad_points, surface_quad_by_tag, surface_normals_by_tag)``,
    computed from the native Lagrange element + facet machinery. The geometry is affine-simplex
    (P1 vertices), so the cell Jacobian, ``JxW`` and physical gradients are exact for P1/P2 nodal
    bases.
    """
    dim = int(domain.dimension)
    # element-type label -> polynomial order: TRI6/TET10 == P2; generic "TRI-P{k}"/"TET-P{k}" carries k.
    order = 2 if element_type in ("TRI6", "TET10") else (int(element_type.split("-P")[1]) if "-P" in element_type else 1)
    quad_degree = max(quad_degree, 2 * order)

    pts_p1, cells_p1, pts_f, cells_f = _get_mesh(domain, dim, order)
    spec = lagrange_triangle(order, quad_degree)
    ref_vals = jnp.asarray(spec.ref_values)  # (n_q, n_dof, 1)
    ref_grads = jnp.asarray(spec.ref_grads)  # (n_q, n_dof, 1, dim)
    qp = jnp.asarray(spec.quad_points)  # (n_q, dim)
    qw = jnp.asarray(spec.quad_weights)  # (n_q,)
    n_q, n_dof = int(qw.shape[0]), int(ref_vals.shape[1])
    test_vec = int(vec)

    pts_j = jnp.asarray(pts_p1)
    cells_p1_j = jnp.asarray(cells_p1, dtype=jnp.int32)
    cells_f_j = jnp.asarray(cells_f, dtype=jnp.int32)
    n_cells = int(cells_f.shape[0])
    num_total_nodes = int(pts_f.shape[0])

    def _cell(c):
        verts = pts_j[cells_p1_j[c]]  # (dim+1, dim) — P1 geometry vertices
        J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim)
        detJ = jnp.linalg.det(J)
        phi, dphi = identity_pushforward(ref_vals, ref_grads, J, detJ)  # (n_q,n_dof), (n_q,n_dof,dim)
        JxW = qw * jnp.abs(detJ)  # (n_q,)
        xq = verts[0] + qp @ J.T  # (n_q, dim)
        return phi, dphi, JxW, xq

    phis, dphis, JxWs, xqs = jax.vmap(_cell)(jnp.arange(n_cells))

    N_flat = phis.reshape(-1, n_dof)  # (n_cells*n_q, n_dof)
    dN_dx_flat = dphis.reshape(-1, n_dof, dim)  # (n_cells*n_q, n_dof, dim)
    # v_grads_JxW = physical test gradient * JxW, broadcast over the test-vec component axis
    vg = (dphis * JxWs[:, :, None, None])[:, :, :, None, :]  # (n_cells,n_q,n_dof,1,dim)
    v_grads_JxW_flat = jnp.broadcast_to(vg, (n_cells, n_q, n_dof, test_vec, dim)).reshape(-1, n_dof, test_vec, dim)
    quad_points = xqs.reshape(-1, dim)

    local_areas = jnp.einsum("cq,cqa->ca", JxWs, phis)  # lumped nodal areas
    global_areas = jax.ops.segment_sum(local_areas.reshape(-1), cells_f_j.reshape(-1), num_segments=num_total_nodes)

    dirichlet_nodes = (
        jnp.asarray(sorted(set(int(i) for i in dirichlet_node_ids)), dtype=jnp.int32)
        if dirichlet_node_ids
        else jnp.asarray([], dtype=jnp.int32)
    )

    fem_context = {
        "cells": cells_f_j,
        "flat_cells": cells_f_j,
        "global_areas": global_areas,
        "N_flat": N_flat,
        "dN_dx_flat": dN_dx_flat,
        "v_grads_JxW_flat": v_grads_JxW_flat,
        "JxW": JxWs,
        "quad_points": quad_points,
        "test_vec": test_vec,
        "num_total_nodes": num_total_nodes,
        "dirichlet_nodes": dirichlet_nodes,
        "surface_data": {},
    }

    # ---- surface_data per Neumann tag (boundary weak terms) ----
    surface_quad_by_tag: dict = {}
    surface_normals_by_tag: dict = {}
    if neumann_tags:
        cell_key = "triangle" if dim == 2 else "tetrahedron"
        conn = build_facet_connectivity(cells_p1, cell_key)
        normals_all = compute_face_normals(pts_p1, conn, cells_p1, cell_key) if conn.n_bfaces > 0 else np.zeros((0, dim))
        fp_phi, fp_dphi_ref, fp_qp, fp_tangs, gw_face = _build_face_tables(order, quad_degree, dim)
        for tag in neumann_tags:
            region_nodes = {int(n) for n in _region_node_ids_from_pts(domain, tag, pts_p1)}
            face_ids = [
                fi
                for fi in range(conn.n_bfaces)
                if all(int(conn.face_nodes[fi, j]) in region_nodes for j in range(conn.face_nodes.shape[1]))
            ]
            if not face_ids:
                continue
            face_ids_j = jnp.asarray(face_ids, dtype=jnp.int32)
            parent = jnp.asarray(conn.parent_cell, dtype=jnp.int32)[face_ids_j]
            lface = jnp.asarray(conn.local_face, dtype=jnp.int32)[face_ids_j]
            normals_j = jnp.asarray(normals_all)[face_ids_j]

            def _face(c, k, n_vec, _fp=fp_phi, _fd=fp_dphi_ref, _fq=fp_qp, _ft=fp_tangs, _gw=gw_face):
                verts = pts_j[cells_p1_j[c]]
                J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)
                K = jnp.linalg.inv(J)
                phi_f = _fp[k]  # (n_fq, n_dof)
                dphi_f = jnp.einsum("qnd,dD->qnD", _fd[k], K)  # (n_fq, n_dof, dim)
                jac_f = _facet_area_element(J, _ft[k])  # edge length (2D) / face area (3D)
                nanson = _gw * jac_f  # (n_fq,)
                xq_f = verts[0] + _fq[k] @ J.T  # (n_fq, dim)
                return phi_f, dphi_f, nanson, xq_f

            phi_fs, dphi_fs, nanson_fs, xq_fs = jax.vmap(_face)(parent, lface, normals_j)
            # (n_faces, n_fq, n_dof), (n_faces, n_fq, n_dof, dim), (n_faces, n_fq), (n_faces, n_fq, dim)
            n_fq = int(phi_fs.shape[1])
            parent_nodes = cells_f_j[parent]  # (n_faces, n_loc) global parent-cell node ids
            local_b_areas = jnp.einsum("fq,fqn->fn", nanson_fs, phi_fs)
            global_b_areas = jax.ops.segment_sum(
                local_b_areas.reshape(-1), parent_nodes.reshape(-1), num_segments=num_total_nodes
            )
            quad_pts_flat = xq_fs.reshape(-1, dim)
            # outward normals broadcast to every face quad point
            quad_normals = jnp.broadcast_to(normals_j[:, None, :], (len(face_ids), n_fq, dim)).reshape(-1, dim)

            fem_context["surface_data"][tag] = {
                "flat_parent_nodes": parent_nodes.reshape(-1),
                "face_shape_vals": phi_fs,
                "face_shape_grads": dphi_fs,
                "nanson_scale": nanson_fs,
                "global_boundary_areas": global_b_areas,
                "quad_points": quad_pts_flat,
                "quad_normals": quad_normals,
            }
            surface_quad_by_tag[tag] = np.asarray(quad_pts_flat)
            surface_normals_by_tag[tag] = np.asarray(quad_normals)

    return fem_context, np.asarray(quad_points), surface_quad_by_tag, surface_normals_by_tag


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


#: Element-chunk policy for the assembly currently being built. Set by ``jno.fem(chunk=...)`` for the
#: duration of one assembly and captured (not read lazily) by :func:`assemble_fem_native`, because the
#: residual/jacobian closures are CALLED long afterwards -- at solve time, outside any context.
#: ``None`` = automatic (device-derived), ``False``/``0`` = no chunking, positive int = cells per chunk.
#: A list rather than a plain global so the context manager can restore the previous value on nesting.
_CHUNK_OVERRIDE = [None]
#: Set when an assembly consumed the override, so ``jno.fem`` can refuse an explicit ``chunk=`` that
#: reached an assembler with no element loop instead of silently ignoring it.
_CHUNK_CONSUMED = [False]


def normalize_chunk(chunk):
    """Validate a user ``chunk=`` value. ``None`` -> automatic, ``False``/``0`` -> off, int -> cells."""
    if chunk is None:
        return None
    if chunk is False or (isinstance(chunk, int) and not isinstance(chunk, bool) and chunk == 0):
        return 0
    if chunk is True:
        return None  # "yes, chunk" == automatic
    if isinstance(chunk, (int, np.integer)) and int(chunk) > 0:
        return int(chunk)
    raise ValueError(
        f"jno.fem: chunk={chunk!r} is not a valid element-chunk size. Pass a positive int (cells per "
        "chunk), False to disable chunking, or None (the default) to size it from the device."
    )


def assemble_fem_native(
    domain,
    volume_terms: List[Any],
    boundary_terms: Dict[str, List[Any]],
    dirichlet_raw: List[Tuple],
    ic_residuals: List[Any],
    *,
    vec: int,
    quad_degree: int,
    evolution: Optional[Dict[Any, Any]] = None,
) -> Tuple[Any, str]:
    """Assemble a Lagrange FEM system into ``(op, mode, offs)`` for :class:`FEM`.

    ``mode`` is ``"linear"``, ``"nonlinear"``, or ``"transient"``; ``op`` matches the
    return-type contract of :func:`fem_1d.assemble_fem_1d` and
    :func:`fem_nonnodal.assemble_fem_nonnodal`.

    Scope: scalar/vector Lagrange P1/P2 fields on 2D triangle and 3D tetrahedral meshes
    (single- and multi-field), with Dirichlet and Neumann/Robin boundary conditions (2D edge /
    3D tet-face surface quadrature).
    """
    from ...trace import FemResidualOperator

    dim = int(domain.dimension)
    if dim not in (2, 3):
        raise NotImplementedError(f"assemble_fem_native: only dim=2 and dim=3 are supported; got dim={dim}.")
    cell_key = "triangle" if dim == 2 else "tetrahedron"

    ctx = dict(getattr(domain, "context", {}) or {})
    ctx.pop("cell_size", None)  # `dom.cell_size` placeholder; the real per-cell h is packed per volume element below

    # -------------------------------------------------------------------------
    # Field layout inference
    # -------------------------------------------------------------------------

    fields: List[Dict] = []
    field_index: Dict[Any, int] = {}
    for bare in volume_terms:
        for _, sub in _split_additive_terms(domain, bare):
            lowered = _lower_statefield_to_trial(sub, {})
            fs, _ = _infer_fields(lowered)
            for f in fs:
                if f["field_key"] not in field_index:
                    field_index[f["field_key"]] = len(fields)
                    fields.append(f)

    if not fields:
        raise ValueError("assemble_fem_native: no trial fields found in volume_terms.")

    for f in fields:
        sp = f.get("space", "Lagrange")
        if sp not in ("Lagrange", ""):
            raise NotImplementedError(
                f"assemble_fem_native: only Lagrange fields are supported; got space={sp!r}. "
                "Use assemble_fem_nonnodal for RT/N1E/P0 fields."
            )

    # -------------------------------------------------------------------------
    # Per-field mesh data
    # -------------------------------------------------------------------------

    mesh_data = [_get_mesh(domain, dim, f["order"]) for f in fields]
    pts_p1 = mesh_data[0][0]  # (n_pts_p1, dim)    — P1 node coordinates (shared)
    cells_p1 = mesh_data[0][1]  # (n_cells, dim+1)  — P1 simplex connectivity (shared)

    pts_f_all = [d[2] for d in mesh_data]  # per-field node coords (P2 or P1)
    cells_f_all = [d[3] for d in mesh_data]  # per-field connectivity
    n_nodes_f = [d[2].shape[0] for d in mesh_data]  # number of DOF nodes per field
    vecs = [int(f["vec"]) for f in fields]

    # Global DOF block offsets: [0, n0, n0+n1, ...]
    offs = [0]
    for i in range(len(fields)):
        offs.append(offs[-1] + n_nodes_f[i] * vecs[i])
    total = offs[-1]

    # Tell region-mask machinery which mesh to classify against
    domain._fem_assembly_points = pts_p1
    domain._fem_assembly_cells = cells_p1

    # The DOF coordinates the flat solution lives on (vertices + edge midpoints for P2).
    # ``FEM.points`` reads ``[0]`` so the solution can be interpreted at the right coordinates --
    # the first field's nodes. The full per-field list backs ``FEM.field_points`` for coupled
    # problems (e.g. Taylor-Hood velocity vs pressure nodes).
    domain._fem_native_dof_points = np.asarray(pts_f_all[0])
    domain._fem_native_dof_points_all = [np.asarray(p) for p in pts_f_all]

    # Field-0 assembly cells + element order, for the periodic-tie reduction (``_build_periodic_
    # reduction`` reads the assembly mesh's cells to extract boundary facets); ``_finalize`` reads
    # these. The full per-field lists back the heterogeneous-order coupled periodic reduction
    # (Taylor-Hood: per-field P_i from each field's own cells/order, matched to its ties by field_key).
    domain._fem_native_assembly_cells = np.asarray(cells_f_all[0])
    domain._fem_native_assembly_order = int(fields[0]["order"])
    domain._fem_native_assembly_cells_all = [np.asarray(cf) for cf in cells_f_all]
    domain._fem_native_field_orders = [int(f["order"]) for f in fields]
    domain._fem_native_field_keys = [f["field_key"] for f in fields]

    # -------------------------------------------------------------------------
    # Element specs and JAX constants
    # -------------------------------------------------------------------------

    _lagrange_simplex = lagrange_tet if dim == 3 else lagrange_triangle
    specs = [_lagrange_simplex(f["order"], quad_degree) for f in fields]
    # All specs share the same simplex quadrature rule (basix is deterministic)
    qp_shared = jnp.asarray(specs[0].quad_points)  # (n_quad, dim)
    qw_shared = jnp.asarray(specs[0].quad_weights)  # (n_quad,)

    pts_j = jnp.asarray(pts_p1)
    cells_j = jnp.asarray(cells_p1, dtype=jnp.int32)
    n_cells = int(cells_p1.shape[0])

    ref_vals_all = [jnp.asarray(s.ref_values) for s in specs]  # list of (n_quad, n_dof_i, 1)
    ref_grads_all = [jnp.asarray(s.ref_grads) for s in specs]  # list of (n_quad, n_dof_i, 1, dim)
    # reference Hessians (for 4th-order / biharmonic weak forms); None if a spec doesn't tabulate them
    ref_hess_all = [None if s.ref_hess is None else jnp.asarray(s.ref_hess) for s in specs]
    cells_f_j = [jnp.asarray(cf, dtype=jnp.int32) for cf in cells_f_all]  # list of (n_cells, n_local_i)

    # Per-field cell DOF index arrays: (n_cells, n_local_i * vec_i)
    cdofs = []
    for i in range(len(fields)):
        comp = jnp.arange(vecs[i])
        cd = offs[i] + cells_f_j[i][:, :, None] * vecs[i] + comp[None, None, :]
        cdofs.append(cd.reshape(n_cells, -1))

    # -------------------------------------------------------------------------
    # Per-region masks (collected from all volume terms)
    # -------------------------------------------------------------------------

    def _collect_masks(terms):
        return tuple(
            sorted(
                {
                    r
                    for bare in terms
                    for _, sub in _split_additive_terms(domain, bare)
                    for r in _collect_region_mask_names(_lower_statefield_to_trial(sub, {}))
                }
            )
        )

    region_mask_names: Tuple[str, ...] = _collect_masks(volume_terms)
    region_mask_arrays = [
        jnp.asarray(_cell_region_mask(domain, r), dtype=qw_shared.dtype).reshape(-1) for r in region_mask_names
    ]
    _region_mask_index = {r: i for i, r in enumerate(region_mask_names)}  # O(1) lookup vs list().index() per cell

    # Temporal variable tags (e.g. "__time__") used inside the weak form's coefficients -- a
    # time-dependent source s(x,t) or operator. The residual/Jacobian builders thread the runtime time
    # `t` into the kernel's volume_vars at the matching slots so `_eval_integrand` resolves them;
    # the packing order is [temporal..., runtime_param..., region_mask...] (see _make_internal_vars).
    _temporal_tag_set: set = set()
    for bare in volume_terms:
        _gather_temporal_tags(bare, _temporal_tag_set)
    for _exprs in boundary_terms.values():
        for bare in _exprs:
            _gather_temporal_tags(bare, _temporal_tag_set)
    temporal_tags: Tuple[str, ...] = tuple(sorted(_temporal_tag_set))

    # Runtime parameters (trainable ``jno.np.parameter(...)`` coefficients, e.g. an unknown diffusivity
    # in an inverse problem). Their values arrive at solve time in an ``args`` dict; the builders pack
    # them into volume_vars right AFTER the temporal slots so ``_eval_integrand`` resolves each
    # parameter node (layout [temporal..., runtime_param..., region_mask...]). A SCALAR parameter is
    # broadcast; a nodal FIELD parameter k(x) (``jno.np.parameter(phi)``) has its per-cell nodal values
    # gathered and interpolated to the quad points via the field's shape functions.
    from .parametric_helpers import _is_fem_field_parameter

    _rt_param_exprs: Dict[str, Any] = {}
    for bare in volume_terms:
        _collect_runtime_parameter_exprs(bare, _rt_param_exprs)
    for _exprs in boundary_terms.values():
        for bare in _exprs:
            _collect_runtime_parameter_exprs(bare, _rt_param_exprs)
    runtime_parameter_tags: Tuple[str, ...] = tuple(sorted(_rt_param_exprs))
    _field_param_names: set = {n for n, expr in _rt_param_exprs.items() if _is_fem_field_parameter(expr)}

    # Neural coefficients (``jno.nn.wrap(net)`` called inside the weak form, e.g. ``net(x,y)*u.dx*v.dx``).
    # Unlike scalar/nodal parameters they never enter the per-cell ``volume_vars`` -- a weight pytree is
    # cell-independent -- the kernel instead re-evaluates the network at the quad points from the
    # {name: module} table (``neural_local_table``) threaded via ``loc["neural_coefficients"]``. A neural
    # coefficient needs NO per-field resolution (unlike a nodal FIELD parameter, which gathers on one
    # field's mesh): the net is evaluated at the shared physical quad points, and a trial-input
    # ``net(u_i)`` resolves its field through ``_field_data`` (op_id/field_key) inside the kernel -- so a
    # coupled (multi-field) form threads it unchanged. The collect / crux-delivery / kernel-table
    # mechanism lives in ``parametric_helpers`` (shared with the non-nodal assembler).
    from .parametric_helpers import collect_neural_slots, neural_local_table, neural_operator_exprs

    _neural = collect_neural_slots(volume_terms, boundary_terms, runtime_parameter_tags=runtime_parameter_tags)
    neural_param_names, _neural_models = _neural.param_names, _neural.models
    _param_and_neural_exprs = neural_operator_exprs(_rt_param_exprs, _neural)

    # Trainable mesh-coordinate parameters (geometry design variables) registered by
    # ``Variable.trainable()`` on a spatial coordinate: their value is scattered into the P1 geometry
    # points before the cell Jacobian is formed (``_apply_coord_params`` below), so ``∂(solve)/∂X``
    # flows through the ordinary assembly. They ride ``runtime_parameter_exprs`` (so crux discovers them
    # and their value arrives in ``args``) but stay OUT of ``runtime_parameter_tags`` -- they are not term
    # coefficients (``_runtime_vals`` must not pack them). See plans/differentiable-r-adaptivity.md (Feature 2).
    _coord_specs: List[Tuple[Any, int, str]] = []
    for _cspec in getattr(domain, "_trainable_coords", None) or []:
        _cname = str(_cspec["name"])
        _param_and_neural_exprs = {**_param_and_neural_exprs, _cname: _cspec["expr"]}
        _coord_specs.append((jnp.asarray(_cspec["ids"], dtype=jnp.int32), int(_cspec["axis"]), _cname))

    # Frozen fields (ui.freeze(values)): KNOWN nodal vectors whose value/gradient are delivered at the
    # quad points (e.g. as neural-coefficient inputs). Collected once; their per-cell nodal slice is a
    # compile-time constant gathered below and threaded via loc["frozen_fields"].
    def _collect_frozen_fields(terms):
        from ...trace import FrozenField
        from .solver_helper import iter_children

        found: Dict[Any, Any] = {}
        seen: set = set()
        stack = list(terms)
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            if isinstance(n, FrozenField):
                found[n.frozen_id] = n
                continue
            stack.extend(iter_children(n))
        return found

    _frozen_nodes = _collect_frozen_fields(list(volume_terms) + list(boundary_terms))

    # Per-quadrature-point STEP HISTORY (``v.i(k)``): scan the terms for HistoryRef nodes and record, per
    # base variable, how many past states to buffer (the most-negative offset). The buffer itself lives on
    # the runtime ``args`` (so it UPDATES each load step without re-assembly and rides the driver's scan
    # carry differentiably); here we only fix the layout so the driver can allocate and thread it. Buffer
    # shape per variable: ``(n_cells, n_quad, depth, *value_shape)`` -- per Gauss point, exactly `depth`
    # deep, so it is memory-minimal. Presence of any history forces the args-threading (parametric) path.
    from ...trace import history_variables as _history_variables

    # Evolution updates (``state.evolves(formula)``) advance internal states between load steps. Their
    # formulas are walked here too, so a ``state.i(-1)`` that appears ONLY inside an evolution formula
    # (not in the weak form) still allocates its buffer with the right depth.
    _evolution = dict(evolution or {})  # {history_key: StateUpdate}
    # WHERE each step-history state lives: a state read at ``.i(k)`` inside a BOUNDARY term buffers on that
    # region's face quadrature points (a *surface* state — e.g. a friction slip on the contact face);
    # otherwise it buffers on the cell quadrature points (a *volume* state — e.g. a plastic strain). Each
    # state's evolution formula is walked together with the reads it belongs to. (Boundary terms were
    # previously not scanned at all — ``list(boundary_terms)`` yielded the region keys, not the terms.)
    _bterm_list = [t for terms in boundary_terms.values() for t in terms]
    _surf_read_regions: Dict[Any, str] = {}  # history key -> the boundary region it is read on
    for _R, _rterms in boundary_terms.items():
        for _k in _history_variables(_rterms):
            _surf_read_regions[_k] = _R
    _surf_read_keys = set(_surf_read_regions)  # history keys read on ANY boundary
    _vol_evo = [su.formula for k, su in _evolution.items() if k not in _surf_read_keys]
    _surf_evo = [su.formula for k, su in _evolution.items() if k in _surf_read_keys]
    _vol_history_raw = _history_variables(list(volume_terms) + _vol_evo)  # {key: (base, depth)}
    _surf_history_raw = _history_variables(_bterm_list + _surf_evo)
    _both = set(_vol_history_raw) & set(_surf_history_raw)
    if _both:
        raise ValueError(
            "jno.fem: a step-history state is read at `.i(k)` on BOTH a volume and a boundary term; a "
            "state lives on one quadrature set (cells or faces). Split it into separate states."
        )
    history_specs = {
        key: {
            "name": str(getattr(base, "name", "hist")),
            "depth": int(depth),
            "value_shape": tuple(getattr(base, "value_shape", ())),
            "shape": (n_cells, int(qp_shared.shape[0]), int(depth)) + tuple(getattr(base, "value_shape", ())),
        }
        for key, (base, depth) in _vol_history_raw.items()
    }
    # Surface-state buffer specs are allocated below, once the boundary facet tables (face count +
    # per-face quadrature width) are built; kept here so the role/readout pass sees every state.
    _history_raw = {**_vol_history_raw, **_surf_history_raw}

    # Per-history-key READOUT + role, for the load-step march. Every buffered state advances one of two
    # ways between steps: (1) a *primary unknown* read at ``.i(-1)`` (its base is one of the solved fields
    # — e.g. a BDF2 ``u.i(-1)``) auto-buffers the just-solved ``u``, so its readout is the bare field
    # interpolated to the quad points; (2) an *internal state* (``ep``) advances by its
    # ``state.evolves(formula)`` update. A state read at ``.i(-1)`` that is NEITHER solved NOR has an
    # ``.evolves`` would leave its buffer frozen at zero — a silently wrong (deformation-theory) result —
    # so that is a hard build error (never a silent freeze). ``readout_formulas`` maps each key to the
    # trace expression the march evaluates per quad point to produce that state's next value.
    _solved_field_keys = {f["field_key"] for f in fields}
    _is_march = bool(getattr(domain, "_is_pseudo_time", False))
    history_roles: Dict[Any, str] = {}
    readout_formulas: Dict[Any, Any] = {}
    for key, (base, _depth) in _history_raw.items():
        if key in _evolution:
            history_roles[key] = "internal"
            readout_formulas[key] = _lower_statefield_to_trial(_evolution[key].formula, {})
        elif getattr(base, "field_key", None) in _solved_field_keys:
            history_roles[key] = "primary"  # auto-buffered from the solved unknown (the bare field at QPs)
            readout_formulas[key] = _lower_statefield_to_trial(base, {})
        elif _is_march:
            # A ``tau=`` domain signals a load-step MARCH: a buffered internal state with no ``.evolves``
            # would stay frozen at zero every step (a silently wrong, deformation-theory result). Fail
            # loud. (On a plain domain the same read is allowed — a residual you thread history into by
            # hand, e.g. to verify the zero-history reduction — so this only fires when marching.)
            raise ValueError(
                f"jno.fem: internal state {str(getattr(base, 'name', 'state'))!r} is read at `.i(-1)` but "
                "has no `.evolves(...)` update — on a `domain(tau=...)` march its history buffer would stay "
                "frozen at zero (a silently wrong, deformation-theory result). Add "
                "`state.evolves(<formula>)` to the `jno.fem([...])` list describing how it advances; or, if "
                "it is really the primary unknown, solve for it (give it a test function)."
            )
        else:
            history_roles[key] = "frozen"  # plain-domain history read, threaded by hand; not marchable

    # A trainable DIRICHLET VALUE ``u(region) - net(x)`` (an unknown boundary profile). The net is not an
    # integrand coefficient -- it is evaluated at the boundary NODES to form the Dirichlet lift -- so it is
    # collected here from ``dirichlet_raw`` (a bare net node; the front-end already rejected compound values)
    # and joins ``_param_and_neural_exprs`` as its own ``ModelWeights`` slot. The lift is (re-)built from the
    # runtime args in ``_dirichlet_pairs_at`` so ``∂b/∂weights`` flows through the solve.
    from ..._fem import _bare as _bare_node
    from ..._fem import _essential_spec as _essential_spec_node
    from ...trace import ModelWeights
    from .parametric_helpers import _is_neural_coefficient, _neural_coefficient_name

    _dir_net_models: Dict[str, Any] = {}
    for _fk, _rg, _comp, _val, _vnode in dirichlet_raw:
        _vn = _bare_node(_vnode) if _vnode is not None else None
        if _vn is not None and _is_neural_coefficient(_vn):
            _dir_net_models[_neural_coefficient_name(_vn)] = _vn.model
    # A net-valued INITIAL condition ``u(initial) - net(x)`` (a trainable starting state, recovered from a
    # trajectory): its weights join the runtime slots the same way, and the initial state is (re-)formed
    # from the runtime args in ``_state0_at`` so ``∂traj/∂weights`` flows through the IC.
    _ic_net_models: Dict[str, Any] = {}
    for _ic in ic_residuals:
        _icv = _essential_spec_node(_bare_node(_ic))[1]
        _vn = _bare_node(_icv) if _icv is not None else None
        if _vn is not None and _is_neural_coefficient(_vn):
            _ic_net_models[_neural_coefficient_name(_vn)] = _vn.model
    if _dir_net_models or _ic_net_models:
        _param_and_neural_exprs = {
            **_param_and_neural_exprs,
            **{n: ModelWeights(m) for n, m in _dir_net_models.items()},
            **{n: ModelWeights(m) for n, m in _ic_net_models.items()},
        }
    # A nodal FIELD parameter k(x) interpolates on one field's FE space. Single field -> field 0. For a
    # coupled (multi-field) problem, associate it with the field whose test function appears in the
    # term(s) that reference it (e.g. mu(x)*(grad u . grad v) -> the velocity field), so its nodal values
    # gather/interpolate on THAT field's mesh. All field params must resolve to one field (a single shared
    # ``shape_vals`` threads the interpolation), else it is rejected.
    _field_param_field_idx = 0
    if _field_param_names and len(fields) > 1:
        from .parametric_helpers import _contains_fem_field_parameter

        _pf_idxs: set = set()
        for bare in volume_terms:
            for sign, sub in _split_additive_terms(domain, bare):
                coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                if _contains_fem_field_parameter(coeff):
                    tfi = _test_field_index(coeff, field_index)
                    if tfi is not None:
                        _pf_idxs.add(int(tfi))
        if len(_pf_idxs) == 1:
            _field_param_field_idx = _pf_idxs.pop()
        else:
            # A field parameter shared across several fields' terms (a material property common to coupled
            # equations, e.g. a conductivity in both a thermal and a coupling term) is allowed WHEN those
            # fields share ONE FE space: same element order on the one mesh -> identical nodes, connectivity
            # and shape_vals, so k(x) interpolates the same regardless of which field's space we pick. Only
            # DIFFERING orders are ambiguous (no shared node set) and stay rejected.
            _orders = {int(fields[i]["order"]) for i in _pf_idxs}
            if not _pf_idxs or len(_orders) != 1:
                raise NotImplementedError(
                    "jno.fem (native): a FEM field parameter k(x) on a coupled (multi-field) problem must "
                    "appear in the terms of fields sharing ONE FE space (same element order) — its nodal "
                    "values interpolate on that shared space. Resolved to fields "
                    f"{sorted(_pf_idxs)} with orders {sorted(_orders)} (differing orders share no node set)."
                )
            _field_param_field_idx = min(_pf_idxs)

    # -------------------------------------------------------------------------
    # Surface integration setup
    # -------------------------------------------------------------------------

    # Per-field facet tables (one set per distinct element order). These tabulate the parent basis on
    # the simplex facets for surface (Neumann/Robin) integration -- a triangle's 3 edges in 2D, a tet's
    # 4 triangular faces in 3D -- so they are skipped when there are no boundary terms.
    face_tables_per_field = (
        [_build_face_tables(f["order"], quad_degree, dim) for f in fields] if boundary_terms else [None] * len(fields)
    )
    # face_tables_per_field[i] = (face_phi, face_dphi_ref, face_ref_qp, face_ref_tangs, face_w);
    # shapes: (n_faces, n_q, n_dof_i), (..., dim), (n_faces, n_q, dim), (n_faces, dim-1, dim), (n_q,)

    conn = build_facet_connectivity(cells_p1, cell_key)
    normals_np = compute_face_normals(pts_p1, conn, cells_p1, cell_key) if conn.n_bfaces > 0 else np.zeros((0, dim))
    # Frozen outward-orientation sign per boundary facet, so the normals can be recomputed differentiably
    # from moved vertices (``_face_normals_jax``) when coordinates are trainable (Feature 3). The sign is
    # locally constant (flips only at element inversion), so freezing it keeps the normal smooth on valid
    # meshes; the raw (unsigned) normal is built the same way as ``compute_face_normals``.
    if conn.n_bfaces > 0:
        _fv = np.asarray(pts_p1)[conn.face_nodes]  # (n_bfaces, n_face_nodes, dim)
        if dim == 2:
            _nraw = np.stack([(_fv[:, 1] - _fv[:, 0])[:, 1], -(_fv[:, 1] - _fv[:, 0])[:, 0]], axis=1)
        else:
            _nraw = np.cross(_fv[:, 1] - _fv[:, 0], _fv[:, 2] - _fv[:, 0])
        _facet_sign_j = jnp.asarray(np.where(np.sum(_nraw * np.asarray(normals_np), axis=1) >= 0, 1.0, -1.0))
        _facet_verts_j = jnp.asarray(conn.face_nodes, dtype=jnp.int32)
    else:
        _facet_sign_j = jnp.zeros((0,))
        _facet_verts_j = jnp.zeros((0, dim), dtype=jnp.int32)

    # Surface step-history buffer layout (now that the facet tables give the per-face quadrature width). A
    # state read at ``.i(k)`` on a boundary term (e.g. a friction slip on the contact face) lives on the
    # boundary FACE quadrature points: shape ``(n_bfaces, n_quad_surf, depth, *value_shape)``, indexed by
    # the global boundary-face id in ``_surf_elem_res`` (faces outside the term's region keep unused,
    # zeroed slots -- cheap, and avoids per-region local re-indexing). Threaded on ``args`` under a key
    # distinct from the volume history so the two never collide.
    _n_quad_surf = int(face_tables_per_field[0][4].shape[0]) if face_tables_per_field[0] is not None else 0
    surface_history_specs = {
        key: {
            "name": str(getattr(base, "name", "hist")),
            "depth": int(depth),
            "value_shape": tuple(getattr(base, "value_shape", ())),
            "shape": (int(conn.n_bfaces), _n_quad_surf, int(depth)) + tuple(getattr(base, "value_shape", ())),
            "surface": True,
        }
        for key, (base, depth) in _surf_history_raw.items()
    }
    # Boundary-face ids per region that carries a surface state (the faces its readout advances). Same
    # all-nodes-in-region face mask the residual's ``surface_work`` uses; computed once here.
    _surf_region_faces: Dict[str, np.ndarray] = {}
    if surface_history_specs and conn.n_bfaces > 0:
        for _R in set(_surf_read_regions.values()):
            _rnodes = {int(n) for n in _region_node_ids(domain, _R)}
            _mask = np.array(
                [
                    all(int(conn.face_nodes[fi, j]) in _rnodes for j in range(conn.face_nodes.shape[1]))
                    for fi in range(conn.n_bfaces)
                ]
            )
            _surf_region_faces[_R] = np.where(_mask)[0].astype(np.int32)

    # -------------------------------------------------------------------------
    # Cell-level field data builder (called inside vmap'd kernels)
    # -------------------------------------------------------------------------

    def _apply_coord_params(pts, args):
        """Scatter trainable mesh-coordinate parameters (``Variable.trainable()`` on a spatial coordinate)
        into the P1 geometry points, so the cell Jacobian and quad-point coordinates become differentiable
        in them. A no-op (returns ``pts`` unchanged) when there are no coordinate parameters. Called once
        per residual/Jacobian evaluation; the resulting dynamic points thread down into ``_cell_fields``."""
        if not _coord_specs or args is None:
            return pts
        for _ids, _axis, _name in _coord_specs:
            if _name in args:
                pts = pts.at[_ids, _axis].set(jnp.asarray(args[_name], dtype=pts.dtype).reshape(-1))
        return pts

    def _cell_fields(c, cell_sols, pts=pts_j):
        """Per-field ``(phi, dphi_phys, cell_sol)`` and shared ``(xq, meas)`` for cell c.

        ``cell_sols`` is a list of this cell's local DOF values per field, shape
        ``(n_local_i, vec_i)``. The residual path gathers them from the global state; the
        per-cell Jacobian path passes a *differentiated* local slice so ``jax.jacfwd`` sees
        an element-sized (not global) input — keeping the AD intermediate O(n_local), not
        O(n_dofs). ``pts`` is the (possibly coordinate-parameter-scattered) P1 geometry points;
        it defaults to the static mesh and is overridden per-eval when coordinates are trainable."""
        verts = pts[cells_j[c]]  # (dim+1, dim)
        J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim) columns = edges
        detJ = jnp.linalg.det(J)
        xq = verts[0][None, :] + qp_shared @ J.T  # (n_quad, dim) physical qp
        meas = jnp.abs(detJ)

        per = []
        for i in range(len(fields)):
            phi, dphi = identity_pushforward(ref_vals_all[i], ref_grads_all[i], J, detJ)
            fd = {"shape_vals": phi, "shape_grads": dphi, "cell_sol": cell_sols[i], "space": "Lagrange"}
            if ref_hess_all[i] is not None:  # physical shape Hessian for 4th-order weak forms
                fd["shape_hess"] = identity_pushforward_hess(ref_hess_all[i], J)
            per.append(fd)

        return per, xq, meas

    # Cell-local DOF bookkeeping for per-cell element-Jacobian assembly. ``cell_all_dofs[c]`` lists
    # every global DOF (all fields, node-major) the cell couples, so an element matrix's columns map
    # straight back to the global matrix; ``loc_seg`` splits a gathered local vector per field.
    n_local_f = [int(cells_f_j[i].shape[1]) for i in range(len(fields))]
    loc_seg = [0]
    for i in range(len(fields)):
        loc_seg.append(loc_seg[-1] + n_local_f[i] * vecs[i])
    cell_all_dofs = jnp.concatenate(cdofs, axis=1) if len(cdofs) > 1 else cdofs[0]  # (n_cell, n_local_all)

    # A LOAD-PATH field (``freeze_path``) is a FrozenField whose nodal values vary per load step: split it
    # out of the compile-time frozen gather, keep only its per-cell connectivity, and let the load-step
    # driver deliver each step's nodal slice through ``args["__loadpath__"]`` (like ``__history__``).
    from ...trace import LoadPathField as _LoadPathField

    _path_nodes = {fid: n for fid, n in _frozen_nodes.items() if isinstance(n, _LoadPathField)}
    _frozen_nodes = {fid: n for fid, n in _frozen_nodes.items() if not isinstance(n, _LoadPathField)}

    # Per-cell gather of each frozen field's nodal slice (n_cell, n_local, 1) -- a compile-time constant
    # (no args threading, no jacfwd tangent), gathered on the frozen field's own FE space via the same
    # connectivity as the live state, so its shape-gradient contraction matches the trial gradient.
    _frozen_gathered: Dict[Any, Any] = {}
    for _fid, _fnode in _frozen_nodes.items():
        _ffidx = field_index[_fnode.field_key]
        _fconn = cells_f_j[_ffidx]  # (n_cell, n_local)
        _fvals = jnp.asarray(_fnode.values)
        # scalar frozen field (n_nodes,) -> per-cell (n_local, 1); VECTOR (n_nodes, vec) -> (n_local, vec).
        # The kernel interpolation ``shape_vals . cell_nodal`` handles either (the trailing axis is carried).
        _frozen_gathered[_fid] = (
            _fvals[_fconn].reshape(_fconn.shape[0], _fconn.shape[1], 1) if _fvals.ndim == 1 else _fvals[_fconn]
        )

    # Load-path fields are scalar P1 fields on the mesh vertices (a temperature history, say) that are not
    # among the solved unknowns, so they have no assembled basis of their own. They borrow the nodal basis
    # and vertex connectivity of a P1 Lagrange field already in the problem (both live on the same mesh
    # vertices): we alias the load-path field's key to that P1 field's index so the kernel resolves its
    # shape functions, and gather its per-cell nodal slice on the same connectivity. Values arrive per step
    # from args; a spec (the full frame stack) rides the driver's scan.
    _path_conn: Dict[Any, Any] = {}
    path_specs: Dict[Any, Any] = {}
    if _path_nodes:
        _p1_idx = next(
            (i for i, f in enumerate(fields) if int(f["order"]) == 1 and str(f.get("space", "Lagrange")) == "Lagrange"),
            None,
        )
        if _p1_idx is None:
            raise NotImplementedError(
                "freeze_path(...): the load-path field is scalar P1 on the mesh vertices and borrows the "
                "nodal basis of a P1 Lagrange field in the problem, but this form has none. Give the primary "
                "unknown order=1 (P1)."
            )
        for _fid, _fnode in _path_nodes.items():
            # scalar and VECTOR load-path fields both borrow the P1 nodal basis (per-component interpolation
            # uses the same shape functions); the driver delivers the per-step slice (n_nodes[, vec]).
            field_index[_fnode.field_key] = _p1_idx  # resolve the load-path field's basis to the P1 field
            _path_conn[_fid] = cells_f_j[_p1_idx]  # scalar P1 vertex connectivity (n_cell, n_local)
            path_specs[_fid] = {"name": _fnode.name, "frames": jnp.asarray(_fnode.path_frames), "n_steps": _fnode.n_steps}

    if path_specs and not (_is_march and history_specs):
        # A load-path field's per-step slice is delivered by the load-step driver; without a march (a
        # `tau=` grid + step-history to drive it) it would never be supplied. Fail loud, name the fix.
        raise ValueError(
            "jno.fem: a `freeze_path(...)` load-path field requires a load-step march — build the domain "
            "with `domain(tau=(start, end, n))` and include step-history (a `.i(-1)` state advanced by "
            "`.evolves`, e.g. the plastic strain εₚ) so `fem.solve()` marches the load path and delivers "
            "each step's field slice. On a plain/steady domain the per-step values are never threaded."
        )

    def _add_loadpath_fields(loc, c, args):
        """Merge this load step's per-cell nodal slice for each load-path field into
        ``loc['frozen_fields']`` — so the FrozenField kernel path interpolates it to the quad points.
        The per-step nodal values come from the driver on ``args['__loadpath__']`` (like ``__history__``);
        without them (e.g. a non-march assembly) the field is simply absent, and a build-time guard has
        already required a march when a load-path field is present."""
        if not _path_conn or not isinstance(args, dict):
            return
        pbuf = args.get("__loadpath__")
        if not pbuf:
            return
        fz = dict(loc.get("frozen_fields", {}))
        for _fid, _conn in _path_conn.items():
            if _fid in pbuf:
                _arr = jnp.asarray(pbuf[_fid])
                # scalar field: (n_nodes,) -> per-cell (n_local, 1); vector field (prev-state mass):
                # (n_nodes, vec) -> per-cell (n_local, vec). The kernel interpolation handles either.
                if _arr.ndim <= 1:
                    fz[_fid] = _arr.reshape(-1)[_conn[c]].reshape(_conn.shape[1], 1)
                else:
                    fz[_fid] = _arr[_conn[c]]
        loc["frozen_fields"] = fz

    def _split_cell_local(local_vals):
        """Split a cell's gathered all-field local vector into per-field ``(n_local_i, vec_i)``."""
        return [local_vals[loc_seg[i] : loc_seg[i + 1]].reshape(n_local_f[i], vecs[i]) for i in range(len(fields))]

    def _gather_cell_local(u_blocks, c):
        """This cell's local DOFs across all fields, concatenated (matches ``cell_all_dofs[c]``)."""
        return jnp.concatenate([u_blocks[i][cells_f_j[i][c]].reshape(-1) for i in range(len(fields))])

    def _runtime_vals(c, t, args, dtype):
        """Cell ``c``'s runtime values for the kernel's volume_vars prefix, ordered
        ``[temporal..., runtime_param...]`` (region masks follow). Temporal + scalar parameters are
        single ``(1,)`` values (read back as scalars); a nodal FIELD parameter contributes this cell's
        local nodal slice ``(n_local,)`` which ``_runtime_parameter_value_from_internal_vars``
        interpolates to the quad points. Empty prefix when the form is autonomous and non-parametric."""
        tv = tuple(jnp.reshape(jnp.asarray(t, dtype=dtype), (-1,))[:1] for _ in temporal_tags)
        a = args or {}
        pv = []
        for name in runtime_parameter_tags:
            if name not in a:
                # Parameter not supplied for this assembly (e.g. the mass matrix, which references no
                # parameter): pack a zero placeholder of the right width. It is only ever read back if
                # the term actually contains the parameter node, in which case args carries its value.
                pv.append(jnp.zeros((n_local_f[_field_param_field_idx] if name in _field_param_names else 1,), dtype))
                continue
            flat = jnp.reshape(jnp.asarray(a[name], dtype=dtype), (-1,))
            # Nodal field parameter -> this cell's local nodal values on its field's mesh (field 0 for a
            # single-field problem; the resolved field for a coupled one).
            pv.append(flat[cells_f_j[_field_param_field_idx][c]] if name in _field_param_names else flat[:1])
        return tv + tuple(pv)

    # -------------------------------------------------------------------------
    # Generic residual builder (volume + optional surface terms)
    # -------------------------------------------------------------------------

    # Surface connectivity (hoisted: shared by the residual and Jacobian builders).
    normals_j = jnp.asarray(normals_np)
    parent_j = jnp.asarray(conn.parent_cell, dtype=jnp.int32)
    lface_j = jnp.asarray(conn.local_face, dtype=jnp.int32)

    def _surface_normals(pts):
        """Outward unit facet normals for the current geometry ``pts``: the frozen static normals when no
        coordinates are trainable (fast path), else recomputed differentiably from the moved vertices."""
        if conn.n_bfaces == 0 or not _coord_specs:
            return normals_j
        return _face_normals_jax(pts, _facet_verts_j, _facet_sign_j)

    def _vol_elem_res(c, local_all, coeff, tfi, rnames, t=0.0, args=None, pts=None):
        """Element residual of one volume term on cell ``c`` as a function of that cell's gathered
        all-field local DOFs ``local_all`` -> ``(n_test_dofs_tfi,)``. Driving the AD off this
        element-sized input (not the global state) is what keeps the per-cell Jacobian's intermediate
        O(n_local) instead of O(n_dofs). ``t`` / ``args`` carry the runtime time and parameters, packed
        per cell into volume_vars BEFORE the region masks (layout [temporal..., runtime_param...,
        region_mask...]). ``pts`` is the coordinate-parameter-scattered geometry (``None`` -> static mesh)."""
        cell_sols = _split_cell_local(local_all)
        per, xq, meas = _cell_fields(c, cell_sols, pts_j if pts is None else pts)
        # Element size h = |detJ|^(1/dim) at the quad points -> the `dom.cell_size` symbol (SUPG/GLS).
        # Constant w.r.t. the cell DOFs (geometry only), so the per-cell Jacobian sees it as a constant.
        h_qp = jnp.broadcast_to(meas ** (1.0 / dim), (qw_shared.shape[0], 1))
        cell_masks = tuple(region_mask_arrays[_region_mask_index[r]][c] for r in rnames)
        loc = {
            "physical_quad_points": xq,
            "fields": per,
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": {**ctx, "cell_size": h_qp},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": rnames,
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype) + cell_masks,
            "trial_value_shape": fields[tfi]["value_shape"],
            "trial_vec": vecs[tfi],
        }
        if _field_param_names:
            # The field parameter's nodal slice is interpolated to the quad points with its field's shape
            # functions (field 0 single-field; the resolved field for a coupled problem).
            # _runtime_parameter_value_from_internal_vars reads this top-level shape_vals.
            loc["shape_vals"] = per[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:  # trainable nets ride args (crux weights); frozen/placeholder -> stored module
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:  # known-field (ui.freeze) per-cell nodal slices for this cell
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if history_specs and args is not None:
            # This cell's per-quad-point history slice (n_quad, depth, *shape), gathered from the buffers on
            # ``args`` -- a plain per-cell index, so ``jacfwd`` treats it as a frozen constant.
            hbuf = args.get("__history__") if isinstance(args, dict) else None
            if hbuf:
                loc["qp_history"] = {k: hbuf[k][c] for k in history_specs if k in hbuf}
        _add_loadpath_fields(loc, c, args)  # per-step load-path field slices -> loc["frozen_fields"]
        return _integrate_term(domain, coeff, loc, qw_shared * meas)

    def _vol_elem_readout(c, local_all, formula, t=0.0, args=None):
        """Per-quadrature-point VALUE of an evolution formula on cell ``c`` -> ``(n_quad, *value_shape)``.

        Same field / parameter / frozen / history ``loc`` as :func:`_vol_elem_res` (so the formula reads
        the solved unknown through ``ε(u)`` and the previous state through ``ep.i(-1)``), but the formula
        carries NO test function, so it is *evaluated* at the quad points (``_eval_integrand``) rather than
        integrated. This is the internal-state update the load-step march applies after each solve.
        Reverse-mode differentiable in ``local_all`` (the solved DOFs) and the history buffers."""
        cell_sols = _split_cell_local(local_all)
        per, xq, meas = _cell_fields(c, cell_sols)
        h_qp = jnp.broadcast_to(meas ** (1.0 / dim), (qw_shared.shape[0], 1))
        loc = {
            "physical_quad_points": xq,
            "fields": per,
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": {**ctx, "cell_size": h_qp},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": (),
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype),
            "trial_value_shape": fields[0]["value_shape"],
            "trial_vec": vecs[0],
        }
        if _field_param_names:
            loc["shape_vals"] = per[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if history_specs and args is not None:
            hbuf = args.get("__history__") if isinstance(args, dict) else None
            if hbuf:
                loc["qp_history"] = {k: hbuf[k][c] for k in history_specs if k in hbuf}
        _add_loadpath_fields(loc, c, args)  # per-step load-path field slices -> loc["frozen_fields"]
        return _eval_integrand(domain, formula, loc)

    def state_readout(u_flat, t=0.0, args=None):
        """Advance every buffered state one load step: evaluate each key's readout formula at the
        quadrature points, given the just-solved ``u_flat`` and the current history buffers (on
        ``args['__history__']``). Returns ``{history_key: (n_cells, n_quad, *value_shape)}`` — the value
        that becomes each state's ``.i(-1)`` at the NEXT step. The load-step march rolls these into the
        depth buffers. Whole-domain: the readout runs on every cell (sub-region-restricted plasticity is
        not wired — a future masked readout)."""
        local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)
        out: Dict[Any, Any] = {}
        for key, formula in readout_formulas.items():
            if key not in history_specs:  # VOLUME states only; surface states advance in surface_state_readout
                continue
            out[key] = jax.vmap(lambda c, la, _f=formula: _vol_elem_readout(c, la, _f, t, args))(
                jnp.arange(n_cells), local_all
            )
        return out

    def _surf_elem_res(fi, local_all, bcoeff, btfi, region, t=0.0, args=None, pts=None, normals=None):
        """Element residual of one surface term on boundary face ``fi`` as a function of the parent
        cell's gathered all-field local DOFs ``local_all`` -> ``(n_test_dofs_btfi,)``. ``pts`` / ``normals``
        are the coordinate-parameter-scattered geometry and its facet normals (``None`` -> static mesh)."""
        c = parent_j[fi]
        k = lface_j[fi]
        n_vec = (normals_j if normals is None else normals)[fi]  # (dim,) outward unit normal
        cell_sols = _split_cell_local(local_all)
        verts = (pts_j if pts is None else pts)[cells_j[c]]
        J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim)
        K = jnp.linalg.inv(J)

        # All-field surface data (needed for coupled Robin terms).
        per_f = []
        for i in range(len(fields)):
            fp_i, fd_i, _, _, _ = face_tables_per_field[i]
            per_f.append(
                {
                    "shape_vals": fp_i[k],
                    "shape_grads": jnp.einsum("qnd,dD->qnD", fd_i[k], K),
                    "cell_sol": cell_sols[i],
                    "space": "Lagrange",
                }
            )

        _, _, fp_qp, fp_tangs, face_w = face_tables_per_field[btfi]
        jac_f = _facet_area_element(J, fp_tangs[k])  # physical edge length (2D) / face area (3D)
        xq_f = verts[0] + fp_qp[k] @ J.T  # (n_q, dim)
        loc = {
            "physical_quad_points": xq_f,
            "fields": per_f,
            "field_index": field_index,
            "tag": f"gauss_{region}",
            "surface": True,
            "domain_context": {**ctx, f"n_{region}": n_vec},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": (),
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype),
            "trial_value_shape": fields[btfi]["value_shape"],
            "trial_vec": vecs[btfi],
        }
        if _field_param_names:
            loc["shape_vals"] = per_f[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:  # known-field (ui.freeze) per-cell nodal slices for the parent cell
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if surface_history_specs and args is not None:
            # This face's per-quad-point surface-history slice (n_quad_surf, depth, *shape), gathered from
            # the buffers on ``args`` by the global boundary-face id -- a per-face constant, so ``jacfwd``
            # treats it as frozen (the tangent is ``∂t_fric/∂u`` with the slip history held, exactly like
            # the volume return map holds the plastic strain).
            sbuf = args.get("__surface_history__") if isinstance(args, dict) else None
            if sbuf:
                loc["qp_history"] = {k: sbuf[k][fi] for k in surface_history_specs if k in sbuf}
        return _integrate_term(domain, bcoeff, loc, face_w * jac_f)

    def _surf_elem_readout(fi, local_all, formula, region, t=0.0, args=None):
        """Per-quad-point VALUE of a surface evolution formula on boundary face ``fi`` -> (n_q, *shape).

        The surface analogue of :func:`_vol_elem_readout`: the same surface ``loc`` as ``_surf_elem_res``
        (fields, outward normal, per-face surface history), but the formula carries no test function, so it
        is *evaluated* (``_eval_integrand``), not integrated -- the advance for a surface state (a slip)."""
        c = parent_j[fi]
        k = lface_j[fi]
        n_vec = normals_j[fi]
        cell_sols = _split_cell_local(local_all)
        verts = pts_j[cells_j[c]]
        J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)
        Kmat = jnp.linalg.inv(J)
        per_f = []
        for i in range(len(fields)):
            fp_i, fd_i, _, _, _ = face_tables_per_field[i]
            per_f.append(
                {
                    "shape_vals": fp_i[k],
                    "shape_grads": jnp.einsum("qnd,dD->qnD", fd_i[k], Kmat),
                    "cell_sol": cell_sols[i],
                    "space": "Lagrange",
                }
            )
        _, _, fp_qp, _fp_tangs, _fw = face_tables_per_field[0]
        xq_f = verts[0] + fp_qp[k] @ J.T
        loc = {
            "physical_quad_points": xq_f,
            "fields": per_f,
            "field_index": field_index,
            "tag": f"gauss_{region}",
            "surface": True,
            "domain_context": {**ctx, f"n_{region}": n_vec},
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": (),
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype),
            "trial_value_shape": fields[0]["value_shape"],
            "trial_vec": vecs[0],
        }
        if _field_param_names:
            loc["shape_vals"] = per_f[_field_param_field_idx]["shape_vals"]
        _nt = neural_local_table(_neural, args)
        if _nt is not None:
            loc["neural_coefficients"] = _nt
        if _frozen_gathered:
            loc["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
        if surface_history_specs and args is not None:
            sbuf = args.get("__surface_history__") if isinstance(args, dict) else None
            if sbuf:
                loc["qp_history"] = {kk: sbuf[kk][fi] for kk in surface_history_specs if kk in sbuf}
        return _eval_integrand(domain, formula, loc)

    def surface_state_readout(u_flat, t=0.0, args=None):
        """Advance each SURFACE state one load step: evaluate its evolves formula on its region's faces.

        Returns ``{key: (n_bfaces, n_quad_surf, *value_shape)}`` -- the region's faces filled, every other
        boundary face zero (unused). The march rolls these into the surface depth buffers."""
        out: Dict[Any, Any] = {}
        for key, spec in surface_history_specs.items():
            formula = readout_formulas.get(key)
            region = _surf_read_regions[key]
            faces = _surf_region_faces.get(region)
            full = jnp.zeros(
                (int(spec["shape"][0]), int(spec["shape"][1])) + tuple(spec["value_shape"]), dtype=u_flat.dtype
            )
            if formula is None or faces is None or len(faces) == 0:
                out[key] = full
                continue
            fids = jnp.asarray(faces, dtype=jnp.int32)
            lv = u_flat[cell_all_dofs[parent_j[fids]]]  # (n_face_R, n_local_all)
            vals = jax.vmap(lambda fi, la, _f=formula, _r=region: _surf_elem_readout(fi, la, _f, _r, t, args))(fids, lv)
            # Normalize to the state's declared per-face shape (n_faces, n_quad_surf, *value_shape): a
            # scalar update written with `inner(dir, u.bind(...), 1)` keeps a spurious trailing size-1 axis
            # (harmless in the residual, where it contracts with the test) that must be squeezed here.
            vals = vals.reshape((fids.shape[0], int(spec["shape"][1])) + tuple(spec["value_shape"]))
            out[key] = full.at[fids].set(vals)
        return out

    def _classify_one(coeff, where: str) -> List[Tuple[Any, int]]:
        """``[(coeff, test_field_idx), ...]`` for one lowered term. Normally one entry; a term that
        welds several test fields inside a product (the real part of a ``complex=True`` form, e.g.
        ``c·(u_r·w_r − u_i·w_i)``) is distributed over its sums into single-test sub-terms, so one
        complex form lowers onto the coupled blocks."""
        from ...trace import BinaryOp, Literal
        from .fem_utils import _expand_product_terms

        tfi = _test_field_index(coeff, field_index)
        if tfi is not None:
            return [(coeff, tfi)]
        expanded = _expand_product_terms(coeff)
        if len(expanded) > 1:
            split: List[Tuple[Any, int]] = []
            for s, sub in expanded:
                sub_signed = sub if s >= 0 else BinaryOp("*", Literal(-1.0), sub)
                sfi = _test_field_index(sub_signed, field_index)
                if sfi is None:
                    split = None
                    break
                split.append((sub_signed, sfi))
            if split is not None:
                return split
        raise ValueError(
            f"jno.fem (native): each {where} weak-form term must contain exactly one test field "
            "(it determines the equation block)."
        )

    def _preprocess_terms(terms, bterms):
        """``(typed_with_masks, surface_work)``: lower each additive sub-term to
        ``(coeff, test_field_idx[, mask_names])`` and bucket boundary faces per region."""
        typed: List[Tuple[Any, int]] = []
        for bare in terms:
            for sign, sub in _split_additive_terms(domain, bare):
                coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                typed.extend(_classify_one(coeff, "volume"))
        typed_with_masks = [(coeff, tfi, tuple(sorted(_collect_region_mask_names(coeff)))) for coeff, tfi in typed]

        surface_work: List[Tuple[str, np.ndarray, List[Tuple[Any, int]]]] = []
        if bterms and conn.n_bfaces > 0:
            for region, bexprs in bterms.items():
                region_nodes = {int(n) for n in _region_node_ids(domain, region)}
                face_mask = np.array(
                    [
                        all(int(conn.face_nodes[fi, j]) in region_nodes for j in range(conn.face_nodes.shape[1]))
                        for fi in range(conn.n_bfaces)
                    ]
                )
                face_ids = np.where(face_mask)[0]
                if len(face_ids) == 0:
                    continue
                btyped = []
                for bexpr in bexprs:
                    for sign, sub in _split_additive_terms(domain, bexpr):
                        bcoeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                        btyped.extend(_classify_one(bcoeff, f"boundary ({region!r})"))
                surface_work.append((region, np.asarray(face_ids, dtype=np.int32), btyped))
        return typed_with_masks, surface_work

    # --- element-loop chunking -----------------------------------------------------------------
    # A single `vmap` over every cell materialises the whole batched intermediate at once, and on a 3-D
    # mesh that intermediate -- not the assembled operator -- is what sets the memory ceiling. Measured
    # on a 31k-DOF 3-D nonlinear problem: the jacfwd tangent tensor `f64[n_cells, 4, 4, 4]` was 82.2 MiB
    # against a 6.8 MiB operator, and the residual's own temp (182 MiB) is the unit every other cost is
    # a multiple of -- each Krylov matvec, and the linearization the matrix-free inner solve holds live.
    #
    # `lax.map(..., batch_size=C)` vmaps C cells at a time and scans over the chunks, so the batched
    # intermediate is capped at C cells regardless of mesh size. Remainders are handled (verified at
    # several non-dividing C) and gradients match the unchunked form exactly.
    # An explicit `jno.fem(chunk=...)` is captured HERE, once, rather than read when the closures run:
    # they are called at solve time, long outside the context that set it.
    _chunk_setting = _CHUNK_OVERRIDE[0]
    _CHUNK_CONSUMED[0] = True

    # Sized in CELLS, capped by a fraction of the DEVICE's memory. GPU saturation depends on how many
    # independent work items a chunk has, not on how many bytes it occupies, so a pure byte budget
    # starves the device as soon as the per-cell block grows (P2/P3, vector fields).
    #
    # Swept on an RTX 3070 at 97824 cells, measuring the full solve rather than the assembly alone:
    #
    #   cells/chunk   2048    4096    8192   16384   32768   unchunked
    #   solve peak   287.6   256.0   273.6   279.5   378.6     801.8 MiB
    #   jacobian      6.08    4.51    3.81    3.69    3.50      3.24 ms
    #
    # Three things that sweep settles and reasoning would not: the cliff is between "chunked at all"
    # and "not" (one chunk costs 802 MiB, any split more than halves it), so the exact size matters far
    # less than whether it splits; the peak is NOT monotonic in chunk size (2048 is worse than 4096),
    # so "smaller is safer" is the wrong instinct; and below ~8k cells the device runs dry and assembly
    # nearly doubles, while above ~16k the extra memory buys almost no speed.
    #
    # The cap is expressed RELATIVE TO THE DEVICE so it is not tuned to one machine: a chunk may use
    # ~0.15% of device memory, which reproduces the measured optimum here (0.15% of 5.7 GiB = 8.8 MiB
    # = 17.9k P1 cells) and scales on its own to a larger card, which has both more memory to spend and
    # more cores to feed.
    #
    # The saturation FLOOR is the one number that cannot be derived: JAX exposes device memory
    # (`bytes_limit`) but not the SM/core count, so there is nothing portable to compute it from. It is
    # therefore set conservatively LOW, where it binds only for large per-cell blocks -- and when it
    # binds it deliberately overruns the memory cap, because the measured alternative is a ~2x
    # slowdown. A bigger card would want a higher floor, but on a bigger card the memory-derived cap is
    # already well above it, so the floor stops mattering exactly where it would have been wrong.
    _CHUNK_MEMORY_FRACTION = 0.0015
    _CHUNK_MIN_CELLS = 8192  # saturation floor; see above -- not derivable, so kept low on purpose
    _CHUNK_FALLBACK_BYTES = 8 << 20  # CPU / unknown device: no saturation pressure, just bound memory

    def _chunk_budget_bytes():
        """Bytes one element chunk may occupy, taken from the device rather than tuned to one."""
        try:
            limit = jax.local_devices()[0].memory_stats().get("bytes_limit")
        except Exception:  # noqa: BLE001 -- CPU backends expose no memory stats
            limit = None
        if not limit:
            return _CHUNK_FALLBACK_BYTES
        return max(_CHUNK_FALLBACK_BYTES // 2, int(limit * _CHUNK_MEMORY_FRACTION))

    def _cell_chunk(n_items: int, n_test: int, n_local: int):
        """Cells per chunk, or ``None`` to keep the plain single `vmap`.

        The per-cell cost is the element block *including its AD tangent* (`n_test * n_local**2`), the
        jacobian's dominant intermediate. The residual's is smaller, so it gets chunked somewhat more
        finely than it strictly needs -- a deliberate simplification: one policy, one explanation, and
        the cost of an extra chunk is a scan step, not a re-computation."""
        if _chunk_setting == 0:
            return None  # explicitly disabled: one vmap over every cell
        if _chunk_setting is not None:
            return None if n_items <= _chunk_setting else int(_chunk_setting)  # explicit cells/chunk
        per = max(1, int(n_test) * int(n_local) * int(n_local) * 8)
        chunk = max(1, _chunk_budget_bytes() // per)
        chunk = max(chunk, _CHUNK_MIN_CELLS)  # never starve the device to honour the byte cap
        if n_items <= chunk:
            return None  # one chunk anyway -- skip the scan overhead entirely
        return int(chunk)

    def _elem_map(fn, xs, chunk):
        """``vmap(fn)`` over the leading axis, in chunks of ``chunk`` when one is set."""
        if chunk is None:
            return jax.vmap(fn)(*xs)
        return jax.lax.map(lambda z: fn(*z), xs, batch_size=int(chunk))

    def _make_residual(terms, bterms=None):
        """Build the free global residual ``R(u_flat) -> (total,)`` (volume + optional surface).

        ``bterms`` is an optional ``{region: [exprs]}`` dict for surface (Neumann/Robin) terms; pass
        ``None`` (the default) to assemble volume terms only — used for the transient mass matrix,
        where boundary contributions must not appear."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        def residual(u_flat, t=0.0, args=None):
            R = jnp.zeros(total, dtype=u_flat.dtype)
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)
            pts_dyn = _apply_coord_params(pts_j, args)  # trainable coords -> differentiable geometry

            for coeff, tfi, rnames in typed_with_masks:
                elem = _elem_map(
                    lambda c, la, _e=coeff, _t=tfi, _r=rnames: _vol_elem_res(c, la, _e, _t, _r, t, args, pts_dyn),
                    (jnp.arange(n_cells), local_all),
                    _cell_chunk(n_cells, cdofs[tfi].shape[1], cell_all_dofs.shape[1]),
                )
                R = R.at[cdofs[tfi].reshape(-1)].add(elem.reshape(-1))

            normals_dyn = _surface_normals(pts_dyn)  # differentiable facet normals under coordinate motion
            for region, face_ids, btyped in surface_work:
                fids = jnp.asarray(face_ids, dtype=jnp.int32)
                pcells = parent_j[fids]
                lv = u_flat[cell_all_dofs[pcells]]  # (n_face, n_local_all)
                for bcoeff, btfi in btyped:
                    contribs = _elem_map(
                        lambda fi, la, _e=bcoeff, _t=btfi, _r=region: _surf_elem_res(
                            fi, la, _e, _t, _r, t, args, pts_dyn, normals_dyn
                        ),
                        (fids, lv),
                        _cell_chunk(int(fids.shape[0]), cdofs[btfi].shape[1], cell_all_dofs.shape[1]),
                    )
                    R = R.at[cdofs[btfi][pcells].reshape(-1)].add(contribs.reshape(-1))
            return R

        return residual

    def _make_jacobian(terms, bterms=None):
        """Build the dense Jacobian ``J(u_flat) -> (total, total)`` by *per-element* forward-mode AD.

        Each cell's (and boundary face's) element matrix is ``jacfwd`` of its element residual w.r.t.
        that element's local DOFs — an ``(n_test, n_local)`` block — then scatter-added into the global
        matrix. The AD never sees the global state, so the intermediate is element-sized; this is what
        a single global ``jacfwd(residual)`` cannot do (it materialises an ``O(n_dofs × n_cells)``
        tangent tensor and OOMs on any non-trivial mesh). The dense result is entry-for-entry
        identical to that global ``jacfwd``, just assembled within a per-element memory budget."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        # --- hoist the (row, col) pattern out of the trace -------------------------------------
        # The triplet INDICES come exclusively from host-static mesh connectivity (`cdofs`,
        # `cell_all_dofs`, `parent_j`, `face_ids`) and the term list; only the element blocks `Ke`
        # depend on `u_flat`/`t`/`args`. So the pattern -- and therefore the compressed nonzero count
        # -- is the same at every state, time and parameter value this is ever traced at.
        #
        # Building it once here is what lets the TRACED assemblies compress: `sum_duplicates` needs a
        # static `nse` under jit, and inferring one requires concrete indices it does not have inside
        # the trace. This mirrors `fem_nonnodal._make_sparse_assembler`, which already hoists its
        # `_vol_idx` the same way. Order must match the append order in `jacobian` exactly.
        _idx_rows, _idx_cols = [], []
        for _coeff_s, _tfi_s, _rn_s in typed_with_masks:
            _sh = (n_cells, int(cdofs[_tfi_s].shape[1]), int(cell_all_dofs.shape[1]))
            _idx_rows.append(jnp.broadcast_to(cdofs[_tfi_s][:, :, None], _sh).reshape(-1))
            _idx_cols.append(jnp.broadcast_to(cell_all_dofs[:, None, :], _sh).reshape(-1))
        for _region_s, _face_ids_s, _btyped_s in surface_work:
            _pc = parent_j[jnp.asarray(_face_ids_s, dtype=jnp.int32)]
            _fcols = cell_all_dofs[_pc]
            for _bcoeff_s, _btfi_s in _btyped_s:
                _sh = (int(_pc.shape[0]), int(cdofs[_btfi_s].shape[1]), int(cell_all_dofs.shape[1]))
                _idx_rows.append(jnp.broadcast_to(cdofs[_btfi_s][_pc][:, :, None], _sh).reshape(-1))
                _idx_cols.append(jnp.broadcast_to(_fcols[:, None, :], _sh).reshape(-1))
        _blk_sizes = [int(r.shape[0]) for r in _idx_rows]  # per-term flat lengths, in append order
        _idx_static = (
            jnp.stack([jnp.concatenate(_idx_rows).astype(jnp.int32), jnp.concatenate(_idx_cols).astype(jnp.int32)], axis=1)
            if _idx_rows
            else None
        )
        try:
            _plan = compress_plan(_idx_static) if _idx_static is not None else None
        except Exception:  # noqa: BLE001 -- a traced pattern would break the static-count invariant
            _idx_static, _plan = None, None  # fall back to the uncompressed (still correct) path

        def jacobian(u_flat, t=0.0, args=None):
            # Assemble into COO triplets and build a BCOO -- never materialises the dense (total, total)
            # matrix (O(nnz), GPU-able at large N). Each per-element block is element-sized; duplicate
            # (i, j) triplets from neighbouring cells are summed by BCOO on matvec / todense, so the
            # per-cell blocks are simply concatenated (no pre-summation).
            # With a plan in force each element block is scattered STRAIGHT into its compressed slots
            # and then dropped, so the concatenated raw-triplet array is never built. That array and
            # the transposed copy XLA made of it were the two largest buffers in the compiled jacobian
            # after the element blocks themselves (61.6 MiB each on a 31k-DOF 3-D problem, against a
            # 6.8 MiB operator), and every per-term `Ke` had to stay alive waiting for the concatenate.
            # The row/column arrays are skipped for the same reason: the plan already has the pattern.
            _acc = [None]
            _off = [0]
            _nblk = [0]
            rows_l, cols_l, data_l = [], [], []
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)
            pts_dyn = _apply_coord_params(pts_j, args)  # trainable coords -> differentiable geometry

            def _emit(flat, rows_fn, cols_fn):
                if _plan is None:
                    data_l.append(flat)
                    rows_l.append(rows_fn())
                    cols_l.append(cols_fn())
                    return
                _inv, _nse = _plan[1], _plan[2]
                k = _blk_sizes[_nblk[0]]
                part = jax.ops.segment_sum(flat, _inv[_off[0] : _off[0] + k], num_segments=_nse)
                _acc[0] = part if _acc[0] is None else _acc[0] + part
                _off[0] += k
                _nblk[0] += 1

            for coeff, tfi, rnames in typed_with_masks:

                def _ke(c, la, _e=coeff, _t=tfi, _r=rnames, _p=pts_dyn):
                    return jax.jacfwd(lambda v: _vol_elem_res(c, v, _e, _t, _r, t, args, _p))(la)

                Ke = _elem_map(  # (n_cell, n_test_tfi, n_local_all)
                    _ke,
                    (jnp.arange(n_cells), local_all),
                    _cell_chunk(n_cells, cdofs[tfi].shape[1], cell_all_dofs.shape[1]),
                )
                _emit(
                    Ke.reshape(-1),
                    lambda _K=Ke, _t=tfi: jnp.broadcast_to(cdofs[_t][:, :, None], _K.shape).reshape(-1),
                    lambda _K=Ke: jnp.broadcast_to(cell_all_dofs[:, None, :], _K.shape).reshape(-1),
                )

            normals_dyn = _surface_normals(pts_dyn)  # differentiable facet normals under coordinate motion
            for region, face_ids, btyped in surface_work:
                fids = jnp.asarray(face_ids, dtype=jnp.int32)
                pcells = parent_j[fids]
                lv = u_flat[cell_all_dofs[pcells]]  # (n_face, n_local_all)
                fcols = cell_all_dofs[pcells]  # (n_face, n_local_all)
                for bcoeff, btfi in btyped:

                    def _kef(fi, la, _e=bcoeff, _t=btfi, _r=region, _p=pts_dyn, _n=normals_dyn):
                        return jax.jacfwd(lambda v: _surf_elem_res(fi, v, _e, _t, _r, t, args, _p, _n))(la)

                    Kef = _elem_map(  # (n_face, n_test_btfi, n_local_all)
                        _kef,
                        (fids, lv),
                        _cell_chunk(int(fids.shape[0]), cdofs[btfi].shape[1], cell_all_dofs.shape[1]),
                    )
                    _emit(
                        Kef.reshape(-1),
                        lambda _K=Kef, _t=btfi, _p=pcells: jnp.broadcast_to(cdofs[_t][_p][:, :, None], _K.shape).reshape(
                            -1
                        ),
                        lambda _K=Kef, _f=fcols: jnp.broadcast_to(_f[:, None, :], _K.shape).reshape(-1),
                    )

            if _plan is not None:
                if _acc[0] is None:  # no terms -> empty operator
                    return jsparse.BCOO((jnp.zeros((0,), u_flat.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(total, total))
                # ~12x fewer stored triplets, INSIDE the trace -- so it reaches the nonlinear Jacobian
                # and the per-step/parametric re-assemblies, not just eager assembly. The host-decided
                # plan makes this an O(nnz) scatter-add rather than an O(nnz log nnz) sort, which
                # matters because it runs once per Newton step / timestep / parameter value.
                return jsparse.BCOO((_acc[0], _plan[0]), shape=(total, total))
            if not data_l:  # no terms -> empty operator
                return jsparse.BCOO((jnp.zeros((0,), u_flat.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(total, total))
            data = jnp.concatenate(data_l)
            if _idx_static is not None:
                idx = _idx_static
            else:  # pattern could not be hoisted -> rebuild it in-trace, uncompressed but correct
                idx = jnp.stack(
                    [jnp.concatenate(rows_l).astype(jnp.int32), jnp.concatenate(cols_l).astype(jnp.int32)], axis=1
                )
            return jsparse.BCOO((data, idx), shape=(total, total))

        # Published so a wrapper that APPENDS triplets (the Dirichlet row-replacement below) can
        # derive its own plan from the same pattern instead of re-deriving one that might disagree.
        # It must describe what `jacobian` ACTUALLY RETURNS -- the COMPRESSED pattern when a plan is
        # in force, not the raw one it was derived from. Publishing the raw pattern here silently
        # mismatched the wrapper's `inverse` against the compressed data length: the recurring shape
        # of bug in this repo is a representation changing while one of its readers does not move.
        jacobian._jno_static_idx = _plan[0] if _plan is not None else _idx_static  # type: ignore[attr-defined]
        return jacobian

    def _dirichlet_jac_rows(jac_fn, pairs):
        """Wrap an assembled-Jacobian callable so Dirichlet rows become the identity row — the
        matrix-level analogue of :func:`_apply_dirichlet_rows` (row-replacement, columns kept), so it
        matches ``jacfwd`` of the row-replaced residual that the Newton step expects."""
        if not pairs:
            return jac_fn
        dofs = jnp.asarray([p[0] for p in pairs], dtype=jnp.int32)

        # `bcoo_set_dirichlet_rows` zeroes the constrained rows and then APPENDS one (d, d, 1) triplet
        # per constrained DOF, so its output carries up to `len(dofs)` duplicates however well the
        # inner Jacobian was compressed. That count is static too: the union of the inner pattern with
        # the Dirichlet diagonal. Derived from the inner assembler's own published pattern so the two
        # cannot disagree; without it, fall back to leaving the appended duplicates in place.
        _inner_idx = getattr(jac_fn, "_jno_static_idx", None)
        _dir_plan = None
        if _inner_idx is not None:
            try:
                _d_np = np.asarray(dofs, dtype=np.int64).reshape(-1)
                _dir_plan = compress_plan(
                    np.concatenate([np.asarray(_inner_idx), np.stack([_d_np, _d_np], axis=1)], axis=0)
                )
            except Exception:  # noqa: BLE001 -- no static plan available; correctness is unaffected
                _dir_plan = None

        def jac(u_flat):
            A = bcoo_set_dirichlet_rows(jac_fn(jnp.asarray(u_flat)), dofs)
            # Same host-decided plan, so this is an O(nnz) scatter-add per Newton step, not a sort.
            return apply_compress_plan(A.data, _dir_plan, A.shape) if _dir_plan is not None else A

        return jac

    # -------------------------------------------------------------------------
    # Dirichlet pair builder
    # -------------------------------------------------------------------------

    def _boundary_node_ids(fidx: int, region: str) -> List[int]:
        """Robust boundary-DOF-node ids of ``region`` for field ``fidx``.

        Boundary nodes are taken from the assembly mesh's boundary FACETS (a node on a boundary
        facet -- P2 edge-midpoints attached by coordinate), not a geometric containment test: the
        latter can miss a P2 midpoint sitting exactly on a face (the discrete proximity test catches
        the P1 vertices but not the new midpoint). The catch-all ``"boundary"`` region is every
        boundary-facet node; a named region filters those by its spatial predicate (exact even for an
        on-face midpoint) or, lacking one, by geometric containment -- but only ever among true
        boundary nodes, so an on-boundary node is never lost to a flaky test. Falls back to the
        plain predicate-over-all-nodes finder when there are no facets (degenerate mesh) or the
        boundary set is empty (e.g. an interior pin), which keeps interior Dirichlet points working.
        """
        from ..._fem import _boundary_facets

        pts_all = np.asarray(pts_f_all[fidx])
        # A named interior SUB-REGION (`domain.region(name, poly)`) pins its WHOLE node set (interior +
        # boundary), by point-in-polygon — not just its boundary nodes (which is empty for an interior
        # region and would silently drop the pin). This is the subdomain / domain-decomposition pin.
        ptags = getattr(domain, "_polygon_tags", {})
        if region in (getattr(domain, "_source_regions", {}) or {}) and ptags.get(region, (None,))[0] == "interior":
            return list(_region_node_ids_from_pts(domain, region, pts_all))
        bf = _boundary_facets(pts_all, np.asarray(cells_f_all[fidx]), dim, fields[fidx]["order"])
        if bf is None:
            return list(_region_node_ids_from_pts(domain, region, pts_all))
        bnodes = np.unique(np.asarray(bf).reshape(-1))
        if region != "boundary":
            coords = pts_all[bnodes]
            pred = getattr(domain, "_tag_predicates", {}).get(region)
            if pred is not None:
                mask = np.asarray(pred(*(coords[:, i] for i in range(dim))), dtype=bool).reshape(-1)
            else:
                loc = domain._make_tag_location_fn(region)
                if loc is None:
                    return []
                mask = np.asarray(jax.vmap(loc)(jnp.asarray(coords)), dtype=bool).reshape(-1)
            bnodes = bnodes[mask]
        if bnodes.size == 0:  # interior pin (no boundary facet matched) -> predicate over all nodes
            return list(_region_node_ids_from_pts(domain, region, pts_all))
        return [int(n) for n in bnodes]

    def _build_dirichlet_pairs() -> List[Tuple[int, float]]:
        from ..._fem import _eval_value_node_at, _is_temporal_value_node
        from ...trace import ModelCall

        pairs: List[Tuple[int, float]] = []
        tv_stash: List[Tuple[Any, Any, Any]] = []  # (dofs, value_node, coords) for time-varying g(x,t)
        for field_key, region, comp, value, value_node in dirichlet_raw:
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            # Time-varying Dirichlet g(x,t): no constant pair — stash (dofs, value_node, coords) so a
            # transient caller (e.g. the second-order augmented block) writes g(x_d, t) each step.
            if value_node is not None and _is_temporal_value_node(value_node):
                vt = vecs[fidx]
                pts_all = np.asarray(pts_f_all[fidx])
                nids = list(_boundary_node_ids(fidx, region))
                coords = jnp.asarray(pts_all[np.asarray(nids, dtype=int)]) if nids else jnp.zeros((0, dim))
                for c in range(vt) if comp is None else [int(comp)]:
                    dofs = jnp.asarray([offs[fidx] + nid * vt + c for nid in nids], dtype=jnp.int32)
                    tv_stash.append((dofs, value_node, coords))
                continue
            _vn = _bare_node(value_node) if value_node is not None else None
            # A nodal DATA-field value (a `jno.np.parameter` carrying a field with NO optimizer — e.g. a
            # neighbour's field in a coupled/domain-decomposition solve) → gather its per-node values by
            # node index. Checked before the neural-coefficient branch so a bare data-field is a value,
            # not a runtime net profile.
            _field_vals = None
            if (
                isinstance(_vn, ModelCall)
                and getattr(_vn.model, "_is_parameter", False)
                and getattr(_vn.model, "_opt_fn", None) is None
            ):
                _field_vals = np.asarray(_vn.model.module.value).reshape(-1)
            elif _vn is not None and _is_neural_coefficient(_vn):
                continue  # a net-valued Dirichlet is (re-)built per args in _dirichlet_pairs_at
            vt = vecs[fidx]
            pts_all = pts_f_all[fidx]
            for nid in _boundary_node_ids(fidx, region):
                p = np.asarray(pts_all[nid])
                if _field_vals is not None:
                    g = float(_field_vals[nid])
                elif value_node is not None:
                    raw = _eval_value_node_at(value_node, jnp.asarray(p)[None])
                    g = float(jnp.asarray(raw).reshape(-1)[0])
                elif callable(value):
                    g = float(value(p))
                else:
                    g = float(value)
                comps_range = range(vt) if comp is None else [int(comp)]
                for c in comps_range:
                    pairs.append((offs[fidx] + nid * vt + c, g))
        # Expose the (dof, value) pairs for callers that compose their own system from native blocks
        # (e.g. the second-order-in-time augmented [u, v] block applies them to the 2N system itself).
        # The time-varying entries ride a companion stash: the caller writes g(x_d, t) (and, for a
        # second-order block, the velocity ġ) per step.
        domain._fem_native_dirichlet_pairs = pairs
        domain._fem_native_dirichlet_tv = tv_stash
        return pairs

    def _dirichlet_pairs_at(args):
        """Dirichlet ``(dof, value)`` pairs with the net-valued profiles evaluated from the runtime
        ``args`` (an unknown BC ``u(region) - net(x)``): the net is called on the region's boundary-node
        coordinates, so the value stays a differentiable JAX scalar and ``∂b/∂weights`` flows through the
        symmetric elimination. Non-net conditions reuse the concrete ``_build_dirichlet_pairs`` values."""
        a = args or {}
        pairs = list(_build_dirichlet_pairs())  # concrete (non-net) conditions
        for field_key, region, comp, value, value_node in dirichlet_raw:
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            _vn = _bare_node(value_node) if value_node is not None else None
            if _vn is None or not _is_neural_coefficient(_vn):
                continue
            vt = vecs[fidx]
            pts_all = np.asarray(pts_f_all[fidx])
            node_ids = _boundary_node_ids(fidx, region)
            module = a.get(_neural_coefficient_name(_vn), _vn.model.module)
            coords = jnp.asarray(pts_all[np.asarray(node_ids, dtype=np.int64)])  # (n_bnodes, dim)
            n_in = len(_vn.args)  # net(xb, yb[, zb]) -> per-coordinate columns (foundax MLP arity)
            gvals = jnp.asarray(module(*[coords[:, i : i + 1] for i in range(n_in)])).reshape(-1)
            for i, nid in enumerate(node_ids):
                for c in range(vt) if comp is None else [int(comp)]:
                    pairs.append((offs[fidx] + nid * vt + c, gvals[i]))
        return pairs

    def _build_dirichlet_tv_entries():
        """Time-varying Dirichlet ``g(x, t)`` entries: a list of ``(dofs, value_node, coords)`` for each
        ``dirichlet_raw`` whose value carries the temporal variable. The transient block evaluates
        ``g`` at ``coords`` and time ``t`` each step (``_eval_value_node_at_time``) and writes it onto
        ``dofs`` in the forcing. The constant-valued conditions are returned separately as ordinary
        pairs (their ``t``-independent value goes in the affine bias)."""
        from ..._fem import _eval_value_node_at, _is_temporal_value_node

        const_pairs: List[Tuple[int, float]] = []
        tv_entries: List[Tuple[Any, Any, Any]] = []
        for field_key, region, comp, value, value_node in dirichlet_raw:
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            vt = vecs[fidx]
            pts_all = np.asarray(pts_f_all[fidx])
            nids = _boundary_node_ids(fidx, region)
            comps_range = range(vt) if comp is None else [int(comp)]
            if value_node is not None and _is_temporal_value_node(value_node):
                coords = jnp.asarray(pts_all[np.asarray(nids, dtype=int)]) if nids else jnp.zeros((0, dim))
                for c in comps_range:
                    dofs = jnp.asarray([offs[fidx] + nid * vt + c for nid in nids], dtype=jnp.int32)
                    tv_entries.append((dofs, value_node, coords))
                continue
            for nid in nids:
                p = pts_all[nid]
                if value_node is not None:
                    g = float(jnp.asarray(_eval_value_node_at(value_node, jnp.asarray(p)[None])).reshape(-1)[0])
                elif callable(value):
                    g = float(value(p))
                else:
                    g = float(value)
                for c in comps_range:
                    const_pairs.append((offs[fidx] + nid * vt + c, g))
        return const_pairs, tv_entries

    # -------------------------------------------------------------------------
    # Mode detection
    # -------------------------------------------------------------------------

    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]
    zeros = jnp.zeros(total)

    # === transient (Mu̇ + Au = c or M u̇ + R(u) = 0) ===
    if ic_residuals or any(_contains_temporal_derivative(t) for t in all_terms):
        from ..._fem import _bare, _essential_spec, _eval_value_node_at, _field_key_of
        from .backend_blocks import SemidiscreteTimeBlock
        from .time_route import (
            _infer_time_window,
            _replace_temporal_with_backward_euler,
            _strip_temporal_trial_derivative,
        )

        sub_signed = [
            _apply_sign(domain, sign, sub) for bare in volume_terms for sign, sub in _split_additive_terms(domain, bare)
        ]
        temporal = [t for t in sub_signed if _contains_temporal_derivative(t)]
        spatial = [t for t in sub_signed if not _contains_temporal_derivative(t)]
        if not temporal:
            raise ValueError(
                "jno.fem (native): an initial condition was provided but no temporal term "
                "(e.g. ``inner(u.t, v)``) was found in the volume weak form."
            )
        # A trainable net on the u̇ term. A COORDINATE ``net(x)`` (an unknown density ``rho(x)*u_t``) keeps
        # the mass a *matrix* -- just parametric in the weights -- so it is re-assembled from ``args`` each
        # step (``mass_fn`` below). A SOLUTION-DEPENDENT ``net(u)`` would make the mass itself nonlinear
        # (``C(u)*u_t``), which the semidiscrete matrix form cannot express -- reject that.
        _parametric_mass = collect_neural_slots(temporal).any_trainable
        if _parametric_mass and any(_is_obviously_nonlinear_in_unknown(domain, t) for t in temporal):
            raise NotImplementedError(
                "jno.fem (native): a solution-dependent neural coefficient net(u) on the mass (u_t) term is a "
                "nonlinear mass C(u)*u_t, which the semidiscrete matrix form cannot express. A coordinate "
                "net(x) mass coefficient (an unknown density) is supported."
            )

        mass_terms = [_strip_temporal_trial_derivative(t) for t in temporal]
        # Mass matrix: volume only (no boundary); spatial residual: volume + boundary
        _mass_jac = _make_jacobian(mass_terms)
        M = _mass_jac(zeros)
        spatial_res = _make_residual(spatial, boundary_terms)
        spatial_jac = _make_jacobian(spatial, boundary_terms)

        t0, t1, dt = _infer_time_window(domain)
        common = dict(
            backend="transient",
            mode="implicit",
            time_order=1,
            spatial_kind="weak_form",
            state0=None,
            t0=t0,
            t1=t1,
            dt=dt,
            eval_context=getattr(domain, "_fem_eval_context", {}) or {},
        )

        # --- initial state: nodal interpolation (exact for Lagrange). ``params`` re-forms a net-valued IC
        # ``u(initial) - net(x)`` from the runtime weights; ``None`` (no IC net) is byte-identical to the
        # old eager build. When an IC net is present the closure also rides the block as ``state0_fn`` so
        # ``∂traj/∂weights`` flows through the initial state. ---
        def _state0_at(params=None):
            s0 = zeros
            for ic in ic_residuals:
                comp, u0_node = _essential_spec(_bare(ic))
                fidx = field_index.get(_field_key_of(ic))
                if fidx is None:
                    raise ValueError("jno.fem (native): IC does not match any known trial field.")
                pts_ic = pts_f_all[fidx]  # (n_nodes_f[fidx], 2)
                nn, vv = n_nodes_f[fidx], vecs[fidx]
                raw = jnp.reshape(jnp.asarray(_eval_value_node_at(u0_node, jnp.asarray(pts_ic), params=params)), (-1,))
                if comp is not None:
                    # Per-component IC (e.g. ``u(initial)[0] - g0``): set just component ``comp`` at every
                    # node of the field. ``raw`` is the per-node value (or a single constant to broadcast).
                    vals = jnp.broadcast_to(raw, (nn,)) if raw.size == 1 else raw.reshape(nn)
                    idx = offs[fidx] + jnp.arange(nn) * vv + int(comp)
                    s0 = s0.at[idx].set(vals)
                else:
                    # Whole-field IC. A constant evaluates to a single value (no coordinate Variables to
                    # sample) -> broadcast to every node; a per-component constant broadcasts across nodes;
                    # otherwise it is the per-node field already.
                    if raw.size == 1:
                        u0_vals = jnp.full((nn, vv), raw[0])
                    elif raw.size == vv:
                        u0_vals = jnp.broadcast_to(raw[None, :], (nn, vv))
                    else:
                        u0_vals = raw.reshape(nn, vv)
                    s0 = s0.at[offs[fidx] : offs[fidx + 1]].set(u0_vals.reshape(-1))
            return s0

        common["state0"] = _state0_at({n: m.module for n, m in _ic_net_models.items()} if _ic_net_models else None)
        if _ic_net_models:
            common["state0_fn"] = _state0_at

        dirichlet_pairs = _build_dirichlet_pairs()
        d_dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
        d_vals = jnp.asarray([p[1] for p in dirichlet_pairs], dtype=zeros.dtype) if dirichlet_pairs else None

        # ---- STATE-DEPENDENT (nonlinear) MASS: ``c(u)·u_t`` with a coefficient depending on the unknown.
        # The fixed ``M = _mass_jac(zeros)`` freezes ``c`` at ``u=0`` (silently wrong; see jno-fem-hard-limits).
        # Reformulate each temporal term to backward-Euler *residual* form ``c(u)·(u − u_prev)·v`` — with
        # ``u_prev`` the previous step's nodal values delivered per step on the load-path channel — so the
        # ordinary residual/Jacobian assembly captures the exact mass action ``M(u)(u−u_prev)`` AND its exact
        # ``∂/∂u`` (both the ``M`` block and the ``∫c′(u)(u−u_prev)·v`` coefficient coupling). The ``1/dt``
        # factor is applied by the stepper. Backward Euler (θ=1) only; scalar fields only (load-path is scalar).
        _nonlinear_mass = (not _parametric_mass) and any(
            _is_obviously_nonlinear_in_unknown(domain, mt) for mt in mass_terms
        )
        prev_state_slices: List[Tuple[int, int, int, int]] = []  # (frozen_id, dof_start, dof_stop, n_components)
        mass_res_bc = mass_jac_bc = None
        if _nonlinear_mass:
            from ...trace import PrevStateField as _PrevStateField

            _prev_by_field: Dict[Any, Any] = {}

            def _prev_for(trial, _cache=_prev_by_field):
                fkey = trial.field_key
                pf = _cache.get(fkey)
                if pf is None:
                    fidx = field_index[fkey]
                    pf = _PrevStateField(trial)
                    _cache[fkey] = pf
                    # The prev-state field carries the source field's OWN key/basis, so it resolves the field's
                    # own shape data (P1 or P2, scalar or vector) — no P1 aliasing (unlike a load-path field).
                    _path_conn[pf.frozen_id] = cells_f_j[fidx]  # the field's own vertex connectivity
                    # (frozen_id, dof-slice into the flat state, n_components) — the step delivers this slice
                    # reshaped to (n_nodes, vec) on the load-path channel each backward-Euler step.
                    prev_state_slices.append((int(pf.frozen_id), int(offs[fidx]), int(offs[fidx + 1]), int(vecs[fidx])))
                return pf

            temporal_be = [_replace_temporal_with_backward_euler(t, _prev_for) for t in temporal]
            _mass_res_raw = _make_residual(temporal_be)  # ∫ c(u)·(u − u_prev)·v  (volume only; mass has no boundary)
            _mass_jac_raw = _make_jacobian(temporal_be)

            def mass_res_bc(u, t, args=None, _d=d_dofs, _f=_mass_res_raw):
                R = jnp.asarray(_f(jnp.asarray(u), t, args)).reshape(-1)
                return R if _d is None else R.at[_d].set(0.0)  # a constrained DOF carries no mass equation

            def mass_jac_bc(u, t, args=None, _d=d_dofs, _f=_mass_jac_raw):
                J = _f(jnp.asarray(u), t, args)
                return J if _d is None else bcoo_zero_rows(J, _d)

        # Parametric mass ``mass_fn(t, args)`` (unknown density net(x)*u_t): re-assemble M from args each
        # step with the Dirichlet rows/cols zeroed (a constrained DOF carries no time derivative). ``None``
        # keeps the static ``M_bc`` for a non-parametric mass.
        def _mass_cb(t, args=None, _d=d_dofs):
            Mt = _mass_jac(zeros, t, args)
            return Mt if _d is None else bcoo_zero_rows_cols(Mt, _d)

        # A mass-only nonlinearity (state-dependent mass) also requires the nonlinear step path, even when
        # every spatial term is linear — the mass action lives in the residual there (``mass_residual``).
        nonlinear = _nonlinear_mass or any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial)
        if nonlinear:
            if _ic_net_models:
                raise NotImplementedError(
                    "jno.fem: a net-valued initial condition u(initial) - net(x) on a *nonlinear transient* "
                    "form is not wired yet (state0_fn threads only the linear stepper). Use a linear "
                    "transient form (a net IC threads there)."
                )
            if _dir_net_models and _nonlinear_mass:
                raise NotImplementedError(
                    "jno.fem: a net-valued Dirichlet with a state-dependent (nonlinear) mass c(u)·u_t on a "
                    "transient form is not supported (the mass residual holds a static Dirichlet dof set). "
                    "Use a linear/parametric mass."
                )

            if _dir_net_models:
                # net-valued Dirichlet u(∂Ω) - net(x): the held value is re-formed from the net weights each
                # Newton residual (mirrors the nonlinear STEADY path ``res_p``); the dof set is static, only
                # the held values ride the weights, and ``∂/∂weights`` flows through the step's custom_root.
                _tnpd = jnp.asarray(
                    [p[0] for p in _dirichlet_pairs_at({n: m.module for n, m in _dir_net_models.items()})],
                    dtype=jnp.int32,
                )

                def _tnp_hold(args):
                    return jnp.stack([jnp.asarray(p[1]).reshape(()) for p in _dirichlet_pairs_at(args)])

                def res_bc(u, t, args=None, _d=_tnpd):
                    R = spatial_res(jnp.asarray(u), t, args)
                    return R.at[_d].set(jnp.asarray(u)[_d] - _tnp_hold(args))

                def jac_bc(u, t, args=None, _d=_tnpd):
                    return bcoo_set_dirichlet_rows(spatial_jac(jnp.asarray(u), t, args), _d)

                _mdofs = _tnpd
            else:
                # Row-replacement Dirichlet (constant g), threaded through the runtime time t AND the
                # runtime args so a time-dependent / parametric spatial coefficient is re-evaluated each step.
                def res_bc(u, t, args=None, _d=d_dofs, _g=d_vals):
                    R = spatial_res(jnp.asarray(u), t, args)
                    return R if _d is None else R.at[_d].set(jnp.asarray(u)[_d] - _g)

                def jac_bc(u, t, args=None, _d=d_dofs):
                    J = spatial_jac(jnp.asarray(u), t, args)
                    return J if _d is None else bcoo_set_dirichlet_rows(J, _d)

                _mdofs = d_dofs

            M_bc = M if _mdofs is None else bcoo_zero_rows_cols(M, _mdofs)
            return (
                SemidiscreteTimeBlock(
                    # A state-dependent mass carries no fixed matrix; the mass action is in mass_residual.
                    mass=None
                    if _nonlinear_mass
                    else (_mass_cb if _parametric_mass else (lambda t, args=None, _M=M_bc: _M)),
                    mass_residual=mass_res_bc,
                    mass_residual_jac=mass_jac_bc,
                    residual=res_bc,
                    jacobian=jac_bc,
                    runtime_parameter_exprs=dict(_param_and_neural_exprs),
                    metadata={"prev_state_slices": prev_state_slices} if _nonlinear_mass else {},
                    **common,
                ),
                "transient",
                offs,
            )

        # ---- linear parametric transient: the operator A(t, args) is re-evaluated each step.
        # Row-replacement Dirichlet (rows -> identity, columns kept) needs no args-dependent lift for a
        # CONSTANT g -- the held value sits in the affine bias. A net-valued Dirichlet u(∂Ω)-net(x) has an
        # args-dependent held value (differentiable in the weights): its whole held vector rides the
        # forcing each step instead (mirrors the g(x,t) path), so the constant bias drops to zero. ----
        if runtime_parameter_tags or neural_param_names or _dir_net_models or _ic_net_models:
            if _dir_net_models:
                if getattr(domain, "_fem_native_dirichlet_tv", None):
                    raise NotImplementedError(
                        "jno.fem: a net-valued Dirichlet combined with a time-varying g(x, t) Dirichlet on a "
                        "transient form is not supported yet (the net value rides the forcing; the g(x, t) lift "
                        "needs the temporal evaluator on those same rows). Use one or the other."
                    )
                # const + net Dirichlet dofs (static boundary-node layout); held values re-formed from args.
                _dd = jnp.asarray(
                    [p[0] for p in _dirichlet_pairs_at({n: m.module for n, m in _dir_net_models.items()})],
                    dtype=jnp.int32,
                )

                def _dhold(args):  # held value on every Dirichlet dof (net entries live in the weights)
                    return jnp.stack([jnp.asarray(p[1]).reshape(()) for p in _dirichlet_pairs_at(args)])
            else:
                _dd = d_dofs
            M_bc = M if _dd is None else bcoo_zero_rows_cols(M, _dd)
            free_mask = jnp.ones((total,), dtype=zeros.dtype)
            if _dd is not None:
                free_mask = free_mask.at[_dd].set(0.0)

            def operator_fn(t, args=None, _d=_dd):
                A = spatial_jac(zeros, t, args)
                return A if _d is None else bcoo_set_dirichlet_rows(A, _d)

            if _dir_net_models:
                c_bias = zeros  # every held value (const + net) rides the forcing

                def forcing_vector_fn(t, args=None, _mask=free_mask, _d=_dd):
                    f = _mask * (-spatial_res(zeros, t, args))
                    return f.at[_d].set(_dhold(args))
            else:
                c_bias = zeros if d_dofs is None else zeros.at[d_dofs].set(d_vals)

                def forcing_vector_fn(t, args=None, _mask=free_mask):
                    return _mask * (-spatial_res(zeros, t, args))

            return (
                SemidiscreteTimeBlock(
                    M=M_bc,
                    mass_fn=_mass_cb if _parametric_mass else None,  # parametric mass (unknown density net(x)*u_t)
                    operator_fn=operator_fn,
                    affine_bias=c_bias,
                    forcing_vector_fn=forcing_vector_fn,
                    runtime_parameter_exprs=dict(_param_and_neural_exprs),
                    # The operator is re-assembled at each (t, args) -- a general (non-affine) operator,
                    # so it covers a parameter inside a nonlinear coefficient (e.g. exp(logk)) too.
                    metadata={
                        "runtime_parameter_names": list(runtime_parameter_tags)
                        + list(neural_param_names)
                        + list(_dir_net_models)
                        + list(_ic_net_models),
                        "nonaffine_operator": True,
                    },
                    **common,
                ),
                "transient",
                offs,
            )

        # ---- time-varying Dirichlet g(x, t) (linear, non-parametric): row-replacement Dirichlet whose
        # held value is supplied by the forcing each step. Constant conditions go to the affine bias;
        # the constrained dofs carry no time derivative (their mass row is zeroed) and the held value
        # u[d] = g(x_d, t) is written into forcing_vector_fn(t) (the per-step Dirichlet lift). ----
        _const_pairs, _tv_entries = _build_dirichlet_tv_entries()
        if _tv_entries:
            from ..._fem import _eval_value_node_at_time

            _cd = jnp.asarray([p[0] for p in _const_pairs], dtype=jnp.int32) if _const_pairs else jnp.zeros((0,), jnp.int32)
            _cv = jnp.asarray([p[1] for p in _const_pairs], dtype=zeros.dtype) if _const_pairs else zeros[:0]
            _tvd = jnp.concatenate([e[0] for e in _tv_entries])
            _all_d = jnp.concatenate([_cd, _tvd])
            # Compress AFTER the Dirichlet edit: bcoo_set_unit_diag appends its own (d, d, 1)
            # triplets, which must merge with whatever the assembly already put on that diagonal.
            A_tv = compress_eager(bcoo_set_dirichlet_rows(spatial_jac(zeros, 0.0), _all_d))
            M_tv = compress_eager(bcoo_zero_rows(M, _all_d))
            c_tv = zeros.at[_cd].set(_cv)
            free_tv = jnp.ones((total,), dtype=zeros.dtype).at[_all_d].set(0.0)

            def forcing_vector_fn(t, args=None, _mask=free_tv, _tv=_tv_entries):
                f = _mask * (-spatial_res(zeros, t))  # source load on the free rows
                for dofs, vnode, coords in _tv:
                    f = f.at[dofs].set(jnp.asarray(_eval_value_node_at_time(vnode, coords, t)).reshape(-1))
                return f

            return (
                SemidiscreteTimeBlock(M=M_tv, A=A_tv, affine_bias=c_tv, forcing_vector_fn=forcing_vector_fn, **common),
                "transient",
                offs,
            )

        # linear transient.  The operator A is assembled at t=0 (autonomous operator); a
        # time-dependent SOURCE is carried by forcing_vector_fn(t).  The constant Dirichlet lift +
        # the t=0 load go into the affine bias via symmetric elimination; forcing_vector_fn supplies
        # only the time-varying increment on the free rows (Dirichlet rows handled by the bias).
        A = spatial_jac(zeros, 0.0)
        c0 = -spatial_res(zeros, 0.0)
        M, A, c = _apply_dirichlet_transient(M, A, c0, dirichlet_pairs)
        # Both operators are applied on EVERY timestep, so the ~19x triplet redundancy is paid once
        # per step for the whole march. Compressing here is the single highest-leverage site.
        M, A = compress_eager(M), compress_eager(A)
        if temporal_tags:
            free_mask = jnp.ones((total,), dtype=zeros.dtype)
            if d_dofs is not None:
                free_mask = free_mask.at[d_dofs].set(0.0)

            def forcing_vector_fn(t, args=None, _c0=c0, _mask=free_mask):
                return _mask * (-spatial_res(zeros, t) - _c0)

            return (
                SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, forcing_vector_fn=forcing_vector_fn, **common),
                "transient",
                offs,
            )
        return SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, **common), "transient", offs

    # === steady ===
    dirichlet_pairs = _build_dirichlet_pairs()
    residual = _make_residual(volume_terms, boundary_terms)
    jacobian = _make_jacobian(volume_terms, boundary_terms)
    nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms)
    s_d_dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
    s_d_vals = jnp.asarray([p[1] for p in dirichlet_pairs], dtype=zeros.dtype) if dirichlet_pairs else None

    # ---- runtime-parametric (inverse): the operator/residual is re-evaluated at the runtime args
    # each call, kept differentiable in args -- the parameter flows as a JAX array through the kernel
    # coefficient into the per-cell assembly (no float() cast). The same re-assembly handles affine,
    # non-affine and (scalar) parameters uniformly. ----
    if (
        runtime_parameter_tags
        or neural_param_names
        or _dir_net_models
        or history_specs
        or surface_history_specs
        or _coord_specs
    ):
        from ...trace import FemLinearSystem

        if nonlinear:
            # ``t`` carries the pseudo-time (load) coordinate τ for the history march — the load written
            # as a function of τ in the weak form varies through it. Defaults to 0.0, so the ordinary
            # (non-marching) parametric/inverse call sites are unchanged.
            if _dir_net_models:
                # net-valued Dirichlet u(∂Ω) - net(x): the held value is a differentiable function of the
                # net weights (delivered on args), so the row-replacement value is re-evaluated from args
                # each residual call (mirrors the linear parametric path's ``_dirichlet_pairs_at``). The
                # dof set is static (boundary-node layout); only the held values ride the weights.
                _npd = jnp.asarray(
                    [p[0] for p in _dirichlet_pairs_at({n: m.module for n, m in _dir_net_models.items()})],
                    dtype=jnp.int32,
                )

                def _np_hold(args):  # held value on every Dirichlet dof (const + net), net entries live
                    return jnp.stack([jnp.asarray(p[1]).reshape(()) for p in _dirichlet_pairs_at(args)])

                def res_p(u, args=None, t=0.0, _d=_npd):
                    R = residual(jnp.asarray(u), t, args)
                    return R.at[_d].set(jnp.asarray(u)[_d] - _np_hold(args))

                def jac_p(u, args=None, t=0.0, _d=_npd):
                    return bcoo_set_dirichlet_rows(jacobian(jnp.asarray(u), t, args), _d)
            else:

                def res_p(u, args=None, t=0.0, _d=s_d_dofs, _g=s_d_vals):
                    R = residual(jnp.asarray(u), t, args)
                    return R if _d is None else R.at[_d].set(jnp.asarray(u)[_d] - _g)

                def jac_p(u, args=None, t=0.0, _d=s_d_dofs):
                    J = jacobian(jnp.asarray(u), t, args)
                    return J if _d is None else bcoo_set_dirichlet_rows(J, _d)

            _op = FemResidualOperator(res_p, jac_p, total, runtime_parameter_exprs=dict(_param_and_neural_exprs))
            _op.history_specs = history_specs  # VOLUME step-history buffer layout for the load-step driver
            _op.surface_history_specs = surface_history_specs  # SURFACE (per-face) step-history layout
            _op.history_roles = history_roles  # {key: "primary" | "internal"} — how each state advances
            _op.state_readout = state_readout  # (u, t, args) -> {key: next per-QP VOLUME state}; march driver
            _op.surface_state_readout = surface_state_readout  # (u, t, args) -> {key: next per-FACE state}
            _op.path_specs = path_specs  # {fid: {frames (n_steps, n_nodes), ...}} — per-step load-path fields
            return (_op, "nonlinear", offs)

        def _assemble_at(args):
            A = jacobian(zeros, 0.0, args)
            b = -residual(zeros, 0.0, args)
            # a net-valued Dirichlet re-forms the lift from args each call; otherwise the static pairs.
            pairs = _dirichlet_pairs_at(args) if _dir_net_models else dirichlet_pairs
            if pairs:
                A, b = _apply_dirichlet_symmetric(A, b, pairs)
            return A, b

        # Static placeholder for .A/.b: scalar params at 0, networks (coefficient + Dirichlet) at stored weights.
        a0, b0 = _assemble_at(
            {n: 0.0 for n in runtime_parameter_tags}
            | {n: _neural_models[n].module for n in neural_param_names}
            | {n: m.module for n, m in _dir_net_models.items()}
        )
        op = FemLinearSystem(
            a0,
            b0,
            operator_fn=lambda args=None: _assemble_at(args)[0],
            rhs_fn=lambda args=None: _assemble_at(args)[1],
            runtime_parameter_exprs=dict(_param_and_neural_exprs),
            # The native parametric path re-assembles the operator at each args (it builds no affine
            # parameter basis), so every runtime parameter -- affine or not -- takes the re-assembly route.
            metadata={"nonaffine_operator": True},
        )
        return op, "linear", offs

    # nonlinear (non-parametric)
    if nonlinear:
        res_bc = _apply_dirichlet_rows(residual, dirichlet_pairs)
        jac = _dirichlet_jac_rows(jacobian, dirichlet_pairs)
        return (
            FemResidualOperator(
                lambda u, args=None: res_bc(jnp.asarray(u)),
                lambda u, args=None: jac(jnp.asarray(u)),
                total,
            ),
            "nonlinear",
            offs,
        )

    # linear (non-parametric)
    A = jacobian(zeros)
    b = -residual(zeros)
    if dirichlet_pairs:
        A, b = _apply_dirichlet_symmetric(A, jnp.asarray(b).reshape(-1), dirichlet_pairs)
    # Collapse duplicate triplets ONCE, after Dirichlet (which appends its own). The assembly emits
    # one block per term and never pre-sums, and each interior DOF pair gets a contribution per
    # incident element -- ~19x redundancy on a 3-D P1 mesh, paid again on every matvec.
    A = compress_eager(A)
    return (A, b), "linear", offs
