"""Native Lagrange assembler for ``jno.fem`` (replaces the feax-backed path).

Implements the full assembly pipeline for scalar/vector Lagrange P1/P2 fields on 2D
triangle and 3D tetrahedral meshes (single- and multi-field, linear/nonlinear/transient),
mirroring the contract of :func:`fem_1d.assemble_fem_1d` and
:func:`fem_nonnodal.assemble_fem_nonnodal`. The assembler is dimension-generic: the cell
Jacobian, element factory and facet machinery all key off ``dim``.

Key components re-used without change:

* :func:`feax_utils._eval_expr_for_feax` — the DSL integrand evaluator.
* :func:`fem_1d._integrate_term` — weighted sum over quad points.
* :func:`fem_1d._apply_dirichlet_*` — Dirichlet enforcement (symmetric/row/transient).
* :func:`feax_utils._promote_to_quadratic` — P1→P2 mesh promotion.
* :func:`feax_utils._cell_region_mask` — per-cell sub-region indicator.

New components (this module only; no feax imports):

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
(2D edge / 3D tet-face surface quadrature).  Runtime FEM *field* parameters and complex FEM
remain on the feax path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .feax_utils import (
    _cell_region_mask,
    _collect_region_mask_names,
    _collect_temporal_tags_for_feax,
    _infer_fields,
    _lower_statefield_to_trial,
    _promote_to_quadratic,
    _test_field_index,
)
from .fem_1d import (
    _apply_dirichlet_rows,
    _apply_dirichlet_symmetric,
    _apply_dirichlet_transient,
    _integrate_term,
    _line_quadrature,
    _region_node_ids,
)
from .fem_facets import _LOCAL_FACES_TET, build_facet_connectivity, compute_face_normals
from .fem_lagrange import BASIX_TET_EDGES, identity_pushforward, lagrange_tet, lagrange_triangle
from .fem_topology import BASIX_TRIANGLE_EDGES
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
    if dim == 2:
        edge_local = BASIX_TRIANGLE_EDGES
    elif dim == 3:
        edge_local = BASIX_TET_EDGES
    else:
        raise NotImplementedError(f"Dimension {dim} not supported by native assembler.")
    pts_f, cells_f = _promote_to_quadratic(pts_p1, cells_p1, edge_local)
    return pts_p1, cells_p1, pts_f, cells_f


def _region_node_ids_from_pts(domain, region: str, pts_all: np.ndarray) -> List[int]:
    """Node ids in ``pts_all`` satisfying the location function for ``region``."""
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
    from basix import CellType, ElementFamily

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

    elem = basix.create_element(ElementFamily.P, cell, elem_degree)
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


# ---------------------------------------------------------------------------
# Native fem_context (feax-free) for the VPINN / grouped-weak-form path
# ---------------------------------------------------------------------------


def build_native_fem_context(domain, *, element_type, quad_degree, vec=1, neumann_tags=(), dirichlet_node_ids=None):
    """Native, feax-free equivalent of ``init_fem``'s ``domain.fem_context`` for the VPINN /
    grouped-weak-form evaluator (``trace_evaluator._eval_grouped_assembly``).

    Returns ``(fem_context, vol_quad_points, surface_quad_by_tag, surface_normals_by_tag)``:
    the same tensor layout the feax path cached, computed from the native Lagrange element +
    facet machinery instead of a feax ``Problem``. The geometry is affine-simplex (P1 vertices),
    so the cell Jacobian, ``JxW`` and physical gradients are exact for P1/P2 nodal bases.
    """
    dim = int(domain.dimension)
    order = 2 if element_type in ("TRI6", "TET10") else 1
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


def assemble_fem_native(
    domain,
    volume_terms: List[Any],
    boundary_terms: Dict[str, List[Any]],
    dirichlet_raw: List[Tuple],
    ic_residuals: List[Any],
    *,
    vec: int,
    quad_degree: int,
) -> Tuple[Any, str]:
    """Assemble a Lagrange FEM system into ``(op, mode, offs)`` for :class:`FEM`.

    ``mode`` is ``"linear"``, ``"nonlinear"``, or ``"transient"``; ``op`` matches the
    return-type contract of :func:`fem_1d.assemble_fem_1d` and
    :func:`fem_nonnodal.assemble_fem_nonnodal`.

    Scope: scalar/vector Lagrange P1/P2 fields on 2D triangle and 3D tetrahedral meshes
    (single- and multi-field), with Dirichlet and Neumann/Robin boundary conditions (2D edge /
    3D tet-face surface quadrature).  Complex FEM and runtime FEM *field* parameters remain on
    the feax path.
    """
    from ...trace import FemResidualOperator

    dim = int(domain.dimension)
    if dim not in (2, 3):
        raise NotImplementedError(f"assemble_fem_native: only dim=2 and dim=3 are supported; got dim={dim}.")
    cell_key = "triangle" if dim == 2 else "tetrahedron"

    ctx = dict(getattr(domain, "context", {}) or {})

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
    # ``FEM.points`` reads ``[0]`` on the native path (there is no feax problem to query) so the
    # solution can be interpreted at the right coordinates -- the first field's nodes, matching
    # the feax convention (``problem.mesh[0].points``). The full per-field list backs
    # ``FEM.field_points`` for coupled problems (e.g. Taylor-Hood velocity vs pressure nodes).
    domain._fem_native_dof_points = np.asarray(pts_f_all[0])
    domain._fem_native_dof_points_all = [np.asarray(p) for p in pts_f_all]

    # Field-0 assembly cells + element order, for the periodic-tie reduction (``_build_periodic_
    # reduction`` reads the assembly mesh's cells to extract boundary facets). On the native path
    # there is no feax problem to query, so ``_finalize`` reads these instead of ``_assembly_cells``.
    domain._fem_native_assembly_cells = np.asarray(cells_f_all[0])
    domain._fem_native_assembly_order = int(fields[0]["order"])

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

    # Temporal variable tags (e.g. "__time__") used inside the weak form's coefficients -- a
    # time-dependent source s(x,t) or operator. The residual/Jacobian builders thread the runtime time
    # `t` into the kernel's volume_vars at the matching slots so `_eval_expr_for_feax` resolves them;
    # the packing order is [temporal..., runtime_param..., region_mask...] (see _make_internal_vars).
    _temporal_tag_set: set = set()
    for bare in volume_terms:
        _collect_temporal_tags_for_feax(bare, _temporal_tag_set)
    for _exprs in boundary_terms.values():
        for bare in _exprs:
            _collect_temporal_tags_for_feax(bare, _temporal_tag_set)
    temporal_tags: Tuple[str, ...] = tuple(sorted(_temporal_tag_set))

    # Runtime parameters (trainable ``jno.np.parameter(...)`` coefficients, e.g. an unknown diffusivity
    # in an inverse problem). Their values arrive at solve time in an ``args`` dict; the builders pack
    # them into volume_vars right AFTER the temporal slots so ``_eval_expr_for_feax`` resolves each
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
    if _field_param_names and len(fields) > 1:
        raise NotImplementedError(
            "jno.fem (native): a FEM field parameter k(x) is supported on single-field problems only."
        )

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

    # -------------------------------------------------------------------------
    # Cell-level field data builder (called inside vmap'd kernels)
    # -------------------------------------------------------------------------

    def _cell_fields(c, cell_sols):
        """Per-field ``(phi, dphi_phys, cell_sol)`` and shared ``(xq, meas)`` for cell c.

        ``cell_sols`` is a list of this cell's local DOF values per field, shape
        ``(n_local_i, vec_i)``. The residual path gathers them from the global state; the
        per-cell Jacobian path passes a *differentiated* local slice so ``jax.jacfwd`` sees
        an element-sized (not global) input — keeping the AD intermediate O(n_local), not
        O(n_dofs)."""
        verts = pts_j[cells_j[c]]  # (dim+1, dim)
        J = jnp.stack([verts[i + 1] - verts[0] for i in range(dim)], axis=1)  # (dim, dim) columns = edges
        detJ = jnp.linalg.det(J)
        xq = verts[0][None, :] + qp_shared @ J.T  # (n_quad, dim) physical qp
        meas = jnp.abs(detJ)

        per = []
        for i in range(len(fields)):
            phi, dphi = identity_pushforward(ref_vals_all[i], ref_grads_all[i], J, detJ)
            per.append({"shape_vals": phi, "shape_grads": dphi, "cell_sol": cell_sols[i], "space": "Lagrange"})

        return per, xq, meas

    # Cell-local DOF bookkeeping for per-cell element-Jacobian assembly. ``cell_all_dofs[c]`` lists
    # every global DOF (all fields, node-major) the cell couples, so an element matrix's columns map
    # straight back to the global matrix; ``loc_seg`` splits a gathered local vector per field.
    n_local_f = [int(cells_f_j[i].shape[1]) for i in range(len(fields))]
    loc_seg = [0]
    for i in range(len(fields)):
        loc_seg.append(loc_seg[-1] + n_local_f[i] * vecs[i])
    cell_all_dofs = jnp.concatenate(cdofs, axis=1) if len(cdofs) > 1 else cdofs[0]  # (n_cell, n_local_all)

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
                pv.append(jnp.zeros((n_local_f[0] if name in _field_param_names else 1,), dtype))
                continue
            flat = jnp.reshape(jnp.asarray(a[name], dtype=dtype), (-1,))
            # Single-field nodal field parameter -> this cell's local nodal values (field 0).
            pv.append(flat[cells_f_j[0][c]] if name in _field_param_names else flat[:1])
        return tv + tuple(pv)

    # -------------------------------------------------------------------------
    # Generic residual builder (volume + optional surface terms)
    # -------------------------------------------------------------------------

    # Surface connectivity (hoisted: shared by the residual and Jacobian builders).
    normals_j = jnp.asarray(normals_np)
    parent_j = jnp.asarray(conn.parent_cell, dtype=jnp.int32)
    lface_j = jnp.asarray(conn.local_face, dtype=jnp.int32)

    def _vol_elem_res(c, local_all, coeff, tfi, rnames, t=0.0, args=None):
        """Element residual of one volume term on cell ``c`` as a function of that cell's gathered
        all-field local DOFs ``local_all`` -> ``(n_test_dofs_tfi,)``. Driving the AD off this
        element-sized input (not the global state) is what keeps the per-cell Jacobian's intermediate
        O(n_local) instead of O(n_dofs). ``t`` / ``args`` carry the runtime time and parameters, packed
        per cell into volume_vars BEFORE the region masks (layout [temporal..., runtime_param...,
        region_mask...])."""
        cell_sols = _split_cell_local(local_all)
        per, xq, meas = _cell_fields(c, cell_sols)
        cell_masks = tuple(region_mask_arrays[list(region_mask_names).index(r)][c] for r in rnames)
        loc = {
            "physical_quad_points": xq,
            "fields": per,
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": ctx,
            "temporal_tags": temporal_tags,
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": rnames,
            "volume_vars": _runtime_vals(c, t, args, local_all.dtype) + cell_masks,
            "trial_value_shape": fields[tfi]["value_shape"],
            "trial_vec": vecs[tfi],
        }
        if _field_param_names:
            # The field parameter's nodal slice is interpolated to the quad points with the field's
            # shape functions (single-field: field 0). _runtime_parameter_value_from_internal_vars
            # reads this top-level shape_vals.
            loc["shape_vals"] = per[0]["shape_vals"]
        return _integrate_term(domain, coeff, loc, qw_shared * meas)

    def _surf_elem_res(fi, local_all, bcoeff, btfi, region, t=0.0, args=None):
        """Element residual of one surface term on boundary face ``fi`` as a function of the parent
        cell's gathered all-field local DOFs ``local_all`` -> ``(n_test_dofs_btfi,)``."""
        c = parent_j[fi]
        k = lface_j[fi]
        n_vec = normals_j[fi]  # (dim,) outward unit normal
        cell_sols = _split_cell_local(local_all)
        verts = pts_j[cells_j[c]]
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
            loc["shape_vals"] = per_f[0]["shape_vals"]
        return _integrate_term(domain, bcoeff, loc, face_w * jac_f)

    def _classify_one(coeff, where: str) -> List[Tuple[Any, int]]:
        """``[(coeff, test_field_idx), ...]`` for one lowered term. Normally one entry; a term that
        welds several test fields inside a product (the real part of a ``complex=True`` form, e.g.
        ``c·(u_r·w_r − u_i·w_i)``) is distributed over its sums into single-test sub-terms -- the same
        fallback the feax multifield path uses, so one complex form lowers onto the coupled blocks."""
        from ...trace import BinaryOp, Literal
        from .feax_utils import _expand_product_terms

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

    def _make_residual(terms, bterms=None):
        """Build the free global residual ``R(u_flat) -> (total,)`` (volume + optional surface).

        ``bterms`` is an optional ``{region: [exprs]}`` dict for surface (Neumann/Robin) terms; pass
        ``None`` (the default) to assemble volume terms only — used for the transient mass matrix,
        where boundary contributions must not appear."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        def residual(u_flat, t=0.0, args=None):
            R = jnp.zeros(total, dtype=u_flat.dtype)
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)

            for coeff, tfi, rnames in typed_with_masks:
                elem = jax.vmap(lambda c, la, _e=coeff, _t=tfi, _r=rnames: _vol_elem_res(c, la, _e, _t, _r, t, args))(
                    jnp.arange(n_cells), local_all
                )
                R = R.at[cdofs[tfi].reshape(-1)].add(elem.reshape(-1))

            for region, face_ids, btyped in surface_work:
                fids = jnp.asarray(face_ids, dtype=jnp.int32)
                pcells = parent_j[fids]
                lv = u_flat[cell_all_dofs[pcells]]  # (n_face, n_local_all)
                for bcoeff, btfi in btyped:
                    contribs = jax.vmap(
                        lambda fi, la, _e=bcoeff, _t=btfi, _r=region: _surf_elem_res(fi, la, _e, _t, _r, t, args)
                    )(fids, lv)
                    R = R.at[cdofs[btfi][pcells].reshape(-1)].add(contribs.reshape(-1))
            return R

        return residual

    def _make_jacobian(terms, bterms=None):
        """Build the dense Jacobian ``J(u_flat) -> (total, total)`` by *per-element* forward-mode AD.

        Each cell's (and boundary face's) element matrix is ``jacfwd`` of its element residual w.r.t.
        that element's local DOFs — an ``(n_test, n_local)`` block — then scatter-added into the global
        matrix. The AD never sees the global state, so the intermediate is element-sized; this is what
        a single global ``jacfwd(residual)`` cannot do (it materialises an ``O(n_dofs × n_cells)``
        tangent tensor and OOMs on any non-trivial mesh). The dense result matches the feax matrix
        entry-for-entry on a matched DOF numbering."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        def jacobian(u_flat, t=0.0, args=None):
            A = jnp.zeros((total, total), dtype=u_flat.dtype)
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)

            for coeff, tfi, rnames in typed_with_masks:

                def _ke(c, la, _e=coeff, _t=tfi, _r=rnames):
                    return jax.jacfwd(lambda v: _vol_elem_res(c, v, _e, _t, _r, t, args))(la)

                Ke = jax.vmap(_ke)(jnp.arange(n_cells), local_all)  # (n_cell, n_test_tfi, n_local_all)
                rows = cdofs[tfi]  # (n_cell, n_test_tfi)
                A = A.at[rows[:, :, None], cell_all_dofs[:, None, :]].add(Ke)

            for region, face_ids, btyped in surface_work:
                fids = jnp.asarray(face_ids, dtype=jnp.int32)
                pcells = parent_j[fids]
                lv = u_flat[cell_all_dofs[pcells]]  # (n_face, n_local_all)
                fcols = cell_all_dofs[pcells]  # (n_face, n_local_all)
                for bcoeff, btfi in btyped:

                    def _kef(fi, la, _e=bcoeff, _t=btfi, _r=region):
                        return jax.jacfwd(lambda v: _surf_elem_res(fi, v, _e, _t, _r, t, args))(la)

                    Kef = jax.vmap(_kef)(fids, lv)  # (n_face, n_test_btfi, n_local_all)
                    frows = cdofs[btfi][pcells]  # (n_face, n_test_btfi)
                    A = A.at[frows[:, :, None], fcols[:, None, :]].add(Kef)
            return A

        return jacobian

    def _dirichlet_jac_rows(jac_fn, pairs):
        """Wrap an assembled-Jacobian callable so Dirichlet rows become the identity row — the
        matrix-level analogue of :func:`_apply_dirichlet_rows` (row-replacement, columns kept), so it
        matches ``jacfwd`` of the row-replaced residual that the Newton step expects."""
        if not pairs:
            return jac_fn
        dofs = jnp.asarray([p[0] for p in pairs], dtype=jnp.int32)

        def jac(u_flat):
            return jac_fn(jnp.asarray(u_flat)).at[dofs, :].set(0.0).at[dofs, dofs].set(1.0)

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
        from ..._fem import _eval_value_node_at

        pairs: List[Tuple[int, float]] = []
        for field_key, region, comp, value, value_node in dirichlet_raw:
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            vt = vecs[fidx]
            pts_all = pts_f_all[fidx]
            for nid in _boundary_node_ids(fidx, region):
                p = np.asarray(pts_all[nid])
                if value_node is not None:
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
        domain._fem_native_dirichlet_pairs = pairs
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
        from .backend_blocks import FeaxTimeBlock
        from .time_route import _infer_time_window, _strip_temporal_trial_derivative

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

        mass_terms = [_strip_temporal_trial_derivative(t) for t in temporal]
        # Mass matrix: volume only (no boundary); spatial residual: volume + boundary
        M = _make_jacobian(mass_terms)(zeros)
        spatial_res = _make_residual(spatial, boundary_terms)
        spatial_jac = _make_jacobian(spatial, boundary_terms)

        t0, t1, dt = _infer_time_window(domain)
        common = dict(
            backend="feax_time",
            mode="implicit",
            time_order=1,
            spatial_kind="weak_form",
            state0=None,
            t0=t0,
            t1=t1,
            dt=dt,
            feax_context=getattr(domain, "_feax_context", {}) or {},
        )

        # --- initial state: nodal interpolation (exact for Lagrange) ---
        state0 = zeros
        for ic in ic_residuals:
            comp, u0_node = _essential_spec(_bare(ic))
            fidx = field_index.get(_field_key_of(ic))
            if fidx is None:
                raise ValueError("jno.fem (native): IC does not match any known trial field.")
            pts_ic = pts_f_all[fidx]  # (n_nodes_f[fidx], 2)
            nn, vv = n_nodes_f[fidx], vecs[fidx]
            raw = jnp.reshape(jnp.asarray(_eval_value_node_at(u0_node, jnp.asarray(pts_ic))), (-1,))
            if comp is not None:
                # Per-component IC (e.g. ``u(initial)[0] - g0``): set just component ``comp`` at every
                # node of the field. ``raw`` is the per-node value (or a single constant to broadcast).
                vals = jnp.broadcast_to(raw, (nn,)) if raw.size == 1 else raw.reshape(nn)
                idx = offs[fidx] + jnp.arange(nn) * vv + int(comp)
                state0 = state0.at[idx].set(vals)
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
                state0 = state0.at[offs[fidx] : offs[fidx + 1]].set(u0_vals.reshape(-1))
        common["state0"] = state0

        dirichlet_pairs = _build_dirichlet_pairs()
        d_dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
        d_vals = jnp.asarray([p[1] for p in dirichlet_pairs], dtype=zeros.dtype) if dirichlet_pairs else None

        nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial)
        if nonlinear:
            # Row-replacement Dirichlet (constant g), threaded through the runtime time t AND the
            # runtime args so a time-dependent / parametric spatial coefficient is re-evaluated each step.
            def res_bc(u, t, args=None, _d=d_dofs, _g=d_vals):
                R = spatial_res(jnp.asarray(u), t, args)
                return R if _d is None else R.at[_d].set(jnp.asarray(u)[_d] - _g)

            def jac_bc(u, t, args=None, _d=d_dofs):
                J = spatial_jac(jnp.asarray(u), t, args)
                return J if _d is None else J.at[_d, :].set(0.0).at[_d, _d].set(1.0)

            M_bc = M if d_dofs is None else M.at[d_dofs, :].set(0.0).at[:, d_dofs].set(0.0)
            return (
                FeaxTimeBlock(
                    mass=lambda t, args=None, _M=M_bc: _M,
                    residual=res_bc,
                    jacobian=jac_bc,
                    runtime_parameter_exprs=dict(_rt_param_exprs),
                    **common,
                ),
                "transient",
                offs,
            )

        # ---- linear parametric transient: the operator A(t, args) is re-evaluated each step.
        # Row-replacement Dirichlet (rows -> identity, columns kept) needs no args-dependent lift --
        # the coupling to the held Dirichlet value sits on the LHS, the constant g in the affine bias. ----
        if runtime_parameter_tags:
            M_bc = M if d_dofs is None else M.at[d_dofs, :].set(0.0).at[:, d_dofs].set(0.0)
            c_bias = zeros if d_dofs is None else zeros.at[d_dofs].set(d_vals)
            free_mask = jnp.ones((total,), dtype=zeros.dtype)
            if d_dofs is not None:
                free_mask = free_mask.at[d_dofs].set(0.0)

            def operator_fn(t, args=None, _d=d_dofs):
                A = spatial_jac(zeros, t, args)
                return A if _d is None else A.at[_d, :].set(0.0).at[_d, _d].set(1.0)

            def forcing_vector_fn(t, args=None, _mask=free_mask):
                return _mask * (-spatial_res(zeros, t, args))

            return (
                FeaxTimeBlock(
                    M=M_bc,
                    operator_fn=operator_fn,
                    affine_bias=c_bias,
                    forcing_vector_fn=forcing_vector_fn,
                    runtime_parameter_exprs=dict(_rt_param_exprs),
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
            A_tv = spatial_jac(zeros, 0.0).at[_all_d, :].set(0.0).at[_all_d, _all_d].set(1.0)
            M_tv = M.at[_all_d, :].set(0.0)
            c_tv = zeros.at[_cd].set(_cv)
            free_tv = jnp.ones((total,), dtype=zeros.dtype).at[_all_d].set(0.0)

            def forcing_vector_fn(t, args=None, _mask=free_tv, _tv=_tv_entries):
                f = _mask * (-spatial_res(zeros, t))  # source load on the free rows
                for dofs, vnode, coords in _tv:
                    f = f.at[dofs].set(jnp.asarray(_eval_value_node_at_time(vnode, coords, t)).reshape(-1))
                return f

            return (
                FeaxTimeBlock(M=M_tv, A=A_tv, affine_bias=c_tv, forcing_vector_fn=forcing_vector_fn, **common),
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
        if temporal_tags:
            free_mask = jnp.ones((total,), dtype=zeros.dtype)
            if d_dofs is not None:
                free_mask = free_mask.at[d_dofs].set(0.0)

            def forcing_vector_fn(t, args=None, _c0=c0, _mask=free_mask):
                return _mask * (-spatial_res(zeros, t) - _c0)

            return FeaxTimeBlock(M=M, A=A, affine_bias=c, forcing_vector_fn=forcing_vector_fn, **common), "transient", offs
        return FeaxTimeBlock(M=M, A=A, affine_bias=c, **common), "transient", offs

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
    if runtime_parameter_tags:
        from ...trace import FemLinearSystem

        if nonlinear:

            def res_p(u, args=None, _d=s_d_dofs, _g=s_d_vals):
                R = residual(jnp.asarray(u), 0.0, args)
                return R if _d is None else R.at[_d].set(jnp.asarray(u)[_d] - _g)

            def jac_p(u, args=None, _d=s_d_dofs):
                J = jacobian(jnp.asarray(u), 0.0, args)
                return J if _d is None else J.at[_d, :].set(0.0).at[_d, _d].set(1.0)

            return (
                FemResidualOperator(res_p, jac_p, total, runtime_parameter_exprs=dict(_rt_param_exprs)),
                "nonlinear",
                offs,
            )

        def _assemble_at(args):
            A = jacobian(zeros, 0.0, args)
            b = -residual(zeros, 0.0, args)
            if dirichlet_pairs:
                A, b = _apply_dirichlet_symmetric(A, b, dirichlet_pairs)
            return A, b

        a0, b0 = _assemble_at({n: 0.0 for n in runtime_parameter_tags})  # static placeholder for .A/.b
        op = FemLinearSystem(
            a0,
            b0,
            operator_fn=lambda args=None: _assemble_at(args)[0],
            rhs_fn=lambda args=None: _assemble_at(args)[1],
            runtime_parameter_exprs=dict(_rt_param_exprs),
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
        A, b = _apply_dirichlet_symmetric(jnp.asarray(A), jnp.asarray(b), dirichlet_pairs)
    return (A, b), "linear", offs
