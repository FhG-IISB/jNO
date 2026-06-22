"""Native 2D Lagrange assembler for ``jno.fem`` (replaces the feax-backed path).

Implements the full assembly pipeline for scalar/vector Lagrange P1/P2 fields on 2D
triangle meshes (single- and multi-field, linear/nonlinear/transient),
mirroring the contract of :func:`fem_1d.assemble_fem_1d` and
:func:`fem_nonnodal.assemble_fem_nonnodal`.

Key components re-used without change:

* :func:`feax_utils._eval_expr_for_feax` — the DSL integrand evaluator.
* :func:`fem_1d._integrate_term` — weighted sum over quad points.
* :func:`fem_1d._apply_dirichlet_*` — Dirichlet enforcement (symmetric/row/transient).
* :func:`feax_utils._promote_to_quadratic` — P1→P2 mesh promotion.
* :func:`feax_utils._cell_region_mask` — per-cell sub-region indicator.

New components (this module only; no feax imports):

* :func:`fem_lagrange.lagrange_triangle` / :func:`fem_lagrange.identity_pushforward`
  — basix-backed Lagrange reference tabulation + isoparametric gradient map.
* :func:`fem_facets.build_facet_connectivity` / :func:`fem_facets.compute_face_normals`
  — boundary face connectivity + outward normals for surface integration.

References
----------
Matrix extraction via ``jax.jacfwd(residual)(zeros)`` follows Griewank & Walther,
*Evaluating Derivatives*, SIAM (2008), §3.5 — the same pattern as :mod:`fem_1d`
and :mod:`fem_nonnodal`.

Scope
-----
Lagrange P1/P2 fields on 2D triangle meshes (single- and multi-field, linear,
nonlinear, and transient).  3D (tet), runtime parameters, VPINN (:class:`ModelCall`),
complex FEM, and periodic BCs remain on the feax path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .feax_utils import (
    _cell_region_mask,
    _collect_region_mask_names,
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
from .fem_facets import build_facet_connectivity, compute_face_normals
from .fem_lagrange import BASIX_TET_EDGES, identity_pushforward, lagrange_triangle
from .fem_topology import BASIX_TRIANGLE_EDGES
from .weak_form import (
    _apply_sign,
    _contains_temporal_derivative,
    _is_obviously_nonlinear_in_unknown,
    _split_additive_terms,
)

# Reference triangle vertex coordinates (basix convention): v0=(0,0), v1=(1,0), v2=(0,1).
_REF_TRI_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

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
    cell_key = "triangle" if dim == 2 else "tetrahedron"
    pts_p1 = np.asarray(domain.mesh.points)[:, :dim]
    cells_p1 = np.asarray(domain.mesh.cells_dict[cell_key], dtype=np.int64)
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


def _build_face_tables(elem_degree: int, quad_degree: int):
    """Pre-tabulate the parent-cell Lagrange basis at the quad points of each local face.

    Returns ``(face_phi, face_dphi_ref, face_ref_qp, face_ref_tang, gw_1d)``:

    * ``face_phi``       ``(3, n_q, n_dof)``     parent basis values at face qp.
    * ``face_dphi_ref``  ``(3, n_q, n_dof, 2)``  reference-domain gradients.
    * ``face_ref_qp``    ``(3, n_q, 2)``          reference coords of face qp.
    * ``face_ref_tang``  ``(3, 2)``               reference edge tangent vectors.
    * ``gw_1d``          ``(n_q,)``               1-D Gauss weights on [0, 1].
    """
    import basix
    from basix import CellType, ElementFamily

    gp_1d, gw_1d = (np.asarray(x) for x in _line_quadrature(quad_degree))
    elem = basix.create_element(ElementFamily.P, CellType.triangle, elem_degree)

    phi_list, dphi_list, qp_list, tang_list = [], [], [], []
    for node_a, node_b, _ in _LOCAL_FACES_TRI:
        va, vb = _REF_TRI_VERTS[node_a], _REF_TRI_VERTS[node_b]
        ref_qp = va[None, :] * (1.0 - gp_1d[:, None]) + vb[None, :] * gp_1d[:, None]  # (n_q, 2)
        tab = elem.tabulate(1, ref_qp)  # (3, n_q, n_dof, 1)
        phi_list.append(tab[0, :, :, 0])  # (n_q, n_dof)
        dphi_list.append(np.stack([tab[1, :, :, 0], tab[2, :, :, 0]], axis=-1))  # (n_q, n_dof, 2)
        qp_list.append(ref_qp)
        tang_list.append(vb - va)

    return (
        jnp.asarray(np.stack(phi_list)),  # (3, n_q, n_dof)
        jnp.asarray(np.stack(dphi_list)),  # (3, n_q, n_dof, 2)
        jnp.asarray(np.stack(qp_list)),  # (3, n_q, 2)
        jnp.asarray(np.stack(tang_list)),  # (3, 2)
        jnp.asarray(gw_1d),  # (n_q,)
    )


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
    """Assemble a 2D Lagrange FEM system into ``(op, mode, offs)`` for :class:`FEM`.

    ``mode`` is ``"linear"``, ``"nonlinear"``, or ``"transient"``; ``op`` matches the
    return-type contract of :func:`fem_1d.assemble_fem_1d` and
    :func:`fem_nonnodal.assemble_fem_nonnodal`.

    Scope: scalar/vector Lagrange P1/P2 fields on 2D triangle meshes (single- and
    multi-field, with Dirichlet and Neumann/Robin boundary conditions).  3D (tet), runtime
    parameters, VPINN, complex FEM, and periodic BCs remain on the feax path.
    """
    from ...trace import FemResidualOperator

    dim = int(domain.dimension)
    if dim != 2:
        raise NotImplementedError(f"assemble_fem_native: only dim=2 is supported; got dim={dim}.")

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
    pts_p1 = mesh_data[0][0]  # (n_pts_p1, 2)  — P1 node coordinates (shared)
    cells_p1 = mesh_data[0][1]  # (n_cells, 3)   — P1 triangle connectivity (shared)

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

    # -------------------------------------------------------------------------
    # Element specs and JAX constants
    # -------------------------------------------------------------------------

    specs = [lagrange_triangle(f["order"], quad_degree) for f in fields]
    # All specs share the same triangle quadrature rule (basix is deterministic)
    qp_shared = jnp.asarray(specs[0].quad_points)  # (n_quad, 2)
    qw_shared = jnp.asarray(specs[0].quad_weights)  # (n_quad,)

    pts_j = jnp.asarray(pts_p1)
    cells_j = jnp.asarray(cells_p1, dtype=jnp.int32)
    n_cells = int(cells_p1.shape[0])

    ref_vals_all = [jnp.asarray(s.ref_values) for s in specs]  # list of (n_quad, n_dof_i, 1)
    ref_grads_all = [jnp.asarray(s.ref_grads) for s in specs]  # list of (n_quad, n_dof_i, 1, 2)
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

    # -------------------------------------------------------------------------
    # Surface integration setup
    # -------------------------------------------------------------------------

    # Per-field face tables (one set per distinct element order)
    face_tables_per_field = [_build_face_tables(f["order"], quad_degree) for f in fields]
    # face_tables_per_field[i] = (face_phi, face_dphi_ref, face_ref_qp, face_ref_tang, gw_1d)
    # shapes: (3, n_q, n_dof_i), (3, n_q, n_dof_i, 2), (3, n_q, 2), (3, 2), (n_q,)

    conn = build_facet_connectivity(cells_p1, "triangle")
    normals_np = compute_face_normals(pts_p1, conn, cells_p1, "triangle") if conn.n_bfaces > 0 else np.zeros((0, 2))

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
        verts = pts_j[cells_j[c]]  # (3, 2)
        J = jnp.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)  # (2, 2) columns = edges
        detJ = jnp.linalg.det(J)
        xq = verts[0][None, :] + qp_shared @ J.T  # (n_quad, 2) physical qp
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

    # -------------------------------------------------------------------------
    # Generic residual builder (volume + optional surface terms)
    # -------------------------------------------------------------------------

    # Surface connectivity (hoisted: shared by the residual and Jacobian builders).
    normals_j = jnp.asarray(normals_np)
    parent_j = jnp.asarray(conn.parent_cell, dtype=jnp.int32)
    lface_j = jnp.asarray(conn.local_face, dtype=jnp.int32)

    def _vol_elem_res(c, local_all, coeff, tfi, rnames):
        """Element residual of one volume term on cell ``c`` as a function of that cell's gathered
        all-field local DOFs ``local_all`` -> ``(n_test_dofs_tfi,)``. Driving the AD off this
        element-sized input (not the global state) is what keeps the per-cell Jacobian's intermediate
        O(n_local) instead of O(n_dofs)."""
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
            "temporal_tags": (),
            "runtime_parameter_tags": (),
            "region_mask_names": rnames,
            "volume_vars": cell_masks,
            "trial_value_shape": fields[tfi]["value_shape"],
            "trial_vec": vecs[tfi],
        }
        return _integrate_term(domain, coeff, loc, qw_shared * meas)

    def _surf_elem_res(fi, local_all, bcoeff, btfi, region):
        """Element residual of one surface term on boundary face ``fi`` as a function of the parent
        cell's gathered all-field local DOFs ``local_all`` -> ``(n_test_dofs_btfi,)``."""
        c = parent_j[fi]
        k = lface_j[fi]
        n_vec = normals_j[fi]  # (2,) outward unit normal
        cell_sols = _split_cell_local(local_all)
        verts = pts_j[cells_j[c]]
        J = jnp.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)
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

        _, _, fp_qp, fp_tang, gw_face = face_tables_per_field[btfi]
        tang = fp_tang[k]  # (2,) ref tangent
        jac_f = jnp.linalg.norm(J @ tang)  # physical edge length
        xq_f = verts[0] + fp_qp[k] @ J.T  # (n_q, 2)
        loc = {
            "physical_quad_points": xq_f,
            "fields": per_f,
            "field_index": field_index,
            "tag": f"gauss_{region}",
            "surface": True,
            "domain_context": {**ctx, f"n_{region}": n_vec},
            "temporal_tags": (),
            "runtime_parameter_tags": (),
            "region_mask_names": (),
            "volume_vars": (),
            "trial_value_shape": fields[btfi]["value_shape"],
            "trial_vec": vecs[btfi],
        }
        return _integrate_term(domain, bcoeff, loc, gw_face * jac_f)

    def _preprocess_terms(terms, bterms):
        """``(typed_with_masks, surface_work)``: lower each additive sub-term to
        ``(coeff, test_field_idx[, mask_names])`` and bucket boundary faces per region."""
        typed: List[Tuple[Any, int]] = []
        for bare in terms:
            for sign, sub in _split_additive_terms(domain, bare):
                coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                tfi = _test_field_index(coeff, field_index)
                if tfi is None:
                    raise ValueError(
                        "jno.fem (native): each weak-form term must contain exactly one test "
                        "field (it determines the equation block)."
                    )
                typed.append((coeff, tfi))
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
                    bcoeff = _lower_statefield_to_trial(_apply_sign(domain, 1, bexpr), {})
                    btfi = _test_field_index(bcoeff, field_index)
                    if btfi is None:
                        raise ValueError(
                            f"jno.fem (native): boundary term in region {region!r} must contain exactly one test field."
                        )
                    btyped.append((bcoeff, btfi))
                surface_work.append((region, np.asarray(face_ids, dtype=np.int32), btyped))
        return typed_with_masks, surface_work

    def _make_residual(terms, bterms=None):
        """Build the free global residual ``R(u_flat) -> (total,)`` (volume + optional surface).

        ``bterms`` is an optional ``{region: [exprs]}`` dict for surface (Neumann/Robin) terms; pass
        ``None`` (the default) to assemble volume terms only — used for the transient mass matrix,
        where boundary contributions must not appear."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        def residual(u_flat):
            R = jnp.zeros(total, dtype=u_flat.dtype)
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)

            for coeff, tfi, rnames in typed_with_masks:
                elem = jax.vmap(lambda c, la, _e=coeff, _t=tfi, _r=rnames: _vol_elem_res(c, la, _e, _t, _r))(
                    jnp.arange(n_cells), local_all
                )
                R = R.at[cdofs[tfi].reshape(-1)].add(elem.reshape(-1))

            for region, face_ids, btyped in surface_work:
                fids = jnp.asarray(face_ids, dtype=jnp.int32)
                pcells = parent_j[fids]
                lv = u_flat[cell_all_dofs[pcells]]  # (n_face, n_local_all)
                for bcoeff, btfi in btyped:
                    contribs = jax.vmap(lambda fi, la, _e=bcoeff, _t=btfi, _r=region: _surf_elem_res(fi, la, _e, _t, _r))(
                        fids, lv
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
        tangent tensor and OOMs on any non-trivial mesh). The dense result matches the feax matrix
        entry-for-entry on a matched DOF numbering."""
        typed_with_masks, surface_work = _preprocess_terms(terms, bterms)

        def jacobian(u_flat):
            A = jnp.zeros((total, total), dtype=u_flat.dtype)
            local_all = u_flat[cell_all_dofs]  # (n_cell, n_local_all)

            for coeff, tfi, rnames in typed_with_masks:

                def _ke(c, la, _e=coeff, _t=tfi, _r=rnames):
                    return jax.jacfwd(lambda v: _vol_elem_res(c, v, _e, _t, _r))(la)

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
                        return jax.jacfwd(lambda v: _surf_elem_res(fi, v, _e, _t, _r))(la)

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

    def _build_dirichlet_pairs() -> List[Tuple[int, float]]:
        from ..._fem import _eval_value_node_at

        pairs: List[Tuple[int, float]] = []
        for field_key, region, comp, value, value_node in dirichlet_raw:
            fidx = field_index.get(field_key)
            if fidx is None:
                continue
            vt = vecs[fidx]
            pts_all = pts_f_all[fidx]
            for nid in _region_node_ids_from_pts(domain, region, pts_all):
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
        return pairs

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
            _comp, u0_node = _essential_spec(_bare(ic))
            fidx = field_index.get(_field_key_of(ic))
            if fidx is None:
                raise ValueError("jno.fem (native): IC does not match any known trial field.")
            pts_ic = pts_f_all[fidx]  # (n_nodes_f[fidx], 2)
            u0_vals = jnp.asarray(_eval_value_node_at(u0_node, jnp.asarray(pts_ic))).reshape(n_nodes_f[fidx], vecs[fidx])
            state0 = state0.at[offs[fidx] : offs[fidx + 1]].set(u0_vals.reshape(-1))
        common["state0"] = state0

        dirichlet_pairs = _build_dirichlet_pairs()

        nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial)
        if nonlinear:
            res_bc = _apply_dirichlet_rows(spatial_res, dirichlet_pairs)
            jac = _dirichlet_jac_rows(spatial_jac, dirichlet_pairs)
            d_dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
            M_bc = M if d_dofs is None else M.at[d_dofs, :].set(0.0).at[:, d_dofs].set(0.0)
            return (
                FeaxTimeBlock(
                    mass=lambda t, args=None, _M=M_bc: _M,
                    residual=lambda u, t, args=None: res_bc(u),
                    jacobian=lambda u, t, args=None: jac(u),
                    **common,
                ),
                "transient",
                offs,
            )

        # linear transient
        A = spatial_jac(zeros)
        c = -spatial_res(zeros)
        M, A, c = _apply_dirichlet_transient(M, A, c, dirichlet_pairs)
        return FeaxTimeBlock(M=M, A=A, affine_bias=c, **common), "transient", offs

    # === steady ===
    dirichlet_pairs = _build_dirichlet_pairs()
    residual = _make_residual(volume_terms, boundary_terms)

    # nonlinear
    if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms):
        res_bc = _apply_dirichlet_rows(residual, dirichlet_pairs)
        jac = _dirichlet_jac_rows(_make_jacobian(volume_terms, boundary_terms), dirichlet_pairs)
        return (
            FemResidualOperator(
                lambda u, args=None: res_bc(jnp.asarray(u)),
                lambda u, args=None: jac(jnp.asarray(u)),
                total,
            ),
            "nonlinear",
            offs,
        )

    # linear
    A = _make_jacobian(volume_terms, boundary_terms)(zeros)
    b = -residual(zeros)
    if dirichlet_pairs:
        A, b = _apply_dirichlet_symmetric(jnp.asarray(A), jnp.asarray(b), dirichlet_pairs)
    return (A, b), "linear", offs
