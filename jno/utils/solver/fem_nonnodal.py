"""Native assembler for non-nodal (push-forward) element families.

This is the n-D analogue of the native 1D path (:mod:`fem_1d`): feax cannot
assemble derivative/edge-DOF elements (it has no push-forward), so the element
zoo assembles on the jNO side from :mod:`fem_topology` (global edge numbering +
orientation) and :mod:`fem_elements` (basix reference tabulation + per-cell
push-forward).

Two entry points:

* :func:`assemble_mixed_poisson_rt` — a *direct* RT–P0 mixed-Poisson assembler, kept to
  validate the engine (edge DOFs, orientation, contravariant Piola, divergence, saddle-block
  assembly) end-to-end against a manufactured solution by *convergence rate*.
* :func:`assemble_fem_nonnodal` — the DSL-driven assembler ``jno.fem`` routes RT/N1E/P0 fields to.
  It covers the H(div)/H(curl) mass and L²-projection, the mixed-Poisson saddle system, the essential
  normal-flux BC ``u·n = g`` (pins boundary-edge DOFs), and the natural pressure BC ``p = p_D``.

Mixed Poisson ``u = -∇p``, ``div u = f``. Flux ``u ∈ RT``, scalar ``p ∈ P0``; weak form
``∫u·v − ∫p div v = 0`` ∀v∈RT, ``∫q div u = ∫f q`` ∀q∈P0. Global DOFs are
``[edge DOFs (n_edges)] ++ [cell DOFs (n_cells)]``; block system ``[[M, −Bᵀ], [B, 0]] [u; p]``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .fem_elements import ElementSpec, piola_contravariant, raviart_thomas_triangle
from .fem_topology import EdgeTopology, build_edge_topology

ScalarField = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # (x, y) -> values


def _cell_jacobian(verts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Affine triangle Jacobian ``J = [v1-v0, v2-v0]`` and its (signed) determinant."""
    J = jnp.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)  # (2, 2)
    return J, jnp.linalg.det(J)


def assemble_mixed_poisson_rt(
    points: np.ndarray, cells: np.ndarray, source_fn: ScalarField, *, quad_degree: int = 4
) -> Tuple[jnp.ndarray, jnp.ndarray, EdgeTopology, ElementSpec]:
    """Assemble the RT–P0 mixed Poisson saddle system ``A x = b`` on a triangle mesh.

    ``points`` ``(n_pts, 2)``, ``cells`` ``(n_cells, 3)`` (P1 triangles), ``source_fn``
    the volume source ``f(x, y)``. Returns ``(A, b, edge_topology, element_spec)`` with
    ``A`` of size ``n_edges + n_cells`` (edge/flux DOFs first, then cell/pressure DOFs).
    """
    spec = raviart_thomas_triangle(degree=1, quad_degree=quad_degree)
    top = build_edge_topology(cells, spec.local_edges)
    n_edges, n_cells = top.n_edges, int(cells.shape[0])

    pts = jnp.asarray(points)
    cells_j = jnp.asarray(np.asarray(cells), dtype=jnp.int32)
    qp, qw = jnp.asarray(spec.quad_points), jnp.asarray(spec.quad_weights)
    rv, rd = jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_div)
    signs = jnp.asarray(top.cell_edge_signs.astype(np.float64))  # (n_cells, 3)

    def _local(cell, sgn):
        verts = pts[cell]  # (3, 2)
        J, detJ = _cell_jacobian(verts)
        meas = jnp.abs(detJ)
        phi, div = piola_contravariant(rv, rd, J, detJ, sgn)  # (nq, 3, 2), (nq, 3)
        w = qw * meas
        Mc = jnp.einsum("q,qad,qbd->ab", w, phi, phi)  # (3, 3) RT mass
        Bc = jnp.einsum("q,qa->a", w, div)  # (3,) ∫ div φ_a
        xq = verts[0][None, :] + qp @ J.T  # (nq, 2) physical quad points
        fc = jnp.sum(w * source_fn(xq[:, 0], xq[:, 1]))  # ∫ f
        return Mc, Bc, fc

    Mc, Bc, fc = jax.vmap(_local)(cells_j, signs)  # (nc,3,3), (nc,3), (nc,)

    n = n_edges + n_cells
    A = jnp.zeros((n, n))
    b = jnp.zeros((n,))
    ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)  # (nc, 3)
    cell_dof = n_edges + jnp.arange(n_cells)

    # RT mass block: A[edge_a, edge_b] += Mc
    ia = jnp.broadcast_to(ce[:, :, None], (n_cells, 3, 3)).reshape(-1)
    ib = jnp.broadcast_to(ce[:, None, :], (n_cells, 3, 3)).reshape(-1)
    A = A.at[ia, ib].add(Mc.reshape(-1))
    # coupling: continuity row (cell dof) gets +B; momentum row (edge dof) gets -Bᵀ
    rows = jnp.broadcast_to(cell_dof[:, None], (n_cells, 3)).reshape(-1)
    cols = ce.reshape(-1)
    A = A.at[rows, cols].add(Bc.reshape(-1))
    A = A.at[cols, rows].add(-Bc.reshape(-1))
    b = b.at[cell_dof].add(fc)
    return A, b, top, spec


def assemble_fem_nonnodal(domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, *, flux_bcs=(), quad_degree=4):
    """Native push-forward assembler for non-nodal (RT, ...) fields, driven by the weak-form DSL.

    The n-D analogue of :func:`fem_1d.assemble_fem_1d_multifield`: it lowers each weak term, builds a
    per-cell ``local`` carrying the field's *physical* (push-forward) shape data, and evaluates the term
    through the shared integrand evaluator (:func:`feax_utils._eval_expr_for_feax`, which now has
    space-guarded RT branches). Returns ``(A, b)`` for the linear system (matrices-only contract).

    Scope: RT (H(div)) and N1E (H(curl)) edge-DOF fields plus P0 (cell DOFs) -- the H(div)/H(curl) mass /
    L²-projection, the RT-P0 mixed-Poisson saddle system, the essential normal-flux BC ``u·n = g``
    (``flux_bcs``, pinned via :func:`_apply_flux_bcs`) and the natural pressure BC ``p = p_D``
    (``boundary_terms``, via :func:`_apply_natural_boundary_terms`). RT and N1E share the edge topology and
    DOF map; they differ only in the push-forward (contravariant vs covariant). Dirichlet/IC are not wired;
    the H(curl) curl-curl operator and the tangential BC ``u·t = g`` come next.
    """
    from .feax_utils import _infer_fields, _lower_statefield_to_trial, _test_field_index
    from .fem_1d import _integrate_term
    from .fem_elements import (
        nedelec_triangle,
        piola_contravariant,
        piola_contravariant_grad,
        piola_covariant,
        piola_covariant_grad,
        raviart_thomas_triangle,
    )
    from .fem_topology import build_edge_topology
    from .weak_form import _apply_sign, _split_additive_terms

    if dirichlet_raw or ic_residuals:
        raise NotImplementedError("jno.fem (non-nodal): Dirichlet / IC terms are not wired yet.")

    # --- field layout: RT/N1E (edge DOFs) and/or P0 (cell DOFs) ---
    fields: List[Any] = []
    field_index: dict = {}
    for bare in volume_terms:
        fs, _ = _infer_fields(_lower_statefield_to_trial(bare, {}))
        for f in fs:
            if f["field_key"] not in field_index:
                field_index[f["field_key"]] = len(fields)
                fields.append(f)
    spaces = [f["space"] for f in fields]
    if any(s not in ("RT", "N1E", "P0") for s in spaces):
        raise NotImplementedError(f"jno.fem (non-nodal): supported element spaces are RT, N1E and P0; got {spaces}.")

    # --- mesh + edge element(s) + topology. RT (H(div), contravariant Piola) and N1E (H(curl),
    # covariant Piola) share the edge ordering, topology and global edge DOFs; they differ only in the
    # push-forward and the per-DOF reference shape data, so one dispatch (edge_ref) serves both. ---
    pts = jnp.asarray(np.asarray(domain.mesh.points))[:, :2]
    cells = np.asarray(domain.mesh.cells_dict["triangle"], dtype=np.int64)
    n_cells = int(cells.shape[0])
    cells_j = jnp.asarray(cells, dtype=jnp.int32)
    edge_ref = {}  # family -> (ref_values, ref_diffop, ref_grads, piola_fn, piola_grad_fn)
    specs = {}
    if "RT" in spaces:
        specs["RT"] = raviart_thomas_triangle(degree=1, quad_degree=quad_degree)
        s = specs["RT"]
        edge_ref["RT"] = (s.ref_values, s.ref_div, s.ref_grads, piola_contravariant, piola_contravariant_grad)
    if "N1E" in spaces:
        specs["N1E"] = nedelec_triangle(degree=1, quad_degree=quad_degree)
        s = specs["N1E"]
        edge_ref["N1E"] = (s.ref_values, s.ref_curl, s.ref_grads, piola_covariant, piola_covariant_grad)
    edge_ref = {k: tuple(jnp.asarray(a) for a in v[:3]) + v[3:] for k, v in edge_ref.items()}
    ref_spec = specs.get("RT") or specs.get("N1E") or raviart_thomas_triangle(degree=1, quad_degree=quad_degree)
    top = build_edge_topology(cells, ref_spec.local_edges)
    ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)  # (n_cells, 3) global edge ids
    esigns = jnp.asarray(top.cell_edge_signs.astype(np.float64))  # (n_cells, 3)
    qp, qw = jnp.asarray(ref_spec.quad_points), jnp.asarray(ref_spec.quad_weights)
    n_quad = int(qw.shape[0])
    ctx = getattr(domain, "context", {}) or {}

    # per-field DOF count (RT/N1E -> n_edges, P0 -> n_cells), block offsets, and per-cell global DOF map
    ndof = [top.n_edges if s in ("RT", "N1E") else n_cells for s in spaces]
    offs = [0]
    for n in ndof:
        offs.append(offs[-1] + n)
    total = offs[-1]
    cdofs = [
        (offs[i] + ce) if spaces[i] in ("RT", "N1E") else (offs[i] + jnp.arange(n_cells)[:, None])
        for i in range(len(fields))
    ]  # (n_cells, 3) for RT/N1E, (n_cells, 1) for P0

    # typed terms: (lowered coeff, test field index -> equation block)
    typed = []
    for bare in volume_terms:
        for sign, sub in _split_additive_terms(domain, bare):
            coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
            tfi = _test_field_index(coeff, field_index)
            if tfi is None:
                raise ValueError("jno.fem (non-nodal): each weak term must contain exactly one test field.")
            typed.append((coeff, tfi))

    def _cell_fields(c, u_blocks):
        verts = pts[cells_j[c]]
        J = jnp.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)
        detJ = jnp.linalg.det(J)
        per = []
        for i, s in enumerate(spaces):
            if s in edge_ref:  # RT (contravariant) or N1E (covariant): same edge DOFs, family-specific push-forward
                rval, rdop, rgr, pf, pgf = edge_ref[s]
                phi, _d = pf(rval, rdop, J, detJ, esigns[c])  # (n_quad, 3, 2)
                grad = pgf(rgr, J, detJ, esigns[c])  # (n_quad, 3, 2, 2)
                per.append({"shape_vals": phi, "shape_grads": grad, "cell_sol": u_blocks[i][ce[c]], "space": s})
            else:  # P0: a single constant DOF per cell
                per.append(
                    {
                        "shape_vals": jnp.ones((n_quad, 1)),
                        "shape_grads": jnp.zeros((n_quad, 1, 2, 2)),
                        "cell_sol": u_blocks[i][c][None],
                        "space": "P0",
                    }
                )
        return per, verts[0][None, :] + qp @ J.T, jnp.abs(detJ)

    def residual(u_flat):
        u_blocks = [u_flat[offs[i] : offs[i + 1]] for i in range(len(fields))]
        R = jnp.zeros(total, dtype=u_flat.dtype)
        for coeff, tfi in typed:

            def _cell(c, e=coeff):
                per, xq, meas = _cell_fields(c, u_blocks)
                local = {
                    "physical_quad_points": xq,
                    "fields": per,
                    "field_index": field_index,
                    "tag": "fem_gauss",
                    "surface": False,
                    "domain_context": ctx,
                    "temporal_tags": (),
                    "runtime_parameter_tags": (),
                    "volume_vars": (),
                }
                return _integrate_term(domain, e, local, qw * meas)  # (ndof of the test field,)

            elem = jax.vmap(_cell)(jnp.arange(n_cells))  # (n_cells, ndof_tfi)
            R = R.at[cdofs[tfi].reshape(-1)].add(elem.reshape(-1))
        return R

    zeros = jnp.zeros(total)
    A = jax.jacfwd(residual)(zeros)
    b = -residual(zeros)

    # natural (weak) boundary terms -- for RT the natural pressure BC p_D*(v·n) (contributes to b)
    if boundary_terms:
        b = _apply_natural_boundary_terms(
            b, boundary_terms, domain, field_index, spaces, top, np.asarray(pts), offs, n_cells, quad_degree
        )
    # essential normal-flux BCs (u·n = g): pin boundary-edge DOFs, then symmetric-eliminate
    if flux_bcs:
        A, b = _apply_flux_bcs(
            A, b, flux_bcs, domain, field_index, spaces, top, np.asarray(pts), offs, n_cells, quad_degree
        )
    return (A, b), "linear", offs


def _apply_natural_boundary_terms(b, boundary_terms, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree):
    """Assemble RT natural (weak) boundary terms into ``b``. Supports the natural pressure BC
    ``p_D · (v·n)`` (mixed Poisson with prescribed ``p = p_D``): in the momentum residual this is
    ``+∮ p_D (v·n) ds``, and since the RT0 basis has ``v_e·n`` constant on its own edge it reduces to
    ``b[edge_e] += sign_topo · avg_edge(p_D)`` (sign validated empirically; the ``1/L`` density cancels
    the edge integral, leaving the average). Other weak boundary forms (e.g. Robin) raise."""
    from ..._fem import _bare, _contains, _eval_value_node_at, _walk
    from ...trace import BinaryOp, TestFunction, Variable
    from .fem_1d import _line_quadrature, _region_node_ids

    cell_edges = np.asarray(top.cell_edges)
    counts = np.bincount(cell_edges.reshape(-1), minlength=top.n_edges)
    boundary = {int(e) for e in np.where(counts == 1)[0]}
    loc = {int(cell_edges[c, k]): (c, k) for c in range(n_cells) for k in range(3) if int(cell_edges[c, k]) in boundary}
    gp, gw = (np.asarray(x).reshape(-1) for x in _line_quadrature(quad_degree))

    b = np.asarray(b).copy()
    for region, terms in boundary_terms.items():
        region_nodes = {int(n) for n in _region_node_ids(domain, region)}
        for term in terms:
            bare = _bare(term)
            # recognise p_D * (v·n): a product with the test on one side, p_D on the other
            ok = isinstance(bare, BinaryOp) and bare.op == "*"
            if ok and _contains(bare.left, TestFunction) and not _contains(bare.right, TestFunction):
                vn_side, pd_node = bare.left, bare.right
            elif ok and _contains(bare.right, TestFunction) and not _contains(bare.left, TestFunction):
                vn_side, pd_node = bare.right, bare.left
            else:
                raise NotImplementedError(
                    "jno.fem (non-nodal): only the natural pressure BC `p_D * (v·n)` weak boundary term is "
                    "supported on an RT field (Robin / general surface terms are not wired yet)."
                )
            walked = list(_walk(vn_side))
            if not any(isinstance(n, Variable) and str(getattr(n, "tag", "")).startswith("n_") for n in walked):
                raise NotImplementedError("jno.fem (non-nodal): expected a normal projection `v·n` in the boundary term.")
            fkeys = {n.field_key for n in walked if isinstance(n, TestFunction)}
            fidx = field_index.get(next(iter(fkeys))) if fkeys else None
            if fidx is None or spaces[fidx] != "RT":
                raise NotImplementedError("jno.fem (non-nodal): a natural p_D*(v·n) BC is only supported on an RT field.")
            for eid in boundary:
                va, vb = (int(x) for x in top.edge_vertices[eid])
                if va not in region_nodes or vb not in region_nodes:
                    continue
                c, k = loc[eid]
                pa, pb = pts_np[va], pts_np[vb]
                xq = pa[None, :] * (1.0 - gp[:, None]) + pb[None, :] * gp[:, None]
                pd = np.asarray(_eval_value_node_at(pd_node, jnp.asarray(xq))).reshape(-1)
                b[offs[fidx] + eid] += int(top.cell_edge_signs[c, k]) * float(np.sum(gw * pd))  # sign * avg_edge(p_D)
    return jnp.asarray(b)


def _apply_flux_bcs(A, b, flux_bcs, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree):
    """Pin boundary-edge DOFs for essential normal-flux BCs ``u·n = g``, then symmetric-eliminate.

    The RT0 edge DOF *is* the edge normal flux, so the pin (orientation sign locked empirically) is
    ``σ_e = -sign_topo · ∫_edge g ds`` with ``sign_topo = top.cell_edge_signs[c, k]`` for the boundary
    edge's single incident cell. Boundary edges are the globally single-use edges, filtered to the BC's
    region by node membership. ``g`` is constant for now (general ``g(x)`` is a later extension)."""
    from ..._fem import _eval_value_node_at
    from .fem_1d import _apply_dirichlet_symmetric, _line_quadrature, _region_node_ids

    cell_edges = np.asarray(top.cell_edges)
    counts = np.bincount(cell_edges.reshape(-1), minlength=top.n_edges)
    boundary = {int(e) for e in np.where(counts == 1)[0]}
    loc = {}  # boundary edge id -> (cell, local k) of its single incident cell
    for c in range(n_cells):
        for k in range(3):
            eid = int(cell_edges[c, k])
            if eid in boundary:
                loc[eid] = (c, k)

    gp, gw = (np.asarray(x).reshape(-1) for x in _line_quadrature(quad_degree))  # 1-D Gauss on [0, 1]
    pins = []
    for field_key, region, value_node in flux_bcs:
        fidx = field_index.get(field_key)
        if fidx is None or spaces[fidx] != "RT":
            raise NotImplementedError("jno.fem (non-nodal): a normal-flux BC is only supported on an RT field.")
        region_nodes = {int(n) for n in _region_node_ids(domain, region)}
        for eid in boundary:
            va, vb = (int(x) for x in top.edge_vertices[eid])
            if va not in region_nodes or vb not in region_nodes:
                continue
            c, k = loc[eid]
            pa, pb = pts_np[va], pts_np[vb]
            length = float(np.linalg.norm(pb - pa))
            xq = pa[None, :] * (1.0 - gp[:, None]) + pb[None, :] * gp[:, None]  # physical edge quad points
            g_vals = np.asarray(_eval_value_node_at(value_node, jnp.asarray(xq))).reshape(-1)
            moment = length * float(np.sum(gw * g_vals))  # ∫_edge g ds
            pins.append((offs[fidx] + eid, -int(top.cell_edge_signs[c, k]) * moment))
    return _apply_dirichlet_symmetric(jnp.asarray(A), jnp.asarray(b), pins)


def rt_flux_at_centroids(points: np.ndarray, cells: np.ndarray, top: EdgeTopology, u_edge: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the RT flux field ``u_h`` at each triangle centroid -> ``(n_cells, 2)``.

    Tabulates the RT basis once at the reference centroid ``(1/3, 1/3)``, Piola-maps it
    per cell (with the edge-orientation signs used in assembly), and contracts with the
    cell's three edge-DOF coefficients ``u_edge[cell_edges]``.
    """
    import basix

    elem = basix.create_element(basix.ElementFamily.RT, basix.CellType.triangle, 1)
    tab = elem.tabulate(1, np.array([[1.0 / 3.0, 1.0 / 3.0]]))  # (3, 1, 3, 2)
    rv = jnp.asarray(tab[0])  # (1, 3, 2)
    rd = jnp.asarray(tab[1][:, :, 0] + tab[2][:, :, 1])  # (1, 3)
    pts = jnp.asarray(points)
    cells_j = jnp.asarray(np.asarray(cells), dtype=jnp.int32)
    signs = jnp.asarray(top.cell_edge_signs.astype(np.float64))
    ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)
    coeffs = u_edge[ce]  # (n_cells, 3)

    def _flux(cell, sgn, c):
        J, detJ = _cell_jacobian(pts[cell])
        phi, _ = piola_contravariant(rv, rd, J, detJ, sgn)  # (1, 3, 2)
        return jnp.einsum("a,ad->d", c, phi[0])  # (2,)

    return jax.vmap(_flux)(cells_j, signs, coeffs)


def n1e_field_at_centroids(points: np.ndarray, cells: np.ndarray, top: EdgeTopology, u_edge: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Nédélec (H(curl)) field ``u_h`` at each triangle centroid -> ``(n_cells, 2)``.

    The H(curl) counterpart of :func:`rt_flux_at_centroids`: tabulates N1E at the reference centroid,
    covariant-Piola-maps it per cell (with the edge-orientation signs used in assembly), and contracts
    with the cell's three edge-DOF coefficients ``u_edge[cell_edges]``.
    """
    import basix

    from .fem_elements import piola_covariant

    elem = basix.create_element(basix.ElementFamily.N1E, basix.CellType.triangle, 1)
    tab = elem.tabulate(1, np.array([[1.0 / 3.0, 1.0 / 3.0]]))  # (3, 1, 3, 2)
    rv = jnp.asarray(tab[0])  # (1, 3, 2)
    rc = jnp.asarray(tab[1][:, :, 1] - tab[2][:, :, 0])  # (1, 3) reference curl d Phi_y/dxi0 - d Phi_x/dxi1
    pts = jnp.asarray(points)
    cells_j = jnp.asarray(np.asarray(cells), dtype=jnp.int32)
    signs = jnp.asarray(top.cell_edge_signs.astype(np.float64))
    ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)
    coeffs = u_edge[ce]  # (n_cells, 3)

    def _val(cell, sgn, c):
        J, detJ = _cell_jacobian(pts[cell])
        phi, _ = piola_covariant(rv, rc, J, detJ, sgn)  # (1, 3, 2)
        return jnp.einsum("a,ad->d", c, phi[0])  # (2,)

    return jax.vmap(_val)(cells_j, signs, coeffs)
