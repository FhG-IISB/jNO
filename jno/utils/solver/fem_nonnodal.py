"""Native assembler for non-nodal (push-forward) element families.

n-D assembler for derivative/edge-DOF element families, which need a per-cell
push-forward. The element zoo assembles from :mod:`fem_topology` (global edge
numbering + orientation) and :mod:`fem_elements` (basix reference tabulation +
per-cell push-forward).

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
    through the shared integrand evaluator (:func:`fem_utils._eval_integrand`, which now has
    space-guarded RT branches). Returns ``(A, b)`` for the linear system (matrices-only contract).

    Scope: RT (H(div)) and N1E (H(curl)) edge-DOF fields plus P0 (cell DOFs) -- the H(div)/H(curl) mass /
    L²-projection, the RT-P0 mixed-Poisson saddle system, the essential normal-flux BC ``u·n = g``
    (``flux_bcs``, pinned via :func:`_apply_flux_bcs`) and the natural pressure BC ``p = p_D``
    (``boundary_terms``, via :func:`_apply_natural_boundary_terms`). RT and N1E share the edge topology and
    DOF map; they differ only in the push-forward (contravariant vs covariant). Dirichlet/IC are not wired;
    the H(curl) curl-curl operator and the tangential BC ``u·t = g`` come next.
    """
    from ...trace import FemResidualOperator
    from .fem_1d import _apply_dirichlet_rows, _apply_dirichlet_symmetric, _integrate_term
    from .fem_elements import (
        nedelec_triangle,
        piola_contravariant,
        piola_contravariant_grad,
        piola_covariant,
        piola_covariant_grad,
        raviart_thomas_triangle,
    )
    from .fem_topology import build_edge_topology
    from .fem_utils import _infer_fields, _lower_statefield_to_trial, _test_field_index
    from .weak_form import (
        _apply_sign,
        _contains_temporal_derivative,
        _is_obviously_nonlinear_in_unknown,
        _split_additive_terms,
    )

    if dirichlet_raw:
        raise NotImplementedError(
            "jno.fem (non-nodal): nodal Dirichlet is not applicable to RT/N1E; the essential BC is the "
            "edge trace (u·n for RT, u×n for N1E) -- write it as `dot(u(region), n_region) - g`."
        )

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

    def _make_residual(terms):
        """Build a ``residual(u_flat)`` closure for the given weak ``terms`` over the shared field
        layout. Reused for the steady system (``jacfwd`` -> A, ``-residual(0)`` -> b), the steady
        nonlinear :class:`FemResidualOperator`, and the transient mass/spatial split (mirrors the 1D
        :func:`fem_1d._make_residual`)."""
        typed = []  # (lowered coeff, test field index -> equation block)
        for bare in terms:
            for sign, sub in _split_additive_terms(domain, bare):
                coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
                tfi = _test_field_index(coeff, field_index)
                if tfi is None:
                    raise ValueError("jno.fem (non-nodal): each weak term must contain exactly one test field.")
                typed.append((coeff, tfi))

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

        return residual

    # --- BCs computed once, separate from per-mode application: the natural BC (p_D·(v·n)) is a constant
    #     RHS load; the essential edge-trace BC is a set of (dof, value) pins (mirrors 1D Dirichlet). ---
    nat_load = (
        _apply_natural_boundary_terms(
            jnp.zeros(total), boundary_terms, domain, field_index, spaces, top, np.asarray(pts), offs, n_cells, quad_degree
        )
        if boundary_terms
        else jnp.zeros(total)
    )
    pins = (
        _flux_bc_pins(flux_bcs, domain, field_index, spaces, top, np.asarray(pts), offs, n_cells, quad_degree)
        if flux_bcs
        else []
    )
    zeros = jnp.zeros(total)

    # === transient: M u̇ + A u = c (mirrors fem_1d._assemble_1d_transient) -- split the temporal term
    #     (∫ ∂ₜu·v -> mass M) from the spatial operator, project the IC onto the edge DOFs, time-block it. ===
    if ic_residuals or any(_contains_temporal_derivative(t) for t in volume_terms):
        from ..._fem import _bare, _essential_spec, _eval_value_node_at, _field_key_of
        from .backend_blocks import SemidiscreteTimeBlock
        from .fem_1d import _apply_dirichlet_transient
        from .time_route import _infer_time_window, _strip_temporal_trial_derivative

        sub = [_apply_sign(domain, s, t) for bare in volume_terms for s, t in _split_additive_terms(domain, bare)]
        temporal = [t for t in sub if _contains_temporal_derivative(t)]
        spatial = [t for t in sub if not _contains_temporal_derivative(t)]
        if not temporal:
            raise ValueError("jno.fem (non-nodal): a transient weak form must contain a temporal term, e.g. inner(u.t, v).")
        M = jax.jacfwd(_make_residual([_strip_temporal_trial_derivative(t) for t in temporal]))(zeros)  # block mass
        spatial_res = _make_residual(spatial)
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

        # Initial state: L²-project each IC field's u0 onto its edge DOFs by solving that field's *mass
        # block* M[blk, blk]. The full M is singular for a mixed/saddle transient (the algebraic field --
        # e.g. RT flux in Darcy -- carries no ∂ₜ), so projecting per field avoids that; algebraic fields
        # with no IC stay 0 and the implicit first step recovers their constraint-consistent value.
        def _project_ic(ic):
            _comp, u0_node = _essential_spec(_bare(ic))
            fidx = field_index.get(_field_key_of(ic))
            if fidx is None:
                raise ValueError("jno.fem (non-nodal): the initial condition does not match any field.")
            u0_blocks = [jnp.zeros(offs[i + 1] - offs[i]) for i in range(len(fields))]

            def _ic_cell(cidx):
                per, xq, meas = _cell_fields(cidx, u0_blocks)
                phi = per[fidx]["shape_vals"]
                u0 = jnp.asarray(_eval_value_node_at(u0_node, xq))
                if phi.ndim == 3:  # RT/N1E vector basis (n_quad, n_dof, vsize): ∫ u0·Φ
                    return jnp.einsum("q,qnc,qc->n", qw * meas, phi, u0.reshape(n_quad, -1))
                return jnp.einsum("q,qn,q->n", qw * meas, phi, u0.reshape(n_quad))  # P0 scalar basis: ∫ u0 q

            local = (cdofs[fidx] - offs[fidx]).reshape(-1)  # field-local DOFs (ce for RT/N1E, cell index for P0)
            load = jnp.zeros(offs[fidx + 1] - offs[fidx]).at[local].add(jax.vmap(_ic_cell)(jnp.arange(n_cells)).reshape(-1))
            sl = slice(offs[fidx], offs[fidx + 1])
            return fidx, jnp.linalg.solve(M[sl, sl], load)  # the differential field's mass block is non-singular

        state0 = zeros
        for ic in ic_residuals:
            fidx, block_sol = _project_ic(ic)
            state0 = state0.at[offs[fidx] : offs[fidx + 1]].set(block_sol)
        common["state0"] = state0

        if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial):
            # nonlinear transient: M(t) u̇ + R(u) = 0 (matrix-free Newton-Krylov per step). Natural load
            # folds into R; essential pins are residual rows + zeroed M rows/cols (the 1D pattern).
            res_bc = _apply_dirichlet_rows(lambda u: spatial_res(u) - nat_load, pins)
            jac = jax.jacfwd(res_bc)
            pin_dofs = jnp.asarray([p[0] for p in pins], dtype=jnp.int32) if pins else None
            M_nl = M if pin_dofs is None else M.at[pin_dofs, :].set(0.0).at[:, pin_dofs].set(0.0)
            block = SemidiscreteTimeBlock(
                mass=lambda t, args=None, _M=M_nl: _M,
                residual=lambda u, t, args=None: res_bc(u),
                jacobian=lambda u, t, args=None: jac(u),
                **common,
            )
            return block, "transient", offs

        # linear transient: M u̇ + A u = c
        A = jax.jacfwd(spatial_res)(zeros)
        c = -spatial_res(zeros) + nat_load  # spatial load + natural-BC constant load
        M, A, c = _apply_dirichlet_transient(M, A, c, pins)  # essential edge-trace pins -> M/A/c rows
        return SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, **common), "transient", offs

    residual = _make_residual(volume_terms)

    def full_residual(u_flat):  # the natural BC is a constant load on the RHS: R(u) = assembled(u) - load
        return residual(u_flat) - nat_load

    # --- steady nonlinear: a genuinely nonlinear weak term -> a Newton residual operator ---
    if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in volume_terms):
        res_bc = _apply_dirichlet_rows(full_residual, pins)  # essential pins as residual rows R[d]=u[d]-g
        jac = jax.jacfwd(res_bc)
        # FemResidualOperator.solve calls residual(u, args)/jacobian(u, args); the non-nodal residual has no
        # runtime parameters, so accept-and-ignore args (args=None keeps `fem.residual(u)` single-arg too).
        return (
            FemResidualOperator(lambda u, args=None: res_bc(u), lambda u, args=None: jac(u), total),
            "nonlinear",
            offs,
        )

    # --- steady linear: A u = b (byte-identical to before the refactor) ---
    A = jax.jacfwd(full_residual)(zeros)
    b = -full_residual(zeros)
    if pins:
        A, b = _apply_dirichlet_symmetric(jnp.asarray(A), jnp.asarray(b), pins)
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


def _flux_bc_pins(flux_bcs, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree):
    """Compute the ``(dof, value)`` boundary pins for essential edge-trace BCs, separated from
    *application* so the same pins can be enforced per solver mode (symmetric elimination for steady
    linear, residual rows for nonlinear, M/A/c rows for transient).

    The lowest-order edge DOF *is* an edge moment, so the BC is a value pin ``σ_e = sgn · ∫_edge g ds``
    (``∫_edge g`` via 1-D edge quadrature; ``g`` may be constant or ``g(x)``). The trace and its sign
    depend on the family:

    * **RT (H(div))** — normal flux ``u·n = g``; ``σ_e = -sign_topo · ∫g`` with
      ``sign_topo = top.cell_edge_signs[c, k]`` (locked empirically).
    * **N1E (H(curl))** — tangential trace ``u×n = g`` (the canonical 2-D tangential component via the
      *outward* normal). The DOF is the edge-topological tangential moment ``∫ u·t_topo``, so the sign
      reconciles ``t_topo`` (low→high vertex) with ``(n_y, -n_x)``: ``sgn`` = orientation of the +90°
      rotation of the edge vector relative to the outward direction (away from the opposite vertex).
      Derived geometrically and checked against the (exact) projection of a constant field on every
      boundary edge.

    Boundary edges are the globally single-use edges, filtered to the BC's region by node membership."""
    from ..._fem import _eval_value_node_at
    from .fem_1d import _line_quadrature, _region_node_ids

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
        if fidx is None or spaces[fidx] not in ("RT", "N1E"):
            raise NotImplementedError(
                "jno.fem (non-nodal): an essential edge-trace BC is supported on RT (u·n) and N1E (u×n) only."
            )
        is_n1e = spaces[fidx] == "N1E"
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
            if is_n1e:  # tangential trace: reconcile the edge tangent with (n_y, -n_x) via the outward direction
                vc = (set(int(x) for ek in cell_edges[c] for x in top.edge_vertices[ek]) - {va, vb}).pop()
                rot90 = np.array([-(pb[1] - pa[1]), pb[0] - pa[0]])  # +90° rotation of the edge vector
                sgn = 1.0 if float(np.dot(rot90, 0.5 * (pa + pb) - pts_np[vc])) > 0 else -1.0
            else:  # RT normal flux
                sgn = -float(top.cell_edge_signs[c, k])
            pins.append((offs[fidx] + eid, sgn * moment))
    return pins


def _apply_flux_bcs(A, b, flux_bcs, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree):
    """Steady-linear application of the essential edge-trace pins: symmetric elimination on ``(A, b)``."""
    from .fem_1d import _apply_dirichlet_symmetric

    pins = _flux_bc_pins(flux_bcs, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree)
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
