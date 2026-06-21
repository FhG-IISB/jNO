"""Native assembler for non-nodal (push-forward) element families.

This is the n-D analogue of the native 1D path (:mod:`fem_1d`): feax cannot
assemble derivative/edge-DOF elements (it has no push-forward), so the element
zoo assembles on the jNO side from :mod:`fem_topology` (global edge numbering +
orientation) and :mod:`fem_elements` (basix reference tabulation + per-cell
push-forward).

The first entry point is the **RT–P0 mixed Poisson** system — the canonical
H(div) test problem — assembled directly (not yet via the weak-form DSL) so the
engine (edge DOFs, orientation, contravariant Piola, divergence, saddle-block
assembly) can be validated end-to-end against a manufactured solution by
*convergence rate*. The DSL/``fem_symbols(space=...)`` routing builds on this
once the engine is proven.

Mixed Poisson ``u = -∇p``, ``div u = f`` with ``p = 0`` on ∂Ω (Dirichlet on ``p``
is *natural* in the mixed form → no essential flux BC). Flux ``u ∈ RT``, scalar
``p ∈ P0``; weak form ``∫u·v − ∫p div v = 0`` ∀v∈RT, ``∫q div u = ∫f q`` ∀q∈P0.
Global DOFs are ``[edge DOFs (n_edges)] ++ [cell DOFs (n_cells)]`` and the block
system is ``[[M, −Bᵀ], [B, 0]] [u; p] = [0; f]``.
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


def assemble_fem_nonnodal(domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, *, quad_degree=4):
    """Native push-forward assembler for non-nodal (RT, ...) fields, driven by the weak-form DSL.

    The n-D analogue of :func:`fem_1d.assemble_fem_1d_multifield`: it lowers each weak term, builds a
    per-cell ``local`` carrying the field's *physical* (push-forward) shape data, and evaluates the term
    through the shared integrand evaluator (:func:`feax_utils._eval_expr_for_feax`, which now has
    space-guarded RT branches). Returns ``(A, b)`` for the linear system (matrices-only contract).

    Scope (step #2a): a single RT field, volume terms only (the H(div) mass / L²-projection system); no
    essential BC. P0, the expanded ``div``, multifield coupling and BCs come next (#2b).
    """
    from .feax_utils import _infer_fields, _lower_statefield_to_trial
    from .fem_1d import _integrate_term
    from .fem_elements import piola_contravariant, piola_contravariant_grad, raviart_thomas_triangle
    from .fem_topology import build_edge_topology
    from .weak_form import _apply_sign, _split_additive_terms

    # --- field layout (single RT field for now) ---
    fields: List[Any] = []
    field_index: dict = {}
    for bare in volume_terms:
        fs, _ = _infer_fields(_lower_statefield_to_trial(bare, {}))
        for f in fs:
            if f["field_key"] not in field_index:
                field_index[f["field_key"]] = len(fields)
                fields.append(f)
    if boundary_terms or dirichlet_raw or ic_residuals:
        raise NotImplementedError("jno.fem (non-nodal): boundary / Dirichlet / IC terms are not wired yet (#2b).")
    if len(fields) != 1 or fields[0]["space"] != "RT":
        raise NotImplementedError(
            "jno.fem (non-nodal): only a single RT field (volume terms) is wired so far; "
            f"got fields {[f['space'] for f in fields]}."
        )

    # --- mesh + RT element + edge topology ---
    pts = jnp.asarray(np.asarray(domain.mesh.points))[:, :2]
    cells = np.asarray(domain.mesh.cells_dict["triangle"], dtype=np.int64)
    spec = raviart_thomas_triangle(degree=1, quad_degree=quad_degree)
    top = build_edge_topology(cells, spec.local_edges)
    n_dofs = top.n_edges
    cells_j = jnp.asarray(cells, dtype=jnp.int32)
    ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)  # (n_cells, 3) global edge DOFs
    signs = jnp.asarray(top.cell_edge_signs.astype(np.float64))  # (n_cells, 3)
    qp, qw = jnp.asarray(spec.quad_points), jnp.asarray(spec.quad_weights)
    rv, rd, rg = jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_div), jnp.asarray(spec.ref_grads)
    ctx = getattr(domain, "context", {}) or {}

    # lower each weak term (sign-split) into evaluable expressions
    terms = [
        _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
        for bare in volume_terms
        for sign, sub in _split_additive_terms(domain, bare)
    ]

    def _cell_residual(c, u_edge):
        verts = pts[cells_j[c]]  # (3, 2)
        J = jnp.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)
        detJ = jnp.linalg.det(J)
        sgn = signs[c]
        phi, _div = piola_contravariant(rv, rd, J, detJ, sgn)  # (n_quad, 3, 2)
        grad = piola_contravariant_grad(rg, J, detJ, sgn)  # (n_quad, 3, 2, 2)
        local = {
            "physical_quad_points": verts[0][None, :] + qp @ J.T,  # (n_quad, 2)
            "fields": [{"shape_vals": phi, "shape_grads": grad, "cell_sol": u_edge[ce[c]], "space": "RT"}],
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": ctx,
            "temporal_tags": (),
            "runtime_parameter_tags": (),
            "volume_vars": (),
        }
        out = jnp.zeros(spec.n_dof)
        for expr in terms:
            out = out + _integrate_term(domain, expr, local, qw * jnp.abs(detJ))  # (3,)
        return out

    def residual(u_flat):
        elem = jax.vmap(lambda c: _cell_residual(c, u_flat))(jnp.arange(cells.shape[0]))  # (n_cells, 3)
        return jnp.zeros(n_dofs, dtype=u_flat.dtype).at[ce.reshape(-1)].add(elem.reshape(-1))

    zeros = jnp.zeros(n_dofs)
    A = jax.jacfwd(residual)(zeros)
    b = -residual(zeros)
    return (A, b), "linear"


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
