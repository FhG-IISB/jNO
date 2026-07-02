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


def assemble_fem_nonnodal(
    domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, *, flux_bcs=(), rotation_bcs=(), quad_degree=4
):
    """Native push-forward assembler for non-nodal (RT, ...) fields, driven by the weak-form DSL.

    It lowers each weak term, builds a per-cell ``local`` carrying the field's *physical* (push-forward)
    shape data, and evaluates it through the shared integrand evaluator (:func:`fem_utils._eval_integrand`).
    Returns ``(op, mode, offsets)``: ``mode`` is ``"linear"`` (``op = (A, b)`` or a parametric
    :class:`FemLinearSystem`), ``"nonlinear"`` (a :class:`FemResidualOperator`) or ``"transient"`` (a
    :class:`SemidiscreteTimeBlock`).

    Families (``fem_symbols(space=...)``):

    * **Edge DOFs** -- RT (H(div)) and N1E (H(curl)), plus P0 (cell DOFs): the H(div)/H(curl) mass /
      L²-projection, the RT-P0 mixed-Poisson saddle system, the essential normal-flux BC ``u·n = g``
      (``flux_bcs``), and the natural pressure BC ``p = p_D``. (The H(curl) curl-curl operator and the
      tangential BC ``u·t = g`` are still to come.)
    * **Vertex DOFs** -- Hermite (C⁰, value+∇) and **Argyris (C¹ conforming, for the biharmonic)**: the
      per-cell ``M(cell)`` DOF-transform path (:func:`fem_elements.hermite_pushforward` /
      :func:`argyris_pushforward`). Hermite takes a value-Dirichlet; Argyris takes the **proper clamped BC**
      ``u=g, ∂u/∂n=∂g/∂n`` with free boundary curvature (:func:`_argyris_dirichlet_pins`).

    All families share the solver modes: steady linear/nonlinear, first- and (single-field) **second-order**
    (``u_tt`` -> augmented ``[u, v]`` block) transient with IC L²-projection, and the differentiable
    **inverse** path -- a runtime scalar *or* P1 field parameter ``k(x)`` in a volume term is threaded so
    ``crux.solve`` recovers it (steady and transient). A parameter in a boundary term is rejected (the
    natural-BC load is non-differentiable).
    """
    from ...trace import FemResidualOperator
    from .fem_1d import _apply_dirichlet_rows, _apply_dirichlet_symmetric, _integrate_term
    from .fem_elements import (
        argyris_pushforward,
        argyris_triangle,
        hermite_pushforward,
        hermite_triangle,
        morley_pushforward,
        morley_triangle,
        nedelec_tet,
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
    if any(s not in ("RT", "N1E", "P0", "Hermite", "Argyris", "Morley") for s in spaces):
        raise NotImplementedError(
            f"jno.fem (non-nodal): supported element spaces are RT, N1E, P0, Hermite, Argyris and Morley; got {spaces}."
        )
    has_edge = ("RT" in spaces) or ("N1E" in spaces)
    has_hermite = "Hermite" in spaces
    has_argyris = "Argyris" in spaces
    has_morley = "Morley" in spaces  # non-conforming biharmonic (vertex value + edge-normal DOFs)
    if has_morley and any(s != "Morley" for s in spaces):
        raise NotImplementedError("jno.fem (non-nodal): a Morley field cannot be mixed with other element families.")
    # vertex-DOF families (take a nodal/derivative Dirichlet): the M(cell) DOF-transform path
    has_vertex = has_hermite or has_argyris or has_morley
    if has_edge and (has_hermite or has_argyris):
        raise NotImplementedError(
            "jno.fem (non-nodal): mixing edge (RT/N1E) and vertex (Hermite/Argyris) fields is not supported."
        )
    if has_hermite and has_argyris:
        raise NotImplementedError("jno.fem (non-nodal): mixing Hermite and Argyris fields is not supported.")
    if dirichlet_raw and not has_vertex:
        raise NotImplementedError(
            "jno.fem (non-nodal): nodal Dirichlet is not applicable to RT/N1E; the essential BC is the "
            "edge trace (u·n for RT, u×n for N1E) -- write it as `dot(u(region), n_region) - g`. "
            "(A Hermite field DOES take a nodal value Dirichlet u(region) - g.)"
        )

    # --- runtime parameters (inverse problems): collect the parameter name->expr map so the assembler can
    # return a *parametric* FemLinearSystem / FemResidualOperator / time block whose operator is re-assembled
    # at each args and stays differentiable in the parameter (mirrors the native path). Both SCALAR parameters
    # and a spatially-varying **P1 field** parameter k(x) are supported: a field parameter is gathered at the
    # mesh vertices (`cells[c]`) and interpolated with P1 shape functions at the quad points -- independent of
    # the non-nodal trial's own DOF layout (the parameter carries its own P1 field via ``_fem_field_domain``). ---
    from .parametric_helpers import _collect_runtime_parameter_exprs, _is_fem_field_parameter

    _rt_param_exprs: dict = {}
    for bare in volume_terms:
        _collect_runtime_parameter_exprs(bare, _rt_param_exprs)
    runtime_parameter_tags: Tuple[str, ...] = tuple(sorted(_rt_param_exprs))
    _field_param_names = frozenset(n for n, e in _rt_param_exprs.items() if _is_fem_field_parameter(e))
    # A parameter in a boundary (Neumann/Robin) term would silently bake to a constant: the natural-BC load
    # is assembled once, non-differentiably (`_apply_natural_boundary_terms`). Reject it rather than mislead.
    _bdry_param_exprs: dict = {}
    for _terms in boundary_terms.values():
        for bare in _terms:
            _collect_runtime_parameter_exprs(bare, _bdry_param_exprs)
    if _bdry_param_exprs:
        raise NotImplementedError(
            "jno.fem (non-nodal): a runtime parameter in a boundary (Neumann/Robin) term is not supported for "
            "inverse problems -- the natural-BC load is assembled non-differentiably. Put the parameter in a "
            "volume term."
        )

    # Simplex dimension from the mesh: 2D triangle vs 3D tetrahedron. The edge (RT/N1E) push-forward and
    # topology are dimension-agnostic; the vertex families (Hermite/Argyris/Morley) and RT are 2D-only, so
    # in 3D only Nédélec (N1E, edge/H(curl)) is wired -- everything else raises rather than silently mis-map.
    dim = 3 if "tetra" in domain.mesh.cells_dict else 2
    if dim == 3 and any(s != "N1E" for s in spaces):
        raise NotImplementedError(
            "jno.fem (non-nodal): on a 3D (tetrahedral) mesh only the Nédélec `N1E` element is supported "
            f"(H(curl), for Maxwell / curl-curl); got spaces {spaces}. RT / P0 / Hermite / Argyris / Morley "
            "are 2D-triangle only."
        )
    if dim == 3 and _field_param_names:
        raise NotImplementedError(
            "jno.fem (non-nodal): a field parameter k(x) in a 3D N1E problem is not wired yet "
            "(the P1 interpolation of the parameter is 2D-triangle only)."
        )
    pts = jnp.asarray(np.asarray(domain.mesh.points))[:, :dim]
    cells = np.asarray(domain.mesh.cells_dict["tetra" if dim == 3 else "triangle"], dtype=np.int64)
    n_cells = int(cells.shape[0])
    n_verts = int(pts.shape[0])
    cells_j = jnp.asarray(cells, dtype=jnp.int32)

    # --- edge families (RT/N1E): contravariant/covariant Piola over a shared edge topology (one
    # ``edge_ref`` dispatch serves both -- same edge DOFs/topology, family-specific push-forward) ---
    edge_ref = {}  # family -> (ref_values, ref_diffop, ref_grads, piola_fn, piola_grad_fn)
    specs = {}
    if "RT" in spaces:
        specs["RT"] = raviart_thomas_triangle(degree=1, quad_degree=quad_degree)
        s = specs["RT"]
        edge_ref["RT"] = (s.ref_values, s.ref_div, s.ref_grads, piola_contravariant, piola_contravariant_grad)
    if "N1E" in spaces:
        specs["N1E"] = (
            nedelec_tet(degree=1, quad_degree=quad_degree)
            if dim == 3
            else nedelec_triangle(degree=1, quad_degree=quad_degree)
        )
        s = specs["N1E"]
        edge_ref["N1E"] = (s.ref_values, s.ref_curl, s.ref_grads, piola_covariant, piola_covariant_grad)
    # ``ref_curl`` is ``None`` for the 3-D tet N1E (its curl is a vector recovered from the physical
    # gradient, not a tabulated scalar) -- keep it ``None`` through the jnp conversion.
    edge_ref = {k: tuple(jnp.asarray(a) if a is not None else None for a in v[:3]) + v[3:] for k, v in edge_ref.items()}

    # --- vertex-DOF families (the M(cell) DOF-transform path): Hermite (C0) and Argyris (C1) ---
    hermite_ref = None
    if has_hermite:
        hs = hermite_triangle(quad_degree=max(quad_degree, 6))  # cubic mass needs a degree-6 rule
        hermite_ref = (jnp.asarray(hs.ref_values), jnp.asarray(hs.ref_grads), jnp.asarray(hs.ref_hess))
    argyris_ref = None
    if has_argyris:
        as_ = argyris_triangle(quad_degree=max(quad_degree, 12))  # quintic mass needs a degree-10+ rule
        argyris_ref = (
            jnp.asarray(as_.ref_values),
            jnp.asarray(as_.ref_grads),
            jnp.asarray(as_.ref_hess),
            tuple(jnp.asarray(a) for a in as_.ref_aux),  # (nv_val, nv_grad, nv_hess, ne_grad) for M(cell)
        )
    morley_ref = None
    if has_morley:
        ms_ = morley_triangle(quad_degree=max(quad_degree, 4))  # quadratic mass needs a degree-4 rule
        morley_ref = (
            jnp.asarray(ms_.ref_values),
            jnp.asarray(ms_.ref_grads),
            jnp.asarray(ms_.ref_hess),
            tuple(jnp.asarray(a) for a in ms_.ref_aux),  # (nv_val, ne_grad) for M(cell)
        )

    # Shared quadrature: a vertex-family problem uses its element rule; an edge/P0 problem uses the edge rule.
    ref_spec = (
        as_
        if has_argyris
        else ms_
        if has_morley
        else hs
        if has_hermite
        else (specs.get("RT") or specs.get("N1E") or raviart_thomas_triangle(1, quad_degree))
    )
    qp, qw = jnp.asarray(ref_spec.quad_points), jnp.asarray(ref_spec.quad_weights)
    n_quad = int(qw.shape[0])
    ctx = getattr(domain, "context", {}) or {}
    # P1 (linear) shape functions at the quad points -- used ONLY to interpolate a P1 field parameter k(x)
    # (its 3 mesh-vertex values per cell), independent of the trial element's own basis.
    p1_shape_vals = jnp.stack([1.0 - qp[:, 0] - qp[:, 1], qp[:, 0], qp[:, 1]], axis=1) if _field_param_names else None

    # Edge topology: edge families (RT/N1E) need it for their edge DOFs; Argyris/Morley need it for the global
    # id + orientation of their edge-normal DOFs (Hermite, pure-vertex, does not).
    if has_edge or has_argyris or has_morley:
        top = build_edge_topology(cells, ref_spec.local_edges)
        ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)  # (n_cells, 3) global edge ids
        esigns = jnp.asarray(top.cell_edge_signs.astype(np.float64)) if has_edge else None  # (n_cells, 3)
        n_edges = int(top.n_edges)
    else:
        top, ce, esigns, n_edges = None, None, None, 0

    # Argyris/Morley edge-normal DOFs: a per-cell, GLOBALLY-oriented physical unit normal per local edge so the
    # two cells sharing an edge agree on the sign of the normal-derivative DOF. Orientation is fixed by the
    # canonical (low, high) global vertex pair: n = R90·(P[hi] - P[lo]) = (-(Δy), Δx) (Kirby 2018 / cross-cell C1).
    argyris_normals = None
    if has_argyris or has_morley:
        _ev = np.asarray(top.edge_vertices)  # (n_edges, 2) canonical (lo, hi)
        _d = np.asarray(pts)[_ev[:, 1]] - np.asarray(pts)[_ev[:, 0]]  # (n_edges, 2)
        _en = np.stack([-_d[:, 1], _d[:, 0]], axis=1)  # R90·d per global edge
        _en = _en / np.linalg.norm(_en, axis=1, keepdims=True)
        argyris_normals = jnp.asarray(_en[np.asarray(top.cell_edges)])  # (n_cells, 3, 2)

    # Hermite per-cell global DOF map: 3 DOFs per vertex (value, ∂x, ∂y, in basix order) + 1 interior
    # (centroid) DOF per cell. Continuity is automatic from shared global vertex ids (point functionals --
    # no edge dedup, no orientation sign; the M(cell) transform takes that role).
    hermite_cdofs = None
    if has_hermite:
        _vdofs = (3 * cells_j[:, :, None] + jnp.arange(3)[None, None, :]).reshape(n_cells, 9)  # (n_cells, 9)
        _idofs = (3 * n_verts + jnp.arange(n_cells))[:, None]  # (n_cells, 1) interior
        hermite_cdofs = jnp.concatenate([_vdofs, _idofs], axis=1)  # (n_cells, 10), block-local DOF ids

    # Argyris per-cell global DOF map: 6 DOFs per vertex (value, ∂x, ∂y, ∂xx, ∂xy, ∂yy) over shared global
    # vertex ids, then 1 normal-derivative DOF per global edge at base ``6·n_verts`` (deduped + sign-consistent
    # via the edge topology). Continuity of value/derivatives is automatic from shared vertex ids; the
    # edge-normal DOF is shared through ``cell_edges``.
    argyris_cdofs = None
    if has_argyris:
        _vdofs = (6 * cells_j[:, :, None] + jnp.arange(6)[None, None, :]).reshape(n_cells, 18)  # (n_cells, 18)
        _edofs = 6 * n_verts + ce  # (n_cells, 3) global edge-DOF ids
        argyris_cdofs = jnp.concatenate([_vdofs, _edofs], axis=1)  # (n_cells, 21)

    # Morley per-cell global DOF map: 1 value DOF per vertex (shared global vertex id) + 1 normal-derivative
    # DOF per global edge at base ``n_verts``. Local order [v0, v1, v2, e0, e1, e2] matches the M(cell) rows.
    morley_cdofs = None
    if has_morley:
        _edofs = n_verts + ce  # (n_cells, 3) global edge-DOF ids
        morley_cdofs = jnp.concatenate([cells_j, _edofs], axis=1)  # (n_cells, 6)

    def _field_ndof(s):
        if s == "Hermite":
            return 3 * n_verts + n_cells
        if s == "Argyris":
            return 6 * n_verts + n_edges
        if s == "Morley":
            return n_verts + n_edges
        return n_edges if s in ("RT", "N1E") else n_cells

    ndof = [_field_ndof(s) for s in spaces]
    offs = [0]
    for n in ndof:
        offs.append(offs[-1] + n)
    total = offs[-1]

    def _field_cdofs(i):
        if spaces[i] == "Hermite":
            return offs[i] + hermite_cdofs  # (n_cells, 10)
        if spaces[i] == "Argyris":
            return offs[i] + argyris_cdofs  # (n_cells, 21)
        if spaces[i] == "Morley":
            return offs[i] + morley_cdofs  # (n_cells, 6)
        if spaces[i] in ("RT", "N1E"):
            return offs[i] + ce  # (n_cells, 3)
        return offs[i] + jnp.arange(n_cells)[:, None]  # (n_cells, 1) P0

    cdofs = [_field_cdofs(i) for i in range(len(fields))]

    def _cell_fields(c, u_blocks):
        verts = pts[cells_j[c]]
        J = jnp.stack([verts[k] - verts[0] for k in range(1, dim + 1)], axis=1)  # (dim, dim): 2x2 tri / 3x3 tet
        detJ = jnp.linalg.det(J)
        per = []
        for i, s in enumerate(spaces):
            if s == "Hermite":  # C0 vertex value+derivative DOFs via the M(cell) DOF-transform
                rv, rg, rh = hermite_ref
                phi, grad, hess = hermite_pushforward(rv, rg, rh, J, detJ, None)
                # Tag "Lagrange": the M(cell) transform is baked into phi/grad/hess, so this SCALAR field's
                # shape data matches nodal Lagrange and the shared evaluator (value / .x / Hessian) serves
                # it unchanged. cell_sol = this cell's 10 local DOF values, (10, 1).
                per.append(
                    {
                        "shape_vals": phi,
                        "shape_grads": grad,
                        "shape_hess": hess,
                        "cell_sol": u_blocks[i][cdofs[i][c] - offs[i]][:, None],
                        "space": "Lagrange",
                    }
                )
            elif s == "Argyris":  # C1 conforming: M(cell) DOF-transform + globally-oriented edge-normal DOFs
                rv, rg, rh, nodal = argyris_ref
                phi, grad, hess = argyris_pushforward(rv, rg, rh, J, detJ, argyris_normals[c], nodal)
                per.append(
                    {
                        "shape_vals": phi,
                        "shape_grads": grad,
                        "shape_hess": hess,
                        "cell_sol": u_blocks[i][cdofs[i][c] - offs[i]][:, None],  # this cell's 21 local DOFs
                        "space": "Lagrange",
                    }
                )
            elif s == "Morley":  # non-conforming biharmonic: M(cell) DOF-transform + globally-oriented edge normal
                rv, rg, rh, nodal = morley_ref
                phi, grad, hess = morley_pushforward(rv, rg, rh, J, detJ, argyris_normals[c], nodal)
                per.append(
                    {
                        "shape_vals": phi,
                        "shape_grads": grad,
                        "shape_hess": hess,
                        "cell_sol": u_blocks[i][cdofs[i][c] - offs[i]][:, None],  # this cell's 6 local DOFs
                        "space": "Lagrange",
                    }
                )
            elif s in edge_ref:  # RT (contravariant) or N1E (covariant): same edge DOFs, family-specific push-forward
                rval, rdop, rgr, pf, pgf = edge_ref[s]
                phi, _d = pf(rval, rdop, J, detJ, esigns[c])  # (n_quad, 3, 2)
                grad = pgf(rgr, J, detJ, esigns[c])  # (n_quad, 3, 2, 2)
                per.append({"shape_vals": phi, "shape_grads": grad, "cell_sol": u_blocks[i][ce[c]], "space": s})
            else:  # P0: a single constant DOF per cell
                per.append(
                    {
                        "shape_vals": jnp.ones((n_quad, 1)),
                        "shape_grads": jnp.zeros((n_quad, 1, dim, dim)),
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

        def residual(u_flat, args=None):
            u_blocks = [u_flat[offs[i] : offs[i + 1]] for i in range(len(fields))]
            # Pack each runtime parameter into volume_vars (steady: no temporal prefix), so the shared evaluator
            # resolves it via `_runtime_parameter_value_from_internal_vars`. A SCALAR parameter is a single
            # cell-independent `(1,)` value; a P1 FIELD parameter k(x) is this cell's 3 mesh-vertex values
            # (gathered at `cells_j[c]`), which the evaluator interpolates with the P1 `shape_vals` supplied in
            # `local`. A parameter absent from `args` packs a zero placeholder (right width per kind).
            _a = args or {}
            _zero_field = jnp.zeros((n_verts,), dtype=u_flat.dtype)
            rt_scalar = {
                name: jnp.reshape(jnp.asarray(_a.get(name, 0.0), dtype=u_flat.dtype), (-1,))[:1]
                for name in runtime_parameter_tags
                if name not in _field_param_names
            }
            field_vals = {name: jnp.asarray(_a.get(name, _zero_field), dtype=u_flat.dtype) for name in _field_param_names}
            R = jnp.zeros(total, dtype=u_flat.dtype)
            for coeff, tfi in typed:

                def _cell(c, e=coeff, _sc=rt_scalar, _fv=field_vals):
                    per, xq, meas = _cell_fields(c, u_blocks)
                    vol_vars = tuple(
                        (_fv[name][cells_j[c]] if name in _field_param_names else _sc[name])  # field: 3 vertex values
                        for name in runtime_parameter_tags
                    )
                    local = {
                        "physical_quad_points": xq,
                        "fields": per,
                        "field_index": field_index,
                        "tag": "fem_gauss",
                        "surface": False,
                        "domain_context": ctx,
                        "temporal_tags": (),
                        "runtime_parameter_tags": runtime_parameter_tags,
                        "volume_vars": vol_vars,
                    }
                    if _field_param_names:  # P1 basis to interpolate a field parameter (top-level, param-only key)
                        local["shape_vals"] = p1_shape_vals
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
        _flux_bc_pins(flux_bcs, domain, field_index, spaces, top, np.asarray(pts), offs, n_cells, quad_degree, dim=dim)
        if flux_bcs
        else []
    )
    # Essential plate BCs compose from two independent traces: the **deflection** ``u(region)-g`` (value)
    # and the **rotation** ``u.dn(region)-h`` (normal derivative). Clamped = both; simply-supported = value
    # only; guided = rotation only; free = neither. Value BCs arrive in ``dirichlet_raw``; rotation BCs in
    # ``rotation_bcs`` (normalised to the same 5-tuple shape, value node = the prescribed ∂u/∂n).
    _rot_as_dir = [(fk, region, None, None, vn) for (fk, region, vn) in rotation_bcs]
    if has_hermite and dirichlet_raw:  # Hermite value-Dirichlet: pin boundary-vertex value DOFs to g
        pins = pins + _hermite_dirichlet_pins(dirichlet_raw, domain, field_index, spaces, np.asarray(pts), offs)
    if has_argyris and dirichlet_raw:  # deflection: pin the boundary value + tangential trace, free the normal
        pins = pins + _argyris_dirichlet_pins(
            dirichlet_raw, domain, field_index, spaces, np.asarray(pts), offs, top, n_verts, kind="value"
        )
    if has_argyris and _rot_as_dir:  # rotation ∂u/∂n: pin the normal-derivative DOFs
        pins = pins + _argyris_dirichlet_pins(
            _rot_as_dir, domain, field_index, spaces, np.asarray(pts), offs, top, n_verts, kind="rotation"
        )
    if has_morley and dirichlet_raw:  # deflection: pin the boundary vertex value DOFs
        pins = pins + _morley_dirichlet_pins(
            dirichlet_raw, domain, field_index, spaces, np.asarray(pts), offs, top, n_verts, kind="value"
        )
    if has_morley and _rot_as_dir:  # rotation ∂u/∂n: pin the boundary edge-normal DOFs
        pins = pins + _morley_dirichlet_pins(
            _rot_as_dir, domain, field_index, spaces, np.asarray(pts), offs, top, n_verts, kind="rotation"
        )
    # Deduplicate by DOF: a corner DOF can be pinned by both the value and the rotation trace (e.g. the
    # gradient is a tangential DOF for one edge and a normal DOF for the perpendicular one). A repeated pin
    # would apply the boundary lift twice; collapse to one (consistent value) before enforcement.
    if pins:
        pins = list(dict(pins).items())
    zeros = jnp.zeros(total)

    # === transient: M u̇ + A u = c (mirrors fem_1d._assemble_1d_transient) -- split the temporal term
    #     (∫ ∂ₜu·v -> mass M) from the spatial operator, project the IC onto the edge DOFs, time-block it. ===
    if ic_residuals or any(_contains_temporal_derivative(t) for t in volume_terms):
        from ..._fem import _bare, _essential_spec, _eval_value_node_at, _field_key_of
        from .backend_blocks import SemidiscreteTimeBlock
        from .fem_1d import _apply_dirichlet_transient
        from .solver_helper import max_temporal_derivative_order as _mto
        from .time_route import _infer_time_window, _strip_temporal_trial_derivative

        sub = [_apply_sign(domain, s, t) for bare in volume_terms for s, t in _split_additive_terms(domain, bare)]
        temporal = [t for t in sub if _contains_temporal_derivative(t)]
        spatial = [t for t in sub if not _contains_temporal_derivative(t)]
        if not temporal:
            raise ValueError("jno.fem (non-nodal): a transient weak form must contain a temporal term, e.g. inner(u.t, v).")

        # === second-order-in-time (u_tt): the augmented first-order block y = [u; v], v = u̇, integrated by
        #     the trapezoidal (θ=½, energy-conserving) rule — the non-nodal analogue of the native
        #     _assemble_second_order_time, reusing this path's push-forward mass / pins / IC-projection. ===
        if max((_mto(t) for t in temporal), default=1) >= 2:
            if runtime_parameter_tags:
                raise NotImplementedError(
                    "jno.fem (non-nodal): a runtime parameter in a second-order-in-time form is not supported."
                )
            if len(fields) > 1:
                raise NotImplementedError("jno.fem (non-nodal): second-order-in-time (u_tt) is single-field only.")

            def _strip_n(t, k):
                for _ in range(k):
                    t = _strip_temporal_trial_derivative(t)
                return t

            M2 = jax.jacfwd(_make_residual([_strip_n(t, 2) for t in temporal if _mto(t) >= 2]))(zeros)  # ∫ü·v ⇒ mass
            damp = [_strip_n(t, 1) for t in temporal if _mto(t) == 1]  # ∫u̇·v ⇒ damping (optional)
            Cmat = jax.jacfwd(_make_residual(damp))(zeros) if damp else jnp.zeros((total, total), zeros.dtype)
            spatial_res2 = _make_residual(spatial)
            K = jax.jacfwd(spatial_res2)(zeros)  # spatial operator (stiffness)
            F = -spatial_res2(zeros) + nat_load  # load (natural BC folded in)
            n = total
            Z = jnp.zeros((n, n), zeros.dtype)
            M_aug = jnp.block([[M2, Z], [Z, M2]])
            A_aug = jnp.block([[Z, -M2], [K, Cmat]])  # M2 u̇ = M2 v ; M2 v̇ + C v + K u = F
            c_aug = jnp.concatenate([jnp.zeros((n,), zeros.dtype), F])
            if pins:  # essential BC on the augmented rows: u[d] = g (constant) and v[d] = 0
                dd = jnp.asarray([p[0] for p in pins], dtype=jnp.int32)
                dg = jnp.asarray([p[1] for p in pins], dtype=zeros.dtype)
                M_aug = M_aug.at[dd, :].set(0.0).at[dd + n, :].set(0.0)
                A_aug = A_aug.at[dd, :].set(0.0).at[dd, dd].set(1.0).at[dd + n, :].set(0.0).at[dd + n, dd + n].set(1.0)
                c_aug = c_aug.at[dd].set(dg).at[dd + n].set(0.0)

            def _project_onto(u0_node):  # L²-project a value node onto the field DOFs via the mass block M2
                u0_blocks = [jnp.zeros(offs[i + 1] - offs[i]) for i in range(len(fields))]

                def _ic_cell(cidx):
                    per, xq, meas = _cell_fields(cidx, u0_blocks)
                    u0 = jnp.asarray(_eval_value_node_at(u0_node, xq)).reshape(n_quad)
                    return jnp.einsum("q,qn,q->n", qw * meas, per[0]["shape_vals"], u0)

                loc = (cdofs[0] - offs[0]).reshape(-1)
                load = jnp.zeros((total,), zeros.dtype).at[loc].add(jax.vmap(_ic_cell)(jnp.arange(n_cells)).reshape(-1))
                return jnp.linalg.solve(M2, load)

            u0_dofs = jnp.zeros((n,), zeros.dtype)
            v0_dofs = jnp.zeros((n,), zeros.dtype)  # start from rest unless a velocity IC u̇(0)=v0 is given
            for ic in ic_residuals:
                _c, u0_node = _essential_spec(_bare(ic))
                proj = _project_onto(u0_node)
                if _mto(_bare(ic)) >= 1:
                    v0_dofs = proj
                else:
                    u0_dofs = proj
            if pins:  # make the initial state consistent with the essential BC (u[d]=g, v[d]=0)
                _pd = jnp.asarray([p[0] for p in pins], dtype=jnp.int32)
                u0_dofs = u0_dofs.at[_pd].set(jnp.asarray([p[1] for p in pins], dtype=zeros.dtype))
                v0_dofs = v0_dofs.at[_pd].set(0.0)
            t0, t1, dt = _infer_time_window(domain)
            return (
                SemidiscreteTimeBlock(
                    backend="transient",
                    mode="implicit",
                    time_order=2,
                    spatial_kind="weak_form",
                    M=M_aug,
                    A=A_aug,
                    affine_bias=c_aug,
                    state0=jnp.concatenate([u0_dofs, v0_dofs]),
                    t0=t0,
                    t1=t1,
                    dt=dt,
                    eval_context=getattr(domain, "_fem_eval_context", {}) or {},
                    metadata={"theta": 0.5, "second_order": True},
                ),
                "transient",
                [0, n, 2 * n],
            )

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

        pin_dofs = jnp.asarray([p[0] for p in pins], dtype=jnp.int32) if pins else None
        if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial):
            # nonlinear transient: M(t) u̇ + R(u) = 0 (matrix-free Newton-Krylov per step). Natural load
            # folds into R; essential pins are residual rows + zeroed M rows/cols (the 1D pattern).
            M_nl = M if pin_dofs is None else M.at[pin_dofs, :].set(0.0).at[:, pin_dofs].set(0.0)
            if runtime_parameter_tags:  # transient inverse: thread args through the residual + its Jacobian

                def res_pt(u, t, args=None):
                    return _apply_dirichlet_rows(lambda uu: spatial_res(uu, args) - nat_load, pins)(jnp.asarray(u))

                def jac_pt(u, t, args=None):
                    return jax.jacfwd(lambda uu: res_pt(uu, t, args))(jnp.asarray(u))

                block = SemidiscreteTimeBlock(
                    mass=lambda t, args=None, _M=M_nl: _M,
                    residual=res_pt,
                    jacobian=jac_pt,
                    runtime_parameter_exprs=dict(_rt_param_exprs),
                    **common,
                )
                return block, "transient", offs
            res_bc = _apply_dirichlet_rows(lambda u: spatial_res(u) - nat_load, pins)
            jac = jax.jacfwd(res_bc)
            block = SemidiscreteTimeBlock(
                mass=lambda t, args=None, _M=M_nl: _M,
                residual=lambda u, t, args=None: res_bc(u),
                jacobian=lambda u, t, args=None: jac(u),
                **common,
            )
            return block, "transient", offs

        if runtime_parameter_tags:  # linear transient inverse: A(args) re-assembled each step; dense Dirichlet
            #   row-replacement inside operator_fn (the dense analogue of the native bcoo_set_dirichlet_rows).
            M_bc = M if pin_dofs is None else M.at[pin_dofs, :].set(0.0).at[:, pin_dofs].set(0.0)
            c_bias = zeros if pin_dofs is None else zeros.at[pin_dofs].set(jnp.asarray([p[1] for p in pins]))
            free_mask = (
                jnp.ones((total,), dtype=zeros.dtype)
                if pin_dofs is None
                else jnp.ones((total,), dtype=zeros.dtype).at[pin_dofs].set(0.0)
            )

            def operator_fn(t, args=None, _d=pin_dofs):
                A = jax.jacfwd(lambda u: spatial_res(u, args))(zeros)
                return A if _d is None else A.at[_d, :].set(0.0).at[_d, _d].set(1.0)  # Dirichlet rows -> identity

            def forcing_vector_fn(t, args=None, _mask=free_mask):
                return _mask * (-spatial_res(zeros, args) + nat_load)  # source on free rows; Dirichlet via the bias

            return (
                SemidiscreteTimeBlock(
                    M=M_bc,
                    operator_fn=operator_fn,
                    affine_bias=c_bias,
                    forcing_vector_fn=forcing_vector_fn,
                    runtime_parameter_exprs=dict(_rt_param_exprs),
                    metadata={"nonaffine_operator": True},
                    **common,
                ),
                "transient",
                offs,
            )

        # linear transient: M u̇ + A u = c
        A = jax.jacfwd(spatial_res)(zeros)
        c = -spatial_res(zeros) + nat_load  # spatial load + natural-BC constant load
        M, A, c = _apply_dirichlet_transient(M, A, c, pins)  # essential edge-trace pins -> M/A/c rows
        return SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, **common), "transient", offs

    residual = _make_residual(volume_terms)

    def full_residual(u_flat, args=None):  # the natural BC is a constant load on the RHS: R(u) = assembled(u) - load
        return residual(u_flat, args) - nat_load

    # --- steady nonlinear: a genuinely nonlinear weak term -> a Newton residual operator ---
    if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in volume_terms):
        if runtime_parameter_tags:  # parametric (inverse): thread args into the residual AND its Jacobian

            def res_p(u, args=None):
                return _apply_dirichlet_rows(lambda uu: full_residual(uu, args), pins)(jnp.asarray(u))

            def jac_p(u, args=None):
                return jax.jacfwd(lambda uu: res_p(uu, args))(jnp.asarray(u))

            return (
                FemResidualOperator(res_p, jac_p, total, runtime_parameter_exprs=dict(_rt_param_exprs)),
                "nonlinear",
                offs,
            )
        res_bc = _apply_dirichlet_rows(full_residual, pins)  # essential pins as residual rows R[d]=u[d]-g
        jac = jax.jacfwd(res_bc)
        # FemResidualOperator.solve calls residual(u, args)/jacobian(u, args); non-parametric here, so
        # accept-and-ignore args (args=None keeps `fem.residual(u)` single-arg too).
        return (
            FemResidualOperator(lambda u, args=None: res_bc(u), lambda u, args=None: jac(u), total),
            "nonlinear",
            offs,
        )

    # --- steady linear parametric (inverse): re-assemble A(args), b(args) each call, kept differentiable in
    #     the parameter (mirrors the native path); the Dirichlet pins are RE-APPLIED per args. Returns a
    #     FemLinearSystem so `fem.solve()` gives a differentiable trace node crux can optimise. ---
    if runtime_parameter_tags:
        from ...trace import FemLinearSystem

        def _assemble_at(args):
            A = jax.jacfwd(lambda u: full_residual(u, args))(zeros)
            b = -full_residual(zeros, args)
            if pins:
                A, b = _apply_dirichlet_symmetric(jnp.asarray(A), jnp.asarray(b), pins)
            return A, b

        # static placeholder for `.A` / `.b` (right width per parameter kind: a field param is (n_verts,))
        _ph = {n: (jnp.zeros((n_verts,)) if n in _field_param_names else 0.0) for n in runtime_parameter_tags}
        a0, b0 = _assemble_at(_ph)
        return (
            FemLinearSystem(
                a0,
                b0,
                operator_fn=lambda args=None: _assemble_at(args)[0],
                rhs_fn=lambda args=None: _assemble_at(args)[1],
                runtime_parameter_exprs=dict(_rt_param_exprs),
                metadata={"nonaffine_operator": True},  # re-assembles at each args; no affine parameter basis
            ),
            "linear",
            offs,
        )

    # --- steady linear (non-parametric): A u = b (byte-identical to before the refactor) ---
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
            # prescribed edge-moment `M_n * v.dn(region)` on an Argyris/Morley plate field -> boundary integral
            mom = _match_plate_moment(bare, field_index, spaces)
            if mom is not None:
                b = np.asarray(
                    _plate_moment_load(
                        b, mom[1], mom[0], region_nodes, spaces[mom[0]], domain, top, pts_np, offs, boundary, loc
                    )
                )
                continue
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


def _match_plate_moment(bare, field_index, spaces):
    """Recognise a prescribed edge-moment term ``M_n * v.dn(region)`` on an Argyris/Morley plate field: a
    product with the test's **normal derivative** (``v.dn``) on one side and the moment coefficient ``M_n``
    (no test) on the other. Returns ``(field_index, M_n node)``, or ``None`` if it is not a moment term."""
    from ..._fem import _contains, _walk
    from ...trace import BinaryOp, NormalDerivative, TestFunction

    if not (isinstance(bare, BinaryOp) and bare.op == "*"):
        return None
    for side, other in ((bare.left, bare.right), (bare.right, bare.left)):
        has_nd_test = any(isinstance(n, NormalDerivative) for n in _walk(side)) and _contains(side, TestFunction)
        if has_nd_test and not _contains(other, TestFunction):
            fkeys = {n.field_key for n in _walk(side) if isinstance(n, TestFunction)}
            fidx = field_index.get(next(iter(fkeys))) if fkeys else None
            if fidx is not None and spaces[fidx] in ("Argyris", "Morley"):
                return fidx, other
    return None


def _plate_moment_load(b, mn_node, fidx, region_nodes, space, domain, top, pts_np, offs, boundary, loc):
    """Assemble the prescribed edge-**moment** load ``+∮_region M_n (∇v · n_out) ds`` for one Argyris/Morley
    plate field into the constant natural-BC load ``b``.

    The bending moment is the natural quantity conjugate to the plate **rotation** ``∂v/∂n``: minimising
    ``½ a(w,w) − ∫ f w − ∮ M_n ∂w/∂n`` adds ``+∮ M_n ∂v/∂n`` to the weak load, and on a straight simply-
    supported edge (``w=0`` essential) the emergent natural condition is ``Δw = M_n`` (Timoshenko &
    Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., McGraw-Hill 1959, §2 — edge bending moment).
    Only ``M_n`` (conjugate to rotation) is wired; the effective Kirchhoff shear ``V_n = Q_n + ∂M_{nt}/∂t``
    (conjugate to deflection, with corner forces) is **not**. The element basis' physical normal derivative
    is tabulated at boundary-edge quadrature points via the same ``M(cell)`` push-forward the volume assembly
    uses (:func:`argyris_pushforward` / :func:`morley_pushforward`), with the reference basis re-tabulated at
    the edge nodes (one tabulation per local edge, reused across all boundary edges of that orientation)."""
    from ..._fem import _eval_value_node_at
    from .fem_1d import _line_quadrature
    from .fem_elements import (
        argyris_pushforward,
        argyris_ref_basis_at,
        argyris_triangle,
        morley_pushforward,
        morley_ref_basis_at,
        morley_triangle,
    )
    from .fem_topology import BASIX_TRIANGLE_EDGES

    # M_n · ∂ₙφ is high-degree on the edge (∂ₙ of a quintic basis is a quartic, ×M_n); use a rule that
    # integrates it exactly for the common polynomial data (degree 14 -> 8 Gauss nodes, exact to degree 15).
    gp, gw = (np.asarray(x).reshape(-1) for x in _line_quadrature(14))
    cells = np.asarray(domain.mesh.cells_dict["triangle"])
    ev = np.asarray(top.edge_vertices)  # (n_edges, 2) canonical (lo, hi)
    ce = np.asarray(top.cell_edges)  # (n_cells, 3) global edge ids
    n_verts = pts_np.shape[0]
    base = offs[fidx]
    ref_v = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    if space == "Argyris":
        nodal = tuple(jnp.asarray(a) for a in argyris_triangle().ref_aux)
        pushf, tabber = argyris_pushforward, argyris_ref_basis_at
    else:
        nodal = tuple(jnp.asarray(a) for a in morley_triangle().ref_aux)
        pushf, tabber = morley_pushforward, morley_ref_basis_at
    # reference-edge basis tabulations (value+grad at the 1-D edge nodes), one per local edge k, reused below
    ref_edge_tab = {}
    for k, (a, bb) in enumerate(BASIX_TRIANGLE_EDGES):
        ref_pts = ref_v[a][None, :] * (1.0 - gp[:, None]) + ref_v[bb][None, :] * gp[:, None]
        ref_edge_tab[k] = tabber(np.asarray(ref_pts))

    b = np.asarray(b)
    for eid in boundary:
        va, vb = (int(x) for x in ev[eid])
        if va not in region_nodes or vb not in region_nodes:
            continue
        c, k = loc[eid]
        cverts = pts_np[cells[c]]  # (3, 2)
        J = np.stack([cverts[1] - cverts[0], cverts[2] - cverts[0]], axis=1)  # columns = edge vectors from v0
        detJ = float(np.linalg.det(J))
        # per-cell globally-oriented edge normals for M(cell) (canonical lo->high, matching the volume assembler)
        _d = pts_np[ev[ce[c], 1]] - pts_np[ev[ce[c], 0]]
        _en = np.stack([-_d[:, 1], _d[:, 0]], axis=1)
        cell_normals = jnp.asarray(_en / np.linalg.norm(_en, axis=1, keepdims=True))
        rv, rg, rh = ref_edge_tab[k]
        _phi, grad, _h = pushf(rv, rg, rh, jnp.asarray(J), detJ, cell_normals, nodal)
        grad = np.asarray(grad)  # (nq, ndof, 2) physical gradient of the cell basis at the edge nodes
        # physical edge nodes via the SAME ref->phys map the push-forward uses (affine): x = v0 + xi @ J.T
        la, lb = BASIX_TRIANGLE_EDGES[k]
        pa, pb = pts_np[cells[c][la]], pts_np[cells[c][lb]]
        xq = pa[None, :] * (1.0 - gp[:, None]) + pb[None, :] * gp[:, None]  # (nq, 2)
        L = float(np.linalg.norm(pb - pa))
        # outward normal: R90 of the edge, flipped to point away from the cell's third (opposite) vertex
        evec = pb - pa
        n_out = np.array([-evec[1], evec[0]], dtype=float)
        n_out /= np.linalg.norm(n_out)
        third = next(int(w) for w in cells[c] if int(w) not in (int(cells[c][la]), int(cells[c][lb])))
        if float(n_out @ (0.5 * (pa + pb) - pts_np[third])) < 0.0:
            n_out = -n_out
        dphi_dn = grad @ n_out  # (nq, ndof)
        mn = np.asarray(_eval_value_node_at(mn_node, jnp.asarray(xq))).reshape(-1)  # (nq,)
        contrib = L * np.einsum("q,q,qn->n", gw, mn, dphi_dn)  # ∮ M_n ∂ₙφ over this edge, per cell DOF
        if space == "Argyris":
            gdofs = np.concatenate([(6 * cells[c][:, None] + np.arange(6)).reshape(-1), 6 * n_verts + ce[c]])
        else:
            gdofs = np.concatenate([cells[c], n_verts + ce[c]])
        np.add.at(b, np.asarray(int(base) + gdofs, dtype=np.intp), contrib)
    return jnp.asarray(b)


def _hermite_dirichlet_pins(dirichlet_raw, domain, field_index, spaces, pts_np, offs):
    """Value-Dirichlet ``(dof, value)`` pins for a Hermite field: pin the **value** DOF (``3·v``) at each
    boundary vertex of the region to ``g(vertex)``. The two derivative DOFs stay free -- this is a true
    value BC (the normal/tangential derivatives are natural), the analogue of nodal Lagrange Dirichlet.
    ``g`` is the Dirichlet value node, evaluated at the vertex coordinate. (A clamped BC -- additionally
    pinning the gradient DOFs to ``∇g`` -- is a follow-on; needed for optimal rates and the C¹ elements.)"""
    from ..._fem import _eval_value_node_at
    from .fem_1d import _region_node_ids

    pins = []
    for fk, region, _comp, _value, value_node in dirichlet_raw:
        fidx = field_index.get(fk)
        if fidx is None or spaces[fidx] != "Hermite":
            continue
        for v in _region_node_ids(domain, region):
            v = int(v)
            g = jnp.asarray(_eval_value_node_at(value_node, jnp.asarray(pts_np[v][None, :]))).reshape(-1)
            pins.append((offs[fidx] + 3 * v, float(g[0])))
    return pins


def _morley_dirichlet_pins(dirichlet_raw, domain, field_index, spaces, pts_np, offs, top, n_verts, kind="value"):
    """Essential BC for a Morley field, one trace at a time (compose for clamped):

    * ``kind="value"`` — the **deflection** ``u(region) - g``: pin the value DOF at each boundary vertex to
      ``g`` (Morley's non-conforming value trace lives at the vertices). Alone ⇒ simply-supported.
    * ``kind="rotation"`` — the **rotation** ``u.dn(region) - h``: pin the edge-normal DOF at each boundary
      edge to the prescribed ``∂u/∂n = h`` (evaluated at the edge midpoint). Alone ⇒ guided.

    Both work on **any** boundary orientation — Morley's DOFs are already the vertex value and the edge-normal
    derivative, so no ``(n, t)`` rotation is needed (unlike the Argyris element). The stored edge DOF is the
    derivative along the assembler's globally-oriented normal ``R90·(P[hi]-P[lo])``, so ``h`` (given as the
    *outward* normal derivative) is multiplied by the per-edge sign relating the two."""
    from ..._fem import _eval_value_node_at
    from .fem_1d import _region_node_ids

    def _val(value_node, xy):
        return float(jnp.asarray(_eval_value_node_at(value_node, jnp.asarray(xy)[None, :])).reshape(()))

    cell_edges = np.asarray(top.cell_edges)
    counts = np.bincount(cell_edges.reshape(-1), minlength=top.n_edges)
    boundary_edges = {int(e) for e in np.where(counts == 1)[0]}
    cells_arr = np.asarray(domain.mesh.cells_dict["triangle"])
    edge_cell = {}  # boundary edge -> its single incident cell (for the outward-normal sign)
    for c in range(cells_arr.shape[0]):
        for k in range(3):
            e = int(cell_edges[c, k])
            if e in boundary_edges:
                edge_cell[e] = c

    pins = []
    for fk, region, _comp, _value, value_node in dirichlet_raw:
        fidx = field_index.get(fk)
        if fidx is None or spaces[fidx] != "Morley":
            continue
        base = offs[fidx]
        region_nodes = {int(v) for v in _region_node_ids(domain, region)}
        if kind == "value":
            for v in region_nodes:  # deflection: value DOF at each boundary vertex
                pins.append((base + v, _val(value_node, pts_np[v])))
            continue
        edge_base = base + n_verts
        for eid in boundary_edges:  # rotation: edge-normal DOF = (outward sign)·h at the midpoint
            va, vb = (int(x) for x in top.edge_vertices[eid])
            if va not in region_nodes or vb not in region_nodes:
                continue
            evec = pts_np[vb] - pts_np[va]
            n_glob = np.array([-evec[1], evec[0]])
            n_glob = n_glob / np.linalg.norm(n_glob)
            mid = 0.5 * (pts_np[va] + pts_np[vb])
            vc = next(int(w) for w in cells_arr[edge_cell[eid]] if int(w) not in (va, vb))
            sign = 1.0 if float(n_glob @ (mid - pts_np[vc])) > 0 else -1.0  # outward = away from the interior vertex
            pins.append((edge_base + eid, sign * _val(value_node, mid)))
    return pins


def _argyris_dirichlet_pins(dirichlet_raw, domain, field_index, spaces, pts_np, offs, top, n_verts, kind="value"):
    """Essential BC for an Argyris field, one boundary trace at a time (compose for clamped):

    * ``kind="value"`` — the **deflection** ``u(region) - g``: pin the value and the **tangential** derivatives
      (``∂ₜ``, ``∂ₜₜ``) so ``u = g`` along the edge, while the **normal** derivatives (``∂ₙ``, ``∂ₙₜ``, ``∂ₙₙ``)
      stay free — i.e. simply-supported. ``g`` and its tangential derivatives come from autodiff of the value.
    * ``kind="rotation"`` — the **rotation** ``u.dn(region) - h``: pin the normal-derivative DOFs (``∂ₙ`` at the
      vertices, ``∂ₙₜ``, and the edge-midpoint normal derivative) to the prescribed ``∂u/∂n = h``.

    Compose the two for a clamped edge (``∂ₙₙ`` still left free — the physical clamped plate leaves the boundary
    curvature as a natural BC). On an **axis-aligned** edge the ``(n, t)`` frame is ``(x, y)`` so each of these
    is a single Argyris DOF; a non-axis-aligned edge needs the ``(n, t)`` rotation (not wired) and is **rejected
    loudly**. Pins are dict-unioned per edge, so a corner is consistently pinned by both incident edges. (For
    Morley — no Hessian DOFs, edge-normal DOF is ``∂ₙ`` — the same two BCs work on *any* orientation.)"""
    from ..._fem import _eval_value_node_at
    from .fem_1d import _region_node_ids

    def _derivs(value_node, xy):
        def gfun(p):
            return jnp.asarray(_eval_value_node_at(value_node, p[None, :])).reshape(())

        p = jnp.asarray(xy, dtype=jnp.float64)
        return float(gfun(p)), np.asarray(jax.grad(gfun)(p))

    cell_edges = np.asarray(top.cell_edges)
    counts = np.bincount(cell_edges.reshape(-1), minlength=top.n_edges)
    boundary_edges = {int(e) for e in np.where(counts == 1)[0]}
    cells_arr = np.asarray(domain.mesh.cells_dict["triangle"])
    edge_cell = {}  # boundary edge -> single incident cell (for the outward-normal orientation)
    for c in range(cells_arr.shape[0]):
        for k in range(3):
            e = int(cell_edges[c, k])
            if e in boundary_edges:
                edge_cell[e] = c

    pins: dict = {}  # dof -> value; dict-unioned so corners are pinned consistently by both edges
    for fk, region, _comp, _value, value_node in dirichlet_raw:
        fidx = field_index.get(fk)
        if fidx is None or spaces[fidx] != "Argyris":
            continue
        base = offs[fidx]
        edge_base = base + 6 * n_verts
        region_nodes = {int(v) for v in _region_node_ids(domain, region)}
        for eid in boundary_edges:
            va, vb = (int(x) for x in top.edge_vertices[eid])  # canonical (lo, hi)
            if va not in region_nodes or vb not in region_nodes:
                continue
            evec = pts_np[vb] - pts_np[va]  # edge tangent
            atol = 1e-9 * (float(np.linalg.norm(evec)) + 1.0)
            if abs(evec[0]) < atol:  # edge along y, on x=const -> normal axis x(0), tangent axis y(1)
                nax, tax = 0, 1
            elif abs(evec[1]) < atol:  # edge along x, on y=const -> normal axis y(1), tangent axis x(0)
                nax, tax = 1, 0
            else:
                raise NotImplementedError(
                    "jno.fem (non-nodal): Argyris essential plate BCs are wired for axis-aligned boundary "
                    f"edges only; got a boundary edge with tangent {tuple(np.round(evec, 4))}. The general "
                    "(n,t)-rotation treatment is not wired -- use an axis-aligned domain, or the Morley element "
                    "(space='Morley'), which supports any orientation."
                )
            hxx, hyy = 3, 5  # Argyris Hessian DOFs: ∂ₓₓ=3, ∂ₓᵧ=4, ∂ᵧᵧ=5
            H_tt, H_nt = (hxx if tax == 0 else hyy), 4  # ∂ₜₜ (single DOF), ∂ₙₜ = ∂ₓᵧ
            g_t, g_n = (1 + tax), (1 + nax)  # gradient DOFs: ∂ₓ=1, ∂ᵧ=2
            n_glob = np.array([-evec[1], evec[0]]) / np.linalg.norm(evec)  # assembler's edge-DOF normal
            mid = 0.5 * (pts_np[va] + pts_np[vb])
            vc = next(int(w) for w in cells_arr[edge_cell[eid]] if int(w) not in (va, vb))
            n_out = n_glob if float(n_glob @ (mid - pts_np[vc])) > 0 else -n_glob  # outward normal
            s = float(n_out[nax])  # ±1: outward normal along +/- the normal axis (axis-aligned)
            for v in (va, vb):
                val, grad = _derivs(value_node, pts_np[v])
                if kind == "value":  # pin value + tangential trace (Cartesian DOFs; free the normal ones)
                    pins[base + 6 * v + 0] = val
                    pins[base + 6 * v + g_t] = float(grad[tax])
                    pins[base + 6 * v + H_tt] = _second_tangential(value_node, pts_np[v], tax)
                else:  # rotation: pin ∂ₙ (= s·h) and ∂ₙₜ (= s·∂ₜh); leave value/tangential/∂ₙₙ free
                    pins[base + 6 * v + g_n] = s * val
                    pins[base + 6 * v + H_nt] = s * float(grad[tax])
            if kind == "rotation":  # edge-midpoint normal derivative along the assembler's global normal
                hmid, _gmid = _derivs(value_node, mid)
                pins[edge_base + eid] = float(n_glob @ n_out) * hmid
    return list(pins.items())


def _second_tangential(value_node, xy, tax):
    """``∂²g/∂(tax)²`` at a point via autodiff of the value node (for the Argyris value-BC tangential trace)."""
    from ..._fem import _eval_value_node_at

    def gfun(p):
        return jnp.asarray(_eval_value_node_at(value_node, p[None, :])).reshape(())

    return float(np.asarray(jax.hessian(gfun)(jnp.asarray(xy, dtype=jnp.float64)))[tax, tax])


def _n1e_tangential_pins_3d(flux_bcs, domain, field_index, spaces, top, offs):
    """PEC tangential pins ``n × E = 0`` for a **3-D** N1E field: pin every boundary-face edge DOF in the BC
    region to 0. Boundary faces are the tet faces used by exactly one cell (:func:`build_facet_connectivity`);
    each contributes its 3 edges, mapped to the global N1E edge id via the edge topology. This is the correct
    3-D criterion — the 2-D "edge used once" / "both endpoints on the region" tests are wrong on a tet mesh
    (an interior edge can join two boundary vertices through the volume). Homogeneous PEC only: a nonzero
    tangential trace ``n × E = g`` raises (its per-edge value ``∫_e g·t`` is a follow-on)."""
    from ..._fem import _constant_of
    from .fem_1d import _region_node_ids
    from .fem_facets import build_facet_connectivity

    cells = np.asarray(domain.mesh.cells_dict["tetra"])
    fc = build_facet_connectivity(cells, "tetrahedron")
    edge_id = {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(top.edge_vertices))}  # canonical (lo,hi) -> eid
    pins = []
    for field_key, region, value_node in flux_bcs:
        fidx = field_index.get(field_key)
        if fidx is None or spaces[fidx] != "N1E":
            raise NotImplementedError(
                "jno.fem (non-nodal, 3D): the only essential edge-trace BC is the N1E tangential trace `n×E`."
            )
        if _constant_of(value_node) != 0.0:
            raise NotImplementedError(
                "jno.fem (non-nodal, 3D): only the homogeneous PEC tangential BC `n×E = 0` is wired; a "
                "prescribed nonzero tangential trace is a follow-on."
            )
        region_nodes = {int(n) for n in _region_node_ids(domain, region)}
        for f in range(fc.n_bfaces):
            fn = [int(x) for x in fc.face_nodes[f]]
            if not all(v in region_nodes for v in fn):
                continue  # this boundary face is not in the BC region
            for a, b in ((fn[0], fn[1]), (fn[1], fn[2]), (fn[0], fn[2])):
                eid = edge_id.get((min(a, b), max(a, b)))
                if eid is not None:
                    pins.append((offs[fidx] + eid, 0.0))
    return list(dict(pins).items())  # dedup edges shared by two boundary faces of the region


def _flux_bc_pins(flux_bcs, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree, *, dim=2):
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

    Boundary edges are the globally single-use edges, filtered to the BC's region by node membership. In
    **3-D** (tet mesh) this "single-use / both-endpoints-in-region" criterion is wrong, so the N1E tangential
    (PEC) pins are computed facet-based via :func:`_n1e_tangential_pins_3d`."""
    if dim == 3:  # 3-D N1E tangential (PEC) — facet-based boundary edges, not the 2-D single-use rule
        return _n1e_tangential_pins_3d(flux_bcs, domain, field_index, spaces, top, offs)
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
