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
from .small_linalg import small_det, small_inv

ScalarField = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]  # (x, y) -> values


def _cell_jacobian(verts: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Affine triangle Jacobian ``J = [v1-v0, v2-v0]`` and its (signed) determinant."""
    J = jnp.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)  # (2, 2)
    return J, small_det(J)


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
    from .fem_1d import _apply_dirichlet_projected, _apply_dirichlet_symmetric, _integrate_term, dirichlet_projection
    from .fem_elements import (
        argyris_pushforward,
        argyris_triangle,
        hermite_pushforward,
        hermite_triangle,
        morley_pushforward,
        morley_triangle,
        piola_contravariant,
        piola_contravariant_grad,
        piola_covariant,
        piola_covariant_grad,
    )
    from .fem_topology import BASIX_TET_EDGES, BASIX_TRIANGLE_EDGES, build_edge_topology
    from .fem_utils import (
        _infer_fields,
        _lower_statefield_to_trial,
        _test_field_index,
        bcoo_eliminate_dirichlet,
        bcoo_zero_rows_cols,
        compress_eager,
    )
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
    # Block layout in ASSEMBLY (= offsets) order, for ``FEM.block_index``. ``_finalize`` snapshots
    # this attribute; without it a non-nodal problem falls back to the CONSTRAINT-WALK order, which
    # is a different order. Measured on a mixed N1E x Lagrange form: the assembler laid the blocks
    # out as [u (n_edges), p (n_verts)] but the walk reported [p, u], so ``block_index(u)`` returned
    # 1 and ``jno.precond.triangular((u, ams()), (p, amg()))`` handed AMS the scalar block. AMS
    # forms G^T A G with G sized (n_edges, n_verts), so it raised a shape mismatch and the whole
    # triangular(AMS, AMG) path was unreachable on any mixed non-nodal system. Where the two orders
    # happen to agree it worked by luck; where the block SIZES also agree it would have been a
    # silently wrong preconditioner rather than a shape error.
    domain._fem_native_field_keys = [f["field_key"] for f in fields]

    spaces = [f["space"] for f in fields]
    if any(s not in ("RT", "N1E", "N2E", "P0", "DG", "Hermite", "Argyris", "Morley", "Lagrange") for s in spaces):
        raise NotImplementedError(
            f"jno.fem (non-nodal): supported element spaces are RT, N1E, N2E, P0, DG, Hermite, Argyris, "
            f"Morley and Lagrange; got {spaces}. (Lagrange is admitted so a nodal scalar can be MIXED "
            f"with a non-nodal field -- the A-V pair is N1E x Lagrange. A Lagrange-only form belongs on "
            f"the native nodal assembler, which this path never sees.)"
        )
    # ``order=`` selects the basix DEGREE of the H(curl)/H(div) families (``order=1`` is the lowest
    # order: N1E0 / RT0 in the "k-1" naming, ``order=k`` converges at rate k in the energy norm) and of
    # a Lagrange field mixed in here. Every other family here has an order INTRINSIC to its definition,
    # so a request for another one is refused rather than ignored: an ignored order= was measured to
    # return the SAME lowest-order space with no warning, which on a wave problem is found only when a
    # convergence study stalls.
    _INTRINSIC_ORDER = {
        "P0": "0 (piecewise constant; use space='DG', order=k for discontinuous P_k)",
        "Hermite": "3 (cubic)",
        "Argyris": "5 (quintic)",
        "Morley": "2 (quadratic)",
    }
    _orders = []
    for _f in fields:
        _sp = str(_f["space"])
        _raw = _f.get("order", 1)
        _o = 1 if _raw is None else int(_raw)
        if _sp == "DG" and _o < 0:
            raise ValueError(f"jno.fem: order={_o} on a DG field; the lowest order is 0 (piecewise constant).")
        if _o > 1 and _sp in _INTRINSIC_ORDER:
            raise NotImplementedError(
                f"jno.fem: order={_o} is not selectable on the {_sp} element — its order is intrinsic "
                f"to the family ({_INTRINSIC_ORDER[_sp]}) and jNO builds only that one. Drop order= "
                "(the default) to get it. order= does apply to N1E, N2E, RT, DG and Lagrange."
            )
        if _o < 1 and _sp != "DG":
            if _sp in _INTRINSIC_ORDER:
                _o = 1  # the intrinsic families ignore a zero/None default order -- historic behaviour
            else:
                raise ValueError(f"jno.fem: order={_o} on the {_sp} field; the lowest order is 1.")
        _orders.append(_o)
    _EDGE_FAMILIES = ("RT", "N1E", "N2E")
    has_edge = any(s in _EDGE_FAMILIES for s in spaces)
    has_hermite = "Hermite" in spaces
    has_argyris = "Argyris" in spaces
    has_morley = "Morley" in spaces  # non-conforming biharmonic (vertex value + edge-normal DOFs)
    if has_morley and any(s != "Morley" for s in spaces):
        raise NotImplementedError("jno.fem (non-nodal): a Morley field cannot be mixed with other element families.")
    # vertex-DOF families (take a nodal/derivative Dirichlet): the M(cell) DOF-transform path
    has_vertex = has_hermite or has_argyris or has_morley
    # The vertex families used to hand back a DENSE operator, so `fem.solve()` landed on a direct
    # `jnp.linalg.solve`. Now that they assemble sparsely the default would silently become the
    # Jacobi-preconditioned BiCGStab that serves real elliptic systems -- and these are 4th-order
    # biharmonic operators (`test_fem_morley.py` asserts the WELL-conditioned form is only cond < 1e12),
    # where it does not converge. Carry the direct choice over rather than inherit the sparse default;
    # `_solve_dispatch` already does the same for 1-D tridiagonal and complex/indefinite systems.
    domain._fem_prefer_direct = bool(has_vertex)
    if has_edge and (has_hermite or has_argyris):
        raise NotImplementedError(
            "jno.fem (non-nodal): mixing edge (RT/N1E) and vertex (Hermite/Argyris) fields is not supported."
        )
    if has_hermite and has_argyris:
        raise NotImplementedError("jno.fem (non-nodal): mixing Hermite and Argyris fields is not supported.")
    # A nodal Dirichlet pins VERTEX-VALUE DOFs, so its applicability is a property of the FIELD it
    # targets, not of the problem: in the mixed A-V pair (N1E x Lagrange) the Dirichlet legitimately
    # constrains the Lagrange potential V (terminal voltage) while the N1E field A takes its essential
    # BC as the edge trace. The old check was GLOBAL ("an edge field exists somewhere, therefore
    # reject every nodal Dirichlet"), which made V = g on a terminal unreachable. Resolve each
    # constraint's target field and judge per space instead.
    for _fk, _dregion, _comp, _dval, _dvnode in dirichlet_raw:
        _fi = field_index.get(_fk)
        if _fi is None:
            raise ValueError(
                f"jno.fem (non-nodal): a Dirichlet condition on region {_dregion!r} targets a field "
                f"that appears in no volume term (known fields: {list(field_index)}). An essential "
                "condition must constrain one of the solved unknowns -- is the field's trial missing "
                "from the weak form?"
            )
        if spaces[_fi] in ("RT", "N1E", "N2E"):
            raise NotImplementedError(
                "jno.fem (non-nodal): nodal Dirichlet is not applicable to RT/N1E/N2E; the essential BC is "
                "the edge trace -- edge and face trace at degree k -- (u·n for RT, u×n for N1E/N2E) -- write it as `dot(u(region), n_region) - g` "
                "(RT) or `u(region).vector.cross(n) - g` (3-D N1E/N2E). "
                "(A vertex-valued field -- Lagrange / Hermite / Argyris / Morley -- DOES take a nodal "
                "value Dirichlet u(region) - g.)"
            )
        if spaces[_fi] in ("P0", "DG"):
            raise NotImplementedError(
                "jno.fem (non-nodal): a P0 (cell-DOF) field carries no vertex values, so a nodal "
                "Dirichlet u(region) - g does not apply to it; constrain it weakly instead."
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
    # A runtime parameter may also live in a BOUNDARY term. The N1E tangential-trace surface/incident BCs
    # re-assemble their coefficient per args differentiably (`_assemble_n1e_surface_mass/_load(params=)`);
    # the host-assembled RT natural-pressure / plate BCs cannot, and are rejected below after classification.
    for _terms in (boundary_terms or {}).values():
        for bare in _terms:
            _collect_runtime_parameter_exprs(bare, _rt_param_exprs)
    runtime_parameter_tags: Tuple[str, ...] = tuple(sorted(_rt_param_exprs))
    # `domain.by_tag` is a per-FACET coefficient, threaded only by the nodal (fem_native) surface
    # kernel. Reject it here explicitly rather than trusting that it happens to reach the evaluator:
    # this path CLASSIFIES boundary terms by pattern (`_n1e_surface_mass_spec` and friends) and lifts
    # the matched coefficient out, so an unevaluated TagMask inside one could be carried into a
    # host-assembled surface mass and silently weight every facet alike.
    from .fem_utils import _collect_tag_mask_names as _tag_names_nn

    _bad_tags = sorted({t for _terms in (boundary_terms or {}).values() for bare in _terms for t in _tag_names_nn(bare)})
    if _bad_tags:
        raise NotImplementedError(
            f"jno.fem: domain._by_tag({_bad_tags}) is not supported on a non-nodal space (N1E / RT / "
            f"Morley / Argyris) -- its per-facet mask is threaded only by the nodal Lagrange surface "
            f"kernel. Write one boundary term per tag instead, or use a Lagrange space."
        )
    # Per-region volume integration: a `RegionMask` node restricts a term to one material's cells. The
    # evaluator reads its 0/1 per-cell value out of `volume_vars` at the slot AFTER the temporal and
    # runtime-parameter slots (layout [temporal..., runtime_param..., region_mask...]) and raises loudly
    # if the assembly path did not thread it -- which this (non-nodal) path previously never did, so
    # per-material eps on an N1E/RT form was unreachable. Mirrors `fem_native`.
    from .fem_utils import _CHUNK_CONSUMED, _CHUNK_OVERRIDE, _cell_region_mask, _collect_region_mask_names
    from .fem_utils import cell_chunk as _cell_chunk_of
    from .fem_utils import elem_map as _elem_map

    # Same element-chunk policy as the native assembler (see `fem_utils`): a single `vmap` over every
    # cell materialises the whole batched intermediate at once, and on a 3-D vector mesh that is what
    # sets the ceiling. Captured here, once, because the closures below run at solve time.
    _chunk_setting = _CHUNK_OVERRIDE[0]
    _CHUNK_CONSUMED[0] = True

    region_mask_names: Tuple[str, ...] = tuple(
        sorted(
            {
                r
                for bare in volume_terms
                for _, sub in _split_additive_terms(domain, bare)
                for r in _collect_region_mask_names(_lower_statefield_to_trial(sub, {}))
            }
        )
    )
    region_mask_arrays = [jnp.asarray(_cell_region_mask(domain, r)).reshape(-1) for r in region_mask_names]
    _region_mask_index = {r: i for i, r in enumerate(region_mask_names)}  # O(1) lookup vs list().index() per cell

    def _cell_masks(c, rnames):  # this cell's 0/1 indicator per region the term references
        return tuple(region_mask_arrays[_region_mask_index[r]][c] for r in rnames)

    _field_param_names = frozenset(n for n, e in _rt_param_exprs.items() if _is_fem_field_parameter(e))
    # A P0 ("cell") field parameter has no gather on this path: `_field_param_names` is used below as
    # if every field parameter were P1 VERTEX data (`_fv[name][cells_j[c]]`, interpolated with
    # `p1_shape_vals`). Handing it a per-cell array reads the values at vertex ids -- wrong numbers,
    # out-of-range indices clamped by JAX, and no error. Refuse it, and name what does work here.
    from .parametric_helpers import _fem_field_kind

    _cell_param_names = sorted(n for n in _field_param_names if _fem_field_kind(_rt_param_exprs[n]) == "cell")
    if _cell_param_names:
        raise NotImplementedError(
            f"jno.fem (non-nodal): the P0 (per-cell) field parameter(s) {_cell_param_names} are not "
            "wired on this assembler -- only nodal P1 field parameters are, and a per-cell array would "
            "be gathered at VERTEX indices and silently give the wrong coefficient. For a per-region "
            'material use d._by_region({"steel": 16.0, ...}) (or d.attach), which is one value per cell '
            "and is threaded through the per-cell region masks; for a smooth field, use a P1 "
            "(Lagrange) parameter."
        )

    # Neural coefficients (``jno.nn.wrap(net)`` in the weak form) on a non-nodal element: like the native
    # path, the network is re-evaluated at the quad points and its weights ride the runtime ``args`` as a
    # ``ModelWeights`` slot (frozen nets evaluate from their stored module and keep the system
    # non-parametric). The network is independent of the C¹ trial's DOF layout -- exactly the property the
    # P1 field parameter relies on -- so ``net(x)`` and a constitutive ``net(u)`` both thread unchanged. The
    # collect / crux-delivery / kernel-table mechanism is shared with the native assembler
    # (``parametric_helpers``); the non-nodal difference is only the boundary policy: a trainable net in a
    # natural-BC (Neumann/Robin) term is rejected, since that load is assembled non-differentiably here.
    from .parametric_helpers import collect_neural_slots, neural_local_table, neural_operator_exprs

    # Collect boundary networks into the slots too (their weights ride ``args`` as ModelWeights): the N1E
    # tangential-trace surface/incident coefficient is re-evaluated per args differentiably (like a scalar
    # parameter). A trainable net in a host-assembled RT-pressure / plate term is rejected below (after
    # classification), so it is never baked non-differentiably.
    _neural = collect_neural_slots(
        volume_terms, boundary_terms, runtime_parameter_tags=runtime_parameter_tags, reject_trainable_boundary=False
    )
    neural_param_names, _neural_models = _neural.param_names, _neural.models
    _param_and_neural_exprs = neural_operator_exprs(_rt_param_exprs, _neural)

    # Trainable mesh coordinates (`domain.variable(region).trainable()`) -- the shape-derivative
    # handle. They are scattered into the geometry points before the cell Jacobian is formed, so
    # d(solve)/dX flows through the ordinary assembly. They ride `runtime_parameter_exprs` (so crux
    # discovers them and their value arrives in `args`) but stay OUT of `runtime_parameter_tags`:
    # they are not term coefficients, and `rt_scalar` must not pack them. Collected after
    # `runtime_parameter_tags` is frozen above, which keeps that separation automatic.
    _coord_specs: List[Tuple[Any, int, str]] = []
    for _cspec in getattr(domain, "_trainable_coords", None) or []:
        _cname = str(_cspec["name"])
        _param_and_neural_exprs = {**_param_and_neural_exprs, _cname: _cspec["expr"]}
        _coord_specs.append((jnp.asarray(_cspec["ids"], dtype=jnp.int32), int(_cspec["axis"]), _cname))

    # Simplex dimension from the mesh: 2D triangle vs 3D tetrahedron. The H(curl)/H(div) push-forwards
    # and the entity DOF maps (:mod:`fem_dofmap`) are dimension-agnostic, so N1E / N2E / RT / P0 /
    # Lagrange assemble on both. The vertex families (Hermite/Argyris/Morley) are 2-D plate elements.
    dim = 3 if "tetra" in domain.mesh.cells_dict else 2
    if dim == 3 and any(s not in ("N1E", "N2E", "RT", "P0", "DG", "Lagrange") for s in spaces):
        raise NotImplementedError(
            "jno.fem (non-nodal): on a 3D (tetrahedral) mesh the supported spaces are N1E, N2E, RT, P0, DG "
            f"and Lagrange; got spaces {spaces}. Hermite / Argyris / Morley are 2-D (triangle) plate "
            "elements."
        )
    _cell_name = "tetrahedron" if dim == 3 else "triangle"
    pts = jnp.asarray(np.asarray(domain.mesh.points))[:, :dim]

    def _apply_coord_params(p, args):
        """Scatter trainable coordinate parameters into the geometry points.

        A no-op without them. With them, the returned points are TRACED, so the cell Jacobian, detJ
        and the quadrature coordinates downstream all become differentiable in the node positions --
        which is the whole shape derivative. The static `pts` above is still what every DOF-locating
        helper (Dirichlet pins, surface bucketing) uses: those pick *which* DOFs, not *where*, and
        must stay concrete."""
        if not _coord_specs or args is None:
            return p
        for _ids, _axis, _name in _coord_specs:
            if _name in args:
                p = p.at[_ids, _axis].set(jnp.asarray(args[_name], dtype=p.dtype).reshape(-1))
        return p

    cells = np.asarray(domain.mesh.cells_dict["tetra" if dim == 3 else "triangle"], dtype=np.int64)
    n_cells = int(cells.shape[0])
    n_verts = int(pts.shape[0])
    cells_j = jnp.asarray(cells, dtype=jnp.int32)

    # --- frozen (known) fields: `ui.bind(...).freeze(values)` used as a COEFFICIENT ------------------
    # Mirrors the native path (fem_native `_collect_frozen_fields` / `_frozen_gathered`), with one
    # difference: there, a frozen field is a pinned copy of a field that is already among the solved
    # unknowns, so it gathers on `cells_f_j[field_index[key]]`. Here the trial is N1E (edge DOFs) and a
    # frozen coefficient lives on the P1 VERTEX space instead — the same space the P1 field parameters
    # already use — so it gathers on `cells_j` and is interpolated with `p1_shape_vals`. That is what
    # lets a computed source (e.g. J_s from an electrokinetic pre-solve) enter an H(curl) form.
    def _collect_frozen_fields(terms):
        from ...trace import FrozenField
        from .solver_helper import iter_children

        found: dict = {}
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

    _frozen_nodes = _collect_frozen_fields(
        list(volume_terms) + [e for exprs in (boundary_terms or {}).values() for e in exprs]
    )
    # Per-cell nodal slice, a compile-time constant (no args threading, no jacfwd tangent).
    # scalar (n_nodes,) -> (n_cell, n_local, 1); VECTOR (n_nodes, vec) -> (n_cell, n_local, vec).
    _frozen_gathered: dict = {}
    for _fid, _fnode in _frozen_nodes.items():
        _fvals = jnp.asarray(_fnode.values)
        if _fvals.shape[0] != n_verts:
            raise ValueError(
                f"jno.fem (non-nodal): a frozen field coefficient must carry one value per MESH VERTEX "
                f"({n_verts}); got {_fvals.shape[0]}. On an N1E form the frozen field is a P1 coefficient, "
                "not a copy of the (edge-DOF) unknown."
            )
        _frozen_gathered[_fid] = (
            _fvals[cells_j].reshape(n_cells, cells_j.shape[1], 1) if _fvals.ndim == 1 else _fvals[cells_j]
        )

    # --- quadrature for the polynomial degree in play. Degree 1 keeps the caller's rule untouched (the
    # lowest-order operators stay bit-identical); degree k needs the mass (degree 2k) exact plus room
    # for a coefficient, so the rule is raised to 2k + 2 when the caller's is lower. ---
    _kmax = max([o for o, s_ in zip(_orders, spaces) if s_ in _EDGE_FAMILIES + ("Lagrange", "DG")] + [1])
    if _kmax > 1:
        quad_degree = max(int(quad_degree), 2 * _kmax + 2)

    # --- H(curl)/H(div) families (RT, N1E, N2E), any degree: contravariant/covariant Piola after the
    # per-cell basix DOF transformation (:mod:`fem_dofmap`). One entry per FIELD, so two fields of the
    # same family may carry different degrees. ---
    from .fem_dofmap import build_dofmap
    from .fem_elements import vector_element_spec

    edge_ref = {}  # field index -> (ref_values, ref_diffop, ref_grads, piola_fn, piola_grad_fn)
    specs = {}  # (family, degree) -> ElementSpec
    dofmaps = {}  # field index -> DofMap (edge families, and Lagrange of order >= 2)
    _dm_cache = {}
    _cells_for_dm = np.asarray(domain.mesh.cells_dict["tetra" if dim == 3 else "triangle"], dtype=np.int64)
    _n_pts_dm = int(np.asarray(domain.mesh.points).shape[0])

    _edge_top_dm = []  # one edge numbering shared by every map (the same one `top` below uses)

    def _dofmap_for(fam, k):
        if (fam, k) not in _dm_cache:
            if not _edge_top_dm:
                _t = build_edge_topology(_cells_for_dm, BASIX_TET_EDGES if dim == 3 else BASIX_TRIANGLE_EDGES)
                _edge_top_dm.append(_t)
            _t = _edge_top_dm[0]
            _dm_cache[(fam, k)] = build_dofmap(
                _cells_for_dm, fam, k, n_verts=_n_pts_dm, edges=(_t.cell_edges, _t.edge_vertices)
            )
        return _dm_cache[(fam, k)]

    for _i, _s in enumerate(spaces):
        if _s in _EDGE_FAMILIES:
            _key = (_s, _orders[_i])
            if _key not in specs:
                specs[_key] = vector_element_spec(_s, _cell_name, _orders[_i], quad_degree)
            _sp_ = specs[_key]
            if _s == "RT":
                edge_ref[_i] = (
                    _sp_.ref_values,
                    _sp_.ref_div,
                    _sp_.ref_grads,
                    piola_contravariant,
                    piola_contravariant_grad,
                )
            else:
                edge_ref[_i] = (_sp_.ref_values, _sp_.ref_curl, _sp_.ref_grads, piola_covariant, piola_covariant_grad)
            dofmaps[_i] = _dofmap_for(_s, _orders[_i])
        elif _s == "Lagrange" and _orders[_i] > 1:
            dofmaps[_i] = _dofmap_for("Lagrange", _orders[_i])
        elif _s == "DG":  # discontinuous P_k: all DOFs interior to a cell, no orientation
            dofmaps[_i] = _dofmap_for("DG", _orders[_i])
    # ``ref_curl`` is ``None`` for the 3-D H(curl) families (their curl is a vector recovered from the
    # physical gradient, not a tabulated scalar) -- keep it ``None`` through the jnp conversion.
    edge_ref = {k: tuple(jnp.asarray(a) if a is not None else None for a in v[:3]) + v[3:] for k, v in edge_ref.items()}
    # Per-field orientation: exact ±1 signs while the transform is diagonal (degree 1 -- the historic
    # sign path, bit for bit), else the per-cell block transform B is applied to the reference data.
    edge_signs = {
        i: (jnp.asarray(dm_.signs) if dm_.is_diagonal else None)
        for i, dm_ in dofmaps.items()
        if spaces[i] in _EDGE_FAMILIES
    }
    # The per-cell transform tables, built HERE (eagerly) and held as plain tuples so `elem_map` lifts
    # their mesh-sized orientation arrays out of the compiled kernel rather than baking them in.
    from .fem_dofmap import transform_from_tables as _B_of

    dm_tables = {i: (dm_.jax_tables(), int(dm_.ndof_local)) for i, dm_ in dofmaps.items() if not dm_.is_diagonal}

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
        else (next(iter(specs.values()), None) or vector_element_spec("RT", _cell_name, 1, quad_degree))
    )
    qp, qw = jnp.asarray(ref_spec.quad_points), jnp.asarray(ref_spec.quad_weights)
    n_quad = int(qw.shape[0])
    ctx = getattr(domain, "context", {}) or {}
    # P1 (linear) shape functions at the quad points -- used ONLY to interpolate a P1 field parameter k(x)
    # from its mesh-vertex values per cell (3 barycentric on a triangle, 4 on a tetrahedron), independent of
    # the trial element's own basis. The kernel contraction ``shape_vals . cell_nodal`` is vertex-count
    # agnostic, so only this reference-basis table is dimension-specific.
    if _field_param_names or _frozen_gathered or "Lagrange" in spaces:
        _bary0 = 1.0 - qp[:, 0] - qp[:, 1] - (qp[:, 2] if dim == 3 else 0.0)
        p1_shape_vals = jnp.stack([_bary0, qp[:, 0], qp[:, 1]] + ([qp[:, 2]] if dim == 3 else []), axis=1)
    else:
        p1_shape_vals = None
    # Lagrange P_k (k >= 2) mixed into the non-nodal path: basix GLL-warped tabulation at the shared
    # quadrature, values (n_quad, n_dof) and reference gradients (n_quad, n_dof, dim).
    lagrange_ref = {}
    for _i, _dm in dofmaps.items():
        if spaces[_i] in ("Lagrange", "DG"):
            _tab = _dm.element.tabulate(1, np.asarray(ref_spec.quad_points))
            lagrange_ref[_i] = (
                jnp.asarray(_tab[0][..., 0]),
                jnp.asarray(np.stack([_tab[1 + a][..., 0] for a in range(dim)], axis=-1)),
                None if _dm.is_diagonal else True,
            )

    # Edge topology: edge families (RT/N1E) need it for their edge DOFs; Argyris/Morley need it for the global
    # id + orientation of their edge-normal DOFs (Hermite, pure-vertex, does not).
    if has_edge or has_argyris or has_morley:
        top = (
            _edge_top_dm[0]
            if _edge_top_dm
            else build_edge_topology(cells, BASIX_TET_EDGES if dim == 3 else ref_spec.local_edges)
        )
        ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)  # (n_cells, 3) global edge ids
        n_edges = int(top.n_edges)
    else:
        top, ce, n_edges = None, None, 0

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
        # Stash the C¹ topology the periodic (non-nodal) prolongation builder needs: Morley value DOFs
        # live at the mesh vertices, its normal-derivative DOFs at the global edges (midpoint + oriented
        # normal ``_en``). Read back in ``_fem._build_periodic_reduction_nonnodal`` when ties are present.
        _pts_np = np.asarray(pts)
        domain._fem_nonnodal_topology = {
            "n_verts": int(_pts_np.shape[0]),
            "n_edges": int(n_edges),
            "vertex_points": _pts_np,
            "edge_vertices": _ev,
            "edge_midpoints": 0.5 * (_pts_np[_ev[:, 0]] + _pts_np[_ev[:, 1]]),
            "edge_normals": _en,
            "family": "Morley" if has_morley else ("Argyris" if has_argyris else "Hermite"),
        }

    # N1E (H(curl) edge) topology for periodic (Floquet/Bloch) ties: each DOF is one edge's tangential
    # moment, so a periodic tie matches boundary edges across faces (by midpoint) with an orientation sign
    # (the lo→hi edge direction). Read back in ``_fem._build_periodic_reduction_n1e`` when ties are present.
    _hcurl_i = next((i for i, s_ in enumerate(spaces) if s_ in ("N1E", "N2E")), None)
    _edge_i = _hcurl_i if _hcurl_i is not None else next((i for i, s_ in enumerate(spaces) if s_ == "RT"), None)
    if _edge_i is not None and top is not None:
        _evn = np.asarray(top.edge_vertices)
        _ptsn = np.asarray(pts)
        _dm_e = dofmaps[_edge_i]
        domain._fem_nonnodal_topology = {
            "n_verts": int(_ptsn.shape[0]),
            "n_edges": int(n_edges),
            "vertex_points": _ptsn,
            "edge_vertices": _evn,
            "edge_midpoints": 0.5 * (_ptsn[_evn[:, 0]] + _ptsn[_evn[:, 1]]),
            "edge_dirs": _ptsn[_evn[:, 1]] - _ptsn[_evn[:, 0]],  # lo→hi direction (sets the tangential sign)
            # The entity DOF map of the edge field: degree, every DOF's entity and orientation. The
            # degree-k consumers (AMS transfer operators, periodic ties, field read-back) build from it;
            # the lowest-order fields above are kept so degree-1 consumers are untouched.
            "family": spaces[_edge_i],
            "degree": int(_orders[_edge_i]),
            "n_dofs": int(_dm_e.n_dofs),
            "dofmap": _dm_e,
            "cells": np.asarray(cells),
            # every field's (space, DofMap or None) -- the degree-k periodic reduction ties each field
            # through its own map (None: P1 Lagrange / P0, whose maps are the vertex / cell ids)
            "field_dofmaps": [(spaces[i], dofmaps.get(i)) for i in range(len(spaces))],
        }
        if spaces[_edge_i] != "N1E" or _orders[_edge_i] > 1:
            # the lowest-order `n_edges`-sized fields above would size a G / P wrongly; consumers that
            # need the DOF count read `n_dofs`, and the degree-1-only builders check this flag.
            domain._fem_nonnodal_topology["lowest_order_n1e"] = False
        else:
            domain._fem_nonnodal_topology["lowest_order_n1e"] = True

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

    def _field_ndof(i):
        s = spaces[i]
        if i in dofmaps:  # edge families (any degree) and Lagrange P_k, k >= 2: entity DOF map
            return int(dofmaps[i].n_dofs)
        if s == "Hermite":
            return 3 * n_verts + n_cells
        if s == "Argyris":
            return 6 * n_verts + n_edges
        if s == "Morley":
            return n_verts + n_edges
        if s == "Lagrange":
            return n_verts
        return n_cells

    ndof = [_field_ndof(i) for i in range(len(spaces))]
    offs = [0]
    for n in ndof:
        offs.append(offs[-1] + n)
    total = offs[-1]

    def _field_cdofs(i):
        if i in dofmaps:  # (n_cells, n_local) -- equals the edge ids `ce` for degree-1 N1E / 2-D RT
            return offs[i] + jnp.asarray(dofmaps[i].cell_dofs, dtype=jnp.int32)
        if spaces[i] == "Hermite":
            return offs[i] + hermite_cdofs  # (n_cells, 10)
        if spaces[i] == "Argyris":
            return offs[i] + argyris_cdofs  # (n_cells, 21)
        if spaces[i] == "Morley":
            return offs[i] + morley_cdofs  # (n_cells, 6)
        if spaces[i] == "Lagrange":
            return offs[i] + cells_j  # (n_cells, dim+1)
        return offs[i] + jnp.arange(n_cells)[:, None]  # (n_cells, 1) P0

    cdofs = [_field_cdofs(i) for i in range(len(fields))]

    def _cell_local_sols(c, u_blocks):
        """This cell's LOCAL DOF values per field: ``u_blocks[i][cdofs[i][c] - offs[i]]``. Decoupling the
        per-cell slice from the global vector is what lets the matrix be assembled per element (``jacfwd``
        w.r.t. these local dofs) instead of via one global ``jacfwd`` that materialises an O(n_dof × n_cell)
        tangent."""
        return [u_blocks[i][cdofs[i][c] - offs[i]] for i in range(len(fields))]

    # Reference tables every field reads in `_cell_fields`: the volume quadrature by default; the
    # general boundary path hands the same structure tabulated at FACE points (`_tabulate_at`).
    _VOL_TABS = {
        "pts": qp,
        "n": n_quad,
        "p1": p1_shape_vals,
        "edge": {i: v[:3] for i, v in edge_ref.items()},
        "lag": {i: v[:2] for i, v in lagrange_ref.items()},
        "hermite": hermite_ref,
        "argyris": None if argyris_ref is None else argyris_ref[:3],
        "morley": None if morley_ref is None else morley_ref[:3],
    }

    def _cell_fields(c, cell_sols, pts_dyn=None, tabs=None):
        T = _VOL_TABS if tabs is None else tabs
        # `pts_dyn` is the coordinate-parameter-scattered geometry; it defaults to the static mesh.
        # Defaulting is REFUSED when coordinates are trainable rather than silently falling back --
        # a static fallback there returns a shape derivative of exactly zero, which looks like a
        # converged design instead of a missing feature.
        if pts_dyn is None:
            if _coord_specs:
                raise NotImplementedError(
                    "jno.fem (non-nodal): trainable mesh coordinates are threaded through the steady "
                    "linear assembly only. This residual / transient / nonlinear path still reads the "
                    "static mesh, so d(solve)/dX would come back as exactly zero. Use a steady linear "
                    "form, or extend the threading to this path."
                )
            pts_dyn = pts
        verts = pts_dyn[cells_j[c]]
        J = jnp.stack([verts[k] - verts[0] for k in range(1, dim + 1)], axis=1)  # (dim, dim): 2x2 tri / 3x3 tet
        detJ = small_det(J)
        per = []
        for i, s in enumerate(spaces):
            if s in ("Lagrange", "DG") and i in lagrange_ref:
                # P_k (k >= 2): basix tabulation, the per-cell DOF transform (a permutation from P3 on,
                # identity for P2), and the isoparametric chain rule for the gradient.
                lv, lg = T["lag"][i]
                _needs_B = lagrange_ref[i][2]
                if _needs_B is not None:
                    Bc = _B_of(dm_tables[i][0], dm_tables[i][1], c)
                    lv = jnp.einsum("ij,qj->qi", Bc, lv)
                    lg = jnp.einsum("ij,qjd->qid", Bc, lg)
                per.append(
                    {
                        "shape_vals": lv,
                        "shape_grads": lg @ jnp.linalg.inv(J),
                        "cell_sol": cell_sols[i][:, None],
                        "space": "Lagrange",
                    }
                )
                continue
            if s == "Lagrange":
                # P1 on a simplex: barycentric values (already tabulated as `p1_shape_vals` for the
                # field-parameter path) and CONSTANT gradients, J^-T @ ref. Tagged "Lagrange" for the
                # same reason Hermite is — the shared evaluator reads value / .x from scalar shape data.
                # `cell_sol` takes the [:, None] SCALAR convention Hermite/Argyris/Morley use (vec = 1
                # as an explicit column); only the vector families (RT/N1E) pass it flat.
                g_ref = jnp.concatenate([-jnp.ones((1, dim)), jnp.eye(dim)], axis=0)  # (dim+1, dim)
                g = g_ref @ small_inv(J)  # (dim+1, dim)
                per.append(
                    {
                        "shape_vals": T["p1"],  # (n_quad, dim+1)
                        "shape_grads": jnp.broadcast_to(g, (T["n"], dim + 1, dim)),
                        "cell_sol": cell_sols[i][:, None],
                        "space": "Lagrange",
                    }
                )
                continue
            if s == "Hermite":  # C0 vertex value+derivative DOFs via the M(cell) DOF-transform
                rv, rg, rh = T["hermite"]
                phi, grad, hess = hermite_pushforward(rv, rg, rh, J, detJ, None)
                # Tag "Lagrange": the M(cell) transform is baked into phi/grad/hess, so this SCALAR field's
                # shape data matches nodal Lagrange and the shared evaluator (value / .x / Hessian) serves
                # it unchanged. cell_sol = this cell's 10 local DOF values, (10, 1).
                per.append(
                    {
                        "shape_vals": phi,
                        "shape_grads": grad,
                        "shape_hess": hess,
                        "cell_sol": cell_sols[i][:, None],
                        "space": "Lagrange",
                    }
                )
            elif s == "Argyris":  # C1 conforming: M(cell) DOF-transform + globally-oriented edge-normal DOFs
                (rv, rg, rh), nodal = T["argyris"], argyris_ref[3]
                phi, grad, hess = argyris_pushforward(rv, rg, rh, J, detJ, argyris_normals[c], nodal)
                per.append(
                    {
                        "shape_vals": phi,
                        "shape_grads": grad,
                        "shape_hess": hess,
                        "cell_sol": cell_sols[i][:, None],  # this cell's 21 local DOFs
                        "space": "Lagrange",
                    }
                )
            elif s == "Morley":  # non-conforming biharmonic: M(cell) DOF-transform + globally-oriented edge normal
                (rv, rg, rh), nodal = T["morley"], morley_ref[3]
                phi, grad, hess = morley_pushforward(rv, rg, rh, J, detJ, argyris_normals[c], nodal)
                per.append(
                    {
                        "shape_vals": phi,
                        "shape_grads": grad,
                        "shape_hess": hess,
                        "cell_sol": cell_sols[i][:, None],  # this cell's 6 local DOFs
                        "space": "Lagrange",
                    }
                )
            elif i in edge_ref:  # RT (contravariant) or N1E/N2E (covariant): entity DOFs, family-specific push-forward
                (rval, rdop, rgr), (pf, pgf) = T["edge"][i], edge_ref[i][3:]
                _sg = edge_signs[i]
                if _sg is not None:  # diagonal ±1 transform (degree 1): the historic sign path
                    phi, _d = pf(rval, rdop, J, detJ, _sg[c])  # (n_quad, n_dof, d)
                    grad = pgf(rgr, J, detJ, _sg[c])  # (n_quad, n_dof, d, d)
                else:  # degree k: B = T(cell_info)^-1 mixes each entity's DOFs, then the Piola map
                    Bc = _B_of(dm_tables[i][0], dm_tables[i][1], c)
                    _ones = jnp.ones((Bc.shape[0],), dtype=Bc.dtype)
                    phi, _d = pf(
                        jnp.einsum("ij,qjv->qiv", Bc, rval),
                        None if rdop is None else jnp.einsum("ij,qj->qi", Bc, rdop),
                        J,
                        detJ,
                        _ones,
                    )
                    grad = pgf(jnp.einsum("ij,qjvm->qivm", Bc, rgr), J, detJ, _ones)
                per.append({"shape_vals": phi, "shape_grads": grad, "cell_sol": cell_sols[i], "space": s})
            else:  # P0: a single constant DOF per cell
                per.append(
                    {
                        "shape_vals": jnp.ones((T["n"], 1)),
                        "shape_grads": jnp.zeros((T["n"], 1, dim, dim)),
                        "cell_sol": cell_sols[i],
                        "space": "P0",
                    }
                )
        return per, verts[0][None, :] + T["pts"] @ J.T, jnp.abs(detJ)

    # ---- GENERAL boundary integrands (any family, any degree, 2-D and 3-D) ------------------------
    # A boundary term the legacy patterns below do not own is evaluated like a volume term: per
    # boundary facet, every field's trace (its owner cell's basis at the facet's quadrature points,
    # through the same DOF transform and Piola map), the facet's outward normal as `n_<region>`, and
    # the shared evaluator. `u.across(main)` adds the value of `u` on ANOTHER face at the facing
    # points -- linear in that face's cell DOFs, which become extra columns of the facet's block.
    _all_cdofs = jnp.concatenate([cdofs[i] for i in range(len(fields))], axis=1)  # (n_cells, n_local_all)
    _all_splits = list(np.cumsum([0] + [int(cdofs[i].shape[1]) for i in range(len(fields))]))
    _GS: dict = {"groups": []}  # filled after the boundary classification

    def _tabulate_at(Xr):
        """Every field's reference tables at reference-cell points ``Xr`` -- the `_cell_fields` input."""
        Xr = np.asarray(Xr, dtype=float)
        T = {"pts": jnp.asarray(Xr), "n": int(Xr.shape[0]), "edge": {}, "lag": {}}
        T["p1"] = jnp.asarray(np.column_stack([1.0 - Xr.sum(axis=1), Xr]))
        for _i, _s in enumerate(spaces):
            if _i in edge_ref:
                _tab = dofmaps[_i].element.tabulate(1, Xr)
                _rv = np.asarray(_tab[0])
                _rg = np.stack([np.asarray(_tab[1 + a]) for a in range(dim)], axis=-1)
                if _s == "RT":
                    _rd = sum(_rg[:, :, a, a] for a in range(dim))
                elif dim == 2:
                    _rd = _rg[:, :, 1, 0] - _rg[:, :, 0, 1]
                else:
                    _rd = None
                T["edge"][_i] = (jnp.asarray(_rv), None if _rd is None else jnp.asarray(_rd), jnp.asarray(_rg))
            elif _i in lagrange_ref:
                _tab = dofmaps[_i].element.tabulate(1, Xr)
                T["lag"][_i] = (
                    jnp.asarray(_tab[0][..., 0]),
                    jnp.asarray(np.stack([_tab[1 + a][..., 0] for a in range(dim)], axis=-1)),
                )
        T["hermite"] = T["argyris"] = T["morley"] = None
        if has_hermite:
            import basix as _bx

            from .fem_lagrange import _ref_hessian_from_tab

            _tab = _bx.create_element(_bx.ElementFamily.Hermite, _bx.CellType.triangle, 3).tabulate(2, Xr)
            T["hermite"] = (
                jnp.asarray(_tab[0]),
                jnp.asarray(np.stack([_tab[1], _tab[2]], axis=-1)),
                jnp.asarray(_ref_hessian_from_tab(_tab, 2)),
            )
        if has_argyris:
            from .fem_elements import argyris_ref_basis_at

            T["argyris"] = argyris_ref_basis_at(Xr)
        if has_morley:
            from .fem_elements import morley_ref_basis_at

            T["morley"] = morley_ref_basis_at(Xr)
        return T

    def _select_tabs(TS, k):
        """The tables of local facet ``k`` (traced) out of the per-local-facet stack ``TS``."""

        def _sel(a):
            return None if a is None else a[k]

        return {
            "pts": TS["pts"][k],
            "n": TS["n"],
            "p1": TS["p1"][k],
            "edge": {i: tuple(_sel(a) for a in v) for i, v in TS["edge"].items()},
            "lag": {i: tuple(_sel(a) for a in v) for i, v in TS["lag"].items()},
            "hermite": None if TS["hermite"] is None else tuple(_sel(a) for a in TS["hermite"]),
            "argyris": None if TS["argyris"] is None else tuple(_sel(a) for a in TS["argyris"]),
            "morley": None if TS["morley"] is None else tuple(_sel(a) for a in TS["morley"]),
        }

    def _facet_res(g, f, la, lbs, coeffs, rt_scalar, field_vals, nt, pts_dyn):
        """Residual (test DOFs of the group's field) of the group's terms on facet ``f`` from the owner
        cell's all-field local DOFs ``la`` and, per ``across`` key, the main cells' DOF values ``lbs``."""
        c, k = g["cell"][f], g["kf"][f]
        tabs = _select_tabs(g["tabs"], k)
        cell_sols = [la[_all_splits[i] : _all_splits[i + 1]] for i in range(len(fields))]
        per, xq, _m = _cell_fields(c, cell_sols, pts_dyn, tabs)
        fv = pts_dyn[cells_j[c]][g["lv"][k]]  # facet vertices
        if dim == 3:
            nr = jnp.cross(fv[1] - fv[0], fv[2] - fv[0])
        else:
            _t = fv[1] - fv[0]
            nr = jnp.stack([-_t[1], _t[0]])
        meas = jnp.linalg.norm(nr)
        nvec = g["sgn"][f] * nr / meas  # outward from the field domain
        dctx = {**ctx, f"n_{g['region']}": nvec}
        for key, (_dofs, phiB) in g["across"].items():
            dctx[key] = jnp.einsum("qj,qjv->qv", lbs[key], phiB[f])
        vol_vars = tuple(
            (field_vals[name][cells_j[c]] if name in _field_param_names else rt_scalar[name])
            for name in runtime_parameter_tags
        )
        local = {
            "physical_quad_points": xq,
            "fields": per,
            "field_index": field_index,
            "tag": f"gauss_{g['region']}",
            "surface": True,
            "domain_context": dctx,
            "temporal_tags": (),
            "runtime_parameter_tags": runtime_parameter_tags,
            "region_mask_names": (),
            "volume_vars": vol_vars,
        }
        if _field_param_names or _frozen_gathered:
            local["shape_vals"] = tabs["p1"]
        if _frozen_gathered:
            local["frozen_fields"] = {fid: gg[c] for fid, gg in _frozen_gathered.items()}
        if nt is not None:
            local["neural_coefficients"] = nt
        w = g["fqw"] * meas
        r = None
        for e in coeffs:
            t_ = _integrate_term(domain, e, local, w)
            r = t_ if r is None else r + t_
        return r

    def _unflatten_lbs(flat, shapes):
        """Inverse of the per-facet concatenation of the across blocks: ``{key: (n_q, n_dof_main)}``."""
        out, o = {}, 0
        for k_, shp in shapes:
            n_ = int(np.prod(shp))
            out[k_] = flat[o : o + n_].reshape(shp)
            o += n_
        return out

    def _facet_inputs(g, u_flat):
        """Per facet: owner-cell local DOFs ``(n_f, n_local_all)`` and, per across key, main DOFs."""
        la = jnp.asarray(u_flat)[_all_cdofs[g["cell"]]]
        lbs = {key: jnp.asarray(u_flat)[dofs] for key, (dofs, _phi) in g["across"].items()}
        return la, lbs

    # A complex COEFFICIENT with a real basis: the element residual is complex while the
    # linearisation point stays real, so ONE data pass carries the full complex operator. The flag
    # sizes every accumulator below; without it the scatter-add into a real buffer RAISES on dtype
    # (it never silently casts), which is why the historic route split each term into Re/Im legs
    # and assembled twice.
    from ..._fem import _bares_have_complex_coeff as _cx_bares
    from ...trace import FunctionCall as _FC_cx

    def _provably_real(b):
        # A top-level ``.real``/``.imag`` wrap is the historic two-leg split's leg shape: the value
        # is real BY CONSTRUCTION even though the complex literal survives inside the wrapped tree.
        # Without this the split's legs would assemble complex-dtype -- twice the data for a zero
        # imaginary part, handed to consumers that expect real legs.
        return isinstance(b, _FC_cx) and getattr(b, "_name", None) in ("real", "imag") and len(b.args) == 1

    _cx_form = _cx_bares(
        [b for b in volume_terms if not _provably_real(b)]
        + [e for _bt in (boundary_terms or {}).values() for e in _bt if not _provably_real(e)]
    )

    def _make_residual(terms, with_gsurf=False):
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
                typed.append((coeff, tfi, tuple(sorted(_collect_region_mask_names(coeff)))))

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
            R = jnp.zeros(total, dtype=jnp.result_type(u_flat.dtype, np.complex64) if _cx_form else u_flat.dtype)
            _nt = neural_local_table(_neural, args)  # once per call, not per (term × cell) inside `_cell`
            # Scatter trainable coordinates once per residual evaluation, not per (term x cell).
            # The steady linear operator is built FROM this residual (jacfwd -> A, -residual(0) -> b),
            # so without this the shape derivative is lost before `assemble` is ever reached.
            _pts_r = _apply_coord_params(pts, _a)
            for coeff, tfi, rnames in typed:

                def _cell(c, e=coeff, _sc=rt_scalar, _fv=field_vals, _rn=rnames, _p=_pts_r):
                    per, xq, meas = _cell_fields(c, _cell_local_sols(c, u_blocks), _p)
                    _ctx_c = _with_cell_size(ctx, meas, xq, (pts if _p is None else _p)[cells_j[c]])
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
                        "domain_context": _ctx_c,
                        "temporal_tags": (),
                        "runtime_parameter_tags": runtime_parameter_tags,
                        "region_mask_names": _rn,
                        "volume_vars": vol_vars + _cell_masks(c, _rn),
                    }
                    if (
                        _field_param_names or _frozen_gathered
                    ):  # P1 basis to interpolate a field parameter (top-level, param-only key)
                        local["shape_vals"] = p1_shape_vals
                    if _frozen_gathered:  # known-field (ui.freeze) per-cell nodal slices for this cell
                        local["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
                    if _nt is not None:  # trainable nets ride args (crux weights); frozen/placeholder -> stored
                        local["neural_coefficients"] = _nt
                    return _integrate_term(domain, e, local, qw * meas)  # (ndof of the test field,)

                elem = _elem_map(  # (n_cells, ndof_tfi)
                    _cell,
                    (jnp.arange(n_cells),),
                    _cell_chunk_of(n_cells, int(cdofs[tfi].shape[1]), int(cdofs[tfi].shape[1]), _chunk_setting),
                )
                R = R.at[cdofs[tfi].reshape(-1)].add(elem.reshape(-1).astype(R.dtype))
            for g in _GS["groups"] if with_gsurf else ():  # general boundary integrands
                la, lbs = _facet_inputs(g, u_flat)

                _shp_r = [(k_, tuple(lbs[k_].shape[1:])) for k_ in lbs]  # per-facet shape of each across block
                _lb_flat = (
                    jnp.concatenate([lbs[k_].reshape(g["n_facets"], -1) for k_, _s in _shp_r], axis=1)
                    if _shp_r
                    else jnp.zeros((g["n_facets"], 0), la.dtype)
                )

                def _fr(f, la_f, lb_f, _g=g, _sc=rt_scalar, _fv=field_vals, _p=_pts_r, _ss=tuple(_shp_r)):
                    return _facet_res(_g, f, la_f, _unflatten_lbs(lb_f, _ss), _g["coeffs"], _sc, _fv, _nt, _p)

                elem = _elem_map(  # chunked like the volume path: bounded memory on a large face
                    _fr,
                    (jnp.arange(g["n_facets"]), la, _lb_flat),
                    _cell_chunk_of(
                        g["n_facets"],
                        int(cdofs[g["tfi"]].shape[1]),
                        int(la.shape[1]) + int(_lb_flat.shape[1]),
                        _chunk_setting,
                    ),
                )
                R = R.at[cdofs[g["tfi"]][g["cell"]].reshape(-1)].add(elem.reshape(-1).astype(R.dtype))
            return R

        return residual

    # --- BCs computed once, separate from per-mode application: the natural BC (p_D·(v·n)) is a constant
    #     RHS load; the essential edge-trace BC is a set of (dof, value) pins (mirrors 1D Dirichlet). ---
    # A weak boundary term is either a LOAD (trial-free → into b) or a BILINEAR surface term (trial+test →
    # into A). The only bilinear surface term wired is the N1E tangential-trace mass `c·inner(n×u, n×v)`
    # (the impedance / first-order absorbing Maxwell BC): split it off and assemble it into the stiffness.
    from ..._fem import _bare as _bare_nn
    from ...trace import FunctionCall as _FC
    from ...trace import Literal as _Lit

    # A weak boundary term is one of: an N1E tangential-trace surface MASS `c·inner(n×u, n×v)` (bilinear →
    # into A, the impedance/absorbing BC); an N1E incident LOAD `inner(g, n×v)` (trial-free → into b, the
    # source); or an RT pressure load `p·(v·n)` (→ into b). Split each term additively and classify each
    # summand, so a combined `i k₀·inner(n×u,n×v) + 2 i k₀·inner(g,n×v)` (absorbing + incident on one face)
    # is routed correctly. The complex leg wraps the whole term as `real(…)`/`imag(…)` and `.real` does NOT
    # distribute over `+`, so peel that wrapper first, split inside, then re-wrap each summand.
    def _signed(x, sign):  # fold a −1 summand sign into the extracted coefficient/source
        return (_Lit(-1.0) if x is None else (-1.0) * x) if sign < 0 else x

    pressure_terms, surface_terms, incident_terms, gsurf_terms = {}, {}, {}, {}
    _body_owned = set(getattr(domain, "_tag_regions", {}) or {})
    for region, terms in (boundary_terms or {}).items():
        for t in terms:
            bt = _bare_nn(t)
            wrap, inner_expr = None, bt
            if isinstance(bt, _FC) and getattr(bt, "_name", None) in ("real", "imag") and len(bt.args) == 1:
                wrap, inner_expr = bt._name, bt.args[0]
            for sign, sub in _split_additive_terms(domain, inner_expr):
                sub_w = sub if wrap is None else (sub.real if wrap == "real" else sub.imag)
                # non-local, or a body-owned tag that may lie INSIDE the mesh: never one of the
                # host-assembled patterns (those walk boundary facets only)
                if region in _body_owned or _reads_across(sub_w):
                    gsurf_terms.setdefault(region, []).append(_apply_sign(domain, sign, sub_w))
                elif dim == 3 and (mass := _n1e_surface_mass_spec(sub_w)) is not None:
                    surface_terms.setdefault(region, []).append(_signed(mass[0], sign))
                elif dim == 3 and (load := _n1e_surface_load_spec(sub_w)) is not None:
                    incident_terms.setdefault(region, []).append(_signed(load[0], sign))
                elif _legacy_natural(sub_w, field_index, spaces):
                    pressure_terms.setdefault(region, []).append(_apply_sign(domain, sign, sub_w))
                else:  # anything else: the general boundary-integrand path
                    gsurf_terms.setdefault(region, []).append(_apply_sign(domain, sign, sub_w))
    if gsurf_terms:
        _GS["groups"] = _build_general_surface(
            gsurf_terms,
            domain,
            dim,
            _cell_name,
            cells,
            spaces,
            field_index,
            dofmaps,
            offs,
            quad_degree,
            _dofmap_for,
            _tabulate_at,
        )
    # A runtime parameter in a host-assembled RT pressure / plate BC cannot be threaded differentiably
    # (the load is baked once); only the N1E surface/incident coefficient re-assembles per args. Reject the
    # former loudly (rather than silently freeze it) now that classification separates the two.
    from .parametric_helpers import _collect_neural_coefficient_exprs

    _pressure_param: dict = {}
    for _terms in pressure_terms.values():
        for _t in _terms:
            _collect_runtime_parameter_exprs(_t, _pressure_param)
            _collect_neural_coefficient_exprs(_t, _pressure_param)  # a trainable net is baked here too
    if _pressure_param:
        raise NotImplementedError(
            "jno.fem (non-nodal): a runtime parameter or trainable neural coefficient in a host-assembled "
            "natural-BC (RT pressure / plate moment) term is not supported (its load is baked "
            f"non-differentiably); got {sorted(_pressure_param)}. Put it in a volume term, or use an N1E "
            "tangential-trace surface/incident BC (which IS parametric / neural-differentiable)."
        )
    # RT natural pressure BC (host-assembled once, constant). The N1E surface mass (→ A) and incident load
    # (→ b) are wrapped in ``_of(params)`` closures so the parametric path re-assembles them differentiably
    # in a boundary-term parameter (an inverse-design impedance / incident source); ``params=None`` is the
    # plain forward pass for the non-parametric build.
    nat_load_rt = (
        _apply_natural_boundary_terms(
            jnp.zeros(total),
            pressure_terms,
            domain,
            field_index,
            spaces,
            top,
            np.asarray(pts),
            offs,
            n_cells,
            quad_degree,
            dofmaps=dofmaps,
        )
        if pressure_terms
        else jnp.zeros(total)
    )

    # Host-static face geometry / region membership for the N1E surface terms, built ONCE. It does not
    # depend on the runtime parameters, and rebuilding it per evaluation was both the dominant cost of a
    # parametric assembly and what made the path un-`jit`-able (a host `np.where` cannot see a tracer).
    _inc_static = (
        _n1e_surface_static("load", incident_terms, domain, spaces, top, np.asarray(pts), offs, quad_degree, dim, dofmaps)
        if incident_terms
        else None
    )
    _surf_static = (
        _n1e_surface_static("mass", surface_terms, domain, spaces, top, np.asarray(pts), offs, quad_degree, dim, dofmaps)
        if surface_terms
        else None
    )

    def _incident_of(params):  # N1E incident forcing ∫ g·(n×v) → b
        if not incident_terms:
            return jnp.zeros(total)
        return _assemble_n1e_surface_load(total, _inc_static, params=params)

    def _surf_mass_of(params, sparse=False):  # N1E tangential-trace impedance mass ∫ c(n×u)·(n×v) → A
        if not surface_terms:
            return None
        return _assemble_n1e_surface_mass(total, _surf_static, params=params, sparse=sparse)

    _incident_const = _incident_of(None)
    nat_load = nat_load_rt + _incident_const  # the constant boundary load (used by the non-parametric path)
    # SPARSE: the dense (total x total) form is O(n_dof^2) -- measured 10 GB at 36k DOFs, 284 GB asked
    # for at 188k -- and only the two small dense (C1 / second-order) paths below add it to a dense
    # matrix; they densify it there. Every sparse path re-assembles it per args (`_surf_mass_of(args,
    # sparse=True)`), so this eager copy exists only for those, and for the presence check.
    surf_mass = _surf_mass_of(None, sparse=True)

    def _surf_mass_dense():
        return None if surf_mass is None else jnp.asarray(surf_mass.todense())

    if surf_mass is not None or incident_terms:
        # The surface mass is a fixed LINEAR block added to the spatial operator A (and the incident load to
        # b), so it composes with the steady AND transient linear problem (M u̇ + A u = c: the impedance is a
        # boundary contribution to A, the incident source rides the forcing). A NONLINEAR form is not wired —
        # the surface block is not re-linearised per Newton step. Raise rather than silently drop it.
        if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in volume_terms):
            raise NotImplementedError(
                "jno.fem (non-nodal): the N1E tangential-trace surface terms (impedance / absorbing / incident BC) "
                "compose with the steady and transient LINEAR problem, but not a NONLINEAR form yet (the surface "
                "mass is a fixed linear block, not re-linearised per step). (Raises rather than silently dropping it.)"
            )
    pins = (
        _flux_bc_pins(
            flux_bcs,
            domain,
            field_index,
            spaces,
            top,
            np.asarray(pts),
            offs,
            n_cells,
            quad_degree,
            dim=dim,
            dofmaps=dofmaps,
        )
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
    if ("Lagrange" in spaces) and dirichlet_raw:  # Lagrange value-Dirichlet: pin the region's vertex DOFs to g
        pins = pins + _lagrange_dirichlet_pins(
            dirichlet_raw, domain, field_index, spaces, np.asarray(pts), offs, dofmaps=dofmaps, cells=cells
        )
    extra = getattr(domain, "_extra_dof_pins", None)
    if extra:  # caller-supplied (dof, value) pins — e.g. a tree-cotree gauge + air-V restriction,
        _bad = [d_ for d_, _v in extra if not (0 <= int(d_) < total)]
        if _bad:
            raise ValueError(
                f"jno.fem (non-nodal): domain._extra_dof_pins contains {len(_bad)} DOF indices "
                f"outside [0, {total}) (first: {_bad[0]}) — the pin list was built against a "
                "different mesh or DOF layout. Rebuild it for this domain; refusing to pin "
                "arbitrary DOFs silently."
            )
        pins = pins + [(int(d_), float(v_)) for d_, v_ in extra]  # applied identically to BOTH complex legs
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
    # Publish the essential (dof, value) pins where `FEM.eigs` (and the other consumers that compose
    # their own reduced system) read them. Written EVERY time, empty included: left unset, eigs on a
    # non-nodal form skipped the elimination -- identity rows against full mass rows inject spurious
    # pairs -- and a stash left behind by an earlier nodal build on the same domain was read instead.
    domain._fem_native_dirichlet_pairs = list(pins)
    zeros = jnp.zeros(total)

    # --- SPARSE per-element assembly of a linear operator from a bare-term list -------------------------
    # Edge/cell DOF families (N1E, RT, P0) assemble one element at a time: each cell's block is ``jacfwd`` of
    # its ELEMENT residual w.r.t. that cell's LOCAL dofs — an ``(n_test, n_local_all)`` block — scattered as
    # COO triplets into a BCOO. A single global ``jacfwd`` instead materialises an ``O(n_dof × n_cells)``
    # tangent tensor that overflows the 2³¹ XLA element limit past ~10⁴ edges. Mirrors the native (Lagrange)
    # assembler ``fem_native._make_jacobian``, and is the ONE assembler behind the steady operator A(args),
    # the transient block mass M, and the transient spatial operator A — so a 3-D vector transient (eddy
    # currents, time-domain Maxwell) marches sparse instead of hitting the dense-assembly wall.
    def _make_sparse_assembler(term_source, with_gsurf=False):
        """Return ``assemble(args=None, *, surface=True) -> BCOO`` for the bare weak terms ``term_source``.

        Only the element data and the tangential-trace surface-mass values depend on ``args``; the BCOO
        row/col PATTERN depends only on the mesh connectivity, so it is built ONCE here (host-static) rather
        than on every ``assemble(args)`` — i.e. every optimizer step of an inverse solve AND every implicit
        time step of a parametric march. ``surface=False`` omits the surface mass (used for the pure temporal
        MASS block, which carries no boundary impedance)."""
        import jax.experimental.sparse as _jsparse

        _typed = []  # (lowered coeff, test-field index, region-mask names) — identical typing to `_make_residual`
        for _bare_term in term_source:
            for _sign, _sub in _split_additive_terms(domain, _bare_term):
                _coeff = _lower_statefield_to_trial(_apply_sign(domain, _sign, _sub), {})
                _tfi = _test_field_index(_coeff, field_index)
                if _tfi is None:
                    raise ValueError("jno.fem (non-nodal): each weak term must contain exactly one test field.")
                _typed.append((_coeff, _tfi, tuple(sorted(_collect_region_mask_names(_coeff)))))

        _cell_all_dofs = jnp.concatenate([cdofs[i] for i in range(len(fields))], axis=1)  # (n_cells, n_local_all)
        _field_splits = list(np.cumsum([0] + [int(cdofs[i].shape[1]) for i in range(len(fields))]))
        _asm_zero_field = jnp.zeros((n_verts,), dtype=zeros.dtype)  # placeholder for an absent field parameter
        _asm_local_zero = jnp.zeros((n_cells, _cell_all_dofs.shape[1]), zeros.dtype)  # u=0: the LINEAR default
        # TERMS SHARING A TEST FIELD are one GROUP: their element blocks have identical (row, col)
        # layout, so they are summed at the element level and emit ONE pattern block -- the old
        # per-term emission stored an identical full triplet set for every additive term, which is
        # a factor of ~n_terms in peak memory for nothing.
        _groups: list = []  # (tfi, [(coeff, rnames), ...]) in first-occurrence order
        _gpos: dict = {}
        for _cf_p, _tfi_p, _rn_p in _typed:
            if _tfi_p not in _gpos:
                _gpos[_tfi_p] = len(_groups)
                _groups.append((_tfi_p, []))
            _groups[_gpos[_tfi_p]][1].append((_cf_p, _rn_p))

        # THE SYMBOLIC PASS, once, on the host, in numpy end to end: the unique sparsity pattern and
        # the inverse map from every emitted entry to its final slot. Element values then SCATTER-ADD
        # straight into a buffer of exactly the final size -- no stored duplicates, no in-trace sort
        # (the old `sum_duplicate_triplets` collapse, whose argsort workspace over the unreduced
        # triplet array was the dominant assembly temporary). The full jnp triplet pattern is built
        # ONLY on the unplanned fallback: with a plan, nothing downstream needs it, and at ~20M
        # triplets it alone is hundreds of MB. Intermediates are freed as soon as consumed -- the
        # audit that motivated this found the peak made of stacked full-length copies, not one villain.
        _uidx = _vinv = _vol_idx = None
        _vol_nse = None
        _grp_sizes = []
        # General boundary-integrand groups: rows = the owner cell's test DOFs, columns = the owner
        # cell's all-field DOFs followed by every across key's main-cell DOFs (per quadrature point).
        _sgs = list(_GS["groups"]) if with_gsurf else []
        _s_cols = []
        for _g in _sgs:
            _cc = [np.asarray(_cell_all_dofs)[_g["cell_np"]]]
            for _key, (_dofs, _phi) in _g["across"].items():
                _cc.append(np.asarray(_dofs).reshape(_g["n_facets"], -1))
            _s_cols.append(np.concatenate(_cc, axis=1))
        if _groups or _sgs:
            try:
                _enc_l = []
                for _tfi_p, _terms_p in _groups:
                    _nt = int(cdofs[_tfi_p].shape[1])
                    _nl = int(_cell_all_dofs.shape[1])
                    _r = np.broadcast_to(np.asarray(cdofs[_tfi_p])[:, :, None], (n_cells, _nt, _nl))
                    _c = np.broadcast_to(np.asarray(_cell_all_dofs)[:, None, :], (n_cells, _nt, _nl))
                    _enc_l.append((_r.astype(np.int64) * np.int64(total) + _c.astype(np.int64)).reshape(-1))
                    _grp_sizes.append(n_cells * _nt * _nl)
                    del _r, _c
                for _g, _cols in zip(_sgs, _s_cols):
                    _nt = int(cdofs[_g["tfi"]].shape[1])
                    _r = np.broadcast_to(
                        np.asarray(cdofs[_g["tfi"]])[_g["cell_np"]][:, :, None], (_g["n_facets"], _nt, _cols.shape[1])
                    )
                    _c = np.broadcast_to(_cols[:, None, :], _r.shape)
                    _enc_l.append((_r.astype(np.int64) * np.int64(total) + _c.astype(np.int64)).reshape(-1))
                    _grp_sizes.append(int(_r.size))
                    del _r, _c
                _enc = np.concatenate(_enc_l) if len(_enc_l) > 1 else _enc_l[0]
                del _enc_l
                _unq, _inv = np.unique(_enc, return_inverse=True)
                del _enc
                _uidx = jnp.stack(
                    [
                        jnp.asarray((_unq // np.int64(total)).astype(np.int32)),
                        jnp.asarray((_unq % np.int64(total)).astype(np.int32)),
                    ],
                    axis=1,
                )  # sorted row-major by construction of np.unique
                _vol_nse = int(_unq.size)
                del _unq
                _inv = _inv.astype(np.int32)
                # per-group slices of the inverse map, so values scatter INSIDE the group loop and the
                # full-length data vector never exists
                _off = np.cumsum([0] + _grp_sizes)
                _vinv = [jnp.asarray(_inv[_off[i] : _off[i + 1]]) for i in range(len(_grp_sizes))]
                del _inv
            except Exception:  # noqa: BLE001 -- a traced pattern cannot be planned on the host
                _uidx = _vinv = None
                _vol_nse = None  # fall back to the uncompressed (still correct) operator
        if _vinv is None and (_groups or _sgs):
            _vol_rows_l, _vol_cols_l = [], []
            for _tfi_p, _terms_p in _groups:
                _kshape = (n_cells, int(cdofs[_tfi_p].shape[1]), _cell_all_dofs.shape[1])
                _vol_rows_l.append(jnp.broadcast_to(cdofs[_tfi_p][:, :, None], _kshape).reshape(-1))
                _vol_cols_l.append(jnp.broadcast_to(_cell_all_dofs[:, None, :], _kshape).reshape(-1))
            for _g, _cols in zip(_sgs, _s_cols):
                _rr = jnp.asarray(cdofs[_g["tfi"]])[_g["cell"]]
                _kshape = (_g["n_facets"], int(_rr.shape[1]), _cols.shape[1])
                _vol_rows_l.append(jnp.broadcast_to(_rr[:, :, None], _kshape).reshape(-1))
                _vol_cols_l.append(jnp.broadcast_to(jnp.asarray(_cols)[:, None, :], _kshape).reshape(-1))
            _vol_idx = jnp.stack(
                [jnp.concatenate(_vol_rows_l).astype(jnp.int32), jnp.concatenate(_vol_cols_l).astype(jnp.int32)],
                axis=1,
            )
        elif not (_groups or _sgs):
            _vol_idx = jnp.zeros((0, 2), jnp.int32)

        def assemble(args=None, *, surface=True, u_flat=None):
            # ``u_flat=None`` linearises at u=0, which is exact for a LINEAR form (the tangent does not
            # depend on the state) and is the historic behaviour. Passing the current iterate gives the
            # NONLINEAR tangent J(u_k) instead -- per-element `jacfwd` at that state, exactly what
            # `fem_native._make_jacobian` does. The sparsity PATTERN is unchanged either way: it comes
            # from mesh connectivity, so the hoisted `_vol_idx` and its compression plan carry over
            # untouched and only the element DATA becomes state-dependent.
            _local_u = _asm_local_zero if u_flat is None else jnp.asarray(u_flat)[_cell_all_dofs]
            _a = args or {}
            rt_scalar = {
                name: jnp.reshape(jnp.asarray(_a.get(name, 0.0), dtype=zeros.dtype), (-1,))[:1]
                for name in runtime_parameter_tags
                if name not in _field_param_names
            }
            field_vals = {
                name: jnp.asarray(_a.get(name, _asm_zero_field), dtype=zeros.dtype) for name in _field_param_names
            }
            _nt = neural_local_table(_neural, args)
            _pts_dyn = _apply_coord_params(pts, _a)  # traced iff coordinates are trainable

            def _elem_res(c, la, coeff, tfi, rnames):
                """This cell's element residual (n_test of field ``tfi``,) from its local dof vector ``la``."""
                cell_sols = [la[_field_splits[i] : _field_splits[i + 1]] for i in range(len(fields))]
                per, xq, meas = _cell_fields(c, cell_sols, _pts_dyn)
                _ctx_c = _with_cell_size(ctx, meas, xq, (pts if _pts_dyn is None else _pts_dyn)[cells_j[c]])
                vol_vars = tuple(
                    (field_vals[name][cells_j[c]] if name in _field_param_names else rt_scalar[name])
                    for name in runtime_parameter_tags
                )
                local = {
                    "physical_quad_points": xq,
                    "fields": per,
                    "field_index": field_index,
                    "tag": "fem_gauss",
                    "surface": False,
                    "domain_context": _ctx_c,
                    "temporal_tags": (),
                    "runtime_parameter_tags": runtime_parameter_tags,
                    "region_mask_names": rnames,
                    "volume_vars": vol_vars + _cell_masks(c, rnames),
                }
                if (
                    _field_param_names or _frozen_gathered
                ):  # P1 basis to interpolate a field parameter (top-level, param-only key)
                    local["shape_vals"] = p1_shape_vals
                if _frozen_gathered:  # known-field (ui.freeze) per-cell nodal slices for this cell
                    local["frozen_fields"] = {fid: g[c] for fid, g in _frozen_gathered.items()}
                if _nt is not None:  # trainable nets ride args (crux weights); frozen/placeholder -> stored
                    local["neural_coefficients"] = _nt
                return _integrate_term(domain, coeff, local, qw * meas)

            # values scatter into their final slots PER GROUP: the full-length data vector exists
            # only on the unplanned fallback, where the uncompressed triplet path still needs it
            _acc_dtype = jnp.result_type(zeros.dtype, np.complex64) if _cx_form else zeros.dtype
            _acc = jnp.zeros((_vol_nse,), _acc_dtype) if _vinv is not None else None
            data_l = []  # fallback only; the (row, col) pattern is `_vol_idx`
            for gi, (tfi, terms_g) in enumerate(_groups):

                def _ke(c, la, _g=tuple(terms_g), _t=tfi):
                    # ONE jacfwd of the SUMMED residual: linearity makes it equal to the sum of the
                    # per-term jacobians, at one element-block buffer per GROUP instead of per term
                    # (and `_cell_fields` is shared across the group's terms instead of recomputed).
                    def _res_sum(v):
                        r = None
                        for _e, _rn in _g:
                            t = _elem_res(c, v, _e, _t, _rn)
                            r = t if r is None else r + t
                        return r

                    return jax.jacfwd(_res_sum)(la)

                Ke = _elem_map(  # (n_cells, n_test_tfi, n_local_all)
                    _ke,
                    (jnp.arange(n_cells), _local_u),
                    _cell_chunk_of(n_cells, int(cdofs[tfi].shape[1]), int(_cell_all_dofs.shape[1]), _chunk_setting),
                )
                if _acc is not None:
                    _acc = _acc.at[_vinv[gi]].add(Ke.reshape(-1).astype(_acc.dtype))
                else:
                    data_l.append(Ke.reshape(-1))

            for sgi, _g in enumerate(_sgs):  # general boundary integrands: one jacfwd per facet
                _la, _lbs = _facet_inputs(_g, jnp.zeros((total,), zeros.dtype) if u_flat is None else u_flat)
                _keys = list(_g["across"])
                _nl = int(_la.shape[1])

                _ss = tuple((k_, tuple(_lbs[k_].shape[1:])) for k_ in _keys)
                _lbf = (
                    jnp.concatenate([_lbs[k_].reshape(_g["n_facets"], -1) for k_ in _keys], axis=1)
                    if _keys
                    else jnp.zeros((_g["n_facets"], 0), _la.dtype)
                )

                def _kf(f, la_f, lb_f, _g=_g, _nl=_nl, _ss=_ss):
                    def _r(vv):
                        return _facet_res(
                            _g,
                            f,
                            vv[:_nl],
                            _unflatten_lbs(vv[_nl:], _ss),
                            _g["coeffs"],
                            rt_scalar,
                            field_vals,
                            _nt,
                            _pts_dyn,
                        )

                    return jax.jacfwd(_r)(jnp.concatenate([la_f, lb_f]))

                Kf = _elem_map(  # (n_f, n_test, n_cols); chunked: the AD tangent is n_cols wide per facet
                    _kf,
                    (jnp.arange(_g["n_facets"]), _la, _lbf),
                    _cell_chunk_of(
                        _g["n_facets"], int(cdofs[_g["tfi"]].shape[1]), _nl + int(_lbf.shape[1]), _chunk_setting
                    ),
                )
                if _acc is not None:
                    _acc = _acc.at[_vinv[len(_groups) + sgi]].add(Kf.reshape(-1).astype(_acc.dtype))
                else:
                    data_l.append(Kf.reshape(-1))
            data = _acc if _acc is not None else (jnp.concatenate(data_l) if data_l else jnp.zeros((0,), zeros.dtype))
            # Tangential-trace surface mass (impedance BC), re-evaluated per args. Taken SPARSE: densifying it
            # would cost O(n_dof²) just to re-sparsify, the memory wall for a 3-D vector run.
            sm = _surf_mass_of(args, sparse=True) if surface else None
            if _vinv is not None:
                # values already landed in their FINAL slots inside the group loop; nothing to sort,
                # nothing to collapse, and the flags let every matvec skip the lazy re-summation.
                _emit_idx = _uidx
                _emit_flags = dict(indices_sorted=True, unique_indices=True)
            else:
                _emit_idx = _vol_idx  # unplanned fallback: uncompressed, still correct
                _emit_flags = {}
            if sm is None:
                return _jsparse.BCOO((data, _emit_idx), shape=(total, total), **_emit_flags)
            idx = jnp.concatenate([_emit_idx, sm.indices], axis=0)  # BCOO sums duplicate (i, j) on materialisation
            data = jnp.concatenate([data, sm.data])
            # Deliberately NOT compressed. The static count would have to cover `sm.indices`, which is
            # produced inside the trace by `_surf_mass_of(args)`; assuming it is args-independent and
            # caching a count on that assumption is exactly the failure mode a static `nse` has -- too
            # small drops entries, silently, and returns a wrong operator rather than a slow one.
            # Left uncompressed (correct, just redundant) until that pattern is hoisted as well.
            return _jsparse.BCOO((data, idx), shape=(total, total))

        return assemble

    # --- the NONLINEAR tangent, assembled per element ------------------------------------------
    # `_apply_dirichlet_projected` replaces the pinned rows of the residual with `u[d] - g`, so the tangent
    # of that is the free tangent with those rows set to the identity -- `bcoo_eliminate_dirichlet`,
    # the same row-replacement `fem_native._dirichlet_jac_rows` performs.
    #
    # `surface=False` matches the dense `jacfwd(res_bc)` it replaces EXACTLY: `_make_residual` is
    # volume-only and the nonlinear path never added `surf_mass` to its tangent. Whether it should is
    # a separate question about the residual too, not something a storage change may decide quietly.
    _pin_dofs_j, _, _pin_project = dirichlet_projection(pins) if pins else (None, None, None)

    def _sparse_tangent(asm, u, args=None):
        # At the PROJECTED state and eliminated on both sides, matching `_apply_dirichlet_projected`.
        _u = jnp.asarray(u).reshape(-1)
        _u = _u if _pin_project is None else _pin_project(_u)
        J = asm(args, surface=False, u_flat=_u)
        return J if _pin_dofs_j is None else bcoo_eliminate_dirichlet(J, _pin_dofs_j)

    #
    # Defined BEFORE the transient branch below: that branch returns early, and its Jacobian closures call
    # `_sparse_tangent` when the march runs -- defined after it, the name was never bound (NameError).

    # === transient: M u̇ + A u = c (mirrors fem_1d._assemble_1d_transient) -- split the temporal term
    #     (∫ ∂ₜu·v -> mass M) from the spatial operator, project the IC onto the edge DOFs, time-block it. ===
    if ic_residuals or any(_contains_temporal_derivative(t) for t in volume_terms):
        from ..._fem import _bare, _essential_spec, _eval_value_node_at, _field_key_of
        from .backend_blocks import SemidiscreteTimeBlock
        from .fem_1d import _apply_dirichlet_transient
        from .solver_helper import max_temporal_derivative_order as _mto
        from .time_route import _infer_time_window, _strip_temporal_trial_derivative

        sub = [_apply_sign(domain, s, t) for bare in volume_terms for s, t in _split_additive_terms(domain, bare)]
        from .weak_form_helpers import refuse_mixed_temporal_group

        for _t in sub:
            refuse_mixed_temporal_group(_t, where="jno.fem (non-nodal)")
        temporal = [t for t in sub if _contains_temporal_derivative(t)]
        spatial = [t for t in sub if not _contains_temporal_derivative(t)]
        if not temporal:
            raise ValueError("jno.fem (non-nodal): a transient weak form must contain a temporal term, e.g. inner(u.t, v).")
        if collect_neural_slots(temporal).any_trainable:
            # The mass block is assembled once (args=None) -- a trainable net on the u̇ term would silently
            # freeze at its initial weights. Fail loud (a frozen net evaluates from its stored module by design).
            raise NotImplementedError(
                "jno.fem (non-nodal): a trainable neural coefficient on the mass (u_t) term is not supported -- "
                "the mass block is assembled once. Use it on spatial terms, or .freeze() the network."
            )

        # === second-order-in-time (u_tt): the augmented first-order block y = [u; v], v = u̇, integrated by
        #     the trapezoidal (θ=½, energy-conserving) rule — the non-nodal analogue of the native
        #     _assemble_second_order_time, reusing this path's push-forward mass / pins / IC-projection. ===
        if max((_mto(t) for t in temporal), default=1) >= 2:
            if len(fields) > 1:
                raise NotImplementedError("jno.fem (non-nodal): second-order-in-time (u_tt) is single-field only.")
            # Runtime/trainable parameters are supported on the SPATIAL operator (a bending stiffness, a wave
            # speed, a k(x) field) — the differentiable inverse for C¹ plates/beams (mirrors the native path).
            # A parameter on the mass/damping (u_tt / u_t) term is rejected: the mass block, and the IC
            # L²-projection through it, are assembled once, so a runtime density would silently freeze.
            _temporal_param: dict = {}
            for _t in temporal:
                _collect_runtime_parameter_exprs(_t, _temporal_param)
            if _temporal_param or collect_neural_slots(temporal).any_trainable:
                raise NotImplementedError(
                    "jno.fem (non-nodal): a runtime/trainable parameter on the mass or damping (u_tt / u_t) "
                    "term of a second-order-in-time form is not supported; put it on the spatial operator."
                )
            has_param = bool(runtime_parameter_tags or neural_param_names)  # all remaining params are spatial

            def _strip_n(t, k):
                for _ in range(k):
                    t = _strip_temporal_trial_derivative(t)
                return t

            M2 = jax.jacfwd(_make_residual([_strip_n(t, 2) for t in temporal if _mto(t) >= 2]))(zeros)  # ∫ü·v ⇒ mass
            damp = [_strip_n(t, 1) for t in temporal if _mto(t) == 1]  # ∫u̇·v ⇒ damping (optional)
            Cmat = jax.jacfwd(_make_residual(damp))(zeros) if damp else jnp.zeros((total, total), zeros.dtype)
            spatial_res2 = _make_residual(
                spatial, with_gsurf=True
            )  # residual(u, args) -> threads a spatial runtime parameter
            n = total
            Z = jnp.zeros((n, n), zeros.dtype)
            dd = jnp.asarray([p[0] for p in pins], dtype=jnp.int32) if pins else None
            dg = jnp.asarray([p[1] for p in pins], dtype=zeros.dtype) if pins else None

            def _dir_A(A):  # essential rows of the augmented system: u[d]=g (constant) and v[d]=0 (identity rows)
                if dd is None:
                    return A
                return A.at[dd, :].set(0.0).at[dd, dd].set(1.0).at[dd + n, :].set(0.0).at[dd + n, dd + n].set(1.0)

            def _A_of(args):  # A_aug(args) = [[0, -M2], [K(args), C]]: M2 u̇ = M2 v ; M2 v̇ + C v + K u = F
                K = jax.jacfwd(lambda u: spatial_res2(u, args))(zeros)
                if surf_mass is not None:  # N1E tangential-trace surface mass (impedance) → the stiffness K
                    K = K + _surf_mass_dense()
                return _dir_A(jnp.block([[Z, -M2], [K, Cmat]]))

            def _f_of(args):  # load F(args) on the v-block rows (Dirichlet g rides affine_bias, not the forcing)
                f = jnp.concatenate([jnp.zeros((n,), zeros.dtype), -spatial_res2(zeros, args) + nat_load])
                return f if dd is None else f.at[dd].set(0.0).at[dd + n].set(0.0)

            M_aug = jnp.block([[M2, Z], [Z, M2]])
            if dd is not None:
                M_aug = M_aug.at[dd, :].set(0.0).at[dd + n, :].set(0.0)

            def _project_onto(u0_node):  # L²-project a value node onto the field DOFs via the mass block M2
                u0_blocks = [jnp.zeros(offs[i + 1] - offs[i]) for i in range(len(fields))]

                def _ic_cell(cidx):
                    per, xq, meas = _cell_fields(cidx, _cell_local_sols(cidx, u0_blocks))
                    phi = per[0]["shape_vals"]
                    u0 = jnp.asarray(_eval_value_node_at(u0_node, xq, t=_infer_time_window(domain)[0]))
                    if phi.ndim == 3:  # H(curl)/H(div) vector basis (n_quad, n_dof, d): ∫ u0·Φ (a wave equation)
                        u0 = jnp.broadcast_to(
                            u0.reshape(-1) if u0.size == phi.shape[-1] else u0.reshape(n_quad, -1), (n_quad, phi.shape[-1])
                        )
                        return jnp.einsum("q,qnc,qc->n", qw * meas, phi, u0)
                    return jnp.einsum("q,qn,q->n", qw * meas, phi, jnp.broadcast_to(u0.reshape(-1), (n_quad,)))

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
            common2 = dict(
                backend="transient",
                mode="implicit",
                time_order=2,
                spatial_kind="weak_form",
                M=M_aug,
                state0=jnp.concatenate([u0_dofs, v0_dofs]),
                t0=t0,
                t1=t1,
                dt=dt,
                eval_context=getattr(domain, "_fem_eval_context", {}) or {},
            )
            if not has_param:
                c_aug = jnp.concatenate([jnp.zeros((n,), zeros.dtype), -spatial_res2(zeros) + nat_load])
                if dd is not None:
                    c_aug = c_aug.at[dd].set(dg).at[dd + n].set(0.0)
                block = SemidiscreteTimeBlock(
                    A=_A_of(None), affine_bias=c_aug, metadata={"theta": 0.5, "second_order": True}, **common2
                )
            else:
                # re-form A_aug/forcing from the runtime args each step; the θ=½ stepper differentiates through
                # its own scan (no stepper change), exactly like the native second-order parametric path.
                _ph = {nm: (jnp.zeros((n_verts,)) if nm in _field_param_names else 0.0) for nm in runtime_parameter_tags}
                _ph.update({nm: _neural_models[nm].module for nm in neural_param_names})
                c_dir = jnp.zeros((2 * n,), zeros.dtype) if dd is None else jnp.zeros((2 * n,), zeros.dtype).at[dd].set(dg)
                block = SemidiscreteTimeBlock(
                    A=_A_of(_ph),
                    affine_bias=c_dir,
                    operator_fn=lambda t, args=None: _A_of(args),
                    forcing_vector_fn=lambda t, args=None: _f_of(args),
                    runtime_parameter_exprs=dict(_param_and_neural_exprs),
                    metadata={
                        "theta": 0.5,
                        "second_order": True,
                        "runtime_parameter_names": list(_param_and_neural_exprs),
                        "nonaffine_operator": True,
                    },
                    **common2,
                )
            return block, "transient", [0, n, 2 * n]

        # Block mass M and spatial operator A: assemble the edge/cell DOF families (N1E/RT/P0) SPARSELY (per
        # element, BCOO) so a 3-D vector transient never forms the dense global jacfwd — the eddy-current /
        # time-domain-Maxwell OOM (the ``O(n_dof × n_cells)`` tangent overflows the 2³¹ XLA limit past ~10⁴
        # edges). ``spatial_res`` (a cheap O(n) residual eval) still supplies the RHS load and the nonlinear
        # residual; only the operator MATRICES go sparse. Vertex C0/C1 families keep the dense path (small 2-D).
        _strip_mass = [_strip_temporal_trial_derivative(t) for t in temporal]
        _assemble_spatial = _make_sparse_assembler(spatial, with_gsurf=True)
        M = _make_sparse_assembler(_strip_mass)(None, surface=False)  # BCOO mass — no impedance on the u̇ term
        spatial_res = _make_residual(spatial, with_gsurf=True)
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
                u0 = jnp.asarray(_eval_value_node_at(u0_node, xq, t=_infer_time_window(domain)[0]))
                if phi.ndim == 3:  # RT/N1E vector basis (n_quad, n_dof, vsize): ∫ u0·Φ
                    u0 = jnp.broadcast_to(
                        u0.reshape(-1) if u0.size == 1 else u0.reshape(n_quad, -1), (n_quad, phi.shape[-1])
                    )
                    return jnp.einsum("q,qnc,qc->n", qw * meas, phi, u0)
                # scalar basis ∫ u0 q — broadcast so a *constant* IC (shape (1,)) fills all n_quad points
                return jnp.einsum("q,qn,q->n", qw * meas, phi, jnp.broadcast_to(u0.reshape(-1), (n_quad,)))

            local = (cdofs[fidx] - offs[fidx]).reshape(-1)  # field-local DOFs (ce for RT/N1E, cell index for P0)
            load = jnp.zeros(offs[fidx + 1] - offs[fidx]).at[local].add(jax.vmap(_ic_cell)(jnp.arange(n_cells)).reshape(-1))
            sl = slice(offs[fidx], offs[fidx + 1])
            if hasattr(M, "indices"):  # BCOO mass: solve the (SPD) field mass block matrix-free — never slice/
                #   densify the block (that is the very O(n²) wall the sparse assembly avoids). The mass block is
                #   block-diagonal in the field, so ``(M @ embed(x))[sl] == M_block @ x``; CG is exact-to-tol.
                nblk = offs[fidx + 1] - offs[fidx]

                def _mv(x, _sl=sl):
                    return (M @ jnp.zeros((total,), zeros.dtype).at[_sl].set(x))[_sl]

                block_sol, _ = jax.scipy.sparse.linalg.cg(_mv, load, tol=1e-12, atol=0.0, maxiter=nblk + 50)
                return fidx, block_sol
            return fidx, jnp.linalg.solve(M[sl, sl], load)  # the differential field's mass block is non-singular

        state0 = zeros
        for ic in ic_residuals:
            fidx, block_sol = _project_ic(ic)
            state0 = state0.at[offs[fidx] : offs[fidx + 1]].set(block_sol)
        common["state0"] = state0

        pin_dofs = jnp.asarray([p[0] for p in pins], dtype=jnp.int32) if pins else None
        if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial):
            # nonlinear transient: M(t) u̇ + R(u) = 0 (matrix-free Newton-Krylov per step). Natural load
            # folds into R; essential pins are residual rows + zeroed M rows/cols (the 1D pattern). The mass is
            # BCOO for edge/cell families (kept sparse through the step matvec); dense for C¹ vertex families.
            M_nl = (
                M
                if pin_dofs is None
                else bcoo_zero_rows_cols(M, pin_dofs)
                if hasattr(M, "indices")
                else M.at[pin_dofs, :].set(0.0).at[:, pin_dofs].set(0.0)
            )
            if runtime_parameter_tags or neural_param_names:  # transient inverse: thread args through res + jac

                def res_pt(u, t, args=None):
                    return _apply_dirichlet_projected(lambda uu: spatial_res(uu, args) - nat_load, pins)(jnp.asarray(u))

                def jac_pt(u, t, args=None):
                    return _sparse_tangent(_assemble_spatial, u, args)

                block = SemidiscreteTimeBlock(
                    mass=lambda t, args=None, _M=M_nl: _M,
                    residual=res_pt,
                    jacobian=jac_pt,
                    runtime_parameter_exprs=dict(_param_and_neural_exprs),
                    **common,
                )
                return block, "transient", offs
            res_bc = _apply_dirichlet_projected(lambda u: spatial_res(u) - nat_load, pins)

            def jac(u, _asm=_assemble_spatial):
                return _sparse_tangent(_asm, u)

            block = SemidiscreteTimeBlock(
                mass=lambda t, args=None, _M=M_nl: _M,
                residual=lambda u, t, args=None: res_bc(u),
                jacobian=lambda u, t, args=None: jac(u),
                **common,
            )
            return block, "transient", offs

        if runtime_parameter_tags or neural_param_names:  # linear transient inverse: A(args) re-assembled each
            #   step (SPARSELY for edge/cell families), Dirichlet rows -> identity via `bcoo_eliminate_dirichlet`.
            M_bc = (
                M
                if pin_dofs is None
                else bcoo_zero_rows_cols(M, pin_dofs)
                if hasattr(M, "indices")
                else M.at[pin_dofs, :].set(0.0).at[:, pin_dofs].set(0.0)
            )
            c_bias = zeros if pin_dofs is None else zeros.at[pin_dofs].set(jnp.asarray([p[1] for p in pins]))
            free_mask = (
                jnp.ones((total,), dtype=zeros.dtype)
                if pin_dofs is None
                else jnp.ones((total,), dtype=zeros.dtype).at[pin_dofs].set(0.0)
            )

            def operator_fn(t, args=None, _d=pin_dofs, _sm=surf_mass, _asm=_assemble_spatial):
                if _asm is not None:  # edge/cell families: per-element sparse A(args), folds in the surface mass
                    A = _asm(args)
                    return A if _d is None else bcoo_eliminate_dirichlet(A, _d)  # Dirichlet rows -> identity
                A = jax.jacfwd(lambda u: spatial_res(u, args))(zeros)  # C¹ vertex families: dense (small 2-D)
                if _sm is not None:  # N1E tangential-trace surface mass (impedance) → the spatial operator A
                    A = A + _sm.todense()
                return A if _d is None else A.at[_d, :].set(0.0).at[_d, _d].set(1.0)  # Dirichlet rows -> identity

            def forcing_vector_fn(t, args=None, _mask=free_mask):
                return _mask * (-spatial_res(zeros, args) + nat_load)  # source on free rows; Dirichlet via the bias

            return (
                SemidiscreteTimeBlock(
                    M=M_bc,
                    operator_fn=operator_fn,
                    affine_bias=c_bias,
                    forcing_vector_fn=forcing_vector_fn,
                    runtime_parameter_exprs=dict(_param_and_neural_exprs),
                    metadata={"nonaffine_operator": True},
                    **common,
                ),
                "transient",
                offs,
            )

        # linear transient: M u̇ + A u = c. Edge/cell families assemble A SPARSELY (BCOO, folds in the surface
        # mass); vertex families keep the dense jacfwd. `_apply_dirichlet_transient` is BCOO-aware (a sparse M/A
        # stays sparse) and the SemidiscreteTimeBlock applies M/A only as matvecs — so a large 3-D vector
        # transient marches without ever densifying the operator (the eddy-current / time-domain-Maxwell fix).
        if _assemble_spatial is not None:
            A = _assemble_spatial(None)  # BCOO; the tangential-trace surface mass is folded in by the assembler
        else:
            A = jax.jacfwd(spatial_res)(zeros)  # C¹ vertex families: dense (small 2-D)
            if surf_mass is not None:  # N1E tangential-trace surface mass (impedance) → the spatial operator A
                A = A + _surf_mass_dense()
        c = -spatial_res(zeros) + nat_load  # spatial load + natural-BC constant load
        M, A, c = _apply_dirichlet_transient(M, A, c, pins)  # essential edge-trace pins -> M/A/c rows
        # Applied on every timestep, so the per-term/per-element triplet redundancy is paid the whole
        # march. A no-op on the dense vertex-family branch (no `.indices`), which is what we want.
        M, A = compress_eager(M), compress_eager(A)
        return SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, **common), "transient", offs

    residual = _make_residual(volume_terms, with_gsurf=True)

    def full_residual(u_flat, args=None):  # the natural BC is a constant load on the RHS: R(u) = assembled(u) - load
        return residual(u_flat, args) - nat_load

    # Steady operator A(args): assemble the RT/N1E/P0 edge & cell DOF families SPARSELY, one element at a time
    # (never a dense global ``jacfwd``) — the SAME ``_make_sparse_assembler`` that backs the transient mass and
    # spatial split above. ``args`` threads the runtime parameters so the PARAMETRIC (inverse) path assembles
    # sparsely too (it used to take the dense global jacfwd — a ~10⁴-edge ceiling on 3-D vector inverse design,
    # re-run every optimizer step); ``args=None`` reduces to the non-parametric assembly exactly. Vertex C0/C1
    # families keep the dense path below (their Hessian-shape element assembly is not ported; small 2-D problems).

    _assemble_sparse_A = _make_sparse_assembler(volume_terms, with_gsurf=True)

    # --- steady nonlinear: a genuinely nonlinear weak term -> a Newton residual operator ---
    if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in volume_terms):
        if runtime_parameter_tags or neural_param_names:  # parametric (inverse): thread args through res + jac

            def res_p(u, args=None):
                return _apply_dirichlet_projected(lambda uu: full_residual(uu, args), pins)(jnp.asarray(u))

            def jac_p(u, args=None):
                return _sparse_tangent(_assemble_sparse_A, u, args)

            return (
                FemResidualOperator(res_p, jac_p, total, runtime_parameter_exprs=dict(_param_and_neural_exprs)),
                "nonlinear",
                offs,
            )
        res_bc = _apply_dirichlet_projected(full_residual, pins)  # essential pins as residual rows R[d]=u[d]-g

        def jac(u):
            return _sparse_tangent(_assemble_sparse_A, u)

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
    # `_coord_specs` belongs in this gate for the same reason it does in the native assembler: a
    # trainable mesh coordinate makes the operator parameter-dependent even though it is not a term
    # COEFFICIENT, so it never appears in `runtime_parameter_tags`. Without it here the assembler
    # returns a static (A, b) and the shape derivative is silently zero -- which reads as a
    # converged design rather than a missing feature.
    if runtime_parameter_tags or neural_param_names or _coord_specs:
        from ...trace import FemLinearSystem

        def _assemble_at(args):
            # per-element sparse assembly for every family; folds in the tangential-trace surface mass
            A = _assemble_sparse_A(args)
            # ``full_residual`` folds in the CONSTANT nat_load; add the parametric change of the N1E incident
            # load so a runtime parameter in the incident source is differentiable in b too.
            b = -full_residual(zeros, args) + (_incident_of(args) - _incident_const)
            if pins:  # `_apply_dirichlet_symmetric` keeps a BCOO sparse; only the dense path needs the cast
                A, b = _apply_dirichlet_symmetric(A, jnp.asarray(b), pins)
            return A, b

        # static placeholder for `.A` / `.b` (right width per parameter kind: a field param is (n_verts,),
        # a neural coefficient rides its stored module)
        _ph = {n: (jnp.zeros((n_verts,)) if n in _field_param_names else 0.0) for n in runtime_parameter_tags}
        _ph.update({n: _neural_models[n].module for n in neural_param_names})
        # a coordinate parameter's placeholder is the CURRENT node positions, not zeros: `.A` / `.b`
        # must describe the mesh as it stands, and zeros would collapse those vertices onto the origin.
        _ph.update({_nm: pts[_ids, _ax] for _ids, _ax, _nm in _coord_specs})
        a0, b0 = _assemble_at(_ph)
        return (
            FemLinearSystem(
                a0,
                b0,
                operator_fn=lambda args=None: _assemble_at(args)[0],
                rhs_fn=lambda args=None: _assemble_at(args)[1],
                runtime_parameter_exprs=dict(_param_and_neural_exprs),
                metadata={"nonaffine_operator": True},  # re-assembles at each args; no affine parameter basis
            ),
            "linear",
            offs,
        )

    # --- steady linear (non-parametric): A u = b ---
    b = -full_residual(zeros)  # spatial + natural-BC load (constant part of the residual)

    # EVERY family assembles SPARSELY, one element at a time -- see `_assemble_sparse_A`. The vertex
    # C0/C1 families (Hermite/Argyris/Morley) used a global dense `jacfwd` here, which was never a
    # property of C0/C1 elements: `_elem_res` calls the same `_cell_fields` that carries their
    # per-cell M(cell) DOF-transform and shape_hess, and the RESIDUAL already assembled them per
    # element through it. The dense form's cost is the `O(n_dofs x n_cells)` tangent, not the matrix --
    # measured on Argyris at 635 dofs: a 3.1 MiB operator with a 2279 MiB peak, 741x the stored size.
    A = _assemble_sparse_A(None)  # no runtime parameters on this branch -> identical to the args-threaded form
    if pins:  # symmetric elimination; `_apply_dirichlet_symmetric` keeps a BCOO sparse
        A, b = _apply_dirichlet_symmetric(A, b, pins)
    return (compress_eager(A), b), "linear", offs


def _reads_across(node) -> bool:
    """Does a boundary term read ``u.across(...)`` (a ``across_*`` Variable)?"""
    from ..._fem import _walk
    from ...trace import Variable

    return any(isinstance(n, Variable) and str(getattr(n, "tag", "")).startswith("across_") for n in _walk(node))


def _legacy_natural(bare, field_index, spaces) -> bool:
    """The two host-assembled natural terms the old path owns: a plate edge moment ``M_n * v.dn`` on an
    Argyris/Morley field, and the RT pressure load ``p_D * (v·n)``. Everything else is general."""
    from ..._fem import _contains, _walk
    from ...trace import BinaryOp, TestFunction, TrialFunction, Variable

    if _match_plate_moment(bare, field_index, spaces) is not None:
        return True
    if not (isinstance(bare, BinaryOp) and bare.op == "*") or _contains(bare, TrialFunction):
        return False
    for side, other in ((bare.left, bare.right), (bare.right, bare.left)):
        if _contains(side, TestFunction) and not _contains(other, TestFunction):
            walked = list(_walk(side))
            if not any(isinstance(n, Variable) and str(getattr(n, "tag", "")).startswith("n_") for n in walked):
                return False
            fk = {n.field_key for n in walked if isinstance(n, TestFunction)}
            fi = field_index.get(next(iter(fk))) if fk else None
            return fi is not None and spaces[fi] == "RT"
    return False


def _with_cell_size(ctx, meas, xq, verts=None):
    """``ctx`` with the cell-geometry symbols resolved for one cell, as the nodal assembler packs them:
    ``dom.cell_size`` = ``|det J|^(1/dim)`` and ``dom.cell_metric`` = ``G = J^-T J^-1`` at every quadrature
    point. Only those the form reads -- otherwise their placeholders would stand in for h / G without a
    word. ``verts`` are the cell's vertex coordinates (needed for ``G``; affine simplices)."""
    want_h, want_g = "cell_size" in ctx, "cell_metric" in ctx
    if not (want_h or want_g):
        return ctx
    nq, dim = xq.shape[0], xq.shape[-1]
    out = dict(ctx)
    if want_h:
        out["cell_size"] = jnp.broadcast_to(jnp.reshape(meas ** (1.0 / dim), (-1, 1)), (nq, 1))
    if want_g:
        if verts is None:
            raise NotImplementedError("dom.cell_metric on this non-nodal path needs the cell's vertices.")
        J = jnp.stack([verts[k] - verts[0] for k in range(1, dim + 1)], axis=1)
        K = jnp.linalg.inv(J)
        out["cell_metric"] = jnp.broadcast_to(K.T @ K, (nq, dim, dim))
    return out


def _build_general_surface(
    gsurf_terms, domain, dim, cell_name, cells, spaces, field_index, dofmaps, offs, quad_degree, dofmap_for, tabulate
):
    """Host-static structure of the general boundary-integrand path, built once.

    One group per (region, test field): the region's boundary facets (all vertices in the region's
    node mask, used by one cell), their owner cell / local facet, the outward-orientation sign, the
    facet quadrature, every field's reference tables per local facet, and for each ``u.across(main)``
    the term reads: per facet quadrature point, the facing point of ``main`` (its projection), that
    point's owner cell, the field's DOFs there and its physical basis -- so the across value is
    ``sum_j u[dofs_j] * phi_j`` and its operator columns are exactly those main DOFs."""
    import basix

    from ..._fem import _walk
    from .fem_1d import _line_quadrature
    from .fem_dofmap import _reference_topology, basis_at_points_batch, facet_outward_normals, facet_owners
    from .fem_utils import _lower_statefield_to_trial, _test_field_index
    from .weak_form import _apply_sign, _split_additive_terms

    tdm = dofmap_for("Lagrange", 1)  # numbering of every entity dim (facets included); P1 = vertex ids
    count, fcell, flocal = facet_owners(tdm)
    fverts = tdm.entity_vertices[dim - 1]
    P = np.asarray(domain.mesh.points)[:, :dim]
    topo = _reference_topology(cell_name)
    lv = [list(v) for v in topo[dim - 1]]
    ref_cell = np.vstack([np.zeros(dim), np.eye(dim)])
    if dim == 2:
        fq, fw = (np.asarray(a).reshape(-1) for a in _line_quadrature(quad_degree))
        fq = fq[:, None]
    else:
        fq, fw = (np.asarray(a) for a in basix.make_quadrature(basix.CellType.triangle, quad_degree))
    Xref = [ref_cell[lv[k][0]] + fq @ (ref_cell[lv[k][1:]] - ref_cell[lv[k][0]]) for k in range(len(lv))]
    per_k = [tabulate(X) for X in Xref]

    def _stack(get):
        vals = [get(T) for T in per_k]
        if vals[0] is None:
            return None
        if isinstance(vals[0], tuple):
            return tuple(None if vals[0][j] is None else jnp.stack([v[j] for v in vals]) for j in range(len(vals[0])))
        return jnp.stack(vals)

    TS = {
        "pts": _stack(lambda T: T["pts"]),
        "n": per_k[0]["n"],
        "p1": _stack(lambda T: T["p1"]),
        "edge": {i: _stack(lambda T, i=i: T["edge"][i]) for i in per_k[0]["edge"]},
        "lag": {i: _stack(lambda T, i=i: T["lag"][i]) for i in per_k[0]["lag"]},
        "hermite": _stack(lambda T: T["hermite"]),
        "argyris": _stack(lambda T: T["argyris"]),
        "morley": _stack(lambda T: T["morley"]),
    }

    def _region_facets(region):
        mask = domain.tag_node_mask(region, np.asarray(domain.mesh.points))
        if mask is None:
            raise ValueError(f"jno.fem (non-nodal): boundary region {region!r} has no location function.")
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        inside = mask[fverts].all(axis=1)
        fac = np.flatnonzero((count == 1) & inside)
        cel, loc = fcell[fac], flocal[fac]
        owner = (getattr(domain, "_tag_regions", {}) or {}).get(region)
        if owner is not None:
            # A body-owned tag (`d.tag(name, where, region=body)`): only that body's facets, including
            # its surface INSIDE a conforming mesh (against another region), each integrated from the
            # body's side -- so the normal points out of the body, as it does on a boundary face.
            from ...domain.mesh_utils import mesh_cell_region_membership

            memb = mesh_cell_region_membership(domain.mesh, dim).get(owner)
            if memb is None:
                raise ValueError(f"jno.fem (non-nodal): tag {region!r} names region {owner!r}, which has no cells.")
            memb = np.asarray(memb, dtype=bool)
            keep = memb[cel]
            fac, cel, loc = fac[keep], cel[keep], loc[keep]
            fi2 = np.flatnonzero((count == 2) & inside)
            if fi2.size:
                pos = np.full(int(count.size), -1, dtype=np.int64)
                pos[fi2] = np.arange(len(fi2))
                n_in = np.zeros(len(fi2), dtype=np.int64)
                oc = np.full(len(fi2), -1, dtype=np.int64)
                ol = np.full(len(fi2), -1, dtype=np.int64)
                ce = np.asarray(tdm.cell_entities[dim - 1])
                for kk in range(ce.shape[1]):  # every (cell, local facet) on a candidate facet
                    p_ = pos[ce[:, kk]]
                    hit = np.flatnonzero((p_ >= 0) & memb)
                    np.add.at(n_in, p_[hit], 1)
                    oc[p_[hit]], ol[p_[hit]] = hit, kk
                one = n_in == 1  # exactly one side in the body: its surface, not a facet inside it
                fac = np.concatenate([fac, fi2[one]])
                cel = np.concatenate([cel, oc[one]])
                loc = np.concatenate([loc, ol[one]])
        if fac.size == 0:
            raise ValueError(
                f"jno.fem (non-nodal): boundary region {region!r} contains no boundary facet (a facet counts "
                "when ALL its vertices lie in the region) -- its terms would be silently dropped."
            )
        return fac, cel, loc

    def _raw_normals(c, k):
        V = P[np.asarray(cells)[c]]
        fvv = np.take_along_axis(V, np.asarray(lv)[k][:, :, None], axis=1)
        if dim == 3:
            return np.cross(fvv[:, 1] - fvv[:, 0], fvv[:, 2] - fvv[:, 0])
        t = fvv[:, 1] - fvv[:, 0]
        return np.stack([-t[:, 1], t[:, 0]], axis=1)

    def _project(X, main_fac):
        """Closest point of the facets ``main_fac`` to each point of ``X`` -> (index into ``main_fac``, point)."""
        from scipy.spatial import cKDTree

        FV = P[fverts[main_fac]]  # (m, dim, dim) facet vertices
        tree = cKDTree(FV.mean(axis=1))
        kq = min(len(main_fac), 12)
        _d, cand = tree.query(X, k=kq)
        cand = np.asarray(cand).reshape(len(X), -1)
        best_f = np.zeros(len(X), dtype=np.int64)
        best_x = np.zeros_like(X)
        best_d = np.full(len(X), np.inf)
        for j in range(cand.shape[1]):
            fvj = FV[cand[:, j]]
            A0 = fvj[:, 0]
            E = fvj[:, 1:] - A0[:, None, :]  # (n, dim-1, dim)
            G = np.einsum("nad,nbd->nab", E, E)
            rhs = np.einsum("nad,nd->na", E, X - A0)
            lam = np.linalg.solve(G, rhs[..., None])[..., 0]  # barycentric of the in-plane projection
            # clamp into the simplex (cheap, exact for points over the facet, a bound otherwise)
            lam = np.clip(lam, 0.0, 1.0)
            sm = lam.sum(axis=1, keepdims=True)
            lam = np.where(sm > 1.0, lam / np.maximum(sm, 1e-300), lam)
            Xp = A0 + np.einsum("na,nad->nd", lam, E)
            dd = np.linalg.norm(Xp - X, axis=1)
            better = dd < best_d
            best_d[better], best_f[better], best_x[better] = dd[better], cand[better, j], Xp[better]
        return best_f, best_x  # index INTO main_fac

    across_pairs = dict(getattr(domain, "_across_pairs", {}) or {})
    groups = []
    for region, terms in gsurf_terms.items():
        fac, c, k = _region_facets(region)
        nout = facet_outward_normals(tdm, P, cells, c, k)
        sgn = np.sign(np.einsum("nd,nd->n", _raw_normals(c, k), nout))
        # physical quadrature points of every facet (host, static mesh -- the across pairing)
        J = np.stack([P[np.asarray(cells)[c][:, a + 1]] - P[np.asarray(cells)[c][:, 0]] for a in range(dim)], axis=2)
        X = P[np.asarray(cells)[c][:, 0]][:, None, :] + np.einsum("nqa,nda->nqd", np.asarray([Xref[kk] for kk in k]), J)
        keys = sorted({str(n.tag) for t in terms for n in _walk(t) if str(getattr(n, "tag", "")).startswith("across_")})
        across = {}
        for key in keys:
            if key not in across_pairs:
                raise ValueError(f"jno.fem: {key!r} was not declared through u.across(...) on this domain.")
            fk, main = across_pairs[key]
            fi = field_index.get(fk)
            if fi is None:
                raise ValueError(f"u.across: its field {fk!r} is not one of this form's unknowns.")
            sp = spaces[fi]
            if sp in ("Hermite", "Argyris", "Morley"):
                raise NotImplementedError(
                    f"u.across on a {sp} (plate) field: its basis needs the per-cell M(cell) transform at the "
                    "main points, which this pairing does not build. The H(curl)/H(div)/Lagrange/DG/P0 "
                    "families are supported."
                )
            mf, mcell, _mloc = _region_facets(main)
            Xf = X.reshape(-1, dim)
            mfi, Xp = _project(Xf, mf)
            cB = mcell[mfi]
            if sp == "P0":
                dofs = offs[fi] + cB[:, None]
                phi = np.ones((len(cB), 1, 1))
            else:
                dmB = dofmaps.get(fi) or dofmap_for("Lagrange", 1)
                phi = basis_at_points_batch(dmB, P, np.asarray(cells), cB, Xp)
                dofs = offs[fi] + dmB.cell_dofs[cB]
            nq = X.shape[1]
            across[key] = (
                jnp.asarray(dofs.reshape(len(fac), nq, -1), dtype=jnp.int32),
                jnp.asarray(phi.reshape(len(fac), nq, phi.shape[1], phi.shape[2])),
            )
        by_test: dict = {}
        for t in terms:
            for sign_, sub in _split_additive_terms(domain, t):
                coeff = _lower_statefield_to_trial(_apply_sign(domain, sign_, sub), {})
                tfi = _test_field_index(coeff, field_index)
                if tfi is None:
                    raise ValueError("jno.fem (non-nodal): a boundary term must contain exactly one test field.")
                by_test.setdefault(tfi, []).append(coeff)
        for tfi, coeffs in by_test.items():
            groups.append(
                {
                    "region": region,
                    "tfi": tfi,
                    "coeffs": coeffs,
                    "cell": jnp.asarray(c, dtype=jnp.int32),
                    "kf": jnp.asarray(k, dtype=jnp.int32),
                    "sgn": jnp.asarray(sgn),
                    "lv": jnp.asarray(np.asarray(lv), dtype=jnp.int32),
                    "fqw": jnp.asarray(fw),
                    "tabs": TS,
                    "across": across,
                    "n_facets": int(len(fac)),
                    "cell_np": np.asarray(c),
                }
            )
    return groups


def _n1e_surface_mass_spec(bare):
    """Recognise a tangential-trace surface-mass boundary term ``[c *] inner(n×u, n×v)`` (the natural
    N1E impedance / first-order absorbing BC for Maxwell) → a 1-tuple ``(coeff_node,)`` (``coeff_node``
    is ``None`` when the coefficient is 1); ``None`` if the term is not of this form.

    The tangential trace ``n×u`` is authored as ``u.vector.cross(nvec)`` (a ``cross`` FunctionCall over
    the trial and the region normal), paired with the test's ``n×v`` inside an ``inner`` FunctionCall,
    optionally scaled by a trial/test-free scalar coefficient (e.g. ``i·k₀``). Distinct from the
    essential PEC ``n×u = 0`` (no test function → pinned, not assembled)."""
    from ..._fem import _contains
    from ...trace import BinaryOp, FunctionCall, Literal, TestFunction, TrialFunction

    node, coeff = bare, None
    wrap = None  # a `.real`/`.imag` wrapper from the complex leg split (Re/Im of `c·inner(n×u,n×v)`)
    if isinstance(node, FunctionCall) and node._name in ("real", "imag") and len(node.args) == 1:
        wrap, node = node._name, node.args[0]
    if isinstance(node, BinaryOp) and node.op == "*":  # peel a leading/trailing scalar coefficient
        left, right = node.left, node.right
        l_uv = _contains(left, TrialFunction) or _contains(left, TestFunction)
        r_uv = _contains(right, TrialFunction) or _contains(right, TestFunction)
        if isinstance(right, FunctionCall) and right._name == "inner" and not l_uv:
            coeff, node = left, right
        elif isinstance(left, FunctionCall) and left._name == "inner" and not r_uv:
            coeff, node = right, left
    if not (isinstance(node, FunctionCall) and node._name == "inner" and len(node.args) == 2):
        return None

    def _is_cross(x, which):
        return isinstance(x, FunctionCall) and x._name == "cross" and len(x.args) == 2 and _contains(x, which)

    a0, a1 = node.args
    ok = (_is_cross(a0, TrialFunction) and _is_cross(a1, TestFunction)) or (
        _is_cross(a0, TestFunction) and _is_cross(a1, TrialFunction)
    )
    if not ok:
        return None
    if wrap is not None:  # fold the leg's Re/Im into the coefficient: Re(i·k₀)=0, Im(i·k₀)=k₀
        base = coeff if coeff is not None else Literal(1.0)
        coeff = base.real if wrap == "real" else base.imag
    return (coeff,)


def _n1e_surface_load_spec(bare):
    """Recognise an N1E incident/forcing surface term ``inner(g, n×v)`` (trial-FREE, into ``b``) → a
    1-tuple ``(g_node,)`` (the prescribed tangential source, e.g. the incident wave ``2 i k₀ E_inc``);
    ``None`` otherwise. Distinct from the bilinear impedance mass (which carries the trial as well)."""
    from ..._fem import _contains
    from ...trace import BinaryOp, FunctionCall, TestFunction, TrialFunction

    def _source(x):  # trial/test-free factor (the source g or a scalar coefficient)
        return not _contains(x, TrialFunction) and not _contains(x, TestFunction)

    node, wrap, coeff = bare, None, None
    if isinstance(node, FunctionCall) and node._name in ("real", "imag") and len(node.args) == 1:
        wrap, node = node._name, node.args[0]
    if isinstance(node, BinaryOp) and node.op == "*":  # peel a scalar coefficient (e.g. 2 i k₀) into the source
        left, right = node.left, node.right
        if isinstance(right, FunctionCall) and right._name == "inner" and _source(left):
            coeff, node = left, right
        elif isinstance(left, FunctionCall) and left._name == "inner" and _source(right):
            coeff, node = right, left
    if not (isinstance(node, FunctionCall) and node._name == "inner" and len(node.args) == 2):
        return None

    def _cross_test(x):
        return isinstance(x, FunctionCall) and x._name == "cross" and len(x.args) == 2 and _contains(x, TestFunction)

    a0, a1 = node.args
    if _cross_test(a0) and _source(a1):
        g = a1
    elif _cross_test(a1) and _source(a0):
        g = a0
    else:
        return None
    if coeff is not None:  # fold the scalar coefficient into the source: c·inner(g, n×v) = inner(c·g, n×v)
        g = coeff * g
    if wrap is not None:  # complex leg split: Re/Im of the (complex) source
        g = g.real if wrap == "real" else g.imag
    return (g,)


def _n1e_surface_precompute(domain, top, quad_degree, spaces, dim, dofmaps=None):
    """Shared precompute for H(curl) boundary-face surface integrals (impedance mass + incident load), any
    degree: validate 3-D/H(curl), build facet connectivity, tabulate the field's basix element at reference
    face-quadrature points (per local face), and return the DOF map pieces. Returns ``(fidx, cells, fc,
    fqp, fqw, face_rv, signs, cell_dofs, dm)``; ``signs`` is the ±1 orientation when the field's DOF
    transform is diagonal (degree 1 -- the historic path), else ``None`` and ``dm`` carries the per-cell
    transform."""
    import basix
    from basix import CellType

    from .fem_facets import _LOCAL_FACES_TET, build_facet_connectivity

    if dim != 3:
        raise NotImplementedError(
            "jno.fem (non-nodal): N1E tangential-trace surface terms (impedance / absorbing / incident BC) are "
            "wired for 3-D H(curl) (Maxwell) only — the 2-D H(curl) tangential trace is a scalar u·t, not yet wired."
        )
    fidx = next((i for i, s in enumerate(spaces) if s in ("N1E", "N2E")), None)
    if fidx is None:
        raise NotImplementedError(
            "jno.fem (non-nodal): N1E surface terms are only supported on an H(curl) (N1E/N2E) field."
        )

    cells = np.asarray(domain.mesh.cells_dict["tetra"], dtype=np.int64)
    fc = build_facet_connectivity(cells, "tetrahedron")
    dm = None if dofmaps is None else dofmaps.get(fidx)
    if dm is None:
        from .fem_dofmap import build_dofmap

        dm = build_dofmap(cells, "N1E", 1, n_verts=int(np.asarray(domain.mesh.points).shape[0]))
    elem = dm.element
    fqp, fqw = (np.asarray(a) for a in basix.make_quadrature(CellType.triangle, quad_degree))  # (nqf,2),(nqf,)
    ref_tet = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    b0, b1 = fqp[:, 0:1], fqp[:, 1:2]  # face barycentric weights (ξ, η); the third is 1−ξ−η
    face_rv = []  # per local face f: basis (nqf, n_dof, 3) at the face-mapped reference points
    for f in range(4):
        V = ref_tet[list(_LOCAL_FACES_TET[f][:3])]
        face_rv.append(np.asarray(elem.tabulate(0, (1.0 - b0 - b1) * V[0] + b0 * V[1] + b1 * V[2])[0]))
    signs = np.asarray(dm.signs) if dm.is_diagonal else None  # (nc, n_dof) — equals the edge signs at degree 1
    cell_dofs = np.asarray(dm.cell_dofs)  # (nc, n_dof) — equals the global edge ids at degree 1
    return fidx, cells, fc, fqp, fqw, face_rv, signs, cell_dofs, dm


def _n1e_face_geometry_batch(bfs, cells, fc, pts_np, face_rv, fqp, signs, dm=None):
    """Per boundary face in ``bfs``: the physical H(curl) basis ``(n, nqf, n_dof, 3)``, the OUTWARD unit
    normal ``(n, 3)``, the face measure |detJ_face| (= 2·area) ``(n,)`` and the physical quad points
    ``(n, nqf, 3)``, with the owner cells ``(n,)``. Covariant-Piola push with the same cell Jacobian and DOF
    orientation (±1 signs, or the degree-k transform ``B``) the volume assembly uses."""
    from .fem_facets import _LOCAL_FACES_TET

    c = np.asarray(fc.parent_cell)[bfs].astype(np.int64)
    f = np.asarray(fc.local_face)[bfs].astype(np.int64)
    cverts = np.asarray(pts_np)[np.asarray(cells)[c]]  # (n, 4, 3)
    J = np.stack([cverts[:, k] - cverts[:, 0] for k in (1, 2, 3)], axis=2)
    rv = np.asarray(face_rv)[f]  # (n, nqf, n_dof, 3)
    if signs is not None:
        rv_c = rv * np.asarray(signs)[c][:, None, :, None]
    else:
        rv_c = np.array(rv)
        for d_, blk in dm.blocks.items():
            for k, idx in enumerate(dm.entity_dofs[d_]):
                Bk = blk[k][np.asarray(dm.orient[d_])[c, k].astype(np.int64)]  # (n, nd, nd)
                rv_c[:, :, idx] = np.einsum("nij,nqjv->nqiv", Bk, rv[:, :, idx])
    phi = np.einsum("nji,nqaj->nqai", np.linalg.inv(J), rv_c)
    lv = np.asarray([list(_LOCAL_FACES_TET[k][:3]) for k in range(4)])[f]  # (n, 3)
    P = np.take_along_axis(cverts, lv[:, :, None], axis=1)  # physical face vertices, reference order
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    measure = np.linalg.norm(nrm, axis=1)  # = 2·area = |det of the face map|
    nhat = nrm / measure[:, None]
    opp = cverts[np.arange(len(c)), 6 - lv.sum(axis=1)]  # the 4th vertex (local ids sum to 6) → outward
    nhat = np.where((np.einsum("ni,ni->n", nhat, P[:, 0] - opp) < 0)[:, None], -nhat, nhat)
    b0, b1 = np.asarray(fqp)[:, 0:1], np.asarray(fqp)[:, 1:2]
    xq = (1.0 - b0 - b1)[None] * P[:, None, 0] + b0[None] * P[:, None, 1] + b1[None] * P[:, None, 2]
    return c, phi, nhat, measure, xq


def _bcast_surface_vals(vals, n, nq, comp=None):
    """Broadcast an evaluated surface coefficient to ``(n, nq)`` (or ``(n, nq, comp)``).

    ``_eval_value_node_at`` returns as few values as the node needs: a CONSTANT coefficient collapses to
    a single entry however many points it was handed, a per-point one returns all of them. Batching the
    faces means the caller must accept either, so dispatch on size rather than assuming the full shape."""
    v = jnp.asarray(vals).reshape(-1)
    tail = () if comp is None else (comp,)
    full = n * nq * (1 if comp is None else comp)
    if v.size == full:
        return v.reshape((n, nq) + tail)
    if comp is not None and v.size == comp:  # one constant vector for every point
        return jnp.broadcast_to(v.reshape((1, 1, comp)), (n, nq, comp))
    if v.size == nq * (1 if comp is None else comp):  # per-quad-point, face-independent
        return jnp.broadcast_to(v.reshape((1, nq) + tail), (n, nq) + tail)
    return jnp.broadcast_to(v.reshape((1,) * (2 + len(tail))), (n, nq) + tail)  # scalar


def _n1e_surface_static(kind, terms, domain, spaces, top, pts_np, offs, quad_degree, dim, dofmaps=None):
    """HOST-STATIC per-(face, term) structure for an N1E surface term — built ONCE.

    The mesh, the region membership and the face geometry do not depend on the runtime parameters, yet
    this was rebuilt on EVERY operator evaluation: a Python loop over every boundary face, plus a host
    ``np.where`` region lookup, per assembly. That both dominated the parametric assembly cost and made
    the whole path un-``jit``-able (a host ``np.where`` cannot see a tracer). Hoisting it leaves the
    per-args path pure JAX.

    ``kind`` is ``"mass"`` (tangential ``φ·φ − (φ·n)(φ·n)``, the impedance BC) or ``"load"`` (``φ×n``,
    the incident source). Entries are grouped by coefficient node so each group evaluates and scatters
    in ONE batched op instead of one per face.

    Returns ``(groups, fqw)`` with ``groups = [(coeff_node, gdofs(n,n_dof), tens(n,nq,...), xq(n,nq,3),
    measure(n,)), ...]``."""
    from .fem_1d import _region_node_ids

    fidx, cells, fc, fqp, fqw, face_rv, signs, cell_edges, dm = _n1e_surface_precompute(
        domain, top, quad_degree, spaces, dim, dofmaps
    )
    fqw = np.asarray(fqw)
    groups = {}  # id(coeff node) -> [node, gdofs[], tens[], xq[], measure[]]  (insertion-ordered)
    for region, nodes in terms.items():
        rmask = np.zeros(len(pts_np), dtype=bool)
        rmask[np.asarray(list(_region_node_ids(domain, region)), dtype=np.int64)] = True
        bfs = np.flatnonzero(rmask[np.asarray(fc.face_nodes)[: fc.n_bfaces]].all(axis=1))
        if bfs.size == 0:
            continue
        c, phi, nhat, measure, xq = _n1e_face_geometry_batch(bfs, cells, fc, pts_np, face_rv, fqp, signs, dm)
        if kind == "mass":
            pn = np.einsum("nqai,ni->nqa", phi, nhat)  # normal component
            tens = np.einsum("nqai,nqbi->nqab", phi, phi) - np.einsum("nqa,nqb->nqab", pn, pn)
        else:
            tens = np.cross(phi, nhat[:, None, None, :])  # φ_a × n (the authored `v.vector.cross(nvec)`)
        gdofs = np.asarray(offs[fidx] + np.asarray(cell_edges)[c], dtype=np.int64)
        for node in nodes:
            g = groups.setdefault(id(node), [node, [], [], [], []])
            g[1].append(gdofs)
            g[2].append(tens)
            g[3].append(xq)
            g[4].append(measure)
    out = [
        (node, np.concatenate(gd), np.concatenate(tn), np.concatenate(xq), np.concatenate(ms).astype(float))
        for node, gd, tn, xq, ms in groups.values()
    ]
    return out, fqw


def _assemble_n1e_surface_mass(total, static, params=None, sparse=False):
    """Tangential-trace surface mass ``∑ ∫_Γ c (n×φ_a)·(n×φ_b) dS`` (the N1E impedance / absorbing BC),
    to ADD to the stiffness ``A``.

    Pure JAX over the host-static structure from :func:`_n1e_surface_static`, so it is differentiable in
    a runtime parameter inside ``c`` (inverse design of a surface impedance) AND ``jit``-able. With
    ``sparse=True`` it returns COO triplets as a BCOO instead of a dense ``(total, total)`` matrix —
    a dense one is ``O(n_dof²)``, which for a 3-D vector problem is the memory wall, not a detail."""
    from ..._fem import _eval_value_node_at

    groups, fqw = static
    if not groups:
        return None if sparse else jnp.zeros((total, total))
    nq = len(fqw)
    rows_l, cols_l, data_l = [], [], []
    for node, gd, tn, xq, ms in groups:
        n = gd.shape[0]
        pts_flat = jnp.asarray(xq).reshape(-1, xq.shape[-1])
        cval = (
            jnp.ones((n, nq))
            if node is None
            else _bcast_surface_vals(_eval_value_node_at(node, pts_flat, params=params), n, nq)
        )
        wq = jnp.asarray(fqw)[None, :] * jnp.asarray(ms)[:, None] * cval  # (n, nq) quad weight x measure x c
        blk = jnp.einsum("nq,nqab->nab", wq, jnp.asarray(tn))  # (n, 6, 6) element surface-mass blocks
        g = jnp.asarray(gd)
        rows_l.append(jnp.broadcast_to(g[:, :, None], blk.shape).reshape(-1))
        cols_l.append(jnp.broadcast_to(g[:, None, :], blk.shape).reshape(-1))
        data_l.append(blk.reshape(-1))
    rows, cols, data = jnp.concatenate(rows_l), jnp.concatenate(cols_l), jnp.concatenate(data_l)
    if sparse:
        from jax.experimental import sparse as _jsp

        idx = jnp.stack([rows.astype(jnp.int32), cols.astype(jnp.int32)], axis=1)
        return _jsp.BCOO((data, idx), shape=(total, total))  # duplicate (i,j) sum on materialisation
    return jnp.zeros((total, total), data.dtype).at[rows, cols].add(data)


def _assemble_n1e_surface_load(total, static, params=None):
    """Tangential incident/forcing surface term ``∑ ∫_Γ g·(φ_a×n) dS`` (the ``inner(g, n×v)`` RHS of the
    Silver-Müller ABC) as a ``(total,)`` load to ADD to ``b``. Pure JAX over the host-static structure,
    so a runtime parameter in ``g`` stays differentiable (inverse design of an incident source)."""
    from ..._fem import _eval_value_node_at

    groups, fqw = static
    if not groups:
        return jnp.zeros(total)
    nq = len(fqw)
    idx_l, val_l = [], []
    for node, gd, tn, xq, ms in groups:
        n = gd.shape[0]
        pts_flat = jnp.asarray(xq).reshape(-1, xq.shape[-1])
        gval = _bcast_surface_vals(_eval_value_node_at(node, pts_flat, params=params), n, nq, comp=3)
        integrand = jnp.einsum("nqi,nqai->nqa", gval, jnp.asarray(tn))  # g·(φ_a×n)
        wq = jnp.asarray(fqw)[None, :] * jnp.asarray(ms)[:, None]  # (n, nq)
        idx_l.append(jnp.asarray(gd).reshape(-1))
        val_l.append(jnp.einsum("nq,nqa->na", wq, integrand).reshape(-1))
    ids, vals = jnp.concatenate(idx_l), jnp.concatenate(val_l)
    return jnp.zeros(total, vals.dtype).at[ids].add(vals)


def _facet_basis(dm, pts_np, cells, facets, quad_degree):
    """Physical basis of ``dm``'s element on boundary facets ``facets`` (global facet ids), host numpy.

    Returns ``(owner_cells, xq (n, nq, gdim), wq (n, nq) = weight · facet measure, normals (n, gdim)
    outward, phi (n, nq, n_dof, value_size))`` -- the reference basis tabulated at the facet's quadrature
    points mapped into the owner cell, oriented by the cell's DOF transform and Piola-mapped. One
    tabulation per local facet index, so the cost is a few einsums per facet group."""
    import basix

    from .fem_dofmap import _reference_topology, cell_jacobians, facet_outward_normals, facet_owners, map_type

    tdim = dm.tdim
    topo = _reference_topology(dm.cell)
    fd = tdim - 1
    ref_cell = np.vstack([np.zeros(tdim), np.eye(tdim)])
    if tdim == 2:
        from .fem_1d import _line_quadrature

        fqp, fqw = (np.asarray(a).reshape(-1) for a in _line_quadrature(quad_degree))
        fqp = fqp[:, None]
    else:
        fqp, fqw = (np.asarray(a) for a in basix.make_quadrature(basix.CellType.triangle, quad_degree))
    count, fcell, flocal = facet_owners(dm)
    facets = np.asarray(facets, dtype=np.int64)
    cs, ks = fcell[facets], flocal[facets]
    J, x0 = cell_jacobians(pts_np, cells)
    normals = facet_outward_normals(dm, pts_np, cells, cs, ks)
    mt = map_type(dm.family)
    nq = fqw.shape[0]
    gd = pts_np.shape[1]
    xq = np.zeros((len(facets), nq, gd))
    wq = np.zeros((len(facets), nq))
    phi = np.zeros((len(facets), nq, dm.ndof_local, dm.element.value_size))
    for kf in range(len(topo[fd])):
        sel = np.flatnonzero(ks == kf)
        if sel.size == 0:
            continue
        V = ref_cell[list(topo[fd][kf])]  # reference facet vertices
        Xr = V[0][None, :] + fqp @ (V[1:] - V[0])  # (nq, tdim) reference-cell points on the facet
        tab = np.asarray(dm.element.tabulate(0, Xr)[0])  # (nq, n_dof, vs)
        c = cs[sel]
        Jc = J[c]
        xq[sel] = x0[c][:, None, :] + np.einsum("qa,nda->nqd", Xr, Jc)
        Pf = pts_np[np.asarray(cells)[c][:, list(topo[fd][kf])]]  # physical facet vertices
        if tdim == 2:
            meas = np.linalg.norm(Pf[:, 1] - Pf[:, 0], axis=1)
        else:
            meas = np.linalg.norm(np.cross(Pf[:, 1] - Pf[:, 0], Pf[:, 2] - Pf[:, 0]), axis=1)
        wq[sel] = fqw[None, :] * meas[:, None]
        if dm.is_diagonal:
            tc = tab[None] * dm.signs[c][:, None, :, None]
        else:
            tc = np.einsum("nij,qjv->nqiv", np.stack([dm.cell_transform_np(int(ci)) for ci in c]), tab)
        if mt == "covariant":
            phi[sel] = np.einsum("nji,nqaj->nqai", np.linalg.inv(Jc), tc)
        elif mt == "contravariant":
            phi[sel] = np.einsum("nij,nqaj->nqai", Jc, tc) / np.linalg.det(Jc)[:, None, None, None]
        else:
            phi[sel] = tc
    return cs, xq, wq, normals, phi


def _rt_pressure_load_general(b, pd_node, fidx, region, dm, domain, pts_np, cells, offs, quad_degree):
    """``b -= ∮_region p_D (φ·n_out) ds`` for an H(div) field of ANY degree, in 2-D or 3-D -- the natural
    pressure BC of mixed Poisson, by genuine facet quadrature (the degree-1 2-D path keeps its exact
    RT0 shortcut; this reproduces it, see the tests)."""
    from ..._fem import _eval_value_node_at
    from .fem_dofmap import facet_owners

    mask = domain.tag_node_mask(region, np.asarray(domain.mesh.points))
    if mask is None:
        raise ValueError(f"jno.fem (non-nodal): natural-BC region {region!r} has no location function.")
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    count, _c, _l = facet_owners(dm)
    fd = dm.tdim - 1
    facets = np.flatnonzero((count == 1) & mask[dm.entity_vertices[fd]].all(axis=1))
    if facets.size == 0:
        return b
    cs, xq, wq, normals, phi = _facet_basis(dm, pts_np, cells, facets, quad_degree)
    pd = np.asarray(_eval_value_node_at(pd_node, jnp.asarray(xq.reshape(-1, xq.shape[-1])))).reshape(-1)
    pd = np.broadcast_to(pd, (xq.shape[0] * xq.shape[1],)) if pd.size == 1 else pd
    pd = pd.reshape(xq.shape[:2])
    # The pressure term ∮ p_D (v·n) belongs to the residual R(u), and b = -R(0): it enters b with a
    # MINUS. (The legacy RT0 shortcut `sign_topo · ⟨p_D⟩` is the same thing; the degree-1 test pins it.)
    contrib = -np.einsum("nq,nq,nqav,nv->na", wq, pd, phi, normals)
    b = np.asarray(b).copy()
    np.add.at(b, (offs[fidx] + dm.cell_dofs[cs]).reshape(-1), contrib.reshape(-1))
    return b


def _apply_natural_boundary_terms(
    b, boundary_terms, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree, dofmaps=None
):
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
            _dm = None if dofmaps is None else dofmaps.get(fidx)
            if _dm is not None and (_dm.degree > 1 or _dm.tdim == 3):  # degree k / tetrahedra: facet quadrature
                _cells_g = np.asarray(domain.mesh.cells_dict["tetra" if _dm.tdim == 3 else "triangle"], dtype=np.int64)
                b = _rt_pressure_load_general(
                    b, pd_node, fidx, region, _dm, domain, pts_np[:, : _dm.tdim], _cells_g, offs, quad_degree
                )
                continue
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


def _lagrange_dirichlet_pins(dirichlet_raw, domain, field_index, spaces, pts_np, offs, dofmaps=None, cells=None):
    """Value-Dirichlet ``(dof, value)`` pins for a P1 Lagrange field on the non-nodal path: one DOF per
    mesh vertex, so ``u = g`` on a region pins DOF ``offs[fidx] + v`` to ``g(vertex v)`` for every vertex
    the region's location predicate selects. This is the scalar-potential half of the A-V (N1E x Lagrange)
    pair -- V = g on a terminal face -- and the exact analogue of :func:`_hermite_dirichlet_pins` with a
    1-DOF-per-vertex layout (Hermite's value DOF sits at ``3*v``; Lagrange's IS ``v``).

    Unlike the Hermite loop, ``g`` is evaluated in ONE :func:`_eval_value_node_at` call over the whole
    region (a terminal face of a production mesh has thousands of vertices; a per-vertex TraceEvaluator
    round-trip is minutes of pure Python). Every degenerate case raises rather than degrades: an empty
    region (a predicate that matched nothing -- the float32-tolerance trap), a complex ``g`` (the
    complex non-nodal split assembles two REAL legs whose fused Dirichlet rows solve to ``u_r = g,
    u_i = 0``, correct only for real ``g``), and a value/vertex-count mismatch.
    """
    from ..._fem import _eval_value_node_at
    from .fem_1d import _region_node_ids

    pins = []
    for fk, region, _comp, _value, value_node in dirichlet_raw:
        fidx = field_index.get(fk)
        if fidx is None or spaces[fidx] != "Lagrange":
            continue
        if dofmaps is not None and fidx in dofmaps:  # P_k, k >= 2: every DOF on the region, by interpolation
            pins.extend(_general_trace_pins(dofmaps[fidx], "value", region, value_node, domain, pts_np, cells, offs[fidx]))
            continue
        vs = np.asarray(_region_node_ids(domain, region), dtype=np.int64)
        if vs.size == 0:
            raise ValueError(
                f"jno.fem (non-nodal): the Dirichlet region {region!r} on the Lagrange field matched "
                "NO mesh vertex -- the essential condition would be silently dropped. Check the region "
                "name / tag predicate (and its tolerance: a predicate finer than float32 eps needs the "
                "float64 tag path, see domain.tag_node_mask)."
            )
        g = np.asarray(_eval_value_node_at(value_node, jnp.asarray(pts_np[vs])))
        if np.iscomplexobj(g):
            if np.abs(g.imag).max() > 0.0:
                raise NotImplementedError(
                    "jno.fem (non-nodal): a COMPLEX Dirichlet value on the Lagrange field is not wired -- "
                    "the complex split assembles two real legs sharing one real pin value, whose fused "
                    "rows enforce u_re = g, u_im = 0. Drive with a real terminal value (phase-reference "
                    "the source) or split the constraint yourself."
                )
            g = g.real
        g = g.reshape(-1)
        if g.shape[0] == 1 and vs.size > 1:  # a constant value node evaluates once; broadcast it
            g = np.full(vs.size, float(g[0]))
        if g.shape[0] != vs.size:
            raise ValueError(
                f"jno.fem (non-nodal): the Dirichlet value on region {region!r} evaluated to "
                f"{g.shape[0]} values for {vs.size} region vertices -- the value expression must be "
                "scalar per point (bind the boundary coordinate variables, not a vector expression)."
            )
        base = int(offs[fidx])
        pins.extend((base + int(v), float(gv)) for v, gv in zip(vs, g))
    return pins


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
    (an interior edge can join two boundary vertices through the volume). Homogeneous PEC, lowest order;
    every other case (nonzero ``u×n = g``, degree k, N2E, RT) goes through :func:`_general_trace_pins`."""
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


def _general_trace_pins(dm, kind, region, value_node, domain, pts_np, cells, base):
    """Essential-trace pins of ANY degree on ``region``, by interpolation (:mod:`fem_dofmap`).

    ``kind``: ``"tangential"`` (H(curl): ``u×n = g``; 2-D scalar ``g = u_x n_y - u_y n_x``, 3-D vector
    ``g``), ``"normal"`` (H(div): ``u·n = g``) or ``"value"`` (Lagrange: ``u = g``). The trace data is
    turned into a field whose trace is ``g`` -- ``g·(n_y, -n_x)`` in 2-D, ``n×g`` in 3-D (the
    tangential part of a ``g`` given as ``u×n``), ``g·n``, ``g`` -- and every DOF in the closure of the
    region's boundary facets is set to the corresponding entity functional of it, oriented by ``B⁻ᵀ``.
    The functionals see only the trace (tangential moments on edges/faces, normal moments on faces), so
    the extension off the facet never matters. A homogeneous trace pins those DOFs to 0 without
    evaluating anything. Facet criterion: boundary facets whose vertices all lie in the region (for a
    value BC on a region with no boundary facet, every facet, plus whole cells, of the region)."""
    from ..._fem import _constant_of, _eval_value_node_at
    from .fem_dofmap import (
        entity_dofs_of_tasks,
        facet_outward_normals,
        facet_owners,
        interpolate_tasks,
        region_trace_entities,
    )

    mask = domain.tag_node_mask(region, np.asarray(domain.mesh.points))
    if mask is None:
        raise ValueError(f"jno.fem (non-nodal): essential region {region!r} has no location function.")
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not mask.any():
        raise ValueError(
            f"jno.fem (non-nodal): the essential region {region!r} matched NO mesh vertex -- the condition "
            "would be silently dropped. Check the region name / tag predicate and its tolerance."
        )
    tasks = region_trace_entities(dm, mask, boundary_only=True)
    if kind == "value" and not any(f >= 0 for (_c, _j, f) in tasks.values()):
        tasks = region_trace_entities(dm, mask, boundary_only=False, cells_too=True)
    if not tasks:
        raise ValueError(
            f"jno.fem (non-nodal): the essential region {region!r} contains no boundary facet of the mesh, "
            "so its trace constrains nothing. A facet counts when ALL its vertices lie in the region."
        )
    try:
        homogeneous = _constant_of(value_node) == 0.0
    except Exception:  # noqa: BLE001 -- a non-constant node simply is not homogeneous
        homogeneous = False
    if homogeneous:
        dofs = entity_dofs_of_tasks(dm, tasks)
        return [(int(base + d_), 0.0) for d_ in dofs]
    tdim = dm.tdim
    count, fcell, flocal = facet_owners(dm)
    normals = facet_outward_normals(dm, pts_np, cells, fcell, flocal)

    def _g(X, comp):
        v = np.asarray(_eval_value_node_at(value_node, jnp.asarray(X)))
        if np.iscomplexobj(v):
            if np.abs(v.imag).max() > 0.0:
                raise NotImplementedError(
                    "jno.fem (non-nodal): a COMPLEX essential trace value is not wired -- the complex form "
                    "is solved as two real legs sharing one real pin value (u_re = g, u_im = 0). Drive with "
                    "a real boundary value or split the constraint yourself."
                )
            v = v.real
        v = v.reshape(-1)
        n = X.shape[0]
        if v.size == n * comp:
            return v.reshape(n, comp)
        if v.size == comp:
            return np.broadcast_to(v.reshape(1, comp), (n, comp))
        if v.size == 1:
            return np.full((n, comp), float(v[0]))
        raise ValueError(
            f"jno.fem (non-nodal): the essential value on region {region!r} evaluated to {v.size} values at "
            f"{n} points; expected {comp} per point."
        )

    if kind == "value":

        def fn(X, N):
            return _g(X, 1)

    elif kind == "normal":

        def fn(X, N):
            return _g(X, 1) * N

    elif tdim == 2:

        def fn(X, N):  # u×n = u_x n_y - u_y n_x = g  <=  u = g·(n_y, -n_x)
            return _g(X, 1) * np.stack([N[:, 1], -N[:, 0]], axis=1)

    else:

        def fn(X, N):  # u×n = g  <=  u_tan = n×g
            return np.cross(N, _g(X, 3))

    dofs, vals = interpolate_tasks(dm, pts_np[:, :tdim], cells, tasks, fn, normals=normals)
    return [(int(base + d_), float(v_)) for d_, v_ in zip(dofs, vals)]


def _flux_bc_pins(flux_bcs, domain, field_index, spaces, top, pts_np, offs, n_cells, quad_degree, *, dim=2, dofmaps=None):
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
    # Degree-k / N2E / 3-D RT / nonzero 3-D tangential traces: the general interpolation route. The two
    # lowest-order legacy routines below keep serving exactly the cases they always did, so those pins
    # (and the operators built from them) are unchanged.
    from ..._fem import _constant_of

    def _legacy(bc):
        fi = field_index.get(bc[0])
        if fi is None or dofmaps is None or fi not in dofmaps:
            return True  # let the legacy routine raise its own message
        dm_ = dofmaps[fi]
        if not (dm_.degree == 1 and spaces[fi] in ("RT", "N1E")):
            return False
        if dim == 3:
            if spaces[fi] != "N1E":
                return False
            try:
                return _constant_of(bc[2]) == 0.0
            except Exception:  # noqa: BLE001
                return False
        return True

    general = [bc for bc in flux_bcs if not _legacy(bc)]
    flux_bcs = [bc for bc in flux_bcs if _legacy(bc)]
    out = []
    if general:
        cells_g = np.asarray(domain.mesh.cells_dict["tetra" if dim == 3 else "triangle"], dtype=np.int64)
        for field_key, region, value_node in general:
            fi = field_index[field_key]
            kind = "normal" if spaces[fi] == "RT" else "tangential"
            out.extend(_general_trace_pins(dofmaps[fi], kind, region, value_node, domain, pts_np, cells_g, offs[fi]))
    if not flux_bcs:
        return out
    if dim == 3:  # 3-D N1E tangential (PEC) — facet-based boundary edges, not the 2-D single-use rule
        return out + _n1e_tangential_pins_3d(flux_bcs, domain, field_index, spaces, top, offs)
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
    return out + pins


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
    from .fem_dofmap import DofMap

    if isinstance(top, DofMap):  # any degree: the general read-back (``top`` is the field's DofMap)
        return nonnodal_field_at(points, cells, top, u_edge)[:, 0, :]
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
    from .fem_dofmap import DofMap

    if isinstance(top, DofMap):  # any degree: the general read-back (``top`` is the field's DofMap)
        return nonnodal_field_at(points, cells, top, u_edge)[:, 0, :]
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


def n1e_field_at_tet_centroids(points: np.ndarray, cells: np.ndarray, top, u_edge: jnp.ndarray, *, curl: bool = False):
    """Evaluate the 3-D Nédélec (H(curl)) field ``u_h`` (and optionally its curl) at each tet centroid.

    The tetrahedral counterpart of :func:`n1e_field_at_centroids`. Tabulates the 6-DOF N1E tet basis at
    the reference centroid, covariant-Piola-maps value and (physical) gradient per cell with the same
    edge-orientation signs used in assembly, and contracts with the cell's six edge-DOF coefficients
    ``u_edge[cell_edges]``. With ``curl=True`` the vector curl is recovered from the antisymmetric parts
    of the physical gradient (the invariant :func:`piola_covariant_grad` provides) — this is how one reads
    ``B = curl A`` off a magnetic-vector-potential solve.

    Returns ``values`` ``(n_cells, 3)``, or ``(values, curls)`` (each ``(n_cells, 3)``) when ``curl=True``.
    Complex ``u_edge`` gives complex outputs (the map is linear).
    """
    from .fem_dofmap import DofMap

    if isinstance(top, DofMap):  # any degree: pass the field's DofMap as ``top``
        val = nonnodal_field_at(points, cells, top, u_edge)[:, 0, :]
        return (val, nonnodal_field_at(points, cells, top, u_edge, derivative="curl")[:, 0, :]) if curl else val
    import basix

    from .fem_elements import piola_covariant, piola_covariant_grad

    elem = basix.create_element(basix.ElementFamily.N1E, basix.CellType.tetrahedron, 1)
    tab = elem.tabulate(1, np.array([[0.25, 0.25, 0.25]]))  # (4, 1, 6, 3): [values, d/dξ0, d/dξ1, d/dξ2]
    rv = jnp.asarray(tab[0])  # (1, 6, 3)
    rg = jnp.asarray(np.stack([tab[1], tab[2], tab[3]], axis=-1))  # (1, 6, 3, 3): ref grad d(Phi)_i/dξ_m
    pts = jnp.asarray(points)
    cells_j = jnp.asarray(np.asarray(cells), dtype=jnp.int32)
    signs = jnp.asarray(top.cell_edge_signs.astype(np.float64))
    ce = jnp.asarray(top.cell_edges, dtype=jnp.int32)
    coeffs = u_edge[ce]  # (n_cells, 6)

    def _tet_jac(verts):
        J = jnp.stack([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]], axis=1)  # (3, 3)
        return J, small_det(J)

    def _val(cell, sgn, c):
        J, detJ = _tet_jac(pts[cell])
        phi, _ = piola_covariant(rv, None, J, detJ, sgn)  # (1, 6, 3)
        val = jnp.einsum("a,ad->d", c, phi[0])  # (3,)
        if not curl:
            return val
        grad = piola_covariant_grad(rg, J, detJ, sgn)  # (1, 6, 3, 3): d(Phi)_i/dx_l
        g = jnp.einsum("a,ail->il", c, grad[0])  # (3, 3) physical gradient of u_h
        crl = jnp.stack([g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]])  # curl = antisymmetric parts
        return val, crl

    return jax.vmap(_val)(cells_j, signs, coeffs)


def _readback_kernel(dofmap, pts, derivative):
    """The per-(cell, points) evaluator shared by the read-backs: ``(c, cell_vertex_ids, coeffs, rv, rg)``
    -> field data at the points whose reference tabulation is ``rv``/``rg``."""
    from .fem_dofmap import map_type, transform_from_tables

    tdim = dofmap.tdim
    mt = map_type(dofmap.family)
    if derivative not in (None, "curl", "div", "grad"):
        raise ValueError(f"nonnodal read-back: derivative must be None, 'curl', 'div' or 'grad'; got {derivative!r}.")
    if derivative == "curl" and mt != "covariant":
        raise ValueError(f"nonnodal read-back: 'curl' is the H(curl) derivative; {dofmap.family} is {mt}-mapped.")
    if derivative == "div" and mt != "contravariant":
        raise ValueError(f"nonnodal read-back: 'div' is the H(div) derivative; {dofmap.family} is {mt}-mapped.")
    signs = None if not dofmap.is_diagonal else jnp.asarray(dofmap.signs)
    tables = None if signs is not None else dofmap.jax_tables()  # eager: never created under the vmap trace

    def _one(c, cell, cf, rv, rg):
        V = pts[cell]
        J = jnp.stack([V[k] - V[0] for k in range(1, tdim + 1)], axis=1)
        K = jnp.linalg.inv(J)
        B = jnp.diag(signs[c]) if signs is not None else transform_from_tables(tables, dofmap.ndof_local, c)
        v = jnp.einsum("ij,qjv->qiv", B, rv)
        g = jnp.einsum("ij,qjvm->qivm", B, rg)
        if mt == "covariant":
            phi = jnp.einsum("ji,qnj->qni", K, v)
            gp = jnp.einsum("ji,qnjm,ml->qnil", K, g, K)
        elif mt == "contravariant":
            dJ = jnp.linalg.det(J)
            phi = jnp.einsum("ij,qnj->qni", J, v) / dJ
            gp = jnp.einsum("ik,qnkm,ml->qnil", J, g, K) / dJ
        else:
            phi = v
            gp = jnp.einsum("qnvm,ml->qnvl", g, K)
        if derivative is None:
            out = jnp.einsum("n,qnv->qv", cf, phi)
            return out[..., 0] if mt == "identity" else out
        G = jnp.einsum("n,qnil->qil", cf, gp)  # (n_pts, vs, tdim) physical gradient of the field
        if derivative == "grad":
            return G[:, 0, :] if mt == "identity" else G
        if derivative == "div":
            return jnp.trace(G, axis1=1, axis2=2)
        if tdim == 2:
            return G[:, 1, 0] - G[:, 0, 1]
        return jnp.stack([G[:, 2, 1] - G[:, 1, 2], G[:, 0, 2] - G[:, 2, 0], G[:, 1, 0] - G[:, 0, 1]], axis=-1)

    return _one


def _tabulate(dofmap, ref_points):
    tdim = dofmap.tdim
    tab = dofmap.element.tabulate(1, np.atleast_2d(np.asarray(ref_points, dtype=float)))
    rv = jnp.asarray(tab[0])
    rg = jnp.asarray(np.stack([tab[1 + a] for a in range(tdim)], axis=-1))  # (n_pts, n_dof, vs, tdim)
    return rv, rg


def nonnodal_field_at(points, cells, dofmap, u, *, ref_points=None, cell_ids=None, derivative=None):
    """Evaluate a solved non-nodal field of ANY degree inside its cells -- the degree-k read-back.

    ``dofmap`` is the field's :class:`fem_dofmap.DofMap` (``domain._fem_nonnodal_topology["dofmap"]``
    after a solve, or :func:`fem_dofmap.build_dofmap`), ``u`` the field's block of the solution
    (``n_dofs``). ``ref_points`` ``(n_pts, tdim)`` are reference-cell points (default: the centroid),
    evaluated in every cell of ``cell_ids`` (default: all). The basis is the assembler's: basix reference
    tabulation, the per-cell DOF transform ``B`` (or the ±1 signs at degree 1), then the family's Piola
    map, so a read-back always describes the field that was solved.

    ``derivative``: ``None`` -> values ``(n_sel, n_pts, value_size)`` (scalar Lagrange: ``(n_sel,
    n_pts)``); ``"curl"`` (H(curl): ``(n_sel, n_pts)`` in 2-D, ``(n_sel, n_pts, 3)`` in 3-D); ``"div"``
    (H(div): ``(n_sel, n_pts)``); ``"grad"`` (Lagrange: ``(n_sel, n_pts, tdim)``; vector families:
    ``(n_sel, n_pts, value_size, tdim)``). Linear in ``u`` (complex ``u`` gives complex output) and pure
    JAX after the host tabulation, so it differentiates in ``u`` and in ``points``.
    :func:`nonnodal_field_at_points` evaluates at physical points instead."""
    pts = jnp.asarray(points)
    cells = np.asarray(cells, dtype=np.int64)
    tdim = cells.shape[1] - 1
    if ref_points is None:
        ref_points = np.full((1, tdim), 1.0 / (tdim + 1))
    sel = np.arange(cells.shape[0]) if cell_ids is None else np.asarray(cell_ids, dtype=np.int64)
    rv, rg = _tabulate(dofmap, ref_points)
    one = _readback_kernel(dofmap, pts, derivative)
    cells_j = jnp.asarray(cells[sel], dtype=jnp.int32)
    coeffs = jnp.asarray(u)[jnp.asarray(dofmap.cell_dofs[sel], dtype=jnp.int32)]
    sel_j = jnp.asarray(sel, dtype=jnp.int32)
    return jax.vmap(lambda c, cell, cf: one(c, cell, cf, rv, rg))(sel_j, cells_j, coeffs)


def nonnodal_field_at_points(points, cells, dofmap, u, X, *, derivative=None, tol=1e-9):
    """Evaluate a solved non-nodal field of any degree at PHYSICAL points ``X`` (``(n, tdim)``).

    Each point is located in a cell on the host (candidates from a KD-tree on the cell centroids,
    accepted by their barycentric coordinates within ``tol``, with a full scan as the fallback), then
    the assembler's basis is evaluated there -- so a point on a shared facet takes its value from one of
    the cells containing it (the conforming trace agrees; the other components of an H(curl)/H(div)
    field may jump, which is the physics). Raises for a point outside the mesh rather than
    extrapolating. Output shapes as :func:`nonnodal_field_at` without the cell axis: ``(n, ...)``.
    Differentiable in ``u``; the point location is host-side (``X`` is data)."""
    from scipy.spatial import cKDTree

    P = np.asarray(points)
    cells = np.asarray(cells, dtype=np.int64)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    tdim = cells.shape[1] - 1
    V = P[cells][:, :, :tdim]
    J = np.stack([V[:, k] - V[:, 0] for k in range(1, tdim + 1)], axis=2)
    Jinv = np.linalg.inv(J)
    _d, cand = cKDTree(V.mean(axis=1)).query(X, k=min(len(cells), 32))
    cand = np.asarray(cand).reshape(X.shape[0], -1)
    found_c = np.full(X.shape[0], -1, dtype=np.int64)
    found_xi = np.zeros((X.shape[0], tdim))
    for i in range(X.shape[0]):
        xi = np.einsum("cij,cj->ci", Jinv[cand[i]], X[i] - V[cand[i], 0])
        ok = np.flatnonzero((xi.min(axis=1) >= -tol) & (xi.sum(axis=1) <= 1.0 + tol))
        if ok.size:
            found_c[i], found_xi[i] = cand[i][ok[0]], xi[ok[0]]
            continue
        xi = np.einsum("cij,cj->ci", Jinv, X[i] - V[:, 0])
        ok = np.flatnonzero((xi.min(axis=1) >= -tol) & (xi.sum(axis=1) <= 1.0 + tol))
        if ok.size == 0:
            raise ValueError(f"nonnodal_field_at_points: point {X[i]} lies outside the mesh.")
        found_c[i], found_xi[i] = ok[0], xi[ok[0]]
    rv, rg = _tabulate(dofmap, found_xi)  # (n, n_dof, vs): one reference point per located point
    one = _readback_kernel(dofmap, jnp.asarray(P), derivative)
    coeffs = jnp.asarray(u)[jnp.asarray(dofmap.cell_dofs[found_c], dtype=jnp.int32)]
    out = jax.vmap(lambda c, cell, cf, v, g: one(c, cell, cf, v[None], g[None])[0])(
        jnp.asarray(found_c, dtype=jnp.int32), jnp.asarray(cells[found_c], dtype=jnp.int32), coeffs, rv, rg
    )
    return out
