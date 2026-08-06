"""Native 1D (segment) FEM assembly for ``jno.fem`` — ``LINE2`` (P1) and ``LINE3`` (P2).

Small native assembler for a 1D ``jno.domain.line(...)`` problem that reuses
jNO's weak-form integrand evaluator (:func:`_eval_integrand`) — it adds the
1D geometry (Lagrange shape functions + Gauss quadrature), an element loop with
a global scatter, and native boundary handling. Same matrices-only contract as
the 2D/3D path: it returns the assembled system, never a solve.

Assembly strategy (how an element matrix is formed):
- build a global residual ``R(u)`` by evaluating each weak term at the 1D
  quadrature points and scattering element contributions;
- boundary (Neumann/Robin) terms ride the *same* residual path as a degenerate
  one-node "element" (so a Robin ``a*u`` lands in the matrix and ``-g`` in the
  load by construction — no ad-hoc load patching);
- Dirichlet rows are replaced by ``u[d] - g`` so the tangent row is the identity
  and the load entry is ``g``;
- the operator is scattered from **per-element** Jacobians into a **BCOO**: a LINE2
  element couples only its own ``2*vec`` dofs, so differentiating the element
  residual w.r.t. that element's dofs gives the element matrix directly
  (``sparse_jacobian``). ``b = -R(0)``; nonlinear keeps ``R``/``jac`` as callables;
  transient separates the mass term.

Assembly is ``O(nnz)``, not ``O(N²)``. It previously recovered the global operator as
``jacfwd(R)(0)`` over the whole residual, whose ``(n_elem, n_dof, ...)`` intermediate
exhausted GPU memory at ~10k nodes — so 1D, the *cheapest* dimension, had the library's
lowest DOF ceiling while 2D/3D scattered sparsely. 50k nodes now assemble in ~2 s.

P2 (``order=2``) adds one dof per element **midpoint**, laid out after all vertices so a vertex dof
keeps its mesh-node index — which is why the boundary/Dirichlet lookup needs no P2 awareness (a 1D
boundary is an endpoint, hence always a vertex). Geometry stays LINE2: a straight element with a
centred midpoint has a constant Jacobian, so the linear map is exact, not an approximation.

Runtime parameters (``jno.np.parameter``, scalar or nodal field) thread through the element kernels
via ``local["volume_vars"]``, the same channel the 2D/3D and non-nodal assemblers use, so a **steady
linear** 1D form is parametric and differentiable in the parameter — a 1D inverse problem runs through
``crux.solve`` like any other. A **neural** (``jno.nn.wrap``) coefficient rides the same steady-linear
path by a different route: a weight pytree is cell-independent, so a network never enters
``volume_vars`` — the kernel re-evaluates it at the quad points from ``local["neural_coefficients"]``
while its weights arrive in ``args`` as a ``ModelWeights`` slot, which is what makes ∂solve/∂weights
flow. Both work on the **transient** path too (the operator and load re-form per step, so a 1D
time-series inverse trains), with one rule enforced: the transient MASS must be parameter-free, since
it is assembled once and a parameter there would be silently frozen. A **nonlinear** form is parametric
too: the residual re-evaluates its coefficients from ``args`` and ``FemResidualOperator`` takes
``R(u, args)``, so Newton runs on ``R(·, θ)`` and implicit differentiation supplies ∂u/∂θ. Not wired:
anything parametric on a **coupled** system (that block builder threads no parameters — it fails loud
rather than dropping the value, which is what it used to do).

Scope: scalar unknown (``vec == 1``), order 1 or 2, single field (the coupled block assembler is
P1 only). Boundary terms must
be value/Robin terms (``g*phi`` / ``(a*u - g)*phi``) — they carry no spatial
derivative, since a 0D facet has no element to differentiate over.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .fem_utils import (
    _eval_integrand,
    bcoo_set_dirichlet_rows,
    bcoo_set_unit_diag,
    bcoo_zero_rows,
    bcoo_zero_rows_cols,
    compress_eager,
)

_COMPONENT_NAMES = {"x": 0, "y": 1, "z": 2}


# --------------------------------------------------------------------------
# 1D reference element (LINE2) + quadrature
# --------------------------------------------------------------------------
def _line_quadrature(quad_degree: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Legendre points/weights on the reference interval [0, 1]."""
    n = max(1, quad_degree // 2 + 1)
    x, w = np.polynomial.legendre.leggauss(n)  # on [-1, 1]
    gp = 0.5 * (x + 1.0)
    gw = 0.5 * w
    return jnp.asarray(gp), jnp.asarray(gw)


def _line_shape(gp: jnp.ndarray, order: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Lagrange shape values ``N`` and reference gradients ``dN/dxi`` at ``gp`` on [0, 1].

    ``order=1`` (LINE2): nodes at ``0, 1`` — ``N = [1-xi, xi]``.
    ``order=2`` (LINE3): nodes at ``0, 1, 1/2`` — the endpoints FIRST, then the midpoint, so a
    global DOF layout of "all vertices, then all element midpoints" keeps every vertex dof at its
    mesh-node index (which is what lets the boundary/Dirichlet node lookup stay unchanged)::

        N_0 = (1-xi)(1-2xi),  N_1 = xi(2xi-1),  N_2 = 4 xi (1-xi)

    Returns ``N`` and ``dN_dxi``, each ``(n_quad, order+1)``."""
    one = jnp.ones_like(gp)
    if int(order) == 1:
        return jnp.stack([1.0 - gp, gp], axis=-1), jnp.stack([-one, one], axis=-1)
    if int(order) != 2:
        raise NotImplementedError(f"jno.fem: 1D Lagrange elements are implemented for order 1 and 2; got {order}.")
    N = jnp.stack([(1.0 - gp) * (1.0 - 2.0 * gp), gp * (2.0 * gp - 1.0), 4.0 * gp * (1.0 - gp)], axis=-1)
    dN_dxi = jnp.stack([4.0 * gp - 3.0, 4.0 * gp - 1.0, 4.0 - 8.0 * gp], axis=-1)
    return N, dN_dxi


# --------------------------------------------------------------------------
# boundary node lookup
# --------------------------------------------------------------------------
def _region_node_ids(domain: Any, region: str) -> List[int]:
    """Mesh node ids whose coordinates satisfy ``region``'s location predicate."""
    loc = domain._make_tag_location_fn(region)
    if loc is None:
        raise ValueError(f"jno.fem (1D): boundary region {region!r} has no location function.")
    pts = jnp.asarray(domain.mesh.points)
    n = int(pts.shape[0])
    num_args = loc.__code__.co_argcount if hasattr(loc, "__code__") else 1

    # Map over the points in CHUNKS, not all at once. A geometric region predicate
    # (`BoundaryRegion.contains`) tests ONE point against every boundary facet, so vmapping it over the
    # whole mesh materialises an (n_points x n_facets) intermediate -- on a realistic 3-D mesh that is
    # 11820 x 8502 f64 = 804 MB, which exhausts the device before the problem is even assembled.
    # Chunking bounds the peak at O(chunk x n_facets) and the result is identical (a pointwise predicate
    # does not couple points). This runs eagerly (the caller wants concrete ids), so the Python loop is free.
    chunk = 512

    def _eval(lo, hi):
        blk = pts[lo:hi]
        return jax.vmap(loc)(blk) if num_args == 1 else jax.vmap(loc)(blk, jnp.arange(lo, hi))

    if n <= chunk:
        hits = np.asarray(_eval(0, n)).reshape(-1)
    else:
        hits = np.concatenate([np.asarray(_eval(s, min(s + chunk, n))).reshape(-1) for s in range(0, n, chunk)])
    return list(np.where(hits)[0])


def _dirichlet_dofs(domain: Any, dirichlet_values: Dict[str, Any], vec: int) -> List[Tuple[int, float]]:
    """Resolve ``{region: value}`` into a list of ``(global_dof, value)`` pairs."""
    pts = np.asarray(domain.mesh.points)
    pairs: List[Tuple[int, float]] = []
    for region, value in dirichlet_values.items():
        for nid in _region_node_ids(domain, region):
            p = pts[nid]
            if isinstance(value, dict):
                for key, v in value.items():
                    comp = _COMPONENT_NAMES[key] if isinstance(key, str) else int(key)
                    g = float(v(p)) if callable(v) else float(v)
                    pairs.append((nid * vec + comp, g))
            else:
                g = float(value(p)) if callable(value) else float(value)
                for comp in range(vec):
                    pairs.append((nid * vec + comp, g))
    return pairs


# --------------------------------------------------------------------------
# per-term local evaluation (reuses the jNO integrand evaluator)
# --------------------------------------------------------------------------
def _integrate_term(domain: Any, expr: Any, local: dict, weights: jnp.ndarray) -> jnp.ndarray:
    """Evaluate one weak term on one element and integrate over its quad points.

    Mirrors ``_eval_volume_integrand``: ``sum_q val(q) * weight(q)`` flattened to
    the element's local DOF layout (node-major)."""
    val = _eval_integrand(domain, expr, local)
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    return ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]


def _param_context(volume_terms, boundary_terms):
    """``(tags, field_names, neural_slots, operator_exprs)`` for a 1D form.

    One collector for the steady and transient paths, using the same helpers as the 2D/3D and
    non-nodal assemblers so a parameter behaves identically across dimensions. ``operator_exprs`` is
    what the resulting ``FemLinearSystem`` / ``SemidiscreteTimeBlock`` carries: the scalar/field
    parameters plus any network's ``ModelWeights`` slot. Networks stay OUT of ``tags`` — a weight
    pytree is cell-independent, so it never enters the per-cell ``volume_vars``."""
    from .parametric_helpers import (
        _collect_runtime_parameter_exprs,
        _is_fem_field_parameter,
        collect_neural_slots,
        neural_operator_exprs,
    )

    exprs: Dict[str, Any] = {}
    for bare in list(volume_terms) + [t for ts in (boundary_terms or {}).values() for t in ts]:
        _collect_runtime_parameter_exprs(bare, exprs)
    tags: Tuple[str, ...] = tuple(sorted(exprs))
    fields = frozenset(n for n, e in exprs.items() if _is_fem_field_parameter(e))
    slots = collect_neural_slots(volume_terms, boundary_terms, runtime_parameter_tags=tags)
    return tags, fields, slots, neural_operator_exprs(exprs, slots)


def dof_layout_1d(domain: Any, order: int):
    """``(n_dof_nodes, dof_coords)`` for a 1D Lagrange space of ``order``.

    P1 dofs are the mesh vertices. P2 adds one dof per element **midpoint**, laid out after all
    vertices, so a vertex dof keeps its mesh-node index. ``dof_coords`` is what ``fem.points`` must
    report — the solution vector lives on these, not on the linear mesh the domain keeps."""
    verts = np.asarray(domain.mesh.points)[:, 0]
    if int(order) == 1:
        return int(verts.shape[0]), verts.reshape(-1, 1)
    cells = np.asarray(domain.mesh.cells_dict["line"])
    mids = 0.5 * (verts[cells[:, 0]] + verts[cells[:, 1]])
    return int(verts.shape[0] + cells.shape[0]), np.concatenate([verts, mids]).reshape(-1, 1)


def _make_residual(
    domain: Any,
    volume_terms: List[Any],
    boundary_terms: Dict[str, List[Any]],
    *,
    n_nodes: int,
    vec: int,
    quad_degree: int,
    order: int = 1,
    runtime_parameter_tags: Tuple[str, ...] = (),
    field_param_names: Any = frozenset(),
    neural_slots: Any = None,
) -> Any:
    """Build the *free* global residual ``R(u_flat) -> (n_dof*vec,)``.

    Covers the volume + boundary (Neumann/Robin) weak terms; Dirichlet is applied
    by the caller (symmetric elimination for the linear ``(A, b)``, row-replacement
    for the nonlinear residual).

    ``order`` selects LINE2 (P1) or LINE3 (P2). ``n_nodes`` is the caller's *vertex* count; the
    residual is sized by the element's dof count, which for P2 adds one midpoint dof per element
    (see :func:`n_dofs_1d`)."""
    verts = jnp.asarray(domain.mesh.points)[:, 0]
    cells = jnp.asarray(domain.mesh.cells_dict["line"], dtype=jnp.int32)  # (n_elem, 2) — the GEOMETRY
    n_vert, n_elem = int(verts.shape[0]), int(cells.shape[0])
    order = int(order)
    gp, gw = _line_quadrature(quad_degree)
    N, dN_dxi = _line_shape(gp, order)
    N_geom, _ = _line_shape(gp, 1)  # geometry stays LINE2: a straight element with a centred
    # midpoint has a constant Jacobian h, so the linear map is EXACT — subparametric, not an
    # approximation, and it keeps `h` a scalar per element.
    ctx = getattr(domain, "context", {}) or {}
    comp_offsets = jnp.arange(vec)

    # P2 adds one dof per element midpoint. Layout: ALL vertices, then all midpoints — so a vertex
    # dof keeps its mesh-node index and the boundary/Dirichlet node lookup needs no P2 awareness
    # (a 1D boundary is an endpoint, hence always a vertex).
    if order == 2:
        elem_nodes = jnp.concatenate([cells, (n_vert + jnp.arange(n_elem, dtype=jnp.int32))[:, None]], axis=1)
        n_nodes_dof = n_vert + n_elem
    else:
        elem_nodes = cells
        n_nodes_dof = n_vert
    n_nodes = n_nodes_dof  # the residual is sized by DOF nodes, not mesh vertices

    # element-local -> global DOF map, node-major: dof = node*vec + comp
    cell_dofs = (elem_nodes[:, :, None] * vec + comp_offsets[None, None, :]).reshape(n_elem, -1)  # (n_elem, nen*vec)

    # precompute boundary (region, term, node_id) triples
    boundary_apps: List[Tuple[Any, int]] = []
    for region, terms in boundary_terms.items():
        nids = _region_node_ids(domain, region)
        for term in terms:
            for nid in nids:
                boundary_apps.append((term, int(nid)))

    def _pack_params(node_ids, args, dtype, width_default=1):
        """This element's runtime-parameter values, in ``runtime_parameter_tags`` order.

        The evaluator reads them out of ``volume_vars`` at ``[temporal..., runtime_param...]``: a length-1
        entry is a **scalar** coefficient (broadcast to the quad points), a length-``n_local`` entry is a
        nodal **field** coefficient which it interpolates with this element's own ``shape_vals``. A tag the
        current assembly does not supply gets a zero placeholder of the right width — it is only ever read
        back when the term actually contains that node, in which case ``args`` carries it."""
        a = args or {}
        out = []
        for name in runtime_parameter_tags:
            is_field = name in field_param_names
            if name not in a:
                out.append(jnp.zeros((node_ids.shape[0] if is_field else width_default,), dtype))
                continue
            flat = jnp.reshape(jnp.asarray(a[name], dtype=dtype), (-1,))
            out.append(flat[node_ids] if is_field else flat[:1])
        return tuple(out)

    def _neural_table(args):
        """``local['neural_coefficients']`` for this call: the ``{name: module}`` table the evaluator
        re-evaluates each network from at the quad points. Unlike a scalar/field parameter a network
        never enters ``volume_vars`` (a weight pytree is cell-independent) — trainable weights arrive
        through ``args``, a frozen net falls back to its stored module."""
        if neural_slots is None:
            return None
        from .parametric_helpers import neural_local_table

        return neural_local_table(neural_slots, args)

    def _volume_local(cell, u_local, expr, pvals=(), ntable=None):
        """Element residual ``(2*vec,)`` from the element's OWN dofs ``u_local`` ``(2, vec)``.

        Taking the local dofs rather than indexing a global vector is what makes the element
        Jacobian available: differentiating this w.r.t. ``u_local`` gives the ``(2*vec, 2*vec)``
        element matrix directly, so the global operator can be scattered sparsely instead of
        recovered by an ``O(N²)`` ``jacfwd`` of the global residual (see :func:`_sparse_jacobian`)."""
        xc = verts[cell[:2]]  # (2,) — the element's endpoints carry the geometry
        h = xc[1] - xc[0]
        phys = (N_geom @ xc)[:, None]  # (n_quad, 1)
        shape_grads = (dN_dxi / h)[:, :, None]  # (n_quad, nen, 1)
        local = {
            "physical_quad_points": phys,
            "shape_vals": N,
            "shape_grads": shape_grads,
            "cell_sol": u_local,  # (2, vec)
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": ctx,
            "trial_value_shape": (),
            "trial_vec": vec,
            "temporal_tags": (),
            "runtime_parameter_tags": runtime_parameter_tags,
            "volume_vars": pvals,
            "neural_coefficients": ntable,
        }
        return _integrate_term(domain, expr, local, gw * h)  # (nen*vec,)

    def residual(u_flat, args=None):
        u = u_flat.reshape(n_nodes, vec)
        _nt = _neural_table(args)
        R = jnp.zeros(n_nodes * vec, dtype=u_flat.dtype)

        # --- volume terms: vmap over elements, then scatter ---
        for expr in volume_terms:
            elem_res = jax.vmap(lambda c, e=expr: _volume_local(c, u[c], e, _pack_params(c, args, u_flat.dtype), _nt))(
                elem_nodes
            )  # (n_elem, nen*vec)
            R = R.at[cell_dofs.reshape(-1)].add(elem_res.reshape(-1))

        # --- boundary terms: degenerate one-node "element" (shape val 1, weight 1) ---
        for expr, nid in boundary_apps:
            _bp = _pack_params(jnp.asarray([nid]), args, u_flat.dtype)
            contrib = _boundary_local(nid, u[nid : nid + 1], expr, u_flat.dtype, _bp, _nt)  # (vec,)
            R = R.at[nid * vec + comp_offsets].add(contrib.reshape(vec))

        return R

    def _boundary_local(nid, u_local, expr, dtype, pvals=(), ntable=None):
        """Boundary contribution ``(vec,)`` at node ``nid`` from its own dof ``u_local`` ``(1, vec)``."""
        local = {
            "physical_quad_points": jnp.asarray([[verts[nid]]]),
            "shape_vals": jnp.ones((1, 1), dtype=dtype),
            "shape_grads": jnp.zeros((1, 1, 1), dtype=dtype),
            "cell_sol": u_local,  # (1, vec)
            "tag": "fem_gauss",
            "surface": True,
            "domain_context": ctx,
            "trial_value_shape": (),
            "trial_vec": vec,
            "temporal_tags": (),
            "runtime_parameter_tags": runtime_parameter_tags,
            "volume_vars": pvals,
            "neural_coefficients": ntable,
        }
        return _integrate_term(domain, expr, local, jnp.ones((1,), dtype=dtype))  # (vec,)

    def sparse_jacobian(u_flat, args=None):
        """``dR/du`` as a **BCOO**, scattered from per-element blocks — never an ``O(N²)`` dense array.

        Each LINE2 element couples only its own ``2*vec`` dofs, so its Jacobian is a ``(2*vec, 2*vec)``
        block obtained by differentiating :func:`_volume_local` w.r.t. that element's dofs (vmapped over
        elements); a boundary term couples one node to itself, a ``(vec, vec)`` block. Emitting those
        blocks as triplets is the same element-scatter the 2D/3D native assembler does, and it is what
        lets a 1D problem scale past the few-thousand-node ceiling that a global ``jacfwd`` imposed
        (its ``(n_elem, n_dof, ...)`` intermediate exhausted GPU memory at ~10k nodes).
        """
        from jax.experimental import sparse as jsp

        u = u_flat.reshape(n_nodes, vec)
        _nt = _neural_table(args)
        idx, dat = [], []

        for expr in volume_terms:
            # (n_elem, 2*vec, 2, vec) -> (n_elem, 2*vec, 2*vec): the element matrices
            Ke = jax.vmap(
                lambda c, e=expr: jax.jacfwd(lambda ul: _volume_local(c, ul, e, _pack_params(c, args, u_flat.dtype), _nt))(
                    u[c]
                )
            )(elem_nodes)
            nen = elem_nodes.shape[1]
            Ke = Ke.reshape(n_elem, nen * vec, nen * vec)
            rows = jnp.broadcast_to(cell_dofs[:, :, None], Ke.shape)
            cols = jnp.broadcast_to(cell_dofs[:, None, :], Ke.shape)
            idx.append(jnp.stack([rows.reshape(-1), cols.reshape(-1)], axis=1))
            dat.append(Ke.reshape(-1))

        for expr, nid in boundary_apps:
            _bp = _pack_params(jnp.asarray([nid]), args, u_flat.dtype)
            Kb = jax.jacfwd(lambda ul: _boundary_local(nid, ul, expr, u_flat.dtype, _bp, _nt))(u[nid : nid + 1])
            Kb = Kb.reshape(vec, vec)
            dofs = nid * vec + comp_offsets
            rows = jnp.broadcast_to(dofs[:, None], (vec, vec))
            cols = jnp.broadcast_to(dofs[None, :], (vec, vec))
            idx.append(jnp.stack([rows.reshape(-1), cols.reshape(-1)], axis=1))
            dat.append(Kb.reshape(-1))

        nd = n_nodes * vec
        if not idx:
            return jsp.BCOO((jnp.zeros((0,), u_flat.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(nd, nd))
        # duplicate (i, j) entries from shared nodes are summed by BCOO on matvec / todense
        return jsp.BCOO((jnp.concatenate(dat), jnp.concatenate(idx).astype(jnp.int32)), shape=(nd, nd))

    residual.sparse_jacobian = sparse_jacobian
    return residual


def _apply_dirichlet_symmetric(A, b, dirichlet_pairs: List[Tuple[int, float]]):
    """Symmetric Dirichlet elimination on a linear system ``A u = b``.

    Moves known columns to the RHS, then zeros the constrained rows *and* columns
    and sets a unit diagonal — so ``A`` stays symmetric (as in the 2D/3D
    path), unlike a row-only replacement."""
    if not dirichlet_pairs:
        return A, b
    dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32)
    vals = jnp.asarray([p[1] for p in dirichlet_pairs], dtype=b.dtype)
    if hasattr(A, "indices"):  # BCOO (native 2D/3D assembler) — keep it sparse, never densify
        e = jnp.zeros(A.shape[0], b.dtype).at[dofs].set(vals)  # the known-column lift
        b = b - A @ e  # carry the known columns to the load (a BCOO matvec, no dense column slice)
        A = bcoo_set_unit_diag(bcoo_zero_rows_cols(A, dofs), dofs)
        b = b.at[dofs].set(vals)
        return A, b
    b = b - A[:, dofs] @ vals  # carry the known columns to the load
    A = A.at[dofs, :].set(0.0).at[:, dofs].set(0.0).at[dofs, dofs].set(1.0)
    b = b.at[dofs].set(vals)
    return A, b


def _apply_dirichlet_rows(residual_free, dirichlet_pairs: List[Tuple[int, float]]):
    """Wrap a free residual so Dirichlet rows read ``u[d] - g`` (row-replacement).

    Used for the nonlinear residual: the tangent row becomes the identity and the
    Newton step drives ``u[d] -> g``.

    ``residual_free`` takes the state ALONE. That single-argument contract is deliberate: this helper
    is shared with the non-nodal and native assemblers, whose free residuals have *different* trailing
    signatures (``R(u, args)`` in one, ``R(u, t=0.0, args=None)`` in the other) — so a wrapper that
    forwarded a second positional would bind it to ``t`` in the native case and silently evaluate the
    form at the wrong time rather than fail. A caller that needs runtime args closes over them and
    wraps per call, ``_apply_dirichlet_rows(lambda uu: free(uu, args), pins)(u)``, which is what the
    parametric paths here and in ``fem_nonnodal`` do."""
    if not dirichlet_pairs:
        return residual_free
    dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32)
    vals = jnp.asarray([p[1] for p in dirichlet_pairs])

    def residual(u_flat):
        R = residual_free(u_flat)
        return R.at[dofs].set(u_flat[dofs] - vals.astype(u_flat.dtype))

    return residual


# --------------------------------------------------------------------------
# public entry: assemble a 1D problem into (op, mode) for the FEM container
# --------------------------------------------------------------------------
def assemble_fem_1d(
    domain: Any,
    volume_terms: List[Any],
    boundary_terms: Dict[str, List[Any]],
    dirichlet_values: Dict[str, Any],
    ic_residuals: List[Any],
    *,
    vec: int,
    quad_degree: int,
    order: int = 1,
) -> Tuple[Any, str]:
    """Assemble a 1D (``LINE2``) weak form into ``(op, mode)`` for :class:`FEM`.

    ``mode`` is ``"linear"`` (``op = (A, b)``), ``"nonlinear"`` (``op`` a
    :class:`FemResidualOperator`), or ``"transient"`` (``op`` a ``SemidiscreteTimeBlock``).
    """
    from ...trace import FemLinearSystem, FemResidualOperator
    from .weak_form import _contains_temporal_derivative, _is_obviously_nonlinear_in_unknown

    if int(vec) != 1:
        raise NotImplementedError(f"jno.fem: 1D (LINE2) assembly supports a scalar unknown (vec=1) only; got vec={vec}.")

    # ---- runtime parameters: the differentiable-inverse path. A `jno.np.parameter` in the form makes
    # the system PARAMETRIC -- the operator and load are re-formed from the runtime args on every call
    # and stay differentiable in them, so `crux.solve` can recover the parameter from data. Collected
    # with the same helpers the 2D/3D and non-nodal assemblers use, so scalar and nodal-field
    # parameters behave identically across dimensions. ----
    runtime_parameter_tags, field_param_names, _neural, _param_and_neural_exprs = _param_context(
        volume_terms, boundary_terms
    )

    # Neural coefficients (``jno.nn.wrap(net)`` inside the weak form). They deliberately stay OUT of
    # ``runtime_parameter_tags``: a weight pytree is cell-independent, so unlike a scalar/field
    # parameter a network never enters the per-cell ``volume_vars`` — the kernel re-evaluates it at the
    # quad points from the ``{name: module}`` table instead. Its weights still ride ``args`` as a
    # ``ModelWeights`` slot, which is what makes ∂solve/∂weights flow. Same three touch-points as the
    # native and non-nodal assemblers (collect / merge into the operator exprs / per-call table).
    # NOTE: a nodal field parameter is interpolated with the element's own shape functions, so its nodal
    # layout must match the trial's — a P1 parameter field cannot ride a LINE3 element. No guard is needed
    # here: `jno.np.parameter(<symbol>)` already refuses a non-P1 symbol at construction, which is the
    # earlier and better place to catch it.

    # P2 adds a midpoint dof per element, so the dof count is NOT the vertex count. Publishing the
    # dof coordinates is what makes `fem.points` line up with the solution vector (the same
    # contract the 2D/3D P2 path already has).
    n_nodes, dof_coords = dof_layout_1d(domain, order)
    ndof = n_nodes * vec
    domain._fem_native_dof_points = dof_coords
    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]

    if any(_contains_temporal_derivative(t) for t in all_terms):
        return _assemble_1d_transient(
            domain,
            volume_terms,
            boundary_terms,
            dirichlet_values,
            ic_residuals,
            vec=vec,
            quad_degree=quad_degree,
            order=order,
        )

    if ic_residuals:
        raise ValueError("jno.fem: an initial condition was given but the 1D weak form has no time derivative.")

    dirichlet_pairs = _dirichlet_dofs(domain, dirichlet_values, vec)
    residual_free = _make_residual(
        domain,
        volume_terms,
        boundary_terms,
        n_nodes=n_nodes,
        vec=vec,
        quad_degree=quad_degree,
        order=order,
        runtime_parameter_tags=runtime_parameter_tags,
        field_param_names=field_param_names,
        neural_slots=_neural,
    )

    nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms)
    if nonlinear:
        # The residual already re-evaluates its coefficients from ``args`` (that is what made the linear
        # path parametric), and ``FemResidualOperator`` takes ``R(u, args)`` — so a parameter or network
        # in a NONLINEAR form threads with no extra machinery: Newton runs on R(·, θ) and the implicit
        # derivative gives ∂u/∂θ. NOTE the jacobian_fn stays a dense global jacfwd; the matrix-free
        # Newton-Krylov default never calls it, so it is a latent cost rather than an active one.
        def res_p(u, args=None):
            return _apply_dirichlet_rows(lambda uu: residual_free(uu, args), dirichlet_pairs)(jnp.asarray(u))

        op = FemResidualOperator(
            residual_fn=res_p,
            jacobian_fn=lambda u, args=None: jax.jacfwd(lambda uu: res_p(uu, args))(jnp.asarray(u)),
            size=ndof,
            runtime_parameter_exprs=_param_and_neural_exprs,
        )
        return op, "nonlinear"

    # linear: R(u) = A u - b  ->  A = dR/du, b = -R(0), then symmetric Dirichlet.
    # A is scattered from per-element blocks (BCOO), not recovered by a global jacfwd — see
    # `sparse_jacobian`. Memory is O(nnz) instead of O(N²), which is what lets a 1D problem run
    # at the node counts 1D actually wants.
    zeros = jnp.zeros(ndof)

    def _system(args=None):
        A = residual_free.sparse_jacobian(zeros, args)
        b = -residual_free(zeros, args)
        return _apply_dirichlet_symmetric(A, b, dirichlet_pairs)

    A0, b0 = _system(None)
    if not _param_and_neural_exprs:
        return (compress_eager(A0), b0), "linear"
    # Parametric: re-form A(θ), b(θ) from the runtime args per call, so ∂u/∂θ flows through the solve.
    # Dirichlet elimination couples A and b (known columns move to the load), so each accessor assembles
    # the pair and takes its half — correct by construction, at the cost of assembling twice per call.
    return (
        FemLinearSystem(
            A0,
            b0,
            operator_fn=lambda args: _system(args)[0],
            rhs_fn=lambda args: _system(args)[1],
            runtime_parameter_exprs=_param_and_neural_exprs,
        ),
        "linear",
    )


def _apply_dirichlet_transient(M, A, c, dirichlet_pairs: List[Tuple[int, float]]):
    """Apply Dirichlet to a semidiscrete linear block ``M u̇ + A u = c``.

    ``A``/``c`` get symmetric elimination (so ``A`` stays symmetric); ``M`` has the
    constrained rows *and* columns zeroed (no diagonal) so the constrained DOFs
    carry no time derivative and the row equation reduces to ``u[d] = g``."""
    A, c = _apply_dirichlet_symmetric(A, c, dirichlet_pairs)
    if dirichlet_pairs:
        dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32)
        M = bcoo_zero_rows_cols(M, dofs) if hasattr(M, "indices") else M.at[dofs, :].set(0.0).at[:, dofs].set(0.0)
    return M, A, c


def _assemble_1d_transient(
    domain, volume_terms, boundary_terms, dirichlet_values, ic_residuals, *, vec, quad_degree, order=1
):
    """Assemble a first-order transient 1D weak form into a ``SemidiscreteTimeBlock``.

    Splits the volume terms into a temporal part (``u_t``) and a spatial part: the
    temporal part becomes the mass matrix ``M`` (rewriting ``u_t`` -> ``u`` via
    ``_strip_temporal_trial_derivative``); the spatial part + boundary terms become
    the operator. Linear -> ``M``/``A`` payload; nonlinear -> ``mass``/``residual``/
    ``jacobian`` callables."""
    from ..._fem import _bare, _ic_value_at_nodes
    from .backend_blocks import SemidiscreteTimeBlock
    from .solver_helper import max_temporal_derivative_order as _mto
    from .time_route import _infer_time_window, _strip_temporal_trial_derivative
    from .weak_form import (
        _apply_sign,
        _contains_temporal_derivative,
        _is_obviously_nonlinear_in_unknown,
        _split_additive_terms,
    )

    # P2 adds a midpoint dof per element, so the dof count is NOT the vertex count. Publishing the
    # dof coordinates is what makes `fem.points` line up with the solution vector (the same
    # contract the 2D/3D P2 path already has).
    n_nodes, dof_coords = dof_layout_1d(domain, order)
    ndof = n_nodes * vec
    domain._fem_native_dof_points = dof_coords
    _tags, _fields, _neural, _param_exprs = _param_context(volume_terms, boundary_terms)

    # Split each weak constraint into additive sub-terms first, so the temporal
    # term (u_t * phi) can be separated from the spatial terms in the same sum.
    sub_terms = [
        _apply_sign(domain, sign, sub) for bare in volume_terms for sign, sub in _split_additive_terms(domain, bare)
    ]
    temporal_terms = [t for t in sub_terms if _contains_temporal_derivative(t)]
    spatial_terms = [t for t in sub_terms if not _contains_temporal_derivative(t)]
    if not temporal_terms:
        raise ValueError("jno.fem: a transient 1D weak form must contain a temporal-derivative term (u_t * phi).")

    # === second-order-in-time (u_tt): the augmented first-order block y = [u; v], v = u̇, integrated
    #     by the trapezoidal (θ=½) rule — the 1D analogue of the native _assemble_second_order_time. ===
    if max((_mto(t) for t in temporal_terms), default=1) >= 2:
        # A nonlinear SPATIAL operator (sine-Gordon, cubic Klein-Gordon, a nonlinear string) is carried by
        # the residual/jacobian form of the same augmented block, exactly as the native path does — see
        # the ``nonlinear_spatial`` branch below. The temporal side must stay linear either way: M2 and C
        # are matrices, so a state-dependent inertia or damping has no representation here.
        nonlinear_spatial = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in spatial_terms)
        # The TEMPORAL side must stay linear whatever the spatial side does: M2 and C are extracted as
        # matrices via ``sparse_jacobian(0)``, so a state-dependent inertia or damping would be silently
        # linearised about u=0 and integrated as a constant — a wrong answer with no error. Checked
        # unconditionally, NOT only when the spatial part is nonlinear, or the linear-spatial case keeps
        # that silent path open.
        if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in temporal_terms):
            raise NotImplementedError(
                "jno.fem: a state-dependent MASS or DAMPING on a second-order-in-time 1D form is not "
                "supported — the augmented block carries M2 and C as matrices, so a c(u) there would be "
                "frozen at its u=0 value. A nonlinear SPATIAL operator (u_tt + N(u) = f) IS supported."
            )

        def _strip_n(t, k):
            for _ in range(k):
                t = _strip_temporal_trial_derivative(t)
            return t

        z = jnp.zeros(ndof)
        m2_terms = [_strip_n(t, 2) for t in temporal_terms if _mto(t) >= 2]  # u_tt·φ ⇒ mass M2
        d_terms = [_strip_n(t, 1) for t in temporal_terms if _mto(t) == 1]  # u_t·φ ⇒ damping C (optional)
        M2 = _make_residual(
            domain, m2_terms, {}, n_nodes=n_nodes, vec=vec, quad_degree=quad_degree, order=order
        ).sparse_jacobian(z)
        from jax.experimental import sparse as _jsp

        Cmat = (
            _make_residual(
                domain, d_terms, {}, n_nodes=n_nodes, vec=vec, quad_degree=quad_degree, order=order
            ).sparse_jacobian(z)
            if d_terms
            # an absent damping term is the EMPTY BCOO, not a dense zero block — a dense (ndof, ndof)
            # of zeros would reintroduce the O(N²) allocation this whole path exists to avoid
            else _jsp.BCOO((jnp.zeros((0,), z.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(ndof, ndof))
        )
        spatial_res2 = _make_residual(
            domain, spatial_terms, boundary_terms, n_nodes=n_nodes, vec=vec, quad_degree=quad_degree, order=order
        )
        # Compose the augmented 2n system SPARSELY. M2/C/K are BCOO (per-element scatter), so ``jnp.block``
        # and ``.at[]`` are both unavailable here — the former cannot mix sparse with dense at all, and
        # BCOO has no ``.at[]``. Reuse the same three helpers the native u_tt path uses, so the 1D and
        # native augmented systems are assembled and constrained by identical code.
        #   [M2  0 ] [u']   [ 0   -M2] [u]   [0]
        #   [0   M2] [v'] + [ K    C ] [v] = [F]
        from ..._fem import _bcoo_block

        n = ndof
        M_aug = _bcoo_block([(M2, 0, 0, 1.0), (M2, n, n, 1.0)], (2 * n, 2 * n), z.dtype)
        dpairs = _dirichlet_dofs(domain, dirichlet_values, vec)
        dd = dg = aug_d = None
        if dpairs:  # u[d]=g (constant) on displacement rows, v[d]=0 on velocity rows
            dd = jnp.asarray([p[0] for p in dpairs], dtype=jnp.int32)
            dg = jnp.asarray([p[1] for p in dpairs], dtype=z.dtype)
            aug_d = jnp.concatenate([dd, dd + n])  # the constrained rows in BOTH blocks
            M_aug = bcoo_zero_rows(M_aug, aug_d)  # mass rows are algebraic -> zeroed
        # the initial state lives on the DOF nodes, which for P2 include the element midpoints
        pts = jnp.asarray(dof_coords)
        u0 = jnp.zeros((n,), z.dtype)
        v0 = jnp.zeros((n,), z.dtype)
        for ic in ic_residuals:  # displacement IC u(0)=u0 + optional velocity IC u̇(0)=v0
            val = jnp.asarray(_ic_value_at_nodes(_bare(ic), domain, pts, ndof, vec), z.dtype)
            if _mto(_bare(ic)) >= 1:
                v0 = val
            else:
                u0 = val
        if dpairs:
            u0, v0 = u0.at[dd].set(dg), v0.at[dd].set(0.0)
        t0, t1, dt = _infer_time_window(domain)
        common2 = dict(
            backend="transient",
            mode="implicit",
            time_order=2,
            spatial_kind="weak_form",
            state0=jnp.concatenate([u0, v0]),
            t0=t0,
            t1=t1,
            dt=dt,
            eval_context=getattr(domain, "_fem_eval_context", {}) or {},
            metadata={"theta": 0.5, "second_order": True},
        )
        if nonlinear_spatial:
            # Newton on the augmented residual ``M_aug ẏ + R_aug(y) = 0`` with
            # ``R_aug(y) = [−M2 v ; N(u) + C v]``, ``N(u) = S(u) − F`` the 1D nonlinear spatial residual.
            # Identical in shape to the native u_tt nonlinear path, so the two cannot drift. The θ=½
            # stepper still applies, which is what keeps an undamped nonlinear wave from bleeding energy.
            def _residual_aug(y, t=0.0, args=None):
                y = jnp.asarray(y).reshape(-1)
                u_, v_ = y[:n], y[n:]
                r = jnp.concatenate([-(M2 @ v_), jnp.asarray(spatial_res2(u_, args)).reshape(-1) + (Cmat @ v_)])
                if dpairs:  # u[d] = g on the displacement rows, v[d] = 0 on the velocity rows
                    r = r.at[dd].set(u_[dd] - dg).at[dd + n].set(v_[dd])
                return r

            def _jacobian_aug(y, t=0.0, args=None):
                # ∂N/∂u at the CURRENT state, scattered per element (never a global dense jacfwd), then
                # composed into the augmented block by the same helper the linear branch uses.
                jn = spatial_res2.sparse_jacobian(jnp.asarray(y).reshape(-1)[:n], args)
                A = _bcoo_block([(M2, 0, n, -1.0), (jn, n, 0, 1.0), (Cmat, n, n, 1.0)], (2 * n, 2 * n), z.dtype)
                return bcoo_set_dirichlet_rows(A, aug_d) if dpairs else A

            block = SemidiscreteTimeBlock(
                mass=lambda t, args=None, _M=M_aug: _M,
                residual=_residual_aug,
                jacobian=_jacobian_aug,
                **common2,
            )
            return block, "transient"

        K = spatial_res2.sparse_jacobian(z)
        F = -spatial_res2(z)  # spatial load + natural-BC load
        A_aug = _bcoo_block([(M2, 0, n, -1.0), (K, n, 0, 1.0), (Cmat, n, n, 1.0)], (2 * n, 2 * n), z.dtype)
        c_aug = jnp.concatenate([jnp.zeros((n,), z.dtype), F])
        if dpairs:
            A_aug = bcoo_set_dirichlet_rows(A_aug, aug_d)  # -> identity rows (columns kept)
            c_aug = c_aug.at[dd].set(dg).at[dd + n].set(0.0)
        block = SemidiscreteTimeBlock(M=M_aug, A=A_aug, affine_bias=c_aug, **common2)
        return block, "transient"

    mass_terms = [_strip_temporal_trial_derivative(t) for t in temporal_terms]

    dirichlet_pairs = _dirichlet_dofs(domain, dirichlet_values, vec)
    if ic_residuals:
        # the initial state lives on the DOF nodes, which for P2 include the element midpoints
        pts = jnp.asarray(dof_coords)
        state0 = _ic_value_at_nodes(_bare(ic_residuals[0]), domain, pts, ndof, vec)
    else:
        state0 = jnp.zeros(ndof)
    t0, t1, dt = _infer_time_window(domain)

    # The MASS must be parameter-free. It is assembled once, outside the per-args re-forming, so a
    # parameter sitting on `u_t * phi` would be read at its zero placeholder and silently baked in --
    # a wrong answer with no error. Same rule the 2D/3D path documents; here it is enforced.
    _mass_tags, _, _mass_slots, _ = _param_context(mass_terms, {})
    if _mass_tags or _mass_slots.all_names:
        raise NotImplementedError(
            "jno.fem: a runtime parameter or neural coefficient on the 1D transient MASS term "
            f"({sorted(set(_mass_tags) | set(_mass_slots.all_names))}) is not supported — the mass is "
            "assembled once, so a parameter there would be silently frozen. Put the parameter on the "
            "spatial operator or the load instead."
        )
    mass_res = _make_residual(domain, mass_terms, {}, n_nodes=n_nodes, vec=vec, quad_degree=quad_degree, order=order)
    spatial_res = _make_residual(
        domain,
        spatial_terms,
        boundary_terms,
        n_nodes=n_nodes,
        vec=vec,
        quad_degree=quad_degree,
        order=order,
        runtime_parameter_tags=_tags,
        field_param_names=_fields,
        neural_slots=_neural,
    )

    all_terms = spatial_terms + [t for ts in boundary_terms.values() for t in ts]
    nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms)

    M = mass_res.sparse_jacobian(jnp.zeros(ndof))

    common = dict(
        backend="transient",
        mode="implicit",
        time_order=1,
        spatial_kind="weak_form",
        state0=state0,
        t0=t0,
        t1=t1,
        dt=dt,
        eval_context=getattr(domain, "_fem_eval_context", {}) or {},
    )

    if nonlinear:
        # M(t) u̇ + R(u, args) = 0 with R the (Dirichlet-enforced) spatial residual, re-evaluated at the
        # runtime args each step — so a parameter threads a nonlinear transient too (the mass stays
        # parameter-free, guarded above).
        def res_pt(u, t, args=None):
            return _apply_dirichlet_rows(lambda uu: spatial_res(uu, args), dirichlet_pairs)(jnp.asarray(u))

        dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
        # the mass is a BCOO now, which has no `.at[]` — zero the constrained rows/cols sparsely
        # (mirrors the linear branch); a dense fallback keeps the old indexing path.
        if dofs is None:
            M_nl = M
        elif hasattr(M, "indices"):
            M_nl = bcoo_zero_rows_cols(M, dofs)
        else:
            M_nl = M.at[dofs, :].set(0.0).at[:, dofs].set(0.0)
        block = SemidiscreteTimeBlock(
            mass=lambda t, args=None, _M=compress_eager(M_nl): _M,
            residual=res_pt,
            jacobian=lambda u, t, args=None: jax.jacfwd(lambda uu: res_pt(uu, t, args))(jnp.asarray(u)),
            runtime_parameter_exprs=_param_exprs,
            **common,
        )
        return block, "transient"

    # linear: M u̇ + A u = c
    def _lin_sys(args=None):
        A = spatial_res.sparse_jacobian(jnp.zeros(ndof), args)
        c = -spatial_res(jnp.zeros(ndof), args)
        return _apply_dirichlet_transient(M, A, c, dirichlet_pairs)

    M0, A0, c0 = _lin_sys(None)
    M0 = compress_eager(M0)  # parameter-free (guarded above), so this holds on both branches
    if not _param_exprs:
        return SemidiscreteTimeBlock(M=M0, A=compress_eager(A0), affine_bias=c0, **common), "transient"
    # Parametric transient: the operator and load re-form from the runtime args at every step, so
    # ∂traj/∂θ flows through the marcher -- a 1D time-series inverse (recover a diffusivity from
    # u(x, t)) trains through `crux.solve` like the steady one. The mass is parameter-free (guarded
    # above), so it stays the statically assembled M.
    block = SemidiscreteTimeBlock(
        M=M0,
        operator_fn=lambda t, args=None: _lin_sys(args)[1],
        forcing_vector_fn=lambda t, args=None: _lin_sys(args)[2],
        runtime_parameter_exprs=_param_exprs,
        **common,
    )
    return block, "transient"


# ==========================================================================
# coupled / mixed multi-field 1D (block native assembly)
# ==========================================================================
# The 1D analogue of the 2D/3D block path: hand-build the block residual for
# LINE2 elements. Each field is a scalar/vector unknown on the
# shared LINE2 mesh; the global DOF vector is laid out by field block —
# ``offset[i] = sum_{j<i} n_nodes * vec_j`` — so the user slices the solution
# exactly as in 2D/3D coupled.


def _block_offsets(fields: List[Any], n_nodes: int) -> List[int]:
    """Per-field block start offsets into the flat DOF vector (cumulative size)."""
    offs = [0]
    for f in fields:
        offs.append(offs[-1] + n_nodes * int(f["vec"]))
    return offs


def _multifield_dirichlet_dofs_1d(domain, dirichlet_raw, fields, field_index, n_nodes):
    """``dirichlet_raw`` ``(field_key, region, comp, value, value_node)`` -> block ``(dof, g)`` pairs.

    Mirrors :func:`_dirichlet_dofs` per field but offsets each DOF into the field's
    block; ``value`` is a constant or a ``value(point)`` callable (so coordinate-
    dependent Dirichlet such as ``u(xb) - xb`` works). ``value_node`` (the raw trace
    node) is consumed only by the 2D/3D time-varying path; the 1D path handles
    coordinate dependence through ``value`` and ignores it."""
    pts = np.asarray(domain.mesh.points)
    offs = _block_offsets(fields, n_nodes)
    pairs: List[Tuple[int, float]] = []
    for field_key, region, comp, value, _value_node in dirichlet_raw:
        fidx = field_index.get(field_key)
        if fidx is None:
            continue
        vt = int(fields[fidx]["vec"])
        for nid in _region_node_ids(domain, region):
            p = pts[nid]
            g = float(value(p)) if callable(value) else float(value)
            comps = range(vt) if comp is None else [comp]
            for c in comps:
                pairs.append((offs[fidx] + nid * vt + c, g))
    return pairs


def _make_multifield_residual_1d(domain, term_list, boundary_term_list, fields, field_index, *, n_nodes, quad_degree):
    """Free block residual ``R(u_flat) -> (sum_i n_nodes*vec_i,)`` for coupled 1D.

    The native analogue of ``_eval_multifield_volume_integrand``: per element it builds
    one ``local`` with per-field shape data (``local["fields"]``) and evaluates each
    ``(coeff, test_field_index)`` via the shared ``_eval_integrand``, scattering the
    element residual into the **test field's** block DOFs. Boundary (Neumann/Robin)
    terms ride the degenerate one-node element, same as the single-field path."""
    nodes = jnp.asarray(domain.mesh.points)[:, 0]
    cells = jnp.asarray(domain.mesh.cells_dict["line"], dtype=jnp.int32)  # (n_elem, 2)
    gp, gw = _line_quadrature(quad_degree)
    N, dN_dxi = _line_shape(gp)
    ctx = getattr(domain, "context", {}) or {}
    nfields = len(fields)
    vecs = [int(f["vec"]) for f in fields]
    offs = _block_offsets(fields, n_nodes)
    total = offs[-1]

    # element-local -> global block DOF map per field (node-major, then block offset)
    cell_dofs = []
    for i in range(nfields):
        comp = jnp.arange(vecs[i])
        cd = (cells[:, :, None] * vecs[i] + comp[None, None, :]).reshape(cells.shape[0], -1) + offs[i]
        cell_dofs.append(cd)  # (n_elem, 2*vec_i)

    def _elem_local(cell, locs, e):
        """Element residual ``(2*vec_test,)`` from every field's OWN element dofs ``locs[i]`` ``(2, vec_i)``.

        Taking the local dofs explicitly is what exposes the element Jacobian: a coupled term's element
        block couples the *test* field's element dofs to **each** field's, so differentiating this w.r.t.
        ``locs`` yields one ``(2*vec_test, 2*vec_i)`` block per field — all still element-local, hence
        scatterable (see ``sparse_jacobian``)."""
        xc = nodes[cell]
        h = xc[1] - xc[0]
        sg = (dN_dxi / h)[:, :, None]
        per_field = [{"shape_vals": N, "shape_grads": sg, "cell_sol": locs[i]} for i in range(nfields)]
        local = {
            "physical_quad_points": (N @ xc)[:, None],
            "fields": per_field,
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": ctx,
            "temporal_tags": (),
            "runtime_parameter_tags": (),
            "volume_vars": (),
        }
        return _integrate_term(domain, e, local, gw * h)  # (2*vec_test,)

    def _bnd_local(nid, locs, e, dtype):
        """Boundary contribution ``(vec_test,)`` at node ``nid`` from each field's own dof ``locs[i]``."""
        per_field = [
            {
                "shape_vals": jnp.ones((1, 1), dtype=dtype),
                "shape_grads": jnp.zeros((1, 1, 1), dtype=dtype),
                "cell_sol": locs[i],
            }
            for i in range(nfields)
        ]
        local = {
            "physical_quad_points": jnp.asarray([[nodes[nid]]]),
            "fields": per_field,
            "field_index": field_index,
            "tag": "fem_gauss",
            "surface": True,
            "domain_context": ctx,
            "temporal_tags": (),
            "runtime_parameter_tags": (),
            "volume_vars": (),
        }
        return _integrate_term(domain, e, local, jnp.ones((1,), dtype=dtype))  # (vec_test,)

    def residual(u_flat):
        u_list = [u_flat[offs[i] : offs[i + 1]].reshape(n_nodes, vecs[i]) for i in range(nfields)]
        R = jnp.zeros(total, dtype=u_flat.dtype)

        # --- volume terms: per element, evaluate the coeff against all fields, scatter
        #     into the test field's block ---
        for coeff, test_idx in term_list:
            elem_res = jax.vmap(lambda c, e=coeff: _elem_local(c, [u_list[i][c] for i in range(nfields)], e))(cells)
            R = R.at[cell_dofs[test_idx].reshape(-1)].add(elem_res.reshape(-1))

        # --- boundary terms: degenerate one-node element (shape val 1, weight 1) ---
        for coeff, test_idx, nids in boundary_term_list:
            vt = vecs[test_idx]
            for nid in nids:
                locs = [u_list[i][nid : nid + 1] for i in range(nfields)]
                contrib = _bnd_local(nid, locs, coeff, u_flat.dtype)  # (vec_test,)
                R = R.at[offs[test_idx] + nid * vt + jnp.arange(vt)].add(contrib.reshape(vt))

        return R

    def sparse_jacobian(u_flat, args=None):
        """``dR/du`` as a **BCOO** for the coupled block system, scattered from element blocks.

        A coupled term contributes, per element, one ``(2*vec_test, 2*vec_i)`` block per field ``i`` it
        references — the off-diagonal blocks are exactly the field couplings. Every one is element-local,
        so the whole block operator is assembled in ``O(nnz)`` instead of by an ``O(total²)`` global
        ``jacfwd`` over the concatenated block vector."""
        from jax.experimental import sparse as jsp

        u_list = [u_flat[offs[i] : offs[i + 1]].reshape(n_nodes, vecs[i]) for i in range(nfields)]
        idx, dat = [], []

        for coeff, test_idx in term_list:

            def _jac(cell, e=coeff):
                locs = [u_list[i][cell] for i in range(nfields)]
                return jax.jacfwd(lambda L: _elem_local(cell, L, e))(locs)  # list of (2*vt, 2, vec_i)

            Js = jax.vmap(_jac)(cells)
            n_el, vt = cells.shape[0], vecs[test_idx]
            for i in range(nfields):
                Ji = jnp.asarray(Js[i]).reshape(n_el, 2 * vt, 2 * vecs[i])
                rows = jnp.broadcast_to(cell_dofs[test_idx][:, :, None], Ji.shape)
                cols = jnp.broadcast_to(cell_dofs[i][:, None, :], Ji.shape)
                idx.append(jnp.stack([rows.reshape(-1), cols.reshape(-1)], axis=1))
                dat.append(Ji.reshape(-1))

        for coeff, test_idx, nids in boundary_term_list:
            vt = vecs[test_idx]
            for nid in nids:
                locs = [u_list[i][nid : nid + 1] for i in range(nfields)]
                Jb = jax.jacfwd(lambda L: _bnd_local(nid, L, coeff, u_flat.dtype))(locs)
                for i in range(nfields):
                    Bi = jnp.asarray(Jb[i]).reshape(vt, vecs[i])
                    r = offs[test_idx] + nid * vt + jnp.arange(vt)
                    c = offs[i] + nid * vecs[i] + jnp.arange(vecs[i])
                    rows = jnp.broadcast_to(r[:, None], Bi.shape)
                    cols = jnp.broadcast_to(c[None, :], Bi.shape)
                    idx.append(jnp.stack([rows.reshape(-1), cols.reshape(-1)], axis=1))
                    dat.append(Bi.reshape(-1))

        if not idx:
            return jsp.BCOO((jnp.zeros((0,), u_flat.dtype), jnp.zeros((0, 2), jnp.int32)), shape=(total, total))
        return jsp.BCOO((jnp.concatenate(dat), jnp.concatenate(idx).astype(jnp.int32)), shape=(total, total))

    residual.sparse_jacobian = sparse_jacobian
    return residual


def _typed_terms_1d(domain, bares, field_index):
    """Lower + sign-split each bare into ``(coeff, test_field_index)`` pairs."""
    from .fem_utils import _lower_statefield_to_trial, _test_field_index
    from .weak_form import _apply_sign, _split_additive_terms

    out = []
    for bare in bares:
        for sign, sub in _split_additive_terms(domain, bare):
            coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
            tfi = _test_field_index(coeff, field_index)
            if tfi is None:
                raise ValueError(
                    "jno.fem: each coupled 1D weak term must contain exactly one test field "
                    "(it determines the equation block); got a term with zero or several."
                )
            out.append((coeff, tfi))
    return out


def assemble_fem_1d_multifield(domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, *, quad_degree):
    """Assemble a coupled (multi-field) 1D weak form into ``(op, mode, offsets)`` for :class:`FEM`.

    1D block analogue of the 2D/3D ``_assemble_multifield``. The
    field layout ``(fields, field_index)`` is inferred once from the volume trial
    functions and threaded into every block builder so the mass / operator / residual
    blocks share one ordering. ``offsets`` reports that ordering to the caller in the same
    ``[0, n_0, n_0+n_1, ...]`` form the 2D/3D path uses."""
    from ...trace import FemResidualOperator
    from .fem_utils import _infer_fields, _lower_statefield_to_trial
    from .weak_form import _contains_temporal_derivative, _is_obviously_nonlinear_in_unknown

    # The coupled block builder threads no runtime parameters — its element kernels publish no
    # ``runtime_parameter_tags``/``volume_vars``. Without this guard the parameter's value simply never
    # reaches the kernel and the solve dies with an internal KeyError about InternalVars, which is what
    # the single-field path did before it grew a parameter path. Fail loud and say so instead.
    _c_tags, _, _c_neural, _c_exprs = _param_context(volume_terms, boundary_terms)
    if _c_exprs:
        raise NotImplementedError(
            "jno.fem: a runtime parameter or neural coefficient on a COUPLED 1D system "
            f"({sorted(_c_exprs)}) is not supported — the coupled 1D block assembler threads no runtime "
            "parameters. A single-field 1D form is fully parametric (steady, transient, linear and "
            "nonlinear); use one field, or a 2D/3D domain for the coupled case."
        )

    n_nodes = int(np.asarray(domain.mesh.points).shape[0])

    # Shared field layout: first appearance across the volume terms (same convention as
    # _infer_fields on the summed volume expr in the 2D/3D path).
    fields: List[Any] = []
    field_index: Dict[Any, int] = {}
    for bare in volume_terms:
        fs, _ = _infer_fields(_lower_statefield_to_trial(bare, {}))
        for f in fs:
            if f["field_key"] not in field_index:
                field_index[f["field_key"]] = len(fields)
                fields.append(f)
    if any(int(f["vec"]) != 1 for f in fields):
        raise NotImplementedError(
            "jno.fem: coupled 1D (LINE2) assembly supports scalar fields (vec=1) per field only; "
            "got a vector field. Use scalar fields, or a 2D/3D domain for vector unknowns."
        )

    # Per-field block offsets, published to `FEM` exactly as the 2D/3D block path publishes them.
    # They are not a convenience: `fem.offsets` is what every consumer slices a coupled solution by
    # (the periodic reduction reduces block-wise through it, and post-processing splits the flat DOF
    # vector with it). Returning them here is what keeps a coupled 1D system indistinguishable from a
    # coupled 2D one on the outside.
    offsets = _block_offsets(fields, n_nodes)

    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]
    if any(_contains_temporal_derivative(t) for t in all_terms):
        dirichlet_pairs = _multifield_dirichlet_dofs_1d(domain, dirichlet_raw, fields, field_index, n_nodes)
        op, mode = _assemble_1d_multifield_transient(
            domain,
            volume_terms,
            boundary_terms,
            dirichlet_pairs,
            ic_residuals,
            fields,
            field_index,
            quad_degree=quad_degree,
        )
        return op, mode, offsets
    if ic_residuals:
        raise ValueError("jno.fem: an initial condition was given but the 1D weak form has no time derivative.")

    volume_tl = _typed_terms_1d(domain, volume_terms, field_index)
    boundary_tl = [
        (coeff, tfi, _region_node_ids(domain, region))
        for region, bares in boundary_terms.items()
        for (coeff, tfi) in _typed_terms_1d(domain, bares, field_index)
    ]
    dirichlet_pairs = _multifield_dirichlet_dofs_1d(domain, dirichlet_raw, fields, field_index, n_nodes)
    total = offsets[-1]
    residual_free = _make_multifield_residual_1d(
        domain, volume_tl, boundary_tl, fields, field_index, n_nodes=n_nodes, quad_degree=quad_degree
    )

    if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms):
        residual = _apply_dirichlet_rows(residual_free, dirichlet_pairs)
        op = FemResidualOperator(
            residual_fn=lambda u, args=None: residual(jnp.asarray(u)),
            jacobian_fn=lambda u, args=None: jax.jacfwd(residual)(jnp.asarray(u)),
            size=total,
        )
        return op, "nonlinear", offsets

    zeros = jnp.zeros(total)
    A = residual_free.sparse_jacobian(zeros)
    b = -residual_free(zeros)
    A, b = _apply_dirichlet_symmetric(A, b, dirichlet_pairs)
    A = compress_eager(A)  # ~19x redundant triplets otherwise; see the helper
    return (A, b), "linear", offsets


def _multifield_initial_state_1d(domain, fields, field_index, ic_residuals, n_nodes):
    """Block initial state from per-field IC residuals (1D; all fields share the mesh)."""
    from ..._fem import _bare, _constant_of, _essential_spec, _eval_value_node_at, _field_key_of

    offs = _block_offsets(fields, n_nodes)
    state0 = jnp.zeros((offs[-1],))
    pts = jnp.asarray(domain.mesh.points)[:, : domain.dimension]
    for ic in ic_residuals:
        fidx = field_index.get(_field_key_of(ic))
        if fidx is None:
            continue
        _comp, node = _essential_spec(_bare(ic))
        const = _constant_of(node)
        if const is not None:
            vals = jnp.full((n_nodes,), float(const))
        else:
            v = jnp.asarray(_eval_value_node_at(node, pts))
            vals = jnp.broadcast_to(v, (n_nodes,)) if v.shape[0] == 1 else v
        block = jnp.asarray(vals).reshape(-1)
        if block.shape[0] != offs[fidx + 1] - offs[fidx]:
            raise NotImplementedError(
                "jno.fem: vector-field initial conditions in coupled 1D transient are not supported yet."
            )
        state0 = state0.at[offs[fidx] : offs[fidx + 1]].set(block)
    return state0


def _assemble_1d_multifield_transient(
    domain, volume_terms, boundary_terms, dirichlet_pairs, ic_residuals, fields, field_index, *, quad_degree
):
    """Coupled 1D first-order transient: block mass + block spatial operator/residual.

    Mirrors the 2D/3D coupled transient path: split the volume terms into a mass list
    (temporal-derivative terms, ``u_t``->``u``) and a spatial list, build a block ``M`` and the
    block spatial operator/residual sharing one field index. Scope (as in 2D/3D): every field must
    carry a time derivative; homogeneous Dirichlet."""
    from .backend_blocks import SemidiscreteTimeBlock
    from .fem_utils import _lower_statefield_to_trial, _test_field_index
    from .time_route import _infer_time_window, _strip_temporal_trial_derivative
    from .weak_form import (
        _apply_sign,
        _contains_temporal_derivative,
        _is_obviously_nonlinear_in_unknown,
        _split_additive_terms,
    )

    n_nodes = int(np.asarray(domain.mesh.points).shape[0])
    total = _block_offsets(fields, n_nodes)[-1]

    mass_tl, spatial_tl = [], []
    for bare in volume_terms:
        for sign, sub in _split_additive_terms(domain, bare):
            coeff = _lower_statefield_to_trial(_apply_sign(domain, sign, sub), {})
            tfi = _test_field_index(coeff, field_index)
            if tfi is None:
                raise ValueError("jno.fem: each coupled 1D weak term must contain exactly one test field.")
            if _contains_temporal_derivative(coeff):
                mass_tl.append((_strip_temporal_trial_derivative(coeff), tfi))
            else:
                spatial_tl.append((coeff, tfi))
    if not mass_tl:
        raise ValueError("jno.fem: a transient 1D weak form must contain a temporal-derivative term (u_t * phi).")
    if {fields[tfi]["field_key"] for _, tfi in mass_tl} != {f["field_key"] for f in fields}:
        raise NotImplementedError(
            "jno.fem: coupled transient requires every field to carry a time derivative (u_t * test); "
            "algebraic (DAE) fields are not supported yet."
        )

    boundary_tl = [
        (coeff, tfi, _region_node_ids(domain, region))
        for region, bares in boundary_terms.items()
        for (coeff, tfi) in _typed_terms_1d(domain, bares, field_index)
    ]

    mass_res = _make_multifield_residual_1d(
        domain, mass_tl, [], fields, field_index, n_nodes=n_nodes, quad_degree=quad_degree
    )
    spatial_res = _make_multifield_residual_1d(
        domain, spatial_tl, boundary_tl, fields, field_index, n_nodes=n_nodes, quad_degree=quad_degree
    )
    M = mass_res.sparse_jacobian(jnp.zeros(total))
    state0 = _multifield_initial_state_1d(domain, fields, field_index, ic_residuals, n_nodes)
    t0, t1, dt = _infer_time_window(domain)
    common = dict(
        backend="transient",
        mode="implicit",
        time_order=1,
        spatial_kind="weak_form",
        state0=state0,
        t0=t0,
        t1=t1,
        dt=dt,
        eval_context=getattr(domain, "_fem_eval_context", {}) or {},
    )

    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]
    if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms):
        residual = _apply_dirichlet_rows(spatial_res, dirichlet_pairs)
        dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
        # the mass is a BCOO now, which has no `.at[]` — zero the constrained rows/cols sparsely
        # (mirrors the linear branch); a dense fallback keeps the old indexing path.
        if dofs is None:
            M_nl = M
        elif hasattr(M, "indices"):
            M_nl = bcoo_zero_rows_cols(M, dofs)
        else:
            M_nl = M.at[dofs, :].set(0.0).at[:, dofs].set(0.0)
        block = SemidiscreteTimeBlock(
            mass=lambda t, args=None, _M=compress_eager(M_nl): _M,
            residual=lambda u, t, args=None: residual(jnp.asarray(u)),
            jacobian=lambda u, t, args=None: jax.jacfwd(residual)(jnp.asarray(u)),
            **common,
        )
        return block, "transient"

    A = spatial_res.sparse_jacobian(jnp.zeros(total))
    c = -spatial_res(jnp.zeros(total))
    M, A, c = _apply_dirichlet_transient(M, A, c, dirichlet_pairs)
    M, A = compress_eager(M), compress_eager(A)
    block = SemidiscreteTimeBlock(M=M, A=A, affine_bias=c, **common)
    return block, "transient"
