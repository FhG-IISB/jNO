"""Native 1D (segment / ``LINE2``) FEM assembly for ``jno.fem``.

feax has no 1D volume element (its ``get_elements`` covers 2D/3D only), so a 1D
``jno.domain.line(...)`` problem can't go through the feax assembly path. This
module is a small native assembler for the 1D case that reuses jNO's existing
weak-form integrand evaluator (:func:`_eval_expr_for_feax`) — it only adds the
1D geometry (``LINE2`` shape functions + Gauss quadrature), an element loop with
a global scatter, and native boundary handling. Same matrices-only contract as
the 2D/3D path: it returns the assembled system, never a solve.

Assembly strategy (mirrors how feax forms an element matrix):
- build a global residual ``R(u)`` by evaluating each weak term at the 1D
  quadrature points and scattering element contributions;
- boundary (Neumann/Robin) terms ride the *same* residual path as a degenerate
  one-node "element" (so a Robin ``a*u`` lands in the matrix and ``-g`` in the
  load by construction — no ad-hoc load patching);
- Dirichlet rows are replaced by ``u[d] - g`` so the tangent row is the identity
  and the load entry is ``g``;
- for a linear problem ``A = jacfwd(R)(0)`` and ``b = -R(0)``; nonlinear keeps
  ``R``/``jac`` as callables; transient separates the mass term.

Scope: scalar unknown (``vec == 1``) on ``LINE2`` elements. Boundary terms must
be value/Robin terms (``g*phi`` / ``(a*u - g)*phi``) — they carry no spatial
derivative, since a 0D facet has no element to differentiate over.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .feax_utils import _eval_expr_for_feax

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


def _line_shape(gp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """LINE2 shape values ``N`` and reference gradients ``dN/dxi`` at ``gp``.

    ``N0 = 1 - xi``, ``N1 = xi`` on [0, 1]. Returns ``N`` (n_quad, 2) and
    ``dN_dxi`` (n_quad, 2)."""
    one = jnp.ones_like(gp)
    N = jnp.stack([1.0 - gp, gp], axis=-1)
    dN_dxi = jnp.stack([-one, one], axis=-1)
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
    if num_args == 1:
        hits = jax.vmap(loc)(pts)
    else:
        hits = jax.vmap(loc)(pts, jnp.arange(n))
    return list(np.where(np.asarray(hits).reshape(-1))[0])


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
    val = _eval_expr_for_feax(domain, expr, local)
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    return ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]


def _make_residual(
    domain: Any,
    volume_terms: List[Any],
    boundary_terms: Dict[str, List[Any]],
    *,
    n_nodes: int,
    vec: int,
    quad_degree: int,
) -> Any:
    """Build the *free* global residual ``R(u_flat) -> (n_nodes*vec,)``.

    Covers the volume + boundary (Neumann/Robin) weak terms; Dirichlet is applied
    by the caller (symmetric elimination for the linear ``(A, b)``, row-replacement
    for the nonlinear residual)."""
    nodes = jnp.asarray(domain.mesh.points)[:, 0]
    cells = jnp.asarray(domain.mesh.cells_dict["line"], dtype=jnp.int32)  # (n_elem, 2)
    gp, gw = _line_quadrature(quad_degree)
    N, dN_dxi = _line_shape(gp)
    ctx = getattr(domain, "context", {}) or {}
    comp_offsets = jnp.arange(vec)

    # element-local -> global DOF map, node-major: dof = node*vec + comp
    cell_dofs = (cells[:, :, None] * vec + comp_offsets[None, None, :]).reshape(cells.shape[0], -1)  # (n_elem, 2*vec)

    # precompute boundary (region, term, node_id) triples
    boundary_apps: List[Tuple[Any, int]] = []
    for region, terms in boundary_terms.items():
        nids = _region_node_ids(domain, region)
        for term in terms:
            for nid in nids:
                boundary_apps.append((term, int(nid)))

    def _volume_local(cell, u, expr):
        xc = nodes[cell]  # (2,)
        h = xc[1] - xc[0]
        phys = (N @ xc)[:, None]  # (n_quad, 1)
        shape_grads = (dN_dxi / h)[:, :, None]  # (n_quad, 2, 1)
        local = {
            "physical_quad_points": phys,
            "shape_vals": N,
            "shape_grads": shape_grads,
            "cell_sol": u[cell],  # (2, vec)
            "tag": "fem_gauss",
            "surface": False,
            "domain_context": ctx,
            "trial_value_shape": (),
            "trial_vec": vec,
            "temporal_tags": (),
            "runtime_parameter_tags": (),
            "volume_vars": (),
        }
        return _integrate_term(domain, expr, local, gw * h)  # (2*vec,)

    def residual(u_flat):
        u = u_flat.reshape(n_nodes, vec)
        R = jnp.zeros(n_nodes * vec, dtype=u_flat.dtype)

        # --- volume terms: vmap over elements, then scatter ---
        for expr in volume_terms:
            elem_res = jax.vmap(lambda c, e=expr: _volume_local(c, u, e))(cells)  # (n_elem, 2*vec)
            R = R.at[cell_dofs.reshape(-1)].add(elem_res.reshape(-1))

        # --- boundary terms: degenerate one-node "element" (shape val 1, weight 1) ---
        for expr, nid in boundary_apps:
            local = {
                "physical_quad_points": jnp.asarray([[nodes[nid]]]),
                "shape_vals": jnp.ones((1, 1), dtype=u_flat.dtype),
                "shape_grads": jnp.zeros((1, 1, 1), dtype=u_flat.dtype),
                "cell_sol": u[nid : nid + 1],  # (1, vec)
                "tag": "fem_gauss",
                "surface": True,
                "domain_context": ctx,
                "trial_value_shape": (),
                "trial_vec": vec,
                "temporal_tags": (),
                "runtime_parameter_tags": (),
                "volume_vars": (),
            }
            contrib = _integrate_term(domain, expr, local, jnp.ones((1,), dtype=u_flat.dtype))  # (vec,)
            R = R.at[nid * vec + comp_offsets].add(contrib.reshape(vec))

        return R

    return residual


def _apply_dirichlet_symmetric(A, b, dirichlet_pairs: List[Tuple[int, float]]):
    """Symmetric Dirichlet elimination on a linear system ``A u = b``.

    Moves known columns to the RHS, then zeros the constrained rows *and* columns
    and sets a unit diagonal — so ``A`` stays symmetric (matching the 2D/3D feax
    path), unlike a row-only replacement."""
    if not dirichlet_pairs:
        return A, b
    dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32)
    vals = jnp.asarray([p[1] for p in dirichlet_pairs], dtype=b.dtype)
    b = b - A[:, dofs] @ vals  # carry the known columns to the load
    A = A.at[dofs, :].set(0.0).at[:, dofs].set(0.0).at[dofs, dofs].set(1.0)
    b = b.at[dofs].set(vals)
    return A, b


def _apply_dirichlet_rows(residual_free, dirichlet_pairs: List[Tuple[int, float]]):
    """Wrap a free residual so Dirichlet rows read ``u[d] - g`` (row-replacement).

    Used for the nonlinear residual: the tangent row becomes the identity and the
    Newton step drives ``u[d] -> g``."""
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
) -> Tuple[Any, str]:
    """Assemble a 1D (``LINE2``) weak form into ``(op, mode)`` for :class:`FEM`.

    ``mode`` is ``"linear"`` (``op = (A, b)``), ``"nonlinear"`` (``op`` a
    :class:`FemResidualOperator`), or ``"transient"`` (``op`` a ``FeaxTimeBlock``).
    """
    from ...trace import FemResidualOperator
    from .weak_form import _contains_temporal_derivative, _is_obviously_nonlinear_in_unknown

    if int(vec) != 1:
        raise NotImplementedError(f"jno.fem: 1D (LINE2) assembly supports a scalar unknown (vec=1) only; got vec={vec}.")

    n_nodes = int(np.asarray(domain.mesh.points).shape[0])
    ndof = n_nodes * vec
    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]

    if any(_contains_temporal_derivative(t) for t in all_terms):
        return _assemble_1d_transient(
            domain, volume_terms, boundary_terms, dirichlet_values, ic_residuals, vec=vec, quad_degree=quad_degree
        )

    if ic_residuals:
        raise ValueError("jno.fem: an initial condition was given but the 1D weak form has no time derivative.")

    dirichlet_pairs = _dirichlet_dofs(domain, dirichlet_values, vec)
    residual_free = _make_residual(domain, volume_terms, boundary_terms, n_nodes=n_nodes, vec=vec, quad_degree=quad_degree)

    nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms)
    if nonlinear:
        residual = _apply_dirichlet_rows(residual_free, dirichlet_pairs)
        op = FemResidualOperator(
            residual_fn=lambda u, args=None: residual(jnp.asarray(u)),
            jacobian_fn=lambda u, args=None: jax.jacfwd(residual)(jnp.asarray(u)),
            size=ndof,
        )
        return op, "nonlinear"

    # linear: R(u) = A u - b  ->  A = dR/du, b = -R(0), then symmetric Dirichlet
    zeros = jnp.zeros(ndof)
    A = jax.jacfwd(residual_free)(zeros)
    b = -residual_free(zeros)
    A, b = _apply_dirichlet_symmetric(A, b, dirichlet_pairs)
    return (A, b), "linear"


def _apply_dirichlet_transient(M, A, c, dirichlet_pairs: List[Tuple[int, float]]):
    """Apply Dirichlet to a semidiscrete linear block ``M u̇ + A u = c``.

    ``A``/``c`` get symmetric elimination (so ``A`` stays symmetric); ``M`` has the
    constrained rows *and* columns zeroed (no diagonal) so the constrained DOFs
    carry no time derivative and the row equation reduces to ``u[d] = g``."""
    A, c = _apply_dirichlet_symmetric(A, c, dirichlet_pairs)
    if dirichlet_pairs:
        dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32)
        M = M.at[dofs, :].set(0.0).at[:, dofs].set(0.0)
    return M, A, c


def _assemble_1d_transient(domain, volume_terms, boundary_terms, dirichlet_values, ic_residuals, *, vec, quad_degree):
    """Assemble a first-order transient 1D weak form into a ``FeaxTimeBlock``.

    Splits the volume terms into a temporal part (``u_t``) and a spatial part: the
    temporal part becomes the mass matrix ``M`` (rewriting ``u_t`` -> ``u`` via
    ``_strip_temporal_trial_derivative``); the spatial part + boundary terms become
    the operator. Linear -> ``M``/``A`` payload; nonlinear -> ``mass``/``residual``/
    ``jacobian`` callables."""
    from ..._fem import _initial_state
    from .backend_blocks import FeaxTimeBlock
    from .time_route import _infer_time_window, _strip_temporal_trial_derivative
    from .weak_form import (
        _apply_sign,
        _contains_temporal_derivative,
        _is_obviously_nonlinear_in_unknown,
        _split_additive_terms,
    )

    n_nodes = int(np.asarray(domain.mesh.points).shape[0])
    ndof = n_nodes * vec

    # Split each weak constraint into additive sub-terms first, so the temporal
    # term (u_t * phi) can be separated from the spatial terms in the same sum.
    sub_terms = [
        _apply_sign(domain, sign, sub) for bare in volume_terms for sign, sub in _split_additive_terms(domain, bare)
    ]
    temporal_terms = [t for t in sub_terms if _contains_temporal_derivative(t)]
    spatial_terms = [t for t in sub_terms if not _contains_temporal_derivative(t)]
    if not temporal_terms:
        raise ValueError("jno.fem: a transient 1D weak form must contain a temporal-derivative term (u_t * phi).")
    mass_terms = [_strip_temporal_trial_derivative(t) for t in temporal_terms]

    dirichlet_pairs = _dirichlet_dofs(domain, dirichlet_values, vec)
    state0 = _initial_state(ic_residuals[0], domain) if ic_residuals else jnp.zeros(ndof)
    t0, t1, dt = _infer_time_window(domain)

    mass_res = _make_residual(domain, mass_terms, {}, n_nodes=n_nodes, vec=vec, quad_degree=quad_degree)
    spatial_res = _make_residual(domain, spatial_terms, boundary_terms, n_nodes=n_nodes, vec=vec, quad_degree=quad_degree)

    all_terms = spatial_terms + [t for ts in boundary_terms.values() for t in ts]
    nonlinear = any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms)

    M = jax.jacfwd(mass_res)(jnp.zeros(ndof))

    common = dict(
        backend="feax_time",
        mode="implicit",
        time_order=1,
        spatial_kind="weak_form",
        state0=state0,
        t0=t0,
        t1=t1,
        dt=dt,
        feax_context=getattr(domain, "_feax_context", {}) or {},
    )

    if nonlinear:
        # M(t) u̇ + R(u) = 0 with R the (Dirichlet-enforced) spatial residual.
        residual = _apply_dirichlet_rows(spatial_res, dirichlet_pairs)
        dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
        M_nl = M if dofs is None else M.at[dofs, :].set(0.0).at[:, dofs].set(0.0)
        block = FeaxTimeBlock(
            mass=lambda t, args=None, _M=M_nl: _M,
            residual=lambda u, t, args=None: residual(jnp.asarray(u)),
            jacobian=lambda u, t, args=None: jax.jacfwd(residual)(jnp.asarray(u)),
            **common,
        )
        return block, "transient"

    # linear: M u̇ + A u = c
    A = jax.jacfwd(spatial_res)(jnp.zeros(ndof))
    c = -spatial_res(jnp.zeros(ndof))
    M, A, c = _apply_dirichlet_transient(M, A, c, dirichlet_pairs)
    block = FeaxTimeBlock(M=M, A=A, affine_bias=c, **common)
    return block, "transient"


# ==========================================================================
# coupled / mixed multi-field 1D (block native assembly)
# ==========================================================================
# The native analogue of the 2D/3D feax block path: feax has no LINE2 element, so
# we hand-build the block residual. Each field is a scalar/vector unknown on the
# shared LINE2 mesh; the global DOF vector is laid out by field block —
# ``offset[i] = sum_{j<i} n_nodes * vec_j`` (matching the feax ``problem.offset``
# convention) — so the user slices the solution exactly as in 2D/3D coupled.


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
    ``(coeff, test_field_index)`` via the shared ``_eval_expr_for_feax``, scattering the
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

    def residual(u_flat):
        u_list = [u_flat[offs[i] : offs[i + 1]].reshape(n_nodes, vecs[i]) for i in range(nfields)]
        R = jnp.zeros(total, dtype=u_flat.dtype)

        # --- volume terms: per element, evaluate the coeff against all fields, scatter
        #     into the test field's block ---
        for coeff, test_idx in term_list:

            def _elem(cell, e=coeff):
                xc = nodes[cell]
                h = xc[1] - xc[0]
                sg = (dN_dxi / h)[:, :, None]
                per_field = [{"shape_vals": N, "shape_grads": sg, "cell_sol": u_list[i][cell]} for i in range(nfields)]
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

            elem_res = jax.vmap(_elem)(cells)  # (n_elem, 2*vec_test)
            R = R.at[cell_dofs[test_idx].reshape(-1)].add(elem_res.reshape(-1))

        # --- boundary terms: degenerate one-node element (shape val 1, weight 1) ---
        for coeff, test_idx, nids in boundary_term_list:
            vt = vecs[test_idx]
            for nid in nids:
                per_field = [
                    {
                        "shape_vals": jnp.ones((1, 1), dtype=u_flat.dtype),
                        "shape_grads": jnp.zeros((1, 1, 1), dtype=u_flat.dtype),
                        "cell_sol": u_list[i][nid : nid + 1],
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
                contrib = _integrate_term(domain, coeff, local, jnp.ones((1,), dtype=u_flat.dtype))  # (vec_test,)
                R = R.at[offs[test_idx] + nid * vt + jnp.arange(vt)].add(contrib.reshape(vt))

        return R

    return residual


def _typed_terms_1d(domain, bares, field_index):
    """Lower + sign-split each bare into ``(coeff, test_field_index)`` pairs."""
    from .feax_utils import _lower_statefield_to_trial, _test_field_index
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
    """Assemble a coupled (multi-field) 1D weak form into ``(op, mode)`` for :class:`FEM`.

    Native block analogue of ``_assemble_multifield`` (which uses feax for 2D/3D). The
    field layout ``(fields, field_index)`` is inferred once from the volume trial
    functions and threaded into every block builder so the mass / operator / residual
    blocks share one ordering."""
    from ...trace import FemResidualOperator
    from .feax_utils import _infer_fields, _lower_statefield_to_trial
    from .weak_form import _contains_temporal_derivative, _is_obviously_nonlinear_in_unknown

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

    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]
    if any(_contains_temporal_derivative(t) for t in all_terms):
        dirichlet_pairs = _multifield_dirichlet_dofs_1d(domain, dirichlet_raw, fields, field_index, n_nodes)
        return _assemble_1d_multifield_transient(
            domain,
            volume_terms,
            boundary_terms,
            dirichlet_pairs,
            ic_residuals,
            fields,
            field_index,
            quad_degree=quad_degree,
        )
    if ic_residuals:
        raise ValueError("jno.fem: an initial condition was given but the 1D weak form has no time derivative.")

    volume_tl = _typed_terms_1d(domain, volume_terms, field_index)
    boundary_tl = [
        (coeff, tfi, _region_node_ids(domain, region))
        for region, bares in boundary_terms.items()
        for (coeff, tfi) in _typed_terms_1d(domain, bares, field_index)
    ]
    dirichlet_pairs = _multifield_dirichlet_dofs_1d(domain, dirichlet_raw, fields, field_index, n_nodes)
    total = _block_offsets(fields, n_nodes)[-1]
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
        return op, "nonlinear"

    zeros = jnp.zeros(total)
    A = jax.jacfwd(residual_free)(zeros)
    b = -residual_free(zeros)
    A, b = _apply_dirichlet_symmetric(A, b, dirichlet_pairs)
    return (A, b), "linear"


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

    Mirrors ``_assemble_multifield_transient`` (2D/3D) natively: split the volume terms
    into a mass list (temporal-derivative terms, ``u_t``->``u``) and a spatial list,
    build a block ``M`` and the block spatial operator/residual sharing one field index.
    Scope (as in 2D/3D): every field must carry a time derivative; homogeneous Dirichlet."""
    from .backend_blocks import FeaxTimeBlock
    from .feax_utils import _lower_statefield_to_trial, _test_field_index
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
    M = jax.jacfwd(mass_res)(jnp.zeros(total))
    state0 = _multifield_initial_state_1d(domain, fields, field_index, ic_residuals, n_nodes)
    t0, t1, dt = _infer_time_window(domain)
    common = dict(
        backend="feax_time",
        mode="implicit",
        time_order=1,
        spatial_kind="weak_form",
        state0=state0,
        t0=t0,
        t1=t1,
        dt=dt,
        feax_context=getattr(domain, "_feax_context", {}) or {},
    )

    all_terms = list(volume_terms) + [t for ts in boundary_terms.values() for t in ts]
    if any(_is_obviously_nonlinear_in_unknown(domain, t) for t in all_terms):
        residual = _apply_dirichlet_rows(spatial_res, dirichlet_pairs)
        dofs = jnp.asarray([p[0] for p in dirichlet_pairs], dtype=jnp.int32) if dirichlet_pairs else None
        M_nl = M if dofs is None else M.at[dofs, :].set(0.0).at[:, dofs].set(0.0)
        block = FeaxTimeBlock(
            mass=lambda t, args=None, _M=M_nl: _M,
            residual=lambda u, t, args=None: residual(jnp.asarray(u)),
            jacobian=lambda u, t, args=None: jax.jacfwd(residual)(jnp.asarray(u)),
            **common,
        )
        return block, "transient"

    A = jax.jacfwd(spatial_res)(jnp.zeros(total))
    c = -spatial_res(jnp.zeros(total))
    M, A, c = _apply_dirichlet_transient(M, A, c, dirichlet_pairs)
    block = FeaxTimeBlock(M=M, A=A, affine_bias=c, **common)
    return block, "transient"
