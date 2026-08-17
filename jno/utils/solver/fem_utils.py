from __future__ import annotations

"""
Internal FEM assembly utilities.

This module contains low-level helpers shared by the steady FEM route and the
transient semidiscrete-time route. It is intentionally not a public API.

Responsibilities:
- lower jNO weak-form symbols into assembly kernels,
- evaluate symbolic expressions at quadrature points inside volume/surface kernels,
- normalize Dirichlet data and prepare residual/Jacobian runtime objects.
"""
import hashlib as _hashlib
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ...trace import (
    BinaryOp,
    Constant,
    Diff,
    DiffSlot,
    FrozenField,
    FunctionCall,
    Hessian,
    HistoryRef,
    Jacobian,
    Literal,
    ModelCall,
    OperationCall,
    OperationDef,
    RegionMask,
    StateField,
    TagMask,
    TensorTag,
    TestFunction,
    Tracker,
    TrialFunction,
    Variable,
)
from ...utils.logger import get_logger
from .solver_helper import contains_node_type, iter_children


def _default_float_dtype():
    return jnp.asarray(0.0).dtype


def _lower_statefield_to_trial(expr, trial_cache=None):
    """Final safety pass: replace any remaining StateField with one shared TrialFunction.

    This is backend-side insurance for the Phase-1 NN-first weak route.
    """
    if trial_cache is None:
        trial_cache = {}

    if expr is None:
        return None

    if isinstance(expr, StateField):
        key = (int(expr.state_id), str(expr.name), tuple(expr.value_shape))
        if key not in trial_cache:
            trial_cache[key] = TrialFunction(name=expr.name, value_shape=expr.value_shape)
        return trial_cache[key]

    if isinstance(expr, BinaryOp):
        left = _lower_statefield_to_trial(expr.left, trial_cache)
        right = _lower_statefield_to_trial(expr.right, trial_cache)
        if left is not expr.left or right is not expr.right:
            return BinaryOp(expr.op, left, right)
        return expr

    if isinstance(expr, FunctionCall):
        new_args = []
        changed = False
        for a in expr.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                na = _lower_statefield_to_trial(a, trial_cache)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            return expr.copy_with_args(new_args)
        return expr

    if isinstance(expr, ModelCall):
        new_args = []
        changed = False
        for a in expr.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                na = _lower_statefield_to_trial(a, trial_cache)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            rebuilt = ModelCall(expr.model, new_args)
            rebuilt.op_id = expr.op_id
            return rebuilt
        return expr

    if isinstance(expr, OperationDef):
        new_expr = _lower_statefield_to_trial(expr.expr, trial_cache)
        if new_expr is not expr.expr:
            rebuilt = OperationDef.__new__(OperationDef)
            rebuilt.expr = new_expr
            rebuilt.input_vars = expr.input_vars
            rebuilt.name = getattr(expr, "name", None)
            rebuilt.op_id = expr.op_id
            return rebuilt
        return expr

    if isinstance(expr, OperationCall):
        new_args = []
        changed = False
        for a in expr.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                na = _lower_statefield_to_trial(a, trial_cache)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            rebuilt = OperationCall(expr.operation, tuple(new_args))
            rebuilt.op_id = expr.op_id
            return rebuilt
        return expr

    if isinstance(expr, Jacobian):
        new_target = _lower_statefield_to_trial(expr.target, trial_cache)
        new_vars = []
        changed = new_target is not expr.target
        for v in expr.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                nv = _lower_statefield_to_trial(v, trial_cache)
            else:
                nv = v
            changed = changed or (nv is not v)
            new_vars.append(nv)
        if changed:
            return Jacobian(new_target, new_vars, expr.scheme)
        return expr

    if isinstance(expr, Hessian):
        new_target = _lower_statefield_to_trial(expr.target, trial_cache)
        new_vars = []
        changed = new_target is not expr.target
        for v in expr.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                nv = _lower_statefield_to_trial(v, trial_cache)
            else:
                nv = v
            changed = changed or (nv is not v)
            new_vars.append(nv)
        if changed:
            return Hessian(new_target, new_vars, expr.scheme, trace=expr.trace)
        return expr

    return expr


def _const_bc_fn(value):
    value = float(value)
    return lambda p, c=value: c


def _normalize_dirichlet_value(value, vec: int):
    """
    Normalize user Dirichlet data into value functions.

    Accepts None, scalar, callable, component list/tuple, or component dict.
    Returns one callable for scalar fields, a length-``vec`` list of callables for
    a fully specified vector field, or a partial ``{component_index: callable}`` dict
    when only some components are constrained (e.g. a roller/symmetry BC
    ``{"y": 0.0}`` pins only ``u_y`` and leaves the other components free).
    """
    if value is None:
        value = 0.0

    if vec < 1:
        raise ValueError(f"'vec' must be >= 1, got {vec}.")

    if callable(value):
        if vec == 1:
            return value
        return [value for _ in range(vec)]

    if np.isscalar(value):
        fn = _const_bc_fn(value)
        if vec == 1:
            return fn
        return [fn for _ in range(vec)]

    if isinstance(value, (list, tuple)):
        if len(value) != vec:
            raise ValueError(f"Dirichlet BC has {len(value)} entries, but vec={vec}.")
        out = []
        for v in value:
            if callable(v):
                out.append(v)
            elif np.isscalar(v):
                out.append(_const_bc_fn(v))
            else:
                raise TypeError("Dirichlet list/tuple entries must be callables or scalars.")
        if vec == 1:
            return out[0]
        return out

    if isinstance(value, dict):
        keymap = {"x": 0, "y": 1, "z": 2}
        # Partial spec: only the named components are constrained; the rest stay
        # free. (Zero-filling here would silently clamp the other components, which
        # breaks roller/symmetry BCs and conflicts at shared corner nodes.)
        out: dict[int, Any] = {}
        for k, v in value.items():
            c = keymap[k.lower()] if isinstance(k, str) else int(k)
            if c < 0 or c >= vec:
                raise ValueError(f"Component index {c} out of range for vec={vec}.")
            if callable(v):
                out[c] = v
            elif np.isscalar(v):
                out[c] = _const_bc_fn(v)
            else:
                raise TypeError("Dirichlet dict entries must be callables or scalars.")
        if vec == 1:
            return out.get(0, _const_bc_fn(0.0))
        return out

    raise TypeError(f"Unsupported Dirichlet BC value type: {type(value).__name__}")


# --------------------------------
# small expression-inspection helpers
# --------------------------------


def _strip_test_function_factor(expr):
    factors = []

    def collect_mul_factors(node):
        if isinstance(node, BinaryOp) and node.op == "*":
            collect_mul_factors(node.left)
            collect_mul_factors(node.right)
        else:
            factors.append(node)

    collect_mul_factors(expr)
    test_factors = [f for f in factors if isinstance(f, TestFunction)]
    if len(test_factors) != 1:
        return None

    coeff_factors = [f for f in factors if not isinstance(f, TestFunction)]
    if len(coeff_factors) == 0:
        return Literal(1.0)

    coeff = coeff_factors[0]
    for f in coeff_factors[1:]:
        coeff = BinaryOp("*", coeff, f)
    return coeff


def _is_simple_neumann_load(expr):
    if not contains_node_type(expr, TestFunction):
        return False
    if contains_node_type(expr, TrialFunction):
        return False
    if contains_node_type(expr, Jacobian):
        return False
    coeff = _strip_test_function_factor(expr)
    return coeff is not None


def _value_shape_num_components(value_shape) -> int:
    if value_shape is None or len(value_shape) == 0:
        return 1
    n = 1
    for s in value_shape:
        n *= int(s)
    return n


def _reshape_components_last(arr, value_shape):
    if value_shape is None or len(value_shape) == 0:
        return arr
    return jnp.reshape(arr, arr.shape[:-1] + tuple(value_shape))


def _drop_scalar_component_axis(grad_list):
    """Per-direction gradients of a ``value_shape == ()`` field, without a component axis.

    Each entry of ``grad_list`` is ``(n_quad, vec)`` and ``vec == 1`` for a scalar field, so the naive
    stack carries a phantom component axis: ``(n_quad, 1)`` for one direction, ``(n_quad, 1, n_dims)``
    for several. ``value_shape == ()`` says there is no component axis, so drop it — the value branch
    makes exactly the same promise for the field itself, and the TEST-function branch already uses this
    convention (``grads[..., d]``, stacked last), so trial and test now agree.

    In a weak term the phantom axis was harmless (``_prefix_align`` inserts singletons after the quad
    axis, absorbing it), which is why it survived. It is *not* harmless in a readout: an evolution
    formula like ``maximum(H.i(-1), 0.5*inner(grad(u,X), grad(u,X), 1))`` compares a genuinely scalar
    ``(n_quad,)`` buffer slice against a ``(n_quad, 1)`` energy and rank-broadcasts to
    ``(n_quad, n_quad)`` — the phase-field driving force, silently the wrong shape."""
    comps = [g[..., 0] for g in grad_list]
    return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)


def _expand_test_shape_vals(shape_vals, n_comp):
    if n_comp == 1:
        return shape_vals
    eye = jnp.eye(n_comp, dtype=shape_vals.dtype)
    return shape_vals[:, :, None, None] * eye[None, None, :, :]


def _infer_trial_metadata(expr) -> Dict[str, Any]:
    """
    Infer the FEM unknown metadata from TrialFunction nodes.

    Returns the unique trial symbol, its value shape, vector size, and whether
    the expression contains a TrialFunction.
    """
    trial_nodes = {}

    def walk(node):
        if node is None:
            return

        if isinstance(node, TrialFunction):
            trial_nodes[node.op_id] = node
            return

        for child in iter_children(node):
            walk(child)

    walk(expr)

    unique_trials = list(trial_nodes.values())
    if len(unique_trials) > 1:
        raise NotImplementedError(
            "The FEM assembler currently supports exactly one TrialFunction "
            "(scalar or vector valued). Multiple coupled FEM unknowns will "
            "come in the next refactor step."
        )

    trial = unique_trials[0] if unique_trials else None
    value_shape = getattr(trial, "value_shape", ()) if trial is not None else ()
    vec = _value_shape_num_components(value_shape)

    return {
        "trial": trial,
        "value_shape": value_shape,
        "vec": vec,
        "has_trial": trial is not None,
    }


def _infer_fields(expr) -> Tuple[List[Dict[str, Any]], Dict[Any, int]]:
    """All distinct trial *fields* in ``expr``, ordered by first appearance.

    A field is one coupled unknown (one ``fem_symbols()`` call — its trial and test
    share a ``field_key``). Returns ``(fields, field_key->index)`` where each field is
    ``{field_key, value_shape, vec, order}``. The index order is the single source of
    truth threaded into the per-field DOF blocks and the multi-field kernel.
    """
    fields: List[Dict[str, Any]] = []
    seen: Dict[Any, int] = {}

    def walk(node):
        if node is None:
            return
        if isinstance(node, TrialFunction):
            key = getattr(node, "field_key", node.op_id)
            if key not in seen:
                seen[key] = len(fields)
                vs = getattr(node, "value_shape", ())
                fields.append(
                    {
                        "field_key": key,
                        "value_shape": vs,
                        "vec": _value_shape_num_components(vs),
                        "order": int(getattr(node, "order", 1)),
                        "space": str(getattr(node, "space", "Lagrange")),
                    }
                )
            return
        for child in iter_children(node):
            walk(child)

    walk(expr)
    return fields, dict(seen)


def _test_field_index(expr, field_index: Dict[Any, int]) -> Optional[int]:
    """Index of the single test field in an additive weak term (or ``None``).

    Each additive term must contain exactly one test field (it determines the
    equation/row block); a term with zero or several distinct test fields is
    ambiguous and the caller errors."""
    keys = set()

    def walk(node):
        if node is None:
            return
        if isinstance(node, TestFunction):
            keys.add(getattr(node, "field_key", node.op_id))
            return
        for child in iter_children(node):
            walk(child)

    walk(expr)
    if len(keys) != 1:
        return None
    return field_index.get(next(iter(keys)))


def _expand_product_terms(node, sign: float = 1.0):
    """Distribute products/quotients over sums → a list of ``(sign, product_node)``.

    Used only as a fallback when a coupled weak term carries several test fields
    welded inside a product — e.g. the **real part of a complex weak form**, where
    a coefficient multiplies a sum that straddles two test fields,
    ``c·(u_r·p_r − u_i·q_i)``. Fully distributing yields ``c·u_r·p_r`` and
    ``−c·u_i·q_i``, each of which classifies to a single test field/equation block.
    The cross-product can grow, but weak-form terms are small and this runs only on
    the (rare) terms that would otherwise be rejected."""
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _expand_product_terms(node.left, sign) + _expand_product_terms(node.right, sign)
        if node.op == "-":
            return _expand_product_terms(node.left, sign) + _expand_product_terms(node.right, -sign)
        if node.op == "*":
            left = _expand_product_terms(node.left, 1.0)
            right = _expand_product_terms(node.right, 1.0)
            return [(sign * sl * sr, BinaryOp("*", lt, rt)) for sl, lt in left for sr, rt in right]
        if node.op == "/":
            # distribute the numerator only; the denominator is treated as a coefficient
            return [(s, BinaryOp("/", t, node.right)) for s, t in _expand_product_terms(node.left, sign)]
    return [(sign, node)]


def _gather_runtime_parameter_tags(node, out=None):
    """Collect names of trainable runtime parameters (ModelCall) used in a coeff."""
    from .parametric_helpers import _is_runtime_scalar_parameter, _parameter_name

    if out is None:
        out = []
    if _is_runtime_scalar_parameter(node):
        name = _parameter_name(node)
        if name not in out:
            out.append(name)
        return out
    for child in iter_children(node) or ():
        _gather_runtime_parameter_tags(child, out)
    return out


def _collect_region_mask_names(node, out=None):
    """Sorted-unique region names appearing as ``RegionMask`` leaves in a lowered expression/IR.

    Used so the volume kernel knows the per-cell mask layout and the assembler builds the matching
    masks (see ``_cell_region_mask``). Order is fixed by sorting so kernel and assembler agree."""
    if out is None:
        out = set()
    if isinstance(node, RegionMask):
        out.add(node.region)
    for child in iter_children(node) or ():
        _collect_region_mask_names(child, out)
    return out


def _collect_tag_mask_names(node, out=None):
    """Sorted-unique tag names appearing as ``TagMask`` leaves in a lowered expression/IR.

    The surface twin of :func:`_collect_region_mask_names`: the assembler uses it to build only the
    per-facet masks a term actually references."""
    if out is None:
        out = set()
    if isinstance(node, TagMask):
        out.add(node.tag)
    for child in iter_children(node) or ():
        _collect_tag_mask_names(child, out)
    return out


def _cell_region_mask(domain, region):
    """``(num_cells,)`` 0/1 indicator: a mesh cell is in ``region`` iff its **centroid** is.

    ``region`` is a geometry part (``domain._source_regions`` shapely polygon) or a ``domain.tag``
    predicate. Concrete (numpy/shapely), built once per assembly -- exact when the mesh respects the
    region boundaries (gmsh meshes each part separately, so no cell straddles a material interface).

    Classifies against the **assembly mesh** (the cell order the kernel vmaps over, stashed by
    ``_build_fem_problem``) so the per-cell mask aligns with the kernel's cells; falls back to the
    domain mesh only if no assembly mesh is recorded yet."""
    dim = int(domain.dimension)
    a_pts = getattr(domain, "_fem_assembly_points", None)
    a_cells = getattr(domain, "_fem_assembly_cells", None)
    if a_pts is not None and a_cells is not None:
        pts = np.asarray(a_pts)[:, :dim]
        cells = np.asarray(a_cells)
    else:
        mesh = domain.mesh
        pts = np.asarray(mesh.points)[:, :dim]
        cells = np.asarray(mesh.cells_dict["triangle" if dim == 2 else "tetra"])
    centroids = pts[cells].mean(axis=1)  # (num_cells, dim)
    src = getattr(domain, "_source_regions", {}) or {}
    preds = getattr(domain, "_tag_predicates", {}) or {}
    shape_regions = getattr(domain, "_shape_regions", {}) or {}
    if region in src:
        from shapely import contains_xy

        m = np.asarray(contains_xy(src[region], centroids[:, 0], centroids[:, 1]))
    elif region in preds:
        m = np.asarray(preds[region](*[centroids[:, i] for i in range(dim)]))
    elif region in shape_regions:
        # A Shape.regions sub-region: analytic CSG membership of the cell centroid (2-D and 3-D),
        # MINUS every higher-priority region, because Shape.regions lets regions overlap and resolves
        # them by declaration order -- a cell belongs to the FIRST region containing it, which is how
        # the mesh itself is labelled (`emit._to_meshio`).
        #
        # Without the subtraction these masks are not a partition, and `domain.by_region` (a sum of
        # RegionMask * value) silently double counts. The failure is quiet and physical rather than an
        # error: an enclosing background region -- e.g. a bounding rect declared last to pick up the
        # leftover void -- contains EVERY cell, so its coefficient is added to every other region's.
        # Measured on the furnace: the graphite-felt insulation ran at k = 0.5 + 0.186 instead of 0.5,
        # a 37% leak that pulled the whole crystal region 780 K cold.
        names = list(shape_regions)
        m = np.asarray(shape_regions[region].contains(centroids), dtype=bool)
        for earlier in names[: names.index(region)]:
            try:
                m &= ~np.asarray(shape_regions[earlier].contains(centroids), dtype=bool)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"jno.fem per-region integration: region {region!r} is declared after {earlier!r}, "
                    f"so resolving it needs to exclude {earlier!r}'s cells -- but {earlier!r} has no "
                    f"closed-form point membership. Declare it first, or give it a CSG-representable "
                    f"shape."
                ) from exc
    else:
        raise ValueError(
            f"jno.fem per-region integration: unknown region {region!r}. Define it with "
            f"domain.tag(name, predicate), a Shape.regions() sub-region, or a geometry part."
        )
    return np.asarray(m, dtype=bool).astype(np.float64)


def _gather_temporal_tags(node, out=None):
    """
    Collect temporal Variable tags used inside the kernels.

    These tags determine which time values must be passed through the
    batched volume-variable arrays during transient assembly.
    """
    if out is None:
        out = set()

    if isinstance(node, Variable) and getattr(node, "axis", None) == "temporal":
        out.add(str(node.tag))
        return out

    if isinstance(node, BinaryOp):
        _gather_temporal_tags(node.left, out)
        _gather_temporal_tags(node.right, out)
        return out

    if isinstance(node, FunctionCall):
        for a in node.args:
            if isinstance(
                a,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                _gather_temporal_tags(a, out)
        return out

    if isinstance(node, Jacobian):
        _gather_temporal_tags(node.target, out)
        for v in node.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                _gather_temporal_tags(v, out)
        return out

    if isinstance(node, Hessian):
        _gather_temporal_tags(node.target, out)
        for v in node.variables:
            if isinstance(
                v,
                (
                    Literal,
                    Constant,
                    TensorTag,
                    Variable,
                    TestFunction,
                    TrialFunction,
                    Jacobian,
                    Hessian,
                    BinaryOp,
                    FunctionCall,
                    ModelCall,
                    OperationDef,
                    OperationCall,
                    Tracker,
                    StateField,
                ),
            ):
                _gather_temporal_tags(v, out)
        return out

    return out


def _runtime_parameter_value_from_internal_vars(local, name):
    """Read a runtime parameter scalar from volume_vars.

    Parameter values are packed AFTER the temporal values, so the volume_vars
    layout is:  [ temporal_tags ... , runtime_parameter_tags ... ].
    """
    temporal_tags = local.get("temporal_tags", ())
    param_tags = local.get("runtime_parameter_tags", ())
    volume_vars = local.get("volume_vars", ())
    if name not in param_tags:
        return None
    idx = len(temporal_tags) + param_tags.index(name)
    if idx >= len(volume_vars):
        return None
    arr = jnp.asarray(volume_vars[idx])
    flat = jnp.reshape(arr, (-1,))
    # Scalar coefficient (incl. per-cell-constant): one value -> broadcast to quad.
    if flat.shape[0] == 1:
        return flat[0]
    # Node-based field coefficient: the kernel has gathered the cell's local nodal values
    # (size = nodes-per-element); interpolate to quadrature points with the field's
    # shape functions, mirroring the solution interpolation (shape_vals . nodal).
    # Returns (n_quad, 1) like a field value.
    shape_vals = local.get("shape_vals")  # (n_quad, n_local)
    cell_nodal = flat.reshape(flat.shape[0], 1)  # (n_local, 1)
    return jnp.sum(shape_vals[:, :, None] * cell_nodal[None, :, :], axis=1)


def _temporal_value_from_internal_vars(local, tag, dim_start=0, dim_end=1):
    """
    Read a temporal variable value from the batched volume-variable arrays.

    Returns None when the requested temporal tag is not part of the current
    kernel call.
    """
    temporal_tags = local.get("temporal_tags", ())
    volume_vars = local.get("volume_vars", ())

    if tag not in temporal_tags:
        return None

    idx = temporal_tags.index(tag)
    if idx >= len(volume_vars):
        raise IndexError(
            f"Temporal variable tag '{tag}' mapped to slot {idx}, but only {len(volume_vars)} volume_vars were provided."
        )

    arr = jnp.asarray(volume_vars[idx])
    # scalar / (1,) / (1,1) -> one scalar time for this assembly call
    t_scalar = jnp.reshape(arr, (-1,))[0]
    out = jnp.asarray([t_scalar])
    return out[dim_start:dim_end]


# --------------------------------
# Expression evaluation helpers
# --------------------------------


def _prefix_align(a, b):
    """Broadcast-align two kernel quantities for an elementwise op.

    In the kernel, trial-derived quantities are laid out as ``(n_quad,
    *value)`` while test-derived ones carry extra leading test-DOF axes
    ``(n_quad, *dof, *value)`` (the per-DOF basis expansion). Their shared axis
    is the leading quadrature axis and their value axes are trailing, so when the
    ranks differ we pad the lower-rank operand with singleton axes **right after
    the quad axis** until the ranks match. The trailing value axes then align by
    normal right-broadcasting and the test-DOF axes broadcast against the inserted
    singletons. This mirrors the prefix-padding ``jno.np.inner`` already does, so
    arbitrary tensor algebra between trial- and test-derived quantities works
    (e.g. ``div(u) * div(phi)``), not just explicit contractions.

    Only activates when ranks differ (and both operands are arrays), so
    equal-rank expressions keep their exact current broadcasting.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if a.ndim == b.ndim or a.ndim == 0 or b.ndim == 0:
        return a, b
    if a.ndim < b.ndim:
        pad = (1,) * (b.ndim - a.ndim)
        a = jnp.reshape(a, a.shape[:1] + pad + a.shape[1:])
    else:
        pad = (1,) * (a.ndim - b.ndim)
        b = jnp.reshape(b, b.shape[:1] + pad + b.shape[1:])
    return a, b


def _field_data(local, node):
    """``(shape_vals, shape_grads, cell_sol)`` for ``node``'s field.

    A multi-field kernel puts per-field arrays in ``local["fields"]`` (indexed by
    field index) and a ``field_key -> index`` map in ``local["field_index"]``. A
    single-field kernel has neither, so this falls back to the flat ``local`` entries
    — leaving the single-field evaluation path byte-identical."""
    fields = local.get("fields")
    if fields is None:
        return local["shape_vals"], local.get("shape_grads"), local.get("cell_sol")
    key = getattr(node, "field_key", getattr(node, "op_id", None))
    fd = fields[local["field_index"][key]]
    return fd["shape_vals"], fd["shape_grads"], fd["cell_sol"]


def _frozen_cell_values(local, node):
    """This cell's frozen nodal slice ``(n_local, vec)`` for a :class:`FrozenField`.

    The native kernel gathers each frozen field's per-cell nodal values (a compile-time
    constant -- no ``args`` threading, no ``jacfwd`` tangent) into
    ``local["frozen_fields"][frozen_id]``; here we just read it back to interpolate the
    known value / contract the known gradient with the field's shape data."""
    table = local.get("frozen_fields")
    if table is None or node.frozen_id not in table:
        raise NotImplementedError(
            "A frozen field (ui.freeze(values)) was used in a weak form assembled by a path that does "
            "not thread frozen fields. It is currently wired for the native steady/linear volume path."
        )
    return table[node.frozen_id]


def _field_hess(local, node):
    """Physical shape-function Hessian ``(n_quad, n_dof, dim, dim)`` for ``node``'s field, or ``None`` if
    its element does not tabulate second derivatives (only nodal Lagrange does). Mirrors :func:`_field_data`."""
    fields = local.get("fields")
    if fields is None:
        return local.get("shape_hess")
    key = getattr(node, "field_key", getattr(node, "op_id", None))
    return fields[local["field_index"][key]].get("shape_hess")


def _field_space(local, node):
    """Element family of ``node``'s field (``"Lagrange"`` default).

    Non-nodal families (``"RT"``, ...) are assembled by the native push-forward path
    (:mod:`fem_nonnodal`), which tags each ``local["fields"]`` entry with its ``space`` and
    supplies *physical* (push-forward) shape data. The value branches below switch on this so
    the Lagrange path stays byte-identical (a single-field Lagrange kernel has no ``fields``)."""
    fields = local.get("fields")
    if fields is None:
        return local.get("space", "Lagrange")
    key = getattr(node, "field_key", getattr(node, "op_id", None))
    return fields[local["field_index"][key]].get("space", "Lagrange")


def _eval_frozen_coefficient(domain, model, local):
    """Value of a ``.freeze()``d (known) coefficient at the kernel's quadrature points.

    The known value is supplied via ``.initialize`` and read here at assembly time:

    * a **coordinate function** ``(x, y[, z]) -> value`` (``.initialize(lambda x, y: ...)``) is called
      on the physical quadrature coordinates — identical to ``jno.fn`` — and may return a scalar or a
      vector (tuple / last-axis array) field;
    * a **scalar constant** (``.initialize(0.8)``) is returned as-is.

    A raw per-node value array is *not* supported (turning scattered nodal data into a coefficient
    needs mesh interpolation — use ``jno.fn`` or a function instead); a frozen parameter with no value
    fails loud."""
    fn = getattr(model, "_initializer_fn", None)
    if fn is not None:  # coordinate function, evaluated at the quad points (like jno.fn)
        import inspect

        try:
            params = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = set()
        if {"key", "shape"} & params:  # a JAX initializer (key, shape, dtype), not a coordinate fn
            raise ValueError(
                "jno.fem: a frozen (.freeze()d) coefficient takes a *scalar* or a coordinate function "
                "(x, y[, z]) -> value — not a JAX initializer. For a uniform value use .initialize(<number>); "
                "JAX initializers (jax.nn.initializers.*) are for *trainable* parameters."
            )
        qp = local["physical_quad_points"]
        dim = int(getattr(domain, "dimension", qp.shape[-1]))
        return jnp.asarray(fn(*(qp[..., i] for i in range(dim))))

    weight = getattr(model, "_weight_tree", None)
    if weight is not None:
        arr = jnp.asarray(weight)
        if arr.ndim == 0:  # scalar constant
            return arr
        raise NotImplementedError(
            "jno.fem: a frozen (.freeze()d) coefficient must be a constant or a (x, y[, z]) -> value "
            "function — a raw per-node array is not supported (use jno.fn(...) or a function)."
        )

    raise ValueError(
        f"jno.fem: frozen parameter {getattr(model, '_parameter_name', '<param>')!r} has no value — "
        "call .initialize(<scalar or (x, y[, z]) -> value function>) before .freeze()."
    )


def _call_neural_coefficient(model, module, arg_vals, local):
    """Evaluate one neural coefficient (``jno.nn.wrap(net)`` inside a weak form) at the quad points.

    ``arg_vals`` are the network's arguments already evaluated by ``_eval_integrand`` — coordinate
    Variables arrive as ``(n_quad, 1)`` slices of the physical quadrature points, an interpolated
    trial value as ``(n_quad, c)``, its gradient as ``(n_quad, len(dims))`` — so ``k(x)``, ``k(u)``
    and ``k(∇u)`` all share this call. Everything is normalised to a ``(n_quad, feat)`` batch (the
    same convention as the point-cloud evaluator ``TraceEvaluator._eval_flax_module_call``, which
    foundax MLPs concatenate internally), cast to the model's explicit ``.dtype`` opt-in when set,
    and the network output is returned as ``(n_quad, k)`` — the shape a nodal field parameter
    returns, so it composes with test-expanded factors via ``_prefix_align`` identically.

    ``module`` is the *current* weight pytree threaded through the runtime ``args`` (a
    ``ModelWeights`` slot), so under ``jax.jacfwd`` w.r.t. the cell DOFs a trial-dependent argument
    carries tangents straight through the network — ∂k(u)/∂u enters the element Jacobian
    automatically — and under the outer training loop the weights are the crux-recombined
    trainable leaves, so gradients reach the optimizer.

    Neural-coefficient support follows the unsupervised coefficient/constitutive-recovery setting
    of NN-EUCLID (M. Flaschel, S. Kumar, L. De Lorenzis, "NN-EUCLID: Deep-learning hyperelasticity
    without stress data", J. Mech. Phys. Solids 165 (2022) 105076, §2.2–2.3) and Tartakovsky et
    al., "Learning Parameters and Constitutive Relationships with Physics-Informed Deep Neural
    Networks" (Water Resour. Res. 56, 2020, §2).
    """
    n_quad = int(local["physical_quad_points"].shape[0])

    def _as_quad_batch(v):
        v = jnp.asarray(v)
        if v.ndim == 0:
            return jnp.broadcast_to(v.reshape(1, 1), (n_quad, 1))
        if v.ndim == 1:
            if v.shape[0] == n_quad:
                return v.reshape(n_quad, 1)
            if v.shape[0] == 1:
                return jnp.broadcast_to(v.reshape(1, 1), (n_quad, 1))
        if v.ndim == 2 and v.shape[0] == n_quad:
            return v
        if v.ndim == 2 and v.shape[0] == 1:
            return jnp.broadcast_to(v, (n_quad, v.shape[1]))
        raise NotImplementedError(
            "jno.fem: a neural coefficient's arguments must be per-quad-point values (coordinates, "
            f"the trial or its derivatives, scalars); got an argument of shape {v.shape}. A test "
            "function cannot appear inside a network argument."
        )

    vals = [_as_quad_batch(v) for v in arg_vals]

    # Mixed precision: honour an explicit Model.dtype() opt-in exactly like the point evaluator —
    # cast floating inputs so the net computes in its declared dtype; a plain-f32 net under x64
    # promotes through the surrounding arithmetic instead.
    compute_dtype = getattr(model, "_dtype", None)
    if compute_dtype is not None:
        vals = [
            v.astype(compute_dtype) if jnp.issubdtype(v.dtype, jnp.floating) and v.dtype != compute_dtype else v
            for v in vals
        ]

    out = module(*vals)
    if hasattr(out, "output"):  # structured foundation-model outputs
        out = out.output
    out = jnp.asarray(out)
    if out.ndim == 0:
        return jnp.broadcast_to(out.reshape(1, 1), (n_quad, 1))
    if out.ndim == 1 and out.shape[0] == n_quad:
        return out.reshape(n_quad, 1)
    if out.ndim == 2 and out.shape[0] == n_quad:
        return out
    raise NotImplementedError(
        f"jno.fem: a neural coefficient must return one value (or a feature vector) per quadrature "
        f"point — got output shape {out.shape} for {n_quad} quad points."
    )


def _eval_integrand(domain, node, local):
    """
    Evaluate a jNO symbolic expression inside a local kernel.

    The `local` dictionary contains quadrature coordinates, shape values,
    shape gradients, local cell DOFs, domain context, and the optional batched
    temporal volume-variable arrays. This evaluator supports literals, constants,
    variables, tensor tags, TrialFunction/TestFunction values, their Jacobians,
    binary operations, and FunctionCall nodes.
    """
    if not isinstance(
        node,
        (
            Literal,
            Constant,
            TensorTag,
            RegionMask,
            TagMask,
            Variable,
            TestFunction,
            TrialFunction,
            FrozenField,
            Jacobian,
            Hessian,
            BinaryOp,
            FunctionCall,
            ModelCall,
        ),
    ):
        try:
            return jnp.asarray(node)
        except Exception:
            pass

    if isinstance(node, Literal):
        return jnp.asarray(node.value)

    if isinstance(node, Constant):
        return jnp.asarray(node.value)

    if isinstance(node, RegionMask):
        # Per-cell sub-region indicator: read the current cell's 0/1 value from the constant per-cell
        # volume_var packed AFTER the temporal + runtime-parameter slots. Fail loud (never silently
        # integrate over the whole domain) if this assembly path did not thread the mask.
        mask_names = local.get("region_mask_names", ())
        volume_vars = local.get("volume_vars", ())
        idx = (
            len(local.get("temporal_tags", ()))
            + len(local.get("runtime_parameter_tags", ()))
            + (list(mask_names).index(node.region) if node.region in mask_names else -1)
        )
        if node.region not in mask_names or idx >= len(volume_vars):
            raise NotImplementedError(
                f"jno.fem per-region integration: the per-cell mask for region '{node.region}' was not "
                f"threaded into this assembly path. Per-region terms ARE supported in steady, nonlinear, "
                f"transient, multifield, parametric and 3D forms (see tests/test_fem_per_region.py); if you "
                f"reach this on such a form, the mask wiring for that specific path is missing — please report it."
            )
        return jnp.reshape(jnp.asarray(volume_vars[idx]), (-1,))[0]

    if isinstance(node, TagMask):
        # Per-facet tag indicator, already sliced to THIS boundary face by the surface kernel. Fail
        # loud rather than defaulting to 1 (which would integrate the term over the whole boundary) or
        # to 0 (which would drop it) -- both are silent physics errors. This is also what rejects a
        # TagMask in a VOLUME term, where `tag_masks` is never populated because there is no facet.
        tag_masks = local.get("tag_masks", None)
        if not tag_masks or node.tag not in tag_masks:
            raise NotImplementedError(
                f"jno.fem per-tag surface integration: the per-facet mask for tag '{node.tag}' was not "
                f"threaded into this assembly path. `domain.by_tag` builds a coefficient for a SURFACE "
                f"term -- one bound to a boundary tag's coordinates. Using it in a volume term, on a "
                f"non-nodal space, or in 1-D is not supported (see docs/fem.md)."
            )
        return jnp.asarray(tag_masks[node.tag])

    if isinstance(node, TensorTag):
        if node.tag not in local["domain_context"]:
            raise KeyError(f"TensorTag '{node.tag}' not found in FEM domain context.")
        tensor = jnp.asarray(local["domain_context"][node.tag])
        if tensor.ndim >= 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.ndim >= 1 and tensor.shape[0] > 1:
            raise NotImplementedError(
                "The FEM assembler currently supports singleton-batch TensorTag coefficients only. "
                f"Got shape {tensor.shape} for tag '{node.tag}'."
            )
        if node.dim_index is not None and tensor.ndim >= 1:
            tensor = tensor[..., node.dim_index]
        return tensor

    if isinstance(node, Variable):
        dim_start, dim_end = node.dim

        # Local quadrature coordinates
        if local.get("surface", False):
            if isinstance(node.tag, str) and node.tag.startswith("gauss_"):
                return local["physical_quad_points"][..., dim_start:dim_end]
        else:
            if node.tag == "fem_gauss":
                return local["physical_quad_points"][..., dim_start:dim_end]

        # Temporal variable in assembly:
        # prefer the batched volume_vars arrays (pure JAX / no domain mutation),
        # then fall back to domain.context for the steady-context path.
        if getattr(node, "axis", None) == "temporal":
            from_iv = _temporal_value_from_internal_vars(
                local,
                str(node.tag),
                dim_start=dim_start,
                dim_end=dim_end,
            )
            if from_iv is not None:
                return from_iv

            if node.tag not in local["domain_context"]:
                raise KeyError(f"Temporal Variable tag '{node.tag}' not found in local/domain context.")
            arr = jnp.asarray(local["domain_context"][node.tag])
            t_scalar = jnp.reshape(arr, (-1,))[0]
            out = jnp.asarray([t_scalar])
            return out[dim_start:dim_end]

        # Fallback to stored tensor/point-data context
        if node.tag in local["domain_context"]:
            arr = jnp.asarray(local["domain_context"][node.tag])
            if arr.ndim >= 1 and arr.shape[0] == 1:
                arr = arr[0]
            return arr[..., dim_start:dim_end]

        raise KeyError(f"Variable tag '{node.tag}' not found in local/domain context.")

    if isinstance(node, TestFunction):
        shape_vals, _, _ = _field_data(local, node)
        # Contact reaction: the main body's share of an interface traction is the SAME integrand
        # tested against the main's projected trace. Test and trial otherwise read one table, so the
        # substitution has to happen here -- the trial values must keep the secondary face's own basis.
        _ov = local.get("test_shape_vals")
        if _ov:
            _key = getattr(node, "field_key", getattr(node, "op_id", None))
            _fi = (local.get("field_index") or {}).get(_key)
            if _fi in _ov:
                shape_vals = _ov[_fi]
        if _field_space(local, node) != "Lagrange":
            # non-nodal: shape_vals is already the per-DOF *physical* basis (n_quad, n_dof, *value)
            return shape_vals
        n_comp = _value_shape_num_components(getattr(node, "value_shape", ()))
        return _expand_test_shape_vals(shape_vals, n_comp)

    if isinstance(node, TrialFunction):
        vals, _, cell_sol = _field_data(local, node)
        if _field_space(local, node) != "Lagrange":
            # non-nodal: u = sum_n cell_sol[n] Phi_n(x). Vector basis (RT) is (n_quad, n_dof, value_size);
            # scalar basis (P0/DG) is (n_quad, n_dof).
            if vals.ndim == 2:
                return jnp.einsum("qn,n->q", vals, cell_sol)
            return jnp.einsum("qnc,n->qc", vals, cell_sol)
        flat_interp = jnp.sum(vals[:, :, None] * cell_sol[None, :, :], axis=1)
        value_shape = getattr(node, "value_shape", ())
        if len(value_shape) == 0:
            # A scalar field is ``(n_quad,)``, NOT ``(n_quad, 1)`` -- ``value_shape == ()`` says there is
            # no component axis, so the interpolation must not invent one. Keeping it is invisible in a
            # weak term (it contracts with the test function) but RANK-BROADCASTS against anything
            # genuinely scalar: a scalar ``state.i(-1)`` buffer slice is ``(n_quad,)``, so
            # ``maximum(H.i(-1), u)`` silently became ``(n_quad, n_quad)`` and the state readout then
            # produced a buffer of the wrong rank. Squeeze here, at the one place the axis is created.
            return flat_interp[..., 0]
        return _reshape_components_last(flat_interp, value_shape)

    if isinstance(node, DiffSlot):
        # The hole a `Diff` differentiates through: its value is injected by the branch below.
        slots = local.get("__diff_slots__") or {}
        if node.key not in slots:
            raise RuntimeError(
                f"jno.fem: a diff value slot ({node.key}) was evaluated outside its `jno.np.diff(...)`. "
                "A DiffSlot is internal — it must not appear in a weak form on its own."
            )
        return slots[node.key]

    if isinstance(node, Diff):
        # d(target)/d(wrt), evaluated POINTWISE at this cell's quadrature points.
        #
        # `wrt` is evaluated to a concrete array (n_quad, *value_shape); `target` is re-evaluated with
        # `wrt` swapped for a value slot, so it becomes an ordinary JAX function of that array. Because a
        # constitutive energy is pointwise in `wrt`, the quadrature axis is a batch axis and the Jacobian
        # of the SUMMED scalar is block-diagonal — so `grad(sum(...))` returns exactly d(target_q)/d(wrt_q)
        # per point, with `wrt`'s own shape and no vmap or reshape. Forward-over-reverse through the
        # element `jacfwd` then yields the consistent tangent d(P)/d(F) for free.
        wrt_val = _eval_integrand(domain, node.wrt, local)
        rewritten = node.rewritten()

        def _scalar_of(value):
            sub = {**(local.get("__diff_slots__") or {}), node._slot.key: value}
            out = _eval_integrand(domain, rewritten, {**local, "__diff_slots__": sub})
            return jnp.sum(out)

        return jax.grad(_scalar_of)(wrt_val)

    if isinstance(node, HistoryRef):
        # STEP-history read ``v.i(k)``: the driver threads this cell's per-quadrature-point buffer slice
        # (shape (n_quad, depth, *value_shape)) as a per-cell constant. We just pick the offset's slot —
        # no shape-function interpolation (history lives AT the quad points, not at nodes). Because the
        # slice is indexed by the cell (not the local DOFs), the per-element ``jacfwd`` sees it as a
        # constant, so the tangent is ``∂σ/∂ε`` with history frozen (correct within a load step).
        table = local.get("qp_history")
        if table is None or node.history_key not in table:
            raise NotImplementedError(
                f"jno.fem: history read {node.name!r} has no per-quadrature-point buffer — this assembly "
                "path does not allocate or thread step history. `.i(k)` history is wired on the real, "
                "steady, native-Lagrange path (2D/3D, single-field or coupled), marched over a "
                "`domain(tau=(start, end, n))` pseudo-time grid by a plain `fem.solve()` — nothing is "
                "passed to it. Not carried by: 1D, non-nodal (Argyris/Morley/edge) elements, VPINN, "
                "periodic ties, a `u.t` transient, or a complex form."
            )
        buf_c = table[node.history_key]  # (n_quad, depth, *value_shape) for this cell
        return buf_c[:, -node.offset - 1]  # offset -1 -> slot 0, -2 -> slot 1, ...  -> (n_quad, *value_shape)

    if isinstance(node, FrozenField):
        # KNOWN field: interpolate its frozen nodal slice at the quad points, exactly like the
        # TrialFunction value branch but with frozen values instead of the live cell solution.
        if _field_space(local, node) != "Lagrange":
            raise NotImplementedError("ui.freeze(values): frozen fields are supported for nodal Lagrange only.")
        vals, _, _ = _field_data(local, node)
        fz = _frozen_cell_values(local, node)  # (n_local, vec)
        flat_interp = jnp.sum(vals[:, :, None] * fz[None, :, :], axis=1)
        value_shape = getattr(node, "value_shape", ())
        if len(value_shape) == 0:
            return flat_interp
        return _reshape_components_last(flat_interp, value_shape)

    if isinstance(node, Jacobian):
        dims = []
        for var in node.variables:
            if not isinstance(var, Variable):
                raise NotImplementedError(
                    "The FEM assembler expects Jacobian variables to be domain.variable(...) placeholders."
                )
            dims.append(var.dim[0])
        if len(dims) == 0:
            raise ValueError("Jacobian node has no differentiation variables")

        if isinstance(node.target, TestFunction):
            _, grads, _ = _field_data(local, node.target)
            if _field_space(local, node.target) != "Lagrange":
                # non-nodal: grads is the per-DOF *physical* gradient (n_quad, n_dof, n_comp, n_dims);
                # pick the requested directions -> (n_quad, n_dof, n_comp[, len(dims)]). trace() then gives div.
                g = jnp.stack([grads[..., d] for d in dims], axis=-1)
                return g[..., 0] if len(dims) == 1 else g
            n_comp = _value_shape_num_components(getattr(node.target, "value_shape", ()))
            if n_comp == 1:
                comps = [grads[..., dim0] for dim0 in dims]
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)
            eye = jnp.eye(n_comp, dtype=grads.dtype)
            comps = [grads[..., dim0][:, :, None, None] * eye[None, None, :, :] for dim0 in dims]
            if len(comps) == 1:
                return comps[0]
            return jnp.stack(comps, axis=-1)

        if isinstance(node.target, FrozenField):
            # gradient of a KNOWN field: contract the SAME physical shape gradients with the frozen
            # nodal slice instead of the live cell solution (byte-identical to the TrialFunction case).
            if _field_space(local, node.target) != "Lagrange":
                raise NotImplementedError("ui.freeze(values): frozen-field gradients are nodal-Lagrange only.")
            _, grads, _ = _field_data(local, node.target)
            fz = _frozen_cell_values(local, node.target)  # (n_local, vec)
            grad_list = [jnp.sum(grads[:, :, dim0 : dim0 + 1] * fz[None, :, :], axis=1) for dim0 in dims]
            value_shape = getattr(node.target, "value_shape", ())
            if len(value_shape) == 0:
                return _drop_scalar_component_axis(grad_list)
            flat = grad_list[0] if len(dims) == 1 else jnp.stack(grad_list, axis=-1)
            if len(dims) == 1:
                return _reshape_components_last(flat, value_shape)
            return jnp.reshape(flat, flat.shape[:1] + tuple(value_shape) + (len(dims),))

        if isinstance(node.target, TrialFunction):
            _, grads, cell_sol = _field_data(local, node.target)
            if _field_space(local, node.target) != "Lagrange":
                # non-nodal: du_i/dx_l = sum_n cell_sol[n] grad[n, i, l] -> (n_quad, n_comp[, len(dims)])
                g = jnp.stack([grads[..., d] for d in dims], axis=-1)
                contracted = jnp.einsum("qn...,n->q...", g, cell_sol)
                return contracted[..., 0] if len(dims) == 1 else contracted
            grad_list = [jnp.sum(grads[:, :, dim0 : dim0 + 1] * cell_sol[None, :, :], axis=1) for dim0 in dims]
            value_shape = getattr(node.target, "value_shape", ())
            if len(value_shape) == 0:
                return _drop_scalar_component_axis(grad_list)
            flat = grad_list[0] if len(dims) == 1 else jnp.stack(grad_list, axis=-1)
            if len(dims) == 1:
                return _reshape_components_last(flat, value_shape)
            return jnp.reshape(flat, flat.shape[:1] + tuple(value_shape) + (len(dims),))

        # Component-of-field gradient: ``u[i].d(x)`` lowers to ``Jacobian(getitem(field, i), [x])``.
        # For a NON-NODAL field the value-component cannot be differentiated directly, but
        # ``d(u_i)/dx_l`` IS the (component i, direction l) entry of the whole-field *physical*
        # gradient -- so select that row (this is what makes ``.div()`` / ``.curl()`` sugar work for
        # RT and N1E). For a NODAL Lagrange vector field the same quantity is the scalar-basis
        # gradient contracted against column ``i`` of the node-major cell solution -- which is what
        # lets a tensor-nonlinear form (finite-strain elasticity: ``F = I + du_i/dx_j``, ``det F``,
        # ``F^{-T}``) be written in the natural component spelling instead of forcing every vector
        # problem through ``inner``/``symgrad`` or a coupled-scalar workaround.
        tgt = node.target
        if isinstance(tgt, FunctionCall) and getattr(tgt, "getitem_key", None) is not None and len(tgt.args) == 1:
            field = tgt.args[0]
            ints = [k for k in tgt.getitem_key if isinstance(k, int)]
            if ints and isinstance(field, (TrialFunction, TestFunction)) and _field_space(local, field) != "Lagrange":
                comp = ints[-1]
                _, grads, cell_sol = _field_data(local, field)  # grads: (n_quad, n_dof, n_comp, n_dims)
                gc = grads[:, :, comp, :]  # physical gradient of component `comp` -> (n_quad, n_dof, n_dims)
                g = jnp.stack([gc[..., d] for d in dims], axis=-1)  # (n_quad, n_dof, len(dims))
                g = g[..., 0] if len(dims) == 1 else g
                if isinstance(field, TestFunction):
                    return g  # per-DOF directional derivative of the component
                return jnp.einsum("qn...,n->q...", g, cell_sol)  # contract the trial DOFs
            if ints and isinstance(field, (TrialFunction, TestFunction)):
                # Lagrange component gradient. Conventions match the whole-field branches exactly, so
                # component and whole-field spellings mix freely in one term:
                #   trial ``u[i].d(x_l)`` -> (n_quad,)                  a number per quad point
                #   test  ``v[i].d(x_l)`` -> (n_quad, n_local, n_comp)  nonzero only on the DOF-component
                #                                                       column ``i`` (node-major ravel)
                # A scalar field (n_comp == 1) takes the whole-field shapes, so ``u[0]`` on a scalar is
                # the field itself rather than a differently-shaped twin.
                comp = ints[-1]
                n_comp = _value_shape_num_components(getattr(field, "value_shape", ()))
                if not 0 <= comp < n_comp:
                    raise IndexError(
                        f"jno.fem: component {comp} is out of range for a field with "
                        f"value_shape={getattr(field, 'value_shape', ())} ({n_comp} component(s))."
                    )
                _, grads, cell_sol = _field_data(local, field)  # grads (n_quad, n_local, gdim); cell_sol (n_local, vec)
                if isinstance(field, TestFunction):
                    if n_comp == 1:
                        comps = [grads[..., d] for d in dims]  # (n_quad, n_local): the scalar convention
                    else:
                        onehot = jnp.zeros((n_comp,), dtype=grads.dtype).at[comp].set(1.0)
                        comps = [grads[..., d][:, :, None] * onehot[None, None, :] for d in dims]
                else:
                    comps = [jnp.sum(grads[..., d] * cell_sol[None, :, comp], axis=1) for d in dims]  # (n_quad,)
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

        if isinstance(node.target, ModelCall):
            # Gradient of a KNOWN (frozen) network coefficient: evaluate the network's spatial
            # gradient CONTINUOUSLY at the quad points (autodiff of the module w.r.t. its coordinate
            # inputs) — unlike a FrozenField, which is only a P1 nodal projection and carries no
            # sub-grid content. With no live trial in the term it is constant in the unknown, so it
            # lands in the RHS (b = -residual(0)), giving a(u_h, v) = L(v) - a(u_NN, v): the
            # finite-element correction to a network prior (FE-basis enrichment, Barucq et al. 2025).
            from .parametric_helpers import _neural_coefficient_name

            neural_modules = local.get("neural_coefficients")
            module = None if neural_modules is None else neural_modules.get(_neural_coefficient_name(node.target))
            if module is None:
                raise NotImplementedError(
                    "jno.fem: the gradient of a network coefficient needs the neural-coefficient assembly "
                    "path — a known/frozen net (net.freeze()) used alongside a real trial in the weak form."
                )
            args = node.target.args
            if not all(isinstance(a, Variable) for a in args):
                raise NotImplementedError(
                    "jno.fem: ∂net/∂x is supported for a network of coordinate variables only, e.g. jnn.grad(net(x, y), x)."
                )
            arg_dims = [a.dim[0] for a in args]
            pts = local["physical_quad_points"]  # (n_quad, gdim)

            def _net_scalar(pt):  # scalar network value at one physical point, differentiable in pt
                out = module(*[pt[d].reshape(1, 1) for d in arg_dims])
                out = out.output if hasattr(out, "output") else out
                return jnp.reshape(jnp.asarray(out), (-1,))[0]

            full_grad = jax.vmap(jax.grad(_net_scalar))(pts)  # (n_quad, gdim): ∂net/∂x_l at each quad point
            comps = [full_grad[:, dim0 : dim0 + 1] for dim0 in dims]  # each (n_quad, 1)
            return comps[0] if len(comps) == 1 else jnp.concatenate(comps, axis=-1)

        raise NotImplementedError("The FEM assembler supports gradients of TrialFunction/TestFunction only.")

    if isinstance(node, Hessian):
        dims = []
        for var in node.variables:
            if not isinstance(var, Variable):
                raise NotImplementedError(
                    "The FEM assembler expects Hessian variables to be domain.variable(...) placeholders."
                )
            if getattr(var, "axis", None) == "temporal":
                raise NotImplementedError(
                    "A temporal second derivative (u_tt) is handled by the second-order-in-time route, "
                    "not shape-Hessian assembly."
                )
            dims.append(var.dim[0])
        if len(dims) == 0:
            raise ValueError("Hessian node has no differentiation variables")
        if not isinstance(node.target, (TestFunction, TrialFunction)):
            raise NotImplementedError("The FEM assembler supports Hessians of TrialFunction/TestFunction only.")
        if _field_space(local, node.target) != "Lagrange":
            raise NotImplementedError("Second derivatives are assembled for nodal Lagrange fields only.")
        if _value_shape_num_components(getattr(node.target, "value_shape", ())) != 1:
            raise NotImplementedError("Hessian/Laplacian assembly currently supports scalar fields only.")
        hess = _field_hess(local, node.target)  # (n_quad, n_dof, dim, dim) physical shape Hessian
        if hess is None:
            raise NotImplementedError(
                "This element does not tabulate second derivatives -- use an order>=2 Lagrange field "
                "(a P1 Hessian is identically zero)."
            )
        da = jnp.asarray(dims)
        hsub = jnp.take(jnp.take(hess, da, axis=2), da, axis=3)  # (n_quad, n_dof, L, L) over requested dirs
        is_test = isinstance(node.target, TestFunction)
        if node.trace:  # Laplacian = sum over the selected diagonal directions
            lap = jnp.einsum("qnii->qn", hsub)  # (n_quad, n_dof) per-DOF Laplacian
            if is_test:
                return lap
            _, _, cell_sol = _field_data(local, node.target)
            return jnp.sum(lap[:, :, None] * cell_sol[None, :, :], axis=1)  # (n_quad, 1) trial Laplacian
        if is_test:
            return hsub  # (n_quad, n_dof, L, L) per-DOF Hessian (e.g. inner(hessian(u), hessian(v)))
        _, _, cell_sol = _field_data(local, node.target)
        return jnp.einsum("qnij,n->qij", hsub, cell_sol[:, 0])  # (n_quad, L, L) trial Hessian

    if isinstance(node, BinaryOp):
        a = _eval_integrand(domain, node.left, local)
        b = _eval_integrand(domain, node.right, local)
        a, b = _prefix_align(a, b)
        if node.op == "+":
            return a + b
        if node.op == "-":
            return a - b
        if node.op == "*":
            return a * b
        if node.op == "/":
            return a / b
        if node.op == "**":
            return a**b
        raise NotImplementedError(f"Unsupported binary operator: {node.op}")

    if isinstance(node, ModelCall):
        from .parametric_helpers import (
            _is_frozen_parameter,
            _is_runtime_scalar_parameter,
            _neural_coefficient_name,
            _parameter_name,
        )

        if _is_frozen_parameter(node):
            # A .freeze()d (known) coefficient: evaluate its constant / coordinate function at the
            # quadrature points -- exactly like jno.fn -- so it needs no runtime-parameter threading
            # and works in every assembly path (steady/transient/nonlinear/coupled).
            return _eval_frozen_coefficient(domain, node.model, local)
        if _is_runtime_scalar_parameter(node):
            name = _parameter_name(node)
            val = _runtime_parameter_value_from_internal_vars(local, name)
            if val is None:
                raise KeyError(
                    f"Runtime parameter '{name}' not supplied to the kernel. "
                    "Ensure it was registered in runtime_parameter_tags and packed "
                    "into InternalVars.volume_vars."
                )
            return val
        # Neural coefficient (``jno.nn.wrap(net)`` called inside the weak form): the assembler
        # threads a {name: module} table into ``local`` (trainable modules arrive through the
        # runtime ``args``; frozen ones fall back to their stored weights) and the network is
        # (re-)evaluated here on its kernel-evaluated arguments at the quadrature points.
        neural_modules = local.get("neural_coefficients")
        if neural_modules is not None:
            module = neural_modules.get(_neural_coefficient_name(node))
            if module is not None:
                arg_vals = [_eval_integrand(domain, a, local) for a in node.args]
                return _call_neural_coefficient(node.model, module, arg_vals, local)
        raise NotImplementedError(
            "jno.fem: a neural coefficient (jno.nn.wrap(net) inside the weak form) is supported on the "
            "native 2D/3D Lagrange assembler (single or coupled) and the scalar C¹ non-nodal families "
            "(Argyris/Morley/Hermite) — this assembly path does not thread network weights yet."
        )

    if isinstance(node, FunctionCall):
        args = [_eval_integrand(domain, arg, local) for arg in node.args]
        kwargs = node.kwargs if node.kwargs else {}
        return node.fn(*args, **kwargs)

    raise NotImplementedError(f"Unsupported weak-form node for the FEM assembler: {type(node).__name__}")


# --------------------------------
# Kernel builders
# --------------------------------


def _eval_volume_integrand(
    domain,
    expr,
    value_shape,
    cell_sol_flat,
    physical_quad_points,
    cell_shape_grads,
    cell_JxW,
    cell_v_grads_JxW,
    temporal_tags,
    runtime_parameter_tags,
    region_mask_names,
    problem_ref,
    *cell_internal_vars,
):
    """
    Evaluate and integrate one volume weak-form expression on one cell.

    Returns the flattened cell residual contribution the assembler expects.
    """
    num_nodes = cell_shape_grads.shape[1]
    vec = _value_shape_num_components(value_shape)
    cell_sol = cell_sol_flat.reshape(num_nodes, vec)

    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("The assembly problem_ref['problem'] was not initialized before kernel evaluation.")

    shape_vals = problem.fes[0].shape_vals
    local = {
        "physical_quad_points": physical_quad_points,
        "shape_vals": shape_vals,
        "shape_grads": cell_shape_grads,
        "cell_sol": cell_sol,
        "tag": "fem_gauss",
        "surface": False,
        "domain_context": domain.context,
        "trial_value_shape": value_shape,
        "trial_vec": vec,
        "temporal_tags": tuple(temporal_tags),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
        "region_mask_names": tuple(region_mask_names),
        "volume_vars": tuple(cell_internal_vars),
    }

    val = _eval_integrand(domain, expr, local)
    weights = cell_JxW[0]
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]


def _lookup_boundary_normals(domain, tag, physical_surface_quad_points):
    """Outward normals at the face quad points for ``tag`` (or ``None``).

    Nearest-neighbour lookup against ``domain.normals_by_tag`` (keyed by the sampled
    ``gauss_<tag>`` or ``<tag>``), matching the single-field surface path. Shared by
    the single- and multi-field surface integrands so normal-dependent boundary terms
    (e.g. a pressure traction ``p_ext * n``) behave identically in coupled problems."""
    if not hasattr(domain, "normals_by_tag"):
        return None
    normal_lookup_tag = f"gauss_{tag}" if f"gauss_{tag}" in domain.normals_by_tag else tag
    if normal_lookup_tag in domain.normals_by_tag and normal_lookup_tag in getattr(domain, "_mesh_pool", {}):
        normal_pts = jnp.asarray(np.asarray(domain._mesh_pool[normal_lookup_tag])[:, : domain.dimension])
        normal_vals = jnp.asarray(np.asarray(domain.normals_by_tag[normal_lookup_tag])[:, : domain.dimension])
        if len(normal_pts) > 0 and len(normal_pts) == len(normal_vals):
            x_use = physical_surface_quad_points[:, : domain.dimension]
            d2 = jnp.sum((normal_pts[None, :, :] - x_use[:, None, :]) ** 2, axis=-1)
            nn_idx = jnp.argmin(d2, axis=1)
            return normal_vals[nn_idx]
    return None


def _eval_surface_integrand(
    domain,
    expr,
    tag,
    value_shape,
    cell_sol_flat,
    physical_surface_quad_points,
    face_shape_vals,
    face_shape_grads,
    face_nanson_scale,
    temporal_tags,
    runtime_parameter_tags,
    *cell_internal_vars_surface,
):
    """
    Evaluate and integrate one boundary weak-form expression on one face.

    Returns the flattened surface residual contribution the assembler expects.
    """
    vec = _value_shape_num_components(value_shape)

    cell_sol_flat = jnp.asarray(cell_sol_flat)
    physical_surface_quad_points = jnp.asarray(physical_surface_quad_points)
    face_shape_vals = jnp.asarray(face_shape_vals)
    face_shape_grads = jnp.asarray(face_shape_grads)
    face_nanson_scale = jnp.asarray(face_nanson_scale)

    if cell_sol_flat.ndim != 1:
        cell_sol_flat = cell_sol_flat.reshape(-1)

    if cell_sol_flat.size % vec != 0:
        raise ValueError(f"Surface kernel DOF size {cell_sol_flat.size} is not divisible by vec={vec} for tag '{tag}'.")

    n_parent_nodes = cell_sol_flat.size // vec
    cell_sol = cell_sol_flat.reshape(n_parent_nodes, vec)

    if face_shape_vals.ndim != 2:
        raise ValueError(f"Expected face_shape_vals.ndim == 2, got shape {face_shape_vals.shape} for tag '{tag}'.")
    if face_shape_grads.ndim != 3:
        raise ValueError(f"Expected face_shape_grads.ndim == 3, got shape {face_shape_grads.shape} for tag '{tag}'.")
    if physical_surface_quad_points.ndim != 2:
        raise ValueError(
            f"Expected physical_surface_quad_points.ndim == 2, got shape {physical_surface_quad_points.shape} for tag '{tag}'."
        )

    nq = face_shape_vals.shape[0]
    if face_shape_vals.shape[1] != n_parent_nodes:
        raise ValueError(
            f"Boundary shape/node mismatch on '{tag}': "
            f"face_shape_vals.shape={face_shape_vals.shape}, "
            f"but cell_sol implies n_parent_nodes={n_parent_nodes}."
        )
    if face_shape_grads.shape[0] != nq or face_shape_grads.shape[1] != n_parent_nodes:
        raise ValueError(
            f"Boundary grad shape mismatch on '{tag}': "
            f"face_shape_grads.shape={face_shape_grads.shape}, "
            f"expected (nq={nq}, n_parent_nodes={n_parent_nodes}, dim)."
        )
    if physical_surface_quad_points.shape[0] != nq:
        raise ValueError(
            f"Boundary quadrature mismatch on '{tag}': "
            f"physical_surface_quad_points.shape={physical_surface_quad_points.shape}, "
            f"face_shape_vals.shape={face_shape_vals.shape}."
        )

    if face_nanson_scale.ndim == 2:
        weights = face_nanson_scale[0]
    elif face_nanson_scale.ndim == 1:
        weights = face_nanson_scale
    else:
        raise ValueError(f"Unsupported face_nanson_scale shape {face_nanson_scale.shape} for tag '{tag}'.")

    if weights.shape[0] != nq:
        raise ValueError(f"Boundary weight/quadrature mismatch on '{tag}': weights.shape={weights.shape}, nq={nq}.")

    boundary_normals = _lookup_boundary_normals(domain, tag, physical_surface_quad_points)

    local = {
        "physical_quad_points": physical_surface_quad_points,
        "shape_vals": face_shape_vals,
        "shape_grads": face_shape_grads,
        "cell_sol": cell_sol,
        "tag": tag,
        "surface": True,
        "domain_context": domain.context,
        "trial_value_shape": value_shape,
        "trial_vec": vec,
        "boundary_normals": boundary_normals,
        "temporal_tags": tuple(temporal_tags),
        "volume_vars": tuple(cell_internal_vars_surface),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
    }

    val = _eval_integrand(domain, expr, local)
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]


def _make_universal_volume_kernel(
    domain, expr, value_shape, temporal_tags, runtime_parameter_tags, region_mask_names, problem_ref
):
    """
    Create the universal volume kernel for a lowered weak-form expression.
    """

    def kernel(
        cell_sol_flat,
        physical_quad_points,
        cell_shape_grads,
        cell_JxW,
        cell_v_grads_JxW,
        *cell_internal_vars,
    ):
        return _eval_volume_integrand(
            domain,
            expr,
            value_shape,
            cell_sol_flat,
            physical_quad_points,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            temporal_tags,
            runtime_parameter_tags,
            region_mask_names,
            problem_ref,
            *cell_internal_vars,
        )

    return kernel


def _eval_multifield_volume_integrand(
    domain,
    term_list,
    fields,
    field_index,
    temporal_tags,
    runtime_parameter_tags,
    region_mask_names,
    problem_ref,
    cell_sol_flat,
    physical_quad_points,
    cell_shape_grads,
    cell_JxW,
    cell_v_grads_JxW,
    *cell_internal_vars,
):
    """Evaluate all coupled volume terms on one cell -> flat block-local residual.

    Splits the cell DOFs per field (``unflatten_fn_dof``), evaluates each additive
    term ``(coeff, test_field_index)`` with per-field shape data (reusing
    ``_eval_integrand``), and accumulates into its test field's residual slot.
    ``ravel_pytree`` concatenates the per-field residuals in field order (matching
    ``unflatten_fn_dof``) so the assembler autodiffs the full block matrix."""
    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("The assembly problem_ref['problem'] was not initialized before kernel evaluation.")

    cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
    nfields = len(fields)
    nnodes = [int(problem.fes[i].shape_vals.shape[1]) for i in range(nfields)]
    nc = [0]
    for n in nnodes:
        nc.append(nc[-1] + n)
    per_field = [
        {
            "shape_vals": problem.fes[i].shape_vals,
            "shape_grads": cell_shape_grads[:, nc[i] : nc[i + 1], :],
            "cell_sol": cell_sol_list[i],
        }
        for i in range(nfields)
    ]
    local = {
        "physical_quad_points": physical_quad_points,
        "fields": per_field,
        "field_index": field_index,
        "tag": "fem_gauss",
        "surface": False,
        "domain_context": domain.context,
        "temporal_tags": tuple(temporal_tags),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
        "region_mask_names": tuple(region_mask_names),
        "volume_vars": tuple(cell_internal_vars),
    }

    residuals = [jnp.zeros((nnodes[i] * int(fields[i]["vec"]),), dtype=cell_sol_flat.dtype) for i in range(nfields)]
    for coeff, test_idx in term_list:
        val = _eval_integrand(domain, coeff, local)
        weights = cell_JxW[test_idx]
        wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
        contrib = ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]
        residuals[test_idx] = residuals[test_idx] + contrib
    return ravel_pytree(residuals)[0]


def _make_multifield_volume_kernel(
    domain, term_list, fields, field_index, temporal_tags, runtime_parameter_tags, region_mask_names, problem_ref
):
    """Universal volume kernel for a coupled (multi-field) weak form."""

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        return _eval_multifield_volume_integrand(
            domain,
            term_list,
            fields,
            field_index,
            temporal_tags,
            runtime_parameter_tags,
            region_mask_names,
            problem_ref,
            cell_sol_flat,
            physical_quad_points,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            *cell_internal_vars,
        )

    return kernel


def _eval_multifield_surface_integrand(
    domain,
    term_list,
    fields,
    field_index,
    tag,
    temporal_tags,
    runtime_parameter_tags,
    problem_ref,
    cell_sol_flat,
    physical_surface_quad_points,
    face_shape_vals,
    face_shape_grads,
    face_nanson_scale,
    *cell_internal_vars_surface,
):
    """Evaluate all coupled boundary terms on one face -> flat block-local residual.

    The surface analogue of :func:`_eval_multifield_volume_integrand`. The kernel receives the
    full multi-field parent-cell DOFs and the per-field face shape data concatenated
    along the node axis (assembler concatenates in ``problem.fes`` order, identical to
    the volume ``shape_grads``), so we split DOFs with ``unflatten_fn_dof`` and slice the
    face shape arrays per field. Each term ``(coeff, test_field_index)`` accumulates into
    its test field's residual slot; ``ravel_pytree`` concatenates in field order so the assembler
    autodiffs the surface contribution into the right block(s) of the load and matrix."""
    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("The assembly problem_ref['problem'] was not initialized before kernel evaluation.")

    cell_sol_list = problem.unflatten_fn_dof(cell_sol_flat)
    nfields = len(fields)
    nnodes = [int(problem.fes[i].shape_vals.shape[1]) for i in range(nfields)]
    nc = [0]
    for n in nnodes:
        nc.append(nc[-1] + n)
    per_field = [
        {
            "shape_vals": face_shape_vals[:, nc[i] : nc[i + 1]],
            "shape_grads": face_shape_grads[:, nc[i] : nc[i + 1], :],
            "cell_sol": cell_sol_list[i],
        }
        for i in range(nfields)
    ]
    local = {
        "physical_quad_points": physical_surface_quad_points,
        "fields": per_field,
        "field_index": field_index,
        "tag": tag,
        "surface": True,
        "domain_context": domain.context,
        "boundary_normals": _lookup_boundary_normals(domain, tag, jnp.asarray(physical_surface_quad_points)),
        "temporal_tags": tuple(temporal_tags),
        "runtime_parameter_tags": tuple(runtime_parameter_tags),
        "volume_vars": tuple(cell_internal_vars_surface),
    }

    face_nanson_scale = jnp.asarray(face_nanson_scale)
    weights = face_nanson_scale[0] if face_nanson_scale.ndim == 2 else face_nanson_scale

    residuals = [jnp.zeros((nnodes[i] * int(fields[i]["vec"]),), dtype=cell_sol_flat.dtype) for i in range(nfields)]
    for coeff, test_idx in term_list:
        val = _eval_integrand(domain, coeff, local)
        wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
        contrib = ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]
        residuals[test_idx] = residuals[test_idx] + contrib
    return ravel_pytree(residuals)[0]


def _make_multifield_surface_kernel(
    domain, term_list, fields, field_index, tag, temporal_tags, runtime_parameter_tags, problem_ref
):
    """Universal surface kernel for the coupled boundary terms on one tag."""

    def kernel(
        cell_sol_flat,
        physical_surface_quad_points,
        face_shape_vals,
        face_shape_grads,
        face_nanson_scale,
        *cell_internal_vars_surface,
    ):
        return _eval_multifield_surface_integrand(
            domain,
            term_list,
            fields,
            field_index,
            tag,
            temporal_tags,
            runtime_parameter_tags,
            problem_ref,
            cell_sol_flat,
            physical_surface_quad_points,
            face_shape_vals,
            face_shape_grads,
            face_nanson_scale,
            *cell_internal_vars_surface,
        )

    return kernel


def _make_universal_surface_kernel(domain, expr, tag, value_shape, runtime_parameter_tags, temporal_tags):
    def kernel(
        cell_sol_flat,
        physical_surface_quad_points,
        face_shape_vals,
        face_shape_grads,
        face_nanson_scale,
        *cell_internal_vars_surface,
    ):
        return _eval_surface_integrand(
            domain,
            expr,
            tag,
            value_shape,
            cell_sol_flat,
            physical_surface_quad_points,
            face_shape_vals,
            face_shape_grads,
            face_nanson_scale,
            temporal_tags,
            runtime_parameter_tags,
            *cell_internal_vars_surface,
        )

    return kernel


# --------------------------------
# Problem assembly
# --------------------------------


def _meshio_type_for_element(element_type: str) -> str:
    meshio_type_map = {
        "TRI3": "triangle",
        "TRI6": "triangle6",
        "QUAD4": "quad",
        "QUAD8": "quad8",
        "TET4": "tetra",
        "TET10": "tetra10",
        "HEX8": "hexahedron",
        "HEX20": "hexahedron20",
        "HEX27": "hexahedron27",
    }
    if element_type not in meshio_type_map:
        raise KeyError(f"Unsupported element type '{element_type}'.")
    return meshio_type_map[element_type]


# Quadratic (P2) meshio cell type -> (linear source type, per-element edge node list
# in meshio ordering). jno's domain machinery assumes linear cells, so the domain mesh
# stays linear (P1) and we promote *only* the assembly mesh here: insert edge-midpoint
# nodes, vertices preserved (so a P1 field on a P2 problem -- e.g. Taylor-Hood pressure
# -- is just the vertex block).
_P2_FROM_P1 = {
    "triangle6": ("triangle", [(0, 1), (1, 2), (2, 0)]),
    "tetra10": ("tetra", [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]),
}


def _promote_to_quadratic(points, cells_p1, edge_local):
    """Promote a linear simplex mesh to quadratic (P1 -> P2).

    Inserts one node at each *unique* edge midpoint (canonical sorted-vertex-pair
    key, so an edge shared by two cells maps to a single node), preserving the
    original vertices (indices ``0..nverts-1``) and appending the edge nodes.
    Returns ``(points_p2, cells_p2)`` with cells in meshio ``*6``/``*10`` ordering
    (corners first, then ``edge_local`` midpoints)."""
    points = np.asarray(points)
    cells_p1 = np.asarray(cells_p1)
    ncorner = cells_p1.shape[1]
    edge_map: dict[tuple[int, int], int] = {}
    edge_nodes: List[Any] = []
    cells_p2 = np.zeros((cells_p1.shape[0], ncorner + len(edge_local)), dtype=np.int64)
    cells_p2[:, :ncorner] = cells_p1
    nid = points.shape[0]
    for c in range(cells_p1.shape[0]):
        for k, (i, j) in enumerate(edge_local):
            a, b = int(cells_p1[c, i]), int(cells_p1[c, j])
            key = (a, b) if a < b else (b, a)
            n = edge_map.get(key)
            if n is None:
                n = nid
                edge_map[key] = nid
                edge_nodes.append(0.5 * (points[a] + points[b]))
                nid += 1
            cells_p2[c, ncorner + k] = n
    pts = np.vstack([points, np.asarray(edge_nodes)]) if edge_nodes else points
    return pts, cells_p2


def _promote_to_degree(points, cells_p1, ref_pts, cell_type=None):
    """Promote a linear mesh to a degree-``k`` Lagrange node mesh (P1 -> P{k} / Q1 -> Q{k}, any ``k``).

    ``cell_type`` names the cell for a tensor-product mesh (``"quad"`` / ``"hexahedron"``), whose
    nodes are placed with its degree-1 basis; omitted, the cell is a simplex and the placement is
    barycentric. A tensor-product cell's ``cells_p1`` must be in basix vertex order.

    ``ref_pts`` are the element's reference interpolation points in **basix DOF order** (shape
    ``(n_dof, tdim)``; the first ``ncorner`` are the cell vertices). Each cell's nodes are the affine
    image of ``ref_pts`` through that cell's P1 geometry (a barycentric combination of its vertices);
    nodes are deduplicated by **physical coordinate** (a scale-aware grid hash). So an edge/face shared
    by two cells collapses to one global node *regardless of the cells' local orientation* -- for C0
    Lagrange a DOF is a point value, so coordinate coincidence on a shared entity is exactly the
    conformity condition (no orientation sign or per-edge node ordering, unlike RT/Nedelec; this is what
    lets a single midpoint suffice at P2 and a clean generalisation hold at P3+). Original vertices keep
    ids ``0..nv-1`` (so a P1 field on a P{k} problem is the leading vertex block, e.g. Taylor-Hood
    pressure). Returns ``(points_k, cells_k)`` with ``cells_k`` columns in the same DOF order as
    ``ref_pts`` -- hence as the basis tabulated from the same basix element."""
    points = np.asarray(points, dtype=float)
    cells_p1 = np.asarray(cells_p1)
    ref_pts = np.asarray(ref_pts, dtype=float)
    ncell, ncorner = cells_p1.shape
    ndof = ref_pts.shape[0]
    if cell_type in ("quad", "quadrilateral", "hexahedron", "hex"):
        # The general form: x(xi) = sum_a N_a(xi) x_a, tabulating the cell's DEGREE-1 (geometry)
        # basis at the reference points. The barycentric block below is this same operation
        # specialised to a simplex, where the P1 basis is [1 - sum(xi), xi]. It is kept rather than
        # replaced because the two differ by one ulp (1.1e-16, measured), and a simplex mesh should
        # not shift at the last bit for a change that is about quadrilaterals.
        #
        # `cells_p1` must be in BASIX vertex order here, since that is the order the tabulated basis
        # is written in -- pairing it with meshio/VTK order makes the cell a bow-tie.
        from .fem_lagrange import _lagrange_basix, basix_cell

        weights = np.asarray(_lagrange_basix(basix_cell(cell_type)[0], 1).tabulate(0, ref_pts))[0, :, :, 0]
    else:
        # barycentric weights of each reference point: l0 = 1 - sum(xi), l_i = xi_i
        weights = np.empty((ndof, ncorner), dtype=float)
        weights[:, 0] = 1.0 - ref_pts.sum(axis=1)
        weights[:, 1:] = ref_pts
    phys = np.einsum("dc,ncg->ndg", weights, points[cells_p1])  # (ncell, ndof, gdim) physical node coords
    # scale-aware coordinate hash: coincident nodes differ only by FP roundoff (<< tol); distinct nodes
    # are separated by ~mesh spacing (>> tol).
    extent = float(np.max(points.max(axis=0) - points.min(axis=0))) if points.shape[0] else 1.0
    tol = 1e-7 * (extent or 1.0)

    def _key(p):
        return tuple(np.round(np.asarray(p) / tol).astype(np.int64))

    coord_to_gid: dict = {}
    out_pts: List[Any] = []
    for vid in range(points.shape[0]):  # seed with the original vertices so they keep ids 0..nv-1
        coord_to_gid[_key(points[vid])] = vid
        out_pts.append(points[vid])
    cells_k = np.zeros((ncell, ndof), dtype=np.int64)
    nid = points.shape[0]
    for c in range(ncell):
        for d in range(ndof):
            k = _key(phys[c, d])
            gid = coord_to_gid.get(k)
            if gid is None:
                gid, coord_to_gid[k] = nid, nid
                out_pts.append(phys[c, d])
                nid += 1
            cells_k[c, d] = gid
    return np.asarray(out_pts), cells_k


def _zero_mass_dirichlet_rows(M, bc):
    """Zero a mass matrix's Dirichlet **rows** so ``M u̇ + A u = c`` reads ``u[d]=g``.

    Symmetric Dirichlet (identity rows) is applied to *every* assembled matrix — correct
    for a stiffness operator, but wrong for the **mass**: a constrained DOF must carry no
    time derivative. Zeroing the Dirichlet rows (``A``'s Dirichlet rows are identity, ``c``
    carries ``g``) makes the Dirichlet row of ``(M + dt A) w = M w_old + dt c`` reduce to
    ``u[d] = g``. The Dirichlet *columns* are deliberately kept: they couple a free row to a
    constrained DOF's time derivative (``M_fd·ġ``), which the stepper's ``M(w_new−w_old)``
    captures — essential for **time-varying** Dirichlet and harmless when ``ġ=0`` (the
    constant case). So M is asymmetric here, by design."""
    rows = None if bc is None else getattr(bc, "bc_rows", None)
    if rows is None:
        return M
    rows = jnp.asarray(rows).reshape(-1)
    if rows.shape[0] == 0:
        return M
    return jnp.asarray(M).at[rows, :].set(0.0)


def _zero_mass_dirichlet_rows_sparse(M, bc):
    """BCOO version of :func:`_zero_mass_dirichlet_rows` — zeros the entries in Dirichlet rows
    without densifying, so the transient mass stays sparse (matrix-free matvec, O(nnz) memory)."""
    rows = None if bc is None else getattr(bc, "bc_rows", None)
    if rows is None:
        return M
    rows = jnp.asarray(rows).reshape(-1)
    if rows.shape[0] == 0:
        return M
    keep = jnp.logical_not(jnp.isin(M.indices[:, 0], rows)).astype(M.data.dtype)
    return type(M)((M.data * keep, M.indices), shape=M.shape)


def _zero_forcing_dirichlet_rows(forcing_fn, bc):
    """Wrap a forcing callback ``f(t)`` to zero its Dirichlet rows.

    The transient forcing is the *raw* source load (assembled without Dirichlet
    elimination), so it has entries at constrained DOFs. The Dirichlet rows must read
    ``u[d]=g`` from the load ``c`` alone — a source contributes only to free DOFs — so the
    forcing is zeroed there before it enters the stepper ``M w_old + dt·(c + f(t))``."""
    rows = None if bc is None else getattr(bc, "bc_rows", None)
    if forcing_fn is None or rows is None:
        return forcing_fn
    rows = jnp.asarray(rows).reshape(-1)
    if rows.shape[0] == 0:
        return forcing_fn

    def cleaned(t, args=None):
        return jnp.asarray(forcing_fn(t, args)).reshape(-1).at[rows].set(0.0)

    return cleaned


def _dense_array(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    try:
        return jnp.asarray(A)
    except Exception:
        return jnp.asarray(np.asarray(A))


def _region_mask_arrays_for_domain(domain):
    """Per-cell sub-region masks for the regions this domain's FEM problem uses, in the kernel's fixed
    (sorted) order -- the order ``_build_fem_problem`` recorded in ``domain._fem_region_mask_order``.

    Each is a constant ``(num_cells, 1)`` 0/1 array (the kernel slices a scalar per cell). Cached per region
    on the domain (depends only on the mesh + region geometry), so transient steps and parametric
    re-evaluations don't rebuild them. Returns ``()`` when the problem has no sub-region terms."""
    names = tuple(getattr(domain, "_fem_region_mask_order", ()) or ())
    if not names:
        return ()
    a_cells = getattr(domain, "_fem_assembly_cells", None)
    n_cells = int(np.asarray(a_cells).shape[0]) if a_cells is not None else None
    cache = getattr(domain, "_fem_region_mask_cache", None)
    if cache is None or cache.get("__ncells__") != n_cells:  # invalidate if the assembly mesh changed
        cache = {"__ncells__": n_cells}
        domain._fem_region_mask_cache = cache
    out = []
    for r in names:
        if r not in cache:
            cache[r] = jnp.asarray(_cell_region_mask(domain, r), dtype=_default_float_dtype()).reshape(-1, 1)
        out.append(cache[r])
    return tuple(out)


def _make_internal_vars(
    fe_module,
    temporal_tags,
    t,
    *,
    n_cells: int,
    dtype=None,
    runtime_parameter_tags=(),
    runtime_parameter_values=None,
    region_mask_arrays=(),
    extra_volume_vars=(),
):
    """
    Build the batched volume-variable arrays the kernel slices per cell.

    Volume-var layout (the order every kernel's ``local`` indexing assumes):
    ``[ temporal ... , runtime_parameter ... , region_mask ... , extra ... ]``. Each temporal variable
    is broadcast to shape (n_cells, 1); region masks are constant per-cell arrays (see
    ``_region_mask_arrays_for_domain``) so any path that threads them gets per-region integration.
    """
    vol = []

    if temporal_tags:
        t0 = jnp.asarray(t, dtype=dtype)
        t_batched = jnp.full((int(n_cells), 1), t0, dtype=t0.dtype)
        vol.extend([t_batched for _ in temporal_tags])
    # Parameter values, in runtime_parameter_tags order, broadcast per cell.
    rpv = runtime_parameter_values or {}
    for name in runtime_parameter_tags:
        p = jnp.asarray(rpv[name], dtype=dtype)
        flat = p.reshape(-1)
        if flat.shape[0] == 1:
            # scalar parameter -> same value in every cell (broadcast to quad in-kernel)
            vol.append(jnp.full((int(n_cells), 1), flat[0], dtype=p.dtype))
        else:
            # field parameter (node- or cell-based) -> pass the global array through;
            # the per-cell gather slices it, the kernel interpolates.
            vol.append(flat)
    # Sub-region masks, after the runtime parameters (matches the evaluator's RegionMask index).
    for m in region_mask_arrays:
        vol.append(jnp.asarray(m, dtype=dtype).reshape(int(n_cells), 1))
    for v in extra_volume_vars:
        arr = jnp.asarray(v, dtype=dtype)
        if arr.ndim == 0:
            arr = jnp.full((int(n_cells), 1), arr, dtype=arr.dtype)
        vol.append(arr)

    return fe_module.InternalVars(volume_vars=tuple(vol))


def _interface_frame(m_pts: np.ndarray, s_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal tangential frame ``(dim-1, dim)`` and origin for a tied/periodic interface pair.

    Matching a secondary node against the main face means comparing the two **in the interface**, with
    the across-interface coordinate removed. This used to be done by dropping the single global axis
    whose tag means differed most, which silently assumes the faces are planar, axis-aligned, and
    separated by a pure translation along that axis. A **tied** interface breaks that outright — both
    faces are coincident, so every mean difference is ~0 and the dropped axis is whichever coordinate
    happened to carry the largest rounding error — and so does any periodic cell whose lattice vector
    is not a global axis.

    The frame generalises it: its rows are the ``dim-1`` directions of greatest variance of the main
    face's own point cloud — an orthonormal basis of the face's tangent plane, from an SVD. Projecting
    onto them removes exactly the across-interface coordinate and nothing else.

    For the planar axis-aligned case the rows span the same plane the transverse axes did, and
    nearest-neighbour distances, edge parameters and barycentric weights are all invariant under a
    rotation *within* that plane — so the existing conforming and non-matching periodic paths return
    bit-comparable results.

    **Limitation:** one frame per tie, so main and secondary must be (near-)parallel. A wedge-shaped or
    closed (cylindrical) interface has no single tangent plane; the degeneracy guard below catches the
    fully-collapsed cases, but a strongly curved face is merely projected, not refused.
    """
    dim = int(m_pts.shape[1])
    ref = m_pts if m_pts.shape[0] >= dim else np.concatenate([m_pts, s_pts], axis=0)
    origin = ref.mean(axis=0)
    if dim == 1:
        # A 1-D "interface" is a single point: there is no in-interface coordinate at all. The empty
        # (0, 1) frame projects every node to a zero-length vector, so every secondary sits at distance 0
        # from the main and ties exactly -- which is what dropping the only axis did before.
        return np.zeros((0, 1)), origin
    _u, sv, vt = np.linalg.svd(ref - origin, full_matrices=True)
    if sv.size < dim - 1 or float(sv[0]) <= 0.0:
        raise ValueError(
            "Periodic/tied interface: the main face collapses to a single point, so no tangent "
            "plane can be fitted. Check that the tagged region actually selects a boundary face."
        )
    if dim > 2 and float(sv[dim - 2]) <= 1.0e-8 * float(sv[0]):
        raise ValueError(
            "Periodic/tied interface: the main face is degenerate (its nodes are collinear), so no "
            "tangent plane can be fitted. In 3-D a tied face must be a surface, not an edge — check "
            "the tag predicate selects a 2-D patch of the boundary."
        )
    return np.ascontiguousarray(vt[: dim - 1]), origin


def _edge_shape(xi: np.ndarray, k: int) -> np.ndarray:
    """Lagrange shape values ``(..., k)`` on a reference edge at local coordinate(s) ``xi in [0, 1]``.

    ``k == 2`` is the P1 edge ``(a, b)``; ``k == 3`` the P2 edge ``(a, b, mid)`` -- the node order the
    facet tables use, with the midside node LAST."""
    xi = np.asarray(xi, dtype=float)
    if k == 2:
        return np.stack([1.0 - xi, xi], axis=-1)
    if k == 3:
        return np.stack([2.0 * (xi - 0.5) * (xi - 1.0), 2.0 * xi * (xi - 0.5), -4.0 * xi * (xi - 1.0)], axis=-1)
    raise ValueError(f"edge facets carry 2 (P1) or 3 (P2) nodes, got {k}")


def _facet_dual_coeffs(k: int, qp: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Element-local **dual** basis coefficients ``A``: ``psi_i = sum_k A_ik N_k`` is biorthogonal to
    the primal basis, ``int psi_i N_j = delta_ij int N_i``.

    Rather than hard-coding the dual functions per element order, build them from the facet mass
    matrix: with ``Mass_ij = int N_i N_j`` and ``d_i = int N_i``, taking ``A = diag(d) Mass^-1`` gives
    ``int psi_i N_j = (A Mass)_ij = d_i delta_ij`` by construction. For the P1 edge this recovers the
    textbook ``psi_0 = 2 - 3xi``, ``psi_1 = 3xi - 1`` (Wohlmuth, SIAM J. Numer. Anal. 38(3):989-1012,
    2000, §3), and it extends to P2 with no extra algebra.

    Biorthogonality is **element-local**, so the assembled ``D`` is diagonal only when the segments
    cover each secondary facet completely -- which :func:`_mortar_rows_2d` checks and refuses otherwise.
    The reference-element coefficients carry over unchanged to a straight physical edge, whose
    Jacobian is constant and cancels between ``Mass`` and ``d``."""
    return _dual_coeffs(_edge_shape(qp, k), qw)


def _dual_coeffs(shape_vals: np.ndarray, qw: np.ndarray) -> np.ndarray:
    """Dual coefficients ``A = diag(d) Mass^-1`` from tabulated shape values ``(n_q, n_loc)`` and
    quadrature weights -- the element-shape-agnostic core of :func:`_facet_dual_coeffs`, shared by the
    2-D (edge) and 3-D (triangle) mortar paths."""
    mass = shape_vals.T @ (qw[:, None] * shape_vals)
    return np.diag(qw @ shape_vals) @ np.linalg.inv(mass)


def _tri_shape(bary: np.ndarray, k: int) -> np.ndarray:
    """Lagrange shape values ``(..., k)`` on a triangle from barycentric coordinates ``(..., 3)``.

    ``k == 3`` is the P1 triangle ``(a, b, c)``; ``k == 6`` the P2 triangle
    ``(a, b, c, mab, mbc, mca)`` -- the node order the facet tables use, matching the 3-D branch of
    :func:`_periodic_facet_weights`."""
    l0, l1, l2 = bary[..., 0], bary[..., 1], bary[..., 2]
    if k == 3:
        return np.stack([l0, l1, l2], axis=-1)
    if k == 6:
        return np.stack(
            [
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            ],
            axis=-1,
        )
    if k == 4:
        # A hexahedron's facet reaches here only on the INTEGRATED (mortar) path, which clips triangles
        # and has no quadrilateral analogue yet. The COLLOCATED path does support it -- bilinear weights
        # from the inverse of the facet's own map, see `_periodic_facet_weights` -- so a non-matching tie
        # across a hex facet works; it is the mortar coupling specifically that does not.
        raise NotImplementedError(
            "an INTEGRATED (mortar) tie across a HEXAHEDRAL facet is not supported: the mortar rows clip "
            "triangles, and these weights are barycentric, which would interpolate a quadrilateral facet "
            "from three of its four nodes. The collocated coupling handles it and is what a hex tie uses; "
            "reaching this means the mortar path was selected for a quad facet, which is a bug."
        )
    raise ValueError(f"triangle facets carry 3 (P1) or 6 (P2) nodes, got {k}")


def _tri_bary(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Barycentric coordinates ``(n, 3)`` of planar points ``x`` ``(n, 2)`` in the triangle ``v``
    ``(3, 2)``. Affine, so it is invariant to the in-plane rotation the interface frame is free in."""
    jac = np.column_stack([v[1] - v[0], v[2] - v[0]])  # (2, 2)
    lam = np.linalg.solve(jac, (np.atleast_2d(x) - v[0]).T)  # (2, n)
    return np.stack([1.0 - lam[0] - lam[1], lam[0], lam[1]], axis=-1)


def _signed_area(poly: np.ndarray) -> float:
    """Shoelace signed area of a planar polygon ``(n, 2)``; positive when counter-clockwise."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _as_ccw(poly: np.ndarray) -> np.ndarray:
    """The same polygon wound counter-clockwise, which is what :func:`_clip_convex` assumes."""
    return poly[::-1] if _signed_area(poly) < 0.0 else poly


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersection of two **convex, counter-clockwise** planar polygons (Sutherland & Hodgman,
    *Commun. ACM* 17(1):32-42, 1974): clip the subject successively against each half-plane of the
    clip polygon.

    Two triangles intersect in a convex polygon of at most 6 vertices. Returns ``(n, 2)``; fewer than
    3 vertices means the overlap is empty or degenerate (a touching edge or corner), which the caller
    drops -- such a contact carries zero area and contributes nothing to the mortar integral.
    """
    out = [np.asarray(p, dtype=float) for p in subject]
    n_clip = len(clip)
    for i in range(n_clip):
        if len(out) < 3:
            return np.zeros((0, 2))
        a, b = clip[i], clip[(i + 1) % n_clip]
        edge = b - a
        prev, out = out, []
        for j in range(len(prev)):
            p, q = prev[j - 1], prev[j]
            # >= 0 is "left of / on" the directed edge a->b, i.e. inside a CCW clip polygon
            sp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
            sq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            if sq >= 0.0:
                if sp < 0.0:
                    out.append(p + (q - p) * (sp / (sp - sq)))
                out.append(q)
            elif sp >= 0.0:
                out.append(p + (q - p) * (sp / (sp - sq)))
    return np.asarray(out) if len(out) >= 3 else np.zeros((0, 2))


def _tri_quadrature(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss x Gauss on the unit square, Duffy-mapped to the reference triangle.

    ``(u, v) -> (u, v(1-u))`` has Jacobian ``1-u``, so an ``n x n`` tensor rule integrates a degree
    ``2n-2`` polynomial exactly (the Jacobian costs one degree in ``u``). Returns barycentric
    coordinates ``(n_q, 3)`` and weights summing to 1/2 -- the reference triangle's area.

    Chosen over a tabulated symmetric rule because it is one formula at every order, so the P1 and P2
    facet cases share it and the exactness is checkable by integrating monomials."""
    g, gw = np.polynomial.legendre.leggauss(n)
    g, gw = 0.5 * (g + 1.0), 0.5 * gw
    u, v = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
    wu, wv = (a.ravel() for a in np.meshgrid(gw, gw, indexing="ij"))
    x, y = u, v * (1.0 - u)
    return np.stack([1.0 - x - y, x, y], axis=-1), wu * wv * (1.0 - u)


def _tri_dual_available(k: int) -> bool:
    """Can a biorthogonal dual basis be built on a ``k``-node triangle by ``A = diag(d) Mass^-1``?

    Only when every ``d_i = int N_i`` is non-zero. That holds for edges at any order and for the P1
    triangle, but **the P2 triangle's vertex functions integrate to exactly zero**::

        int_T L_a (2 L_a - 1) dA = 2/12 - 1/6 = 0

    so ``diag(d)`` is singular and this construction collapses.

    **Rescaling does not rescue it, and no better basis exists for this architecture.** Taking
    ``c_i = 1`` (``psi = Mass^-1 N``) does give an exactly biorthogonal basis with ``D = I``, and it
    transfers constant/linear/quadratic fields exactly -- but its span does not contain the linear
    functions, because the coefficients needed for that are ``Mass_e . 1``, which varies per element
    while the basis coefficients are global. Lamichhane's thesis proves this is unavoidable: under
    ``supp phi_i == supp mu_i`` (a locally supported dual basis, one multiplier per secondary DOF, exactly
    jNO's structure) **Lemma 3.4** shows there is *no* dual multiplier space containing the piecewise
    linear hat functions for quadratic simplicial elements in 3-D -- and containing them is what the
    optimal a priori error estimate needs. The published remedy uses a multiplier space of *lower*
    dimension than the secondary trace space, which makes ``D`` rectangular and the tie a constrained
    solve rather than an elimination; jNO's prolongation cannot express that.

    Note the boundary matches exactly: the same source records that in **two** dimensions the
    quadratic dual space *does* contain the linear hats, which is why the P2 edge path above is sound.

    B. Lamichhane, *Higher Order Mortar Finite Elements with Dual Lagrange Multiplier Spaces and
    Applications*, PhD thesis, Univ. Stuttgart 2006, Remark 2.10 and Lemma 3.4; see also Lamichhane,
    Stevenson & Wohlmuth, *Numer. Math.* 102:93-121, 2005.

    So a P2 triangular interface keeps the collocated coupling and the ``coupling`` key reports it.
    The check is computed rather than hard-coded to the node count so it also guards any element type
    added later whose shape functions have a zero integral.
    """
    if k not in (3, 6):
        # This is an availability QUERY, so an unsupported facet is a False, not a raise. A
        # quadrilateral facet (k = 4, a hexahedron's) has no triangular shape functions to integrate
        # -- asking `_tri_shape` for them aborted the whole periodic build here, before the caller
        # ever reached its node-to-node matching, which is what a conforming hex mesh needs and
        # which does not involve a dual basis at all.
        return False
    bary, w = _tri_quadrature(k // 3 + 2)
    d = np.abs(w @ _tri_shape(bary, k))
    return bool(d.min() > 1.0e-12 * float(d.max()))


def _main_covers_secondary_3d(s_facets: np.ndarray, m_facets: np.ndarray, loc: np.ndarray) -> bool:
    """Does every secondary facet vertex lie inside some main triangle? The 3-D counterpart of
    :func:`_faces_span_the_same_extent`.

    A mortar integral over the secondary face needs the main basis defined underneath all of it. In 2-D
    that reduces to an interval containment; in 3-D the faces can also be *ragged* (equal bounding
    boxes, mismatched boundaries), so the test is per-vertex containment rather than an extent
    comparison. Failing it means the tie keeps the collocated coupling and reports so.
    """
    s = np.asarray(s_facets, int)[:, :3]
    m = np.asarray(m_facets, int)[:, :3]
    if s.size == 0 or m.size == 0:
        return False
    pts = np.asarray(loc, dtype=float)[np.unique(s)]
    best = np.full(len(pts), -np.inf)
    for f in range(m.shape[0]):
        v = np.asarray(loc, dtype=float)[m[f]]
        if abs(_signed_area(v)) <= 0.0:
            continue
        best = np.maximum(best, _tri_bary(pts, v).min(axis=1))
    return bool(np.all(best >= -1.0e-9))


def _mortar_rows_3d(
    s_facets: np.ndarray,
    m_facets: np.ndarray,
    loc: np.ndarray,
    *,
    span: float,
) -> Dict[int, List[Tuple[int, float]]]:
    """Dual-mortar prolongation rows for a **3-D** interface, whose facets are triangles.

    Same constraint as :func:`_mortar_rows_2d` -- ``D u_s = M u_m`` with a diagonal ``D`` from the
    dual basis -- but the segmentation is genuine polygon geometry rather than an interval
    intersection: each secondary triangle is clipped against every main triangle it overlaps
    (:func:`_clip_convex`), the clip polygon is fan-triangulated, and each sub-triangle carries its
    own quadrature (Puso & Laursen, *CMAME* 193:601-629, 2004).

    This is the case that motivates the whole coupling: point-in-triangle collocation is not an L2
    projection on a surface, so it fails the constant-stress patch test in 3-D where the 2-D edge case
    happens to pass it.

    **Areas are measured in the interface frame**, so the interface must be (near-)planar -- the same
    assumption :func:`_interface_frame` already makes. A curved tied surface is projected, and its
    segment areas are the projected ones.
    """
    s_facets = np.asarray(s_facets, dtype=int)
    m_facets = np.asarray(m_facets, dtype=int)
    ks, km = int(s_facets.shape[1]), int(m_facets.shape[1])
    xy = np.asarray(loc, dtype=float)

    s_nodes, m_nodes = np.unique(s_facets), np.unique(m_facets)
    s_at = {int(v): i for i, v in enumerate(s_nodes)}
    m_at = {int(v): i for i, v in enumerate(m_nodes)}
    diag = np.zeros(len(s_nodes))
    cross = np.zeros((len(s_nodes), len(m_nodes)))

    bary_q, w_q = _tri_quadrature(max(ks, km) // 3 + 2)
    dual = _dual_coeffs(_tri_shape(bary_q, ks), w_q)
    area_tol = 1.0e-12 * max(span, 1.0) ** 2

    # Bounding boxes make the facet pairing O(n_s + n_m) per secondary in practice instead of a full
    # O(n_s * n_m) Python double loop, which a face of a few thousand triangles would not survive.
    m_v = xy[m_facets[:, :3]]  # (n_m, 3, 2)
    m_lo, m_hi = m_v.min(axis=1), m_v.max(axis=1)

    for e in range(s_facets.shape[0]):
        sv = xy[s_facets[e, :3]]
        area_e = abs(_signed_area(sv))
        if area_e <= area_tol:
            continue
        s_lo, s_hi = sv.min(axis=0), sv.max(axis=0)
        near = np.flatnonzero(np.all(m_lo <= s_hi, axis=1) & np.all(m_hi >= s_lo, axis=1))
        rows = [s_at[int(v)] for v in s_facets[e]]
        sv_ccw = _as_ccw(sv)
        covered = 0.0
        for f in near:
            mv = xy[m_facets[f, :3]]
            poly = _clip_convex(sv_ccw, _as_ccw(mv))
            if poly.shape[0] < 3:
                continue
            cols = [m_at[int(v)] for v in m_facets[f]]
            for i in range(1, poly.shape[0] - 1):  # fan-triangulate the convex clip polygon
                sub = poly[[0, i, i + 1]]
                a_sub = abs(_signed_area(sub))
                if a_sub <= area_tol:
                    continue
                covered += a_sub
                x_q = bary_q @ sub  # barycentric -> cartesian on the sub-triangle
                n_s = _tri_shape(_tri_bary(x_q, sv), ks)
                n_m = _tri_shape(_tri_bary(x_q, mv), km)
                psi = n_s @ dual.T
                w = w_q * (a_sub / 0.5)  # reference weights sum to 1/2; rescale to the real area
                diag[rows] += np.einsum("qi,qi,q->i", psi, n_s, w)
                cross[np.ix_(rows, cols)] += np.einsum("qi,qj,q->ij", psi, n_m, w)

        if abs(covered - area_e) > 1.0e-8 * area_e:
            raise ValueError(
                f"Mortar segmentation covered {covered:.6g} of a secondary facet of area {area_e:.6g}. "
                "The main face contains the secondary face's vertices, so this is a HOLE in the main "
                "face -- its triangles do not tile it. Check the main tag selects a connected set "
                "of whole boundary facets."
            )

    if np.any(np.abs(diag) <= area_tol):
        raise ValueError(
            "Mortar coupling produced a singular secondary mass diagonal; a tied secondary node carries no "
            "facet area. Check the secondary tag selects whole boundary facets, not isolated nodes."
        )
    rows_out: Dict[int, List[Tuple[int, float]]] = {}
    for node, i in s_at.items():
        w_row = cross[i] / diag[i]
        nz = np.flatnonzero(np.abs(w_row) > 1.0e-14)
        rows_out[node] = [(int(m_nodes[j]), float(w_row[j])) for j in nz]
    return rows_out


def _faces_span_the_same_extent(s_facets: np.ndarray, m_facets: np.ndarray, loc: np.ndarray, *, span: float) -> bool:
    """Does the main face cover the secondary face, so a mortar integral over the secondary is well posed?

    A mortar constraint integrates over the secondary face and needs the main basis defined everywhere
    under it. Two tagged faces do **not** always cover the same extent: a face tagged from geometry
    can exclude its corner nodes, leaving the two sides of a periodic pair with different lengths
    (measured here: a main spanning ``y in [0.1, 0.9]`` against a secondary spanning
    ``[0.0435, 0.9565]``). Collocation tolerates that by clamping each stray secondary node to the nearest
    main facet's endpoint; an integral cannot, so such a tie keeps the collocated coupling and says
    so through the ``coupling`` key rather than integrating over a domain the main does not cover.

    Faces built from a ``domain.tag`` predicate include their corners by construction and pass.
    """
    t = np.asarray(loc, dtype=float)[:, 0]
    s, m = t[np.asarray(s_facets, int)[:, :2]], t[np.asarray(m_facets, int)[:, :2]]
    if s.size == 0 or m.size == 0:
        return False
    tol = 1.0e-8 * max(span, 1.0)
    return bool(m.min() <= s.min() + tol and m.max() >= s.max() - tol)


def _mortar_rows_2d(
    s_facets: np.ndarray,
    m_facets: np.ndarray,
    loc: np.ndarray,
    *,
    span: float,
) -> Dict[int, List[Tuple[int, float]]]:
    """Dual-mortar prolongation rows for a **2-D** interface, whose facets are edges.

    The tie is imposed in the integral sense of the mortar method (Bernardi/Maday/Patera 1994; dual
    multiplier spaces from Wohlmuth 2000) rather than collocated at secondary nodes:

        ``int_G psi_i (u_s - u_m . Phi) dG = 0``   =>   ``D u_s = M u_m``,
        ``D_ij = int_G psi_i N_j^s``,  ``M_ij = int_G psi_i (N_j^m . Phi)``

    With the dual ``psi`` of :func:`_facet_dual_coeffs`, ``D`` is **diagonal**, so the secondary DOFs
    eliminate explicitly as ``u_s = D^-1 M u_m`` -- the same one-row-per-secondary shape the collocated
    path produces, and the same shape the existing reduction consumes. Unlike collocation this passes
    the constant-stress patch test and transfers momentum in a variationally balanced way.

    Both integrals are taken over the **segments**: each secondary edge clipped against every main edge
    it overlaps (Puso & Laursen, CMAME 193:601-629, 2004). In 2-D the interface coordinate is scalar,
    so clipping is an interval intersection and is exact -- no polygon geometry is involved.

    ``loc`` is the ``(n_nodes, 1)`` projection onto the interface frame; ``span`` is the interface
    extent, used to scale the degeneracy and coverage tolerances. Returns
    ``{secondary_node_id: [(main_node_id, weight), ...]}``.
    """
    s_facets = np.asarray(s_facets, dtype=int)
    m_facets = np.asarray(m_facets, dtype=int)
    ks, km = int(s_facets.shape[1]), int(m_facets.shape[1])
    t = np.asarray(loc, dtype=float)[:, 0]

    # Columns 0, 1 are the edge endpoints at every order (a midside node is interior to the edge).
    sa, sb = t[s_facets[:, 0]], t[s_facets[:, 1]]
    ma, mb = t[m_facets[:, 0]], t[m_facets[:, 1]]

    s_nodes = np.unique(s_facets)
    m_nodes = np.unique(m_facets)
    s_at = {int(n): i for i, n in enumerate(s_nodes)}
    m_at = {int(n): i for i, n in enumerate(m_nodes)}
    diag = np.zeros(len(s_nodes))
    cross = np.zeros((len(s_nodes), len(m_nodes)))

    # psi (degree p) x N^m (degree p) is degree 2p; this Gauss rule is exact through 2*max(ks,km)+1.
    qp, qw = np.polynomial.legendre.leggauss(max(ks, km) + 2)
    qp, qw = 0.5 * (qp + 1.0), 0.5 * qw  # [-1, 1] -> [0, 1]
    dual = _facet_dual_coeffs(ks, qp, qw)
    seg_tol = 1.0e-10 * max(span, 1.0)

    for e in range(s_facets.shape[0]):
        e_lo, e_hi = min(sa[e], sb[e]), max(sa[e], sb[e])
        length = sb[e] - sa[e]  # signed: the local coordinate must follow the facet's own orientation
        if abs(length) <= seg_tol:
            continue
        rows = [s_at[int(n)] for n in s_facets[e]]
        covered = 0.0
        for f in range(m_facets.shape[0]):
            c0 = max(e_lo, min(ma[f], mb[f]))
            c1 = min(e_hi, max(ma[f], mb[f]))
            seg = c1 - c0
            if seg <= seg_tol:
                continue
            covered += seg
            tq = c0 + qp * seg  # quadrature points along the interface, in interface coordinates
            n_s = _edge_shape((tq - sa[e]) / length, ks)  # (n_q, ks)
            n_m = _edge_shape((tq - ma[f]) / (mb[f] - ma[f]), km)  # (n_q, km)
            psi = n_s @ dual.T
            w = qw * seg  # ds along a straight interface is dt, so these are physical lengths
            cols = [m_at[int(n)] for n in m_facets[f]]
            diag[rows] += np.einsum("qi,qi,q->i", psi, n_s, w)
            cross[np.ix_(rows, cols)] += np.einsum("qi,qj,q->ij", psi, n_m, w)

        if abs(covered - (e_hi - e_lo)) > 1.0e-8 * max(e_hi - e_lo, seg_tol):
            raise ValueError(
                f"Mortar segmentation covered {covered:.6g} of a secondary facet of length "
                f"{e_hi - e_lo:.6g}. The two faces span the same extent, so this is a HOLE in the "
                "main face -- its facets do not tile it. Check the main tag selects a connected "
                "set of whole boundary facets."
            )

    if np.any(np.abs(diag) <= seg_tol):
        raise ValueError(
            "Mortar coupling produced a singular secondary mass diagonal; a tied secondary node carries no "
            "facet area. Check the secondary tag selects whole boundary facets, not isolated nodes."
        )
    rows_out: Dict[int, List[Tuple[int, float]]] = {}
    for node, i in s_at.items():
        w_row = cross[i] / diag[i]
        nz = np.flatnonzero(np.abs(w_row) > 1.0e-14)
        rows_out[node] = [(int(m_nodes[j]), float(w_row[j])) for j in nz]
    return rows_out


def main_trace_weights(
    query: np.ndarray,
    m_facets: np.ndarray,
    loc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weights that evaluate a **main-side** field at arbitrary points of the interface.

    A tie eliminates secondary DOFs, so its weights are only ever needed *at secondary nodes*. Contact needs
    the same trace at **quadrature points**: the signed gap ``g = g0 + n.(u_s - u_m . Phi)`` compares
    the two sides wherever the surface integral samples them, not just where the secondary mesh happens to
    put a node. This is that generalisation, batched.

    ``query`` is ``(n_q, dim-1)`` in the interface frame of :func:`_interface_frame`; ``m_facets`` is
    ``(n_f, k)`` main facet node ids; ``loc`` is every node projected into that frame. Returns
    ``(ids, w)``, both ``(n_q, k)``: the main nodes each query point reads and their shape values.
    ``sum(w, axis=1) == 1``, so a constant main field is reproduced exactly.

    Host/NumPy by design -- locating a point in a facet is a discrete search, the same eager-setup
    exception the rest of the tie machinery takes. The *result* is a plain gather, so a field read
    through it stays differentiable in the DOF values. It is **not** differentiable in the mesh
    coordinates: the weights are frozen at build time, so a shape derivative through a gap would need
    them re-derived in JAX.

    A query outside every facet is clamped to the nearest one, matching
    :func:`_periodic_facet_weights` -- at a rounding-width overhang that is the intended answer, and a
    genuine overhang is refused earlier by the coverage checks.
    """
    q = np.atleast_2d(np.asarray(query, dtype=float))
    m_facets = np.asarray(m_facets, dtype=int)
    k = int(m_facets.shape[1])
    if q.shape[0] == 0 or m_facets.shape[0] == 0:
        return np.zeros((0, k), dtype=int), np.zeros((0, k))

    if q.shape[1] == 1:  # 2-D interface: facets are edges, locate by interval containment
        t = q[:, 0]
        a, b = loc[m_facets[:, 0], 0], loc[m_facets[:, 1], 0]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        eps = 1.0e-9 * float(np.max(hi - lo)) if m_facets.shape[0] else 0.0
        inside = (t[:, None] >= lo[None, :] - eps) & (t[:, None] <= hi[None, :] + eps)
        # the containing edge, else the nearest one (overhang by a rounding width)
        dist = np.minimum(np.abs(t[:, None] - lo[None, :]), np.abs(t[:, None] - hi[None, :]))
        idx = np.where(inside.any(axis=1), np.argmax(inside, axis=1), np.argmin(dist, axis=1))
        span = b[idx] - a[idx]
        xi = np.clip(np.where(np.abs(span) < 1e-300, 0.0, (t - a[idx]) / np.where(span == 0, 1.0, span)), 0.0, 1.0)
        return m_facets[idx], _edge_shape(xi, k)

    # 3-D interface: facets are triangles, locate by barycentric containment
    v = loc[m_facets[:, :3]]  # (n_f, 3, 2)
    v0, v1 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    det = np.where(np.abs(det) < 1e-300, 1e-300, det)
    d = q[:, None, :] - v[None, :, 0, :]  # (n_q, n_f, 2)
    l1 = (d[..., 0] * v1[None, :, 1] - d[..., 1] * v1[None, :, 0]) / det[None, :]
    l2 = (v0[None, :, 0] * d[..., 1] - v0[None, :, 1] * d[..., 0]) / det[None, :]
    l0 = 1.0 - l1 - l2
    viol = np.maximum(0.0, -l0) + np.maximum(0.0, -l1) + np.maximum(0.0, -l2)
    idx = np.argmin(viol, axis=1)  # containing triangle, else the least-violating one
    rows = np.arange(q.shape[0])
    bary = np.stack([l0[rows, idx], l1[rows, idx], l2[rows, idx]], axis=-1)
    bary = np.clip(bary, 0.0, 1.0)
    bary = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-300)  # renormalise after clamping
    return m_facets[idx], _tri_shape(bary, k)


def interface_gap_data(
    secondary_qp: np.ndarray,
    m_facets: np.ndarray,
    points: np.ndarray,
    secondary_normals: np.ndarray,
    *,
    frame: np.ndarray | None = None,
    origin: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Everything a signed contact gap needs, precomputed on the host: ``(ids, w, g0)``.

    The gap between two surfaces is ``g = g0 - n . (u_s - u_m . Phi)``, split into a part fixed by the
    geometry and a part that moves with the solution:

    * ``g0 = n . (Phi(x_s) - x_s)`` — the **initial** separation at each secondary quadrature point, i.e.
      how far it stands off the main surface along the normal. Positive = a gap, negative = initial
      penetration. Zero for two coincident (tied) faces.
    * ``(ids, w)`` — the :func:`main_trace_weights` gather, so ``u_m . Phi`` at those points is a
      plain weighted sum of main DOFs and therefore differentiable in the solution.

    **Orientation — the one thing to get right.** ``n`` is the **secondary face's outward normal**, which on
    a contacting pair points *toward* the main body. Secondary motion along ``n`` therefore *closes* the
    gap, which is where the minus sign on the displacement term comes from. Handed the opposite normal
    this returns ``-g``, and every downstream sign follows it: a penetrating body reads as open, and
    ``max(0, -c*g)`` never activates, so free interpenetration is an exact root of the residual and
    Newton converges to it without complaint. The parameter is named ``secondary_normals`` rather than
    ``normals`` because there is nothing in the arrays themselves that distinguishes the two cases.

    The convention that follows is ``g > 0`` separated, ``g < 0`` interpenetrating, contact pressure
    ``p = max(0, -c*g) >= 0``, and a traction term ``+p * inner(n, phi)`` — the sign that adds a
    *positive*-definite ``+c (n.du)(n.phi)`` to the tangent, since ``dg/du_s = -n``.

    ``secondary_qp`` is ``(..., dim)`` physical quadrature points on the secondary face; ``secondary_normals`` is the
    matching outward unit normal per point (or one per face, broadcast by the caller). ``frame`` /
    ``origin`` default to a fit of the **main** facets' own tangent plane (:func:`_interface_frame`),
    which is what makes this work for coincident faces where no separating axis exists.

    Host/NumPy: locating each point on the main face is a discrete search, frozen at build time. So
    the gap is differentiable in the DOF values but **not** in the mesh coordinates -- shape-optimising
    through a contact gap would need the projection re-derived in JAX. It also assumes **small
    sliding**: the pairing is fixed, so a caller that slides must rebuild it per load step.
    """
    q = np.asarray(secondary_qp, dtype=float)
    pts = np.asarray(points, dtype=float)
    m_facets = np.asarray(m_facets, dtype=int)
    flat = q.reshape(-1, q.shape[-1])
    if frame is None or origin is None:
        m_pts = pts[np.unique(m_facets)]
        frame, origin = _interface_frame(m_pts, flat if flat.size else m_pts)

    ids, w = main_trace_weights((flat - origin) @ np.asarray(frame).T, m_facets, (pts - origin) @ np.asarray(frame).T)
    proj = np.einsum("qk,qkd->qd", w, pts[ids])  # Phi(x_s): the main-surface point under each query
    n = np.asarray(secondary_normals, dtype=float).reshape(-1, q.shape[-1])
    # Measured from the secondary TOWARD the main (``proj - x_s``), because ``n`` points that way -- see
    # the orientation paragraph above, which is the whole reason this line is not the other way round.
    g0 = np.einsum("qd,qd->q", n, proj - flat)
    lead = q.shape[:-1]
    return ids.reshape(*lead, -1), w.reshape(*lead, -1), g0.reshape(*lead)


def _periodic_facet_weights(
    t_query: np.ndarray,
    facet_node_ids: np.ndarray,
    loc: np.ndarray,
) -> List[Tuple[int, float]] | None:
    """Interpolation weights for a secondary at in-interface coord ``t_query`` on the
    main boundary facets (node-to-segment / mortar-lite identification).

    ``facet_node_ids`` is ``(n_facets, k)`` of global node ids; ``loc`` is the ``(n_nodes, dim-1)``
    projection of every node onto the interface tangent frame (:func:`_interface_frame`), so this
    routine never sees the across-interface coordinate. **2D** (``loc`` 1-D): facets are edges --
    columns 0,1 are the vertices, optional column 2 the midside node (``k == 3`` ⇒ P2). **3D**
    (``loc`` 2-D): facets are triangles -- columns 0,1,2 the vertices, optional columns 3,4,5 the
    edge midpoints (``k == 6`` ⇒ P2). Returns ``[(node_id, weight), ...]`` whose weights sum to 1
    (partition of unity ⇒ constants reproduced; linear/quadratic-on-the-facet reproduced exactly).
    """
    tq = np.atleast_1d(np.asarray(t_query, dtype=float))
    facet_node_ids = np.asarray(facet_node_ids, dtype=int)
    if facet_node_ids.ndim != 2 or facet_node_ids.shape[0] == 0:
        return None
    k = facet_node_ids.shape[1]
    # Only P1 and P2 facets have shape functions here. The branches below used to read `k < 3` /
    # `k < 6` and fall through to the P2 formulas for ANYTHING larger, so a P3 edge was interpolated
    # as if its node at 1/3 were the midpoint and its node at 2/3 did not exist. Weights still summed
    # to 1, so a constant transferred and nothing complained -- but a LINEAR field came out wrong by
    # 0.25-0.33 on a field of range 2. Refuse instead.
    _allowed = (2, 3) if tq.shape[0] == 1 else (3, 4, 6)  # edges | triangle P1, QUAD Q1, triangle P2
    if k not in _allowed:
        raise NotImplementedError(
            f"A non-matching periodic/tied interface supports {'P1/P2 edges' if tq.shape[0] == 1 else 'P1/P2 triangles and Q1 quadrilaterals'}; "
            f"this one has {k}-node facets. Use a conforming mesh for the tie, or drop the element order -- "
            "interpolating it with the formulas below would silently misplace the extra nodes."
        )

    if tq.shape[0] == 1:  # 2D: locate the main edge spanning the secondary's in-interface coord
        t = float(tq[0])
        a_ids, b_ids = facet_node_ids[:, 0], facet_node_ids[:, 1]
        ta, tb = loc[a_ids, 0], loc[b_ids, 0]
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        span = hi - lo
        eps = 1.0e-9 * (float(np.max(span)) if span.size else 1.0)
        inside = (t >= lo - eps) & (t <= hi + eps)
        if inside.any():
            idx = int(np.argmax(inside))
        else:  # outside every edge (rounding at a face end) -> nearest edge
            idx = int(np.argmin(np.minimum(np.abs(t - lo), np.abs(t - hi))))
        a, b = int(a_ids[idx]), int(b_ids[idx])
        L = float(tb[idx] - ta[idx])
        xi = 0.0 if abs(L) < eps else (t - float(ta[idx])) / L  # local coord, a:0 -> b:1
        xi = min(1.0, max(0.0, xi))
        if k < 3:  # P1 edge: linear
            return [(a, 1.0 - xi), (b, xi)]
        m = int(facet_node_ids[idx, 2])  # P2 edge (a, b, mid): quadratic Lagrange at xi = 0, 1, 0.5
        return [(a, 2.0 * (xi - 0.5) * (xi - 1.0)), (b, 2.0 * xi * (xi - 0.5)), (m, -4.0 * xi * (xi - 1.0))]

    if tq.shape[0] == 2 and k == 4:  # 3D: a QUADRILATERAL facet (a hexahedron's face)
        # A triangle's map is affine, so its inverse is the closed-form barycentric solve below. A
        # quadrilateral's is BILINEAR and has no closed-form inverse, so the reference coordinates come
        # from Newton (:func:`fem_lagrange._invert_tensor_map`, the same inverse the quad solution
        # transfer uses). Reaching the branch below instead would interpolate the facet from three of
        # its four nodes -- silently, and wrong by O(1) on anything but a linear field.
        from .fem_lagrange import _invert_tensor_map

        V = loc[facet_node_ids]  # (F, 4, 2) facet corners in the interface frame, VTK cyclic order
        perm = [0, 1, 3, 2]  # VTK walks the perimeter; basix orders lexicographically
        xi, N = _invert_tensor_map(V[:, perm], np.broadcast_to(tq[None, :], (V.shape[0], 2)), "quadrilateral")
        # the containing facet (0 <= xi <= 1 on both axes); else the least-violating one, mirroring
        # the barycentric branch's handling of a shared edge or rounding at a face end
        viol = (np.maximum(0.0, -xi) + np.maximum(0.0, xi - 1.0)).sum(axis=1)
        idx = int(np.argmin(viol))
        ids = facet_node_ids[idx][perm]
        return [(int(i), float(wt)) for i, wt in zip(ids, N[idx])]

    if tq.shape[0] == 2:  # 3D: locate the main triangle containing the secondary, barycentric weights
        pa = loc[facet_node_ids[:, 0]]
        pb = loc[facet_node_ids[:, 1]]
        pc = loc[facet_node_ids[:, 2]]
        v0, v1, v2 = pb - pa, pc - pa, tq[None, :] - pa
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        denom = d00 * d11 - d01 * d01
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        l1 = (d11 * d20 - d01 * d21) / denom  # vertex b
        l2 = (d00 * d21 - d01 * d20) / denom  # vertex c
        l0 = 1.0 - l1 - l2  # vertex a
        # the containing triangle (all barycentrics >= 0); else the least-violating one (shared edge / rounding)
        viol = np.maximum(0.0, -l0) + np.maximum(0.0, -l1) + np.maximum(0.0, -l2)
        idx = int(np.argmin(viol))
        a, b, c = (int(facet_node_ids[idx, j]) for j in range(3))
        L0, L1, L2 = float(l0[idx]), float(l1[idx]), float(l2[idx])
        if k < 6:  # P1 triangle: barycentric
            return [(a, L0), (b, L1), (c, L2)]
        # P2 triangle (a, b, c, mab, mbc, mca): quadratic shape functions in barycentric coords
        mab, mbc, mca = (int(facet_node_ids[idx, j]) for j in range(3, 6))
        return [
            (a, L0 * (2.0 * L0 - 1.0)),
            (b, L1 * (2.0 * L1 - 1.0)),
            (c, L2 * (2.0 * L2 - 1.0)),
            (mab, 4.0 * L0 * L1),
            (mbc, 4.0 * L1 * L2),
            (mca, 4.0 * L2 * L0),
        ]
    return None


def build_periodic_prolongation(
    points: np.ndarray,
    pairs: Sequence[Tuple[str, str]],
    tag_indices: Dict[str, np.ndarray],
    *,
    vec: int = 1,
    tol: float | None = None,
    facets: Dict[str, np.ndarray] | None = None,
    phases: Sequence[complex] | None = None,
) -> Dict[str, object]:
    """Build the node-level periodic prolongation matrix ``P``.

    Identifies each secondary-face node with the main face. When the two faces have
    the **same node layout** (structured / conforming) this is an exact 0/1
    node-to-node map. When they **don't** (unstructured / non-matching), a secondary
    that has no main node within ``tol`` is tied to the main face by one of two
    couplings, and the returned ``coupling`` key says which was used:

    * ``"mortar"`` — both faces carry facet connectivity and the main face covers
      the secondary face, so the integrated dual-mortar constraint applies:
      :func:`_mortar_rows_2d` in 2-D (edge facets, interval clipping) or
      :func:`_mortar_rows_3d` in 3-D (triangle facets, polygon clipping).
      Variationally consistent, momentum-balanced, and it passes the constant-stress
      patch test. Faces tagged by a ``domain.tag`` predicate include their corners and
      qualify; a face tagged from geometry that drops its corners does not — see
      :func:`_faces_span_the_same_extent` / :func:`_main_covers_secondary_3d`.
    * ``"collocated"`` — otherwise (native 1-D chains, a tag that selects nodes but
      no whole facet, or two faces that do not cover each other) the secondary is tied to
      the main *facet* it lands on by **node-to-segment interpolation** (linear for
      P1, quadratic for P2). Consistent (partition of unity) and exact for fields the
      main facet can represent, but collocated rather than integrated. See
      :func:`_periodic_facet_weights`.

    Both couplings pass the **linear patch test**: this is a main-secondary elimination,
    which reproduces a linear solution exactly whenever ``P`` does, and node-to-segment
    interpolation is linearly complete. What separates them is that mortar imposes the
    integral (L2) constraint, so it is more accurate for fields the main space cannot
    represent — measured at 4-40% lower RMS error on a non-matching 3-D interface.

    Parameters
    ----------
    points:
        ``(n_nodes, dim)`` array of FEM node coordinates (the assembly mesh).
    pairs:
        Ordered ``(main_tag, secondary_tag)`` boundary pairings, e.g.
        ``[("left", "right"), ("bottom", "top")]``.
    tag_indices:
        Mapping from boundary tag name to the global node ids on that tag.
    vec:
        Number of scalar components per node. ``vec > 1`` returns the
        node-major expansion ``kron(P_node, I_vec)``.
    tol:
        Coordinate-matching tolerance for the in-interface coordinates. When
        ``None`` it is derived from the bounding-box diagonal.
    facets:
        Optional ``{main_tag: (n_facets, k) node-id array}`` of the main
        boundary facets, required only for the non-matching (interpolatory) path.

    Returns
    -------
    dict with keys:
        ``P``               : ``(n_full, n_red)`` dense jnp prolongation matrix.
        ``P_node``          : node-level prolongation (``vec == 1`` form).
        ``kept_nodes``      : sorted global ids of the retained (main/free) nodes.
        ``secondary_to_main`` : resolved secondary-node -> main-node mapping (exact ties).
        ``n_full``          : full node count.
        ``n_red``           : reduced node count.
        ``vec``             : component count used.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_nodes = pts.shape[0]
    facets = facets or {}
    if phases is None:
        phases = [1.0] * len(pairs)
    # Bloch/quasi-periodic: a non-unit phase makes P complex; a plain periodic cell keeps P real 0/1
    # so the fast (selection) reduction path is untouched.
    is_bloch = any(abs(complex(p) - 1.0) > 1e-12 for p in phases)

    if tol is None:
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        tol = max(span, 1.0) * 1.0e-6

    # secondary -> main node (exact tie, weight = Bloch phase); secondary -> [(main node, weight)] (interp).
    secondary_to_main: Dict[int, int] = {}
    secondary_phase: Dict[int, complex] = {}
    secondary_interp: Dict[int, List[Tuple[int, float]]] = {}
    n_mortar = n_collocated = 0  # how each non-matching secondary was tied -> reported in ``coupling``

    for (main_tag, secondary_tag), ph in zip(pairs, phases):
        if main_tag not in tag_indices or secondary_tag not in tag_indices:
            raise KeyError(
                f"Periodic pair ({main_tag!r}, {secondary_tag!r}) refers to a tag "
                f"that is not present in the mesh. Known tags: {sorted(tag_indices)}."
            )

        m_ids = np.asarray(tag_indices[main_tag], dtype=int).reshape(-1)
        s_ids = np.asarray(tag_indices[secondary_tag], dtype=int).reshape(-1)
        if s_ids.size == 0:  # nothing to eliminate on this pair
            continue

        # THE SECONDARY MUST BE THE FINER SIDE. Its DOFs are eliminated in favour of an interpolation from
        # the main, so eliminating the fine side onto a coarse one is right, and the reverse discards
        # exactly the resolution the fine mesh was built for. Measured on a coating/substrate tie with
        # 81 nodes against 10: correct order gave the exact interface value, reversed was off by 10.62%
        # -- with no error, which is why this reorders rather than trusting the caller to know.
        # Only when the side that would BECOME the main carries facet connectivity: the main is
        # what a non-matching secondary interpolates from, so swapping without it just moves the
        # problem (and a caller that supplied one side's facets meant that side to be the main).
        _can_swap = (facets or {}).get(secondary_tag) is not None
        if _can_swap and s_ids.size < 0.9 * m_ids.size:
            if abs(complex(ph) - 1.0) > 1e-12:
                # A Bloch tie carries a phase e^{ik.L} that is direction-dependent: swapping the two
                # sides requires conjugating it. Rather than do that silently, say so and leave it.
                _log.warning(
                    f"periodic tie ({secondary_tag!r} -> {main_tag!r}): the eliminated (secondary) side has "
                    f"{s_ids.size} nodes against the main's {m_ids.size}, which discards the finer "
                    "side's interface resolution. This is a BLOCH tie, whose phase is direction-"
                    "dependent, so it was NOT reordered automatically -- swap the operands and "
                    "conjugate the phase to fix it."
                )
            else:
                _log.warning(
                    f"periodic tie ({secondary_tag!r} -> {main_tag!r}): the eliminated (secondary) side has "
                    f"{s_ids.size} nodes against the main's {m_ids.size}. The secondary must be the finer "
                    f"side, so the two were swapped -- write u({main_tag}) - u({secondary_tag}) to make "
                    "that explicit."
                )
                main_tag, secondary_tag = secondary_tag, main_tag
                m_ids, s_ids = s_ids, m_ids
        m_pts = pts[m_ids]
        s_pts = pts[s_ids]

        # Project every node onto the interface's own tangent plane. That removes the across-interface
        # coordinate whether the two faces are separated by a lattice vector (periodic) or coincident
        # (tied) -- the axis-dropping this replaces could only express the former. See _interface_frame.
        frame, origin = _interface_frame(m_pts, s_pts)
        loc = (pts - origin) @ frame.T  # (n_nodes, dim-1) in-interface coordinates
        m_loc, s_loc = loc[m_ids], loc[s_ids]

        # Nearest in-interface main node for every secondary node.
        d2 = np.sum((s_loc[:, None, :] - m_loc[None, :, :]) ** 2, axis=-1)
        nn = np.argmin(d2, axis=1) if m_ids.size else np.zeros(len(s_ids), dtype=int)
        dist = np.sqrt(d2[np.arange(len(s_ids)), nn]) if m_ids.size and len(s_ids) else np.zeros(len(s_ids))

        # A non-matching secondary is tied by a MORTAR coupling when the interface is 2-D (edge facets) and
        # both faces carry facet connectivity -- an integrated constraint that passes the patch test.
        # Without secondary facets (native 1-D chains, a tag that selects nodes but no whole facet) there is
        # nothing to integrate over, so those nodes keep the collocated node-to-segment weights. Which
        # one each tie used is reported back in ``coupling`` rather than left to guesswork.
        mortar: Dict[int, List[Tuple[int, float]]] = {}
        s_fc, m_fc = facets.get(secondary_tag), facets.get(main_tag)
        if s_fc is not None and m_fc is not None and loc.shape[1] in (1, 2):
            span = float(np.ptp(loc)) if loc.size else 1.0
            if loc.shape[1] == 1:  # 2-D interface: edge facets, clipping is an interval intersection
                if _faces_span_the_same_extent(s_fc, m_fc, loc, span=span):
                    mortar = _mortar_rows_2d(s_fc, m_fc, loc, span=span)
            # 3-D: triangle facets, polygon clipping. P2 triangles have no dual basis of this form
            # (their vertex functions integrate to zero) -- see _tri_dual_available.
            elif _tri_dual_available(int(np.shape(s_fc)[1])) and _main_covers_secondary_3d(s_fc, m_fc, loc):
                mortar = _mortar_rows_3d(s_fc, m_fc, loc, span=span)

        for k, sid in enumerate(s_ids):
            if m_ids.size and dist[k] <= tol:  # conforming: exact node-to-node (corners land here too)
                secondary_to_main[int(sid)] = int(m_ids[nn[k]])
                secondary_phase[int(sid)] = complex(ph)
                continue
            # non-matching: mortar rows when available, else collocated node-to-segment interpolation
            # (either way the weights are scaled by the Bloch phase)
            w = mortar.get(int(sid))
            if w is not None:
                n_mortar += 1
            else:
                w = _periodic_facet_weights(s_loc[k], facets.get(main_tag), loc) if facets else None
                n_collocated += w is not None
            if w is None:
                raise ValueError(
                    f"Periodic matching for ({main_tag!r}, {secondary_tag!r}) failed at secondary node {int(sid)}: "
                    f"nearest main node is {float(dist[k]):.3e} away (tol {tol:.3e}) and no main facet "
                    "connectivity was supplied for interpolation. Pass `facets=` (unstructured) or use a "
                    "conforming mesh."
                )
            secondary_interp[int(sid)] = [(int(m), complex(ph) * wt) for (m, wt) in w]

    secondary_set = set(secondary_to_main) | set(secondary_interp)

    # Each secondary is a linear combination of other nodes (exact: one main, weight 1; interpolated:
    # facet shape-function weights). Those nodes may themselves be secondarys — a corner is a secondary in
    # several directions, and an interpolation can land on a main edge whose endpoint is itself a
    # secondary — so resolve every secondary **transitively** to kept (main) nodes. This handles any number
    # of periodic directions (e.g. a doubly-periodic cell) with a single general mechanism.
    raw: Dict[int, List[Tuple[int, complex]]] = {
        sid: [(m, secondary_phase.get(sid, 1.0))] for sid, m in secondary_to_main.items()
    }
    raw.update({sid: list(ws) for sid, ws in secondary_interp.items()})
    resolved: Dict[int, Dict[int, float]] = {}

    def _expand(node: int, stack: frozenset) -> Dict[int, float]:
        if node not in secondary_set:
            return {node: 1.0}
        if node in resolved:
            return resolved[node]
        if node in stack:
            raise ValueError(f"Periodic identification is cyclic at node {node}; check the tie directions.")
        down = stack | {node}
        out: Dict[int, float] = {}
        for n2, w in raw[node]:
            for kept_node, wk in _expand(n2, down).items():
                out[kept_node] = out.get(kept_node, 0.0) + w * wk
        resolved[node] = out
        return out

    kept_nodes: List[int] = [i for i in range(n_nodes) if i not in secondary_set]
    reduced_index = {node: r for r, node in enumerate(kept_nodes)}
    n_red = len(kept_nodes)

    # The prolongation is a SELECTION matrix (0/1, plus interpolation weights for non-matching faces)
    # -- it must be SPARSE, never dense: O(n_full) nonzeros vs O(n_full * n_red) dense (GBs at large
    # N). Collect (row, col, weight) entries node-by-node, then a BCOO; a vector field expands each
    # node entry to ``vec`` component entries (a Kronecker of the node map with I_vec).
    import jax.experimental.sparse as jsparse

    rows: List[int] = []
    cols: List[int] = []
    data: List[complex] = []
    for i in range(n_nodes):
        if i in secondary_set:
            for kept_node, weight in _expand(i, frozenset()).items():
                rows.append(i)
                cols.append(reduced_index[kept_node])
                data.append(weight if is_bloch else float(np.real(weight)))
        else:
            rows.append(i)
            cols.append(reduced_index[i])
            data.append(1.0)

    # Informational exact-chain map (single kept main per exact secondary); interpolated secondarys omitted.
    final_main = {sid: next(iter(_expand(sid, frozenset()))) for sid in secondary_to_main}

    rows_a = np.asarray(rows, dtype=np.int64)
    cols_a = np.asarray(cols, dtype=np.int64)
    data_a = np.asarray(data, dtype=np.complex128 if is_bloch else np.float64)
    if vec > 1:  # kron(P_node, I_vec): each node entry -> vec component entries
        comp = np.arange(vec, dtype=np.int64)
        rows_a = (rows_a[:, None] * vec + comp[None, :]).reshape(-1)
        cols_a = (cols_a[:, None] * vec + comp[None, :]).reshape(-1)
        data_a = np.repeat(data_a, vec)
    n_full, n_red_full = int(n_nodes * vec), int(n_red * vec)
    P = jsparse.BCOO((jnp.asarray(data_a), jnp.asarray(np.stack([rows_a, cols_a], axis=1))), shape=(n_full, n_red_full))

    return {
        "P": P,
        "P_node": P,  # sparse; equals the node-level map when vec == 1
        "kept_nodes": np.asarray(kept_nodes, dtype=np.int64),
        "secondary_to_main": final_main,
        "n_full": n_full,
        "n_red": n_red_full,
        "vec": int(vec),
        # whether P is a one-main-per-secondary selection (conforming) -> the sparse remap reduction is
        # exact; computed once here (eager, concrete P) so the reduce path never inspects P under trace.
        "is_selection": _is_selection(P),
        # Bloch/quasi-periodic: P is complex, so the reduction is Hermitian (P^H A P) and the reduced
        # complex system can't be split into independent real/imag legs.
        "is_bloch": bool(is_bloch),
        # How the NON-matching secondarys were tied: "conforming" (there were none), "mortar" (integrated,
        # passes the patch test), "collocated" (node-to-segment; no secondary facets to integrate over) or
        # "mixed". Reported rather than inferred, so a caller never has to guess which it got.
        "coupling": (
            "conforming"
            if not (n_mortar or n_collocated)
            else "mortar"
            if not n_collocated
            else "collocated"
            if not n_mortar
            else "mixed"
        ),
    }


def build_periodic_prolongation_nonnodal(
    n_verts: int,
    n_edges: int,
    vertex_points: np.ndarray,
    edge_midpoints: np.ndarray,
    edge_normals: np.ndarray,
    vtags: Dict[str, np.ndarray],
    etags: Dict[str, np.ndarray],
    pairs: Sequence[Tuple[str, str]],
    *,
    tol: float | None = None,
) -> Dict[str, object]:
    """DOF-level periodic prolongation for a **Morley** (C¹ non-nodal) field.

    Morley DOFs: one *value* DOF per vertex at ids ``[0, n_verts)``; one *normal-derivative* DOF per
    global edge at ids ``[n_verts, n_verts + n_edges)``. The node-based ``build_periodic_prolongation``
    can't represent this (it assumes DOFs = points × components). Here the two DOF blocks are tied
    separately and block-diagonally combined:

    * **value DOFs** — tie by vertex coordinate, weight ``+1`` (delegated to the nodal builder on the
      vertex points, so corners / transitive chains are handled identically to Lagrange);
    * **edge normal-derivative DOFs** — tie boundary-edge → boundary-edge by *midpoint*, weight
      ``sign(n_secondary · n_main)`` where each edge's normal is its globally-oriented reference normal
      (``fem_nonnodal``: ``n = R90·(P[hi] − P[lo])``). For axis-aligned periodic boundaries this dot is
      ``±1`` — the tie **sign is derived from geometry, not assumed** (and gated by an MMS test).

    Non-conforming periodic boundaries (a secondary edge with no transverse-matching main edge) raise a
    clear error rather than silently mis-coupling (mortar interpolation of derivative DOFs is out of
    scope). Returns the same dict shape as :func:`build_periodic_prolongation`.
    """
    import jax.experimental.sparse as jsparse

    # --- value-DOF block: reuse the nodal builder on the vertex points (weight +1) ---
    vred = build_periodic_prolongation(np.asarray(vertex_points, dtype=np.float64), pairs, vtags, vec=1, tol=tol)
    Pv = vred["P"]
    n_vred = int(vred["n_red"])
    vkept = np.asarray(vred["kept_nodes"], dtype=np.int64)

    # --- edge-derivative block: signed boundary-edge ties by midpoint ---
    emid = np.asarray(edge_midpoints, dtype=np.float64).reshape(n_edges, -1)
    enrm = np.asarray(edge_normals, dtype=np.float64).reshape(n_edges, -1)
    span = float(np.linalg.norm(emid.max(0) - emid.min(0))) if n_edges else 1.0
    etol = tol if tol is not None else max(span, 1.0) * 1.0e-6
    e_secondary_main: Dict[int, Tuple[int, float]] = {}
    for mtag, stag in pairs:
        me = np.asarray(etags.get(mtag, []), dtype=int).reshape(-1)
        se = np.asarray(etags.get(stag, []), dtype=int).reshape(-1)
        if me.size == 0 or se.size == 0:
            continue
        mp, sp = emid[me], emid[se]
        axis = int(np.argmax(np.abs(mp.mean(0) - sp.mean(0))))
        trans = [d for d in range(mp.shape[1]) if d != axis]
        d2 = np.sum((sp[:, None, :][:, :, trans] - mp[None, :, :][:, :, trans]) ** 2, axis=-1)
        nn = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(se)), nn])
        for kk, sid in enumerate(se):
            if dist[kk] > etol:
                raise ValueError(
                    f"jno.fem periodic (non-nodal C¹): secondary edge {int(sid)} on {stag!r} has no "
                    f"transverse-matching main edge on {mtag!r} (nearest {dist[kk]:.3e} > tol {etol:.3e}). "
                    "Periodic C¹ needs a conforming (matching) periodic boundary; non-matching edge "
                    "coupling (mortar interpolation of derivative DOFs) is not supported."
                )
            mid = int(me[nn[kk]])
            dot = float(np.dot(enrm[int(sid)], enrm[mid]))
            e_secondary_main[int(sid)] = (mid, 1.0 if dot >= 0.0 else -1.0)

    def _resolve_edge(e: int) -> Tuple[int, float]:
        seen: set = set()
        sign = 1.0
        while e in e_secondary_main:
            if e in seen:
                raise ValueError(f"jno.fem periodic (non-nodal C¹): cyclic edge tie at edge {e}.")
            seen.add(e)
            e, s = e_secondary_main[e]
            sign *= s
        return e, sign

    ekept = [e for e in range(n_edges) if e not in e_secondary_main]
    e_red_idx = {e: r for r, e in enumerate(ekept)}
    n_ered = len(ekept)

    # --- combined DOF-level BCOO  [[Pv, 0], [0, Pe]]  (edge rows/cols offset past the value block) ---
    Pv_idx = np.asarray(Pv.indices)
    Pv_dat = np.asarray(Pv.data)
    rows = list(Pv_idx[:, 0])
    cols = list(Pv_idx[:, 1])
    data = list(Pv_dat)
    for e in range(n_edges):
        m, sign = _resolve_edge(e)
        rows.append(n_verts + e)
        cols.append(n_vred + e_red_idx[m])
        data.append(sign)
    n_full = n_verts + n_edges
    n_red = n_vred + n_ered
    P = jsparse.BCOO(
        (
            jnp.asarray(np.asarray(data, dtype=np.float64)),
            jnp.asarray(np.stack([np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)], axis=1)),
        ),
        shape=(n_full, n_red),
    )
    kept = np.concatenate([vkept, n_verts + np.asarray(ekept, dtype=np.int64)]) if n_ered else vkept
    return {
        "P": P,
        "P_node": P,
        "kept_nodes": kept,
        "secondary_to_main": {},
        "n_full": n_full,
        "n_red": n_red,
        "vec": 1,
        # signed edge ties (weight −1) are not a 0/1 selection -> take the general PᵀAP reduce path.
        "is_selection": bool(np.all(np.abs(np.asarray(data)) == 1.0) and np.all(np.asarray(data) >= 0.0)),
    }


# ---------------------------------------------------------------------------
# Sparse (BCOO) Dirichlet row/column operations
#
# The native assembler returns the global matrix as a BCOO (``O(nnz)``, never the
# dense ``O(n^2)`` array). These keep the three dense Dirichlet primitives
# (``.at[d,:].set(0)``, ``.at[:,d].set(0)``, ``.at[d,d].set(1)``) sparse: a row/col
# is "zeroed" by masking the stored values (nse unchanged — zeroed entries stay as
# explicit-zero triplets, harmless to matvec); a unit diagonal is appended as
# ``(d, d, 1)`` triplets (BCOO sums duplicates on matvec / todense / sum_duplicates,
# so the zeroed original ``(d,d)`` plus the appended ``1`` is exactly ``1``). All are
# static-``nse``, ``jit``-safe and differentiable in the stored values.
# ---------------------------------------------------------------------------


def bcoo_zero_rows(A, dofs):
    """``A.at[dofs, :].set(0)`` for a BCOO ``A`` — mask out every stored entry on a ``dofs`` row."""
    isd = jnp.zeros(A.shape[0], A.data.dtype).at[dofs].set(1.0)
    keep = 1.0 - isd[A.indices[:, 0]]
    return jsparse.BCOO((A.data * keep, A.indices), shape=A.shape)


def bcoo_zero_rows_cols(A, dofs):
    """``A.at[dofs, :].set(0).at[:, dofs].set(0)`` for a BCOO ``A`` (symmetric elimination, no diagonal)."""
    isd = jnp.zeros(A.shape[0], A.data.dtype).at[dofs].set(1.0)
    keep = (1.0 - isd[A.indices[:, 0]]) * (1.0 - isd[A.indices[:, 1]])
    return jsparse.BCOO((A.data * keep, A.indices), shape=A.shape)


def bcoo_identity_rows(A, mask):
    """Replace the rows of a BCOO ``A`` selected by a **traced boolean mask** with identity rows.

    The index-array helpers above cannot serve here: a min-map's active set is a function of the current
    iterate, so it is a traced mask of static length, not a list of DOF numbers. Same two moves — scale
    the stored entries by the row's keep factor, then append a full diagonal carrying the mask as its
    value (0 on an inactive row, so those appended entries are exact no-ops and duplicate indices sum)."""
    m = jnp.asarray(mask).astype(A.data.dtype)
    n = A.shape[0]
    data = A.data * (1.0 - m[A.indices[:, 0]])
    diag = jnp.arange(n, dtype=A.indices.dtype)
    eye_idx = jnp.stack([diag, diag], axis=1)
    return jsparse.BCOO((jnp.concatenate([data, m]), jnp.concatenate([A.indices, eye_idx])), shape=A.shape)


def bcoo_set_unit_diag(A, dofs):
    """Append unit-diagonal triplets ``(d, d, 1)`` for ``d in dofs`` to a BCOO ``A`` (after the rows
    were zeroed, this makes ``A[d, d] == 1`` exactly — duplicate indices are summed)."""
    eye_idx = jnp.stack([dofs, dofs], axis=1).astype(A.indices.dtype)
    eye_dat = jnp.ones(jnp.asarray(dofs).shape[0], A.data.dtype)
    return jsparse.BCOO((jnp.concatenate([A.data, eye_dat]), jnp.concatenate([A.indices, eye_idx])), shape=A.shape)


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


def chunk_budget_bytes():
    """Bytes one element chunk may occupy, taken from the device rather than tuned to one."""
    try:
        limit = jax.local_devices()[0].memory_stats().get("bytes_limit")
    except Exception:  # noqa: BLE001 -- CPU backends expose no memory stats
        limit = None
    if not limit:
        return _CHUNK_FALLBACK_BYTES
    return max(_CHUNK_FALLBACK_BYTES // 2, int(limit * _CHUNK_MEMORY_FRACTION))


def _balanced_chunk(n_items: int, chunk: int) -> int:
    """The smallest chunk that still splits ``n_items`` into the SAME number of pieces.

    ``lax.map(..., batch_size=c)`` compiles the element kernel **twice** when ``c`` does not divide
    the item count: once as the scan body, and once more, unrolled, for the leftover tail. That
    duplicate is invisible in the source and expensive in XLA -- on a Taylor-Hood Stokes assembly
    (21,138 cells, chunk 8,192, so two full chunks and a 4,754 tail) it was 4.9x the element-program
    compile time and **2.0x the whole build, 10.3 s -> 5.3 s**, for a bit-identical answer.

    Rebalancing is free in both directions that matter: the chunk COUNT is unchanged, so there is no
    extra scan step, and each chunk is no larger than the one asked for, so peak memory only ever
    falls. It cannot always divide evenly (100 items in 3 chunks is 34+34+32), and then this is
    simply a no-op -- the tail is at most one chunk either way.
    """
    k = -(-int(n_items) // int(chunk))  # pieces at the requested size, unchanged by the rebalance
    return max(1, -(-int(n_items) // k))


def cell_chunk(n_items: int, n_test: int, n_local: int, setting=None):
    """Cells per chunk, or ``None`` to keep the plain single `vmap`.

    The per-cell cost is the element block *including its AD tangent* (`n_test * n_local**2`), the
    jacobian's dominant intermediate. The residual's is smaller, so it gets chunked somewhat more
    finely than it strictly needs -- a deliberate simplification: one policy, one explanation, and
    the cost of an extra chunk is a scan step, not a re-computation.

    Whatever size that policy (or an explicit ``setting``) lands on is then rebalanced by
    :func:`_balanced_chunk` so the chunks come out even where they can. That is a compile-time
    concern, not a memory one, and it never raises either cost -- see there."""
    if setting == 0:
        return None  # explicitly disabled: one vmap over every cell
    if setting is not None:  # explicit cells/chunk: an upper bound, so rebalancing still honours it
        return None if n_items <= setting else _balanced_chunk(n_items, int(setting))
    per = max(1, int(n_test) * int(n_local) * int(n_local) * 8)
    chunk = max(1, chunk_budget_bytes() // per)
    chunk = max(chunk, _CHUNK_MIN_CELLS)  # never starve the device to honour the byte cap
    if n_items <= chunk:
        return None  # one chunk anyway -- skip the scan overhead entirely
    return _balanced_chunk(n_items, chunk)


_ELEM_MAP_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
#: LRU bound. One assembled problem registers ~6 entries and a nonlinear solve ~4 more, so this holds
#: roughly twenty problems -- comfortably more than the repeated-solve case it exists for. It is not
#: unbounded because an entry pins the values baked into its compilation (see
#: :func:`_bake_fingerprint`), which for a discarded problem means keeping that mesh's arrays alive;
#: at 128 that is bounded by ~20 dead problems rather than by the whole session.
_ELEM_MAP_CACHE_MAX = 128

#: Duplicate-collapse plans, keyed on the CONTENT of the triplet pattern they were computed from.
#:
#: See :func:`compress_plan` for why content rather than identity (a remesh changes the content and
#: misses, so staleness is impossible) and for the measured hash-vs-work ratio.
#:
#: Bounded at 4 because an entry pins DEVICE arrays, and the ``inverse`` leg is one int32 per RAW
#: triplet -- the largest array the plan holds (38 MiB at 9.5M triplets, against ~3 MiB for the unique
#: indices). Four covers the two-to-three patterns one build registers plus a neighbour, which is the
#: repeated-build case this exists for; holding a dead problem's pattern any longer costs device
#: memory for nothing.
_PLAN_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_PLAN_CACHE_MAX = 4


#: Content digests of baked array leaves, keyed by object id. Each entry PINS the array it was
#: computed from -- numpy arrays are not weakref-able, and pinning is what keeps the id in the key
#: from being recycled onto a different array while the entry is live (the same pattern
#: ``_ELEM_MAP_CACHE`` documents for its baked leaves). Bounded: an entry is one array pin + a small
#: digest tuple, and one build touches ~20 array leaves.
_LEAF_DIGEST_CACHE: "OrderedDict[int, tuple]" = OrderedDict()
_LEAF_DIGEST_CACHE_MAX = 4096

#: The CONTENT-keyed twin of ``_ELEM_MAP_CACHE`` -- same values (``(jitted, pins)`` tuples), keyed on
#: what the compilation depends on *by value* rather than by object identity. This is what makes a
#: REBUILD of an identical problem (fresh mesh arrays, fresh closures, identical content) reuse the
#: compiled programs instead of paying trace + XLA compile again: measured on 3-D Poisson at 10k
#: nodes, the 8 recompiles were 1.25 s of a 1.56 s warm rebuild.
_ELEM_MAP_CONTENT: "OrderedDict[tuple, tuple]" = OrderedDict()
_ELEM_MAP_CONTENT_MAX = 128

#: Diagnostics for tests and for finding out WHY a rebuild failed to hit: hit/miss counters plus a
#: per-type tally of the leaves that defeated content keying (``content_bail``).
_ELEM_MAP_STATS: Dict[str, Any] = {"id_hits": 0, "content_hits": 0, "misses": 0, "content_bail": {}}


def _array_digest(leaf):
    """Content identity of a baked array leaf, memoized by object id (arrays are immutable).

    Includes dtype, shape and -- for JAX arrays -- ``weak_type`` and the sharding string, because each
    of those changes the program a jit would bake, not just the numbers in it. ``np.asarray`` on a
    device array is a host transfer; it happens once per array object (the memo), at build time, and
    only on the rebuild-miss path -- never per solve."""
    key = id(leaf)
    hit = _LEAF_DIGEST_CACHE.get(key)
    if hit is not None and hit[0] is leaf:
        _LEAF_DIGEST_CACHE.move_to_end(key)
        return hit[1]
    if isinstance(leaf, jax.core.Tracer):
        return None
    extra = ()
    if isinstance(leaf, jax.Array):
        extra = (bool(getattr(leaf, "weak_type", False)), str(getattr(leaf, "sharding", "")))
    arr = np.ascontiguousarray(np.asarray(leaf))
    digest = (
        "arr",
        arr.dtype.str,
        arr.shape,
        extra,
        _hashlib.blake2b(memoryview(arr).cast("B"), digest_size=16).digest(),
    )
    _LEAF_DIGEST_CACHE[key] = (leaf, digest)
    while len(_LEAF_DIGEST_CACHE) > _LEAF_DIGEST_CACHE_MAX:
        _LEAF_DIGEST_CACHE.popitem(last=False)
    return digest


def _callable_token(fn, seen):
    """Content token of a plain Python function: its code object plus the recursive tokens of its
    closure -- ``_fn_content_key`` without the treedef packaging. Bound methods and builtins hash by
    their qualified name (their behaviour is version-stable within a process)."""
    import types as _types

    if isinstance(fn, _types.FunctionType):
        sub = _fn_content_key(fn, None, seen)
        return None if sub is None else ("fn",) + sub
    qual = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    mod = getattr(fn, "__module__", None)
    if qual and mod:
        return ("callable", mod, qual)
    return None


def _expr_digest(node, seen=None):
    """Structural content digest of a traced expression -- the coefficient trees the element
    functions bake in. **Allow-list, strict**: a node type this walker does not positively know how
    to fingerprint returns ``None``, which disables content keying for that closure (safe: a miss
    costs a recompile, a wrong hit would be a wrong operator). The bail is tallied by type in
    ``_ELEM_MAP_STATS["content_bail"]`` so coverage gaps are measurable, not guessed.

    Deliberately EXCLUDED from every token: per-build counters (``op_id``, ``frozen_id``,
    ``layer_id``) -- they differ across rebuilds without changing the compiled program, and any node
    whose counter DOES key runtime lookups (``FrozenField``'s gather table, ``ModelCall``) is not on
    the allow-list at all."""
    from ...trace import RegionMask as _RM
    from ...trace import TagMask as _TM

    if seen is None:
        seen = set()
    if id(node) in seen:
        return None  # a true back-edge (ancestor on the CURRENT path); a DAG is fine -- see below
    seen.add(id(node))

    def _val(v):
        if isinstance(v, (np.ndarray, jax.Array)):
            return _array_digest(v)
        if isinstance(v, (bool, int, float, complex, str, bytes, type(None), np.integer, np.floating)):
            return ("v", type(v).__name__, v)
        if isinstance(v, (tuple, list)):
            parts = tuple(_val(x) for x in v)
            return None if any(p is None for p in parts) else ("seq", parts)
        return None

    t = type(node).__name__
    if isinstance(node, Literal) or isinstance(node, Constant):
        head = _val(node.value)
    elif isinstance(node, _RM):
        head = ("region", node.region)
    elif isinstance(node, _TM):
        head = ("tag", node.tag)
    elif isinstance(node, Variable):
        head = _val((node.tag, tuple(np.atleast_1d(node.dim).tolist()), getattr(node, "axis", "spatial")))
    elif isinstance(node, (TrialFunction, TestFunction)):
        head = _val(
            (
                getattr(node, "name", ""),
                tuple(getattr(node, "value_shape", ()) or ()),
                int(getattr(node, "order", 1)),
                str(getattr(node, "space", "Lagrange")),
            )
        )
    elif isinstance(node, (Jacobian, Hessian)):
        head = ("d", str(getattr(node, "scheme", "")))
    elif isinstance(node, BinaryOp):
        head = ("op", node.op)
    elif isinstance(node, FunctionCall):
        fn_tok = _callable_token(node.fn, seen)
        kw = _val(tuple(sorted((node.kwargs or {}).items()))) if getattr(node, "kwargs", None) else ()
        if fn_tok is None or kw is None:
            fn_tok = None
        head = None if fn_tok is None else ("call", node._name, fn_tok, getattr(node, "reduces_axis", None), kw)
    else:
        head = None
    if head is None:
        _ELEM_MAP_STATS["content_bail"][t] = _ELEM_MAP_STATS["content_bail"].get(t, 0) + 1
        return None
    child_digests = []
    from .solver_helper import iter_children

    for child in iter_children(node) or ():
        d = _expr_digest(child, seen)
        if d is None:
            return None
        child_digests.append(d)
    # PATH-based guard, not visited-based: an expression is a DAG (`ui` appears once per term but is
    # ONE object), so a shared node must digest normally on every path -- only a genuine cycle, where
    # a node is its own ancestor, is refused. Hence the discard on exit.
    seen.discard(id(node))
    return (t, head, tuple(child_digests))


def _leaf_content_token(leaf, seen):
    """Content token of ONE baked closure leaf, or ``None`` when this leaf cannot be keyed by value.

    The probe over a real build's element functions found exactly these kinds: arrays, plain scalars,
    coefficient expression trees, nested functions, empty sets, an empty ``NeuralSlots``, and the
    owning ``domain``. Everything else bails -- tallied, so the next kind to support is measured."""
    import types as _types

    from ...trace import Placeholder as _Ph

    if isinstance(leaf, (np.ndarray, jax.Array)):
        return _array_digest(leaf)
    if isinstance(leaf, (bool, int, float, complex, str, bytes, type(None), np.integer, np.floating, np.bool_)):
        return ("v", type(leaf).__name__, leaf)
    if isinstance(leaf, (set, frozenset)):
        try:
            return ("set", frozenset(leaf))
        except TypeError:
            return None
    if isinstance(leaf, _types.FunctionType):
        sub = _fn_content_key(leaf, None, seen)
        return None if sub is None else ("fn",) + sub
    if isinstance(leaf, _Ph):
        d = _expr_digest(leaf)
        return None if d is None else ("expr", d)
    tname = type(leaf).__name__
    # MRO walk, not a name match on the leaf class: ``PolygonDomain(domain)`` and any other subclass
    # must token the same way -- matching only the base name made every shapely-built domain bail,
    # which the tally surfaced as ``{'PolygonDomain': 12}`` on the first multifield rebuild measured.
    if any(c.__name__ == "domain" and c.__module__.startswith("jno.") for c in type(leaf).__mro__):
        return _domain_content_token(leaf)
    if tname == "NeuralSlots" and not getattr(leaf, "models", None):
        return ("neural", tuple(getattr(leaf, "all_names", ())), tuple(getattr(leaf, "param_names", ())))
    _ELEM_MAP_STATS["content_bail"][tname] = _ELEM_MAP_STATS["content_bail"].get(tname, 0) + 1
    return None


def _domain_content_token(d):
    """Mesh-content identity of a ``jno.domain``: dimension, point and cell digests, and the tag
    names. Memoized on the instance (a domain's mesh is fixed after build; adaptivity REPLACES the
    domain object). Everything an element function reads from the domain at trace time is either
    covered here or reaches the closure as a separately-digested array (masks, facet ids, context
    tensors), so identical tokens imply identical compiled programs."""
    cached = d.__dict__.get("_elem_map_content_token")
    if cached is not None:
        return cached
    try:
        mesh = d.mesh
        parts = [("dim", int(d.dimension)), _array_digest(np.asarray(mesh.points))]
        for name in sorted(mesh.cells_dict):
            parts.append((name, _array_digest(np.asarray(mesh.cells_dict[name]))))
        parts.append(("tags", tuple(sorted(map(str, getattr(d, "avaiable_mesh_tags", ()) or ())))))
        token = ("domain", tuple(parts))
    except Exception:  # noqa: BLE001 -- no mesh (point cloud), exotic state: just do not key it
        return None
    d.__dict__["_elem_map_content_token"] = token
    return token


def _structure_token(obj, seen, renumber):
    """Canonical content token of one baked value -- container structure INCLUDED, walked by hand
    rather than through ``tree_flatten``, for one reason: **integer dict keys are per-build counters**.

    The assembler's closures carry tables keyed by ``field_key`` (= ``op_id``, a process-global
    counter), so two builds of the identical problem capture ``{1: ...}`` and ``{3: ...}`` -- same
    structure, different key -- and a treedef-based key can never match across builds. Those keys are
    pure within-problem lookup indices: the compiled program depends on WHICH entry a trace-time
    lookup resolved to, never on the integer's value. So they are alpha-renumbered by first
    appearance (De Bruijn-style), and an int LEAF equal to a renumbered key is renumbered with it (it
    is the same field key stored as a value, e.g. ``fields[i]['field_key']``). A literal int that
    merely collides with a field key makes the maps diverge and the lookup MISS -- the safe
    direction; a false hit would need two different programs with identical canonical forms, which
    the consistent renumbering excludes.

    String dict keys and everything else compare by value. Unknown container/leaf types fall through
    to :func:`_leaf_content_token`, whose bail disables keying for the whole closure."""
    if isinstance(obj, dict):
        # NUMERIC sort for int keys, in both the renumber assignment and the item order: counters
        # increase monotonically per build, so numeric order preserves the cross-build semantic
        # correspondence (1<->3, 10<->12), where a repr sort would flip it ("12" < "3").
        def _is_int(k):
            return isinstance(k, (int, np.integer)) and not isinstance(k, bool)

        for k in sorted(k for k in obj if _is_int(k)):
            renumber.setdefault(int(k), len(renumber))
        # A field key also travels as a VALUE under its own name (``fields[i]['field_key']``), where
        # no int dict key ever introduces it. The entry name identifies the semantics, so it joins
        # the same renumbering -- consistently with any table keyed by the same counter.
        for k in sorted(obj, key=lambda k: (0, int(k)) if _is_int(k) else (1, repr(k))):
            if k == "field_key" and _is_int(obj[k]):
                renumber.setdefault(int(obj[k]), len(renumber))
        items = []
        for k in sorted(obj, key=lambda k: (0, int(k)) if _is_int(k) else (1, repr(k))):
            kk = ("#", renumber[int(k)]) if _is_int(k) else k
            if k == "field_key" and _is_int(obj[k]):
                # Renumbered HERE, at the semantically-identified site -- never as a bare int leaf,
                # where a literal 1 colliding with field_key=1 would be renumbered in one build and
                # not the other, guaranteeing a miss for the commonest small-int literals.
                v = ("#", renumber[int(obj[k])])
            else:
                v = _structure_token(obj[k], seen, renumber)
            if v is None:
                return None
            items.append((kk, v))
        return ("dict", tuple(items))
    if isinstance(obj, (list, tuple)):
        parts = tuple(_structure_token(x, seen, renumber) for x in obj)
        return None if any(p is None for p in parts) else (type(obj).__name__, parts)
    import types as _types

    if isinstance(obj, _types.FunctionType):
        sub = _fn_content_key(obj, None, seen, renumber)
        return None if sub is None else ("fn",) + sub
    return _leaf_content_token(obj, seen)


def _fn_content_key(fn, chunk, seen=None, renumber=None):
    """The CONTENT twin of :func:`_bake_fingerprint`: everything a jit of ``fn`` would bake, keyed by
    value -- recursing through nested functions (fresh objects per build, identical code and captures
    across rebuilds), containers walked canonically (see :func:`_structure_token`). ``None`` disables
    content keying for this call; the reason is tallied in ``_ELEM_MAP_STATS['content_bail']``."""
    if seen is None:
        seen = set()
    if renumber is None:
        renumber = {}
    if id(fn) in seen:
        _ELEM_MAP_STATS["content_bail"]["<recursive-fn>"] = _ELEM_MAP_STATS["content_bail"].get("<recursive-fn>", 0) + 1
        return None
    seen.add(id(fn))
    try:
        captured = tuple(c.cell_contents for c in (fn.__closure__ or ()))
    except ValueError:
        _ELEM_MAP_STATS["content_bail"]["<unfilled-cell>"] = _ELEM_MAP_STATS["content_bail"].get("<unfilled-cell>", 0) + 1
        return None
    tok = _structure_token((captured, fn.__defaults__ or ()), seen, renumber)
    if tok is None:
        return None
    return (fn.__code__, tok, chunk)


def _bake_fingerprint(fn, chunk):
    """Identity of everything a ``jit`` of ``fn`` would BAKE IN: its code object, and the *leaves* of
    its closure cells and default arguments.

    Two element functions with the same fingerprint produce the same jaxpr, so they can share one
    compilation. The leaves are compared **by identity**, which is sound because JAX arrays are
    immutable -- the same array object cannot have become different values. Flattening as a pytree
    rather than fingerprinting the containers is what makes a mutated-in-place ``args`` dict miss:
    its leaves are new objects, so it gets a new key instead of a stale compilation.

    Returns ``(key, pins)``, or ``None`` when the function must not be cached -- an empty cell, or a
    **tracer** among the captures. Tracers are excluded for two reasons: under an enclosing trace the
    inner ``jit`` is inlined and compiles nothing anyway, so caching buys zero; and pinning a tracer
    in a module-level dict would keep it alive past the trace that owns it.
    """
    try:
        captured = tuple(c.cell_contents for c in (fn.__closure__ or ()))
    except ValueError:  # a cell that is not filled yet (recursive closure)
        return None
    baked = (captured, fn.__defaults__ or ())
    leaves, treedef = jax.tree_util.tree_flatten(baked)
    if any(isinstance(leaf, jax.core.Tracer) for leaf in leaves):
        return None
    return (fn.__code__, treedef, tuple(id(leaf) for leaf in leaves), chunk), leaves


def elem_map(fn, xs, chunk):
    """``vmap(fn)`` over the leading axis, in chunks of ``chunk`` when one is set -- COMPILED.

    ``jax.vmap`` batches, it does not compile. Without an enclosing ``jit`` every batched primitive
    inside ``fn`` executes eagerly and is compiled as its own single-primitive program: assembling
    one 2-D Poisson problem issued **187 such programs, 3.5 s**, of which 187 were under 60 MLIR
    lines. They are keyed on shapes, so a remesh pays the whole bill again -- which is what made an
    AFEM step cost ~4 s. (The chunked branch escaped this only by accident: ``lax.map`` lowers to
    ``scan``, a single primitive.)

    The reason this could not simply be ``jax.jit(jax.vmap(fn))`` is that callers build ``fn`` as a
    fresh lambda per evaluation. ``jax.jit`` keys its cache on the function object, so a fresh lambda
    is a fresh entry: measured, that turned a repeat nonlinear solve from 522 ms into 1233 ms with 9
    recompilations -- trading a 2.5x faster first solve for a 2.4x slower every-solve-after. Hence
    :func:`_bake_fingerprint`: the cache is keyed on what the compilation actually depends on -- the
    code object and the identities of the values baked into it -- so the freshly-built lambda that
    wraps identical captures reuses the identical compiled program.

    The cache is bounded and holds its baked leaves, which both caps memory and keeps the ``id``\\ s
    in the key from being recycled onto different objects while an entry is live.

    Measured cold, one variant per process (running both in one process lets whichever goes second
    inherit the other's warm XLA cache, which is how an earlier comparison flattered itself):

        cold build     4282 -> 2758 ms     remesh (the AFEM step)   4134 -> 2764 ms
        first solve    2745 -> 1739 ms     repeat solve              502 ->  355 ms

    **What this costs.** Rebuilding an identical problem from scratch allocates fresh mesh arrays, so
    the key legitimately misses and that path pays a trace plus a compile where the old eager route
    hit JAX's per-op cache (which keys on shapes): 283 -> 710 ms. Shape-keying instead would be faster
    and WRONG -- it would hand a compilation baked with one mesh's coordinates to a different mesh of
    the same size. The honest fix for that case is to stop baking the geometry in at all and pass it
    as an argument, a restructure of how ``fem_native`` builds these closures. Until then the trade is
    net favourable: a build-then-remesh-twice-then-rebuild sequence goes 13.0 s -> 9.1 s.

    The regression the naive ``jax.jit(jax.vmap(fn))`` caused -- repeat solve 522 -> 1233 ms -- does
    not appear; repeats are faster than before the change, not slower.
    """
    if chunk is None:
        run, arrays = jax.vmap(fn), xs
    else:
        c = int(chunk)
        run, arrays = (lambda *a: jax.lax.map(lambda z: fn(*z), a, batch_size=c)), xs

    fp = _bake_fingerprint(fn, chunk)
    if fp is None:  # traced or unpinnable: jit anyway (inlined under a trace), just do not cache
        return jax.jit(run)(*arrays)

    key, pins = fp

    # Cache entries are ``(jit_fn, pins, baked)``: ``pins`` guard THIS call's ids against recycling,
    # ``baked`` are the leaves of the build that COMPILED the function -- the arrays actually
    # captured in its closure. The two differ exactly for content-hit ALIASES, and the distinction
    # is what makes the donated-buffer check below possible.
    def _baked_dead(entry):
        return entry[2] is not entry[1] and any(
            isinstance(_l, jax.Array) and _l.is_deleted() for _l in jax.tree_util.tree_leaves(entry[2])
        )

    hit = _ELEM_MAP_CACHE.get(key)
    if hit is not None and _baked_dead(hit):
        # An id-keyed ALIAS whose original baked device buffers were DONATED and deleted since (a
        # training loop's optimizer step donates the old parameter buffer). Executing the shared
        # compilation reads a corpse: measured "Array has been deleted with shape=float64[1]" out of
        # a kernel shared across tests. Recompile with this build's live leaves.
        _ELEM_MAP_CACHE.pop(key, None)
        hit = None
    if hit is None:
        # Identity miss. Before compiling, try the CONTENT key: a rebuild of an identical problem
        # bakes fresh objects with identical values, and identical values compile to the identical
        # program. On a hit the id key is inserted as an ALIAS, so every subsequent call from this
        # build fast-paths without touching a digest again. A ``None`` content key (a leaf the
        # tokenizer cannot key by value) falls through to compile -- a miss is safe, a wrong hit
        # would be a wrong operator.
        ckey = _fn_content_key(fn, chunk)
        chit = None
        if ckey is not None:
            chit = _ELEM_MAP_CONTENT.get(ckey)
            if chit is not None and any(
                isinstance(_l, jax.Array) and _l.is_deleted() for _l in jax.tree_util.tree_leaves(chit[2])
            ):
                # Same corpse check for the content table (its entries keep the ORIGINAL build's
                # leaves as ``baked``, which are exactly the compiled closure's buffers).
                del _ELEM_MAP_CONTENT[ckey]
                _ELEM_MAP_STATS["content_bail"]["<deleted-buffer>"] = (
                    _ELEM_MAP_STATS["content_bail"].get("<deleted-buffer>", 0) + 1
                )
                chit = None
        if chit is not None:
            _ELEM_MAP_STATS["content_hits"] += 1
            _ELEM_MAP_CONTENT.move_to_end(ckey)
            # Share the COMPILED program but pin THIS call's leaves: the id key contains this build's
            # object ids, and the entry's pins are what stop those ids from being recycled onto
            # different objects after a GC. ``baked`` stays the ORIGINAL build's leaves -- the
            # closure's actual buffers -- so the liveness checks above test the right arrays.
            hit = (chit[0], pins, chit[2])
        else:
            _ELEM_MAP_STATS["misses"] += 1
            hit = (jax.jit(run), pins, pins)
            if ckey is not None:
                _ELEM_MAP_CONTENT[ckey] = hit
                while len(_ELEM_MAP_CONTENT) > _ELEM_MAP_CONTENT_MAX:
                    _ELEM_MAP_CONTENT.popitem(last=False)
        _ELEM_MAP_CACHE[key] = hit
        if len(_ELEM_MAP_CACHE) > _ELEM_MAP_CACHE_MAX:
            _ELEM_MAP_CACHE.popitem(last=False)
    else:
        _ELEM_MAP_STATS["id_hits"] += 1
        _ELEM_MAP_CACHE.move_to_end(key)
    return hit[0](*arrays)


def sum_duplicate_triplets(A, nse: int | None = None):
    """Collapse duplicate ``(row, col)`` triplets in an assembled BCOO. Exact, and a large win.

    The assemblers append one triplet block **per additive weak-form term** and never pre-sum, and on
    top of that every interior DOF pair receives a contribution from each element sharing it — for P1
    tets that is ~20 elements. BCOO sums the duplicates lazily on every ``@``, so results were always
    correct; they just cost ~19x the work, invisibly, on every one of a Krylov solve's hundreds of
    matvecs.

    Measured on a real 3-D Poisson (P1 tets), stored vs unique triplets, and the redundancy GROWS
    with the mesh::

        h=0.30   21882 -> 1311  (16.7x)   0.33 -> 0.02 MiB   matvec 3.5x faster
        h=0.20   47176 -> 2567  (18.4x)   0.72 -> 0.04 MiB   matvec 5.6x faster
        h=0.16   96473 -> 4999  (19.3x)   1.47 -> 0.08 MiB   matvec 5.7x faster

    Operator unchanged to 2.2e-15. The one-time sort amortises immediately against the matvec count.

    **This changes the sparsity PATTERN, not only the storage.** ``BCOO.sum_duplicates`` defaults to
    ``remove_zeros=True``, and on a Dirichlet-constrained operator that is the larger of the two
    effects: symmetric elimination and mass row/column zeroing leave whole rows of numerically-zero
    triplets behind. On a 3-D transient heat block (341 dofs, h=0.16) the mass goes 18240 stored ->
    3841 unique-with-zeros -> **753** actually nonzero, and the operator 54992 -> 3841 -> 1025. So the
    output is the true nonzero pattern, not a re-indexed copy of the input one.

    That is safe for every current consumer, and was checked rather than assumed: ``matrix_diagonal``
    scatter-adds only on-diagonal triplets, so a *dropped* zero and a *stored* zero give the same
    diagonal; ``jacobi`` guards with ``where(|d| > 1e-30, d, 1.0)`` and documents leaving those rows
    unscaled; the AMG/AMS CSR conversions only see a sparser graph. Solutions were compared before and
    after across steady 3-D, forced 2-D transient with inhomogeneous Dirichlet, 1-D steady and 1-D
    transient: max relative change 1.4e-15. A consumer that needs the *structural* pattern (an entry
    that is zero now but nonzero at another parameter value) must not read it off a compressed
    operator.

    **Concrete operators only.** ``sum_duplicates`` needs a static ``nse`` under ``jit``, and the
    unique count is data-dependent, so a traced/parametric assembly is returned untouched (correct,
    just uncompressed). The sparsity pattern is fixed by mesh and terms, so threading a precomputed
    ``nse`` through the parametric path is a worthwhile follow-up rather than a limitation in
    principle. Note ``sparse_lu_solve`` already calls ``sum_duplicates(nse=A.nse)`` — that sorts but
    deliberately keeps the *unshrunk* count, so it gets none of this compression.
    """
    if not hasattr(A, "indices"):
        return A
    if nse is not None:
        return _sum_duplicates_static(A, nse=int(nse))
    try:
        return A.sum_duplicates()  # nse inferred -> requires concrete indices
    except Exception:  # noqa: BLE001 — traced operator: leave it alone rather than fail the assembly
        return A


@partial(jax.jit, static_argnames=("nse",))
def _sum_duplicates_static(A, *, nse: int):
    """``sum_duplicates`` at a STATIC count, as one compiled program.

    ``BCOO.sum_duplicates`` is a sort plus segment work, and run uncompiled each of those primitives
    becomes its own XLA module. The non-nodal assembler calls this from inside its re-assembly, which
    is not itself jitted, so a Morley build spent **~1150 ms across 41 programs** here -- more than the
    element kernels it was compressing. The 2-D Lagrange path never showed it because that route uses
    :func:`compress_eager` instead.

    ``nse`` is static and the operator arrives as an argument, so ``jax.jit`` keys this on shapes
    natively -- no cache-key machinery, and under the parametric/per-step traces that call it, the jit
    simply inlines.

    ``remove_zeros=False`` is deliberate and load-bearing. With removal on, the surviving count is
    data-dependent, so ``sum_duplicates`` pads the result out to ``nse`` using OUT-OF-BOUND indices
    ``(shape[0], shape[1])``. BCOO's own matvec ignores those, but jNO hands operators to consumers
    that read the triplets directly -- the AMG/AMS CSR conversion would build a row ``n`` that does not
    exist. Off, the output is exactly the unique-pair set with every index in bounds: purely
    structural, data-independent, and identical for every parameter value the assembler is traced at.
    """
    return A.sum_duplicates(nse=nse, remove_zeros=False)


def unique_triplet_count(indices) -> int:
    """Host-side count of distinct ``(row, col)`` pairs -- the exact ``nse`` for the static path above.

    Exactness is a correctness requirement, not an optimisation: ``sum_duplicates`` *drops* entries
    when the requested ``nse`` is too small, which is a silently wrong operator rather than a slower
    one. Computed with ``np.unique`` over the concrete index array the assembler is about to close
    over, so the count and the pattern come from the same source and cannot drift apart.

    Raises on a traced index array rather than guessing. Every caller builds its pattern from
    host-static mesh connectivity, so a tracer here means that invariant broke, and the caller must
    fall back to the uncompressed path rather than ship a wrong count."""
    plan = compress_plan(indices)
    return 0 if plan is None else int(plan[2])


def compress_plan(indices):
    """Host-side compression plan ``(unique_indices, inverse, nse)``, or ``None`` for an empty pattern.

    The pattern is fixed by mesh and terms, so the *same* duplicate-collapse happens on every call.
    Deciding it once host-side turns the in-trace work from a sort into a scatter-add:
    ``segment_sum(data, inverse)`` is ``O(nnz)`` where ``sum_duplicates`` is ``O(nnz log nnz)``, and it
    compiles to no sort at all. That matters because the traced assemblers re-run per Newton step, per
    timestep and per parameter value -- measured on an 18k-DOF 3-D nonlinear problem, paying the sort
    on every call cost more in assembly than the compression saved in matvecs.

    Keys are flattened to a single ``int64`` (``row * (max_col + 1) + col``) rather than sorted
    lexicographically as an ``(nnz, 2)`` array: ``np.unique(..., axis=0)`` builds a structured view and
    is ~2x slower on a 3M-triplet pattern.

    Raises on a traced index array rather than guessing. Every caller builds its pattern from
    host-static mesh connectivity, so a tracer here means that invariant broke, and the caller must
    fall back to the uncompressed path rather than ship a wrong count.

    **Memoized on the pattern's CONTENT** (:data:`_PLAN_CACHE`). The sentence at the top of this
    docstring -- the pattern is fixed by mesh and terms -- is also true *across builds*, and it was
    being ignored: rebuilding the same problem on the same mesh recomputed an identical plan. Measured
    on 3-D Poisson at 27,833 nodes, that was **0.68 s of a 1.5 s build, 45%**, and this path is
    entered two to three times per build (here, the Dirichlet-row wrapper, and again inside
    :func:`compress_eager` on the augmented pattern).

    Content-keyed rather than keyed on the identity of the arrays the pattern was derived from, which
    is the safer of the two: identical content provably yields an identical plan, and a **remeshed
    domain changes the content and simply misses**, so no staleness is possible. Identity-keying would
    have needed the mesh threaded into the key and a test to prove an adaptive remesh invalidates it.
    The hash is not free but it is not close to the cost either -- measured at the 9.5M-triplet size,
    ``blake2b`` 60 ms against the 800 ms ``np.unique`` it skips, i.e. **7.5%**."""
    arr = np.asarray(indices)  # raises on a tracer -- deliberately not caught
    if arr.size == 0 or arr.shape[0] == 0:
        return None
    digest = _hashlib.blake2b(memoryview(np.ascontiguousarray(arr)).cast("B"), digest_size=16).digest()
    key = (digest, arr.shape, arr.dtype.str)
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        _PLAN_CACHE.move_to_end(key)
        return hit
    rows = arr[:, 0].astype(np.int64)
    cols = arr[:, 1].astype(np.int64)
    stride = int(cols.max()) + 1
    uniq, inverse = np.unique(rows * stride + cols, return_inverse=True)
    idx = np.stack([uniq // stride, uniq % stride], axis=1).astype(np.int32)
    # int32 inverse: it is one entry per RAW triplet, so on a large 3-D operator it is the biggest
    # array the plan holds -- halving it against numpy's int64 default is worth the cast.
    plan = jnp.asarray(idx), jnp.asarray(inverse.reshape(-1).astype(np.int32)), int(uniq.shape[0])
    _PLAN_CACHE[key] = plan
    while len(_PLAN_CACHE) > _PLAN_CACHE_MAX:
        _PLAN_CACHE.popitem(last=False)
    return plan


def compress_eager(A):
    """Collapse duplicates in a CONCRETE operator without an on-device sort.

    ``sum_duplicates`` sorts the triplet axis on the device, and on a large operator that workspace
    costs more than the compression saves. Measured on a 3-D transient heat block (18k DOFs, 6.27M raw
    triplets): the operator fell 95.6 -> 5.9 MiB but PEAK memory rose 500 -> 666 MiB. The stored
    operator got 16x smaller and the machine could fit a smaller problem -- the opposite of the point.

    The pattern is host data, so the collapse can be decided in numpy and applied on device as an
    ``O(nnz)`` ``segment_sum``: no device sort, no workspace spike. Explicit zeros are still dropped,
    but the check runs on the ALREADY-COMPRESSED values, which are ~16x smaller than the raw triplets,
    so it is a cheap transfer rather than a full round trip.

    Falls back to the sorting path for anything non-concrete -- correctness never depends on this."""
    if not hasattr(A, "indices"):
        return A
    try:
        plan = compress_plan(A.indices)
    except Exception:  # noqa: BLE001 -- traced operator: no host plan available
        return sum_duplicate_triplets(A)
    if plan is None:
        return A
    idx, inverse, nse = plan
    data = jax.ops.segment_sum(A.data, inverse, num_segments=nse)
    keep = np.asarray(jnp.abs(data) > 0.0)  # concrete and already compressed -> cheap
    if not bool(keep.all()):
        sel = jnp.asarray(np.flatnonzero(keep))
        data, idx = data[sel], idx[sel]
    return jsparse.BCOO((data, idx), shape=A.shape)


def apply_compress_plan(data, plan, shape):
    """Build the compressed BCOO from raw triplet ``data`` and a :func:`compress_plan`.

    Works under trace (no data-dependent shapes) and is exact: every raw triplet is added into its
    unique slot exactly once, which is the same accumulation ``sum_duplicates`` performs and the same
    one BCOO performs lazily on each matvec."""
    idx, inverse, nse = plan
    return jsparse.BCOO((jax.ops.segment_sum(data, inverse, num_segments=nse), idx), shape=shape)


def bcoo_set_dirichlet_rows(A, dofs):
    """``A.at[dofs, :].set(0).at[dofs, dofs].set(1)`` for a BCOO ``A`` — row-replacement (identity row,
    **columns kept**), the matrix-level analogue of a row-replacement residual.

    Retained only for the paths still on row replacement (the second-order-in-time augmented block and
    the transient stepper); everything on the steady residual path uses
    :func:`bcoo_eliminate_dirichlet`, which is symmetric. Keeping the two apart is deliberate: a
    residual and its tangent must agree on which convention they use, and mixing them silently breaks
    Newton rather than erroring."""
    return bcoo_set_unit_diag(bcoo_zero_rows(A, dofs), dofs)


def bcoo_eliminate_dirichlet(A, dofs):
    """Symmetric Dirichlet elimination of a BCOO tangent: zero the constrained rows **and columns**,
    then set a unit diagonal.

    The matrix-level companion of :func:`~jno.utils.solver.fem_1d._apply_dirichlet_projected` — that
    residual evaluates the free form at the projected point ``P(u)``, whose derivative is
    ``M·J·M + (I−M)``, and this is exactly that. It replaced a row-only version, which left the
    constrained columns populated and so made the tangent non-symmetric even for a symmetric operator:
    jNO tests symmetry *bitwise* (see ``linear._matrix_structure``), so every Dirichlet nonlinear
    problem was factored as a general LU instead of LDLᵀ.

    Both maskers keep the index array and only zero ``A.data``, so the sparsity pattern is unchanged —
    which is why the static ``compress_plan`` machinery around this needs no rework."""
    return bcoo_set_unit_diag(bcoo_zero_rows_cols(A, dofs), dofs)


# ---------------------------------------------------------------------------
# Operator / state reduction and prolongation
# ---------------------------------------------------------------------------


_log = get_logger()


def _is_selection(P):
    """True iff BCOO ``P`` has exactly one nonzero per full row — a periodic *selection* (each secondary
    DOF equals a single main), for which the remap-sum reduction is exact. A **nonconforming** tie
    builds an *interpolation* ``P`` (several weighted mains per secondary row), which needs a genuine
    ``P^T M P`` (the dense fallback). ``P.indices`` is static (built from connectivity), so this is a
    trace-time constant even when ``reduce_matrix`` runs inside a jitted ``operator_fn``."""
    if not hasattr(P, "indices"):
        return False
    rows = np.asarray(P.indices[:, 0])
    n = int(P.shape[0])
    return rows.shape[0] == n and int(np.bincount(rows, minlength=n).max()) == 1


def _selection_maps(P, dtype):
    """For a periodic selection ``P`` (BCOO, one nonzero per full row), return ``(main, pval)``:
    ``main[i]`` = the reduced DOF that full DOF ``i`` maps to; ``pval[i]`` = the tie coefficient
    (``1``, or ``-1`` for an antiperiodic tie)."""
    main = jnp.zeros(P.shape[0], P.indices.dtype).at[P.indices[:, 0]].set(P.indices[:, 1])
    pval = jnp.zeros(P.shape[0], dtype).at[P.indices[:, 0]].set(jnp.asarray(P.data, dtype))
    return main, pval


def _remap_bcoo(mat, m_row, p_row, m_col, p_col, shape):
    """Sparse Galerkin remap: send each BCOO triplet ``(r, c, v)`` of ``mat`` to
    ``(m_row[r], m_col[c], v·p_row[r]·p_col[c])`` and let BCOO sum duplicates. This is exactly
    ``P_row^T mat P_col`` for selection matrices, in ``O(nnz(mat))`` and **without** ever forming a
    dense ``n_full × n_full`` intermediate (``nnz`` is static = ``nnz(mat)`` → ``jit``-safe)."""
    r, c = mat.indices[:, 0], mat.indices[:, 1]
    ridx = jnp.stack([m_row[r], m_col[c]], axis=1)
    rdata = mat.data * p_row[r] * p_col[c]
    return jsparse.BCOO((rdata, ridx), shape=shape)


def _remap_bcoo_weighted(mat, P, conj=False):
    """Sparse Galerkin reduction ``P^T mat P`` (or ``P^H mat P`` when ``conj``) for an *interpolation*
    (nonconforming) periodic ``P``
    -- a few mains per secondary, weighted. Generalises :func:`_remap_bcoo` (one main, weight 1) by
    spreading each ``mat`` triplet ``(r, c, v)`` across the ``D x D`` main pairs of its row and
    column with the interpolation weights: ``v -> v · w_r[a] · w_c[b]`` at ``(main_r[a], main_c[b])``.
    Stays sparse (no dense ``n_full × n_full`` intermediate); ``D = max mains/secondary`` (small: the
    nodes of a main facet). Returns ``None`` if ``P``'s indices are not concrete (built under trace),
    so the caller falls back to the dense product -- the nonconforming reduction is built eagerly."""
    try:
        pidx = np.asarray(P.indices)  # concrete only; a tracer raises
        pdat = np.asarray(P.data)
    except Exception:
        return None
    n_full, n_red = int(mat.shape[0]), int(P.shape[1])
    rows, cols = pidx[:, 0], pidx[:, 1]
    order = np.argsort(rows, kind="stable")
    rows, cols, wts = rows[order], cols[order], pdat[order]
    if len(rows) == 0:
        return jsparse.BCOO((jnp.zeros(0), jnp.zeros((0, 2), jnp.int32)), shape=(n_red, n_red))
    is_new = np.concatenate([[True], rows[1:] != rows[:-1]])
    slot = np.arange(len(rows)) - np.maximum.accumulate(np.where(is_new, np.arange(len(rows)), 0))
    D = int(slot.max()) + 1
    main = np.zeros((n_full, D), np.int64)
    main[rows, slot] = cols
    weight = np.zeros((n_full, D), np.complex128 if np.iscomplexobj(pdat) else np.float64)
    weight[rows, slot] = wts
    main, weight = jnp.asarray(main), jnp.asarray(weight)
    r, c, v = mat.indices[:, 0], mat.indices[:, 1], mat.data
    mr, wr, mc, wc = main[r], weight[r], main[c], weight[c]  # (nnz, D)
    if conj:  # P^H mat P conjugates the row (left) weights
        wr = jnp.conj(wr)
    nnz = r.shape[0]
    a = jnp.broadcast_to(mr[:, :, None], (nnz, D, D)).reshape(-1)
    b = jnp.broadcast_to(mc[:, None, :], (nnz, D, D)).reshape(-1)
    data = (v[:, None, None] * wr[:, :, None] * wc[:, None, :]).reshape(-1)
    idx = jnp.stack([a, b], axis=1).astype(mat.indices.dtype)
    return jsparse.BCOO((data, idx), shape=(n_red, n_red)).sum_duplicates()


def reduce_matrix(P, mat, is_selection=None, conj=False):
    """Galerkin reduction ``P^T mat P`` (or the Hermitian ``P^H mat P`` when ``conj`` — for a complex
    Bloch/quasi-periodic ``P``, where the left factor must be conjugated).

    When ``mat`` is BCOO and ``P`` is a BCOO *selection* (conforming/structured tie, one main per
    secondary — e.g. the doubly-periodic PEB) the reduction remaps ``mat``'s triplets to their main
    indices: it stays sparse and never materialises the dense ``n_full × n_full`` matrix (``O(nnz)``;
    the reduction is otherwise the dominant memory peak of a periodic solve). An *interpolation* ``P``
    (nonconforming tie) or a dense ``P``/``mat`` (1D path) falls back to the exact dense ``P^T mat P``.

    ``is_selection`` (whether ``P`` is a one-main-per-secondary selection) is passed precomputed by the
    periodic builders so the reduction never inspects ``P.indices`` at run time — that matters because
    a parametric ``operator_fn`` reduces inside a jitted ``scan`` where ``P.indices`` is a tracer. When
    ``None`` (a direct/eager call) it is computed once here."""
    if is_selection is None:
        is_selection = _is_selection(P)

    def _dt(x):
        # Read the dtype OFF the object. ``np.asarray(x).dtype`` would materialise it, which throws
        # TracerArrayConversionError when ``P`` is traced -- and ``P`` IS traced whenever the basis is
        # differentiated through (``jax.grad`` w.r.t. a Galerkin/POD basis, the learned-basis case).
        # numpy, jnp, BCOO and tracers all carry ``.dtype`` directly, so no sniffing is needed. NB a
        # ``hasattr(x, "data")`` test does NOT identify a BCOO: a numpy array has ``.data`` too (a
        # memoryview, which has no ``.dtype``), so that route crashed on a plain numpy ``P``.
        return x.dtype if hasattr(x, "dtype") else np.asarray(x).dtype

    pdtype = np.result_type(_dt(P), _dt(mat))
    if hasattr(mat, "indices") and is_selection:
        main, pval = _selection_maps(P, pdtype)
        prow = jnp.conj(pval) if conj else pval  # P^H remap conjugates the row (left) factor
        return _remap_bcoo(mat, main, prow, main, pval, (int(P.shape[1]), int(P.shape[1])))
    if hasattr(mat, "indices") and hasattr(P, "indices"):
        # nonconforming (interpolation) tie: weighted triplet-remap, stays sparse (falls back to the
        # dense product below only if P was built under trace, which the nonconforming path never is)
        remapped = _remap_bcoo_weighted(mat, P, conj=conj)
        if remapped is not None:
            return remapped
    mat = mat.todense() if hasattr(mat, "todense") else mat
    Pd = jnp.asarray(P.todense() if hasattr(P, "todense") else P, pdtype)
    left = jnp.conj(Pd).T if conj else Pd.T
    return left @ jnp.asarray(mat, pdtype) @ Pd


def reduce_vector(P, vec, conj=False):
    """Reduce a full-space load/bias vector via ``P^T vec`` (or ``P^H vec`` when ``conj``)."""
    Pd = P if hasattr(P, "todense") else jnp.asarray(P)
    left = (jnp.conj(Pd).T if conj else Pd.T) if not hasattr(Pd, "indices") else None
    if left is not None:
        return left @ jnp.asarray(vec, Pd.dtype).reshape(-1)
    Pc = jsparse.BCOO((jnp.conj(Pd.data), Pd.indices), shape=Pd.shape) if conj else Pd
    return Pc.T @ jnp.asarray(vec, Pc.dtype).reshape(-1)


def restrict_state(P, state_full, kept_nodes, vec: int = 1):
    """Restrict a full initial state to the reduced main DOFs.

    For a consistent periodic initial condition (matching values on opposite
    faces) gathering the kept-node entries is exact and avoids the doubling
    that ``P^T`` would introduce on main nodes.

    ``kept_nodes=None`` means ``P`` is a **Galerkin basis** (``fem.solve(basis=...)``) rather than a
    main/secondary selection: its columns are not a subset of the full DOFs, so there is nothing to
    gather and the restriction is the projection ``P^T state``. For an orthonormal ``P`` that is the
    best approximation of the state within the span, which is exactly what the reduced solve wants.
    """
    state_full = jnp.asarray(state_full).reshape(-1)
    if kept_nodes is None:
        Pd = P if hasattr(P, "todense") else jnp.asarray(P)
        return Pd.T @ jnp.asarray(state_full, Pd.dtype)
    kept = np.asarray(kept_nodes, dtype=int)
    if vec == 1:
        return state_full[jnp.asarray(kept)]
    dof = (kept[:, None] * vec + np.arange(vec)[None, :]).reshape(-1)
    return state_full[jnp.asarray(dof)]


def prolong(P, reduced):
    """Map reduced DOFs back to the full space: ``u_full = P @ u_red``.

    Accepts a single ``(n_red,)`` vector or a batched ``(..., n_red)`` array
    (e.g. a Diffrax ``solution.ys`` trajectory of shape ``(T, n_red)``).
    """
    P = P if hasattr(P, "todense") else jnp.asarray(P)  # keep a BCOO P sparse
    reduced = jnp.asarray(reduced, P.dtype)
    if reduced.ndim == 1:
        return P @ reduced
    # (..., n_red) @ (n_red, n_full) -> (..., n_full)
    return reduced @ P.T


# ---------------------------------------------------------------------------
# Periodic reduction over a (possibly multifield) block.  A `periodic` dict is
# either single-field (legacy `{"P", "kept_nodes", "vec"}`) or multifield
# (`{"blocks": [{"P", "kept", "vec"}, ...], "off_full": [...], "off_red": [...]}`).
# The Galerkin reduction of a blocked operator decomposes per field-pair,
# `reduced[i,j] = P_i^T M[i,j] P_j`, so we NEVER materialise a block-diagonal P
# (which would be O(F^2) dense and force one global vec/order); each field keeps
# its own P_i / kept / vec and only one P_i is held at a time.
# ---------------------------------------------------------------------------


def _periodic_blocks(periodic):
    """Normalise a periodic dict to ``(blocks, off_full, off_red)``; a legacy single-field
    dict is wrapped as a single block (so the multifield path is the general case)."""
    if "blocks" in periodic:
        return periodic["blocks"], np.asarray(periodic["off_full"]), np.asarray(periodic["off_red"])
    P = periodic["P"]
    nf, nr = int(P.shape[0]), int(P.shape[1])
    block = [
        {
            "P": P,
            "kept": periodic["kept_nodes"],
            "vec": periodic.get("vec", 1),
            "is_selection": periodic.get("is_selection"),
        }
    ]
    return block, np.array([0, nf]), np.array([0, nr])


def build_periodic_prolongation_n1e(n_edges, edge_midpoints, edge_dirs, etags, pairs, phases=None, tol=None):
    """DOF-level periodic (Floquet/Bloch) prolongation for a lowest-order Nédélec (N1E) edge field.

    Each edge carries one tangential-moment DOF. A secondary-face edge ties to the main-face edge at the same
    transverse position; the tie weight is the Bloch phase times an orientation **sign** (+1 if the two
    edges point the same way along their lo→hi canonical direction, −1 if opposed — the tangential moment
    flips with the edge orientation). Corner edges shared by two periodic faces are tied twice; the chain is
    resolved to a single retained main DOF. Returns a legacy single-field prolongation dict."""
    mid = np.asarray(edge_midpoints, dtype=np.float64)
    dirs = np.asarray(edge_dirs, dtype=np.float64)
    if phases is None:
        phases = [1.0] * len(pairs)
    is_bloch = any(abs(complex(p) - 1.0) > 1e-12 for p in phases)
    if tol is None:
        span = float(np.linalg.norm(mid.max(axis=0) - mid.min(axis=0)))
        tol = max(span, 1.0) * 1.0e-6

    s2m: Dict[int, int] = {}  # secondary edge -> main edge
    weight: Dict[int, complex] = {}  # secondary edge -> tie weight (orientation sign × Bloch phase)
    for (mtag, stag), ph in zip(pairs, phases):
        me, se = np.asarray(etags[mtag], dtype=int), np.asarray(etags[stag], dtype=int)
        if me.size == 0 or se.size == 0:
            raise ValueError(f"periodic N1E: boundary tag {mtag!r}/{stag!r} has no edges to tie.")
        mm, sm = mid[me], mid[se]
        axis = int(np.argmax(np.abs(mm.mean(axis=0) - sm.mean(axis=0))))  # periodic axis = largest mean gap
        tr = [d for d in range(mid.shape[1]) if d != axis]
        mt, st = mm[:, tr], sm[:, tr]
        d2 = np.sum((st[:, None, :] - mt[None, :, :]) ** 2, axis=-1)
        nn = np.argmin(d2, axis=1)
        dist = np.sqrt(d2[np.arange(len(se)), nn])
        for k, s in enumerate(se):
            if dist[k] > tol:
                raise ValueError(
                    f"periodic N1E: secondary edge {int(s)} has no transverse main match "
                    f"(nearest {dist[k]:.2e} > tol {tol:.2e}); a conforming periodic mesh is required."
                )
            m = int(me[nn[k]])
            sign = 1.0 if float(dirs[int(s)] @ dirs[m]) >= 0.0 else -1.0
            s2m[int(s)], weight[int(s)] = m, sign * complex(ph)

    secondarys = set(s2m)
    kept = np.array([e for e in range(n_edges) if e not in secondarys], dtype=int)
    col = {int(e): j for j, e in enumerate(kept)}
    P = np.zeros((n_edges, len(kept)), dtype=(np.complex128 if is_bloch else np.float64))
    for j, e in enumerate(kept):
        P[e, j] = 1.0
    for s in secondarys:  # resolve a possibly-chained tie (corner edge tied via two faces) to a kept main
        m, w = s2m[s], weight[s]
        while m in s2m:
            w *= weight[m]
            m = s2m[m]
        P[s, col[m]] = w if is_bloch else w.real  # non-Bloch weight is a real ±1 sign
    return {
        "P": jnp.asarray(P),
        "kept_nodes": kept,
        "vec": 1,
        "is_selection": False,  # ±1 signs (and complex phases) → not a pure 0/1 selection
        "is_bloch": is_bloch,
        "n_full": int(n_edges),
        "n_red": int(len(kept)),
    }


def reduce_matrix_periodic(periodic, mat, conj=False):
    """``P^T mat P`` (or Hermitian ``P^H mat P`` when ``conj``) for a single- or multi-field reduction.

    When ``mat`` and every field's ``P_i`` are BCOO, the whole blocked reduction
    ``reduced[i,j] = P_i^T mat[i,j] P_j`` is one global triplet-remap (the per-field offsets are
    folded into a single full→reduced main map), so it stays sparse and never densifies — the fix
    for the ``O(n^2)`` reduction peak at large ``N``. A dense ``mat`` falls back to the per-block
    dense reduction (no block-diagonal ``P`` is ever materialised)."""
    blocks, off_f, off_r = _periodic_blocks(periodic)

    def _sel(b):  # prefer the precomputed flag (built eagerly); compute only when called eagerly
        return b["is_selection"] if b.get("is_selection") is not None else _is_selection(b["P"])

    if len(blocks) == 1:
        return reduce_matrix(blocks[0]["P"], mat, is_selection=_sel(blocks[0]), conj=conj)  # single-field
    if hasattr(mat, "indices") and all(_sel(b) for b in blocks):
        n_full, n_red = int(off_f[-1]), int(off_r[-1])
        pdtype = np.result_type(*[b["P"].data.dtype for b in blocks], mat.data.dtype)
        gmain = jnp.zeros(n_full, mat.indices.dtype)
        gpval = jnp.zeros(n_full, pdtype)
        for i, b in enumerate(blocks):
            Pi = b["P"]
            full_local = Pi.indices[:, 0]  # field-i local full DOF
            gmain = gmain.at[int(off_f[i]) + full_local].set((int(off_r[i]) + Pi.indices[:, 1]).astype(mat.indices.dtype))
            gpval = gpval.at[int(off_f[i]) + full_local].set(jnp.asarray(Pi.data, pdtype))
        prow = jnp.conj(gpval) if conj else gpval
        return _remap_bcoo(mat, gmain, prow, gmain, gpval, (n_red, n_red))
    mat = jnp.asarray(mat.todense()) if hasattr(mat, "todense") else jnp.asarray(mat)
    pdtype = np.result_type(
        *[np.asarray(b["P"].todense() if hasattr(b["P"], "todense") else b["P"]).dtype for b in blocks], mat.dtype
    )
    out = jnp.zeros((int(off_r[-1]), int(off_r[-1])), pdtype)
    _sp = lambda P: P if hasattr(P, "todense") else jnp.asarray(P, pdtype)  # keep BCOO sparse  # noqa: E731
    for i, bi in enumerate(blocks):
        Pi = _sp(bi["P"])
        left = (jnp.conj(Pi).T if conj else Pi.T) if not hasattr(Pi, "indices") else Pi.T
        for j, bj in enumerate(blocks):
            Pj = _sp(bj["P"])
            blk = jnp.asarray(mat[off_f[i] : off_f[i + 1], off_f[j] : off_f[j + 1]], pdtype)
            out = out.at[off_r[i] : off_r[i + 1], off_r[j] : off_r[j + 1]].set(left @ blk @ Pj)
    return out


def reduce_vector_periodic(periodic, vec, conj=False):
    """``P^T vec`` (or ``P^H vec`` when ``conj``) for a single- or multi-field reduction (per-field block)."""
    blocks, off_f, _ = _periodic_blocks(periodic)
    if len(blocks) == 1:
        return reduce_vector(blocks[0]["P"], vec, conj=conj)
    vec = jnp.asarray(vec).reshape(-1)
    _sp = lambda P: P if hasattr(P, "todense") else jnp.asarray(P, vec.dtype)  # keep BCOO sparse  # noqa: E731
    out = []
    for i, b in enumerate(blocks):
        Pi = _sp(b["P"])
        Pc = jsparse.BCOO((jnp.conj(Pi.data), Pi.indices), shape=Pi.shape) if (conj and hasattr(Pi, "indices")) else Pi
        out.append(Pc.T @ vec[off_f[i] : off_f[i + 1]])
    return jnp.concatenate(out)


def restrict_state_periodic(periodic, state):
    """Restrict a full state to the reduced main DOFs (per-field block)."""
    blocks, off_f, _ = _periodic_blocks(periodic)
    if len(blocks) == 1:
        return restrict_state(blocks[0]["P"], state, blocks[0]["kept"], blocks[0]["vec"])
    state = jnp.asarray(state).reshape(-1)
    return jnp.concatenate(
        [restrict_state(b["P"], state[off_f[i] : off_f[i + 1]], b["kept"], b["vec"]) for i, b in enumerate(blocks)]
    )


def prolong_periodic(periodic, reduced):
    """Prolong reduced DOFs to the full space (per-field block); supports a batched leading axis."""
    blocks, _, off_r = _periodic_blocks(periodic)
    if len(blocks) == 1:
        return prolong(blocks[0]["P"], reduced)
    reduced = jnp.asarray(reduced)
    parts = [prolong(b["P"], reduced[..., off_r[i] : off_r[i + 1]]) for i, b in enumerate(blocks)]
    return jnp.concatenate(parts, axis=-1)
