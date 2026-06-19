from __future__ import annotations

"""
Internal FEAX backend utilities.

This module contains low-level helpers shared by the steady FEM route and the
transient FEAX-time route. It is intentionally not a public API.

Responsibilities:
- convert jNO weak-form symbols into FEAX-compatible kernels,
- build FEAX meshes/problems/Dirichlet BC configs,
- evaluate symbolic expressions inside FEAX volume/surface kernels,
- prepare residual/Jacobian runtime objects for time-dependent assembly.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ...trace import (
    BinaryOp,
    Constant,
    FunctionCall,
    Hessian,
    Jacobian,
    Literal,
    ModelCall,
    OperationCall,
    OperationDef,
    StateField,
    TensorTag,
    TestFunction,
    Tracker,
    TrialFunction,
    Variable,
)
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
    Normalize user Dirichlet data into FEAX-compatible value functions.

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
            "FEAX backend currently supports exactly one TrialFunction "
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
    truth threaded into the feax ``Problem`` lists and the multi-field kernel.
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


def _collect_runtime_parameter_tags_for_feax(node, out=None):
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
        _collect_runtime_parameter_tags_for_feax(child, out)
    return out


def _collect_temporal_tags_for_feax(node, out=None):
    """
    Collect temporal Variable tags used inside FEAX kernels.

    These tags determine which time values must be passed through FEAX
    InternalVars during transient assembly.
    """
    if out is None:
        out = set()

    if isinstance(node, Variable) and getattr(node, "axis", None) == "temporal":
        out.add(str(node.tag))
        return out

    if isinstance(node, BinaryOp):
        _collect_temporal_tags_for_feax(node.left, out)
        _collect_temporal_tags_for_feax(node.right, out)
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
                _collect_temporal_tags_for_feax(a, out)
        return out

    if isinstance(node, Jacobian):
        _collect_temporal_tags_for_feax(node.target, out)
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
                _collect_temporal_tags_for_feax(v, out)
        return out

    if isinstance(node, Hessian):
        _collect_temporal_tags_for_feax(node.target, out)
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
                _collect_temporal_tags_for_feax(v, out)
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
    # Node-based field coefficient: feax has gathered the cell's local nodal values
    # (size = nodes-per-element); interpolate to quadrature points with the field's
    # shape functions, mirroring the solution interpolation (cf. feax interpolate_var:
    # shape_vals . nodal). Returns (n_quad, 1) like a field value.
    shape_vals = local.get("shape_vals")  # (n_quad, n_local)
    cell_nodal = flat.reshape(flat.shape[0], 1)  # (n_local, 1)
    return jnp.sum(shape_vals[:, :, None] * cell_nodal[None, :, :], axis=1)


def _temporal_value_from_internal_vars(local, tag, dim_start=0, dim_end=1):
    """
    Read a temporal variable value from FEAX InternalVars.

    Returns None when the requested temporal tag is not part of the current
    FEAX kernel call.
    """
    temporal_tags = local.get("temporal_tags", ())
    volume_vars = local.get("volume_vars", ())

    if tag not in temporal_tags:
        return None

    idx = temporal_tags.index(tag)
    if idx >= len(volume_vars):
        raise IndexError(
            f"Temporal FEAX variable tag '{tag}' mapped to slot {idx}, "
            f"but only {len(volume_vars)} volume_vars were provided."
        )

    arr = jnp.asarray(volume_vars[idx])
    # scalar / (1,) / (1,1) -> one scalar time for this assembly call
    t_scalar = jnp.reshape(arr, (-1,))[0]
    out = jnp.asarray([t_scalar])
    return out[dim_start:dim_end]


# --------------------------------
# FEAX expression evaluation helpers
# --------------------------------


def _prefix_align(a, b):
    """Broadcast-align two kernel quantities for an elementwise op.

    In the FEAX kernel, trial-derived quantities are laid out as ``(n_quad,
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


def _eval_expr_for_feax(domain, node, local):
    """
    Evaluate a jNO symbolic expression inside a FEAX local kernel.

    The `local` dictionary contains quadrature coordinates, shape values,
    shape gradients, local cell DOFs, domain context, and optional temporal
    InternalVars. This evaluator supports literals, constants, variables,
    tensor tags, TrialFunction/TestFunction values, their Jacobians, binary
    operations, and FunctionCall nodes.
    """
    if not isinstance(
        node,
        (
            Literal,
            Constant,
            TensorTag,
            Variable,
            TestFunction,
            TrialFunction,
            Jacobian,
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

    if isinstance(node, TensorTag):
        if node.tag not in local["domain_context"]:
            raise KeyError(f"TensorTag '{node.tag}' not found in FEM domain context.")
        tensor = jnp.asarray(local["domain_context"][node.tag])
        if tensor.ndim >= 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.ndim >= 1 and tensor.shape[0] > 1:
            raise NotImplementedError(
                "FEAX backend currently supports singleton-batch TensorTag coefficients only. "
                f"Got shape {tensor.shape} for tag '{node.tag}'."
            )
        if node.dim_index is not None and tensor.ndim >= 1:
            tensor = tensor[..., node.dim_index]
        return tensor

    if isinstance(node, Variable):
        dim_start, dim_end = node.dim

        # FEAX local quadrature coordinates
        if local.get("surface", False):
            if isinstance(node.tag, str) and node.tag.startswith("gauss_"):
                return local["physical_quad_points"][..., dim_start:dim_end]
        else:
            if node.tag == "fem_gauss":
                return local["physical_quad_points"][..., dim_start:dim_end]

        # Temporal variable in FEAX assembly:
        # prefer FEAX InternalVars volume_vars (pure JAX / no domain mutation),
        # then fall back to domain.context for older steady / legacy paths.
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
                raise KeyError(f"Temporal Variable tag '{node.tag}' not found in FEAX local/domain context.")
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

        raise KeyError(f"Variable tag '{node.tag}' not found in FEAX local/domain context.")

    if isinstance(node, TestFunction):
        n_comp = _value_shape_num_components(getattr(node, "value_shape", ()))
        shape_vals, _, _ = _field_data(local, node)
        return _expand_test_shape_vals(shape_vals, n_comp)

    if isinstance(node, TrialFunction):
        vals, _, cell_sol = _field_data(local, node)
        flat_interp = jnp.sum(vals[:, :, None] * cell_sol[None, :, :], axis=1)
        value_shape = getattr(node, "value_shape", ())
        if len(value_shape) == 0:
            return flat_interp
        return _reshape_components_last(flat_interp, value_shape)

    if isinstance(node, Jacobian):
        dims = []
        for var in node.variables:
            if not isinstance(var, Variable):
                raise NotImplementedError(
                    "FEAX backend expects Jacobian variables to be domain.variable(...) placeholders."
                )
            dims.append(var.dim[0])
        if len(dims) == 0:
            raise ValueError("Jacobian node has no differentiation variables")

        if isinstance(node.target, TestFunction):
            n_comp = _value_shape_num_components(getattr(node.target, "value_shape", ()))
            _, grads, _ = _field_data(local, node.target)
            if n_comp == 1:
                comps = [grads[..., dim0] for dim0 in dims]
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)
            eye = jnp.eye(n_comp, dtype=grads.dtype)
            comps = [grads[..., dim0][:, :, None, None] * eye[None, None, :, :] for dim0 in dims]
            if len(comps) == 1:
                return comps[0]
            return jnp.stack(comps, axis=-1)

        if isinstance(node.target, TrialFunction):
            _, grads, cell_sol = _field_data(local, node.target)
            grad_list = [jnp.sum(grads[:, :, dim0 : dim0 + 1] * cell_sol[None, :, :], axis=1) for dim0 in dims]
            flat = grad_list[0] if len(dims) == 1 else jnp.stack(grad_list, axis=-1)
            value_shape = getattr(node.target, "value_shape", ())
            if len(value_shape) == 0:
                return flat
            if len(dims) == 1:
                return _reshape_components_last(flat, value_shape)
            return jnp.reshape(flat, flat.shape[:1] + tuple(value_shape) + (len(dims),))

        raise NotImplementedError("FEAX backend supports gradients of TrialFunction/TestFunction only.")

    if isinstance(node, BinaryOp):
        a = _eval_expr_for_feax(domain, node.left, local)
        b = _eval_expr_for_feax(domain, node.right, local)
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
        from .parametric_helpers import _is_runtime_scalar_parameter, _parameter_name

        if _is_runtime_scalar_parameter(node):
            name = _parameter_name(node)
            val = _runtime_parameter_value_from_internal_vars(local, name)
            if val is None:
                raise KeyError(
                    f"Runtime parameter '{name}' not supplied to the FEAX kernel. "
                    "Ensure it was registered in runtime_parameter_tags and packed "
                    "into InternalVars.volume_vars."
                )
            return val
        # Non-parameter ModelCall (e.g. a neural coefficient) -> not handled here.
        raise NotImplementedError("FEAX kernel cannot evaluate non-parameter ModelCall coefficients yet.")

    if isinstance(node, FunctionCall):
        args = [_eval_expr_for_feax(domain, arg, local) for arg in node.args]
        kwargs = node.kwargs if node.kwargs else {}
        return node.fn(*args, **kwargs)

    raise NotImplementedError(f"Unsupported weak-form node for FEAX backend: {type(node).__name__}")


# --------------------------------
# FEAX kernel builders
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
    problem_ref,
    *cell_internal_vars,
):
    """
    Evaluate and integrate one volume weak-form expression on one FEAX cell.

    Returns the flattened cell residual contribution expected by FEAX.
    """
    num_nodes = cell_shape_grads.shape[1]
    vec = _value_shape_num_components(value_shape)
    cell_sol = cell_sol_flat.reshape(num_nodes, vec)

    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("FEAX problem_ref['problem'] was not initialized before kernel evaluation.")

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
        "volume_vars": tuple(cell_internal_vars),
    }

    val = _eval_expr_for_feax(domain, expr, local)
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
    Evaluate and integrate one boundary weak-form expression on one FEAX face.

    Returns the flattened surface residual contribution expected by FEAX.
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

    val = _eval_expr_for_feax(domain, expr, local)
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]


def _make_universal_volume_kernel(domain, expr, value_shape, temporal_tags, runtime_parameter_tags, problem_ref):
    """
    Create the FEAX universal volume kernel for a lowered weak-form expression.
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
    ``_eval_expr_for_feax``), and accumulates into its test field's residual slot.
    ``ravel_pytree`` concatenates the per-field residuals in field order (matching
    ``unflatten_fn_dof``) so feax autodiffs the full block matrix."""
    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("FEAX problem_ref['problem'] was not initialized before kernel evaluation.")

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
        "volume_vars": tuple(cell_internal_vars),
    }

    residuals = [jnp.zeros((nnodes[i] * int(fields[i]["vec"]),), dtype=cell_sol_flat.dtype) for i in range(nfields)]
    for coeff, test_idx in term_list:
        val = _eval_expr_for_feax(domain, coeff, local)
        weights = cell_JxW[test_idx]
        wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
        contrib = ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]
        residuals[test_idx] = residuals[test_idx] + contrib
    return ravel_pytree(residuals)[0]


def _make_multifield_volume_kernel(
    domain, term_list, fields, field_index, temporal_tags, runtime_parameter_tags, problem_ref
):
    """FEAX universal volume kernel for a coupled (multi-field) weak form."""

    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
        return _eval_multifield_volume_integrand(
            domain,
            term_list,
            fields,
            field_index,
            temporal_tags,
            runtime_parameter_tags,
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

    The surface analogue of :func:`_eval_multifield_volume_integrand`. feax passes the
    full multi-field parent-cell DOFs and the per-field face shape data concatenated
    along the node axis (assembler concatenates in ``problem.fes`` order, identical to
    the volume ``shape_grads``), so we split DOFs with ``unflatten_fn_dof`` and slice the
    face shape arrays per field. Each term ``(coeff, test_field_index)`` accumulates into
    its test field's residual slot; ``ravel_pytree`` concatenates in field order so feax
    autodiffs the surface contribution into the right block(s) of the load and matrix."""
    problem = problem_ref["problem"]
    if problem is None:
        raise RuntimeError("FEAX problem_ref['problem'] was not initialized before kernel evaluation.")

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
        val = _eval_expr_for_feax(domain, coeff, local)
        wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
        contrib = ravel_pytree(jnp.sum(val * weights.reshape(wshape), axis=0))[0]
        residuals[test_idx] = residuals[test_idx] + contrib
    return ravel_pytree(residuals)[0]


def _make_multifield_surface_kernel(
    domain, term_list, fields, field_index, tag, temporal_tags, runtime_parameter_tags, problem_ref
):
    """FEAX universal surface kernel for the coupled boundary terms on one tag."""

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
# FEAX problem assembly
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
# in meshio ordering). feax has no P2 mesh source and jno's domain machinery assumes
# linear cells, so the domain mesh stays linear (P1) and we promote *only* the feax
# assembly mesh here: insert edge-midpoint nodes, vertices preserved (so a P1 field on
# a P2 problem -- e.g. Taylor-Hood pressure -- is just the vertex block).
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


def _build_feax_mesh(domain, element_type: str):
    import feax as fe

    meshio_type = _meshio_type_for_element(element_type)
    dim = domain.dimension
    cells_dict = domain.mesh.cells_dict
    if meshio_type in cells_dict:
        points = np.asarray(domain.mesh.points)[:, :dim]
        cells = np.asarray(cells_dict[meshio_type])
    elif meshio_type in _P2_FROM_P1:  # promote the (linear) domain mesh for assembly only
        p1_type, edge_local = _P2_FROM_P1[meshio_type]
        if p1_type not in cells_dict:
            raise ValueError(f"Cannot build '{element_type}': no '{p1_type}' cells to promote to '{meshio_type}'.")
        points, cells = _promote_to_quadratic(np.asarray(domain.mesh.points)[:, :dim], cells_dict[p1_type], edge_local)
    else:
        raise KeyError(f"No mesh cells of type '{meshio_type}' for element '{element_type}'.")
    return fe.Mesh(jnp.asarray(points), jnp.asarray(cells, dtype=jnp.int32), ele_type=element_type)


def _make_feax_dirichlet_specs(domain, vec: int):
    import feax as fe

    specs = []
    tags = list(getattr(domain, "_fem_dirichlet_tags", []))
    value_fns = getattr(domain, "_fem_dirichlet_value_fns", {}) or {}

    component_names = {0: "x", 1: "y", 2: "z"}

    for tag in tags:
        loc_fn = domain._make_tag_location_fn(tag)
        if loc_fn is None:
            domain.log.warning(f"Dirichlet tag '{tag}' not found in mesh tags. Skipping.")
            continue

        normalized = _normalize_dirichlet_value(value_fns.get(tag, 0.0), vec)
        if vec == 1:
            fn = normalized if callable(normalized) else _const_bc_fn(normalized)
            specs.append(fe.DirichletBCSpec(location=loc_fn, component="all", value=fn))
            continue

        if callable(normalized):
            specs.append(fe.DirichletBCSpec(location=loc_fn, component="all", value=normalized))
            continue

        if isinstance(normalized, dict):
            # Partial / per-component (roller) BC: one spec per named component only.
            for comp, fn in normalized.items():
                specs.append(fe.DirichletBCSpec(location=loc_fn, component=component_names.get(comp, comp), value=fn))
            continue

        if isinstance(normalized, (list, tuple)):
            for comp, fn in enumerate(normalized):
                specs.append(
                    fe.DirichletBCSpec(
                        location=loc_fn,
                        component=component_names.get(comp, comp),
                        value=fn,
                    )
                )
            continue

        raise TypeError(f"Unsupported normalized Dirichlet value type for tag '{tag}': {type(normalized).__name__}")

    return specs


_ELEMENT_FOR_ORDER = {(2, 1): "TRI3", (2, 2): "TRI6", (3, 1): "TET4", (3, 2): "TET10"}


def _element_for_order(dimension: int, order: int) -> str:
    """Simplex element type for a coupled field's ``(dimension, order)``."""
    et = _ELEMENT_FOR_ORDER.get((int(dimension), int(order)))
    if et is None:
        raise ValueError(
            f"jno.fem: no element for dimension {dimension}, order {order} (coupled fields support order 1, 2)."
        )
    return et


def _make_multifield_dirichlet_specs(domain, fields, field_index):
    """Per-field Dirichlet specs (with ``variable_index``) for a coupled problem.

    Reads ``domain._fem_dirichlet_by_field`` = ``{field_index: {region: value}}`` and
    mirrors ``_make_feax_dirichlet_specs`` per field, tagging each spec with its
    ``variable_index`` so feax constrains the right block."""
    import feax as fe

    component_names = {0: "x", 1: "y", 2: "z"}
    by_field = getattr(domain, "_fem_dirichlet_by_field", {}) or {}
    specs = []
    for fidx, region_values in by_field.items():
        vec = int(fields[fidx]["vec"])
        for tag, value in region_values.items():
            loc_fn = domain._make_tag_location_fn(tag)
            if loc_fn is None:
                domain.log.warning(f"Dirichlet tag '{tag}' not found in mesh tags. Skipping.")
                continue
            normalized = _normalize_dirichlet_value(value, vec)
            if vec == 1:
                fn = normalized if callable(normalized) else _const_bc_fn(normalized)
                specs.append(fe.DirichletBCSpec(location=loc_fn, component="all", value=fn, variable_index=fidx))
            elif callable(normalized):
                specs.append(fe.DirichletBCSpec(location=loc_fn, component="all", value=normalized, variable_index=fidx))
            elif isinstance(normalized, dict):
                for comp, fn in normalized.items():
                    specs.append(
                        fe.DirichletBCSpec(
                            location=loc_fn, component=component_names.get(comp, comp), value=fn, variable_index=fidx
                        )
                    )
            elif isinstance(normalized, (list, tuple)):
                for comp, fn in enumerate(normalized):
                    specs.append(
                        fe.DirichletBCSpec(
                            location=loc_fn, component=component_names.get(comp, comp), value=fn, variable_index=fidx
                        )
                    )
    return specs


def _build_multifield_feax_problem(domain, ir, fields, field_index, *, apply_dirichlet, store_on_domain):
    """Build a multi-variable FEAX Problem + block Dirichlet BC for coupled fields.

    Each field gets its own mesh/vec/element; one universal kernel groups the weak
    terms by their test field. feax autodiffs this into the block matrix downstream
    (``_assemble_fem_system_concrete`` is unchanged)."""
    import feax as fe

    dim = domain.dimension
    quad_degree = getattr(domain, "_fem_quad_degree", None) or 2
    eles = [_element_for_order(dim, f["order"]) for f in fields]
    meshes = [_build_feax_mesh(domain, et) for et in eles]
    vecs = [int(f["vec"]) for f in fields]

    def _typed_term(t):
        """Lower one IR weak term to ``(coeff, test_field_index)`` for the block kernel."""
        coeff = _lower_statefield_to_trial(t.coeff, {})
        tfi = _test_field_index(coeff, field_index)
        if tfi is None:
            raise ValueError(
                "jno.fem: each coupled weak term must contain exactly one test field "
                "(it determines the equation block); got a term with zero or several."
            )
        return coeff, tfi

    term_list = []
    boundary_terms_by_tag: dict[str, list] = {}
    for t in ir.terms:
        if t.channel != "raw":
            continue
        if t.support == "volume":
            term_list.append(_typed_term(t))
        elif t.support == "boundary":
            boundary_terms_by_tag.setdefault(t.region_id, []).append(_typed_term(t))

    # Temporal tags (and runtime parameters) used by the coupled coefficients, so a
    # t-dependent coefficient (e.g. a body force e^{-t}, a source f(x,t)) is evaluated at
    # the runtime time via feax InternalVars rather than baked at a fixed t.
    temporal_tags: set = set()
    runtime_parameter_tags: list = []
    for _coeff, _tfi in term_list + [t for terms in boundary_terms_by_tag.values() for t in terms]:
        temporal_tags.update(_collect_temporal_tags_for_feax(_coeff))
        _collect_runtime_parameter_tags_for_feax(_coeff, runtime_parameter_tags)
    temporal_tags = tuple(sorted(temporal_tags))
    runtime_parameter_tags = tuple(runtime_parameter_tags)

    problem_ref: dict[str, Any] = {"problem": None}
    kernel = _make_multifield_volume_kernel(
        domain, term_list, fields, field_index, temporal_tags, runtime_parameter_tags, problem_ref
    )

    # Coupled surface (Neumann/Robin) terms: one universal surface kernel per boundary
    # tag, grouping that tag's terms by test field into the block residual. feax matches
    # location_fns[i] <-> get_universal_kernels_surface()[i] by index, so both lists are
    # built in lockstep over the same ordered tags.
    location_fns = []
    surface_kernels = []
    for tag, tag_terms in boundary_terms_by_tag.items():
        loc_fn = domain._make_tag_location_fn(tag)
        if loc_fn is None:
            domain.log.warning(f"Boundary tag '{tag}' not found while building coupled surface locations. Skipping.")
            continue
        location_fns.append(loc_fn)
        surface_kernels.append(
            _make_multifield_surface_kernel(
                domain, tag_terms, fields, field_index, tag, temporal_tags, runtime_parameter_tags, problem_ref
            )
        )

    class GeneratedMultifieldProblem(fe.Problem):
        def get_universal_kernel(self_inner):
            return kernel

        def get_universal_kernels_surface(self_inner):
            return surface_kernels

    problem = GeneratedMultifieldProblem(
        meshes, vec=vecs, dim=dim, ele_type=eles, gauss_order=quad_degree, location_fns=location_fns
    )
    problem_ref["problem"] = problem

    bc_specs = _make_multifield_dirichlet_specs(domain, fields, field_index) if apply_dirichlet else []
    bc = fe.DirichletBCConfig(bc_specs).create_bc(problem)

    if store_on_domain:
        domain._feax_problem = problem
        domain._feax_bc = bc

    return problem, bc


def _zero_mass_dirichlet_rows(M, bc):
    """Zero a mass matrix's Dirichlet **rows** so ``M u̇ + A u = c`` reads ``u[d]=g``.

    feax applies symmetric Dirichlet (identity rows) to *every* assembled matrix — correct
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


def _build_feax_problem(domain, ir, *, apply_dirichlet: bool = True, store_on_domain: bool = True, fields_override=None):
    """
    Build a FEAX Problem and Dirichlet BC object from lowered weak-form IR.

    The returned FEAX problem owns the generated volume and surface kernels.
    When `store_on_domain=True`, the FEAX problem and BC are cached on the
    domain for later reuse.

    ``fields_override`` is an optional ``(fields, field_index)`` pair forcing the
    multi-field block layout instead of inferring it from this IR's own terms. The
    transient route passes it so the separately-assembled mass and operator blocks
    share one field ordering (and so a block is built for every field even if this
    IR only mentions some of them).
    """
    import feax as fe

    trial_cache: dict[tuple[int, str, tuple[Any, ...]], TrialFunction] = {}

    volume_expr = _lower_statefield_to_trial(ir.volume_expr, trial_cache)
    boundary_exprs = {k: _lower_statefield_to_trial(v, trial_cache) for k, v in ir.boundary_exprs.items()}

    if volume_expr is None and len(boundary_exprs) == 0:
        raise ValueError("No terms found for FEM assembly.")

    # Coupled (multi-field) weak form -> multi-variable block assembly.
    _mf_fields, _mf_index = fields_override if fields_override is not None else _infer_fields(volume_expr)
    if len(_mf_fields) > 1:
        return _build_multifield_feax_problem(
            domain, ir, _mf_fields, _mf_index, apply_dirichlet=apply_dirichlet, store_on_domain=store_on_domain
        )

    metadata = _infer_trial_metadata(volume_expr if volume_expr is not None else next(iter(boundary_exprs.values())))
    vec = int(metadata["vec"])
    value_shape = metadata["value_shape"]

    element_type = getattr(domain, "_fem_element_type", None)
    quad_degree = getattr(domain, "_fem_quad_degree", None)

    if element_type is None:
        element_type = "TRI3"
    if quad_degree is None:
        quad_degree = 2

    mesh = _build_feax_mesh(domain, element_type)

    temporal_tags_set = set()
    if volume_expr is not None:
        temporal_tags_set.update(_collect_temporal_tags_for_feax(volume_expr))
    for expr in boundary_exprs.values():
        temporal_tags_set.update(_collect_temporal_tags_for_feax(expr))
    temporal_tags = tuple(sorted(temporal_tags_set))

    runtime_parameter_tags_set = []
    if volume_expr is not None:
        _collect_runtime_parameter_tags_for_feax(volume_expr, runtime_parameter_tags_set)
    for expr in boundary_exprs.values():
        _collect_runtime_parameter_tags_for_feax(expr, runtime_parameter_tags_set)
    runtime_parameter_tags = tuple(runtime_parameter_tags_set)

    problem_ref: dict[str, Any] = {"problem": None}

    active_boundary_tags: List[str] = []
    location_fns = []
    surface_kernels = []
    for tag, expr in boundary_exprs.items():
        loc_fn = domain._make_tag_location_fn(tag)
        if loc_fn is None:
            domain.log.warning(f"Boundary tag '{tag}' not found while building FEAX surface locations. Skipping.")
            continue
        active_boundary_tags.append(tag)
        location_fns.append(loc_fn)
        surface_kernels.append(
            _make_universal_surface_kernel(domain, expr, tag, value_shape, runtime_parameter_tags, temporal_tags)
        )

    # FEAX always evaluates the volume kernel before adding optional surface
    # contributions. A boundary-only source, such as a pure Neumann load,
    # therefore still needs a valid zero-valued volume kernel.
    if volume_expr is not None:
        volume_kernel = _make_universal_volume_kernel(
            domain,
            volume_expr,
            value_shape,
            temporal_tags,
            runtime_parameter_tags,
            problem_ref,
        )
    else:

        def volume_kernel(
            cell_sol_flat,
            physical_quad_points,
            cell_shape_grads,
            cell_JxW,
            cell_v_grads_JxW,
            *cell_internal_vars,
        ):
            del (
                physical_quad_points,
                cell_shape_grads,
                cell_JxW,
                cell_v_grads_JxW,
                cell_internal_vars,
            )

            # FEAX expects one local residual vector per cell. For a
            # boundary-only weak form, the volume contribution is identically
            # zero and the actual load is assembled by the surface kernels.
            return jnp.zeros_like(jnp.asarray(cell_sol_flat))

    class GeneratedProblem(fe.Problem):
        def get_universal_kernel(self_inner):
            return volume_kernel

        def get_universal_kernels_surface(self_inner):
            return surface_kernels

    problem = GeneratedProblem(
        mesh,
        vec=vec,
        dim=domain.dimension,
        ele_type=element_type,
        gauss_order=quad_degree,
        location_fns=location_fns,
    )
    problem_ref["problem"] = problem

    bc_specs = _make_feax_dirichlet_specs(domain, vec) if apply_dirichlet else []
    bc = fe.DirichletBCConfig(bc_specs).create_bc(problem)

    if store_on_domain:
        domain._feax_problem = problem
        domain._feax_bc = bc

    return problem, bc


def _dense_array(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    try:
        return jnp.asarray(A)
    except Exception:
        return jnp.asarray(np.asarray(A))


def _make_internal_vars(
    fe_module,
    temporal_tags,
    t,
    *,
    n_cells: int,
    dtype=None,
    runtime_parameter_tags=(),
    runtime_parameter_values=None,
    extra_volume_vars=(),
):
    """
    Build FEAX InternalVars in a batched shape FEAX can slice.

    Each temporal variable is broadcast to shape (n_cells, 1).
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
            # feax's gather_internal_vars slices it per cell, the kernel interpolates.
            vol.append(flat)
    for v in extra_volume_vars:
        arr = jnp.asarray(v, dtype=dtype)
        if arr.ndim == 0:
            arr = jnp.full((int(n_cells), 1), arr, dtype=arr.dtype)
        vol.append(arr)

    return fe_module.InternalVars(volume_vars=tuple(vol))


def _prepare_feax_runtime(
    domain,
    ir,
    *,
    apply_dirichlet=True,
    need_jacobian=True,
    symmetric_bc=True,
    fields_override=None,
):
    """
    Prepare reusable FEAX residual/Jacobian runtime objects for an IR.

    Returns a dictionary containing the FEAX problem, BC, residual callable,
    optional Jacobian callable, reference state, dtype, temporal tags, and
    number of mesh cells. ``fields_override`` forces the multi-field block layout
    (used by the coupled transient forcing path).
    """
    import feax as fe

    problem, bc = _build_feax_problem(
        domain,
        ir,
        apply_dirichlet=apply_dirichlet,
        store_on_domain=False,
        fields_override=fields_override,
    )

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = fe.create_J_bc_function(problem, bc, symmetric=symmetric_bc) if need_jacobian else None

    size = int(problem.num_total_dofs_all_vars)
    dtype = _default_float_dtype()

    try:
        u_ref = fe.zero_like_initial_guess(problem, bc)
    except Exception:
        u_ref = jnp.zeros((size,), dtype=dtype)

    u_ref = jnp.asarray(u_ref, dtype=dtype)

    temporal_tags = set()
    for term in getattr(ir, "terms", []):
        temporal_tags.update(_collect_temporal_tags_for_feax(term.coeff))
    temporal_tags = tuple(sorted(temporal_tags))
    runtime_parameter_tags = []
    for term in getattr(ir, "terms", []):
        _collect_runtime_parameter_tags_for_feax(term.coeff, runtime_parameter_tags)
    runtime_parameter_tags = tuple(runtime_parameter_tags)

    element_type = getattr(domain, "_fem_element_type", None)
    if element_type is None:
        element_type = "TRI3"

    meshio_type = _meshio_type_for_element(element_type)

    if meshio_type not in domain.mesh.cells_dict:
        raise KeyError(
            f"Mesh cell type '{meshio_type}' for element_type='{element_type}' "
            f"not found in domain.mesh.cells_dict. "
            f"Available: {list(domain.mesh.cells_dict.keys())}"
        )

    n_cells = int(np.asarray(domain.mesh.cells_dict[meshio_type]).shape[0])

    return {
        "problem": problem,
        "bc": bc,
        "res_bc": res_bc,
        "jac_bc": jac_bc,
        "size": size,
        "dtype": dtype,
        "u_ref": u_ref,
        "temporal_tags": temporal_tags,
        "runtime_parameter_tags": runtime_parameter_tags,
        "n_cells": n_cells,
    }


"""
Periodic boundary-condition support for the FEAX-backed time/static routes.

The FEAX assembly produces *full* unconstrained Galerkin operators (mass M,
operator A, parametric basis K_i, affine bias, forcing). Periodicity is not a
property of those matrices: left/right (or bottom/top) boundary nodes are
independent free DOFs that, left alone, satisfy a natural zero-flux condition
rather than a periodic one.

Periodicity is enforced algebraically through a prolongation matrix ``P`` that
identifies slave DOFs with their master DOFs:

    u_full = P @ u_red                          (n_full x n_red)

The reduced semidiscrete system that the time integrator actually solves is

    (P^T M P) u_red_dot + (P^T A P) u_red = P^T (c + f).

This module builds ``P`` by coordinate matching on the user-declared periodic
tag pairs and provides the small reduction/prolongation helpers used by the
time route and the Diffrax adapter.

Only node-level (scalar, vec=1) and node-major vector layouts
(dof = node*vec + comp) are handled; the vector case is obtained from the
node-level ``P`` via a Kronecker product with the identity.
"""


def _periodic_facet_weights(
    t_query: np.ndarray,
    facet_node_ids: np.ndarray,
    pts: np.ndarray,
    transverse: List[int],
) -> List[Tuple[int, float]] | None:
    """Interpolation weights for a slave at transverse coord ``t_query`` on the
    master boundary facets (node-to-segment / mortar-lite identification).

    ``facet_node_ids`` is ``(n_facets, k)`` of global node ids. **2D** (transverse
    1-D): facets are edges -- columns 0,1 are the vertices, optional column 2 the
    midside node (``k == 3`` ⇒ P2). **3D** (transverse 2-D): facets are triangles --
    columns 0,1,2 the vertices, optional columns 3,4,5 the edge midpoints (``k == 6``
    ⇒ P2). Returns ``[(node_id, weight), ...]`` whose weights sum to 1 (partition of
    unity ⇒ constants reproduced; linear/quadratic-on-the-facet reproduced exactly).
    """
    tq = np.atleast_1d(np.asarray(t_query, dtype=float))
    facet_node_ids = np.asarray(facet_node_ids, dtype=int)
    if facet_node_ids.ndim != 2 or facet_node_ids.shape[0] == 0:
        return None
    k = facet_node_ids.shape[1]

    if tq.shape[0] == 1:  # 2D: locate the master edge spanning the slave's transverse coord
        t = float(tq[0])
        tr = transverse[0]
        a_ids, b_ids = facet_node_ids[:, 0], facet_node_ids[:, 1]
        ta, tb = pts[a_ids, tr], pts[b_ids, tr]
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

    if tq.shape[0] == 2:  # 3D: locate the master triangle containing the slave, barycentric weights
        tr = transverse
        pa = pts[facet_node_ids[:, 0]][:, tr]
        pb = pts[facet_node_ids[:, 1]][:, tr]
        pc = pts[facet_node_ids[:, 2]][:, tr]
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

    raise NotImplementedError("3D periodic interpolation (triangle facets) is milestone M2.")


def build_periodic_prolongation(
    points: np.ndarray,
    pairs: Sequence[Tuple[str, str]],
    tag_indices: Dict[str, np.ndarray],
    *,
    vec: int = 1,
    tol: float | None = None,
    facets: Dict[str, np.ndarray] | None = None,
) -> Dict[str, object]:
    """Build the node-level periodic prolongation matrix ``P``.

    Identifies each slave-face node with the master face. When the two faces have
    the **same node layout** (structured / conforming) this is an exact 0/1
    node-to-node map. When they **don't** (unstructured / non-matching), a slave
    that has no master node within ``tol`` is instead tied to the master *facet*
    it lands on by **node-to-segment interpolation** (linear for P1, quadratic for
    P2) — master–slave MPC elimination, consistent (partition of unity) though not
    a full dual-mortar coupling. See :func:`_periodic_facet_weights`.

    Parameters
    ----------
    points:
        ``(n_nodes, dim)`` array of FEM node coordinates (the assembly mesh).
    pairs:
        Ordered ``(master_tag, slave_tag)`` boundary pairings, e.g.
        ``[("left", "right"), ("bottom", "top")]``.
    tag_indices:
        Mapping from boundary tag name to the global node ids on that tag.
    vec:
        Number of scalar components per node. ``vec > 1`` returns the
        node-major expansion ``kron(P_node, I_vec)``.
    tol:
        Coordinate-matching tolerance for the transverse coordinates. When
        ``None`` it is derived from the bounding-box diagonal.
    facets:
        Optional ``{master_tag: (n_facets, k) node-id array}`` of the master
        boundary facets, required only for the non-matching (interpolatory) path.

    Returns
    -------
    dict with keys:
        ``P``               : ``(n_full, n_red)`` dense jnp prolongation matrix.
        ``P_node``          : node-level prolongation (``vec == 1`` form).
        ``kept_nodes``      : sorted global ids of the retained (master/free) nodes.
        ``slave_to_master`` : resolved slave-node -> master-node mapping (exact ties).
        ``n_full``          : full node count.
        ``n_red``           : reduced node count.
        ``vec``             : component count used.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_nodes = pts.shape[0]
    facets = facets or {}

    if tol is None:
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        tol = max(span, 1.0) * 1.0e-6

    # slave -> master node (exact 0/1 tie); slave -> [(master node, weight)] (interpolated tie).
    slave_to_master: Dict[int, int] = {}
    slave_interp: Dict[int, List[Tuple[int, float]]] = {}

    for master_tag, slave_tag in pairs:
        if master_tag not in tag_indices or slave_tag not in tag_indices:
            raise KeyError(
                f"Periodic pair ({master_tag!r}, {slave_tag!r}) refers to a tag "
                f"that is not present in the mesh. Known tags: {sorted(tag_indices)}."
            )

        m_ids = np.asarray(tag_indices[master_tag], dtype=int).reshape(-1)
        s_ids = np.asarray(tag_indices[slave_tag], dtype=int).reshape(-1)
        m_pts = pts[m_ids]
        s_pts = pts[s_ids]

        # Periodic axis = coordinate whose tag means differ the most.
        axis = int(np.argmax(np.abs(m_pts.mean(axis=0) - s_pts.mean(axis=0))))
        transverse = [d for d in range(pts.shape[1]) if d != axis]

        m_trans = m_pts[:, transverse]
        s_trans = s_pts[:, transverse]

        # Nearest transverse master node for every slave node.
        d2 = np.sum((s_trans[:, None, :] - m_trans[None, :, :]) ** 2, axis=-1)
        nn = np.argmin(d2, axis=1) if m_ids.size else np.zeros(len(s_ids), dtype=int)
        dist = np.sqrt(d2[np.arange(len(s_ids)), nn]) if m_ids.size and len(s_ids) else np.zeros(len(s_ids))

        for k, sid in enumerate(s_ids):
            if m_ids.size and dist[k] <= tol:  # conforming: exact node-to-node (corners land here too)
                slave_to_master[int(sid)] = int(m_ids[nn[k]])
                continue
            # non-matching: tie to the master facet by interpolation
            w = _periodic_facet_weights(s_trans[k], facets.get(master_tag), pts, transverse) if facets else None
            if w is None:
                raise ValueError(
                    f"Periodic matching for ({master_tag!r}, {slave_tag!r}) failed at slave node {int(sid)}: "
                    f"nearest master node is {float(dist[k]):.3e} away (tol {tol:.3e}) and no master facet "
                    "connectivity was supplied for interpolation. Pass `facets=` (unstructured) or use a "
                    "conforming mesh."
                )
            slave_interp[int(sid)] = w

    slave_set = set(slave_to_master) | set(slave_interp)

    # Each slave is a linear combination of other nodes (exact: one master, weight 1; interpolated:
    # facet shape-function weights). Those nodes may themselves be slaves — a corner is a slave in
    # several directions, and an interpolation can land on a master edge whose endpoint is itself a
    # slave — so resolve every slave **transitively** to kept (master) nodes. This handles any number
    # of periodic directions (e.g. a doubly-periodic cell) with a single general mechanism.
    raw: Dict[int, List[Tuple[int, float]]] = {sid: [(m, 1.0)] for sid, m in slave_to_master.items()}
    raw.update({sid: list(ws) for sid, ws in slave_interp.items()})
    resolved: Dict[int, Dict[int, float]] = {}

    def _expand(node: int, stack: frozenset) -> Dict[int, float]:
        if node not in slave_set:
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

    kept_nodes: List[int] = [i for i in range(n_nodes) if i not in slave_set]
    reduced_index = {node: r for r, node in enumerate(kept_nodes)}
    n_red = len(kept_nodes)

    P_node = np.zeros((n_nodes, n_red), dtype=np.float64)
    for i in range(n_nodes):
        if i in slave_set:
            for kept_node, weight in _expand(i, frozenset()).items():
                P_node[i, reduced_index[kept_node]] += weight
        else:
            P_node[i, reduced_index[i]] = 1.0

    # Informational exact-chain map (single kept master per exact slave); interpolated slaves omitted.
    final_master = {sid: next(iter(_expand(sid, frozenset()))) for sid in slave_to_master}

    P_node_j = jnp.asarray(P_node)
    if vec == 1:
        P = P_node_j
    else:
        P = jnp.kron(P_node_j, jnp.eye(vec, dtype=P_node_j.dtype))

    return {
        "P": P,
        "P_node": P_node_j,
        "kept_nodes": np.asarray(kept_nodes, dtype=np.int64),
        "slave_to_master": final_master,
        "n_full": int(n_nodes * vec),
        "n_red": int(n_red * vec),
        "vec": int(vec),
    }


# ---------------------------------------------------------------------------
# Operator / state reduction and prolongation
# ---------------------------------------------------------------------------


def reduce_matrix(P, mat):
    """Galerkin reduction ``P^T mat P``."""
    P = jnp.asarray(P)
    mat = jnp.asarray(mat, dtype=P.dtype)
    return P.T @ mat @ P


def reduce_vector(P, vec):
    """Reduce a full-space load/bias vector via ``P^T vec``."""
    P = jnp.asarray(P)
    vec = jnp.asarray(vec, dtype=P.dtype).reshape(-1)
    return P.T @ vec


def restrict_state(P, state_full, kept_nodes, vec: int = 1):
    """Restrict a full initial state to the reduced master DOFs.

    For a consistent periodic initial condition (matching values on opposite
    faces) gathering the kept-node entries is exact and avoids the doubling
    that ``P^T`` would introduce on master nodes.
    """
    state_full = jnp.asarray(state_full).reshape(-1)
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
    P = jnp.asarray(P)
    reduced = jnp.asarray(reduced, dtype=P.dtype)
    if reduced.ndim == 1:
        return P @ reduced
    # (..., n_red) @ (n_red, n_full) -> (..., n_full)
    return reduced @ P.T
