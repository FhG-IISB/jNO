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
from typing import Any, Dict, List

import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ...trace import (
    Placeholder,
    Literal,
    BinaryOp,
    FunctionCall,
    Variable,
    ModelCall,
    OperationDef,
    OperationCall,
    Jacobian,
    Hessian,
    Tracker,
    TrialFunction,
    TestFunction,
    TensorTag,
    Constant,
    Assembly,
    GroupedAssembly,
    StateField,
)
from .solver_helper import contains_node_type , iter_children
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
            if isinstance(a, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
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
            if isinstance(a, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
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
            if isinstance(a, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
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
            if isinstance(v, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
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
            if isinstance(v, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
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
    Returns one callable for scalar fields or a list of callables for vector fields.
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
        out = [_const_bc_fn(0.0) for _ in range(vec)]
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
            return out[0]
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
            if isinstance(a, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
                _collect_temporal_tags_for_feax(a, out)
        return out

    if isinstance(node, Jacobian):
        _collect_temporal_tags_for_feax(node.target, out)
        for v in node.variables:
            if isinstance(v, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
                _collect_temporal_tags_for_feax(v, out)
        return out

    if isinstance(node, Hessian):
        _collect_temporal_tags_for_feax(node.target, out)
        for v in node.variables:
            if isinstance(v, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction,
                              Jacobian, Hessian, BinaryOp, FunctionCall, ModelCall,
                              OperationDef, OperationCall, Tracker, StateField)):
                _collect_temporal_tags_for_feax(v, out)
        return out

    return out


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

def _eval_expr_for_feax(domain, node, local):
    """
    Evaluate a jNO symbolic expression inside a FEAX local kernel.

    The `local` dictionary contains quadrature coordinates, shape values,
    shape gradients, local cell DOFs, domain context, and optional temporal
    InternalVars. This evaluator supports literals, constants, variables,
    tensor tags, TrialFunction/TestFunction values, their Jacobians, binary
    operations, and FunctionCall nodes.
    """
    if not isinstance(node, (Literal, Constant, TensorTag, Variable, TestFunction, TrialFunction, Jacobian, BinaryOp, FunctionCall)):
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
        return _expand_test_shape_vals(local["shape_vals"], n_comp)

    if isinstance(node, TrialFunction):
        vals = local["shape_vals"]
        flat_interp = jnp.sum(vals[:, :, None] * local["cell_sol"][None, :, :], axis=1)
        value_shape = getattr(node, "value_shape", ())
        if len(value_shape) == 0:
            return flat_interp
        return _reshape_components_last(flat_interp, value_shape)

    if isinstance(node, Jacobian):
        dims = []
        for var in node.variables:
            if not isinstance(var, Variable):
                raise NotImplementedError("FEAX backend expects Jacobian variables to be domain.variable(...) placeholders.")
            dims.append(var.dim[0])
        if len(dims) == 0:
            raise ValueError("Jacobian node has no differentiation variables")

        if isinstance(node.target, TestFunction):
            n_comp = _value_shape_num_components(getattr(node.target, "value_shape", ()))
            grads = local["shape_grads"]
            if n_comp == 1:
                comps = [grads[..., dim0] for dim0 in dims]
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)
            eye = jnp.eye(n_comp, dtype=grads.dtype)
            comps = [grads[..., dim0][:, :, None, None] * eye[None, None, :, :] for dim0 in dims]
            if len(comps) == 1:
                return comps[0]
            return jnp.stack(comps, axis=-1)

        if isinstance(node.target, TrialFunction):
            grads = local["shape_grads"]
            cell_sol = local["cell_sol"]
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
        if node.op == "+":
            return a + b
        if node.op == "-":
            return a - b
        if node.op == "*":
            return a * b
        if node.op == "/":
            return a / b
        if node.op == "**":
            return a ** b
        raise NotImplementedError(f"Unsupported binary operator: {node.op}")

    if isinstance(node, FunctionCall):
        args = [_eval_expr_for_feax(domain, arg, local) for arg in node.args]
        kwargs = node.kwargs if node.kwargs else {}
        return node.fn(*args, **kwargs)

    raise NotImplementedError(f"Unsupported weak-form node for FEAX backend: {type(node).__name__}")


# --------------------------------
# FEAX kernel builders
# --------------------------------

def _eval_volume_integrand(domain, expr,value_shape, cell_sol_flat, physical_quad_points, cell_shape_grads,cell_JxW, cell_v_grads_JxW,temporal_tags, problem_ref,*cell_internal_vars,):
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
        "volume_vars": tuple(cell_internal_vars),
    }

    val = _eval_expr_for_feax(domain, expr, local)
    weights = cell_JxW[0]
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]


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
        raise ValueError(
            f"Surface kernel DOF size {cell_sol_flat.size} is not divisible by vec={vec} for tag '{tag}'."
        )

    n_parent_nodes = cell_sol_flat.size // vec
    cell_sol = cell_sol_flat.reshape(n_parent_nodes, vec)

    if face_shape_vals.ndim != 2:
        raise ValueError(
            f"Expected face_shape_vals.ndim == 2, got shape {face_shape_vals.shape} for tag '{tag}'."
        )
    if face_shape_grads.ndim != 3:
        raise ValueError(
            f"Expected face_shape_grads.ndim == 3, got shape {face_shape_grads.shape} for tag '{tag}'."
        )
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
        raise ValueError(
            f"Unsupported face_nanson_scale shape {face_nanson_scale.shape} for tag '{tag}'."
        )

    if weights.shape[0] != nq:
        raise ValueError(
            f"Boundary weight/quadrature mismatch on '{tag}': "
            f"weights.shape={weights.shape}, nq={nq}."
        )

    boundary_normals = None
    if hasattr(domain, "normals_by_tag"):
        normal_lookup_tag = f"gauss_{tag}" if f"gauss_{tag}" in domain.normals_by_tag else tag
        if normal_lookup_tag in domain.normals_by_tag and normal_lookup_tag in getattr(domain, "_mesh_pool", {}):
            normal_pts = jnp.asarray(np.asarray(domain._mesh_pool[normal_lookup_tag])[:, : domain.dimension])
            normal_vals = jnp.asarray(np.asarray(domain.normals_by_tag[normal_lookup_tag])[:, : domain.dimension])
            if len(normal_pts) > 0 and len(normal_pts) == len(normal_vals):
                x_use = physical_surface_quad_points[:, : domain.dimension]
                d2 = jnp.sum((normal_pts[None, :, :] - x_use[:, None, :]) ** 2, axis=-1)
                nn_idx = jnp.argmin(d2, axis=1)
                boundary_normals = normal_vals[nn_idx]

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
    }

    val = _eval_expr_for_feax(domain, expr, local)
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]

def _make_universal_volume_kernel(domain, expr, value_shape, temporal_tags, problem_ref):
    """
    Create the FEAX universal volume kernel for a lowered weak-form expression.
    """
    def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW, *cell_internal_vars):
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
            problem_ref,
            *cell_internal_vars,
        )
    return kernel


def _make_universal_surface_kernel(domain, expr, tag, value_shape, temporal_tags):
    def kernel(cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, *cell_internal_vars_surface):
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


def _build_feax_mesh(domain, element_type: str):
    import feax as fe

    meshio_type = _meshio_type_for_element(element_type)
    points = jnp.asarray(domain.mesh.points[:, : domain.dimension])
    cells = jnp.asarray(domain.mesh.cells_dict[meshio_type], dtype=jnp.int32)
    return fe.Mesh(points, cells, ele_type=element_type)


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

        if isinstance(normalized, (list, tuple)):
            for comp, fn in enumerate(normalized):
                specs.append(fe.DirichletBCSpec(location=loc_fn, component=component_names.get(comp, comp), value=fn))
            continue

        raise TypeError(f"Unsupported normalized Dirichlet value type for tag '{tag}': {type(normalized).__name__}")

    return specs


def _build_feax_problem(domain, ir, *, apply_dirichlet: bool = True, store_on_domain: bool = True):
    """
    Build a FEAX Problem and Dirichlet BC object from lowered weak-form IR.

    The returned FEAX problem owns the generated volume and surface kernels.
    When `store_on_domain=True`, the FEAX problem and BC are cached on the
    domain for later reuse.
    """
    import feax as fe
    trial_cache = {}

    volume_expr = _lower_statefield_to_trial(ir.volume_expr, trial_cache)
    boundary_exprs = {
        k: _lower_statefield_to_trial(v, trial_cache)
        for k, v in ir.boundary_exprs.items()
    }

    if volume_expr is None and len(boundary_exprs) == 0:
        raise ValueError("No terms found for FEM assembly.")

    metadata = _infer_trial_metadata(
        volume_expr if volume_expr is not None else next(iter(boundary_exprs.values()))
    )
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

    problem_ref = {"problem": None}

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
        surface_kernels.append(_make_universal_surface_kernel(domain, expr, tag, value_shape, temporal_tags))

    volume_kernel = None
    if volume_expr is not None:
        volume_kernel = _make_universal_volume_kernel(domain, volume_expr, value_shape, temporal_tags, problem_ref)

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


def _make_internal_vars(fe_module, temporal_tags, t, *, n_cells: int, dtype=None, extra_volume_vars=()):
    """
    Build FEAX InternalVars in a batched shape FEAX can slice.

    Each temporal variable is broadcast to shape (n_cells, 1).
    """
    vol = []

    if temporal_tags:
        t0 = jnp.asarray(t, dtype=dtype)
        t_batched = jnp.full((int(n_cells), 1), t0, dtype=t0.dtype)
        vol.extend([t_batched for _ in temporal_tags])

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
):
    """
    Prepare reusable FEAX residual/Jacobian runtime objects for an IR.

    Returns a dictionary containing the FEAX problem, BC, residual callable,
    optional Jacobian callable, reference state, dtype, temporal tags, and
    number of mesh cells.
    """
    import feax as fe

    problem, bc = _build_feax_problem(
        domain,
        ir,
        apply_dirichlet=apply_dirichlet,
        store_on_domain=False,
    )

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = (
        fe.create_J_bc_function(problem, bc, symmetric=symmetric_bc)
        if need_jacobian
        else None
    )

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
        "n_cells": n_cells,
    }