from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from ..trace import (
    Literal,
    Constant,
    TensorTag,
    Variable,
    TestFunction,
    TrialFunction,
    Jacobian,
    BinaryOp,
    FunctionCall,
    FemResidualOperator,
)

from .weak_form import _sum_terms


# --------------------------------
# FEM boundary-condition helpers
# --------------------------------

def _default_float_dtype():
    return jnp.asarray(0.0).dtype


@dataclass(frozen=True)
class DirichletBC:
    tags: tuple[str, ...]
    values: object = None


@dataclass(frozen=True)
class NeumannBC:
    tags: tuple[str, ...]


def _as_tags(tags) -> tuple[str, ...]:
    if isinstance(tags, str):
        return (tags,)
    if isinstance(tags, Sequence):
        out = tuple(str(t) for t in tags)
        if len(out) == 0:
            raise ValueError("Boundary tag list cannot be empty.")
        return out
    raise TypeError(f"Boundary tags must be a string or a sequence of strings, got {type(tags).__name__}.")


def dirichlet(tags, values=None):
    return DirichletBC(tags=_as_tags(tags), values=values)


def neumann(tags):
    return NeumannBC(tags=_as_tags(tags))


def _const_bc_fn(value):
    value = float(value)
    return lambda p, c=value: c


def _normalize_dirichlet_value(value, vec: int):
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


def expand_bcs(bcs, vec: int):
    dirichlet_tags = []
    dirichlet_value_fns = {}
    neumann_tags = []

    for bc in bcs:
        if isinstance(bc, DirichletBC):
            for tag in bc.tags:
                if tag not in dirichlet_tags:
                    dirichlet_tags.append(tag)
                dirichlet_value_fns[tag] = _normalize_dirichlet_value(bc.values, vec)
        elif isinstance(bc, NeumannBC):
            for tag in bc.tags:
                if tag not in neumann_tags:
                    neumann_tags.append(tag)
        else:
            raise TypeError(f"Unsupported BC entry '{type(bc).__name__}'. Use dirichlet(...) or neumann(...).")

    return dirichlet_tags, dirichlet_value_fns, neumann_tags


# --------------------------------
# small expression-inspection helpers
# --------------------------------

def _contains_node_type(node, cls) -> bool:
    if isinstance(node, cls):
        return True
    if isinstance(node, BinaryOp):
        return _contains_node_type(node.left, cls) or _contains_node_type(node.right, cls)
    if isinstance(node, FunctionCall):
        return any(_contains_node_type(a, cls) for a in node.args)
    if isinstance(node, Jacobian):
        return _contains_node_type(node.target, cls) or any(_contains_node_type(v, cls) for v in node.variables)
    return False


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
    if not _contains_node_type(expr, TestFunction):
        return False
    if _contains_node_type(expr, TrialFunction):
        return False
    if _contains_node_type(expr, Jacobian):
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
    trial_nodes = {}

    def walk(node):
        if node is None:
            return
        if isinstance(node, TrialFunction):
            trial_nodes[node.op_id] = node
            return
        for attr in ("left", "right", "target", "expr"):
            child = getattr(node, attr, None)
            if child is not None:
                walk(child)
        for attr in ("args", "variables"):
            vals = getattr(node, attr, None)
            if vals is None:
                continue
            for v in vals:
                if isinstance(v, (list, tuple)):
                    for vv in v:
                        walk(vv)
                else:
                    walk(v)

    walk(expr)
    unique_trials = list(trial_nodes.values())
    if len(unique_trials) > 1:
        raise NotImplementedError(
            "FEAX backend currently supports exactly one TrialFunction (scalar or vector valued). "
            "Multiple coupled FEM unknowns will come in the next refactor step."
        )
    trial = unique_trials[0] if unique_trials else None
    value_shape = getattr(trial, "value_shape", ()) if trial is not None else ()
    vec = _value_shape_num_components(value_shape)
    return {"trial": trial, "value_shape": value_shape, "vec": vec, "has_trial": trial is not None}


# --------------------------------
# FEAX expression evaluation helpers
# --------------------------------

def _eval_expr_for_feax(domain, node, local):
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
            # Any boundary quadrature variable like gauss_right, gauss_top, gauss_wall
            # should read from the current surface quad points inside that surface kernel.
            if isinstance(node.tag, str) and node.tag.startswith("gauss_"):
                return local["physical_quad_points"][..., dim_start:dim_end]

        else:
            # Volume quadrature variable
            if node.tag == "fem_gauss":
                return local["physical_quad_points"][..., dim_start:dim_end]

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

def _eval_volume_integrand(domain, expr, value_shape, cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
    num_nodes = cell_shape_grads.shape[1]
    vec = _value_shape_num_components(value_shape)
    cell_sol = cell_sol_flat.reshape(num_nodes, vec)

    # FEAX/JAX-FEM compatible local context
    shape_vals = domain._feax_problem.fes[0].shape_vals
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
):
    vec = _value_shape_num_components(value_shape)

    cell_sol_flat = jnp.asarray(cell_sol_flat)
    physical_surface_quad_points = jnp.asarray(physical_surface_quad_points)
    face_shape_vals = jnp.asarray(face_shape_vals)
    face_shape_grads = jnp.asarray(face_shape_grads)
    face_nanson_scale = jnp.asarray(face_nanson_scale)

    # FEAX surface kernels use parent-cell DOFs, so derive node count from cell_sol_flat
    if cell_sol_flat.ndim != 1:
        cell_sol_flat = cell_sol_flat.reshape(-1)

    if cell_sol_flat.size % vec != 0:
        raise ValueError(
            f"Surface kernel DOF size {cell_sol_flat.size} is not divisible by vec={vec} for tag '{tag}'."
        )

    n_parent_nodes = cell_sol_flat.size // vec
    cell_sol = cell_sol_flat.reshape(n_parent_nodes, vec)

    # Normalize FEAX inputs after boundary-wise vmap:
    # expected:
    #   physical_surface_quad_points : (nq, dim)
    #   face_shape_vals              : (nq, n_parent_nodes)
    #   face_shape_grads             : (nq, n_parent_nodes, dim)
    #   face_nanson_scale            : (num_vars, nq) or (nq,)
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

    # FEAX: face_nanson_scale after vmap over boundary faces is typically (num_vars, nq)
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
    }

    val = _eval_expr_for_feax(domain, expr, local)
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return ravel_pytree(jnp.sum(weighted, axis=0))[0]

def _make_universal_volume_kernel(domain, expr, value_shape):
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
        )
    return kernel


def _make_universal_surface_kernel(domain, expr, tag, value_shape):
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


def _build_feax_problem(domain, ir):
    import feax as fe

    volume_expr = ir.volume_expr
    boundary_exprs = ir.boundary_exprs

    if volume_expr is None and len(boundary_exprs) == 0:
        raise ValueError("No terms found for FEM assembly.")

    metadata = _infer_trial_metadata(volume_expr if volume_expr is not None else next(iter(boundary_exprs.values())))
    vec = int(metadata["vec"])
    value_shape = metadata["value_shape"]

    element_type = getattr(domain, "_fem_element_type", None)
    quad_degree = getattr(domain, "_fem_quad_degree", None)
    if element_type is None or quad_degree is None:
        solver_ctx = getattr(domain, "_jaxfem_solver_context", None)
        if solver_ctx is not None:
            element_type = solver_ctx.get("element_type", element_type)
            quad_degree = solver_ctx.get("quad_degree", quad_degree)
    if element_type is None:
        element_type = "TRI3"
    if quad_degree is None:
        quad_degree = 2

    mesh = _build_feax_mesh(domain, element_type)

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
        surface_kernels.append(_make_universal_surface_kernel(domain, expr, tag, value_shape))

    volume_kernel = None if volume_expr is None else _make_universal_volume_kernel(domain, volume_expr, value_shape)

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

    bc_specs = _make_feax_dirichlet_specs(domain, vec)
    bc = fe.DirichletBCConfig(bc_specs).create_bc(problem) if len(bc_specs) > 0 else fe.DirichletBCConfig([]).create_bc(problem)

    domain._feax_problem = problem
    domain._feax_bc = bc

    return problem, bc


# --------------------------------
# public FEAX-backed entry points
# --------------------------------

def _assemble_fem_residual_from_ir(domain, ir, **kwargs):
    import feax as fe

    problem, bc = _build_feax_problem(domain, ir)
    internal_vars = fe.InternalVars()

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = fe.create_J_bc_function(problem, bc, symmetric=kwargs.get("symmetric_bc", True))
    size = int(problem.num_total_dofs_all_vars)

    def residual_fn(u_flat):
        u_flat = jnp.asarray(u_flat, dtype=_default_float_dtype())
        return jnp.asarray(res_bc(u_flat, internal_vars))

    def jacobian_fn(u_flat):
        u_flat = jnp.asarray(u_flat, dtype=_default_float_dtype())
        return jac_bc(u_flat, internal_vars)

    return FemResidualOperator(
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        size=size,
    )


def _assemble_fem_system_from_ir(domain, ir, **kwargs):
    import feax as fe

    problem, bc = _build_feax_problem(domain, ir)
    internal_vars = fe.InternalVars()

    try:
        u0 = fe.zero_like_initial_guess(problem, bc)
    except Exception:
        u0 = jnp.zeros((problem.num_total_dofs_all_vars,), dtype=_default_float_dtype())

    u0 = jnp.asarray(u0, dtype=_default_float_dtype())

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = fe.create_J_bc_function(problem, bc, symmetric=kwargs.get("symmetric_bc", True))

    # FEAX gives the correction system:
    #     A du = -r(u0)
    # Convert to the full-state system expected by jNO examples:
    #     A u = A u0 - r(u0)
    A = jac_bc(u0, internal_vars)
    r0 = jnp.asarray(res_bc(u0, internal_vars), dtype=_default_float_dtype())

    if hasattr(A, "__matmul__"):
        b = A @ u0 - r0
    else:
        A_dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A.toarray())
        b = A_dense @ u0 - r0
        A = A_dense

    return A, b
