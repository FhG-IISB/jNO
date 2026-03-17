from __future__ import annotations
from dataclasses import dataclass

from typing import Any, Dict, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from .trace import (
    Literal,
    Constant,
    TensorTag,
    Variable,
    TestFunction,
    TrialFunction,
    Jacobian,
    BinaryOp,
    FunctionCall,
    FemLinearSystem,
)
from .weak_form import _sum_terms


import numpy as onp
import scipy.sparse as sp
from .weak_form import _contains_node_type

# --------------------------------
# FEM boundary-condition helpers
# --------------------------------
def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
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
    raise TypeError(
        f"Boundary tags must be a string or a sequence of strings, got {type(tags).__name__}."
    )


def dirichlet(tags, values=None):
    """
    Create a Dirichlet BC spec.

    Examples
    --------
    dirichlet("left")
    dirichlet("left", 0.0)
    dirichlet(["left", "right"], (0.0, 0.0))
    dirichlet("left", {"x": 0.0})
    """
    return DirichletBC(tags=_as_tags(tags), values=values)


def neumann(tags):
    """
    Create a Neumann boundary-region spec.

    Examples
    --------
    neumann("right")
    neumann(["top", "right"])
    """
    return NeumannBC(tags=_as_tags(tags))


def _const_bc_fn(value):
    value = float(value)
    return lambda p, c=value: c


def _normalize_dirichlet_value(value, vec: int):
    """
    Normalize the new BC value syntax into the legacy format already expected by
    domain._build_dirichlet_bc_info(...):

    scalar field (vec=1):
        callable

    vector field (vec>1):
        [fn0, fn1, ...]

    Supported inputs
    ----------------
    None
        -> zero on all components

    scalar
        -> broadcast constant to all components

    callable
        -> scalar: fn
           vector: [fn, fn, ..., fn]

    list/tuple
        -> explicit componentwise values/callables

    dict
        -> sparse componentwise specification with keys 0/1/2 or x/y/z
           unspecified components default to zero
    """
    if value is None:
        value = 0.0

    if vec < 1:
        raise ValueError(f"'vec' must be >= 1, got {vec}.")

    # callable
    if callable(value):
        if vec == 1:
            return value
        return [value for _ in range(vec)]

    # scalar constant
    if np.isscalar(value):
        fn = _const_bc_fn(value)
        if vec == 1:
            return fn
        return [fn for _ in range(vec)]

    # explicit componentwise list/tuple
    if isinstance(value, (list, tuple)):
        if len(value) != vec:
            raise ValueError(
                f"Dirichlet BC has {len(value)} entries, but vec={vec}."
            )

        out = []
        for v in value:
            if callable(v):
                out.append(v)
            elif np.isscalar(v):
                out.append(_const_bc_fn(v))
            else:
                raise TypeError(
                    "Dirichlet list/tuple entries must be callables or scalars."
                )

        if vec == 1:
            return out[0]
        return out

    # sparse dict by component
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
                raise TypeError(
                    "Dirichlet dict entries must be callables or scalars."
                )

        if vec == 1:
            return out[0]
        return out

    raise TypeError(
        f"Unsupported Dirichlet BC value type: {type(value).__name__}"
    )


def expand_bcs(bcs, vec: int):
    """
    Convert the new BC API into the legacy init_fem inputs:
        dirichlet_tags, dirichlet_value_fns, neumann_tags
    """
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
            raise TypeError(
                f"Unsupported BC entry '{type(bc).__name__}'. "
                "Use dirichlet(...) or neumann(...)."
            )

    return dirichlet_tags, dirichlet_value_fns, neumann_tags
# --------------------------------
# small expression-inspection helpers
# --------------------------------

def _strip_test_function_factor(domain, expr):
    """
    Extract the scalar coefficient from a product containing exactly one
    TestFunction factor.

    Examples handled:
    phi * g
    g * phi
    (-1) * (g * phi)
    (a * b) * phi
    phi * (a * b)

    Returns:
        coeff_expr  (with TestFunction removed)
        or None if expr is not a pure multiplicative term with exactly one TestFunction.
    """
  
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

def _is_simple_neumann_load(domain, expr):
    """
    True for a pure boundary load term scalar(x)*phi with:
    - contains TestFunction
    - no TrialFunction
    - no grad(phi)
    - no grad(u)
    """
    

    if not _contains_node_type(domain, expr, TestFunction):
        return False
    if _contains_node_type(domain, expr, TrialFunction):
        return False
    if _contains_node_type(domain, expr, Jacobian):
        return False

    coeff = _strip_test_function_factor(domain, expr)
    return coeff is not None

def _matrix_to_jax_bcoo(A):
    """
    Convert a PETSc-like or SciPy sparse matrix returned by jax_fem.get_A(...)
    into a pure-JAX BCOO sparse matrix.
    """
    import numpy as np
    import scipy.sparse as sp
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    # PETSc-like path
    if hasattr(A, "getValuesCSR"):
        indptr, indices, data = A.getValuesCSR()
        shape = A.getSize()
        A_csr = sp.csr_matrix((data, indices, indptr), shape=shape)

    # SciPy sparse path
    elif hasattr(A, "tocsr"):
        A_csr = A.tocsr()

    else:
        raise TypeError(
            f"Unsupported matrix type from get_A(problem): {type(A)}"
        )

    A_coo = A_csr.tocoo()

    data = jnp.asarray(A_coo.data)
    indices = jnp.stack(
        [
            jnp.asarray(A_coo.row, dtype=jnp.int32),
            jnp.asarray(A_coo.col, dtype=jnp.int32),
        ],
        axis=1,
    )

    return BCOO((data, indices), shape=A_coo.shape)

def _get_problem_vec(problem) -> int:
    """
    Robustly extract the field component count from a JAX-FEM Problem.
    """
    vec = getattr(problem, "vec", 1)

    if isinstance(vec, (list, tuple)):
        if len(vec) == 0:
            return 1
        return int(vec[0])

    return int(vec)
# --------------------------------
# jax-fem lowering helpers
# --------------------------------
def _make_native_surface_map_from_expr(domain, coeff_expr, tag):
    def _contains_normal_variable(node):
        if isinstance(node, Variable):
            return isinstance(node.tag, str) and node.tag.startswith("n_")
        if isinstance(node, (Literal, Constant, TestFunction, TrialFunction)):
            return False
        if isinstance(node, Jacobian):
            return _contains_normal_variable(node.target) or any(
                _contains_normal_variable(v) for v in node.variables
            )
        if isinstance(node, BinaryOp):
            return _contains_normal_variable(node.left) or _contains_normal_variable(node.right)
        if isinstance(node, FunctionCall):
            return any(_contains_normal_variable(arg) for arg in node.args)
        return False

    needs_normals = _contains_normal_variable(coeff_expr)

    normal_pts_jax = None
    normal_vals_jax = None

    if needs_normals:
        normal_lookup_tag = f"gauss_{tag}" if f"gauss_{tag}" in domain.normals_by_tag else tag

        if normal_lookup_tag in domain.normals_by_tag and normal_lookup_tag in domain._mesh_pool:
            normal_pts = np.asarray(domain._mesh_pool[normal_lookup_tag])[:, : domain.dimension]
            normal_vals = np.asarray(domain.normals_by_tag[normal_lookup_tag])[:, : domain.dimension]

            if len(normal_pts) > 0 and len(normal_pts) == len(normal_vals):
                normal_pts_jax = jnp.asarray(normal_pts)
                normal_vals_jax = jnp.asarray(normal_vals)

    def surface_map(u, x):
        boundary_normals = None

        if needs_normals and normal_pts_jax is not None:
            x_use = x[: domain.dimension]
            d2 = jnp.sum((normal_pts_jax - x_use[None, :]) ** 2, axis=1)
            idx = jnp.argmin(d2)
            boundary_normals = normal_vals_jax[idx:idx + 1]

        local = {
            "physical_quad_points": x[None, :],
            "shape_vals": jnp.ones((1, 1)),
            "shape_grads": jnp.zeros((1, 1, x.shape[0])),
            "cell_sol": jnp.zeros((1, 1)),
            "tag": tag,
            "surface": True,
            "boundary_normals": boundary_normals,
            "domain_context": domain.context,
        }

        coeff_val = _eval_expr_for_jaxfem(domain, coeff_expr, local)
        coeff_val = jnp.asarray(coeff_val).reshape(())
        return jnp.array([coeff_val])

    return surface_map

def _value_shape_num_components(value_shape) -> int:
    """
    Flatten a value_shape tuple into the number of components.

    Examples
    --------
    ()    -> 1
    (2,)  -> 2
    (3,)  -> 3
    (2,2) -> 4
    """
    if value_shape is None or len(value_shape) == 0:
        return 1

    n = 1
    for s in value_shape:
        n *= int(s)
    return n


def _reshape_components_last(arr, value_shape):
    """
    Reshape a trailing flattened component axis into value_shape.

    Parameters
    ----------
    arr : jax array
        Shape (..., n_comp)
    value_shape : tuple
        Desired trailing value shape.

    Returns
    -------
    jax array
        Shape (...,) + value_shape
    """
    if value_shape is None or len(value_shape) == 0:
        return arr

    return jnp.reshape(arr, arr.shape[:-1] + tuple(value_shape))


def _infer_trial_vec_from_compiled(compiled_expr):
    """
    Infer JAX-FEM vec from the compiled trial symbol metadata.
    """
    trial = compiled_expr.get("trial", None)
    if trial is None:
        return 1
    return _value_shape_num_components(getattr(trial, "value_shape", ()))


def _expand_test_shape_vals(shape_vals, n_comp):
    """
    Expand scalar FE basis values into vector-valued test basis values.

    Input
    -----
    shape_vals : (n_quad, n_local_nodes)
    n_comp     : int

    Output
    ------
    scalar case:
        (n_quad, n_local_nodes)

    vector case:
        (n_quad, n_local_nodes, n_comp, n_comp)

    Interpretation
    --------------
    The last two axes are:
      - basis component index
      - physical field component index

    Using an identity in those axes preserves the test-function component
    axis during tensor contractions.
    """
    if n_comp == 1:
        return shape_vals

    eye = jnp.eye(n_comp, dtype=shape_vals.dtype)          # (n_comp, n_comp)
    return shape_vals[:, :, None, None] * eye[None, None, :, :]

def _compile_weakform_for_jaxfem(domain, expr, tag: str = "fem_gauss") -> Dict[str, Any]:
    trial_nodes = {}

    def walk(node):
        if node is None:
            return

        if isinstance(node, TrialFunction):
            # Deduplicate by op_id so repeated uses of the same trial field
            # (e.g. grad(u,x) and grad(u,y)) count as one unknown.
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
            "fem_solver=True currently supports exactly one TrialFunction "
            "(scalar or vector valued). Multiple coupled FEM unknowns are not yet supported."
        )

    trial = unique_trials[0] if unique_trials else None
    value_shape = getattr(trial, "value_shape", ()) if trial is not None else ()
    vec = _value_shape_num_components(value_shape)

    return {
        "expr": expr,
        "tag": tag,
        "has_trial": len(unique_trials) == 1,
        "trial": trial,
        "value_shape": value_shape,
        "vec": vec,
    }

def _eval_compiled_volume_integrand(
    domain,
    compiled: Dict[str, Any],
    cell_sol_flat,
    physical_quad_points,
    cell_shape_grads,
    cell_JxW,
    cell_v_grads_JxW,
):
    num_nodes = cell_shape_grads.shape[1]
    vec = int(compiled.get("vec", 1))

    cell_sol = cell_sol_flat.reshape(num_nodes, vec)
    shape_vals = domain._jaxfem_solver_context["dummy_problem"].fes[0].shape_vals

    local = {
        "physical_quad_points": physical_quad_points,
        "shape_vals": shape_vals,
        "shape_grads": cell_shape_grads,
        "cell_sol": cell_sol,
        "tag": compiled["tag"],
        "surface": False,
        "domain_context": domain.context,
        "trial_value_shape": compiled.get("value_shape", ()),
        "trial_vec": vec,
    }

    val = _eval_expr_for_jaxfem(domain, compiled["expr"], local)
    weights = cell_JxW[0]
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return jax.flatten_util.ravel_pytree(jnp.sum(weighted, axis=0))[0]

def _eval_compiled_surface_integrand(
    domain,
    compiled: Dict[str, Any],
    cell_sol_flat,
    physical_surface_quad_points,
    face_shape_vals,
    face_shape_grads,
    face_nanson_scale,
):
    num_nodes = face_shape_vals.shape[1]
    vec = int(compiled.get("vec", 1))

    cell_sol = cell_sol_flat.reshape(num_nodes, vec)

    local = {
        "physical_quad_points": physical_surface_quad_points,
        "shape_vals": face_shape_vals,
        "shape_grads": face_shape_grads,
        "cell_sol": cell_sol,
        "tag": compiled["tag"],
        "surface": True,
        "domain_context": domain.context,
        "trial_value_shape": compiled.get("value_shape", ()),
        "trial_vec": vec,
    }

    val = _eval_expr_for_jaxfem(domain, compiled["expr"], local)
    weights = face_nanson_scale[0]
    wshape = (weights.shape[0],) + (1,) * (val.ndim - 1)
    weighted = val * weights.reshape(wshape)
    return jax.flatten_util.ravel_pytree(jnp.sum(weighted, axis=0))[0]

def _eval_expr_for_jaxfem(domain, node, local):

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
            raise KeyError(
                f"TensorTag '{node.tag}' not found in FEM domain context. "
                f"Available: {list(local['domain_context'].keys())}"
            )

        tensor = jnp.asarray(local["domain_context"][node.tag])

        # FEM assembly currently works on a single concrete domain instance.
        # Accept plain tensors (...) or a singleton batch (1, ...).
        if tensor.ndim >= 1 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.ndim >= 1 and tensor.shape[0] > 1:
            raise NotImplementedError(
                "fem_solver=True currently supports only singleton-batch TensorTag "
                f"coefficients. Got shape {tensor.shape} for tag '{node.tag}'."
            )

        if node.dim_index is not None and tensor.ndim >= 1:
            tensor = tensor[..., node.dim_index]
        return tensor

    if isinstance(node, Variable):
        # Optional boundary normals, e.g. tag == "n_gauss_wall"
        if isinstance(node.tag, str) and node.tag.startswith("n_"):
            if "boundary_normals" not in local or local["boundary_normals"] is None:
                raise ValueError(
                    f"Normal variable '{node.tag}' requested, but no boundary_normals "
                    f"were provided in the local FEM surface context."
                )
            dim0 = node.dim[0]
            return local["boundary_normals"][:, dim0:dim0 + 1]

        pts = local["physical_quad_points"]
        dim0 = node.dim[0]
        return pts[:, dim0:dim0 + 1]

    if isinstance(node, TestFunction):
        n_comp = _value_shape_num_components(getattr(node, "value_shape", ()))
        return _expand_test_shape_vals(local["shape_vals"], n_comp)

    if isinstance(node, TrialFunction):
        vals = local["shape_vals"]  # (n_quad, n_local_nodes)
        flat_interp = jnp.sum(vals[:, :, None] * local["cell_sol"][None, :, :], axis=1)  # (n_quad, vec)

        value_shape = getattr(node, "value_shape", ())
        if len(value_shape) == 0:
            return flat_interp

        return _reshape_components_last(flat_interp, value_shape)

    if isinstance(node, Jacobian):
        dims = []
        for var in node.variables:
            if not isinstance(var, Variable):
                raise NotImplementedError(
                    "fem_solver=True currently expects Jacobian variables to be "
                    "domain.variable(...) placeholders."
                )
            dims.append(var.dim[0])

        if len(dims) == 0:
            raise ValueError("Jacobian node has no differentiation variables")

        if isinstance(node.target, TestFunction):
            n_comp = _value_shape_num_components(getattr(node.target, "value_shape", ()))
            grads = local["shape_grads"]  # (n_quad, n_local_nodes, dim)

            # scalar test case
            if n_comp == 1:
                comps = [grads[..., dim0] for dim0 in dims]
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

            # vector-valued test case:
            # return shape with explicit basis-component axis preserved
            #
            # one derivative:
            #   (n_quad, n_local_nodes, n_comp, n_comp)
            #
            # many derivatives:
            #   (n_quad, n_local_nodes, n_comp, n_comp, n_dim_requested)
            #
            # axes are:
            #   quad, local_node, basis_comp, physical_comp, derivative_dim
            eye = jnp.eye(n_comp, dtype=grads.dtype)  # (n_comp, n_comp)

            comps = [
                grads[..., dim0][:, :, None, None] * eye[None, None, :, :]
                for dim0 in dims
            ]

            if len(comps) == 1:
                return comps[0]

            return jnp.stack(comps, axis=-1)

        if isinstance(node.target, TrialFunction):
            grads = local["shape_grads"]   # (n_quad, n_local_nodes, dim)
            cell_sol = local["cell_sol"]   # (n_local_nodes, vec)

            grad_list = [
                jnp.sum(
                    grads[:, :, dim0:dim0 + 1] * cell_sol[None, :, :],
                    axis=1,
                )  # (n_quad, vec)
                for dim0 in dims
            ]

            if len(dims) == 1:
                flat = grad_list[0]  # (n_quad, vec)
            else:
                flat = jnp.stack(grad_list, axis=-1)  # (n_quad, vec, n_dim_requested)

            value_shape = getattr(node.target, "value_shape", ())

            # scalar
            if len(value_shape) == 0:
                if len(dims) == 1:
                    return flat
                return flat

            # vector/tensor-valued unknown
            if len(dims) == 1:
                return _reshape_components_last(flat, value_shape)
            else:
                return jnp.reshape(
                    flat,
                    flat.shape[:1] + tuple(value_shape) + (len(dims),)
                )

        raise NotImplementedError(
            "fem_solver=True currently supports gradients of TrialFunction/TestFunction only."
        )

    if isinstance(node, BinaryOp):
        a = _eval_expr_for_jaxfem(domain, node.left, local)
        b = _eval_expr_for_jaxfem(domain, node.right, local)

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
        args = [_eval_expr_for_jaxfem(domain, arg, local) for arg in node.args]
        kwargs = node.kwargs if node.kwargs else {}
        return node.fn(*args, **kwargs)

    raise NotImplementedError(
        f"Unsupported weak-form node for fem_solver=True: {type(node).__name__}"
    )

def _build_grouped_problem(domain, volume_terms, boundary_terms):
    """
    Build one grouped JAX-FEM Problem from grouped weak-form terms.

    Returns
    -------
    problem : jax_fem.problem.Problem
    solver_ctx : dict
    """
    from jax_fem.problem import Problem

    solver_ctx = domain._jaxfem_solver_context

    vol_expr = _sum_terms(domain, volume_terms) if len(volume_terms) > 0 else None
    boundary_exprs = {
        region_id: _sum_terms(domain, terms)
        for region_id, terms in boundary_terms.items()
    }

    if vol_expr is None and len(boundary_exprs) == 0:
        raise ValueError("No terms found for FEM assembly.")

    compiled_volume_expr = None
    if vol_expr is not None:
        compiled_volume_expr = _compile_weakform_for_jaxfem(domain, vol_expr, tag="fem_gauss")

    active_boundary_tags = list(boundary_exprs.keys())
    # --------------------------------------------------
    # infer vec FIRST
    # --------------------------------------------------
    vec = 1
    if compiled_volume_expr is not None:
        vec = int(compiled_volume_expr.get("vec", 1))
    elif len(active_boundary_tags) > 0:
        first_tag = active_boundary_tags[0]
        trial_nodes = {}

        def walk_boundary(node):
            if node is None:
                return
            if isinstance(node, TrialFunction):
                trial_nodes[node.op_id] = node
                return
            for attr in ("left", "right", "target", "expr"):
                child = getattr(node, attr, None)
                if child is not None:
                    walk_boundary(child)
            for attr in ("args", "variables"):
                vals = getattr(node, attr, None)
                if vals is None:
                    continue
                for v in vals:
                    if isinstance(v, (list, tuple)):
                        for vv in v:
                            walk_boundary(vv)
                    else:
                        walk_boundary(v)

        walk_boundary(boundary_exprs[first_tag])
        if len(trial_nodes) == 1:
            only_trial = list(trial_nodes.values())[0]
            vec = _value_shape_num_components(getattr(only_trial, "value_shape", ()))
        else:
            vec = int(solver_ctx.get("default_vec", 1))
    else:
        vec = int(solver_ctx.get("default_vec", 1))
    native_surface_maps = []
    universal_surface_kernels = []

    for tag in active_boundary_tags:
        expr = boundary_exprs[tag]
        is_simple = _is_simple_neumann_load(domain, expr)

        if is_simple:
            coeff_expr = _strip_test_function_factor(domain, expr)
            native_surface_maps.append(_make_native_surface_map_from_expr(domain, coeff_expr, tag))

            def zero_surface_kernel(
                cell_sol_flat,
                physical_surface_quad_points,
                face_shape_vals,
                face_shape_grads,
                face_nanson_scale,
                *cell_internal_vars_surface,
            ):
                return jnp.zeros_like(cell_sol_flat)

            universal_surface_kernels.append(zero_surface_kernel)

        else:
            def zero_surface_map(u, x):
                return jnp.array([0.0])

            native_surface_maps.append(zero_surface_map)

            compiled_expr = _compile_weakform_for_jaxfem(domain, expr, tag=tag)

            def make_surface_kernel(compiled_expr_local):
                def kernel(
                    cell_sol_flat,
                    physical_surface_quad_points,
                    face_shape_vals,
                    face_shape_grads,
                    face_nanson_scale,
                    *cell_internal_vars_surface,
                ):
                    return _eval_compiled_surface_integrand(
                        domain,
                        compiled_expr_local,
                        cell_sol_flat,
                        physical_surface_quad_points,
                        face_shape_vals,
                        face_shape_grads,
                        face_nanson_scale,
                    )
                return kernel

            universal_surface_kernels.append(make_surface_kernel(compiled_expr))

     # --------------------------------------------------
    # build dirichlet BC info using inferred vec
    # --------------------------------------------------
    dirichlet_bc_info = domain._build_dirichlet_bc_info(
        solver_ctx["dirichlet_tags"],
        getattr(domain, "_fem_dirichlet_value_fns", None),
        vec=vec,
    )

    location_fns = []
    for tag in active_boundary_tags:
        loc_fn = domain._make_tag_location_fn(tag)
        if loc_fn is None:
            domain.log.warning(
                f"Boundary tag '{tag}' not found while building FEM surface locations. Skipping."
            )
            continue
        location_fns.append(loc_fn)

    class GeneratedProblem(Problem):
        def get_universal_kernel(self_inner):
            if compiled_volume_expr is None:
                return None

            def kernel(
                cell_sol_flat,
                physical_quad_points,
                cell_shape_grads,
                cell_JxW,
                cell_v_grads_JxW,
                *cell_internal_vars,
            ):
                return _eval_compiled_volume_integrand(
                    domain,
                    compiled_volume_expr,
                    cell_sol_flat,
                    physical_quad_points,
                    cell_shape_grads,
                    cell_JxW,
                    cell_v_grads_JxW,
                )

            return kernel

        def get_surface_maps(self_inner):
            return native_surface_maps

        def get_universal_kernels_surface(self_inner):
            return universal_surface_kernels


    problem = GeneratedProblem(
        solver_ctx["mesh"],
        vec=vec,
        dim=solver_ctx["dim"],
        ele_type=solver_ctx["element_type"],
        gauss_order=solver_ctx["quad_degree"],
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
    )

    return problem, solver_ctx

def _flat_to_sol_list(problem, u_flat, dtype=None):
    """
    Convert a flat DOF vector into JAX-FEM's expected [array(num_nodes, vec)] format.

    If dtype is None, preserve the dtype of u_flat / script-level JAX defaults.
    """
    u_flat = jnp.asarray(u_flat) if dtype is None else jnp.asarray(u_flat, dtype=dtype)
    n = problem.fes[0].num_total_nodes
    vec = _get_problem_vec(problem)
    expected = n * vec

    if u_flat.ndim != 1 or u_flat.shape[0] != expected:
        raise ValueError(f"Expected flat state of shape ({expected},), got {u_flat.shape}")

    return [u_flat.reshape(n, vec)]

# --------------------------------
# grouped fem assembly
# --------------------------------
def _assemble_fem_residual_grouped(domain, volume_terms, boundary_terms, **kwargs):
    """
    Build a nonlinear FEM residual operator R(u)=0 from grouped weak-form terms.
    """
    import jax
    import jax.numpy as jnp
    from .trace import FemResidualOperator

    try:
        from jax_fem.solver import apply_bc_vec, get_A
    except Exception as exc:
        raise ImportError(
            "jax_fem and jax_fem.solver.apply_bc_vec/get_A are required for fem_residual."
        ) from exc

    problem, solver_ctx = _build_grouped_problem(domain, volume_terms, boundary_terms)
    n_dofs = problem.fes[0].num_total_nodes * _get_problem_vec(problem)

    def residual_fn(u_flat):
        sol_list = _flat_to_sol_list(problem, u_flat)
        raw_res_list = problem.newton_update(sol_list)
        raw_res_vec = jax.flatten_util.ravel_pytree(raw_res_list)[0]
        dofs_vec = jax.flatten_util.ravel_pytree(sol_list)[0]
        res_vec_bc = apply_bc_vec(raw_res_vec, dofs_vec, problem)
        return jnp.asarray(res_vec_bc)

    def jacobian_fn(u_flat):
        sol_list = _flat_to_sol_list(problem, u_flat)
        _ = problem.newton_update(sol_list)
        A = get_A(problem)
        return _matrix_to_jax_bcoo(A)

    return FemResidualOperator(
        residual_fn=residual_fn,
        jacobian_fn=jacobian_fn,
        size=n_dofs,
    )


def _assemble_fem_system_grouped(domain, volume_terms, boundary_terms, **kwargs):
    """
    Linear grouped FEM assembly returning A, b.
    """
    import numpy as onp
    import jax
    import jax.numpy as jnp
    import scipy.sparse as sp

    try:
        from jax_fem.solver import get_A, apply_bc_vec
    except Exception as exc:
        raise ImportError(
            "jax_fem and jax_fem.solver.get_A/apply_bc_vec are required for fem_system."
        ) from exc

    problem, solver_ctx = _build_grouped_problem(domain, volume_terms, boundary_terms)

    vec = _get_problem_vec(problem)
    dtype = _default_float_dtype()
    zero_sol = [jnp.zeros((problem.fes[0].num_total_nodes, vec), dtype=dtype)]

    res_list = problem.newton_update(zero_sol)

    dofs0 = jax.flatten_util.ravel_pytree(zero_sol)[0]
    raw_res_vec = jax.flatten_util.ravel_pytree(res_list)[0]

    res_vec_bc = apply_bc_vec(raw_res_vec, dofs0, problem)
    A = get_A(problem)

    A_jax = _matrix_to_jax_bcoo(A)
    b_jax = jnp.asarray(-res_vec_bc)

    return A_jax, b_jax
