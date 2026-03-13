from __future__ import annotations

from typing import Any, Dict
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
            "fem_solver=True currently supports a single scalar TrialFunction only."
        )

    return {
        "expr": expr,
        "tag": tag,
        "has_trial": len(unique_trials) == 1,
        "trial": unique_trials[0] if unique_trials else None,
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
    cell_sol = cell_sol_flat.reshape(num_nodes, 1)
    shape_vals = domain._jaxfem_solver_context["dummy_problem"].fes[0].shape_vals

    local = {
        "physical_quad_points": physical_quad_points,
        "shape_vals": shape_vals,
        "shape_grads": cell_shape_grads,
        "cell_sol": cell_sol,
        "tag": compiled["tag"],
        "surface": False,
        "domain_context": domain.context,
    }

    val = _eval_expr_for_jaxfem(domain, compiled["expr"], local)
    return jax.flatten_util.ravel_pytree(jnp.sum(val * cell_JxW[0][:, None], axis=0))[0]

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
    cell_sol = cell_sol_flat.reshape(num_nodes, 1)

    local = {
        "physical_quad_points": physical_surface_quad_points,
        "shape_vals": face_shape_vals,
        "shape_grads": face_shape_grads,
        "cell_sol": cell_sol,
        "tag": compiled["tag"],
        "surface": True,
        "domain_context": domain.context,
    }

    val = _eval_expr_for_jaxfem(domain, compiled["expr"], local)
    weights = face_nanson_scale[0]

    return jax.flatten_util.ravel_pytree(jnp.sum(val * weights[:, None], axis=0))[0]

def _eval_expr_for_jaxfem(domain, node, local):

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
        return local["shape_vals"]

    if isinstance(node, TrialFunction):
        vals = local["shape_vals"]
        return jnp.sum(vals[:, :, None] * local["cell_sol"][None, :, :], axis=1)

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
            comps = [local["shape_grads"][..., dim0] for dim0 in dims]
            return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

        if isinstance(node.target, TrialFunction):
            grads = local["shape_grads"]
            comps = [
                jnp.sum(
                    grads[:, :, dim0:dim0 + 1] * local["cell_sol"][None, :, :],
                    axis=1,
                )
                for dim0 in dims
            ]
            return comps[0] if len(comps) == 1 else jnp.concatenate(comps, axis=-1)

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

    dirichlet_bc_info = domain._build_dirichlet_bc_info(
        solver_ctx["dirichlet_tags"],
        getattr(domain, "_fem_dirichlet_value_fns", None),
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
        vec=1,
        dim=solver_ctx["dim"],
        ele_type=solver_ctx["element_type"],
        gauss_order=solver_ctx["quad_degree"],
        dirichlet_bc_info=dirichlet_bc_info,
        location_fns=location_fns,
    )

    return problem, solver_ctx

def _flat_to_sol_list(problem, u_flat, dtype=jnp.float64):
    """Convert a flat DOF vector into JAX-FEM's expected [array(num_nodes,1)] format."""
    u_flat = jnp.asarray(u_flat, dtype=dtype)
    n = problem.fes[0].num_total_nodes
    if u_flat.ndim != 1 or u_flat.shape[0] != n:
        raise ValueError(f"Expected flat state of shape ({n},), got {u_flat.shape}")
    return [u_flat.reshape(n, 1)]

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
    n_dofs = problem.fes[0].num_total_nodes

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

    zero_sol = [jnp.zeros((problem.fes[0].num_total_nodes, 1), dtype=jnp.float64)]

    res_list = problem.newton_update(zero_sol)

    dofs0 = jax.flatten_util.ravel_pytree(zero_sol)[0]
    raw_res_vec = jax.flatten_util.ravel_pytree(res_list)[0]

    res_vec_bc = apply_bc_vec(raw_res_vec, dofs0, problem)
    A = get_A(problem)

    A_jax = _matrix_to_jax_bcoo(A)
    b_jax = jnp.asarray(-res_vec_bc)

    return A_jax, b_jax
# --------------------------------
# temporary legacy wrapper (delete later)
# --------------------------------
# def _fem_assemble_bucket(self, expr, support: str, region_id: str):
#     """Assemble a linear system contribution using JAX-FEM kernels.

#     Returns a FemLinearSystem which can be combined with + / - and
#     unpacked as A, b = ...
#     """
#     if not getattr(self, "_fem_solver_enabled", False):
#         raise ValueError("Call init_fem(..., fem_solver=True) to enable the FEM solver route.")

#     try:
#         from jax_fem.problem import Problem
#     except Exception as exc:
#         raise ImportError("jax_fem is required for fem_solver=True, but it could not be imported.") from exc

#     import numpy as onp
#     import jax
#     import jax.numpy as jnp

#     try:
#         from jax_fem.solver import get_A, apply_bc_vec
#     except Exception as exc:
#         raise ImportError(
#             "Could not import get_A/apply_bc_vec from jax_fem.solver."
#         ) from exc

#     if support == "volume":
#         tag = "fem_gauss"
#     elif support == "boundary":
#         tag = region_id
#     else:
#         raise ValueError(f"Unknown support '{support}'")

#     compiled = self._compile_weakform_for_jaxfem(expr, tag=tag)
#     solver_ctx = self._jaxfem_solver_context

#     active_surface_idx = None
#     if support == "boundary":
#         try:
#             active_surface_idx = solver_ctx["valid_neumann_tags"].index(tag)
#         except ValueError as exc:
#             raise ValueError(
#                 f"Unknown FEM boundary region '{tag}'. Known tags: {solver_ctx['valid_neumann_tags']}"
#             ) from exc

#     parent = self

#     class GeneratedProblem(Problem):
#         def get_universal_kernel(self_nonlocal):
#             if support != "volume":
#                 return lambda cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW: jnp.zeros_like(cell_sol_flat)

#             def kernel(cell_sol_flat, physical_quad_points, cell_shape_grads, cell_JxW, cell_v_grads_JxW):
#                 return parent._eval_compiled_volume_integrand(
#                     compiled,
#                     cell_sol_flat,
#                     physical_quad_points,
#                     cell_shape_grads,
#                     cell_JxW,
#                     cell_v_grads_JxW,
#                 )

#             return kernel

#         def get_universal_kernels_surface(self_nonlocal):
#             kernels = []
#             n_surfaces = len(solver_ctx["location_fns"])
#             for i in range(n_surfaces):
#                 if support == "boundary" and i == active_surface_idx:
#                     kernels.append(
#                         lambda cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, _i=i:
#                             parent._eval_compiled_surface_integrand(
#                                 compiled,
#                                 cell_sol_flat,
#                                 physical_surface_quad_points,
#                                 face_shape_vals,
#                                 face_shape_grads,
#                                 face_nanson_scale,
#                             )
#                     )
#                 else:
#                     kernels.append(
#                         lambda cell_sol_flat, physical_surface_quad_points, face_shape_vals, face_shape_grads, face_nanson_scale, _i=i:
#                             jnp.zeros_like(cell_sol_flat)
#                     )
#             return kernels

#     problem = GeneratedProblem(
#         solver_ctx["mesh"],
#         vec=1,
#         dim=solver_ctx["dim"],
#         ele_type=solver_ctx["element_type"],
#         gauss_order=solver_ctx["quad_degree"],
#         dirichlet_bc_info=solver_ctx["dirichlet_bc_info"],
#         location_fns=solver_ctx["location_fns"],
#     )
#     node_inds_list = getattr(problem.fes[0], "node_inds_list", None)
#     if node_inds_list is None:
#         print("Dirichlet node count: 0")
#     else:
#         total_dirichlet_nodes = sum(len(onp.asarray(inds).reshape(-1)) for inds in node_inds_list)
#         print("Dirichlet node count (raw, all groups):", total_dirichlet_nodes)
#     zero_sol = [jnp.zeros((problem.fes[0].num_total_nodes, 1), dtype=jnp.float64)]

#     # Assemble raw residual/tangent around zero state
#     res_list = problem.newton_update(zero_sol)

#     # Flatten dofs and residual in the same format JAX-FEM solver expects
#     dofs0 = jax.flatten_util.ravel_pytree(zero_sol)[0]
#     raw_res_vec = jax.flatten_util.ravel_pytree(res_list)[0]

#     # Let JAX-FEM apply its own Dirichlet row-elimination formulation
#     res_vec_bc = apply_bc_vec(raw_res_vec, dofs0, problem)
#     A = get_A(problem)

#     # For a linear problem at zero state:
#     #   A u + res_vec_bc = 0   =>   A u = -res_vec_bc
#     b = -onp.asarray(res_vec_bc)

#     # Normalize matrix type for downstream use with Lineax / SciPy conversions
#     if hasattr(A, "getValuesCSR"):
#         # PETSc-like matrix from jax_fem path
#         indptr, indices, data = A.getValuesCSR()
#         import scipy.sparse as sp
#         A = sp.csr_matrix((data, indices, indptr), shape=A.getSize())
#     else:
#         # SciPy path from jax_fem.get_A pure-scipy bypass
#         A = A.tocsr() if hasattr(A, "tocsr") else A

#     return FemLinearSystem(A, b)

# def fem_assemble(self, expr, tag: str = "fem_gauss"):
#     """
#     Temporary compatibility wrapper for old code.
#     """
#     if tag == "fem_gauss":
#         return self._fem_assemble_bucket(expr, support="volume", region_id="volume")
#     return self._fem_assemble_bucket(expr, support="boundary", region_id=tag)

