"""CORE solver using new tracing system - NO INNER VMAPS version."""

from typing import Dict, List, Callable, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import inspect

from .trace import (
    Placeholder,
    NewAxis,
    Reshape,
    Slice,
    Concat,
    FunctionCall,
    Literal,
    ConstantNamespace,
    Constant,
    Variable,
    TensorTag,
    BinaryOp,
    FlaxModule,
    TunableModule,
    TunableModuleCall,
    FlaxModuleCall,
    OperationDef,
    OperationCall,
    Hessian,
    Jacobian,
)
from .utils import get_logger
import equinox as eqx


# ============================================================
# Finite Difference helpers
# ============================================================


class DifferentialOperators:

    @staticmethod
    def compute_fd_gradient_2d_simple(u_values: jnp.ndarray, points: jnp.ndarray, triangles: np.ndarray, dim: int) -> jnp.ndarray:
        n_points = len(u_values)
        triangles = jnp.array(triangles)
        i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        p0, p1, p2 = points[i_idx], points[j_idx], points[k_idx]
        u0, u1, u2 = u_values[i_idx], u_values[j_idx], u_values[k_idx]
        dx1, dy1 = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        dx2, dy2 = p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1]
        areas = 0.5 * jnp.abs(dx1 * dy2 - dx2 * dy1)
        if dim == 0:
            grads = ((u1 - u0) * dy2 - (u2 - u0) * dy1) / (2 * areas + 1e-12)
        else:
            grads = ((u2 - u0) * dx1 - (u1 - u0) * dx2) / (2 * areas + 1e-12)
        grads = jnp.where(areas > 1e-12, grads, 0.0)
        area_weights = jnp.where(areas > 1e-12, areas, 0.0)
        contributions = grads * area_weights
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions).at[k_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(area_weights).at[j_idx].add(area_weights).at[k_idx].add(area_weights)
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_fd_gradient_3d_simple(u_values: jnp.ndarray, points: jnp.ndarray, tetrahedra: np.ndarray, dim: int) -> jnp.ndarray:
        n_points = len(u_values)
        tetrahedra = jnp.array(tetrahedra)
        i_idx, j_idx, k_idx, l_idx = (
            tetrahedra[:, 0],
            tetrahedra[:, 1],
            tetrahedra[:, 2],
            tetrahedra[:, 3],
        )
        p0, p1, p2, p3 = points[i_idx], points[j_idx], points[k_idx], points[l_idx]
        u0, u1, u2, u3 = (
            u_values[i_idx],
            u_values[j_idx],
            u_values[k_idx],
            u_values[l_idx],
        )
        v1, v2, v3 = p1 - p0, p2 - p0, p3 - p0
        volumes = jnp.abs(v1[:, 0] * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]) - v1[:, 1] * (v2[:, 0] * v3[:, 2] - v2[:, 2] * v3[:, 0]) + v1[:, 2] * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])) / 6.0
        if dim == 0:
            grads = ((u1 - u0) * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]) + (u2 - u0) * (v3[:, 1] * v1[:, 2] - v3[:, 2] * v1[:, 1]) + (u3 - u0) * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1])) / (6 * volumes + 1e-12)
        elif dim == 1:
            grads = ((u1 - u0) * (v2[:, 2] * v3[:, 0] - v2[:, 0] * v3[:, 2]) + (u2 - u0) * (v3[:, 2] * v1[:, 0] - v3[:, 0] * v1[:, 2]) + (u3 - u0) * (v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2])) / (6 * volumes + 1e-12)
        else:
            grads = ((u1 - u0) * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0]) + (u2 - u0) * (v3[:, 0] * v1[:, 1] - v3[:, 1] * v1[:, 0]) + (u3 - u0) * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])) / (6 * volumes + 1e-12)
        grads = jnp.where(volumes > 1e-12, grads, 0.0)
        volume_weights = jnp.where(volumes > 1e-12, volumes, 0.0)
        contributions = grads * volume_weights
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions).at[k_idx].add(contributions).at[l_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(volume_weights).at[j_idx].add(volume_weights).at[k_idx].add(volume_weights).at[l_idx].add(volume_weights)
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_fd_laplacian_1d_simple(u_values, points, lines):
        """
        Compute finite difference Laplacian on a 1D line mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 1) or (N,)
            lines: Line element connectivity, shape (M, 2)

        Returns:
            Laplacian (d²u/dx²) at each point, shape (N,)
        """
        # In 1D, Laplacian is just the second derivative: d²u/dx²
        grad = DifferentialOperators.compute_fd_gradient_1d_simple(u_values, points, lines)
        return DifferentialOperators.compute_fd_gradient_1d_simple(grad, points, lines)

    @staticmethod
    def compute_fd_laplacian_2d_simple(u_values, points, triangles, dims):
        result = jnp.zeros_like(u_values)
        for dim in dims:
            grad = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, dim)
            result = result + DifferentialOperators.compute_fd_gradient_2d_simple(grad, points, triangles, dim)
        return result

    @staticmethod
    def compute_fd_laplacian_3d_simple(u_values, points, tetrahedra, dims):
        result = jnp.zeros_like(u_values)
        for dim in dims:
            grad = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, dim)
            result = result + DifferentialOperators.compute_fd_gradient_3d_simple(grad, points, tetrahedra, dim)
        return result

    @staticmethod
    def compute_fd_hessian_2d_simple(u_values, points, triangles, var_dims):
        """
        Compute finite difference Hessian on a 2D triangular mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 2)
            triangles: Triangle connectivity, shape (M, 3)
            var_dims: List of tuples (i, vi_dim, j, vj_dim) specifying which
                      Hessian components to compute

        Returns:
            Hessian values at each point, shape (N, n_vars, n_vars)
        """
        N = points.shape[0]
        n_vars = int(jnp.sqrt(len(var_dims)))

        # First compute gradients
        grad_x = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, 0)
        grad_y = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, 1)

        # Then compute gradients of gradients (second derivatives)
        # d²u/dx² = d/dx(du/dx)
        d2u_dx2 = DifferentialOperators.compute_fd_gradient_2d_simple(grad_x, points, triangles, 0)
        # d²u/dxdy = d/dy(du/dx)
        d2u_dxdy = DifferentialOperators.compute_fd_gradient_2d_simple(grad_x, points, triangles, 1)
        # d²u/dy² = d/dy(du/dy)
        d2u_dy2 = DifferentialOperators.compute_fd_gradient_2d_simple(grad_y, points, triangles, 1)

        # Build full Hessian matrix for each point
        # hess_full[i, j, k] = d²u/dx_j dx_k at point i
        hess_full = jnp.zeros((N, 2, 2))
        hess_full = hess_full.at[:, 0, 0].set(d2u_dx2)
        hess_full = hess_full.at[:, 0, 1].set(d2u_dxdy)
        hess_full = hess_full.at[:, 1, 0].set(d2u_dxdy)  # Symmetry
        hess_full = hess_full.at[:, 1, 1].set(d2u_dy2)

        # Extract only the requested components
        result = jnp.zeros((N, n_vars, n_vars))
        for i, vi_dim, j, vj_dim in var_dims:
            result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])

        return result

    @staticmethod
    def compute_fd_hessian_3d_simple(u_values, points, tetrahedra, var_dims):
        """
        Compute finite difference Hessian on a 3D tetrahedral mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 3)
            tetrahedra: Tetrahedra connectivity, shape (M, 4)
            var_dims: List of tuples (i, vi_dim, j, vj_dim) specifying which
                      Hessian components to compute

        Returns:
            Hessian values at each point, shape (N, n_vars, n_vars)
        """
        N = points.shape[0]
        n_vars = int(jnp.sqrt(len(var_dims)))

        # First compute gradients
        grad_x = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 0)
        grad_y = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 1)
        grad_z = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 2)

        # Compute all second derivatives
        d2u_dx2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 0)
        d2u_dxdy = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 1)
        d2u_dxdz = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 2)
        d2u_dy2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_y, points, tetrahedra, 1)
        d2u_dydz = DifferentialOperators.compute_fd_gradient_3d_simple(grad_y, points, tetrahedra, 2)
        d2u_dz2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_z, points, tetrahedra, 2)

        # Build full Hessian matrix
        hess_full = jnp.zeros((N, 3, 3))
        hess_full = hess_full.at[:, 0, 0].set(d2u_dx2)
        hess_full = hess_full.at[:, 0, 1].set(d2u_dxdy)
        hess_full = hess_full.at[:, 0, 2].set(d2u_dxdz)
        hess_full = hess_full.at[:, 1, 0].set(d2u_dxdy)  # Symmetry
        hess_full = hess_full.at[:, 1, 1].set(d2u_dy2)
        hess_full = hess_full.at[:, 1, 2].set(d2u_dydz)
        hess_full = hess_full.at[:, 2, 0].set(d2u_dxdz)  # Symmetry
        hess_full = hess_full.at[:, 2, 1].set(d2u_dydz)  # Symmetry
        hess_full = hess_full.at[:, 2, 2].set(d2u_dz2)

        # Extract only the requested components
        result = jnp.zeros((N, n_vars, n_vars))
        for i, vi_dim, j, vj_dim in var_dims:
            result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])

        return result

    @staticmethod
    def compute_fd_gradient_1d_simple(u_values: jnp.ndarray, points: jnp.ndarray, lines: np.ndarray) -> jnp.ndarray:
        """
        Compute finite difference gradient on a 1D line mesh.

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 1) or (N,)
            lines: Line element connectivity, shape (M, 2)

        Returns:
            Gradient (du/dx) at each point, shape (N,)
        """
        n_points = len(u_values)
        lines = jnp.array(lines)

        # Get node indices for each line element
        i_idx, j_idx = lines[:, 0], lines[:, 1]

        # Get coordinates (handle both (N,1) and (N,) shapes)
        if points.ndim == 2:
            x0, x1 = points[i_idx, 0], points[j_idx, 0]
        else:
            x0, x1 = points[i_idx], points[j_idx]

        # Get function values at element nodes
        u0, u1 = u_values[i_idx], u_values[j_idx]

        # Compute element lengths
        dx = x1 - x0
        lengths = jnp.abs(dx)

        # Compute gradient on each element: du/dx = (u1 - u0) / (x1 - x0)
        grads = (u1 - u0) / (dx + 1e-12)

        # Zero out gradients for degenerate elements
        grads = jnp.where(lengths > 1e-12, grads, 0.0)
        length_weights = jnp.where(lengths > 1e-12, lengths, 0.0)

        # Weight contributions by element length
        contributions = grads * length_weights

        # Accumulate contributions to nodes (each element contributes to both its nodes)
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(length_weights).at[j_idx].add(length_weights)

        # Return weighted average
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_fd_hessian_1d_simple(u_values, points, lines, var_dims=None):
        """
        Compute finite difference Hessian on a 1D line mesh.

        In 1D, the Hessian is simply the second derivative d²u/dx².

        Args:
            u_values: Function values at mesh points, shape (N,)
            points: Mesh point coordinates, shape (N, 1) or (N,)
            lines: Line element connectivity, shape (M, 2)
            var_dims: Optional list of tuples (i, vi_dim, j, vj_dim) for compatibility.
                      In 1D, only (0, 0, 0, 0) is valid.

        Returns:
            Hessian values at each point, shape (N, 1, 1)
        """
        N = len(u_values)

        # Compute second derivative: d²u/dx²
        grad_x = DifferentialOperators.compute_fd_gradient_1d_simple(u_values, points, lines)
        d2u_dx2 = DifferentialOperators.compute_fd_gradient_1d_simple(grad_x, points, lines)

        # Build Hessian matrix (1x1 in 1D)
        hess_full = jnp.zeros((N, 1, 1))
        hess_full = hess_full.at[:, 0, 0].set(d2u_dx2)

        # If var_dims specified, extract requested components
        if var_dims is not None:
            n_vars = int(jnp.sqrt(len(var_dims)))
            result = jnp.zeros((N, n_vars, n_vars))
            for i, vi_dim, j, vj_dim in var_dims:
                # In 1D, vi_dim and vj_dim should both be 0
                result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])
            return result

        return hess_full


class TraceEvaluator:
    """Evaluates traced expressions - designed for JIT compilation.

    This version has NO inner vmaps. All operations are batched over N points.
    The outer vmap in core.py handles the batch dimension B.

    Shapes inside evaluate():
        context[tag]:  (N, D) for spatial points, (F,) or (1, F) for parameters

    All intermediate results should be (N,) or (N, K).

    Node handlers are registered via the ``_HANDLERS`` class-level dispatch
    table.  To add support for a new trace node type, define a method
    ``_eval_<NodeType>(self, expr, ctx)`` and add an entry in ``_HANDLERS``.
    """

    def __init__(self, params: Dict):
        self.params = params
        self.log = get_logger()
        self._logged_schemes = {}

    # ------------------------------------------------------------------
    # Evaluation context — lightweight carrier replacing 5 positional args
    # ------------------------------------------------------------------
    class _EvalCtx:
        """Bundles the read-only state that every handler needs."""

        __slots__ = ("context", "var_bindings", "key")

        def __init__(self, context, var_bindings, key):
            self.context = context
            self.var_bindings = var_bindings
            self.key = key

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        expr,
        context: Dict[str, jnp.ndarray] = None,
        var_bindings: Dict = None,
        key=None,
    ) -> jnp.ndarray:
        """Evaluate expression for a SINGLE batch (no batch dimension)."""
        ctx = self._EvalCtx(
            context=context or {},
            var_bindings=var_bindings or {},
            key=key,
        )
        return self._dispatch(expr, ctx)

    # Dispatch table — maps node type → handler method name.
    # ORDER MATTERS: more specific types (Constant, Literal) come first
    # so they aren't shadowed by their base class (Placeholder).
    _HANDLERS: List[tuple] = [
        (Constant, "_eval_constant"),
        (Literal, "_eval_literal"),
        (TensorTag, "_eval_tensor_tag"),
        (Variable, "_eval_variable"),
        (Concat, "_eval_concat"),
        (Reshape, "_eval_reshape"),
        (FunctionCall, "_eval_function_call"),
        (Slice, "_eval_slice"),
        (BinaryOp, "_eval_binary_op"),
        (OperationCall, "_eval_operation_call"),
        (FlaxModuleCall, "_eval_flax_module_call"),
        (TunableModule, "_eval_tunable_module"),
        (TunableModuleCall, "_eval_tunable_module_call"),
        (Jacobian, "_eval_jacobian"),
        (Hessian, "_eval_hessian"),
        (OperationDef, "_eval_operation_def"),
    ]

    def _dispatch(self, expr, ctx):
        """Look up handler in the dispatch table and call it."""
        for node_type, method_name in self._HANDLERS:
            if isinstance(expr, node_type):
                return getattr(self, method_name)(expr, ctx)
        raise ValueError(f"Cannot evaluate: {type(expr)}")

    # ------------------------------------------------------------------
    # Helpers shared by differential-operator handlers
    # ------------------------------------------------------------------
    def _build_local_context(self, idx, tag, points, context):
        """Build dynamically-sliced local context for a single point ``idx``.

        Used by AD-based Jacobian and Hessian handlers
        to construct per-point evaluation contexts.
        """
        local = {}
        for k, v in context.items():
            if v.ndim < 1:
                local[k] = v
            elif v.ndim == 1:
                if k == tag or v.shape[0] == points.shape[0]:
                    local[k] = jax.lax.dynamic_slice(v, (idx,), (1,))
                else:
                    local[k] = v
            else:
                # v.ndim >= 2
                if k == tag or v.shape[0] == points.shape[0]:
                    local[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                else:
                    local[k] = v

        return local

    def _map_mesh_to_sampled(self, mesh_points, sampled_points, values):
        """Map values computed at mesh vertices back to sampled points via
        nearest-neighbour lookup."""
        dists = jnp.sum(
            (mesh_points[jnp.newaxis, :, :] - sampled_points[:, jnp.newaxis, :]) ** 2,
            axis=-1,
        )
        vertex_indices = jnp.argmin(dists, axis=1)
        return values[vertex_indices]

    # ------------------------------------------------------------------
    # Individual node handlers
    # ------------------------------------------------------------------

    def _eval_constant(self, expr, ctx):
        return expr.value

    def _eval_literal(self, expr, ctx):
        return expr.value

    def _eval_tensor_tag(self, expr, ctx):
        if expr.tag not in ctx.context:
            raise ValueError(f"TensorTag '{expr.tag}' not found. Available:  {list(ctx.context.keys())}")
        tensor = jnp.asarray(ctx.context[expr.tag])
        if expr.dim_index is not None and tensor.ndim >= 1:
            tensor = tensor[..., expr.dim_index]
        return tensor

    def _eval_variable(self, expr, ctx):
        bound_var = ctx.var_bindings.get(id(expr), expr)
        tag = bound_var.tag

        if tag in ctx.context:
            tag_data = ctx.context[tag]
            dim_start, dim_end = bound_var.dim
            result = tag_data[..., dim_start:dim_end]
            if dim_end is None:
                pass  # keep full shape
            # elif result.ndim >= 1 and result.shape[-1] == 1:
            #    result = result[..., 0]
            return result

        else:
            self.log.error(f"Variable tag '{tag}' not found. context: {list(ctx.context.keys())}")
            raise KeyError(f"Variable tag '{tag}' not found in context")

    def _eval_concat(self, expr, ctx):
        items = [self._dispatch(item, ctx) for item in expr.items]
        items = [i[..., jnp.newaxis] if i.ndim == 1 else i for i in items]
        return jnp.concatenate(items, axis=-1)

    def _eval_reshape(self, expr, ctx):
        target = self._dispatch(expr.target, ctx)
        return target.reshape(expr.target_shape)

    def _eval_function_call(self, expr, ctx):
        args = [(self._dispatch(arg, ctx) if isinstance(arg, Placeholder) else arg) for arg in expr.args]
        sig = inspect.signature(expr.fn)
        if "key" in sig.parameters:
            return expr.fn(*args, key=ctx.key)
        else:
            return expr.fn(*args)

    def _eval_slice(self, expr, ctx):
        target = self._dispatch(expr.target, ctx)
        concrete_key = []
        for k in expr.key:
            if isinstance(k, NewAxis):
                concrete_key.append(None)
            else:
                concrete_key.append(k)
        result = target[tuple(concrete_key)]
        if result.ndim == 0:
            return result
        return result

    def _eval_binary_op(self, expr, ctx):
        left = self._dispatch(expr.left, ctx)
        right = self._dispatch(expr.right, ctx)
        _BINARY_FNS = {
            "+": jnp.add,
            "-": jnp.subtract,
            "*": jnp.multiply,
            "/": jnp.divide,
            "**": jnp.power,
        }
        res = _BINARY_FNS[expr.op](left, right)
        return res

    def _eval_operation_call(self, expr, ctx):
        op = expr.operation
        new_bindings = dict(ctx.var_bindings)
        op_vars = op._collected_vars

        for op_var, call_arg in zip(op_vars, expr.args):
            if isinstance(call_arg, Variable):
                bound_arg = ctx.var_bindings.get(id(call_arg), call_arg)
                new_bindings[id(op_var)] = bound_arg
            elif isinstance(call_arg, TensorTag):
                pass
            else:
                raise ValueError(f"Unsupported OperationCall argument type: {type(call_arg)}")

        new_ctx = self._EvalCtx(ctx.context, new_bindings, ctx.key)
        return self._dispatch(op.expr, new_ctx)

    def _eval_flax_module_call(self, expr, ctx):
        arg_values = []
        arg_sources = []

        for arg in expr.args:
            if isinstance(arg, (Placeholder, TensorTag)):
                val = self._dispatch(arg, ctx)
                arg_values.append(val)

                is_spatial = False
                if isinstance(arg, Variable) and arg.tag in ctx.context:
                    is_spatial = True

                arg_sources.append(is_spatial)
            else:
                arg_values.append(jnp.asarray(arg))
                arg_sources.append(False)

        flax_mod = expr.model
        model = self.params.get(flax_mod.layer_id)

        if model is None:
            raise ValueError(f"No model for FlaxModule {flax_mod.layer_id}")

        N = 1
        for val, is_spatial in zip(arg_values, arg_sources):
            if is_spatial:
                val = jnp.asarray(val)
                if val.ndim >= 1:
                    N = max(N, val.shape[0])

        def normalize_arg(val, is_spatial):
            val = jnp.asarray(val)
            if is_spatial:
                if val.ndim == 0:
                    return jnp.full((N, 1), val)
                elif val.ndim == 1:
                    return val[:, jnp.newaxis]
                else:
                    return val
            else:
                if val.ndim == 0:
                    return val[jnp.newaxis]
                else:
                    return val

        shaped_args = [normalize_arg(v, s) for v, s in zip(arg_values, arg_sources)]

        # Call equinox model directly (it IS the pytree, no init/apply split)
        import inspect

        sig = inspect.signature(model.__call__)
        if "key" in sig.parameters:
            result = model(*shaped_args, key=ctx.key)
        else:
            result = model(*shaped_args)

        # while result.ndim >= 2 and result.shape[-1] == 1:
        #    result = result[..., 0]

        return result

    def _eval_tunable_module(self, expr, ctx):
        if expr._current_instance is None:
            raise ValueError(f"TunableModule {expr} has no current instance.  " "This should be set by core.solve() before evaluation.")
        return self._dispatch(expr._current_instance, ctx)

    def _eval_tunable_module_call(self, expr, ctx):
        tunable = expr.model
        if tunable._current_instance is None:
            raise ValueError("TunableModule has no current instance. " "This should be set by core.solve() before evaluation.")
        concrete_call = FlaxModuleCall(tunable._current_instance, expr.args)
        concrete_call.op_id = expr.op_id
        return self._dispatch(concrete_call, ctx)

    def _eval_jacobian(self, expr, ctx):
        """Evaluate Jacobian (first-order derivatives).

        With a single variable this acts as a gradient and the result
        is squeezed to a scalar per point.
        """
        target = expr.target
        variables = expr.variables
        scheme = expr.scheme

        first_var = variables[0]
        bound_var = ctx.var_bindings.get(id(first_var), first_var)
        points = ctx.context[bound_var.tag]
        tag = bound_var.tag
        n_vars = len(variables)
        var_dims = [(i, vi.dim[0]) for i, vi in enumerate(variables)]

        # Ensure points is 2D (N, D) — after vmap it may be 1D (D,)
        if points.ndim == 1:
            points = points[jnp.newaxis, :]

        if scheme == "finite_difference":
            domain = bound_var._domain
            if domain is None or domain.mesh_connectivity is None:
                raise ValueError("FD scheme requires domain with mesh connectivity")
            mesh_points = jnp.array(domain.mesh_connectivity["points"])
            mesh_dim = domain.mesh_connectivity["dimension"]

            def u_at_pts(pts):
                ctx_dict = {**ctx.context, tag: pts}
                new_ctx = self._EvalCtx(ctx_dict, ctx.var_bindings, ctx.key)
                return self._dispatch(target, new_ctx)

            u_full = u_at_pts(mesh_points)

            jac_components = []
            for _i, vi_dim in var_dims:
                if mesh_dim == 1:
                    grad_full = DifferentialOperators.compute_fd_gradient_1d_simple(u_full, mesh_points, domain.mesh_connectivity["lines"])
                elif mesh_dim == 2:
                    grad_full = DifferentialOperators.compute_fd_gradient_2d_simple(u_full, mesh_points, domain.mesh_connectivity["triangles"], vi_dim)
                elif mesh_dim == 3:
                    grad_full = DifferentialOperators.compute_fd_gradient_3d_simple(u_full, mesh_points, domain.mesh_connectivity["tetrahedra"], vi_dim)
                jac_components.append(grad_full)

            if n_vars == 1:
                return self._map_mesh_to_sampled(mesh_points, points, jac_components[0])
            jac_full = jnp.stack(jac_components, axis=-1)
            return self._map_mesh_to_sampled(mesh_points, points, jac_full)

        elif scheme == "automatic_differentiation":
            evaluator_self = self

            if n_vars == 1:
                # Single-variable gradient: use jax.grad for efficiency
                dim = var_dims[0][1]

                def grad_single(idx):
                    pt = jax.lax.dynamic_slice(points, (idx, 0), (1, points.shape[1]))[0]
                    local_ctx = evaluator_self._build_local_context(idx, tag, points, ctx.context)

                    def u_scalar(p):
                        ctx_dict = {**local_ctx, tag: p[jnp.newaxis, ...]}
                        new_ctx = evaluator_self._EvalCtx(ctx_dict, ctx.var_bindings, ctx.key)
                        return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                    return jax.grad(u_scalar)(pt)[dim]

                return jax.vmap(grad_single)(jnp.arange(points.shape[0]))
            else:
                # Multi-variable Jacobian
                def jac_single(pt):
                    def u_fn(p):
                        new_ctx = evaluator_self._EvalCtx(
                            {**ctx.context, tag: p[jnp.newaxis, :]},
                            ctx.var_bindings,
                            ctx.key,
                        )
                        return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                    jac = jax.jacobian(u_fn)(pt)
                    result = jnp.zeros((n_vars,))
                    for i, vi_dim in var_dims:
                        result = result.at[i].set(jac[vi_dim])
                    return result

                return jax.vmap(jac_single)(points)

    def _eval_hessian(self, expr, ctx):
        """Evaluate Hessian (second-order derivatives).

        When ``expr.trace is True`` this computes the Laplacian
        (sum of diagonal Hessian entries) instead of the full matrix.
        """
        target = expr.target
        variables = expr.variables
        scheme = expr.scheme
        compute_trace = getattr(expr, "trace", False)

        first_var = variables[0]
        bound_var = ctx.var_bindings.get(id(first_var), first_var)

        points = None
        if bound_var.tag in ctx.context:
            points = ctx.context[bound_var.tag]
            # Ensure points is 2D (N, D) — after vmap it may be 1D (D,)
            if points.ndim == 1:
                points = points[jnp.newaxis, :]
            dims = tuple(v.dim[0] for v in variables)
        else:
            dims = tuple(0 for _ in variables)
        tag = bound_var.tag
        n = len(variables)
        var_dims = [(i, vi.dim[0], j, vj.dim[0]) for i, vi in enumerate(variables) for j, vj in enumerate(variables)]

        if scheme == "finite_difference":
            domain = bound_var._domain
            if domain is None or domain.mesh_connectivity is None:
                raise ValueError("FD scheme requires domain with mesh connectivity")
            mesh_points = jnp.array(domain.mesh_connectivity["points"])
            mesh_dim = domain.mesh_connectivity["dimension"]

            def u_at_pts(pts):
                ctx_dict = {**ctx.context, tag: pts}
                new_ctx = self._EvalCtx(ctx_dict, ctx.var_bindings, ctx.key)
                return self._dispatch(target, new_ctx)

            u_full = u_at_pts(mesh_points)

            if compute_trace:
                # Laplacian: sum of second derivatives on diagonal
                if mesh_dim == 1:
                    lap_full = DifferentialOperators.compute_fd_laplacian_1d_simple(u_full, mesh_points, domain.mesh_connectivity["lines"])
                elif mesh_dim == 2:
                    lap_full = DifferentialOperators.compute_fd_laplacian_2d_simple(u_full, mesh_points, domain.mesh_connectivity["triangles"], dims)
                elif mesh_dim == 3:
                    lap_full = DifferentialOperators.compute_fd_laplacian_3d_simple(u_full, mesh_points, domain.mesh_connectivity["tetrahedra"], dims)
                if points is not None:
                    return self._map_mesh_to_sampled(mesh_points, points, lap_full)
                return lap_full
            else:
                # Full Hessian matrix
                if mesh_dim == 1:
                    hess_full = DifferentialOperators.compute_fd_hessian_1d_simple(u_full, mesh_points, domain.mesh_connectivity["lines"])
                elif mesh_dim == 2:
                    hess_full = DifferentialOperators.compute_fd_hessian_2d_simple(u_full, mesh_points, domain.mesh_connectivity["triangles"], var_dims)
                elif mesh_dim == 3:
                    hess_full = DifferentialOperators.compute_fd_hessian_3d_simple(u_full, mesh_points, domain.mesh_connectivity["tetrahedra"], var_dims)
                return self._map_mesh_to_sampled(mesh_points, points, hess_full)

        elif scheme == "automatic_differentiation":
            evaluator_self = self

            if compute_trace:
                # Laplacian via AD
                def lap_single(idx):
                    pt = jax.lax.dynamic_slice(points, (idx, 0), (1, points.shape[1]))[0]
                    local_ctx = evaluator_self._build_local_context(idx, tag, points, ctx.context)

                    def u_scalar(p):
                        ctx_dict = {**local_ctx, tag: p[jnp.newaxis, :]}
                        new_ctx = evaluator_self._EvalCtx(ctx_dict, ctx.var_bindings, ctx.key)
                        return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                    hess = jax.hessian(u_scalar)(pt)
                    return sum(hess[d, d] for d in dims)

                return jax.vmap(lap_single)(jnp.arange(points.shape[0]))
            else:
                # Full Hessian via AD
                def hess_single(pt):
                    def u_scalar(p):
                        new_ctx = evaluator_self._EvalCtx(
                            {**ctx.context, tag: p[jnp.newaxis, :]},
                            ctx.var_bindings,
                            ctx.key,
                        )
                        return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                    hess = jax.hessian(u_scalar)(pt)
                    result = jnp.zeros((n, n))
                    for i, vi_dim, j, vj_dim in var_dims:
                        result = result.at[i, j].set(hess[vi_dim, vj_dim])
                    return result

                return jax.vmap(hess_single)(points)

    def _eval_operation_def(self, expr, ctx):
        return self._dispatch(expr.expr, ctx)

    @staticmethod
    def collect_dense_layers(expr: Placeholder) -> List:
        """Collect all FlaxModule nodes and their call arguments from expression tree.

        Traverses depth-first so that dependencies (modules whose outputs feed
        into other modules) are collected before the modules that consume them.

        Returns:
            List of ``(FlaxModule, call_args | None)`` tuples.
            ``call_args`` is ``None`` for standalone parameter modules.
        """
        layers = []
        seen = set()

        def visit(node):
            if isinstance(node, FlaxModule):
                if node.layer_id not in seen:
                    seen.add(node.layer_id)
                    layers.append((node, None))

            elif isinstance(node, TunableModule):
                if node._current_instance is not None:
                    inst = node._current_instance
                    if inst.layer_id not in seen:
                        seen.add(inst.layer_id)
                        layers.append((inst, None))

            elif isinstance(node, TunableModuleCall):
                # Visit args first (dependency order)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
                tunable = node.model
                if tunable._current_instance is not None:
                    flax_mod = tunable._current_instance
                    if flax_mod.layer_id not in seen:
                        seen.add(flax_mod.layer_id)
                        layers.append((flax_mod, node.args))

            elif isinstance(node, FlaxModuleCall):
                # Visit args first (dependency order)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
                flax_mod = node.model
                if flax_mod.layer_id not in seen:
                    seen.add(flax_mod.layer_id)
                    layers.append((flax_mod, node.args))

            elif isinstance(node, Concat):
                for item in node.items:
                    visit(item)
            elif isinstance(node, BinaryOp):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, OperationCall):
                visit(node.operation.expr)
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, (Hessian, Jacobian)):
                visit(node.target)
            elif isinstance(node, Slice):
                visit(node.target)
            elif isinstance(node, Reshape):
                visit(node.target)

        visit(expr)
        return layers

    @staticmethod
    def _infer_arg_shapes(
        call_args: List,
        tensor_dims: Dict[str, tuple],
        existing_params: Dict,
    ) -> List[tuple]:
        """Infer the *normalised* argument shapes for a FlaxModuleCall.

        Only needed for legacy Flax modules that require dummy inputs
        for ``module.init()``.  Equinox modules are constructed eagerly
        and never reach this code path.
        """
        abstract_ctx = {tag: jax.ShapeDtypeStruct(tuple(shape), jnp.float32) for tag, shape in tensor_dims.items()}

        def eval_and_normalize(context):
            evaluator = TraceEvaluator(existing_params)
            ctx = evaluator._EvalCtx(context, {}, jax.random.PRNGKey(0))

            arg_values = []
            arg_sources = []
            for arg in call_args:
                val = evaluator._dispatch(arg, ctx)
                arg_values.append(val)
                is_spatial = isinstance(arg, Variable) and arg.tag in context
                arg_sources.append(is_spatial)

            N = 1
            for val, is_spatial in zip(arg_values, arg_sources):
                if is_spatial:
                    val = jnp.asarray(val)
                    if val.ndim >= 1:
                        N = max(N, val.shape[0])

            def normalize_arg(val, is_spatial):
                val = jnp.asarray(val)
                if is_spatial:
                    if val.ndim == 0:
                        return jnp.full((N, 1), val)
                    elif val.ndim == 1:
                        return val[:, jnp.newaxis]
                    else:
                        return val
                else:
                    if val.ndim == 0:
                        return val[jnp.newaxis]
                    else:
                        return val

            return tuple(normalize_arg(v, s) for v, s in zip(arg_values, arg_sources))

        abstract_results = jax.eval_shape(eval_and_normalize, abstract_ctx)
        return [r.shape for r in abstract_results]

    @staticmethod
    def _cast_model_dtype(model, dtype, logger):
        """Cast all floating-point arrays in *model* to *dtype*.

        Works for both Equinox modules and ``FlaxModelWrapper`` (which
        wraps a Flax param dict).  Integer arrays are left unchanged.
        """

        def cast_leaf(x):
            if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(dtype)
            return x

        model = jax.tree_util.tree_map(cast_leaf, model)
        logger.info(f"Cast model parameters to {dtype}")
        return model

    @staticmethod
    def merge_pretrained_params(pretrained_params: dict, new_params: dict, logger) -> dict:
        """
        Merge pretrained weights with new params, replacing embedding/recovery layers
        when shapes don't match (for different channel dimensions).

        A concise summary (counts) is logged to the main logger.  Detailed
        per-parameter information is written to ``weight_merge.log`` in the
        same directory as the logger's output path.
        """
        stats = {"matched": 0, "replaced": 0}
        details: list = []  # collect per-param detail lines

        def count_params(arr):
            return arr.size if hasattr(arr, "size") else 0

        def merge(pretrained, new, path=""):
            if isinstance(pretrained, dict) and isinstance(new, dict):
                result = {}
                all_keys = set(list(pretrained.keys()) + list(new.keys()))

                for key in all_keys:
                    current_path = f"{path}/{key}" if path else key

                    if key in pretrained and key in new:
                        if isinstance(pretrained[key], dict):
                            result[key] = merge(pretrained[key], new[key], current_path)
                        else:
                            if pretrained[key].shape == new[key].shape:
                                result[key] = pretrained[key]
                                stats["matched"] += count_params(pretrained[key])
                                details.append(
                                    f"  MATCHED  {current_path}  "
                                    f"shape={pretrained[key].shape}  "
                                    f"params={count_params(pretrained[key]):,}"
                                )
                            else:
                                result[key] = new[key]
                                n = count_params(new[key])
                                stats["replaced"] += n
                                details.append(
                                    f"  MISMATCH {current_path}  "
                                    f"{pretrained[key].shape} -> {new[key].shape}  "
                                    f"params={n:,}  (reinitialized)"
                                )
                    elif key in pretrained:
                        result[key] = pretrained[key]
                        if not isinstance(pretrained[key], dict):
                            n = count_params(pretrained[key])
                            stats["matched"] += n
                            details.append(
                                f"  MATCHED  {current_path}  "
                                f"shape={pretrained[key].shape}  "
                                f"params={n:,}  (pretrained only)"
                            )
                    else:
                        result[key] = new[key]
                        if not isinstance(new[key], dict):
                            n = count_params(new[key])
                            stats["replaced"] += n
                            details.append(
                                f"  NEW      {current_path}  "
                                f"shape={new[key].shape}  "
                                f"params={n:,}  (new only)"
                            )

                return result
            else:
                if hasattr(pretrained, "shape") and hasattr(new, "shape"):
                    if pretrained.shape == new.shape:
                        stats["matched"] += count_params(pretrained)
                        return pretrained
                    else:
                        stats["replaced"] += count_params(new)
                        return new
                return new if new is not None else pretrained

        merged = merge(pretrained_params, new_params)

        total = stats["matched"] + stats["replaced"]
        pct = 100 * stats["matched"] / total if total else 0
        n_mismatch = sum(1 for d in details if "MISMATCH" in d)
        n_new = sum(1 for d in details if "NEW" in d)

        summary = (
            f"Pretrained weights: {stats['matched']:,}/{total:,} params matched "
            f"({pct:.4f}%), {stats['replaced']:,} reinitialized "
            f"({n_mismatch} shape mismatches, {n_new} new)"
        )
        logger.info(summary)

        # Write detailed per-parameter report to a text file next to log.txt
        from pathlib import Path
        log_dir = getattr(logger, "path", None)
        if log_dir is not None:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            detail_path = log_dir / "weight_merge.log"
            with open(detail_path, "w") as f:
                f.write(summary + "\n\n")
                f.write("\n".join(details) + "\n")

        return merged

    @staticmethod
    def build_single_layer_params(layer, arg_shapes, rng, logger):
        """Retrieve or construct the model for a single layer.

        The model was already fully constructed at factory time — we just
        return ``layer.module``, optionally loading pretrained weights.
        """
        if not isinstance(layer, FlaxModule):
            raise ValueError(f"Unknown layer type: {type(layer)}")

        module = layer.module

        # ---- Flax wrapper path (msgpack weights) --------------------
        from .architectures.common import FlaxModelWrapper

        if isinstance(module, FlaxModelWrapper) and layer.weight_path is not None:
            logger.info(f"Loading pretrained Flax weights from {layer.weight_path}")
            from flax.serialization import from_bytes

            with open(layer.weight_path, "rb") as f:
                pretrained_params = from_bytes(module.params, f.read())

            # Merge pretrained weights with fresh params
            merged = TraceEvaluator.merge_pretrained_params(
                pretrained_params,
                module.params,
                logger,
            )
            # Return a new FlaxModelWrapper with merged params
            model = FlaxModelWrapper(
                module.apply_fn,
                merged,
                post_fn=module.post_fn,
                **module.default_kwargs,
            )

            # ---- optional dtype cast --------------------------------
            if getattr(layer, "_dtype", None) is not None:
                model = TraceEvaluator._cast_model_dtype(model, layer._dtype, logger)

            if layer.show:
                leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
                total = sum(l.size for l in leaves)
                logger.info(f"  {type(model).__name__}: {total:,} parameters")

            return model

        # ---- Equinox path (normal) ----------------------------------
        if isinstance(module, eqx.Module):
            model = module

            if layer.weight_path is not None:
                logger.info(f"Loading pretrained weights from {layer.weight_path}")
                model = eqx.tree_deserialise_leaves(layer.weight_path, model)

            # ---- optional dtype cast --------------------------------
            if getattr(layer, "_dtype", None) is not None:
                model = TraceEvaluator._cast_model_dtype(model, layer._dtype, logger)

            if layer.show:
                leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
                total = sum(l.size for l in leaves)
                logger.info(f"  {type(model).__name__}: {total:,} parameters")

            return model

    @staticmethod
    def compile_traced_expression(expr: Placeholder, all_ops: List[OperationDef]) -> Callable:
        """Compile traced expression into a JAX-compatible function."""

        def evaluate_single_batch(params, context_single, key):
            """Evaluate for a single batch - no batch dimension."""
            evaluator = TraceEvaluator(params)
            return evaluator.evaluate(expr, context_single, {}, key)

        def compiled_fn(params, context=None, batchsize=None, key=None):
            """
            Evaluate the compiled expression.

            Args:
                params: Model parameters
                context: Unified dictionary of all input arrays by tag
                batchsize: If provided, randomly select this many samples from the batch dimension.
                        If None, use all samples.
                key: JAX random key for selecting random subset. Required if batchsize is provided.
            """
            context = context or {}

            # Collect all unique tags from all constraints to standardize ordering
            all_tags = set()
            for op in all_ops:
                if hasattr(op, "_collected_vars"):
                    for var in op._collected_vars:
                        all_tags.add(var.tag)

            tag_order = tuple(sorted(context.keys(), key=lambda t: (t not in all_tags, t)))
            ctx_tuple = tuple(context[tag] for tag in tag_order) if tag_order else ()

            # ============================================================
            # STEP 1: Determine the PRIMARY batch size B
            # First dimension is always batch — check all entries
            # ============================================================
            batched_sizes = []
            for arr in ctx_tuple:
                if hasattr(arr, "ndim") and arr.ndim >= 1:
                    batched_sizes.append(arr.shape[0])

            if not batched_sizes:
                # No batched data → no vmap needed
                return evaluate_single_batch(params, context, key=key)

            # Use the maximum as the primary batch size
            B = max(batched_sizes)

            # ============================================================
            # STEP 1.5: Handle random subset selection if batchsize is provided
            # ============================================================
            if batchsize is not None:
                if key is None:
                    raise ValueError("A JAX random key must be provided when batchsize is specified.")

                if batchsize > B:
                    print("WARNING: batchsize smaller then sampling -> replace=True")
                    indices = jax.random.choice(key, B, shape=(batchsize,), replace=True)
                    indices = jnp.sort(indices)

                if batchsize < B:
                    indices = jax.random.choice(key, B, shape=(batchsize,), replace=False)
                    indices = jnp.sort(indices)

                if batchsize == B:
                    indices = jnp.arange(0, B, 1)

                def subset_entry(arr):
                    if hasattr(arr, "ndim") and arr.ndim >= 1 and arr.shape[0] == B:
                        return arr[indices]
                    return arr

                ctx_tuple = tuple(subset_entry(a) for a in ctx_tuple)
                B = batchsize

            # ============================================================
            # STEP 2: Normalize — only vmap over arrays with batch size == B
            # ============================================================
            def normalize_entry(arg):
                if not hasattr(arg, "ndim") or arg.ndim == 0:
                    return arg, None

                bs = arg.shape[0]
                if bs == B:
                    return arg, 0
                elif bs == 1:
                    return jnp.squeeze(arg, axis=0), None
                else:
                    return arg, None

            new_ctx = []
            ctx_in_axes = []
            for a in ctx_tuple:
                a2, ax = normalize_entry(a)
                new_ctx.append(a2)
                ctx_in_axes.append(ax)
            ctx_tuple = tuple(new_ctx)
            ctx_in_axes = tuple(ctx_in_axes)

            # ============================================================
            # STEP 3: vmap — only over axes marked with 0
            # ============================================================
            def eval_single_batch_tuple(ctx_vals, rng_key):
                ctx_dict = dict(zip(tag_order, ctx_vals))
                return evaluate_single_batch(params, ctx_dict, key=rng_key)

            if key is not None:
                keys = jax.random.split(key, B)
                vmapped_fn = jax.vmap(
                    eval_single_batch_tuple,
                    in_axes=(ctx_in_axes, 0),
                )
                return vmapped_fn(ctx_tuple, keys)
            else:

                def eval_single_batch_tuple_no_key(ctx_vals):
                    ctx_dict = dict(zip(tag_order, ctx_vals))
                    return evaluate_single_batch(params, ctx_dict, key=None)

                vmapped_fn = jax.vmap(
                    eval_single_batch_tuple_no_key,
                    in_axes=(ctx_in_axes,),
                )
                return vmapped_fn(ctx_tuple)

        return compiled_fn

    @staticmethod
    def init_layer_params(
        all_ops: List,
        domain_dim: int,
        tensor_dims: Dict[str, int],
        rng: jax.Array,
        logger,
    ) -> Tuple[Dict, jax.Array]:
        """Collect / initialise models for all layers.

        For equinox modules (the normal path), the model was already
        constructed eagerly at factory time — we just store it directly
        (with optional pretrained weight loading).

        For legacy Flax modules (ScOT / Poseidon), we fall back to
        shape inference via ``jax.eval_shape`` and ``module.init``.

        Returns:
            all_models: Dict mapping layer_id -> callable model
            rng: Updated RNG key
        """
        all_models = {}
        seen = set()

        for op in all_ops:
            layers_with_args = TraceEvaluator.collect_dense_layers(op.expr)
            for layer, call_args in layers_with_args:
                if layer.layer_id in seen:
                    continue
                seen.add(layer.layer_id)

                rng, init_rng = jax.random.split(rng)

                # Fast path: equinox modules are already fully constructed
                if isinstance(layer.module, eqx.Module):
                    model = TraceEvaluator.build_single_layer_params(layer, None, init_rng, logger)
                else:
                    # Legacy Flax: need shape inference first
                    if call_args is not None:
                        arg_shapes = TraceEvaluator._infer_arg_shapes(call_args, tensor_dims, all_models)
                    else:
                        arg_shapes = None
                    model = TraceEvaluator.build_single_layer_params(layer, arg_shapes, init_rng, logger)

                all_models[layer.layer_id] = model

        return all_models, rng
