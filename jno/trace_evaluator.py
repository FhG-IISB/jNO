"""CORE solver using new tracing system - NO INNER VMAPS version."""

from typing import Dict, List, Callable, Tuple
import jax
import jax.numpy as jnp
import inspect

from .trace import (
    Placeholder,
    FunctionCall,
    Literal,
    ConstantNamespace,
    Constant,
    Variable,
    TensorTag,
    BinaryOp,
    Model,
    TunableModule,
    TunableModuleCall,
    ModelCall,
    OperationDef,
    OperationCall,
    Hessian,
    Jacobian,
    TestFunction, 
    Assembly,
)
from .utils import get_logger
import equinox as eqx


from .differential_operators import DifferentialOperators


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
        self._logged_schemes: Dict[str, str] = {}

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

    # ------------------------------------------------------------------
    # Shape tracing — walk the expression tree and record output shapes
    # ------------------------------------------------------------------
    def trace_shapes(
        self,
        expr,
        context: Dict[str, jnp.ndarray],
        var_bindings: Dict = None,
        key=None,
    ) -> str:
        """Return a human-readable tree showing the output shape at every node.

        This wraps :meth:`_dispatch` so that each handler's output
        shape is captured and printed alongside a concise node label.
        The tree is indented to reflect nesting.

        Typical usage (called from ``core._log_constraint_shapes``)::

            evaluator = TraceEvaluator(params)
            print(evaluator.trace_shapes(expr, ctx_dict))

        The output looks like::

            BinaryOp(-)                                  → (513, 1)
              Jacobian([Var(t)], fd)                      → (513, 1)
                BinaryOp(*)                               → (513, 1)
                  ModelCall(DeepONet)                 → (513, 1)
                    Variable(__time__[0:1])                → (1,)
                    Concat(axis=-1)                        → (513, 2)
                      Variable(interior[0:1])              → (513, 1)
                      Variable(interior[1:2])              → (513, 1)
                  Variable(interior[0:1])                  → (513, 1)
              BinaryOp(*)                                  → (513, 1)
                Literal(0.1)                               → scalar
                Laplacian([Var(x), Var(y)], fd)            → (513, 1)
                  ...
        """
        ctx = self._EvalCtx(
            context=context or {},
            var_bindings=var_bindings or {},
            key=key,
        )
        lines: list = []
        self._trace_visit(expr, ctx, depth=0, lines=lines, seen=set())
        return "\n".join(lines)

    def _trace_visit(self, node, ctx, depth, lines, seen):
        """Recursively visit *node*, evaluate it, and record its shape."""
        pad = "  " * depth
        uid, label = self._node_label(node)

        try:
            abstract = jax.eval_shape(lambda: self._dispatch(node, ctx))
            shape_str = str(abstract.shape) if hasattr(abstract, "shape") else "scalar"
        except Exception:
            shape_str = self._infer_shape_from_children(node, ctx)

        # Layout:
        #   #3fa2c1  │    BinaryOp(-)                  → (513, 1)
        #   ^uid^    ^indent + label^                  ^shape^
        tree_part = f"{pad}{label}"
        shape_col = max(60, len(tree_part) + 2)
        tree_part = tree_part.ljust(shape_col) + f"→ {shape_str}"

        # uid column is fixed 10 chars wide, separated by │
        entry = f"{uid}  │  {tree_part}"
        lines.append(entry)

        self._trace_children(node, ctx, depth, lines, seen)

    def _trace_children(self, node, ctx, depth, lines, seen):
        """Descend into the children of *node* for shape tracing."""
        if isinstance(node, BinaryOp):
            self._trace_visit(node.left, ctx, depth + 1, lines, seen)
            self._trace_visit(node.right, ctx, depth + 1, lines, seen)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    self._trace_visit(arg, ctx, depth + 1, lines, seen)
        elif isinstance(node, ModelCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    self._trace_visit(arg, ctx, depth + 1, lines, seen)
        elif isinstance(node, TunableModuleCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    self._trace_visit(arg, ctx, depth + 1, lines, seen)
        elif isinstance(node, OperationDef):
            if node.op_id not in seen:
                seen.add(node.op_id)
                self._trace_visit(node.expr, ctx, depth + 1, lines, seen)
        elif isinstance(node, OperationCall):
            # Build rebound context then trace the inner OperationDef
            self._trace_visit(node.operation, ctx, depth + 1, lines, seen)
        elif isinstance(node, (Jacobian, Hessian)):
            self._trace_visit(node.target, ctx, depth + 1, lines, seen)
        elif isinstance(node, Assembly):
            self._trace_visit(node.expr, ctx, depth + 1, lines, seen)
        # Leaf nodes (Variable, TensorTag, Constant, Literal) — no children

    def _infer_shape_from_children(self, node, ctx):
        """Best-effort shape inference when jax.eval_shape fails.

        Falls back to simple broadcast / passthrough rules based on the
        node type and its children's shapes.
        """

        def _child_shape(child):
            try:
                a = jax.eval_shape(lambda: self._dispatch(child, ctx))
                return a.shape if hasattr(a, "shape") else ()
            except Exception:
                return None

        if isinstance(node, BinaryOp):
            ls = _child_shape(node.left)
            rs = _child_shape(node.right)
            if ls is not None and rs is not None:
                try:
                    out = jnp.broadcast_shapes(ls, rs)
                    return str(out)
                except Exception:
                    return f"broadcast({ls}, {rs}) ??"
            return f"({ls} {node.op} {rs}) ??"

        if isinstance(node, (Jacobian, Hessian)):
            # Derivative output typically has the same leading shape as
            # the target expression.
            ts = _child_shape(node.target)
            if ts is not None:
                return f"~{ts}  (derivative)"
            return "??"
        if isinstance(node, Assembly):
            return f"({node.num_total_nodes},)"
        if isinstance(node, FunctionCall):
            # Reductions like .mse produce ()
            name = node._name or getattr(node.fn, "__name__", "")
            if name in ("mse", "mean", "sum", "max", "min"):
                return "()"
            # Element-wise functions keep input shape
            if node.args:
                cs = _child_shape(node.args[0])
                if cs is not None:
                    return str(cs)
            return "??"

        if isinstance(node, ModelCall):
            # Try to get the model output shape by actually running the
            # forward pass.  Model calls are cheap; it is only AD
            # derivatives that are expensive.
            try:
                result = self._dispatch(node, ctx)
                return str(result.shape) if hasattr(result, "shape") else "scalar"
            except Exception:
                return "??"

        if isinstance(node, OperationDef):
            cs = _child_shape(node.expr)
            if cs is not None:
                return str(cs)
            # Try running the whole OperationDef
            try:
                result = self._dispatch(node, ctx)
                return str(result.shape) if hasattr(result, "shape") else "()"
            except Exception:
                return "??"

        return "??"

    # Dispatch table — maps node type → handler method name.
    # ORDER MATTERS: more specific types (Constant, Literal) come first
    # so they aren't shadowed by their base class (Placeholder).
    _HANDLERS: List[tuple] = [
        (Constant, "_eval_constant"),
        (Literal, "_eval_literal"),
        (TensorTag, "_eval_tensor_tag"),
        (Variable, "_eval_variable"),
        (FunctionCall, "_eval_function_call"),
        (BinaryOp, "_eval_binary_op"),
        (OperationCall, "_eval_operation_call"),
        (ModelCall, "_eval_flax_module_call"),
        (TunableModule, "_eval_tunable_module"),
        (TunableModuleCall, "_eval_tunable_module_call"),
        (Jacobian, "_eval_jacobian"),
        (Hessian, "_eval_hessian"),
        (OperationDef, "_eval_operation_def"),
        (TestFunction, "_eval_test_function"), 
        (Assembly, "_eval_assembly"),
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
            # ---Safely pass through FEM dictionaries and scalars ---
            if isinstance(v, dict) or not hasattr(v, "ndim"):
                local[k] = v
                continue
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
                        # Handle N-dimensional arrays by padding start_indices and slice_sizes
                        start_indices = (idx,) + (0,) * (v.ndim - 1)
                        slice_sizes = (1,) + v.shape[1:]
                        local[k] = jax.lax.dynamic_slice(v, start_indices, slice_sizes)
                    else:
                        local[k] = v

        return local

    def _map_mesh_to_sampled(self, mesh_points, sampled_points, values):
        """Map values computed at mesh vertices back to sampled points via
        nearest-neighbour lookup.  If the point sets have the same size
        and are identical (common when n_samples == n_mesh), return
        values directly to avoid a costly O(N×M) distance matrix."""
        if mesh_points.shape == sampled_points.shape:
            # Fast path: when shapes match, check if points are identical
            # (this is the common case when all mesh points are sampled).
            same = jnp.all(jnp.abs(mesh_points - sampled_points) < 1e-8)
            return jax.lax.cond(
                same,
                lambda _: values,
                lambda _: self._nearest_neighbour_lookup(mesh_points, sampled_points, values),
                operand=None,
            )
        return self._nearest_neighbour_lookup(mesh_points, sampled_points, values)

    @staticmethod
    def _nearest_neighbour_lookup(mesh_points, sampled_points, values):
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
        axis = getattr(bound_var, "axis", "spatial")

        if tag in ctx.context:
            tag_data = ctx.context[tag]
            dim_start, dim_end = bound_var.dim
            result = tag_data[..., dim_start:dim_end]
            if dim_end is None:
                pass  # keep full shape
            return result
        elif axis == "temporal" and "__time__" in ctx.context:
            # Temporal variable reads from the shared time context.
            # After T-scan peels the T axis this is a scalar (1,).
            tag_data = ctx.context["__time__"]
            dim_start, dim_end = bound_var.dim
            result = tag_data[..., dim_start:dim_end]
            return result
        else:
            self.log.error(f"Variable tag '{tag}' not found. context: {list(ctx.context.keys())}")
            raise KeyError(f"Variable tag '{tag}' not found in context")

    def _eval_function_call(self, expr, ctx):
        args = [(self._dispatch(arg, ctx) if isinstance(arg, Placeholder) else arg) for arg in expr.args]
        kwargs = expr.kwargs if expr.kwargs else {}
        sig = inspect.signature(expr.fn)
        if "key" in sig.parameters:
            return expr.fn(*args, key=ctx.key, **kwargs)
        else:
            return expr.fn(*args, **kwargs)

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
                if isinstance(arg, Variable):
                    bound_arg = ctx.var_bindings.get(id(arg), arg)
                    axis = getattr(bound_arg, "axis", "spatial")
                    if axis == "spatial" and bound_arg.tag in ctx.context:
                        is_spatial = True
                    elif axis == "temporal":
                        # Temporal variable — will be broadcast to (N, 1)
                        is_spatial = False

                arg_sources.append(is_spatial)
            else:
                arg_values.append(jnp.asarray(arg))
                arg_sources.append(False)

        flax_mod = expr.model
        model = self.params.get(flax_mod.layer_id)

        if model is None:
            raise ValueError(f"No model for Model {flax_mod.layer_id}")

        def normalize_arg(val, is_spatial):
            """Minimal normalization: scalars → (1,), 1-D spatial → (N,1).

            No cross-argument broadcasting — that is the network's job.
            """
            val = jnp.asarray(val)
            if is_spatial:
                if val.ndim == 0:
                    return val.reshape(1, 1)
                elif val.ndim == 1:
                    return val[:, jnp.newaxis]  # (N,) → (N, 1)
                return val
            else:
                if val.ndim == 0:
                    return val.reshape(1)  # scalar → (1,)
                return val

        shaped_args = [normalize_arg(v, s) for v, s in zip(arg_values, arg_sources)]

        # Call equinox model directly (it IS the pytree, no init/apply split)
        import inspect

        sig = inspect.signature(model.__call__)
        if "key" in sig.parameters:
            result = model(*shaped_args, key=ctx.key)
        else:
            result = model(*shaped_args)

        # Ensure result is (N, 1) when model flattens to (N,).
        # Networks like DeepONet squeeze n_outputs=1 → (N,), but the
        # expression tree expects (N, 1) to match variable shapes.
        if result.ndim == 1 and result.shape[0] > 1:
            result = result[:, jnp.newaxis]

        return result

    def _eval_tunable_module(self, expr, ctx):
        if expr._current_instance is None:
            raise ValueError(f"TunableModule {expr} has no current instance.  " "This should be set by core.solve() before evaluation.")
        return self._dispatch(expr._current_instance, ctx)

    def _eval_tunable_module_call(self, expr, ctx):
        tunable = expr.model
        if tunable._current_instance is None:
            raise ValueError("TunableModule has no current instance. " "This should be set by core.solve() before evaluation.")
        concrete_call = ModelCall(tunable._current_instance, expr.args)
        concrete_call.op_id = expr.op_id
        return self._dispatch(concrete_call, ctx)

    def _eval_jacobian(self, expr, ctx):
        """Evaluate Jacobian (first-order derivatives).

        With a single variable this acts as a gradient and the result
        is squeezed to a scalar per point.

        Handles both spatial and temporal variables:
        - Spatial variables: differentiate w.r.t. columns of the spatial
          context ``(N, D_spatial)`` using either AD or FD.
        - Temporal variables: differentiate w.r.t. the scalar time value
          using AD (default) or central FD when scheme='finite_difference'.
        """
        target = expr.target
        variables = expr.variables
        scheme = expr.scheme
        if isinstance(target, TestFunction):
            var = variables[0]
            dim_idx = 0
            if hasattr(var, "dim") and isinstance(var.dim, (list, tuple)):
                ints = [d for d in var.dim if isinstance(d, int)]
                if ints:
                    dim_idx = ints[0]
            if target.tag == "fem_gauss":
                dN = ctx.context["dN_dx_flat"]
                
                #print(f"TRACING {target.tag} - dN native shape: {dN.shape}")
                #print(f"TRACING {target.tag} - Extracting dim {dim_idx}")
                return dN[..., dim_idx]

        first_var = variables[0]
        bound_var = ctx.var_bindings.get(id(first_var), first_var)
        first_axis = getattr(bound_var, "axis", "spatial")

        # ── Temporal derivative ──
        if first_axis == "temporal":
            evaluator_self = self
            time_key = "__time__"
            time_val = ctx.context.get(time_key)  # (1,)

            if scheme == "finite_difference":
                # Central difference: (u(t+eps) - u(t-eps)) / (2*eps)
                # Two forward passes through the network — much cheaper
                # than N jax.grad calls for AD.
                eps = jnp.float32(1e-3)
                t_fwd = time_val + eps
                t_bwd = time_val - eps

                ctx_fwd = {**ctx.context, time_key: t_fwd}
                ctx_bwd = {**ctx.context, time_key: t_bwd}

                u_fwd = self._dispatch(target, self._EvalCtx(ctx_fwd, ctx.var_bindings, ctx.key))
                u_bwd = self._dispatch(target, self._EvalCtx(ctx_bwd, ctx.var_bindings, ctx.key))

                result = (u_fwd - u_bwd) / (2.0 * eps)
                # Ensure (N, 1) shape
                if result.ndim == 1:
                    result = result[:, jnp.newaxis]
                return result

            # ── Temporal derivative via AD (default) ──
            # Find any spatial tag to determine N
            N = 1
            for k, v in ctx.context.items():
                if k != time_key and hasattr(v, "ndim") and v.ndim >= 1:
                    N = max(N, v.shape[0])

            def grad_time_single(idx):
                """Grad w.r.t. time for spatial point idx."""
                # Build local context with point idx sliced out of spatial arrays
                local_ctx = {}
                for k, v in ctx.context.items():
                    if k == time_key:
                        continue  # will be replaced by differentiable t
                    if hasattr(v, "ndim") and v.ndim >= 2 and v.shape[0] == N:
                        local_ctx[k] = jax.lax.dynamic_slice(v, (idx, 0), (1, v.shape[1]))
                    elif hasattr(v, "ndim") and v.ndim == 1 and v.shape[0] == N:
                        local_ctx[k] = jax.lax.dynamic_slice(v, (idx,), (1,))
                    else:
                        local_ctx[k] = v

                def u_of_t(t_arr):
                    new_ctx_dict = {**local_ctx, time_key: t_arr}
                    new_ctx = evaluator_self._EvalCtx(new_ctx_dict, ctx.var_bindings, ctx.key)
                    return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                return jax.grad(u_of_t)(time_val)[0]

            result = jax.vmap(grad_time_single)(jnp.arange(N))
            # Ensure (N,) → (N, 1) to match variable / model-output shapes
            return result[:, jnp.newaxis]

        # ── Spatial derivative ──
        tag = bound_var.tag
        points = ctx.context[bound_var.tag]
        n_vars = len(variables)
        var_dims = [(i, vi.dim[0]) for i, vi in enumerate(variables)]

        # Ensure points is 2D (N, D) — after vmap it may be 1D (D,)
        if points.ndim == 1:
            points = points[jnp.newaxis, :]

        if scheme.startswith("finite_difference"):
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
            N_mesh = mesh_points.shape[0]

            # FD operators expect flat 1-D (N,) values.  Operator-learning
            # models (Poseidon, FNO, …) return image-shaped tensors whose
            # total size equals N_mesh.  Auto-flatten them and remember the
            # original shape so we can restore it afterwards.
            u_squeezed = u_full.squeeze(-1) if (u_full.ndim > 1 and u_full.shape[-1] == 1) else u_full
            image_shape = None
            if u_squeezed.ndim > 1 and u_squeezed.size == N_mesh:
                image_shape = u_full.shape
                u_full_1d = u_squeezed.ravel()
            else:
                u_full_1d = u_squeezed if u_squeezed.ndim == 1 else u_squeezed.ravel()

            _, grad_method, _lap_method = DifferentialOperators.parse_fd_scheme(scheme)
            jac_components = []
            for _i, vi_dim in var_dims:
                if mesh_dim == 1:
                    grad_full = DifferentialOperators.compute_fd_gradient_1d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["lines"], method=grad_method)
                elif mesh_dim == 2:
                    grad_full = DifferentialOperators.compute_fd_gradient_2d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["triangles"], vi_dim, method=grad_method)
                elif mesh_dim == 3:
                    grad_full = DifferentialOperators.compute_fd_gradient_3d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["tetrahedra"], vi_dim, method=grad_method)
                jac_components.append(grad_full)

            if image_shape is not None:
                # Return in the same image shape as the model output
                if n_vars == 1:
                    return jac_components[0].reshape(image_shape)
                return jnp.stack(jac_components, axis=-1).reshape(*image_shape[:-1], n_vars)

            if n_vars == 1:
                result = self._map_mesh_to_sampled(mesh_points, points, jac_components[0])
                return result[:, jnp.newaxis] if result.ndim == 1 else result
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

                return jax.vmap(grad_single)(jnp.arange(points.shape[0]))[:, jnp.newaxis]
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

        Hessian/Laplacian is expected to be purely spatial.  After the
        T-scan peels the time axis, context entries are ``(N, D_spatial)``
        and ``__time__`` is ``(1,)`` — the FD path can now safely
        replace the spatial context with the full mesh points without
        losing the time value.
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

        if scheme.startswith("finite_difference"):
            domain = bound_var._domain
            if domain is None or domain.mesh_connectivity is None:
                raise ValueError("FD scheme requires domain with mesh connectivity")
            mesh_points = jnp.array(domain.mesh_connectivity["points"])
            mesh_dim = domain.mesh_connectivity["dimension"]

            def u_at_pts(pts):
                # Replace only the spatial tag — __time__ stays intact
                ctx_dict = {**ctx.context, tag: pts}
                new_ctx = self._EvalCtx(ctx_dict, ctx.var_bindings, ctx.key)
                return self._dispatch(target, new_ctx)

            u_full = u_at_pts(mesh_points)
            N_mesh = mesh_points.shape[0]

            # Auto-flatten image-shaped model outputs to (N_mesh,).
            # See the Jacobian FD path for the same logic.
            u_squeezed = u_full.squeeze(-1) if (u_full.ndim > 1 and u_full.shape[-1] == 1) else u_full
            image_shape = None
            if u_squeezed.ndim > 1 and u_squeezed.size == N_mesh:
                image_shape = u_full.shape
                u_full_1d = u_squeezed.ravel()
            else:
                u_full_1d = u_squeezed if u_squeezed.ndim == 1 else u_squeezed.ravel()

            _main, _grad_method, lap_method = DifferentialOperators.parse_fd_scheme(scheme)

            if compute_trace:
                # Laplacian: sum of second derivatives on diagonal
                if mesh_dim == 1:
                    lap_full = DifferentialOperators.compute_fd_laplacian_1d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["lines"], method=lap_method)
                elif mesh_dim == 2:
                    lap_full = DifferentialOperators.compute_fd_laplacian_2d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["triangles"], dims, method=lap_method)
                elif mesh_dim == 3:
                    lap_full = DifferentialOperators.compute_fd_laplacian_3d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["tetrahedra"], dims, method=lap_method)

                if image_shape is not None:
                    # Return in the same image shape as the model output
                    return lap_full.reshape(image_shape)

                if points is not None:
                    result = self._map_mesh_to_sampled(mesh_points, points, lap_full)
                    return result[:, jnp.newaxis] if result.ndim == 1 else result
                return lap_full[:, jnp.newaxis] if lap_full.ndim == 1 else lap_full
            else:
                # Full Hessian matrix
                if mesh_dim == 1:
                    hess_full = DifferentialOperators.compute_fd_hessian_1d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["lines"])
                elif mesh_dim == 2:
                    hess_full = DifferentialOperators.compute_fd_hessian_2d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["triangles"], var_dims)
                elif mesh_dim == 3:
                    hess_full = DifferentialOperators.compute_fd_hessian_3d_simple(u_full_1d, mesh_points, domain.mesh_connectivity["tetrahedra"], var_dims)
                if image_shape is not None:
                    return hess_full.reshape(*image_shape[:-1], n, n)
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

                return jax.vmap(lap_single)(jnp.arange(points.shape[0]))[:, jnp.newaxis]
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

    def _eval_test_function(self, expr, ctx):
        """Returns the precomputed shape function values for volume or surface."""
        if expr.tag == "fem_gauss":
            # Volume shape functions: (N_quads_total, 3)
            return ctx.context["N_flat"]
        else:
            if "surface_data" not in ctx.context or expr.tag not in ctx.context["surface_data"]:
                raise KeyError(f"Surface tag '{expr.tag}' not found in fem_context.")
            
            vals = ctx.context["surface_data"][expr.tag]["face_shape_vals"]
            # reshape(-1, ...) is safe whether vals is (F, Q, 3) or (1, F, Q, 3)
            return vals.reshape(-1, vals.shape[-1])

    def _eval_assembly(self, expr, ctx):
        integrand = self._dispatch(expr.expr, ctx)
        
        if expr.tag == "fem_gauss":
            weights = ctx.context["JxW"] 
            flat_cells = ctx.context["flat_cells"].flatten()
        else:
            surf_data = ctx.context["surface_data"][expr.tag]
            weights = surf_data["nanson_scale"]
            flat_cells = surf_data["flat_parent_nodes"].flatten()

        num_entities, num_quads = weights.shape[-2], weights.shape[-1]
        weights_flat = weights.flatten()[:, jnp.newaxis]

        local_residuals = integrand * weights_flat
        num_local_nodes = integrand.shape[-1]
        
        # This reshape is the ultimate safeguard. If num_local_nodes is 
        # wrong (like the Jacobian bug), this will CRASH instead of cheating.
        local_residuals = local_residuals.reshape(num_entities, num_quads, num_local_nodes)
        cell_residuals = jnp.sum(local_residuals, axis=1) 
        
        global_residual = jax.ops.segment_sum(
            cell_residuals.flatten(), 
            flat_cells, 
            num_segments=expr.num_total_nodes
        )
        if global_residual.ndim == 1:
            global_residual = global_residual[:, jnp.newaxis]

        # NORMALIZATION FIX
        if "global_areas" in ctx.context:
            areas = ctx.context["global_areas"].reshape(-1,1)
            global_residual = global_residual / (areas + 1e-12)

        # GALERKIN FIX: If using Hard BCs, zeroing residuals is optional but 
        # standard. If you want maximum accuracy, try commenting this out!
        if "dirichlet_nodes" in ctx.context:
            d_nodes = jnp.asarray(ctx.context["dirichlet_nodes"]).flatten().astype(jnp.int32)
            if d_nodes.size > 0:
                global_residual = global_residual.at[d_nodes].set(0.0)

        if global_residual.ndim == 1:
            global_residual = global_residual[:, jnp.newaxis]
        # jax.debug.print(
        #     "[{tag}] Max: {max_val:.2e} | Mean abs: {mean_val:.2e} | Non-zero nodes: {nnz}",
        #     tag=getattr(expr, "tag", "unknown_tag"),
        #     max_val=jnp.max(jnp.abs(global_residual)),
        #     mean_val=jnp.mean(jnp.abs(global_residual)),
        #     nnz=jnp.sum(jnp.abs(global_residual) > 1e-8)
        # )

        return global_residual
   
    @staticmethod
    def _node_label(node) -> Tuple[str, str]:
        """Return (uid, label) — rendered separately by _trace_visit."""
        uid = f"#{id(node) % 0xFFFFFF:06x}"
        if isinstance(node, Variable):
            tag = node.tag
            dim = node.dim
            axis = getattr(node, "axis", "spatial")
            axis_str = f", {axis}" if axis != "spatial" else ""
            return uid, f"Variable({tag}[{dim[0]}:{dim[1]}]{axis_str})"
        if isinstance(node, TensorTag):
            return uid, f"TensorTag({node.tag})"
        if isinstance(node, Constant):
            val = node.value
            if hasattr(val, "shape") and val.shape == ():
                val = float(val)
            return uid, f"Constant({node.tag}.{node.key}={val})"
        if isinstance(node, Literal):
            v = node.value
            if hasattr(v, "shape"):
                v = float(v) if v.shape == () else v.shape
            return uid, f"Literal({v})"
        if isinstance(node, BinaryOp):
            return uid, f"BinaryOp({node.op})"
        if isinstance(node, FunctionCall):
            name = node._name or getattr(node.fn, "__name__", "fn")
            return uid, f"FunctionCall({name})"
        if isinstance(node, ModelCall):
            mod = node.model
            mod_name = type(mod.module).__name__ if hasattr(mod, "module") else str(mod)
            lid = getattr(mod, "layer_id", "?")
            return uid, f"ModelCall({mod_name}, layer={lid})"
        if isinstance(node, TunableModuleCall):
            return uid, f"TunableModuleCall(id={node.model.layer_id})"
        if isinstance(node, OperationDef):
            vars_str = ", ".join(str(v) for v in node._collected_vars)
            return uid, f"OperationDef[{node.op_id}]({vars_str})"
        if isinstance(node, OperationCall):
            return uid, f"OperationCall[{node.operation.op_id}]"
        if isinstance(node, Jacobian):
            vars_str = ", ".join(str(v) for v in node.variables)
            scheme_str = f", {node.scheme[:2]}" if node.scheme else ""
            return uid, f"Jacobian([{vars_str}]{scheme_str})"
        if isinstance(node, Hessian):
            kind = "Laplacian" if node.trace else "Hessian"
            vars_str = ", ".join(str(v) for v in node.variables)
            scheme_str = f", {node.scheme[:2]}" if node.scheme else ""
            return uid, f"{kind}([{vars_str}]{scheme_str})"
        return uid, type(node).__name__
    
    