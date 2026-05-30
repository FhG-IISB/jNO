"""PDEformer-2 bridge: auto-build the PDE computational graph from jNO traces.

When a user writes a normal jNO PINN program that uses a ``foundax.pdeformer2``
model as the backbone, ``jno.core`` calls ``maybe_attach_pdeformer2_graphs``
automatically. This module:

1. Walks the loss expression trees to find ``ModelCall`` nodes whose underlying
   ``eqx.Module`` is a ``jax_pdeformer2.PDEformer``.
2. Translates the symbolic PDE residual into a PDEformer-2 DAG (uf, dt, dx, dy,
   add, mul, neg, coef, ic, eq0, Mod*, Branch*, ...).
3. Evaluates the IC residual ``ini = u0 - f(x0, y0)`` at a sampled grid to
   produce the function-node values.
4. Wraps the ``PDEformer`` instance in ``PDEformer2Wrapper`` which carries the
   static graph tensors and exposes a normal ``(t, x, y) -> u`` interface.

The user writes zero PDEformer-specific code — detection is purely
``isinstance(model.module, PDEformer)``.

Constraint classification
-------------------------
Each loss term is routed by the variable tag of its spatial Variables:

  * ``interior``       → PDE residual, walked into the DAG
  * ``initial``        → IC residual, RHS sampled on a grid as function values
  * any other tag      → treated as a soft constraint and **excluded** from
                          the DAG (the standard pattern for boundary terms)

Why hard BC ansätze are NOT supported
-------------------------------------
The PINN trick ``u = net(t,x,y) * x(1-x)y(1-y)`` cannot be used with a
PDEformer-2 backbone, for two reasons:

  1. Expanding ``∇²(NN · ansatz)`` via the product rule introduces raw
     ``Variable(x), Variable(y)`` nodes, which are outside PDEformer-2's
     operator vocabulary {add, mul, neg, square, dt, dx, dy, sin, cos,
     exp10, log10}.
  2. Even if (1) were patched (e.g. by hiding the ansatz inside the
     wrapper), the trainer would force the foundation model to learn
     ``NN ≈ u / ansatz`` instead of the ``u`` it was pre-trained on,
     largely throwing away the pre-trained warm start.

The recommended pattern is therefore soft BCs (a separate ``boundary`` tag
plus ``ub.mse``), which match the foundation model's pre-training
distribution and let the encoder receive the canonical PDE graph.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..trace import (
    BinaryOp,
    FunctionCall,
    Hessian,
    Jacobian,
    Literal,
    Model,
    ModelCall,
    Placeholder,
    Variable,
)

# ---------------------------------------------------------------------------
# Node type indices — taken verbatim from functoreality/pdeformer-2/src/data/pde_dag.py
# ---------------------------------------------------------------------------

VAR_NODE_TYPES = ["uf"]
COEF_NODE_TYPES = ["coef"]
FUNCTION_NODE_TYPES = ["ic", "cf", "bv", "sdf", "eval"]
OPERATOR_NODE_TYPES = [
    "add",
    "mul",
    "eq0",
    "dt",
    "dx",
    "dy",
    "dz",
    "dn",
    "avg_int",
    "neg",
    "square",
    "exp10",
    "log10",
    "sin",
    "cos",
]
RESERVED_NODE_TYPES = ["vc", "at"] + [f"Reserved{i}" for i in range(14)]
FUNCTION_BRANCH_NODE_TYPES = [f"Branch{i}" for i in range(16)]
INR_NODE_TYPES = [f"Mod{i}" for i in range(32)]
DAG_NODE_TYPES = (
    ["pad"]
    + VAR_NODE_TYPES
    + COEF_NODE_TYPES
    + FUNCTION_NODE_TYPES
    + OPERATOR_NODE_TYPES
    + RESERVED_NODE_TYPES
    + FUNCTION_BRANCH_NODE_TYPES
    + INR_NODE_TYPES
)
NODE_TYPE_DICT = {t: i for i, t in enumerate(DAG_NODE_TYPES)}

# Defaults that match jax_pdeformer2.PDEformer
DEFAULT_NUM_SPATIAL = 16
DEFAULT_NUM_DEGREE = 32
DISCONN_ATTN_BIAS = -1.0e9


class UnsupportedPDEOperatorError(ValueError):
    """Raised when a jNO trace node cannot be mapped to a PDEformer-2 DAG node."""


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------


class PDEformer2Wrapper(eqx.Module):
    """Bakes the static PDE DAG into a ``PDEformer`` and exposes ``(t, x, y) -> u``.

    The graph tensors are stored as numpy arrays and converted to JAX arrays
    via ``jax.lax.stop_gradient`` inside ``__call__`` so they are never updated
    by the optimizer.
    """

    pde_model: Any  # jax_pdeformer2.PDEformer — fully trainable

    _node_type: Any
    _node_scalar: Any
    _node_function: Any
    _in_degree: Any
    _out_degree: Any
    _attn_bias: Any
    _spatial_pos: Any

    t_arg_idx: int = eqx.field(static=True)
    x_arg_idx: int = eqx.field(static=True)
    y_arg_idx: int = eqx.field(static=True)

    def __init__(
        self,
        pde_model,
        node_type,
        node_scalar,
        node_function,
        in_degree,
        out_degree,
        attn_bias,
        spatial_pos,
        t_arg_idx: int,
        x_arg_idx: int,
        y_arg_idx: int,
    ):
        self.pde_model = pde_model
        self._node_type = jnp.asarray(node_type)
        self._node_scalar = jnp.asarray(node_scalar)
        self._node_function = jnp.asarray(node_function)
        self._in_degree = jnp.asarray(in_degree)
        self._out_degree = jnp.asarray(out_degree)
        self._attn_bias = jnp.asarray(attn_bias)
        self._spatial_pos = jnp.asarray(spatial_pos)
        self.t_arg_idx = t_arg_idx
        self.x_arg_idx = x_arg_idx
        self.y_arg_idx = y_arg_idx

    def __call__(self, *args):
        t = args[self.t_arg_idx]
        x = args[self.x_arg_idx]
        y = args[self.y_arg_idx]
        # broadcast to a common (N, 1) shape so concatenation is well-defined
        t_arr = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(t).reshape(-1)), x.reshape(-1).shape)
        x_arr = jnp.asarray(x).reshape(-1)
        y_arr = jnp.asarray(y).reshape(-1)
        coord = jnp.stack([t_arr, x_arr, y_arr, jnp.zeros_like(x_arr)], axis=-1)  # (N, 4)

        node_type = jax.lax.stop_gradient(self._node_type)
        node_scalar = jax.lax.stop_gradient(self._node_scalar)
        node_function = jax.lax.stop_gradient(self._node_function)
        in_degree = jax.lax.stop_gradient(self._in_degree)
        out_degree = jax.lax.stop_gradient(self._out_degree)
        attn_bias = jax.lax.stop_gradient(self._attn_bias)
        spatial_pos = jax.lax.stop_gradient(self._spatial_pos)

        return self.pde_model(
            node_type,
            node_scalar,
            node_function,
            in_degree,
            out_degree,
            attn_bias,
            spatial_pos,
            coord,
        )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class PDEGraphBuilder:
    """Builds a PDEformer-2 DAG node-by-node and produces static tensors."""

    def __init__(self, n_inr_nodes: int, function_num_branches: int):
        self.n_inr_nodes = int(n_inr_nodes)
        self.function_num_branches = int(function_num_branches)

        # scalar nodes (Mod*, uf, coef, operator, ic-marker, ...)
        self._scalar_types: List[str] = []
        self._scalar_values: List[float] = []
        self._scalar_node_ids: List[int] = []

        # function nodes (one per ic/cf/bv/sdf, each with `function_num_branches` Branch* tokens)
        self._function_arrays: List[np.ndarray] = []
        self._function_marker_ids: List[int] = []  # id of the 'ic' marker (sits in scalar section)
        self._function_kinds: List[str] = []  # 'ic' / 'cf' / etc.

        # edges: (src_id, dst_id) — direction matches the original repo's _build_dag
        self._edges: List[Tuple[int, int]] = []

        self._next_id = 0

    # -- node primitives --------------------------------------------------

    def _add_scalar(self, type_name: str, scalar: float = 0.0) -> int:
        nid = self._next_id
        self._next_id += 1
        self._scalar_types.append(type_name)
        self._scalar_values.append(float(scalar))
        self._scalar_node_ids.append(nid)
        return nid

    def add_uf(self) -> int:
        """Adds an unknown-field node plus ``n_inr_nodes`` Mod nodes wired to it."""
        uf_id = self._add_scalar("uf")
        for j in range(self.n_inr_nodes):
            mod_id = self._add_scalar(f"Mod{j}")
            # Mod -> uf
            self._edges.append((mod_id, uf_id))
        return uf_id

    def add_op(self, type_name: str, *parent_ids: int) -> int:
        if type_name not in OPERATOR_NODE_TYPES:
            raise ValueError(f"Unknown operator type '{type_name}'")
        nid = self._add_scalar(type_name)
        for pid in parent_ids:
            self._edges.append((pid, nid))
        return nid

    def add_coef(self, value: float) -> int:
        return self._add_scalar("coef", value)

    def add_ic(self, values: np.ndarray, x_pts, y_pts, t_pts) -> int:
        """Add an IC function node + ``function_num_branches`` Branch nodes.

        Args:
            values: shape ``(n_pts,)`` sampled IC values.
            x_pts, y_pts, t_pts: shape ``(n_pts,)`` coordinate arrays.

        Returns:
            Scalar-section id of the 'ic' marker node.
        """
        ic_id = self._add_scalar("ic")
        # Function array column layout: [t, x, y, z, f]
        arr = np.stack(
            [
                np.asarray(t_pts, dtype=np.float32),
                np.asarray(x_pts, dtype=np.float32),
                np.asarray(y_pts, dtype=np.float32),
                np.zeros_like(np.asarray(x_pts, dtype=np.float32)),
                np.asarray(values, dtype=np.float32),
            ],
            axis=-1,
        )
        self._function_arrays.append(arr)
        self._function_marker_ids.append(ic_id)
        self._function_kinds.append("ic")
        return ic_id

    def add_eq0(self, lhs_id: int) -> int:
        return self.add_op("eq0", lhs_id)

    # -- tensor assembly --------------------------------------------------

    def build_tensors(self) -> dict:
        """Produce the numpy tensors that ``PDEformer`` consumes."""
        n_scalar = len(self._scalar_types)
        n_func = len(self._function_arrays)
        n_branches_per_func = self.function_num_branches

        # Allocate Branch* node ids after all scalar nodes (they live in the
        # "function" tail of the node_type tensor).
        func_branch_ids: List[List[int]] = []  # per function: list of branch node ids
        next_id = n_scalar
        for fi in range(n_func):
            ids = list(range(next_id, next_id + n_branches_per_func))
            func_branch_ids.append(ids)
            next_id += n_branches_per_func

        n_total = n_scalar + n_func * n_branches_per_func

        # node_type
        node_type = np.zeros((n_total,), dtype=np.int32)
        for i, type_name in enumerate(self._scalar_types):
            node_type[i] = NODE_TYPE_DICT[type_name]
        for fi in range(n_func):
            for j, bid in enumerate(func_branch_ids[fi]):
                node_type[bid] = NODE_TYPE_DICT[f"Branch{j}"]

        # node_scalar (scalar section only)
        node_scalar = np.zeros((n_scalar, 1), dtype=np.float32)
        for i, v in enumerate(self._scalar_values):
            node_scalar[i, 0] = v

        # node_function (function section only) — needs (n_func, n_pts, 5)
        if n_func == 0:
            node_function = np.zeros((0, 1, 5), dtype=np.float32)
        else:
            n_pts = self._function_arrays[0].shape[0]
            for arr in self._function_arrays:
                if arr.shape[0] != n_pts:
                    raise ValueError(
                        f"All function nodes must share the same number of sample points; got {arr.shape[0]} vs {n_pts}."
                    )
            node_function = np.stack(self._function_arrays, axis=0).astype(np.float32)

        # Build all edges (scalar PDE edges + ic→Branch edges).
        all_edges: List[Tuple[int, int]] = list(self._edges)
        for fi, marker_id in enumerate(self._function_marker_ids):
            for bid in func_branch_ids[fi]:
                all_edges.append((marker_id, bid))

        # Floyd-Warshall in pure numpy.
        INF = np.float32(np.inf)
        dist = np.full((n_total, n_total), INF, dtype=np.float32)
        np.fill_diagonal(dist, 0.0)
        for src, dst in all_edges:
            dist[src, dst] = 1.0
        for k in range(n_total):
            dk = dist[:, k : k + 1] + dist[k : k + 1, :]
            np.minimum(dist, dk, out=dist)

        # spatial_pos: +1 because 0 is reserved for padding; clamped to num_spatial-1.
        sp_clamped = np.where(np.isfinite(dist), dist, DEFAULT_NUM_SPATIAL - 2)
        sp_clamped = np.clip(sp_clamped, 0, DEFAULT_NUM_SPATIAL - 2)
        spatial_pos = (1 + sp_clamped).astype(np.int32)

        # attn_bias: 0 for connected (finite distance), DISCONN_ATTN_BIAS for unreachable.
        attn_bias = np.zeros((n_total, n_total), dtype=np.float32)
        unreachable = ~np.isfinite(dist)
        unreachable = np.logical_and(unreachable, unreachable.T)
        attn_bias[unreachable] = DISCONN_ATTN_BIAS

        # in/out degree (1-indexed to mirror the original repo).
        in_deg = np.ones((n_total,), dtype=np.int32)
        out_deg = np.ones((n_total,), dtype=np.int32)
        for src, dst in all_edges:
            out_deg[src] += 1
            in_deg[dst] += 1
        in_deg = np.clip(in_deg, 0, DEFAULT_NUM_DEGREE - 1)
        out_deg = np.clip(out_deg, 0, DEFAULT_NUM_DEGREE - 1)

        # Add the n_graph=1 leading axis and reshape to the encoder's expected shapes.
        return {
            "node_type": node_type[None, :, None],  # (1, n_total, 1)
            "node_scalar": node_scalar[None, :, :],  # (1, n_scalar, 1)
            "node_function": node_function[None, ...],  # (1, n_func, n_pts, 5)
            "in_degree": in_deg[None, :],  # (1, n_total)
            "out_degree": out_deg[None, :],  # (1, n_total)
            "attn_bias": attn_bias[None, :, :],  # (1, n_total, n_total)
            "spatial_pos": spatial_pos[None, :, :],  # (1, n_total, n_total)
        }


# ---------------------------------------------------------------------------
# Trace walker — turn a jNO Placeholder tree into DAG nodes
# ---------------------------------------------------------------------------


class PDETraceWalker:
    """Maps jNO trace nodes to PDEformer-2 DAG nodes."""

    def __init__(self, target_model: Model, builder: PDEGraphBuilder, uf_id: int):
        self.target_model = target_model
        self.builder = builder
        self.uf_id = uf_id

    def walk(self, expr) -> int:
        # ModelCall referencing the PDEformer model → uf
        if isinstance(expr, ModelCall):
            if expr.model is self.target_model:
                return self.uf_id
            raise UnsupportedPDEOperatorError(
                f"ModelCall to a different model {expr.model} inside a PDEformer-2 PDE expression."
            )

        # Pass through identity-like FunctionCall wrappers like `.mse`
        if isinstance(expr, FunctionCall):
            name = getattr(expr, "_name", None)
            fn = getattr(expr, "fn", None)

            # Recognize sin / cos / square via fn identity or name.
            if name == "mse" or name == "mae":
                # Should have been unwrapped already; defensive fallback.
                return self.walk(expr.args[0])
            if fn is jnp.sin or name == "sin":
                return self.builder.add_op("sin", self.walk(expr.args[0]))
            if fn is jnp.cos or name == "cos":
                return self.builder.add_op("cos", self.walk(expr.args[0]))
            if name == "square" or fn is jnp.square:
                return self.builder.add_op("square", self.walk(expr.args[0]))
            raise UnsupportedPDEOperatorError(
                f"FunctionCall '{name or fn}' is not supported in the PDEformer-2 vocabulary."
            )

        if isinstance(expr, Jacobian):
            if len(expr.variables) != 1:
                raise UnsupportedPDEOperatorError(
                    "Multi-variable Jacobian cannot be mapped to a single PDEformer-2 derivative."
                )
            var = expr.variables[0]
            type_name = _jacobian_axis_type(var)
            return self.builder.add_op(type_name, self.walk(expr.target))

        if isinstance(expr, Hessian):
            if not expr.trace:
                raise UnsupportedPDEOperatorError("Full Hessian is not supported; use trace=True (Laplacian) only.")
            # Decompose ∇² into Σ d/dx_i (d/dx_i (target)).
            inner_ids = []
            for var in expr.variables:
                type_name = _jacobian_axis_type(var)
                inner = self.builder.add_op(type_name, self.walk(expr.target))
                outer = self.builder.add_op(type_name, inner)
                inner_ids.append(outer)
            if len(inner_ids) == 1:
                return inner_ids[0]
            # Reduce via successive 'add'.
            acc = inner_ids[0]
            for nid in inner_ids[1:]:
                acc = self.builder.add_op("add", acc, nid)
            return acc

        if isinstance(expr, BinaryOp):
            op = expr.op
            left, right = expr.left, expr.right

            if op == "+":
                return self.builder.add_op("add", self.walk(left), self.walk(right))

            if op == "-":
                return self.builder.add_op(
                    "add",
                    self.walk(left),
                    self.builder.add_op("neg", self.walk(right)),
                )

            if op == "*":
                # Scalar coefficient * subexpression → mul(coef, walk(other))
                if isinstance(left, Literal):
                    return self.builder.add_op(
                        "mul",
                        self.builder.add_coef(float(jnp.asarray(left.value))),
                        self.walk(right),
                    )
                if isinstance(right, Literal):
                    return self.builder.add_op(
                        "mul",
                        self.walk(left),
                        self.builder.add_coef(float(jnp.asarray(right.value))),
                    )
                return self.builder.add_op("mul", self.walk(left), self.walk(right))

            if op == "**" and isinstance(right, Literal):
                power = int(jnp.asarray(right.value))
                if power == 2:
                    return self.builder.add_op("square", self.walk(left))
                raise UnsupportedPDEOperatorError(f"Power {power} is not supported (only 2 maps to 'square').")

            raise UnsupportedPDEOperatorError(f"BinaryOp '{op}' is not supported.")

        if isinstance(expr, Literal):
            return self.builder.add_coef(float(jnp.asarray(expr.value)))

        raise UnsupportedPDEOperatorError(
            f"Trace node of type {type(expr).__name__} is not supported in PDEformer-2 graphs."
        )


def _jacobian_axis_type(var: Variable) -> str:
    if getattr(var, "axis", "spatial") == "temporal":
        return "dt"
    start, _ = var.dim
    if start == 0:
        return "dx"
    if start == 1:
        return "dy"
    if start == 2:
        return "dz"
    raise UnsupportedPDEOperatorError(f"Cannot infer derivative axis for Variable with dim={var.dim}, axis={var.axis}.")


# ---------------------------------------------------------------------------
# IC: pure-expression evaluator + RHS extraction
# ---------------------------------------------------------------------------


def _pure_eval(expr, var_values: dict, model_to_skip: Model) -> jnp.ndarray:
    """Evaluate a Placeholder subtree that contains no ModelCalls."""
    if isinstance(expr, ModelCall):
        if expr.model is model_to_skip:
            return jnp.zeros_like(next(iter(var_values.values())))
        raise UnsupportedPDEOperatorError(f"Cannot evaluate IC RHS containing call to model {expr.model}")

    if isinstance(expr, Variable):
        key = _variable_key(expr)
        if key not in var_values:
            raise UnsupportedPDEOperatorError(f"IC RHS references variable {key} but no sample values were provided.")
        return var_values[key]

    if isinstance(expr, Literal):
        return jnp.asarray(expr.value, dtype=jnp.float32)

    if isinstance(expr, BinaryOp):
        l = _pure_eval(expr.left, var_values, model_to_skip)
        r = _pure_eval(expr.right, var_values, model_to_skip)
        return {
            "+": l + r,
            "-": l - r,
            "*": l * r,
            "/": l / r,
            "**": l**r,
        }[expr.op]

    if isinstance(expr, FunctionCall):
        args = [_pure_eval(a, var_values, model_to_skip) if isinstance(a, Placeholder) else a for a in expr.args]
        return expr.fn(*args)

    raise UnsupportedPDEOperatorError(f"Cannot evaluate IC RHS — unsupported node type {type(expr).__name__}.")


def _variable_key(v: Variable) -> tuple:
    return (v.tag, tuple(v.dim) if v.dim else None, v.axis)


def _contains_model_call(expr, model: Model) -> bool:
    if isinstance(expr, ModelCall) and expr.model is model:
        return True
    for attr in ("left", "right", "target", "expr"):
        child = getattr(expr, attr, None)
        if isinstance(child, Placeholder) and _contains_model_call(child, model):
            return True
    for attr in ("args", "variables", "options"):
        vals = getattr(expr, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, Placeholder) and _contains_model_call(v, model):
                return True
    return False


def _ic_target_sign(ini_expr, model: Model) -> int:
    """Return +1 if ``ini = f - u0`` (target is +eval), -1 if ``ini = u0 - f`` (target is -eval).

    Walks the top-level BinaryOp tree:
      * ``BinaryOp('-', a, b)``: if ``a`` contains the model, target = -eval(ini, u0=0)
      * ``BinaryOp('+', a, b)``: target = -eval (sign-flipped on the other side)
      * Anything else: assume the common ``u0 - f`` convention, target = -eval.
    """
    if isinstance(ini_expr, BinaryOp) and ini_expr.op == "-":
        if _contains_model_call(ini_expr.left, model):
            return -1  # ini = u0 - f → f = -eval(ini, u0=0)
        if _contains_model_call(ini_expr.right, model):
            return +1  # ini = f - u0 → f = +eval(ini, u0=0)
    return -1


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _unwrap_loss(expr):
    """Strip ``.mse``/``.mae`` wrappers to get the underlying residual."""
    while isinstance(expr, FunctionCall) and getattr(expr, "_name", None) in {"mse", "mae"}:
        expr = expr.args[0]
    return expr


def _collect_modelcalls(expr, model: Model, out: list):
    if isinstance(expr, ModelCall) and expr.model is model:
        out.append(expr)
    for attr in ("left", "right", "target", "expr"):
        child = getattr(expr, attr, None)
        if isinstance(child, Placeholder):
            _collect_modelcalls(child, model, out)
    for attr in ("args", "variables", "options"):
        vals = getattr(expr, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, Placeholder):
                _collect_modelcalls(v, model, out)


def _collect_models(expr, out: list, seen: set):
    if isinstance(expr, ModelCall):
        if id(expr.model) not in seen:
            seen.add(id(expr.model))
            out.append(expr.model)
    for attr in ("left", "right", "target", "expr"):
        child = getattr(expr, attr, None)
        if isinstance(child, Placeholder):
            _collect_models(child, out, seen)
    for attr in ("args", "variables", "options"):
        vals = getattr(expr, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, Placeholder):
                _collect_models(v, out, seen)


def _term_has_tag(expr, tag: str) -> bool:
    if isinstance(expr, Variable) and getattr(expr, "axis", "spatial") == "spatial" and expr.tag == tag:
        return True
    for attr in ("left", "right", "target", "expr"):
        child = getattr(expr, attr, None)
        if isinstance(child, Placeholder) and _term_has_tag(child, tag):
            return True
    for attr in ("args", "variables", "options"):
        vals = getattr(expr, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, Placeholder) and _term_has_tag(v, tag):
                return True
    return False


def _is_pdeformer(module) -> bool:
    try:
        from jax_pdeformer2 import PDEformer
    except Exception:
        return False
    return isinstance(module, PDEformer)


def _model_call_arg_layout(mc: ModelCall) -> Tuple[int, int, int]:
    """Inspect ModelCall(net, [t, x, y]) and return (t_idx, x_idx, y_idx)."""
    t_idx = x_idx = y_idx = None
    for i, a in enumerate(mc.args):
        if not isinstance(a, Variable):
            continue
        if a.axis == "temporal":
            t_idx = i
        else:
            start = a.dim[0]
            if start == 0:
                x_idx = i
            elif start == 1:
                y_idx = i
    if t_idx is None or x_idx is None or y_idx is None:
        raise UnsupportedPDEOperatorError(
            "PDEformer-2 backbone requires the model to be called as net(t, x, y) "
            "with separate temporal and spatial Variables."
        )
    return t_idx, x_idx, y_idx


def _ic_sample_grid(domain, resolution: int = 64) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pick a spatial grid for IC sampling and a t-start from the domain."""
    # Spatial bounds: probe context["interior"] if present, else default [0,1]^2.
    try:
        ctx = domain.context
        spatial = np.asarray(ctx.get("interior", None))
    except Exception:
        spatial = None
    if spatial is None or spatial.size == 0:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    else:
        # ``interior`` is shaped (..., dim); collapse to (N, dim).
        flat = np.asarray(spatial).reshape(-1, spatial.shape[-1])
        x_min, x_max = float(flat[:, 0].min()), float(flat[:, 0].max())
        if flat.shape[-1] >= 2:
            y_min, y_max = float(flat[:, 1].min()), float(flat[:, 1].max())
        else:
            y_min, y_max = 0.0, 1.0

    t_start = 0.0
    time_attr = getattr(domain, "time", None)
    if time_attr is not None:
        try:
            t_start = float(time_attr[0])
        except Exception:
            pass

    xs = np.linspace(x_min, x_max, resolution, dtype=np.float32)
    ys = np.linspace(y_min, y_max, resolution, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel(), t_start


def _build_for_model(
    model: Model,
    pde_terms: List[Placeholder],
    ic_terms: List[Placeholder],
    domain,
    resolution: int = 64,
) -> PDEformer2Wrapper:
    pde_inst = model.module
    n_inr_nodes = int(getattr(pde_inst, "n_inr_nodes"))
    # function_num_branches = (resolution // 64)^2 from Conv2dFuncEncoderV3 strides (3 × 4 = 64).
    func_res = int(getattr(pde_inst.pde_encoder.function_encoder, "resolution", 128))
    n_branches = max(1, (func_res // 64)) ** 2

    builder = PDEGraphBuilder(n_inr_nodes=n_inr_nodes, function_num_branches=n_branches)
    uf_id = builder.add_uf()

    # Walk every PDE term and append its LHS to the equation list.
    walker = PDETraceWalker(target_model=model, builder=builder, uf_id=uf_id)
    for term in pde_terms:
        try:
            lhs_id = walker.walk(term)
            builder.add_eq0(lhs_id)
        except UnsupportedPDEOperatorError as e:
            raise UnsupportedPDEOperatorError(f"Failed to translate PDE residual to PDEformer-2 graph: {e}") from e

    # IC: sample on a grid and store as a function node.
    if ic_terms:
        x_pts, y_pts, t_start = _ic_sample_grid(domain, resolution=func_res)
        var_values = {
            # spatial vars on the "initial" tag carry tag="initial" and dim slices [0,1] / [1,2]
            ("initial", (0, 1), "spatial"): jnp.asarray(x_pts),
            ("initial", (1, 2), "spatial"): jnp.asarray(y_pts),
            # temporal var key is the time-tag from domain.variable; we set both common forms
        }
        # Match whatever temporal tag the user's domain produced.
        # (initial-domain time var typically has tag "__time_initial__" or "__time__".)
        time_tags = ["__time_initial__", "__time__"]
        for tt in time_tags:
            var_values[(tt, (0, 1), "temporal")] = jnp.asarray(np.full_like(x_pts, t_start))

        for term in ic_terms:
            sign = _ic_target_sign(term, model)
            try:
                evaluated = _pure_eval(term, var_values, model_to_skip=model)
            except UnsupportedPDEOperatorError as e:
                raise UnsupportedPDEOperatorError(f"Failed to evaluate IC RHS: {e}") from e
            ic_vals_np = sign * np.asarray(evaluated, dtype=np.float32).reshape(-1)
            builder.add_ic(
                values=ic_vals_np,
                x_pts=x_pts,
                y_pts=y_pts,
                t_pts=np.full_like(x_pts, t_start),
            )

    tensors = builder.build_tensors()

    # Determine argument order from any ModelCall referencing this model.
    sample_calls: List[ModelCall] = []
    for term in pde_terms + ic_terms:
        _collect_modelcalls(term, model, sample_calls)
    if not sample_calls:
        raise UnsupportedPDEOperatorError("No ModelCall found for the PDEformer-2 model in the supplied constraints.")
    t_idx, x_idx, y_idx = _model_call_arg_layout(sample_calls[0])

    return PDEformer2Wrapper(
        pde_model=pde_inst,
        t_arg_idx=t_idx,
        x_arg_idx=x_idx,
        y_arg_idx=y_idx,
        **tensors,
    )


def maybe_attach_pdeformer2_graphs(constraints: List[Placeholder], domain) -> None:
    """Detect PDEformer models in ``constraints`` and replace ``model.module``
    with a baked-graph ``PDEformer2Wrapper`` for each one. No-op if none found.
    """
    # Discover all PDEformer-backed Models referenced by any constraint.
    models: List[Model] = []
    seen: set = set()
    for c in constraints:
        _collect_models(c, models, seen)

    pdef_models = [m for m in models if _is_pdeformer(m.module)]
    if not pdef_models:
        return

    for model in pdef_models:
        # Classify constraints that touch this model.  Each constraint is one of:
        #   * IC term:   contains a spatial Variable with tag "initial"
        #   * PDE term:  contains a spatial Variable with tag "interior"
        #   * BC / other: any other tag (e.g. "boundary"); skipped for graph-building
        #                — still trained as a normal soft constraint by jno.core.
        pde_terms: List[Placeholder] = []
        ic_terms: List[Placeholder] = []
        for c in constraints:
            mcs: List[ModelCall] = []
            _collect_modelcalls(c, model, mcs)
            if not mcs:
                continue
            term = _unwrap_loss(c)
            if _term_has_tag(term, "initial"):
                ic_terms.append(term)
            elif _term_has_tag(term, "interior"):
                pde_terms.append(term)
            # else: a boundary/other constraint — fine, just not in the DAG.

        if not pde_terms:
            raise UnsupportedPDEOperatorError(
                "PDEformer-2 backbone was used but no PDE residual was found in the constraints."
            )

        wrapper = _build_for_model(model, pde_terms, ic_terms, domain)
        model.module = wrapper
