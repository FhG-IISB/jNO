"""CORE solver using new tracing system - NO INNER VMAPS version."""

import functools
import inspect
from typing import Any, Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .trace import (
    Assembly,
    BinaryOp,
    Choice,
    Constant,
    FrozenField,
    FunctionCall,
    GroupedAssembly,
    Hessian,
    Integral,
    IntegralTime,
    Jacobian,
    Literal,
    ModelCall,
    ModelWeights,
    NetworkGradient,
    Noise,
    OperationCall,
    OperationDef,
    Placeholder,
    StateField,
    TemporalDerivative,
    TensorTag,
    TestFunction,
    TunableModule,
    TunableModuleCall,
    Variable,
    collect_tags,
)


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


@functools.lru_cache(maxsize=1024)
def _accepts_key(fn) -> bool:
    """Whether ``fn`` declares a ``key=`` parameter.

    Cached on the *unbound* function: ``inspect.signature`` costs ~6 µs and the
    evaluator asks this once per model / custom-function node per trace — a few
    thousand times for a multi-field residual, none of which can change answer
    between calls.
    """
    return "key" in inspect.signature(fn).parameters


from .differential_operators import DifferentialOperators
from .integration_operators import IntegrationOperators
from .utils import get_logger
from .utils.ad_mode import ad_fn, parse_ad_scheme, parse_hessian_scheme
from .utils.schemes import resolve_scheme, scheme_family


def _uniform_grid_spec(domain, n_values: int):
    """``(shape, spacing)`` of the uniform grid a nodal field of length ``n_values`` lives on.

    Prefers the descriptor ``jno.domain(..., structured=True)`` stamps onto
    ``mesh_connectivity["grid"]``. Otherwise derives the spacing from ``_grid_shape`` and the mesh
    points, which the generators lay out in C order (``idx = i*Ny + j``), so a nodal field reshapes
    cleanly. Raises rather than guessing -- a spectral derivative on a non-uniform grid is silently
    wrong, not merely inaccurate.
    """
    if domain is None:
        raise ValueError("scheme='spectral' needs a domain with a uniform grid; this expression has none.")

    grid = (getattr(domain, "mesh_connectivity", None) or {}).get("grid")
    if grid is not None:
        shape, spacing = tuple(int(s) for s in grid["shape"]), tuple(float(h) for h in grid["spacing"])
    else:
        shape = getattr(domain, "_grid_shape", None)
        if not shape:
            raise ValueError(
                "scheme='spectral' requires a uniform grid, and this domain has none. Build it with "
                "Shape.structured() -- e.g. jno.Shape.rect(0, 0, 1, 1, size=h).structured().domain() "
                "-- which records the grid descriptor, or use scheme='finite_difference' on an "
                "unstructured mesh."
            )
        shape = tuple(int(s) for s in shape)
        pts = np.asarray(domain.mesh_connectivity["points"])[:, : len(shape)]
        if pts.shape[0] != int(np.prod(shape)):
            raise ValueError(
                f"scheme='spectral': the mesh has {pts.shape[0]} points but the grid is {shape} "
                f"({int(np.prod(shape))} nodes), so the field does not reshape onto it."
            )
        grid_pts = pts.reshape(*shape, len(shape))
        spacing = []
        for ax in range(len(shape)):
            if shape[ax] < 2:
                spacing.append(1.0)
                continue
            step = np.diff(np.take(grid_pts[..., ax], indices=range(shape[ax]), axis=ax).reshape(shape[ax], -1)[:, 0])
            if not np.allclose(step, step[0], rtol=1e-6, atol=1e-12):
                raise ValueError(
                    f"scheme='spectral': axis {ax} of this grid is not uniformly spaced "
                    f"(min {step.min():.3e}, max {step.max():.3e}). FFT differentiation assumes uniform spacing."
                )
            spacing.append(float(step[0]))
        spacing = tuple(spacing)

    if int(np.prod(shape)) != int(n_values):
        raise ValueError(
            f"scheme='spectral': the field has {n_values} values but the grid {shape} has {int(np.prod(shape))} nodes."
        )
    return shape, spacing


def _mirror_axis(u, axis: int):
    """Even (mirror) extension along ``axis``: ``[u0..u_{N-1}, u_{N-2}..u_1]``, length ``2N-2``.

    The result is periodic, so the ordinary FFT machinery applies unchanged -- which is why the
    cosine sub-scheme needs no DCT/DST. (JAX implements only DCT-2 and has no DST at all, so the
    transform-pair route would have meant hand-rolling one.)
    """
    n = u.shape[axis]
    tail = jnp.flip(jnp.take(u, jnp.arange(1, n - 1), axis=axis), axis=axis)
    return jnp.concatenate([u, tail], axis=axis)


def _unmirror_axis(u, axis: int, n: int):
    """Take the first ``n`` entries along ``axis`` — the original half of a mirrored array."""
    return jnp.take(u, jnp.arange(n), axis=axis)


def _spectral_second_moments(values_flat, shape, spacing, pairs, *, trace: bool, mirror: bool = False):
    """Second derivatives along grid axes, from **one** transform pair.

    ``pairs`` is the list of ``(a, b)`` axis pairs wanted. With ``trace=True`` the diagonal is summed
    into the Laplacian, which is a single multiply by ``-(kx^2 + ky^2 + ...)`` -- the reason a
    spectral Laplacian is cheap where a stencil needs a pass per axis and automatic differentiation
    needs a full Hessian per point.

    Same endpoint convention as :func:`_spectral_diff`: the duplicated periodic node is dropped on
    every transformed axis and re-attached afterwards.
    """
    was_real = not jnp.iscomplexobj(values_flat)
    u = jnp.asarray(values_flat).reshape(shape)
    axes = tuple(sorted({int(a) for pair in pairs for a in pair}))

    orig_n = {a: u.shape[a] for a in axes}
    if mirror:
        for a in axes:
            u = _mirror_axis(u, a)
    else:
        sl = [slice(None)] * len(shape)
        for a in axes:
            sl[a] = slice(0, -1)
        u = u[tuple(sl)]

    def _k(a, *, odd: bool):
        n = u.shape[a]
        k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=spacing[a])
        if odd and n % 2 == 0:
            k = k.at[n // 2].set(0.0)  # Nyquist has no well-defined odd derivative
        return k.reshape([n if i == a else 1 for i in range(len(shape))])

    uh = jnp.fft.fftn(u, axes=axes)

    def _restore(arr):
        for a in axes:
            arr = (
                _unmirror_axis(arr, a, orig_n[a])
                if mirror
                else jnp.concatenate([arr, jnp.take(arr, jnp.array([0]), axis=a)], axis=a)
            )
        return arr

    if trace:
        mult = sum(-(_k(a, odd=False) ** 2) for a, _ in pairs)
        out = jnp.fft.ifftn(mult * uh, axes=axes)
        out = out.real if was_real else out
        return _restore(out).reshape(-1)

    comps = []
    for a, b in pairs:
        mult = -(_k(a, odd=False) ** 2) if a == b else -(_k(a, odd=True) * _k(b, odd=True))
        o = jnp.fft.ifftn(mult * uh, axes=axes)
        comps.append(_restore(o.real if was_real else o).reshape(-1))
    return comps


def _spectral_diff(values_flat, shape, spacing, axis: int, order: int, *, mirror: bool = False):
    """``d^order/dx_axis^order`` of a flat nodal field, via the FFT along that grid axis.

    Exact for a band-limited periodic field; on a non-periodic one the implied periodic extension
    has a jump and the result rings (Gibbs). Periodicity is the caller's claim -- it is not read
    from the domain, because the only per-axis periodic flag jNO records is a residue of whether a
    periodic FDM problem happened to be built earlier in the process.

    Trefethen, *Spectral Methods in MATLAB* (SIAM 2000), ch. 3.
    """
    was_real = not jnp.iscomplexobj(values_flat)
    u = jnp.asarray(values_flat).reshape(shape)
    h = spacing[axis]

    if mirror:
        # Even extension: exact when the field's odd derivatives vanish at both ends (Neumann-like),
        # which is a real but NARROWER class than "non-periodic" -- a field with u' != 0 at an end
        # still has a kink in the extension and still rings, just far less (measured 44x on a ramp).
        n_orig = u.shape[axis]
        u = _mirror_axis(u, axis)
        n = u.shape[axis]
        k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=h)
        if order % 2 == 1 and n % 2 == 0:
            k = k.at[n // 2].set(0.0)
        mult = ((1j * k) ** order).reshape([n if a == axis else 1 for a in range(len(shape))])
        out = jnp.fft.ifft(mult * jnp.fft.fft(u, axis=axis), axis=axis)
        out = out.real if was_real else out
        return _unmirror_axis(out, axis, n_orig).reshape(-1)

    # jNO's structured grids span the interval INCLUSIVE of both ends, so the last node along an
    # axis is the periodic image of the first: `Shape.rect(0,0,1,1, size=1/16)` gives 17 nodes for
    # 16 intervals. The FFT wants exactly one period with no duplicate, so drop that node, transform
    # over the remaining n, and put it back afterwards. The finite-difference periodic stencils do
    # the same thing (`differential_operators.py`, `uu = moveaxis(U, dim, 0)[:-1]`). Without this
    # the transform assumes a period of (n+1)*h and every derivative is wrong -- measurably worse
    # than a 2nd-order stencil, which is how it was caught.
    u = jnp.moveaxis(u, axis, 0)[:-1]
    u = jnp.moveaxis(u, 0, axis)
    n = u.shape[axis]

    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=h)
    if order % 2 == 1 and n % 2 == 0:
        # The Nyquist mode has no well-defined odd derivative (it is its own alias); the standard
        # convention drops it rather than letting an arbitrary sign through.
        k = k.at[n // 2].set(0.0)
    mult = (1j * k) ** order
    mult = mult.reshape([n if a == axis else 1 for a in range(len(shape))])

    out = jnp.fft.ifft(mult * jnp.fft.fft(u, axis=axis), axis=axis)
    out = out.real if was_real else out
    # Re-attach the duplicated endpoint so the result lines up with the mesh's node ordering.
    first = jnp.take(out, jnp.array([0]), axis=axis)
    return jnp.concatenate([out, first], axis=axis).reshape(-1)


class _MeshCtx(NamedTuple):
    """What a mesh-field kernel needs from the domain: the nodes, the count, and the domain itself."""

    points: Any
    n: int
    domain: Any
    dim: int


class _MeshFieldBackend(NamedTuple):
    """A family that differentiates stored values on the mesh.

    Each entry builds a per-channel kernel from ``(mesh, scheme)``. Adding a family is adding an
    entry here plus its kernels -- ``_eval_jacobian`` / ``_eval_hessian`` do not change.

    ``gradient(mesh, scheme) -> f(u_flat, axis) -> (N,)``
    ``laplacian(mesh, scheme, dims) -> f(u_flat) -> (N,)``
    ``hessian(mesh, scheme, var_dims) -> f(u_flat) -> (N, n, n)``
    """

    gradient: Any
    laplacian: Any
    hessian: Any


def _fd_gradient(mesh, scheme):
    _, grad_method, _ = DifferentialOperators.parse_fd_scheme(scheme)
    mc = mesh.domain.mesh_connectivity
    cells = {1: "lines", 2: "triangles", 3: "tetrahedra"}[mesh.dim]
    fn = {
        1: DifferentialOperators.compute_fd_gradient_1d_simple,
        2: DifferentialOperators.compute_fd_gradient_2d_simple,
        3: DifferentialOperators.compute_fd_gradient_3d_simple,
    }[mesh.dim]

    def _k(u_1d, axis):
        if mesh.dim == 1:
            return fn(u_1d, mesh.points, mc[cells], method=grad_method, grid=mc.get("grid"))
        return fn(u_1d, mesh.points, mc[cells], axis, method=grad_method, grid=mc.get("grid"))

    return _k


def _spectral_gradient(mesh, scheme):
    g_shape, g_spacing = _uniform_grid_spec(mesh.domain, mesh.n)
    mirror = scheme.split(":", 1)[1].strip() == "cosine" if ":" in scheme else False

    def _k(u_1d, axis):
        return _spectral_diff(u_1d, g_shape, g_spacing, axis, order=1, mirror=mirror)

    return _k


def _fd_laplacian(mesh, scheme, dims):
    _, _, lap_method = DifferentialOperators.parse_fd_scheme(scheme)
    mc = mesh.domain.mesh_connectivity
    cells = {1: "lines", 2: "triangles", 3: "tetrahedra"}[mesh.dim]
    fn = {
        1: DifferentialOperators.compute_fd_laplacian_1d_simple,
        2: DifferentialOperators.compute_fd_laplacian_2d_simple,
        3: DifferentialOperators.compute_fd_laplacian_3d_simple,
    }[mesh.dim]

    def _k(u_1d):
        if mesh.dim == 1:
            return fn(u_1d, mesh.points, mc[cells], grid=mc.get("grid"))
        return fn(u_1d, mesh.points, mc[cells], dims, method=lap_method, grid=mc.get("grid"))

    return _k


def _fd_hessian(mesh, scheme, var_dims):
    mc = mesh.domain.mesh_connectivity
    cells = {1: "lines", 2: "triangles", 3: "tetrahedra"}[mesh.dim]
    fn = {
        1: DifferentialOperators.compute_fd_hessian_1d_simple,
        2: DifferentialOperators.compute_fd_hessian_2d_simple,
        3: DifferentialOperators.compute_fd_hessian_3d_simple,
    }[mesh.dim]

    def _k(u_1d):
        if mesh.dim == 1:
            return fn(u_1d, mesh.points, mc[cells])
        return fn(u_1d, mesh.points, mc[cells], var_dims, grid=mc.get("grid"))

    return _k


def _spectral_laplacian(mesh, scheme, dims):
    g_shape, g_spacing = _uniform_grid_spec(mesh.domain, mesh.n)
    mirror = scheme.split(":", 1)[1].strip() == "cosine" if ":" in scheme else False
    pairs = [(int(d), int(d)) for d in dims]

    def _k(u_1d):
        return _spectral_second_moments(u_1d, g_shape, g_spacing, pairs, trace=True, mirror=mirror)

    return _k


def _spectral_hessian(mesh, scheme, var_dims):
    g_shape, g_spacing = _uniform_grid_spec(mesh.domain, mesh.n)
    mirror = scheme.split(":", 1)[1].strip() == "cosine" if ":" in scheme else False
    pairs = [(vi, vj) for _i, vi, _j, vj in var_dims]
    n = int(np.sqrt(len(var_dims)))

    def _k(u_1d):
        comps = _spectral_second_moments(u_1d, g_shape, g_spacing, pairs, trace=False, mirror=mirror)
        return jnp.stack(comps, axis=-1).reshape(-1, n, n)

    return _k


#: Families that differentiate stored values on the mesh. A new one is an entry here plus its
#: kernels; neither `_eval_jacobian` nor `_eval_hessian` changes. (Automatic differentiation is
#: deliberately absent -- it differentiates a FUNCTION per point and never touches the mesh.)
_MESH_FIELD_FAMILIES: Dict[str, "_MeshFieldBackend"] = {
    "finite_difference": _MeshFieldBackend(_fd_gradient, _fd_laplacian, _fd_hessian),
    "spectral": _MeshFieldBackend(_spectral_gradient, _spectral_laplacian, _spectral_hessian),
}


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

    # Sentinel: the sampled points ARE the mesh vertices, so the mesh→sampled map is the identity and the
    # values pass through unremapped (the common collocation-on-mesh case, decided once on the host).
    _NN_IDENTITY = object()

    def __init__(self, params: Dict):
        self.params = params
        self.log = get_logger()
        self._logged_schemes: Dict[str, str] = {}
        self._nn_index_cache: Dict = {}  # (mesh, sampled) point-set geometry -> nearest-vertex gather index
        self._mesh_eval_cache: Dict = {}  # (target, tag, points, context) -> target sampled on the mesh

    # ------------------------------------------------------------------
    # Evaluation context — lightweight carrier replacing 5 positional args
    # ------------------------------------------------------------------
    class _EvalCtx:
        """Bundles the read-only state that every handler needs."""

        __slots__ = ("context", "var_bindings", "key", "active_region")

        def __init__(self, context, var_bindings, key, active_region=None):
            self.context = context
            self.var_bindings = var_bindings
            self.key = key
            self.active_region = active_region

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------
    def evaluate(
        self,
        expr,
        context: Dict[str, jnp.ndarray] = None,
        var_bindings: Dict = None,
        key=None,
        active_region=None,
    ) -> jnp.ndarray:
        """Evaluate expression for a SINGLE batch (no batch dimension)."""
        ctx = self._EvalCtx(
            context=context or {},
            var_bindings=var_bindings or {},
            key=key,
            active_region=None,
        )
        # Sharing between FD derivative nodes is scoped to one expression: the cache
        # keys hold identities that only exist inside this call.
        self._mesh_eval_cache.clear()
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
            active_region=None,
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
        except Exception as exc:
            # eval_shape can fail for any number of reasons (unbound vars,
            # non-jit-traceable ops, etc.); fall back to structural inference
            # rather than aborting constraint-shape logging.
            self.log.debug(f"eval_shape fallback for {label}: {type(exc).__name__}: {exc}")
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
        elif isinstance(node, Choice):
            for opt in node.options:
                if isinstance(opt, Placeholder):
                    self._trace_visit(opt, ctx, depth + 1, lines, seen)
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
        elif isinstance(node, GroupedAssembly):
            if node.volume_expr is not None:
                self._trace_visit(node.volume_expr, ctx, depth + 1, lines, seen)
            for bnd_expr in node.boundary_exprs.values():
                self._trace_visit(bnd_expr, ctx, depth + 1, lines, seen)
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
        if isinstance(node, GroupedAssembly):
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
        (FrozenField, "_eval_frozen_field"),
        (Variable, "_eval_variable"),
        (FunctionCall, "_eval_function_call"),
        (BinaryOp, "_eval_binary_op"),
        (OperationCall, "_eval_operation_call"),
        (ModelCall, "_eval_flax_module_call"),
        (ModelWeights, "_eval_model_weights"),
        (TunableModule, "_eval_tunable_module"),
        (TunableModuleCall, "_eval_tunable_module_call"),
        (Choice, "_eval_choice"),
        (Jacobian, "_eval_jacobian"),
        (Hessian, "_eval_hessian"),
        (Integral, "_eval_integral"),
        (IntegralTime, "_eval_integral_time"),
        (TemporalDerivative, "_eval_temporal_derivative"),
        (OperationDef, "_eval_operation_def"),
        (TestFunction, "_eval_test_function"),
        (Assembly, "_eval_assembly"),
        (StateField, "_eval_state_field"),
        (GroupedAssembly, "_eval_grouped_assembly"),
        (NetworkGradient, "_eval_network_gradient"),
        (Noise, "_eval_noise"),
    ]

    def _dispatch(self, expr, ctx):
        """Look up handler in the dispatch table and call it."""
        for node_type, method_name in self._HANDLERS:
            if isinstance(expr, node_type):
                # Assembly defines the active variational bucket for its subtree.
                if isinstance(expr, Assembly):
                    new_ctx = self._EvalCtx(
                        ctx.context,
                        ctx.var_bindings,
                        ctx.key,
                        active_region={
                            "support": expr.support,
                            "region_id": expr.region_id,
                        },
                    )
                    return getattr(self, method_name)(expr, new_ctx)

                return getattr(self, method_name)(expr, ctx)

        raise ValueError(f"Cannot evaluate: {type(expr)}")

    # ------------------------------------------------------------------
    # Helpers shared by differential-operator handlers
    # ------------------------------------------------------------------
    def _build_local_context(self, idx, tag, points, context):
        """Build dynamically-sliced local context for a single point ``idx``."""
        local = {
            "__active_spatial_n__": 1,
        }

        for k, v in context.items():
            # keep helper key / scalars / dicts
            if k == "__active_spatial_n__":
                continue
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
                    start_indices = (idx,) + (0,) * (v.ndim - 1)
                    slice_sizes = (1,) + tuple(v.shape[1:])
                    local[k] = jax.lax.dynamic_slice(v, start_indices, slice_sizes)
                else:
                    local[k] = v

        return local

    def _mesh_field_values(self, target, tag, bound_var, ctx, family):
        """Evaluate ``target`` on the mesh and flatten it for a per-channel kernel.

        Returns ``(mesh, u_flat, image_shape, n_channels)``. Operator-learning models return
        image-shaped tensors whose spatial axes flatten to the node count; the original shape is
        remembered so it can be restored, and a multi-channel output is vmapped over its channel
        axis by the caller. The ``ndim > 2`` gate keeps a plain ``(N, 1)`` per-point scalar from
        being reinterpreted as an image.
        """
        domain = getattr(bound_var, "_domain", None)
        if domain is None or getattr(domain, "mesh_connectivity", None) is None:
            raise ValueError(f"scheme={family!r} requires a domain with mesh connectivity.")
        mesh_points = jnp.array(domain.mesh_connectivity["points"])
        mesh = _MeshCtx(mesh_points, int(mesh_points.shape[0]), domain, int(domain.mesh_connectivity["dimension"]))

        u_full = self._target_on_mesh(target, tag, mesh_points, ctx)
        u_squeezed = u_full.squeeze(-1) if (u_full.ndim > 1 and u_full.shape[-1] == 1) else u_full
        if u_squeezed.ndim > 1 and u_squeezed.size == mesh.n:
            return mesh, u_squeezed.reshape(mesh.n), u_full.shape, 1
        if u_full.ndim > 2 and int(np.prod(u_full.shape[:-1])) == mesh.n:
            return mesh, u_full.reshape(mesh.n, u_full.shape[-1]), u_full.shape, u_full.shape[-1]
        return mesh, (u_squeezed if u_squeezed.ndim == 1 else u_squeezed.ravel()), None, 1

    def _finish_mesh_jacobian(self, comps, image_shape, n_vars, mesh_points, points):
        """Shape a mesh-field gradient back to the caller's convention."""
        if image_shape is not None:
            if n_vars == 1:
                return comps[0].reshape(image_shape)
            return jnp.stack(comps, axis=-1).reshape(*image_shape[:-1], n_vars)
        if n_vars == 1:
            result = self._map_mesh_to_sampled(mesh_points, points, comps[0])
            return result[:, jnp.newaxis] if result.ndim == 1 else result
        return self._map_mesh_to_sampled(mesh_points, points, jnp.stack(comps, axis=-1))

    def _target_on_mesh(self, target, tag, mesh_points, ctx):
        """Evaluate ``target`` with the spatial tag ``tag`` bound to the whole mesh point set.

        Every finite-difference derivative applies its stencil to the target sampled
        on the mesh, and one residual routinely carries several of them over the same
        field — ``laplacian(u, fd) + u.d(x, fd) + u`` re-traced the network once per
        node.  Siblings share a single evaluation here: the value depends only on the
        target, the tag, the point set and the surrounding context, and all four are
        the same object for siblings of one expression.  Only the spatial tag is
        replaced, so ``__time__`` and the rest of the context stay intact.
        """
        key = (
            id(target),
            tag,
            id(mesh_points),
            id(ctx.context),
            id(ctx.var_bindings),
            id(ctx.active_region),
        )
        cached = self._mesh_eval_cache.get(key)
        if cached is not None:
            return cached[-1]

        new_ctx = self._EvalCtx(
            {**ctx.context, tag: mesh_points},
            ctx.var_bindings,
            ctx.key,
            active_region=ctx.active_region,
        )
        value = self._dispatch(target, new_ctx)
        # Keep the identified objects alive alongside the value so no id() in the key
        # can be recycled onto a different object while the entry is live.
        self._mesh_eval_cache[key] = (target, mesh_points, ctx.context, ctx.var_bindings, ctx.active_region, value)
        return value

    def _map_mesh_to_sampled(self, mesh_points, sampled_points, values):
        """Map values computed at mesh vertices back to sampled points via nearest-neighbour lookup.

        The nearest-vertex assignment is a STRUCTURAL map — it depends only on the two point sets, never on
        ``values`` — so it is resolved ONCE on the host (a scipy cKDTree, ``O((N+M) log N)``) and reused as a
        constant gather index. Every derivative readout at non-mesh collocation points (`u.x`, `∇u·n`, the
        Jacobian / Laplacian / Hessian) is then a `values[index]` gather, instead of rebuilding an
        ``O(N_sampled × N_mesh)`` distance matrix in the compiled graph on every call. When the sampled points
        ARE the mesh vertices (the common case) the map is the identity and ``values`` pass through unremapped.
        Dynamic (traced) coordinates — e.g. differentiating an objective w.r.t. the sample points — fall back
        to the original in-graph ``jnp.argmin`` (structural precompute needs concrete coordinates)."""
        idx = self._nn_index(mesh_points, sampled_points)
        if idx is None:  # traced coordinates: the host cannot resolve the index -> in-graph argmin
            return self._nearest_neighbour_lookup(mesh_points, sampled_points, values)
        if idx is self._NN_IDENTITY:  # sampled points == mesh vertices -> no remap
            return values
        return values[idx]

    def _nn_index(self, mesh_points, sampled_points):
        """The nearest mesh-vertex index for each sampled point, resolved on the host and cached by the two
        point sets' geometry. Returns ``None`` if either set is a JAX tracer (dynamic coordinates), the
        ``_NN_IDENTITY`` sentinel if the sampled points are the mesh vertices, else the gather index as a
        JAX array (a jaxpr constant, so it never re-runs a distance reduction at solve time)."""
        if isinstance(mesh_points, jax.core.Tracer) or isinstance(sampled_points, jax.core.Tracer):
            return None
        mp, sp = np.asarray(mesh_points), np.asarray(sampled_points)
        key = (mp.shape, sp.shape, hash(mp.tobytes()), hash(sp.tobytes()))
        cached = self._nn_index_cache.get(key)
        if cached is None:
            if mp.shape == sp.shape and np.allclose(mp, sp, rtol=0.0, atol=1e-8):
                cached = self._NN_IDENTITY  # collocation on the mesh vertices — the direct-return fast path
            else:
                from scipy.spatial import cKDTree

                cached = jnp.asarray(cKDTree(mp).query(sp)[1])  # (N_sampled,) nearest mesh-vertex indices
            if len(self._nn_index_cache) > 128:  # bound memory if the geometry changes every call (eager resampling)
                self._nn_index_cache.clear()
            self._nn_index_cache[key] = cached
        return cached

    @staticmethod
    def _nearest_neighbour_lookup(mesh_points, sampled_points, values):
        """In-graph nearest-neighbour fallback for traced coordinates: an ``O(N_sampled × N_mesh)`` distance
        reduction. The host path (:meth:`_nn_index`) avoids this whenever the coordinates are concrete."""
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

        def _broadcast_temporal(result):
            result = jnp.asarray(result)

            # Find target ndim from spatial arrays in context
            target_ndim = None
            for k, v in ctx.context.items():
                if str(k).startswith("__"):
                    continue
                if hasattr(v, "ndim") and v.ndim >= 2:
                    target_ndim = v.ndim
                    break

            if target_ndim is None:
                target_ndim = 2  # default: (N, D)

            # Pad trailing 1s so temporal matches spatial ndim.
            # E.g. spatial (N, D) → temporal (1,) becomes (1, 1);
            #      spatial (W, N, D) → temporal (W,) becomes (W, 1, 1).
            # JAX broadcasting then handles (1, 1) * (N, 1) → (N, 1)
            # in arithmetic, while models receive compact temporal input.
            while result.ndim < target_ndim:
                result = jnp.expand_dims(result, axis=-1)

            return result

        if tag in ctx.context:
            tag_data = ctx.context[tag]
            dim_start, dim_end = bound_var.dim
            result = tag_data[..., dim_start:dim_end]
            if axis == "temporal":
                return _broadcast_temporal(result)
            return result

        elif axis == "temporal" and "__time__" in ctx.context:
            tag_data = ctx.context["__time__"]
            dim_start, dim_end = bound_var.dim
            result = tag_data[..., dim_start:dim_end]
            return _broadcast_temporal(result)

        else:
            available = sorted(k for k in ctx.context.keys() if not k.startswith("__"))
            self.log.error(f"Variable tag '{tag}' not found. context: {list(ctx.context.keys())}")
            raise KeyError(
                f"Variable tag '{tag}' not found in evaluation context. "
                f"Available tags: {available}. "
                f"If this tag should exist, ensure you called domain.variable('{tag}') "
                f"before constructing the expression (variable() triggers sampling)."
            )

    def _eval_function_call(self, expr, ctx):
        args = [(self._dispatch(arg, ctx) if isinstance(arg, Placeholder) else arg) for arg in expr.args]
        kwargs = expr.kwargs if expr.kwargs else {}
        try:
            wants_key = _accepts_key(expr.fn)
        except TypeError:  # unhashable callable (e.g. a functools.partial over a dict)
            wants_key = "key" in inspect.signature(expr.fn).parameters
        if wants_key:
            return expr.fn(*args, key=ctx.key, **kwargs)
        else:
            return expr.fn(*args, **kwargs)

    def _eval_model_weights(self, expr, ctx):
        """Resolve a ``ModelWeights`` node to the model's *current* module pytree.

        Under ``crux.solve`` this is the live trainable module (``self.params`` is
        ``eqx.combine(trainable, frozen, static)`` rebuilt each step), so gradients of anything
        computed from it — e.g. a FEM solve that re-assembles with these weights — reach the
        optimizer. Outside training (eager evaluation) it falls back to the stored module.
        """
        return self.params.get(expr.model.layer_id, expr.model.module)

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

        new_ctx = self._EvalCtx(ctx.context, new_bindings, ctx.key, active_region=ctx.active_region)
        return self._dispatch(op.expr, new_ctx)

    def _eval_flax_module_call(self, expr, ctx):
        arg_values = []
        arg_sources = []

        for arg in expr.args:
            if isinstance(arg, (Placeholder, TensorTag)):
                val = self._dispatch(arg, ctx)
                arg_values.append(val)

                is_spatial = False
                if isinstance(arg, TensorTag):
                    is_spatial = getattr(val, "ndim", 0) >= 3
                if isinstance(arg, Variable):
                    bound_arg = ctx.var_bindings.get(id(arg), arg)
                    axis = getattr(bound_arg, "axis", "spatial")
                    if axis == "spatial" and bound_arg.tag in ctx.context:
                        is_spatial = True
                    elif axis == "temporal":
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
                while val.ndim > 4 and val.shape[0] == 1:
                    val = val[0]
                return val
            else:
                if val.ndim == 0:
                    return val.reshape(1)  # scalar → (1,)
                return val

        def _is_foundax_pointwise_mlp(model):
            mod_name = type(model).__module__.lower()
            cls_name = type(model).__name__.lower()

            # Do not touch operator architectures such as DeepONet.
            if "deeponet" in mod_name or "deeponet" in cls_name:
                return False

            # Foundax MLP concatenates coordinate arguments internally.
            return "mlp" in mod_name or "mlp" in cls_name

        def _broadcast_pointwise_args(args):
            arrs = [jnp.asarray(a) for a in args]

            if len(arrs) <= 1:
                return arrs

            # Use all args with a feature axis. For coordinate inputs this is
            # usually (..., 1). We broadcast only the leading axes.
            leading_shapes = []
            for a in arrs:
                if a.ndim >= 2:
                    leading_shapes.append(a.shape[:-1])

            if not leading_shapes:
                return arrs

            target_leading = jnp.broadcast_shapes(*leading_shapes)

            out = []
            for a in arrs:
                if a.ndim == 0:
                    a = a.reshape((1,) * len(target_leading) + (1,))
                elif a.ndim == 1:
                    a = a.reshape((1,) * len(target_leading) + (a.shape[0],))

                target_shape = target_leading + (a.shape[-1],)
                out.append(jnp.broadcast_to(a, target_shape))

            return out

        shaped_args = [normalize_arg(v, s) for v, s in zip(arg_values, arg_sources)]

        if _is_foundax_pointwise_mlp(model):
            shaped_args = _broadcast_pointwise_args(shaped_args)

        # Mixed precision: when a model is *explicitly* opted into a dtype via
        # Model.dtype(), cast its inputs to that dtype so it actually computes in
        # it (e.g. bf16 -> real bf16 matmuls, not bf16 storage promoted to f32).
        # Keyed on the explicit opt-in, NOT the inferred param dtype: a plain f32
        # model under jax_enable_x64 must still compute in f64 by promotion rather
        # than be silently downcast.  Applies to training and inference alike.
        _compute_dtype = getattr(flax_mod, "_dtype", None)
        if _compute_dtype is not None:
            shaped_args = [
                a.astype(_compute_dtype)
                if getattr(a, "dtype", None) is not None
                and jnp.issubdtype(a.dtype, jnp.floating)
                and a.dtype != _compute_dtype
                else a
                for a in shaped_args
            ]

        # Call equinox model directly (it IS the pytree, no init/apply split).
        # Keyed on the class's ``__call__`` so the signature lookup caches across
        # instances (the bound method is a fresh object on every attribute access).
        if _accepts_key(type(model).__call__):
            result = model(*shaped_args, key=ctx.key)
        else:
            result = model(*shaped_args)

        # Some foundation models return structured outputs (e.g. ScOTOutput,
        # tuples) instead of a raw array. Normalize to an array-like payload.
        if hasattr(result, "output"):
            result = result.output

        # A pointwise network maps coordinates -> per-point values; give it the (N, 1) channel axis the
        # downstream ops expect. A NODAL FIELD parameter (``jno.np.parameter(<symbol>)``) is that same
        # per-point field kind, so it must ALSO carry the channel axis -- otherwise a (N,) parameter
        # broadcasts against an (N, 1) coefficient field/network into a spurious (N, N) outer product (this
        # broke an eps coefficient built as ``network * f(field_parameter)`` in jno.rcwa). Only a BARE
        # parameter (``jno.np.parameter((N,))``, e.g. a trainable mesh-coordinate leaf) has no field/
        # coordinate nature, so it keeps its declared shape -- ``parameter((N,))`` evaluates to (N,), not a
        # spurious (N, 1) (needed for building a geometry functional from coordinate parameters).
        _bare_parameter = getattr(flax_mod, "_is_parameter", False) and getattr(flax_mod, "_fem_field", None) is None
        if result.ndim == 1 and result.shape[0] > 1 and not _bare_parameter:
            result = result[:, jnp.newaxis]

        return result

    def _eval_tunable_module(self, expr, ctx):
        if expr._current_instance is None:
            raise ValueError(
                f"TunableModule {expr} has no current instance.  This should be set by core.solve() before evaluation."
            )
        return self._dispatch(expr._current_instance, ctx)

    def _eval_state_field(self, expr, ctx):
        return self._dispatch(expr.expr, ctx)

    @staticmethod
    def _partition_by_path_match(model, selector):
        """Partition model using path-matching when LoRA changed the tree structure.

        Falls back to this when eqx.partition(model, selector) fails due to a
        structural mismatch between the selector (built on the pre-LoRA model) and
        the current model (with LoRALinear layers).  Selects model leaves whose
        pytree path exactly matches a True-valued path in the selector.
        """

        def _path_str(path):
            parts = []
            for k in path:
                if hasattr(k, "key"):
                    parts.append(str(k.key))
                elif hasattr(k, "idx"):
                    parts.append(str(k.idx))
                elif hasattr(k, "name"):
                    parts.append(k.name)
                else:
                    parts.append(str(k))
            return "/".join(parts)

        sel_flat, _ = jax.tree_util.tree_flatten_with_path(selector)
        true_paths = {_path_str(path) for path, val in sel_flat if val is True or (isinstance(val, bool) and val)}

        bool_tree = jax.tree_util.tree_map_with_path(
            lambda path, leaf: _path_str(path) in true_paths and isinstance(leaf, jax.Array),
            model,
        )
        import equinox as eqx

        return eqx.partition(model, bool_tree)

    def _eval_network_gradient(self, expr, ctx):
        """Compute ∂target/∂params using jax.jacrev over eqx.partition'd weights.

        Can be used both for post-training analysis via crux.eval() AND as part
        of a training loss.  When used in training the optimizer must differentiate
        through jax.jacrev, which is second-order AD — correct but expensive
        (cost scales as O(P × N × forward_cost)).  To avoid the second-order cost
        while still penalising the Jacobian, wrap the call in jax.lax.stop_gradient:

            loss = (jax.lax.stop_gradient(u.grad(net)) ** 2).mean()

        Returns (B, N, P) for scalar output, (B, N, D, P) for D-dimensional output,
        consistent with other crux.eval() results.  Access J[0] for (N, P).
        """
        import equinox as eqx

        layer_id = expr.model_node.layer_id
        current_model = self.params.get(layer_id)
        if current_model is None:
            raise ValueError(
                f"NetworkGradient: no model registered for layer_id={layer_id!r}. "
                "Make sure to call crux.eval() after crux.solve()."
            )

        # Split into differentiable leaves vs static structure.
        # selector is either None (all params) or a boolean pytree built by the
        # caller via eqx.tree_at and stored on the model with net.mask(mask).
        if expr.selector is not None:
            try:
                trainable, static = eqx.partition(current_model, expr.selector)
            except ValueError:
                # Structural mismatch — model was transformed (e.g. LoRA applied to
                # hidden layers) after the selector was built on the original model.
                # Fall back to selecting leaves by pytree path.
                trainable, static = self._partition_by_path_match(current_model, expr.selector)
        else:
            trainable, static = eqx.partition(current_model, eqx.is_array)

        def forward_fn(trainable_params):
            # Rebuild model from trainable leaves + static structure
            full_model = eqx.combine(trainable_params, static)
            # Swap in the rebuilt model and re-evaluate the full target expression
            new_params = {**self.params, layer_id: full_model}
            return TraceEvaluator(new_params)._dispatch(expr.target, ctx)

        # Infer output shape cheaply (no actual computation).
        # The compiled evaluator may wrap outputs in extra leading singleton
        # dims from the device mesh vmap: (1, N, D) instead of (N, D).
        # Strip them to recover the logical spatial shape.
        raw_out_shape = jax.eval_shape(forward_fn, trainable).shape
        logical_shape = raw_out_shape
        while len(logical_shape) > 1 and logical_shape[0] == 1:
            logical_shape = logical_shape[1:]

        # ── Scalar target (e.g. loss.mse) → gradient vector (P,) ───────────
        if len(logical_shape) == 0:
            # `jax.grad`, not `jax.jacrev`: for a scalar output the two agree exactly, but
            # jacrev still vmaps its pullback over a length-1 basis, and a target containing
            # `fem.solve()` bottoms out in `spsolve`, which has no batching rule.
            grad_pytree = jax.grad(forward_fn)(trainable)
            leaves = jax.tree_util.tree_leaves(grad_pytree)
            cols = [leaf.reshape(-1) for leaf in leaves]
            return jnp.concatenate(cols, axis=-1)  # (P,)

        N = logical_shape[0]
        D = logical_shape[1] if len(logical_shape) > 1 else 1

        # Jacobian: pytree matching trainable, each leaf L of shape S
        # becomes shape (*out_shape, *S)  →  (N, D, *S).
        #
        # Use jacfwd (forward-mode AD) instead of jacrev: jacfwd batches P
        # JVP passes via vmap internally, giving XLA a single compact program.
        # jacrev would create N separate VJP traces — catastrophic for large N
        # or when forward_fn includes high-order spatial derivatives (Laplacian).
        try:
            jac_pytree = jax.jacfwd(forward_fn)(trainable)
        except NotImplementedError as exc:
            if "atching rule" not in str(exc):
                raise
            # Forward mode vmaps P tangents through the target, and a sparse solve has no
            # batching rule for either `spsolve` or `csr_matvec`. Reverse mode would work but
            # costs one pass per OUTPUT point, which for a per-point network Jacobian is the
            # N-way blow-up jacfwd is chosen here to avoid — so refuse rather than silently
            # trade a crash for an unusable runtime.
            raise NotImplementedError(
                f"NetworkGradient: the target has {N} output points and contains a sparse "
                "solve, which has no batching rule — neither forward mode (used here) nor "
                "jax.jacrev can vmap through it. Reduce the target to a scalar (e.g. "
                "`expr.mean().grad(net)`), which takes a single reverse pass."
            ) from exc

        # Flatten all param leaves into a single (N, D, P) array, then squeeze D=1
        leaves = jax.tree_util.tree_leaves(jac_pytree)
        cols = [leaf.reshape(N, D, -1) for leaf in leaves]
        J = jnp.concatenate(cols, axis=-1)  # (N, D, P)

        return J[:, 0, :] if D == 1 else J  # (N, P) or (N, D, P)

    def _eval_noise(self, expr, ctx):
        # Infer the number of active spatial points from context
        n = ctx.context.get("__active_spatial_n__", None)
        if n is None:
            for k, v in ctx.context.items():
                if not k.startswith("__") and hasattr(v, "shape") and len(v.shape) >= 1:
                    n = v.shape[0]
                    break
            if n is None:
                n = 1

        ndim = int(expr.params.get("ndim", 1))
        shape = (n, ndim)

        if ctx.key is None:
            return jnp.zeros(shape)

        # fold_in gives each Noise node a unique, deterministic subkey derived
        # from the step key — reproducible when the solver seed is fixed.
        subkey = jax.random.fold_in(ctx.key, expr._noise_id)

        dist = expr.distribution
        if dist == "gaussian":
            return jax.random.normal(subkey, shape) * expr.params.get("std", 1.0)
        elif dist == "uniform":
            return jax.random.uniform(
                subkey,
                shape,
                minval=expr.params.get("low", -1.0),
                maxval=expr.params.get("high", 1.0),
            )
        elif dist == "laplace":
            std = expr.params.get("std", 1.0)
            # Inverse-CDF of Laplace via Uniform(-0.5, 0.5)
            u = jax.random.uniform(subkey, shape, minval=1e-6, maxval=1.0 - 1e-6) - 0.5
            return -jnp.sign(u) * jnp.log(1.0 - 2.0 * jnp.abs(u)) * std / jnp.sqrt(2.0)
        else:
            raise ValueError(
                f"Unknown noise distribution {expr.distribution!r}. Choose from: 'gaussian', 'uniform', 'laplace'."
            )

    def _eval_tunable_module_call(self, expr, ctx):
        tunable = expr.model
        if tunable._current_instance is None:
            raise ValueError("TunableModule has no current instance. This should be set by core.solve() before evaluation.")
        concrete_call = ModelCall(tunable._current_instance, expr.args)
        concrete_call.op_id = expr.op_id
        return self._dispatch(concrete_call, ctx)

    def _eval_choice(self, expr, ctx):
        idx = int(expr.selected)
        if not (0 <= idx < len(expr.options)):
            raise ValueError(f"Choice index {idx} out of range for '{expr.name}'")
        return self._dispatch(expr.options[idx], ctx)

    def _eval_frozen_field(self, expr, ctx):
        """Evaluate a :class:`~jno.trace.FrozenField` (a field pinned to known nodal ``values``) at the
        active sample points: map its per-vertex values onto the points in context for the region it was
        bound to. Its gradient (``.x`` / ``.y``) is produced by :meth:`_eval_jacobian`'s FD-over-mesh path
        (a nodal mesh field has no analytic coordinate-function to auto-differentiate)."""
        if getattr(expr, "num_components", 1) != 1:
            # A VECTOR frozen field stores (n_nodes, vec); a standalone readout maps ONE scalar per vertex.
            # It works as a coefficient inside a jno.fem form (the assembler gathers the vec-vectors); a
            # standalone .eval() of the whole vector is not wired (fail loud, not a silent reshape).
            raise NotImplementedError(
                "FrozenField.eval(): standalone readout of a VECTOR frozen field is not supported yet — a "
                "vector frozen field works as a coefficient in a jno.fem form. Read a single component."
            )
        # A caller marching a MOVING mesh supplies the nodal values and the mesh geometry through the eval
        # context instead of the graph node and the domain object. Both are otherwise host state captured
        # when the expression was built, so under a `lax.scan` a frozen field would silently keep reading
        # the SEED state on the SEED mesh -- a wrong answer with no symptom. Purely additive: absent these
        # keys, every existing `FrozenField.eval()` caller takes the original path unchanged.
        _ctx_vals = ctx.context.get("__frozen_values__")
        _ctx_vals = None if _ctx_vals is None else _ctx_vals.get(getattr(expr, "frozen_id", None))
        values = jnp.asarray(expr.values if _ctx_vals is None else _ctx_vals).reshape(-1)

        _ctx_pts = ctx.context.get("__mesh_points__")
        domain = getattr(expr, "_domain", None)
        if _ctx_pts is None and (domain is None or getattr(domain, "mesh_connectivity", None) is None):
            raise ValueError(
                "FrozenField.eval() needs the frozen field to carry its mesh domain — build it via "
                "`u.bind(x=..., y=...).freeze(values)` (the coordinate bind supplies the domain)."
            )
        if _ctx_pts is not None:
            mesh_points = jnp.asarray(_ctx_pts)
            mesh_dim = int(mesh_points.shape[-1])
        else:
            mesh_dim = int(domain.mesh_connectivity["dimension"])
            mesh_points = jnp.asarray(domain.mesh_connectivity["points"])[:, :mesh_dim]

        tag = getattr(expr, "_coord_tag", None)
        pts = ctx.context.get(tag) if tag is not None else None
        if pts is None:
            have = sorted(k for k in ctx.context if not str(k).startswith("__"))
            raise KeyError(
                f"FrozenField.eval(): the sample points for its bound region '{tag}' are not in the eval "
                f"context (have {have}); evaluate it on the region it was bound to."
            )
        pts = jnp.asarray(pts)
        # A transient domain time-batches a tag as (B, T, N, D); a frozen field is a spatial snapshot
        # (time-independent), so collapse every leading axis down to the (N, D) spatial points.
        while pts.ndim > 2:
            pts = pts[0]
        if pts.ndim == 1:
            pts = pts[jnp.newaxis, :]
        return self._map_mesh_to_sampled(mesh_points, pts[:, :mesh_dim], values)

    def _eval_normal_derivative(self, target, normal_var, scheme, ctx):
        """``∇(target)·n`` at the eval points, where ``normal_var`` is a boundary/interface normal
        (tag ``n_<region>``). The full spatial FD gradient of ``target`` is computed over the mesh and
        dotted with the node normals — the pointwise value of the normal flux. Requires both the normals
        (``ctx.context['n_<region>']``) and the points (``ctx.context['<region>']``) to be present."""
        # The normal Variable's tag may carry a trailing component suffix (``n_interface_L_R_0``) while
        # the eval context holds the base tags (``n_interface_L_R`` normals, ``interface_L_R`` points),
        # so match the normals key by prefix.
        ntag = normal_var.tag
        normal_key = next(
            (k for k in ctx.context if isinstance(k, str) and k.startswith("n_") and str(ntag).startswith(k)), None
        )
        coord_tag = normal_key[2:] if normal_key is not None else ntag[2:]
        if normal_key is None or coord_tag not in ctx.context:
            have = sorted(k for k in ctx.context if not str(k).startswith("__"))
            raise KeyError(
                f"normal-derivative eval needs the normals ('n_<region>') and points ('<region>') for "
                f"'{ntag}' in the context; have {have}."
            )
        normals = jnp.asarray(ctx.context[normal_key])
        points = jnp.asarray(ctx.context[coord_tag])
        domain = getattr(normal_var, "_domain", None)
        if domain is None or getattr(domain, "mesh_connectivity", None) is None:
            raise ValueError("normal-derivative eval requires the normal variable to carry a mesh domain.")
        mesh_points = jnp.asarray(domain.mesh_connectivity["points"])
        mesh_dim = int(domain.mesh_connectivity["dimension"])

        # nodal values of the target over the WHOLE mesh (FD stencils need the full field, not just the
        # interface nodes), then map the gradient back to the requested points.
        full_ctx = self._EvalCtx(
            {**ctx.context, coord_tag: mesh_points}, ctx.var_bindings, ctx.key, active_region=ctx.active_region
        )
        u_full = jnp.asarray(self._dispatch(target, full_ctx)).reshape(-1)

        sch = "finite_difference" if scheme_family(scheme) == "automatic_differentiation" else scheme
        _, grad_method, _ = DifferentialOperators.parse_fd_scheme(sch)
        if mesh_dim == 1:
            grads = [
                DifferentialOperators.compute_fd_gradient_1d_simple(
                    u_full, mesh_points, domain.mesh_connectivity["lines"], method=grad_method
                )
            ]
        elif mesh_dim == 2:
            cells = domain.mesh_connectivity["triangles"]
            _grid = domain.mesh_connectivity.get("grid")  # structured-grid fast path (else None)
            grads = [
                DifferentialOperators.compute_fd_gradient_2d_simple(
                    u_full, mesh_points, cells, i, method=grad_method, grid=_grid
                )
                for i in range(2)
            ]
        else:
            cells = domain.mesh_connectivity["tetrahedra"]
            _grid = domain.mesh_connectivity.get("grid")  # structured-grid fast path (else None)
            grads = [
                DifferentialOperators.compute_fd_gradient_3d_simple(
                    u_full, mesh_points, cells, i, method=grad_method, grid=_grid
                )
                for i in range(3)
            ]
        grad_full = jnp.stack([jnp.asarray(g).reshape(-1) for g in grads], axis=-1)  # (N_mesh, D)
        pts2d = points.reshape(-1, points.shape[-1])
        grad_at = self._map_mesh_to_sampled(mesh_points, pts2d, grad_full)  # (N_pts, D)
        n = normals.reshape(-1, normals.shape[-1])
        return jnp.sum(grad_at * n, axis=-1)  # (N_pts,)

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
        # A bare 'automatic_differentiation' means 'whatever the run configured'
        # (jno.setup(diff_type=...)); an explicit family:submethod passes through.
        scheme = resolve_scheme(expr.scheme)
        # A FrozenField is known nodal values on a mesh — there is no analytic coordinate-function to
        # auto-differentiate, so its spatial gradient is the FD-over-mesh gradient of those values.
        # Force FD only for AUTOMATIC differentiation, which is what is meaningless on stored nodal
        # values. Any other family keeps its own backend.
        if isinstance(target, FrozenField) and scheme_family(scheme) == "automatic_differentiation":
            scheme = "finite_difference"
        if isinstance(target, TestFunction):
            if ctx.active_region is None:
                raise ValueError(
                    "Jacobian of TestFunction requires an active_region. Use grad(phi, x) only inside Assembly(...)."
                )

            requested_dims = []
            for var in variables:
                dim_idx = 0
                if hasattr(var, "dim") and isinstance(var.dim, (list, tuple)):
                    ints = [d for d in var.dim if isinstance(d, int)]
                    if ints:
                        dim_idx = ints[0]
                requested_dims.append(dim_idx)

            support = ctx.active_region["support"]
            region_id = ctx.active_region["region_id"]
            value_shape = getattr(target, "value_shape", ())
            n_comp = self._value_shape_num_components(value_shape)

            if support == "volume":
                dN = ctx.context["dN_dx_flat"]  # (Nq_total, nloc, dim)

                if n_comp == 1:
                    comps = [dN[..., dim_idx] for dim_idx in requested_dims]
                    return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

                eye = jnp.eye(n_comp, dtype=dN.dtype)
                comps = [dN[..., dim_idx][:, :, None, None] * eye[None, None, :, :] for dim_idx in requested_dims]

                # one derivative:
                #   (Nq_total, nloc, basis_comp, phys_comp)
                # many derivatives:
                #   (Nq_total, nloc, basis_comp, phys_comp, n_vars)
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

            if support == "boundary":
                if "surface_data" not in ctx.context or region_id not in ctx.context["surface_data"]:
                    raise KeyError(f"Boundary region '{region_id}' not found in fem_context['surface_data'].")

                surf_data = ctx.context["surface_data"][region_id]

                if "face_shape_grads" not in surf_data:
                    raise NotImplementedError(
                        f"Boundary TestFunction gradients requested on region '{region_id}', "
                        "but 'face_shape_grads' is not stored in fem_context['surface_data']'. "
                        "Add boundary shape gradients in domain.init_fem() first."
                    )

                dN_face = surf_data["face_shape_grads"]
                # flatten to (Nq_total, nloc, dim)
                dN_face = dN_face.reshape(-1, dN_face.shape[-2], dN_face.shape[-1])

                if n_comp == 1:
                    comps = [dN_face[..., dim_idx] for dim_idx in requested_dims]
                    return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

                eye = jnp.eye(n_comp, dtype=dN_face.dtype)
                comps = [dN_face[..., dim_idx][:, :, None, None] * eye[None, None, :, :] for dim_idx in requested_dims]
                return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

            raise ValueError(f"Unknown active support '{support}'")

        first_var = variables[0]
        bound_var = ctx.var_bindings.get(id(first_var), first_var)
        first_axis = getattr(bound_var, "axis", "spatial")

        # ── Normal derivative: ``u.d(n)`` where ``n`` is a boundary/interface normal ──
        # (tag ``n_<region>`` from ``domain.variable(region, normals=True)``, a vector). Returns the
        # pointwise flux value ``∇(target)·n`` — distinct from the affine BC-assembly decomposition —
        # so an interface condition like ``k*uA.d(n) - k*uB.d(n)`` can be *evaluated* at the interface
        # nodes given each subdomain's computed nodal field (the coupling residual for an interface solve).
        _ndim = getattr(bound_var, "dim", None)
        if (
            len(variables) == 1
            and isinstance(getattr(bound_var, "tag", None), str)
            and bound_var.tag.startswith("n_")
            and isinstance(_ndim, (list, tuple))
            and len(_ndim) == 2
            and (_ndim[1] - _ndim[0]) >= 2  # a normal is a vector; a coordinate spans one component
        ):
            return self._eval_normal_derivative(target, bound_var, scheme, ctx)

        # ── Temporal derivative ──
        if first_axis == "temporal":
            evaluator_self = self

            def is_time_tag(tag_name):
                s = str(tag_name)
                return s == "__time__" or s.startswith("__time")

            def spatial_tag_for_time_key(tkey):
                s = str(tkey)
                if s == "__time__":
                    return None
                if s.startswith("__time_") and s.endswith("__"):
                    return s[len("__time_") : -2]
                return None

            # Use the actual temporal tag of the differentiated variable if present
            active_time_key = getattr(bound_var, "tag", "__time__")
            if active_time_key not in ctx.context:
                active_time_key = "__time__"

            active_spatial_tag = spatial_tag_for_time_key(active_time_key)

            time_arr = jnp.asarray(ctx.context[active_time_key])
            time_dtype = time_arr.dtype
            time_scalar0 = jnp.reshape(time_arr, (-1,))[0]

            if scheme == "finite_difference":
                eps = jnp.asarray(1e-3, dtype=_default_float_dtype())

                def _set_active_time_tags(base_ctx, t_scalar):
                    t_box = jnp.asarray([[t_scalar]], dtype=time_dtype)
                    out = dict(base_ctx)

                    # keep the global time consistent
                    out["__time__"] = t_box
                    # and also the specific active time tag
                    out[active_time_key] = t_box
                    return out

                ctx_fwd = _set_active_time_tags(ctx.context, time_scalar0 + eps)
                ctx_bwd = _set_active_time_tags(ctx.context, time_scalar0 - eps)

                u_fwd = self._dispatch(
                    target,
                    self._EvalCtx(
                        ctx_fwd,
                        ctx.var_bindings,
                        ctx.key,
                        active_region=ctx.active_region,
                    ),
                )
                u_bwd = self._dispatch(
                    target,
                    self._EvalCtx(
                        ctx_bwd,
                        ctx.var_bindings,
                        ctx.key,
                        active_region=ctx.active_region,
                    ),
                )

                result = (u_fwd - u_bwd) / (2.0 * eps)
                if result.ndim == 1:
                    result = result[:, jnp.newaxis]
                return result

            # ── Temporal derivative via AD (default) ──
            #
            # Supports both:
            #   grad(u, t)                  -> first temporal derivative
            #   grad(grad(u, t), t)         -> second temporal derivative
            #
            # Important:
            # For the nested case, do NOT evaluate the inner Jacobian as the
            # target of an outer jax.grad. That can return a vector over all
            # spatial points and causes:
            #   "Temporal derivative expected scalar output per point"
            #
            # Instead, collapse the nested temporal Jacobian into one scalar
            # function u_i(t) per spatial point and apply grad(grad(...)).
            domain = getattr(bound_var, "_domain", None)
            param_tags = set(getattr(domain, "_param_tags", set())) if domain is not None else set()

            def is_spatial_pointset(tag_name, value):
                if is_time_tag(tag_name):
                    return False
                if tag_name in param_tags:
                    return False
                if not hasattr(value, "ndim"):
                    return False
                return value.ndim >= 2

            def point_axis(value):
                # During TraceCompiler time-window evaluation, spatial arrays
                # are typically (W, N, D). For steady/single-time local contexts,
                # they are often (N, D). In both cases the point axis is ndim - 2.
                return value.ndim - 2

            def _is_temporal_variable(v):
                bv = ctx.var_bindings.get(id(v), v)
                return isinstance(bv, Variable) and getattr(bv, "axis", None) == "temporal"

            # Detect nested temporal derivative:
            #   Jacobian(Jacobian(u, [t]), [t])
            temporal_derivative_order = 1
            base_target = target

            if isinstance(target, Jacobian) and len(getattr(target, "variables", [])) == 1:
                inner_var = target.variables[0]
                if _is_temporal_variable(inner_var):
                    temporal_derivative_order = 2
                    base_target = target.target

            # Determine active N from the matching spatial tag first.
            N = int(ctx.context.get("__active_spatial_n__", 1))
            if (
                active_spatial_tag is not None
                and active_spatial_tag in ctx.context
                and is_spatial_pointset(active_spatial_tag, ctx.context[active_spatial_tag])
            ):
                v = ctx.context[active_spatial_tag]
                N = int(v.shape[point_axis(v)])
            else:
                # Scope N to the expression's own spatial tags so that tags
                # with different point counts (e.g. "initial" vs "interior")
                # don't bleed into each other's temporal-derivative vmaps.
                # Reset to 1 rather than using __active_spatial_n__: the
                # compiler sets that from the first alphabetical spatial tag,
                # which may differ from this expression's actual tag.
                expr_spatial_tags = [
                    t for t in collect_tags(base_target) if t in ctx.context and is_spatial_pointset(t, ctx.context[t])
                ]
                source_tags = (
                    expr_spatial_tags
                    if expr_spatial_tags
                    else [k for k, v in ctx.context.items() if is_spatial_pointset(k, v)]
                )
                N = 1
                for t in source_tags:
                    v = ctx.context[t]
                    N = max(N, int(v.shape[point_axis(v)]))

            def _set_active_time_tags(base_ctx, t_scalar):
                t_box = jnp.asarray([[t_scalar]], dtype=time_dtype)
                out = dict(base_ctx)

                # Always keep the global time coherent.
                out["__time__"] = t_box

                # Also keep the specific active time tag coherent.
                out[active_time_key] = t_box
                return out

            def _scalar_from_point_output(out):
                out = jnp.asarray(out)
                out = jnp.squeeze(out)

                if out.ndim == 0:
                    return out

                # Accept any single-entry shape, e.g. (1,), (1,1), (1,1,1).
                if out.size == 1:
                    return jnp.reshape(out, (-1,))[0]

                raise ValueError(f"Temporal derivative expected scalar output per point, got shape {out.shape}")

            def _local_context_for_point(idx):
                local_ctx = {"__active_spatial_n__": 1}

                for k, v in ctx.context.items():
                    # Skip time tags; they are rebuilt for each t_scalar.
                    if is_time_tag(k):
                        continue

                    if is_spatial_pointset(k, v):
                        ax = point_axis(v)

                        should_slice = False

                        # If the temporal tag is tied to a specific spatial tag,
                        # slice that tag.
                        if active_spatial_tag is not None and k == active_spatial_tag:
                            should_slice = True

                        # If the temporal tag is the global "__time__", slice all
                        # spatial point sets whose point axis matches N.
                        elif active_spatial_tag is None and int(v.shape[ax]) == N:
                            should_slice = True

                        if should_slice:
                            start_indices = [0] * v.ndim
                            slice_sizes = list(v.shape)
                            start_indices[ax] = idx
                            slice_sizes[ax] = 1

                            local_ctx[k] = jax.lax.dynamic_slice(
                                v,
                                tuple(start_indices),
                                tuple(slice_sizes),
                            )
                        else:
                            local_ctx[k] = v
                    else:
                        local_ctx[k] = v

                return local_ctx

            def temporal_derivative_single_point(idx):
                local_ctx = _local_context_for_point(idx)

                def u_of_t_scalar(t_scalar):
                    new_ctx_dict = _set_active_time_tags(local_ctx, t_scalar)
                    new_ctx = evaluator_self._EvalCtx(
                        new_ctx_dict,
                        ctx.var_bindings,
                        ctx.key,
                        active_region=ctx.active_region,
                    )

                    out = evaluator_self._dispatch(base_target, new_ctx)
                    return _scalar_from_point_output(out)

                if temporal_derivative_order == 1:
                    _grad = ad_fn(parse_ad_scheme(scheme))
                    return _grad(u_of_t_scalar)(time_scalar0)

                if temporal_derivative_order == 2:
                    _outer, _inner = parse_hessian_scheme(scheme)
                    return ad_fn(_outer)(ad_fn(_inner)(u_of_t_scalar))(time_scalar0)

                _grad = ad_fn(parse_ad_scheme(scheme))
                fn = u_of_t_scalar
                for _ in range(temporal_derivative_order):
                    fn = _grad(fn)
                return fn(time_scalar0)

            result = jax.vmap(temporal_derivative_single_point)(jnp.arange(N))
            return result[:, jnp.newaxis]

        # ── Spatial derivative ──
        tag = bound_var.tag
        points = ctx.context[bound_var.tag]
        while hasattr(points, "ndim") and points.ndim > 2 and points.shape[0] == 1:
            points = jnp.squeeze(points, axis=0)
        # The gradient of a FrozenField is a spatial functional; on a transient domain its target points
        # are time-batched (B, T, N, D) — collapse the leading axes to the (N, D) spatial points so the
        # mesh→points map is not entangled with the time scan (which silently collapses it to a constant).
        if isinstance(target, FrozenField):
            while hasattr(points, "ndim") and points.ndim > 2:
                points = points[0]
        n_vars = len(variables)
        var_dims = [(i, vi.dim[0]) for i, vi in enumerate(variables)]

        # Ensure points is 2D (N, D) — after vmap it may be 1D (D,)
        if points.ndim == 1:
            points = points[jnp.newaxis, :]

        # Resolving the family raises on an unknown one -- the `else` this chain never had. An
        # unrecognised scheme used to fall off the end of the function and return None, surfacing
        # much later as `TypeError: 'NoneType' object is not subscriptable`.
        family = scheme_family(scheme)
        if family in _MESH_FIELD_FAMILIES:
            # Every family that differentiates STORED VALUES on the mesh shares this shape: evaluate
            # the target on the mesh, flatten an operator's image-shaped output, differentiate each
            # channel, put the shape back. Only the per-channel kernel differs, so that is the only
            # thing a new family supplies -- see `_MESH_FIELD_FAMILIES`.
            mesh, u_flat, image_shape, n_channels = self._mesh_field_values(target, tag, bound_var, ctx, family)
            kernel = _MESH_FIELD_FAMILIES[family].gradient(mesh, scheme)
            comps = [
                jax.vmap(lambda u_c, _a=vi_dim: kernel(u_c, _a))(u_flat.T).T
                if (image_shape is not None and n_channels > 1)
                else kernel(u_flat, vi_dim)
                for _i, vi_dim in var_dims
            ]
            return self._finish_mesh_jacobian(comps, image_shape, n_vars, mesh.points, points)
        elif family == "automatic_differentiation":
            evaluator_self = self
            _jac = ad_fn(parse_ad_scheme(scheme))

            def make_u_fn(local_ctx):
                def u_fn(p):
                    ctx_dict = {**local_ctx, tag: p[jnp.newaxis, :]}
                    new_ctx = evaluator_self._EvalCtx(
                        ctx_dict,
                        ctx.var_bindings,
                        ctx.key,
                        active_region=ctx.active_region,
                    )
                    return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                return u_fn

            if n_vars == 1:
                dim = var_dims[0][1]

                def jac_single(idx):
                    pt = jax.lax.dynamic_slice(points, (idx, 0), (1, points.shape[1]))[0]
                    local_ctx = evaluator_self._build_local_context(idx, tag, points, ctx.context)
                    u_fn = make_u_fn(local_ctx)

                    # Only the RANK of the output decides how the Jacobian is sliced
                    # below, so ask for the shape abstractly — calling u_fn here would
                    # trace the whole target a second time per derivative node.
                    val0 = jax.eval_shape(u_fn, pt)
                    jac = _jac(u_fn)(pt)

                    # scalar output -> shape (1,)
                    if jnp.ndim(val0) == 0:
                        return jnp.asarray(jac[dim])[jnp.newaxis]

                    # vector/tensor output -> keep output shape, select derivative dim
                    return jac[..., dim]

                return jax.vmap(jac_single)(jnp.arange(points.shape[0]))

            else:

                def jac_single(idx):
                    pt = jax.lax.dynamic_slice(points, (idx, 0), (1, points.shape[1]))[0]
                    local_ctx = evaluator_self._build_local_context(idx, tag, points, ctx.context)
                    u_fn = make_u_fn(local_ctx)

                    val0 = jax.eval_shape(u_fn, pt)  # rank only — see the n_vars == 1 branch
                    jac = _jac(u_fn)(pt)

                    # scalar output -> (n_vars,)
                    if jnp.ndim(val0) == 0:
                        return jnp.stack([jac[vi_dim] for _, vi_dim in var_dims], axis=-1)

                    # vector/tensor output:
                    # jac shape is value_shape + (input_dim,)
                    # collect requested derivative directions on the last axis
                    comps = [jac[..., vi_dim] for _, vi_dim in var_dims]
                    return comps[0] if len(comps) == 1 else jnp.stack(comps, axis=-1)

                return jax.vmap(jac_single)(jnp.arange(points.shape[0]))

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
        # A bare 'automatic_differentiation' means 'whatever the run configured'
        # (jno.setup(diff_type=...)); an explicit family:submethod passes through.
        scheme = resolve_scheme(expr.scheme)
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

        family = scheme_family(scheme)
        if family in _MESH_FIELD_FAMILIES:
            backend = _MESH_FIELD_FAMILIES[family]
            mesh, u_flat, image_shape, n_channels = self._mesh_field_values(target, tag, bound_var, ctx, family)
            multi = image_shape is not None and n_channels > 1

            if compute_trace:
                lap = backend.laplacian(mesh, scheme, dims)
                lap_full = jax.vmap(lap)(u_flat.T).T if multi else lap(u_flat)
                if image_shape is not None:
                    return lap_full.reshape(image_shape)
                if points is not None:
                    result = self._map_mesh_to_sampled(mesh.points, points, lap_full)
                    return result[:, jnp.newaxis] if result.ndim == 1 else result
                return lap_full[:, jnp.newaxis] if lap_full.ndim == 1 else lap_full

            hess = backend.hessian(mesh, scheme, var_dims)
            hess_full = jnp.moveaxis(jax.vmap(hess)(u_flat.T), 0, 1) if multi else hess(u_flat)
            if image_shape is not None:
                if n_channels > 1:
                    return hess_full.reshape(*image_shape[:-1], n_channels, n, n)
                return hess_full.reshape(*image_shape[:-1], n, n)
            return self._map_mesh_to_sampled(mesh.points, points, hess_full)
        elif family == "automatic_differentiation":
            evaluator_self = self
            _outer_mode, _inner_mode = parse_hessian_scheme(scheme)
            _outer = ad_fn(_outer_mode)
            _inner = ad_fn(_inner_mode)

            def _hess(f):
                return _outer(_inner(f))

            if points is not None and points.ndim == 3:
                n_time, n_points, _ = points.shape

                def _windowed_scalar(t_idx, p_idx, p):
                    """``target`` at the single ``(t_idx, p_idx)`` entry, as a function of ``p``.

                    The context is sliced down to that one entry — mirroring
                    :meth:`_build_local_context` for the 2-D point set, but peeling
                    the (time, point) pair — so the target is evaluated once per
                    derivative instead of over the whole window.

                    Restricting the point set this way assumes a *pointwise* target,
                    which is what the AD derivative paths require throughout (the
                    2-D branches below evaluate one point at a time for the same
                    reason).  Evaluating the full window and reading one entry back
                    out would keep a non-local target honest, but costs a forward
                    pass over all ``n_time * n_points`` entries for every one of the
                    ``n_time * n_points`` derivatives — quadratic in the mesh.
                    """
                    local = {"__active_spatial_n__": 1}
                    for k, v in ctx.context.items():
                        if k == "__active_spatial_n__":
                            continue
                        if isinstance(v, dict) or not hasattr(v, "ndim"):
                            local[k] = v
                        elif v.ndim >= 3 and v.shape[0] == n_time and v.shape[1] == n_points:
                            # a (T, N, ...) point set — take the single (t, p) entry
                            local[k] = jax.lax.dynamic_slice(
                                v,
                                (t_idx, p_idx) + (0,) * (v.ndim - 2),
                                (1, 1) + tuple(v.shape[2:]),
                            )
                        elif k != tag and v.ndim >= 1 and v.shape[0] == n_time:
                            # a per-timestep entry (e.g. __time__ at (T, 1))
                            local[k] = jax.lax.dynamic_slice(v, (t_idx,) + (0,) * (v.ndim - 1), (1,) + tuple(v.shape[1:]))
                        else:
                            local[k] = v
                    new_ctx = evaluator_self._EvalCtx(
                        {**local, tag: p[jnp.newaxis, jnp.newaxis, :]},
                        ctx.var_bindings,
                        ctx.key,
                        active_region=ctx.active_region,
                    )
                    return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                if compute_trace:

                    def lap_time_point(t_idx, p_idx):
                        pt = points[t_idx, p_idx]
                        hess = _hess(lambda p: _windowed_scalar(t_idx, p_idx, p))(pt)
                        return sum(hess[d, d] for d in dims)

                    result = jax.vmap(
                        lambda t_idx: jax.vmap(lambda p_idx: lap_time_point(t_idx, p_idx))(jnp.arange(n_points))
                    )(jnp.arange(n_time))
                    return result[..., jnp.newaxis]

                def hess_time_point(t_idx, p_idx):
                    pt = points[t_idx, p_idx]
                    hess = _hess(lambda p: _windowed_scalar(t_idx, p_idx, p))(pt)
                    result = jnp.zeros((n, n))
                    for i, vi_dim, j, vj_dim in var_dims:
                        result = result.at[i, j].set(hess[vi_dim, vj_dim])
                    return result

                return jax.vmap(lambda t_idx: jax.vmap(lambda p_idx: hess_time_point(t_idx, p_idx))(jnp.arange(n_points)))(
                    jnp.arange(n_time)
                )

            if compute_trace:
                # Laplacian via AD
                def lap_single(idx):
                    pt = jax.lax.dynamic_slice(points, (idx, 0), (1, points.shape[1]))[0]
                    local_ctx = evaluator_self._build_local_context(idx, tag, points, ctx.context)

                    def u_scalar(p):
                        ctx_dict = {**local_ctx, tag: p[jnp.newaxis, :]}
                        new_ctx = evaluator_self._EvalCtx(
                            ctx_dict,
                            ctx.var_bindings,
                            ctx.key,
                            active_region=ctx.active_region,
                        )
                        return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                    hess = _hess(u_scalar)(pt)
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
                            active_region=ctx.active_region,
                        )
                        return jnp.squeeze(evaluator_self._dispatch(target, new_ctx))

                    hess = _hess(u_scalar)(pt)
                    result = jnp.zeros((n, n))
                    for i, vi_dim, j, vj_dim in var_dims:
                        result = result.at[i, j].set(hess[vi_dim, vj_dim])
                    return result

                return jax.vmap(hess_single)(points)

    def _eval_integral(self, expr: "Integral", ctx):
        """Evaluate a mesh-based integral reduction.

        **Scalar path** (``expr.integration_var is None``): walks the tree for
        the first spatial Variable, evaluates the integrand at all mesh nodes of
        that region, and returns a weighted sum (scalar).

        **Vectorized path** (``expr.integration_var = x``): ``x`` is the outer
        collocation Variable.  All other spatial Variables in the expression are
        aliased to a fresh integration mesh via ``var_bindings``, then
        ``jax.vmap`` evaluates the weighted sum for every outer point, returning
        ``(N, 1)`` — a function of ``x``.  No special flag on
        ``domain.variable()`` is needed; just call it twice::

            x, _ = domain.variable("interior")
            t, _ = domain.variable("interior")
            integral = (K(x, t) * net(t)).integrate(var=x)
        """
        if expr.integration_var is not None:
            return self._eval_integral_vectorized(expr, ctx)

        # ── Scalar path ────────────────────────────────────────────────────────
        first_spatial_var = None

        def _find_spatial(node):
            nonlocal first_spatial_var
            if first_spatial_var is not None:
                return
            if isinstance(node, Variable) and getattr(node, "axis", "spatial") == "spatial":
                bound = ctx.var_bindings.get(id(node), node)
                b_tag = getattr(bound, "tag", "")
                if getattr(bound, "_domain", None) is not None and not b_tag.startswith("n_"):
                    first_spatial_var = bound
                    return
            for attr in ("target", "left", "right"):
                child = getattr(node, attr, None)
                if isinstance(child, Placeholder):
                    _find_spatial(child)
            for attr in ("args", "variables"):
                for child in getattr(node, attr, []):
                    if isinstance(child, Placeholder):
                        _find_spatial(child)

        _find_spatial(expr.target)

        if first_spatial_var is None:
            # Fall back: accept normal vars too (e.g. nx.integrate())
            def _find_any_spatial(node):
                nonlocal first_spatial_var
                if first_spatial_var is not None:
                    return
                if isinstance(node, Variable) and getattr(node, "axis", "spatial") == "spatial":
                    bound = ctx.var_bindings.get(id(node), node)
                    if getattr(bound, "_domain", None) is not None:
                        first_spatial_var = bound
                        return
                for attr in ("target", "left", "right"):
                    child = getattr(node, attr, None)
                    if isinstance(child, Placeholder):
                        _find_any_spatial(child)
                for attr in ("args", "variables"):
                    for child in getattr(node, attr, []):
                        if isinstance(child, Placeholder):
                            _find_any_spatial(child)

            _find_any_spatial(expr.target)

        if first_spatial_var is None:
            raise ValueError(
                "Integral: no spatial Variable with a domain found in expression. "
                "Make sure to call .integrate() on an expression containing spatial variables."
            )

        raw_tag = first_spatial_var.tag
        tag = raw_tag[2:] if raw_tag.startswith("n_") else raw_tag
        domain = first_spatial_var._domain
        mc = domain.mesh_connectivity

        if mc is None:
            raise ValueError(
                f"Integral: domain for tag '{tag}' has no mesh_connectivity. "
                "Mesh-based integration requires an unstructured mesh domain."
            )

        cached = self._get_integral_cache(domain, tag, mc, getattr(expr, "quadrature", "nodal"))
        is_boundary = cached["is_boundary"]
        all_region_pts = jnp.array(cached["region_pts"])
        weights = jnp.array(cached["weights"])

        normal_tag = f"n_{tag}"
        new_ctx_dict = dict(ctx.context)
        new_ctx_dict[tag] = all_region_pts
        if raw_tag != tag:
            new_ctx_dict[raw_tag] = new_ctx_dict.get(raw_tag, all_region_pts)
        if is_boundary and cached["normals"] is not None:
            new_ctx_dict[normal_tag] = jnp.array(cached["normals"])

        new_ctx = self._EvalCtx(
            new_ctx_dict,
            ctx.var_bindings,
            ctx.key,
            active_region=ctx.active_region,
        )

        u_full = self._dispatch(expr.target, new_ctx)

        if u_full.ndim > 1 and u_full.shape[-1] == 1:
            u_full = u_full.squeeze(-1)

        if u_full.ndim != 1:
            raise ValueError(
                f"Integral: inner expression must be scalar-valued (shape (N,)) after squeezing, "
                f"got shape {u_full.shape}. Compute F·n explicitly before calling .integrate()."
            )

        return jnp.sum(u_full * weights)

    def _eval_integral_time(self, expr: "IntegralTime", ctx):
        """Trapezoidal time integral over the current window.

        Reads ``context["__time_window__"]`` (shape ``(W, 1)``) injected by the
        compiler's ``eval_window`` into ``passive_ctx`` before the per-step vmap.
        Evaluates ``expr.target`` at each of the W time values with the current
        spatial context, applies trapezoidal weights, and reduces over the time
        axis.

        Requires ``W >= 2`` — the solve()-level guard ensures this when
        ``min_consecutive < 2`` is passed with an ``IntegralTime`` constraint.
        """
        time_window_key = "__time_window__"
        if time_window_key not in ctx.context:
            raise RuntimeError(
                "IntegralTime: '__time_window__' not found in context. "
                "This key should be injected by the compiler before the per-step vmap. "
                "Ensure you are using a time-dependent domain and solve() with min_consecutive >= 2."
            )

        time_window = jnp.asarray(ctx.context[time_window_key])  # (W, 1)
        t_vals = time_window[:, 0]  # (W,)

        # Trapezoidal weights: w[0] = dt0/2, w[i] = (dt_{i-1} + dt_i)/2, w[-1] = dt_{W-2}/2
        dt = jnp.diff(t_vals)  # (W-1,)
        w = jnp.zeros_like(t_vals)
        w = w.at[:-1].add(dt / 2)
        w = w.at[1:].add(dt / 2)

        def eval_at_t(t_i):
            new_ctx_dict = dict(ctx.context)
            new_ctx_dict["__time__"] = t_i[jnp.newaxis]  # (1,) to match single-step context
            new_ctx = self._EvalCtx(new_ctx_dict, ctx.var_bindings, ctx.key, ctx.active_region)
            return self._dispatch(expr.target, new_ctx)

        results = jax.vmap(eval_at_t)(t_vals)  # (W, *inner_shape)

        # Broadcast w to match results shape and sum over time axis
        w_idx = (slice(None),) + (None,) * (results.ndim - 1)
        return jnp.sum(results * w[w_idx], axis=0)

    def _eval_temporal_derivative(self, expr: "TemporalDerivative", ctx):
        """First-order time derivative via cross-step finite differences.

        The compiler pre-computes ``expr.target`` on all W consecutive time
        steps before the per-step vmap and injects the resulting window into
        ``ctx["__temporal_fd_cache__"][id(expr.target)]``.  Here we read that
        window, identify the current step via ``ctx["__step_index__"]``, and
        return a clamped central difference.  At the interior of the window
        this is a 2-point central diff ``(u[i+1] - u[i-1]) / (2*dt)``; at the
        first/last step the indices clamp to the boundary, yielding a one-
        sided forward/backward diff ``(u[1] - u[0]) / dt`` or its mirror.

        Raises if the compiler did not populate the cache (i.e.
        ``min_consecutive`` is missing or ``< 2`` for the current solve).
        """
        cache = ctx.context.get("__temporal_fd_cache__")
        t_window = ctx.context.get("__time_window__")
        step_idx = ctx.context.get("__step_index__")

        if cache is None or t_window is None or step_idx is None:
            raise RuntimeError(
                "TemporalDerivative: temporal-FD cache not populated. "
                "Use crux.solve(..., min_consecutive>=2) (>=3 recommended) "
                "on a time-dependent domain to enable cross-step temporal FD."
            )

        tid = id(expr.target)
        if tid not in cache:
            raise RuntimeError(
                "TemporalDerivative: target was not pre-computed by the compiler. "
                "This indicates a bug in trace traversal — please report it."
            )

        u_window = cache[tid]  # (W, ...)
        W = u_window.shape[0]
        if W < 2:
            raise RuntimeError(
                f"TemporalDerivative requires window size W >= 2 but got W={W}. Pass min_consecutive >= 2 to crux.solve()."
            )

        step_idx = jnp.asarray(step_idx, dtype=jnp.int32)
        i_prev = jnp.maximum(step_idx - 1, 0)
        i_next = jnp.minimum(step_idx + 1, W - 1)
        t_prev = t_window[i_prev, 0]
        t_next = t_window[i_next, 0]
        dt_eff = t_next - t_prev  # 2*dt interior, dt at edge
        u_prev = u_window[i_prev]
        u_next = u_window[i_next]
        return (u_next - u_prev) / dt_eff

    @staticmethod
    def _get_integral_cache(domain, tag: str, mc: dict, quadrature: "str | int" = "nodal") -> dict:
        """Populate and return the per-(tag, quadrature) integration region cache on *domain*."""
        if not hasattr(domain, "_integral_region_cache"):
            domain._integral_region_cache = {}

        cache_key = (tag, quadrature)
        if cache_key not in domain._integral_region_cache:
            global_bi = np.asarray(mc["boundary_indices"])
            is_boundary = False
            if tag in domain._boundary_registry:
                reg_pts_idx = np.asarray(domain._boundary_registry[tag]["point_indices"])
                if len(reg_pts_idx) > 0:
                    n_on_bnd = int(np.sum(np.isin(reg_pts_idx, global_bi)))
                    is_boundary = n_on_bnd / len(reg_pts_idx) > 0.5

            use_gauss = quadrature != "nodal"
            if use_gauss and is_boundary:
                raise NotImplementedError(
                    f"Gauss-quadrature integration (quadrature={quadrature!r}) is volume-only; "
                    f"region '{tag}' is a boundary. Use the default nodal rule for boundary integrals."
                )
            gauss_degree = 4 if quadrature == "gauss" else quadrature

            if is_boundary:
                reg_indices = np.asarray(domain._boundary_registry[tag]["point_indices"])
                region_pts_np = np.asarray(mc["points"])[reg_indices]
                # A boundary face that carries its own edge subset (a Shape/CSG source-edge tag such
                # as ``boundary_chamber_0``) shares its two end corners with the perpendicular faces.
                # The global ``nodal_ds`` lumps a corner's measure over *all* boundary edges meeting at
                # that node, so counting the corner in this face over-attributes its perpendicular-edge
                # share (one mesh segment) to this face's length. Recompute the nodal measure from just
                # this tag's own edges -- mirroring the interior-subregion branch below -- so a shared
                # corner contributes only its own-face half. (Corners must stay in ``tag_indices`` for
                # multidirectional periodicity; only the integration weight needs the per-face split.)
                tag_edges = getattr(domain, "_tag_edges", {}).get(tag)
                if tag_edges is not None and len(tag_edges) > 0:
                    points_arr = np.asarray(mc["points"])
                    ds_local = np.zeros(points_arr.shape[0], dtype=np.float64)
                    te = np.asarray(tag_edges)
                    seg = np.linalg.norm(points_arr[te[:, 1]] - points_arr[te[:, 0]], axis=1)
                    np.add.at(ds_local, te[:, 0], 0.5 * seg)
                    np.add.at(ds_local, te[:, 1], 0.5 * seg)
                    weights_np = ds_local[reg_indices]
                else:
                    weights_np = np.asarray(mc["nodal_ds"])[reg_indices]
                normals_np = None
                stored = getattr(domain, "normals_by_tag", {}).get(tag)
                if stored is not None:
                    normals_np = np.asarray(stored)
            else:
                tag_tris = getattr(domain, "_tag_triangles", {}).get(tag)
                full_tris = mc.get("triangles") if hasattr(mc, "get") else None
                has_subregion = (
                    tag_tris is not None and full_tris is not None and len(tag_tris) > 0 and len(tag_tris) < len(full_tris)
                )
                if use_gauss:
                    # Element Gauss quadrature: physical quad points + JxW weights over this
                    # region's cells (all cells, or a sub-region's triangle subset).
                    sub_cells = np.asarray(tag_tris) if has_subregion else None
                    region_pts_np, weights_np = IntegrationOperators.gauss_points_and_weights(
                        mc, degree=gauss_degree, cells=sub_cells
                    )
                    normals_np = None
                elif has_subregion:
                    # Non-boundary tag with its own triangle subset (e.g.
                    # ``interior_<name>`` on a CSG mesh): compute weights
                    # from just those triangles so disjoint sub-regions
                    # don't share nodal volume from each other's incident
                    # elements.
                    tris = np.asarray(tag_tris)
                    points_arr = np.asarray(mc["points"])
                    vols = np.zeros(points_arr.shape[0], dtype=np.float64)
                    a = points_arr[tris[:, 0]]
                    b = points_arr[tris[:, 1]]
                    c = points_arr[tris[:, 2]]
                    areas = 0.5 * np.abs(
                        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
                    )
                    share = areas / 3.0
                    np.add.at(vols, tris[:, 0], share)
                    np.add.at(vols, tris[:, 1], share)
                    np.add.at(vols, tris[:, 2], share)
                    reg_indices = np.unique(tris.flatten()).astype(int)
                    region_pts_np = points_arr[reg_indices]
                    weights_np = vols[reg_indices]
                else:
                    region_pts_np = np.asarray(mc["points"])
                    weights_np = IntegrationOperators.nodal_volumes(mc)
                normals_np = None

            domain._integral_region_cache[cache_key] = {
                "is_boundary": is_boundary,
                "region_pts": region_pts_np,
                "weights": weights_np,
                "normals": normals_np,
            }

        return domain._integral_region_cache[cache_key]

    def _eval_integral_vectorized(self, expr, ctx):
        """Vectorized integral — result is (N_outer, 1), a function of the outer variable.

        ``expr.integration_var`` is the outer (collocation) Variable.  All
        other spatial Variables in the expression are aliased to the full
        integration mesh via ``var_bindings`` so the two sets of points live in
        separate context slots, even when they share the same tag.
        ``jax.vmap`` then evaluates the weighted sum for every outer point.
        """
        import types

        outer_var = expr.integration_var
        outer_tag = outer_var.tag
        outer_id = id(outer_var)

        # Walk the expression tree, collect (node, bound_var) for all spatial vars.
        seen: list[tuple] = []  # (node, bound_var)

        def _collect(node):
            if isinstance(node, Variable) and getattr(node, "axis", "spatial") == "spatial":
                bound = ctx.var_bindings.get(id(node), node)
                b_tag = getattr(bound, "tag", "")
                if getattr(bound, "_domain", None) is not None and not b_tag.startswith("n_"):
                    seen.append((node, bound))
            for attr in ("target", "left", "right"):
                child = getattr(node, attr, None)
                if isinstance(child, Placeholder):
                    _collect(child)
            for attr in ("args", "variables"):
                for child in getattr(node, attr, []):
                    if isinstance(child, Placeholder):
                        _collect(child)

        _collect(expr.target)

        # Split into outer entries (matching integration_var by id) and inner.
        inner_entries = [(node, bound) for node, bound in seen if id(bound) != outer_id]

        if not inner_entries:
            raise ValueError(
                "integrate(var=x): no inner variable found in the expression. "
                "Call domain.variable(tag) a second time to create the dummy integration "
                "variable and use it in the integrand."
            )

        # Determine the integration region from the first inner variable.
        _, inner_var = inner_entries[0]
        inner_tag = inner_var.tag
        domain = inner_var._domain
        mc = domain.mesh_connectivity

        if mc is None:
            raise ValueError(f"integrate(var=x): domain for inner tag '{inner_tag}' has no mesh_connectivity.")

        cached = self._get_integral_cache(domain, inner_tag, mc)
        all_iv_pts = jnp.array(cached["region_pts"])  # (N_iv, D)
        weights = jnp.array(cached["weights"])  # (N_iv,)

        # Create a lightweight alias that redirects inner Variable reads to the
        # integration mesh (alias_tag), leaving the outer tag free for vmap.
        alias_tag = f"__jno_iv_{inner_tag}__"
        alias_bindings: dict = {}
        for node, bound in inner_entries:
            alias = types.SimpleNamespace(tag=alias_tag, dim=bound.dim, axis="spatial")
            alias_bindings[id(node)] = alias  # keyed by node id, as _eval_variable expects

        outer_pts = jnp.array(ctx.context[outer_tag])  # (N_outer, D)
        evaluator_self = self

        def integral_at(outer_pt):
            new_ctx_dict = dict(ctx.context)
            new_ctx_dict[outer_tag] = outer_pt[None, :]  # (1, D) — single outer point
            new_ctx_dict[alias_tag] = all_iv_pts  # (N_iv, D) — full integration mesh
            new_bindings = {**ctx.var_bindings, **alias_bindings}
            new_ctx = evaluator_self._EvalCtx(
                new_ctx_dict,
                new_bindings,
                ctx.key,
                active_region=ctx.active_region,
            )
            u_full = evaluator_self._dispatch(expr.target, new_ctx)
            if u_full.ndim > 1 and u_full.shape[-1] == 1:
                u_full = u_full.squeeze(-1)
            return jnp.sum(u_full * weights)

        result = jax.vmap(integral_at)(outer_pts)  # (N_outer,)
        return result[:, None]  # (N_outer, 1) — matches standard jno shape

    def _eval_operation_def(self, expr, ctx):
        return self._dispatch(expr.expr, ctx)

    @staticmethod
    def _value_shape_num_components(value_shape) -> int:
        if value_shape is None or len(value_shape) == 0:
            return 1
        n = 1
        for s in value_shape:
            n *= int(s)
        return n

    @staticmethod
    def _expand_test_basis(shape_vals, value_shape):
        """
        Scalar test:
            shape_vals -> (Nq, nloc)

        Vector test (e.g. value_shape=(2,)):
            -> (Nq, nloc, ncomp, ncomp)

        Last two axes are:
        - basis component index
        - physical component index

        This preserves the test-component axis under contractions like
        inner(f, phi) or inner(sigma, eps(phi), n_contract=2).
        """
        n_comp = TraceEvaluator._value_shape_num_components(value_shape)

        if n_comp == 1:
            return shape_vals

        eye = jnp.eye(n_comp, dtype=shape_vals.dtype)
        return shape_vals[:, :, None, None] * eye[None, None, :, :]

    def _assemble_value_basis_integrand(self, coeff, shape_vals_flat, weights, flat_entity_nodes, num_total_nodes):
        coeff = jnp.asarray(coeff)
        shape_vals_flat = jnp.asarray(shape_vals_flat)
        flat_entity_nodes = jnp.asarray(flat_entity_nodes, dtype=jnp.int32).reshape(-1)
        weights = jnp.asarray(weights)
        while weights.ndim > 2 and weights.shape[0] == 1:
            weights = jnp.squeeze(weights, axis=0)
        while coeff.ndim > 2 and coeff.shape[0] == 1:
            coeff = jnp.squeeze(coeff, axis=0)

        if coeff.ndim == 0:
            coeff = coeff[None]
        elif coeff.ndim == 2 and coeff.shape[1] == 1:
            coeff = coeff[:, 0]
        elif coeff.ndim > 2:
            raise ValueError(f"Unsupported coeff rank {coeff.ndim} for value assembly; got shape {coeff.shape}")

        if shape_vals_flat.ndim != 2:
            raise ValueError(f"Expected shape_vals_flat.ndim == 2, got shape {shape_vals_flat.shape}")

        n_q_total, n_loc = shape_vals_flat.shape
        n_cell_times_nloc = flat_entity_nodes.shape[0]

        if n_cell_times_nloc % n_loc != 0:
            raise ValueError(f"flat_entity_nodes size {n_cell_times_nloc} is not divisible by n_loc={n_loc}")

        num_entities = n_cell_times_nloc // n_loc

        if n_q_total % num_entities != 0:
            raise ValueError(f"Number of quad rows {n_q_total} is not divisible by num_entities={num_entities}")

        num_quads = n_q_total // num_entities

        if weights.ndim == 2:
            weights_flat = weights.reshape(-1)
        elif weights.ndim == 1:
            weights_flat = weights
        else:
            raise ValueError(f"Unsupported weights rank {weights.ndim}; got shape {weights.shape}")

        # scalar case -> (Nq_total, n_loc)
        if coeff.ndim == 1:
            if coeff.shape[0] != n_q_total:
                raise ValueError(f"coeff shape {coeff.shape} incompatible with shape_vals_flat {shape_vals_flat.shape}")

            local_q = coeff[:, None] * shape_vals_flat * weights_flat[:, None]
            local_entity = local_q.reshape(num_entities, num_quads, n_loc).sum(axis=1)

            global_residual = jax.ops.segment_sum(
                local_entity.reshape(-1),
                flat_entity_nodes,
                num_segments=num_total_nodes,
            )
            return global_residual[:, None]

        # vector case -> (Nq_total, vec)
        elif coeff.ndim == 2:
            if coeff.shape[0] != n_q_total:
                raise ValueError(f"coeff shape {coeff.shape} incompatible with shape_vals_flat {shape_vals_flat.shape}")

            vec = coeff.shape[1]

            # (Nq_total, n_loc, vec)
            local_q = coeff[:, None, :] * shape_vals_flat[:, :, None] * weights_flat[:, None, None]

            # (num_entities, num_quads, n_loc, vec) -> sum quads -> (num_entities, n_loc, vec)
            local_entity = local_q.reshape(num_entities, num_quads, n_loc, vec).sum(axis=1)

            # assemble each component separately
            comps = []
            for c in range(vec):
                gc = jax.ops.segment_sum(
                    local_entity[:, :, c].reshape(-1),
                    flat_entity_nodes,
                    num_segments=num_total_nodes,
                )
                comps.append(gc)

            return jnp.stack(comps, axis=-1)

        else:
            raise ValueError(f"Unsupported normalized coeff shape {coeff.shape}")

    def _assemble_grad_basis_integrand(self, coeff_vec, v_grads_JxW_flat, flat_entity_nodes, num_total_nodes):
        """
        Assemble coeff : grad(phi)-type terms.

        Supported canonical cases
        -------------------------
        Scalar:
            coeff_vec         : (Nq_total, dim)
            v_grads_JxW_flat  : (Nq_total, n_loc, dim)

        Vector / multi-component:
            coeff_vec         : (Nq_total, vec, dim)
            v_grads_JxW_flat  : (Nq_total, n_loc, test_vec, dim)

        Notes
        -----
        The assembler may provide test_vec = 1 even for vector-valued problems. In that case
        the test gradient weights are broadcast over the coefficient component axis.
        """
        coeff_vec = jnp.asarray(coeff_vec)
        v_grads_JxW_flat = jnp.asarray(v_grads_JxW_flat)
        flat_entity_nodes = jnp.asarray(flat_entity_nodes, dtype=jnp.int32).reshape(-1)

        # Strip leading singleton batch/time axes if present
        while coeff_vec.ndim > 2 and coeff_vec.shape[0] == 1:
            coeff_vec = jnp.squeeze(coeff_vec, axis=0)

        # --------------------------------------------------
        # Scalar grad-channel case
        # --------------------------------------------------
        if v_grads_JxW_flat.ndim == 3:
            # coeff: (Nq_total, dim)
            if coeff_vec.ndim == 3 and coeff_vec.shape[1] == 1:
                coeff_vec = coeff_vec[:, 0, :]

            if coeff_vec.ndim == 1:
                coeff_vec = coeff_vec[:, None]

            if coeff_vec.ndim != 2:
                raise ValueError(
                    f"Scalar grad assembly expects coeff_vec.ndim == 2 after normalization, "
                    f"got shape {coeff_vec.shape} with v_grads_JxW_flat {v_grads_JxW_flat.shape}"
                )

            if coeff_vec.shape[0] != v_grads_JxW_flat.shape[0]:
                raise ValueError(
                    f"grad coeff shape {coeff_vec.shape} incompatible with v_grads_JxW_flat {v_grads_JxW_flat.shape}"
                )

            n_q_total, n_loc, dim = v_grads_JxW_flat.shape
            if coeff_vec.shape[1] != dim:
                raise ValueError(f"Scalar grad coeff last dimension {coeff_vec.shape[1]} does not match dim={dim}")

            # (Nq_total, n_loc)
            local_q = jnp.sum(coeff_vec[:, None, :] * v_grads_JxW_flat, axis=-1)

        # --------------------------------------------------
        # Vector / multi-component grad-channel case
        # --------------------------------------------------
        elif v_grads_JxW_flat.ndim == 4:
            # coeff should be (Nq_total, vec, dim)
            if coeff_vec.ndim == 2:
                coeff_vec = coeff_vec[:, None, :]

            if coeff_vec.ndim != 3:
                raise ValueError(
                    f"Vector grad assembly expects coeff_vec.ndim == 3 after normalization, "
                    f"got shape {coeff_vec.shape} with v_grads_JxW_flat {v_grads_JxW_flat.shape}"
                )

            if coeff_vec.shape[0] != v_grads_JxW_flat.shape[0]:
                raise ValueError(
                    f"grad coeff shape {coeff_vec.shape} incompatible with v_grads_JxW_flat {v_grads_JxW_flat.shape}"
                )

            n_q_total, n_loc, test_vec, dim = v_grads_JxW_flat.shape
            coeff_nq, coeff_vec_dim, coeff_dim = coeff_vec.shape

            if coeff_dim != dim:
                raise ValueError(f"Vector grad coeff last dimension {coeff_dim} does not match dim={dim}")

            # the assembler may provide singleton component axis in v_grads_JxW_flat.
            # Broadcast it to the coefficient component count if needed.
            if test_vec != coeff_vec_dim:
                if test_vec == 1:
                    v_grads_JxW_flat = jnp.broadcast_to(
                        v_grads_JxW_flat,
                        (n_q_total, n_loc, coeff_vec_dim, dim),
                    )
                    test_vec = coeff_vec_dim
                else:
                    raise ValueError(
                        f"grad coeff shape {coeff_vec.shape} incompatible with v_grads_JxW_flat {v_grads_JxW_flat.shape}"
                    )

            # double contraction over component and spatial direction
            # -> local_q shape (Nq_total, n_loc)
            local_q = jnp.sum(
                coeff_vec[:, None, :, :] * v_grads_JxW_flat,
                axis=-1,
            )

        else:
            raise ValueError(
                f"Unsupported v_grads_JxW_flat rank {v_grads_JxW_flat.ndim}; "
                f"expected 3 (scalar) or 4 (vector/multi-component)."
            )

        # --------------------------------------------------
        # Reconstruct cell structure and sum quadrature
        # --------------------------------------------------
        n_cell_times_nloc = flat_entity_nodes.shape[0]

        if n_cell_times_nloc % n_loc != 0:
            raise ValueError(f"flat_entity_nodes size {n_cell_times_nloc} is not divisible by n_loc={n_loc}")

        num_cells = n_cell_times_nloc // n_loc

        if n_q_total % num_cells != 0:
            raise ValueError(f"Number of quad rows {n_q_total} is not divisible by num_cells={num_cells}")

        num_quads = n_q_total // num_cells

        # Scalar case: local_q shape (Nq_total, n_loc)
        if local_q.ndim == 2:
            local_cell = local_q.reshape(num_cells, num_quads, n_loc).sum(axis=1)

            global_residual = jax.ops.segment_sum(
                local_cell.reshape(-1),
                flat_entity_nodes,
                num_segments=num_total_nodes,
            )
            return global_residual[:, None]

        # Vector case: local_q shape (Nq_total, n_loc, vec)
        elif local_q.ndim == 3:
            vec = local_q.shape[-1]
            local_cell = local_q.reshape(num_cells, num_quads, n_loc, vec).sum(axis=1)

            comps = []
            for c in range(vec):
                gc = jax.ops.segment_sum(
                    local_cell[:, :, c].reshape(-1),
                    flat_entity_nodes,
                    num_segments=num_total_nodes,
                )
                comps.append(gc)

            return jnp.stack(comps, axis=-1)

        else:
            raise ValueError(f"Unsupported local_q shape {local_q.shape}")

    def _assemble_basis_integrand(self, integrand, weights, flat_entity_nodes, num_total_nodes):
        """
        Assemble a basis-weighted integral into nodal residuals.

        Expected scalar-field shapes:
        integrand         : (Nq_total, n_loc)        or (n_cells, n_q, n_loc)
        weights           : (n_cells, n_q)
        flat_entity_nodes : (n_cells * n_loc,)
        Returns:
        global_residual   : (num_total_nodes, 1)
        """
        integrand = jnp.asarray(integrand)
        weights = jnp.asarray(weights)
        flat_entity_nodes = jnp.asarray(flat_entity_nodes, dtype=jnp.int32).reshape(-1)
        while integrand.ndim > 2 and integrand.shape[0] == 1:
            integrand = jnp.squeeze(integrand, axis=0)

        num_entities, num_quads = weights.shape[-2], weights.shape[-1]
        # Expected local-node count comes from connectivity, not integrand shape.
        # This guards grouped FEM paths where integrand is emitted with a flattened
        # (nloc * feature) axis instead of an explicit nloc axis.
        if flat_entity_nodes.size % num_entities != 0:
            raise ValueError(
                "Inconsistent FEM connectivity: flat_entity_nodes.size is not divisible "
                f"by num_entities ({flat_entity_nodes.size} vs {num_entities})."
            )
        expected_n_local_nodes = int(flat_entity_nodes.size // num_entities)

        if integrand.ndim < 2:
            raise ValueError(f"Assembly integrand must have at least 2 dims, got shape {integrand.shape}.")

        if integrand.shape[0] != weights.size:
            raise ValueError(
                "Assembly integrand leading axis must match total quadrature points "
                f"({integrand.shape[0]} vs {weights.size})."
            )

        if integrand.shape[1] != expected_n_local_nodes:
            if integrand.shape[1] % expected_n_local_nodes != 0:
                raise ValueError(
                    "Assembly integrand local-node axis is incompatible with connectivity: "
                    f"shape={integrand.shape}, expected nloc={expected_n_local_nodes}."
                )
            # Recover explicit local-node axis from flattened representation.
            split = integrand.shape[1] // expected_n_local_nodes
            integrand = integrand.reshape((integrand.shape[0], expected_n_local_nodes, split) + tuple(integrand.shape[2:]))

        trailing_shape = integrand.shape[2:]

        num_entities, num_quads = weights.shape

        if integrand.ndim == 2:
            if integrand.shape[0] != num_entities * num_quads:
                raise ValueError(
                    f"Integrand first dim {integrand.shape[0]} does not match "
                    f"num_entities*num_quads = {num_entities * num_quads}"
                )
            n_loc = integrand.shape[1]
            integrand = integrand.reshape(num_entities, num_quads, n_loc)

        elif integrand.ndim >= 3:
            if integrand.shape[0] != num_entities or integrand.shape[1] != num_quads:
                raise ValueError(
                    f"Structured integrand shape {integrand.shape} is incompatible with weights shape {weights.shape}"
                )
            n_loc = integrand.shape[2]

        else:
            raise ValueError(f"Unsupported integrand shape: {integrand.shape}")

        weighted = integrand * weights[..., None]
        local_residual = jnp.sum(weighted, axis=1)

        if local_residual.ndim == 2:
            if flat_entity_nodes.size != local_residual.size:
                raise ValueError(
                    f"flat_entity_nodes size {flat_entity_nodes.size} does not match "
                    f"local_residual size {local_residual.size}"
                )

            global_residual = jax.ops.segment_sum(
                local_residual.reshape(-1),
                flat_entity_nodes,
                num_segments=num_total_nodes,
            )
            return global_residual[:, None]

        trailing_shape = local_residual.shape[2:]
        n_comp = int(np.prod(trailing_shape))
        local_flat = local_residual.reshape(num_entities, n_loc, n_comp)

        outs = []
        for c in range(n_comp):
            outs.append(
                jax.ops.segment_sum(
                    local_flat[:, :, c].reshape(-1),
                    flat_entity_nodes,
                    num_segments=num_total_nodes,
                )
            )

        global_residual = jnp.stack(outs, axis=-1)
        return global_residual.reshape(num_total_nodes, *trailing_shape)

    def _eval_test_function(self, expr, ctx):
        """
        Resolve the generic TestFunction against the currently active variational region.

        Scalar test:
            volume   -> (Nq_total, nloc)
            boundary -> (Nq_total, nloc)

        Vector test:
            volume   -> (Nq_total, nloc, ncomp, ncomp)
            boundary -> (Nq_total, nloc, ncomp, ncomp)
        """
        if ctx.active_region is None:
            raise ValueError("TestFunction evaluation requires an active_region. Use it inside Assembly(...).")

        support = ctx.active_region["support"]
        region_id = ctx.active_region["region_id"]
        value_shape = getattr(expr, "value_shape", ())

        if support == "volume":
            vals = ctx.context["N_flat"]  # (Nq_total, nloc)
            return self._expand_test_basis(vals, value_shape)

        if support == "boundary":
            if "surface_data" not in ctx.context or region_id not in ctx.context["surface_data"]:
                raise KeyError(f"Boundary region '{region_id}' not found in fem_context['surface_data'].")

            vals = ctx.context["surface_data"][region_id]["face_shape_vals"]
            vals = vals.reshape(-1, vals.shape[-1])  # (Nq_total, nloc)
            return self._expand_test_basis(vals, value_shape)

        raise ValueError(f"Unknown active support '{support}'")

    def _eval_assembly(self, expr, ctx):
        integrand = self._dispatch(expr.expr, ctx)

        if expr.support == "volume":
            weights = ctx.context["JxW"]
            flat_cells = ctx.context["flat_cells"].flatten()

        elif expr.support == "boundary":
            if "surface_data" not in ctx.context or expr.region_id not in ctx.context["surface_data"]:
                raise KeyError(f"Boundary region '{expr.region_id}' not found in fem_context['surface_data'].")
            surf_data = ctx.context["surface_data"][expr.region_id]
            weights = surf_data["nanson_scale"]
            flat_cells = surf_data["flat_parent_nodes"].flatten()

        else:
            raise ValueError(f"Unknown assembly support '{expr.support}'")

        global_residual = self._assemble_basis_integrand(
            integrand,
            weights,
            flat_cells,
            expr.num_total_nodes,
        )

        if "dirichlet_nodes" in ctx.context:
            d_nodes = jnp.asarray(ctx.context["dirichlet_nodes"]).flatten().astype(jnp.int32)
            if d_nodes.size > 0:
                global_residual = global_residual.at[d_nodes].set(0.0)

        return global_residual

    def _eval_grouped_assembly(self, expr, ctx):
        total = None

        # -------------------------
        # Volume value contribution
        # -------------------------
        if expr.volume_value_expr is not None:
            vol_ctx = self._EvalCtx(
                ctx.context,
                ctx.var_bindings,
                ctx.key,
                active_region={"support": "volume", "region_id": "volume"},
            )

            coeff_val = self._dispatch(expr.volume_value_expr, vol_ctx)

            vol_val_res = self._assemble_value_basis_integrand(
                coeff_val,
                ctx.context["N_flat"],
                ctx.context["JxW"],
                ctx.context["flat_cells"].reshape(-1),
                expr.num_total_nodes,
            )

            # Value terms act like RHS/load contributions in the residual,
            # so they enter with a minus sign.
            total = vol_val_res if total is None else (total + vol_val_res)

        # -------------------------
        # Volume grad contribution
        # -------------------------
        if expr.volume_grad_expr is not None:
            vol_ctx = self._EvalCtx(
                ctx.context,
                ctx.var_bindings,
                ctx.key,
                active_region={"support": "volume", "region_id": "volume"},
            )

            coeff_grad = self._dispatch(expr.volume_grad_expr, vol_ctx)

            vol_grad_res = self._assemble_grad_basis_integrand(
                coeff_grad,
                ctx.context["v_grads_JxW_flat"],
                ctx.context["flat_cells"].reshape(-1),
                expr.num_total_nodes,
            )

            total = vol_grad_res if total is None else (total + vol_grad_res)

        # -------------------------
        # Boundary value contributions
        # -------------------------
        for region_id, bnd_expr in expr.boundary_value_exprs.items():
            bnd_ctx = self._EvalCtx(
                ctx.context,
                ctx.var_bindings,
                ctx.key,
                active_region={"support": "boundary", "region_id": region_id},
            )

            coeff_val = self._dispatch(bnd_expr, bnd_ctx)

            surf_data = ctx.context["surface_data"][region_id]

            face_shape_vals = jnp.asarray(surf_data["face_shape_vals"])
            while face_shape_vals.ndim > 3 and face_shape_vals.shape[0] == 1:
                face_shape_vals = jnp.squeeze(face_shape_vals, axis=0)

            nanson_scale = jnp.asarray(surf_data["nanson_scale"])
            while nanson_scale.ndim > 2 and nanson_scale.shape[0] == 1:
                nanson_scale = jnp.squeeze(nanson_scale, axis=0)

            flat_parent_nodes = jnp.asarray(surf_data["flat_parent_nodes"], dtype=jnp.int32)
            while flat_parent_nodes.ndim > 1 and flat_parent_nodes.shape[0] == 1:
                flat_parent_nodes = jnp.squeeze(flat_parent_nodes, axis=0)

            bnd_res = self._assemble_value_basis_integrand(
                coeff_val,
                face_shape_vals.reshape(-1, face_shape_vals.shape[-1]),
                nanson_scale,
                flat_parent_nodes.reshape(-1),
                expr.num_total_nodes,
            )

            # if "global_boundary_areas" in surf_data and "global_areas" in ctx.context:
            #     gb = jnp.asarray(surf_data["global_boundary_areas"]).reshape(-1, 1)
            #     gv = jnp.asarray(ctx.context["global_areas"]).reshape(-1, 1)
            #     bnd_res = bnd_res * (gv / (gb + 1e-12))

            # Boundary value terms (e.g. Neumann loads) also act like RHS/load contributions
            # print(f"DEBUG boundary tag={region_id}, ||bnd_res|| =", jnp.linalg.norm(bnd_res))
            total = bnd_res if total is None else (total + bnd_res)

        if total is None:
            raise ValueError("GroupedAssembly has neither value nor grad nor boundary terms.")

        if "global_areas" in ctx.context:
            areas = jnp.asarray(ctx.context["global_areas"]).reshape(-1, 1)
            total = total / (areas + 1e-12)

        if "dirichlet_nodes" in ctx.context:
            d_nodes = jnp.asarray(ctx.context["dirichlet_nodes"]).flatten().astype(jnp.int32)
            if d_nodes.size > 0:
                if total.ndim == 1:
                    total = total.at[d_nodes].set(0.0)
                else:
                    total = total.at[d_nodes, :].set(0.0)

        return total

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
        if isinstance(node, Choice):
            return uid, f"Choice(name={node.name}, selected={node.selected})"
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
        if isinstance(node, NetworkGradient):
            return uid, f"NetworkGradient(model={node.model_node!r})"
        if isinstance(node, Noise):
            params_str = ", ".join(f"{k}={v}" for k, v in node.params.items())
            return uid, f"Noise({node.distribution}, {params_str})"
        return uid, type(node).__name__
