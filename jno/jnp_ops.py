import builtins
import warnings
from pathlib import Path
from typing import List, Union

import jax
import jax.numpy as jnp

from .architectures.models import nn, parameter  # noqa: F401

# Keep import so people can use jno.numpy as jno -> jno.model, jno.tune
from .integration_operators import IntegrationOperators  # noqa: F401
from .trace import (
    Cellwise,
    Choice,
    ConstantNamespace,
    Diff,
    FunctionCall,
    Hessian,
    Integral,
    Jacobian,
    Placeholder,
    TestFunction,
    Tracker,
    TrialFunction,
    Variable,
    _guard_ad_on_fd,
)
from .tuner import Arch, ArchSpace, tune  # noqa: F401


def _u(x):
    """Strip a typed-view wrapper, if any, to get the underlying Placeholder.

    All top-level ``jno.np.*`` wrappers call this on Placeholder arguments so
    users can pass ``u.scalar`` / ``u.vector`` / etc. directly without ``.expr``.
    """
    from .trace.views import _VIEW_TYPES

    return x._expr if isinstance(x, _VIEW_TYPES) else x


def _coord_vars_of(*operands) -> dict:
    """Merge ``_coord_vars`` (name → Variable) from any view operands.

    A reduction like ``inner``/``trace`` unwraps its operands with :func:`_u`,
    which discards a bound view's coordinate registration. Re-collecting it here
    lets a region binding survive the reduction, so a bound boundary term such as
    ``inner(t, phi.bind(x=xr, y=yr))`` still exposes its region to the FEM driver.

    A name mapping to two *different* Variables across operands is ambiguous and
    is dropped (order-independent), mirroring the view ``_rewrap`` merge.
    """
    merged: dict = {}
    conflicting: set = set()
    for o in operands:
        cv = getattr(o, "_coord_vars", None)
        if not cv:
            continue
        for name, var in cv.items():
            if name in merged and merged[name] is not var:
                conflicting.add(name)
            else:
                merged[name] = var
    for name in conflicting:
        merged.pop(name, None)
    return merged


def _attach_coords(call, raw_operands):
    """Attach merged operand ``_coord_vars`` to a freshly built ``FunctionCall``.

    ``raw_operands`` are the *pre-unwrap* arguments (views), so their coordinate
    bindings can be read before :func:`_u` strips them.
    """
    cv = _coord_vars_of(*raw_operands)
    if cv:
        call._coord_vars = cv
    return call


def _guard(target, scheme: str = "automatic_differentiation") -> None:
    """Functional-API mirror of ``Placeholder``'s method-level guard: block an
    automatic-differentiation differential operator over a FieldView
    finite-difference partial (which would silently return 0).

    Also catches the coordinates being passed positionally — the functional
    operators take the variables as a *list*, so ``laplacian(u, x, y)`` lands ``y``
    in the ``scheme`` slot and would otherwise fail much later inside the compiler
    with an unrelated ``AttributeError``.
    """
    if not isinstance(scheme, str):
        raise TypeError(
            f"scheme must be a string, got {type(scheme).__name__}. The functional operators take the "
            f"coordinates as one list — write laplacian(u, [x, y]), or use the method form u.laplacian(x, y)."
        )
    _guard_ad_on_fd(_u(target), scheme)


# ============================================================================
# Constants
# ============================================================================

pi = jnp.pi
π = jnp.pi
e = jnp.e
inf = jnp.inf
nan = jnp.nan


def tracker(op: Placeholder, interval: int = 1, reduce=None) -> Tracker:
    """Mark an expression as a tracked metric.

    Trackers are monitored during training but do NOT contribute to
    the loss function or gradient computation.

    Args:
        op: The expression to monitor.
        interval: Evaluate every *interval* epochs (default: every epoch).
        reduce: Optional callable applied to the numpy array after device
            transfer to produce a scalar for W&B and the progress line.
            Defaults to ``np.mean`` for non-scalar outputs.
    """

    return Tracker(_u(op), interval, reduce=reduce)


def constant(tag: str, data: Union[dict, str, Path]) -> ConstantNamespace:
    """Create a ConstantNamespace from a dict or file.

    Args:
        tag: Name for this constant group
        data: Dict of values, or path to .json/.yaml/.yml/.toml/.pkl/.npz file

    Returns:
        ConstantNamespace with attribute access to constants

    Example::

        # From dict with nested parameters
        C = constant("C", {
            "k": 1.0,
            "m": 2.0,
            "physics": {
                "gravity": 9.81,
                "friction": 0.1
            },
            "model": {
                "layers": {
                    "hidden": 64,
                    "output": 10
                }
            }
        })

        # Access flat constants
        C.k  # -> Constant(C.k=1.0)

        # Access nested constants
        C.physics.gravity  # -> Constant(C.physics.gravity=9.81)
        C.model.layers.hidden  # -> Constant(C.model.layers.hidden=64)

        # From JSON file
        C = constant("C", "params.json")

        # From YAML file
        C = constant("C", "config.yaml")
    """
    return ConstantNamespace(tag, data)


def function(fn, args: list = [], name: str = "", reduces_axis: int = None):
    """Wrap a pure JAX function of traced arguments as a node (the general escape hatch).

    Example::

        r = jno.np.function(lambda a, b: jnp.where(a > b, a, 0.0), [u, v], name="clip")

    Args:
        fn: the pure function, called with the evaluated ``args``.
        args: its arguments (traced expressions, arrays or numbers).
        name: display name in printing and ``jno.info``.
        reduces_axis: set when ``fn`` reduces its input (a mean, a sum, ...), so the weak-form mark of
            the input does not pass to the result.

    Returns:
        FunctionCall: the node.
    """
    return FunctionCall(fn, args, name, reduces_axis)


def choice(options, name: str | None = None, default: int = 0) -> Choice:
    """Create a tunable categorical choice over traced expressions.

    The tuner switches between the options; otherwise ``default`` is used.

    Example::

        u = jnn.choice([net(inp) * x * (1 - x), net(inp)], name="bc_form")

    Args:
        options: the candidate expressions (constants are wrapped as literals).
        name: name under which the tuner reports the choice.
        default: index of the option used until another is selected.

    Returns:
        Choice: the choice node.
    """
    return Choice(options=options, name=name, default=default)


# ============================================================================
# Element-wise wrappers
# ============================================================================


def _unary_call(jnp_fn, x) -> FunctionCall:
    """``jnp_fn(x)`` as a traced node; typed views are unwrapped and ``x``'s coordinates carried over."""
    return _attach_coords(FunctionCall(jnp_fn, [_u(x)]), [x])


def _binary_call(jnp_fn, x, y) -> FunctionCall:
    """``jnp_fn(x, y)`` as a traced node; typed views are unwrapped and coordinates carried over."""
    return _attach_coords(FunctionCall(jnp_fn, [_u(x), _u(y)]), [x, y])


# Trigonometric
def sin(x) -> FunctionCall:
    """Element-wise sine, ``jnp.sin``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.sin, x)


def cos(x) -> FunctionCall:
    """Element-wise cosine, ``jnp.cos``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.cos, x)


def tan(x) -> FunctionCall:
    """Element-wise tangent, ``jnp.tan``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.tan, x)


def arcsin(x) -> FunctionCall:
    """Element-wise inverse sine, ``jnp.arcsin``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.arcsin, x)


def arccos(x) -> FunctionCall:
    """Element-wise inverse cosine, ``jnp.arccos``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.arccos, x)


def arctan(x) -> FunctionCall:
    """Element-wise inverse tangent, ``jnp.arctan``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.arctan, x)


def arctan2(x, y) -> FunctionCall:
    """Element-wise four-quadrant inverse tangent ``arctan(x / y)`` with the sign of both arguments, ``jnp.arctan2``, as a traced node.

    Args:
        x: the first argument: a traced expression, view, array or number.
        y: the second argument; broadcast against ``x``.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _binary_call(jnp.arctan2, x, y)


atan2 = arctan2


# Hyperbolic
def sinh(x) -> FunctionCall:
    """Element-wise hyperbolic sine, ``jnp.sinh``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.sinh, x)


def cosh(x) -> FunctionCall:
    """Element-wise hyperbolic cosine, ``jnp.cosh``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.cosh, x)


def tanh(x) -> FunctionCall:
    """Element-wise hyperbolic tangent, ``jnp.tanh``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.tanh, x)


def arcsinh(x) -> FunctionCall:
    """Element-wise inverse hyperbolic sine, ``jnp.arcsinh``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.arcsinh, x)


def arccosh(x) -> FunctionCall:
    """Element-wise inverse hyperbolic cosine, ``jnp.arccosh``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.arccosh, x)


def arctanh(x) -> FunctionCall:
    """Element-wise inverse hyperbolic tangent, ``jnp.arctanh``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.arctanh, x)


# Exponential / logarithmic
def exp(x) -> FunctionCall:
    """Element-wise exponential ``e**x``, ``jnp.exp``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.exp, x)


def exp2(x) -> FunctionCall:
    """Element-wise base-2 exponential ``2**x``, ``jnp.exp2``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.exp2, x)


def expm1(x) -> FunctionCall:
    """Element-wise ``exp(x) - 1``, accurate for small ``x``, ``jnp.expm1``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.expm1, x)


def log(x) -> FunctionCall:
    """Element-wise natural logarithm, ``jnp.log``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.log, x)


def log2(x) -> FunctionCall:
    """Element-wise base-2 logarithm, ``jnp.log2``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.log2, x)


def log10(x) -> FunctionCall:
    """Element-wise base-10 logarithm, ``jnp.log10``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.log10, x)


def log1p(x) -> FunctionCall:
    """Element-wise ``log(1 + x)``, accurate for small ``x``, ``jnp.log1p``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.log1p, x)


# Power / root
def sqrt(x) -> FunctionCall:
    """Element-wise square root, ``jnp.sqrt``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.sqrt, x)


def cbrt(x) -> FunctionCall:
    """Element-wise cube root, ``jnp.cbrt``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.cbrt, x)


def square(x) -> FunctionCall:
    """Element-wise square ``x**2``, ``jnp.square``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.square, x)


def power(x, y) -> FunctionCall:
    """Element-wise power ``x**y``, ``jnp.power``, as a traced node.

    Args:
        x: the first argument: a traced expression, view, array or number.
        y: the second argument; broadcast against ``x``.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _binary_call(jnp.power, x, y)


# Rounding / absolute
def abs(x) -> FunctionCall:
    """Element-wise absolute value, ``jnp.abs``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.abs, x)


def floor(x) -> FunctionCall:
    """Element-wise floor (round down), ``jnp.floor``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.floor, x)


def ceil(x) -> FunctionCall:
    """Element-wise ceiling (round up), ``jnp.ceil``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.ceil, x)


def round(x) -> FunctionCall:
    """Element-wise rounding to the nearest integer, ``jnp.round``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.round, x)


def sign(x) -> FunctionCall:
    """Element-wise sign (-1, 0 or 1), ``jnp.sign``, as a traced node.

    Args:
        x: a traced expression, view, array or number.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _unary_call(jnp.sign, x)


# ============================================================================
# Array manipulation
# ============================================================================


@jax.tree_util.register_pytree_node_class
class ViewFactorOp:
    """Linear boundary radiation operator: a view-factor matrix ``F`` usable in traced expressions.

    Supports ``F @ x`` and ``x @ F`` with arrays or traced expressions, and a radiosity solve.

    Args:
        F: the view-factor matrix, an array or a traced expression ``(n, n)``.
    """

    def __init__(self, F: Union["Placeholder", jnp.ndarray]):
        self.F = F

    # -----------------------
    # Matrix multiply: F @ x
    # -----------------------
    def __matmul__(self, x):
        return FunctionCall(lambda A, b: A @ b, [self.F, x])

    # -----------------------
    # Left multiply: x @ F
    # -----------------------
    def __rmatmul__(self, x):
        return FunctionCall(lambda b, A: b @ A, [x, self.F])

    # -----------------------
    # Apply operator explicitly
    # -----------------------
    def apply(self, x):
        """``F @ x``.

        Args:
            x: the vector (array or traced expression).

        Returns:
            The product.
        """
        return self @ x

    # -----------------------
    # Solve (I - αF)x = rhs
    # -----------------------
    def solve(self, rhs, alpha):
        """Solve the radiosity system ``(I - alpha F) q = rhs``.

        Args:
            rhs: the right-hand side vector.
            alpha: the scalar factor (e.g. ``1 - emissivity`` for grey-body radiosity).

        Returns:
            The solution ``q``, traced when any input is traced.
        """

        def solve_fn(A, b, a):
            eye_mat = jnp.eye(A.shape[0])
            return jnp.linalg.solve(eye_mat - a * A, b)

        return FunctionCall(solve_fn, [self.F, rhs, alpha])

    # -----------------------
    # PyTree support for JAX
    # -----------------------
    def tree_flatten(self):
        """Pytree flattening: the children ``(F,)`` and no auxiliary data."""
        return (self.F,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """Pytree unflattening: rebuild the operator from its children.

        Args:
            aux: unused auxiliary data.
            children: ``(F,)``.

        Returns:
            ViewFactorOp: the rebuilt operator.
        """
        return cls(*children)


def view_factor(F: Union["Placeholder", jnp.ndarray]) -> ViewFactorOp:
    """Create a view-factor operator, :class:`ViewFactorOp`.

    Args:
        F: the view-factor matrix, an array or traced expression.

    Returns:
        ViewFactorOp: the operator.
    """
    return ViewFactorOp(F)


def _align_temporal(items):
    """Broadcast a bare temporal variable onto the spatial point axis before concatenating.

    ``dom.variable(tag)`` hands back spatial coords carrying a points axis, but ``t`` as a
    *scalar* per time slice (it reads ``context["__time__"]``). Concatenating them raises
    nothing — the scalar is broadcast — but on a domain with two or more spatial dimensions it
    builds a graph that is pathologically slow to compile, with no error to point at. Feeding a
    PINN/DeepONet trunk ``concat([x, y, t])`` is the usual way in.

    Doing the broadcast here, on the trace (``t + 0*x``), is what users otherwise have to
    remember to write by hand.
    """
    axes = [getattr(i, "axis", None) for i in items]
    if "temporal" not in axes or "spatial" not in axes:
        return items
    ref = next(i for i, a in zip(items, axes) if a == "spatial")
    return [(i + 0.0 * ref) if a == "temporal" else i for i, a in zip(items, axes)]


def concat(items, axis: int = -1) -> FunctionCall:
    """Concatenate traced expressions along the last axis.

    Args:
        items: the expressions (views, Placeholders or arrays).
        axis: accepted for NumPy compatibility; concatenation is always along the last axis at
            evaluation time.

    Returns:
        FunctionCall: the concatenation.
    """
    items = _align_temporal(items)

    def _fn(*args):
        expanded = []
        for a in args:
            a = jnp.asarray(a)
            if a.ndim == 0:
                a = a[jnp.newaxis]
            elif a.ndim == 1:
                a = a[..., jnp.newaxis]
            expanded.append(a)

        # A SINGLE operand needs no concatenation at all — and must not fall through to the
        # rank-alignment fallback below, which is written for two or more. It becomes reachable on a
        # **1D** domain, where `canonicalize_grad_coeff` stacks one component per dimension: with
        # dim=1 that is a one-item stack, and the fallback then re-entered trace-node construction
        # inside the evaluator, so a 1D VPINN never finished assembling its loss.
        if len(expanded) == 1:
            return expanded[0]

        # Fast path when shapes already match on non-concatenation dims.
        ref = expanded[0].shape[:-1]
        if all(a.shape[:-1] == ref for a in expanded[1:]):
            return jnp.concatenate(expanded, axis=-1)

        # Fallback: align ranks and only broadcast singleton dimensions. This is where a bare number
        # meets a per-point operand (``vector(1.0, x)``): the number is (1,), the coordinate (N, 1).
        # ``builtins.max``, not ``max``: this module rebinds ``max`` to the trace reduction
        # ``jno.np.max``, which here built a trace node into the reshape shape -- and JAX's formatter
        # for that invalid shape then iterated the node forever (``__getitem__`` never ends).
        max_ndim = builtins.max(a.ndim for a in expanded)
        aligned = []
        for a in expanded:
            if a.ndim < max_ndim:
                a = a.reshape((1,) * (max_ndim - a.ndim) + a.shape)
            aligned.append(a)

        target_prefix = list(aligned[0].shape[:-1])
        for a in aligned[1:]:
            shp = a.shape[:-1]
            if len(shp) != len(target_prefix):
                raise ValueError(f"concat rank mismatch: {len(shp)} vs {len(target_prefix)}")
            for i, (t, s) in enumerate(zip(target_prefix, shp)):
                if t == s:
                    continue
                if t == 1:
                    target_prefix[i] = s
                elif s == 1:
                    continue
                else:
                    raise ValueError(f"concat cannot broadcast dim {i}: {t} vs {s}")

        target_prefix = tuple(target_prefix)
        broadcasted = [jnp.broadcast_to(a, target_prefix + (a.shape[-1],)) for a in aligned]
        return jnp.concatenate(broadcasted, axis=-1)

    return FunctionCall(_fn, [_u(i) for i in items], name="concat")


# docstring-from: jno.jnp_ops.concat
def concatenate(items, axis: int = -1) -> FunctionCall:
    """Concatenate traced expressions along the last axis.

    Args:
        items: the expressions (views, Placeholders or arrays).
        axis: accepted for NumPy compatibility; concatenation is always along the last axis at
            evaluation time.

    Returns:
        FunctionCall: the concatenation.
    """
    return concat(items, axis=axis)


def vector(*components):
    """Build a :class:`VectorView` directly from scalar component Placeholders.

    Equivalent to ``VectorView(concat([*components]))`` but skips the manual
    list and the ``.vector`` re-wrap step::

        n    = jno.np.vector(nx, ny)          # 2-D normal as VectorView
        flux = jno.np.vector(jx, jy, jz)      # 3-D current density
        prob = jno.np.vector(x * p, y * p)    # OU drift flux
        e1   = jno.np.vector(1.0, 0.0, 0.0)    # a constant vector
        g    = jno.np.vector(1.0, 0.0, x)      # numbers broadcast over the points like `1.0 + 0.0*x`
    """
    from .trace.views import VectorView

    return VectorView(concat(list(components)))


def stack(items, axis: int = 0) -> FunctionCall:
    """Stack traced expressions along a new axis.

    Args:
        items: the expressions (views, Placeholders or arrays).
        axis: position of the new axis; ``-1`` concatenates along the last axis (as :func:`concat`).

    Returns:
        FunctionCall: the stacked node.
    """
    if axis == -1:
        return concat(items, axis=-1)
    return FunctionCall(lambda *args: jnp.stack(args, axis=axis), [_u(i) for i in items], name="stack")


def reshape(x, shape: tuple) -> FunctionCall:
    """Reshape a traced expression.

    Args:
        x: a traced expression, view or array.
        shape: the new shape.

    Returns:
        FunctionCall: the reshaped node.
    """
    return FunctionCall(lambda a: jnp.reshape(a, shape), [_u(x)])


def squeeze(x, axis: int = None) -> FunctionCall:
    """Remove axes of length one.

    Args:
        x: a traced expression, view or array.
        axis: the axis (or axes) to remove; ``None`` removes every length-one axis.

    Returns:
        FunctionCall: the squeezed node.
    """
    return FunctionCall(lambda a: jnp.squeeze(a, axis=axis), [_u(x)])


def expand_dims(x, axis: int) -> FunctionCall:
    """Insert an axis of length one.

    Args:
        x: a traced expression, view or array.
        axis: where the new axis goes.

    Returns:
        FunctionCall: the expanded node.
    """
    return FunctionCall(lambda a: jnp.expand_dims(a, axis=axis), [_u(x)])


def transpose(x, axes: tuple = None) -> FunctionCall:
    """
    Transpose a second-order tensor **over its last two axes** — the TENSOR transpose.

    This is the convention its siblings already use: ``sym``/``antisym`` are built on
    ``jnp.swapaxes(a, -1, -2)``, ``trace`` contracts ``axis1=-2, axis2=-1``, and ``identity`` carries a
    leading singleton so it broadcasts as a constant tensor. A weak form's operands carry batch axes a
    kernel prepends — the quadrature axis, and a local-basis axis on a trial/test function — so
    ``jnp.transpose``'s default of reversing *every* axis transposes those too and the expression dies
    on a shape mismatch (or, worse, silently means something else). Hence the trailing-axes default::

        sigma_T = transpose(sigma)                  # tensor transpose, batch axes untouched
        C       = einsum("...ki,...kj->...ij", F, F)  # F^T F, the same convention

    Pass ``axes`` explicitly to get the raw ``jnp.transpose`` behaviour for a genuine axis permutation.

    Args:
        x: tensor-valued expression, ``(..., n, n)``.
        axes: explicit permutation. Omit it for the tensor transpose.
    """
    if axes is not None:
        return _attach_coords(FunctionCall(lambda a, _ax=axes: jnp.transpose(a, axes=_ax), [_u(x)], name="transpose"), [x])
    return _attach_coords(
        FunctionCall(lambda a: jnp.swapaxes(a, -1, -2), [_u(x)], name="transpose"),
        [x],
    )


def trace(x) -> FunctionCall:
    """Trace of a matrix/tensor over the last two axes.

    Example::

        trace(A)              -> scalar trace for (..., n, n)
        trace(symgrad(u))     -> volumetric strain

    Args:
        x: a second-order tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the trace per point.
    """
    return _attach_coords(
        FunctionCall(
            lambda a: jnp.trace(a, axis1=-2, axis2=-1),
            [_u(x)],
            name="trace",
        ),
        [x],
    )


def sym(x) -> FunctionCall:
    """Symmetric part of a second-order tensor over the last two axes, ``sym(A) = (A + Aᵀ)/2``.

    Args:
        x: a second-order tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the symmetric part.
    """
    return _attach_coords(
        FunctionCall(
            lambda a: 0.5 * (a + jnp.swapaxes(a, -1, -2)),
            [_u(x)],
            name="sym",
        ),
        [x],
    )


def antisym(x) -> FunctionCall:
    """Skew-symmetric part of a second-order tensor over the last two axes, ``antisym(A) = (A - Aᵀ)/2``.

    Args:
        x: a second-order tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the skew-symmetric part.
    """
    return _attach_coords(
        FunctionCall(
            lambda a: 0.5 * (a - jnp.swapaxes(a, -1, -2)),
            [_u(x)],
            name="antisym",
        ),
        [x],
    )


def identity(n: int) -> FunctionCall:
    """Symbolic identity matrix helper — an ``(n, n)`` identity carried with a **leading singleton axis**,
    ``(1, n, n)``, so it broadcasts as a *constant tensor* over the batch axis a kernel prepends (the
    quadrature axis in an FE weak form; the points axis in a collocation residual). A bare ``(n, n)``
    would be mis-read by the elementwise-broadcast alignment as if its first axis were the quadrature
    axis; the leading singleton makes ``I`` compose correctly inside any tensor formula.

    Example::

        I = jnn.identity(3)
        sigma = lam * jnn.trace(eps) * I + 2.0 * mu * eps          # tensor stress in a weak form
        F     = I + jnn.grad(u, [x, y, z])                          # deformation gradient (finite strain)
    """
    return FunctionCall(
        lambda: jnp.eye(n)[None],  # (1, n, n): constant over the leading (quadrature / points) axis
        [],
        name="identity",
    )


def symgrad(
    target: Placeholder,
    variables: List[Variable],
    scheme: str = "automatic_differentiation",
) -> FunctionCall:
    """Symmetric gradient of a vector/tensor-valued field.

    For a vector field ``u`` in R^dim: ``grad(u)`` is ``(..., n_comp, dim)`` and
    ``symgrad(u) = (grad(u) + grad(u)ᵀ)/2``. In small-strain elasticity::

        eps = symgrad(u, [x, y])       # 2-D
        eps = symgrad(u, [x, y, z])    # 3-D

    This uses jNO's Jacobian convention, ``jacobian(u, [x, y])`` with trailing shape
    ``(..., value_shape, dim)``: the last axis is the derivative direction.

    Args:
        target: the vector field.
        variables: the coordinate Variables, one per dimension.
        scheme: how the derivatives are computed: ``"automatic_differentiation"`` (default, the global
            mode set via :func:`jno.setup`; ``":forward"`` / ``":reverse"`` and, for second order,
            ``":fwd-over-rev"`` style suffixes select the nesting) or ``"finite_difference"`` (optional
            sub-schemes ``":lsq"``, ``":uniform"``, ``":inverse_distance"``, ``":cotangent"``).

    Returns:
        FunctionCall: the symmetric gradient, ``(..., dim, dim)``.
    """
    G = jacobian(_u(target), variables, scheme=scheme)
    return FunctionCall(
        lambda a: 0.5 * (a + jnp.swapaxes(a, -1, -2)),
        [G],
        name="symgrad",
    )


# ============================================================================
# Linear algebra on second-order tensors
# ============================================================================
# All act on the **last two axes** and broadcast over any leading (e.g.
# quadrature) axes — exactly like ``trace``/``sym`` above. ``inv``/``det``/
# ``eigvalsh`` are thin ``jax.numpy.linalg`` wrappers; the symmetric matrix
# functions ``logm``/``expm``/``sqrtm`` route through a Daleckiĭ–Kreĭn
# ``custom_jvp`` (see ``jno.trace.views``) so their gradient stays finite at
# *repeated* eigenvalues — equal principal stretches, the common case in
# finite-strain plasticity. Together these are the tensor primitives a
# finite-strain constitutive update needs: ``inv`` for the multiplicative split
# ``F = Fₑ·Fₚ``, ``det`` for the volumetric part ``J = det F``, ``logm``/``expm``
# for a log-strain (Hencky) return map.


def inv(x) -> FunctionCall:
    """Inverse of a second-order tensor over the last two axes → tensor.

    Broadcasts over any leading (quadrature/batch) axes.

    Example::

        Fe = jnn.matmul(F, jnn.inv(Fp))       # elastic part of F = Fe · Fp

    Args:
        x: a second-order tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the inverse.
    """
    return _attach_coords(FunctionCall(jnp.linalg.inv, [_u(x)], name="inv"), [x])


def det(x) -> FunctionCall:
    """Determinant over the last two axes → scalar (per leading index).

    Example::

        J = jnn.det(F)                        # volumetric part of the deformation

    Args:
        x: a second-order tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the determinant per point.
    """
    return _attach_coords(FunctionCall(jnp.linalg.det, [_u(x)], name="det"), [x])


def diff(target, wrt) -> Diff:
    """Differentiate a **scalar** expression w.r.t. another **expression** — ``∂target/∂wrt``.

    The constitutive counterpart of :func:`grad`, which differentiates w.r.t. a spatial coordinate.
    This is what lets a hyperelastic material be written as the thing it actually is — a stored
    energy — instead of a hand-derived stress::

        F   = I + jacobian(u, X)                      # bind it ONCE, then reuse
        I1b = det(F) ** (-2 / 3) * trace(matmul(transpose(F), F))
        psi = C10 * (I1b - 3) + C20 * (I1b - 3) ** 2  # Yeoh, the energy from the paper
        P   = diff(psi, F)                            # 1st Piola-Kirchhoff, by autodiff

    ``P`` then goes into the weak form as usual, ``inner(P, jacobian(phi, X), 2)``. The element tangent
    (``∂P/∂F``, the consistent modulus) comes out of the assembler's own differentiation for free.

    It is **pointwise**: the derivative is taken at each quadrature point independently, which is what a
    constitutive law is. An ``Integral`` inside ``target`` is therefore refused — differentiate the
    integrand, then integrate.

    ``wrt`` is matched **by identity**, so bind it to a variable and pass that same object; a rebuilt
    copy is a different node and is rejected rather than silently differentiating to zero.

    Works for any energy-derived law — Neo-Hookean, Mooney-Rivlin, Yeoh, Ogden, Gent, anisotropic
    tissue models — and equally for a chemical potential ``mu = diff(f, c)`` or an electro/magneto-
    strictive coupling.
    """
    return Diff(_u(target), _u(wrt))


def cellwise(x) -> Cellwise:
    """Project an expression onto the **piecewise constants**: its quadrature-weighted mean over each
    cell, ``(∫_K x dx)/(∫_K dx)``, broadcast back to that cell's quadrature points.

    This is the one primitive the locking-free strain measures are built from. Volumetric locking is
    what a standard displacement element does as ``nu -> 0.5`` — and J2 plastic flow is isochoric, so it
    is the *default* regime for plasticity, not an edge case. B-bar replaces the volumetric part of the
    strain by its cell mean; written the way the elastic forms in this codebase are written, as a scalar
    contraction rather than with an identity tensor (``sigma(ebar) : ebar(phi)``, using
    ``dev(A):dev(B) = A:B - tr(A)tr(B)/dim``)::

        eps = lambda w: sym(w, X)
        eu, ev   = eps(ui), eps(vi)
        tru, trv = trace(eu), trace(ev)
        cu, cv   = cellwise(tru), cellwise(trv)          # <- the projection, and the whole change
        dev_uv   = inner(eu, ev, 2) - tru * trv / dim    # dev(eu) : dev(ev)
        mech     = lam * cu * cv + 2 * mu * (dev_uv + cu * cv / dim)

    Dropping the two ``cellwise`` calls recovers the standard form, so B-bar is a one-line edit to a
    form you already have. Note the identity-tensor spelling ``e - vol*I + cellwise(vol)*I`` works on
    the *trial* side but not the test side: a test-bearing quantity carries an extra DOF axis that a
    constant ``(1, d, d)`` tensor will not broadcast against. Contract to a scalar, as above.

    F-bar is the finite-strain counterpart — project the determinant instead::

        F    = I + grad(u, X)
        Fbar = (cellwise(det(F)) / det(F)) ** (1 / dim) * F
        mech = inner(diff(psi(Fbar), Fbar), grad(phi, X), 2)

    **It is quadrature-weighted, not a plain mean over quadrature points.** The two coincide only when
    the rule's weights are equal — false for higher-degree simplex rules, and false on curved
    (``shape.curved()``) or tensor-product cells, where the measure varies inside the cell. The weighted
    form is the actual L2 projection, so it is the actual B-bar.

    **Where it does and does not help, measured.** B-bar cures locking only where the strain is not
    already constant within a cell, so it is for **quadrilateral / hexahedral** cells and higher-order
    simplices. On **P1 triangles and tets the strain is element-wise constant, so this projection is
    the identity and changes nothing** — that is not a defect, but it means B-bar is not the cure for
    P1 volumetric locking (use Q1/hex cells, a higher order, or a mixed u-p formulation there). On a
    plane-strain Q1 cantilever at ``nu = 0.4999`` (measured, 16x4 cells): standard ``2.68e-02`` against
    B-bar ``3.03e-01``, an 11.3x difference, where B-bar is already converged (it moves 0.7% out to
    128x32) and the standard element is still 2.3x too stiff at 128x32 — refinement does not cure
    locking, which is why this exists. At ``nu = 0.3`` the same comparison differs by 5.6%.

    Scope, enforced rather than documented around: **native-Lagrange volume terms**. The 1-D, non-nodal
    (Argyris/Morley/RT/N1E) and VPINN assemblers do not thread the per-cell quadrature weights, and a
    collocation/PINN residual has no cells at all — each raises rather than averaging over the wrong set
    of points. Boundary/surface terms raise for the same reason. It also may not appear inside a
    :func:`diff` target (that would make ``diff`` return a cell-summed derivative);
    ``cellwise(diff(...))`` and the F-bar spelling above are the supported orderings, and the error
    says so.
    """
    return _attach_coords(Cellwise(_u(x)), [x])


def eigvalsh(x) -> FunctionCall:
    """Ascending eigenvalues of a **symmetric** tensor over the last two axes →
    vector of principal values (per leading index).

    Uses the symmetric solver ``jnp.linalg.eigh``. Useful for principal
    stretches / principal stresses and isotropic yield functions. (Eigenvalues
    only — eigenvectors are not returned, as they are ill-defined at repeated
    eigenvalues; for a spectral matrix function use :func:`logm`/:func:`expm`/
    :func:`sqrtm`, which stay differentiable there.)

    Args:
        x: a symmetric tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the eigenvalues, ascending, ``(..., n)``.
    """
    return _attach_coords(FunctionCall(lambda a: jnp.linalg.eigh(a)[0], [_u(x)], name="eigvalsh"), [x])


def logm(x) -> FunctionCall:
    """Symmetric matrix logarithm ``logm(A) = V diag(log λ) Vᵀ`` over the last two
    axes → tensor (SPD input).

    The gradient stays finite at **repeated eigenvalues** (Daleckiĭ–Kreĭn form),
    so a log-strain return map is differentiable through equal principal
    stretches. Distinct from the element-wise scalar :func:`log`.

    Example::

        E = 0.5 * jnn.logm(jnn.matmul(F.T, F))   # Hencky (logarithmic) strain

    Args:
        x: a symmetric positive-definite tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the matrix logarithm.
    """
    from .trace.views import _spectral_matrix_function

    return _attach_coords(FunctionCall(lambda a: _spectral_matrix_function(a, "log", 0.0), [_u(x)], name="logm"), [x])


def expm(x) -> FunctionCall:
    """Symmetric matrix exponential ``expm(A) = V diag(exp λ) Vᵀ`` over the last
    two axes → tensor.

    Gradient stable at repeated eigenvalues; the inverse of :func:`logm` on SPD
    inputs. Distinct from the element-wise scalar :func:`exp`.

    Example::

        Fp = jnn.matmul(jnn.expm(dgamma * N), Fp_prev)   # exponential map of plastic flow

    Args:
        x: a symmetric tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the matrix exponential.
    """
    from .trace.views import _spectral_matrix_function

    return _attach_coords(FunctionCall(lambda a: _spectral_matrix_function(a, "exp", 0.0), [_u(x)], name="expm"), [x])


def sqrtm(x) -> FunctionCall:
    """Symmetric matrix square root ``A^(1/2) = V diag(√λ) Vᵀ`` over the last two
    axes → tensor (SPD input).

    Gradient stable at repeated eigenvalues. E.g. the right stretch
    ``U = sqrtm(FᵀF)`` of a polar decomposition. Distinct from the element-wise
    scalar :func:`sqrt`.

    Args:
        x: a symmetric positive-semidefinite tensor expression, ``(..., n, n)``.

    Returns:
        FunctionCall: the principal square root.
    """
    from .trace.views import _spectral_matrix_function

    return _attach_coords(FunctionCall(lambda a: _spectral_matrix_function(a, "pow", 0.5), [_u(x)], name="sqrtm"), [x])


# ============================================================================
# Reduction operations
# ============================================================================


def _grid_shape_of(domain):
    """A domain's structured-grid extent as a tuple, or ``None`` when it has no grid."""
    gs = getattr(domain, "_grid_shape", None)
    if gs is None:
        sg = getattr(domain, "_structured_grid", None)
        gs = sg.get("shape") if isinstance(sg, dict) else None
    return tuple(int(s) for s in gs) if gs else None


def _axis_extent(var: Variable) -> int:
    """How many points ``var`` spans along its own axis.

    A ``Variable`` does not carry this: ``.size`` is its *component* span (the ``[lo, hi]`` slice
    into ``context[tag]``), not a point count. So it comes from the owning domain — the time axis
    from ``context["__time__"]``, a spatial axis from the structured grid, indexed by the
    coordinate's own component index ``dim[0]`` (x=0, y=1, z=2).
    """
    dom = getattr(var, "_domain", None)
    if dom is None:
        raise ValueError(f"axis={var!r}: this Variable has no owning domain, so its extent is unknown.")

    if getattr(var, "axis", "spatial") == "temporal":
        t = getattr(dom, "context", {}).get("__time__")
        if t is None or not hasattr(t, "shape"):
            raise ValueError(f"axis={var!r}: the domain is not time-dependent, so there is no time axis to reduce.")
        return int(t.shape[0])

    gs = _grid_shape_of(dom)
    if gs is None:
        raise ValueError(
            f"axis={var!r}: tag {getattr(var, 'tag', '?')!r} has no structured grid, so its points live on a "
            "single flat axis and no coordinate names one of them. `axis=<Variable>` is defined only on a "
            "structured-grid domain (Geometries.poseidon / Geometries.equi_distant_rect / a structured mesh). "
            "Reduce the whole point axis with axis=None."
        )
    k = int(var.dim[0])
    if k >= len(gs):
        raise ValueError(f"axis={var!r}: component index {k} is out of range for grid {gs}.")
    return gs[k]


def _resolve_axis(shape, var: Variable) -> int:
    """The integer axis of ``shape`` that ``var`` names.

    Matches the coordinate's extent against the array's axes. When several axes share that extent —
    the *common* case, since grids are usually square — the whole grid block is located instead and
    the coordinate indexed within it. Genuinely ambiguous layouts raise rather than guess.
    """
    shape = tuple(int(s) for s in shape)
    extent = _axis_extent(var)
    cand = [i for i, s in enumerate(shape) if s == extent]

    if len(cand) == 1:
        return cand[0]
    if not cand:
        raise ValueError(
            f"axis={var!r}: no axis of an array with shape {shape} has extent {extent}. Either the "
            "expression is not defined on that coordinate's grid, or its layout was transposed."
        )

    gs = _grid_shape_of(getattr(var, "_domain", None))
    offs = []
    if gs and len(gs) <= len(shape):
        offs = [o for o in range(len(shape) - len(gs) + 1) if shape[o : o + len(gs)] == gs]
    if len(offs) == 1:
        return offs[0] + int(var.dim[0])

    raise ValueError(
        f"axis={var!r} is ambiguous on an array with shape {shape}: axes {cand} all have extent "
        f"{extent}, and the grid block {gs} matches at offsets {offs or 'nowhere'}. Reduce with an "
        "explicit integer axis instead."
    )


def _resolve_axes(shape, axis):
    """Turn any ``Variable`` in ``axis`` into an integer axis of ``shape``; ints pass through."""
    if axis is None:
        return None
    if isinstance(axis, (tuple, list)):
        out = tuple(_resolve_axis(shape, a) if isinstance(_u(a), Variable) else int(a) for a in axis)
        if len(set(out)) != len(out):
            raise ValueError(f"axis={axis!r}: resolves to duplicate axes {out} of shape {tuple(shape)}.")
        return out
    return _resolve_axis(shape, _u(axis)) if isinstance(_u(axis), Variable) else axis


def _reduction_call(jnp_fn, name, x, axis, keepdims) -> FunctionCall:
    """``jnp_fn(x, axis=axis, keepdims=keepdims)`` as a traced reduction node (typed views unwrapped)."""

    # `axis` may be a coordinate Variable (or a tuple containing them). It is resolved to an
    # integer HERE, inside the closure, because that is the first point where the array — and
    # therefore its shape — actually exists; a trace expression carries no shape. `reduces_axis`
    # keeps the UNRESOLVED value: it is what the user wrote.
    def _fn(a, _axis=axis, _keepdims=keepdims):
        return jnp_fn(a, axis=_resolve_axes(a.shape, _axis), keepdims=_keepdims)

    return FunctionCall(_fn, [_u(x)], name=name, reduces_axis=axis)


def sum(x, axis=None, keepdims=False) -> FunctionCall:
    """Sum of a traced expression, ``jnp.sum``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.sum, "sum", x, axis, keepdims)


def mean(x, axis=None, keepdims=False) -> FunctionCall:
    """Arithmetic mean of a traced expression, ``jnp.mean``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.mean, "mean", x, axis, keepdims)


def median(x, axis=None, keepdims=False) -> FunctionCall:
    """Median of a traced expression, ``jnp.median``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.median, "median", x, axis, keepdims)


def std(x, axis=None, keepdims=False) -> FunctionCall:
    """Standard deviation of a traced expression, ``jnp.std``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.std, "std", x, axis, keepdims)


def var(x, axis=None, keepdims=False) -> FunctionCall:
    """Variance of a traced expression, ``jnp.var``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.var, "var", x, axis, keepdims)


def min(x, axis=None, keepdims=False) -> FunctionCall:
    """Minimum of a traced expression, ``jnp.min``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.min, "min", x, axis, keepdims)


def max(x, axis=None, keepdims=False) -> FunctionCall:
    """Maximum of a traced expression, ``jnp.max``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.max, "max", x, axis, keepdims)


def prod(x, axis=None, keepdims=False) -> FunctionCall:
    """Product of a traced expression, ``jnp.prod``, over all or some axes.

    Args:
        x: a traced expression, view, array or number.
        axis: ``None`` reduces over every axis; an int or tuple of ints picks axes; a coordinate
            ``Variable`` (or a tuple containing them) picks the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the reduced node.
    """
    return _reduction_call(jnp.prod, "prod", x, axis, keepdims)


def norm(x, ord=None, axis=None, keepdims=False) -> FunctionCall:
    """Vector or matrix norm, ``jnp.linalg.norm``, as a traced node.

    Args:
        x: a traced expression, view or array.
        ord: the norm order, as for ``jnp.linalg.norm`` (``None`` is the 2-norm / Frobenius norm).
        axis: ``None``, an int or tuple of ints, or a coordinate ``Variable`` (or tuple containing
            them) for the axis that coordinate runs along.
        keepdims: keep the reduced axes with size 1.

    Returns:
        FunctionCall: the norm.
    """

    def _fn(a, _axis=axis):
        return jnp.linalg.norm(a, ord=ord, axis=_resolve_axes(a.shape, _axis), keepdims=keepdims)

    return FunctionCall(_fn, [_u(x)], name="norm", reduces_axis=axis)


# ============================================================================
# Comparison operations
# ============================================================================


def maximum(x, y) -> FunctionCall:
    """Element-wise maximum of the two arguments, ``jnp.maximum``, as a traced node.

    Args:
        x: the first argument: a traced expression, view, array or number.
        y: the second argument; broadcast against ``x``.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _binary_call(jnp.maximum, x, y)


def minimum(x, y) -> FunctionCall:
    """Element-wise minimum of the two arguments, ``jnp.minimum``, as a traced node.

    Args:
        x: the first argument: a traced expression, view, array or number.
        y: the second argument; broadcast against ``x``.

    Returns:
        FunctionCall: the result, broadcast to the arguments' shape.
    """
    return _binary_call(jnp.minimum, x, y)


def where(condition, x, y) -> FunctionCall:
    """Return elements chosen from x or y depending on condition."""
    return FunctionCall(jnp.where, [_u(condition), _u(x), _u(y)])


# ============================================================================
# Linear algebra
# ============================================================================


def _small_matmul(a, b):
    from .utils.solver.small_linalg import small_matmul  # lazy: jno.utils.solver imports this module's users

    return small_matmul(a, b)


_small_matmul.__name__ = "matmul"
_small_matmul.__doc__ = jnp.matmul.__doc__

#: FunctionCall names whose result has the rank of their first operand.
_RANK_PRESERVING = {"transpose", "sym", "antisym", "inv", "matmul", "dot"}


def _value_rank(x):
    """Tensor rank of ``x`` per point (its TRAILING axes), or ``None`` when the trace does not say.

    Read from the trace, never from a runtime shape: inside a kernel ``(4, 3)`` may be four 3-vectors or
    one 4x3 matrix. Known: a field's ``value_shape``, a ``grad`` over several coordinates (one more than its
    target), a typed view, a constant (its ``ndim``; ``identity`` is a matrix), elementwise arithmetic (the
    larger rank), ``transpose``/``sym``/``inv`` (unchanged), ``trace``/``det`` (two fewer) and ``dot``/``matmul``
    of known ranks. Anything else -- an opaque ``function``, a single-coordinate derivative -- is ``None``."""
    from .trace import BinaryOp, Literal
    from .trace.views import MatrixView, ScalarView, VectorView

    if isinstance(x, VectorView):
        return 1
    if isinstance(x, MatrixView):
        return 2
    if isinstance(x, ScalarView):
        return 0
    if not isinstance(x, Placeholder):
        return jnp.ndim(x)
    if isinstance(x, Literal):
        return jnp.ndim(x.value)
    if isinstance(x, Jacobian):
        r = _value_rank(x.target)
        return r + 1 if r is not None and len(x.variables) > 1 else None
    if isinstance(x, BinaryOp):
        if x.op not in ("+", "-", "*", "/"):
            return None
        ra, rb = _value_rank(x.left), _value_rank(x.right)
        return None if ra is None or rb is None else builtins.max(ra, rb)
    if isinstance(x, FunctionCall):
        name = getattr(x, "_name", "")
        if name == "identity":
            return 2
        if name == "dot" and getattr(x, "_value_rank", None) is not None:
            return x._value_rank
        if name in _RANK_PRESERVING and x.args:
            return _value_rank(x.args[0])
        if name in ("trace", "det") and x.args:
            r = _value_rank(x.args[0])
            return r - 2 if r is not None and r >= 2 else None
        return None
    shape = getattr(x, "value_shape", None)  # trial / test functions, state fields
    return len(shape) if shape is not None else None


def _per_point_dot(a, b, ra: int, rb: int):
    """``dot`` of one point's rank-``ra`` and rank-``rb`` tensors (``<= 2``), over the trailing axes, with
    the leading batch axes aligned like :func:`inner` aligns them: a test function's extra local-basis
    axis is matched by a singleton inserted after the other operand's batch axes."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    if ra == 0 or rb == 0:
        return a * b
    pa, pb = a.ndim - ra, b.ndim - rb
    if pa < pb:
        a = jnp.reshape(a, a.shape[:pa] + (1,) * (pb - pa) + a.shape[pa:])
    elif pb < pa:
        b = jnp.reshape(b, b.shape[:pb] + (1,) * (pa - pb) + b.shape[pb:])
    if ra == 1 and rb == 1:
        return jnp.sum(a * b, axis=-1)
    from .utils.solver.small_linalg import small_matmul  # lazy: see _small_matmul

    if ra == 2 and rb == 1:
        return small_matmul(a, b[..., None])[..., 0]
    if ra == 1 and rb == 2:
        return small_matmul(a[..., None, :], b)[..., 0, :]
    return small_matmul(a, b)


def dot(x, y):
    """Dot product of per-point tensors: vector·vector (a scalar), matrix·vector, vector·matrix,
    matrix·matrix, or a scaling when one side is a scalar -- NumPy's ``dot`` for each point.

    In a weak form every operand carries leading batch axes (quadrature points, a test function's basis
    axis) and the tensor sits in the trailing ones, the convention of ``transpose`` and ``trace``. Which
    product is meant depends on each operand's tensor rank, which a shape alone cannot tell, so it is read
    from the trace (see :func:`_value_rank`). An operand whose rank the trace does not carry, or a tensor of
    rank above 2, raises: write ``inner(a, b)`` (vectors, any contraction count), ``matmul(a, b)`` /
    ``a @ b`` (matrices) or ``einsum`` instead. Two plain arrays (no trace, no batch axes) are ``jnp.dot``.

    Args:
        x: the first operand (scalar, vector or matrix per point).
        y: the second operand.

    Returns:
        FunctionCall: the product.

    Raises:
        ValueError: if an operand's tensor rank is unknown or above 2.
    """
    if not any(isinstance(v, (Placeholder,) + _view_types()) for v in (x, y)):
        return FunctionCall(jnp.dot, [x, y], name="dot")
    ra, rb = _value_rank(x), _value_rank(y)
    if ra is None or rb is None or ra > 2 or rb > 2:
        which = " and ".join(
            f"{n} (rank {'unknown' if r is None else r})"
            for n, r in (("the first operand", ra), ("the second operand", rb))
            if r is None or r > 2
        )
        raise ValueError(
            f"jno.np.dot: cannot tell which product is meant -- {which}. dot picks vector·vector, "
            "matrix·vector or matrix·matrix from each operand's tensor rank, read from the trace (a field's "
            "value_shape, grad over several coordinates, a view, a constant), and this operand does not carry "
            "one. Write the product explicitly: inner(a, b) for vectors, matmul(a, b) or a @ b for matrices, "
            "or einsum('...ij,...j->...i', A, v)."
        )
    call = _attach_coords(
        FunctionCall(lambda a, b, _ra=ra, _rb=rb: _per_point_dot(a, b, _ra, _rb), [_u(x), _u(y)], name="dot"),
        [x, y],
    )
    call._value_rank = builtins.max(ra, rb) if 0 in (ra, rb) else ra + rb - 2  # a scaling keeps the rank
    return call


def _view_types():
    from .trace.views import _VIEW_TYPES

    return _VIEW_TYPES


def matmul(x, y) -> FunctionCall:
    """Matrix product over the last two axes, per point (``jnp.matmul`` semantics), as a traced node.

    A small per-point product (``M·N·K`` up to 8x8x8) is computed as a broadcast multiply and a sum
    rather than a ``dot_general``, which a GPU would run as a padded batched GEMM once the points are
    batched (see :func:`jno.utils.solver.small_linalg.small_matmul`). The values are the same up to
    summation order.

    Args:
        x: the left matrix (or batch of matrices), ``(..., M, K)``.
        y: the right matrix, ``(..., K, N)``.

    Returns:
        FunctionCall: the product, ``(..., M, N)``.
    """
    return _binary_call(_small_matmul, x, y)


def cross(x, y) -> FunctionCall:
    """Cross product of 3-vectors along the last axis, ``jnp.cross``, as a traced node.

    Args:
        x: the first vector field, ``(..., 3)``.
        y: the second vector field, ``(..., 3)``.

    Returns:
        FunctionCall: the cross product, ``(..., 3)``.
    """
    return _binary_call(jnp.cross, x, y)


# ============================================================================
# Differential operators (pino-specific)
# ============================================================================
def _broadcasts(s1, s2) -> bool:
    """Whether two shapes broadcast under numpy rules (right-aligned, 1s stretch)."""
    for x, y in zip(reversed(s1), reversed(s2)):
        if x != y and x != 1 and y != 1:
            return False
    return True


def inner(x, y, n_contract: int = 1, keepdims: bool = False) -> FunctionCall:
    """Generalized inner product: a contraction over the last ``n_contract`` axes.

    Shape-friendly for weak forms: the lower-rank operand is padded with singleton axes *before* the
    contracted trailing axes, so ``inner(grad_u, grad_phi)`` works both pointwise and in FEM mode,
    where ``grad_phi`` carries an extra basis-function axis.

    Example::

        inner(a, b)               # vector inner product over the last axis
        inner(A, B, n_contract=2) # Frobenius product A : B

    Args:
        x: the first operand.
        y: the second operand.
        n_contract: how many trailing axes to contract (1 for vectors, 2 for ``A : B``).
        keepdims: keep the contracted axes with size 1.

    Returns:
        FunctionCall: the contraction.
    """

    def _fn(a, b, _n=n_contract, _keep=keepdims):
        a = jnp.asarray(a)
        b = jnp.asarray(b)

        if _n < 1:
            return a * b
        if a.ndim < _n or b.ndim < _n:
            raise ValueError("inner(...): n_contract exceeds operand rank")

        a_prefix_ndim = a.ndim - _n
        b_prefix_ndim = b.ndim - _n

        if a_prefix_ndim < b_prefix_ndim:
            pad = (1,) * (b_prefix_ndim - a_prefix_ndim)
            a = jnp.reshape(a, a.shape[:-_n] + pad + a.shape[-_n:])
        elif b_prefix_ndim < a_prefix_ndim:
            pad = (1,) * (a_prefix_ndim - b_prefix_ndim)
            b = jnp.reshape(b, b.shape[:-_n] + pad + b.shape[-_n:])

        axes = tuple(range(-_n, 0))
        if not _broadcasts(a.shape, b.shape):
            # Before the raw broadcast error: the usual cause is a COMPONENT-FIRST operand, which is
            # what `jno.np.stack([f0, f1])` builds (stack defaults to axis=0) while the value axis is
            # trailing everywhere in the assemblers. Detect it by asking whether swapping the first
            # two axes would align, and name it -- the same diagnosis the VPINN lowering gives for the
            # same weak form, so one spelling means one thing on either path.
            for name, arr, other in (("first", a, b), ("second", b, a)):
                if arr.ndim >= 2 and _broadcasts(jnp.moveaxis(arr, 0, -1).shape, other.shape):
                    raise ValueError(
                        f"inner(...): the {name} operand, shape {tuple(arr.shape)}, is COMPONENT-FIRST "
                        f"against shape {tuple(other.shape)} -- its first two axes are the wrong way "
                        "round. That is what `jno.np.stack([f0, f1])` builds, since stack defaults to "
                        "axis=0, while the value axis is trailing here. Write "
                        "`jno.np.stack([f0, f1], axis=-1)`."
                    )
        return jnp.sum(a * b, axis=axes, keepdims=_keep)

    return _attach_coords(FunctionCall(_fn, [_u(x), _u(y)], name="inner", reduces_axis=-1), [x, y])


def double_dot(x, y) -> FunctionCall:
    """Double contraction (Frobenius product) ``A : B = Σ_ij A_ij B_ij`` over the last two axes.

    The same as ``inner(x, y, n_contract=2)``.

    Args:
        x: the first tensor, ``(..., n, m)``.
        y: the second tensor, ``(..., n, m)``.

    Returns:
        FunctionCall: the contraction per point.
    """
    return inner(x, y, n_contract=2)


def _small_einsum(subscripts, *args):
    from .utils.solver.small_linalg import small_einsum  # lazy: jno.utils.solver imports this module's users

    return small_einsum(subscripts, *args)


def einsum(subscripts: str, *operands) -> FunctionCall:
    """Traced jnp.einsum wrapper for compact tensor/vector contractions.

    A contraction of small per-point tensors (``M·N·K`` at most 8x8x8 per pairwise step) is evaluated as a
    broadcast multiply and a sum instead of a ``dot_general``, which a GPU runs as a padded batched GEMM
    once the points are batched (see :func:`jno.utils.solver.small_linalg.small_einsum`); larger ones are
    plain ``jnp.einsum``. Same values up to summation order.

    Args:
        subscripts: the ``jnp.einsum`` subscript string, e.g. ``"...ij,...j->...i"``.
        *operands: the operands (traced expressions or arrays).

    Returns:
        FunctionCall: the contraction.
    """
    return _attach_coords(
        FunctionCall(
            lambda *args, _subs=subscripts: _small_einsum(_subs, *args),
            [_u(o) for o in operands],
            name="einsum",
        ),
        list(operands),
    )


# docstring-from: jno.jnp_ops.divergence
def div(vector_field: List[Placeholder], variables: List[Variable]) -> Placeholder:
    """Compute the divergence of a vector field.

    div(F) = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ...

    Args:
        vector_field: List of expressions [F₁, F₂, ...]
        variables: Corresponding variables [x₁, x₂, ...]

    Returns:
        Divergence as a sum of gradients

    Example::

        div_F = pnp.divergence([Fx, Fy], [x, y])
    """
    return divergence(vector_field, variables)


def grad(target: Placeholder, variable: Variable, scheme: str = "automatic_differentiation") -> Jacobian:
    """Compute the gradient of target with respect to variable.

    Implemented as a single-variable Jacobian.

    Prefer the method-style shorthand on the target expression::

        u_x  = u.d(x)          # ∂u/∂x
        u_xx = u.d(x).d(x)     # ∂²u/∂x² (chainable)

    Args:
        target: Expression to differentiate
        variable: Variable to differentiate with respect to
        scheme: 'automatic_differentiation' (default) or 'finite_difference'

    Returns:
        Jacobian placeholder representing ∂target/∂variable

    Example::

        u_x = pnp.grad(u(x, y), x)  # ∂u/∂x
    """
    _guard(target, scheme)
    if isinstance(variable, (list, tuple)):
        if len(variable) == 0:
            raise ValueError("grad(..., variables) requires at least one variable")
        return Jacobian(_u(target), list(variable), scheme)
    return Jacobian(_u(target), [variable], scheme)


def laplacian(
    target: Placeholder,
    variables: List[Variable] = None,
    scheme: str = "automatic_differentiation",
) -> Hessian:
    """Compute the Laplacian of target with respect to variables.

    Implemented as a Hessian with trace=True (sum of diagonal second derivatives):
    ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...

    Prefer the method-style shorthand::

        lap_u = u.laplacian(x, y)   # ∂²u/∂x² + ∂²u/∂y²

    Args:
        target: Expression to differentiate
        variables: List of variables to differentiate with respect to
        scheme: 'automatic_differentiation' (default) or 'finite_difference'

    Returns:
        Hessian placeholder with trace=True

    Example::

        lap_u = pnp.laplacian(u(x, y), [x, y])  # ∂²u/∂x² + ∂²u/∂y²
    """
    _guard(target, scheme)
    if scheme == "finite_difference" and variables is not None:
        warnings.warn(
            "jno.np.laplacian: the variables passed are not used by the finite-difference laplacian; its "
            "derivatives are computed on the entire spatial grid.",
            stacklevel=2,
        )

    return Hessian(_u(target), variables, scheme, trace=True)


# docstring-from: jno.jnp_ops.laplacian
def laplace(
    target: Placeholder,
    variables: List[Variable],
    scheme: str = "automatic_differentiation",
) -> Hessian:
    """Compute the Laplacian of target with respect to variables.

    Implemented as a Hessian with trace=True (sum of diagonal second derivatives):
    ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...

    Prefer the method-style shorthand::

        lap_u = u.laplacian(x, y)   # ∂²u/∂x² + ∂²u/∂y²

    Args:
        target: Expression to differentiate
        variables: List of variables to differentiate with respect to
        scheme: 'automatic_differentiation' (default) or 'finite_difference'

    Returns:
        Hessian placeholder with trace=True

    Example::

        lap_u = pnp.laplacian(u(x, y), [x, y])  # ∂²u/∂x² + ∂²u/∂y²
    """
    _guard(target, scheme)
    return Hessian(_u(target), variables, scheme, trace=True)


def hessian(
    target: Placeholder,
    variables: List[Variable],
    scheme: str = "automatic_differentiation",
) -> Hessian:
    """Hessian matrix of ``target`` with respect to ``variables``, ``H[i,j] = ∂²u/∂xᵢ∂xⱼ``.

    Example::

        H = jno.np.hessian(u, [x, y])   # 2x2 Hessian per point

    Args:
        target: the expression to differentiate.
        variables: the coordinate Variables.
        scheme: how the derivatives are computed: ``"automatic_differentiation"`` (default, the global
            mode set via :func:`jno.setup`; ``":forward"`` / ``":reverse"`` and, for second order,
            ``":fwd-over-rev"`` style suffixes select the nesting) or ``"finite_difference"`` (optional
            sub-schemes ``":lsq"``, ``":uniform"``, ``":inverse_distance"``, ``":cotangent"``).

    Returns:
        Hessian: the Hessian node.
    """
    _guard(target, scheme)
    return Hessian(_u(target), variables, scheme)


def jacobian(
    target: Placeholder,
    variables: List[Variable],
    scheme: str = "automatic_differentiation",
) -> Jacobian:
    """Jacobian of ``target`` with respect to ``variables``, ``J[i] = ∂u/∂xᵢ``.

    For a vector field the result is ``(..., n_comp, n_vars)``.

    Example::

        J = jno.np.jacobian(u, [x, y])   # gradient (scalar u) or Jacobian (vector u)

    Args:
        target: the expression to differentiate.
        variables: the coordinate Variables.
        scheme: how the derivatives are computed: ``"automatic_differentiation"`` (default, the global
            mode set via :func:`jno.setup`; ``":forward"`` / ``":reverse"`` and, for second order,
            ``":fwd-over-rev"`` style suffixes select the nesting) or ``"finite_difference"`` (optional
            sub-schemes ``":lsq"``, ``":uniform"``, ``":inverse_distance"``, ``":cotangent"``).

    Returns:
        Jacobian: the Jacobian node.
    """
    _guard(target, scheme)
    return Jacobian(_u(target), variables, scheme)


def divergence(vector_field: List[Placeholder], variables: List[Variable]) -> Placeholder:
    """Compute the divergence of a vector field.

    div(F) = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ...

    Args:
        vector_field: List of expressions [F₁, F₂, ...]
        variables: Corresponding variables [x₁, x₂, ...]

    Returns:
        Divergence as a sum of gradients

    Example::

        div_F = pnp.divergence([Fx, Fy], [x, y])
    """
    if len(vector_field) != len(variables):
        raise ValueError("vector_field and variables must have same length")

    for vf in vector_field:
        _guard(vf)
    result: Placeholder = Jacobian(_u(vector_field[0]), [variables[0]])
    for i in range(1, len(vector_field)):
        result = result + Jacobian(_u(vector_field[i]), [variables[i]])
    return result


def curl_2d(Fx: Placeholder, Fy: Placeholder, x: Variable, y: Variable) -> Placeholder:
    """
    Compute the 2D curl (scalar).

    curl(F) = ∂Fy/∂x - ∂Fx/∂y

    Args:
        Fx, Fy: Components of the vector field
        x, y: Spatial variables

    Returns:
        Scalar curl
    """
    _guard(Fx)
    _guard(Fy)
    return Jacobian(_u(Fy), [x]) - Jacobian(_u(Fx), [y])


def curl_3d(
    Fx: Placeholder,
    Fy: Placeholder,
    Fz: Placeholder,
    x: Variable,
    y: Variable,
    z: Variable,
) -> Placeholder:
    """Curl of a 3-D vector field given by components::

        curl(F) = (∂Fz/∂y - ∂Fy/∂z,  ∂Fx/∂z - ∂Fz/∂x,  ∂Fy/∂x - ∂Fx/∂y)

    Args:
        Fx: the x component of ``F(x, y, z)``.
        Fy: the y component.
        Fz: the z component.
        x: the x Variable.
        y: the y Variable.
        z: the z Variable.

    Returns:
        A 3-component expression, the curl vector.
    """
    Fx, Fy, Fz = _u(Fx), _u(Fy), _u(Fz)
    _guard(Fx)
    _guard(Fy)
    _guard(Fz)
    curl_x = Jacobian(Fz, [y]) - Jacobian(Fy, [z])
    curl_y = Jacobian(Fx, [z]) - Jacobian(Fz, [x])
    curl_z = Jacobian(Fy, [x]) - Jacobian(Fx, [y])
    return stack([curl_x, curl_y, curl_z], axis=-1)


def integrate(expr: Placeholder, *, quadrature: "str | int" = "nodal") -> "Integral":
    """Integrate a scalar expression over its mesh domain region.

    Shorthand for ``expr.integrate(quadrature=...)``.  The region (boundary vs
    volume) is auto-detected from the Variable tags inside ``expr`` via
    ``domain._boundary_registry``.

    The expression is evaluated at the region's quadrature points and reduced to
    a scalar via a weighted sum.  ``quadrature`` selects the volume rule:
    ``"nodal"`` (default, the P1 vertex rule using nodal measures) or ``"gauss"``
    / an integer degree (element Gauss quadrature — higher-order, alias-resistant;
    volume regions only).  For flux integrals, compute F·n explicitly first::

        (Fx(x_b, y_b) * nx + Fy(x_b, y_b) * ny).integrate()
        integrate(Fx(x_b, y_b) * nx + Fy(x_b, y_b) * ny)  # equivalent

    Args:
        expr: Scalar-valued Placeholder expression.
        quadrature: ``"nodal"`` (default), ``"gauss"``, or an integer Gauss degree.

    Returns:
        Integral node that evaluates to a scalar.
    """
    return Integral(_u(expr), quadrature=quadrature)


def test(name: str = "phi") -> TestFunction:
    """Create a generic variational test function symbol.

    Args:
        name: the symbol name.

    Returns:
        TestFunction: the symbol. Prefer ``u.test()`` on an unknown.
    """
    return TestFunction(name=name)


def trial(name: str = "u") -> TrialFunction:
    """Create a generic variational unknown symbol.

    Args:
        name: the symbol name.

    Returns:
        TrialFunction: the symbol. Prefer ``d.unknown()``.
    """
    return TrialFunction(name=name)


# ============================================================================
# Array creation and dtypes — plain re-exports from jax.numpy
# ============================================================================
from jax.numpy import (  # noqa: F401, E402
    arange,
    array,
    asarray,
    bool_,
    complex64,
    complex128,
    eye,
    float32,
    float64,
    full,
    int32,
    int64,
    linspace,
    meshgrid,
    ones,
    zeros,
)


def _linalg_call(name, args, kwargs) -> FunctionCall:
    """``jnp.linalg.<name>(*args, **kwargs)`` as a traced node."""
    return FunctionCall(getattr(jnp.linalg, name), list(args), name=name, kwargs=kwargs or None)


class _linalg:
    """``jno.np.linalg``: every ``jax.numpy.linalg`` function, returning a traced node.

    Each function takes the arguments of its ``jax.numpy.linalg`` namesake; traced expressions among
    them are evaluated first, so ``jno.np.linalg.solve(A, b)`` builds a node that solves per point::

        x = jno.np.linalg.solve(A, b)

    For differentiable matrix functions of symmetric tensors at repeated eigenvalues prefer
    :func:`logm`, :func:`expm`, :func:`sqrtm` and :func:`eigvalsh` in ``jno.np``.
    """

    @staticmethod
    def cholesky(*args, **kwargs) -> FunctionCall:
        """Cholesky factor of a Hermitian positive-definite matrix: ``jnp.linalg.cholesky`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.cholesky``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.cholesky(*args, **kwargs)``.
        """
        return _linalg_call("cholesky", args, kwargs)

    @staticmethod
    def cond(*args, **kwargs) -> FunctionCall:
        """Condition number of a matrix: ``jnp.linalg.cond`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.cond``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.cond(*args, **kwargs)``.
        """
        return _linalg_call("cond", args, kwargs)

    @staticmethod
    def cross(*args, **kwargs) -> FunctionCall:
        """Cross product of 3-vectors: ``jnp.linalg.cross`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.cross``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.cross(*args, **kwargs)``.
        """
        return _linalg_call("cross", args, kwargs)

    @staticmethod
    def det(*args, **kwargs) -> FunctionCall:
        """Determinant: ``jnp.linalg.det`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.det``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.det(*args, **kwargs)``.
        """
        return _linalg_call("det", args, kwargs)

    @staticmethod
    def diagonal(*args, **kwargs) -> FunctionCall:
        """Diagonal of a matrix (batched): ``jnp.linalg.diagonal`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.diagonal``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.diagonal(*args, **kwargs)``.
        """
        return _linalg_call("diagonal", args, kwargs)

    @staticmethod
    def eig(*args, **kwargs) -> FunctionCall:
        """Eigenvalues and eigenvectors of a general matrix: ``jnp.linalg.eig`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.eig``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.eig(*args, **kwargs)``.
        """
        return _linalg_call("eig", args, kwargs)

    @staticmethod
    def eigh(*args, **kwargs) -> FunctionCall:
        """Eigenvalues and eigenvectors of a Hermitian/symmetric matrix: ``jnp.linalg.eigh`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.eigh``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.eigh(*args, **kwargs)``.
        """
        return _linalg_call("eigh", args, kwargs)

    @staticmethod
    def eigvals(*args, **kwargs) -> FunctionCall:
        """Eigenvalues of a general matrix: ``jnp.linalg.eigvals`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.eigvals``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.eigvals(*args, **kwargs)``.
        """
        return _linalg_call("eigvals", args, kwargs)

    @staticmethod
    def eigvalsh(*args, **kwargs) -> FunctionCall:
        """Eigenvalues of a Hermitian/symmetric matrix: ``jnp.linalg.eigvalsh`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.eigvalsh``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.eigvalsh(*args, **kwargs)``.
        """
        return _linalg_call("eigvalsh", args, kwargs)

    @staticmethod
    def inv(*args, **kwargs) -> FunctionCall:
        """Matrix inverse: ``jnp.linalg.inv`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.inv``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.inv(*args, **kwargs)``.
        """
        return _linalg_call("inv", args, kwargs)

    @staticmethod
    def lstsq(*args, **kwargs) -> FunctionCall:
        """Least-squares solution of ``a x = b``: ``jnp.linalg.lstsq`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.lstsq``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.lstsq(*args, **kwargs)``.
        """
        return _linalg_call("lstsq", args, kwargs)

    @staticmethod
    def matmul(*args, **kwargs) -> FunctionCall:
        """Matrix product: ``jnp.linalg.matmul`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.matmul``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.matmul(*args, **kwargs)``.
        """
        return _linalg_call("matmul", args, kwargs)

    @staticmethod
    def matrix_norm(*args, **kwargs) -> FunctionCall:
        """Matrix norm: ``jnp.linalg.matrix_norm`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.matrix_norm``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.matrix_norm(*args, **kwargs)``.
        """
        return _linalg_call("matrix_norm", args, kwargs)

    @staticmethod
    def matrix_power(*args, **kwargs) -> FunctionCall:
        """Integer power of a square matrix: ``jnp.linalg.matrix_power`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.matrix_power``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.matrix_power(*args, **kwargs)``.
        """
        return _linalg_call("matrix_power", args, kwargs)

    @staticmethod
    def matrix_rank(*args, **kwargs) -> FunctionCall:
        """Rank of a matrix: ``jnp.linalg.matrix_rank`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.matrix_rank``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.matrix_rank(*args, **kwargs)``.
        """
        return _linalg_call("matrix_rank", args, kwargs)

    @staticmethod
    def matrix_transpose(*args, **kwargs) -> FunctionCall:
        """Transpose of the last two axes: ``jnp.linalg.matrix_transpose`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.matrix_transpose``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.matrix_transpose(*args, **kwargs)``.
        """
        return _linalg_call("matrix_transpose", args, kwargs)

    @staticmethod
    def multi_dot(*args, **kwargs) -> FunctionCall:
        """Chained matrix product in the cheapest order: ``jnp.linalg.multi_dot`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.multi_dot``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.multi_dot(*args, **kwargs)``.
        """
        return _linalg_call("multi_dot", args, kwargs)

    @staticmethod
    def norm(*args, **kwargs) -> FunctionCall:
        """Vector or matrix norm: ``jnp.linalg.norm`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.norm``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.norm(*args, **kwargs)``.
        """
        return _linalg_call("norm", args, kwargs)

    @staticmethod
    def outer(*args, **kwargs) -> FunctionCall:
        """Outer product of two vectors: ``jnp.linalg.outer`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.outer``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.outer(*args, **kwargs)``.
        """
        return _linalg_call("outer", args, kwargs)

    @staticmethod
    def pinv(*args, **kwargs) -> FunctionCall:
        """Moore–Penrose pseudo-inverse: ``jnp.linalg.pinv`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.pinv``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.pinv(*args, **kwargs)``.
        """
        return _linalg_call("pinv", args, kwargs)

    @staticmethod
    def qr(*args, **kwargs) -> FunctionCall:
        """QR decomposition: ``jnp.linalg.qr`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.qr``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.qr(*args, **kwargs)``.
        """
        return _linalg_call("qr", args, kwargs)

    @staticmethod
    def slogdet(*args, **kwargs) -> FunctionCall:
        """Sign and log-absolute-value of the determinant: ``jnp.linalg.slogdet`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.slogdet``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.slogdet(*args, **kwargs)``.
        """
        return _linalg_call("slogdet", args, kwargs)

    @staticmethod
    def solve(*args, **kwargs) -> FunctionCall:
        """Solution of the linear system ``a x = b``: ``jnp.linalg.solve`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.solve``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.solve(*args, **kwargs)``.
        """
        return _linalg_call("solve", args, kwargs)

    @staticmethod
    def svd(*args, **kwargs) -> FunctionCall:
        """Singular value decomposition: ``jnp.linalg.svd`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.svd``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.svd(*args, **kwargs)``.
        """
        return _linalg_call("svd", args, kwargs)

    @staticmethod
    def svdvals(*args, **kwargs) -> FunctionCall:
        """Singular values: ``jnp.linalg.svdvals`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.svdvals``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.svdvals(*args, **kwargs)``.
        """
        return _linalg_call("svdvals", args, kwargs)

    @staticmethod
    def tensordot(*args, **kwargs) -> FunctionCall:
        """Tensor contraction over given axes: ``jnp.linalg.tensordot`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.tensordot``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.tensordot(*args, **kwargs)``.
        """
        return _linalg_call("tensordot", args, kwargs)

    @staticmethod
    def tensorinv(*args, **kwargs) -> FunctionCall:
        """Inverse of a tensor (reshaped as a matrix): ``jnp.linalg.tensorinv`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.tensorinv``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.tensorinv(*args, **kwargs)``.
        """
        return _linalg_call("tensorinv", args, kwargs)

    @staticmethod
    def tensorsolve(*args, **kwargs) -> FunctionCall:
        """Solution of a tensor equation ``a x = b``: ``jnp.linalg.tensorsolve`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.tensorsolve``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.tensorsolve(*args, **kwargs)``.
        """
        return _linalg_call("tensorsolve", args, kwargs)

    @staticmethod
    def trace(*args, **kwargs) -> FunctionCall:
        """Trace of a matrix: ``jnp.linalg.trace`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.trace``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.trace(*args, **kwargs)``.
        """
        return _linalg_call("trace", args, kwargs)

    @staticmethod
    def vecdot(*args, **kwargs) -> FunctionCall:
        """Vector dot product along an axis: ``jnp.linalg.vecdot`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.vecdot``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.vecdot(*args, **kwargs)``.
        """
        return _linalg_call("vecdot", args, kwargs)

    @staticmethod
    def vector_norm(*args, **kwargs) -> FunctionCall:
        """Vector norm: ``jnp.linalg.vector_norm`` as a traced node.

        Args:
            *args: the positional arguments of ``jax.numpy.linalg.vector_norm``; traced expressions are evaluated first.
            **kwargs: its keyword arguments, passed unchanged.

        Returns:
            FunctionCall: the node; it evaluates to ``jnp.linalg.vector_norm(*args, **kwargs)``.
        """
        return _linalg_call("vector_norm", args, kwargs)


# A function JAX adds later is still reachable, without a hover text of its own.
for _name in dir(jnp.linalg):
    if not _name.startswith("_") and callable(getattr(jnp.linalg, _name)) and not hasattr(_linalg, _name):
        setattr(_linalg, _name, staticmethod(lambda *a, _n=_name, **k: _linalg_call(_n, a, k)))

linalg = _linalg
