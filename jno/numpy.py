import jax.numpy as jnp
from typing import Union, List, Sequence, Callable
from .trace import Placeholder, Variable, FunctionCall, Concat, Hessian, Jacobian, Constant, ConstantNamespace, BinaryOp

# Keep import so people can use jno.numpy as jno -> jno.model, jno.tune
from .tuner import Arch, ArchSpace, tune
from .architectures.models import nn, parameter
from .trace import Tracker
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

pi = jnp.pi
π = jnp.pi
e = jnp.e
inf = jnp.inf
nan = jnp.nan


def tracker(op: Placeholder, interval: int = 1) -> Tracker:
    """Mark an expression as a tracked metric.

    Trackers are monitored during training but do NOT contribute to
    the loss function or gradient computation.

    Args:
        op: The expression to monitor.
        interval: Evaluate every *interval* epochs (default: every epoch).
    """

    return Tracker(op, interval)


def constant(tag: str, data: Union[dict, str, Path]) -> ConstantNamespace:
    """
    Create a ConstantNamespace from a dict or file.

    Args:
        tag: Name for this constant group
        data: Dict of values, or path to .json/.yaml/.yml/.toml/.pkl/.npz file

    Returns:
        ConstantNamespace with attribute access to constants

    Examples:
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
    return FunctionCall(fn, args, name, reduces_axis)


# ============================================================================
# Factory for simple unary wrappers
# ============================================================================


def _unary(jnp_fn):
    """Create a unary wrapper that handles Placeholder and plain arrays."""

    def wrapper(x):
        if isinstance(x, Placeholder):
            return FunctionCall(jnp_fn, [x])
        return jnp_fn(x)

    wrapper.__name__ = jnp_fn.__name__
    wrapper.__doc__ = jnp_fn.__doc__
    return wrapper


def _binary(jnp_fn):
    """Create a binary wrapper that handles Placeholder and plain arrays."""

    def wrapper(x, y):
        if isinstance(x, Placeholder) or isinstance(y, Placeholder):
            return FunctionCall(jnp_fn, [x, y])
        return jnp_fn(x, y)

    wrapper.__name__ = jnp_fn.__name__
    wrapper.__doc__ = jnp_fn.__doc__
    return wrapper


# Trigonometric
sin = _unary(jnp.sin)
cos = _unary(jnp.cos)
tan = _unary(jnp.tan)
arcsin = _unary(jnp.arcsin)
arccos = _unary(jnp.arccos)
arctan = _unary(jnp.arctan)
arctan2 = _binary(jnp.arctan2)
atan2 = arctan2

# Hyperbolic
sinh = _unary(jnp.sinh)
cosh = _unary(jnp.cosh)
tanh = _unary(jnp.tanh)
arcsinh = _unary(jnp.arcsinh)
arccosh = _unary(jnp.arccosh)
arctanh = _unary(jnp.arctanh)

# Exponential / logarithmic
exp = _unary(jnp.exp)
exp2 = _unary(jnp.exp2)
expm1 = _unary(jnp.expm1)
log = _unary(jnp.log)
log2 = _unary(jnp.log2)
log10 = _unary(jnp.log10)
log1p = _unary(jnp.log1p)

# Power / root
sqrt = _unary(jnp.sqrt)
cbrt = _unary(jnp.cbrt)
square = _unary(jnp.square)
power = _binary(jnp.power)

# Rounding / absolute
abs = _unary(jnp.abs)
floor = _unary(jnp.floor)
ceil = _unary(jnp.ceil)
round = _unary(jnp.round)
sign = _unary(jnp.sign)


# ============================================================================
# Array manipulation
# ============================================================================
import jax


@jax.tree_util.register_pytree_node_class
class ViewFactorOp:
    """
    Linear boundary radiation operator.

    Supports:
        F @ x
        x @ F

    Works with:
        • jnp.ndarray
        • Placeholder
    """

    def __init__(self, F: Union["Placeholder", jnp.ndarray]):
        self.F = F

    # -----------------------
    # Matrix multiply: F @ x
    # -----------------------
    def __matmul__(self, x: Union["Placeholder", jnp.ndarray]):
        if isinstance(self.F, Placeholder) or isinstance(x, Placeholder):
            return FunctionCall(lambda A, b: A @ b, [self.F, x])
        return self.F @ x

    # -----------------------
    # Left multiply: x @ F
    # -----------------------
    def __rmatmul__(self, x: Union["Placeholder", jnp.ndarray]):
        if isinstance(self.F, Placeholder) or isinstance(x, Placeholder):
            return FunctionCall(lambda b, A: b @ A, [x, self.F])
        return x @ self.F

    # -----------------------
    # Apply operator explicitly
    # -----------------------
    def apply(self, x: Union["Placeholder", jnp.ndarray]):
        return self @ x

    # -----------------------
    # Solve (I - αF)x = rhs
    # -----------------------
    def solve(self, rhs, alpha):
        if isinstance(self.F, Placeholder) or isinstance(rhs, Placeholder):

            def solve_fn(A, b, a):
                I = jnp.eye(A.shape[0])
                return jnp.linalg.solve(I - a * A, b)

            return FunctionCall(solve_fn, [self.F, rhs, alpha])

        I = jnp.eye(self.F.shape[0])
        return jnp.linalg.solve(I - alpha * self.F, rhs)

    # -----------------------
    # PyTree support for JAX
    # -----------------------
    def tree_flatten(self):
        return (self.F,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def view_factor(F: Union["Placeholder", jnp.ndarray]) -> Union[ViewFactorOp, jnp.ndarray]:
    """Create a view factor operator."""
    if isinstance(F, Placeholder):
        return ViewFactorOp(F)
    return ViewFactorOp(F)


def concat(items: Sequence[Union[Placeholder, jnp.ndarray]], axis: int = -1) -> Union[Concat, jnp.ndarray]:
    """Concatenate arrays or placeholders along an axis."""
    has_placeholder = any(isinstance(item, Placeholder) for item in items)
    if has_placeholder:
        return Concat(list(items), axis=axis)
    return jnp.concatenate(items, axis=axis)


def concatenate(items: Sequence[Union[Placeholder, jnp.ndarray]], axis: int = -1) -> Union[Concat, jnp.ndarray]:
    """Alias for concat."""
    return concat(items, axis=axis)


def stack(items: Sequence[Union[Placeholder, jnp.ndarray]], axis: int = 0) -> Union[Concat, FunctionCall, jnp.ndarray]:
    """Stack arrays along a new axis.

    For axis=-1, uses Concat which enables proper dimension inference for neural networks.
    """
    has_placeholder = any(isinstance(item, Placeholder) for item in items)
    if has_placeholder:
        # For axis=-1 (last axis), use Concat for proper dimension tracking
        if axis == -1:
            return Concat(list(items), axis=-1)
        # For other axes, use FunctionCall
        return FunctionCall(lambda *args: jnp.stack(args, axis=axis), list(items))
    return jnp.stack(items, axis=axis)


def reshape(x: Union[Placeholder, jnp.ndarray], shape: tuple) -> Union[FunctionCall, jnp.ndarray]:
    """Reshape array."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.reshape(a, shape), [x])
    return jnp.reshape(x, shape)


def squeeze(x: Union[Placeholder, jnp.ndarray], axis: int = None) -> Union[FunctionCall, jnp.ndarray]:
    """Remove single-dimensional entries."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.squeeze(a, axis=axis), [x])
    return jnp.squeeze(x, axis=axis)


def expand_dims(x: Union[Placeholder, jnp.ndarray], axis: int) -> Union[FunctionCall, jnp.ndarray]:
    """Expand array dimensions."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.expand_dims(a, axis=axis), [x])
    return jnp.expand_dims(x, axis=axis)


def transpose(x: Union[Placeholder, jnp.ndarray], axes: tuple = None) -> Union[FunctionCall, jnp.ndarray]:
    """Transpose array."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.transpose(a, axes=axes), [x])
    return jnp.transpose(x, axes=axes)


# ============================================================================
# Reduction operations
# ============================================================================


def sum(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Sum of array elements."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.sum(a, axis=axis, keepdims=keepdims), [x], name="sum", reduces_axis=axis)
    return jnp.sum(x, axis=axis, keepdims=keepdims)


def mean(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Mean of array elements."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.mean(a, axis=axis, keepdims=keepdims), [x], name="mean", reduces_axis=axis)
    return jnp.mean(x, axis=axis, keepdims=keepdims)


def median(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Median of array elements."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.median(a, axis=axis, keepdims=keepdims), [x], name="median", reduces_axis=axis)
    return jnp.median(x, axis=axis, keepdims=keepdims)


def std(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Standard deviation."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.std(a, axis=axis, keepdims=keepdims), [x], name="std", reduces_axis=axis)
    return jnp.std(x, axis=axis, keepdims=keepdims)


def var(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Variance."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.var(a, axis=axis, keepdims=keepdims), [x], name="var", reduces_axis=axis)
    return jnp.var(x, axis=axis, keepdims=keepdims)


def min(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Minimum value."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.min(a, axis=axis, keepdims=keepdims), [x], name="min", reduces_axis=axis)
    return jnp.min(x, axis=axis, keepdims=keepdims)


def max(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Maximum value."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.max(a, axis=axis, keepdims=keepdims), [x], name="max", reduces_axis=axis)
    return jnp.max(x, axis=axis, keepdims=keepdims)


def prod(x: Union[Placeholder, jnp.ndarray], axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Product of array elements."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.prod(a, axis=axis, keepdims=keepdims), [x], name="prod", reduces_axis=axis)
    return jnp.prod(x, axis=axis, keepdims=keepdims)


def norm(x: Union[Placeholder, jnp.ndarray], ord: int = None, axis: int = None, keepdims: bool = False) -> Union[FunctionCall, jnp.ndarray]:
    """Vector/matrix norm."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims), [x], name="norm", reduces_axis=axis)
    return jnp.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


# ============================================================================
# Comparison operations
# ============================================================================


maximum = _binary(jnp.maximum)
minimum = _binary(jnp.minimum)


def where(condition, x, y) -> Union[FunctionCall, jnp.ndarray]:
    """Return elements chosen from x or y depending on condition."""
    if isinstance(condition, Placeholder) or isinstance(x, (Placeholder, int, float)) or isinstance(y, (Placeholder, int, float)):
        return FunctionCall(jnp.where, [condition, x, y])
    return jnp.where(condition, x, y)


# ============================================================================
# Linear algebra
# ============================================================================


dot = _binary(jnp.dot)
matmul = _binary(jnp.matmul)
cross = _binary(jnp.cross)


# ============================================================================
# Differential operators (pino-specific)
# ============================================================================


def grad(target: Placeholder, variable: Variable, scheme: str = "automatic_differentiation") -> Jacobian:
    """
    Compute the gradient of target with respect to variable.

    Implemented as a single-variable Jacobian.

    Args:
        target: Expression to differentiate
        variable: Variable to differentiate with respect to
        scheme: 'automatic_differentiation' (default) or 'finite_difference'

    Returns:
        Jacobian placeholder representing ∂target/∂variable

    Example:
        u_x = pnp.grad(u(x, y), x)  # ∂u/∂x
    """
    return Jacobian(target, [variable], scheme)


def laplacian(target: Placeholder, variables: List[Variable] = None, scheme: str = "automatic_differentiation") -> Hessian:
    """
    Compute the Laplacian of target with respect to variables.

    Implemented as a Hessian with trace=True (sum of diagonal second derivatives):
    ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...

    Args:
        target: Expression to differentiate
        variables: List of variables to differentiate with respect to
        scheme: 'automatic_differentiation' (default) or 'finite_difference'

    Returns:
        Hessian placeholder with trace=True

    Example:
        lap_u = pnp.laplacian(u(x, y), [x, y])  # ∂²u/∂x² + ∂²u/∂y²
    """
    if scheme == "finite_difference" and variables is not None:
        print("Variables were selected for the finite difference laplacian which are not used. The finite difference derivatives are computed on the entire spatial grid.")

    return Hessian(target, variables, scheme, trace=True)


def laplace(target: Placeholder, variables: List[Variable], scheme: str = "automatic_differentiation") -> Hessian:
    """Alias for laplacian."""
    return Hessian(target, variables, scheme, trace=True)


def hessian(target: Placeholder, variables: List[Variable], scheme: str = "automatic_differentiation") -> Hessian:
    """
    Compute the Hessian matrix of target with respect to variables.

    The Hessian is the matrix of second derivatives:
    H[i,j] = ∂²u/∂xᵢ∂xⱼ

    Args:
        target: Expression to differentiate
        variables: List of variables

    Returns:
        Hessian placeholder representing the full Hessian matrix

    Example:
        H = pnp.hessian(u(x, y), [x, y])  # 2x2 Hessian matrix
    """
    return Hessian(target, variables, scheme)


def jacobian(target: Placeholder, variables: List[Variable], scheme: str = "automatic_differentiation") -> Jacobian:
    """
    Compute the Jacobian matrix of target with respect to variables.

    The Jacobian is the matrix of first derivatives:
    J[i] = \u2202u/\u2202x\u1d62

    Args:
        target: Expression to differentiate
        variables: List of variables

    Returns:
        Jacobian placeholder representing the full Jacobian matrix

    Example:
        J = pnp.jacobian(u(x, y), [x, y])  # 2-element Jacobian vector
    """
    return Jacobian(target, variables, scheme)


def divergence(vector_field: List[Placeholder], variables: List[Variable]) -> Placeholder:
    """
    Compute the divergence of a vector field.

    div(F) = ∂F₁/∂x₁ + ∂F₂/∂x₂ + ...

    Args:
        vector_field: List of expressions [F₁, F₂, ...]
        variables: Corresponding variables [x₁, x₂, ...]

    Returns:
        Divergence as a sum of gradients

    Example:
        div_F = pnp.divergence([Fx, Fy], [x, y])
    """
    if len(vector_field) != len(variables):
        raise ValueError("vector_field and variables must have same length")

    result = Jacobian(vector_field[0], [variables[0]])
    for i in range(1, len(vector_field)):
        result = result + Jacobian(vector_field[i], [variables[i]])
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
    return Jacobian(Fy, [x]) - Jacobian(Fx, [y])


def curl_3d(Fx: Placeholder, Fy: Placeholder, Fz: Placeholder, x: Variable, y: Variable, z: Variable) -> Placeholder:
    """
    Compute the 3D curl (vector).

    curl(F) = [ ∂Fz/∂y - ∂Fy/∂z,
                ∂Fx/∂z - ∂Fz/∂x,
                ∂Fy/∂x - ∂Fx/∂y ]

    Args:
        Fx, Fy, Fz: Components of the vector field F(x, y, z)
        x, y, z:    Spatial variables

    Returns:
        A 3-component Placeholder representing the curl vector
    """
    curl_x = Jacobian(Fz, [y]) - Jacobian(Fy, [z])
    curl_y = Jacobian(Fx, [z]) - Jacobian(Fz, [x])
    curl_z = Jacobian(Fy, [x]) - Jacobian(Fx, [y])
    return stack([curl_x, curl_y, curl_z], axis=-1)


# ============================================================================
# Array creation (these return JAX arrays, not placeholders)
# ============================================================================


def zeros(shape, dtype=None):
    """Create array of zeros."""
    return jnp.zeros(shape, dtype=dtype)


def ones(shape, dtype=None):
    """Create array of ones."""
    return jnp.ones(shape, dtype=dtype)


def ones_like(u):
    return u * 0.0 + 1.0


def full(shape, fill_value, dtype=None):
    """Create array filled with value."""
    return jnp.full(shape, fill_value, dtype=dtype)


def arange(start, stop=None, step=None, dtype=None):
    """Create evenly spaced values."""
    return jnp.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, dtype=None):
    """Create evenly spaced values over interval."""
    return jnp.linspace(start, stop, num, dtype=dtype)


def meshgrid(*xi, indexing="xy"):
    """Create coordinate matrices."""
    return jnp.meshgrid(*xi, indexing=indexing)


def eye(n, m=None, dtype=None):
    """Create identity matrix."""
    return jnp.eye(n, m, dtype=dtype)


# ============================================================================
# Type conversions
# ============================================================================


def array(x, dtype=None):
    """Create array from input."""
    return jnp.array(x, dtype=dtype)


def asarray(x, dtype=None):
    """Convert to array."""
    return jnp.asarray(x, dtype=dtype)


# ============================================================================
# Data types
# ============================================================================

float32 = jnp.float32
float64 = jnp.float64
int32 = jnp.int32
int64 = jnp.int64
bool_ = jnp.bool_
complex64 = jnp.complex64
complex128 = jnp.complex128


def _create_linalg_wrapper():
    """Factory function to create the linalg wrapper class."""

    class _linalg:
        """Wrapper for jax.numpy.linalg that returns FunctionCall objects."""

        pass

    # Get all public functions from jnp.linalg
    for name in dir(jnp.linalg):
        if name.startswith("_"):
            continue

        original = getattr(jnp.linalg, name)

        if callable(original):
            # Create wrapper function
            def make_method(func, func_name):
                def method(*args, **kwargs):
                    return FunctionCall(func, list(args), name=func_name, kwargs=kwargs if kwargs else None)

                method.__doc__ = func.__doc__
                method.__name__ = func_name
                return staticmethod(method)

            setattr(_linalg, name, make_method(original, name))

    return _linalg


linalg = _create_linalg_wrapper()
