import jax.numpy as jnp
from typing import Union, List, Sequence, Callable
from .trace import Placeholder, Variable, FunctionCall, Concat, Laplacian, Gradient, Hessian, Jacobian, Constant, ConstantNamespace, EulerResiduals, Tracker, BinaryOp

# Keep import so people can use jno.numpy as jno -> jno.model, jno.tune
from .tuner import Arch, ArchSpace, tune
from .architectures.models import nn, parameter

from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

pi = jnp.pi
π = jnp.pi
e = jnp.e
inf = jnp.inf
nan = jnp.nan


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


def function(fn, args, name: str = "", reduces_axis: int = None):
    return FunctionCall(fn, args, name, reduces_axis)


def tracker(op: BinaryOp, interval: int = 1) -> Tracker:
    return Tracker(op, interval)


# ============================================================================
# Trigonometric functions
# ============================================================================


def sin(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Sine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.sin, [x])
    return jnp.sin(x)


def cos(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Cosine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.cos, [x])
    return jnp.cos(x)


def tan(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Tangent function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.tan, [x])
    return jnp.tan(x)


def arcsin(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Inverse sine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.arcsin, [x])
    return jnp.arcsin(x)


def arccos(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Inverse cosine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.arccos, [x])
    return jnp.arccos(x)


def arctan(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Inverse tangent function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.arctan, [x])
    return jnp.arctan(x)


def arctan2(y: Union[Placeholder, jnp.ndarray], x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Two-argument arctangent."""
    if isinstance(y, Placeholder) or isinstance(x, Placeholder):
        return FunctionCall(jnp.arctan2, [y, x])
    return jnp.arctan2(y, x)


def atan2(y: Union[Placeholder, jnp.ndarray], x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Two-argument arctangent."""
    return arctan2(y, x)


# ============================================================================
# Hyperbolic functions
# ============================================================================


def sinh(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Hyperbolic sine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.sinh, [x])
    return jnp.sinh(x)


def cosh(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Hyperbolic cosine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.cosh, [x])
    return jnp.cosh(x)


def tanh(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Hyperbolic tangent function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.tanh, [x])
    return jnp.tanh(x)


def arcsinh(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Inverse hyperbolic sine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.arcsinh, [x])
    return jnp.arcsinh(x)


def arccosh(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Inverse hyperbolic cosine function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.arccosh, [x])
    return jnp.arccosh(x)


def arctanh(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Inverse hyperbolic tangent function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.arctanh, [x])
    return jnp.arctanh(x)


# ============================================================================
# Exponential and logarithmic functions
# ============================================================================


def exp(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Exponential function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.exp, [x])
    return jnp.exp(x)


def exp2(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """2**x."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.exp2, [x])
    return jnp.exp2(x)


def expm1(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """exp(x) - 1."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.expm1, [x])
    return jnp.expm1(x)


def log(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Natural logarithm."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.log, [x])
    return jnp.log(x)


def log2(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Base-2 logarithm."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.log2, [x])
    return jnp.log2(x)


def log10(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Base-10 logarithm."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.log10, [x])
    return jnp.log10(x)


def log1p(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """log(1 + x)."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.log1p, [x])
    return jnp.log1p(x)


# ============================================================================
# Power and root functions
# ============================================================================


def sqrt(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Square root."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.sqrt, [x])
    return jnp.sqrt(x)


def cbrt(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Cube root."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.cbrt, [x])
    return jnp.cbrt(x)


def square(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Square."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.square, [x])
    return jnp.square(x)


def power(x: Union[Placeholder, jnp.ndarray], y: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """x**y."""
    if isinstance(x, Placeholder) or isinstance(y, Placeholder):
        return FunctionCall(jnp.power, [x, y])
    return jnp.power(x, y)


# ============================================================================
# Rounding functions
# ============================================================================


def abs(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Absolute value."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.abs, [x])
    return jnp.abs(x)


def floor(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Floor."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.floor, [x])
    return jnp.floor(x)


def ceil(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Ceiling."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.ceil, [x])
    return jnp.ceil(x)


def round(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Round to nearest integer."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.round, [x])
    return jnp.round(x)


def clip(x: Union[Placeholder, jnp.ndarray], min_val=None, max_val=None) -> Union[FunctionCall, jnp.ndarray]:
    """Clip values to range."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.clip(a, min_val, max_val), [x])
    return jnp.clip(x, min_val, max_val)


# ============================================================================
# Sign and special functions
# ============================================================================


def sign(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Sign function."""
    if isinstance(x, Placeholder):
        return FunctionCall(jnp.sign, [x])
    return jnp.sign(x)


def heaviside(x: Union[Placeholder, jnp.ndarray], h0: float = 0.5) -> Union[FunctionCall, jnp.ndarray]:
    """Heaviside step function."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.heaviside(a, h0), [x])
    return jnp.heaviside(x, h0)


def sigmoid(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Sigmoid function: 1 / (1 + exp(-x))."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: 1 / (1 + jnp.exp(-a)), [x])
    return 1 / (1 + jnp.exp(-x))


def softplus(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Softplus function: log(1 + exp(x))."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.log1p(jnp.exp(a)), [x])
    return jnp.log1p(jnp.exp(x))


def relu(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """ReLU activation."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.maximum(a, 0), [x])
    return jnp.maximum(x, 0)


def leaky_relu(x: Union[Placeholder, jnp.ndarray], negative_slope: float = 0.01) -> Union[FunctionCall, jnp.ndarray]:
    """Leaky ReLU activation."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.where(a > 0, a, negative_slope * a), [x])
    return jnp.where(x > 0, x, negative_slope * x)


def elu(x: Union[Placeholder, jnp.ndarray], alpha: float = 1.0) -> Union[FunctionCall, jnp.ndarray]:
    """ELU activation."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: jnp.where(a > 0, a, alpha * (jnp.exp(a) - 1)), [x])
    return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))


def gelu(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """GELU activation."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: 0.5 * a * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (a + 0.044715 * a**3))), [x])
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def swish(x: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Swish activation: x * sigmoid(x)."""
    if isinstance(x, Placeholder):
        return FunctionCall(lambda a: a / (1 + jnp.exp(-a)), [x])
    return x / (1 + jnp.exp(-x))


# ============================================================================
# Array manipulation
# ============================================================================


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
    """Mean of array elements."""
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


def maximum(x: Union[Placeholder, jnp.ndarray], y: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Element-wise maximum."""
    if isinstance(x, Placeholder) or isinstance(y, Placeholder):
        return FunctionCall(jnp.maximum, [x, y])
    return jnp.maximum(x, y)


def minimum(x: Union[Placeholder, jnp.ndarray], y: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Element-wise minimum."""
    if isinstance(x, Placeholder) or isinstance(y, Placeholder):
        return FunctionCall(jnp.minimum, [x, y])
    return jnp.minimum(x, y)


def where(condition, x, y) -> Union[FunctionCall, jnp.ndarray]:
    """Return elements chosen from x or y depending on condition."""
    if isinstance(condition, Placeholder) or isinstance(x, Placeholder) or isinstance(y, Placeholder):
        return FunctionCall(jnp.where, [condition, x, y])
    return jnp.where(condition, x, y)


# ============================================================================
# Linear algebra
# ============================================================================


def dot(x: Union[Placeholder, jnp.ndarray], y: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Dot product."""
    if isinstance(x, Placeholder) or isinstance(y, Placeholder):
        return FunctionCall(jnp.dot, [x, y])
    return jnp.dot(x, y)


def matmul(x: Union[Placeholder, jnp.ndarray], y: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Matrix multiplication."""
    if isinstance(x, Placeholder) or isinstance(y, Placeholder):
        return FunctionCall(jnp.matmul, [x, y])
    return jnp.matmul(x, y)


def cross(x: Union[Placeholder, jnp.ndarray], y: Union[Placeholder, jnp.ndarray]) -> Union[FunctionCall, jnp.ndarray]:
    """Cross product."""
    if isinstance(x, Placeholder) or isinstance(y, Placeholder):
        return FunctionCall(jnp.cross, [x, y])
    return jnp.cross(x, y)


# ============================================================================
# Differential operators (pino-specific)
# ============================================================================


def grad(target: Placeholder, variable: Variable, scheme: str = "automatic_differentiation") -> Gradient:
    """
    Compute the gradient of target with respect to variable.

    Args:
        target: Expression to differentiate
        variable: Variable to differentiate with respect to
        scheme: 'automatic_differentiation' (default) or 'finite_difference'

    Returns:
        Gradient placeholder representing ∂target/∂variable

    Example:
        u_x = pnp.grad(u(x, y), x)  # ∂u/∂x
        u_x_fd = pnp.grad(u(x, y), x, scheme='finite_difference')
    """
    return Gradient(target, variable, scheme)


def laplacian(target: Placeholder, variables: List[Variable] = None, scheme: str = "automatic_differentiation") -> Laplacian:
    """
    Compute the Laplacian of target with respect to variables.

    The Laplacian is the sum of second derivatives:
    ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...

    Args:
        target: Expression to differentiate
        variables: List of variables to differentiate with respect to
        scheme: 'automatic_differentiation' (default) or 'finite_difference'

    Returns:
        Laplacian placeholder

    Example:
        lap_u = pnp.laplacian(u(x, y), [x, y])  # ∂²u/∂x² + ∂²u/∂y²
        lap_u_fd = pnp.laplacian(u(x, y), [x, y], scheme='finite_difference')
    """

    if scheme == "finite_difference" and variables is not None:
        print("Variables were selected for the finite difference laplacian which are not used. The finite difference derivatives are computed on the entire spatial grid.")

    return Laplacian(target, variables, scheme)


def laplace(target: Placeholder, variables: List[Variable], scheme: str = "automatic_differentiation") -> Laplacian:
    """Alias for laplacian."""
    return Laplacian(target, variables, scheme)


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
    Compute the Jac matrix of target with respect to variables.

    The Jac is the matrix of second derivatives:
    H[i,j] = ∂²u/∂xᵢ∂xⱼ

    Args:
        target: Expression to differentiate
        variables: List of variables

    Returns:
        Jac placeholder representing the full Jac matrix

    Example:
        H = pnp.Jacobian(u(x, y), [x, y])  # 2x2 Jac matrix
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

    result = Gradient(vector_field[0], variables[0])
    for i in range(1, len(vector_field)):
        result = result + Gradient(vector_field[i], variables[i])
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
    return Gradient(Fy, x) - Gradient(Fx, y)


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
    curl_x = Gradient(Fz, y) - Gradient(Fy, z)
    curl_y = Gradient(Fx, z) - Gradient(Fz, x)
    curl_z = Gradient(Fy, x) - Gradient(Fx, y)
    return stack([curl_x, curl_y, curl_z], axis=-1)


def helmholtz_curlcurl_residuals(p, q, d, x, y, z, alpha=1.0, scheme="automatic_differentiation"):
    """
    Build the three residual fields for the vector Helmholtz curl–curl form.

    Optimized: exploits symmetry of mixed partials (∂²f/∂x∂y = ∂²f/∂y∂x)

    Residuals:
      r_x = (∂²p/∂y² + ∂²p/∂z²) - (∂²q/∂x∂y + ∂²d/∂x∂z) + α·p
      r_y = (∂²q/∂z² + ∂²q/∂x²) - (∂²d/∂y∂z + ∂²p/∂x∂y) + α·q
      r_z = (∂²d/∂x² + ∂²d/∂y²) - (∂²p/∂x∂z + ∂²q/∂y∂z) + α·d
    """
    # ---- First derivatives (9 total, all needed)
    jac_p = jacobian(p, [x, y, z], scheme)
    jac_q = jacobian(q, [x, y, z], scheme)
    jac_d = jacobian(d, [x, y, z], scheme)

    p_x, p_y, p_z = jac_p[..., 0], jac_p[..., 1], jac_p[..., 2]
    q_x, q_y, q_z = jac_q[..., 0], jac_q[..., 1], jac_q[..., 2]
    d_x, d_y, d_z = jac_d[..., 0], jac_d[..., 1], jac_d[..., 2]

    # ---- Second derivatives (optimized: 10 instead of 12)
    # Pure second derivatives (6)
    p_yy = grad(p_y, y, scheme)
    p_zz = grad(p_z, z, scheme)
    q_xx = grad(q_x, x, scheme)
    q_zz = grad(q_z, z, scheme)
    d_xx = grad(d_x, x, scheme)
    d_yy = grad(d_y, y, scheme)

    # Mixed partials (4 unique, reused)
    p_xy = grad(p_x, y, scheme)  # used in r_x and r_y
    p_xz = grad(p_x, z, scheme)  # used in r_z
    q_yz = grad(q_y, z, scheme)  # used in r_z
    d_xz = grad(d_x, z, scheme)  # used in r_x
    d_yz = grad(d_y, z, scheme)  # used in r_y

    # Note: q_xy = grad(q_x, y) = grad(q_y, x) by symmetry
    # But we need q_xy for r_x, and it's not reused elsewhere
    q_xy = grad(q_x, y, scheme)

    # ---- Residuals
    r_x = (p_yy + p_zz) - (q_xy + d_xz) + alpha * p
    r_y = (q_zz + q_xx) - (d_yz + p_xy) + alpha * q
    r_z = (d_xx + d_yy) - (p_xz + q_yz) + alpha * d

    return stack([r_x, r_y, r_z], axis=-1)


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


def euler_residuals(
    rho: Union[Placeholder, jnp.ndarray],
    u: Union[Placeholder, jnp.ndarray],
    v: Union[Placeholder, jnp.ndarray],
    p: Union[Placeholder, jnp.ndarray],
    rho0: Union[Placeholder, jnp.ndarray],
    u0: Union[Placeholder, jnp.ndarray],
    v0: Union[Placeholder, jnp.ndarray],
    p0: Union[Placeholder, jnp.ndarray],
    n_substeps: int = 1,
):
    """
    Compute 2D compressible Euler equation residuals using central finite differences.

    This function computes the residuals of the conservative form of the
    2D compressible Euler equations:

    Equations:
        E = 0.5 * ρ * (u² + v²) + p / (γ - 1)  # Total energy

        con1 = ∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y = 0           # Continuity
        con2 = ∂(ρu)/∂t + ∂(ρu² + p)/∂x + ∂(ρuv)/∂y = 0  # x-momentum
        con3 = ∂(ρv)/∂t + ∂(ρuv)/∂x + ∂(ρv² + p)/∂y = 0  # y-momentum
        con4 = ∂E/∂t + ∂((E+p)u)/∂x + ∂((E+p)v)/∂y = 0   # Energy

    Args:
        rho: Density field, shape (T, H, W) e.g., (21, 128, 128)
        u: x-velocity field, shape (T, H, W)
        v: y-velocity field, shape (T, H, W)
        p: Pressure field, shape (T, H, W)
        dt: Time step size (default 1.0)
        dx: Grid spacing in x/width direction (default 1.0)
        dy: Grid spacing in y/height direction (default 1.0)
        gamma: Ratio of specific heats (default 1.4 for air)

    Returns:
        Stacked residuals with shape (T-2, H-2, W-2, 4)
        Channel order: [continuity, x-momentum, y-momentum, energy]

        For perfect solutions, all residuals should be zero.

    Example:
        >>> import jno.numpy as jnp
        >>>
        >>> # Fields with shape (21, 128, 128)
        >>> rho = ...  # density
        >>> u = ...    # x-velocity
        >>> v = ...    # y-velocity
        >>> p = ...    # pressure
        >>>
        >>> # Compute residuals with physical grid spacing
        >>> residuals = jnp.euler_residuals(rho, u, v, p, dt=0.01, dx=0.1, dy=0.1)
        >>>
        >>> # Access individual residuals
        >>> continuity = residuals[..., 0]
        >>> x_momentum = residuals[..., 1]
        >>> y_momentum = residuals[..., 2]
        >>> energy = residuals[..., 3]

    Note:
        The output loses 2 points in each dimension due to central differencing:
        - Time: T -> T-2 (loses first and last timestep)
        - Height: H -> H-2 (loses top and bottom row)
        - Width: W -> W-2 (loses left and right column)

    """
    # Check if any input is a Placeholder (traced computation)
    if any(isinstance(arg, Placeholder) for arg in [rho, u, v, p, rho0, u0, v0, p0]):
        return EulerResiduals(rho, u, v, p, rho0, u0, v0, p0, n_substeps)

    # Direct computation for concrete arrays
    from .trace_evaluator import EulerResidualsDiff

    return EulerResidualsDiff.compute_euler_residuals_flat(rho, u, v, p, rho0, u0, v0, p0, n_substeps)
