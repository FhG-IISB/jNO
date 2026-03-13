import jax
import jax.numpy as jnp
from typing import List, Union, Sequence
from .trace import Placeholder, Variable, FunctionCall, Hessian, Jacobian, Constant, ConstantNamespace, TestFunction, TrialFunction

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
    """Create a unary wrapper for Placeholder args."""

    def wrapper(x):
        return FunctionCall(jnp_fn, [x])

    wrapper.__name__ = jnp_fn.__name__
    wrapper.__doc__ = jnp_fn.__doc__
    return wrapper


def _binary(jnp_fn):
    """Create a binary wrapper for Placeholder args."""

    def wrapper(x, y):
        return FunctionCall(jnp_fn, [x, y])

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
        return self @ x

    # -----------------------
    # Solve (I - αF)x = rhs
    # -----------------------
    def solve(self, rhs, alpha):
        def solve_fn(A, b, a):
            I = jnp.eye(A.shape[0])
            return jnp.linalg.solve(I - a * A, b)

        return FunctionCall(solve_fn, [self.F, rhs, alpha])

    # -----------------------
    # PyTree support for JAX
    # -----------------------
    def tree_flatten(self):
        return (self.F,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def view_factor(F: Union["Placeholder", jnp.ndarray]) -> ViewFactorOp:
    """Create a view factor operator."""
    return ViewFactorOp(F)


def concat(items, axis: int = -1) -> FunctionCall:
    """Concatenate placeholders along an axis (always axis=-1 at eval time)."""

    def _fn(*args):
        expanded = [a[..., jnp.newaxis] if a.ndim == 1 else a for a in args]
        return jnp.concatenate(expanded, axis=-1)

    return FunctionCall(_fn, list(items), name="concat")


def concatenate(items, axis: int = -1) -> FunctionCall:
    """Alias for concat."""
    return concat(items, axis=axis)


def stack(items, axis: int = 0) -> FunctionCall:
    """Stack placeholders along a new axis."""
    if axis == -1:
        return concat(items, axis=-1)
    return FunctionCall(lambda *args: jnp.stack(args, axis=axis), list(items), name="stack")


def reshape(x, shape: tuple) -> FunctionCall:
    """Reshape a placeholder to a new shape."""
    return FunctionCall(lambda a: jnp.reshape(a, shape), [x])


def squeeze(x, axis: int = None) -> FunctionCall:
    """Remove single-dimensional entries."""
    return FunctionCall(lambda a: jnp.squeeze(a, axis=axis), [x])


def expand_dims(x, axis: int) -> FunctionCall:
    """Expand array dimensions."""
    return FunctionCall(lambda a: jnp.expand_dims(a, axis=axis), [x])


def transpose(x, axes: tuple = None) -> FunctionCall:
    """Transpose array."""
    return FunctionCall(lambda a: jnp.transpose(a, axes=axes), [x])

def trace(x) -> FunctionCall:
    """
    Trace of a matrix/tensor over the last two axes.

    Examples
    --------
    trace(A)              -> scalar trace for (..., n, n)
    trace(symgrad(u))     -> volumetric strain
    """
    return FunctionCall(
        lambda a: jnp.trace(a, axis1=-2, axis2=-1),
        [x],
        name="trace",
    )


def sym(x) -> FunctionCall:
    """
    Symmetric part of a second-order tensor over the last two axes.

    sym(A) = 0.5 * (A + A^T)
    """
    return FunctionCall(
        lambda a: 0.5 * (a + jnp.swapaxes(a, -1, -2)),
        [x],
        name="sym",
    )


def antisym(x) -> FunctionCall:
    """
    Skew-symmetric part of a second-order tensor over the last two axes.

    antisym(A) = 0.5 * (A - A^T)
    """
    return FunctionCall(
        lambda a: 0.5 * (a - jnp.swapaxes(a, -1, -2)),
        [x],
        name="antisym",
    )


def identity(n: int) -> FunctionCall:
    """
    Symbolic identity matrix helper.

    This returns a traced constant-like FunctionCall so it composes naturally
    inside symbolic expressions.

    Example
    -------
    I = jnn.identity(2)
    sigma = lam * jnn.trace(eps) * I + 2.0 * mu * eps
    """
    return FunctionCall(
        lambda: jnp.eye(n),
        [],
        name="identity",
    )


def symgrad(target: Placeholder, variables: List[Variable], scheme: str = "automatic_differentiation") -> FunctionCall:
    """
    Symmetric gradient of a vector/tensor-valued field.

    For a vector field u in R^dim:
        grad(u)    -> (..., n_comp, dim)
        symgrad(u) -> 0.5 * (grad(u) + grad(u)^T)

    In small-strain elasticity:
        eps(u) = symgrad(u, [x, y])   # 2D
        eps(u) = symgrad(u, [x, y, z])# 3D

    Notes
    -----
    This assumes the Jacobian convention used by jno:
        jacobian(u, [x, y]) has trailing shape (..., value_shape, dim)
    so the last axis is the derivative direction and the second-last block
    corresponds to field components.
    """
    G = jacobian(target, variables, scheme=scheme)
    return FunctionCall(
        lambda a: 0.5 * (a + jnp.swapaxes(a, -1, -2)),
        [G],
        name="symgrad",
    )
# ============================================================================
# Reduction operations
# ============================================================================


def _reduction(jnp_fn, name):
    """Create a reduction wrapper for Placeholder args."""

    def wrapper(x, axis=None, keepdims=False):
        return FunctionCall(
            lambda a: jnp_fn(a, axis=axis, keepdims=keepdims),
            [x],
            name=name,
            reduces_axis=axis,
        )

    wrapper.__name__ = name
    wrapper.__doc__ = jnp_fn.__doc__
    return wrapper


sum = _reduction(jnp.sum, "sum")
mean = _reduction(jnp.mean, "mean")
median = _reduction(jnp.median, "median")
std = _reduction(jnp.std, "std")
var = _reduction(jnp.var, "var")
min = _reduction(jnp.min, "min")
max = _reduction(jnp.max, "max")
prod = _reduction(jnp.prod, "prod")


def norm(x, ord=None, axis=None, keepdims=False) -> FunctionCall:
    """Vector/matrix norm."""
    return FunctionCall(
        lambda a: jnp.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims),
        [x],
        name="norm",
        reduces_axis=axis,
    )


# ============================================================================
# Comparison operations
# ============================================================================


maximum = _binary(jnp.maximum)
minimum = _binary(jnp.minimum)


def where(condition, x, y) -> FunctionCall:
    """Return elements chosen from x or y depending on condition."""
    return FunctionCall(jnp.where, [condition, x, y])


# ============================================================================
# Linear algebra
# ============================================================================


dot = _binary(jnp.dot)
matmul = _binary(jnp.matmul)
cross = _binary(jnp.cross)


# ============================================================================
# Differential operators (pino-specific)
# ============================================================================
def inner(x, y, n_contract: int = 1, keepdims: bool = False) -> FunctionCall:
    """
    Generalized inner product / contraction over the last ``n_contract`` axes.

    This is intentionally shape-friendly for weak forms. It pads the lower-rank
    operand with singleton axes *before* the contracted trailing axes so common
    patterns like

        inner(grad_u, grad_phi)

    work both in pointwise mode and FEM mode, where ``grad_phi`` usually carries
    an extra local basis-function axis.

    Examples:
        inner(a, b)               -> vector inner product over last axis
        inner(A, B, n_contract=2) -> Frobenius product
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
        return jnp.sum(a * b, axis=axes, keepdims=_keep)

    return FunctionCall(_fn, [x, y], name="inner", reduces_axis=-1)

def double_dot(x, y) -> FunctionCall:
    """
    Double contraction / Frobenius product.

    Equivalent to:
        inner(x, y, n_contract=2)
    """
    return inner(x, y, n_contract=2)

def einsum(subscripts: str, *operands) -> FunctionCall:
    """Traced jnp.einsum wrapper for compact tensor/vector contractions."""
    return FunctionCall(
        lambda *args, _subs=subscripts: jnp.einsum(_subs, *args),
        list(operands),
        name="einsum",
    )


def div(vector_field: List[Placeholder], variables: List[Variable]) -> Placeholder:
    """Alias for divergence."""
    return divergence(vector_field, variables)

def grad(target: Placeholder, variable: Variable, scheme: str = "automatic_differentiation") -> Jacobian:
    """
    Compute the gradient of target with respect to variable.

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

    Example:
        u_x = pnp.grad(u(x, y), x)  # ∂u/∂x
    """
    if isinstance(variable, (list, tuple)):
        if len(variable) == 0:
            raise ValueError("grad(..., variables) requires at least one variable")
        return Jacobian(target, list(variable), scheme)
    return Jacobian(target, [variable], scheme)


def laplacian(target: Placeholder, variables: List[Variable] = None, scheme: str = "automatic_differentiation") -> Hessian:
    """
    Compute the Laplacian of target with respect to variables.

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

def test(name: str = "phi") -> TestFunction:
    """Create a generic variational test function symbol."""
    return TestFunction(name=name)

def trial(name: str = "u") -> TrialFunction:
    """Create a generic variational unknown symbol."""
    return TrialFunction(name=name)

# ============================================================================
# Array creation and dtypes — plain re-exports from jax.numpy
# ============================================================================
from jax.numpy import (
    zeros,
    ones,
    full,
    eye,
    arange,
    linspace,
    meshgrid,
    array,
    asarray,
    float32,
    float64,
    int32,
    int64,
    bool_,
    complex64,
    complex128,
)


def _create_linalg_wrapper():
    """Factory function to create the linalg wrapper class."""

    class _linalg:
        """Wrapper for jax.numpy.linalg that always returns FunctionCall nodes."""

        pass

    for name in dir(jnp.linalg):
        if name.startswith("_"):
            continue

        original = getattr(jnp.linalg, name)

        if callable(original):

            def make_method(func, func_name):
                def method(*args, **kwargs):
                    return FunctionCall(
                        func,
                        list(args),
                        name=func_name,
                        kwargs=kwargs if kwargs else None,
                    )

                method.__doc__ = func.__doc__
                method.__name__ = func_name
                return staticmethod(method)

            setattr(_linalg, name, make_method(original, name))

    return _linalg


linalg = _create_linalg_wrapper()
