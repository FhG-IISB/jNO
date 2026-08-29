from pathlib import Path
from typing import List, Union

import jax
import jax.numpy as jnp

from .architectures.models import nn, parameter  # noqa: F401

# Keep import so people can use jno.numpy as jno -> jno.model, jno.tune
from .integration_operators import IntegrationOperators  # noqa: F401
from .trace import (
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


def choice(options, name: str | None = None, default: int = 0) -> Choice:
    """Create a tunable categorical choice over traced expressions.

    Example:
        u = jnn.choice([net(inp) * x * (1 - x), net(inp)], name="bc_form")
    """
    return Choice(options=options, name=name, default=default)


# ============================================================================
# Factory for simple unary wrappers
# ============================================================================


def _unary(jnp_fn):
    """Create a unary wrapper for Placeholder args (auto-unwraps typed views)."""

    def wrapper(x):
        return _attach_coords(FunctionCall(jnp_fn, [_u(x)]), [x])

    wrapper.__name__ = jnp_fn.__name__
    wrapper.__doc__ = jnp_fn.__doc__
    return wrapper


def _binary(jnp_fn):
    """Create a binary wrapper for Placeholder args (auto-unwraps typed views)."""

    def wrapper(x, y):
        return _attach_coords(FunctionCall(jnp_fn, [_u(x), _u(y)]), [x, y])

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
            eye_mat = jnp.eye(A.shape[0])
            return jnp.linalg.solve(eye_mat - a * A, b)

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
    """Concatenate placeholders along an axis (always axis=-1 at eval time)."""
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

        # Fallback: align ranks and only broadcast singleton dimensions.
        max_ndim = max(a.ndim for a in expanded)
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


def concatenate(items, axis: int = -1) -> FunctionCall:
    """Alias for concat."""
    return concat(items, axis=axis)


def vector(*components):
    """Build a :class:`VectorView` directly from scalar component Placeholders.

    Equivalent to ``VectorView(concat([*components]))`` but skips the manual
    list and the ``.vector`` re-wrap step::

        n    = jno.np.vector(nx, ny)          # 2-D normal as VectorView
        flux = jno.np.vector(jx, jy, jz)      # 3-D current density
        prob = jno.np.vector(x * p, y * p)    # OU drift flux
    """
    from .trace.views import VectorView

    return VectorView(concat(list(components)))


def stack(items, axis: int = 0) -> FunctionCall:
    """Stack placeholders along a new axis."""
    if axis == -1:
        return concat(items, axis=-1)
    return FunctionCall(lambda *args: jnp.stack(args, axis=axis), [_u(i) for i in items], name="stack")


def reshape(x, shape: tuple) -> FunctionCall:
    """Reshape a placeholder to a new shape."""
    return FunctionCall(lambda a: jnp.reshape(a, shape), [_u(x)])


def squeeze(x, axis: int = None) -> FunctionCall:
    """Remove single-dimensional entries."""
    return FunctionCall(lambda a: jnp.squeeze(a, axis=axis), [_u(x)])


def expand_dims(x, axis: int) -> FunctionCall:
    """Expand array dimensions."""
    return FunctionCall(lambda a: jnp.expand_dims(a, axis=axis), [_u(x)])


def transpose(x, axes: tuple = None) -> FunctionCall:
    """Transpose array."""
    return FunctionCall(lambda a: jnp.transpose(a, axes=axes), [_u(x)])


def trace(x) -> FunctionCall:
    """
    Trace of a matrix/tensor over the last two axes.

    Examples
    --------
    trace(A)              -> scalar trace for (..., n, n)
    trace(symgrad(u))     -> volumetric strain
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
    """
    Symmetric part of a second-order tensor over the last two axes.

    sym(A) = 0.5 * (A + A^T)
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
    """
    Skew-symmetric part of a second-order tensor over the last two axes.

    antisym(A) = 0.5 * (A - A^T)
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
    """
    Symbolic identity matrix helper — an ``(n, n)`` identity carried with a **leading singleton axis**,
    ``(1, n, n)``, so it broadcasts as a *constant tensor* over the batch axis a kernel prepends (the
    quadrature axis in an FE weak form; the points axis in a collocation residual). A bare ``(n, n)``
    would be mis-read by the elementwise-broadcast alignment as if its first axis were the quadrature
    axis; the leading singleton makes ``I`` compose correctly inside any tensor formula.

    Example
    -------
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

    Example
    -------
    Fe = jnn.matmul(F, jnn.inv(Fp))       # elastic part of F = Fe · Fp
    """
    return _attach_coords(FunctionCall(jnp.linalg.inv, [_u(x)], name="inv"), [x])


def det(x) -> FunctionCall:
    """Determinant over the last two axes → scalar (per leading index).

    Example
    -------
    J = jnn.det(F)                        # volumetric part of the deformation
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


def eigvalsh(x) -> FunctionCall:
    """Ascending eigenvalues of a **symmetric** tensor over the last two axes →
    vector of principal values (per leading index).

    Uses the symmetric solver ``jnp.linalg.eigh``. Useful for principal
    stretches / principal stresses and isotropic yield functions. (Eigenvalues
    only — eigenvectors are not returned, as they are ill-defined at repeated
    eigenvalues; for a spectral matrix function use :func:`logm`/:func:`expm`/
    :func:`sqrtm`, which stay differentiable there.)
    """
    return _attach_coords(FunctionCall(lambda a: jnp.linalg.eigh(a)[0], [_u(x)], name="eigvalsh"), [x])


def logm(x) -> FunctionCall:
    """Symmetric matrix logarithm ``logm(A) = V diag(log λ) Vᵀ`` over the last two
    axes → tensor (SPD input).

    The gradient stays finite at **repeated eigenvalues** (Daleckiĭ–Kreĭn form),
    so a log-strain return map is differentiable through equal principal
    stretches. Distinct from the element-wise scalar :func:`log`.

    Example
    -------
    E = 0.5 * jnn.logm(jnn.matmul(F.T, F))   # Hencky (logarithmic) strain
    """
    from .trace.views import _spectral_matrix_function

    return _attach_coords(FunctionCall(lambda a: _spectral_matrix_function(a, "log", 0.0), [_u(x)], name="logm"), [x])


def expm(x) -> FunctionCall:
    """Symmetric matrix exponential ``expm(A) = V diag(exp λ) Vᵀ`` over the last
    two axes → tensor.

    Gradient stable at repeated eigenvalues; the inverse of :func:`logm` on SPD
    inputs. Distinct from the element-wise scalar :func:`exp`.

    Example
    -------
    Fp = jnn.matmul(jnn.expm(dgamma * N), Fp_prev)   # exponential map of plastic flow
    """
    from .trace.views import _spectral_matrix_function

    return _attach_coords(FunctionCall(lambda a: _spectral_matrix_function(a, "exp", 0.0), [_u(x)], name="expm"), [x])


def sqrtm(x) -> FunctionCall:
    """Symmetric matrix square root ``A^(1/2) = V diag(√λ) Vᵀ`` over the last two
    axes → tensor (SPD input).

    Gradient stable at repeated eigenvalues. E.g. the right stretch
    ``U = sqrtm(FᵀF)`` of a polar decomposition. Distinct from the element-wise
    scalar :func:`sqrt`.
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


def _reduction(jnp_fn, name):
    """Create a reduction wrapper for Placeholder args (auto-unwraps typed views)."""

    def wrapper(x, axis=None, keepdims=False):
        # `axis` may be a coordinate Variable (or a tuple containing them). It is resolved to an
        # integer HERE, inside the closure, because that is the first point where the array — and
        # therefore its shape — actually exists; a trace expression carries no shape. `reduces_axis`
        # keeps the UNRESOLVED value: it is what the user wrote.
        def _fn(a, _axis=axis, _keepdims=keepdims):
            return jnp_fn(a, axis=_resolve_axes(a.shape, _axis), keepdims=_keepdims)

        return FunctionCall(_fn, [_u(x)], name=name, reduces_axis=axis)

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
    """Vector/matrix norm. ``axis`` also accepts a coordinate ``Variable`` (see :func:`mean`)."""

    def _fn(a, _axis=axis):
        return jnp.linalg.norm(a, ord=ord, axis=_resolve_axes(a.shape, _axis), keepdims=keepdims)

    return FunctionCall(_fn, [_u(x)], name="norm", reduces_axis=axis)


# ============================================================================
# Comparison operations
# ============================================================================


maximum = _binary(jnp.maximum)
minimum = _binary(jnp.minimum)


def where(condition, x, y) -> FunctionCall:
    """Return elements chosen from x or y depending on condition."""
    return FunctionCall(jnp.where, [_u(condition), _u(x), _u(y)])


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

    return _attach_coords(FunctionCall(_fn, [_u(x), _u(y)], name="inner", reduces_axis=-1), [x, y])


def double_dot(x, y) -> FunctionCall:
    """
    Double contraction / Frobenius product.

    Equivalent to:
        inner(x, y, n_contract=2)
    """
    return inner(x, y, n_contract=2)


def einsum(subscripts: str, *operands) -> FunctionCall:
    """Traced jnp.einsum wrapper for compact tensor/vector contractions."""
    return _attach_coords(
        FunctionCall(
            lambda *args, _subs=subscripts: jnp.einsum(_subs, *args),
            [_u(o) for o in operands],
            name="einsum",
        ),
        list(operands),
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
    _guard(target, scheme)
    if scheme == "finite_difference" and variables is not None:
        print(
            "Variables were selected for the finite difference laplacian which are not used. The finite difference derivatives are computed on the entire spatial grid."
        )

    return Hessian(_u(target), variables, scheme, trace=True)


def laplace(
    target: Placeholder,
    variables: List[Variable],
    scheme: str = "automatic_differentiation",
) -> Hessian:
    """Alias for laplacian."""
    _guard(target, scheme)
    return Hessian(_u(target), variables, scheme, trace=True)


def hessian(
    target: Placeholder,
    variables: List[Variable],
    scheme: str = "automatic_differentiation",
) -> Hessian:
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
    _guard(target, scheme)
    return Hessian(_u(target), variables, scheme)


def jacobian(
    target: Placeholder,
    variables: List[Variable],
    scheme: str = "automatic_differentiation",
) -> Jacobian:
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
    _guard(target, scheme)
    return Jacobian(_u(target), variables, scheme)


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
    """Create a generic variational test function symbol."""
    return TestFunction(name=name)


def trial(name: str = "u") -> TrialFunction:
    """Create a generic variational unknown symbol."""
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
