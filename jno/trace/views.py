"""Semantic views on :class:`Placeholder` — scalar, vector, complex, matrix, voigt.

These thin Python adapters give traced expressions a typed surface so users
can write ``u.vector.div(x, y)`` instead of ``u[..., 0].d(x) + u[..., 1].d(y)``,
or ``sigma.voigt.von_mises()`` instead of the explicit Voigt formula.

A view is **not** a :class:`Placeholder` — it just wraps one and returns new
:class:`Placeholder` nodes (wrapped in the appropriate view) from its methods.
Cross-type arithmetic (``MatrixView @ VectorView`` → ``VectorView``,
``ScalarView * MatrixView`` → ``MatrixView``, ``v.outer(w)`` → ``MatrixView``,
…) is handled inside the dunder methods.

For operations that don't have a dedicated method (``.mse``, ``.mean``,
``.d(x)``, etc.), ``__getattr__`` falls through to the underlying Placeholder.
``.expr`` exposes the raw Placeholder for ``jno.np`` interop.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from ..jnp_ops import concat
from . import FunctionCall, Placeholder

_VIEW_TYPES: tuple = ()  # filled at end of module
_NAMED_PARTIALS_CLS_FOR: dict = {}  # filled at end of module: type(view) → Named<View>WithPartials
_MAX_PARTIAL_ORDER = 4


def _unwrap(other):
    """Strip ``._expr`` from any view, pass through everything else."""
    if isinstance(other, _VIEW_TYPES):
        return other._expr
    return other


def _parse_partial_sequence(key: str, coord_vars: dict) -> list | None:
    """Parse ``key`` as a sequence of registered coord names (≤ 4 names).

    Two parsing regimes, chosen by inspecting the registered names:
      * **all single-char names** → concatenated form (``xy``, ``xxx``, ``txyt``).
      * **any multi-char name**  → underscore-separated form (``r_theta``).

    Returns the ordered list of names if parseable as 1..MAX_PARTIAL_ORDER
    names, else ``None`` (so ``__getattr__`` can fall through).
    """
    names = set(coord_vars)
    if not names:
        return None
    if all(len(n) == 1 for n in names):
        seq = list(key)
    else:
        seq = key.split("_")
    if 1 <= len(seq) <= _MAX_PARTIAL_ORDER and all(s in names for s in seq):
        return seq
    return None


def _coords_dispatch(view_self, args: tuple, named_vars: dict, *, positional_factory=None):
    """Shared dispatch logic for ``ScalarView.coords``, ``VectorView.coords``, etc.

    * ``coords(**vars)`` (kwargs) → partial-derivative wrapper (all views).
    * ``coords(*strings)`` / ``coords([strings])`` (positional) → component /
      element-access wrapper. Only valid where ``positional_factory`` is
      provided (currently ``MatrixView`` and ``VectorView``).
    * Mixing kwargs with positional strings raises ``TypeError``.
    """
    if named_vars and args:
        raise TypeError("coords() expects either kwargs (name=Variable) or positional names, not both")
    if named_vars:
        cls = _NAMED_PARTIALS_CLS_FOR[type(view_self)]
        return cls(view_self._expr, named_vars)
    if args:
        if positional_factory is None:
            raise TypeError(
                f"{type(view_self).__name__}.coords() expects kwargs (name=Variable); "
                "positional string names are only supported on VectorView and MatrixView"
            )
        # Allow both coords(["x", "y"]) and coords("x", "y")
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        # Backward-compat: if non-string objects are passed (e.g. Variables),
        # fall back to their ``tag`` attribute or ``str(...)`` representation.
        resolved = tuple(n if isinstance(n, str) else getattr(n, "tag", str(n)) for n in args)
        return positional_factory(view_self._expr, resolved)
    raise TypeError("coords() requires at least one argument (kwargs or positional names)")


# ---------------------------------------------------------------------------
# ScalarView
# ---------------------------------------------------------------------------


class ScalarView:
    """Semantic view of a scalar Placeholder.

    Exposes named scalar operations (``.abs()``, ``.exp()``, ...) and acts as
    the cross-type "multiplier" — ``scalar * matrix`` returns a ``MatrixView``,
    ``scalar * vector`` returns a ``VectorView``, etc.

    Unknown attributes fall through to the underlying Placeholder so things
    like ``.mse``, ``.mean``, ``.d(x)`` work transparently.
    """

    def __init__(self, expr: Placeholder) -> None:
        self._expr = expr

    @property
    def expr(self) -> Placeholder:
        """The underlying Placeholder — pass this to ``jno.np`` functions."""
        return self._expr

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_expr"), name)

    def integrate(self, **kwargs) -> "ScalarView":
        """Integrate the underlying scalar, preserving the ScalarView type."""
        return ScalarView(self._expr.integrate(**kwargs))

    def coords(self, *args, **named_vars):
        """Bind coordinate Variables for derivative-by-name access.

        ``u.scalar.coords(x=x_var, y=y_var).x`` → ScalarView of ``∂u/∂x``;
        ``.xy`` is the mixed second derivative; up to 4th order.
        Convenience methods that auto-fill registered coord vars are also
        available — ``.grad()`` returns the full gradient, ``.laplacian()``
        returns Δu.
        """
        return _coords_dispatch(self, args, named_vars)

    # -- scalar operations --
    def abs(self) -> "ScalarView":
        return ScalarView(FunctionCall(jnp.abs, [self._expr], "abs"))

    def sign(self) -> "ScalarView":
        return ScalarView(FunctionCall(jnp.sign, [self._expr], "sign"))

    def log(self) -> "ScalarView":
        return ScalarView(FunctionCall(jnp.log, [self._expr], "log"))

    def exp(self) -> "ScalarView":
        return ScalarView(FunctionCall(jnp.exp, [self._expr], "exp"))

    def sqrt(self) -> "ScalarView":
        return ScalarView(FunctionCall(jnp.sqrt, [self._expr], "sqrt"))

    def pow(self, n: float) -> "ScalarView":
        return ScalarView(self._expr**n)

    # -- arithmetic with cross-type dispatch --
    def __add__(self, other):
        return ScalarView(self._expr + _unwrap(other))

    def __radd__(self, other):
        return ScalarView(_unwrap(other) + self._expr)

    def __sub__(self, other):
        return ScalarView(self._expr - _unwrap(other))

    def __rsub__(self, other):
        return ScalarView(_unwrap(other) - self._expr)

    def __neg__(self):
        return ScalarView(-self._expr)

    def __truediv__(self, other):
        return ScalarView(self._expr / _unwrap(other))

    def __rtruediv__(self, other):
        return ScalarView(_unwrap(other) / self._expr)

    def __mul__(self, other):
        if isinstance(other, VectorView):
            return VectorView(self._expr * other._expr)
        if isinstance(other, MatrixView):
            return MatrixView(self._expr * other._expr)
        if isinstance(other, VoigtView):
            return VoigtView(self._expr * other._expr)
        if isinstance(other, ComplexView):
            return ComplexView(self._expr * other._expr)
        return ScalarView(self._expr * _unwrap(other))

    def __rmul__(self, other):
        return ScalarView(_unwrap(other) * self._expr)


# ---------------------------------------------------------------------------
# VectorView
# ---------------------------------------------------------------------------


class VectorView:
    """Semantic view of a Placeholder as a spatial vector field ``[..., n]``."""

    def __init__(self, expr: Placeholder) -> None:
        self._expr = expr

    @property
    def expr(self) -> Placeholder:
        return self._expr

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_expr"), name)

    def integrate(self, **kwargs) -> "VectorView":
        """Component-wise integral, preserving VectorView type."""
        return VectorView(self._expr.integrate(**kwargs))

    def coords(self, *args, **named_vars):
        """Two forms.

        * ``v.coords(x=x_var, y=y_var)`` (kwargs) → partial-derivative wrapper.
          ``v.x`` returns a ``VectorView`` of ``∂v/∂x`` (component-wise partial).
        * ``v.coords("x", "y")`` or ``v.coords(["x", "y"])`` (positional strings)
          → component-access wrapper. ``v.x`` returns ``ScalarView`` of v[..., 0].
        """
        return _coords_dispatch(self, args, named_vars, positional_factory=NamedVectorView)

    # -- component access --
    def _c(self, i: int) -> "ScalarView":
        return ScalarView(self._expr[..., i])

    def component(self, i: int) -> "ScalarView":
        """i-th component → ScalarView."""
        return self._c(i)

    # -- differential operators --
    def div(self, *vars) -> "ScalarView":
        """Divergence ∑ ∂u_i/∂x_i → ScalarView.

        Number of variables must equal the last dimension of the vector.
        """
        terms = [self._c(i).expr.d(v) for i, v in enumerate(vars)]
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return ScalarView(total)

    def curl(self, *vars):
        """Curl of the vector field.

        2 variables (2-D) → ``ScalarView`` (∂u_y/∂x − ∂u_x/∂y).
        3 variables (3-D) → ``VectorView`` (the curl vector).
        """
        if len(vars) == 2:
            x, y = vars
            return ScalarView(self._c(1).expr.d(x) - self._c(0).expr.d(y))
        if len(vars) == 3:
            x, y, z = vars
            cx = self._c(2).expr.d(y) - self._c(1).expr.d(z)
            cy = self._c(0).expr.d(z) - self._c(2).expr.d(x)
            cz = self._c(1).expr.d(x) - self._c(0).expr.d(y)
            return VectorView(concat([cx, cy, cz]))
        raise ValueError("curl requires 2 or 3 spatial variables")

    # -- reductions / pairwise --
    def norm(self) -> "ScalarView":
        """L2 norm ``sqrt(∑ u_i²)`` → ScalarView."""
        return ScalarView(FunctionCall(lambda x: jnp.linalg.norm(x, axis=-1), [self._expr], "norm", True))

    def dot(self, other) -> "ScalarView":
        """Dot product u·v → ScalarView."""
        product = self._expr * _unwrap(other)
        return ScalarView(FunctionCall(lambda x: jnp.sum(x, axis=-1), [product], "dot", True))

    def cross(self, other) -> "VectorView":
        """Cross product u×v (3-D only) → VectorView."""
        return VectorView(FunctionCall(lambda a, b: jnp.cross(a, b), [self._expr, _unwrap(other)], "cross"))

    def normalize(self) -> "VectorView":
        """Unit vector ``u / (||u|| + ε)`` → VectorView."""
        return VectorView(
            FunctionCall(
                lambda x: x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8),
                [self._expr],
                "normalize",
            )
        )

    def outer(self, other) -> "MatrixView":
        """Outer product v ⊗ w → ``[..., n, m]`` MatrixView."""
        return MatrixView(
            FunctionCall(
                lambda a, b: a[..., :, jnp.newaxis] * b[..., jnp.newaxis, :],
                [self._expr, _unwrap(other)],
                "outer",
            )
        )

    def jacobian(self, *vars) -> "MatrixView":
        """Full Jacobian ``∂u_i/∂x_j`` of this vector field → MatrixView.

        Stacks ``self._expr.d(v)`` for each variable along a new last axis,
        so for an ``[..., n]`` vector and ``m`` variables the result is
        ``[..., n, m]`` with ``J[..., i, j] = ∂u_i/∂x_j``.
        """
        cols = [self._expr.d(v) for v in vars]
        return MatrixView(
            FunctionCall(
                lambda *cs: jnp.stack(cs, axis=-1),
                cols,
                "jacobian",
            )
        )

    def grad(self, *vars) -> "MatrixView":
        """Spatial gradient of a vector field = Jacobian → MatrixView.

        Alias for :meth:`jacobian` so that ``vec.grad(x, y)`` mirrors the
        scalar ``u.grad(x, y) → VectorView`` pattern in dimension.
        """
        return self.jacobian(*vars)

    # -- arithmetic --
    def __add__(self, other):
        return VectorView(self._expr + _unwrap(other))

    def __radd__(self, other):
        return VectorView(_unwrap(other) + self._expr)

    def __sub__(self, other):
        return VectorView(self._expr - _unwrap(other))

    def __rsub__(self, other):
        return VectorView(_unwrap(other) - self._expr)

    def __neg__(self):
        return VectorView(-self._expr)

    def __mul__(self, other):
        return VectorView(self._expr * _unwrap(other))

    def __rmul__(self, other):
        return VectorView(_unwrap(other) * self._expr)

    def __truediv__(self, other):
        return VectorView(self._expr / _unwrap(other))

    def __rtruediv__(self, other):
        return VectorView(_unwrap(other) / self._expr)

    def __matmul__(self, other):
        """``v @ A`` (row-vec times matrix) → VectorView; ``v @ w`` → Placeholder dot."""
        if isinstance(other, MatrixView):
            return VectorView(
                FunctionCall(
                    lambda v, A: (v[..., jnp.newaxis, :] @ A)[..., 0, :],
                    [self._expr, other._expr],
                    "vecmat",
                )
            )
        return FunctionCall(lambda a, b: a @ b, [self._expr, _unwrap(other)], "dot")


# ---------------------------------------------------------------------------
# ComplexView
# ---------------------------------------------------------------------------


class ComplexView:
    """Semantic view of a Placeholder as a complex field, last dim = 2 = [re, im].

    For native JAX ``complex64`` Placeholders, use ``placeholder.real`` /
    ``.imag`` directly (added to :class:`Placeholder`).
    """

    def __init__(self, expr: Placeholder) -> None:
        self._expr = expr

    @property
    def expr(self) -> Placeholder:
        return self._expr

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_expr"), name)

    def integrate(self, **kwargs) -> "ComplexView":
        """Component-wise integral of the [re, im] split, preserving ComplexView."""
        return ComplexView(self._expr.integrate(**kwargs))

    def coords(self, *args, **named_vars):
        """Bind coordinate Variables for partial-derivative-by-name access.

        ``ψ.complex.coords(x=x_var, y=y_var).x`` → ComplexView of ``∂ψ/∂x`` —
        the partial is taken element-wise across the ``[re, im]`` split.
        """
        return _coords_dispatch(self, args, named_vars)

    @property
    def real(self) -> "ScalarView":
        """Real part ``expr[..., 0]`` → ScalarView."""
        return ScalarView(self._expr[..., 0])

    @property
    def imag(self) -> "ScalarView":
        """Imaginary part ``expr[..., 1]`` → ScalarView."""
        return ScalarView(self._expr[..., 1])

    @property
    def abs(self) -> "ScalarView":
        """Modulus ``sqrt(re² + im²)`` → ScalarView."""
        return ScalarView(FunctionCall(lambda x: jnp.linalg.norm(x, axis=-1), [self._expr], "cabs", True))

    @property
    def angle(self) -> "ScalarView":
        """Phase angle ``arctan2(im, re)`` → ScalarView."""
        return ScalarView(
            FunctionCall(
                lambda x: jnp.arctan2(x[..., 1], x[..., 0]),
                [self._expr],
                "angle",
                True,
            )
        )

    @property
    def conj(self) -> "ComplexView":
        """Complex conjugate ``[re, -im]`` → ComplexView."""
        return ComplexView(concat([self._expr[..., 0], -self._expr[..., 1]]))

    def to_native(self) -> Placeholder:
        """Convert split ``[re, im]`` → native JAX complex64 Placeholder."""
        return FunctionCall(lambda x: x[..., 0] + 1j * x[..., 1], [self._expr], "to_native")

    def mul(self, other) -> "ComplexView":
        """Complex multiplication ``(a+bi)(c+di) = (ac-bd) + (ad+bc)i``."""
        o = other if isinstance(other, ComplexView) else ComplexView(_unwrap(other))
        re = self.real.expr * o.real.expr - self.imag.expr * o.imag.expr
        im = self.real.expr * o.imag.expr + self.imag.expr * o.real.expr
        return ComplexView(concat([re, im]))

    # -- elementwise (scalar) arithmetic; for complex product use .mul --
    def __add__(self, other):
        return ComplexView(self._expr + _unwrap(other))

    def __radd__(self, other):
        return ComplexView(_unwrap(other) + self._expr)

    def __sub__(self, other):
        return ComplexView(self._expr - _unwrap(other))

    def __rsub__(self, other):
        return ComplexView(_unwrap(other) - self._expr)

    def __neg__(self):
        return ComplexView(-self._expr)

    def __mul__(self, other):
        return ComplexView(self._expr * _unwrap(other))

    def __rmul__(self, other):
        return ComplexView(_unwrap(other) * self._expr)

    def __truediv__(self, other):
        return ComplexView(self._expr / _unwrap(other))

    def __rtruediv__(self, other):
        return ComplexView(_unwrap(other) / self._expr)


# ---------------------------------------------------------------------------
# MatrixView
# ---------------------------------------------------------------------------


class MatrixView:
    """Semantic view of a Placeholder as a full matrix field ``[..., n, m]``.

    For symmetric tensors in Voigt packing (``[..., 3]`` 2-D, ``[..., 6]`` 3-D)
    use :class:`VoigtView` instead. The packed-format constructors
    (``from_upper_tri``, ``from_lower_tri``, ``from_flat``, ``from_diag``)
    accept compactly-stored data and return a ``MatrixView`` so all other
    methods can chain off the unpacked matrix.
    """

    def __init__(self, expr: Placeholder) -> None:
        self._expr = expr

    @property
    def expr(self) -> Placeholder:
        return self._expr

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_expr"), name)

    def integrate(self, **kwargs) -> "MatrixView":
        """Element-wise integral, preserving MatrixView type."""
        return MatrixView(self._expr.integrate(**kwargs))

    # ------------------------------------------------------------------
    # Basic matrix operations
    # ------------------------------------------------------------------

    def trace(self) -> "ScalarView":
        """Sum of diagonal elements → ScalarView."""
        return ScalarView(FunctionCall(lambda x: jnp.trace(x, axis1=-2, axis2=-1), [self._expr], "trace", True))

    def sym(self) -> "MatrixView":
        """Symmetric part ``(A + Aᵀ)/2`` → MatrixView."""
        return MatrixView(FunctionCall(lambda x: (x + x.swapaxes(-2, -1)) * 0.5, [self._expr], "sym"))

    def skew(self) -> "MatrixView":
        """Skew-symmetric part ``(A − Aᵀ)/2`` → MatrixView."""
        return MatrixView(FunctionCall(lambda x: (x - x.swapaxes(-2, -1)) * 0.5, [self._expr], "skew"))

    def det(self) -> "ScalarView":
        """Determinant → ScalarView."""
        return ScalarView(FunctionCall(jnp.linalg.det, [self._expr], "det", True))

    def diag(self) -> "VectorView":
        """Diagonal elements → VectorView, shape ``[..., n]``."""
        return VectorView(FunctionCall(lambda x: jnp.diagonal(x, axis1=-2, axis2=-1), [self._expr], "diag"))

    def inv(self) -> "MatrixView":
        """Matrix inverse → MatrixView."""
        return MatrixView(FunctionCall(jnp.linalg.inv, [self._expr], "inv"))

    def eigvals(self) -> "VectorView":
        """Eigenvalues via ``jnp.linalg.eigh`` (symmetric input, ascending) → VectorView."""
        return VectorView(FunctionCall(lambda x: jnp.linalg.eigh(x)[0], [self._expr], "eigvals"))

    def norm(self, ord: str = "fro") -> "ScalarView":
        """Matrix norm (Frobenius by default) → ScalarView."""
        return ScalarView(
            FunctionCall(
                lambda x, _o=ord: jnp.linalg.norm(x, ord=_o, axis=(-2, -1)),
                [self._expr],
                "mat_norm",
                True,
            )
        )

    def transpose(self) -> "MatrixView":
        """Swap last two dimensions → MatrixView."""
        return MatrixView(FunctionCall(lambda x: x.swapaxes(-2, -1), [self._expr], "transpose"))

    def log(self) -> "MatrixView":
        """Matrix logarithm via eigendecomposition (symmetric / SPD matrices only)."""

        def _logm(x):
            vals, vecs = jnp.linalg.eigh(x)
            return (vecs * jnp.log(vals)[..., jnp.newaxis, :]) @ vecs.swapaxes(-2, -1)

        return MatrixView(FunctionCall(_logm, [self._expr], "mat_log"))

    def exp(self) -> "MatrixView":
        """Matrix exponential via eigendecomposition (symmetric matrices only)."""

        def _expm(x):
            vals, vecs = jnp.linalg.eigh(x)
            return (vecs * jnp.exp(vals)[..., jnp.newaxis, :]) @ vecs.swapaxes(-2, -1)

        return MatrixView(FunctionCall(_expm, [self._expr], "mat_exp"))

    def pow(self, n: float) -> "MatrixView":
        """Matrix power ``Aⁿ`` via eigendecomposition (symmetric matrices only)."""

        def _powm(x, _n=n):
            vals, vecs = jnp.linalg.eigh(x)
            return (vecs * (vals**_n)[..., jnp.newaxis, :]) @ vecs.swapaxes(-2, -1)

        return MatrixView(FunctionCall(_powm, [self._expr], f"mat_pow_{n}"))

    # ------------------------------------------------------------------
    # Packed-format constructors and converters
    # ------------------------------------------------------------------

    def from_upper_tri(self) -> "MatrixView":
        """Unpack ``[..., k]`` upper-triangle storage → ``[..., n, n]`` symmetric MatrixView.

        Infers ``n`` from ``k = n(n+1)/2`` (k=3 → n=2, k=6 → n=3, k=10 → n=4, …).
        """

        def _unpack(x):
            k = x.shape[-1]
            n = int(round((-1 + (1 + 8 * k) ** 0.5) / 2))
            i_idx, j_idx = jnp.triu_indices(n)
            A = jnp.zeros(x.shape[:-1] + (n, n), dtype=x.dtype)
            A = A.at[..., i_idx, j_idx].set(x)
            A = A + A.swapaxes(-2, -1)
            return jnp.where(jnp.eye(n, dtype=bool), A * 0.5, A)

        return MatrixView(FunctionCall(_unpack, [self._expr], "from_upper_tri"))

    def from_lower_tri(self) -> "MatrixView":
        """Unpack ``[..., k]`` lower-triangle storage → ``[..., n, n]`` symmetric MatrixView."""

        def _unpack(x):
            k = x.shape[-1]
            n = int(round((-1 + (1 + 8 * k) ** 0.5) / 2))
            i_idx, j_idx = jnp.tril_indices(n)
            A = jnp.zeros(x.shape[:-1] + (n, n), dtype=x.dtype)
            A = A.at[..., i_idx, j_idx].set(x)
            A = A + A.swapaxes(-2, -1)
            return jnp.where(jnp.eye(n, dtype=bool), A * 0.5, A)

        return MatrixView(FunctionCall(_unpack, [self._expr], "from_lower_tri"))

    def from_flat(self, n: int, m: Optional[int] = None) -> "MatrixView":
        """Reshape ``[..., n*m]`` → ``[..., n, m]`` (square if ``m`` omitted)."""
        _m = m if m is not None else n
        return MatrixView(
            FunctionCall(
                lambda x, _n=n, _m=_m: x.reshape(x.shape[:-1] + (_n, _m)),
                [self._expr],
                f"from_flat_{n}x{_m}",
            )
        )

    def from_diag(self) -> "MatrixView":
        """Promote ``[..., n]`` diagonal vector → ``[..., n, n]`` diagonal MatrixView."""

        def _fn(x):
            n = x.shape[-1]
            return jnp.einsum("...i,ij->...ij", x, jnp.eye(n, dtype=x.dtype))

        return MatrixView(FunctionCall(_fn, [self._expr], "from_diag"))

    def to_upper_tri(self) -> Placeholder:
        """Pack ``[..., n, n]`` → ``[..., n(n+1)/2]`` upper-triangle entries (Placeholder)."""

        def _fn(x):
            i_idx, j_idx = jnp.triu_indices(x.shape[-1])
            return x[..., i_idx, j_idx]

        return FunctionCall(_fn, [self._expr], "to_upper_tri")

    def to_lower_tri(self) -> Placeholder:
        """Pack ``[..., n, n]`` → ``[..., n(n+1)/2]`` lower-triangle entries (Placeholder)."""

        def _fn(x):
            i_idx, j_idx = jnp.tril_indices(x.shape[-1])
            return x[..., i_idx, j_idx]

        return FunctionCall(_fn, [self._expr], "to_lower_tri")

    def coords(self, *args, **named_vars):
        """Two forms.

        * ``A.coords(["x", "y"])`` or ``A.coords("x", "y")`` (positional strings)
          → element-access wrapper :class:`NamedMatrixView`; ``A.xy`` returns
          ``ScalarView`` of ``A[..., 0, 1]``.
        * ``A.coords(x=x_var, y=y_var)`` (kwargs) → partial-derivative wrapper;
          ``A.x`` returns ``MatrixView`` of ``∂A/∂x_var`` (element-wise partial).
        """
        return _coords_dispatch(self, args, named_vars, positional_factory=NamedMatrixView)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        return MatrixView(self._expr + _unwrap(other))

    def __radd__(self, other):
        return MatrixView(_unwrap(other) + self._expr)

    def __sub__(self, other):
        return MatrixView(self._expr - _unwrap(other))

    def __rsub__(self, other):
        return MatrixView(_unwrap(other) - self._expr)

    def __mul__(self, other):
        return MatrixView(self._expr * _unwrap(other))

    def __rmul__(self, other):
        return MatrixView(_unwrap(other) * self._expr)

    def __neg__(self):
        return MatrixView(-self._expr)

    def __truediv__(self, other):
        return MatrixView(self._expr / _unwrap(other))

    def __rtruediv__(self, other):
        return MatrixView(_unwrap(other) / self._expr)

    def __pow__(self, n):
        """Elementwise power. For matrix power use ``.pow(n)``."""
        return MatrixView(self._expr**n)

    def __matmul__(self, other):
        """``A @ B`` → MatrixView. ``A @ v`` (VectorView) → VectorView."""
        if isinstance(other, VectorView):
            return VectorView(
                FunctionCall(
                    lambda A, v: (A @ v[..., jnp.newaxis])[..., 0],
                    [self._expr, other._expr],
                    "matvec",
                )
            )
        return MatrixView(FunctionCall(lambda a, b: a @ b, [self._expr, _unwrap(other)], "matmul"))


# ---------------------------------------------------------------------------
# NamedMatrixView
# ---------------------------------------------------------------------------


class NamedMatrixView(MatrixView):
    """MatrixView with coordinate labels for named component access.

    Created by :meth:`MatrixView.coords`. Inherits every MatrixView operation
    (``.trace()``, ``.det()``, …); adds attribute-style lookup of components.

    Examples
    --------
    >>> A = sigma.voigt.to_full().coords(["x", "y"])
    >>> A.xy        # ScalarView for σ_xy
    >>> A.trace()   # ScalarView for σ_xx + σ_yy (inherited)
    >>> A.component("x", "y")   # explicit form, also returns ScalarView
    """

    def __init__(self, expr: Placeholder, coord_names: list) -> None:
        super().__init__(expr)
        # Store on __dict__ via object.__setattr__ to be safe even if
        # __getattr__ is consulted during init for some reason.
        object.__setattr__(self, "_coord_names", list(coord_names))
        object.__setattr__(self, "_coord_map", {n: i for i, n in enumerate(coord_names)})

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        # Avoid recursion: pull internals via object.__getattribute__
        names = object.__getattribute__(self, "_coord_names")
        expr = object.__getattribute__(self, "_expr")
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if key == ni + nj or key == f"{ni}_{nj}":
                    return ScalarView(expr[..., i, j])
        # Fall through to underlying Placeholder attributes (.mse, .d, etc.)
        return getattr(expr, key)

    def component(self, name1: str, name2: str) -> "ScalarView":
        """Explicit component lookup → ScalarView."""
        i = self._coord_map[name1]
        j = self._coord_map[name2]
        return ScalarView(self._expr[..., i, j])


# ---------------------------------------------------------------------------
# VoigtView
# ---------------------------------------------------------------------------


class VoigtView:
    """Semantic view of a Placeholder as a symmetric tensor in Voigt notation.

    Last-dim layout selects the dimension:
      * 3 → 2-D symmetric tensor ``[σ_xx, σ_yy, σ_xy]``
      * 6 → 3-D symmetric tensor ``[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]``

    Arithmetic between two VoigtViews returns a VoigtView (componentwise).
    Use ``.to_full()`` to get a full ``[..., n, n]`` MatrixView for general
    matrix operations (``.inv``, ``.log``, ``.exp``, ``.coords``).
    """

    def __init__(self, expr: Placeholder) -> None:
        self._expr = expr

    @property
    def expr(self) -> Placeholder:
        return self._expr

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_expr"), name)

    def integrate(self, **kwargs) -> "VoigtView":
        """Component-wise integral over the Voigt-packed tensor → VoigtView."""
        return VoigtView(self._expr.integrate(**kwargs))

    def coords(self, *args, **named_vars):
        """Bind coordinate Variables for partial-derivative-by-name access.

        ``σ.voigt.coords(x=x_var, y=y_var).x`` → VoigtView of ``∂σ/∂x`` —
        the partial is taken component-wise across the Voigt packing.
        """
        return _coords_dispatch(self, args, named_vars)

    # ------------------------------------------------------------------
    # Invariants and physics-style operations
    # ------------------------------------------------------------------

    def trace(self) -> "ScalarView":
        """Sum of normal stresses (first ``n`` components) → ScalarView."""

        def _fn(x):
            if x.shape[-1] == 3:
                return x[..., 0] + x[..., 1]
            return x[..., 0] + x[..., 1] + x[..., 2]

        return ScalarView(FunctionCall(_fn, [self._expr], "voigt_trace", True))

    def hydrostatic(self) -> "ScalarView":
        """Mean normal stress ``trace / n`` → ScalarView."""

        def _fn(x):
            if x.shape[-1] == 3:
                return (x[..., 0] + x[..., 1]) * 0.5
            return (x[..., 0] + x[..., 1] + x[..., 2]) / 3.0

        return ScalarView(FunctionCall(_fn, [self._expr], "hydrostatic", True))

    def von_mises(self) -> "ScalarView":
        """Von Mises equivalent stress → ScalarView.

        2-D Voigt ``[σ_xx, σ_yy, σ_xy]``::

            √(σ_xx² − σ_xx σ_yy + σ_yy² + 3 σ_xy²)

        3-D Voigt ``[σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy]``::

            √(½((σ_xx−σ_yy)² + (σ_yy−σ_zz)² + (σ_zz−σ_xx)²) + 3(σ_yz² + σ_xz² + σ_xy²))
        """

        def _fn(x):
            if x.shape[-1] == 3:
                return jnp.sqrt(x[..., 0] ** 2 - x[..., 0] * x[..., 1] + x[..., 1] ** 2 + 3.0 * x[..., 2] ** 2)
            return jnp.sqrt(
                0.5 * ((x[..., 0] - x[..., 1]) ** 2 + (x[..., 1] - x[..., 2]) ** 2 + (x[..., 2] - x[..., 0]) ** 2)
                + 3.0 * (x[..., 3] ** 2 + x[..., 4] ** 2 + x[..., 5] ** 2)
            )

        return ScalarView(FunctionCall(_fn, [self._expr], "von_mises", True))

    def deviatoric(self) -> "VoigtView":
        """Deviatoric part ``σ − (tr σ / n) · I`` (Voigt form) → VoigtView."""

        def _fn(x):
            n = 2 if x.shape[-1] == 3 else 3
            p = x[..., :n].sum(-1, keepdims=True) / n
            return x.at[..., :n].add(-p)

        return VoigtView(FunctionCall(_fn, [self._expr], "deviatoric"))

    def to_full(self) -> "MatrixView":
        """Voigt → full ``[..., n, n]`` symmetric matrix → MatrixView."""

        def _fn(x):
            if x.shape[-1] == 3:
                return jnp.stack(
                    [
                        jnp.stack([x[..., 0], x[..., 2]], axis=-1),
                        jnp.stack([x[..., 2], x[..., 1]], axis=-1),
                    ],
                    axis=-2,
                )
            return jnp.stack(
                [
                    jnp.stack([x[..., 0], x[..., 5], x[..., 4]], axis=-1),
                    jnp.stack([x[..., 5], x[..., 1], x[..., 3]], axis=-1),
                    jnp.stack([x[..., 4], x[..., 3], x[..., 2]], axis=-1),
                ],
                axis=-2,
            )

        return MatrixView(FunctionCall(_fn, [self._expr], "voigt_to_full"))

    def principal(self) -> "VectorView":
        """Principal stresses (eigenvalues of full tensor, ascending) → VectorView."""
        return VectorView(FunctionCall(lambda x: jnp.linalg.eigh(x)[0], [self.to_full()._expr], "principal"))

    def invariants(self) -> "VectorView":
        """Stress invariants → VectorView.

        2-D → ``[I1, I2]``. 3-D → ``[I1, I2, I3]`` with ``I3 = det σ``.
        """

        def _fn(v, A):
            if v.shape[-1] == 3:
                I1 = v[..., 0] + v[..., 1]
                I2 = v[..., 0] * v[..., 1] - v[..., 2] ** 2
                return jnp.stack([I1, I2], axis=-1)
            I1 = v[..., 0] + v[..., 1] + v[..., 2]
            I2 = (
                v[..., 0] * v[..., 1]
                + v[..., 1] * v[..., 2]
                + v[..., 2] * v[..., 0]
                - v[..., 3] ** 2
                - v[..., 4] ** 2
                - v[..., 5] ** 2
            )
            I3 = jnp.linalg.det(A)
            return jnp.stack([I1, I2, I3], axis=-1)

        return VectorView(FunctionCall(_fn, [self._expr, self.to_full()._expr], "invariants"))

    def max_shear(self) -> "ScalarView":
        """Maximum shear stress ``(σ_max − σ_min) / 2`` → ScalarView."""
        return ScalarView(
            FunctionCall(
                lambda e: (e[..., -1] - e[..., 0]) * 0.5,
                [self.principal()._expr],
                "max_shear",
                True,
            )
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        return VoigtView(self._expr + _unwrap(other))

    def __radd__(self, other):
        return VoigtView(_unwrap(other) + self._expr)

    def __sub__(self, other):
        return VoigtView(self._expr - _unwrap(other))

    def __rsub__(self, other):
        return VoigtView(_unwrap(other) - self._expr)

    def __mul__(self, other):
        return VoigtView(self._expr * _unwrap(other))

    def __rmul__(self, other):
        return VoigtView(_unwrap(other) * self._expr)

    def __neg__(self):
        return VoigtView(-self._expr)

    def __truediv__(self, other):
        return VoigtView(self._expr / _unwrap(other))

    def __rtruediv__(self, other):
        return VoigtView(_unwrap(other) / self._expr)


# ---------------------------------------------------------------------------
# NamedVectorView — positional component access (mirrors NamedMatrixView)
# ---------------------------------------------------------------------------


class NamedVectorView(VectorView):
    """VectorView with string-labelled components.

    Created by :meth:`VectorView.coords` with positional string arguments.
    Attribute access selects a component::

        v = velocity.vector.coords("x", "y")   # VectorView → NamedVectorView
        v.x   # ScalarView of v[..., 0]
        v.y   # ScalarView of v[..., 1]

    For partial-derivative-by-name semantics use the keyword form
    :meth:`VectorView.coords` with Variables — that returns a
    ``NamedVectorViewWithPartials`` instead.
    """

    def __init__(self, expr: Placeholder, names) -> None:
        super().__init__(expr)
        object.__setattr__(self, "_names", tuple(names))

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        names = object.__getattribute__(self, "_names")
        if key in names:
            return ScalarView(self._expr[..., names.index(key)])
        return getattr(object.__getattribute__(self, "_expr"), key)

    def component(self, name: str) -> "ScalarView":
        """Explicit component lookup by name."""
        i = self._names.index(name)
        return ScalarView(self._expr[..., i])


# ---------------------------------------------------------------------------
# Named<View>WithPartials — partial-derivative-by-name (kwargs form)
# ---------------------------------------------------------------------------


def _make_named_with_partials_cls(view_cls):
    """Build a Named<view_cls>WithPartials class.

    Records ``{name: Variable}`` bindings, and resolves attribute access
    against sequences of registered names (up to 4th order) by chaining
    ``self._expr.d(var)`` and wrapping the result in ``view_cls``.
    """

    class NamedWithPartials(view_cls):
        _base_view = view_cls

        def __init__(self, expr, coord_vars: dict) -> None:
            view_cls.__init__(self, expr)
            object.__setattr__(self, "_coord_vars", dict(coord_vars))

        def __getattr__(self, key: str):
            if key.startswith("_"):
                raise AttributeError(key)
            cv = object.__getattribute__(self, "_coord_vars")
            seq = _parse_partial_sequence(key, cv)
            if seq is not None:
                result = self._expr
                for name in seq:
                    result = result.d(cv[name])
                return type(self)._base_view(result)
            return getattr(object.__getattribute__(self, "_expr"), key)

        # -- convenience overrides that auto-fill the registered coord vars --

        def grad(self, *vars):
            """Spatial gradient — uses registered coords if no args given."""
            cv = object.__getattribute__(self, "_coord_vars")
            if not vars:
                names = tuple(cv.keys())
                vars_used = tuple(cv.values())
                from ..jnp_ops import concat

                inner = concat([self._expr.d(v) for v in vars_used])
                return NamedVectorViewWithPartials(inner, dict(zip(names, vars_used)))
            return view_cls.grad(self, *vars)  # delegate to base view's grad

        def laplacian(self, *vars):
            """Δself — uses registered coords if no args given."""
            cv = object.__getattribute__(self, "_coord_vars")
            vars_used = vars or tuple(cv.values())
            return ScalarView(self._expr.laplacian(*vars_used))

    NamedWithPartials.__name__ = f"Named{view_cls.__name__}WithPartials"
    NamedWithPartials.__qualname__ = NamedWithPartials.__name__
    return NamedWithPartials


NamedScalarViewWithPartials = _make_named_with_partials_cls(ScalarView)
NamedVectorViewWithPartials = _make_named_with_partials_cls(VectorView)
NamedComplexViewWithPartials = _make_named_with_partials_cls(ComplexView)
NamedMatrixViewWithPartials = _make_named_with_partials_cls(MatrixView)
NamedVoigtViewWithPartials = _make_named_with_partials_cls(VoigtView)


# Populate the tuple now that all classes exist (used by _unwrap()).
_VIEW_TYPES = (ScalarView, VectorView, ComplexView, MatrixView, VoigtView)

# Dispatch table used by `_coords_dispatch` to pick the Named<View>WithPartials
# wrapper for each base view type.
_NAMED_PARTIALS_CLS_FOR = {
    ScalarView: NamedScalarViewWithPartials,
    VectorView: NamedVectorViewWithPartials,
    ComplexView: NamedComplexViewWithPartials,
    MatrixView: NamedMatrixViewWithPartials,
    VoigtView: NamedVoigtViewWithPartials,
}


__all__ = [
    "ScalarView",
    "VectorView",
    "ComplexView",
    "MatrixView",
    "NamedMatrixView",
    "NamedVectorView",
    "VoigtView",
    "NamedScalarViewWithPartials",
    "NamedVectorViewWithPartials",
    "NamedComplexViewWithPartials",
    "NamedMatrixViewWithPartials",
    "NamedVoigtViewWithPartials",
]
