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

from functools import partial
from typing import TYPE_CHECKING, Optional

import jax
import jax.numpy as jnp

from ..jnp_ops import concat
from . import FunctionCall, Placeholder

# ---------------------------------------------------------------------------
# Stable spectral matrix functions (logm / expm / Aⁿ) on symmetric matrices.
# ---------------------------------------------------------------------------
# f(A) = V diag(f(λ)) Vᵀ. Its differential is the Daleckiĭ–Kreĭn / Löwner form
# with divided differences L_ij = (f(λ_i)−f(λ_j))/(λ_i−λ_j), which stay finite at
# λ_i = λ_j (the limit is f'(λ_i)) — exactly where ``jnp.linalg.eigh``'s own
# gradient blows up (a bare 1/(λ_i−λ_j)). A ``custom_jvp`` in this form makes
# matrix log/exp/pow differentiable through *repeated* eigenvalues — equal
# principal stretches in finite-strain plasticity, or any isotropic state —
# batched over a leading quadrature axis.
#
# Reference: N. J. Higham, *Functions of Matrices: Theory and Computation*,
# SIAM (2008), Theorem 3.11 (Daleckiĭ–Kreĭn); the Fréchet derivative of a
# symmetric matrix function is the Hadamard product of the Löwner matrix with
# the eigenbasis-rotated perturbation.
_SPECTRAL_DEGENERATE_TOL = 1e-7


def _spectral_scalar(w, kind: str, n: float):
    if kind == "log":
        return jnp.log(w)
    if kind == "exp":
        return jnp.exp(w)
    return w**n  # pow


def _spectral_dscalar(w, kind: str, n: float):
    if kind == "log":
        return 1.0 / w
    if kind == "exp":
        return jnp.exp(w)
    return n * w ** (n - 1.0)  # pow


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _spectral_matrix_function(a, kind: str, n: float):
    """``f(A) = V diag(f(λ)) Vᵀ`` for a symmetric ``A`` (batched on a leading axis)."""
    w, vecs = jnp.linalg.eigh(a)
    return (vecs * _spectral_scalar(w, kind, n)[..., None, :]) @ jnp.swapaxes(vecs, -2, -1)


@_spectral_matrix_function.defjvp
def _spectral_matrix_function_jvp(kind, n, primals, tangents):
    (a,), (da,) = primals, tangents
    w, vecs = jnp.linalg.eigh(a)
    fw, dfw = _spectral_scalar(w, kind, n), _spectral_dscalar(w, kind, n)
    vt = jnp.swapaxes(vecs, -2, -1)
    out = (vecs * fw[..., None, :]) @ vt
    # Löwner matrix: off-diagonal divided difference; diagonal (and near-degenerate) -> f'(λ)
    dw = w[..., :, None] - w[..., None, :]
    close = jnp.abs(dw) < _SPECTRAL_DEGENERATE_TOL
    loewner = jnp.where(
        close,
        0.5 * (dfw[..., :, None] + dfw[..., None, :]),
        (fw[..., :, None] - fw[..., None, :]) / jnp.where(close, 1.0, dw),
    )
    das = 0.5 * (da + jnp.swapaxes(da, -2, -1))  # symmetric part (A is symmetric)
    dout = vecs @ (loewner * (vt @ das @ vecs)) @ vt
    return out, dout


def _is_periodic_tie_combination(expr_a, expr_b) -> bool:
    """True iff ``expr_a - expr_b`` is a **periodic tie** ``u(A) - u(B)``: both sides reference the
    unknown — a ``TrialFunction`` (``jno.fem``) or the strong-form nodal-field unknown from
    ``domain.unknown()`` (``jno.fdm``) — and neither carries a ``TestFunction`` (a weak term would).
    This isolates the tie from ordinary weak forms so the coord-binding merge only relaxes for a true
    tie; unrelated two-field combinations still raise. Imports are local to avoid a trace↔solver cycle."""
    if expr_a is None or expr_b is None:
        return False
    from ..utils.solver.solver_helper import contains_node_type, iter_children
    from . import ModelCall, TestFunction, TrialFunction

    if contains_node_type(expr_a, TestFunction) or contains_node_type(expr_b, TestFunction):
        return False

    def _has_unknown(expr):
        if contains_node_type(expr, TrialFunction):
            return True
        seen = [expr]  # a jno.fdm nodal-field unknown is a ModelCall with model._fem_field == "node"
        while seen:
            node = seen.pop()
            node = getattr(node, "_expr", node)
            if isinstance(node, ModelCall) and getattr(getattr(node, "model", None), "_fem_field", None) == "node":
                return True
            seen.extend(iter_children(node) or ())
        return False

    return _has_unknown(expr_a) and _has_unknown(expr_b)


def _tag_of_coord_vars(cv) -> Optional[str]:
    """The single region tag shared by a bound view's coordinate Variables, or ``None``."""
    tags = {getattr(v, "tag", None) for v in (cv or {}).values()}
    tags = {t for t in tags if isinstance(t, str)}
    return next(iter(tags)) if len(tags) == 1 else None


if TYPE_CHECKING:
    from . import Tracker

    class _DelegatesToPlaceholder:
        """Type-checker-only declaration of the :class:`Placeholder` conveniences
        that every view reaches at runtime through its ``__getattr__`` →
        ``self._expr.<name>``.

        Without this, those fall-through names (``u.scalar.mse``, ``.mean``,
        ``.name(...)``, …) are typed ``Any`` and an IDE shows nothing on hover.
        The class is *empty at runtime* (see the ``else`` branch) — ``__getattr__``
        still does the real delegation; this only informs the type checker. A view
        that defines one of these names itself (e.g. ``ComplexView.real``) overrides
        the stub as usual.
        """

        mse: FunctionCall
        mae: FunctionCall
        mean: FunctionCall
        sum: FunctionCall
        min: FunctionCall
        max: FunctionCall
        std: FunctionCall
        shape: FunctionCall
        T: FunctionCall
        real: FunctionCall
        imag: FunctionCall

        def name(self, label: str) -> Placeholder: ...
        def tracker(self, interval: int = 1, reduce=None) -> Tracker: ...
        def reshape(self, *shape) -> FunctionCall: ...
        def equal(self, other) -> FunctionCall: ...
        def not_equal(self, other) -> FunctionCall: ...

else:

    class _DelegatesToPlaceholder:  # empty at runtime
        pass


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


class _SchemeProxy:
    """Re-uses the partial-derivative parsing of a ``Named<View>WithPartials``,
    but threads a fixed differentiation ``scheme`` through every ``.d(...)`` call.

    Surfaced via ``view.fd`` (finite-difference namespace): ``u.fd.x`` mirrors
    ``u.x`` but evaluates the partial via the FD scheme instead of AD.
    """

    __slots__ = ("_view", "_scheme")

    def __init__(self, view, scheme: str):
        object.__setattr__(self, "_view", view)
        object.__setattr__(self, "_scheme", scheme)

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        view = object.__getattribute__(self, "_view")
        scheme = object.__getattribute__(self, "_scheme")
        cv = object.__getattribute__(view, "_coord_vars")
        seq = _parse_partial_sequence(key, cv)
        if seq is None:
            raise AttributeError(
                f"{key!r} is not a registered partial-name sequence "
                f"(known names: {sorted(cv)}; use .bind(...) to register more)."
            )
        result = view._expr
        for name in seq:
            result = result.d(cv[name], scheme=scheme)
        return type(view)._base_view(result)


def _coords_dispatch(view_self, args: tuple, named_vars: dict, *, positional_factory=None):
    """Shared dispatch for ``.partials``/``.bind`` (kwargs) and ``.coords`` (positional names).

    * ``named_vars`` (kwargs, from ``.partials`` / ``.bind``) → partial-derivative wrapper.
    * ``args`` (positional strings, from ``.coords``) → component / element-access wrapper,
      valid only where ``positional_factory`` is provided (``VectorView`` / ``MatrixView``).
    """
    if named_vars and args:
        raise TypeError("coords() expects either kwargs (name=Variable) or positional names, not both")
    if named_vars:
        # Re-bind: ``view_self`` already carries bindings (it is a Named<View>WithPartials, or a
        # FieldViewWithPartials). Stay on its OWN class and merge, new wins -- which is exactly what
        # ``_rewrap``'s conflict message tells the user ``.bind(...)`` is for. Resolving through
        # ``_base_view`` instead would send a FieldViewWithPartials to NamedScalarViewWithPartials and
        # silently swap its FD-only derivatives for AD ones, which are identically zero for an operator
        # network that never takes x/y as inputs -- a wrong answer, not an error.
        existing = getattr(view_self, "_coord_vars", None)
        if existing is not None:
            return type(view_self)(view_self._expr, {**existing, **named_vars})
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


class ScalarView(_DelegatesToPlaceholder):
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

    def _rewrap(self, new_expr, other=None) -> "ScalarView":
        """Wrap ``new_expr`` in the same view subclass as ``self``.

        The optional ``other`` argument names the original second operand of
        the arithmetic / differential op, so subclasses with extra state
        (``Named*WithPartials``) can validate that the binding remains
        consistent — e.g. raising on a name collision with a different Variable.
        Base view: ignores ``other``.
        """
        return ScalarView(new_expr)

    def integrate(self, **kwargs) -> "ScalarView":
        """Integrate the underlying scalar, preserving the ScalarView type."""
        return self._rewrap(self._expr.integrate(**kwargs))

    @property
    def stop_gradient(self) -> "ScalarView":
        """Block gradient flow through this view, preserving the ScalarView type."""
        return self._rewrap(self._expr.stop_gradient)

    def d(self, v, scheme: str = "automatic_differentiation") -> "ScalarView":
        """``∂self/∂v`` — same view type.

        ``v`` may be a coordinate Variable (``∂self/∂x``) or a boundary/interface **normal** from
        ``domain.variable(tag, normals=True)``, in which case this is the normal derivative ``∂self/∂n``
        (the assembler resolves the flux ``∇self·n``). So an interface flux / material condition reads
        with the same ``.d`` as a plain derivative::

            n = domain.variable("interface_A_B", normals=True)
            kA * uA.d(n) - kB * uB.d(n)          # flux continuity across the interface
        """
        return self._rewrap(self._expr.d(v, scheme=scheme))

    def d2(self, v, scheme: str = "automatic_differentiation") -> "ScalarView":
        """``∂²self/∂v²`` — same view type."""
        return self._rewrap(self._expr.d2(v, scheme=scheme))

    def dd(self, v, w=None, scheme: str = "automatic_differentiation") -> "ScalarView":
        """Mixed second derivative ``∂²self/∂v∂w`` — same view type."""
        return self._rewrap(self._expr.dd(v, w, scheme=scheme))

    def partials(self, **named_vars):
        """Bind Variables to names for partial-derivative-by-attribute access.

        ``u.scalar.partials(x=x, y=y, t=t).x`` → ``ScalarView`` of ``∂u/∂x``;
        ``u.xy`` → ``∂²u/∂x∂y`` (up to 4th order; see :func:`_parse_partial_sequence`).
        ``u.t`` works the same as any other bound name — registration is
        purely lexical, no spatial-vs-temporal distinction.
        """
        return _coords_dispatch(self, (), named_vars)

    # ``bind`` is a synonym — programming flavour, same semantics.
    bind = partials

    def freeze(self, values) -> "ScalarView":
        """Pin this field's DOFs to the known nodal array ``values`` (e.g. a precomputed
        coarse solution). The returned view's value and ``.x`` / ``.y`` give that KNOWN
        field's value / gradient at the quadrature points — usable as a neural-coefficient
        input (``net(xi, yi, ui.freeze(u0).x, ui.freeze(u0).y)``) while the weak form stays
        LINEAR in the live unknown. Coordinate bindings from ``.bind(x=, y=)`` are preserved,
        so ``.x`` / ``.y`` keep working.

        The frozen field (and its gradient) is also **readable standalone via ``.eval()``** — a
        functional of a solved field written as pure traced math. With the region's normals
        (``d.variable(tag, normals=True, split=True)`` → ``…, n_x, n_y``) this gives the boundary
        normal-flux, e.g. a Stefan interface speed::

            x, y, t, nx, ny = d.variable("boundary", normals=True, split=True)
            Tf = u.bind(x=x, y=y).freeze(sol)
            v_n = (-(k / L) * (Tf.x * nx + Tf.y * ny)).eval()   # ∇T·n on the boundary, as an array

        The gradient of a frozen (nodal) field is evaluated by the **FD-over-mesh** scheme (there is no
        analytic coordinate-function to auto-differentiate). See :class:`jno.trace.FrozenField`."""
        from . import FrozenField

        cv = getattr(self, "_coord_vars", None)
        # Carry the mesh domain + the (spatial) region tag so a standalone `.eval()` of the frozen field
        # — or of its `.x`/`.y` gradient — can map the nodal values onto that region's sample points.
        domain = coord_tag = None
        for _var in (cv or {}).values():
            if getattr(_var, "axis", "spatial") == "temporal":
                continue  # x/y share the spatial region tag; skip the time coordinate
            if getattr(_var, "_domain", None) is not None:
                domain, coord_tag = _var._domain, getattr(_var, "tag", None)
                break
        frozen = FrozenField(self._expr, values, domain=domain, coord_tag=coord_tag)
        if cv:
            return _coords_dispatch(ScalarView(frozen), (), dict(cv))
        return ScalarView(frozen)

    def freeze_path(self, frames) -> "ScalarView":
        """Like :meth:`freeze`, but the pinned nodal values vary **per load step** of a
        ``domain(tau=...)`` march: ``frames`` has shape ``(n_load_steps, n_nodes)`` and at step ``k`` the
        field presents ``frames[k]`` at the quadrature points (its ``.x`` / ``.y`` / ``.z`` gradients too).

        This drives a load path with a **precomputed field history** — one nodal field per step, from a
        prior solve or prescribed data — so the field history *is* the load (a one-way coupling)::

            u, phi = d.fem_symbols(value_shape=(3,))
            xi, yi, zi = d.variable("interior", split=True)[:3]
            g = f.bind(x=xi, y=yi, z=zi).freeze_path(field_frames)  # (n_load_steps, n_nodes)
            eigenstrain = beta * g * I3                             # a prescribed per-step eigenstrain
            fem = jno.fem([inner(sigma(u, g), eps(phi), 2), ep.evolves(...), *bcs])
            fem.solve()   # marches the tau= grid; step k sees field_frames[k]

        ``frames``'s leading dimension must equal the number of ``tau=`` load steps. Scalar Lagrange
        fields only. See :class:`jno.trace.LoadPathField`."""
        from . import LoadPathField

        cv = getattr(self, "_coord_vars", None)
        domain = coord_tag = None
        for _var in (cv or {}).values():
            if getattr(_var, "axis", "spatial") == "temporal":
                continue
            if getattr(_var, "_domain", None) is not None:
                domain, coord_tag = _var._domain, getattr(_var, "tag", None)
                break
        field = LoadPathField(self._expr, frames, domain=domain, coord_tag=coord_tag)
        # Return the node itself (a scalar trace expression): a load-path field is consumed by its VALUE in
        # the weak form (e.g. the thermal strain ``β·θ·I``), and the assembler interpolates it at the quad
        # points via the FrozenField path. (Gradients ``.x/.y`` of a per-step field are not exposed.)
        return field

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

    # -- arithmetic with cross-type dispatch (all paths go through _rewrap so
    #    subclasses like NamedScalarViewWithPartials keep their state) --
    def __add__(self, other):
        return self._rewrap(self._expr + _unwrap(other), other=other)

    def __radd__(self, other):
        return self._rewrap(_unwrap(other) + self._expr, other=other)

    def __sub__(self, other):
        return self._rewrap(self._expr - _unwrap(other), other=other)

    def __rsub__(self, other):
        return self._rewrap(_unwrap(other) - self._expr, other=other)

    def __neg__(self):
        return self._rewrap(-self._expr)

    def __pos__(self):
        """``+expr`` is the identity, as it is for a Python number: returns ``self`` unchanged.

        Defined so a term list can write a signed source symmetrically -- ``[-r, -r, +r, +r]`` reads as
        stoichiometry, and without this the ``+r`` legs raise ``TypeError`` while the ``-r`` legs work.
        Returns ``self`` rather than a ``1 * expr`` node, so it costs nothing in the graph."""
        return self

    def __truediv__(self, other):
        return self._rewrap(self._expr / _unwrap(other), other=other)

    def __rtruediv__(self, other):
        return self._rewrap(_unwrap(other) / self._expr, other=other)

    def __mul__(self, other):
        if isinstance(other, VectorView):
            return VectorView(self._expr * other._expr)
        if isinstance(other, MatrixView):
            return MatrixView(self._expr * other._expr)
        if isinstance(other, VoigtView):
            return VoigtView(self._expr * other._expr)
        if isinstance(other, ComplexView):
            return ComplexView(self._expr * other._expr)
        return self._rewrap(self._expr * _unwrap(other), other=other)

    def __rmul__(self, other):
        return self._rewrap(_unwrap(other) * self._expr, other=other)

    def __pow__(self, n):
        return self._rewrap(self._expr ** _unwrap(n))


# ---------------------------------------------------------------------------
# VectorView
# ---------------------------------------------------------------------------


class VectorView(_DelegatesToPlaceholder):
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

    def _rewrap(self, new_expr, other=None) -> "VectorView":
        """Wrap ``new_expr`` in the same view subclass as ``self``."""
        return VectorView(new_expr)

    def _frozen_domain_tag(self):
        """The mesh domain + spatial region tag from this view's ``.bind(...)`` coords, so a frozen field
        can map its nodal values onto the region (shared by :meth:`freeze` / :meth:`freeze_path`)."""
        cv = getattr(self, "_coord_vars", None)
        for _var in (cv or {}).values():
            if getattr(_var, "axis", "spatial") == "temporal":
                continue  # x/y/z share the spatial region tag; skip the time coordinate
            if getattr(_var, "_domain", None) is not None:
                return cv, _var._domain, getattr(_var, "tag", None)
        return cv, None, None

    def freeze(self, values) -> "VectorView":
        """Pin this VECTOR field's DOFs to the known nodal array ``values`` of shape ``(n_nodes, vec)`` --
        the vector analogue of :meth:`ScalarView.freeze`. The returned view (and its component/gradient
        views) give that KNOWN vector field at the quadrature points, so it conditions a coefficient while
        the weak form stays LINEAR in the live unknown -- e.g. a precomputed vector field (a velocity, a
        prior displacement) driving a coefficient. Coordinate bindings from ``.bind(x=, y=, z=)`` are
        preserved. See :class:`jno.trace.FrozenField`."""
        from . import FrozenField

        cv, domain, coord_tag = self._frozen_domain_tag()
        view = VectorView(FrozenField(self._expr, values, domain=domain, coord_tag=coord_tag))
        return _coords_dispatch(view, (), dict(cv)) if cv else view

    def freeze_path(self, frames) -> "VectorView":
        """Like :meth:`freeze`, but the pinned VECTOR nodal values vary **per load step** of a
        ``domain(tau=...)`` march: ``frames`` has shape ``(n_load_steps, n_nodes, vec)`` and at step ``k``
        the field presents ``frames[k]`` at the quadrature points -- the vector analogue of
        :meth:`ScalarView.freeze_path` (e.g. a prescribed per-step field driving an eigenstrain / a
        one-way coupling). See :class:`jno.trace.LoadPathField`."""
        from . import LoadPathField

        cv, domain, coord_tag = self._frozen_domain_tag()
        view = VectorView(LoadPathField(self._expr, frames, domain=domain, coord_tag=coord_tag))
        return _coords_dispatch(view, (), dict(cv)) if cv else view

    @property
    def complex(self) -> "ComplexVectorView":
        """Reinterpret this vector field as a **complex vector** ``[..., d, 2]`` (last axis =
        ``[re, im]``): ``.real`` / ``.imag`` then give the real / imaginary ``d``-vectors. The
        underlying Placeholder must carry that layout. See :class:`ComplexVectorView`."""
        return ComplexVectorView(self._expr)

    def integrate(self, **kwargs) -> "VectorView":
        """Component-wise integral, preserving VectorView type."""
        return self._rewrap(self._expr.integrate(**kwargs))

    @property
    def stop_gradient(self) -> "VectorView":
        """Block gradient flow component-wise, preserving the VectorView type."""
        return self._rewrap(self._expr.stop_gradient)

    def d(self, v, scheme: str = "automatic_differentiation") -> "VectorView":
        """Component-wise ``∂self/∂v`` — same view type."""
        return self._rewrap(self._expr.d(v, scheme=scheme))

    def d2(self, v, scheme: str = "automatic_differentiation") -> "VectorView":
        return self._rewrap(self._expr.d2(v, scheme=scheme))

    def dd(self, v, w=None, scheme: str = "automatic_differentiation") -> "VectorView":
        return self._rewrap(self._expr.dd(v, w, scheme=scheme))

    def partials(self, **named_vars):
        """Bind Variables to names for component-wise partial-derivative-by-attribute access.

        ``v.vector.partials(x=x, y=y).x`` → ``VectorView`` of ``∂v/∂x``.
        """
        return _coords_dispatch(self, (), named_vars, positional_factory=NamedVectorView)

    bind = partials  # synonym

    def coords(self, *args):
        """Name a vector's components for attribute access::

            v.coords("x", "y")        # or v.coords(["x", "y"])

        returns a ``NamedVectorView`` whose components are reachable as ``.x``, ``.y``.
        To bind coordinates for derivatives use ``.bind(**vars)`` / ``.partials(**vars)``.
        """
        return _coords_dispatch(self, args, {}, positional_factory=NamedVectorView)

    # -- component access --
    def _c(self, i: int) -> "ScalarView":
        comp = ScalarView(self._expr[..., i])
        # Preserve the region binding so a bound view's component
        # (e.g. ``u.bind(x=xr, y=yr)[1]``) still carries ``_coord_vars`` — needed
        # for per-component (roller) Dirichlet and vector boundary terms.
        cv = getattr(self, "_coord_vars", None)
        if cv:
            return comp.bind(**cv)
        return comp

    def component(self, i: int) -> "ScalarView":
        """i-th component → ScalarView."""
        return self._c(i)

    def __getitem__(self, i: int) -> "ScalarView":
        """``v[i]`` — i-th component (alias for :meth:`component`)."""
        return self._c(i)

    # -- differential operators --
    def _ops_coords(self, vars):
        """Coordinate Variables for a differential operator: the explicit ``vars`` if any, else the
        spatial coords bound via ``.bind(x=.., y=..)`` -- so ``u.bind(x=x, y=y).curl()`` (and ``.div()`` /
        ``.grad()``) need no re-passing. Only ``.bind(...)`` views carry ``_coord_vars``."""
        if vars:
            return vars
        cv = getattr(self, "_coord_vars", None)
        if not cv:
            raise TypeError(
                "div()/curl()/grad() need spatial coordinate Variables: pass them explicitly "
                "(e.g. u.curl(x, y)) or bind them first (u.bind(x=x, y=y).curl())."
            )
        return tuple(cv[n] for n in ("x", "y", "z") if n in cv)

    def div(self, *vars) -> "ScalarView":
        """Divergence ∑ ∂u_i/∂x_i → ScalarView.

        With no arguments, differentiates against the coordinates bound by ``.bind(x=.., y=..)``.
        Number of variables must equal the last dimension of the vector.
        """
        vars = self._ops_coords(vars)
        terms = [self._c(i).expr.d(v) for i, v in enumerate(vars)]
        total = terms[0]
        for t in terms[1:]:
            total = total + t
        return ScalarView(total)

    def curl(self, *vars):
        """Curl of the vector field.

        With no arguments, differentiates against the coordinates bound by ``.bind(x=.., y=..)``.
        2 variables (2-D) → ``ScalarView`` (∂u_y/∂x − ∂u_x/∂y).
        3 variables (3-D) → ``VectorView`` (the curl vector).
        """
        vars = self._ops_coords(vars)
        if len(vars) == 2:
            x, y = vars
            return ScalarView(self._c(1).expr.d(x) - self._c(0).expr.d(y))
        if len(vars) == 3:
            x, y, z = vars
            cx = self._c(2).expr.d(y) - self._c(1).expr.d(z)
            cy = self._c(0).expr.d(z) - self._c(2).expr.d(x)
            cz = self._c(1).expr.d(x) - self._c(0).expr.d(y)
            # Stack the three scalar components onto a NEW last axis (like ``jacobian``), not ``concat``
            # (concatenate-on-last): for an FEM test function each component is per-DOF ``(n_quad, n_dof)``,
            # which concat would merge into ``(n_quad, 3·n_dof)``; stacking gives the correct
            # ``(n_quad, n_dof, 3)`` vector. Identical to concat for 1-D (nodal/network) components.
            return VectorView(FunctionCall(lambda *cs: jnp.stack(cs, axis=-1), [cx, cy, cz], "stack"))
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
        ``[..., n, m]`` with ``J[..., i, j] = ∂u_i/∂x_j``. With no arguments, uses the
        coordinates bound by ``.bind(x=.., y=..)``.
        """
        vars = self._ops_coords(vars)
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

    # -- arithmetic — all paths go through _rewrap so subclasses keep state --
    def __add__(self, other):
        return self._rewrap(self._expr + _unwrap(other), other=other)

    def __radd__(self, other):
        return self._rewrap(_unwrap(other) + self._expr, other=other)

    def __sub__(self, other):
        return self._rewrap(self._expr - _unwrap(other), other=other)

    def __rsub__(self, other):
        return self._rewrap(_unwrap(other) - self._expr, other=other)

    def __neg__(self):
        return self._rewrap(-self._expr)

    def __pos__(self):
        """``+expr`` is the identity, as it is for a Python number: returns ``self`` unchanged.

        Defined so a term list can write a signed source symmetrically -- ``[-r, -r, +r, +r]`` reads as
        stoichiometry, and without this the ``+r`` legs raise ``TypeError`` while the ``-r`` legs work.
        Returns ``self`` rather than a ``1 * expr`` node, so it costs nothing in the graph."""
        return self

    def __mul__(self, other):
        return self._rewrap(self._expr * _unwrap(other), other=other)

    def __rmul__(self, other):
        return self._rewrap(_unwrap(other) * self._expr, other=other)

    def __truediv__(self, other):
        return self._rewrap(self._expr / _unwrap(other), other=other)

    def __rtruediv__(self, other):
        return self._rewrap(_unwrap(other) / self._expr, other=other)

    def __pow__(self, n):
        return self._rewrap(self._expr ** _unwrap(n))

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


class ComplexView(_DelegatesToPlaceholder):
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

    def _rewrap(self, new_expr, other=None) -> "ComplexView":
        """Wrap ``new_expr`` in the same view subclass as ``self``."""
        return ComplexView(new_expr)

    def integrate(self, **kwargs) -> "ComplexView":
        """Component-wise integral of the [re, im] split, preserving ComplexView."""
        return self._rewrap(self._expr.integrate(**kwargs))

    @property
    def stop_gradient(self) -> "ComplexView":
        """Block gradient flow through the [re, im] pair, preserving ComplexView."""
        return self._rewrap(self._expr.stop_gradient)

    def d(self, v, scheme: str = "automatic_differentiation") -> "ComplexView":
        """Component-wise ``∂self/∂v`` — same view type."""
        return self._rewrap(self._expr.d(v, scheme=scheme))

    def d2(self, v, scheme: str = "automatic_differentiation") -> "ComplexView":
        return self._rewrap(self._expr.d2(v, scheme=scheme))

    def dd(self, v, w=None, scheme: str = "automatic_differentiation") -> "ComplexView":
        return self._rewrap(self._expr.dd(v, w, scheme=scheme))

    def partials(self, **named_vars):
        """Bind Variables to names for partial-derivative-by-attribute access.

        ``ψ.complex.partials(x=x, y=y).x`` → ``ComplexView`` of ``∂ψ/∂x`` —
        the partial is taken element-wise across the ``[re, im]`` split.
        """
        return _coords_dispatch(self, (), named_vars)

    bind = partials  # synonym

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
        return self._rewrap(self._expr + _unwrap(other), other=other)

    def __radd__(self, other):
        return self._rewrap(_unwrap(other) + self._expr, other=other)

    def __sub__(self, other):
        return self._rewrap(self._expr - _unwrap(other), other=other)

    def __rsub__(self, other):
        return self._rewrap(_unwrap(other) - self._expr, other=other)

    def __neg__(self):
        return self._rewrap(-self._expr)

    def __pos__(self):
        """``+expr`` is the identity, as it is for a Python number: returns ``self`` unchanged.

        Defined so a term list can write a signed source symmetrically -- ``[-r, -r, +r, +r]`` reads as
        stoichiometry, and without this the ``+r`` legs raise ``TypeError`` while the ``-r`` legs work.
        Returns ``self`` rather than a ``1 * expr`` node, so it costs nothing in the graph."""
        return self

    def __mul__(self, other):
        return self._rewrap(self._expr * _unwrap(other), other=other)

    def __rmul__(self, other):
        return self._rewrap(_unwrap(other) * self._expr, other=other)

    def __truediv__(self, other):
        return self._rewrap(self._expr / _unwrap(other), other=other)

    def __rtruediv__(self, other):
        return self._rewrap(_unwrap(other) / self._expr, other=other)

    def __pow__(self, n):
        return self._rewrap(self._expr ** _unwrap(n))


# ---------------------------------------------------------------------------
# ComplexVectorView
# ---------------------------------------------------------------------------


class ComplexVectorView(_DelegatesToPlaceholder):
    """Semantic view of a Placeholder as a **complex vector** field, shape ``[..., d, 2]`` (``d``
    vector components; last axis ``= 2 = [re, im]``). Reached via ``placeholder.vector.complex``.

    ``.real`` / ``.imag`` return the real and imaginary parts as :class:`VectorView`\\s (each a real
    ``d``-vector), so ``E.real.dot(n)`` / ``E.imag.div(x, y)`` work. Complex algebra (``.mul``,
    ``.conj``) is componentwise (Hadamard) over the vector; ``.mul`` against a complex *scalar*
    (:class:`ComplexView`) broadcasts. Mirrors :class:`ComplexView`, but each part is a vector. The
    natural FEM realisation is two coupled real vector fields ``(E_r, E_i)``."""

    def __init__(self, expr: Placeholder) -> None:
        self._expr = expr

    @property
    def expr(self) -> Placeholder:
        return self._expr

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_expr"), name)

    def _rewrap(self, new_expr, other=None) -> "ComplexVectorView":
        return ComplexVectorView(new_expr)

    @staticmethod
    def _pack(re_expr, im_expr) -> Placeholder:
        """Rebuild the ``[..., d, 2]`` layout from real/imag vector parts (new last axis = [re, im])."""
        return FunctionCall(lambda a, b: jnp.stack([a, b], axis=-1), [re_expr, im_expr], "cvpack")

    @property
    def real(self) -> "VectorView":
        """Real part ``expr[..., 0]`` (a real ``d``-vector) → VectorView."""
        return VectorView(self._expr[..., 0])

    @property
    def imag(self) -> "VectorView":
        """Imaginary part ``expr[..., 1]`` (a real ``d``-vector) → VectorView."""
        return VectorView(self._expr[..., 1])

    @property
    def conj(self) -> "ComplexVectorView":
        """Complex conjugate ``[re, -im]`` (componentwise) → ComplexVectorView."""
        return ComplexVectorView(self._pack(self.real.expr, -self.imag.expr))

    def mul(self, other) -> "ComplexVectorView":
        """Componentwise complex product ``(a+bi)(c+di)=(ac-bd)+(ad+bc)i``. A real ``other`` scales
        both parts; a :class:`ComplexView` (complex scalar) broadcasts against the vector."""
        if not isinstance(other, (ComplexView, ComplexVectorView)):
            return self._rewrap(self._expr * _unwrap(other))
        re = self.real.expr * other.real.expr - self.imag.expr * other.imag.expr
        im = self.real.expr * other.imag.expr + self.imag.expr * other.real.expr
        return ComplexVectorView(self._pack(re, im))

    @property
    def abs(self) -> "VectorView":
        """Per-component modulus ``sqrt(re² + im²)`` → VectorView."""
        return VectorView(FunctionCall(lambda x: jnp.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2), [self._expr], "cvabs"))

    @property
    def stop_gradient(self) -> "ComplexVectorView":
        return self._rewrap(self._expr.stop_gradient)

    def integrate(self, **kwargs) -> "ComplexVectorView":
        return self._rewrap(self._expr.integrate(**kwargs))

    def d(self, v, scheme: str = "automatic_differentiation") -> "ComplexVectorView":
        return self._rewrap(self._expr.d(v, scheme=scheme))

    def partials(self, **named_vars):
        """Bind Variables for partial-by-attribute access; partials are component-wise over [re, im]."""
        return _coords_dispatch(self, (), named_vars)

    bind = partials

    def to_native(self) -> Placeholder:
        """Convert split ``[..., d, 2]`` → native complex ``[..., d]``."""
        return FunctionCall(lambda x: x[..., 0] + 1j * x[..., 1], [self._expr], "to_native")

    # elementwise (scalar) arithmetic; for the complex product use .mul — mirrors ComplexView
    def __add__(self, other):
        return self._rewrap(self._expr + _unwrap(other), other=other)

    def __radd__(self, other):
        return self._rewrap(_unwrap(other) + self._expr, other=other)

    def __sub__(self, other):
        return self._rewrap(self._expr - _unwrap(other), other=other)

    def __rsub__(self, other):
        return self._rewrap(_unwrap(other) - self._expr, other=other)

    def __neg__(self):
        return self._rewrap(-self._expr)

    def __pos__(self):
        """``+expr`` is the identity, as it is for a Python number: returns ``self`` unchanged.

        Defined so a term list can write a signed source symmetrically -- ``[-r, -r, +r, +r]`` reads as
        stoichiometry, and without this the ``+r`` legs raise ``TypeError`` while the ``-r`` legs work.
        Returns ``self`` rather than a ``1 * expr`` node, so it costs nothing in the graph."""
        return self

    def __mul__(self, other):
        return self._rewrap(self._expr * _unwrap(other), other=other)

    def __rmul__(self, other):
        return self._rewrap(_unwrap(other) * self._expr, other=other)


# ---------------------------------------------------------------------------
# ComplexPair — complex as two SEPARATE real parts (FEM-friendly)
# ---------------------------------------------------------------------------


def _is_complex_pair(o) -> bool:
    return isinstance(o, ComplexPair)


def _radd(a, b):
    """``a + b`` with ``None`` standing for an identically-zero part."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _rsub(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return -b
    if b is None:
        return a
    return a - b


def _rmul(a, b):
    if a is None or b is None:
        return None
    return a * b


def _as_pair(o):
    """Coerce ``o`` to a :class:`ComplexPair` (``NotImplemented`` if not possible).

    A Python ``complex`` becomes its ``(real, imag)`` constants; any real
    expression/number becomes ``(o, 0)``."""
    if isinstance(o, ComplexPair):
        return o
    if isinstance(o, complex):
        return ComplexPair(o.real if o.real != 0 else None, o.imag if o.imag != 0 else None)
    if isinstance(o, (int, float)) or isinstance(o, (Placeholder, ScalarView, VectorView)):
        return ComplexPair(o, None)
    return NotImplemented


def _complex_times_real(real_view, c: complex) -> "ComplexPair":
    """``c * (real field)`` → a :class:`ComplexPair` (``1j·field`` → ``(0, field)``)."""

    def scale(coeff):
        if coeff == 0:
            return None
        if coeff == 1:
            return real_view
        return coeff * real_view

    return ComplexPair(scale(c.real), scale(c.imag))


class ComplexPair:
    """A complex quantity held as two **separate** real parts ``(re, im)``.

    Unlike :class:`ComplexView` (which packs ``[re, im]`` into one Placeholder's
    last axis), each part here is an independent expression — the FEM-friendly
    representation of a complex field built from real ``fem_symbols``::

        E = Er.bind(x=x, y=y) + 1j * Ei.bind(x=x, y=y)      # -> ComplexPair

    ``1j`` is just the imaginary unit; the tracer carries the real and imaginary
    parts through every operation (``*`` is the complex product, ``.conj``, ``.x``,
    ``[i]`` map over both). ``.real`` / ``.imag`` hand back the two parts as the
    user's own real fields, so a complex weak form's ``.real`` lowers directly onto
    the coupled (multifield) real system that ``jno.fem`` already assembles — no
    separate complex machinery. A ``None`` part means "identically zero"."""

    __slots__ = ("_re", "_im")

    # Duck-typed marker for Placeholder._wrap: a pair holds TWO expressions, so it cannot be wrapped
    # as one operand -- the Placeholder-side binary op must yield to ComplexPair's reflected op, which
    # distributes over (re, im). Without this, `parameter * pair` died in jnp.asarray(ComplexPair)
    # INSIDE Placeholder.__mul__, so Python never consulted the pair at all.
    _is_complex_pair = True

    def __init__(self, re, im=None):
        self._re = re
        self._im = im

    # -- accessors --
    @property
    def real(self):
        return self._re

    @property
    def imag(self):
        return self._im if self._im is not None else 0.0

    @property
    def conj(self) -> "ComplexPair":
        return ComplexPair(self._re, None if self._im is None else -self._im)

    # -- field-like passthroughs (map over both parts) --
    def _map(self, fn) -> "ComplexPair":
        return ComplexPair(fn(self._re), None if self._im is None else fn(self._im))

    @property
    def x(self) -> "ComplexPair":
        return self._map(lambda p: p.x)

    @property
    def y(self) -> "ComplexPair":
        return self._map(lambda p: p.y)

    @property
    def z(self) -> "ComplexPair":
        return self._map(lambda p: p.z)

    @property
    def t(self) -> "ComplexPair":
        return self._map(lambda p: p.t)

    def __getitem__(self, i) -> "ComplexPair":
        return self._map(lambda p: p[i])

    def bind(self, **kw) -> "ComplexPair":
        return self._map(lambda p: p.bind(**kw))

    partials = bind

    def d(self, v, **kw) -> "ComplexPair":
        return self._map(lambda p: p.d(v, **kw))

    def dot(self, other) -> "ComplexPair":
        """Complex dot ``∑_i self_i · other_i`` of two complex vectors → complex scalar."""
        o = _as_pair(other)
        if o is NotImplemented:
            return NotImplemented
        rr = self._re.dot(o._re) if (self._re is not None and o._re is not None) else None
        ii = self._im.dot(o._im) if (self._im is not None and o._im is not None) else None
        ri = self._re.dot(o._im) if (self._re is not None and o._im is not None) else None
        ir = self._im.dot(o._re) if (self._im is not None and o._re is not None) else None
        return ComplexPair(_rsub(rr, ii), _radd(ri, ir))

    # -- complex algebra --
    def __add__(self, other) -> "ComplexPair":
        o = _as_pair(other)
        if o is NotImplemented:
            return NotImplemented
        return ComplexPair(_radd(self._re, o._re), _radd(self._im, o._im))

    __radd__ = __add__

    def __sub__(self, other) -> "ComplexPair":
        o = _as_pair(other)
        if o is NotImplemented:
            return NotImplemented
        return ComplexPair(_rsub(self._re, o._re), _rsub(self._im, o._im))

    def __rsub__(self, other) -> "ComplexPair":
        o = _as_pair(other)
        if o is NotImplemented:
            return NotImplemented
        return ComplexPair(_rsub(o._re, self._re), _rsub(o._im, self._im))

    def __neg__(self) -> "ComplexPair":
        return ComplexPair(-self._re, None if self._im is None else -self._im)

    def __pos__(self) -> "ComplexPair":
        """``+expr`` is the identity, as it is for a Python number: returns ``self`` unchanged.

        Defined so a term list can write a signed source symmetrically -- ``[-r, -r, +r, +r]`` reads as
        stoichiometry, and without this the ``+r`` legs raise ``TypeError`` while the ``-r`` legs work.
        Returns ``self`` rather than a ``1 * expr`` node, so it costs nothing in the graph."""
        return self

    def __mul__(self, other) -> "ComplexPair":
        o = _as_pair(other)
        if o is NotImplemented:
            return NotImplemented
        re = _rsub(_rmul(self._re, o._re), _rmul(self._im, o._im))
        im = _radd(_rmul(self._re, o._im), _rmul(self._im, o._re))
        return ComplexPair(re, im)

    __rmul__ = __mul__

    def __truediv__(self, other) -> "ComplexPair":
        if isinstance(other, ComplexPair):
            raise TypeError("ComplexPair: division by a complex quantity is not supported")
        return ComplexPair(self._re / other, None if self._im is None else self._im / other)

    def __repr__(self) -> str:
        return f"ComplexPair(re={self._re!r}, im={self._im!r})"


# ---------------------------------------------------------------------------
# MatrixView
# ---------------------------------------------------------------------------


class MatrixView(_DelegatesToPlaceholder):
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

    def _rewrap(self, new_expr, other=None) -> "MatrixView":
        """Wrap ``new_expr`` in the same view subclass as ``self``."""
        return MatrixView(new_expr)

    def integrate(self, **kwargs) -> "MatrixView":
        """Element-wise integral of each matrix component over the integration
        domain (not a matrix-algebra integral) — MatrixView preserved.

        Useful for things like the spatial average of an anisotropic
        diffusivity field.  Each ``[i, j]`` entry is integrated independently.
        """
        return self._rewrap(self._expr.integrate(**kwargs))

    @property
    def stop_gradient(self) -> "MatrixView":
        """Block gradient flow element-wise, preserving the MatrixView type."""
        return self._rewrap(self._expr.stop_gradient)

    def d(self, v, scheme: str = "automatic_differentiation") -> "MatrixView":
        """Element-wise ``∂self/∂v`` — same view type."""
        return self._rewrap(self._expr.d(v, scheme=scheme))

    def d2(self, v, scheme: str = "automatic_differentiation") -> "MatrixView":
        return self._rewrap(self._expr.d2(v, scheme=scheme))

    def dd(self, v, w=None, scheme: str = "automatic_differentiation") -> "MatrixView":
        return self._rewrap(self._expr.dd(v, w, scheme=scheme))

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
        """Matrix logarithm ``logm(A)`` via eigendecomposition (symmetric / SPD), with a gradient that
        stays finite at repeated eigenvalues (Daleckiĭ–Kreĭn form) → MatrixView."""
        return MatrixView(FunctionCall(lambda x: _spectral_matrix_function(x, "log", 0.0), [self._expr], "mat_log"))

    def exp(self) -> "MatrixView":
        """Matrix exponential ``expm(A)`` via eigendecomposition (symmetric), stable at repeated
        eigenvalues → MatrixView."""
        return MatrixView(FunctionCall(lambda x: _spectral_matrix_function(x, "exp", 0.0), [self._expr], "mat_exp"))

    def pow(self, n: float) -> "MatrixView":
        """Matrix power ``Aⁿ`` via eigendecomposition (symmetric), stable at repeated eigenvalues (so
        ``pow(0.5)`` is a differentiable ``sqrtm``) → MatrixView."""
        return MatrixView(
            FunctionCall(lambda x, _n=float(n): _spectral_matrix_function(x, "pow", _n), [self._expr], f"mat_pow_{n}")
        )

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

    def partials(self, **named_vars):
        """Bind Variables to names for element-wise partial-derivative access.

        ``A.matrix.partials(x=x, y=y).x`` → ``MatrixView`` of ``∂A/∂x``.
        """
        return _coords_dispatch(self, (), named_vars, positional_factory=NamedMatrixView)

    bind = partials  # synonym

    def coords(self, *args):
        """Name a matrix's axes for attribute element access::

            A.coords("x", "y")        # or A.coords(["x", "y"])

        returns a ``NamedMatrixView`` with element access via ``A.xy``, ``A.yy`` (``ScalarView``
        of ``A[..., 0, 1]``, etc.). To bind coordinates for derivatives use ``.bind(**vars)``.
        """
        return _coords_dispatch(self, args, {}, positional_factory=NamedMatrixView)

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        return self._rewrap(self._expr + _unwrap(other), other=other)

    def __radd__(self, other):
        return self._rewrap(_unwrap(other) + self._expr, other=other)

    def __sub__(self, other):
        return self._rewrap(self._expr - _unwrap(other), other=other)

    def __rsub__(self, other):
        return self._rewrap(_unwrap(other) - self._expr, other=other)

    def __mul__(self, other):
        return self._rewrap(self._expr * _unwrap(other), other=other)

    def __rmul__(self, other):
        return self._rewrap(_unwrap(other) * self._expr, other=other)

    def __neg__(self):
        return self._rewrap(-self._expr)

    def __pos__(self):
        """``+expr`` is the identity, as it is for a Python number: returns ``self`` unchanged.

        Defined so a term list can write a signed source symmetrically -- ``[-r, -r, +r, +r]`` reads as
        stoichiometry, and without this the ``+r`` legs raise ``TypeError`` while the ``-r`` legs work.
        Returns ``self`` rather than a ``1 * expr`` node, so it costs nothing in the graph."""
        return self

    def __truediv__(self, other):
        return self._rewrap(self._expr / _unwrap(other), other=other)

    def __rtruediv__(self, other):
        return self._rewrap(_unwrap(other) / self._expr, other=other)

    def __pow__(self, n):
        """Elementwise power. For matrix power use ``.pow(n)``."""
        return self._rewrap(self._expr**n)

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


class VoigtView(_DelegatesToPlaceholder):
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

    def _rewrap(self, new_expr, other=None) -> "VoigtView":
        """Wrap ``new_expr`` in the same view subclass as ``self``."""
        return VoigtView(new_expr)

    def integrate(self, **kwargs) -> "VoigtView":
        """Component-wise integral over the Voigt-packed tensor — VoigtView
        preserved; each Voigt slot is integrated independently.
        """
        return self._rewrap(self._expr.integrate(**kwargs))

    @property
    def stop_gradient(self) -> "VoigtView":
        """Block gradient flow component-wise, preserving the VoigtView type."""
        return self._rewrap(self._expr.stop_gradient)

    def d(self, v, scheme: str = "automatic_differentiation") -> "VoigtView":
        """Component-wise ``∂self/∂v`` — same view type."""
        return self._rewrap(self._expr.d(v, scheme=scheme))

    def d2(self, v, scheme: str = "automatic_differentiation") -> "VoigtView":
        return self._rewrap(self._expr.d2(v, scheme=scheme))

    def dd(self, v, w=None, scheme: str = "automatic_differentiation") -> "VoigtView":
        return self._rewrap(self._expr.dd(v, w, scheme=scheme))

    def partials(self, **named_vars):
        """Bind Variables to names for component-wise partial-derivative access.

        ``σ.voigt.partials(x=x, y=y).x`` → ``VoigtView`` of ``∂σ/∂x``.
        """
        return _coords_dispatch(self, (), named_vars)

    bind = partials  # synonym

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
        return self._rewrap(self._expr + _unwrap(other), other=other)

    def __radd__(self, other):
        return self._rewrap(_unwrap(other) + self._expr, other=other)

    def __sub__(self, other):
        return self._rewrap(self._expr - _unwrap(other), other=other)

    def __rsub__(self, other):
        return self._rewrap(_unwrap(other) - self._expr, other=other)

    def __mul__(self, other):
        return self._rewrap(self._expr * _unwrap(other), other=other)

    def __rmul__(self, other):
        return self._rewrap(_unwrap(other) * self._expr, other=other)

    def __neg__(self):
        return self._rewrap(-self._expr)

    def __pos__(self):
        """``+expr`` is the identity, as it is for a Python number: returns ``self`` unchanged.

        Defined so a term list can write a signed source symmetrically -- ``[-r, -r, +r, +r]`` reads as
        stoichiometry, and without this the ``+r`` legs raise ``TypeError`` while the ``-r`` legs work.
        Returns ``self`` rather than a ``1 * expr`` node, so it costs nothing in the graph."""
        return self

    def __truediv__(self, other):
        return self._rewrap(self._expr / _unwrap(other), other=other)

    def __rtruediv__(self, other):
        return self._rewrap(_unwrap(other) / self._expr, other=other)

    def __pow__(self, n):
        return self._rewrap(self._expr ** _unwrap(n))


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
# FieldView — FD-only partial derivatives on neural-operator field outputs
# ---------------------------------------------------------------------------

# Boundary slices for 2-D spatial fields (H, W, C) — works with any leading
# batch / time dimensions thanks to the Ellipsis prefix.
# Axis ordering follows equi_distant_rect with indexing="ij": axis 0 = x, 1 = y.
_BOUNDARY_SLICES = {
    "left": (..., slice(None, 1), slice(None), slice(None)),
    "right": (..., slice(-1, None), slice(None), slice(None)),
    "bottom": (..., slice(None), slice(None, 1), slice(None)),
    "top": (..., slice(None), slice(-1, None), slice(None)),
}


class FieldView(ScalarView):
    """Semantic view of a Placeholder as a full mesh-shaped field.

    Used when the wrapped expression is the output of a neural operator
    (Poseidon, FNO, etc.) where x, y, z, t are **not** inputs to the
    network — AD-based partials would be 0.  All derivatives returned by
    this view (via ``.bind(...)``) are evaluated with the structured-grid
    finite-difference scheme.

    The actual derivative attribute access lives on
    :class:`FieldViewWithPartials` — created by ``.bind(...)``.

    Example::

        u = NN(a_phys).field.bind(x=x_var, y=y_var, t=t_var)
        residual = u.t - 220.0**2 * (u.xx + u.yy) + u**3 - u
    """

    def partials(self, **named_vars):
        """Bind Variables to names; returns a :class:`FieldViewWithPartials`."""
        return FieldViewWithPartials(self._expr, named_vars)

    bind = partials


_WHOLE_LAPLACIAN_SUBSCHEMES = ("cotangent",)


def _reject_whole_laplacian_scheme(scheme: str, method: str) -> None:
    """A per-axis second derivative cannot carry a sub-scheme that computes the WHOLE Laplacian.

    ``finite_difference:cotangent`` returns ∇²u for every requested dimension, so ``u.d2(x, ...)``
    silently returns the full Laplacian and ``u.d2(x, ...) + u.d2(y, ...)`` silently returns **2∇²u**
    — a solve that converges to half the right answer without raising.
    """
    sub = str(scheme).split(":", 1)[1] if ":" in str(scheme) else ""
    if sub in _WHOLE_LAPLACIAN_SUBSCHEMES:
        raise ValueError(
            f"{method}(scheme={scheme!r}) is not meaningful: the {sub!r} sub-scheme computes the "
            f"WHOLE Laplacian, not a per-axis second derivative, so summing one per axis doubles it. "
            f"Write the Laplacian directly instead: u.laplacian(x, y, scheme={scheme!r})."
        )


class FieldViewWithPartials(ScalarView):
    """Field view with name → Variable bindings and FD-only derivatives.

    Resolves attribute access (``.x``, ``.xx``, ``.t``, ``.tt``, ``.xt``,
    ``.xxt`` …) by parsing the attribute as a sequence of registered names
    and emitting:

    * one :class:`TemporalDerivative` node per occurrence of a temporal
      Variable (``var.axis == "temporal"``);
    * one :class:`Jacobian` (single occurrence) or :class:`Hessian` (two
      occurrences of the same spatial Variable, ``trace=True``) per spatial
      Variable;
    * mixed spatial pairs (``.xy``) → one :class:`Hessian` with
      ``trace=False``;
    * higher-order chains alternate naturally (e.g. ``.xt`` → spatial-FD
      around an inner ``TemporalDerivative``).

    Optional keyword arguments to ``.bind`` of the form ``<name>_coords``
    (numpy / jnp arrays with the same shape as the field) are validated
    against the bound Variable's domain mesh and emit a warning if the
    distance exceeds ``1e-6``.  The validation is purely diagnostic — the
    FD computation always reads coordinates from the domain mesh.
    """

    _base_view = ScalarView

    def __init__(self, expr, coord_vars: dict) -> None:
        # Separate Variable bindings from optional *_coords ndarrays.
        import numpy as np

        spatial_coords: dict = {}
        vars_only: dict = {}
        for k, v in coord_vars.items():
            if k.endswith("_coords"):
                spatial_coords[k[: -len("_coords")]] = v
            else:
                vars_only[k] = v

        ScalarView.__init__(self, expr)
        object.__setattr__(self, "_coord_vars", dict(vars_only))
        object.__setattr__(self, "_spatial_coords", spatial_coords)

        # Diagnostic mismatch check against the domain mesh.
        if spatial_coords:
            self._validate_field_coords(np)

    def _validate_field_coords(self, np_mod) -> None:
        import warnings

        cv = object.__getattribute__(self, "_coord_vars")
        spatial_coords = object.__getattribute__(self, "_spatial_coords")

        domain = None
        for name, coord_arr in spatial_coords.items():
            var = cv.get(name)
            if var is None:
                continue
            d = getattr(var, "_domain", None)
            if d is not None:
                domain = d
                break

        if domain is None or getattr(domain, "mesh_connectivity", None) is None:
            return

        try:
            mesh_pts = np_mod.asarray(domain.mesh_connectivity["points"])
        except Exception:  # pragma: no cover - defensive
            return

        # Order user coords by axis index so the columns line up with
        # mesh_pts (which is N_mesh × ndim, columns ordered by dim index).
        ordered_names = sorted(
            spatial_coords,
            key=lambda nm: cv[nm].dim[0] if nm in cv and hasattr(cv[nm], "dim") else 0,
        )
        try:
            cols = [np_mod.asarray(spatial_coords[nm]).ravel() for nm in ordered_names]
        except Exception:  # pragma: no cover - defensive
            return

        n_user = cols[0].size
        n_mesh = mesh_pts.shape[0]
        if n_user != n_mesh:
            warnings.warn(
                f"FieldView coordinate mismatch: user provided {n_user} points but "
                f"domain mesh has {n_mesh}. FD stencils use the domain mesh — adjust "
                f"the domain resolution (e.g. equi_distant_rect(nx=H-1)) to match."
            )
            return

        ndim_user = len(cols)
        ndim_mesh = mesh_pts.shape[1]
        cmp_dim = min(ndim_user, ndim_mesh)
        diffs = np_mod.stack(cols[:cmp_dim], axis=-1) - mesh_pts[:, :cmp_dim]
        norms = np_mod.linalg.norm(diffs, axis=-1)
        mean = float(norms.mean())
        mx = float(norms.max())
        if mx > 1e-6:
            warnings.warn(
                f"FieldView coordinate mismatch: mean={mean:.3e}, max={mx:.3e}. "
                "FD stencils use the domain mesh coordinates — ensure your field's "
                "grid aligns with the domain (consider equi_distant_rect with the "
                "same range and resolution)."
            )

    # ------------------------------------------------------------------
    # Type preservation through arithmetic and .d(...)
    # ------------------------------------------------------------------

    def _rewrap(self, new_expr, other=None):
        cv = object.__getattribute__(self, "_coord_vars")
        sc = object.__getattribute__(self, "_spatial_coords")
        other_cv = getattr(other, "_coord_vars", None) if other is not None else None
        tie_tags = None
        if other_cv:
            conflict = any(name in cv and cv[name] is not var for name, var in other_cv.items())
            if conflict and _is_periodic_tie_combination(
                object.__getattribute__(self, "_expr"), getattr(other, "_expr", None)
            ):
                # jno.fdm periodic tie `u(A) - u(B)`: the BinaryOp discards the per-side views, so stash
                # the two region tags here (the only place they survive) for the classifier to read.
                tie_tags = (_tag_of_coord_vars(cv), _tag_of_coord_vars(other_cv))
            else:
                merged = dict(cv)
                for name, var in other_cv.items():
                    if name in merged and merged[name] is not var:
                        raise ValueError(
                            f"coord binding conflict for {name!r}: cannot combine "
                            f"two FieldView bindings that map {name!r} to different "
                            f"Variables (left={merged[name]!r}, right={var!r}). "
                            f"Re-bind the result explicitly with .bind(...) to resolve."
                        )
                    merged[name] = var
                cv = merged
        if tie_tags is None:
            # An operation on a stamped tie keeps the stamp: `u(A) - u(B) - g` discards the tie's
            # own view at the OUTER subtraction, and the region tags survive nowhere else. Whether
            # the offset form means anything is the reading solver's call -- jno.fem still refuses
            # it, loudly, because _tie_phase does not recognise the outer `-`.
            tie_tags = getattr(self, "_periodic_tie", None) or getattr(other, "_periodic_tie", None)
        new = FieldViewWithPartials.__new__(FieldViewWithPartials)
        ScalarView.__init__(new, new_expr)
        object.__setattr__(new, "_coord_vars", cv)
        object.__setattr__(new, "_spatial_coords", sc)
        if tie_tags is not None:
            object.__setattr__(new, "_periodic_tie", tie_tags)
        return new

    # ------------------------------------------------------------------
    # Method-style derivatives — default scheme
    # ------------------------------------------------------------------
    # A **nodal parameter** field (``domain.unknown()`` / ``jno.np.parameter(<fem symbol>)``) is a
    # discrete field on the mesh: autodiff w.r.t. a coordinate is meaningless, so its ``.d``/``.d2``
    # default to finite differences — ``ui.d2(x)`` is the FD second derivative, no ``scheme=`` needed.
    # For a neural-operator FieldView the default stays autodiff so the existing AD-on-FD guard keeps
    # forcing an explicit ``scheme="finite_difference"`` (see ``TestFieldViewADGuard``).

    def _default_deriv_scheme(self) -> str:
        expr = object.__getattribute__(self, "_expr")
        model = getattr(expr, "model", None)
        return "finite_difference" if getattr(model, "_fem_field", None) == "node" else "automatic_differentiation"

    def d(self, v, scheme: str | None = None) -> "ScalarView":
        """``∂self/∂v`` — finite differences by default for a nodal field (pass ``scheme=`` to override)."""
        return self._rewrap(self._expr.d(v, scheme=scheme or self._default_deriv_scheme()))

    def d2(self, v, scheme: str | None = None) -> "ScalarView":
        """``∂²self/∂v²`` — finite differences by default for a nodal field."""
        if scheme is not None:
            _reject_whole_laplacian_scheme(scheme, "d2")
        return self._rewrap(self._expr.d2(v, scheme=scheme or self._default_deriv_scheme()))

    def dd(self, v, w=None, scheme: str | None = None) -> "ScalarView":
        """Mixed second derivative ``∂²self/∂v∂w`` — finite differences by default for a nodal field."""
        if scheme is not None:
            _reject_whole_laplacian_scheme(scheme, "dd")
        return self._rewrap(self._expr.dd(v, w, scheme=scheme or self._default_deriv_scheme()))

    def laplacian(self, *variables, scheme: str | None = None) -> "ScalarView":
        """``∇²self`` — finite differences by default for a nodal field.

        Without this override the call falls through to :meth:`Placeholder.laplacian`, whose default is
        ``"automatic_differentiation"``. AD is meaningless on a discrete nodal field, and the AD Hessian
        branch has no ``points`` to differentiate at, so it failed with an opaque ``AttributeError``
        instead of doing the finite-difference thing ``.d`` / ``.d2`` / ``.dd`` already do here.
        """
        return self._rewrap(self._expr.laplacian(*variables, scheme=scheme or self._default_deriv_scheme()))

    # ------------------------------------------------------------------
    # Attribute access — derivative sequence parsing
    # ------------------------------------------------------------------

    @property
    def grid_shape(self) -> "tuple | None":
        """Spatial grid shape ``(H, W)`` (or ``(H, W, D)`` for 3-D) from the
        bound domain, or ``None`` if unavailable.

        Useful for determining boundary indices::

            H, W = u.grid_shape
            # left x-boundary is row 0; right is row H-1.
        """
        cv = object.__getattribute__(self, "_coord_vars")
        for var in cv.values():
            d = getattr(var, "_domain", None)
            if d is not None:
                gs = getattr(d, "_grid_shape", None)
                if gs is not None:
                    return gs
        return None

    def __getattr__(self, key: str):
        if key.startswith("_"):
            raise AttributeError(key)
        # Boundary slicing (2-D spatial): left/right → x-axis, bottom/top → y-axis.
        # Derivative-first is the correct order for Neumann/Robin BCs:
        #   u.x.right (FD on full field then slice) ✓
        #   u.right   (field value at boundary)     ✓
        # Returns ScalarView so that further FD chaining is blocked (calling u.right.x
        # would take FD of a 1-point boundary slice, which gives incorrect results).
        if key in _BOUNDARY_SLICES:
            return ScalarView(self._expr[_BOUNDARY_SLICES[key]])
        cv = object.__getattribute__(self, "_coord_vars")
        seq = _parse_partial_sequence(key, cv)
        if seq is not None:
            return self._build_partial(seq)
        return getattr(object.__getattribute__(self, "_expr"), key)

    def _build_partial(self, seq):
        """Build a derivative node from a parsed name sequence.

        The sequence is consumed left-to-right.  Spatial runs of length 1 or 2
        collapse into a single :class:`Jacobian` / :class:`Hessian` node;
        temporal occurrences each become a :class:`TemporalDerivative`.  This
        yields the natural chained form for mixed spatiotemporal derivatives
        (e.g. ``"xt"`` → ``Jacobian(TemporalDerivative(u, t), [x_var], "fd")``).

        Returns a :class:`FieldViewWithPartials` (via ``_rewrap``) so that
        further boundary slicing chains correctly, e.g.::

            u.x.right   # FieldViewWithPartials(Jacobian) → ScalarView of right slice ✓
            u.t.left    # FieldViewWithPartials(TD) → ScalarView of left slice ✓
        """
        from . import Hessian, Jacobian, TemporalDerivative

        cv = object.__getattribute__(self, "_coord_vars")
        result = self._expr

        i = 0
        n = len(seq)
        while i < n:
            name = seq[i]
            var = cv[name]
            is_temporal = getattr(var, "axis", None) == "temporal"
            if is_temporal:
                result = TemporalDerivative(result, var)
                i += 1
                continue

            # Spatial: greedily group with the next slot if it is also spatial
            # AND it is a (possibly different) spatial name; this collapses
            # `.xx`, `.yy`, `.xy` into a single Hessian per step.
            if i + 1 < n:
                next_name = seq[i + 1]
                next_var = cv[next_name]
                next_is_temporal = getattr(next_var, "axis", None) == "temporal"
                if not next_is_temporal:
                    if next_name == name:
                        # Repeated spatial: ∂²/∂var² via Hessian(trace=True)
                        result = Hessian(result, [var], "finite_difference", trace=True)
                    else:
                        # Mixed spatial: ∂²/∂var₁∂var₂ via Hessian(trace=False)
                        result = Hessian(result, [var, next_var], "finite_difference", trace=False)
                    i += 2
                    continue

            # Single spatial: ∂/∂var via Jacobian
            result = Jacobian(result, [var], "finite_difference")
            i += 1

        # Return FieldViewWithPartials (not plain ScalarView) so boundary attrs
        # like .right and .left chain correctly: u.x.right, u.t.left, etc.
        return self._rewrap(result)


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

        def _rewrap(self, new_expr, other=None):
            """Preserve the ``coord_vars`` binding through arithmetic / ``.d``.

            So ``(u - source).x`` works just like ``u.x``: the result of
            arithmetic is still a ``Named<View>WithPartials`` and inherits
            the original ``{name: Variable}`` registration.

            If ``other`` is itself a ``Named*WithPartials``, its bindings are
            merged into ``self``'s — but only if every shared name maps to the
            *same* Variable. A conflicting name (e.g. ``u.bind(x=x1) + v.bind(x=x2)``)
            raises ``ValueError`` rather than silently picking one side, since
            ``(u + v).x`` would otherwise depend on operand order.
            """
            cv = object.__getattribute__(self, "_coord_vars")
            other_cv = getattr(other, "_coord_vars", None) if other is not None else None
            tie_tags = None
            if other_cv:
                conflict = any(name in cv and cv[name] is not var for name, var in other_cv.items())
                if conflict and _is_periodic_tie_combination(
                    object.__getattribute__(self, "_expr"), getattr(other, "_expr", None)
                ):
                    # FEM periodic tie `u(A) - u(B)`: a constraint we never differentiate, so the
                    # `.x`-ambiguity guard does not apply. The BinaryOp discards the per-side views, so
                    # stash the two region tags here (the only place they survive) for jno.fem to read.
                    tie_tags = (_tag_of_coord_vars(cv), _tag_of_coord_vars(other_cv))
                else:
                    merged = dict(cv)
                    for name, var in other_cv.items():
                        if name in merged and merged[name] is not var:
                            raise ValueError(
                                f"coord binding conflict for {name!r}: cannot combine "
                                f"two named views that map {name!r} to different "
                                f"Variables (left={merged[name]!r}, right={var!r}). "
                                f"Re-bind the result explicitly with .bind(...) to resolve."
                            )
                        merged[name] = var
                    cv = merged
            if tie_tags is None:
                # see the note in FieldViewWithPartials._rewrap: an operation on a stamped tie keeps
                # the stamp, because the outer op discards the tie's own view.
                tie_tags = getattr(self, "_periodic_tie", None) or getattr(other, "_periodic_tie", None)
            res = type(self)(new_expr, cv)
            if tie_tags is not None:
                object.__setattr__(res, "_periodic_tie", tie_tags)
            return res

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

        # No no-args ``grad()`` / ``laplacian()`` convenience methods —
        # tutorials write the explicit form ``u.xx + u.yy`` etc., which reads
        # like the math and is unambiguous about which axes participate.

        @property
        def fd(self):
            """Finite-difference partial-derivative namespace.

            ``u.fd.x``, ``u.fd.xx``, ``u.fd.xy`` etc. mirror the attribute
            syntax of ``u.x`` / ``u.xx`` / ``u.xy`` but evaluate each partial
            via the finite-difference scheme instead of automatic
            differentiation.
            """
            return _SchemeProxy(self, "finite_difference")

    NamedWithPartials.__name__ = f"Named{view_cls.__name__}WithPartials"
    NamedWithPartials.__qualname__ = NamedWithPartials.__name__
    return NamedWithPartials


NamedScalarViewWithPartials = _make_named_with_partials_cls(ScalarView)
NamedVectorViewWithPartials = _make_named_with_partials_cls(VectorView)
NamedComplexViewWithPartials = _make_named_with_partials_cls(ComplexView)
NamedComplexVectorViewWithPartials = _make_named_with_partials_cls(ComplexVectorView)
NamedMatrixViewWithPartials = _make_named_with_partials_cls(MatrixView)
NamedVoigtViewWithPartials = _make_named_with_partials_cls(VoigtView)


# Populate the tuple now that all classes exist (used by _unwrap()).
_VIEW_TYPES = (ScalarView, VectorView, ComplexView, ComplexVectorView, MatrixView, VoigtView, FieldView)

# Dispatch table used by `_coords_dispatch` to pick the Named<View>WithPartials
# wrapper for each base view type.
_NAMED_PARTIALS_CLS_FOR = {
    ScalarView: NamedScalarViewWithPartials,
    VectorView: NamedVectorViewWithPartials,
    ComplexView: NamedComplexViewWithPartials,
    ComplexVectorView: NamedComplexVectorViewWithPartials,
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
    "NamedComplexVectorViewWithPartials",
    "NamedMatrixViewWithPartials",
    "NamedVoigtViewWithPartials",
    "FieldView",
    "FieldViewWithPartials",
]
