"""
Tracing system for pino operations.

Variables are placeholders that get filled during solve iterations.
Operations trace computations and return callable placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp

from ..architectures.lora import LoRAWrapper, _normalize_wrappers
from ..tuner import Arch, ArchSpace
from ..utils.adaptive import LearningRateSchedule
from ..utils.iree import IREEModel
from ..utils.logger import get_logger

__all__ = [
    "Placeholder",
    "FunctionCall",
    "Choice",
    "Literal",
    "ConstantNamespace",
    "Constant",
    "Variable",
    "TensorTag",
    "BinaryOp",
    "Tracker",
    "Model",
    "TunableModule",
    "TunableModuleCall",
    "ModelCall",
    "OperationDef",
    "OperationCall",
    "Hessian",
    "Integral",
    "IntegralTime",
    "Jacobian",
    "TemporalDerivative",
    "NetworkGradient",
    "Noise",
    "collect_operations",
    "collect_tags",
    "get_primary_tag",
    "dump_tree",
    "cse",
    "TestFunction",
    "Assembly",
    "GroupedAssembly",
    "TrialFunction",
    "FemLinearSystem",
    "FemResidualOperator",
    "StateField",
    # Typed semantic views (re-exported at the bottom of this module)
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
    "FieldView",
    "FieldViewWithPartials",
]

# Global counter for unique operation IDs
_operation_counter = 0


def _next_op_id() -> int:
    global _operation_counter
    _operation_counter += 1
    return _operation_counter


def _contains_node_type_local(node, cls) -> bool:
    if isinstance(node, cls):
        return True

    for attr in ("left", "right", "target", "expr", "operation", "model"):
        child = getattr(node, attr, None)
        if child is not None and _contains_node_type_local(child, cls):
            return True

    for attr in ("args", "variables", "options"):
        vals = getattr(node, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if _contains_node_type_local(vv, cls):
                        return True
            else:
                if _contains_node_type_local(v, cls):
                    return True
    return False


def _contains_fd_partial(node) -> bool:
    """True if ``node``'s subtree contains a finite-difference partial.

    FieldView emits FD ``Jacobian`` / ``Hessian`` (``scheme`` starting with
    ``"finite_difference"``) and ``TemporalDerivative`` nodes on a grid output
    whose coordinates are not network inputs.  Re-applying an automatic-
    differentiation operator on top of these silently returns zero, so callers
    detect them and raise instead.
    """
    if isinstance(node, TemporalDerivative):
        return True
    if isinstance(node, (Jacobian, Hessian)) and str(getattr(node, "scheme", "")).startswith("finite_difference"):
        return True
    for attr in ("left", "right", "target", "expr", "operation", "model"):
        child = getattr(node, attr, None)
        if child is not None and _contains_fd_partial(child):
            return True
    for attr in ("args", "variables", "options"):
        vals = getattr(node, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                if any(_contains_fd_partial(vv) for vv in v):
                    return True
            elif _contains_fd_partial(v):
                return True
    return False


def _guard_ad_on_fd(target, scheme: str) -> None:
    """Raise if an automatic-differentiation derivative is requested over a
    FieldView finite-difference partial.

    The field is a grid output (its coordinates are not network inputs), so AD
    over an FD partial silently evaluates to zero — a wrong answer that looks
    plausible.  Requesting an explicit ``finite_difference`` scheme is allowed
    (that is how FieldView builds higher-order partials internally).
    """
    if str(scheme).startswith("finite_difference"):
        return
    if _contains_fd_partial(target):
        raise ValueError(
            "Cannot apply an automatic-differentiation derivative to an expression built "
            "from FieldView finite-difference partials: the field is a grid output (its "
            "coordinates are not network inputs), so AD silently returns 0. Use the "
            "FieldView FD API for higher-order derivatives instead — e.g. `u.xx + u.yy` "
            "rather than `jno.np.vector(u.x, u.y).div(x, y)`, and `u.xx` rather than "
            "`u.x.d(x)`."
        )


def _mark_weak(node, root_id=None):
    if root_id is None:
        root_id = getattr(node, "op_id", id(node))
    setattr(node, "_is_weak_expr", True)
    setattr(node, "_weak_root_id", root_id)
    return node


def _propagate_weak(node, *children):
    # 1) If any child is already marked weak, propagate the same weak root id.
    weak_children = [c for c in children if getattr(c, "_is_weak_expr", False)]
    if weak_children:
        root_id = getattr(weak_children[0], "_weak_root_id", None)
        return _mark_weak(node, root_id=root_id)

    # 2) Otherwise seed the weak marker if this newly built node now contains
    #    TestFunction / TrialFunction / StateField anywhere underneath.
    if _looks_like_weak_expression(node):
        return _mark_weak(node)

    return node


def _looks_like_weak_expression(node) -> bool:
    return (
        _contains_node_type_local(node, TestFunction)
        or _contains_node_type_local(node, TrialFunction)
        or _contains_node_type_local(node, StateField)
    )


class Placeholder:
    """Base node for the traced DSL graph.

    Placeholders behave like symbolic tensors: arithmetic and composition
    operators create new traced nodes instead of executing eagerly. Calling a
    placeholder (`u(x)`) auto-wraps it in an `OperationDef` so it can be reused
    with different inputs. Concrete values are only produced when evaluated by
    the solver/visualizer.

    Note: ``__eq__`` and ``__hash__`` use object identity so that Placeholder
    instances can safely be stored in sets and used as dict keys.  For
    element-wise symbolic equality comparisons use ``Placeholder.equal(other)``.
    """

    # -- identity-based equality so Placeholders work in sets/dicts -----------
    def __eq__(self, other) -> bool:
        return self is other

    def __ne__(self, other) -> bool:
        return self is not other

    def __hash__(self) -> int:
        return id(self)

    # -- symbolic comparison operators (return traced FunctionCall nodes) ------
    def equal(self, other) -> FunctionCall:
        """Element-wise symbolic equality (traced, not Python ``==``)."""
        return FunctionCall(jnp.equal, [self, other])

    def not_equal(self, other) -> FunctionCall:
        """Element-wise symbolic inequality (traced, not Python ``!=``)."""
        return FunctionCall(jnp.not_equal, [self, other])

    def __gt__(self, other) -> FunctionCall:
        return FunctionCall(jnp.greater, [self, other])

    def __lt__(self, other) -> FunctionCall:
        return FunctionCall(jnp.less, [self, other])

    def __ge__(self, other) -> FunctionCall:
        return FunctionCall(jnp.greater_equal, [self, other])

    def __le__(self, other) -> FunctionCall:
        return FunctionCall(jnp.less_equal, [self, other])

    def _wrap(self, other) -> Placeholder:
        """Wrap non-Placeholder types.

        Typed semantic views (``ScalarView``, ``MatrixView``, …) expose their
        underlying Placeholder via ``._expr`` — unwrap those so mixed-direction
        arithmetic like ``placeholder + u.scalar`` works without surprise.
        """
        if isinstance(other, Placeholder):
            return other
        inner = getattr(other, "_expr", None)
        if isinstance(inner, Placeholder):
            return inner
        return Literal(other)

    def name(self, label: str) -> "Placeholder":
        """Tag this expression with a human-readable label for logs and W&B.

        Returns *self* so it can be used inline::

            pde_loss = (k * (u.dd(x) + u.dd(y)) + 1.0).mse.name("pde")
            bc_loss  = u_bc.mse.name("bc")
        """
        self._user_name = label
        return self

    # Graph-time physical unit (dimensional metadata). ``None`` = undeclared.
    # Only ever set explicitly here on a leaf; inferred units for intermediate
    # nodes are computed non-destructively by ``jno.units.check`` and never
    # written back onto the graph.
    _unit: "object | None" = None

    def unit(self, spec: "str") -> "Placeholder":
        """Declare the physical unit of this expression for dimensional analysis.

        Attaches graph-time metadata used by :func:`jno.units.check` to audit
        that a PDE is dimensionally consistent.  Returns *self* so it chains
        inline::

            x = x.unit("m")
            u = net(x, t).unit("K")

        ``spec`` is a unit string such as ``"m"``, ``"m/s"``, ``"kg/m^3"`` or
        ``"Pa"`` (see :meth:`jno.units.Unit.parse`).
        """
        from .units import Unit

        self._unit = Unit.parse(spec)
        return self

    # Graph-time characteristic magnitude (the *scale* that complements the
    # *dimension* set by ``.unit``). ``None`` = undeclared. Like ``_unit`` it is
    # only ever set explicitly on a leaf; intermediate magnitudes are computed
    # non-destructively by ``jno.units.nondimensionalize`` and never written back.
    _scale: "float | None" = None

    def scale(self, value: float) -> "Placeholder":
        """Declare the characteristic *magnitude* of this variable/field.

        Units give the *dimension* (``.unit("m")`` → a length); the scale gives
        the *magnitude* in that unit (``.scale(0.1)`` → the characteristic length
        is 0.1 m).  The two are orthogonal and used together by
        :func:`jno.units.nondimensionalize` to derive dimensionless groups and
        rescale a problem to ``O(1)``.  Returns *self* so it chains inline::

            x = x.unit("m").scale(0.1)        # L  = 0.1 m
            u = net(x, t).unit("K").scale(50.0)
            alpha = alpha.unit("m^2/s").scale(1e-5)
        """
        self._scale = float(value)
        return self

    @property
    def stop_gradient(self) -> "FunctionCall":
        """Block gradient flow through this expression.

        Identity in the forward pass, zero in the backward pass — property form
        of :func:`jno.fn.stop_gradient`. Decouple cooperating models so each
        interaction term only updates *its own* parameters::

            L_int_phy = (u_phy - u_syn.stop_gradient).mse
            L_int_syn = (u_syn - u_phy.stop_gradient).mse

        Or treat a costly quantity (e.g. a parameter Jacobian) as a constant
        when penalising it, without differentiating through its computation::

            loss = (u.grad(net).stop_gradient ** 2).mean()
        """
        return FunctionCall(jax.lax.stop_gradient, [self], name="stop_gradient")

    def __add__(self, other) -> BinaryOp:
        if isinstance(other, _VIEW_TYPES):
            return NotImplemented
        return BinaryOp("+", self, self._wrap(other))

    def __radd__(self, other) -> BinaryOp:
        return BinaryOp("+", self._wrap(other), self)

    def __sub__(self, other) -> BinaryOp:
        if isinstance(other, _VIEW_TYPES):
            return NotImplemented
        return BinaryOp("-", self, self._wrap(other))

    def __rsub__(self, other) -> BinaryOp:
        return BinaryOp("-", self._wrap(other), self)

    def __mul__(self, other) -> BinaryOp:
        if isinstance(other, _VIEW_TYPES):
            return NotImplemented
        return BinaryOp("*", self, self._wrap(other))

    def __rmul__(self, other) -> BinaryOp:
        return BinaryOp("*", self._wrap(other), self)

    def __truediv__(self, other) -> BinaryOp:
        if isinstance(other, _VIEW_TYPES):
            return NotImplemented
        return BinaryOp("/", self, self._wrap(other))

    def __rtruediv__(self, other) -> BinaryOp:
        return BinaryOp("/", self._wrap(other), self)

    def __neg__(self) -> BinaryOp:
        return BinaryOp("*", Literal(-1.0), self)

    def __pow__(self, other) -> BinaryOp:
        return BinaryOp("**", self, self._wrap(other))

    def __rpow__(self, other) -> BinaryOp:
        return BinaryOp("**", self._wrap(other), self)

    def __getitem__(self, key) -> FunctionCall:
        if not isinstance(key, tuple):
            key = (key,)
        concrete_key = tuple(None if k is None else k for k in key)
        fc = FunctionCall(lambda x, k=concrete_key: x[k], [self], name="getitem")
        # Record the key so consumers (e.g. the FEM driver) can recover a
        # component index from `u[..., i]` rather than it being hidden in the closure.
        fc.getitem_key = concrete_key
        return fc

    def __call__(self, *args):
        """Call this expression with different variables (auto-wraps in operation)."""
        # Auto-wrap this expression in an OperationDef if not already one
        if not hasattr(self, "_auto_op"):
            self._auto_op = OperationDef(self)
        return self._auto_op(*args)

    def __matmul__(self, other) -> FunctionCall:
        """Matrix multiplication: self @ other"""
        other = self._wrap(other)

        # symbolic path
        if isinstance(other, Placeholder):
            return FunctionCall(lambda a, b: a @ b, [self, other])

        # eager path (other is ndarray)
        return FunctionCall(lambda a: a @ other, [self])

    def __rmatmul__(self, other) -> FunctionCall:
        """Matrix multiplication: other @ self"""
        other = self._wrap(other)

        if isinstance(other, Placeholder):
            return FunctionCall(lambda a, b: a @ b, [other, self])

        return FunctionCall(lambda b: other @ b, [self])

    def reshape(self, *shape) -> FunctionCall:
        """Reshape this placeholder to a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return FunctionCall(lambda x, s=shape: x.reshape(s), [self], name="reshape")

    def assemble(self, domain=None, target=None, **kwargs):
        """Unified assembly entry point.

        If domain is omitted, try to infer it from Variables/TestFunction/etc.
        target=None lets weak_form.py infer the steady solver route in Phase 1.
        """
        from ..utils.solver.weak_form import assemble_weak_form

        return assemble_weak_form(domain, self, target=target, **kwargs)

    def print(
        self,
        what: Union[str, Callable[[jnp.ndarray], Any]] = "shape",
        label: Optional[str] = None,
    ):
        """Emit runtime debug info for this placeholder and pass it through.

        Args:
            what: Either a preset string or a callable:
                - ``"shape"``: print array shape only.
                - ``"stats"``: print shape + min/max/mean/std.
                - ``"value"``: print full value.
                - ``"all"``: print shape + stats + value.
                - ``callable``: called as ``what(x)`` and the result is printed.
            label: Optional message prefix in the log output.

        Returns:
            A traced node that prints at runtime and returns the unchanged input.

        Example:
            ``constraints.append((NN(*inpt) - u).print("stats", "residual"))``
            ``constraints.append((NN(*inpt) - u).print(lambda x: x.shape, "residual"))``
        """
        prefix = "Placeholder" if label is None else label
        allowed = {"shape", "stats", "value", "all"}

        if callable(what):

            def _debug_print_custom(x, _fn=what, _prefix=prefix):
                arr = jnp.asarray(x)
                value = _fn(arr)
                jax.debug.print("{p}: {v}", p=_prefix, v=value)
                return x

            return FunctionCall(_debug_print_custom, [self], name="print")

        mode = str(what).lower().strip()
        if mode not in allowed:
            raise ValueError(f"Unsupported print mode '{what}'. Use one of {sorted(allowed)} or pass a callable.")

        def _debug_print(x, _mode=mode, _prefix=prefix):
            arr = jnp.asarray(x)

            if _mode in ("shape", "all"):
                jax.debug.print("{p}: shape={s}", p=_prefix, s=arr.shape)

            if _mode in ("stats", "all"):
                arr_f = arr.astype(jnp.float32)
                jax.debug.print(
                    "{p}: min={mn}, max={mx}, mean={mu}, std={sd}",
                    p=_prefix,
                    mn=jnp.min(arr_f),
                    mx=jnp.max(arr_f),
                    mu=jnp.mean(arr_f),
                    sd=jnp.std(arr_f),
                )

            if _mode in ("value", "all"):
                jax.debug.print("{p}: value={v}", p=_prefix, v=arr)

            return x

        return FunctionCall(_debug_print, [self], name="print")

    def tracker(self, interval: int = 1, reduce=None) -> "Tracker":
        """Mark this expression as a tracked metric.

        Trackers are evaluated during training logs but do not contribute to
        the optimization loss.

        Args:
            interval: Evaluate every ``interval`` epochs (must be >= 1).
            reduce: Optional callable applied to the numpy array after device
                transfer to produce a scalar for W&B and the progress line.
                Defaults to ``np.mean`` for non-scalar outputs.  Example::

                    u.d(x).tracker(100, reduce=lambda v: v.max())

        Returns:
            ``Tracker`` wrapper for this expression.
        """
        if not isinstance(interval, int) or interval < 1:
            raise ValueError(f"interval must be an integer >= 1, got {interval}")
        return Tracker(self, interval, reduce=reduce)

    @property
    def shape(self) -> FunctionCall:
        return FunctionCall(lambda x: jnp.ones(x.shape, dtype="bool"), [self], "shape", True)

    @property
    def mean(self) -> FunctionCall:
        return FunctionCall(lambda x: jnp.squeeze(x.mean()), [self], "mean", True)

    @property
    def sum(self) -> FunctionCall:
        return FunctionCall(lambda x: jnp.squeeze(jnp.sum(x)), [self], "sum", True)

    @property
    def min(self) -> FunctionCall:
        return FunctionCall(lambda x: jnp.squeeze(jnp.min(x)), [self], "min", True)

    @property
    def max(self) -> FunctionCall:
        return FunctionCall(lambda x: jnp.squeeze(jnp.max(x)), [self], "max", True)

    @property
    def std(self) -> FunctionCall:
        return FunctionCall(lambda x: jnp.squeeze(jnp.std(x)), [self], "std", True)

    @property
    def mse(self) -> FunctionCall:
        def fn(x):
            return jnp.squeeze(jnp.mean(jnp.square(x)))

        return FunctionCall(fn, [self], "mse", True)

    @property
    def mae(self) -> FunctionCall:
        return FunctionCall(lambda x: jnp.squeeze(jnp.mean(jnp.abs(x))), [self], "mae", True)

    @property
    def T(self) -> FunctionCall:
        return FunctionCall(lambda x: x.T, [self], "transpose", True)

    # ------------------------------------------------------------------
    # Native complex-dtype helpers (work on jnp.complex64/complex128)
    # ------------------------------------------------------------------

    @property
    def real(self) -> FunctionCall:
        """Real part via ``jnp.real`` (works on native complex arrays)."""
        return FunctionCall(jnp.real, [self], "real")

    @property
    def imag(self) -> FunctionCall:
        """Imaginary part via ``jnp.imag``."""
        return FunctionCall(jnp.imag, [self], "imag")

    # ------------------------------------------------------------------
    # Typed semantic views — see jno.trace.views for the full API
    # ------------------------------------------------------------------

    @property
    def scalar(self):
        """Scalar view — typed scalar ops and cross-type ``*`` dispatch."""
        from .views import ScalarView

        return ScalarView(self)

    @property
    def vector(self):
        """Vector field view — ``.div(*v)``, ``.curl(*v)``, ``.norm()``, ``.dot(other)``,
        ``.cross(other)``, ``.normalize()``, ``.outer(other)``, ``v @ A``."""
        from .views import VectorView

        return VectorView(self)

    @property
    def complex(self):
        """Complex field view (last dim = 2, ``[re, im]``) — ``.real``, ``.imag``,
        ``.abs``, ``.angle``, ``.conj``, ``.mul(other)``, ``.to_native()``."""
        from .views import ComplexView

        return ComplexView(self)

    @property
    def matrix(self):
        """Full matrix view (``[..., n, m]``) — ``.trace()``, ``.det()``, ``.inv()``,
        ``.eigvals()``, ``.sym()``, ``.skew()``, ``.log()``, ``.exp()``, ``.pow(n)``,
        plus packed constructors ``.from_upper_tri()`` / ``.from_lower_tri()`` /
        ``.from_flat(n, m)`` / ``.from_diag()`` and ``.coords([names])``."""
        from .views import MatrixView

        return MatrixView(self)

    @property
    def voigt(self):
        """Voigt symmetric-tensor view (last dim = 3 for 2-D, 6 for 3-D) —
        ``.von_mises()``, ``.trace()``, ``.hydrostatic()``, ``.deviatoric()``,
        ``.principal()``, ``.invariants()``, ``.max_shear()``, ``.to_full()``."""
        from .views import VoigtView

        return VoigtView(self)

    @property
    def field(self):
        """Field view for neural-operator outputs — FD-only partial derivatives.

        Use ``.field.bind(x=x_var, y=y_var, t=t_var)`` when the underlying
        ``Placeholder`` is the full mesh-shaped output of a neural operator
        (Poseidon, FNO, etc.) and ``x``/``y``/``t`` are NOT inputs to the
        network. All derivatives via the returned view are evaluated with
        the structured-grid finite-difference scheme.
        """
        from .views import FieldView

        return FieldView(self)

    # ------------------------------------------------------------------
    # Differential operators — method-style API
    # ------------------------------------------------------------------

    def d(self, variable: "Variable", scheme: str = "automatic_differentiation") -> "Jacobian":
        """Return ∂self/∂variable — shorthand for ``jnn.grad(self, variable)``.

        Can be chained for higher-order derivatives::

            u_xx = u.d(x).d(x)   # ∂²u/∂x²
            u_xy = u.d(x).d(y)   # ∂²u/∂x∂y

        Args:
            variable: The Variable to differentiate with respect to.
            scheme: First-order scheme string.

                * ``"automatic_differentiation"`` (default) — uses the global
                  AD mode set via :func:`jno.setup` (see :mod:`jno.utils.ad_mode`).
                * ``"automatic_differentiation:forward"`` — ``jax.jacfwd``.
                * ``"automatic_differentiation:reverse"`` — ``jax.jacrev``.
                * ``"finite_difference"`` (optional sub-schemes:
                  ``":lsq"`` / ``":uniform"`` / ``":inverse_distance"``).
        """
        _guard_ad_on_fd(self, scheme)
        return Jacobian(self, [variable], scheme)

    def diff(self, variable: "Variable", scheme: str = "automatic_differentiation") -> "Jacobian":
        """Alias for :meth:`d`."""
        _guard_ad_on_fd(self, scheme)
        return Jacobian(self, [variable], scheme)

    def d2(self, variable: "Variable", scheme: str = "automatic_differentiation") -> "Hessian":
        """Return ∂²self/∂variable² — shorthand for ``jnn.hessian(self, [variable])``.

        Args:
            variable: The Variable to differentiate with respect to.
            scheme: Second-order scheme string — see :meth:`laplacian`.
        """
        _guard_ad_on_fd(self, scheme)
        return Hessian(self, [variable], scheme, trace=True)

    def dd(self, variable: "Variable", scheme: str = "automatic_differentiation") -> "Hessian":
        """Alias for :meth:`d2`."""
        _guard_ad_on_fd(self, scheme)
        return Hessian(self, [variable], scheme, trace=True)

    def laplacian(
        self,
        *variables: "Variable",
        scheme: str = "automatic_differentiation",
    ) -> "Hessian":
        """Return ∇²self — shorthand for ``jnn.laplacian(self, list(variables))``.

        Example::

            lap_u = u.laplacian(x, y)   # ∂²u/∂x² + ∂²u/∂y²

        Args:
            *variables: Variables to include in the Laplacian.
            scheme: Second-order scheme string.

                * ``"automatic_differentiation"`` (default) — uses the global
                  Hessian mode set via :func:`jno.setup`
                  (see :mod:`jno.utils.ad_mode`).
                * ``"automatic_differentiation:fwd-over-rev"`` —
                  ``jacfwd(jacrev(f))`` (equivalent to historical ``jax.hessian``).
                * ``"automatic_differentiation:fwd-over-fwd"`` —
                  ``jacfwd(jacfwd(f))``.
                * ``"automatic_differentiation:rev-over-rev"`` —
                  ``jacrev(jacrev(f))``.
                * ``"automatic_differentiation:rev-over-fwd"`` —
                  ``jacrev(jacfwd(f))``.
                * ``"finite_difference"`` (optional sub-schemes:
                  ``":cotangent"`` (2-D), ``":lsq"``).
        """
        _guard_ad_on_fd(self, scheme)
        return Hessian(self, list(variables) if variables else None, scheme, trace=True)

    def hessian(
        self,
        *variables: "Variable",
        scheme: str = "automatic_differentiation",
    ) -> "Hessian":
        """Return the full Hessian matrix of self w.r.t. *variables*.

        Example::

            H = u.hessian(x, y)   # 2×2 Hessian matrix per point

        Args:
            *variables: Variables for the Hessian.
            scheme: Second-order scheme string — see :meth:`laplacian`.
        """
        _guard_ad_on_fd(self, scheme)
        return Hessian(self, list(variables), scheme, trace=False)

    # ------------------------------------------------------------------
    # Integration — method-style API
    # ------------------------------------------------------------------

    def integrate(self, var: "Variable | None" = None) -> "Integral | IntegralTime":
        """Integrate this expression over its mesh domain region or over time.

        **Spatial scalar integral** (``var=None``, default):
        The region (boundary vs volume) is auto-detected from the Variable
        tags inside the expression.  The expression is evaluated at all mesh
        nodes and reduced to a scalar via a weighted sum.

        **Vectorized spatial integral** (``var=x`` — the outer/collocation variable):
        When the expression contains two distinct Variable objects from the
        same mesh (e.g. an outer collocation variable ``x`` and an inner
        dummy ``t``), pass the outer one as ``var``.  The integral is then
        evaluated for every collocation point via ``jax.vmap``, returning an
        ``(N, 1)`` array — a function of the outer variable.  This enables
        non-separable Fredholm-type kernels without any special flag on
        ``domain.variable()``::

            x, _ = domain.variable("interior")   # outer (collocation)
            t, _ = domain.variable("interior")   # inner (dummy) — no flag needed
            integral = (K(x, t) * net(t)).integrate(var=x)

        **Temporal integral** (``var=t`` — the temporal Variable):
        When a temporal Variable (``axis='temporal'``) is passed, the integral
        is computed over the time window visible in the current forward pass
        using the trapezoidal rule::

            x, t = domain.variable("interior")
            loss = (u_net(x, t).integrate(t) - target).mse

        Requires ``min_consecutive >= 2`` in ``solve()`` (trapezoidal integration
        over a single point is identically zero — a silent wrong answer).
        Chain with spatial integration for space-time integrals::

            space_time_integral = u_net(x, t).integrate().integrate(t)
        """
        if var is not None and getattr(var, "axis", None) == "temporal":
            return IntegralTime(self, time_var=var)
        return Integral(self, integration_var=var)

    def grad(self, *args):
        """Gradient operator with two forms (dispatched by argument type).

        **1. Spatial gradient** — ``grad(x, y, [z])`` with ``Variable`` arguments
        returns a :class:`VectorView` of the spatial gradient
        ``[∂self/∂x, ∂self/∂y, ...]``. Use inside PDE residuals — e.g.
        ``flux = kappa * u.grad(x, y)`` builds a flux vector that chains
        into ``.div(x, y)``.

        **2. Parameter gradient** — ``grad(model)`` with a single :class:`Model`
        argument returns a :class:`NetworkGradient` ``(N, P)`` array — N
        spatial points × P selected trainable parameters (flattened). For
        multi-dimensional output ``(N, D)`` the result is ``(N, D, P)``.

        To restrict to a subset of parameters, call ``model.mask(bool_pytree)``
        first using a boolean pytree built with :func:`equinox.tree_at`.  Since
        ``mask()`` returns the model itself, you can write the selection inline::

            import equinox as eqx, jax

            # All parameters — Neural Tangent Kernel
            J = crux.eval([u.grad(net)])[0]             # (N, P_total)
            K = J @ J.T                                  # (N, N)

            # Subset — only the output-layer weight matrix
            all_false = jax.tree_util.tree_map(lambda _: False, net.module)
            mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)
            J_w = crux.eval([u.grad(net.mask(mask))])[0]  # (N, P_weight)
        """
        if not args:
            raise TypeError("grad() requires at least one Variable (spatial gradient) or a Model (parameter gradient).")
        if len(args) == 1 and isinstance(args[0], Model):
            # Parameter-gradient form (existing behaviour).
            model = args[0]
            selector = getattr(model, "_param_mask", None)
            return NetworkGradient(self, model, selector=selector)
        # Spatial-gradient form (new): one or more Variables → VectorView.
        from ..jnp_ops import concat
        from .views import VectorView

        return VectorView(concat([self.d(v) for v in args]))


class Literal(Placeholder):
    """Concrete scalar/array embedded in the trace (no trainable params)."""

    def __init__(self, value):
        self.value = jnp.asarray(value)

    def __repr__(self):
        return f"Literal({self.value})"


class FunctionCall(Placeholder):
    """Call to a pure function over traced args."""

    def __init__(
        self,
        fn: Callable,
        args: Union[list, tuple],
        name: str | None = None,
        reduces_axis: Optional[int] = None,
        kwargs: Optional[Dict] = None,
    ):
        self.fn = fn
        self.args = args if isinstance(args, (list, tuple)) else [args]
        self._name = name
        self.reduces_axis = reduces_axis
        self.kwargs = kwargs
        weak_children = [a for a in self.args if isinstance(a, Placeholder)]
        if not self.reduces_axis:
            _propagate_weak(self, *weak_children)

    def __repr__(self):
        name = self._name or getattr(self.fn, "__name__", str(self.fn))
        args_str = ", ".join(str(a) for a in self.args)
        return f"{name}({args_str})"

    def copy_with_args(self, new_args):
        """Create a new instance with different args."""
        return FunctionCall(
            fn=self.fn,
            args=new_args,
            name=self._name,
            reduces_axis=self.reduces_axis,
            kwargs=self.kwargs,
        )

    def __call__(self, args):
        """Return a new FunctionCall with the given args."""
        return self.copy_with_args([args])


class Choice(Placeholder):
    """Tunable categorical choice between multiple traced expressions."""

    def __init__(self, options: Union[list, tuple], name: str | None = None, default: int = 0):
        if not isinstance(options, (list, tuple)) or len(options) == 0:
            raise ValueError("Choice requires a non-empty list/tuple of options")

        self.options = [opt if isinstance(opt, Placeholder) else Literal(opt) for opt in options]
        self.op_id = _next_op_id()
        self.name = name or f"choice_{self.op_id}"
        self.selected = int(default)

        if not (0 <= self.selected < len(self.options)):
            raise ValueError(f"Choice default index {self.selected} out of range for {len(self.options)} option(s)")

    def select(self, index: int) -> "Choice":
        idx = int(index)
        if not (0 <= idx < len(self.options)):
            raise ValueError(f"Choice index {idx} out of range for {len(self.options)} option(s)")
        self.selected = idx
        return self

    def __repr__(self):
        return f"Choice(name={self.name!r}, selected={self.selected}, n={len(self.options)})"


class ConstantNamespace:
    """Wrapper that allows attribute access to constants (P.k syntax).

    Supports loading from:
    - dict: Direct dictionary of key-value pairs
    - str/Path: File path to .json, .yaml, .yml, .toml, .pkl, .pickle files

    All numeric values are pre-converted to JAX arrays at load time.
    Nested dictionaries become nested ConstantNamespace objects.
    """

    def __init__(
        self,
        tag: str | None,
        data: Union[dict, str, Path],
        _parent_tag: str | None = None,
    ):
        self._tag = tag
        self._full_tag = f"{_parent_tag}.{tag}" if _parent_tag else tag
        self._data = self._load_and_convert(data)
        self._constants: Dict[str, Any] = {}

    def _load_and_convert(self, data: Union[dict, str, Path]) -> dict:
        """Load data from dict or file path and convert values to JAX arrays."""
        raw_data = self._load_data(data)
        return self._convert_to_jax(raw_data, self._full_tag)

    @staticmethod
    def _convert_to_jax(data: dict, parent_tag: str | None = None) -> dict:
        """Recursively convert all numeric values to JAX arrays and dicts to namespaces."""
        converted = {}
        for key, value in data.items():
            converted[key] = ConstantNamespace._convert_value(value, key, parent_tag)
        return converted

    @staticmethod
    def _convert_value(value: Any, key: str | None = None, parent_tag: str | None = None) -> Any:
        """Convert a single value to JAX array if numeric, or ConstantNamespace if dict."""
        # Nested dictionary -> nested ConstantNamespace
        if isinstance(value, dict):
            return ConstantNamespace(key, value, _parent_tag=parent_tag)

        # Already a JAX array
        if isinstance(value, jnp.ndarray):
            return value

        # NumPy array -> JAX array
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return jnp.asarray(value)
        except ImportError:
            pass

        # Numeric scalar (int, float)
        if isinstance(value, (int, float)):
            return jnp.asarray(value)

        # List/tuple - check if numeric or contains dicts
        if isinstance(value, (list, tuple)):
            # Check if it contains dicts (don't convert to array)
            if any(isinstance(item, dict) for item in value):
                # Convert each dict to ConstantNamespace, keep others as-is
                return [
                    (
                        ConstantNamespace(f"{key}[{i}]", item, _parent_tag=parent_tag)
                        if isinstance(item, dict)
                        else ConstantNamespace._convert_value(item, f"{key}[{i}]", parent_tag)
                    )
                    for i, item in enumerate(value)
                ]
            # Check if it's numeric (could be nested arrays)
            if ConstantNamespace._is_numeric_sequence(value):
                return jnp.asarray(value)
            # Otherwise keep as-is (e.g., list of strings)
            return value

        # Non-numeric (strings, etc.) -> keep as-is
        return value

    @staticmethod
    def _is_numeric_sequence(seq) -> bool:
        """Check if a sequence contains only numeric values (possibly nested)."""
        if not seq:
            return True

        for item in seq:
            if isinstance(item, (int, float)):
                continue
            elif isinstance(item, (list, tuple)):
                if not ConstantNamespace._is_numeric_sequence(item):
                    return False
            else:
                # numpy arrays are numeric
                try:
                    import numpy as np

                    if isinstance(item, (np.ndarray, np.generic)):
                        continue
                except ImportError:
                    pass

                # JAX arrays are numeric
                if isinstance(item, jnp.ndarray):
                    continue

                return False
        return True

    def _load_data(self, data: Union[dict, str, Path]) -> dict:
        """Load data from dict or file path."""
        # Already a dict
        if isinstance(data, dict):
            return data

        # Convert to Path
        path = Path(data)

        if not path.exists():
            raise FileNotFoundError(f"Constant file not found: {path}")

        suffix = path.suffix.lower()

        # JSON
        if suffix == ".json":
            return self._load_json(path)

        # YAML
        elif suffix in (".yaml", ".yml"):
            return self._load_yaml(path)

        # TOML
        elif suffix == ".toml":
            return self._load_toml(path)

        # Pickle (for numpy arrays, etc.)
        elif suffix in (".pkl", ".pickle"):
            return self._load_pickle(path)

        # NumPy .npz
        elif suffix == ".npz":
            return self._load_npz(path)

        # NumPy .npy (single array - wrap in dict)
        elif suffix == ".npy":
            return self._load_npy(path)

        else:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. "
                f"Supported formats: .json, .yaml, .yml, .toml, .pkl, .pickle, .npz, .npy"
            )

    @staticmethod
    def _load_json(path: Path) -> dict:
        """Load JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load YAML file."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("PyYAML is required to load .yaml/.yml files. Install with: pip install pyyaml")

        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_toml(path: Path) -> dict:
        """Load TOML file."""
        try:
            # Python 3.11+ has tomllib in stdlib
            import tomllib

            with open(path, "rb") as f:
                return tomllib.load(f)
        except ImportError:
            try:
                import toml  # type: ignore[import-untyped]

                with open(path, "r") as f:
                    return toml.load(f)
            except ImportError:
                raise ImportError("toml package is required to load .toml files. Install with: pip install toml")

    @staticmethod
    def _load_pickle(path: Path) -> dict:
        """Load pickle file."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Pickle file must contain a dict, got {type(data).__name__}")
        return data

    @staticmethod
    def _load_npz(path: Path) -> dict:
        """Load NumPy .npz file."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy is required to load .npz files.")

        npz = np.load(path, allow_pickle=True)
        return dict(npz)

    @staticmethod
    def _load_npy(path: Path) -> dict:
        """Load NumPy .npy file (single array, wrapped in dict with filename as key)."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy is required to load .npy files.")

        arr = np.load(path, allow_pickle=True)
        key = path.stem  # Use filename without extension as key
        return {key: arr}

    def __getattr__(self, key: str):
        # Avoid recursion for private attributes
        if key.startswith("_"):
            raise AttributeError(key)

        if key not in self._data:
            available = list(self._data.keys())
            raise AttributeError(f"Constant '{self._full_tag}' has no key '{key}'. Available keys: {available}")

        value = self._data[key]

        # If it's already a ConstantNamespace, return it directly
        if isinstance(value, ConstantNamespace):
            return value

        # Lazy creation of Constant objects for leaf values
        if key not in self._constants:
            self._constants[key] = Constant(self._full_tag, key, value)

        return self._constants[key]

    def __getitem__(self, key: str):
        """Support P["key"] syntax as well."""
        try:
            return self.__getattr__(key)
        except AttributeError as e:
            raise KeyError(str(e))

    def __contains__(self, key: str) -> bool:
        """Support 'key in P' syntax."""
        return key in self._data

    def __iter__(self):
        """Iterate over keys."""
        return iter(self._data.keys())

    def keys(self):
        """Return available keys."""
        return self._data.keys()

    def values(self):
        """Return Constant objects or nested namespaces for all keys."""
        return [self.__getattr__(k) for k in self._data.keys()]

    def items(self):
        """Return (key, Constant/ConstantNamespace) pairs."""
        return [(k, self.__getattr__(k)) for k in self._data.keys()]

    def to_dict(self) -> dict:
        """Recursively convert back to a plain dictionary."""
        result: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, ConstantNamespace):
                result[key] = value.to_dict()
            elif isinstance(value, jnp.ndarray):
                result[key] = value
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, ConstantNamespace) else item for item in value]
            else:
                result[key] = value
        return result

    def __repr__(self):
        def format_keys(data, indent=0):
            lines = []
            for key, value in data.items():
                if isinstance(value, ConstantNamespace):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.extend(format_keys(value._data, indent + 1))
                else:
                    lines.append(f"{'  ' * indent}{key}")
            return lines

        keys_repr = ", ".join(format_keys(self._data))
        return f"ConstantNamespace({self._full_tag}, keys=[{keys_repr}])"

    def __len__(self):
        return len(self._data)


class Constant(Placeholder):
    """Concrete scalar/array embedded in the trace.

    Values are pre-converted to JAX arrays at creation time.
    """

    def __init__(self, tag: str | None, key: str | None, value: Any):
        self.tag = tag
        self.key = key
        self.value = value  # Already a jnp.ndarray from ConstantNamespace

    def __repr__(self):
        # Truncate large arrays for display
        if hasattr(self.value, "shape"):
            if self.value.shape == ():
                val_repr = f"{float(self.value)}"
            else:
                val_repr = f"array{self.value.shape}"
        elif isinstance(self.value, (list, tuple)) and len(self.value) > 5:
            val_repr = f"{type(self.value).__name__}[{len(self.value)}]"
        else:
            val_repr = repr(self.value)
        return f"Constant({self.tag}.{self.key}={val_repr})"


class Variable(Placeholder):
    """Independent variable placeholder (e.g., `x`, `y`, `t`).

    Carries the domain tag and dimension index so the solver can bind sampled
    coordinates when evaluating traced expressions.

    For time-dependent problems, spatial variables (``axis='spatial'``) index
    into the spatial context array ``context[tag]`` shaped ``(N, D_spatial)``
    (after the outer B and T vmaps peel off their axes).  The temporal
    variable (``axis='temporal'``) reads from a separate
    ``context["__time__"]`` entry that is a scalar (after the T vmap).
    """

    def __init__(
        self,
        tag: str,
        dim: list,
        domain: Any,
        axis: str = "spatial",
        fem_meta: dict | None = None,
    ):
        self.tag = tag
        self.dim = dim
        self.axis = axis  # 'spatial' or 'temporal'
        self.fem_meta = fem_meta  # Optional dict for FEM-specific metadata (e.g., element type, node positions)
        if tag in domain.context.keys():
            self.size = dim[1] - dim[0] if dim[1] is not None else domain.context[tag].shape[-1]
        else:
            raise KeyError(f"Variable tag '{tag}' not found in domain.context. Available: {list(domain.context.keys())}")
        self._domain = domain  # Reference to parent domain for inference

    def __repr__(self):
        if self.axis == "temporal":
            return "Var(t)"
        if self.fem_meta is not None:
            support = self.fem_meta.get("support")
            region = self.fem_meta.get("region_id")
            return f"Var({self.tag}[{self.dim}], support={support}, region={region})"
        return f"Var({self.tag}[{self.dim}])"


class TensorTag(Placeholder):
    """Named tensor tag used for parametric inputs or coefficients.

    Backed by arrays attached to the domain; when domains merge, tags stack
    along the batch dimension.
    """

    def __init__(self, tag: str, domain=None, dim_index: int | None = None):
        self.tag = tag
        self._domain = domain
        self.dim_index = dim_index  # For slicing multi-dimensional tensors

    def __repr__(self):
        if self.dim_index is not None:
            return f"Tensor({self.tag})[{self.dim_index}]"
        return f"Tensor({self.tag})"


class BinaryOp(Placeholder):
    """Binary arithmetic/elementwise op (e.g., +, -, *, /, **).

    Stores the operator string plus left/right operands so evaluation and
    visualization can rebuild the expression tree.
    """

    def __init__(self, op: str, left: Placeholder, right: Placeholder):
        self.op = op
        self.left = left
        self.right = right
        _propagate_weak(self, left, right)

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class Tracker(Placeholder):
    """Wraps an expression to be monitored during training without contributing to the loss."""

    def __init__(self, expr: Placeholder, interval: int = 1, reduce=None):
        self.expr = expr
        self.interval = interval
        self.reduce = reduce
        self.op_id = _next_op_id()

    def __repr__(self):
        return f"Tracker({self.expr!r}, interval={self.interval})"


class Model(Placeholder):
    """Wrapper for user-defined Equinox models.

    Allows using any Equinox module within the PINO tracing system.
    The module is initialized lazily when the input dimension is known.

    Example - Direct call style (module takes separate arguments):
        class MLP(eqx.Module):
            ...
            def __call__(self, x, y, *, key=None):
                z = jnp.concat([x, y], axis=-1)
                ...
                return z

        uv_net = pnp.nn.wrap(MLP(..., key=key))
        u = uv_net(x, y)[..., 0]
    """

    def __init__(self, module: Any, name: str = "", weight_path: str | None = None):
        """Create a Model wrapper.

        Args:
            module: An Equinox module instance (already constructed), or a
                    callable / Flax nn.Module for backward compatibility.
        """
        self.module = module
        self.name = name
        self.input_dim = None
        self.weight_path = weight_path
        self.layer_id = _next_op_id()
        self.show = True  # Whether or not to print the model

        # ── training config (plain Python, not JAX arrays) ──
        self._frozen: bool = False
        self._lora_config: list[dict] | None = None  # [{"target", "rank", "alpha", "wrappers"}, ...]
        self._opt_fn = None  # optax optimizer factory / instance
        self._lr = LearningRateSchedule(1.0)
        self._bayesian_cfg = None  # {"factory", "kernel_kwargs", "prior", "warmup", "keep", "thin"}
        self._vi_cfg = None  # {"factory", "optimizer", "num_samples", "posterior_draws", "prior"}
        self._posterior_samples_pytree = None  # stacked module pytree, leading axis = N; populated by solve()
        # Per-step blackjax info aggregated post-solve into a
        # ``{field_name: (K, N) array}`` dict — populated for HMC-family
        # kernels (NUTS/HMC: is_divergent, acceptance_rate, energy; MALA:
        # acceptance_rate).  ``None`` for SG-MCMC / VI / non-Bayesian.
        self._posterior_diagnostics: dict | None = None
        self._dtype = None  # target dtype (e.g. jnp.bfloat16) or None
        self._param_mask = None  # current mask scope for grouped optimizer/lr calls
        self._trainable_param_mask = None  # persistent trainability mask used by mask(...).freeze()
        self._lora_param_mask = None  # param mask passed to mask().lora() → restricts which modules get LoRA
        self._mask_scope_pending: bool = False  # transient flag for mask(...).optimizer()/scale() group scoping
        self._param_groups: list = []  # [{target, mask, opt_fn, lr}] for per-group optimizer config
        self._weight_tree = None  # pretrained weights as a pytree (alternative to weight_path file)
        self._initialize_mask = None  # optional bool pytree consumed by initialize() for partial preload
        self._initializer_fn = None  # callable initializer used at compile time
        self._initializer_key = None  # optional PRNG key for callable initializer
        # Phase 12 — logdensity-aware initializer (jno.bayesian.pathfinder, future Laplace, …).
        # Detected by .initialize() via the requires_logdensity = True marker.
        # Runs *inside* solve() after the loss is built but before the kernel state is finalised.
        self._bayesian_initializer = None
        self._bayesian_initializer_key = None
        self._tunable_opts: Dict[str, list] = {}  # per-model tunable options for sweeps

    # ── public API ───────────────────────────────────────────

    @property
    def params(self):
        """Alias for :attr:`module` — always reflects the current (post-training) weights."""
        return self.module

    @property
    def posterior_samples(self):
        """Post-warmup MCMC samples for this model, or ``None`` if it was not
        configured with :meth:`bayesian`.

        Returns the stacked module pytree (leading axis = number of kept
        samples).  For :func:`jno.np.parameter` models, the single-leaf
        ``_Parameter(value=…)`` wrapper is unwrapped to a plain array, so
        ``a.posterior_samples`` has shape ``(N, *a_shape)`` directly.
        """
        pytree = getattr(self, "_posterior_samples_pytree", None)
        if pytree is None:
            return None
        if getattr(self, "_is_jno_scalar_parameter", False):
            return pytree.value
        return pytree

    @property
    def posterior_diagnostics(self):
        """Per-step blackjax kernel info aggregated across the chain.

        Returns a ``{field: (K, N) array}`` dict — one entry per field
        the kernel surfaces:

        * **NUTS / HMC** — ``is_divergent`` (bool), ``acceptance_rate``
          (float), ``energy`` (float).
        * **MALA** — ``acceptance_rate`` only.
        * **SG-MCMC / VI** — ``None``: those kernels expose no
          per-step info object.

        Returns ``None`` if this model is not Bayesian, or if its
        kernel doesn't surface per-step diagnostics.  Inspect
        ``is_divergent`` first — non-zero counts mean the integrator
        is repeatedly failing and the chain cannot be trusted at the
        posted ``step_size`` / ``inverse_mass_matrix``.
        """
        return getattr(self, "_posterior_diagnostics", None)

    def __call__(self, *args) -> "ModelCall":
        """Call this module with variables and return a traced ``ModelCall``."""
        return ModelCall(self, list(args))

    def __repr__(self):
        return f"Model({type(self.module).__name__})"

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.show = False
        return self

    def summary(self):
        """Print and persist a complete model-control summary.

        The summary is always printed to stdout. If a default jNO logger is
        active, the same content is appended to ``<logger.path>/log.log``.

        Returns:
            self (for chaining).
        """
        log = get_logger(use_default=True)

        # Basic identity
        module_name = type(self.module).__name__
        lines = [
            f"Model Summary (layer {self.layer_id})",
            "=" * 60,
            f"name:                 {self.name or module_name}",
            f"module_type:          {module_name}",
            f"show_architecture:    {self.show}",
            "",
            "Training Controls",
            "-" * 60,
            f"frozen:               {self._frozen}",
            f"dtype:                {self._dtype}",
            f"optimizer:            {getattr(self._opt_fn, '__name__', str(self._opt_fn))}",
            f"lr_schedule:          {self._lr}",
            f"lora_config:          {self._lora_config}",
            f"mask_scope_pending:   {self._mask_scope_pending}",
            "",
            "Initialization",
            "-" * 60,
            f"weight_path:          {self.weight_path}",
            f"weight_tree_set:      {self._weight_tree is not None}",
            f"initialize_mask_set:  {self._initialize_mask is not None}",
            f"initializer_fn_set:   {self._initializer_fn is not None}",
            f"initializer_key_set:  {self._initializer_key is not None}",
            "",
            "Mask Diagnostics",
            "-" * 60,
            f"mask_active:          {self._param_mask is not None}",
            f"trainable_mask_set:   {self._trainable_param_mask is not None}",
            f"lora_mask_set:        {self._lora_param_mask is not None}",
            "",
            "Parameter Groups",
            "-" * 60,
            f"num_groups:           {len(self._param_groups)}",
        ]

        for i, g in enumerate(self._param_groups):
            target = g.get("target")
            opt_fn = g.get("opt_fn")
            lr = g.get("lr")
            mask = g.get("mask")
            n_true = None
            try:
                leaves = jax.tree_util.tree_leaves(mask) if mask is not None else []
                n_true = sum(1 for x in leaves if isinstance(x, bool) and x)
            except Exception:
                n_true = None
            lines.append(
                f"  [{i}] target={target!r}, opt={getattr(opt_fn, '__name__', str(opt_fn))}, "
                f"scale={lr}, matched_leaves={n_true}"
            )

        # Param summary
        try:
            leaves = jax.tree_util.tree_leaves(self.module)
            arr_leaves = [leaf for leaf in leaves if eqx.is_array(leaf)]
            n_params = int(sum(int(leaf.size) for leaf in arr_leaves))
            lines.extend(
                [
                    "",
                    "Parameters",
                    "-" * 60,
                    f"array_leaves:         {len(arr_leaves)}",
                    f"total_parameters:     {n_params}",
                ]
            )
        except Exception as exc:
            lines.extend(
                [
                    "",
                    "Parameters",
                    "-" * 60,
                    f"count_failed:         {exc}",
                ]
            )

        # Always print summary to stdout.
        for line in lines:
            print(line)

        # Also append to files under the active logger path when available.
        summary_file = None
        try:
            if getattr(log, "path", None) is not None:
                summary_file = Path(log.path) / "mode_summary.txt"
        except Exception:
            summary_file = None

        if summary_file is not None:
            try:
                summary_file.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_file, "a", encoding="utf-8") as f:
                    for line in lines:
                        f.write(f"{line}\n")
                    f.write("\n")
                print(f"Model summary appended to {summary_file}")
            except Exception as exc:
                print(f"Model summary could not be written to summary file: {exc}")

        return self

    # ── finetuning helpers ───────────────────────────────────

    def freeze(self):
        """Mark this model as frozen (not trained).

        When preceded by ``mask(...)``, only the currently selected
        parameters are frozen and everything else remains trainable::

            NN.mask(param_mask).freeze()         # True leaves frozen, False leaves trainable
            NN.freeze()                          # whole model frozen

        Order matters: ``mask()`` must be called before ``freeze()``.
        """
        if self._mask_scope_pending and self._param_mask is not None:
            # mask(...).freeze(): invert the current scope mask so
            # selected=True leaves become frozen (False), and non-selected
            # leaves remain trainable (True).
            self._trainable_param_mask = jax.tree_util.tree_map(
                lambda x: (not x) if isinstance(x, bool) else False,
                self._param_mask,
            )
            self._mask_scope_pending = False
            self._frozen = False
        else:
            self._trainable_param_mask = None
            self._frozen = True
            self._mask_scope_pending = False
        return self

    def unfreeze(self):
        """Unfreeze this model so it is trained normally."""
        self._frozen = False
        self._trainable_param_mask = None
        return self

    def constrain(self, transform: Callable) -> "Model":
        """Apply a paramax reparameterization to trainable parameter leaves.

        Parameters are stored in their unconstrained form and transformed by
        ``transform`` before every forward pass via ``paramax.unwrap()``,
        which jno's training loop calls automatically.

        When preceded by ``mask(...)``, only leaves where the mask is ``True``
        are wrapped — all other leaves remain unconstrained::

            k_net.mask(output_mask).constrain(jax.nn.softplus)  # output layer only
            k_net.constrain(jax.nn.softplus)                    # all parameters

        Args:
            transform: A jit-compatible callable (e.g. ``jax.nn.softplus``,
                       ``jax.nn.sigmoid``).

        Returns:
            self  (for chaining)
        """
        import paramax as _pm

        use_mask = self._mask_scope_pending and self._param_mask is not None
        mask = self._param_mask if use_mask else None
        self._mask_scope_pending = False

        def _wrap(leaf, selected=True):
            if eqx.is_inexact_array(leaf) and selected:
                return _pm.Parameterize(transform, leaf)
            return leaf

        if mask is not None:
            self.module = jax.tree_util.tree_map(_wrap, self.module, mask)
        else:
            self.module = jax.tree_util.tree_map(lambda leaf: _wrap(leaf), self.module)
        return self

    def mask(self, param_mask=None):
        """Set the current mask scope using an explicit boolean pytree mask.

        ``param_mask`` must mirror the parameter tree structure and contain
        boolean leaves where ``True`` selects leaves in the masked scope.

        This scope is consumed by grouped optimizer/lr calls and by
        ``mask(...).freeze()``.  It is also read by ``u.grad(net.mask(...))``
        to restrict the Jacobian to only the selected parameters.

        Example::

            import equinox as eqx, jax

            all_false = jax.tree_util.tree_map(lambda _: False, model.module)
            param_mask = eqx.tree_at(
                lambda m: (m.layers[0].weight, m.layers[0].bias),
                all_false, (True, True),
            )
            model.mask(param_mask).optimizer(optax.adam(1e-3))
            J = crux.eval([u.grad(model.mask(param_mask))])[0]  # (N, P_selected)
        """
        self._param_mask = param_mask
        self._mask_scope_pending = self._param_mask is not None
        return self

    def lora(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        *,
        target: str | None = None,
        wrapper: type[LoRAWrapper] | Sequence[type[LoRAWrapper]] | None = None,
        specs: list[dict] | None = None,
    ):
        """Enable LoRA fine-tuning for this model.

        Two calling conventions:

        1. **Uniform**::

               NN.lora(rank=8, alpha=16)
               NN.lora(rank=4, wrapper=MyConvAdapter)          # custom adapter
               NN.lora(rank=4, wrapper=[LoRALinear, MyConv])   # tried in order

        2. **Per-target** — different rank/alpha/adapter per layer group::

               NN.lora(specs=[
                   {"target": "encoder", "rank": 4,  "alpha": 1.0},
                   {"target": "conv",    "rank": 8,  "alpha": 2.0, "wrapper": MyConvAdapter},
               ])

           Each ``target`` is a regex matched against the pytree path.
           The first matching spec wins.

        By default only the low-rank adapters are trained; base weights are
        frozen.  Layers that are NOT wrapped by LoRA remain fully trainable.
        Call ``freeze()`` before ``lora()`` to also freeze any parameters
        outside LoRA-wrapped layers::

            NN.freeze().lora(rank=8, alpha=16)

        Use ``mask(M)`` to restrict which layers receive LoRA adapters::

            NN.mask(M).lora(rank=8, alpha=16)  # only M-selected layers are wrapped

        Args:
            rank:    LoRA rank (uniform mode).
            alpha:   LoRA scaling factor (uniform mode).
            target:  Regex to restrict *which layers get LoRA adapters* (uniform
                     mode only).  Layers whose pytree path does not match are left
                     completely untouched.  Use ``specs=`` for per-group targeting.
            wrapper: Adapter class or list of classes to try in order.
                     Defaults to ``(LoRALinear, LoRAConv)`` — wraps both linear
                     and conv layers.  Pass a single class or list to override.
            specs:   Per-target specs (per-target mode).  Each dict has keys
                     ``target`` (str regex), ``rank`` (int), ``alpha`` (float),
                     and optionally ``wrapper``.
        """
        if self._mask_scope_pending and self._param_mask is not None:
            self._lora_param_mask = self._param_mask
        else:
            self._lora_param_mask = None
        # lora() always clears any stale freeze mask — it overrides mask().freeze() semantics.
        self._trainable_param_mask = None
        self._mask_scope_pending = False

        default_wrappers = _normalize_wrappers(wrapper)

        if specs is not None:
            self._lora_config = [
                {
                    "target": s.get("target"),
                    "rank": s["rank"],
                    "alpha": s["alpha"],
                    "wrappers": _normalize_wrappers(s["wrapper"]) if "wrapper" in s else default_wrappers,
                }
                for s in specs
            ]
        else:
            self._lora_config = [{"target": target, "rank": rank, "alpha": alpha, "wrappers": default_wrappers}]

        return self

    def optimizer(self, opt_fn: Any):
        """Attach an optimizer to this model.

        When preceded by ``mask(param_mask)``, the optimizer applies only
        to matching parameters; everything else uses the global optimizer
        (set via a bare ``optimizer()`` call)::

            NN.mask(mask_decoder).optimizer(optax.adam)  # decoder group
            NN.mask(mask_encoder).optimizer(optax.sgd)   # encoder group
            NN.optimizer(optax.adam)                   # global fallback

        Bake the learning rate into the optax optimizer (e.g. ``optax.adam(1e-3)``);
        use :meth:`scale` to multiply it -- e.g. with a ``dlrs(...)`` schedule for
        loss-adaptive learning-rate scaling. ``mask(...)`` is one-shot, so to scale a
        masked group call ``mask(...)`` again before ``scale(...)``::

            NN.mask(mask_decoder).optimizer(optax.adam(1e-3))
            NN.mask(mask_decoder).scale(my_schedule)

        A bare/global call (not preceded by ``mask(...)``) replaces any
        previously configured parameter groups.

        Args:
            opt_fn: An optax optimizer factory, e.g. ``optax.adam``,
                    or an already-constructed transform.
        """
        if self._mask_scope_pending and self._param_mask is not None:
            # One-shot masked scope: consume mask on this call.
            group = self._get_or_create_group()
            if self._opt_fn is None:
                self._opt_fn = opt_fn
            group["opt_fn"] = opt_fn
            self._mask_scope_pending = False
        else:
            self._opt_fn = opt_fn
            # Global optimizer replacement should discard stale group overrides.
            self._param_groups = []
            self._mask_scope_pending = False

        return self

    def bayesian(
        self,
        kernel_factory,
        *,
        prior=None,
        warmup: int = 500,
        keep: int = 1000,
        thin: int = 1,
        adapt: bool = True,
        num_chains: int = 1,
        init_jitter: float = 0.0,
        likelihood_scale: float = 1.0,
        **kernel_kwargs,
    ):
        """Sample this model's parameters from a posterior via blackjax.

        Mirrors :meth:`optimizer` but uses an MCMC kernel instead of an
        optax update.  The model's parameters become the *position* of a
        chain initialised at the current weights; each training step is
        one transition of ``kernel_factory(logdensity_fn, **kernel_kwargs)``
        for full-data kernels (NUTS/HMC/MALA), or
        ``kernel_factory(grad_estimator, **kernel_kwargs)`` for SG-MCMC
        (SGLD/SGHMC).  jno chooses the dispatch shape by inspecting the
        factory's signature.

        After :meth:`jno.core.core.solve` returns, the chain is available
        on the model via :attr:`posterior_samples` — a pytree mirroring
        the model parameters with a leading sample axis of length
        ``keep // thin``.

        Example — NUTS on a scalar PDE coefficient::

            import blackjax, jax.numpy as jnp
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.bayesian(
                blackjax.nuts,
                step_size=1e-2,
                inverse_mass_matrix=jnp.ones(1),
                warmup=500,
                keep=1000,
            )
            crux = jno.core([residual.mse])
            crux.solve(2000)
            chain = a.posterior_samples            # (1000, 1)

        Example — SGLD on a small MLP (BNN PINN)::

            import blackjax
            net = jno.nn.wrap(foundax.mlp(...))
            net.bayesian(blackjax.sgld, step_size=1e-5, warmup=2000, keep=1000)

        References:
            * NUTS — Hoffman & Gelman (2014), *The No-U-Turn Sampler*, JMLR
              15(1), 1593-1623.
            * SGLD — Welling & Teh (2011), *Bayesian Learning via Stochastic
              Gradient Langevin Dynamics*, ICML 2011, 681-688.

        Args:
            kernel_factory: A blackjax kernel constructor, e.g.
                ``blackjax.nuts`` or ``blackjax.sgld``.  Any callable that
                takes ``logdensity_fn`` (full-data) or ``grad_estimator``
                (SG-MCMC) as its first positional argument is accepted.
            prior: Optional ``pytree -> float`` log-prior.  Default: a wide
                isotropic Gaussian with σ=10 over all inexact-array leaves.
            warmup: For adaptive HMC-family kernels (``blackjax.nuts`` /
                ``blackjax.hmc``) with ``adapt=True``: number of
                ``blackjax.window_adaptation`` steps run before the main
                loop.  Adapted ``step_size`` and ``inverse_mass_matrix``
                replace the initial values, and the main loop collects
                samples from epoch 0.  For non-adaptive kernels (MALA,
                SGLD, SGHMC) and ``adapt=False``: number of initial outer
                epochs whose samples are discarded.  Default 500.
            keep:   Number of post-warmup samples to retain (after thinning).
                Default 1000.
            thin:   Keep one sample every ``thin`` post-warmup steps.
                Default 1.
            adapt:  When ``True`` (default) and the kernel is in the HMC
                family, ``blackjax.window_adaptation`` runs for ``warmup``
                steps before the main loop and adapts step size + inverse
                mass matrix.  Silently no-op for non-adaptive kernels.
                Set ``False`` to revert to the "discard first N samples"
                semantics with the user-supplied hyperparameters.
            num_chains: Number of parallel MCMC chains run via
                ``jax.vmap``.  Default 1.  Output
                :attr:`posterior_samples` has leading axes ``(K, N, *)``
                regardless of K (arviz / `az.from_dict` compatible).
                All ``.bayesian()`` models in a single :meth:`solve` call
                must share the same ``num_chains``.
            init_jitter: When ``num_chains > 1``, perturb each chain's
                initial position by ``N(0, init_jitter)`` for
                over-dispersion (gives a more conservative R-hat).
                Default 0.0 = all chains start from the same point with
                different PRNG keys.
            likelihood_scale: Multiplier on the negative log-likelihood
                term in the per-step logdensity.  Default ``1.0``.  The
                canonical Gaussian-noise log-likelihood is a *sum* over
                data points; jno's ``residual.mse`` returns a *mean*.
                Pass ``N_obs`` (the data-point count) — or
                ``N_obs / sigma**2`` more generally — to recover the
                correct posterior magnitude.  Without this, MCMC chains
                on multi-thousand-point PINN losses move much more
                slowly than they should and VI is often stuck near the
                prior.
            **kernel_kwargs: Forwarded to ``kernel_factory``.  ``step_size``
                is optional for HMC-family kernels (NUTS / HMC) when
                ``adapt=True`` and ``warmup > 0`` — window adaptation
                picks one.  **Required** for ``adapt=False``, MALA, and
                SG-MCMC.  May include e.g. ``inverse_mass_matrix=``
                (NUTS/HMC), ``num_integration_steps=`` (HMC/SGHMC).

        Returns:
            self, for chaining.
        """
        if int(num_chains) < 1:
            raise ValueError(f"num_chains must be >= 1, got {num_chains}.")
        if float(likelihood_scale) <= 0.0:
            raise ValueError(f"likelihood_scale must be positive, got {likelihood_scale!r}.")
        if getattr(self, "_vi_cfg", None) is not None:
            raise ValueError("Model already has .vi(...) configured; .bayesian() and .vi() are mutually exclusive.")
        cfg = {
            "factory": kernel_factory,
            "kernel_kwargs": dict(kernel_kwargs),
            "prior": prior,
            "warmup": int(warmup),
            "keep": int(keep),
            "thin": int(thin),
            "adapt": bool(adapt),
            "num_chains": int(num_chains),
            "init_jitter": float(init_jitter),
            "likelihood_scale": float(likelihood_scale),
        }
        if self._mask_scope_pending and self._param_mask is not None:
            # Masked branch: register this kernel as a per-group backend
            # on the currently pending ``.mask(M)`` scope.  The remaining
            # leaves (not covered by any group) use the global optimizer
            # (if set via a bare ``.optimizer(...)``), or are frozen.
            group = self._get_or_create_group()
            if group.get("backend") not in (None, "optax"):
                raise ValueError(
                    "Model: this mask scope already has a non-optax backend "
                    f"({group['backend']!r}).  One backend per mask scope."
                )
            group["backend"] = "bayesian"
            group["bayesian_cfg"] = cfg
            self._mask_scope_pending = False
        else:
            # Global branch — unchanged.
            self._bayesian_cfg = cfg
        # `.bayesian()` IS the update — clear any prior `.freeze()` flag so
        # solve() does not skip this model.
        self._frozen = False
        self._trainable_param_mask = None
        return self

    def vi(
        self,
        factory,
        *,
        optimizer,
        num_samples: int = 8,
        posterior_draws: int = 500,
        prior=None,
        likelihood_scale: float = 1.0,
        init_log_std: float = -3.0,
        init_mu_at_position: bool = True,
        **factory_kwargs,
    ):
        """Fit a variational approximation to this model's posterior.

        Mirrors :meth:`bayesian` but optimises the **evidence lower bound**
        (ELBO) of a variational family ``q`` instead of running an MCMC
        chain.  After :meth:`jno.core.core.solve` returns, ``posterior_draws``
        i.i.d. samples are drawn from the fitted ``q`` and stored on the
        model as :attr:`posterior_samples` — same ``(1, N, *param)`` layout
        as the MCMC path, so all downstream code (:func:`crux.eval` with
        ``samples="auto"``, :func:`jno.bayesian.rhat`, wandb stats) keeps
        working transparently.

        Example — mean-field VI on a scalar PDE coefficient::

            import blackjax, optax
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.vi(
                blackjax.meanfield_vi,
                optimizer=optax.adam(1e-3),
                num_samples=8,
                posterior_draws=500,
            )
            crux = jno.core([residual.mse])
            crux.solve(2000)            # 2000 ELBO optimisation steps
            chain = a.posterior_samples  # (1, 500, 1) — draws from fitted q

        References:
            * Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei,
              D. M. (2017).  *Automatic Differentiation Variational
              Inference.*  JMLR 18(1), 430–474.
            * Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013).
              *Stochastic Variational Inference.*  JMLR 14(1), 1303–1347.

        Args:
            factory: A blackjax VI factory, e.g. :func:`blackjax.meanfield_vi`.
                Detected via signature (first arg ``logdensity_fn``, second
                ``optimizer``).
            optimizer: An optax ``GradientTransformation`` used to
                optimise the ELBO.  Common choice: ``optax.adam(1e-3)``.
            num_samples: Monte-Carlo sample count used to estimate the ELBO
                at each step.  Higher = lower-variance gradient, slower
                step.  Default 8.
            posterior_draws: After solve(), draw this many i.i.d. samples
                from the fitted variational distribution and store them on
                :attr:`posterior_samples`.  Default 500.
            prior: Optional ``pytree -> float`` log-prior.  Default: the
                same wide isotropic Gaussian (σ=10) used by
                :meth:`bayesian`.
            likelihood_scale: Multiplier on the negative log-likelihood
                term in the ELBO.  Default ``1.0``.  The canonical
                Gaussian-noise log-likelihood is a *sum* over data
                points; jno's ``residual.mse`` returns a *mean*.  For
                mean-field VI in particular, pass ``N_obs`` (or
                ``N_obs / sigma**2``) so the likelihood actually pulls
                the variational mean away from the prior.  Without
                this, VI is often stuck near its initialisation
                because the prior dominates by a factor of
                ``N_obs``.
            init_log_std: Initial value for ``state.rho`` (log-std of the
                variational ``q``) on every weight.  Default ``-3.0``
                (σ ≈ 0.05) — keeps the initial MC ELBO sample tight so
                gradients are low-variance from the start; the
                optimiser then grows rho where the posterior is wide.
                Pass ``0.0`` (σ ≈ 1.0) to restore blackjax's default
                broad init.
            init_mu_at_position: When ``True`` (default), initialise
                ``state.mu`` at the user-supplied position (matches
                numpyro's autoguide).  ``False`` keeps blackjax's
                zero start — only useful on toy problems where the
                MAP is exactly zero.
            **factory_kwargs: Forwarded to ``factory``.  E.g. an explicit
                ``objective=blackjax.vi.meanfield_vi.RenyiAlpha(alpha=0.5)``
                or ``stl_estimator=False``.

        Returns:
            self, for chaining.
        """
        if getattr(self, "_bayesian_cfg", None) is not None:
            raise ValueError("Model already has .bayesian(...) configured; .bayesian() and .vi() are mutually exclusive.")
        if int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
        if int(posterior_draws) < 1:
            raise ValueError(f"posterior_draws must be >= 1, got {posterior_draws}.")
        if float(likelihood_scale) <= 0.0:
            raise ValueError(f"likelihood_scale must be positive, got {likelihood_scale!r}.")
        cfg = {
            "factory": factory,
            "optimizer": optimizer,
            "num_samples": int(num_samples),
            "posterior_draws": int(posterior_draws),
            "prior": prior,
            "factory_kwargs": dict(factory_kwargs),
            "likelihood_scale": float(likelihood_scale),
            "init_log_std": float(init_log_std),
            "init_mu_at_position": bool(init_mu_at_position),
        }
        if self._mask_scope_pending and self._param_mask is not None:
            # Masked branch — register VI as a per-group backend on the
            # currently pending ``.mask(M)`` scope.  Mirror of the
            # masked branch in ``Model.bayesian(...)``.
            group = self._get_or_create_group()
            if group.get("backend") not in (None, "optax"):
                raise ValueError(
                    "Model: this mask scope already has a non-optax backend "
                    f"({group['backend']!r}).  One backend per mask scope."
                )
            group["backend"] = "vi"
            group["vi_cfg"] = cfg
            self._mask_scope_pending = False
        else:
            # Global branch — unchanged.
            self._vi_cfg = cfg
        # .vi() IS the update — clear freeze.
        self._frozen = False
        self._trainable_param_mask = None
        return self

    def scale(self, scale: LearningRateSchedule | float | None):
        """Scale this model's learning rate.

        ``scale`` multiplies the rate the optimizer already carries (applied as
        ``optax.scale(...)``). Pass a loss-adaptive ``dlrs(...)`` schedule here to
        change the scale **dynamically during training**, a static float for a fixed
        factor, or any ``LearningRateSchedule``. For a bare optimizer factory (e.g.
        ``optax.adam`` with no baked-in rate) the scale *is* the learning rate.

        When preceded by ``mask(param_mask)``, the scale applies only to that
        parameter group. ``mask(...)`` is one-shot, so call it immediately before
        ``scale(...)``::

            NN.mask(mask_decoder).scale(dlrs(lr0=1e-3))

        Args:
            scale:  A ``LearningRateSchedule`` / ``dlrs(...)`` (or float) for this
                    model. If *None*, a constant 1e-3 is used.
        """
        if self._mask_scope_pending and self._param_mask is not None:
            # Ensure a global fallback exists for uncovered leaves.
            if self._lr is None:
                self._lr = scale
            self._get_or_create_group()["lr"] = scale
            self._mask_scope_pending = False
        else:
            self._lr = scale
        return self

    def _current_group_key(self) -> str:
        """Stable key for the currently active mask scope."""
        return f"<mask:{id(self._param_mask)}>"

    def _get_or_create_group(self):
        """Return the param group dict for the current mask scope, creating it if needed."""
        key = self._current_group_key()
        for g in self._param_groups:
            if g["target"] == key:
                g["mask"] = self._param_mask  # refresh in case mask() was called again
                return g
        g = {"target": key, "mask": self._param_mask, "opt_fn": None, "lr": None}
        self._param_groups.append(g)
        return g

    def initialize(self, weights: Any, *, key: Any = None) -> "Model":
        """Load pretrained weights into this model at init time.

                Accepted ``weights`` inputs:

                - ``str`` / ``Path``: load from checkpoint path.
                    Supports Equinox ``.eqx`` files and Orbax checkpoint directories
                    (optionally ``"<path>::<model_key>"``).
                - Pytree object: copy array leaves directly from the provided tree.
                - Callable initializer: apply a JAX initializer function to every
                    floating-point array leaf at compile time.

                Examples:

                .. code-block:: python

                        net.initialize("./weights.eqx")
                        net.initialize("./runs/ckpts/2000::1")
                        net.initialize(other_model.module)

                        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0))
                        p.initialize(jax.nn.initializers.ones)

        Args:
            weights: File path / pytree / callable initializer.
            key: Optional PRNG key used when ``weights`` is callable.

        Returns:
            self (for chaining).
        """
        self._initialize_mask = None
        self._initializer_fn = None
        self._initializer_key = None

        # Phase 12 — logdensity-aware initializer.  Detected via the
        # class-level ``requires_logdensity = True`` marker so the existing
        # stateless ``(shape, dtype, key) -> array`` callable path is
        # unaffected (those callables don't carry the attribute).
        if getattr(weights, "requires_logdensity", False):
            self._bayesian_initializer = weights
            self._bayesian_initializer_key = key
            self.weight_path = None
            self._weight_tree = None
            return self

        # Other branches reset the logdensity-aware slot so .initialize()
        # is last-write-wins regardless of which path was previously set.
        self._bayesian_initializer = None
        self._bayesian_initializer_key = None

        if isinstance(weights, (str, Path)):
            self.weight_path = str(weights)
            self._weight_tree = None
        elif isinstance(weights, eqx.Module):
            self._weight_tree = weights
            self.weight_path = None
        elif callable(weights):
            self._initializer_fn = weights
            self.weight_path = None
            self._weight_tree = None
            if key is not None:
                self._initializer_key = key
        else:
            self._weight_tree = weights
            self.weight_path = None
        return self

    def dtype(self, dtype: Any) -> "Model":
        """Set this model's working dtype (parameters **and** compute).

        Casts all floating-point parameters to *dtype* and — at the forward
        seam — casts the model's inputs to match, so the network actually
        *computes* in *dtype* rather than promoting back to float32.  The cast is
        symmetric: it lowers (float32 → bfloat16) **and** promotes (load a
        bfloat16 checkpoint, then ``.dtype(jnp.float32)``), and applies to both
        training and inference.  Integer arrays (e.g. indices) are left unchanged.

        This is the model-precision knob.  **Data** precision (float32 vs
        float64) is JAX's ``jax_enable_x64`` flag — *not* a jNO setting.  Enable
        it before building models/domains (``JAX_ENABLE_X64=1`` or
        ``jax.config.update("jax_enable_x64", True)``).

        Args:
            dtype: A JAX floating dtype object, e.g. ``jnp.bfloat16``,
                ``jnp.float16``, ``jnp.float32`` or ``jnp.float64``.

        Caveats:
            * bfloat16 *compute* degrades autodiff derivatives
              (``.laplacian`` / ``.hessian``) — keep derivative-critical (PINN)
              models in float32 and opt only data-loss / operator backbones into
              bf16.
            * bfloat16 parameters mean the optimizer update also runs in
              bfloat16, which can stall on very small updates.

        Example::

            backbone.dtype(jnp.bfloat16)   # real bf16 compute for this model
            pinn_net.dtype(jnp.float32)    # keep its derivatives full precision
        """
        if isinstance(dtype, str):
            raise ValueError(
                f"Model.dtype() takes a JAX dtype object, not a string {dtype!r}. "
                "Pass e.g. jnp.bfloat16 / jnp.float32 for model precision. "
                "Data precision (float32 vs float64) is JAX's jax_enable_x64 flag "
                "(set JAX_ENABLE_X64=1 or jax.config.update('jax_enable_x64', True) "
                "before building models) — it is not a jNO setting."
            )
        self._dtype = dtype
        return self

    @property
    def model_key(self) -> str:
        """Stable key identifying this model in a sweep space."""
        return self.name if self.name else f"model_{self.layer_id}"

    def tune(
        self,
        *,
        freeze: list | None = None,
        lora: list | None = None,
        optimizer: list | None = None,
        lr: list | None = None,
        dtype: list | None = None,
    ) -> "Model":
        """Declare per-model tunable options for hyperparameter sweeps.

        Each argument accepts a list of candidate values.  During a sweep
        the tuner searches over all combinations.

        Args:
            freeze: List of bool, e.g. ``[True, False]``.
            lora: List of ``(rank, alpha)`` tuples **or** ``None`` values,
                e.g. ``[(4, 1.0), (8, 1.0), None]``.
            optimizer: List of optax factories, e.g. ``[optax.adam]``.
            lr: List of :class:`LearningRateSchedule` objects.
            dtype: List of dtypes, e.g. ``[jnp.float32, jnp.bfloat16]``.

        Returns:
            self (for chaining).

        Example::

            backbone = nn.poseidon(...)
            backbone.initialize("weights.msgpack")
            backbone.tune(
                freeze=[True, False],
                lora=[(4, 1.0), None],
                optimizer=[optax.adam],
                lr=[lrs.constant(1e-4), lrs.constant(1e-5)],
            )
        """
        self._tunable_opts = {}
        if freeze is not None:
            self._tunable_opts["freeze"] = list(freeze)
        if lora is not None:
            self._tunable_opts["lora"] = list(lora)
        if optimizer is not None:
            self._tunable_opts["optimizer"] = list(optimizer)
        if lr is not None:
            self._tunable_opts["lr"] = list(lr)
        if dtype is not None:
            self._tunable_opts["dtype"] = list(dtype)
        return self

    def reset(self):
        """Reset all training configuration to defaults.

        .. note:: This does **not** clear ``_tunable_opts`` — those
           persist across trials so that the tuner can re-apply them.
        """
        self._frozen = False
        self._lora_config = None
        self._opt_fn = None
        self._lr = None
        self._dtype = None
        self._param_mask = None
        self._trainable_param_mask = None
        self._lora_param_mask = None
        self._mask_scope_pending = False
        self._weight_tree = None
        self._initialize_mask = None
        self._initializer_fn = None
        self._bayesian_initializer = None
        self._bayesian_initializer_key = None
        self._merge_lora_flag = False
        # ``_posterior_samples_pytree`` and ``_posterior_diagnostics``
        # are populated by solve() — clearing them here ensures a stale
        # chain from a prior run doesn't bleed into the next.
        self._posterior_samples_pytree = None
        self._posterior_diagnostics = None
        return self

    def to_iree(
        self,
        sample_inputs: tuple,
        *,
        target_backend: str = "llvm-cpu",
        optimization_level: int = 3,
    ) -> IREEModel:
        """Compile this model to an :class:`IREEModel` for deployment.

        The Equinox module's current weights are baked into the compiled
        artefact as constants.  The result is serialisable via
        ``jno.save`` / ``jno.load``.

        Args:
            sample_inputs: Tuple of arrays (NumPy, JAX, or jno.numpy) whose
                shapes match the positional arguments of this call.
            target_backend: IREE target backend (default ``"llvm-cpu"``).
            optimization_level: IREE optimisation level 0–3 (default ``3``).

        Returns:
            A compiled :class:`IREEModel` ready for inference.

        Example::

            iree_m = net.to_iree(
                (jnp.ones((1, 100, 1)), jnp.ones((1, 100, 1)))
            )
            jno.save(iree_m, "model.pkl")
            output = iree_m(x_np, y_np)
        """
        module = self.module
        module_name = type(module).__name__.lower()

        # Convert all inputs to JAX arrays (handles numpy, jnp, jno.numpy)
        jax_inputs = tuple(jnp.asarray(inp) for inp in sample_inputs)

        def infer(*args):
            return module(*args)

        infer.__name__ = module_name

        return IREEModel.compile(
            infer,
            jax_inputs,
            target_backend=target_backend,
            optimization_level=optimization_level,
        )


class ModelCall(Placeholder):
    """Represents a call to a Model with specific arguments.

    This is created when you call a Model directly with variables:
        uv_net = pnp.nn.wrap(MLP())
        result = uv_net(x, y)  # Creates ModelCall

    All training-configuration methods (``freeze``, ``lora``, ``optimizer``,
    ``dtype``, ``initialize``, ``tune``) are proxied to the underlying
    :class:`Model` so you can chain them after the call::

        u = nn.mlp(2, 64, 1)(x, y).freeze()
    """

    def __init__(self, model: Model, args: list):
        self.model = model
        self.args = args
        self.op_id = _next_op_id()

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.model}({args_str})"

    # ── proxied helpers (delegate to Model) ─────────────

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.model.show = False
        return self

    def freeze(self):
        self.model.freeze()
        return self

    def unfreeze(self):
        self.model.unfreeze()
        return self

    def mask(self, param_mask=None):
        """Proxy for :meth:`Model.mask`."""
        self.model.mask(param_mask)
        return self

    def lora(self, rank: int = 4, alpha: float = 1.0, *, target=None, wrapper=None, specs=None):
        self.model.lora(rank, alpha, target=target, wrapper=wrapper, specs=specs)
        return self

    def optimizer(self, opt_fn: Any) -> "ModelCall":
        self.model.optimizer(opt_fn)
        return self

    def scale(self, scale: Any) -> "ModelCall":
        """Proxy for :meth:`Model.scale` -- multiply the learning rate (e.g. with a
        ``dlrs(...)`` schedule for loss-adaptive scaling). Chains after ``optimizer(...)``."""
        self.model.scale(scale)
        return self

    def constrain(self, transform: Callable) -> "ModelCall":
        """Proxy for :meth:`Model.constrain` -- paramax reparameterization applied
        before every forward pass (e.g. ``jax.nn.softplus`` to keep a parameter or a
        ``jno.np.parameter(phi)`` coefficient field positive). Chains after
        :meth:`mask`."""
        self.model.constrain(transform)
        return self

    def regularize(self, kind: str = "h1seminorm", *variables, **kwargs):
        """Regularization loss term for a field -- one surface for FEM and coordinate fields.

        For a **FEM nodal-parameter** field (``jno.np.parameter(<fem symbol>)``) this is the
        FEM-exact penalty assembled on the field's element space. For any **other** field (e.g.
        a coordinate network ``k(x, y)``) it is the autodiff form -- pass the spatial variables
        to differentiate against. Returns a pointwise loss term; reduce with ``.mean`` / ``.sum``
        and weight it::

            crux([data.mse, alpha * k.regularize('smooth', x, y).mean])   # coordinate field
            crux([data.mse, alpha * k.regularize('h1seminorm').mean])     # FEM field parameter

        ``kind``:
          * ``'smooth'`` (``'h1seminorm'``, ``'h1'``) -- H1 seminorm ``∫|∇k|²`` (FEM: ``kᵀLk``).
            Encourages smooth fields.
          * ``'tv'`` -- total variation ``∫|∇k|`` (FEM: eps-smoothed, ``eps=`` kwarg). Sharp interfaces.
          * ``'l2'`` (``'tikhonov'``, ``'ridge'``) -- ``∫(k-ref)²`` (``ref=`` kwarg). **FEM only.**
          * ``'nonneg'`` -- soft positivity ``strength·relu(-k)`` (``strength=`` kwarg). For a hard
            ``k > 0`` use :meth:`constrain` (e.g. ``jax.nn.softplus``).
          * ``'bounded'`` -- soft two-sided barrier outside ``[lo, hi]`` (``lo=``, ``hi=`` kwargs).
        """
        if getattr(self.model, "_fem_field", None) is not None:
            from .._fem import _field_regularizer_term

            return _field_regularizer_term(self, kind, **kwargs)

        # Coordinate field -> autodiff form (no FE space to assemble against).
        k = kind.lower()
        if k in ("smooth", "h1seminorm", "h1"):
            if not variables:
                raise ValueError(
                    "regularize('smooth', ...) on a coordinate field needs the spatial variables, "
                    "e.g. k.regularize('smooth', x, y)."
                )
            acc = self.d(variables[0]) ** 2
            for v in variables[1:]:
                acc = acc + self.d(v) ** 2
            return acc
        if k == "tv":
            if not variables:
                raise ValueError("regularize('tv', ...) needs the spatial variables, e.g. k.regularize('tv', x, y).")
            sq = self.d(variables[0]) ** 2
            for v in variables[1:]:
                sq = sq + self.d(v) ** 2
            return sq**0.5
        if k == "nonneg":
            return FunctionCall(lambda f, _s=kwargs.get("strength", 1.0): _s * jnp.maximum(0.0, -f), [self], name="nonneg")
        if k == "bounded":
            return FunctionCall(
                lambda f, _lo=kwargs["lo"], _hi=kwargs["hi"]: jnp.maximum(0.0, f - _hi) + jnp.maximum(0.0, _lo - f),
                [self],
                name="bounded",
            )
        raise ValueError(
            f"regularize: kind {kind!r} is not available for a coordinate field "
            "('l2'/'tikhonov' is FEM-only); use 'smooth', 'tv', 'nonneg', or 'bounded'."
        )

    def bayesian(self, kernel_factory, *, prior=None, warmup=500, keep=1000, thin=1, adapt=True, **kernel_kwargs):
        """Proxy for :meth:`Model.bayesian`."""
        self.model.bayesian(
            kernel_factory,
            prior=prior,
            warmup=warmup,
            keep=keep,
            thin=thin,
            adapt=adapt,
            **kernel_kwargs,
        )
        return self

    def vi(self, factory, *, optimizer, num_samples=8, posterior_draws=500, prior=None, **factory_kwargs):
        """Proxy for :meth:`Model.vi`."""
        self.model.vi(
            factory,
            optimizer=optimizer,
            num_samples=num_samples,
            posterior_draws=posterior_draws,
            prior=prior,
            **factory_kwargs,
        )
        return self

    @property
    def posterior_samples(self):
        """Shortcut to the underlying :attr:`Model.posterior_samples`."""
        return self.model.posterior_samples

    @property
    def posterior_diagnostics(self):
        """Shortcut to the underlying :attr:`Model.posterior_diagnostics`."""
        return self.model.posterior_diagnostics

    def initialize(self, weights: Any, *, key: Any = None) -> "ModelCall":
        self.model.initialize(weights, key=key)
        return self

    def dtype(self, dtype: Any) -> "ModelCall":
        self.model.dtype(dtype)
        return self

    def summary(self):
        """Proxy for :meth:`Model.summary`."""
        self.model.summary()
        return self

    def tune(self, **kwargs):
        """Proxy for :meth:`Model.tune`."""
        self.model.tune(**kwargs)
        return self


class TunableModule(Placeholder):
    """
    Wraps a Flax module CLASS + ArchSpace.
    Behaves like Model but with lazy instantiation.
    """

    def __init__(self, module_cls: Type, space: "ArchSpace"):
        self.module_cls = module_cls
        self.space = space
        self.layer_id = _next_op_id()
        self._current_instance: Optional[Model] = None

    def __call__(self, *args):
        """Call with variables - creates ModelCall."""
        # If we have a current instance (during solve), use it
        if self._current_instance is not None:
            return self._current_instance(*args)
        # Otherwise create a placeholder call
        return TunableModuleCall(self, list(args))

    def instantiate(self, arch: "Arch", *, key=None):
        """Create module instance with given architecture.

        For equinox modules, pass ``key`` for random initialization.
        Falls back to ``module_cls(arch=arch)`` if the constructor
        does not accept a ``key`` keyword.
        """
        if key is not None:
            try:
                return self.module_cls(arch=arch, key=key)
            except TypeError:
                return self.module_cls(arch=arch)
        return self.module_cls(arch=arch)


class TunableModuleCall(Placeholder):
    """Call to a TunableModule - resolved at solve time."""

    def __init__(self, model: TunableModule, args: list):
        self.model = model
        self.args = args
        self.op_id = _next_op_id()

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.model}({args_str})"

    def dont_show(self):
        """If called will NOT display the network architecture."""
        self.model.module_cls._show = False
        return self


class OperationDef(Placeholder):
    """An operation definition - traces a computation graph.

    When called with variables, returns an OperationCall that can be
    evaluated during solve iterations.
    """

    def __init__(self, expr: Placeholder, input_vars: List[Variable] | None = None):
        self.expr = expr
        self.input_vars = input_vars or []
        self.name: str | None = None
        self.op_id = _next_op_id()

        # Collect all variables from the expression to determine input signature
        self._collected_vars = self._collect_variables(expr)

        # Check if this operation has trainable layers
        self.has_trainable = self._has_trainable_layers(expr)

    def _collect_variables(self, expr) -> List[Variable]:
        """Collect all Variable and TensorTag placeholders from expression."""
        vars_found = []
        seen_ids = set()

        def visit(node):
            if isinstance(node, (Variable, TensorTag)):
                if id(node) not in seen_ids:
                    seen_ids.add(id(node))
                    vars_found.append(node)
            elif isinstance(node, ModelCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, BinaryOp):
                visit(node.left)
                visit(node.right)
            elif isinstance(node, FunctionCall):
                for arg in node.args:
                    if isinstance(arg, Placeholder):
                        visit(arg)
            elif isinstance(node, OperationCall):
                for arg in node.args:
                    visit(arg)
            elif isinstance(node, GroupedAssembly):
                if node.volume_value_expr is not None:
                    visit(node.volume_value_expr)
                if node.volume_grad_expr is not None:
                    visit(node.volume_grad_expr)
                for bnd_expr in node.boundary_value_exprs.values():
                    visit(bnd_expr)
            elif isinstance(node, Assembly):
                visit(node.expr)

        visit(expr)
        return vars_found

    def _has_trainable_layers(self, expr) -> bool:
        """Check if expression contains Model nodes."""

        def visit(node):
            if isinstance(node, Model):
                return True
            elif isinstance(node, ModelCall):
                return True  # ModelCall contains a trainable Model
            elif isinstance(node, BinaryOp):
                return visit(node.left) or visit(node.right)
            elif isinstance(node, FunctionCall):
                return any(visit(arg) for arg in node.args if isinstance(arg, Placeholder))
            elif isinstance(node, OperationCall):
                return node.operation.has_trainable
            elif isinstance(node, Assembly):
                return visit(node.expr)
            elif isinstance(node, GroupedAssembly):
                val_has = visit(node.volume_value_expr) if node.volume_value_expr is not None else False
                grad_has = visit(node.volume_grad_expr) if node.volume_grad_expr is not None else False
                bnd_has = any(visit(expr) for expr in node.boundary_value_exprs.values())
                return val_has or grad_has or bnd_has
            return False

        return visit(expr)

    def __call__(self, *args) -> "OperationCall":
        """Call operation with specific variables.

        If no arguments passed, uses original variables (no substitution).
        If fewer arguments passed than variables, fills remaining from original variables.
        Example: For Op with vars (x, t, a), calling Op(x0, t0) gives (x0, t0, a).
        """
        n_vars = len(self._collected_vars)
        if len(args) > n_vars:
            var_names = [str(v) for v in self._collected_vars]
            raise ValueError(
                f"Op[{self.op_id}] has {n_vars} variable(s) {var_names}, "
                f"but {len(args)} argument(s) were passed: {[str(a) for a in args]}"
            )
        # Fill in missing args from original variables
        if len(args) < n_vars:
            args = args + tuple(self._collected_vars[len(args) :])
        return OperationCall(self, args)

    def __repr__(self):
        return f"Op[{self.op_id}]"


class OperationCall(Placeholder):
    """A call to an operation with specific input variables.

    Example: u(x, y) where u is an OperationDef and x, y are Variables.
    """

    def __init__(self, operation: OperationDef, args: tuple):
        self.operation = operation
        self.args = args  # Variables or other OperationCalls
        self.op_id = _next_op_id()

    def __repr__(self):
        args_str = ", ".join(str(a) for a in self.args)
        return f"{self.operation}({args_str})"


class Hessian(Placeholder):
    """Second-order differential operator.

    When ``trace=False`` (default), computes the full Hessian matrix
    H[i,j] = d²u / (dxᵢ dxⱼ).

    When ``trace=True``, computes the Laplacian (trace of the Hessian):
    ∇²u = Σᵢ d²u/dxᵢ².
    """

    def __init__(
        self,
        target: "Placeholder",
        variables: Optional[List[Variable]] = None,
        scheme: str = "automatic_differentiation",
        trace: bool = False,
    ):
        self.target = target
        self.variables = variables if isinstance(variables, list) else [variables]
        self.scheme = scheme
        self.trace = trace  # True → Laplacian (sum of diagonal)
        weak_vars = [v for v in self.variables if isinstance(v, Placeholder)]
        _propagate_weak(self, target, *weak_vars)

    def __repr__(self):
        var_names = ", ".join(str(v) for v in self.variables)
        kind = "∇²" if self.trace else "Hessian"
        return f"{kind}({self.target}, [{var_names}])"


class Jacobian(Placeholder):
    """First-order differential operator.

    Computes J[i] = du/dxᵢ for each variable xᵢ.  When only one
    variable is supplied this is equivalent to a partial derivative
    (gradient), and the result is squeezed to a scalar per point.
    """

    def __init__(
        self,
        target: "Placeholder",
        variables: List[Union[Variable, None]],
        scheme: str = "automatic_differentiation",
    ):
        self.target = target
        self.variables = variables if isinstance(variables, list) else [variables]
        self.scheme = scheme
        weak_vars = [v for v in self.variables if isinstance(v, Placeholder)]
        _propagate_weak(self, target, *weak_vars)

    def __repr__(self):
        var_names = ", ".join(str(v) for v in self.variables)
        return f"Jacobian({self.target}, [{var_names}])"


class Integral(Placeholder):
    """Mesh-based integral reduction of an expression over its domain region.

    Created by :meth:`Placeholder.integrate`.  The region (boundary vs volume)
    is auto-detected at evaluation time from the Variable tags inside
    ``target`` via ``domain._boundary_registry``.

    When ``integration_var`` is set (the outer/collocation Variable), the
    evaluator uses ``jax.vmap`` to return an ``(N, 1)`` array instead of a
    scalar, enabling non-separable Fredholm kernels.
    """

    def __init__(self, target: "Placeholder", integration_var: "Variable | None" = None):
        self.target = target
        self.integration_var = integration_var
        _propagate_weak(self, target)

    def __repr__(self):
        if self.integration_var is not None:
            return f"Integral({self.target}, outer={self.integration_var})"
        return f"Integral({self.target})"


class IntegralTime(Placeholder):
    """Trapezoidal time integral of an expression over the current time window.

    Created by :meth:`Placeholder.integrate` when passed a temporal Variable.
    At evaluation time the handler reads ``context["__time_window__"]`` (shape
    ``(W, 1)``) injected by the compiler before the per-step vmap, evaluates
    ``target`` at each of the W time values with the current spatial context,
    applies trapezoidal weights, and sums to return the integral.

    Chain with a spatial :class:`Integral` for space-time integrals::

        u.integrate().integrate(t)   # ∫∫ u dΩ dt
    """

    def __init__(self, target: "Placeholder", time_var: "Variable"):
        self.target = target
        self.time_var = time_var
        _propagate_weak(self, target)

    def __repr__(self):
        return f"IntegralTime({self.target})"


class TemporalDerivative(Placeholder):
    """First-order time derivative via cross-step finite differences.

    Created by :class:`FieldView` when a temporally-tagged Variable is bound
    via ``.field.bind(t=t_var)``.  Higher-order temporal derivatives are
    expressed by chaining (``.tt`` → ``TemporalDerivative(TemporalDerivative(u, t), t)``).

    At evaluation time the handler reads:

    * ``ctx["__temporal_fd_cache__"][id(target)]`` — the pre-computed window
      ``u_window`` of shape ``(W, ...)`` injected by the compiler before the
      per-step vmap;
    * ``ctx["__step_index__"]`` — the current step index within the window;
    * ``ctx["__time_window__"]`` — the ``(W, 1)`` array of time values.

    Applies a clamped central difference at interior steps and one-sided
    differences at window edges.  Requires ``min_consecutive >= 2`` in
    :meth:`Crux.solve` (>= 3 recommended for proper central differences).
    """

    def __init__(self, target: "Placeholder", time_var: "Variable"):
        self.target = target
        self.time_var = time_var
        _propagate_weak(self, target)

    def __repr__(self):
        return f"TemporalDerivative({self.target})"


class NetworkGradient(Placeholder):
    """Per-point Jacobian of an expression w.r.t. a model's trainable parameters.

    Created by ``expr.grad(model)``.  Evaluated by :class:`TraceEvaluator`
    using ``jax.jacrev`` over ``eqx.partition``-ed parameters.

    **Post-training analysis** (via ``crux.eval``)::

        J = crux.eval([u.grad(net)])   # (B, N, P) — access J[0] for (N, P)

    **Training loss** (second-order AD, expensive)::

        loss = (u.grad(net) ** 2).mean()   # differentiates through jacrev

    **Training loss, stop-gradient variant** (first-order, cheaper)::

        loss = (jax.lax.stop_gradient(u.grad(net)) ** 2).mean()

    Output shape inside ``crux.eval``: ``(B, N, P)`` for scalar-output expressions;
    ``(B, N, D, P)`` for D-dimensional output.
    N = spatial points, P = number of selected trainable parameters.
    When *selector* is ``None``, P = all trainable array leaves flattened.
    """

    def __init__(self, target: Placeholder, model_node: "Model", selector=None):
        self.target = target
        self.model_node = model_node
        self.selector = selector  # None → all params; callable → eqx.tree_at selector

    def __repr__(self):
        sel = f", selector={self.selector!r}" if self.selector is not None else ""
        return f"NetworkGradient({self.target!r}, model={self.model_node!r}{sel})"


class Noise(Placeholder):
    """Stochastic noise term regenerated every training step.

    Created by :mod:`jno.noise`.  Produces an array of shape ``(N, ndim)``
    where ``N`` is inferred at evaluation time from the number of active
    spatial points and ``ndim`` (default 1) controls the trailing dimension.

    The realisation is derived from the solver's step PRNG key via
    ``jax.random.fold_in``, so it is fully reproducible when the global seed
    is fixed (via :func:`jno.setup` or ``.jno.toml``).

    Parameters
    ----------
    distribution : str
        ``'gaussian'``, ``'uniform'``, or ``'laplace'``.
    **params
        Distribution-specific kwargs: ``std``, ``low``, ``high``, ``ndim``.
    """

    def __init__(self, distribution: str, **params):
        self.distribution = distribution
        self.params = params
        self._noise_id = _next_op_id()

    def __repr__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Noise({self.distribution}, {params_str})"


class FemLinearSystem:
    """Container for a steady linear FEM system ``A(args) x = b(args)``."""

    def __init__(
        self,
        A,
        b,
        *,
        operator_fn=None,
        rhs_fn=None,
        runtime_parameter_exprs=None,
        operator_basis=None,
        rhs_basis=None,
        metadata=None,
    ):
        self.A = A
        self.b = b
        self.operator_fn = operator_fn
        self.rhs_fn = rhs_fn
        self.runtime_parameter_exprs = runtime_parameter_exprs or {}
        self.operator_basis = operator_basis or {}
        self.rhs_basis = rhs_basis or {}
        self.metadata = metadata or {}

    @property
    def is_parametric(self) -> bool:
        return self.operator_fn is not None or self.rhs_fn is not None

    def evaluate(self, args=None):
        A = self.A if self.operator_fn is None else self.operator_fn(args)
        b = self.b if self.rhs_fn is None else self.rhs_fn(args)
        return A, b

    def solve(self, solve_fn=None):
        """Differentiable forward solve ``u = solve_fn(A(θ), b(θ))`` as a trace node.

        Returns a :class:`FunctionCall` field. When it is evaluated (e.g. inside
        ``crux.solve``), any runtime parameters ``θ`` are resolved to their current
        values, :meth:`evaluate` forms ``A(θ), b(θ)``, and ``solve_fn`` solves them.
        Gradients flow back to the parameters through ``solve_fn`` — so an inverse
        problem is just ``crux([(fem.solve() - u_obs).mse], domain=...)`` where
        ``fem = jno.fem([...])`` (see ``docs/inverse-problems.md``).

        ``solve_fn`` is **your** solver: any ``(A, b) -> u`` callable. jNO writes no
        solver code and imposes no library — the default is
        :func:`jax.numpy.linalg.solve` (dense); pass e.g.
        ``lambda A, b: lineax.linear_solve(lineax.MatrixLinearOperator(A), b).value`` to
        choose another. Use a differentiable solver so ``∂u/∂θ`` exists.

        Note: the FEM solve is global (one ``A``, ``b``, ``u``); enable x64
        (``jax_enable_x64``) and set the parameter dtype to match — the feax
        assembly is float64.
        """
        if solve_fn is None:

            def solve_fn(A, b):
                return jnp.linalg.solve(A, b)

        names = list(self.runtime_parameter_exprs)
        params = [self.runtime_parameter_exprs[n] for n in names]

        def _solve(*values):
            A, b = self.evaluate(dict(zip(names, values)))
            return solve_fn(jnp.asarray(A), jnp.asarray(b).reshape(-1))

        return FunctionCall(_solve, params, name="fem_solve")

    def __iter__(self):
        if self.is_parametric:
            raise TypeError(
                "This FEM system depends on runtime parameters. Call system.evaluate(args={...}) before unpacking."
            )
        yield self.A
        yield self.b

    def __add__(self, other):
        if not isinstance(other, FemLinearSystem):
            return NotImplemented
        if self.is_parametric or other.is_parametric:
            raise TypeError(
                "Addition of parameter-aware FemLinearSystem blocks is not "
                "implemented. Combine the weak forms before assembly."
            )
        return FemLinearSystem(self.A + other.A, self.b + other.b)

    def todense(self, args=None):
        A, b = self.evaluate(args=args)
        A_dense = A.todense() if hasattr(A, "todense") else A
        return A_dense, b

    def __sub__(self, other):
        if not isinstance(other, FemLinearSystem):
            return NotImplemented
        if self.is_parametric or other.is_parametric:
            raise TypeError(
                "Subtraction of parameter-aware FemLinearSystem blocks is not "
                "implemented. Combine the weak forms before assembly."
            )
        return FemLinearSystem(self.A - other.A, self.b - other.b)

    def __repr__(self):
        shape = getattr(self.A, "shape", None)
        return f"FemLinearSystem(shape={shape}, b_shape={getattr(self.b, 'shape', None)}, parametric={self.is_parametric})"


class FemResidualOperator:
    """Container for a steady nonlinear FEM residual ``R(u, args) = 0``."""

    def __init__(
        self,
        residual_fn,
        jacobian_fn=None,
        size=None,
        *,
        runtime_parameter_exprs=None,
        residual_basis=None,
        metadata=None,
    ):
        self.residual = residual_fn
        self.jacobian = jacobian_fn
        self.size = size
        self.runtime_parameter_exprs = runtime_parameter_exprs or {}
        self.residual_basis = residual_basis or {}
        self.metadata = metadata or {}

    @property
    def is_parametric(self) -> bool:
        return bool(self.runtime_parameter_exprs)

    def __call__(self, u, args=None):
        return self.residual(u, args)

    def linearize(self, u, args=None):
        if self.jacobian is None:
            raise ValueError("No jacobian function available.")

        return (
            self.jacobian(u, args),
            -self.residual(u, args),
        )

    def solve(self, solve_fn=None, *, u0=None):
        """Differentiable nonlinear forward solve of ``R(u, θ) = 0`` as a trace node.

        Returns a :class:`FunctionCall` field. When evaluated (e.g. inside
        ``crux.solve``), runtime parameters ``θ`` are resolved to their current
        values, ``residual_fn(u) = R(u, θ)`` is built, and ``solve_fn`` solves it.
        Gradients reach the parameters when ``solve_fn`` is implicit-diff aware, so
        an inverse problem is ``crux([(fem.solve() - u_obs).mse], domain=...)`` where
        ``fem = jno.fem([...])``.

        ``solve_fn`` is **your** solver: any ``(residual_fn, u0) -> u`` callable. The
        default is a Newton ``optimistix.root_find`` (implicit differentiation, so
        ``∂u/∂θ`` is exact without unrolling Newton); pass your own to choose a
        different optimistix solver or library. jNO's analytic Jacobian
        (:attr:`jacobian`) is available as an optional speed-up — by default the
        solver auto-differentiates the residual.

        ``u0`` is the initial guess (default: zeros of the operator size; enable
        x64 — the feax residual is float64).
        """
        if u0 is None:
            if self.size is None:
                raise ValueError("FemResidualOperator.solve: pass u0= (operator size is unknown).")
            u0 = jnp.zeros((int(self.size),), dtype=jnp.result_type(float))

        if solve_fn is None:

            def solve_fn(residual_fn, y0):
                import optimistix as optx

                sol = optx.root_find(
                    lambda u, _a: residual_fn(u),
                    optx.Newton(rtol=1e-8, atol=1e-8),
                    y0,
                    args=None,
                    max_steps=100,
                )
                return sol.value

        names = list(self.runtime_parameter_exprs)
        params = [self.runtime_parameter_exprs[n] for n in names]

        def _solve(*values):
            args = dict(zip(names, values))
            return solve_fn(lambda u: self.residual(u, args), u0)

        return FunctionCall(_solve, params, name="fem_solve")

    def __repr__(self):
        return (
            "FemResidualOperator("
            f"size={self.size}, "
            f"has_jacobian={self.jacobian is not None}, "
            f"parametric={self.is_parametric}"
            ")"
        )


class StateField(Placeholder):
    """Internal marker for the primary weak-form unknown.

    This wraps the original NN-valued expression so the same weak graph can be
    lowered either to VPINN (bind back to expr) or FEM (replace by TrialFunction).
    """

    def __init__(
        self,
        expr: Placeholder,
        *,
        state_id: int = 0,
        name: str = "u",
        value_shape: tuple = (),
    ):
        self.expr = expr
        self.state_id = int(state_id)
        self.name = name
        self.value_shape = tuple(value_shape)

    def __repr__(self):
        return f"StateField(name={self.name!r}, id={self.state_id}, shape={self.value_shape})"


class GaugePin:
    """Marker that gauge-fixes a field's constant null space (created by ``trial.pin()``).

    An incompressible pressure -- or any pure-Neumann scalar -- is determined only up to an
    additive constant, so its discrete operator has a one-dimensional (constant) null space and
    the saddle system is singular. ``p.pin(value)`` removes it by fixing a single, *arbitrary*
    degree of freedom to ``value``: this is **gauge-fixing**, not a boundary condition. ``jno.fem``
    lowers each pin to a single-node Dirichlet ``p(node) - value`` at a deterministic vertex
    (nearest the mesh min-corner) -- the same essential path the explicit ``p(xpn, ypn) - value``
    form takes -- so assembly is unchanged. The location is intentionally not user-specified;
    any single DOF removes the null space.
    """

    __slots__ = ("field", "value")

    def __init__(self, field, value=0.0):
        self.field = field
        self.value = value

    def __repr__(self):
        return f"GaugePin(field={getattr(self.field, 'name', '?')!r}, value={self.value!r})"


class TrialFunction(Placeholder):
    """
    Generic variational unknown symbol.

    Interpretation depends on assembly target:
      - vpinn         -> interpreted via model/u_net
      - fem_system    -> interpreted as FE trial function
      - fem_residual  -> interpreted as FE unknown in a nonlinear residual operator

    Parameters
    ----------
    name : str
        Symbol name used for printing/debugging.
    value_shape : tuple
        Shape of the field value at one spatial point:
          ()    -> scalar
          (2,)  -> 2D vector
          (3,)  -> 3D vector
          (2,2) -> second-order tensor, etc.
    """

    def __init__(self, name="u", value_shape=(), order=1):
        self.name = name
        self.value_shape = tuple(value_shape)
        self.order = int(order)  # element polynomial degree for this field (P1=1, P2=2)
        self.op_id = _next_op_id()
        # Identifies the field this symbol belongs to; a (trial, test) pair from one
        # fem_symbols() call shares a key so the coupled kernel can pair u<->v.
        self.field_key = self.op_id

    @property
    def num_components(self) -> int:
        if len(self.value_shape) == 0:
            return 1
        n = 1
        for s in self.value_shape:
            n *= int(s)
        return n

    def _field_view(self):
        n = len(self.value_shape)
        if n == 0:
            return self.scalar
        if n == 1:
            return self.vector
        return self.matrix

    def partials(self, **named_vars):
        """Bind named coordinate Variables for attribute-style derivatives.

        ``u.bind(x=xi, y=yi).x`` -> ``du/dx`` (a ``Jacobian`` node identical to
        ``u.d(xi)``); ``.xx`` / ``.xy`` give higher partials. Mirrors the PINN
        field idiom ``net(x).scalar.bind(x=x)``.
        """
        return self._field_view().partials(**named_vars)

    bind = partials

    def pin(self, value=0.0):
        """Gauge-fix this field's constant null space by pinning one arbitrary DOF to ``value``.

        For an incompressible pressure or a pure-Neumann scalar, whose solution is defined only
        up to an additive constant. Drop the result straight into the ``jno.fem`` constraint
        list -- no ``domain.point_region`` / coordinate plumbing needed::

            fem = jno.fem([momentum, -q * div(u), p.pin(), *wall_bcs])

        ``jno.fem`` pins a deterministic vertex (nearest the mesh min-corner), so the gauge is
        reproducible; the location is intentionally not user-specified -- any single DOF removes
        the null space. See :class:`GaugePin`.
        """
        if self.num_components != 1:
            raise ValueError(
                "jno.fem: pin() gauge-fixes a *scalar* field's constant null space, but "
                f"{self.name!r} has value_shape {self.value_shape}. Pin a scalar field "
                "(e.g. the pressure); a fully Dirichlet vector field has no null space to fix."
            )
        return GaugePin(self, value)

    def __call__(self, *coords, **named):
        """Evaluate this field symbol on the region carried by ``coords``.

        Positional coordinates bind to ``x`` / ``y`` / ``z`` in order; extra
        keyword bindings (e.g. ``t=...``) pass through. ``u(x, y)`` is sugar for
        ``u.bind(x=x, y=y)`` so the same gesture works as in a PINN (``net(x)``).
        """
        binding = {axis: c for axis, c in zip(("x", "y", "z"), coords)}
        binding.update(named)
        if not binding:
            raise TypeError(f"{type(self).__name__} must be called with coordinate variables, e.g. u(x, y).")
        return self.partials(**binding)

    def __repr__(self):
        return f"TrialFunction({self.name}, value_shape={self.value_shape})"


class TestFunction(Placeholder):
    """
    Generic variational test function.

    It is resolved against the active bucket (volume or boundary region)
    during evaluation/assembly.

    Parameters
    ----------
    name : str
        Symbol name used for printing/debugging.
    value_shape : tuple
        Shape of the field value at one spatial point:
          ()    -> scalar
          (2,)  -> 2D vector
          (3,)  -> 3D vector
          (2,2) -> second-order tensor, etc.
    """

    def __init__(self, name="phi", value_shape=(), order=1):
        self.name = name
        self.value_shape = tuple(value_shape)
        self.order = int(order)  # element polynomial degree for this field (P1=1, P2=2)
        self.op_id = _next_op_id()
        # Shared with the paired trial (set by variational_symbols) to identify the field.
        self.field_key = self.op_id

    @property
    def num_components(self) -> int:
        if len(self.value_shape) == 0:
            return 1
        n = 1
        for s in self.value_shape:
            n *= int(s)
        return n

    def _field_view(self):
        n = len(self.value_shape)
        if n == 0:
            return self.scalar
        if n == 1:
            return self.vector
        return self.matrix

    def partials(self, **named_vars):
        """Bind named coordinate Variables for attribute-style derivatives.

        ``phi.bind(x=xi, y=yi).x`` -> ``dphi/dx`` (a ``Jacobian`` node identical
        to ``phi.d(xi)``); ``.xx`` / ``.xy`` give higher partials.
        """
        return self._field_view().partials(**named_vars)

    bind = partials

    def pin(self, value=0.0):
        """Reject ``pin()`` on a test function -- only the unknown has a null space to fix."""
        raise ValueError(
            "jno.fem: pin() gauge-fixes the unknown's constant null space -- call it on the "
            "trial symbol (e.g. p.pin()), not the test function."
        )

    def __call__(self, *coords, **named):
        """Evaluate this test-function symbol on the region carried by ``coords``.

        Positional coordinates bind to ``x`` / ``y`` / ``z`` in order; extra
        keyword bindings pass through. ``phi(x, y)`` is sugar for
        ``phi.bind(x=x, y=y)``.
        """
        binding = {axis: c for axis, c in zip(("x", "y", "z"), coords)}
        binding.update(named)
        if not binding:
            raise TypeError(f"{type(self).__name__} must be called with coordinate variables, e.g. phi(x, y).")
        return self.partials(**binding)

    def __repr__(self):
        return f"TestFunction({self.name}, value_shape={self.value_shape})"


class Assembly(Placeholder):
    """
    Internal assembly node for one already-bucketed variational contribution.
    """

    def __init__(self, expr: Placeholder, domain_or_nodes, support: str, region_id: str):
        self.expr = expr
        self.support = support  # "volume" | "boundary"
        self.region_id = region_id  # e.g. "cells", "right", "wall_3", ...
        if hasattr(domain_or_nodes, "context"):
            self.num_total_nodes = int(domain_or_nodes.context["num_total_nodes"])
        else:
            self.num_total_nodes = int(domain_or_nodes)
        self.op_id = _next_op_id()

    def __repr__(self):
        return f"Assemble({self.expr}, support={self.support}, region={self.region_id}, nodes={self.num_total_nodes})"


class GroupedAssembly(Placeholder):
    """
    Internal node for grouped FEAX-style variational assembly.

    Separate channels:
      - volume_value_expr   : terms multiplied by TestFunction(phi)
      - volume_grad_expr    : terms multiplied by grad(TestFunction(phi))
      - boundary_value_exprs: boundary terms multiplied by TestFunction(phi)
    """

    def __init__(self, volume_value_expr, volume_grad_expr, boundary_value_exprs, domain_or_nodes):
        self.volume_value_expr = volume_value_expr  # Placeholder | None
        self.volume_grad_expr = volume_grad_expr  # Placeholder | None
        self.boundary_value_exprs = boundary_value_exprs or {}  # dict[str, Placeholder]
        if hasattr(domain_or_nodes, "context"):
            self.num_total_nodes = int(domain_or_nodes.context["num_total_nodes"])
        else:
            self.num_total_nodes = int(domain_or_nodes)
        self.op_id = _next_op_id()

    def __repr__(self):
        bkeys = list(self.boundary_value_exprs.keys())
        return (
            f"GroupedAssembly(value={'yes' if self.volume_value_expr is not None else 'no'}, "
            f"grad={'yes' if self.volume_grad_expr is not None else 'no'}, "
            f"boundaries={bkeys}, nodes={self.num_total_nodes})"
        )


# =============================================================================
# Tree optimisation — Common Sub-expression Elimination (CSE)
# =============================================================================


def cse(expr: Placeholder) -> Placeholder:
    """Eliminate common sub-expressions in a traced computation tree.

    Walks *expr* bottom-up and replaces structurally identical sub-trees
    with a single shared Python object.  Two nodes are considered
    identical when they have the same type, the same static attributes
    (operator, function identity, op_id, …) **and** the same children
    (by ``id``).

    The pass is safe to run multiple times and never changes semantics.

    What gets deduplicated:

    * ``OperationCall`` — same ``OperationDef`` + same argument
      ``Variable``/``TensorTag`` objects (by identity).
    * ``BinaryOp`` — same operator + same (already-deduped) children.
    * ``FunctionCall`` — same ``fn`` + same (already-deduped) args.
    * ``Jacobian`` / ``Hessian`` — same target + same variables.
    * ``ModelCall`` — same model + same args.

    Returns:
        A (possibly shared) tree with duplicates collapsed.
    """
    # Maps a structural key → canonical node that was already built.
    _canon: dict = {}

    def _key(node):
        """Return a hashable key for *node* assuming children are already canonical."""
        if isinstance(node, Variable):
            return ("Var", id(node))
        if isinstance(node, TensorTag):
            return ("Tag", id(node))
        if isinstance(node, (Constant, Literal)):
            return ("Const", id(node))
        if isinstance(node, BinaryOp):
            return ("Bin", node.op, id(node.left), id(node.right))
        if isinstance(node, FunctionCall):
            arg_ids = tuple(id(a) for a in node.args)
            return ("Fn", id(node.fn), node._name, arg_ids)
        if isinstance(node, Choice):
            opt_ids = tuple(id(a) for a in node.options)
            return ("Choice", node.name, node.selected, opt_ids)
        if isinstance(node, ModelCall):
            arg_ids = tuple(id(a) for a in node.args)
            return ("Call", node.model.layer_id, arg_ids)
        if isinstance(node, OperationCall):
            arg_ids = tuple(id(a) for a in node.args)
            return ("OpCall", node.operation.op_id, arg_ids)
        if isinstance(node, OperationDef):
            return ("OpDef", node.op_id, id(node.expr))
        if isinstance(node, Jacobian):
            var_ids = tuple(id(v) for v in node.variables)
            return ("Jac", id(node.target), var_ids, node.scheme)
        if isinstance(node, Hessian):
            var_ids = tuple(id(v) for v in node.variables)
            return ("Hess", id(node.target), var_ids, node.trace, node.scheme)
        if isinstance(node, Tracker):
            return ("Track", id(node.expr), node.interval)
        if isinstance(node, TrialFunction):
            return ("TrialFn", node.name, id(node))
        if isinstance(node, TestFunction):
            return ("TestFn", node.name, id(node))
        if isinstance(node, Assembly):
            return (
                "Assembly",
                node.support,
                node.region_id,
                node.num_total_nodes,
                id(node.expr),
            )
        if isinstance(node, GroupedAssembly):
            return (
                "GroupedAssembly",
                id(node.volume_value_expr) if node.volume_value_expr is not None else None,
                id(node.volume_grad_expr) if node.volume_grad_expr is not None else None,
                tuple((k, id(v)) for k, v in sorted(node.boundary_value_exprs.items())),
                node.num_total_nodes,
            )

    def _visit(node):
        """Post-order walk: canonicalise children first, then self."""
        # Leaves — always canonical
        if isinstance(node, (Variable, TensorTag, Constant, Literal, TrialFunction, TestFunction)):
            return node

        # ── recurse into children and rebuild if anything changed ──
        if isinstance(node, BinaryOp):
            left_node = _visit(node.left)
            r = _visit(node.right)
            if left_node is not node.left or r is not node.right:
                node = BinaryOp(node.op, left_node, r)
        elif isinstance(node, FunctionCall):
            new_args = [_visit(a) if isinstance(a, Placeholder) else a for a in node.args]
            if any(n is not o for n, o in zip(new_args, node.args)):
                node = FunctionCall(node.fn, new_args, node._name, node.reduces_axis, node.kwargs)
        elif isinstance(node, Choice):
            new_opts = [_visit(a) if isinstance(a, Placeholder) else a for a in node.options]
            if any(n is not o for n, o in zip(new_opts, node.options)):
                new_node = Choice(new_opts, name=node.name, default=node.selected)
                new_node.op_id = node.op_id
                node = new_node
        elif isinstance(node, ModelCall):
            new_args = [_visit(a) if isinstance(a, Placeholder) else a for a in node.args]
            if any(n is not o for n, o in zip(new_args, node.args)):
                new_node = ModelCall(node.model, new_args)
                new_node.op_id = node.op_id
                node = new_node
        elif isinstance(node, OperationDef):
            new_expr = _visit(node.expr)
            if new_expr is not node.expr:
                new_node = OperationDef.__new__(OperationDef)
                new_node.expr = new_expr
                new_node.input_vars = node.input_vars
                new_node.op_id = node.op_id
                new_node._collected_vars = node._collected_vars
                new_node.has_trainable = node.has_trainable
                node = new_node
        elif isinstance(node, OperationCall):
            new_op = _visit(node.operation)
            new_args = tuple(_visit(a) if isinstance(a, Placeholder) else a for a in node.args)
            if new_op is not node.operation or any(n is not o for n, o in zip(new_args, node.args)):
                node = OperationCall(new_op, new_args)
        elif isinstance(node, Jacobian):
            new_target = _visit(node.target)
            if new_target is not node.target:
                node = Jacobian(new_target, node.variables, node.scheme)
        elif isinstance(node, Hessian):
            new_target = _visit(node.target)
            if new_target is not node.target:
                node = Hessian(new_target, node.variables, node.scheme, node.trace)
        elif isinstance(node, Tracker):
            new_expr = _visit(node.expr)
            if new_expr is not node.expr:
                node = Tracker(new_expr, node.interval)
        elif isinstance(node, Assembly):
            new_expr = _visit(node.expr)
            if new_expr is not node.expr:
                node = Assembly(
                    new_expr,
                    node.num_total_nodes,
                    support=node.support,
                    region_id=node.region_id,
                )
        elif isinstance(node, GroupedAssembly):
            new_volume_value = _visit(node.volume_value_expr) if node.volume_value_expr is not None else None
            new_volume_grad = _visit(node.volume_grad_expr) if node.volume_grad_expr is not None else None
            new_boundary = {}
            changed = new_volume_value is not node.volume_value_expr or new_volume_grad is not node.volume_grad_expr

            for region_id, bnd_expr in node.boundary_value_exprs.items():
                new_expr = _visit(bnd_expr)
                new_boundary[region_id] = new_expr
                if new_expr is not bnd_expr:
                    changed = True

            if changed:
                node = GroupedAssembly(
                    new_volume_value,
                    new_volume_grad,
                    new_boundary,
                    node.num_total_nodes,
                )
        # ── dedup: if we've seen an identical node, return the earlier one ──
        k = _key(node)
        if k in _canon:
            return _canon[k]
        _canon[k] = node
        return node

    return _visit(expr)


# =============================================================================
# Evaluation engine
# =============================================================================


def collect_operations(expr: Placeholder) -> List[OperationDef]:
    """Collect all OperationDef instances from an expression."""
    ops = []
    seen = set()

    def visit(node):
        if isinstance(node, OperationDef):
            if node.op_id not in seen:
                seen.add(node.op_id)
                ops.append(node)
            visit(node.expr)
        elif isinstance(node, OperationCall):
            if node.operation.op_id not in seen:
                seen.add(node.operation.op_id)
                ops.append(node.operation)
            visit(node.operation.expr)
            for arg in node.args:
                visit(arg)
        elif isinstance(node, ModelCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, TunableModuleCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, Choice):
            for opt in node.options:
                if isinstance(opt, Placeholder):
                    visit(opt)
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, (Hessian, Jacobian)):
            visit(node.target)
            for v in node.variables:
                visit(v)
        elif isinstance(node, NetworkGradient):
            visit(node.target)
        elif isinstance(node, (Integral, IntegralTime)):
            visit(node.target)
        elif isinstance(node, TemporalDerivative):
            visit(node.target)
            if isinstance(node.time_var, Placeholder):
                visit(node.time_var)
        elif isinstance(node, Tracker):
            visit(node.expr)
        elif isinstance(node, Assembly):
            visit(node.expr)
        elif isinstance(node, GroupedAssembly):
            if node.volume_value_expr is not None:
                visit(node.volume_value_expr)
            if node.volume_grad_expr is not None:
                visit(node.volume_grad_expr)
            for bnd_expr in node.boundary_value_exprs.values():
                visit(bnd_expr)

    visit(expr)
    return ops


def collect_tags(expr: Placeholder) -> set:
    """Collect all unique tags from Variables in the expression tree."""
    tags = set()

    def visit(node):
        if isinstance(node, Variable):
            tags.add(node.tag)
        elif isinstance(node, TensorTag):
            tags.add(node.tag)
        elif isinstance(node, BinaryOp):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, FunctionCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, OperationDef):
            visit(node.expr)
        elif isinstance(node, OperationCall):
            visit(node.operation.expr)
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, ModelCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, TunableModuleCall):
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    visit(arg)
        elif isinstance(node, Choice):
            for opt in node.options:
                if isinstance(opt, Placeholder):
                    visit(opt)
        elif isinstance(node, (Hessian, Jacobian)):
            visit(node.target)
            for v in node.variables:
                visit(v)
        elif isinstance(node, NetworkGradient):
            visit(node.target)
        elif isinstance(node, (Integral, IntegralTime)):
            visit(node.target)
        elif isinstance(node, TemporalDerivative):
            visit(node.target)
            if isinstance(node.time_var, Placeholder):
                visit(node.time_var)
        elif isinstance(node, Tracker):
            visit(node.expr)
        elif isinstance(node, (TrialFunction, TestFunction)):
            pass
        elif isinstance(node, Assembly):
            visit(node.expr)
        elif isinstance(node, GroupedAssembly):
            if node.volume_value_expr is not None:
                visit(node.volume_value_expr)
            if node.volume_grad_expr is not None:
                visit(node.volume_grad_expr)
            for bnd_expr in node.boundary_value_exprs.values():
                visit(bnd_expr)

    visit(expr)
    return tags


def get_primary_tag(expr: Placeholder) -> str:
    """Return the first Variable tag found in the expression tree."""
    tags = collect_tags(expr)
    return next(iter(tags)) if tags else None


def dump_tree(expr, indent: int = 0, seen: set = None) -> str:
    """Return a human-readable indented string of the expression tree.

    Args:
        expr:   Any trace node (Placeholder subclass).
        indent: Current indentation level (used by recursion).
        seen:   Set of already-visited ``OperationDef.op_id`` values to
                avoid infinite recursion on shared sub-graphs.

    Returns:
        Multi-line string with the full computation tree.

    Example::

        tree_str = dump_tree(pde)
        print(tree_str)
        # or
        with open("tree.txt", "w") as f:
            f.write(tree_str)
    """
    if seen is None:
        seen = set()
    lines: list[str] = []

    def _node_label(node) -> str:
        """One-line label for a node (no children)."""
        if isinstance(node, Variable):
            return f"Variable({node.tag}[{node.dim}])"
        if isinstance(node, TensorTag):
            return f"TensorTag({node.tag})"
        if isinstance(node, Constant):
            val = node.value
            if hasattr(val, "shape") and val.shape == ():
                val = float(val)
            return f"Constant({node.tag}.{node.key}={val})"
        if isinstance(node, Literal):
            return f"Literal({node.value})"
        if isinstance(node, TrialFunction):
            return f"TrialFunction({node.name})"
        if isinstance(node, TestFunction):
            return f"TestFunction({node.name})"
        if isinstance(node, Model):
            return f"Model(id={node.layer_id}, {type(node.module).__name__})"
        if isinstance(node, (int, float)):
            return str(node)
        return type(node).__name__

    def _visit(node, depth):
        p = "  " * depth
        if isinstance(node, Variable):
            lines.append(f"{p}{_node_label(node)}")
        elif isinstance(node, TensorTag):
            lines.append(f"{p}{_node_label(node)}")
        elif isinstance(node, (Constant, Literal)):
            lines.append(f"{p}{_node_label(node)}")
        elif isinstance(node, BinaryOp):
            lines.append(f"{p}BinaryOp({node.op})")
            _visit(node.left, depth + 1)
            _visit(node.right, depth + 1)
        elif isinstance(node, FunctionCall):
            name = node._name or getattr(node.fn, "__name__", "fn")
            lines.append(f"{p}FunctionCall({name})")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _visit(arg, depth + 1)
                else:
                    lines.append(f"{p}  {arg}")
        elif isinstance(node, ModelCall):
            lines.append(f"{p}ModelCall({_node_label(node.model)})")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _visit(arg, depth + 1)
                else:
                    lines.append(f"{p}  {arg}")
        elif isinstance(node, TunableModuleCall):
            lines.append(f"{p}TunableModuleCall(id={node.model.layer_id})")
            for arg in node.args:
                if isinstance(arg, Placeholder):
                    _visit(arg, depth + 1)
        elif isinstance(node, Choice):
            lines.append(f"{p}Choice(name={node.name}, selected={node.selected})")
            for i, opt in enumerate(node.options):
                lines.append(f"{p}  option[{i}]")
                _visit(opt, depth + 2)
        elif isinstance(node, OperationDef):
            if node.op_id in seen:
                vars_str = ", ".join(str(v) for v in node._collected_vars)
                lines.append(f"{p}Op[{node.op_id}]({vars_str})  [already shown]")
                return
            seen.add(node.op_id)
            vars_str = ", ".join(str(v) for v in node._collected_vars)
            lines.append(f"{p}OperationDef[{node.op_id}] vars=({vars_str})")
            _visit(node.expr, depth + 1)
        elif isinstance(node, OperationCall):
            args_str = ", ".join(str(a) for a in node.args)
            lines.append(f"{p}OperationCall[{node.operation.op_id}]({args_str})")
            _visit(node.operation, depth + 1)
        elif isinstance(node, Hessian):
            kind = "Laplacian" if node.trace else "Hessian"
            vars_str = ", ".join(str(v) for v in node.variables)
            lines.append(f"{p}{kind}([{vars_str}])")
            _visit(node.target, depth + 1)
        elif isinstance(node, Jacobian):
            vars_str = ", ".join(str(v) for v in node.variables)
            lines.append(f"{p}Jacobian([{vars_str}])")
            _visit(node.target, depth + 1)
        elif isinstance(node, Tracker):
            lines.append(f"{p}Tracker(interval={node.interval})")
            _visit(node.expr, depth + 1)
        elif isinstance(node, (TrialFunction, TestFunction)):
            lines.append(f"{p}{_node_label(node)}")
        elif isinstance(node, Assembly):
            lines.append(f"{p}Assembly(support={node.support}, region={node.region_id})")
            _visit(node.expr, depth + 1)
        elif isinstance(node, GroupedAssembly):
            lines.append(f"{p}GroupedAssembly(nodes={node.num_total_nodes})")
            if node.volume_value_expr is not None:
                lines.append(f"{p}  volume_value:")
                _visit(node.volume_value_expr, depth + 2)
            if node.volume_grad_expr is not None:
                lines.append(f"{p}  volume_grad:")
                _visit(node.volume_grad_expr, depth + 2)
            for region_id, bnd_expr in node.boundary_value_exprs.items():
                lines.append(f"{p}  boundary_value[{region_id}]:")
                _visit(bnd_expr, depth + 2)
        elif isinstance(node, ConstantNamespace):
            lines.append(f"{p}ConstantNamespace({node._full_tag})")
        elif isinstance(node, Placeholder):
            # Fallback for any unknown Placeholder subclass
            lines.append(f"{p}{repr(node)}")
        else:
            lines.append(f"{p}{node}")

    _visit(expr, indent)
    return "\n".join(lines)


# Re-export typed semantic views. Imported at module bottom so that all
# classes referenced inside views.py (FunctionCall, Placeholder) already exist.
from .views import (  # noqa: E402
    _VIEW_TYPES,
    ComplexView,
    FieldView,
    FieldViewWithPartials,
    MatrixView,
    NamedComplexViewWithPartials,
    NamedMatrixView,
    NamedMatrixViewWithPartials,
    NamedScalarViewWithPartials,
    NamedVectorView,
    NamedVectorViewWithPartials,
    NamedVoigtViewWithPartials,
    ScalarView,
    VectorView,
    VoigtView,
)
