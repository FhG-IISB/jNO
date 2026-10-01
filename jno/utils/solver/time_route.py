from __future__ import annotations

"""
Internal time-dependent solver routing helpers.

Provides the small set of utilities consumed by jNO's FEM transient assembly:
- infer the time window from the domain or kwargs,
- strip temporal trial-derivative wrappers during semidiscrete assembly.

This module is internal. User-facing code assembles transient FEM problems
through ``jno.fem([...])``.
"""
from typing import Any, Tuple

from ...trace import (
    BinaryOp,
    FunctionCall,
    Hessian,
    Jacobian,
    Placeholder,
    TrialFunction,
)
from .solver_helper import (
    is_temporal_var as _is_temporal_var,
)

# -----------------------------------------------------------------------------
# Time metadata helpers
# -----------------------------------------------------------------------------


def _infer_time_window(domain, **kwargs) -> Tuple[float, float, float | None]:
    """
    Infer the time interval and default time-step from the domain or kwargs.
    Priority:
    - `kwargs["t0"]`, `kwargs["t1"]`, `kwargs["dt"]` / `kwargs["dt0"]`
    - `domain.time = (t0, t1, n_steps)`
    - fallback `(0.0, 1.0, None)`

    Returns:
        `(t0, t1, dt_default)`.
    """
    if getattr(domain, "time", None) is not None:
        t0, t1, n_steps = domain.time
        if n_steps is None or int(n_steps) <= 1:
            dt_default = None
        else:
            dt_default = float(t1 - t0) / float(int(n_steps) - 1)
    else:
        t0, t1, dt_default = 0.0, 1.0, None

    t0 = float(kwargs.get("t0", t0))
    t1 = float(kwargs.get("t1", t1))
    dt_default = kwargs.get("dt", kwargs.get("dt0", dt_default))
    if dt_default is not None:
        dt_default = float(dt_default)

    return t0, t1, dt_default


# -----------------------------------------------------------------------------
# Semidiscrete-time weak-form helpers
# -----------------------------------------------------------------------------


def _is_temporal_jacobian_of_trial(node: Any) -> bool:
    return (
        isinstance(node, Jacobian)
        and isinstance(node.target, TrialFunction)
        and any(_is_temporal_var(v) for v in node.variables)
    )


def _strip_temporal_trial_derivative(node: Any) -> Any:
    """
    Replace d/dt(TrialFunction) with TrialFunction.

    Used when converting first-order weak transient terms like
        ∫ u_t * phi
    into a spatial mass operator
        ∫ u * phi
    during semidiscrete assembly.
    """
    if _is_temporal_jacobian_of_trial(node):
        return node.target

    if isinstance(node, BinaryOp):
        return BinaryOp(
            node.op,
            _strip_temporal_trial_derivative(node.left),
            _strip_temporal_trial_derivative(node.right),
        )

    if isinstance(node, FunctionCall):
        new_args = [_strip_temporal_trial_derivative(a) if isinstance(a, Placeholder) else a for a in node.args]
        if hasattr(node, "copy_with_args"):
            return node.copy_with_args(new_args)
        return FunctionCall(
            node.fn,
            new_args,
            name=getattr(node, "_name", None),
            reduces_axis=getattr(node, "reduces_axis", None),
            kwargs=getattr(node, "kwargs", None),
        )

    if isinstance(node, Jacobian):
        return Jacobian(
            _strip_temporal_trial_derivative(node.target),
            [_strip_temporal_trial_derivative(v) if isinstance(v, Placeholder) else v for v in node.variables],
            node.scheme,
        )

    if isinstance(node, Hessian):
        return Hessian(
            _strip_temporal_trial_derivative(node.target),
            [_strip_temporal_trial_derivative(v) if isinstance(v, Placeholder) else v for v in node.variables],
            node.scheme,
            trace=node.trace,
        )

    return node


#: Operations LINEAR in each argument: a rate inside one of them keeps the term linear in the rate (the degree
#: adds up across the arguments, as in a product). Anything else that wraps a rate -- sqrt, sin, abs, where, a
#: power -- is treated as nonlinear in it.
_RATE_LINEAR_CALLS = {"inner", "einsum", "trace", "transpose", "sym", "antisym", "matmul", "dot", "sum", "mean",
                      "reshape", "squeeze", "expand_dims", "getitem", "negative", "multiply", "symgrad"}  # fmt: skip
_RATE_STACKING_CALLS = {"stack", "concat", "concatenate"}  # the degree of the stack is the largest part's


def _rate_degree(node: Any):
    """How many time-derivative factors ``u_t`` the product ``node`` carries -- ``None`` when ``u_t`` sits
    inside an operation that is not linear in it (``sqrt(u_t)``, ``u_t**2``, ``where(..., u_t, ...)``)."""
    if _is_temporal_jacobian_of_trial(node):
        return 1
    if isinstance(node, BinaryOp):
        a, b = _rate_degree(node.left), _rate_degree(node.right)
        if a is None or b is None:
            return None
        if node.op in ("*", "@"):
            return a + b
        if node.op in ("+", "-"):
            return max(a, b)
        if node.op == "/":
            return a if b == 0 else None
        return None if (a or b) else 0  # a power, a comparison, ... of a rate
    if isinstance(node, FunctionCall):
        degs = [_rate_degree(x) if isinstance(x, Placeholder) else 0 for x in node.args]
        if not any(d is None or d > 0 for d in degs):
            return 0
        if any(d is None for d in degs):
            return None
        name = getattr(node, "_name", None) or getattr(getattr(node, "fn", None), "__name__", "")
        if name in _RATE_LINEAR_CALLS:
            return sum(degs)
        if name in _RATE_STACKING_CALLS:
            return max(degs)
        return None
    if isinstance(node, (Jacobian, Hessian)):
        return _rate_degree(node.target)
    from .solver_helper import iter_children

    degs = [_rate_degree(c) for c in (iter_children(node) or ())]
    if any(d is None for d in degs):
        return None
    return max(degs, default=0)


def refuse_nonlinear_in_rate(node: Any, where: str = "jno.fem") -> None:
    """Raise unless the transient term ``node`` is LINEAR in the time derivative ``u_t``.

    A first-order march writes every term as a mass action ``M(u) u_t``: the constant-mass path reads ``M``
    off the derivative of the term at ``u_t = 0``, and the state-dependent path replaces ``u_t`` by
    ``u - u_prev`` and divides the whole action by the step once. Both are exact only for a term linear in
    ``u_t``. A quadratic one came out scaled by ``dt`` instead of ``dt**2`` -- measured on ``u_t + a u_t**2 +
    u = 0``: the march matched the mis-scaled recursion to every digit, not backward Euler -- and the
    constant-mass path would drop it outright (its derivative at ``u_t = 0`` is zero). Refused by name."""
    deg = _rate_degree(node)
    if deg is not None and deg <= 1:
        return
    what = "quadratic (or higher)" if deg is not None else "not linear"
    shown = repr(node)
    shown = shown if len(shown) <= 240 else shown[:240] + " ..."
    raise NotImplementedError(
        f"{where}: the transient term {shown} is {what} in the time derivative u_t. A first-order march "
        "treats every u_t as a mass action M(u)·u_t, which is exact only for a term linear in u_t -- this "
        "one would be marched wrongly (a u_t·u_t piece scaled by dt instead of dt², silently). Write the term "
        "with u_t entering linearly; for a residual-based VMS Reynolds stress -(∇v, u'⊗u'), drop the "
        "u_t⊗u_t piece of u'⊗u' (or evaluate u' quasi-statically there)."
    )


def _replace_temporal_with_backward_euler(node: Any, prev_for) -> Any:
    """Replace ``d/dt(TrialFunction)`` with ``(TrialFunction − u_prev)`` — the backward-Euler
    discretization of a transient term used when the mass coefficient depends on the unknown
    (state-dependent mass). ``prev_for(trial) -> PrevStateField`` supplies the field's previous-step
    frozen values.

    A term ``c(u)·u_t·v`` becomes ``c(u)·(u − u_prev)·v``; assembling its residual/Jacobian gives the
    exact mass action ``M(u)(u − u_prev)`` and its exact ``∂/∂u`` (both ``M`` and the coefficient
    coupling). The ``1/dt`` factor is applied by the stepper, NOT here — so the produced term is
    independent of the (runtime) step size. Mirrors :func:`_strip_temporal_trial_derivative`'s traversal.
    """
    if _is_temporal_jacobian_of_trial(node):
        trial = node.target
        return trial - prev_for(trial)  # (u − u_prev); trace BinaryOp via Placeholder.__sub__

    if isinstance(node, BinaryOp):
        return BinaryOp(
            node.op,
            _replace_temporal_with_backward_euler(node.left, prev_for),
            _replace_temporal_with_backward_euler(node.right, prev_for),
        )

    if isinstance(node, FunctionCall):
        new_args = [
            _replace_temporal_with_backward_euler(a, prev_for) if isinstance(a, Placeholder) else a for a in node.args
        ]
        if hasattr(node, "copy_with_args"):
            return node.copy_with_args(new_args)
        return FunctionCall(
            node.fn,
            new_args,
            name=getattr(node, "_name", None),
            reduces_axis=getattr(node, "reduces_axis", None),
            kwargs=getattr(node, "kwargs", None),
        )

    if isinstance(node, Jacobian):
        return Jacobian(
            _replace_temporal_with_backward_euler(node.target, prev_for),
            [
                _replace_temporal_with_backward_euler(v, prev_for) if isinstance(v, Placeholder) else v
                for v in node.variables
            ],
            node.scheme,
        )

    if isinstance(node, Hessian):
        return Hessian(
            _replace_temporal_with_backward_euler(node.target, prev_for),
            [
                _replace_temporal_with_backward_euler(v, prev_for) if isinstance(v, Placeholder) else v
                for v in node.variables
            ],
            node.scheme,
            trace=node.trace,
        )

    return node
