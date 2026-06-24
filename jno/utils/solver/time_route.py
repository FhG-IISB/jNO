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
