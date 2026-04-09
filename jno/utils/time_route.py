from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from ..trace import BinaryOp, FunctionCall, Hessian, Jacobian, Placeholder, TestFunction, TrialFunction, Variable
from .backend_blocks import DiffraxBlock, FeaxTimeBlock


# -----------------------------------------------------------------------------
# Small graph-inspection helpers
# -----------------------------------------------------------------------------


def _contains_node_type(node: Any, cls) -> bool:
    if isinstance(node, cls):
        return True
    if isinstance(node, BinaryOp):
        return _contains_node_type(node.left, cls) or _contains_node_type(node.right, cls)
    if isinstance(node, FunctionCall):
        return any(_contains_node_type(a, cls) for a in node.args)
    if isinstance(node, Jacobian):
        return _contains_node_type(node.target, cls) or any(_contains_node_type(v, cls) for v in node.variables)
    if isinstance(node, Hessian):
        return _contains_node_type(node.target, cls) or any(_contains_node_type(v, cls) for v in node.variables)
    return False



def _is_temporal_var(node: Any) -> bool:
    return isinstance(node, Variable) and getattr(node, "axis", None) == "temporal"



def _max_temporal_derivative_order(node: Any) -> int:
    if isinstance(node, Jacobian):
        target_order = _max_temporal_derivative_order(node.target)
        local_order = sum(1 for v in node.variables if _is_temporal_var(v))
        if local_order > 0:
            return target_order + local_order
        return target_order

    if isinstance(node, Hessian):
        target_order = _max_temporal_derivative_order(node.target)
        local_order = sum(1 for v in node.variables if _is_temporal_var(v))
        if local_order > 0:
            return target_order + local_order
        return target_order

    if isinstance(node, BinaryOp):
        return max(_max_temporal_derivative_order(node.left), _max_temporal_derivative_order(node.right))

    if isinstance(node, FunctionCall):
        if not node.args:
            return 0
        return max((_max_temporal_derivative_order(a) for a in node.args if isinstance(a, Placeholder)), default=0)

    return 0



def _collect_temporal_tags(node: Any, out: set[str] | None = None) -> set[str]:
    if out is None:
        out = set()
    if isinstance(node, Variable) and getattr(node, "axis", None) == "temporal":
        out.add(str(node.tag))
        return out
    if isinstance(node, BinaryOp):
        _collect_temporal_tags(node.left, out)
        _collect_temporal_tags(node.right, out)
        return out
    if isinstance(node, FunctionCall):
        for a in node.args:
            if isinstance(a, Placeholder):
                _collect_temporal_tags(a, out)
        return out
    if isinstance(node, (Jacobian, Hessian)):
        _collect_temporal_tags(node.target, out)
        for v in node.variables:
            if isinstance(v, Placeholder):
                _collect_temporal_tags(v, out)
        return out
    return out


# -----------------------------------------------------------------------------
# Time metadata helpers
# -----------------------------------------------------------------------------


def _infer_time_window(domain, **kwargs) -> Tuple[float, float, float | None]:
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



def _resolve_initial_data(kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
    initial_conditions = kwargs.get("initial_conditions", kwargs.get("ics", None))
    state0 = kwargs.get("state0", kwargs.get("y0", kwargs.get("initial_state", None)))
    return initial_conditions, state0


# -----------------------------------------------------------------------------
# Public backend lowerers
# -----------------------------------------------------------------------------


def _assemble_diffrax_from_strong_form(domain, expr, **kwargs) -> DiffraxBlock:
    if _contains_node_type(expr, TrialFunction) or _contains_node_type(expr, TestFunction):
        raise ValueError(
            "target='diffrax' currently expects a strong-form expression without "
            "TrialFunction/TestFunction symbols. For transient FEM weak forms, "
            "use target='feax_time'."
        )

    time_order = _max_temporal_derivative_order(expr)
    if time_order <= 0:
        raise ValueError(
            "target='diffrax' could not find a temporal derivative in the provided "
            "strong-form expression. Expected a first- or second-order-in-time problem."
        )
    if time_order > 2:
        raise NotImplementedError(
            "target='diffrax' currently supports only first- or second-order-in-time "
            "strong-form problems in phase 1."
        )

    t0, t1, dt0 = _infer_time_window(domain, **kwargs)
    initial_conditions, state0 = _resolve_initial_data(kwargs)

    metadata = dict(kwargs.get("metadata", {}))
    metadata.setdefault("temporal_tags", sorted(_collect_temporal_tags(expr)))
    metadata.setdefault("rewrite_required", bool(time_order == 2))
    metadata.setdefault("domain_time", getattr(domain, "time", None))
    metadata.setdefault("notes", "Phase 1 returns a backend block contract; full symbolic-to-IVP lowering follows in the next implementation step.")

    form = "explicit_first_order" if time_order == 1 else "rewritten_second_order"
    lowered_rhs = kwargs.get("lowered_rhs", None)

    term = kwargs.get("term", None)
    rhs = kwargs.get("rhs", None)
    if term is None:
        try:
            import diffrax as _diffrax  # type: ignore

            if rhs is not None:
                term = _diffrax.ODETerm(rhs)
        except Exception:
            term = None

    return DiffraxBlock(
        form=form,
        time_order=int(time_order),
        original_expr=expr,
        lowered_rhs=lowered_rhs,
        state0=state0,
        initial_conditions=initial_conditions,
        t0=t0,
        t1=t1,
        dt0=dt0,
        rhs=rhs,
        term=term,
        args=kwargs.get("args", None),
        mass=kwargs.get("mass", None),
        state_meta=dict(kwargs.get("state_meta", {})),
        metadata=metadata,
    )



def _assemble_feax_time_from_ir(domain, ir, **kwargs) -> FeaxTimeBlock:
    if not hasattr(domain, "_feax_context"):
        raise ValueError(
            "target='feax_time' requires domain.init_fem(...) to be called before "
            "assembly so the FEAX mesh and quadrature context are available."
        )

    expr_candidates = []
    if ir.volume_expr is not None:
        expr_candidates.append(ir.volume_expr)
    expr_candidates.extend(ir.boundary_exprs.values())

    time_order = max((_max_temporal_derivative_order(e) for e in expr_candidates), default=0)
    if time_order <= 0:
        raise ValueError(
            "target='feax_time' could not find a temporal derivative in the weak-form "
            "expression. Use target='fem_system' or 'fem_residual' for steady weak forms."
        )
    if time_order > 2:
        raise NotImplementedError(
            "target='feax_time' currently supports only first- or second-order-in-time "
            "weak-form problems in phase 1."
        )

    t0, t1, dt = _infer_time_window(domain, **kwargs)
    initial_conditions, state0 = _resolve_initial_data(kwargs)

    mode = kwargs.get("mode", kwargs.get("scheme", None))
    if mode is None:
        mode = "explicit" if time_order == 2 else "implicit"
    mode = str(mode).lower()
    if mode not in {"implicit", "explicit"}:
        raise ValueError(
            f"Unsupported target='feax_time' mode '{mode}'. Supported: 'implicit', 'explicit'."
        )

    metadata = dict(kwargs.get("metadata", {}))
    metadata.setdefault("temporal_tags", sorted(set().union(*(_collect_temporal_tags(e) for e in expr_candidates))))
    metadata.setdefault("domain_time", getattr(domain, "time", None))
    metadata.setdefault("notes", "Phase 1 returns a transient FEAX block contract; mass/residual splitting and solver-specific wrapping follow in the next implementation step.")

    return FeaxTimeBlock(
        mode=mode,
        time_order=int(time_order),
        spatial_kind="weak_form",
        ir=ir,
        residual_expr=ir.volume_expr,
        boundary_exprs=ir.boundary_exprs,
        mass_expr=kwargs.get("mass_expr", None),
        rhs=kwargs.get("rhs", None),
        jacobian=kwargs.get("jacobian", None),
        state0=state0,
        initial_conditions=initial_conditions,
        t0=t0,
        t1=t1,
        dt=dt,
        feax_context=getattr(domain, "_feax_context", {}),
        metadata=metadata,
    )
