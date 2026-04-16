from __future__ import annotations

from typing import Any, Dict, Tuple
from contextlib import contextmanager
import numpy as np
import jax.numpy as jnp
from ..trace import BinaryOp, FunctionCall, Hessian, Jacobian, Literal, Placeholder, StateField, TestFunction, TrialFunction, Variable
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
        return any(_contains_node_type(a, cls) for a in node.args if isinstance(a, Placeholder))
    if isinstance(node, Jacobian):
        return _contains_node_type(node.target, cls) or any(
            _contains_node_type(v, cls) for v in node.variables if isinstance(v, Placeholder)
        )
    if isinstance(node, Hessian):
        return _contains_node_type(node.target, cls) or any(
            _contains_node_type(v, cls) for v in node.variables if isinstance(v, Placeholder)
        )
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
        # Hessian corresponds to a second derivative operator.
        local_order = 2 * sum(1 for v in node.variables if _is_temporal_var(v))
        if local_order > 0:
            return target_order + local_order
        return target_order

    if isinstance(node, BinaryOp):
        return max(_max_temporal_derivative_order(node.left), _max_temporal_derivative_order(node.right))

    if isinstance(node, FunctionCall):
        if not node.args:
            return 0
        return max(
            (_max_temporal_derivative_order(a) for a in node.args if isinstance(a, Placeholder)),
            default=0,
        )

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


def _contains_temporal_derivative(node: Any) -> bool:
    return _max_temporal_derivative_order(node) > 0


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


def _detect_time_order(expr: Any) -> int:
    return _max_temporal_derivative_order(expr)


def _classify_time_problem(expr: Any, domain, target: str) -> Dict[str, Any]:
    return {
        "target": target,
        "time_order": _detect_time_order(expr),
        "temporal_tags": sorted(_collect_temporal_tags(expr)),
        "has_trial": _contains_node_type(expr, TrialFunction),
        "has_test": _contains_node_type(expr, TestFunction),
        "has_statefield": _contains_node_type(expr, StateField),
        "domain_time": getattr(domain, "time", None),
    }


def _extract_initial_conditions(kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
    return _resolve_initial_data(kwargs)


# -----------------------------------------------------------------------------
# Strong-form Diffrax helpers
# -----------------------------------------------------------------------------


def _rewrite_second_order_to_first_order(expr: Any, **kwargs) -> Dict[str, Any]:
    """Phase-2 metadata-only rewrite placeholder.

    For now this returns structural metadata sufficient to tell the caller that
    a second-order problem should be solved with an augmented first-order state,
    e.g. (u, v) with v = u_t.
    """
    return {
        "implemented": False,
        "strategy": "augment_state_with_velocity",
        "original_expr": expr,
        "state_names": kwargs.get("state_names", ("u", "v")),
    }


# -----------------------------------------------------------------------------
# Weak-form FEAX helpers
# -----------------------------------------------------------------------------

def _split_additive_terms(node):
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _split_additive_terms(node.left) + _split_additive_terms(node.right)
        if node.op == "-":
            return _split_additive_terms(node.left) + [BinaryOp("*", Literal(-1.0), t) for t in _split_additive_terms(node.right)]
    return [node]


def _sum_terms(terms):
    if not terms:
        return None
    out = terms[0]
    for t in terms[1:]:
        out = BinaryOp("+", out, t)
    return out


def _split_mass_and_residual_from_ir(ir) -> Tuple[Any, Any, Dict[str, Any]]:
    """Split weak-form volume expression into transient mass-like and residual parts.

    Rule:
    - additive terms containing temporal derivatives -> mass_expr
    - remaining additive terms -> residual_expr
    - boundary terms stay boundary_exprs
    """
    volume_expr = getattr(ir, "volume_expr", None)
    boundary_exprs = dict(getattr(ir, "boundary_exprs", {}) or {})

    if volume_expr is None:
        return None, None, boundary_exprs

    terms = _split_additive_terms(volume_expr)

    mass_terms = []
    residual_terms = []

    for term in terms:
        if _contains_temporal_derivative(term):
            mass_terms.append(term)
        else:
            residual_terms.append(term)

    mass_expr = _sum_terms(mass_terms)
    residual_expr = _sum_terms(residual_terms)

    return mass_expr, residual_expr, boundary_exprs


def _is_temporal_jacobian_of_trial(node: Any) -> bool:
    return (
        isinstance(node, Jacobian)
        and isinstance(node.target, TrialFunction)
        and any(_is_temporal_var(v) for v in node.variables)
    )


def _strip_temporal_trial_derivative(node: Any) -> Any:
    """
    Convert Jacobian(TrialFunction, [t]) -> TrialFunction.

    This is the key step for building the FE mass matrix from a weak term like
        ∫ u_t * phi
    by turning it into
        ∫ u * phi
    during spatial semidiscretization.

    Phase 1:
    - first-order in time only
    - only strips temporal derivatives directly on TrialFunction
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
        new_args = [
            _strip_temporal_trial_derivative(a) if isinstance(a, Placeholder) else a
            for a in node.args
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
        new_target = _strip_temporal_trial_derivative(node.target)
        new_vars = [
            _strip_temporal_trial_derivative(v) if isinstance(v, Placeholder) else v
            for v in node.variables
        ]
        return Jacobian(new_target, new_vars, node.scheme)

    if isinstance(node, Hessian):
        new_target = _strip_temporal_trial_derivative(node.target)
        new_vars = [
            _strip_temporal_trial_derivative(v) if isinstance(v, Placeholder) else v
            for v in node.variables
        ]
        return Hessian(new_target, new_vars, node.scheme, trace=node.trace)

    return node


def _filter_ir_terms(ir, predicate):
    from .weak_form import LoweredWeakForm

    kept = [t for t in ir.terms if predicate(t)]
    return LoweredWeakForm(domain=ir.domain, terms=kept)


def _merge_residual_and_boundary_ir(ir):
    from .weak_form import LoweredWeakForm

    kept = []
    for t in ir.terms:
        if t.support == "volume":
            kept.append(t)
        elif t.support == "boundary":
            kept.append(t)
    return LoweredWeakForm(domain=ir.domain, terms=kept)


@contextmanager
def _temporary_time_value(domain, t_value: float):
    """
    Temporarily overwrite the domain time context so FEAX assembly sees the
    current time when source terms / coefficients depend on t.
    """
    old_main = None
    had_main = "__time__" in domain.context
    if had_main:
        old_main = np.array(domain.context["__time__"], copy=True)

    old_local = {}
    local_keys = [k for k in domain.context.keys() if str(k).startswith("__time_")]
    for k in local_keys:
        old_local[k] = np.array(domain.context[k], copy=True)

    try:
        if had_main:
            arr = np.asarray(domain.context["__time__"])
            domain.context["__time__"] = np.full(arr.shape, t_value, dtype=arr.dtype)

        for k in local_keys:
            arr = np.asarray(domain.context[k])
            domain.context[k] = np.full(arr.shape, t_value, dtype=arr.dtype)

        yield
    finally:
        if had_main:
            domain.context["__time__"] = old_main
        for k, v in old_local.items():
            domain.context[k] = v


def _build_first_order_semidiscrete_operators(domain, ir, mass_expr, residual_expr, boundary_exprs):
    """
    Build callables for the semidiscrete problem

        M(t) u_dot + R(u,t) = 0

    Phase 1 assumptions:
    - first-order only
    - one unknown
    - FEAX weak route
    """
    from .weak_form import LoweredWeakForm, LoweredChannelTerm
    from .fem_route import _assemble_fem_system_from_ir, _assemble_fem_residual_from_ir

    # -------------------------
    # Build mass IR
    # -------------------------
    mass_terms = []
    for term in ir.terms:
        if term.support == "volume" and term.channel == "raw" and _contains_temporal_derivative(term.coeff):
            stripped = _strip_temporal_trial_derivative(term.coeff)
            mass_terms.append(
                LoweredChannelTerm(
                    sign=term.sign,
                    support=term.support,
                    region_id=term.region_id,
                    channel="raw",
                    coeff=stripped,
                    variable_id=term.variable_id,
                    value_shape=term.value_shape,
                    original_expr=term.original_expr,
                )
            )

    mass_ir = LoweredWeakForm(domain=domain, terms=mass_terms)

    # -------------------------
    # Build residual IR
    # volume residual + boundary residual
    # -------------------------
    residual_terms = []
    for term in ir.terms:
        if term.support == "volume" and term.channel == "raw" and not _contains_temporal_derivative(term.coeff):
            residual_terms.append(term)
        elif term.support == "boundary" and term.channel == "raw":
            residual_terms.append(term)

    residual_ir = LoweredWeakForm(domain=domain, terms=residual_terms)

    def mass_fn(t, args=None):
        with _temporary_time_value(domain, float(t)):
            A, _ = _assemble_fem_system_from_ir(domain, mass_ir)
            return A

    def residual_fn(u_flat, t, args=None):
        with _temporary_time_value(domain, float(t)):
            op = _assemble_fem_residual_from_ir(domain, residual_ir)
            return op.residual(u_flat)

    def jacobian_fn(u_flat, t, args=None):
        with _temporary_time_value(domain, float(t)):
            op = _assemble_fem_residual_from_ir(domain, residual_ir)
            if op.jacobian is None:
                raise ValueError("No jacobian available for semidiscrete residual.")
            return op.jacobian(u_flat)

    return mass_fn, residual_fn, jacobian_fn


# -----------------------------------------------------------------------------
# Public backend lowerers
# -----------------------------------------------------------------------------


def _assemble_diffrax_from_strong_form(domain, expr, **kwargs) -> DiffraxBlock:
    if _contains_node_type(expr, TrialFunction) or _contains_node_type(expr, TestFunction):
        raise ValueError(
            "target='diffrax' expects a strong-form expression without "
            "TrialFunction/TestFunction symbols. For transient weak forms, "
            "use target='feax_time'."
        )

    time_order = _detect_time_order(expr)
    if time_order <= 0:
        raise ValueError(
            "target='diffrax' could not find a temporal derivative in the provided "
            "strong-form expression. Expected a first- or second-order-in-time problem."
        )
    if time_order > 2:
        raise NotImplementedError(
            "target='diffrax' currently supports only first- or second-order-in-time "
            "strong-form problems."
        )

    t0, t1, dt0 = _infer_time_window(domain, **kwargs)
    initial_conditions, state0 = _extract_initial_conditions(kwargs)

    metadata = dict(kwargs.get("metadata", {}))
    metadata.setdefault("phase", "phase_2_contract")
    metadata.setdefault("lowering_complete", False)
    metadata.setdefault("classification", _classify_time_problem(expr, domain, target="diffrax"))
    metadata.setdefault("temporal_tags", sorted(_collect_temporal_tags(expr)))
    metadata.setdefault("rewrite_required", bool(time_order == 2))
    metadata.setdefault("domain_time", getattr(domain, "time", None))
    metadata.setdefault(
        "notes",
        "Phase 2 returns a Diffrax block contract with first/second-order classification. "
        "Full symbolic RHS isolation can be refined next.",
    )

    form = "explicit_first_order" if time_order == 1 else "rewritten_second_order"
    lowered_rhs = kwargs.get("lowered_rhs", None)
    rewritten_system = _rewrite_second_order_to_first_order(expr, **kwargs) if time_order == 2 else None

    rhs = kwargs.get("rhs", None)
    mass = kwargs.get("mass", None)
    term = kwargs.get("term", None)

    if term is None:
        try:
            import diffrax as _diffrax  # type: ignore
            if rhs is not None:
                term = _diffrax.ODETerm(rhs)
        except Exception:
            term = None

    return DiffraxBlock(
        backend="diffrax",
        form=form,
        time_order=int(time_order),
        original_expr=expr,
        lowered_rhs=lowered_rhs,
        rewritten_system=rewritten_system,
        state0=state0,
        initial_conditions=initial_conditions,
        t0=t0,
        t1=t1,
        dt0=dt0,
        rhs=rhs,
        term=term,
        args=kwargs.get("args", None),
        mass=mass,
        state_meta=dict(kwargs.get("state_meta", {})),
        metadata=metadata,
    )

def _contains_trial(node: Any) -> bool:
    return _contains_node_type(node, TrialFunction)


def _strip_temporal_trial_derivative(node: Any) -> Any:
    """
    Replace d/dt(TrialFunction) -> TrialFunction.

    This is used to turn a weak mass term like
        ∫ u_t phi
    into the spatial mass operator
        ∫ u phi
    """
    if isinstance(node, Jacobian):
        if isinstance(node.target, TrialFunction) and any(_is_temporal_var(v) for v in node.variables):
            return node.target
        return Jacobian(
            _strip_temporal_trial_derivative(node.target),
            [
                _strip_temporal_trial_derivative(v) if isinstance(v, Placeholder) else v
                for v in node.variables
            ],
            node.scheme,
        )

    if isinstance(node, Hessian):
        return Hessian(
            _strip_temporal_trial_derivative(node.target),
            [
                _strip_temporal_trial_derivative(v) if isinstance(v, Placeholder) else v
                for v in node.variables
            ],
            node.scheme,
            trace=node.trace,
        )

    if isinstance(node, BinaryOp):
        return BinaryOp(
            node.op,
            _strip_temporal_trial_derivative(node.left),
            _strip_temporal_trial_derivative(node.right),
        )

    if isinstance(node, FunctionCall):
        new_args = [
            _strip_temporal_trial_derivative(a) if isinstance(a, Placeholder) else a
            for a in node.args
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

    return node


def _clone_term_with_coeff(term, new_coeff):
    from .weak_form import LoweredChannelTerm

    return LoweredChannelTerm(
        sign=term.sign,
        support=term.support,
        region_id=term.region_id,
        channel=term.channel,
        coeff=new_coeff,
        variable_id=term.variable_id,
        value_shape=term.value_shape,
        original_expr=term.original_expr,
    )


def _make_ir(domain, terms):
    from .weak_form import LoweredWeakForm
    return LoweredWeakForm(domain=domain, terms=list(terms))


def _split_first_order_linear_terms(ir):
    """
    Split a first-order weak-form IR into:

        mass_ir : terms with time derivative on trial
        op_ir   : spatial/state-dependent linear operator terms
        src_ir  : pure load terms without TrialFunction

    This is intentionally narrow:
    - one state
    - first-order in time
    - linear weak form
    """
    mass_terms = []
    op_terms = []
    src_terms = []

    for term in ir.terms:
        coeff = term.coeff

        if term.channel != "raw":
            raise NotImplementedError(
                "Linear semidiscrete FEAX-time path currently expects raw FEM weak-form IR terms only."
            )

        if term.support == "volume" and _contains_temporal_derivative(coeff):
            stripped = _strip_temporal_trial_derivative(coeff)
            mass_terms.append(_clone_term_with_coeff(term, stripped))
            continue

        if _contains_trial(coeff):
            op_terms.append(term)
        else:
            src_terms.append(term)

    return (
        _make_ir(ir.domain, mass_terms),
        _make_ir(ir.domain, op_terms),
        _make_ir(ir.domain, src_terms),
    )


def _dense_array(A):
    if hasattr(A, "todense"):
        return jnp.asarray(A.todense())
    if hasattr(A, "toarray"):
        return jnp.asarray(A.toarray())
    return jnp.asarray(A)


def _assemble_feax_time_from_ir(domain, ir, **kwargs) -> FeaxTimeBlock:
    if not hasattr(domain, "_feax_context"):
        raise ValueError(
            "target='feax_time' requires domain.init_fem(...) to be called before "
            "assembly so the FEAX mesh and quadrature context are available."
        )

    expr_candidates = []
    if getattr(ir, "volume_expr", None) is not None:
        expr_candidates.append(ir.volume_expr)
    expr_candidates.extend((getattr(ir, "boundary_exprs", {}) or {}).values())

    time_order = max((_detect_time_order(e) for e in expr_candidates), default=0)
    if time_order <= 0:
        raise ValueError(
            "target='feax_time' could not find a temporal derivative in the weak-form "
            "expression. Use target='fem_system' or 'fem_residual' for steady weak forms."
        )

    if time_order != 1:
        raise NotImplementedError(
            "The JAX-native semidiscrete FEAX-time path currently supports only "
            "first-order-in-time weak forms."
        )

    t0, t1, dt = _infer_time_window(domain, **kwargs)
    initial_conditions, state0 = _extract_initial_conditions(kwargs)

    mode = kwargs.get("mode", kwargs.get("scheme", None))
    if mode is None:
        mode = "implicit"
    mode = str(mode).lower()
    if mode not in {"implicit", "explicit"}:
        raise ValueError(
            f"Unsupported target='feax_time' mode '{mode}'. Supported: 'implicit', 'explicit'."
        )

    mass_expr, residual_expr, boundary_exprs = _split_mass_and_residual_from_ir(ir)

    metadata = dict(kwargs.get("metadata", {}))
    metadata.setdefault("classification", _classify_time_problem(ir.volume_expr, domain, target="feax_time"))
    metadata.setdefault("temporal_tags", sorted(set().union(*(_collect_temporal_tags(e) for e in expr_candidates))))
    metadata.setdefault("domain_time", getattr(domain, "time", None))

    # ------------------------------------------------------------------
    # New narrow linear JAX-native path
    # ------------------------------------------------------------------
    if kwargs.get("linear", False):
        from .fem_route import _assemble_fem_system_from_ir

        mass_ir, op_ir, src_ir = _split_first_order_linear_terms(ir)

        if len(mass_ir.terms) == 0:
            raise ValueError(
                "Linear semidiscrete path could not extract a mass term. "
                "Expected something like u_t * phi."
            )

        # We intentionally require the user to provide a JAX forcing vector callback
        # for any nonzero source/load contribution, rather than doing Python-side
        # FEAX reassembly inside the RHS.
        if len(src_ir.terms) > 0 and kwargs.get("forcing_vector_fn", None) is None:
            raise ValueError(
                "Linear semidiscrete FEAX-time path detected source/load terms. "
                "Please pass forcing_vector_fn(t, args) as a pure JAX callback, "
                "or assemble an operator-only weak form and provide the load externally."
            )

        M_sys, bM = _assemble_fem_system_from_ir(domain, mass_ir)
        A_sys, bA = _assemble_fem_system_from_ir(domain, op_ir)

        M = _dense_array(M_sys)
        A = _dense_array(A_sys)
        affine_bias = jnp.asarray(bA).reshape(-1)

        metadata["phase"] = "phase_2_linear_jax"
        metadata["lowering_complete"] = True
        metadata["notes"] = (
            "Linear semidiscrete JAX FEAX block assembled. "
            "Runtime data are pure JAX arrays/callables. "
            "Use block.as_diffrax(...) to obtain a Diffrax ODETerm."
        )

        return FeaxTimeBlock(
            backend="feax_time",
            mode=mode,
            time_order=1,
            spatial_kind="weak_form",
            ir=ir,
            mass_expr=mass_expr,
            residual_expr=residual_expr,
            boundary_exprs=boundary_exprs,
            rhs=None,
            jacobian=None,
            mass=None,
            residual=None,
            state0=state0,
            initial_conditions=initial_conditions,
            t0=t0,
            t1=t1,
            dt=dt,
            feax_context=getattr(domain, "_feax_context", {}),
            metadata=metadata,
            M=M,
            A=A,
            affine_bias=affine_bias,
            forcing_vector_fn=kwargs.get("forcing_vector_fn", None),
        )

    # ------------------------------------------------------------------
    # Fallback: old contract-only path
    # ------------------------------------------------------------------
    metadata.setdefault("phase", "phase_2_contract")
    metadata.setdefault("lowering_complete", False)
    metadata.setdefault(
        "notes",
        "Contract-only FEAX-time block. Pass linear=True for the current JAX-native "
        "linear first-order semidiscrete path.",
    )

    return FeaxTimeBlock(
        backend="feax_time",
        mode=mode,
        time_order=1,
        spatial_kind="weak_form",
        ir=ir,
        mass_expr=mass_expr,
        residual_expr=residual_expr,
        boundary_exprs=boundary_exprs,
        rhs=kwargs.get("rhs", None),
        jacobian=kwargs.get("jacobian", None),
        mass=kwargs.get("mass", None),
        residual=kwargs.get("residual", None),
        state0=state0,
        initial_conditions=initial_conditions,
        t0=t0,
        t1=t1,
        dt=dt,
        feax_context=getattr(domain, "_feax_context", {}),
        metadata=metadata,
    )