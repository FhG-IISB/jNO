from __future__ import annotations

"""
Internal time-dependent solver routing.

This module lowers time-dependent jNO expressions into solver-facing blocks.

Main responsibilities:
- classify temporal order and time metadata,
- lower strong-form time-dependent expressions to DiffraxBlock,
- lower transient weak forms to FeaxTimeBlock,
- split weak-form IR into mass, operator, residual, and source parts,
- build JAX/FEAX semidiscrete runtime callables for transient FEM problems.

This module is internal. User-facing code should normally call:

    expr.assemble(target="diffrax")
    weak_expr.assemble(target="fem_time")
"""
from contextlib import contextmanager
from typing import Any, Dict, Tuple

import jax.numpy as jnp
import numpy as np

from ...trace import (
    BinaryOp,
    FunctionCall,
    Hessian,
    Jacobian,
    Literal,
    Placeholder,
    StateField,
    TensorTag,
    TestFunction,
    TrialFunction,
    Variable,
)
from ...trace_evaluator import TraceEvaluator
from .backend_blocks import DiffraxBlock, FeaxTimeBlock
from .feax_utils import (
    _build_feax_mesh,
    _dense_array,
    _make_internal_vars,
    _prepare_feax_runtime,
)
from .parametric_helpers import (
    _clone_term_with_coeff,
    _collect_runtime_parameter_exprs,
    _contains_runtime_parameter,
    _make_ir,
    _make_zero_ir_like,
    _merge_runtime_parameter_exprs,
    _runtime_parameter_tag,
    _runtime_scalar_arg,
    _split_parametric_operator_ir,
)
from .solver_helper import (
    collect_temporal_tags as _collect_temporal_tags,
)
from .solver_helper import (
    contains_model_call as _contains_model_call,
)
from .solver_helper import (
    contains_node_type as _contains_node_type,
)
from .solver_helper import (
    contains_temporal_derivative as _contains_temporal_derivative,
)
from .solver_helper import (
    is_temporal_var as _is_temporal_var,
)
from .solver_helper import (
    max_temporal_derivative_order as _max_temporal_derivative_order,
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


def _resolve_initial_data(kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Extract initial-condition metadata and solver initial state from kwargs.

    Recognized aliases:
    - initial conditions: `initial_conditions`, `ics`
    - state: `state0`, `y0`, `initial_state`
    """
    initial_conditions = kwargs.get("initial_conditions", kwargs.get("ics", None))
    state0 = kwargs.get("state0", kwargs.get("y0", kwargs.get("initial_state", None)))
    return initial_conditions, state0


def _detect_time_order(expr: Any) -> int:
    return _max_temporal_derivative_order(expr)


def _classify_time_problem(expr: Any, domain, target: str) -> Dict[str, Any]:
    """
    Build lightweight metadata describing a time-dependent expression.

    The result records temporal order, temporal tags, weak-form symbols, and
    domain time information. It is stored in backend block metadata.
    """
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


def _append_identity_unique(out, node):
    if not any(node is x for x in out):
        out.append(node)


def _collect_temporal_state_candidates(node: Any, out=None):
    """
    Collect expression subtrees that appear as targets of temporal derivatives.

    Used to infer the primary state expression for strong-form Diffrax lowering.
    """
    if out is None:
        out = []

    if isinstance(node, Jacobian):
        if any(_is_temporal_var(v) for v in node.variables):
            _append_identity_unique(out, node.target)
        _collect_temporal_state_candidates(node.target, out)
        for v in node.variables:
            if isinstance(v, Placeholder):
                _collect_temporal_state_candidates(v, out)
        return out

    if isinstance(node, Hessian):
        if any(_is_temporal_var(v) for v in node.variables):
            _append_identity_unique(out, node.target)
        _collect_temporal_state_candidates(node.target, out)
        for v in node.variables:
            if isinstance(v, Placeholder):
                _collect_temporal_state_candidates(v, out)
        return out

    if isinstance(node, BinaryOp):
        _collect_temporal_state_candidates(node.left, out)
        _collect_temporal_state_candidates(node.right, out)
        return out

    if isinstance(node, FunctionCall):
        for a in node.args:
            if isinstance(a, Placeholder):
                _collect_temporal_state_candidates(a, out)
        return out

    return out


def _infer_single_state_expr(expr: Any):
    """
    Infer a unique state expression from temporal derivative targets.

    Model-call candidates are preferred. Raises if multiple possible state
    expressions are found and the user must pass `state_expr=...` explicitly.
    """
    candidates = _collect_temporal_state_candidates(expr, out=[])

    # Prefer candidates that actually contain a model call
    model_candidates = [c for c in candidates if _contains_model_call(c)]
    if len(model_candidates) == 1:
        return model_candidates[0]

    if len(candidates) == 1:
        return candidates[0]

    if len(model_candidates) > 1:
        raise ValueError(
            "Could not infer state_expr automatically: multiple model-based temporal candidates found. "
            "Pass state_expr=... explicitly."
        )

    if len(candidates) > 1:
        raise ValueError(
            "Could not infer state_expr automatically: multiple temporal candidates found. Pass state_expr=... explicitly."
        )

    raise ValueError("Could not infer state_expr automatically. Pass state_expr=... explicitly.")


def _resolve_strong_state_expr(expr: Any, **kwargs):
    """
    Resolve the state expression for strong-form lowering.

    Uses explicit `state_expr=...` when provided. If `infer_state_expr=True`,
    attempts automatic inference from temporal derivative targets.
    """
    state_expr = kwargs.get("state_expr", None)
    if state_expr is not None:
        return state_expr

    if bool(kwargs.get("infer_state_expr", False)):
        return _infer_single_state_expr(expr)

    return None


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


def _build_manual_second_order_reduction(expr: Any, **kwargs) -> Dict[str, Any]:
    """
    Validate and describe a user-provided manual second-order reduction.

    Requires:
        second_order="manual"
        rhs=...
        state0=[u0, v0]

    Returns metadata for a reduced first-order system.
    """
    rhs = kwargs.get("rhs", None)
    state0 = kwargs.get("state0", kwargs.get("y0", kwargs.get("initial_state", None)))
    state_names = tuple(kwargs.get("state_names", ("u", "v")))

    if kwargs.get("second_order", None) != "manual":
        return {
            "implemented": False,
            "strategy": "augment_state_with_velocity",
            "original_expr": expr,
            "state_names": state_names,
        }

    if rhs is None:
        raise ValueError(
            "second_order='manual' requires rhs=..., where rhs(t, y, args) returns the reduced first-order system."
        )

    if state0 is None:
        raise ValueError("second_order='manual' requires state0=..., typically [u0, v0].")

    state0_arr = jnp.asarray(state0)
    if state0_arr.ndim != 1:
        raise ValueError(f"second_order='manual' expects a 1D reduced initial state, got shape {state0_arr.shape}.")

    if len(state_names) != 2:
        raise ValueError(f"state_names must contain exactly two names, got {state_names}.")

    if state0_arr.shape[0] != 2:
        raise ValueError("Priority-3 manual second-order support currently expects exactly two reduced states: [u, v].")

    return {
        "implemented": True,
        "strategy": "manual_first_order",
        "original_expr": expr,
        "state_names": state_names,
        "state_size": int(state0_arr.shape[0]),
    }


# -----------------------------------------------------------------------------
# Weak-form FEAX helpers
# -----------------------------------------------------------------------------


def _split_additive_terms(node):
    """
    Split an expression into additive terms while preserving signs.

    Subtractions are rewritten by multiplying the right-hand terms by `-1`.
    """
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _split_additive_terms(node.left) + _split_additive_terms(node.right)
        if node.op == "-":
            return _split_additive_terms(node.left) + [
                BinaryOp("*", Literal(-1.0), t) for t in _split_additive_terms(node.right)
            ]
    return [node]


def _sum_terms(terms):
    """
    Recombine a list of symbolic terms using BinaryOp("+", ...).

    Returns None for an empty list.
    """
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
            dtype = arr.dtype if arr.size > 0 else np.float32
            domain.context["__time__"] = np.asarray([[t_value]], dtype=dtype)

        for k in local_keys:
            arr = np.asarray(domain.context[k])
            dtype = arr.dtype if arr.size > 0 else np.float32
            domain.context[k] = np.asarray([[t_value]], dtype=dtype)

        yield
    finally:
        if had_main:
            domain.context["__time__"] = old_main
        for k, v in old_local.items():
            domain.context[k] = v


def _build_first_order_semidiscrete_operators(
    domain,
    ir,
    mass_expr,
    residual_expr,
    boundary_exprs,
):
    """
    Build nonlinear first-order semidiscrete operators with runtime parameters.

    The generated callables represent:

        M(t) u_dot + R(u, t, args) = 0

    Runtime parameters are supported in the residual through:

        R(u,t,args) = R0(u,t) + sum_i args[name_i] * Ri(u,t)
        J(u,t,args) = J0(u,t) + sum_i args[name_i] * Ji(u,t)

    The mass term deliberately remains parameter-free in this first version.
    """
    del mass_expr, residual_expr, boundary_exprs

    import feax as fe

    from .weak_form import (
        LoweredChannelTerm,
        LoweredWeakForm,
    )

    # --------------------------------------------------
    # Build mass IR
    # --------------------------------------------------
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

    mass_ir = LoweredWeakForm(
        domain=domain,
        terms=mass_terms,
    )

    # --------------------------------------------------
    # Build residual IR
    # --------------------------------------------------
    residual_terms = []

    for term in ir.terms:
        if term.support == "volume" and term.channel == "raw" and not _contains_temporal_derivative(term.coeff):
            residual_terms.append(term)

        elif term.support == "boundary" and term.channel == "raw":
            residual_terms.append(term)

    residual_ir = LoweredWeakForm(
        domain=domain,
        terms=residual_terms,
    )

    if len(mass_ir.terms) == 0:
        raise ValueError("Nonlinear semidiscrete path could not extract a mass term. Expected something like `u_t * phi`.")

    # Runtime parameters in the mass matrix are intentionally rejected
    # in this first implementation.
    if any(_contains_runtime_parameter(term.coeff) for term in mass_ir.terms):
        raise NotImplementedError(
            "Runtime trainable parameters inside nonlinear transient mass "
            "terms are not supported yet. Keep the mass term parameter-free "
            "and place affine parameters inside the residual."
        )

    # --------------------------------------------------
    # Build mass runtime
    # --------------------------------------------------
    mass_rt = _prepare_feax_runtime(
        domain,
        mass_ir,
        apply_dirichlet=True,
        need_jacobian=True,
        symmetric_bc=True,
    )

    # --------------------------------------------------
    # Split residual:
    #
    #     R(u,t,args)
    #       = R0(u,t)
    #       + sum_i args[name_i] * Ri(u,t)
    # --------------------------------------------------
    (
        static_residual_ir,
        parameter_irs,
        runtime_parameter_exprs,
    ) = _split_parametric_operator_ir(residual_ir)

    # FEAX requires one structural IR even when every residual term is
    # parameter dependent.
    if len(static_residual_ir.terms) == 0:
        if parameter_irs:
            static_eval_ir = _make_zero_ir_like(next(iter(parameter_irs.values())))
        else:
            static_eval_ir = _make_zero_ir_like(mass_ir)
    else:
        static_eval_ir = static_residual_ir

    residual_rt = _prepare_feax_runtime(
        domain,
        static_eval_ir,
        apply_dirichlet=True,
        need_jacobian=True,
        symmetric_bc=True,
    )

    # --------------------------------------------------
    # Build BC-safe affine residual bases
    #
    #     effective_basis
    #       = basis_with_bc - zero_basis_with_bc
    # --------------------------------------------------
    basis_runtimes = {}

    for name, basis_ir in parameter_irs.items():
        zero_basis_ir = _make_zero_ir_like(basis_ir)

        basis_rt = _prepare_feax_runtime(
            domain,
            basis_ir,
            apply_dirichlet=True,
            need_jacobian=True,
            symmetric_bc=True,
        )

        zero_rt = _prepare_feax_runtime(
            domain,
            zero_basis_ir,
            apply_dirichlet=True,
            need_jacobian=True,
            symmetric_bc=True,
        )

        if basis_rt["jac_bc"] is None or zero_rt["jac_bc"] is None:
            raise ValueError(f"Nonlinear transient residual basis {name!r} did not produce a Jacobian.")

        if int(basis_rt["size"]) != int(residual_rt["size"]):
            raise ValueError(
                f"Nonlinear transient residual basis {name!r} has size {basis_rt['size']}, expected {residual_rt['size']}."
            )

        if int(zero_rt["size"]) != int(residual_rt["size"]):
            raise ValueError(
                f"Nonlinear transient zero basis {name!r} has size {zero_rt['size']}, expected {residual_rt['size']}."
            )

        basis_runtimes[name] = {
            "basis": basis_rt,
            "zero": zero_rt,
        }

    # --------------------------------------------------
    # FEAX InternalVars helper
    # --------------------------------------------------
    def _internal_vars(rt, t):
        if len(rt["temporal_tags"]) == 0:
            return fe.InternalVars()

        return _make_internal_vars(
            fe,
            rt["temporal_tags"],
            t,
            n_cells=rt["n_cells"],
            dtype=rt["dtype"],
        )

    # --------------------------------------------------
    # Mass matrix
    # --------------------------------------------------
    if mass_rt["jac_bc"] is None:
        raise ValueError("Mass runtime did not produce a Jacobian operator.")

    if len(mass_rt["temporal_tags"]) == 0:
        M_const = _dense_array(
            mass_rt["jac_bc"](
                mass_rt["u_ref"],
                fe.InternalVars(),
            )
        )

        M_const = jnp.asarray(
            M_const,
            dtype=mass_rt["dtype"],
        )

        def mass_fn(t, args=None):
            del t, args
            return M_const

    else:

        def mass_fn(t, args=None):
            del args

            iv = _internal_vars(
                mass_rt,
                t,
            )

            M_t = _dense_array(
                mass_rt["jac_bc"](
                    mass_rt["u_ref"],
                    iv,
                )
            )

            return jnp.asarray(
                M_t,
                dtype=mass_rt["dtype"],
            )

    # --------------------------------------------------
    # Residual and Jacobian
    # --------------------------------------------------
    if residual_rt["jac_bc"] is None:
        raise ValueError("Residual runtime did not produce a Jacobian operator.")

    dtype = residual_rt["dtype"]

    def residual_fn(u_flat, t, args=None):
        u_flat = jnp.asarray(
            u_flat,
            dtype=dtype,
        ).reshape(-1)

        iv_static = _internal_vars(
            residual_rt,
            t,
        )

        out = jnp.asarray(
            residual_rt["res_bc"](
                u_flat,
                iv_static,
            ),
            dtype=dtype,
        ).reshape(-1)

        for name, pair in basis_runtimes.items():
            coeff = _runtime_scalar_arg(
                args,
                name,
                dtype=dtype,
            )

            iv_basis = _internal_vars(
                pair["basis"],
                t,
            )

            iv_zero = _internal_vars(
                pair["zero"],
                t,
            )

            basis_res = jnp.asarray(
                pair["basis"]["res_bc"](
                    u_flat,
                    iv_basis,
                ),
                dtype=dtype,
            ).reshape(-1) - jnp.asarray(
                pair["zero"]["res_bc"](
                    u_flat,
                    iv_zero,
                ),
                dtype=dtype,
            ).reshape(-1)

            out = out + coeff * basis_res

        return out

    def jacobian_fn(u_flat, t, args=None):
        u_flat = jnp.asarray(
            u_flat,
            dtype=dtype,
        ).reshape(-1)

        iv_static = _internal_vars(
            residual_rt,
            t,
        )

        out = jnp.asarray(
            _dense_array(
                residual_rt["jac_bc"](
                    u_flat,
                    iv_static,
                )
            ),
            dtype=dtype,
        )

        for name, pair in basis_runtimes.items():
            coeff = _runtime_scalar_arg(
                args,
                name,
                dtype=dtype,
            )

            iv_basis = _internal_vars(
                pair["basis"],
                t,
            )

            iv_zero = _internal_vars(
                pair["zero"],
                t,
            )

            basis_jac = jnp.asarray(
                _dense_array(
                    pair["basis"]["jac_bc"](
                        u_flat,
                        iv_basis,
                    )
                ),
                dtype=dtype,
            ) - jnp.asarray(
                _dense_array(
                    pair["zero"]["jac_bc"](
                        u_flat,
                        iv_zero,
                    )
                ),
                dtype=dtype,
            )

            out = out + coeff * basis_jac

        return out

    runtime_info = {
        "mass_is_constant": bool(len(mass_rt["temporal_tags"]) == 0),
        "residual_has_time": bool(
            len(residual_rt["temporal_tags"]) > 0
            or any(len(pair["basis"]["temporal_tags"]) > 0 for pair in basis_runtimes.values())
        ),
        "dtype": mass_rt["dtype"],
        "state_size": int(mass_rt["size"]),
        "mass_temporal_tags": tuple(mass_rt["temporal_tags"]),
        "residual_temporal_tags": tuple(residual_rt["temporal_tags"]),
        "runtime_parameter_names": tuple(sorted(runtime_parameter_exprs.keys())),
        "runtime_parameter_exprs": dict(runtime_parameter_exprs),
        "residual_basis_names": tuple(sorted(basis_runtimes.keys())),
    }

    return (
        mass_fn,
        residual_fn,
        jacobian_fn,
        runtime_info,
    )


# -----------------------------------------------------------------------------
# Public backend lowerers
# -----------------------------------------------------------------------------


def _assemble_diffrax_from_strong_form(domain, expr, **kwargs) -> DiffraxBlock:
    """
    Lower a strong-form time-dependent expression into a DiffraxBlock.

    This route is used by:

        expr.assemble(target="diffrax")

    Supported cases:
    - first-order strong forms with explicit `state_expr`, `time_var`, and `state0`
    - first-order contract mode when no symbolic lowering inputs are provided
    - second-order problems through manual first-order reduction

    Returns:
        DiffraxBlock containing RHS, Diffrax term, initial state, time interval,
        optional mass function, and lowering metadata.

    Raises:
        ValueError if the expression contains TestFunction/TrialFunction symbols,
        because transient weak forms must use `target="fem_time"`.
    """
    if _contains_node_type(expr, TrialFunction) or _contains_node_type(expr, TestFunction):
        raise ValueError(
            "target='diffrax' expects a strong-form expression without "
            "TrialFunction/TestFunction symbols. For transient weak forms, "
            "use target='fem_time'."
        )

    time_order = _detect_time_order(expr)
    if time_order <= 0:
        raise ValueError(
            "target='diffrax' could not find a temporal derivative in the provided "
            "strong-form expression. Expected a first- or second-order-in-time problem."
        )
    if time_order > 2:
        raise NotImplementedError(
            "target='diffrax' currently supports only first- or second-order-in-time strong-form problems."
        )

    t0, t1, dt0 = _infer_time_window(domain, **kwargs)
    initial_conditions, state0 = _extract_initial_conditions(kwargs)

    metadata = dict(kwargs.get("metadata", {}))
    metadata.setdefault("classification", _classify_time_problem(expr, domain, target="diffrax"))
    metadata.setdefault("temporal_tags", sorted(_collect_temporal_tags(expr)))
    metadata.setdefault("rewrite_required", bool(time_order == 2))
    metadata.setdefault("domain_time", getattr(domain, "time", None))

    # --------------------------------------------------
    # Priority 2: real first-order symbolic lowering
    # --------------------------------------------------
    if time_order == 1:
        state_expr = _resolve_strong_state_expr(expr, **kwargs)
        time_var = kwargs.get("time_var", None)
        params = kwargs.get("params", None)

        if state_expr is not None:
            if time_var is None or not isinstance(time_var, Variable) or getattr(time_var, "axis", None) != "temporal":
                raise ValueError("Strong-form Diffrax lowering with state_expr=... requires time_var=<temporal Variable>.")

            if state0 is None:
                raise ValueError("Strong-form Diffrax lowering with state_expr=... requires state0=...")

            mass_expr, residual_expr = _split_first_order_strong_form(expr, state_expr, time_var)
            rhs, mass_fn, lowered_rhs, strong_runtime = _build_first_order_strong_diffrax_runtime(
                domain,
                mass_expr=mass_expr,
                residual_expr=residual_expr,
                state_expr=state_expr,
                time_var=time_var,
                state0=state0,
                params=params,
            )

            metadata["runtime_parameter_names"] = list(strong_runtime["runtime_parameter_names"])

            metadata["runtime_parameter_tags"] = dict(strong_runtime["runtime_parameter_tags"])

            metadata["dynamic_parameters"] = bool(strong_runtime["runtime_parameter_names"])

            import diffrax as _diffrax

            metadata["phase"] = "phase_2_first_order_lowered"
            metadata["lowering_complete"] = True
            metadata["state_expr_mode"] = "explicit" if kwargs.get("state_expr", None) is not None else "inferred"
            metadata["notes"] = (
                "First-order strong-form Diffrax lowering completed. "
                "The provided state_expr subtree was replaced internally by a runtime solver state."
            )

            return DiffraxBlock(
                backend="diffrax",
                form="explicit_first_order",
                time_order=1,
                original_expr=expr,
                lowered_rhs=lowered_rhs,
                rewritten_system=None,
                state0=state0,
                initial_conditions=initial_conditions,
                t0=t0,
                t1=t1,
                dt0=dt0,
                rhs=rhs,
                term=_diffrax.ODETerm(rhs),
                args=kwargs.get("args", None),
                mass=mass_fn,
                state_meta=dict(kwargs.get("state_meta", {})),
                metadata=metadata,
            )

        # fallback: keep old contract mode if user did not opt into actual lowering
        metadata.setdefault("phase", "phase_2_contract")
        metadata.setdefault("lowering_complete", False)
        metadata.setdefault(
            "notes",
            "First-order strong-form problem classified, but no symbolic lowering was performed. "
            "Pass state_expr=..., time_var=..., state0=... to enable real Diffrax lowering.",
        )

        rhs = kwargs.get("rhs", None)
        mass = kwargs.get("mass", None)
        term = kwargs.get("term", None)
        if term is None:
            try:
                import diffrax as _diffrax

                if rhs is not None:
                    term = _diffrax.ODETerm(rhs)
            except Exception:
                term = None

        return DiffraxBlock(
            backend="diffrax",
            form="explicit_first_order",
            time_order=1,
            original_expr=expr,
            lowered_rhs=kwargs.get("lowered_rhs", None),
            rewritten_system=None,
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

    # --------------------------------------------------
    # Second-order manual reduction
    # --------------------------------------------------
    if time_order == 2:
        state_expr = _resolve_strong_state_expr(expr, **kwargs)
        reduction = _build_manual_second_order_reduction(expr, **kwargs)

        rhs = kwargs.get("rhs", None)
        mass = kwargs.get("mass", None)
        term = kwargs.get("term", None)

        if kwargs.get("second_order", None) == "manual":
            if rhs is None:
                raise ValueError("second_order='manual' requires rhs=... for the reduced first-order system.")

            if state0 is None:
                raise ValueError("second_order='manual' requires state0=..., typically [u0, v0].")

            state_names = tuple(kwargs.get("state_names", ("u", "v")))
            rewritten_system = {
                "implemented": True,
                "strategy": "manual_first_order",
                "original_expr": expr,
                "state_names": state_names,
                "state_expr_mode": (
                    "explicit"
                    if kwargs.get("state_expr", None) is not None
                    else ("inferred" if state_expr is not None else "none")
                ),
            }

            if term is None:
                import diffrax as _diffrax

                term = _diffrax.ODETerm(rhs)

            metadata["phase"] = "phase_3_second_order_manual_reduction"
            metadata["lowering_complete"] = True
            metadata["reduction_mode"] = "manual"
            metadata["notes"] = (
                "Second-order strong-form problem accepted through manual first-order reduction. "
                "state_expr is used only as the training-expression anchor; the actual solver route uses the reduced rhs."
            )

            return DiffraxBlock(
                backend="diffrax",
                form="manual_second_order_reduced",
                time_order=2,
                original_expr=expr,
                lowered_rhs=None,
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
                state_meta={
                    **dict(kwargs.get("state_meta", {})),
                    "original_order": 2,
                    "reduction_state_names": state_names,
                },
                metadata=metadata,
            )

        # --------------------------------------------------
        # Fallback: keep old contract-only placeholder
        # --------------------------------------------------

        metadata["phase"] = "phase_3_contract"
        metadata["lowering_complete"] = False
        metadata["rewrite_required"] = True
        metadata["notes"] = (
            "Second-order strong-form problem was classified, but no manual reduction "
            "was provided. Pass second_order='manual', rhs=..., state0=[u0, v0] to "
            "build a usable Diffrax block."
        )

        return DiffraxBlock(
            backend="diffrax",
            form="rewritten_second_order",
            time_order=2,
            original_expr=expr,
            lowered_rhs=None,
            rewritten_system=reduction,
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
    raise RuntimeError(f"Internal error: strong-form Diffrax lowering reached an unsupported time_order={time_order}.")


def _split_additive_terms_strong(node):
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _split_additive_terms_strong(node.left) + _split_additive_terms_strong(node.right)
        if node.op == "-":
            return _split_additive_terms_strong(node.left) + [
                BinaryOp("*", Literal(-1.0), t) for t in _split_additive_terms_strong(node.right)
            ]
    return [node]


def _replace_exact_subtree(node: Any, target: Any, replacement: Any) -> Any:
    """
    Replace a specific expression subtree by object identity.

    Used when replacing the symbolic state expression with a runtime state tag
    during strong-form Diffrax lowering.
    """
    if node is target:
        return replacement

    if isinstance(node, BinaryOp):
        left = _replace_exact_subtree(node.left, target, replacement)
        right = _replace_exact_subtree(node.right, target, replacement)
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = [_replace_exact_subtree(a, target, replacement) if isinstance(a, Placeholder) else a for a in node.args]
        if any(a is not b for a, b in zip(new_args, node.args)):
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

    if isinstance(node, Jacobian):
        new_target = _replace_exact_subtree(node.target, target, replacement)
        new_vars = [
            _replace_exact_subtree(v, target, replacement) if isinstance(v, Placeholder) else v for v in node.variables
        ]
        if new_target is not node.target or any(a is not b for a, b in zip(new_vars, node.variables)):
            return Jacobian(new_target, new_vars, node.scheme)
        return node

    if isinstance(node, Hessian):
        new_target = _replace_exact_subtree(node.target, target, replacement)
        new_vars = [
            _replace_exact_subtree(v, target, replacement) if isinstance(v, Placeholder) else v for v in node.variables
        ]
        if new_target is not node.target or any(a is not b for a, b in zip(new_vars, node.variables)):
            return Hessian(new_target, new_vars, node.scheme, trace=node.trace)
        return node

    return node


def _same_temporal_var(a: Any, b: Any) -> bool:
    return (
        isinstance(a, Variable)
        and isinstance(b, Variable)
        and getattr(a, "axis", None) == "temporal"
        and getattr(b, "axis", None) == "temporal"
        and str(a.tag) == str(b.tag)
    )


def _is_state_time_derivative(node: Any, state_expr: Any, time_var: Variable) -> bool:
    return (
        isinstance(node, Jacobian)
        and node.target is state_expr
        and len(node.variables) == 1
        and _same_temporal_var(node.variables[0], time_var)
    )


def _extract_temporal_coeff(term: Any, state_expr: Any, time_var: Variable):
    """
    Extract the coefficient multiplying the first-order state time derivative.

    Supports forms such as:
        u_t
        a * u_t
        u_t * a
        u_t / a
    """
    # u_t
    if _is_state_time_derivative(term, state_expr, time_var):
        return Literal(1.0)

    # a * u_t  or  u_t * a
    if isinstance(term, BinaryOp) and term.op == "*":
        if _is_state_time_derivative(term.left, state_expr, time_var):
            return term.right
        if _is_state_time_derivative(term.right, state_expr, time_var):
            return term.left

    # u_t / a   ->   (1/a) * u_t
    if isinstance(term, BinaryOp) and term.op == "/":
        if _is_state_time_derivative(term.left, state_expr, time_var):
            return BinaryOp("/", Literal(1.0), term.right)

    return None


def _split_first_order_strong_form(expr: Any, state_expr: Any, time_var: Variable):
    """
    Split a first-order strong form into mass coefficient and residual parts.

    For an expression like:

        a(x,t,u) * u_t + F(u,t) = 0

    returns:
        mass_expr = a(x,t,u)
        residual_expr = F(u,t)
    """
    terms = _split_additive_terms_strong(expr)

    mass_terms = []
    residual_terms = []

    for term in terms:
        coeff = _extract_temporal_coeff(term, state_expr, time_var)
        if coeff is not None:
            if _contains_temporal_derivative(coeff):
                raise NotImplementedError(
                    "Strong-form Diffrax lowering does not support temporal derivatives "
                    "inside the coefficient of the state time-derivative term yet."
                )
            mass_terms.append(coeff)
        else:
            residual_terms.append(term)

    if len(mass_terms) == 0:
        raise ValueError(
            "Could not isolate a first-order temporal state term. "
            "Expected something like u_t + F(u,t)=0 or a(x,t,u)*u_t + F(u,t)=0."
        )

    mass_expr = _sum_terms(mass_terms)
    residual_expr = _sum_terms(residual_terms)

    if residual_expr is None:
        residual_expr = Literal(0.0)

    return mass_expr, residual_expr


def _build_first_order_strong_diffrax_runtime(
    domain,
    *,
    mass_expr,
    residual_expr,
    state_expr,
    time_var,
    state0,
    params=None,
):
    """
    Build a strong-form Diffrax RHS with runtime ``jno.np.parameter`` values.

    Symbolic inverse parameters are replaced by private ``TensorTag`` nodes.
    Concrete values remain external and are supplied through:

        diffrax.diffeqsolve(..., args={"parameter_name": value})
    """
    params = {} if params is None else params
    evaluator = TraceEvaluator(params)

    # --------------------------------------------------
    # Replace symbolic state expression by runtime state
    # --------------------------------------------------
    state_tag = "__diffrax_state__"
    state_runtime = TensorTag(
        tag=state_tag,
        domain=domain,
    )

    mass_runtime_expr = _replace_exact_subtree(
        mass_expr,
        state_expr,
        state_runtime,
    )

    residual_runtime_expr = _replace_exact_subtree(
        residual_expr,
        state_expr,
        state_runtime,
    )

    # --------------------------------------------------
    # Detect jno.np.parameter(...) expressions and replace
    # them with runtime TensorTags populated from Diffrax args.
    # --------------------------------------------------
    runtime_parameter_exprs = {}

    _collect_runtime_parameter_exprs(
        mass_runtime_expr,
        runtime_parameter_exprs,
    )

    _collect_runtime_parameter_exprs(
        residual_runtime_expr,
        runtime_parameter_exprs,
    )

    runtime_parameter_tags = {}

    for name, param_expr in runtime_parameter_exprs.items():
        tag = _runtime_parameter_tag(name)

        runtime_parameter_tags[name] = tag

        runtime_tag = TensorTag(
            tag=tag,
            domain=domain,
        )

        mass_runtime_expr = _replace_exact_subtree(
            mass_runtime_expr,
            param_expr,
            runtime_tag,
        )

        residual_runtime_expr = _replace_exact_subtree(
            residual_runtime_expr,
            param_expr,
            runtime_tag,
        )

    state0_arr = jnp.asarray(state0)
    time_tag = str(time_var.tag)

    def _state_to_context(y):
        y = jnp.asarray(y)

        if y.ndim == 0:
            return y.reshape(1)

        if y.ndim == 1:
            return y[:, None]

        return y

    def _set_time_context(ctx, t):
        t_arr = jnp.asarray(
            [[t]],
            dtype=jnp.asarray(t).dtype,
        )

        ctx[time_tag] = t_arr

        if time_tag != "__time__" and "__time__" in domain.context:
            ctx["__time__"] = t_arr

        return ctx

    def _inject_runtime_parameters(ctx, args):
        for name, tag in runtime_parameter_tags.items():
            ctx[tag] = _runtime_scalar_arg(
                args,
                name,
                dtype=state0_arr.dtype,
            )

        return ctx

    def _eval_runtime(expr_rt, y, t, args):
        ctx = dict(domain.context)

        ctx[state_tag] = _state_to_context(y)

        ctx = _set_time_context(
            ctx,
            t,
        )

        ctx = _inject_runtime_parameters(
            ctx,
            args,
        )

        out = evaluator.evaluate(
            expr_rt,
            context=ctx,
        )

        return jnp.asarray(out)

    def mass_fn(t, args=None):
        return _eval_runtime(
            mass_runtime_expr,
            state0_arr,
            t,
            args,
        )

    def residual_eval(y, t, args=None):
        return _eval_runtime(
            residual_runtime_expr,
            y,
            t,
            args,
        )

    def rhs(t, y, args=None):
        y_arr = jnp.asarray(y)

        M_t = jnp.asarray(
            _eval_runtime(
                mass_runtime_expr,
                y_arr,
                t,
                args,
            )
        )

        R_t = jnp.asarray(
            _eval_runtime(
                residual_runtime_expr,
                y_arr,
                t,
                args,
            )
        )

        # Scalar mass.
        if M_t.ndim == 0 or M_t.size == 1:
            return (-R_t / jnp.reshape(M_t, ())).reshape(y_arr.shape)

        # Diagonal or element-wise mass.
        if M_t.shape == y_arr.shape or M_t.shape == _state_to_context(y_arr).shape:
            return (-R_t / M_t).reshape(y_arr.shape)

        # Dense mass matrix.
        return jnp.linalg.solve(
            jnp.asarray(M_t),
            -jnp.asarray(R_t).reshape(-1),
        ).reshape(y_arr.shape)

    runtime_info = {
        "runtime_parameter_names": tuple(sorted(runtime_parameter_exprs.keys())),
        "runtime_parameter_exprs": dict(runtime_parameter_exprs),
        "runtime_parameter_tags": dict(runtime_parameter_tags),
    }

    return (
        rhs,
        mass_fn,
        residual_runtime_expr,
        runtime_info,
    )


def _contains_trial(node: Any) -> bool:
    return _contains_node_type(node, TrialFunction)


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


def _is_linear_first_order_ir(ir) -> bool:
    """
    Narrow structural check for the current JAX-native semidiscrete path.

    Conditions:
    - first-order in time
    - raw weak-form terms
    - no obvious nonlinear function of TrialFunction
    """
    from .weak_form import _is_obviously_nonlinear_in_unknown

    for term in ir.terms:
        if term.channel != "raw":
            return False
        if _is_obviously_nonlinear_in_unknown(ir.domain, term.coeff):
            return False
    return True


def _should_use_linear_semidiscrete_path(ir, kwargs) -> bool:
    if "linear" in kwargs:
        return bool(kwargs["linear"])
    return _is_linear_first_order_ir(ir)


def _ir_temporal_tags(ir) -> list[str]:
    tags = set()
    for term in ir.terms:
        tags.update(_collect_temporal_tags(term.coeff))
    return sorted(tags)


def _prepare_src_runtime(domain, src_ir):
    """
    Build FEAX source-only runtime ONCE.

    Source/load vector is evaluated as:
        b(t) = -r_src(0, t)
    with no Dirichlet elimination.
    """
    rt = _prepare_feax_runtime(
        domain,
        src_ir,
        apply_dirichlet=False,
        need_jacobian=False,
        symmetric_bc=True,
    )

    u_zero = jnp.zeros((rt["size"],), dtype=rt["dtype"])
    rt["u_zero"] = u_zero
    return rt


def _build_source_vector_fn(domain, src_ir, *, size, dtype):
    """Build one transient volume-source or Neumann-load vector callback."""
    import feax as fe

    if src_ir is None or len(src_ir.terms) == 0:
        return None

    rt = _prepare_src_runtime(domain, src_ir)
    if int(rt["size"]) != int(size):
        raise ValueError(f"Auto forcing runtime size mismatch: runtime size={rt['size']}, expected {size}.")

    if len(rt["temporal_tags"]) == 0:
        iv0 = fe.InternalVars()
        const_vec = -jnp.asarray(rt["res_bc"](rt["u_zero"], iv0), dtype=dtype).reshape(-1)

        def source_vector_fn(t):
            del t
            return const_vec

        return source_vector_fn

    def source_vector_fn(t):
        iv = _make_internal_vars(fe, rt["temporal_tags"], t, n_cells=rt["n_cells"], dtype=rt["dtype"])
        return -jnp.asarray(rt["res_bc"](rt["u_zero"], iv), dtype=dtype).reshape(-1)

    return source_vector_fn


def _build_auto_forcing_vector_fn(domain, src_ir, *, size, dtype):
    """Build ``f(t,args) = f0(t) + sum_i args[name_i] * f_i(t)``."""
    if src_ir is None or len(src_ir.terms) == 0:
        return None, {}, {}

    static_src_ir, parameter_irs, runtime_parameter_exprs = _split_parametric_operator_ir(src_ir)
    static_fn = _build_source_vector_fn(domain, static_src_ir, size=size, dtype=dtype)
    forcing_basis = {
        name: _build_source_vector_fn(domain, basis_ir, size=size, dtype=dtype) for name, basis_ir in parameter_irs.items()
    }
    zero = jnp.zeros((int(size),), dtype=dtype)

    def forcing_vector_fn(t, args=None):
        out = zero if static_fn is None else static_fn(t)
        for name, basis_fn in forcing_basis.items():
            coeff = _runtime_scalar_arg(args, name, dtype=dtype)
            out = out + coeff * basis_fn(t)
        return out

    return forcing_vector_fn, forcing_basis, runtime_parameter_exprs


def _assemble_feax_time_from_ir(domain, ir, **kwargs) -> FeaxTimeBlock:
    """
    Lower a transient weak-form IR into a FeaxTimeBlock.

    This route is used by:

        weak_expr.assemble(target="fem_time")

    It supports first-order-in-time weak forms and produces a solver-agnostic
    semidiscrete block.

    Linear path:
        Splits the IR into mass, operator, and source parts and returns a block
        representing:

            M u_dot + A(t, args) u = affine_bias + forcing_vector_fn(t, args)

    Nonlinear path:
        Builds FEAX/JAX runtime callables and returns a block representing:

            M(t) u_dot + R(u, t) = 0

    Returns:
        FemTimeBlock with either linear payload `(M, A, affine_bias,
        forcing_vector_fn)` or nonlinear payload `(mass, residual, jacobian)`.

    Raises:
        ValueError if `domain.init_fem(...)` was not called or if no temporal
        derivative is found.
        NotImplementedError for weak forms that are not first-order in time.
    """
    has_fem_context = (
        getattr(domain, "_variational_initialized", False)
        and getattr(domain, "fem_context", None) is not None
        and getattr(domain, "_feax_context", None) is not None
    )

    if not has_fem_context:
        raise ValueError(
            "target='fem_time' requires domain.init_fem(...) to be called before "
            "assembly so the FEM mesh and quadrature context are available."
        )

    expr_candidates = []
    if getattr(ir, "volume_expr", None) is not None:
        expr_candidates.append(ir.volume_expr)
    expr_candidates.extend((getattr(ir, "boundary_exprs", {}) or {}).values())

    time_order = max((_detect_time_order(e) for e in expr_candidates), default=0)
    if time_order <= 0:
        raise ValueError(
            "target='fem_time' could not find a temporal derivative in the weak-form "
            "expression. Use target='fem_system' or 'fem_residual' for steady weak forms."
        )

    if time_order != 1:
        raise NotImplementedError(
            "The JAX-native semidiscrete FEAX-time path currently supports only first-order-in-time weak forms."
        )

    t0, t1, dt = _infer_time_window(domain, **kwargs)
    initial_conditions, state0 = _extract_initial_conditions(kwargs)

    mode = kwargs.get("mode", kwargs.get("scheme", None))
    if mode is None:
        mode = "implicit"
    mode = str(mode).lower()
    if mode not in {"implicit", "explicit"}:
        raise ValueError(f"Unsupported target='fem_time' mode '{mode}'. Supported: 'implicit', 'explicit'.")

    mass_expr, residual_expr, boundary_exprs = _split_mass_and_residual_from_ir(ir)

    metadata = dict(kwargs.get("metadata", {}))
    metadata.setdefault(
        "classification",
        _classify_time_problem(ir.volume_expr, domain, target="fem_time"),
    )
    metadata.setdefault(
        "temporal_tags",
        sorted(set().union(*(_collect_temporal_tags(e) for e in expr_candidates))),
    )
    metadata.setdefault("domain_time", getattr(domain, "time", None))

    # ------------------------------------------------------------------
    # New narrow linear JAX-native path
    # ------------------------------------------------------------------
    use_linear_path = _should_use_linear_semidiscrete_path(ir, kwargs)

    if use_linear_path:
        from .fem_route import _assemble_fem_system_from_ir

        mass_ir, op_ir, src_ir = _split_first_order_linear_terms(ir)
        if len(mass_ir.terms) == 0:
            raise ValueError("Expected a mass term such as `u_t * phi`.")
        if any(_contains_runtime_parameter(term.coeff) for term in mass_ir.terms):
            raise NotImplementedError("Runtime parameters inside transient mass terms are not supported yet.")

        M_sys, _bM = _assemble_fem_system_from_ir(domain, mass_ir)
        M = _dense_array(M_sys)

        static_op_ir, operator_parameter_irs, operator_parameter_exprs = _split_parametric_operator_ir(op_ir)
        if len(static_op_ir.terms) > 0:
            A0_sys, bA0 = _assemble_fem_system_from_ir(domain, static_op_ir)
            A = _dense_array(A0_sys)
            affine_bias = jnp.asarray(bA0).reshape(-1)
        else:
            A = jnp.zeros_like(M)
            affine_bias = jnp.zeros((M.shape[0],), dtype=M.dtype)

        operator_basis = {}
        for name, basis_ir in operator_parameter_irs.items():
            zero_basis_ir = _make_zero_ir_like(basis_ir)
            K_bc, bK_bc = _assemble_fem_system_from_ir(domain, basis_ir)
            K_zero_bc, bK_zero_bc = _assemble_fem_system_from_ir(domain, zero_basis_ir)
            K = jnp.asarray(_dense_array(K_bc)) - jnp.asarray(_dense_array(K_zero_bc))
            bK = jnp.asarray(bK_bc).reshape(-1) - jnp.asarray(bK_zero_bc).reshape(-1)
            if not np.allclose(np.asarray(bK), 0.0, atol=1.0e-8):
                raise NotImplementedError("Runtime Dirichlet parameters are not supported yet.")
            operator_basis[name] = K

        operator_fn = None
        if operator_basis:

            def operator_fn(t, args):
                del t
                A_t = A
                for name, K in operator_basis.items():
                    coeff = _runtime_scalar_arg(args, name, dtype=A.dtype)
                    A_t = A_t + coeff * jnp.asarray(K, dtype=A.dtype)
                return A_t

        user_operator_fn = kwargs.get("operator_fn", None)
        if user_operator_fn is not None:
            operator_fn = user_operator_fn

        forcing_vector_fn = kwargs.get("forcing_vector_fn", None)
        forcing_basis = {}
        forcing_parameter_exprs = {}
        auto_forcing = False
        forcing_mode = "none"
        if forcing_vector_fn is not None:
            forcing_mode = "user_callback"
        elif len(src_ir.terms) > 0:
            forcing_vector_fn, forcing_basis, forcing_parameter_exprs = _build_auto_forcing_vector_fn(
                domain, src_ir, size=M.shape[0], dtype=M.dtype
            )
            auto_forcing = True
            forcing_mode = "weak_auto"

        combined_runtime_parameter_exprs = _merge_runtime_parameter_exprs(operator_parameter_exprs, forcing_parameter_exprs)

        feax_problem = getattr(domain, "_feax_problem", None)
        feax_mesh = getattr(feax_problem, "mesh", None)
        if feax_mesh is None:
            feax_mesh = _build_feax_mesh(domain, getattr(domain, "_fem_element_type", "TRI3"))

        metadata["phase"] = "phase_2_linear_jax"
        metadata["lowering_complete"] = True
        metadata["forcing_terms"] = int(len(src_ir.terms))
        metadata["forcing_temporal_tags"] = _ir_temporal_tags(src_ir)
        metadata["auto_forcing"] = bool(auto_forcing)
        metadata["forcing_mode"] = forcing_mode
        metadata["linear_inferred"] = "linear" not in kwargs
        metadata["linear_path_selected"] = bool(use_linear_path)
        metadata["runtime_parameter_names"] = sorted(combined_runtime_parameter_exprs)
        metadata["operator_parameter_names"] = sorted(operator_parameter_exprs)
        metadata["forcing_parameter_names"] = sorted(forcing_parameter_exprs)
        metadata["dynamic_operator"] = bool(operator_fn is not None)

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
            operator_fn=operator_fn,
            runtime_parameter_exprs=combined_runtime_parameter_exprs,
            operator_basis=operator_basis,
            affine_bias=affine_bias,
            forcing_vector_fn=forcing_vector_fn,
            forcing_basis=forcing_basis,
            feax_mesh=feax_mesh,
            forcing_mode=forcing_mode,
            nonlinear_runtime={},
        )

    # ------------------------------------------------------------------
    # Nonlinear first-order semidiscrete path
    # ------------------------------------------------------------------
    mass_fn, residual_fn, jacobian_fn, nonlinear_runtime = _build_first_order_semidiscrete_operators(
        domain,
        ir,
        mass_expr,
        residual_expr,
        boundary_exprs,
    )

    feax_mesh = None
    if getattr(domain, "_feax_context", None) is not None:
        feax_mesh = domain._feax_context.get("mesh", None)

    metadata["phase"] = "phase_2_nonlinear_jax"
    metadata["lowering_complete"] = True
    metadata["auto_forcing"] = False
    metadata["forcing_mode"] = "embedded_residual"
    metadata["linear_inferred"] = "linear" not in kwargs
    metadata["linear_path_selected"] = bool(use_linear_path)
    metadata["mass_is_constant"] = bool(nonlinear_runtime["mass_is_constant"])
    metadata["residual_has_time"] = bool(nonlinear_runtime["residual_has_time"])
    metadata["state_size"] = int(nonlinear_runtime["state_size"])

    metadata["runtime_parameter_names"] = list(
        nonlinear_runtime.get(
            "runtime_parameter_names",
            (),
        )
    )

    metadata["dynamic_parameters"] = bool(
        nonlinear_runtime.get(
            "runtime_parameter_names",
            (),
        )
    )

    metadata["residual_basis_names"] = list(
        nonlinear_runtime.get(
            "residual_basis_names",
            (),
        )
    )

    metadata["notes"] = (
        "Nonlinear first-order semidiscrete FEAX-time block assembled. "
        "mass(t), residual(u,t), and jacobian(u,t) are populated for external solvers. "
        "Source and boundary forcing are embedded inside residual(u,t)."
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
        jacobian=jacobian_fn,
        mass=mass_fn,
        residual=residual_fn,
        state0=state0,
        initial_conditions=initial_conditions,
        t0=t0,
        t1=t1,
        dt=dt,
        feax_context=getattr(domain, "_feax_context", {}),
        metadata=metadata,
        M=None,
        A=None,
        runtime_parameter_exprs=dict(
            nonlinear_runtime.get(
                "runtime_parameter_exprs",
                {},
            )
        ),
        affine_bias=None,
        forcing_vector_fn=None,
        feax_mesh=feax_mesh,
        forcing_mode="embedded_residual",
        nonlinear_runtime=nonlinear_runtime,
    )
