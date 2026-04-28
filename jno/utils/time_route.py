from __future__ import annotations

from typing import Any, Dict, Tuple
from contextlib import contextmanager
import numpy as np
import jax
import jax.numpy as jnp
from ..trace import BinaryOp, FunctionCall, Hessian, Jacobian, Literal, Placeholder, StateField, TestFunction, TrialFunction, Variable,TensorTag
from .backend_blocks import DiffraxBlock, FeaxTimeBlock
from ..trace_evaluator import TraceEvaluator
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

def _build_manual_second_order_reduction(expr: Any, **kwargs) -> Dict[str, Any]:
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
            "second_order='manual' requires rhs=..., where rhs(t, y, args) returns "
            "the reduced first-order system."
        )

    if state0 is None:
        raise ValueError(
            "second_order='manual' requires state0=..., typically [u0, v0]."
        )

    state0_arr = jnp.asarray(state0)
    if state0_arr.ndim != 1:
        raise ValueError(
            f"second_order='manual' expects a 1D reduced initial state, got shape {state0_arr.shape}."
        )

    if len(state_names) != 2:
        raise ValueError(
            f"state_names must contain exactly two names, got {state_names}."
        )

    if state0_arr.shape[0] != 2:
        raise ValueError(
            "Priority-3 manual second-order support currently expects exactly "
            "two reduced states: [u, v]."
        )

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

def _make_internal_vars(fe_module, temporal_tags, t, *, n_cells: int, dtype=None, extra_volume_vars=()):
    """
    Build FEAX InternalVars in a batched shape FEAX can slice.

    Each temporal variable is broadcast to shape (n_cells, 1).
    """
    vol = []

    if temporal_tags:
        t0 = jnp.asarray(t, dtype=dtype)
        t_batched = jnp.full((int(n_cells), 1), t0, dtype=t0.dtype)
        vol.extend([t_batched for _ in temporal_tags])

    for v in extra_volume_vars:
        arr = jnp.asarray(v, dtype=dtype)
        if arr.ndim == 0:
            arr = jnp.full((int(n_cells), 1), arr, dtype=arr.dtype)
        vol.append(arr)

    return fe_module.InternalVars(volume_vars=tuple(vol))

def _prepare_feax_runtime(
    domain,
    ir,
    *,
    apply_dirichlet=True,
    need_jacobian=True,
    symmetric_bc=True,
):
    import feax as fe
    from .fem_route import _build_feax_problem, _default_float_dtype, _meshio_type_for_element

    problem, bc = _build_feax_problem(
        domain,
        ir,
        apply_dirichlet=apply_dirichlet,
        store_on_domain=False,
    )

    res_bc = fe.create_res_bc_function(problem, bc)
    jac_bc = (
        fe.create_J_bc_function(problem, bc, symmetric=symmetric_bc)
        if need_jacobian
        else None
    )

    size = int(problem.num_total_dofs_all_vars)
    dtype = _default_float_dtype()

    try:
        u_ref = fe.zero_like_initial_guess(problem, bc)
    except Exception:
        u_ref = jnp.zeros((size,), dtype=dtype)

    u_ref = jnp.asarray(u_ref, dtype=dtype)
    temporal_tags = tuple(_ir_temporal_tags(ir))

    # robust cell count lookup
    # Robust cell count lookup: use the domain mesh directly.
    # This is the same source used by _build_feax_mesh(...) in fem_route.py.
    element_type = getattr(domain, "_fem_element_type", None)
    if element_type is None:
        element_type = "TRI3"

    meshio_type = _meshio_type_for_element(element_type)

    if meshio_type not in domain.mesh.cells_dict:
        raise KeyError(
            f"Mesh cell type '{meshio_type}' for element_type='{element_type}' "
            f"not found in domain.mesh.cells_dict. "
            f"Available: {list(domain.mesh.cells_dict.keys())}"
        )

    n_cells = int(np.asarray(domain.mesh.cells_dict[meshio_type]).shape[0])

    return {
        "problem": problem,
        "bc": bc,
        "res_bc": res_bc,
        "jac_bc": jac_bc,
        "size": size,
        "dtype": dtype,
        "u_ref": u_ref,
        "temporal_tags": temporal_tags,
        "n_cells": n_cells,
    }

def _build_first_order_semidiscrete_operators(domain, ir, mass_expr, residual_expr, boundary_exprs):
    """
    Build pure-JAX callable operators for the semidiscrete problem

        M(t) u_dot + R(u,t) = 0

    This version caches FEAX problem/operator construction ONCE and
    reuses the resulting res_bc/jac_bc callables many times.
    """
    import feax as fe
    from .weak_form import LoweredWeakForm, LoweredChannelTerm

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
    # -------------------------
    residual_terms = []
    for term in ir.terms:
        if term.support == "volume" and term.channel == "raw" and not _contains_temporal_derivative(term.coeff):
            residual_terms.append(term)
        elif term.support == "boundary" and term.channel == "raw":
            residual_terms.append(term)

    residual_ir = LoweredWeakForm(domain=domain, terms=residual_terms)

    if len(mass_ir.terms) == 0:
        raise ValueError(
            "Nonlinear semidiscrete path could not extract a mass term. "
            "Expected something like u_t * phi."
        )

    mass_rt = _prepare_feax_runtime(
        domain,
        mass_ir,
        apply_dirichlet=True,
        need_jacobian=True,
        symmetric_bc=True,
    )
    residual_rt = _prepare_feax_runtime(
        domain,
        residual_ir,
        apply_dirichlet=True,
        need_jacobian=True,
        symmetric_bc=True,
    )

    # -------------------------
    # Mass operator
    # -------------------------
    if mass_rt["jac_bc"] is None:
        raise ValueError("Mass runtime did not produce a Jacobian operator.")

    if len(mass_rt["temporal_tags"]) == 0:
        M_const = _dense_array(mass_rt["jac_bc"](mass_rt["u_ref"], fe.InternalVars()))
        M_const = jnp.asarray(M_const, dtype=mass_rt["dtype"])

        def mass_fn(t, args=None):
            return M_const
    else:
        def mass_fn(t, args=None):
            iv = _make_internal_vars(
                    fe,
                    mass_rt["temporal_tags"],
                    t,
                    n_cells=mass_rt["n_cells"],
                    dtype=mass_rt["dtype"],
                )
            M_t = _dense_array(mass_rt["jac_bc"](mass_rt["u_ref"], iv))
            return jnp.asarray(M_t, dtype=mass_rt["dtype"])

    # -------------------------
    # Residual / Jacobian operators
    # -------------------------
    if residual_rt["jac_bc"] is None:
        raise ValueError("Residual runtime did not produce a Jacobian operator.")

    if len(residual_rt["temporal_tags"]) == 0:
        iv_res0 = fe.InternalVars()

        def residual_fn(u_flat, t, args=None):
            u_flat = jnp.asarray(u_flat, dtype=residual_rt["dtype"]).reshape(-1)
            return jnp.asarray(residual_rt["res_bc"](u_flat, iv_res0), dtype=residual_rt["dtype"]).reshape(-1)

        def jacobian_fn(u_flat, t, args=None):
            u_flat = jnp.asarray(u_flat, dtype=residual_rt["dtype"]).reshape(-1)
            J = _dense_array(residual_rt["jac_bc"](u_flat, iv_res0))
            return jnp.asarray(J, dtype=residual_rt["dtype"])
    else:
        def residual_fn(u_flat, t, args=None):
            u_flat = jnp.asarray(u_flat, dtype=residual_rt["dtype"]).reshape(-1)
            iv = _make_internal_vars(
                    fe,
                    residual_rt["temporal_tags"],
                    t,
                    n_cells=residual_rt["n_cells"],
                    dtype=residual_rt["dtype"],
                )
            return jnp.asarray(residual_rt["res_bc"](u_flat, iv), dtype=residual_rt["dtype"]).reshape(-1)

        def jacobian_fn(u_flat, t, args=None):
            u_flat = jnp.asarray(u_flat, dtype=residual_rt["dtype"]).reshape(-1)
            iv = _make_internal_vars(
                    fe,
                    residual_rt["temporal_tags"],
                    t,
                    n_cells=residual_rt["n_cells"],
                    dtype=residual_rt["dtype"],
                )
            J = _dense_array(residual_rt["jac_bc"](u_flat, iv))
            return jnp.asarray(J, dtype=residual_rt["dtype"])

    runtime_info = {
        "mass_is_constant": bool(len(mass_rt["temporal_tags"]) == 0),
        "residual_has_time": bool(len(residual_rt["temporal_tags"]) > 0),
        "dtype": mass_rt["dtype"],
        "state_size": int(mass_rt["size"]),
        "mass_temporal_tags": tuple(mass_rt["temporal_tags"]),
        "residual_temporal_tags": tuple(residual_rt["temporal_tags"]),
    }

    return mass_fn, residual_fn, jacobian_fn, runtime_info

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
    metadata.setdefault("classification", _classify_time_problem(expr, domain, target="diffrax"))
    metadata.setdefault("temporal_tags", sorted(_collect_temporal_tags(expr)))
    metadata.setdefault("rewrite_required", bool(time_order == 2))
    metadata.setdefault("domain_time", getattr(domain, "time", None))

    # --------------------------------------------------
    # Priority 2: real first-order symbolic lowering
    # --------------------------------------------------
    if time_order == 1:
        state_expr = kwargs.get("state_expr", None)
        time_var = kwargs.get("time_var", None)
        params = kwargs.get("params", None)

        if state_expr is not None:
            if time_var is None or not isinstance(time_var, Variable) or getattr(time_var, "axis", None) != "temporal":
                raise ValueError(
                    "First-order strong-form Diffrax lowering requires time_var=<temporal Variable>."
                )

            if state0 is None:
                raise ValueError(
                    "First-order strong-form Diffrax lowering requires state0=..."
                )

            mass_expr, residual_expr = _split_first_order_strong_form(expr, state_expr, time_var)
            rhs, mass_fn, lowered_rhs = _build_first_order_strong_diffrax_runtime(
                domain,
                mass_expr=mass_expr,
                residual_expr=residual_expr,
                state_expr=state_expr,
                time_var=time_var,
                state0=state0,
                params=params,
            )

            import diffrax as _diffrax

            metadata["phase"] = "phase_2_first_order_lowered"
            metadata["lowering_complete"] = True
            metadata["notes"] = (
                "First-order strong-form Diffrax lowering completed. "
                "Symbolic temporal term was isolated and converted into rhs(t,y)."
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
    # Priority 3: manual second-order reduction
    # --------------------------------------------------
    reduction = _build_manual_second_order_reduction(expr, **kwargs)

    rhs = kwargs.get("rhs", None)
    mass = kwargs.get("mass", None)
    term = kwargs.get("term", None)

    if reduction.get("implemented", False):
        if term is None:
            try:
                import diffrax as _diffrax  # type: ignore
                term = _diffrax.ODETerm(rhs)
            except Exception:
                term = None

        metadata["phase"] = "phase_3_second_order_manual_reduction"
        metadata["lowering_complete"] = True
        metadata["rewrite_required"] = True
        metadata["reduction_mode"] = "manual"
        metadata["notes"] = (
            "Second-order strong-form problem accepted through manual first-order reduction. "
            "User supplied rhs(t,y,args) for the reduced [u, v] system."
        )

        return DiffraxBlock(
            backend="diffrax",
            form="manual_second_order_reduced",
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
            state_meta={
                **dict(kwargs.get("state_meta", {})),
                "original_order": 2,
                "reduction_state_names": reduction["state_names"],
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

def _split_additive_terms_strong(node):
    if isinstance(node, BinaryOp):
        if node.op == "+":
            return _split_additive_terms_strong(node.left) + _split_additive_terms_strong(node.right)
        if node.op == "-":
            return (
                _split_additive_terms_strong(node.left)
                + [BinaryOp("*", Literal(-1.0), t) for t in _split_additive_terms_strong(node.right)]
            )
    return [node]

def _replace_exact_subtree(node: Any, target: Any, replacement: Any) -> Any:
    if node is target:
        return replacement

    if isinstance(node, BinaryOp):
        left = _replace_exact_subtree(node.left, target, replacement)
        right = _replace_exact_subtree(node.right, target, replacement)
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = [
            _replace_exact_subtree(a, target, replacement) if isinstance(a, Placeholder) else a
            for a in node.args
        ]
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
            _replace_exact_subtree(v, target, replacement) if isinstance(v, Placeholder) else v
            for v in node.variables
        ]
        if new_target is not node.target or any(a is not b for a, b in zip(new_vars, node.variables)):
            return Jacobian(new_target, new_vars, node.scheme)
        return node

    if isinstance(node, Hessian):
        new_target = _replace_exact_subtree(node.target, target, replacement)
        new_vars = [
            _replace_exact_subtree(v, target, replacement) if isinstance(v, Placeholder) else v
            for v in node.variables
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
    params = {} if params is None else params
    evaluator = TraceEvaluator(params)

    state_tag = "__diffrax_state__"
    state_runtime = TensorTag(tag=state_tag, domain=domain)

    mass_runtime_expr = _replace_exact_subtree(mass_expr, state_expr, state_runtime)
    residual_runtime_expr = _replace_exact_subtree(residual_expr, state_expr, state_runtime)

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
        t_arr = jnp.asarray([[t]], dtype=jnp.asarray(t).dtype)
        ctx[time_tag] = t_arr
        if time_tag != "__time__" and "__time__" in domain.context:
            ctx["__time__"] = t_arr
        return ctx

    def _eval_runtime(expr_rt, y, t):
        ctx = dict(domain.context)
        ctx[state_tag] = _state_to_context(y)
        ctx = _set_time_context(ctx, t)
        out = evaluator.evaluate(expr_rt, context=ctx)
        return jnp.asarray(out)

    def mass_fn(t, args=None):
        return _eval_runtime(mass_runtime_expr, state0_arr, t)

    def residual_eval(y, t):
        return _eval_runtime(residual_runtime_expr, y, t)

    def rhs(t, y, args=None):
        y_arr = jnp.asarray(y)
        M_t = jnp.asarray(_eval_runtime(mass_runtime_expr, y_arr, t))
        R_t = jnp.asarray(_eval_runtime(residual_runtime_expr, y_arr, t))

        # scalar mass
        if M_t.ndim == 0 or M_t.size == 1:
            return (-R_t / jnp.reshape(M_t, ())).reshape(y_arr.shape)

        # diagonal / elementwise mass
        if M_t.shape == y_arr.shape or M_t.shape == _state_to_context(y_arr).shape:
            return (-R_t / M_t).reshape(y_arr.shape)

        # dense mass matrix
        return jnp.linalg.solve(
            jnp.asarray(M_t),
            -jnp.asarray(R_t).reshape(-1),
        ).reshape(y_arr.shape)

    return rhs, mass_fn, residual_runtime_expr

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
    try:
        return jnp.asarray(A)
    except Exception:
        return jnp.asarray(np.asarray(A))


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


def _to_numpy_dense(A):
    if hasattr(A, "todense"):
        return np.asarray(A.todense())
    if hasattr(A, "toarray"):
        return np.asarray(A.toarray())
    return np.asarray(A)


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

def _eval_src_vector_host(domain, runtime, t_value, dtype):
    """
    Evaluate the pure source/load vector at one time value.

    Since src_ir contains no TrialFunction terms, the source vector is

        b(t) = -r_src(0, t)

    for the FEAX residual assembled from the source-only weak form.
    """
    with _temporary_time_value(domain, float(t_value)):
        r_t = runtime["res_bc"](runtime["u_zero"], runtime["internal_vars"])

    return np.asarray(-np.asarray(r_t), dtype=np.dtype(dtype)).reshape(-1)


def _build_auto_forcing_vector_fn(domain, src_ir, *, size, dtype):
    import feax as fe

    if src_ir is None or len(src_ir.terms) == 0:
        return None

    rt = _prepare_src_runtime(domain, src_ir)
    if int(rt["size"]) != int(size):
        raise ValueError(
            f"Auto forcing runtime size mismatch: runtime size={rt['size']}, expected {size}."
        )

    if len(rt["temporal_tags"]) == 0:
        iv0 = fe.InternalVars()
        const_vec = -jnp.asarray(rt["res_bc"](rt["u_zero"], iv0), dtype=dtype).reshape(-1)

        def forcing_vector_fn(t, args=None):
            return const_vec

        return forcing_vector_fn

    def forcing_vector_fn(t, args=None):
        iv = _make_internal_vars(fe, rt["temporal_tags"], t, n_cells=rt["n_cells"], dtype=rt["dtype"], )
        return -jnp.asarray(rt["res_bc"](rt["u_zero"], iv), dtype=dtype).reshape(-1)

    return forcing_vector_fn

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
    use_linear_path = _should_use_linear_semidiscrete_path(ir, kwargs)

    if use_linear_path:
        from .fem_route import _assemble_fem_system_from_ir, _build_feax_mesh

        mass_ir, op_ir, src_ir = _split_first_order_linear_terms(ir)

        if len(mass_ir.terms) == 0:
            raise ValueError(
                "Linear semidiscrete path could not extract a mass term. "
                "Expected something like u_t * phi."
            )

        M_sys, bM = _assemble_fem_system_from_ir(domain, mass_ir)
        A_sys, bA = _assemble_fem_system_from_ir(domain, op_ir)

        M = _dense_array(M_sys)
        A = _dense_array(A_sys)
        affine_bias = jnp.asarray(bA).reshape(-1)

        feax_problem = getattr(domain, "_feax_problem", None)
        feax_mesh = getattr(feax_problem, "mesh", None)

        if feax_mesh is None:
            element_type = getattr(domain, "_fem_element_type", "TRI3")
            feax_mesh = _build_feax_mesh(domain, element_type)

        forcing_vector_fn = kwargs.get("forcing_vector_fn", None)
        auto_forcing = False
        forcing_mode = "none"

        if forcing_vector_fn is not None:
            forcing_mode = "user_callback"
        elif len(src_ir.terms) > 0:
            forcing_vector_fn = _build_auto_forcing_vector_fn(
                domain,
                src_ir,
                size=M.shape[0],
                dtype=M.dtype,
            )
            auto_forcing = True
            forcing_mode = "weak_auto"

        metadata["phase"] = "phase_2_linear_jax"
        metadata["lowering_complete"] = True
        metadata["forcing_terms"] = int(len(src_ir.terms))
        metadata["forcing_temporal_tags"] = _ir_temporal_tags(src_ir)
        metadata["auto_forcing"] = bool(auto_forcing)
        metadata["forcing_mode"] = "weak_auto" if auto_forcing else ("user_callback" if forcing_vector_fn is not None else "none")
        metadata["linear_inferred"] = "linear" not in kwargs
        metadata["linear_path_selected"] = bool(use_linear_path)

        if auto_forcing:
            metadata["notes"] = (
                "Linear semidiscrete JAX FEAX block assembled. "
                "M, A, affine_bias, and forcing_vector_fn are populated for external solvers. "
                "Forcing was auto-lowered from non-trial weak-form terms."
            )
        elif forcing_vector_fn is not None:
            metadata["notes"] = (
                "Linear semidiscrete JAX FEAX block assembled. "
                "M, A, affine_bias, and forcing_vector_fn are populated for external solvers. "
                "Forcing uses the user-supplied callback."
            )
        else:
            metadata["notes"] = (
                "Linear semidiscrete JAX FEAX block assembled. "
                "M, A, affine_bias, and forcing_vector_fn are populated for external solvers."
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
            forcing_vector_fn=forcing_vector_fn,
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
        affine_bias=None,
        forcing_vector_fn=None,
        feax_mesh=feax_mesh,
        forcing_mode="embedded_residual",
        nonlinear_runtime=nonlinear_runtime,
    )
    