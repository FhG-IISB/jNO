from __future__ import annotations

"""
Weak-form lowering and assembly dispatcher.

This module is the central routing layer behind:

    expr.assemble(target=...)

It converts symbolic jNO weak forms into backend-neutral IR and dispatches that
IR to VPINN, steady FEM, transient FEAX-time, or strong-form Diffrax routes.

Main responsibilities:
- represent lowered weak-form terms through LoweredChannelTerm / LoweredWeakForm,
- infer the solver target when target=None,
- lower weak expressions into VPINN or FEM-compatible IR,
- assemble VPINN GroupedAssembly objects for training,
- dispatch linear/nonlinear steady FEM assembly,
- dispatch transient weak-form FEAX-time assembly,
- dispatch strong-form time-dependent expressions to Diffrax.

Supported assemble targets:
    "vpinn"        -> GroupedAssembly
    "fem_system"   -> (A, b)
    "fem_residual" -> FemResidualOperator
    "feax_time"    -> FeaxTimeBlock
    "diffrax"      -> DiffraxBlock
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, cast
import jax.numpy as jnp
from ...jnp_ops import stack
from ...trace import (
    Placeholder,
    Literal,
    BinaryOp,
    FunctionCall,
    Variable,
    ModelCall,
    OperationDef,
    OperationCall,
    Jacobian,
    Hessian,
    Tracker,
    TrialFunction,
    TestFunction,
    TensorTag,
    Constant,
    Assembly,
    GroupedAssembly,
    StateField,
)
from .solver_helper import (
    sum_terms as _sum_terms,
    apply_sign as _apply_sign,
    contains_node_type as _contains_node_type,
    contains_testfunction as _contains_testfunction,
    contains_trialfunction as _contains_trialfunction,
    iter_placeholder_children as _iter_placeholder_children,
    contains_subexpr as _contains_subexpr,
    contains_model_call as _contains_model_eval,
    depends_on_domain_variables as _depends_on_domain_variables,
    unique_by_id as _unique_by_id,
    contains_temporal_derivative as _contains_temporal_derivative,
)

from .weak_form_helpers import (
    split_weak_additive_terms,
    function_name as _function_name,
    get_grad_axis_from_test_grad as _get_grad_axis_from_test_grad,
    canonicalize_grad_coeff as _canonicalize_grad_coeff,
    value_shape_num_components as _value_shape_num_components,
    is_test_value as _is_test_value,
    is_test_grad as _is_test_grad,
    is_symgrad_test as _is_symgrad_test,
    get_test_value_shape as _get_test_value_shape,
    contains_testfunction_gradient as _contains_testfunction_gradient,
    has_weak_basis_symbols as _has_weak_basis_symbols,
    collect_region_keys as _collect_region_keys,
    collect_variational_metas as _collect_variational_metas,
    infer_term_bucket as _infer_term_bucket,
    get_variational_region_meta as _get_variational_region_meta,
    find_first_statefield as _find_first_statefield,
    infer_state_value_shape as _infer_state_value_shape,
    # state-field / trial rewriters
    wrap_primary_state as _wrap_primary_state,
    ensure_statefield_wrapped as _ensure_statefield_wrapped,
    is_statefield_candidate as _is_statefield_candidate,
    collect_state_field_candidates as _collect_state_field_candidates,
    collect_derivative_based_state_targets as _collect_derivative_based_state_targets,
    detect_primary_state_field as _detect_primary_state_field,
    rebind_variational_variables as _rebind_variational_variables,
    bind_statefield_for_vpinn as _bind_statefield_for_vpinn,
    bind_statefield_for_fem as _bind_statefield_for_fem,
    substitute_trial_for_vpinn as _substitute_trial_for_vpinn,
)

# -----------------------------------------------------------------------------
# Backend-neutral weak-form IR
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweredChannelTerm:
    """
    One lowered weak-form term in backend-neutral IR.

    A symbolic weak-form expression is first split into additive terms. Each
    additive term is then classified by support, region, and channel.

    Fields
    ------
    sign:
        Original additive sign of the term, usually +1.0 or -1.0.
    support:
        Geometric support of the term. Usually `"volume"` or `"boundary"`.
    region_id:
        Region identifier. For volume terms this is usually `"volume"`.
        For boundary terms this is the boundary tag, for example `"left"`,
        `"right"`, or `"top"`.
    channel:
        Lowered channel name.

        For VPINN lowering:
            `"test_value"`:
                volume term multiplying the test function value.
            `"test_grad"`:
                volume term multiplying the test function gradient.
            `"boundary_test_value"`:
                boundary term multiplying the test function value.

        For FEM/FEAX lowering:
            `"raw"`:
                original weak-form term kept in raw symbolic form.
    coeff:
        Symbolic coefficient expression after channel extraction.
    variable_id:
        Reserved id for multi-field systems. Currently usually 0.
    value_shape:
        Value shape of the weak unknown/test function.
    original_expr:
        Original symbolic term before channel extraction.
    """

    sign: float
    support: str  # "volume" | "boundary"
    region_id: str
    channel: str  # "test_value" | "test_grad" | "boundary_test_value" | "raw"
    coeff: Placeholder
    variable_id: int = 0
    value_shape: tuple = ()
    original_expr: Placeholder = None


@dataclass
class LoweredWeakForm:
    """
    Backend-neutral lowered representation of a weak form.

    `LoweredWeakForm` stores a list of `LoweredChannelTerm` objects and exposes
    convenience properties for the different backend routes.

    FEM/FEAX routes use:
        volume_expr:
            Sum of raw volume weak-form expressions.
        boundary_exprs:
            Dictionary mapping boundary region id to summed raw boundary
            expression.

    VPINN routes use:
        volume_value_expr:
            Sum of coefficients multiplying the test-function value.
        volume_grad_expr:
            Sum of coefficients multiplying the test-function gradient.
        boundary_value_exprs:
            Dictionary mapping boundary region id to coefficients multiplying
            boundary test-function values.

    Parameters
    ----------
    domain:
        Domain that owns the quadrature/FEM context.
    terms:
        Lowered weak-form terms.
    """

    domain: Any
    terms: List[LoweredChannelTerm] = field(default_factory=list)

    def select(self, *, support=None, region_id=None, channel=None, variable_id=None):
        """
        Select lowered terms matching optional support, region, channel, and variable id.
        """
        out = []
        for t in self.terms:
            if support is not None and t.support != support:
                continue
            if region_id is not None and t.region_id != region_id:
                continue
            if channel is not None and t.channel != channel:
                continue
            if variable_id is not None and t.variable_id != variable_id:
                continue
            out.append(t)
        return out

    def sum_coeffs(self, *, support=None, region_id=None, channel=None, variable_id=None):
        """
        Sum coefficients of all lowered terms matching the given filters.

        Returns None if no matching terms exist.
        """
        coeffs = [
            t.coeff
            for t in self.select(
                support=support,
                region_id=region_id,
                channel=channel,
                variable_id=variable_id,
            )
        ]
        return _sum_terms(self.domain, coeffs)

    # FEM-facing raw expressions
    @property
    def volume_expr(self):
        """
        Raw summed volume expression used by FEM/FEAX assembly routes.
        """
        exprs = [t.coeff for t in self.terms if t.support == "volume" and t.channel == "raw"]
        return _sum_terms(self.domain, exprs)

    @property
    def boundary_exprs(self) -> Dict[str, Placeholder]:
        """
        Raw summed boundary expressions grouped by boundary region id.

        Returns:
            dict mapping region_id -> symbolic boundary expression.
        """
        out: Dict[str, List[Placeholder]] = {}
        for t in self.terms:
            if t.support == "boundary" and t.channel == "raw":
                out.setdefault(t.region_id, []).append(t.coeff)
        return {k: _sum_terms(self.domain, v) for k, v in out.items()}

    # VPINN-facing canonical channels
    @property
    def volume_value_expr(self):
        return self.sum_coeffs(support="volume", channel="test_value", variable_id=0)

    @property
    def volume_grad_expr(self):
        return self.sum_coeffs(support="volume", channel="test_grad", variable_id=0)

    @property
    def boundary_value_exprs(self) -> Dict[str, Placeholder]:
        out = {}
        region_ids = sorted({t.region_id for t in self.terms if t.support == "boundary"})
        for rid in region_ids:
            coeff = self.sum_coeffs(
                support="boundary",
                region_id=rid,
                channel="boundary_test_value",
                variable_id=0,
            )
            if coeff is not None:
                out[rid] = coeff
        return out


# --------------------------------
# additive-term helpers
# --------------------------------
def _can_bucket_as_single_term(domain, node):
    try:
        return _infer_term_bucket(domain, node)
    except Exception:
        return None


def _split_additive_terms(domain, node, sign=1.0):
    return split_weak_additive_terms(
        domain,
        node,
        sign,
        infer_term_bucket=_infer_term_bucket,
    )


def _extract_test_channel(domain, expr) -> Tuple[str, Placeholder, Dict[str, Any]]:
    """
    Extract the VPINN test-function channel from a signed weak-form term.

    Returns:
        channel:
            `"test_value"` or `"test_grad"`.
        coeff:
            Symbolic coefficient multiplying the test channel.
        meta:
            Metadata such as `variable_id` and `value_shape`.

    Supported patterns include terms multiplying `TestFunction`, spatial
    `Jacobian(TestFunction)`, and selected symmetric-gradient test expressions.

    Raises:
        NotImplementedError if the term cannot be reduced to a supported VPINN
        test-function channel.
    """

    # pure test value
    if isinstance(expr, TestFunction):
        return (
            "test_value",
            Literal(1.0),
            {
                "value_shape": getattr(expr, "value_shape", ()),
                "variable_id": 0,
            },
        )

    # pure grad(test)
    if isinstance(expr, Jacobian) and isinstance(expr.target, TestFunction):
        value_shape = getattr(expr.target, "value_shape", ())
        axis = _get_grad_axis_from_test_grad(expr)
        coeff = _canonicalize_grad_coeff(domain, Literal(1.0), axis, value_shape)
        return (
            "test_grad",
            coeff,
            {
                "value_shape": value_shape,
                "variable_id": 0,
            },
        )
    # additive combination of same-channel terms
    if isinstance(expr, BinaryOp) and expr.op in {"+", "-"}:
        ch_left, coeff_left, meta_left = _extract_test_channel(domain, expr.left)
        ch_right, coeff_right, meta_right = _extract_test_channel(domain, expr.right)

        if ch_left != ch_right:
            raise ValueError("Could not extract a single canonical test channel from additive term: " f"left channel={ch_left}, right channel={ch_right}, expr={expr}")

        if meta_left.get("variable_id", 0) != meta_right.get("variable_id", 0):
            raise ValueError("Additive weak-form term mixes different variable ids: " f"{meta_left.get('variable_id', 0)} vs {meta_right.get('variable_id', 0)}")

        if tuple(meta_left.get("value_shape", ())) != tuple(meta_right.get("value_shape", ())):
            raise ValueError("Additive weak-form term mixes different test value shapes: " f"{meta_left.get('value_shape', ())} vs {meta_right.get('value_shape', ())}")

        if expr.op == "+":
            coeff = coeff_left + coeff_right
        else:
            coeff = coeff_left - coeff_right

        return ch_left, coeff, meta_left
    # direct product tree
    if isinstance(expr, BinaryOp) and expr.op == "*":
        left = expr.left
        right = expr.right

        if _is_test_value(left):
            return (
                "test_value",
                right,
                {
                    "value_shape": getattr(left, "value_shape", ()),
                    "variable_id": 0,
                },
            )
        if _is_test_value(right):
            return (
                "test_value",
                left,
                {
                    "value_shape": getattr(right, "value_shape", ()),
                    "variable_id": 0,
                },
            )

        if _is_test_grad(left):
            left_grad = cast(Jacobian, left)
            value_shape = getattr(left_grad.target, "value_shape", ())
            axis = _get_grad_axis_from_test_grad(left_grad)
            coeff = _canonicalize_grad_coeff(domain, right, axis, value_shape)
            return (
                "test_grad",
                coeff,
                {
                    "value_shape": value_shape,
                    "variable_id": 0,
                },
            )

        if _is_test_grad(right):
            right_grad = cast(Jacobian, right)
            value_shape = getattr(right_grad.target, "value_shape", ())
            axis = _get_grad_axis_from_test_grad(right_grad)
            coeff = _canonicalize_grad_coeff(domain, left, axis, value_shape)
            return (
                "test_grad",
                coeff,
                {
                    "value_shape": value_shape,
                    "variable_id": 0,
                },
            )
        # recurse left
        try:
            channel, coeff_left, meta = _extract_test_channel(domain, left)
            return channel, coeff_left * right, meta
        except ValueError:
            pass

        # recurse right
        try:
            channel, coeff_right, meta = _extract_test_channel(domain, right)
            return channel, left * coeff_right, meta
        except ValueError:
            pass

    # contractions
    if isinstance(expr, FunctionCall):
        name = _function_name(expr)

        if name == "inner" and len(expr.args) >= 2:
            a0, a1 = expr.args[0], expr.args[1]

            # inner(a, phi) or inner(phi, a)
            if _is_test_value(a0):
                return (
                    "test_value",
                    a1,
                    {
                        "value_shape": getattr(a0, "value_shape", ()),
                        "variable_id": 0,
                    },
                )
            if _is_test_value(a1):
                return (
                    "test_value",
                    a0,
                    {
                        "value_shape": getattr(a1, "value_shape", ()),
                        "variable_id": 0,
                    },
                )

            # inner(A, grad(phi)) or inner(grad(phi), A)
            if _is_test_grad(a0):
                return (
                    "test_grad",
                    a1,
                    {
                        "value_shape": getattr(a0.target, "value_shape", ()),
                        "variable_id": 0,
                    },
                )
            if _is_test_grad(a1):
                return (
                    "test_grad",
                    a0,
                    {
                        "value_shape": getattr(a1.target, "value_shape", ()),
                        "variable_id": 0,
                    },
                )

            # elasticity-like inner(sigma, symgrad(phi), n_contract=2)
            # canonical FEAX grad channel uses sigma itself
            if _is_symgrad_test(a0):
                return (
                    "test_grad",
                    a1,
                    {
                        "value_shape": _get_test_value_shape(a0),
                        "variable_id": 0,
                    },
                )
            if _is_symgrad_test(a1):
                return (
                    "test_grad",
                    a0,
                    {
                        "value_shape": _get_test_value_shape(a1),
                        "variable_id": 0,
                    },
                )

    raise ValueError("Could not extract a canonical FEAX-style test channel from weak-form term. " f"Unsupported term structure: {expr}")


# --------------------------------
# domain infer helpers
# --------------------------------
def is_variational_expr(domain, expr) -> bool:
    """Return True if expr still looks like an unassembled weak-form expression."""
    if expr is None:
        return False

    # Already assembled weak object -> not raw anymore
    if isinstance(expr, (Assembly, GroupedAssembly)):
        return False

    return _contains_node_type(expr, TestFunction) or _contains_node_type(expr, TrialFunction) or _contains_node_type(expr, StateField) or _contains_testfunction_gradient(domain, expr)


def _infer_domain_from_expr(expr):
    """
    Infer the owning domain from variables or weak symbols inside an expression.

    This is used when `expr.assemble(domain=None, ...)` is called. The function
    walks the symbolic expression and returns the first attached domain it finds.

    Raises:
        ValueError if no domain can be inferred.
    """
    domains = []

    def walk(node):
        if node is None:
            return

        if isinstance(node, Variable):
            d = getattr(node, "_domain", None)
            if d is not None:
                domains.append(d)

        if isinstance(node, StateField):
            walk(node.expr)
            return

        for attr in ("left", "right", "target", "expr"):
            child = getattr(node, attr, None)
            if child is not None:
                walk(child)

        for attr in ("args", "variables"):
            vals = getattr(node, attr, None)
            if vals is None:
                continue
            for v in vals:
                if isinstance(v, (list, tuple)):
                    for vv in v:
                        walk(vv)
                else:
                    walk(v)

    walk(expr)
    uniq = []
    seen = set()
    for d in domains:
        if id(d) not in seen:
            seen.add(id(d))
            uniq.append(d)

    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) == 0:
        raise ValueError("Could not infer domain from weak expression. Pass domain=... explicitly.")
    raise ValueError("Weak expression references multiple domains. Pass the intended domain explicitly.")


# --------------------------------
# auto solver helpers
# --------------------------------


def _contains_unknown_symbol(domain, expr) -> bool:
    return _contains_node_type(expr, StateField) or _contains_node_type(expr, TrialFunction)


def _is_obviously_nonlinear_in_unknown(domain, expr):
    """
    Heuristically detect nonlinear dependence on the unknown.

    This is used for steady FEM auto-routing. Linear-looking weak forms are sent
    to `target="fem_system"`, while nonlinear forms are sent to
    `target="fem_residual"`.
    """
    if expr is None:
        return False

    if isinstance(expr, BinaryOp):
        left_has = _contains_unknown_symbol(domain, expr.left)
        right_has = _contains_unknown_symbol(domain, expr.right)

        # product/division of two unknown-dependent factors -> nonlinear
        if expr.op in {"*", "/"} and left_has and right_has:
            return True

        # powers of the unknown are nonlinear except u**1
        if expr.op == "**":
            if left_has:
                # u**1 is linear, everything else is nonlinear
                if isinstance(expr.right, Literal):
                    try:
                        p = float(expr.right.value)
                        if p != 1.0:
                            return True
                    except Exception:
                        return True
                else:
                    return True

            # exponent depending on unknown is definitely nonlinear
            if right_has:
                return True

        return _is_obviously_nonlinear_in_unknown(domain, expr.left) or _is_obviously_nonlinear_in_unknown(domain, expr.right)

    if isinstance(expr, FunctionCall):
        unknown_args = [a for a in expr.args if isinstance(a, Placeholder) and _contains_unknown_symbol(domain, a)]

        name = _function_name(expr)

        # Structural / linear-ish wrappers that should not force nonlinear classification
        linearish = {
            "inner",
            "reshape",
            "transpose",
            "getitem",
            "concat",
            "stack",
            "symgrad",
            "trace",
            "einsum",
        }

        # Jacobian/Hessian are handled through their own nodes
        if len(unknown_args) > 0 and name not in linearish:
            return True

        return any(_is_obviously_nonlinear_in_unknown(domain, a) for a in expr.args if isinstance(a, Placeholder))

    if isinstance(expr, (Jacobian, Hessian)):
        return _is_obviously_nonlinear_in_unknown(domain, expr.target)

    return False


def _infer_solver_target(domain, expr):
    """
    Infer the default assembly target for a symbolic expression.

    Rules:
    - strong time-dependent expressions without weak symbols -> `"diffrax"`
    - weak time-dependent expressions -> `"feax_time"`
    - steady weak forms that are linear in the unknown -> `"fem_system"`
    - steady weak forms that are nonlinear in the unknown -> `"fem_residual"`

    The inference is only used when `target=None`.
    """
    if _contains_temporal_derivative(expr):
        if _contains_node_type(expr, TestFunction) or _contains_node_type(expr, TrialFunction) or _contains_node_type(expr, StateField):
            if getattr(domain, "_feax_context", None) is not None:
                return "feax_time"
            raise ValueError("A time-dependent weak form was detected, but domain.init_fem(...) " "has not been called. For transient weak forms, initialize FEM first " "and use target='feax_time' (or let auto-inference choose it).")

        return "diffrax"

    if _is_obviously_nonlinear_in_unknown(domain, expr):
        return "fem_residual"
    return "fem_system"


# -----------------------------------------------------------------------------
# Lower once, dispatch many
# -----------------------------------------------------------------------------
def lower_weak_form(domain, expr, trial_value=None, for_target="vpinn"):
    """
    Lower a symbolic weak-form expression into backend-neutral weak-form IR.

    Parameters
    ----------
    domain:
        Domain that owns the sampled quadrature/FEM context.
    expr:
        Symbolic weak-form expression.
    trial_value:
        Optional neural trial expression used when substituting TrialFunction
        symbols for VPINN training.
    for_target:
        Lowering mode.

        `"vpinn"`:
            Converts each term into canonical VPINN channels:
            test-value, test-gradient, and boundary-test-value.

        `"fem"`:
            Keeps each weak-form term in raw symbolic form so FEAX can evaluate
            TrialFunction/TestFunction values and gradients directly.

    Returns
    -------
    LoweredWeakForm
        Backend-neutral IR containing lowered weak-form terms.

    Notes
    -----
    StateField nodes are rebound differently depending on the target:

    - VPINN:
        StateField is replaced by the wrapped neural expression and rebound to
        the active quadrature region.

    - FEM/FEAX:
        StateField is replaced by a shared TrialFunction so FEAX can assemble
        matrix/residual operators.
    """
    # print("DEBUG lower_weak_form has StateField before wrap:",
    # _contains_node_type(domain, expr, StateField))
    # expr = _ensure_statefield_wrapped(domain, expr)
    # print("DEBUG lower_weak_form has StateField after wrap:",
    # _contains_node_type(domain, expr, StateField))
    shared_trial_symbol = None
    if for_target == "fem":
        sf = _find_first_statefield(expr)
        if sf is not None:
            shared_trial_symbol = TrialFunction(name=sf.name, value_shape=sf.value_shape)

    terms = _split_additive_terms(domain, expr)
    lowered_terms = []

    for sign, term in terms:
        support, region_id = _infer_term_bucket(domain, term)
        term_for_target = term

        if for_target == "vpinn":
            term_for_target = _bind_statefield_for_vpinn(
                domain,
                term_for_target,
                target_support=support,
                target_region_id=region_id,
            )

            if trial_value is not None:
                term_for_target = _substitute_trial_for_vpinn(
                    domain,
                    term_for_target,
                    trial_value,
                    target_support=support,
                    target_region_id=region_id,
                )

        else:
            term_for_target = _bind_statefield_for_fem(
                term_for_target,
                trial_symbol=shared_trial_symbol,
            )

        signed_expr = _apply_sign(domain, sign, term_for_target)

        if for_target == "vpinn":
            channel, coeff, meta = _extract_test_channel(domain, signed_expr)

            if support == "boundary":
                if channel != "test_value":
                    raise NotImplementedError(f"Boundary grad(test)-type terms are not supported on region '{region_id}'.")
                channel = "boundary_test_value"

            lowered_terms.append(
                LoweredChannelTerm(
                    sign=sign,
                    support=support,
                    region_id=region_id,
                    channel=channel,
                    coeff=coeff,
                    variable_id=int(meta.get("variable_id", 0)),
                    value_shape=tuple(meta.get("value_shape", ())),
                    original_expr=term,
                )
            )
        else:
            lowered_terms.append(
                LoweredChannelTerm(
                    sign=sign,
                    support=support,
                    region_id=region_id,
                    channel="raw",
                    coeff=signed_expr,
                    variable_id=0,
                    value_shape=(),
                    original_expr=term,
                )
            )

    return LoweredWeakForm(domain=domain, terms=lowered_terms)


def assemble_weak_form(domain, expr, target=None, **kwargs):
    """
    Assemble or lower a jNO expression into the requested solver backend.

    This is the central implementation behind:

        expr.assemble(target=...)

    Parameters
    ----------
    domain:
        Domain owning the expression and FEM/VPINN context. If None, the domain
        is inferred from symbolic variables inside `expr`.
    expr:
        Symbolic expression to assemble or lower.
    target:
        Backend target. Supported values:

        None:
            Infer the target automatically.

        `"vpinn"`:
            Return a differentiable `GroupedAssembly` object for VPINN training.

        `"fem_system"`:
            Return a steady linear FEM system `(A, b)` such that `A @ u = b`.

        `"fem_residual"`:
            Return a `FemResidualOperator` with residual and Jacobian callables.

        `"feax_time"`:
            Return a `FeaxTimeBlock` representing a transient semidiscrete
            weak-form FEM problem.

        `"diffrax"`:
            Return a `DiffraxBlock` for strong-form time-dependent problems.

    **kwargs:
        Backend-specific options forwarded to the selected route.

    Returns
    -------
    object
        Depending on target:

        - `"vpinn"`        -> GroupedAssembly
        - `"fem_system"`   -> tuple(A, b)
        - `"fem_residual"` -> FemResidualOperator
        - `"feax_time"`    -> FeaxTimeBlock
        - `"diffrax"`      -> DiffraxBlock

    Notes
    -----
    Strong-form Diffrax lowering is dispatched before weak-form StateField
    wrapping. All weak/FEM/VPINN/FEAX-time routes go through the weak-form
    lowering path.
    """
    if domain is None:
        domain = _infer_domain_from_expr(expr)

    # IMPORTANT:
    # Infer target first from the raw user expression.
    if target is None:
        target = _infer_solver_target(domain, expr)

    # Strong-form Diffrax lowering should not go through weak/state wrapping.
    if target == "diffrax":
        from .time_route import _assemble_diffrax_from_strong_form

        return _assemble_diffrax_from_strong_form(domain, expr, **kwargs)

    # All weak/FEM/VPINN routes still use the wrapped weak expression path.
    expr = _ensure_statefield_wrapped(domain, expr)

    if target == "vpinn":
        trial_value = kwargs.pop("trial_value", None)

        # Backward-compatible alias used in examples:
        #     weak.assemble(domain, u_net=u_gauss, target="vpinn")
        if trial_value is None:
            trial_value = kwargs.pop("u_net", None)

        ir = lower_weak_form(
            domain,
            expr,
            trial_value=trial_value,
            for_target="vpinn",
        )
        return _assemble_vpinn_from_ir(ir, **kwargs)

    if target in {"fem_system", "fem_residual"}:
        ir = lower_weak_form(domain, expr, for_target="fem")
        if target == "fem_system":
            from .fem_route import _assemble_fem_system_from_ir

            return _assemble_fem_system_from_ir(domain, ir, **kwargs)

        from .fem_route import _assemble_fem_residual_from_ir

        return _assemble_fem_residual_from_ir(domain, ir, **kwargs)

    if target == "feax_time":
        ir = lower_weak_form(domain, expr, for_target="fem")
        from .time_route import _assemble_feax_time_from_ir

        return _assemble_feax_time_from_ir(domain, ir, **kwargs)

    raise ValueError(f"Unknown assembly target '{target}'. " "Supported: 'vpinn', 'fem_system', 'fem_residual', 'feax_time', 'diffrax'")


def _assemble_vpinn_from_ir(ir: LoweredWeakForm, **kwargs):
    """
    Convert lowered VPINN weak-form IR into a differentiable GroupedAssembly.

    Returns:
        GroupedAssembly containing:

        - volume_value_expr:
            coefficient of volume test-function value terms.
        - volume_grad_expr:
            coefficient of volume test-function gradient terms.
        - boundary_value_exprs:
            boundary value terms grouped by boundary region id.
        - num_total_nodes:
            number of global FEM/test nodes.

    Notes
    -----
    This route does not solve a FEM system. It creates a differentiable weak
    residual object evaluated during neural training.
    """
    if ir.volume_value_expr is None and ir.volume_grad_expr is None and len(ir.boundary_value_exprs) == 0:
        raise ValueError("No terms found for VPINN assembly.")

    num_total_nodes = int(ir.domain.fem_context["num_total_nodes"])

    return GroupedAssembly(
        ir.volume_value_expr,
        ir.volume_grad_expr,
        ir.boundary_value_exprs,
        num_total_nodes,
    )
