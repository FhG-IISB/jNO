from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..trace import (
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
)


# -----------------------------------------------------------------------------
# Backend-neutral weak-form IR
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LoweredWeakTerm:
    sign: float
    expr: Placeholder
    support: str  # "volume" | "boundary"
    region_id: str
    original_expr: Placeholder


@dataclass
class LoweredWeakForm:
    domain: object
    terms: List[LoweredWeakTerm] = field(default_factory=list)

    @property
    def volume_terms(self) -> List[Placeholder]:
        return [t.expr for t in self.terms if t.support == "volume"]

    @property
    def boundary_terms(self) -> Dict[str, List[Placeholder]]:
        out: Dict[str, List[Placeholder]] = {}
        for term in self.terms:
            if term.support == "boundary":
                out.setdefault(term.region_id, []).append(term.expr)
        return out

    @property
    def volume_expr(self):
        return _sum_terms(self.domain, self.volume_terms)

    @property
    def boundary_exprs(self) -> Dict[str, Placeholder]:
        return {k: _sum_terms(self.domain, v) for k, v in self.boundary_terms.items()}


# --------------------------------
# additive-term helpers
# --------------------------------
def _split_additive_terms(domain, node, sign=1.0):
    if isinstance(node, BinaryOp) and node.op == "+":
        return _split_additive_terms(domain, node.left, sign) + _split_additive_terms(domain, node.right, sign)
    if isinstance(node, BinaryOp) and node.op == "-":
        return _split_additive_terms(domain, node.left, sign) + _split_additive_terms(domain, node.right, -sign)
    return [(sign, node)]


def _apply_sign(domain, sign, term):
    if sign == 1.0:
        return term
    return Literal(sign) * term


def _sum_terms(domain, terms):
    if len(terms) == 0:
        return None
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


# --------------------------------
# variational region helpers
# --------------------------------
def _contains_node_type(domain, expr, node_type):
    if isinstance(expr, node_type):
        return True

    for attr in ("left", "right", "operand", "args", "expr", "integrand"):
        if hasattr(expr, attr):
            child = getattr(expr, attr)
            if isinstance(child, (list, tuple)):
                for c in child:
                    if _contains_node_type(domain, c, node_type):
                        return True
            elif child is not None:
                if _contains_node_type(domain, child, node_type):
                    return True

    return False


def _collect_variational_metas(domain, node, out):
    if node is None:
        return

    if isinstance(node, Variable) and getattr(node, "fem_meta", None) is not None:
        out.append(node.fem_meta)
        return

    for attr in ("left", "right", "target", "expr"):
        child = getattr(node, attr, None)
        if child is not None:
            _collect_variational_metas(domain, child, out)

    for attr in ("args", "variables"):
        vals = getattr(node, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                for vv in v:
                    _collect_variational_metas(domain, vv, out)
            else:
                _collect_variational_metas(domain, v, out)


def _infer_term_bucket(domain, term):
    metas = []
    _collect_variational_metas(domain, term, metas)

    if len(metas) > 0:
        support = metas[0]["support"]
        region_id = metas[0]["region_id"]
        for m in metas[1:]:
            if m["support"] != support or m["region_id"] != region_id:
                raise ValueError(
                    "Weak-form term mixes variational regions. "
                    f"Found both ({support}, {region_id}) and ({m['support']}, {m['region_id']})."
                )
        return support, region_id

    if _contains_node_type(domain, term, TrialFunction) or _contains_node_type(domain, term, TestFunction):
        return "volume", "volume"

    raise ValueError(
        "Could not infer weak-form support for term. "
        "Use variables sampled on fem_gauss / gauss_<tag> inside the term or include TrialFunction/TestFunction."
    )


def _get_variational_region_meta(domain, support: str, region_id: str):
    registry = getattr(domain, "_variational_sampling_registry", {})
    for sample_tag, meta in registry.items():
        if meta.get("support") == support and meta.get("region_id") == region_id:
            return meta
    raise KeyError(
        f"No variational sampling meta found for support={support!r}, region_id={region_id!r}. Available: {registry}"
    )


# --------------------------------
# trial substitution / rebind
# --------------------------------
def _rebind_variational_variables(domain, node, target_support: str, target_region_id: str):
    if node is None:
        return None

    target_meta = _get_variational_region_meta(domain, target_support, target_region_id)
    target_tag = target_meta["context_tag"]

    if isinstance(node, Variable) and getattr(node, "fem_meta", None) is not None:
        if node.axis == "temporal":
            return node
        return Variable(
            tag=target_tag,
            dim=list(node.dim),
            domain=domain,
            axis=node.axis,
            fem_meta=target_meta,
        )

    if isinstance(node, (TensorTag, Constant, Literal, TrialFunction, TestFunction)):
        return node

    if isinstance(node, BinaryOp):
        left = _rebind_variational_variables(domain, node.left, target_support, target_region_id)
        right = _rebind_variational_variables(domain, node.right, target_support, target_region_id)
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = [_rebind_variational_variables(domain, a, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            return FunctionCall(node.fn, new_args, node._name, node.reduces_axis, node.kwargs)
        return node

    if isinstance(node, ModelCall):
        new_args = [_rebind_variational_variables(domain, a, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt = ModelCall(node.model, new_args)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, OperationDef):
        new_expr = _rebind_variational_variables(domain, node.expr, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt = OperationDef.__new__(OperationDef)
            rebuilt.expr = new_expr
            rebuilt.input_vars = node.input_vars
            rebuilt.name = getattr(node, "name", None)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, OperationCall):
        new_args = [_rebind_variational_variables(domain, a, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt = OperationCall(node.operation, tuple(new_args))
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, Jacobian):
        new_target = _rebind_variational_variables(domain, node.target, target_support, target_region_id)
        new_vars = [_rebind_variational_variables(domain, v, target_support, target_region_id) if isinstance(v, Placeholder) else v for v in node.variables]
        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Jacobian(new_target, new_vars, node.scheme)
        return node

    if isinstance(node, Hessian):
        new_target = _rebind_variational_variables(domain, node.target, target_support, target_region_id)
        new_vars = [_rebind_variational_variables(domain, v, target_support, target_region_id) if isinstance(v, Placeholder) else v for v in node.variables]
        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Hessian(new_target, new_vars, node.scheme)
        return node

    if isinstance(node, Tracker):
        new_expr = _rebind_variational_variables(domain, node.expr, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt = Tracker(new_expr, interval=node.interval)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, Assembly):
        new_expr = _rebind_variational_variables(domain, node.expr, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt = Assembly(new_expr, node.num_total_nodes, node.support, node.region_id)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, GroupedAssembly):
        vol_expr = _rebind_variational_variables(domain, node.volume_expr, target_support, target_region_id) if node.volume_expr is not None else None
        bnd_exprs = {k: _rebind_variational_variables(domain, v, target_support, target_region_id) for k, v in node.boundary_exprs.items()}
        if vol_expr is not node.volume_expr or any(bnd_exprs[k] is not node.boundary_exprs[k] for k in bnd_exprs):
            rebuilt = GroupedAssembly(vol_expr, bnd_exprs, node.num_total_nodes)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    return node


def _substitute_trial_for_vpinn(domain, node, trial_value, target_support: Optional[str] = None, target_region_id: Optional[str] = None):
    if node is None:
        return None

    if isinstance(node, (Variable, TestFunction, TensorTag, Constant, Literal)):
        return node

    if isinstance(node, TrialFunction):
        out = trial_value
        if target_support is not None and target_region_id is not None:
            out = _rebind_variational_variables(domain, out, target_support, target_region_id)
        return out

    if isinstance(node, BinaryOp):
        left = _substitute_trial_for_vpinn(domain, node.left, trial_value, target_support, target_region_id)
        right = _substitute_trial_for_vpinn(domain, node.right, trial_value, target_support, target_region_id)
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = [_substitute_trial_for_vpinn(domain, a, trial_value, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            return FunctionCall(node.fn, new_args, node._name, node.reduces_axis, node.kwargs)
        return node

    if isinstance(node, ModelCall):
        new_args = [_substitute_trial_for_vpinn(domain, a, trial_value, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt = ModelCall(node.model, new_args)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, OperationDef):
        new_expr = _substitute_trial_for_vpinn(domain, node.expr, trial_value, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt = OperationDef.__new__(OperationDef)
            rebuilt.expr = new_expr
            rebuilt.input_vars = node.input_vars
            rebuilt.name = getattr(node, "name", None)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, OperationCall):
        new_args = [_substitute_trial_for_vpinn(domain, a, trial_value, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt = OperationCall(node.operation, tuple(new_args))
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, Jacobian):
        new_target = _substitute_trial_for_vpinn(domain, node.target, trial_value, target_support, target_region_id)
        new_vars = [_substitute_trial_for_vpinn(domain, v, trial_value, target_support, target_region_id) if isinstance(v, Placeholder) else v for v in node.variables]
        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Jacobian(new_target, new_vars, node.scheme)
        return node

    if isinstance(node, Hessian):
        new_target = _substitute_trial_for_vpinn(domain, node.target, trial_value, target_support, target_region_id)
        new_vars = [_substitute_trial_for_vpinn(domain, v, trial_value, target_support, target_region_id) if isinstance(v, Placeholder) else v for v in node.variables]
        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Hessian(new_target, new_vars, node.scheme)
        return node

    if isinstance(node, Tracker):
        new_expr = _substitute_trial_for_vpinn(domain, node.expr, trial_value, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt = Tracker(new_expr, interval=node.interval)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, Assembly):
        new_expr = _substitute_trial_for_vpinn(domain, node.expr, trial_value, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt = Assembly(new_expr, node.num_total_nodes, node.support, node.region_id)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, GroupedAssembly):
        vol_expr = _substitute_trial_for_vpinn(domain, node.volume_expr, trial_value, target_support, target_region_id) if node.volume_expr is not None else None
        bnd_exprs = {k: _substitute_trial_for_vpinn(domain, v, trial_value, target_support, target_region_id) for k, v in node.boundary_exprs.items()}
        if vol_expr is not node.volume_expr or any(bnd_exprs[k] is not node.boundary_exprs[k] for k in bnd_exprs):
            rebuilt = GroupedAssembly(vol_expr, bnd_exprs, node.num_total_nodes)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    return node


# -----------------------------------------------------------------------------
# Lower once, dispatch many
# -----------------------------------------------------------------------------

def lower_weak_form(domain, expr, *, trial_value=None, for_target: str = "fem") -> LoweredWeakForm:
    terms = _split_additive_terms(domain, expr)
    lowered_terms: List[LoweredWeakTerm] = []

    for sign, term in terms:
        support, region_id = _infer_term_bucket(domain, term)
        term_for_target = term

        if for_target == "vpinn" and trial_value is not None:
            term_for_target = _substitute_trial_for_vpinn(
                domain,
                term,
                trial_value,
                target_support=support,
                target_region_id=region_id,
            )

        lowered_terms.append(
            LoweredWeakTerm(
                sign=sign,
                expr=_apply_sign(domain, sign, term_for_target),
                support=support,
                region_id=region_id,
                original_expr=term,
            )
        )

    return LoweredWeakForm(domain=domain, terms=lowered_terms)


def assemble_weak_form(domain, expr, target="vpinn", **kwargs):
    if target == "vpinn":
        ir = lower_weak_form(domain, expr, trial_value=kwargs.get("u_net", None), for_target="vpinn")
        return _assemble_vpinn_from_ir(ir, **kwargs)

    if target in {"fem_system", "fem_residual"}:
        ir = lower_weak_form(domain, expr, for_target="fem")
        if target == "fem_system":
            from .fem_route import _assemble_fem_system_from_ir
            return _assemble_fem_system_from_ir(domain, ir, **kwargs)
        from .fem_route import _assemble_fem_residual_from_ir
        return _assemble_fem_residual_from_ir(domain, ir, **kwargs)

    raise ValueError(f"Unknown assembly target '{target}'. Supported: 'vpinn', 'fem_system', 'fem_residual'")


def _assemble_vpinn_from_ir(ir: LoweredWeakForm, **kwargs):
    if ir.volume_expr is None and len(ir.boundary_exprs) == 0:
        raise ValueError("No terms found for VPINN assembly.")
    return GroupedAssembly(ir.volume_expr, ir.boundary_exprs, ir.domain)

