from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import jax.numpy as jnp
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

    @property
    def volume_value_terms(self) -> List[Placeholder]:
        out: List[Placeholder] = []
        for t in self.terms:
            if t.support != "volume":
                continue
            kind, coeff = _strip_single_test_basis_factor(self.domain, t.expr)
            if kind == "value":
                out.append(coeff)
        return out


    @property
    def boundary_value_terms(self) -> Dict[str, List[Placeholder]]:
        out: Dict[str, List[Placeholder]] = {}
        for t in self.terms:
            if t.support != "boundary":
                continue
            kind, coeff = _strip_single_test_basis_factor(self.domain, t.expr)
            if kind == "grad":
                raise NotImplementedError(
                    f"Boundary grad(test) terms are not supported yet for VPINN on region '{t.region_id}'."
                )
            out.setdefault(t.region_id, []).append(coeff)
        return out

    @property
    def volume_value_expr(self):
        return _sum_terms(self.domain, self.volume_value_terms)

    @property
    def volume_grad_terms(self) -> Dict[int, List[Placeholder]]:
        """
        Return grad(test) coefficients bucketed by derivative direction.

        Example for 2D scalar Poisson:
            {
                0: [u_x term coeffs...],
                1: [u_y term coeffs...],
            }
        """
        out: Dict[int, List[Placeholder]] = {}
        for t in self.terms:
            if t.support != "volume":
                continue

            if not _contains_testfunction_gradient(self.domain, t.expr):
                continue

            if not (isinstance(t.expr, BinaryOp) and t.expr.op == "*"):
                raise ValueError(
                    f"Expected grad(test) weak-form term to be a product, got {t.expr}"
                )

            left = t.expr.left
            right = t.expr.right

            if _is_testfunction_grad(left):
                dim = _get_testfunction_grad_dim(left)
                coeff = right
            elif _is_testfunction_grad(right):
                dim = _get_testfunction_grad_dim(right)
                coeff = left
            else:
                raise ValueError(
                    f"Expected one grad(TestFunction) factor in term, got {t.expr}"
                )

            out.setdefault(dim, []).append(coeff)

        return out


    @property
    def volume_grad_expr(self):
        """
        Build a vector coefficient field for grad(test) assembly.

        Returns shape-like traced object corresponding to:
            stack([sum(coeffs_dim0), sum(coeffs_dim1), ...], axis=-1)
        """
        by_dim = self.volume_grad_terms
        if len(by_dim) == 0:
            return None

        max_dim = max(by_dim.keys())
        comps = []

        for d in range(max_dim + 1):
            comp = _sum_terms(self.domain, by_dim.get(d, []))
            if comp is None:
                comp = Literal(0.0)
            comps.append(comp)

        return FunctionCall(lambda *xs: jnp.stack(xs, axis=-1), comps, name="stack_grad_coeff")

    @property
    def boundary_value_exprs(self) -> Dict[str, Placeholder]:
        return {k: _sum_terms(self.domain, v) for k, v in self.boundary_value_terms.items()}


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

def _contains_testfunction_gradient(domain, expr):
    if isinstance(expr, Jacobian) and isinstance(expr.target, TestFunction):
        return True

    for attr in ("left", "right", "operand", "args", "expr", "integrand", "target", "variables"):
        if hasattr(expr, attr):
            child = getattr(expr, attr)
            if isinstance(child, (list, tuple)):
                for c in child:
                    if _contains_testfunction_gradient(domain, c):
                        return True
            elif child is not None:
                if _contains_testfunction_gradient(domain, child):
                    return True

    return False

def _is_testfunction_value(node):
    return isinstance(node, TestFunction)


def _is_testfunction_grad(node):
    return isinstance(node, Jacobian) and isinstance(node.target, TestFunction)


def _strip_single_test_basis_factor(domain, expr):
    """
    For scalar weak forms, strip exactly one test-function factor from a term.

    Supported recursively through nested multiplications:
      coeff * phi
      phi * coeff
      coeff * grad(phi)
      grad(phi) * coeff
      a * (b * phi)
      a * (b * grad(phi))
      etc.

    Returns
    -------
    kind, coeff_expr
        kind = "value" or "grad"
        coeff_expr = expr with exactly one test basis factor removed
    """
    if _is_testfunction_value(expr):
        return "value", Literal(1.0)

    if _is_testfunction_grad(expr):
        return "grad", Literal(1.0)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left = expr.left
        right = expr.right

        # direct patterns
        if _is_testfunction_value(left):
            return "value", right
        if _is_testfunction_value(right):
            return "value", left

        if _is_testfunction_grad(left):
            return "grad", right
        if _is_testfunction_grad(right):
            return "grad", left

        # recursive patterns: try stripping from left subtree
        try:
            kind, coeff_left = _strip_single_test_basis_factor(domain, left)
            return kind, coeff_left * right
        except ValueError:
            pass

        # recursive patterns: try stripping from right subtree
        try:
            kind, coeff_right = _strip_single_test_basis_factor(domain, right)
            return kind, left * coeff_right
        except ValueError:
            pass

    raise ValueError(
        "Could not strip a single TestFunction basis factor from weak-form term. "
        f"Unsupported term structure: {expr}"
    )

def _get_testfunction_grad_dim(expr):
    """
    Return the physical derivative direction index for a grad(TestFunction) term.

    Expected form:
        Jacobian(TestFunction(...), [Variable(...)])

    Returns
    -------
    int
        derivative direction, e.g. 0 for x, 1 for y
    """
    if not (isinstance(expr, Jacobian) and isinstance(expr.target, TestFunction)):
        raise ValueError(f"Expected Jacobian(TestFunction(...)), got {expr}")

    if len(expr.variables) != 1:
        raise ValueError(f"Expected one differentiation variable in {expr}")

    var = expr.variables[0]
    if not hasattr(var, "dim") or not isinstance(var.dim, (list, tuple)):
        raise ValueError(f"Could not infer grad direction from variable {var}")

    return int(var.dim[0])

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
        vol_val = _rebind_variational_variables(domain, node.volume_value_expr, target_support, target_region_id) if node.volume_value_expr is not None else None
        vol_grad = _rebind_variational_variables(domain, node.volume_grad_expr, target_support, target_region_id) if node.volume_grad_expr is not None else None
        bnd_exprs = {k: _rebind_variational_variables(domain, v, target_support, target_region_id) for k, v in node.boundary_value_exprs.items()}
        if (
            vol_val is not node.volume_value_expr
            or vol_grad is not node.volume_grad_expr
            or any(bnd_exprs[k] is not node.boundary_value_exprs[k] for k in bnd_exprs)
        ):
            rebuilt = GroupedAssembly(vol_val, vol_grad, bnd_exprs, node.num_total_nodes)
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
        vol_val = _substitute_trial_for_vpinn(domain, node.volume_value_expr, trial_value, target_support, target_region_id) if node.volume_value_expr is not None else None
        vol_grad = _substitute_trial_for_vpinn(domain, node.volume_grad_expr, trial_value, target_support, target_region_id) if node.volume_grad_expr is not None else None
        bnd_exprs = {k: _substitute_trial_for_vpinn(domain, v, trial_value, target_support, target_region_id) for k, v in node.boundary_value_exprs.items()}
        if (
            vol_val is not node.volume_value_expr
            or vol_grad is not node.volume_grad_expr
            or any(bnd_exprs[k] is not node.boundary_value_exprs[k] for k in bnd_exprs)
        ):
            rebuilt = GroupedAssembly(vol_val, vol_grad, bnd_exprs, node.num_total_nodes)
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
            if for_target == "vpinn" and _contains_node_type(domain, term_for_target, TrialFunction):
                raise RuntimeError(
                    "VPINN lowering failed: a TrialFunction is still present after substitution."
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
        u_net = kwargs.get("u_net", None)

        # Only require u_net when the weak form actually contains a TrialFunction.
        has_trial = _contains_node_type(domain, expr, TrialFunction)

        if has_trial and u_net is None:
            raise ValueError(
                "For target='vpinn', you must pass u_net=... when the weak form contains a TrialFunction."
            )

        ir = lower_weak_form(
            domain,
            expr,
            trial_value=u_net,
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

    raise ValueError(
        f"Unknown assembly target '{target}'. "
        "Supported: 'vpinn', 'fem_system', 'fem_residual'"
    )


def _assemble_vpinn_from_ir(ir: LoweredWeakForm, **kwargs):
    if (
        ir.volume_value_expr is None
        and ir.volume_grad_expr is None
        and len(ir.boundary_value_exprs) == 0
    ):
        raise ValueError("No terms found for VPINN assembly.")

    num_total_nodes = int(ir.domain.fem_context["num_total_nodes"])

    return GroupedAssembly(
        ir.volume_value_expr,
        ir.volume_grad_expr,
        ir.boundary_value_exprs,
        num_total_nodes,
    )

