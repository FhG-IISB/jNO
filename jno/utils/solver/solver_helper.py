from __future__ import annotations

from typing import Any

from ...trace import (
    Placeholder,
    Literal,
    BinaryOp,
    FunctionCall,
    Variable,
    Jacobian,
    Hessian,
    ModelCall,
    OperationDef,
    OperationCall,
    Tracker,
    Assembly,
    GroupedAssembly,
    StateField,
    TestFunction,
    TrialFunction,
)


def iter_children(node: Any):
    """
    Generic traced-expression tree walker.

    This is the central traversal utility. Keep it complete so all higher-level
    predicates such as contains_node_type, contains_model_call, and
    depends_on_domain_variables can stay small and consistent.
    """
    if node is None:
        return

    if isinstance(node, (list, tuple)):
        for item in node:
            if isinstance(item, Placeholder):
                yield item
        return

    if isinstance(node, BinaryOp):
        yield node.left
        yield node.right
        return

    if isinstance(node, FunctionCall):
        for a in node.args:
            if isinstance(a, Placeholder):
                yield a
        return

    if isinstance(node, ModelCall):
        for a in node.args:
            if isinstance(a, Placeholder):
                yield a
        return

    if isinstance(node, OperationDef):
        yield node.expr
        return

    if isinstance(node, OperationCall):
        yield node.operation
        for a in node.args:
            if isinstance(a, Placeholder):
                yield a
        return

    if isinstance(node, (Jacobian, Hessian)):
        yield node.target
        for v in node.variables:
            if isinstance(v, Placeholder):
                yield v
        return

    if isinstance(node, Tracker):
        yield node.expr
        return

    if isinstance(node, Assembly):
        yield node.expr
        return

    if isinstance(node, GroupedAssembly):
        if node.volume_value_expr is not None:
            yield node.volume_value_expr
        if node.volume_grad_expr is not None:
            yield node.volume_grad_expr
        for e in node.boundary_value_exprs.values():
            yield e
        return

    if isinstance(node, StateField):
        # Important: StateField wraps the actual symbolic unknown expression.
        yield node.expr
        return


def iter_placeholder_children(node: Any):
    """
    Alias kept for weak-form rewriters that specifically want placeholder children.
    Since iter_children already only yields traced placeholder-like children,
    this can simply forward to iter_children.
    """
    yield from iter_children(node)


def contains_node_type(node: Any, cls) -> bool:
    if isinstance(node, cls):
        return True
    return any(contains_node_type(child, cls) for child in iter_children(node) or ())


def contains_testfunction(node: Any) -> bool:
    return contains_node_type(node, TestFunction)


def contains_trialfunction(node: Any) -> bool:
    return contains_node_type(node, TrialFunction)


def contains_model_call(node: Any) -> bool:
    return contains_node_type(node, ModelCall)


def contains_model_eval(node: Any) -> bool:
    # Backward-compatible alias.
    return contains_model_call(node)


def depends_on_domain_variables(node: Any) -> bool:
    return contains_node_type(node, Variable)


def contains_subexpr(root: Any, target: Any) -> bool:
    if root is target:
        return True

    return any(contains_subexpr(child, target) for child in iter_children(root) or ())


def unique_by_id(nodes):
    out = []
    seen = set()

    for n in nodes:
        if id(n) not in seen:
            seen.add(id(n))
            out.append(n)

    return out


def sum_terms(domain, terms):
    if len(terms) == 0:
        return None

    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def apply_sign(domain, sign, term):
    if sign == 1.0:
        return term
    return Literal(sign) * term


def is_temporal_var(node: Any) -> bool:
    return isinstance(node, Variable) and getattr(node, "axis", None) == "temporal"


def max_temporal_derivative_order(node: Any) -> int:
    if isinstance(node, Jacobian):
        target_order = max_temporal_derivative_order(node.target)
        local_order = sum(1 for v in node.variables if is_temporal_var(v))
        if local_order > 0:
            return target_order + local_order
        return target_order

    if isinstance(node, Hessian):
        target_order = max_temporal_derivative_order(node.target)
        # Hessian corresponds to a second derivative operator.
        local_order = 2 * sum(1 for v in node.variables if is_temporal_var(v))
        if local_order > 0:
            return target_order + local_order
        return target_order

    if isinstance(node, BinaryOp):
        return max(max_temporal_derivative_order(node.left), max_temporal_derivative_order(node.right))

    if isinstance(node, FunctionCall):
        if not node.args:
            return 0
        return max(
            (max_temporal_derivative_order(a) for a in node.args if isinstance(a, Placeholder)),
            default=0,
        )

    return 0


def collect_temporal_tags(node: Any, out: set[str] | None = None) -> set[str]:
    if out is None:
        out = set()

    if isinstance(node, Variable) and getattr(node, "axis", None) == "temporal":
        out.add(str(node.tag))
        return out

    if isinstance(node, BinaryOp):
        collect_temporal_tags(node.left, out)
        collect_temporal_tags(node.right, out)
        return out

    if isinstance(node, FunctionCall):
        for a in node.args:
            if isinstance(a, Placeholder):
                collect_temporal_tags(a, out)
        return out

    if isinstance(node, (Jacobian, Hessian)):
        collect_temporal_tags(node.target, out)
        for v in node.variables:
            if isinstance(v, Placeholder):
                collect_temporal_tags(v, out)
        return out

    return out


def contains_temporal_derivative(node: Any) -> bool:
    return max_temporal_derivative_order(node) > 0
