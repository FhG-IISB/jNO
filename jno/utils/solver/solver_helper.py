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
    Yield direct traced-expression children of a jNO symbolic node.

    This is the central tree-walker used by generic predicates such as
    `contains_node_type`, `contains_model_call`, and temporal derivative
    detection. Keep this function complete when new trace node types are added.
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
        # StateField wraps the actual symbolic unknown expression.
        yield node.expr
        return


def iter_placeholder_children(node: Any):
    """
    Alias for iterating traced children of a symbolic node.

    Kept for readability in weak-form rewriters that conceptually operate on
    placeholder-like trace nodes.
    """
    yield from iter_children(node)


def contains_node_type(node: Any, cls) -> bool:
    """
    Return True if `node` or any traced child is an instance of `cls`.
    """
    if isinstance(node, cls):
        return True

    return any(contains_node_type(child, cls) for child in iter_children(node) or ())


def contains_testfunction(node: Any) -> bool:
    """Return True if the expression tree contains a `TestFunction`."""
    return contains_node_type(node, TestFunction)


def contains_trialfunction(node: Any) -> bool:
    """Return True if the expression tree contains a `TrialFunction`."""
    return contains_node_type(node, TrialFunction)


def contains_model_call(node: Any) -> bool:
    """Return True if the expression tree contains a neural model call."""
    return contains_node_type(node, ModelCall)


def contains_model_eval(node: Any) -> bool:
    """
    Alias for `contains_model_call`.

    Kept because weak-form code historically used the name
    `contains_model_eval`.
    """
    return contains_model_call(node)


def depends_on_domain_variables(node: Any) -> bool:
    """
    Return True if the expression depends on domain variables such as x, y, or t.
    """
    return contains_node_type(node, Variable)


def contains_subexpr(root: Any, target: Any) -> bool:
    """
    Return True if `target` appears by object identity inside `root`.
    """
    if root is target:
        return True

    return any(contains_subexpr(child, target) for child in iter_children(root) or ())


def unique_by_id(nodes):
    """
    Return nodes with duplicates removed by Python object identity.
    """
    out = []
    seen = set()

    for n in nodes:
        if id(n) not in seen:
            seen.add(id(n))
            out.append(n)

    return out


def sum_terms(domain, terms):
    """
    Sum symbolic terms using jNO expression algebra.

    Returns None for an empty term list.
    """
    if len(terms) == 0:
        return None

    out = terms[0]
    for t in terms[1:]:
        out = out + t

    return out


def apply_sign(domain, sign, term):
    """
    Apply a scalar sign to a symbolic term.
    """
    if sign == 1.0:
        return term

    return Literal(sign) * term


def is_temporal_var(node: Any) -> bool:
    """
    Return True if `node` is a temporal domain variable.
    """
    return isinstance(node, Variable) and getattr(node, "axis", None) == "temporal"


def max_temporal_derivative_order(node: Any) -> int:
    """
    Return the maximum temporal derivative order found in an expression tree.

    Examples:
        grad(u, t)          -> 1
        grad(grad(u,t), t)  -> 2
        Hessian(u, [t])     -> 2
    """
    if isinstance(node, Jacobian):
        target_order = max_temporal_derivative_order(node.target)
        local_order = sum(1 for v in node.variables if is_temporal_var(v))
        return target_order + local_order if local_order > 0 else target_order

    if isinstance(node, Hessian):
        target_order = max_temporal_derivative_order(node.target)
        local_order = 2 * sum(1 for v in node.variables if is_temporal_var(v))
        return target_order + local_order if local_order > 0 else target_order

    return max(
        (max_temporal_derivative_order(child) for child in iter_children(node) or ()),
        default=0,
    )


def collect_temporal_tags(node: Any, out: set[str] | None = None) -> set[str]:
    """
    Collect all temporal variable tags used in an expression tree.
    """
    if out is None:
        out = set()

    if isinstance(node, Variable) and getattr(node, "axis", None) == "temporal":
        out.add(str(node.tag))
        return out

    for child in iter_children(node) or ():
        collect_temporal_tags(child, out)

    return out


def contains_temporal_derivative(node: Any) -> bool:
    """
    Return True if the expression contains any time derivative.
    """
    return max_temporal_derivative_order(node) > 0