from __future__ import annotations

from typing import Any

from ...trace import (
    Placeholder,
    Literal,
    BinaryOp,
    FunctionCall,
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
    if isinstance(node, BinaryOp):
        yield node.left
        yield node.right
    elif isinstance(node, FunctionCall):
        for a in node.args:
            if isinstance(a, Placeholder):
                yield a
    elif isinstance(node, ModelCall):
        for a in node.args:
            if isinstance(a, Placeholder):
                yield a
    elif isinstance(node, OperationDef):
        yield node.expr
    elif isinstance(node, OperationCall):
        yield node.operation
        for a in node.args:
            if isinstance(a, Placeholder):
                yield a
    elif isinstance(node, (Jacobian, Hessian)):
        yield node.target
        for v in node.variables:
            if isinstance(v, Placeholder):
                yield v
    elif isinstance(node, Tracker):
        yield node.expr
    elif isinstance(node, Assembly):
        yield node.expr
    elif isinstance(node, GroupedAssembly):
        if node.volume_value_expr is not None:
            yield node.volume_value_expr
        if node.volume_grad_expr is not None:
            yield node.volume_grad_expr
        for e in node.boundary_value_exprs.values():
            yield e


def contains_node_type(node: Any, cls) -> bool:
    if isinstance(node, cls):
        return True
    return any(contains_node_type(child, cls) for child in iter_children(node))


def contains_testfunction(node: Any) -> bool:
    return contains_node_type(node, TestFunction)


def contains_trialfunction(node: Any) -> bool:
    return contains_node_type(node, TrialFunction)


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