from __future__ import annotations

from typing import Optional

from ...jnp_ops import stack
from ...trace import (
    Placeholder,
    Literal,
    BinaryOp,
    FunctionCall,
    Variable,
    Jacobian,
    TestFunction,
)


def split_weak_additive_terms(domain, node, sign=1.0, infer_term_bucket=None):
    if isinstance(node, BinaryOp) and node.op in {"+", "-"}:
        bucket = None
        if infer_term_bucket is not None:
            try:
                bucket = infer_term_bucket(domain, node)
            except Exception:
                bucket = None

        if bucket is not None:
            support, _region_id = bucket
            if support == "boundary":
                return [(sign, node)]

        if node.op == "+":
            return (
                split_weak_additive_terms(domain, node.left, sign, infer_term_bucket)
                + split_weak_additive_terms(domain, node.right, sign, infer_term_bucket)
            )

        if node.op == "-":
            return (
                split_weak_additive_terms(domain, node.left, sign, infer_term_bucket)
                + split_weak_additive_terms(domain, node.right, -sign, infer_term_bucket)
            )

    return [(sign, node)]


def function_name(node) -> Optional[str]:
    if isinstance(node, FunctionCall):
        if getattr(node, "_name", None) is not None:
            return str(node._name)
        if hasattr(node.fn, "__name__"):
            return str(node.fn.__name__)
    return None


def get_grad_axis_from_test_grad(node) -> int:
    if not (isinstance(node, Jacobian) and isinstance(node.target, TestFunction)):
        raise TypeError(f"Expected Jacobian(TestFunction), got {type(node).__name__}")

    if len(node.variables) != 1:
        raise ValueError(
            "Canonical FEAX-style test_grad lowering currently expects exactly one "
            f"spatial variable in Jacobian(TestFunction,...), got {len(node.variables)}"
        )

    var = node.variables[0]
    if not isinstance(var, Variable):
        raise TypeError(f"Expected Variable inside Jacobian(TestFunction), got {type(var).__name__}")

    if not hasattr(var, "dim") or len(var.dim) < 1:
        raise ValueError(f"Cannot infer gradient axis from variable {var}")

    return int(var.dim[0])


def canonicalize_grad_coeff(domain, coeff_expr, axis: int, value_shape: tuple):
    dim = int(domain.dimension)

    if value_shape is None or len(value_shape) == 0:
        return stack(
            [coeff_expr * Literal(1.0 if j == axis else 0.0) for j in range(dim)],
            axis=-1,
        )

    if len(value_shape) == 1:
        return stack(
            [coeff_expr * Literal(1.0 if j == axis else 0.0) for j in range(dim)],
            axis=-1,
        )

    raise NotImplementedError(
        f"Canonical grad coeff inflation not implemented yet for value_shape={value_shape}"
    )


def value_shape_num_components(value_shape) -> int:
    if value_shape is None or len(value_shape) == 0:
        return 1
    n = 1
    for s in value_shape:
        n *= int(s)
    return n


def is_test_value(node):
    return isinstance(node, TestFunction)


def is_test_grad(node):
    return isinstance(node, Jacobian) and isinstance(node.target, TestFunction)


def is_symgrad_test(node) -> bool:
    if not isinstance(node, FunctionCall):
        return False

    name = function_name(node)
    if name != "symgrad":
        return False

    if len(node.args) < 1:
        return False

    arg0 = node.args[0]
    return isinstance(arg0, TestFunction) or (
        isinstance(arg0, Jacobian) and isinstance(arg0.target, TestFunction)
    )


def get_test_value_shape(node) -> tuple:
    if isinstance(node, TestFunction):
        return getattr(node, "value_shape", ())

    if isinstance(node, Jacobian) and isinstance(node.target, TestFunction):
        return getattr(node.target, "value_shape", ())

    if is_symgrad_test(node):
        arg0 = node.args[0]
        if isinstance(arg0, TestFunction):
            return getattr(arg0, "value_shape", ())
        if isinstance(arg0, Jacobian) and isinstance(arg0.target, TestFunction):
            return getattr(arg0.target, "value_shape", ())

    return ()