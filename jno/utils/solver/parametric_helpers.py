from __future__ import annotations

"""
Shared helpers for scalar affine runtime parameters in external solver blocks.

This module is internal solver plumbing. It recognizes zero-argument
``jno.np.parameter(...)`` calls that were marked with ``_is_parameter=True``
and factors direct multiplicative coefficients from lowered FEM weak-form IR.

Supported first scope
---------------------
    parameter * weak_term

Examples:
    k * grad(u) * grad(phi)
    mu * u**3 * phi

Deliberately unsupported for now
--------------------------------
    exp(raw_k) * weak_term
    (k + c) * weak_term
    k1 * k2 * weak_term
    neural_coefficient(x) * weak_term
"""
from typing import Any

import jax.numpy as jnp

from ...trace import BinaryOp, Literal, ModelCall
from .solver_helper import iter_children as _iter_children


def _is_runtime_scalar_parameter(node) -> bool:
    """Return True for zero-argument trainable jNO physical parameters."""
    return (
        isinstance(node, ModelCall)
        and len(node.args) == 0
        and bool(getattr(node.model, "_is_parameter", False))
    )


def _contains_runtime_parameter(node) -> bool:
    """Recursively detect trainable physical parameters in one trace subtree."""
    if _is_runtime_scalar_parameter(node):
        return True

    return any(
        _contains_runtime_parameter(child)
        for child in (_iter_children(node) or ())
    )


def _flatten_product(node):
    """Flatten a symbolic multiplication tree into factors."""
    if isinstance(node, BinaryOp) and node.op == "*":
        return _flatten_product(node.left) + _flatten_product(node.right)

    return [node]


def _multiply_factors(factors):
    """Rebuild a symbolic multiplication tree."""
    if len(factors) == 0:
        return Literal(1.0)

    out = factors[0]
    for factor in factors[1:]:
        out = BinaryOp("*", out, factor)

    return out


def _parameter_name(param: ModelCall) -> str:
    """Return the stable public name used for one runtime parameter."""
    name = getattr(param.model, "_parameter_name", None)
    if name:
        return str(name)

    if getattr(param.model, "name", None):
        return str(param.model.name)

    return f"parameter_{param.model.layer_id}"


def _factor_runtime_parameter_from_term(coeff):
    """
    Extract one direct multiplicative scalar parameter from a lowered FEM term.

    Supported pattern:
        nu * spatial_term
    """
    factors = _flatten_product(coeff)

    params = [
        factor
        for factor in factors
        if _is_runtime_scalar_parameter(factor)
    ]

    if len(params) == 0:
        if _contains_runtime_parameter(coeff):
            raise NotImplementedError(
                "A trainable FEM parameter was found, but it is not a direct "
                "multiplicative factor. The current affine runtime lowering "
                "supports terms such as `nu * grad(u) * grad(phi)` only."
            )

        return None, coeff

    if len(params) > 1:
        raise NotImplementedError(
            "Affine FEM runtime lowering supports one trainable scalar "
            "coefficient per additive weak-form term."
        )

    param = params[0]

    stripped = _multiply_factors(
        [factor for factor in factors if factor is not param]
    )

    if _contains_runtime_parameter(stripped):
        raise NotImplementedError(
            "Nested runtime physical parameters are not supported yet."
        )

    return param, stripped


def _clone_term_with_coeff(term, new_coeff):
    """Clone one lowered weak-form term while replacing only its coefficient."""
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
    """Build a LoweredWeakForm from an iterable of lowered channel terms."""
    from .weak_form import LoweredWeakForm

    return LoweredWeakForm(domain=domain, terms=list(terms))


def _split_parametric_operator_ir(op_ir):
    """
    Split one lowered IR into a static part and affine runtime basis IRs.

    Returns:
        static_ir:
            Terms independent of runtime physical parameters.
        parameter_irs:
            Mapping ``name -> stripped basis IR``.
        parameter_exprs:
            Mapping ``name -> original jNO parameter expression``.
    """
    static_terms = []
    parameter_terms = {}
    parameter_exprs = {}

    for term in op_ir.terms:
        param, stripped_coeff = _factor_runtime_parameter_from_term(
            term.coeff
        )

        if param is None:
            static_terms.append(term)
            continue

        name = _parameter_name(param)

        parameter_exprs[name] = param
        parameter_terms.setdefault(name, []).append(
            _clone_term_with_coeff(term, stripped_coeff)
        )

    parameter_irs = {
        name: _make_ir(op_ir.domain, terms)
        for name, terms in parameter_terms.items()
    }

    return (
        _make_ir(op_ir.domain, static_terms),
        parameter_irs,
        parameter_exprs,
    )


def _make_zero_ir_like(ir):
    """
    Create a zero-valued IR that preserves trial/test structure.

    This is useful when every physical term is parameter-dependent but static
    Dirichlet enforcement still has to be assembled exactly once.
    """
    if len(ir.terms) == 0:
        raise ValueError("Cannot construct a zero IR from an empty IR.")

    first = ir.terms[0]
    zero_coeff = BinaryOp("*", Literal(0.0), first.coeff)

    return _make_ir(
        ir.domain,
        [_clone_term_with_coeff(first, zero_coeff)],
    )


def _runtime_scalar_arg(args, name: str, *, dtype):
    """Read and scalarize one required runtime coefficient from ``args``."""
    if args is None:
        raise ValueError(
            "This FEM block depends on runtime physical parameters. "
            f"Pass args={{'{name}': value, ...}} to the external solver."
        )

    if name not in args:
        raise KeyError(
            f"Missing runtime physical parameter {name!r}. "
            f"Available args: {sorted(args.keys())}"
        )

    return jnp.asarray(args[name], dtype=dtype).reshape(())

# -----------------------------------------------------------------------------
# Strong-form runtime-parameter helpers
# -----------------------------------------------------------------------------


def _collect_runtime_parameter_exprs(node, out=None):
    """
    Collect zero-argument ``jno.np.parameter(...)`` calls by public name.

    Strong-form Diffrax lowering replaces these symbolic parameter ModelCalls
    with private TensorTags. Concrete values are supplied by the external
    solver through:

        diffrax.diffeqsolve(..., args={"parameter_name": value})
    """
    if out is None:
        out = {}

    if _is_runtime_scalar_parameter(node):
        name = _parameter_name(node)
        previous = out.get(name)

        if (
            previous is not None
            and getattr(previous, "model", None)
            is not getattr(node, "model", None)
        ):
            raise ValueError(
                f"Multiple runtime parameter models use the name {name!r}. "
                "Parameter names must be unique inside one solver block."
            )

        out[name] = node
        return out

    for child in (_iter_children(node) or ()):
        _collect_runtime_parameter_exprs(child, out)

    return out


def _runtime_parameter_tag(name: str) -> str:
    """
    Return the private TensorTag name used by strong-form Diffrax lowering.
    """
    return f"__runtime_parameter_{name}__"

