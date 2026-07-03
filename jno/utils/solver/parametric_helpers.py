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

Deliberately unsupported for now (affine factoring)
---------------------------------------------------
    exp(raw_k) * weak_term
    (k + c) * weak_term
    k1 * k2 * weak_term

Neural coefficients (``jno.nn.wrap(net)`` called inside a weak form, e.g.
``net(x, y) * weak_term``) are NOT affine-factored: like nodal FEM field
parameters they stay inside the integrand and take the re-assembly route.
This module supplies their detection/collection predicates
(``_is_neural_coefficient`` / ``_collect_neural_coefficient_exprs``); the
native assembler threads the weight pytree through the runtime ``args`` and
the kernel evaluates the network at the quadrature points
(``fem_utils._eval_integrand``).
"""

import jax.numpy as jnp

from ...trace import BinaryOp, Literal, ModelCall
from .solver_helper import iter_children as _iter_children


def _is_runtime_scalar_parameter(node) -> bool:
    """Return True for zero-argument jNO physical parameters (``jno.np.parameter(...)``)."""
    return isinstance(node, ModelCall) and len(node.args) == 0 and bool(getattr(node.model, "_is_parameter", False))


def _is_frozen_parameter(node) -> bool:
    """A parameter marked ``.freeze()`` — a **known**, non-trainable coefficient. It is *not* a runtime
    (trainable) unknown: the integrand evaluator resolves it as a coordinate function / constant at
    the quadrature points (see ``fem_utils._eval_frozen_coefficient``), so the system assembles
    non-parametrically. Excluded from runtime-parameter detection/collection below."""
    return _is_runtime_scalar_parameter(node) and bool(getattr(getattr(node, "model", None), "_frozen", False))


def _contains_runtime_parameter(node) -> bool:
    """Recursively detect **trainable** physical parameters in one trace subtree (frozen ones don't
    count — they are baked in as known coefficients, keeping the system non-parametric)."""
    if _is_runtime_scalar_parameter(node) and not _is_frozen_parameter(node):
        return True

    return any(_contains_runtime_parameter(child) for child in (_iter_children(node) or ()))


def _is_fem_field_parameter(node) -> bool:
    """True for a trainable parameter that is a FEM coefficient *field*
    (``jno.np.parameter(phi)``) rather than a scalar."""
    return _is_runtime_scalar_parameter(node) and getattr(node.model, "_fem_field", None) is not None


def _contains_fem_field_parameter(node) -> bool:
    """Recursively detect a FEM field parameter in one trace subtree."""
    if _is_fem_field_parameter(node):
        return True
    return any(_contains_fem_field_parameter(child) for child in (_iter_children(node) or ()))


def _is_neural_coefficient(node) -> bool:
    """True for a *network* ModelCall used as a coefficient (``jno.nn.wrap(net)(x, ...)``).

    Everything that is a ModelCall but not a zero-arg ``jno.np.parameter(...)`` counts: the model
    is an arbitrary equinox module evaluated at the quadrature points on its (symbolic) arguments.
    A ``.freeze()``d network is still a neural coefficient — it evaluates through the same kernel
    branch — but it is excluded from *trainable* collection (see
    ``_collect_neural_coefficient_exprs``), so an all-frozen system assembles non-parametrically.
    """
    return isinstance(node, ModelCall) and not _is_runtime_scalar_parameter(node)


def _is_frozen_neural_coefficient(node) -> bool:
    """A neural coefficient whose model is ``.freeze()``d — a known, non-trainable network."""
    return _is_neural_coefficient(node) and bool(getattr(node.model, "_frozen", False))


def _contains_neural_coefficient(node) -> bool:
    """Recursively detect a neural coefficient in one trace subtree."""
    if _is_neural_coefficient(node):
        return True
    return any(_contains_neural_coefficient(child) for child in (_iter_children(node) or ()))


def _neural_coefficient_name(node: ModelCall) -> str:
    """Stable public name for one neural coefficient (mirrors ``_parameter_name``)."""
    name = getattr(node.model, "name", None)
    if name:
        return str(name)
    return f"neural_{node.model.layer_id}"


def _collect_neural_coefficient_exprs(node, out=None, *, include_frozen=False):
    """Collect network ModelCalls (neural coefficients) by public name.

    With the default ``include_frozen=False``, ``.freeze()``d networks are skipped — they are
    known coefficients, evaluated from their stored weights, and must not make the system
    parametric. Pass ``include_frozen=True`` to build the kernel's evaluation table, which needs
    every network (frozen ones evaluate from their stored module). Two *different* models sharing
    one public name are rejected (the name keys the runtime ``args`` slot).
    """
    if out is None:
        out = {}

    if _is_neural_coefficient(node) and (include_frozen or not _is_frozen_neural_coefficient(node)):
        name = _neural_coefficient_name(node)
        previous = out.get(name)

        if previous is not None and getattr(previous, "model", None) is not getattr(node, "model", None):
            raise ValueError(
                f"Multiple neural-coefficient models use the name {name!r}. "
                "Model names must be unique inside one solver block."
            )

        out[name] = node
        # Do NOT return: the network's arguments may nest further coefficients (e.g. net2 inside
        # net1's argument); keep descending.

    for child in _iter_children(node) or ():
        _collect_neural_coefficient_exprs(child, out, include_frozen=include_frozen)

    return out


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


_NONAFFINE = object()  # module-level sentinel; identity-compared, never ==


def _factor_runtime_parameter_from_term(coeff, *, allow_nonaffine=False):
    # A FEM field parameter (nodal coefficient) can never be an affine scalar factor
    # -- it stays inside the integrand and is re-assembled (interpolated to quad).
    if _contains_fem_field_parameter(coeff):
        return _NONAFFINE, coeff

    factors = _flatten_product(coeff)
    params = [f for f in factors if _is_runtime_scalar_parameter(f)]

    if len(params) == 0:
        if _contains_runtime_parameter(coeff):
            if allow_nonaffine:
                return _NONAFFINE, coeff
            raise NotImplementedError(
                "A trainable FEM parameter was found, but it is not a direct "
                "multiplicative factor. The current affine runtime lowering "
                "supports terms such as `nu * grad(u) * grad(phi)` only."
            )
        return None, coeff

    if len(params) > 1:
        if allow_nonaffine:
            return _NONAFFINE, coeff
        raise NotImplementedError(
            "Affine FEM runtime lowering supports one trainable scalar coefficient per additive weak-form term."
        )

    param = params[0]
    stripped = _multiply_factors([f for f in factors if f is not param])
    if _contains_runtime_parameter(stripped):
        if allow_nonaffine:
            return _NONAFFINE, coeff
        raise NotImplementedError("Nested runtime physical parameters are not supported yet.")
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


def _split_parametric_operator_ir(op_ir, *, allow_nonaffine=False):
    static_terms = []
    parameter_terms = {}
    parameter_exprs = {}
    nonaffine_terms = []

    for term in op_ir.terms:
        param, stripped_coeff = _factor_runtime_parameter_from_term(term.coeff, allow_nonaffine=allow_nonaffine)

        if param is _NONAFFINE:
            nonaffine_terms.append(term)  # keep full coeff; parameter stays inside
            continue
        if param is None:
            static_terms.append(term)
            continue

        name = _parameter_name(param)
        previous = parameter_exprs.get(name)
        if previous is not None and getattr(previous, "model", None) is not getattr(param, "model", None):
            raise ValueError(
                f"Multiple runtime parameter models use the name {name!r}. "
                "Parameter names must be unique inside one solver block."
            )
        parameter_exprs[name] = param
        parameter_terms.setdefault(name, []).append(_clone_term_with_coeff(term, stripped_coeff))

    parameter_irs = {name: _make_ir(op_ir.domain, terms) for name, terms in parameter_terms.items()}
    nonaffine_ir = _make_ir(op_ir.domain, nonaffine_terms)

    return (
        _make_ir(op_ir.domain, static_terms),
        parameter_irs,
        parameter_exprs,
        nonaffine_ir,
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
        raise KeyError(f"Missing runtime physical parameter {name!r}. Available args: {sorted(args.keys())}")

    return jnp.asarray(args[name], dtype=dtype).reshape(())


# -----------------------------------------------------------------------------
# Strong-form runtime-parameter helpers
# -----------------------------------------------------------------------------


def _collect_runtime_parameter_exprs(node, out=None):
    """Collect zero-argument ``jno.np.parameter(...)`` calls by public name."""
    if out is None:
        out = {}

    if _is_runtime_scalar_parameter(node) and not _is_frozen_parameter(node):
        name = _parameter_name(node)
        previous = out.get(name)

        if previous is not None and getattr(previous, "model", None) is not getattr(node, "model", None):
            raise ValueError(
                f"Multiple runtime parameter models use the name {name!r}. "
                "Parameter names must be unique inside one solver block."
            )

        out[name] = node
        return out

    for child in _iter_children(node) or ():
        _collect_runtime_parameter_exprs(child, out)

    return out


def _runtime_parameter_tag(name: str) -> str:
    """
    Return the private TensorTag name used by strong-form Diffrax lowering.
    """
    return f"__runtime_parameter_{name}__"


def _merge_runtime_parameter_exprs(*mappings):
    """
    Merge runtime-parameter expression maps while rejecting name collisions.

    The same physical parameter may appear in multiple affine channels, for
    example both in an operator basis and in a forcing basis. Reusing the same
    parameter model is valid; using two different parameter models with the
    same public name is rejected.
    """
    merged = {}

    for mapping in mappings:
        for name, expr in (mapping or {}).items():
            previous = merged.get(name)

            if previous is not None and getattr(previous, "model", None) is not getattr(expr, "model", None):
                raise ValueError(
                    f"Multiple runtime parameter models use the name {name!r}. "
                    "Parameter names must be unique inside one solver block."
                )

            merged[name] = expr

    return merged
