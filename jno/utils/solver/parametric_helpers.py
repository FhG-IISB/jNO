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

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    model = getattr(node, "model", None)
    if getattr(model, "_derived_expr", None) is not None:
        # `.derives(...)` also marks the parameter untrained, but it is NOT a known coefficient:
        # its value is recomputed every solve, so it must keep the runtime-parameter threading
        # (and with it the per-cell / per-node gather) rather than being baked in at assembly.
        return False
    return _is_runtime_scalar_parameter(node) and bool(getattr(model, "_frozen", False))


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


def _fem_field_kind(node):
    """``"node"``, ``"cell"`` or ``None`` — which FE space a field parameter's values live on.

    A ``"cell"`` field is P0: one value per element, broadcast over that element's quadrature
    points. A ``"node"`` field is P1: the cell's vertex values, interpolated with the shape
    functions. They gather differently, so the assembler has to tell them apart.
    """
    if not _is_fem_field_parameter(node):
        return None
    return getattr(node.model, "_fem_field", None)


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


# -----------------------------------------------------------------------------
# Neural-coefficient mechanism (single source of truth for both assemblers)
#
# A neural coefficient reuses the runtime-parameter *infrastructure* (the ``args`` dict,
# ``FemLinearSystem.operator_fn(args)`` / ``FemResidualOperator`` / ``SemidiscreteTimeBlock``,
# ``custom_root`` implicit diff). Only three things are specific to a network, and they are the
# axes along which the mechanism could later change -- so they live here, called identically by the
# native and non-nodal assemblers rather than copied into each:
#
#   1. COLLECT      -- which networks are coefficients, their names, which are trainable
#                      (``collect_neural_slots``);
#   2. CRUX DELIVERY -- how a trainable net's weights reach the solve node: a ``ModelWeights`` slot in
#                      ``runtime_parameter_exprs`` that the trace evaluator resolves to the live module
#                      (``neural_operator_exprs``);
#   3. KERNEL TABLE -- how the module reaches the per-cell integrand: a ``{name: module}`` map placed at
#                      ``local["neural_coefficients"]`` (``neural_local_table``).
#
# Swapping the mechanism (e.g. partition-based weights instead of whole-module-by-closure) touches
# only (2)+(3) here plus the ``ModelWeights`` handler; a network stays user-visible as just
# ``jno.nn.wrap(net)(...)`` inside a weak form, so no user code or behaviour test moves.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NeuralSlots:
    """The neural coefficients of one solver block. ``all_names`` covers every network (frozen +
    trainable) whose module the kernel must evaluate; ``param_names`` is the trainable subset that
    becomes ``ModelWeights`` runtime slots; ``models`` maps each name to its :class:`Model`."""

    all_names: Tuple[str, ...]
    models: Dict[str, Any]
    param_names: Tuple[str, ...]

    @property
    def any(self) -> bool:
        return bool(self.all_names)

    @property
    def any_trainable(self) -> bool:
        return bool(self.param_names)


def collect_neural_slots(
    volume_terms,
    boundary_terms=None,
    *,
    runtime_parameter_tags: Tuple[str, ...] = (),
    reject_trainable_boundary: bool = False,
) -> NeuralSlots:
    """Collect a block's neural coefficients (touch-point 1).

    Volume-term networks (frozen + trainable) are always threaded. ``boundary_terms`` handling is
    the one place the two assemblers legitimately differ, kept explicit here rather than hidden:
    ``reject_trainable_boundary=False`` (native) folds boundary networks into the kernel table too;
    ``True`` (non-nodal, whose natural-BC load is assembled non-differentiably) raises on a trainable
    boundary net. A network sharing a name with a runtime parameter is rejected (names key ``args``).
    """
    exprs: Dict[str, ModelCall] = {}
    for bare in volume_terms:
        _collect_neural_coefficient_exprs(bare, exprs, include_frozen=True)

    if boundary_terms:
        if reject_trainable_boundary:
            bexprs: Dict[str, ModelCall] = {}
            for terms in boundary_terms.values():
                for bare in terms:
                    _collect_neural_coefficient_exprs(bare, bexprs)  # trainable only
            if bexprs:
                raise NotImplementedError(
                    "jno.fem (non-nodal): a trainable neural coefficient in a boundary (Neumann/Robin) term "
                    "is not supported -- the natural-BC load is assembled non-differentiably. Put it in a "
                    "volume term."
                )
        else:
            for terms in boundary_terms.values():
                for bare in terms:
                    _collect_neural_coefficient_exprs(bare, exprs, include_frozen=True)

    all_names = tuple(sorted(exprs))
    if set(all_names) & set(runtime_parameter_tags):
        raise ValueError(
            f"jno.fem: a neural coefficient and a runtime parameter share the name(s) "
            f"{sorted(set(all_names) & set(runtime_parameter_tags))}; names key the runtime args and must be "
            "unique inside one solver block (rename via .name())."
        )
    models = {n: e.model for n, e in exprs.items()}
    param_names = tuple(sorted(n for n, e in exprs.items() if not bool(getattr(e.model, "_frozen", False))))
    return NeuralSlots(all_names=all_names, models=models, param_names=param_names)


def neural_operator_exprs(rt_param_exprs: Dict[str, Any], slots: NeuralSlots) -> Dict[str, Any]:
    """Merge trainable networks into ``runtime_parameter_exprs`` as ``ModelWeights`` slots (touch-point 2).

    The result is what an ``FemLinearSystem`` / ``FemResidualOperator`` / ``SemidiscreteTimeBlock`` carries;
    at solve time the trace evaluator resolves each ``ModelWeights`` to the live (crux-recombined) module,
    which returns here through ``args`` -- so ``∂solve/∂weights`` flows through the implicit-diff solvers."""
    if not slots.param_names:
        return dict(rt_param_exprs)
    from ...trace import ModelWeights

    return {**rt_param_exprs, **{n: ModelWeights(slots.models[n]) for n in slots.param_names}}


def neural_local_table(slots: NeuralSlots, args) -> Optional[Dict[str, Any]]:
    """The per-cell ``{name: module}`` table for ``local['neural_coefficients']`` (touch-point 3), or
    ``None`` when the block has no networks. Trainable modules arrive through ``args``; a frozen (or
    static-placeholder) net falls back to its stored module."""
    if not slots.all_names:
        return None
    a = args or {}
    return {n: a.get(n, slots.models[n].module) for n in slots.all_names}


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
