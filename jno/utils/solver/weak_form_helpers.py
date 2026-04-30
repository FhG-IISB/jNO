from __future__ import annotations

"""
Internal weak-form helper utilities.

This module contains symbolic transformation helpers used by
`solver.weak_form`. It is not a public API.

Responsibilities:
- split weak-form expressions into additive signed terms,
- detect TestFunction / TrialFunction / test-gradient channels,
- infer volume/boundary variational regions,
- detect and wrap neural unknowns as StateField nodes,
- rebind FEM/VPINN variables between volume and boundary quadrature regions,
- substitute TrialFunction symbols with neural trial expressions for VPINN.
"""
from typing import Optional

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
    contains_node_type,
    iter_placeholder_children,
    contains_model_eval,
    depends_on_domain_variables,
    unique_by_id,
)


# ---------------------------------------------------------------------------
# Additive splitting
# ---------------------------------------------------------------------------


def split_weak_additive_terms(domain, node, sign=1.0, infer_term_bucket=None):
    """
    Split weak-form additive expressions into signed terms.

    Boundary terms are kept together if bucket inference identifies the whole
    additive expression as belonging to one boundary region.
    """
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
            return split_weak_additive_terms(domain, node.left, sign, infer_term_bucket) + split_weak_additive_terms(domain, node.right, sign, infer_term_bucket)

        if node.op == "-":
            return split_weak_additive_terms(domain, node.left, sign, infer_term_bucket) + split_weak_additive_terms(domain, node.right, -sign, infer_term_bucket)

    return [(sign, node)]


# ---------------------------------------------------------------------------
# Function/test-symbol helpers
# ---------------------------------------------------------------------------


def function_name(node) -> Optional[str]:
    if isinstance(node, FunctionCall):
        if getattr(node, "_name", None) is not None:
            return str(node._name)
        if hasattr(node.fn, "__name__"):
            return str(node.fn.__name__)
    return None


def get_grad_axis_from_test_grad(node) -> int:
    """
    Infer the spatial derivative axis from Jacobian(TestFunction, variable).

    Used when lowering test-gradient terms into canonical VPINN/FEAX channels.
    """
    if not (isinstance(node, Jacobian) and isinstance(node.target, TestFunction)):
        raise TypeError(f"Expected Jacobian(TestFunction), got {type(node).__name__}")

    if len(node.variables) != 1:
        raise ValueError("Canonical FEAX-style test_grad lowering currently expects exactly one " f"spatial variable in Jacobian(TestFunction,...), got {len(node.variables)}")

    var = node.variables[0]
    if not isinstance(var, Variable):
        raise TypeError(f"Expected Variable inside Jacobian(TestFunction), got {type(var).__name__}")

    if not hasattr(var, "dim") or len(var.dim) < 1:
        raise ValueError(f"Cannot infer gradient axis from variable {var}")

    return int(var.dim[0])


def canonicalize_grad_coeff(domain, coeff_expr, axis: int, value_shape: tuple):
    """
    Convert scalar coefficient multiplying d(phi)/dx_axis into a vector-valued
    coefficient for the canonical test_grad channel.
    """
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

    raise NotImplementedError(f"Canonical grad coeff inflation not implemented yet for value_shape={value_shape}")


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
    return isinstance(arg0, TestFunction) or (isinstance(arg0, Jacobian) and isinstance(arg0.target, TestFunction))


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


def contains_testfunction_gradient(domain, expr):
    if expr is None:
        return False

    if isinstance(expr, Jacobian) and isinstance(expr.target, TestFunction):
        return True

    for child in iter_placeholder_children(expr):
        if contains_testfunction_gradient(domain, child):
            return True

    return False


def has_weak_basis_symbols(domain, node):
    return contains_node_type(node, TestFunction) or contains_node_type(node, TrialFunction) or contains_testfunction_gradient(domain, node)


# ---------------------------------------------------------------------------
# Variational region helpers
# ---------------------------------------------------------------------------


def collect_region_keys(domain, node):
    metas = []
    collect_variational_metas(domain, node, metas)
    return {(m["support"], m["region_id"]) for m in metas}


def collect_variational_metas(domain, node, out):
    """
    Collect FEM/variational metadata from Variable nodes inside an expression.

    StateField nodes are intentionally not expanded, because their wrapped
    expression may have been created on a different quadrature region.
    """
    if node is None:
        return

    # IMPORTANT:
    # StateField is a symbolic unknown placeholder.
    # Do not inspect node.expr here, otherwise Robin terms that contain
    # a volume-built state leak volume tags into boundary bucket inference.
    if isinstance(node, StateField):
        return

    if isinstance(node, Variable) and getattr(node, "fem_meta", None) is not None:
        out.append(node.fem_meta)
        return

    for child in iter_placeholder_children(node):
        collect_variational_metas(domain, child, out)


def infer_term_bucket(domain, term):
    """
    Infer whether a weak-form term belongs to the volume or a boundary region.

    Returns:
        `(support, region_id)`

    Raises:
        ValueError if one term mixes incompatible variational regions.
    """
    metas = []
    collect_variational_metas(domain, term, metas)

    if len(metas) > 0:
        support = metas[0]["support"]
        region_id = metas[0]["region_id"]

        for m in metas[1:]:
            if m["support"] != support or m["region_id"] != region_id:
                raise ValueError("Weak-form term mixes variational regions. " f"Found both ({support}, {region_id}) and " f"({m['support']}, {m['region_id']}).")

        return support, region_id

    if contains_node_type(term, TrialFunction) or contains_node_type(term, TestFunction):
        return "volume", "volume"

    raise ValueError("Could not infer weak-form support for term. " "Use variables sampled on fem_gauss / gauss_<tag> inside the term " "or include TrialFunction/TestFunction.")


def get_variational_region_meta(domain, support: str, region_id: str):
    """
    Return stored variational sampling metadata for a support/region pair.

    Used when rebinding expressions between volume and boundary quadrature
    contexts.
    """
    registry = getattr(domain, "_variational_sampling_registry", {})

    for _sample_tag, meta in registry.items():
        if meta.get("support") == support and meta.get("region_id") == region_id:
            return meta

    raise KeyError(f"No variational sampling meta found for support={support!r}, " f"region_id={region_id!r}. Available: {registry}")


# ---------------------------------------------------------------------------
# State-field detection
# ---------------------------------------------------------------------------


def find_first_statefield(node):
    if node is None:
        return None

    if isinstance(node, StateField):
        return node

    for child in iter_placeholder_children(node):
        found = find_first_statefield(child)
        if found is not None:
            return found

    return None


def infer_state_value_shape(domain, expr) -> tuple:
    shapes = set()

    def walk(node):
        if node is None:
            return

        if isinstance(node, TestFunction):
            shapes.add(tuple(getattr(node, "value_shape", ())))
            return

        if isinstance(node, Jacobian) and isinstance(node.target, TestFunction):
            shapes.add(tuple(getattr(node.target, "value_shape", ())))
            return

        if is_symgrad_test(node):
            shapes.add(tuple(get_test_value_shape(node)))
            return

        for child in iter_placeholder_children(node):
            walk(child)

    walk(expr)

    if len(shapes) == 0:
        return ()

    if len(shapes) == 1:
        return next(iter(shapes))

    raise NotImplementedError("Could not infer a unique state value_shape from the weak form. " f"Found multiple test value shapes: {sorted(shapes)}")


def wrap_primary_state(node, target, *, state_name="u", value_shape=()):
    """
    Replace the selected symbolic unknown subtree by a StateField wrapper.

    The wrapper marks the primary unknown for FEM/FEAX lowering while preserving
    the original neural expression inside the StateField.
    """
    if node is target:
        return StateField(node, state_id=0, name=state_name, value_shape=value_shape)

    if node is None:
        return None

    if isinstance(node, BinaryOp):
        left = wrap_primary_state(
            node.left,
            target,
            state_name=state_name,
            value_shape=value_shape,
        )
        right = wrap_primary_state(
            node.right,
            target,
            state_name=state_name,
            value_shape=value_shape,
        )
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = []
        changed = False

        for a in node.args:
            if isinstance(a, Placeholder):
                na = wrap_primary_state(
                    a,
                    target,
                    state_name=state_name,
                    value_shape=value_shape,
                )
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)

        if changed:
            return node.copy_with_args(new_args)

        return node

    if isinstance(node, ModelCall):
        new_args = []
        changed = False

        for a in node.args:
            if isinstance(a, Placeholder):
                na = wrap_primary_state(
                    a,
                    target,
                    state_name=state_name,
                    value_shape=value_shape,
                )
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)

        if changed:
            rebuilt_model = ModelCall(node.model, new_args)
            rebuilt_model.op_id = node.op_id
            return rebuilt_model

        return node

    if isinstance(node, Jacobian):
        new_target = wrap_primary_state(
            node.target,
            target,
            state_name=state_name,
            value_shape=value_shape,
        )
        new_vars = []
        changed = new_target is not node.target

        for v in node.variables:
            if isinstance(v, Placeholder):
                nv = wrap_primary_state(
                    v,
                    target,
                    state_name=state_name,
                    value_shape=value_shape,
                )
            else:
                nv = v

            changed = changed or (nv is not v)
            new_vars.append(nv)

        if changed:
            return Jacobian(new_target, new_vars, node.scheme)

        return node

    if isinstance(node, Hessian):
        new_target = wrap_primary_state(
            node.target,
            target,
            state_name=state_name,
            value_shape=value_shape,
        )
        new_vars = []
        changed = new_target is not node.target

        for v in node.variables:
            if isinstance(v, Placeholder):
                nv = wrap_primary_state(
                    v,
                    target,
                    state_name=state_name,
                    value_shape=value_shape,
                )
            else:
                nv = v

            changed = changed or (nv is not v)
            new_vars.append(nv)

        if changed:
            return Hessian(new_target, new_vars, node.scheme, trace=node.trace)

        return node

    if isinstance(node, OperationDef):
        new_expr = wrap_primary_state(
            node.expr,
            target,
            state_name=state_name,
            value_shape=value_shape,
        )

        if new_expr is not node.expr:
            rebuilt_opdef = OperationDef.__new__(OperationDef)
            rebuilt_opdef.expr = new_expr
            rebuilt_opdef.input_vars = node.input_vars
            rebuilt_opdef.name = getattr(node, "name", None)
            rebuilt_opdef.op_id = node.op_id
            return rebuilt_opdef

        return node

    if isinstance(node, OperationCall):
        new_args = []
        changed = False

        for a in node.args:
            if isinstance(a, Placeholder):
                na = wrap_primary_state(
                    a,
                    target,
                    state_name=state_name,
                    value_shape=value_shape,
                )
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)

        if changed:
            rebuilt_opcall = OperationCall(node.operation, tuple(new_args))
            rebuilt_opcall.op_id = node.op_id
            return rebuilt_opcall

        return node

    if isinstance(node, Tracker):
        new_expr = wrap_primary_state(
            node.expr,
            target,
            state_name=state_name,
            value_shape=value_shape,
        )

        if new_expr is not node.expr:
            rebuilt_tracker = Tracker(new_expr, interval=node.interval)
            rebuilt_tracker.op_id = node.op_id
            return rebuilt_tracker

        return node

    return node


def ensure_statefield_wrapped(domain, expr):
    """
    Ensure that a weak form has an explicit StateField or TrialFunction.

    If the expression contains a neural unknown but no StateField/TrialFunction,
    this function detects the primary unknown and wraps it as StateField.
    """
    if contains_node_type(expr, StateField) or contains_node_type(expr, TrialFunction):
        return expr

    candidate = detect_primary_state_field(domain, expr)
    if candidate is None:
        return expr

    value_shape = infer_state_value_shape(domain, expr)
    return wrap_primary_state(expr, candidate, state_name="u", value_shape=value_shape)


def is_statefield_candidate(domain, node):
    """
    Return True if a node is a valid candidate for automatic StateField wrapping.

    Candidates must be neural/model-based expressions that depend on domain
    variables and must not already contain weak basis symbols.
    """
    if node is None:
        return False

    # Never wrap existing weak symbols or already-wrapped state fields.
    if isinstance(node, (StateField, TestFunction, TrialFunction)):
        return False

    # Derivatives are not the state field itself.
    if isinstance(node, (Jacobian, Hessian)):
        return False

    # The unknown field itself must not already include weak basis functions.
    if has_weak_basis_symbols(domain, node):
        return False

    # Reject additive composites like (u**3 - u) or whole weak expressions.
    if isinstance(node, BinaryOp) and node.op in {"+", "-"}:
        return False

    # Must actually come from a model / NN evaluation somewhere.
    if not contains_model_eval(node):
        return False

    # Must depend on domain variables.
    if not depends_on_domain_variables(node):
        return False

    # If sampled FEM regions appear, they must all belong to one bucket only.
    keys = collect_region_keys(domain, node)
    if len(keys) > 1:
        return False

    return True


def collect_state_field_candidates(domain, node, out):
    if node is None:
        return

    if is_statefield_candidate(domain, node):
        out.append(node)
        return

    for child in iter_placeholder_children(node):
        collect_state_field_candidates(domain, child, out)


def collect_derivative_based_state_targets(domain, node, out):
    """
    Collect unknown candidates that appear as targets of derivatives.

    This path is preferred because terms like grad(u) identify the unknown more
    reliably than searching arbitrary model-call expressions.
    """
    if node is None:
        return

    if isinstance(node, (Jacobian, Hessian)):
        tgt = node.target

        if not isinstance(tgt, (TrialFunction, TestFunction, StateField)):
            if contains_model_eval(tgt) and depends_on_domain_variables(tgt) and not has_weak_basis_symbols(domain, tgt):
                keys = collect_region_keys(domain, tgt)
                if len(keys) <= 1:
                    out.append(tgt)

    for child in iter_placeholder_children(node):
        collect_derivative_based_state_targets(domain, child, out)


def detect_primary_state_field(domain, expr):
    """
    Detect the unique primary unknown expression in a weak form.

    First searches derivative targets such as grad(u), then falls back to
    model-call candidates. Raises if multiple unknowns are detected.
    """
    # Robust path:
    # If the unknown already appears as target of grad/hessian, use that first.
    deriv_targets = []
    collect_derivative_based_state_targets(domain, expr, deriv_targets)
    deriv_targets = unique_by_id(deriv_targets)

    if len(deriv_targets) == 1:
        return deriv_targets[0]

    if len(deriv_targets) > 1:
        raise NotImplementedError("Multiple derivative-based state targets detected in weak form. " "Phase 1 supports exactly one unknown. " "Phase 3 will support multi-unknown systems.")

    # Fallback for weak forms without grad(u), e.g. pure reaction-like forms.
    candidates = []
    collect_state_field_candidates(domain, expr, candidates)
    candidates = unique_by_id(candidates)

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        raise NotImplementedError("Multiple state-field candidates detected in weak form. " "Phase 1 supports exactly one unknown. " "Phase 3 will support multi-unknown systems.")

    return candidates[0]


# ---------------------------------------------------------------------------
# Trial substitution / variational rebinding
# ---------------------------------------------------------------------------


def rebind_variational_variables(domain, node, target_support: str, target_region_id: str):
    """
    Rebind FEM quadrature Variables to a target variational region.

    Used when an expression created on one quadrature region must be evaluated
    on another region, for example rebinding a neural trial expression from
    volume quadrature to boundary quadrature.
    """
    if node is None:
        return None

    target_meta = get_variational_region_meta(domain, target_support, target_region_id)
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
        left = rebind_variational_variables(
            domain,
            node.left,
            target_support,
            target_region_id,
        )
        right = rebind_variational_variables(
            domain,
            node.right,
            target_support,
            target_region_id,
        )

        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)

        return node

    if isinstance(node, FunctionCall):
        new_args = [rebind_variational_variables(domain, a, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]

        if any(n is not o for n, o in zip(new_args, node.args)):
            return FunctionCall(node.fn, new_args, node._name, node.reduces_axis, node.kwargs)

        return node

    if isinstance(node, ModelCall):
        new_args = [rebind_variational_variables(domain, a, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]

        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt_model = ModelCall(node.model, new_args)
            rebuilt_model.op_id = node.op_id
            return rebuilt_model

        return node

    if isinstance(node, OperationDef):
        new_expr = rebind_variational_variables(
            domain,
            node.expr,
            target_support,
            target_region_id,
        )

        if new_expr is not node.expr:
            rebuilt_opdef = OperationDef.__new__(OperationDef)
            rebuilt_opdef.expr = new_expr
            rebuilt_opdef.input_vars = node.input_vars
            rebuilt_opdef.name = getattr(node, "name", None)
            rebuilt_opdef.op_id = node.op_id
            return rebuilt_opdef

        return node

    if isinstance(node, OperationCall):
        new_args = [rebind_variational_variables(domain, a, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]

        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt_opcall = OperationCall(node.operation, tuple(new_args))
            rebuilt_opcall.op_id = node.op_id
            return rebuilt_opcall

        return node

    if isinstance(node, Jacobian):
        new_target = rebind_variational_variables(
            domain,
            node.target,
            target_support,
            target_region_id,
        )
        new_vars = [rebind_variational_variables(domain, v, target_support, target_region_id) if isinstance(v, Placeholder) else v for v in node.variables]

        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Jacobian(new_target, new_vars, node.scheme)

        return node

    if isinstance(node, Hessian):
        new_target = rebind_variational_variables(
            domain,
            node.target,
            target_support,
            target_region_id,
        )
        new_vars = [rebind_variational_variables(domain, v, target_support, target_region_id) if isinstance(v, Placeholder) else v for v in node.variables]

        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Hessian(new_target, new_vars, node.scheme, trace=node.trace)

        return node

    if isinstance(node, Tracker):
        new_expr = rebind_variational_variables(
            domain,
            node.expr,
            target_support,
            target_region_id,
        )

        if new_expr is not node.expr:
            rebuilt_tracker = Tracker(new_expr, interval=node.interval)
            rebuilt_tracker.op_id = node.op_id
            return rebuilt_tracker

        return node

    if isinstance(node, Assembly):
        new_expr = rebind_variational_variables(
            domain,
            node.expr,
            target_support,
            target_region_id,
        )

        if new_expr is not node.expr:
            rebuilt_assembly = Assembly(new_expr, node.num_total_nodes, node.support, node.region_id)
            rebuilt_assembly.op_id = node.op_id
            return rebuilt_assembly
        return node

    if isinstance(node, GroupedAssembly):
        vol_val = (
            rebind_variational_variables(
                domain,
                node.volume_value_expr,
                target_support,
                target_region_id,
            )
            if node.volume_value_expr is not None
            else None
        )
        vol_grad = (
            rebind_variational_variables(
                domain,
                node.volume_grad_expr,
                target_support,
                target_region_id,
            )
            if node.volume_grad_expr is not None
            else None
        )
        bnd_exprs = {k: rebind_variational_variables(domain, v, target_support, target_region_id) for k, v in node.boundary_value_exprs.items()}

        if vol_val is not node.volume_value_expr or vol_grad is not node.volume_grad_expr or any(bnd_exprs[k] is not node.boundary_value_exprs[k] for k in bnd_exprs):
            rebuilt = GroupedAssembly(vol_val, vol_grad, bnd_exprs, node.num_total_nodes)
            rebuilt.op_id = node.op_id
            return rebuilt

        return node

    return node


def bind_statefield_for_vpinn(domain, node, target_support: str, target_region_id: str):
    if node is None:
        return None

    if isinstance(node, StateField):
        rebound = rebind_variational_variables(
            domain,
            node.expr,
            target_support,
            target_region_id,
        )
        return rebound

    if isinstance(node, BinaryOp):
        left = bind_statefield_for_vpinn(
            domain,
            node.left,
            target_support,
            target_region_id,
        )
        right = bind_statefield_for_vpinn(
            domain,
            node.right,
            target_support,
            target_region_id,
        )

        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)

        return node

    if isinstance(node, FunctionCall):
        new_args = []
        changed = False

        for a in node.args:
            if isinstance(a, Placeholder):
                na = bind_statefield_for_vpinn(
                    domain,
                    a,
                    target_support,
                    target_region_id,
                )
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)

        if changed:
            return node.copy_with_args(new_args)

        return node

    if isinstance(node, ModelCall):
        new_args = []
        changed = False

        for a in node.args:
            if isinstance(a, Placeholder):
                na = bind_statefield_for_vpinn(
                    domain,
                    a,
                    target_support,
                    target_region_id,
                )
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)

        if changed:
            rebuilt = ModelCall(node.model, new_args)
            rebuilt.op_id = node.op_id
            return rebuilt

        return node

    if isinstance(node, (Jacobian, Hessian)):
        new_target = bind_statefield_for_vpinn(
            domain,
            node.target,
            target_support,
            target_region_id,
        )

        if isinstance(node, Jacobian):
            return Jacobian(new_target, node.variables, node.scheme)

        return Hessian(new_target, node.variables, node.scheme, trace=node.trace)

    return node


def bind_statefield_for_fem(node, trial_symbol=None):
    """
    Replace StateField nodes by their neural expression for VPINN evaluation.

    The wrapped expression is rebound to the requested volume/boundary region.
    """
    if node is None:
        return None

    if isinstance(node, StateField):
        if trial_symbol is None:
            trial_symbol = TrialFunction(name=node.name, value_shape=node.value_shape)
        return trial_symbol

    if isinstance(node, BinaryOp):
        left = bind_statefield_for_fem(node.left, trial_symbol)
        right = bind_statefield_for_fem(node.right, trial_symbol)

        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)

        return node

    if isinstance(node, FunctionCall):
        new_args = []
        changed = False

        for a in node.args:
            if isinstance(a, Placeholder):
                na = bind_statefield_for_fem(a, trial_symbol)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)

        if changed:
            return node.copy_with_args(new_args)

        return node

    if isinstance(node, ModelCall):
        new_args = []
        changed = False

        for a in node.args:
            if isinstance(a, Placeholder):
                na = bind_statefield_for_fem(a, trial_symbol)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)

        if changed:
            rebuilt = ModelCall(node.model, new_args)
            rebuilt.op_id = node.op_id
            return rebuilt

        return node

    if isinstance(node, (Jacobian, Hessian)):
        new_target = bind_statefield_for_fem(node.target, trial_symbol)

        if isinstance(node, Jacobian):
            return Jacobian(new_target, node.variables, node.scheme)

        return Hessian(new_target, node.variables, node.scheme, trace=node.trace)

    return node


def substitute_trial_for_vpinn(
    domain,
    node,
    trial_value,
    target_support: Optional[str] = None,
    target_region_id: Optional[str] = None,
):
    """
    Substitute TrialFunction symbols with a neural trial expression for VPINN.

    If a target support/region is provided, the trial expression is rebound to
    that quadrature region before substitution.
    """
    if node is None:
        return None

    if isinstance(node, (Variable, TestFunction, TensorTag, Constant, Literal)):
        return node

    if isinstance(node, TrialFunction):
        out = trial_value

        if target_support is not None and target_region_id is not None:
            out = rebind_variational_variables(
                domain,
                out,
                target_support,
                target_region_id,
            )

        return out

    if isinstance(node, BinaryOp):
        left = substitute_trial_for_vpinn(
            domain,
            node.left,
            trial_value,
            target_support,
            target_region_id,
        )
        right = substitute_trial_for_vpinn(
            domain,
            node.right,
            trial_value,
            target_support,
            target_region_id,
        )

        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)

        return node

    if isinstance(node, FunctionCall):
        new_args = [
            (
                substitute_trial_for_vpinn(
                    domain,
                    a,
                    trial_value,
                    target_support,
                    target_region_id,
                )
                if isinstance(a, Placeholder)
                else a
            )
            for a in node.args
        ]

        if any(n is not o for n, o in zip(new_args, node.args)):
            return FunctionCall(
                node.fn,
                new_args,
                node._name,
                node.reduces_axis,
                node.kwargs,
            )

        return node

    if isinstance(node, ModelCall):
        new_args = [
            (
                substitute_trial_for_vpinn(
                    domain,
                    a,
                    trial_value,
                    target_support,
                    target_region_id,
                )
                if isinstance(a, Placeholder)
                else a
            )
            for a in node.args
        ]

        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt_model = ModelCall(node.model, new_args)
            rebuilt_model.op_id = node.op_id
            return rebuilt_model

        return node

    if isinstance(node, OperationDef):
        new_expr = substitute_trial_for_vpinn(
            domain,
            node.expr,
            trial_value,
            target_support,
            target_region_id,
        )

        if new_expr is not node.expr:
            rebuilt_opdef = OperationDef.__new__(OperationDef)
            rebuilt_opdef.expr = new_expr
            rebuilt_opdef.input_vars = node.input_vars
            rebuilt_opdef.name = getattr(node, "name", None)
            rebuilt_opdef.op_id = node.op_id
            return rebuilt_opdef

        return node

    if isinstance(node, OperationCall):
        new_args = [
            (
                substitute_trial_for_vpinn(
                    domain,
                    a,
                    trial_value,
                    target_support,
                    target_region_id,
                )
                if isinstance(a, Placeholder)
                else a
            )
            for a in node.args
        ]

        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt_opcall = OperationCall(node.operation, tuple(new_args))
            rebuilt_opcall.op_id = node.op_id
            return rebuilt_opcall

        return node

    if isinstance(node, Jacobian):
        new_target = substitute_trial_for_vpinn(
            domain,
            node.target,
            trial_value,
            target_support,
            target_region_id,
        )
        new_vars = [
            (
                substitute_trial_for_vpinn(
                    domain,
                    v,
                    trial_value,
                    target_support,
                    target_region_id,
                )
                if isinstance(v, Placeholder)
                else v
            )
            for v in node.variables
        ]

        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Jacobian(new_target, new_vars, node.scheme)

        return node

    if isinstance(node, Hessian):
        new_target = substitute_trial_for_vpinn(
            domain,
            node.target,
            trial_value,
            target_support,
            target_region_id,
        )
        new_vars = [
            (
                substitute_trial_for_vpinn(
                    domain,
                    v,
                    trial_value,
                    target_support,
                    target_region_id,
                )
                if isinstance(v, Placeholder)
                else v
            )
            for v in node.variables
        ]

        if new_target is not node.target or any(n is not o for n, o in zip(new_vars, node.variables)):
            return Hessian(new_target, new_vars, node.scheme, trace=node.trace)

        return node

    if isinstance(node, Tracker):
        new_expr = substitute_trial_for_vpinn(
            domain,
            node.expr,
            trial_value,
            target_support,
            target_region_id,
        )

        if new_expr is not node.expr:
            rebuilt_tracker = Tracker(new_expr, interval=node.interval)
            rebuilt_tracker.op_id = node.op_id
            return rebuilt_tracker

        return node

    if isinstance(node, Assembly):
        new_expr = substitute_trial_for_vpinn(
            domain,
            node.expr,
            trial_value,
            target_support,
            target_region_id,
        )

        if new_expr is not node.expr:
            rebuilt_assembly = Assembly(
                new_expr,
                node.num_total_nodes,
                node.support,
                node.region_id,
            )
            rebuilt_assembly.op_id = node.op_id
            return rebuilt_assembly

        return node

    if isinstance(node, GroupedAssembly):
        vol_val = (
            substitute_trial_for_vpinn(
                domain,
                node.volume_value_expr,
                trial_value,
                target_support,
                target_region_id,
            )
            if node.volume_value_expr is not None
            else None
        )
        vol_grad = (
            substitute_trial_for_vpinn(
                domain,
                node.volume_grad_expr,
                trial_value,
                target_support,
                target_region_id,
            )
            if node.volume_grad_expr is not None
            else None
        )
        bnd_exprs = {
            k: substitute_trial_for_vpinn(
                domain,
                v,
                trial_value,
                target_support,
                target_region_id,
            )
            for k, v in node.boundary_value_exprs.items()
        }

        if vol_val is not node.volume_value_expr or vol_grad is not node.volume_grad_expr or any(bnd_exprs[k] is not node.boundary_value_exprs[k] for k in bnd_exprs):
            rebuilt_grouped = GroupedAssembly(
                vol_val,
                vol_grad,
                bnd_exprs,
                node.num_total_nodes,
            )
            rebuilt_grouped.op_id = node.op_id
            return rebuilt_grouped

        return node

    return node
