from __future__ import annotations

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


# --------------------------------
# additive-term helpers
# --------------------------------
def _split_additive_terms(domain, node, sign=1.0):
    """
    Split an expression into additive terms:
        a - b + c  ->  [(+1,a), (-1,b), (+1,c)]
    """
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
    """Recursively check whether expr contains a node of type node_type."""
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
    """
    Walk an expression tree and collect fem_meta from Variable nodes.
    """
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
    """
    Infer whether a term belongs to:
        - ("volume", "volume")
        - ("boundary", "<region_id>")
    based on sampled variational variables appearing inside the term.

    Fallback rule:
        If no sampled variational variables are present, but the term contains
        TrialFunction and/or TestFunction, then interpret it as a volume term.
        This makes semilinear reaction terms like (u**3 - u) * phi work
        without needing an artificial factor like (1 + 0*xg).
    """
    metas = []
    _collect_variational_metas(domain, term, metas)

    # Standard path: explicit sampled variational variables exist
    if len(metas) > 0:
        supports = {m["support"] for m in metas}
        region_ids = {m["region_id"] for m in metas}

        if len(supports) != 1:
            raise ValueError(f"Mixed supports inside one term are not allowed. Found supports={supports}")

        if len(region_ids) != 1:
            raise ValueError(f"Mixed region ids inside one term are not allowed. Found region_ids={region_ids}")

        support = next(iter(supports))
        region_id = next(iter(region_ids))
        return support, region_id

    # Fallback path: no sampled FEM variables, but still a variational term.
    # Example: (u**3 - u) * phi
    has_trial = _contains_node_type(domain, term, TrialFunction)
    has_test = _contains_node_type(domain, term, TestFunction)

    if has_trial or has_test:
        return "volume", "volume"

    raise ValueError(
        "Could not infer variational bucket for term. "
        "Each assembled term must contain either:\n"
        "  (a) at least one sampled FEM/variational variable, or\n"
        "  (b) a TrialFunction/TestFunction-only variational term, which is "
        "then assumed to be a volume term."
    )


def _get_variational_region_meta(domain, support: str, region_id: str):
    """
    Find the registered variational-sampling metadata for one region.
    Returns the registry entry, e.g. for ("boundary", "right") -> gauss_right meta.
    """
    registry = getattr(domain, "_variational_sampling_registry", {})
    for sample_tag, meta in registry.items():
        if meta.get("support") == support and meta.get("region_id") == region_id:
            return meta
    raise KeyError(f"No variational sampling meta found for support={support!r}, region_id={region_id!r}. " f"Available: {registry}")


# --------------------------------
# trial substitution / rebind
# --------------------------------


def _rebind_variational_variables(domain, node, target_support: str, target_region_id: str):
    """
    Rebind variational coordinate variables inside an expression to a specific
    variational region.

    This is needed for VPINN boundary bilinear terms such as alpha(xr,yr) * u * phi,
    where the supplied trial expression u_net was originally built on fem_gauss
    but must be evaluated on gauss_right / gauss_top / ... for boundary assembly.

    Only Variable nodes carrying fem_meta are rewritten. Ordinary non-variational
    variables and tensor tags are left untouched.
    """

    if node is None:
        return None

    target_meta = _get_variational_region_meta(domain, target_support, target_region_id)
    target_tag = target_meta["context_tag"]

    # Rewrite only spatial variational coordinate variables
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

    # Leaves that stay unchanged
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
            rebuilt_model_call = ModelCall(node.model, new_args)
            rebuilt_model_call.op_id = node.op_id
            return rebuilt_model_call
        return node

    if isinstance(node, OperationDef):
        new_expr = _rebind_variational_variables(domain, node.expr, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt_operation_def = OperationDef.__new__(OperationDef)
            rebuilt_operation_def.expr = new_expr
            rebuilt_operation_def.input_vars = node.input_vars
            rebuilt_operation_def.name = getattr(node, "name", None)
            rebuilt_operation_def.op_id = node.op_id
            return rebuilt_operation_def
        return node

    if isinstance(node, OperationCall):
        new_args = [_rebind_variational_variables(domain, a, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt_operation_call = OperationCall(node.operation, tuple(new_args))
            rebuilt_operation_call.op_id = node.op_id
            return rebuilt_operation_call
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
            rebuilt_tracker = Tracker(new_expr, interval=node.interval)
            rebuilt_tracker.op_id = node.op_id
            return rebuilt_tracker
        return node

    if isinstance(node, Assembly):
        new_expr = _rebind_variational_variables(domain, node.expr, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt_assembly = Assembly(new_expr, node.num_total_nodes, node.support, node.region_id)
            rebuilt_assembly.op_id = node.op_id
            return rebuilt_assembly
        return node

    if isinstance(node, GroupedAssembly):
        vol_expr = _rebind_variational_variables(domain, node.volume_expr, target_support, target_region_id) if node.volume_expr is not None else None
        bnd_exprs = {k: _rebind_variational_variables(domain, v, target_support, target_region_id) for k, v in node.boundary_exprs.items()}
        if vol_expr is not node.volume_expr or any(bnd_exprs[k] is not node.boundary_exprs[k] for k in bnd_exprs):
            rebuilt_grouped_assembly = GroupedAssembly(vol_expr, bnd_exprs, node.num_total_nodes)
            rebuilt_grouped_assembly.op_id = node.op_id
            return rebuilt_grouped_assembly
        return node

    return node


def _substitute_trial_for_vpinn(
    domain,
    node,
    trial_value,
    target_support: str | None = None,
    target_region_id: str | None = None,
):
    """
    Replace symbolic ``TrialFunction`` nodes with a concrete VPINN trial
    expression.

    Boundary terms may also trigger a rebind of sampled variational
    coordinates to the target region.
    """
    if node is None:
        return None

    # Leaves that stay unchanged
    if isinstance(node, (Variable, TestFunction, TensorTag, Constant, Literal)):
        return node

    # Replace the symbolic unknown by the provided VPINN expression.
    # If the current term belongs to a boundary bucket, first rebind the
    # variational coordinates inside the supplied trial expression from
    # fem_gauss -> gauss_<region>.
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
            rebuilt_model_call = ModelCall(node.model, new_args)
            rebuilt_model_call.op_id = node.op_id
            return rebuilt_model_call
        return node

    if isinstance(node, OperationDef):
        new_expr = _substitute_trial_for_vpinn(domain, node.expr, trial_value, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt_operation_def = OperationDef.__new__(OperationDef)
            rebuilt_operation_def.expr = new_expr
            rebuilt_operation_def.input_vars = node.input_vars
            rebuilt_operation_def.name = getattr(node, "name", None)
            rebuilt_operation_def.op_id = node.op_id
            return rebuilt_operation_def
        return node

    if isinstance(node, OperationCall):
        new_args = [_substitute_trial_for_vpinn(domain, a, trial_value, target_support, target_region_id) if isinstance(a, Placeholder) else a for a in node.args]
        if any(n is not o for n, o in zip(new_args, node.args)):
            rebuilt_operation_call = OperationCall(node.operation, tuple(new_args))
            rebuilt_operation_call.op_id = node.op_id
            return rebuilt_operation_call
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
            rebuilt_tracker = Tracker(new_expr, interval=node.interval)
            rebuilt_tracker.op_id = node.op_id
            return rebuilt_tracker
        return node

    if isinstance(node, Assembly):
        new_expr = _substitute_trial_for_vpinn(domain, node.expr, trial_value, target_support, target_region_id)
        if new_expr is not node.expr:
            rebuilt_assembly = Assembly(new_expr, node.num_total_nodes, node.support, node.region_id)
            rebuilt_assembly.op_id = node.op_id
            return rebuilt_assembly
        return node

    if isinstance(node, GroupedAssembly):
        vol_expr = _substitute_trial_for_vpinn(domain, node.volume_expr, trial_value, target_support, target_region_id) if node.volume_expr is not None else None
        bnd_exprs = {k: _substitute_trial_for_vpinn(domain, v, trial_value, target_support, target_region_id) for k, v in node.boundary_exprs.items()}
        if vol_expr is not node.volume_expr or any(bnd_exprs[k] is not node.boundary_exprs[k] for k in bnd_exprs):
            rebuilt_grouped_assembly = GroupedAssembly(vol_expr, bnd_exprs, node.num_total_nodes)
            rebuilt_grouped_assembly.op_id = node.op_id
            return rebuilt_grouped_assembly
        return node

    return node


# --------------------------------
# grouped weak-form assembly
# --------------------------------


def assemble_weak_form(domain, expr, target="vpinn", **kwargs):
    """
    Assemble a symbolic weak form for the requested backend.

    Parameters
    ----------
    domain : object
        Domain providing geometry and variational metadata.
    expr : object
        Symbolic weak-form expression.
    target : {"vpinn", "fem_system", "fem_residual"}, default="vpinn"
        Assembly target.
    **kwargs
        Backend-specific options.

    Returns
    -------
    object
        Backend-specific assembled representation.
    """
    trial_value = kwargs.get("u_net", None) if target == "vpinn" else None

    terms = _split_additive_terms(domain, expr)

    volume_terms = []
    boundary_terms = {}

    for sign, term in terms:
        support, region_id = _infer_term_bucket(domain, term)

        term_for_target = term
        if target == "vpinn" and trial_value is not None:
            term_for_target = _substitute_trial_for_vpinn(
                domain,
                term,
                trial_value,
                target_support=support,
                target_region_id=region_id,
            )

        signed_term = _apply_sign(domain, sign, term_for_target)

        if support == "volume":
            volume_terms.append(signed_term)
        elif support == "boundary":
            boundary_terms.setdefault(region_id, []).append(signed_term)
        else:
            raise ValueError(f"Unknown support '{support}'")

    if target == "vpinn":
        return _assemble_vpinn_grouped(domain, volume_terms, boundary_terms, **kwargs)

    if target == "fem_system":
        from .fem_route import _assemble_fem_system_grouped

        return _assemble_fem_system_grouped(domain, volume_terms, boundary_terms, **kwargs)

    if target == "fem_residual":
        from .fem_route import _assemble_fem_residual_grouped

        return _assemble_fem_residual_grouped(domain, volume_terms, boundary_terms, **kwargs)

    raise ValueError(f"Unknown assembly target '{target}'. Supported: 'vpinn', 'fem_system', 'fem_residual'")


def _assemble_vpinn_grouped(domain, volume_terms, boundary_terms, **kwargs):
    """
    Assemble grouped VPINN volume and boundary terms into one internal node.
    """

    vol_expr = _sum_terms(domain, volume_terms) if len(volume_terms) > 0 else None
    boundary_exprs = {region_id: _sum_terms(domain, terms) for region_id, terms in boundary_terms.items()}

    if vol_expr is None and len(boundary_exprs) == 0:
        raise ValueError("No terms found for VPINN assembly.")

    return GroupedAssembly(vol_expr, boundary_exprs, domain)
