from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import jax.numpy as jnp
from ..jnp_ops import stack
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
class LoweredChannelTerm:
    sign: float
    support: str          # "volume" | "boundary"
    region_id: str
    channel: str          # "test_value" | "test_grad" | "boundary_test_value" | "raw"
    coeff: Placeholder
    variable_id: int = 0
    value_shape: tuple = ()
    original_expr: Placeholder = None


@dataclass
class LoweredWeakForm:
    domain: object
    terms: List[LoweredChannelTerm] = field(default_factory=list)

    def select(self, *, support=None, region_id=None, channel=None, variable_id=None):
        out = []
        for t in self.terms:
            if support is not None and t.support != support:
                continue
            if region_id is not None and t.region_id != region_id:
                continue
            if channel is not None and t.channel != channel:
                continue
            if variable_id is not None and t.variable_id != variable_id:
                continue
            out.append(t)
        return out

    def sum_coeffs(self, *, support=None, region_id=None, channel=None, variable_id=None):
        coeffs = [
            t.coeff
            for t in self.select(
                support=support,
                region_id=region_id,
                channel=channel,
                variable_id=variable_id,
            )
        ]
        return _sum_terms(self.domain, coeffs)

    # FEM-facing raw expressions
    @property
    def volume_expr(self):
        exprs = [t.coeff for t in self.terms if t.support == "volume" and t.channel == "raw"]
        return _sum_terms(self.domain, exprs)

    @property
    def boundary_exprs(self) -> Dict[str, Placeholder]:
        out: Dict[str, List[Placeholder]] = {}
        for t in self.terms:
            if t.support == "boundary" and t.channel == "raw":
                out.setdefault(t.region_id, []).append(t.coeff)
        return {k: _sum_terms(self.domain, v) for k, v in out.items()}

    # VPINN-facing canonical channels
    @property
    def volume_value_expr(self):
        return self.sum_coeffs(support="volume", channel="test_value", variable_id=0)

    @property
    def volume_grad_expr(self):
        return self.sum_coeffs(support="volume", channel="test_grad", variable_id=0)

    @property
    def boundary_value_exprs(self) -> Dict[str, Placeholder]:
        out = {}
        region_ids = sorted({t.region_id for t in self.terms if t.support == "boundary"})
        for rid in region_ids:
            coeff = self.sum_coeffs(
                support="boundary",
                region_id=rid,
                channel="boundary_test_value",
                variable_id=0,
            )
            if coeff is not None:
                out[rid] = coeff
        return out
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

def _contains_testfunction(node) -> bool:
    return _contains_node_type(None, node, TestFunction)

def _contains_trialfunction(node) -> bool:
    return _contains_node_type(None, node, TrialFunction)

def _function_name(node) -> Optional[str]:
    if isinstance(node, FunctionCall):
        if getattr(node, "_name", None) is not None:
            return str(node._name)
        if hasattr(node.fn, "__name__"):
            return str(node.fn.__name__)
    return None

def _get_grad_axis_from_test_grad(node) -> int:
    """
    Return the spatial derivative axis for Jacobian(TestFunction, [var]).
    For fem_gauss variables in 2D:
      x -> 0
      y -> 1
    """
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

    # For split=True variables:
    # xg has dim like [0,1], yg has dim like [1,2]
    if not hasattr(var, "dim") or len(var.dim) < 1:
        raise ValueError(f"Cannot infer gradient axis from variable {var}")

    axis = int(var.dim[0])
    return axis

def _canonicalize_grad_coeff(domain, coeff_expr, axis: int, value_shape: tuple):
    """
    Convert a directional grad(test) coefficient into canonical FEAX-style grad coeff.

    Scalar test:
        directional coeff shape ~ (...)        -> canonical (..., dim)

    Vector test with value_shape=(vec,):
        directional coeff shape ~ (..., vec)   -> canonical (..., vec, dim)
    """
    dim = int(domain.dimension)

    # scalar-valued test
    if value_shape is None or len(value_shape) == 0:
        comps = []
        for j in range(dim):
            scale = Literal(1.0 if j == axis else 0.0)
            comps.append(coeff_expr * scale)
        return stack(comps, axis=-1)

    # vector-valued test (current common case)
    if len(value_shape) == 1:
        comps = []
        for j in range(dim):
            scale = Literal(1.0 if j == axis else 0.0)
            comps.append(coeff_expr * scale)   # each has shape (..., vec)
        return stack(comps, axis=-1)           # -> (..., vec, dim)

    raise NotImplementedError(
        f"Canonical grad coeff inflation not implemented yet for value_shape={value_shape}"
    )
def _value_shape_num_components(value_shape) -> int:
    if value_shape is None or len(value_shape) == 0:
        return 1
    n = 1
    for s in value_shape:
        n *= int(s)
    return n

def _is_test_value(node):
    return isinstance(node, TestFunction)

def _is_test_grad(node):
    return isinstance(node, Jacobian) and isinstance(node.target, TestFunction)

def _is_symgrad_test(node) -> bool:
    if not isinstance(node, FunctionCall):
        return False
    name = _function_name(node)
    if name != "symgrad":
        return False
    if len(node.args) < 1:
        return False
    arg0 = node.args[0]
    return isinstance(arg0, TestFunction) or (
        isinstance(arg0, Jacobian) and isinstance(arg0.target, TestFunction)
    )

def _get_test_value_shape(node) -> tuple:
    if isinstance(node, TestFunction):
        return getattr(node, "value_shape", ())
    if isinstance(node, Jacobian) and isinstance(node.target, TestFunction):
        return getattr(node.target, "value_shape", ())
    if _is_symgrad_test(node):
        arg0 = node.args[0]
        if isinstance(arg0, TestFunction):
            return getattr(arg0, "value_shape", ())
        if isinstance(arg0, Jacobian) and isinstance(arg0.target, TestFunction):
            return getattr(arg0.target, "value_shape", ())
    return ()

def _extract_test_channel(domain, expr) -> Tuple[str, Placeholder, Dict[str, Any]]:
    """
    Canonicalize one weak-form term into FEAX-style channels:
      - volume/test_value
      - volume/test_grad
      - boundary_test_value (assigned later from support)
    """

    # pure test value
    if isinstance(expr, TestFunction):
        return "test_value", Literal(1.0), {
            "value_shape": getattr(expr, "value_shape", ()),
            "variable_id": 0,
        }

    # pure grad(test)
    if isinstance(expr, Jacobian) and isinstance(expr.target, TestFunction):
        value_shape = getattr(expr.target, "value_shape", ())
        axis = _get_grad_axis_from_test_grad(expr)
        coeff = _canonicalize_grad_coeff(domain, Literal(1.0), axis, value_shape)
        return "test_grad", coeff, {
            "value_shape": value_shape,
            "variable_id": 0,
        }
    # additive combination of same-channel terms
    if isinstance(expr, BinaryOp) and expr.op in {"+", "-"}:
        ch_left, coeff_left, meta_left = _extract_test_channel(domain, expr.left)
        ch_right, coeff_right, meta_right = _extract_test_channel(domain, expr.right)

        if ch_left != ch_right:
            raise ValueError(
                "Could not extract a single canonical test channel from additive term: "
                f"left channel={ch_left}, right channel={ch_right}, expr={expr}"
            )

        if meta_left.get("variable_id", 0) != meta_right.get("variable_id", 0):
            raise ValueError(
                "Additive weak-form term mixes different variable ids: "
                f"{meta_left.get('variable_id', 0)} vs {meta_right.get('variable_id', 0)}"
            )

        if tuple(meta_left.get("value_shape", ())) != tuple(meta_right.get("value_shape", ())):
            raise ValueError(
                "Additive weak-form term mixes different test value shapes: "
                f"{meta_left.get('value_shape', ())} vs {meta_right.get('value_shape', ())}"
            )

        if expr.op == "+":
            coeff = coeff_left + coeff_right
        else:
            coeff = coeff_left - coeff_right

        return ch_left, coeff, meta_left
    # direct product tree
    if isinstance(expr, BinaryOp) and expr.op == "*":
        left = expr.left
        right = expr.right

        if _is_test_value(left):
            return "test_value", right, {
                "value_shape": getattr(left, "value_shape", ()),
                "variable_id": 0,
            }
        if _is_test_value(right):
            return "test_value", left, {
                "value_shape": getattr(right, "value_shape", ()),
                "variable_id": 0,
            }

        if _is_test_grad(left):
            value_shape = getattr(left.target, "value_shape", ())
            axis = _get_grad_axis_from_test_grad(left)
            coeff = _canonicalize_grad_coeff(domain, right, axis, value_shape)
            return "test_grad", coeff, {
                "value_shape": value_shape,
                "variable_id": 0,
            }

        if _is_test_grad(right):
            value_shape = getattr(right.target, "value_shape", ())
            axis = _get_grad_axis_from_test_grad(right)
            coeff = _canonicalize_grad_coeff(domain, left, axis, value_shape)
            return "test_grad", coeff, {
                "value_shape": value_shape,
                "variable_id": 0,
            }

        # recurse left
        try:
            channel, coeff_left, meta = _extract_test_channel(domain, left)
            return channel, coeff_left * right, meta
        except ValueError:
            pass

        # recurse right
        try:
            channel, coeff_right, meta = _extract_test_channel(domain, right)
            return channel, left * coeff_right, meta
        except ValueError:
            pass

    # contractions
    if isinstance(expr, FunctionCall):
        name = _function_name(expr)

        if name == "inner" and len(expr.args) >= 2:
            a0, a1 = expr.args[0], expr.args[1]

            # inner(a, phi) or inner(phi, a)
            if _is_test_value(a0):
                return "test_value", a1, {
                    "value_shape": getattr(a0, "value_shape", ()),
                    "variable_id": 0,
                }
            if _is_test_value(a1):
                return "test_value", a0, {
                    "value_shape": getattr(a1, "value_shape", ()),
                    "variable_id": 0,
                }

            # inner(A, grad(phi)) or inner(grad(phi), A)
            if _is_test_grad(a0):
                return "test_grad", a1, {
                    "value_shape": getattr(a0.target, "value_shape", ()),
                    "variable_id": 0,
                }
            if _is_test_grad(a1):
                return "test_grad", a0, {
                    "value_shape": getattr(a1.target, "value_shape", ()),
                    "variable_id": 0,
                }

            # elasticity-like inner(sigma, symgrad(phi), n_contract=2)
            # canonical FEAX grad channel uses sigma itself
            if _is_symgrad_test(a0):
                return "test_grad", a1, {
                    "value_shape": _get_test_value_shape(a0),
                    "variable_id": 0,
                }
            if _is_symgrad_test(a1):
                return "test_grad", a0, {
                    "value_shape": _get_test_value_shape(a1),
                    "variable_id": 0,
                }

    raise ValueError(
        "Could not extract a canonical FEAX-style test channel from weak-form term. "
        f"Unsupported term structure: {expr}"
    )

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
    lowered_terms: List[LoweredChannelTerm] = []

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
            if _contains_node_type(domain, term_for_target, TrialFunction):
                raise RuntimeError(
                    "VPINN lowering failed: a TrialFunction is still present after substitution."
                )

        signed_expr = _apply_sign(domain, sign, term_for_target)

        if for_target == "vpinn":
            channel, coeff, meta = _extract_test_channel(domain, signed_expr)

            if support == "boundary":
                if channel != "test_value":
                    raise NotImplementedError(
                        f"Boundary grad(test)-type terms are not supported on region '{region_id}'."
                    )
                channel = "boundary_test_value"

            lowered_terms.append(
                LoweredChannelTerm(
                    sign=sign,
                    support=support,
                    region_id=region_id,
                    channel=channel,
                    coeff=coeff,
                    variable_id=int(meta.get("variable_id", 0)),
                    value_shape=tuple(meta.get("value_shape", ())),
                    original_expr=term,
                )
            )
        else:
            # FEM/FEAX keeps the full signed term expression
            lowered_terms.append(
                LoweredChannelTerm(
                    sign=sign,
                    support=support,
                    region_id=region_id,
                    channel="raw",
                    coeff=signed_expr,
                    variable_id=0,
                    value_shape=(),
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