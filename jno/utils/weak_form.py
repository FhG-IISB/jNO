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
    StateField,
    WeakReduction,
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
# domain infer helpers
# --------------------------------
def _infer_domain_from_expr(expr):
    domains = []

    def walk(node):
        if node is None:
            return

        if isinstance(node, Variable):
            d = getattr(node, "_domain", None)
            if d is not None:
                domains.append(d)

        if isinstance(node, StateField):
            walk(node.expr)
            return

        for attr in ("left", "right", "target", "expr"):
            child = getattr(node, attr, None)
            if child is not None:
                walk(child)

        for attr in ("args", "variables"):
            vals = getattr(node, attr, None)
            if vals is None:
                continue
            for v in vals:
                if isinstance(v, (list, tuple)):
                    for vv in v:
                        walk(vv)
                else:
                    walk(v)

    walk(expr)
    uniq = []
    seen = set()
    for d in domains:
        if id(d) not in seen:
            seen.add(id(d))
            uniq.append(d)

    if len(uniq) == 1:
        return uniq[0]
    if len(uniq) == 0:
        raise ValueError("Could not infer domain from weak expression. Pass domain=... explicitly.")
    raise ValueError("Weak expression references multiple domains. Pass the intended domain explicitly.")



def _contains_model_eval(node) -> bool:
    if isinstance(node, ModelCall):
        return True
    for attr in ("left", "right", "target", "expr"):
        child = getattr(node, attr, None)
        if child is not None and _contains_model_eval(child):
            return True
    for attr in ("args", "variables"):
        vals = getattr(node, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if _contains_model_eval(vv):
                        return True
            else:
                if _contains_model_eval(v):
                    return True
    return False


def _depends_on_domain_variables(node) -> bool:
    if isinstance(node, Variable):
        return True
    for attr in ("left", "right", "target", "expr"):
        child = getattr(node, attr, None)
        if child is not None and _depends_on_domain_variables(child):
            return True
    for attr in ("args", "variables"):
        vals = getattr(node, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if _depends_on_domain_variables(vv):
                        return True
            else:
                if _depends_on_domain_variables(v):
                    return True
    return False


def _wrap_primary_state(node, target):
    if node is target:
        return StateField(node, state_id=0, name="u", value_shape=())

    if node is None:
        return None

    if isinstance(node, BinaryOp):
        left = _wrap_primary_state(node.left, target)
        right = _wrap_primary_state(node.right, target)
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = []
        changed = False
        for a in node.args:
            if isinstance(a, Placeholder):
                na = _wrap_primary_state(a, target)
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
                na = _wrap_primary_state(a, target)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            rebuilt = ModelCall(node.model, new_args)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, Jacobian):
        new_target = _wrap_primary_state(node.target, target)
        new_vars = []
        changed = new_target is not node.target
        for v in node.variables:
            if isinstance(v, Placeholder):
                nv = _wrap_primary_state(v, target)
            else:
                nv = v
            changed = changed or (nv is not v)
            new_vars.append(nv)
        if changed:
            return Jacobian(new_target, new_vars, node.scheme)
        return node

    if isinstance(node, Hessian):
        new_target = _wrap_primary_state(node.target, target)
        new_vars = []
        changed = new_target is not node.target
        for v in node.variables:
            if isinstance(v, Placeholder):
                nv = _wrap_primary_state(v, target)
            else:
                nv = v
            changed = changed or (nv is not v)
            new_vars.append(nv)
        if changed:
            return Hessian(new_target, new_vars, node.scheme, trace=node.trace)
        return node

    if isinstance(node, OperationDef):
        new_expr = _wrap_primary_state(node.expr, target)
        if new_expr is not node.expr:
            rebuilt = OperationDef.__new__(OperationDef)
            rebuilt.expr = new_expr
            rebuilt.input_vars = node.input_vars
            rebuilt.name = getattr(node, "name", None)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, OperationCall):
        new_args = []
        changed = False
        for a in node.args:
            if isinstance(a, Placeholder):
                na = _wrap_primary_state(a, target)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            rebuilt = OperationCall(node.operation, tuple(new_args))
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    if isinstance(node, Tracker):
        new_expr = _wrap_primary_state(node.expr, target)
        if new_expr is not node.expr:
            rebuilt = Tracker(new_expr, interval=node.interval)
            rebuilt.op_id = node.op_id
            return rebuilt
        return node

    return node

def _ensure_statefield_wrapped(domain, expr):
    if _contains_node_type(domain, expr, StateField) or _contains_node_type(domain, expr, TrialFunction):
        return expr
    candidate = _detect_primary_state_field(domain, expr)
    #print("DEBUG primary state candidate:", candidate)
    if candidate is None:
        return expr
    return _wrap_primary_state(expr, candidate)

def _iter_placeholder_children(node):
    for attr in ("left", "right", "target", "expr"):
        child = getattr(node, attr, None)
        if child is not None and isinstance(child, Placeholder):
            yield child

    for attr in ("args", "variables"):
        vals = getattr(node, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if isinstance(vv, Placeholder):
                        yield vv
            else:
                if isinstance(v, Placeholder):
                    yield v


def _contains_subexpr(root, target):
    if root is target:
        return True
    for child in _iter_placeholder_children(root):
        if _contains_subexpr(child, target):
            return True
    return False

def _contains_testfunction_gradient(domain, expr):
    if expr is None:
        return False

    if isinstance(expr, Jacobian) and isinstance(expr.target, TestFunction):
        return True

    for attr in ("left", "right", "operand", "expr", "integrand", "target"):
        child = getattr(expr, attr, None)
        if child is not None and _contains_testfunction_gradient(domain, child):
            return True

    for attr in ("args", "variables"):
        vals = getattr(expr, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if _contains_testfunction_gradient(domain, vv):
                        return True
            else:
                if _contains_testfunction_gradient(domain, v):
                    return True

    return False

def _has_weak_basis_symbols(domain, node):
    return (
        _contains_node_type(domain, node, TestFunction)
        or _contains_node_type(domain, node, TrialFunction)
        or _contains_testfunction_gradient(domain, node)
    )


def _collect_region_keys(domain, node):
    metas = []
    _collect_variational_metas(domain, node, metas)
    return {(m["support"], m["region_id"]) for m in metas}


def _is_statefield_candidate(domain, node):
    if node is None:
        return False

    # Never wrap existing weak symbols or already-wrapped state fields.
    if isinstance(node, (StateField, TestFunction, TrialFunction)):
        return False

    # NEW: derivatives are not the state field itself
    if isinstance(node, (Jacobian, Hessian)):
        return False

    # The unknown field itself must not already include weak basis functions.
    if _has_weak_basis_symbols(domain, node):
        return False

    # Reject additive composites like (u**3 - u) or whole weak expressions.
    if isinstance(node, BinaryOp) and node.op in {"+", "-"}:
        return False

    # Must actually come from a model / NN evaluation somewhere.
    if not _contains_model_eval(node):
        return False

    # Must depend on domain variables.
    if not _depends_on_domain_variables(node):
        return False

    # If sampled FEM regions appear, they must all belong to one bucket only.
    keys = _collect_region_keys(domain, node)
    if len(keys) > 1:
        return False

    return True


def _collect_state_field_candidates(domain, node, out):
    if node is None:
        return

    if _is_statefield_candidate(domain, node):
        out.append(node)
        return   # IMPORTANT: do not descend further

    for child in _iter_placeholder_children(node):
        _collect_state_field_candidates(domain, child, out)


# def _prune_to_minimal_candidates(candidates):
#     """Keep only the smallest candidates, e.g. keep u, discard u**3."""
#     uniq = []
#     seen = set()
#     for c in candidates:
#         if id(c) not in seen:
#             seen.add(id(c))
#             uniq.append(c)

#     minimal = []
#     for c in uniq:
#         contains_other = any(
#             (other is not c) and _contains_subexpr(c, other)
#             for other in uniq
#         )
#         if not contains_other:
#             minimal.append(c)

#     return minimal
def _unique_by_id(nodes):
    out = []
    seen = set()
    for n in nodes:
        if id(n) not in seen:
            seen.add(id(n))
            out.append(n)
    return out


def _collect_derivative_based_state_targets(domain, node, out):
    if node is None:
        return

    if isinstance(node, (Jacobian, Hessian)):
        tgt = node.target

        if not isinstance(tgt, (TrialFunction, TestFunction, StateField)):
            if (
                _contains_model_eval(tgt)
                and _depends_on_domain_variables(tgt)
                and not _has_weak_basis_symbols(domain, tgt)
            ):
                keys = _collect_region_keys(domain, tgt)
                if len(keys) <= 1:
                    out.append(tgt)

    for child in _iter_placeholder_children(node):
        _collect_derivative_based_state_targets(domain, child, out)

def _detect_primary_state_field(domain, expr):
    # Phase 1 robust path:
    # if the unknown already appears as target of grad/hessian, use that first.
    deriv_targets = []
    _collect_derivative_based_state_targets(domain, expr, deriv_targets)
    deriv_targets = _unique_by_id(deriv_targets)

    if len(deriv_targets) == 1:
        return deriv_targets[0]

    if len(deriv_targets) > 1:
        raise NotImplementedError(
            "Multiple derivative-based state targets detected in weak form. "
            "Phase 1 supports exactly one unknown. "
            "Phase 3 will support multi-unknown systems."
        )

    # Fallback for weak forms without grad(u), e.g. pure reaction-like forms
    candidates = []
    _collect_state_field_candidates(domain, expr, candidates)
    candidates = _unique_by_id(candidates)

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        raise NotImplementedError(
            "Multiple state-field candidates detected in weak form. "
            "Phase 1 supports exactly one unknown. "
            "Phase 3 will support multi-unknown systems."
        )

    return candidates[0]
# --------------------------------
# variational region helpers
# --------------------------------
def _contains_node_type(domain, expr, node_type):
    if expr is None:
        return False

    if isinstance(expr, node_type):
        return True

    # recurse through common single-child attrs
    for attr in ("left", "right", "operand", "expr", "integrand", "target"):
        child = getattr(expr, attr, None)
        if child is not None and _contains_node_type(domain, child, node_type):
            return True

    # recurse through list-like attrs
    for attr in ("args", "variables"):
        vals = getattr(expr, attr, None)
        if vals is None:
            continue
        for v in vals:
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if _contains_node_type(domain, vv, node_type):
                        return True
            else:
                if _contains_node_type(domain, v, node_type):
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

def _find_first_statefield(node):
    if node is None:
        return None

    if isinstance(node, StateField):
        return node

    for child in _iter_placeholder_children(node):
        found = _find_first_statefield(child)
        if found is not None:
            return found

    return None

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

def _bind_statefield_for_vpinn(domain, node, target_support: str, target_region_id: str):
    if node is None:
        return None

    if isinstance(node, StateField):
        rebound = _rebind_variational_variables(domain, node.expr, target_support, target_region_id)
        return rebound

    if isinstance(node, BinaryOp):
        left = _bind_statefield_for_vpinn(domain, node.left, target_support, target_region_id)
        right = _bind_statefield_for_vpinn(domain, node.right, target_support, target_region_id)
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = []
        changed = False
        for a in node.args:
            if isinstance(a, Placeholder):
                na = _bind_statefield_for_vpinn(domain, a, target_support, target_region_id)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            return node.copy_with_args(new_args)
        return node

    if isinstance(node, (Jacobian, Hessian)):
        new_target = _bind_statefield_for_vpinn(domain, node.target, target_support, target_region_id)
        if isinstance(node, Jacobian):
            return Jacobian(new_target, node.variables, node.scheme)
        return Hessian(new_target, node.variables, node.scheme, trace=node.trace)

    return node


def _bind_statefield_for_fem(node, trial_symbol=None):
    if node is None:
        return None

    if isinstance(node, StateField):
        if trial_symbol is None:
            trial_symbol = TrialFunction(name=node.name, value_shape=node.value_shape)
        return trial_symbol

    if isinstance(node, BinaryOp):
        left = _bind_statefield_for_fem(node.left, trial_symbol)
        right = _bind_statefield_for_fem(node.right, trial_symbol)
        if left is not node.left or right is not node.right:
            return BinaryOp(node.op, left, right)
        return node

    if isinstance(node, FunctionCall):
        new_args = []
        changed = False
        for a in node.args:
            if isinstance(a, Placeholder):
                na = _bind_statefield_for_fem(a, trial_symbol)
                changed = changed or (na is not a)
                new_args.append(na)
            else:
                new_args.append(a)
        if changed:
            return node.copy_with_args(new_args)
        return node

    if isinstance(node, (Jacobian, Hessian)):
        new_target = _bind_statefield_for_fem(node.target, trial_symbol)
        if isinstance(node, Jacobian):
            return Jacobian(new_target, node.variables, node.scheme)
        return Hessian(new_target, node.variables, node.scheme, trace=node.trace)

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


# --------------------------------
# auto solver helpers
# --------------------------------
def _contains_temporal_derivative(expr) -> bool:
    from .time_route import _max_temporal_derivative_order
    return _max_temporal_derivative_order(expr) > 0


def _contains_unknown_symbol(domain, expr) -> bool:
    return (
        _contains_node_type(domain, expr, StateField)
        or _contains_node_type(domain, expr, TrialFunction)
    )


def _is_obviously_nonlinear_in_unknown(domain, expr):
    if expr is None:
        return False

    if isinstance(expr, BinaryOp):
        left_has = _contains_unknown_symbol(domain, expr.left)
        right_has = _contains_unknown_symbol(domain, expr.right)

        # product/division of two unknown-dependent factors -> nonlinear
        if expr.op in {"*", "/"} and left_has and right_has:
            return True

        # powers of the unknown are nonlinear except u**1
        if expr.op == "**":
            if left_has:
                # u**1 is linear, everything else is nonlinear
                if isinstance(expr.right, Literal):
                    try:
                        p = float(expr.right.value)
                        if p != 1.0:
                            return True
                    except Exception:
                        return True
                else:
                    return True

            # exponent depending on unknown is definitely nonlinear
            if right_has:
                return True

        return (
            _is_obviously_nonlinear_in_unknown(domain, expr.left)
            or _is_obviously_nonlinear_in_unknown(domain, expr.right)
        )

    if isinstance(expr, FunctionCall):
        unknown_args = [
            a for a in expr.args
            if isinstance(a, Placeholder) and _contains_unknown_symbol(domain, a)
        ]

        name = _function_name(expr)

        # Structural / linear-ish wrappers that should not force nonlinear classification
        linearish = {
            "inner",
            "reshape",
            "transpose",
            "getitem",
            "concat",
            "stack",
        }

        # Jacobian/Hessian are handled through their own nodes
        if len(unknown_args) > 0 and name not in linearish:
            return True

        return any(
            _is_obviously_nonlinear_in_unknown(domain, a)
            for a in expr.args
            if isinstance(a, Placeholder)
        )

    if isinstance(expr, (Jacobian, Hessian)):
        return _is_obviously_nonlinear_in_unknown(domain, expr.target)

    return False

def _infer_solver_target(domain, expr):
    if _contains_temporal_derivative(expr):
        raise NotImplementedError(
            "Automatic time-target inference belongs to Phase 2. "
            "For now, pass target='diffrax' or target='feax_time' after Phase 2 is added."
        )

    if _is_obviously_nonlinear_in_unknown(domain, expr):
        return "fem_residual"
    return "fem_system"
# -----------------------------------------------------------------------------
# Lower once, dispatch many
# -----------------------------------------------------------------------------
def lower_weak_form(domain, expr, trial_value=None, for_target="vpinn"):
    print("DEBUG lower_weak_form has StateField before wrap:",
      _contains_node_type(domain, expr, StateField))
    expr = _ensure_statefield_wrapped(domain, expr)
    print("DEBUG lower_weak_form has StateField after wrap:",
      _contains_node_type(domain, expr, StateField))
    shared_trial_symbol = None
    if for_target == "fem":
        sf = _find_first_statefield(expr)
        if sf is not None:
            shared_trial_symbol = TrialFunction(name=sf.name, value_shape=sf.value_shape)

    terms = _split_additive_terms(domain, expr)
    lowered_terms = []

    for sign, term in terms:
        support, region_id = _infer_term_bucket(domain, term)
        term_for_target = term

        if for_target == "vpinn":
            term_for_target = _bind_statefield_for_vpinn(
                domain,
                term_for_target,
                target_support=support,
                target_region_id=region_id,
            )

            if trial_value is not None:
                term_for_target = _substitute_trial_for_vpinn(
                    domain,
                    term_for_target,
                    trial_value,
                    target_support=support,
                    target_region_id=region_id,
                )

        else:
            term_for_target = _bind_statefield_for_fem(
                term_for_target,
                trial_symbol=shared_trial_symbol,
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

def assemble_weak_form(domain, expr, target=None, **kwargs):
    if domain is None:
        domain = _infer_domain_from_expr(expr)

    expr = _ensure_statefield_wrapped(domain, expr)

    if target is None:
        target = _infer_solver_target(domain, expr)

    if target == "vpinn":
        u_net = kwargs.get("u_net", None)

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
        "Supported in Phase 1: 'vpinn', 'fem_system', 'fem_residual'"
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