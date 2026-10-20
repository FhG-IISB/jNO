"""A structural key of a form's OPERATOR: two forms whose keys match differ at most in their load.

Forms that share their operator (the same bilinear terms over the same unknowns, coefficients and boundary
terms) and differ only in their sources -- one form per drive, load case, incident direction, sample -- can
be solved with ONE factorisation and a block of right-hand sides. The key is how jNO recognises them.

It is built from the weak-form terms with every load summand (a test function but no trial function) left
out. Structure is compared, not object identity, because the same bilinear expression is usually rebuilt per
form (``[jno.fem([a(u, v) - l_k(v), bc]) for l_k in loads]``); but every LEAF that carries data counts by
identity -- a parameter, a frozen field, a symbol's field -- so two keys only match when the operator really
is the same function of the same inputs. A node type this module does not know keys by identity, which can
only ever MISS a sharing, never invent one. The solve that shares a factorisation still checks every
member's residual against that member's own matrix, so a key that matched wrongly raises.
"""

from __future__ import annotations

from typing import Any


def _hashable(v):
    import numpy as np

    if isinstance(v, (int, float, complex, str, bool, type(None))):
        return v
    if isinstance(v, (tuple, list)):
        return tuple(_hashable(x) for x in v)
    try:
        a = np.asarray(v)
        if a.dtype != object:
            return ("arr", a.shape, str(a.dtype), a.tobytes())
    except Exception:  # noqa: BLE001
        pass
    return ("id", id(v))


def _fn_key(fn):
    """A callable by what it computes: module, qualified name, captured cells and defaults (a lambda built
    per call with different captures is a different function)."""
    import functools

    if isinstance(fn, functools.partial):
        return ("partial", _fn_key(fn.func), _hashable(fn.args), _hashable(sorted((fn.keywords or {}).items())))
    code = getattr(fn, "__code__", None)
    if code is None:
        return ("id", id(fn))
    cells = tuple(_hashable(c.cell_contents) for c in (fn.__closure__ or ()))
    return (
        getattr(fn, "__module__", None),
        getattr(fn, "__qualname__", None),
        code.co_code,
        cells,
        _hashable(getattr(fn, "__defaults__", None)),
    )


def structural_key(node, memo=None, leaf=None, commutative=False) -> Any:
    """The structural key of an expression tree (see the module notes). ``leaf(node)``, when given, may key a
    node itself (return a key) or leave it to the rules below (return ``None``) -- e.g. a frozen field as a
    numbered slot, so two integrands that differ only in which fields they read get one key. ``commutative``
    compares chains of ``*`` and ``+`` and the two arguments of ``inner`` as unordered (for :func:`symmetric_operator`)."""
    from ...trace import (
        BinaryOp,
        FrozenField,
        FunctionCall,
        Jacobian,
        Literal,
        ModelCall,
        RegionMask,
        TestFunction,
        TrialFunction,
        Variable,
    )
    from ...trace.views import _VIEW_TYPES

    memo = {} if memo is None else memo
    if id(node) in memo:
        return memo[id(node)]
    n = node._expr if isinstance(node, _VIEW_TYPES) else node
    if leaf is not None:
        k = leaf(n)
        if k is not None:
            memo[id(node)] = k
            return k
    sk = lambda c: structural_key(c, memo, leaf, commutative)  # noqa: E731
    t = type(n).__name__
    if isinstance(n, (TrialFunction, TestFunction)):
        k = (
            t,
            n.field_key,
            tuple(getattr(n, "value_shape", ()) or ()),
            getattr(n, "order", None),
            getattr(n, "space", None),
        )
    elif isinstance(n, FrozenField):
        k = (t, n.frozen_id)
    elif isinstance(n, Variable):
        k = (t, getattr(n, "tag", None), _hashable(getattr(n, "dim", None)), id(getattr(n, "_domain", None)))
    elif isinstance(n, Literal):
        k = (t, _hashable(n.value))
    elif isinstance(n, RegionMask):
        k = (t, n.region)
    elif isinstance(n, BinaryOp) and commutative and n.op in ("*", "+"):
        parts = []

        def flat(m):
            m_ = m._expr if isinstance(m, _VIEW_TYPES) else m
            if isinstance(m_, BinaryOp) and m_.op == n.op:
                flat(m_.left)
                flat(m_.right)
            else:
                parts.append(sk(m))

        flat(n.left)
        flat(n.right)
        k = (t, n.op, tuple(sorted(parts, key=repr)))
    elif isinstance(n, BinaryOp):
        k = (t, n.op, sk(n.left), sk(n.right))
    elif isinstance(n, Jacobian):
        k = (t, sk(n.target), tuple(sk(v) for v in n.variables), _hashable(getattr(n, "scheme", None)))
    elif isinstance(n, ModelCall):
        k = (t, id(n.model), tuple(sk(a) for a in (n.args or ())))
    elif isinstance(n, FunctionCall):
        args = tuple(
            sk(a) if hasattr(a, "__class__") and a.__class__.__module__.startswith("jno") else _hashable(a)
            for a in (n.args or ())
        )
        if commutative and getattr(n, "_name", None) == "inner" and len(args) == 2 and not getattr(n, "kwargs", None):
            args = tuple(sorted(args, key=repr))
        k = (
            t,
            getattr(n, "_name", None),
            _fn_key(n.fn),
            _hashable(getattr(n, "getitem_key", None)),
            args,
            _hashable(sorted((getattr(n, "kwargs", None) or {}).items())),
        )
    else:
        k = (t, id(n))  # unknown node: identity -- may miss a sharing, never invents one
    memo[id(node)] = k
    return k


def operator_key(domain, constraints, extra=()) -> Any:
    """The key of a form's operator: its constraints with the load summands (test function, no trial) left out.
    ``extra`` (mode, complex size, quadrature, ...) is part of the key."""
    from ...trace import TestFunction, TrialFunction
    from ...trace.views import _VIEW_TYPES
    from .solver_helper import contains_node_type
    from .weak_form import _split_additive_terms

    memo: dict = {}
    parts = []
    for c in constraints:
        bare = c._expr if isinstance(c, _VIEW_TYPES) else c
        if contains_node_type(bare, TestFunction):
            for sign, sub in _split_additive_terms(domain, bare):
                if contains_node_type(sub, TrialFunction):
                    parts.append(("op", sign, structural_key(sub, memo)))
        else:  # a constraint without a test function: an essential / boundary condition, part of the operator
            parts.append(("bc", structural_key(bare, memo)))
    return (id(domain), tuple(extra), tuple(sorted(parts, key=repr)))


def symmetric_operator(domain, constraints) -> bool:
    """Is the operator of the form ``constraints`` provably symmetric?

    Yes when the bilinear part -- the additive terms with a trial function -- is the same set of terms when each
    trial function and the test function of its field trade places, with products, sums and ``inner`` compared
    as commutative. Essential conditions (constraints without a test function) do not enter: the steady linear
    path eliminates them symmetrically (known columns lifted to the right-hand side, rows and columns zeroed,
    unit diagonal; see tests/test_fem_dirichlet_symmetry.py). For the steady linear mode only -- the transient
    blocks replace rows. ``False`` whenever symmetry cannot be shown: a form this does not recognise is solved as
    a general one, never factored as symmetric when it is not."""
    from ...trace import TestFunction, TrialFunction
    from ...trace.views import _VIEW_TYPES
    from .solver_helper import contains_node_type
    from .weak_form import _split_additive_terms

    def key(sub, swap):
        def leaf(n):
            if isinstance(n, TrialFunction):
                role = "test" if swap else "trial"
            elif isinstance(n, TestFunction):
                role = "trial" if swap else "test"
            else:
                return None
            return (role, n.field_key, tuple(n.value_shape or ()), int(n.order), str(n.space))

        return repr(structural_key(sub, {}, leaf, commutative=True))

    terms = []
    for c in constraints:
        bare = c._expr if isinstance(c, _VIEW_TYPES) else c
        if not contains_node_type(bare, TestFunction):
            continue  # an essential condition: eliminated symmetrically on the steady linear path
        for sign, sub in _split_additive_terms(domain, bare):
            if contains_node_type(sub, TrialFunction):
                terms.append((repr(sign), sub))
    if not terms:
        return False
    return sorted((sg, key(t, False)) for sg, t in terms) == sorted((sg, key(t, True)) for sg, t in terms)
