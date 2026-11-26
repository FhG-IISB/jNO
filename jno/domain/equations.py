"""Terms written as math that jno.fem lowers before assembly: complex terms, integral equations, initial values.

* **Complex terms.** A term built from ``fem_symbols(complex=True)`` fields is a :class:`~jno.trace.views.ComplexPair`.
  A weak term (one with a test function) enters as its real part: with W = W_re + i W_im, Re[a(A; W)] holds the
  equations of both parts. Any other term (a boundary value ``A(x, y) - 0``) enters as both parts. The result is a
  REAL form in (Re A, Im A), so a complex field works in nonlinear and coupled forms too: ``abs(A)**2``, sigma(T).
  The real part holds both test functions (W_re and W_im); it is split into one term per test function (the form is
  linear in them), so the native assembler takes it like any form in two real fields -- constant unknowns included.

* **Integral equations.** ``(expr * dV).integrate() - P`` with no test function is the equation of a constant
  unknown (``d.unknown.scalar(constant=True)``): int expr dV = P. It is assigned to the constant whose own test
  function appears in no other term -- exactly one must qualify, or the term raises and asks for the explicit form
  ``((expr * dV).integrate() - P) * I.test()``. It lowers to the volume term ``expr * mu`` (mu = the constant's test
  function, 1 everywhere) and ``-P`` on the constant's own row.

* **Initial values.** ``u.initialize(u0)`` on a field of a time-dependent domain is the initial condition
  ``u(initial) - u0``; it is added to the term list here. On a steady problem it is the solve's starting point,
  read by ``fem.solve()`` (``FEM._initialized_x0``), not a term.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..trace import BinaryOp, Integral, Literal, Placeholder, TestFunction, TrialFunction


def _walk(e):
    from ..trace.views import _VIEW_TYPES

    stack, seen = [e], set()
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, _VIEW_TYPES):
            stack.append(x._expr)
            continue
        if not isinstance(x, Placeholder):
            continue
        yield x
        if isinstance(x, (TrialFunction, TestFunction)):
            continue  # a field symbol is a leaf: a trial keeps its own test function as an attribute
        for v in x.__dict__.values():
            if isinstance(v, (list, tuple)):
                stack.extend(v)
            else:
                stack.append(v)


def _has(e, cls):
    return any(isinstance(n, cls) for n in _walk(e))


def _test_key(n):
    return getattr(n, "field_key", id(n))


def _tests_in(e):
    return {_test_key(n): n for n in _walk(e) if isinstance(n, TestFunction)}


def split_by_test(e):
    """``{test key: part}`` with ``e`` = the sum of the parts and each part holding one test function. ``e`` must
    be linear in its test functions (a weak form is): sums and differences are split, a product or quotient is
    split through the factor that holds the tests."""
    from ..trace.views import _VIEW_TYPES

    e = e._expr if isinstance(e, _VIEW_TYPES) else e
    keys = _tests_in(e)
    if len(keys) <= 1:
        return {next(iter(keys), None): e}
    if isinstance(e, BinaryOp) and e.op in ("+", "-"):
        left, right = split_by_test(e.left), split_by_test(e.right)
        out = dict(left)
        for k, p in right.items():
            if k in out:
                out[k] = BinaryOp(e.op, out[k], p)
            else:
                out[k] = p if e.op == "+" else BinaryOp("*", Literal(-1.0), p)
        return out
    if isinstance(e, BinaryOp) and e.op in ("*", "/"):
        in_left, in_right = bool(_tests_in(e.left)), bool(_tests_in(e.right))
        if in_left and not in_right:
            return {k: BinaryOp(e.op, p, e.right) for k, p in split_by_test(e.left).items()}
        if in_right and not in_left and e.op == "*":
            return {k: BinaryOp("*", e.left, p) for k, p in split_by_test(e.right).items()}
    raise ValueError(
        "jno.fem: a weak term with the test functions of a complex field must be linear in them; found "
        f"{type(e).__name__} {getattr(e, 'op', '')!s} holding {len(keys)} test functions together."
    )


def lower_complex_terms(constraints):
    from ..trace.views import ComplexPair

    if not isinstance(constraints, (list, tuple)):
        return constraints
    out = []
    for t in constraints:
        if isinstance(t, ComplexPair):
            parts = [p for p in (t.real, t.imag) if p is not None]
            if any(_has(p, TestFunction) for p in parts):
                if t.real is None:
                    raise ValueError("jno.fem: a complex weak term with no real part has no equations; check the form.")
                # Re[a(A; W_re + i W_im)]: the equations of both parts, one term per test function
                out.extend(split_by_test(t.real).values())
            else:
                # a boundary value: both parts, each as `part - value`. `A(x, y) - 0.0` subtracts the 0 from the
                # real part only, which leaves the imaginary part a bare `A_im(x, y)` -- not an equation to jno.fem
                for p in parts:
                    q = p._expr if hasattr(p, "_expr") else p
                    out.append(p if isinstance(q, BinaryOp) and q.op == "-" else p - 0.0)
        else:
            out.append(t)
    return out


def _additive(e, sign=1.0):
    """Split a top-level sum into [(sign, part)]."""
    from ..trace.views import _VIEW_TYPES

    e = e._expr if isinstance(e, _VIEW_TYPES) else e
    if isinstance(e, BinaryOp) and e.op in ("+", "-"):
        return _additive(e.left, sign) + _additive(e.right, sign if e.op == "+" else -sign)
    return [(sign, e)]


def _constant_value(e):
    if isinstance(e, (int, float)):
        return float(e)
    if isinstance(e, Literal):
        v = np.asarray(e.value)
        if v.size == 1 and not np.iscomplexobj(v):
            return float(v.reshape(()))
    return None


def lower_integral_equations(constraints):
    from .._fem import Coupling

    if not isinstance(constraints, (list, tuple)):
        return constraints
    pending, out = [], []
    for t in constraints:
        if isinstance(t, Placeholder) or hasattr(t, "_expr"):
            if _has(t, Integral) and not _has(t, TestFunction):
                pending.append(t)
                continue
        out.append(t)
    if not pending:
        return constraints
    # constants whose test function appears in no term: the candidates to own an integral equation
    constants, tested = {}, set()
    for t in out + pending:
        for n in _walk(t) if (isinstance(t, Placeholder) or hasattr(t, "_expr")) else ():
            if isinstance(n, TrialFunction) and getattr(n, "_is_constant", False):
                constants[getattr(n, "field_key", id(n))] = n
            if isinstance(n, TestFunction):
                tested.add(getattr(n, "field_key", id(n)))
    for t in pending:
        free = [c for k, c in constants.items() if k not in tested]
        if len(free) != 1:
            raise ValueError(
                "jno.fem: an integral equation (expr.integrate() - value, with no test function) belongs to the one "
                f"constant unknown that has no equation yet; found {len(free)} such constants. Write it with the "
                "constant's test function instead: ((expr * dV).integrate() - value) * I.test()."
            )
        const = free[0]
        mu = const.test()
        tested.add(getattr(const, "field_key", id(const)))
        volume, value = None, 0.0
        for sign, part in _additive(t):
            c = _constant_value(part)
            if c is not None:
                value += sign * c
                continue
            if isinstance(part, Integral) and part.integration_var is None:
                piece = part.target * mu if sign > 0 else -(part.target * mu)
                volume = piece if volume is None else volume + piece
                continue
            raise ValueError(
                "jno.fem: an integral equation must be a sum of integrals and constants, e.g. "
                "(joule * dV).integrate() - P; got a term with " + type(part).__name__ + "."
            )
        if volume is None:
            raise ValueError("jno.fem: an integral equation without an integral.")
        out.append(volume)
        if value != 0.0:
            cst = Coupling(
                lambda u, _v=value: jnp.full(jnp.shape(u), _v, dtype=jnp.result_type(u)),
                name="integral-equation value",
                field_key=getattr(const, "field_key", None),
            )
            cst._support = np.array([0])
            out.append(cst)
    return out


def _is_initial_condition_of(t, key):
    """Whether term ``t`` is an initial condition ``u(initial) - g`` of the field ``key`` (not a velocity one)."""
    from ..trace import Variable

    nodes = list(_walk(t))
    if any(isinstance(n, TestFunction) for n in nodes):
        return False
    if not any(isinstance(n, TrialFunction) and _test_key(n) == key for n in nodes):
        return False
    tags = {getattr(n, "_jno_region_tag", getattr(n, "tag", None)) for n in nodes if isinstance(n, Variable)}
    return "initial" in tags and "__time_initial__" not in tags


def lower_initial_values(constraints):
    """Add ``u(initial) - u0`` for every field initialized on a time-dependent domain. Returns the terms and
    ``[(field, version)]`` of the values it used (``fem.solve()`` refuses a value changed after the build)."""
    if not isinstance(constraints, (list, tuple)):
        return constraints, []
    fields = {}
    for t in constraints:
        if isinstance(t, Placeholder) or hasattr(t, "_expr"):
            for n in _walk(t):
                if isinstance(n, TrialFunction) and "_initial" in n.__dict__:
                    fields.setdefault(_test_key(n), n)
    out, used = list(constraints), []
    for key, u in fields.items():
        dom = u.__dict__.get("_domain")
        if dom is None or not getattr(dom, "_is_time_dependent", False):
            continue  # steady: the starting point, read by fem.solve()
        if getattr(u, "_is_constant", False):
            raise ValueError(
                f"jno.fem: {u.name}.initialize(...) on a constant unknown of a time-dependent problem: a constant "
                "unknown has no initial state of its own."
            )
        if any(_is_initial_condition_of(t, key) for t in constraints if isinstance(t, Placeholder) or hasattr(t, "_expr")):
            raise ValueError(
                f"jno.fem: the initial state of {u.name} is given twice, by {u.name}.initialize(...) and by a term "
                f"{u.name}(initial) - u0. Keep one."
            )
        coords = dom.variable("initial", split=True)[: dom.dimension]
        out.append(u(*coords) - u.__dict__["_initial"])
        used.append((u, u.__dict__.get("_initial_version", 0)))
    return out, used
