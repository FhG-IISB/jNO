"""Expressions in mathematical notation, for :func:`jno.info`.

``render(expr, names)`` writes a traced expression the way it would be written on paper: ``∂u/∂t + u ∂u/∂x
− 0.00318 ∂²u/∂x²``. Coordinates print as ``x, y, z`` by column and ``t`` for time, network outputs by the
names :func:`field_names` gives them (``u``, or ``u₀, u₁, ...`` for the components of one network, ``.name()``
where the user set one), derivative chains collapse (``∂²u/∂x∂t``, ``ΔΔu``). Where the terminal cannot encode
Unicode the same text is written in ASCII (``d2u/dx2``, ``lap(u)``).

A node this module does not know prints as its class name, so a report never fails on a new node type.
"""

from __future__ import annotations

import math
import sys

import numpy as np

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_FIELD_LETTERS = "uvwpqrs"
_GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "eps": "ε",
    "epsilon": "ε",
    "eta": "η",
    "theta": "θ",
    "kappa": "κ",
    "lambda": "λ",
    "lam": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
}
_COORDS = "xyz"

# precedence: sums < products < unary minus < powers < atoms
_SUM, _PROD, _NEG, _POW, _ATOM = 1, 2, 3, 4, 5


def unicode_ok(stream=None) -> bool:
    """Whether ``stream`` (default ``sys.stdout``) can encode the mathematical symbols."""
    enc = getattr(stream or sys.stdout, "encoding", None) or ""
    try:
        "∂∇Δ∫²₀π−·≤".encode(enc)
        return True
    except (LookupError, UnicodeEncodeError):
        return False


class _Style:
    def __init__(self, ascii_only: bool):
        self.ascii = ascii_only

    def symbol_const(self, sym: str) -> str:
        """A known constant: ``pi^2`` -> ``π²``, ``sqrt2`` -> ``√2`` (ASCII keeps the spelling)."""
        if self.ascii:
            return {"sqrt2": "sqrt(2)", "sqrt3": "sqrt(3)"}.get(sym, sym)
        return {"pi": "π", "pi^2": "π²", "pi^3": "π³", "sqrt2": "√2", "sqrt3": "√3", "e": "e"}[sym]

    def symbol(self, name: str) -> str:
        """A user's name as a symbol: ``phi`` -> ``φ`` where the terminal can show it."""
        return name if self.ascii else _GREEK.get(name, name)

    def sup(self, n) -> str:
        return f"^{n}" if self.ascii else str(n).translate(_SUP)

    def sub(self, n) -> str:
        return f"_{n}" if self.ascii else str(n).translate(_SUB)

    def __getattr__(self, key):
        table = {
            "d": ("∂", "d"),
            "lap": ("Δ", "lap "),
            "grad": ("∇", "grad "),
            "int": ("∫", "int "),
            "dot": ("·", "*"),
            "minus": ("−", "-"),
            "le": ("≤", "<="),
            "ge": ("≥", ">="),
            "pi": ("π", "pi"),
            "sqrt": ("√", "sqrt"),
            "hess": ("∇∇", "hess "),
        }
        if key in table:
            return table[key][1 if self.ascii else 0]
        raise AttributeError(key)


_NODE_NAMES = (
    "Assembly",
    "BinaryOp",
    "Constant",
    "Constraint",
    "FunctionCall",
    "Hessian",
    "Integral",
    "Jacobian",
    "Literal",
    "ModelCall",
    "OperationCall",
    "OperationDef",
    "Placeholder",
    "TemporalDerivative",
    "TestFunction",
    "Tracker",
    "TrialFunction",
    "Variable",
)


def _node_types():
    """The trace node classes by name (imported late: this module is imported by ``jno.trace``)."""
    from .. import trace

    return {n: getattr(trace, n) for n in _NODE_NAMES}


def _user_name(node):
    nm = getattr(node, "_user_name", None)
    return nm if isinstance(nm, str) and nm else None


def _fn_name(fc) -> str:
    name = getattr(fc, "_name", None) or getattr(getattr(fc, "fn", None), "__name__", "") or "f"
    return name.lstrip("_")


def _component_index(fc):
    """``i`` for ``a[..., i]`` (the component read every jno trace uses), else ``None``."""
    key = getattr(fc, "getitem_key", None)
    if isinstance(key, tuple) and len(key) >= 2 and key[0] is Ellipsis and isinstance(key[1], (int, np.integer)):
        if all(k is None for k in key[2:]):
            return int(key[1])
    return None


def field_names(exprs):
    """Names for the fields the expressions differentiate and the networks they read.

    A differentiated expression that is not a bare network output (``net(x, t) * x * (1 - x)``, a field built on
    a network) is a *field* of its own: it gets the next letter (``u``, ``v``, ...) and a ``where`` definition,
    and is printed by that letter everywhere it appears. A network read anywhere else gets a letter too (``u``,
    or ``u₀, u₁, ...`` for its components); one that only appears inside such fields prints as its class name.
    ``.name(...)`` on a node wins.

    Returns ``(names, definitions)``: ``names`` maps ``("node", id)``, ``("model", id)`` and ``("component",
    id, i)`` to symbols; ``definitions`` are ``(symbols, rhs)`` pairs, ``rhs`` a node or ``(label, args)``.
    """
    T = _node_types()
    deriv = (T["Jacobian"], T["Hessian"], T["TemporalDerivative"])

    def contains_model(n, seen=None):
        seen = set() if seen is None else seen
        if not isinstance(n, T["Placeholder"]) or id(n) in seen:
            return False
        seen.add(id(n))
        return isinstance(n, T["ModelCall"]) or any(contains_model(c, seen) for c in _children(n, T))

    def is_network(n):
        return isinstance(n, T["ModelCall"]) or (
            isinstance(n, T["FunctionCall"]) and _fn_name(n) == "getitem" and isinstance(n.args[0], T["ModelCall"])
        )

    # fields: the innermost non-derivative targets of derivative chains that contain a network
    fields, direct, seen = [], set(), set()

    def find(n):
        if not isinstance(n, T["Placeholder"]) or id(n) in seen:
            return
        seen.add(id(n))
        if isinstance(n, deriv):
            base = n.target
            while isinstance(base, deriv):
                base = base.target
            if is_network(base):
                direct.add(id(base.model) if isinstance(base, T["ModelCall"]) else id(base.args[0].model))
            elif contains_model(base) and all(f is not base for f in fields):
                fields.append(base)
        for c in _children(n, T):
            find(c)

    for e in exprs:
        find(e)
    field_ids = {id(f) for f in fields}

    # networks: in order of first appearance; `outside` when read other than inside a field
    order, calls, components, outside, named = [], {}, {}, set(), {}
    seen = set()

    def walk(n, inside):
        if not isinstance(n, T["Placeholder"]) or (id(n), inside) in seen:
            return
        seen.add((id(n), inside))
        inside = inside or id(n) in field_ids
        if isinstance(n, T["FunctionCall"]) and _fn_name(n) == "getitem" and isinstance(n.args[0], T["ModelCall"]):
            i = _component_index(n)
            if i is not None:
                mid = id(n.args[0].model)
                components.setdefault(mid, set()).add(i)
                if _user_name(n):
                    named[("component", mid, i)] = _user_name(n)
        if isinstance(n, T["ModelCall"]):
            mid = id(n.model)
            if mid not in calls:
                calls[mid] = n
                order.append(mid)
            if not inside:
                outside.add(mid)
            if _user_name(n):
                named[("model", mid)] = _user_name(n)
        for c in _children(n, T):
            walk(c, inside)

    for e in exprs:
        walk(e, False)

    letters = iter([c for c in _FIELD_LETTERS] + [f"u{k}" for k in range(len(_FIELD_LETTERS), 99)])
    names, defs = {}, []
    for f in fields:  # the fields the equations are about take the first letters
        sym = _user_name(f) or next(letters)
        names[("node", id(f))] = sym
        defs.append(([sym], f, ("node", id(f))))
    for mid in order:
        call = calls[mid]
        module = getattr(call.model, "module", None)
        label = getattr(call.model, "name", None)
        label = label if isinstance(label, str) and label else (type(module).__name__ if module is not None else "net")
        if label.startswith("_Parameter"):  # a grid unknown or a trainable parameter: values, not a network
            label = "unknown"
        names[("label", mid)] = label
        # a network takes a letter when it is differentiated itself, or when nothing else is (a data fit);
        # otherwise it prints as its class name inside the fields built on it
        if ("model", mid) not in named and mid not in direct and (fields or mid not in outside):
            continue
        base = named.get(("model", mid)) or next(letters)
        names[("model", mid)] = base
        comps = sorted(components.get(mid, ()))
        for i in comps:
            names[("component", mid, i)] = named.get(("component", mid, i)) or (
                base if comps == [0] and _single_output(call) else f"{base}#{i}"
            )
        symbols = [names[("component", mid, i)] for i in comps] if comps else [base]
        defs.append((symbols, (label, list(call.args)), ("model", mid)))
    return names, defs


def _single_output(call) -> bool:
    out = getattr(getattr(call.model, "module", None), "output_dim", None)
    return out == 1


def _children(n, T):
    if isinstance(n, T["BinaryOp"]):
        return [n.left, n.right]
    if isinstance(n, (T["FunctionCall"], T["ModelCall"])):
        return [a for a in n.args if isinstance(a, T["Placeholder"])]
    if isinstance(n, T["OperationCall"]):
        return [n.operation.expr] + [a for a in n.args if isinstance(a, T["Placeholder"])]
    if isinstance(n, T["OperationDef"]):
        return [n.expr]
    if isinstance(n, (T["Jacobian"], T["Hessian"], T["TemporalDerivative"], T["Integral"])):
        return [n.target]
    if isinstance(n, T["Constraint"]):
        return [n.residual]
    if isinstance(n, (T["Tracker"], T["Assembly"])):
        return [n.expr]
    return []


# constants a number is recognised as a small fraction of: (value, symbol, in the denominator?)
_KNOWN = (
    (math.pi, "pi", False),
    (math.pi**2, "pi^2", False),
    (math.pi**3, "pi^3", False),
    (math.pi, "pi", True),
    (math.pi**2, "pi^2", True),
    (math.sqrt(2.0), "sqrt2", False),
    (math.sqrt(3.0), "sqrt3", False),
    (math.e, "e", False),
)


def _fraction(r: float, rel: float):
    """``r`` as ``Fraction(p, q)`` with ``q <= 12`` or a power of ten (and ``|p| <= 1000``), if one agrees with it
    to ``rel``; else ``None``."""
    from fractions import Fraction

    for limit in (12, 1000):
        f = Fraction(r).limit_denominator(limit)
        if f.denominator > 12 and f.denominator not in (100, 1000):
            continue
        if abs(f.numerator) <= 1000 and abs(float(f) - r) <= rel * max(abs(r), 1e-300):
            return f
    return None


def _coefficient(f) -> str:
    """A fraction as written in a formula: ``3``, ``3/4``, ``0.01``."""
    if f.denominator == 1:
        return str(f.numerator)
    if f.denominator in (10, 100, 1000):
        return f"{float(f):g}"
    return f"{f.numerator}/{f.denominator}"


def _number(v, st: _Style, rel: float = 1e-6) -> str:
    """A number as written in a formula: an integer, a small fraction (``1/3``), a small fraction of a known
    constant (``3π²``, ``0.01/π``, ``√2/2``), or else four significant digits.

    A match must agree with the stored value to ``rel`` -- its precision (float32 about 1e-7, float64 about
    1e-16, with room for the arithmetic that produced it) -- so a number typed to four digits (``0.3183``) is
    never promoted to ``1/π``.
    """
    v = float(v)
    if v < 0:
        return st.minus + _number(-v, st, rel)
    if v == 0 or (v == int(v) and v < 1e6):
        return str(int(v))
    f = _fraction(v, rel)
    if f is not None and f.denominator not in (10, 100, 1000):
        return _coefficient(f)
    for value, sym, below in _KNOWN:
        f = _fraction(v * value if below else v / value, rel)
        if f is None:
            continue
        k = st.symbol_const(sym)
        j = "*" if st.ascii else ""  # 3π², 3*pi^2
        if below:
            return (
                f"{_coefficient(f)}/{k}"
                if f.denominator == 1 or f.denominator in (10, 100, 1000)
                else f"{f.numerator}/({f.denominator}{j}{k})"
            )
        if f.denominator == 1:
            return k if f.numerator == 1 else f"{f.numerator}{j}{k}"
        if f.denominator in (10, 100, 1000):
            return f"{_coefficient(f)}{j or ' '}{k}"
        return (k if f.numerator == 1 else f"{f.numerator}{j}{k}") + f"/{f.denominator}"
    return f"{v:.4g}"


class _Renderer:
    def __init__(self, names, st: _Style):
        self.names, self.st, self.T = names, st, _node_types()
        self.defining = None  # the field whose definition is being written: printed in full, not by its name

    def __call__(self, n) -> str:
        return self.go(n)[0]

    def wrap(self, n, prec):
        text, p = self.go(n)
        return f"({text})" if p < prec else text

    def go(self, n):
        T, st = self.T, self.st
        if not isinstance(n, T["Placeholder"]):
            if isinstance(n, (int, float, np.floating, np.integer)):
                txt = _number(n, st)
                return (txt, _NEG if txt.startswith("-") else _ATOM)
            return (str(n), _ATOM)
        if ("node", id(n)) in self.names and self.defining is not n:
            return (self.names[("node", id(n))], _ATOM)
        if getattr(n, "_attached_name", None):  # a property attached to the domain: d.k
            return (st.symbol(n._attached_name), _ATOM)
        if _user_name(n) and not isinstance(n, (T["ModelCall"], T["FunctionCall"])):
            return (st.symbol(_user_name(n)), _ATOM)
        if isinstance(n, T["Literal"]):
            v = np.asarray(getattr(n, "value", np.nan))
            rel = 1e-6 if v.dtype.itemsize <= 4 else 1e-12
            if v.size == 1:
                txt = _number(v.reshape(()), st, rel)
                if txt.startswith(("-", "−")):
                    return (txt, _NEG)
                return (txt, _PROD if ("/" in txt or " " in txt) else _ATOM)
            return ("[" + ", ".join(_number(x, st) for x in v.reshape(-1)[:4]) + (", …]" if v.size > 4 else "]"), _ATOM)
        if isinstance(n, T["Constant"]):
            return (str(getattr(n, "key", None) or getattr(n, "tag", "c")), _ATOM)
        if isinstance(n, T["Variable"]):
            return (self.variable(n), _ATOM)
        if isinstance(n, T["ModelCall"]):
            return (self.model_call(n), _ATOM)
        if isinstance(n, (T["TrialFunction"], T["TestFunction"])):
            return (st.symbol(str(getattr(n, "name", None) or ("u" if isinstance(n, T["TrialFunction"]) else "v"))), _ATOM)
        if isinstance(n, T["BinaryOp"]):
            return self.binary(n)
        if isinstance(n, T["FunctionCall"]):
            return self.function(n)
        if isinstance(n, (T["Jacobian"], T["TemporalDerivative"])):
            return self.derivative(n)
        if isinstance(n, T["Hessian"]) and len(n.variables) == 1 and self._single_coord(n.variables[0]):
            c = self.variable(n.variables[0])  # one coordinate: ∂²u/∂x², whatever node carries it
            target, tp = self.go(n.target)
            if tp == _ATOM and len(target) <= 12:
                return (f"{st.d}{st.sup(2)}{target}/{st.d}{c}{st.sup(2)}", _ATOM)
            return (f"{st.d}{st.sup(2)}/{st.d}{c}{st.sup(2)} ({target})", _PROD)
        if isinstance(n, T["Hessian"]):
            if not getattr(n, "trace", False):
                return (st.hess + self.wrap(n.target, _ATOM), _NEG)
            k, node = 0, n
            while isinstance(node, T["Hessian"]) and getattr(node, "trace", False):
                k, node = k + 1, node.target
            return (st.lap + (st.sup(k) if k > 1 else "") + self.wrap(node, _ATOM), _NEG)
        if isinstance(n, T["Integral"]):
            return (f"{st.int} {self.wrap(n.target, _PROD)} dx", _PROD)
        if isinstance(n, T["Assembly"]):
            region = getattr(n, "region_id", None)
            d = "ds" if getattr(n, "support", "") == "boundary" else "dx"
            sub = f"_{region}" if region else ""
            return (f"{st.int}{sub} {self.wrap(n.expr, _PROD)} {d}", _PROD)
        if isinstance(n, T["OperationDef"]):
            return self.go(n.expr)
        if isinstance(n, T["OperationCall"]):
            return self.go(n.operation.expr)
        if isinstance(n, T["Tracker"]):
            return self.go(n.expr)
        if isinstance(n, T["Constraint"]):
            return (self.constraint(n), _SUM)
        return (f"{type(n).__name__}(…)", _ATOM)

    def variable(self, v) -> str:
        if _user_name(v):
            return _user_name(v)
        if getattr(v, "axis", "spatial") == "temporal":
            return "t"
        tag = str(getattr(v, "tag", ""))
        dim = getattr(v, "dim", None)
        if tag.startswith("n_"):
            return "n"
        if isinstance(dim, (list, tuple)) and len(dim) == 2 and dim[1] is not None:
            if dim[1] - dim[0] == 1:
                i = int(dim[0])
                return _COORDS[i] if i < len(_COORDS) else "x" + self.st.sub(i)
            return "X"
        return "X"

    def model_call(self, call) -> str:
        args = [a for a in call.args]
        name = self.names.get(("model", id(call.model)))
        if name is None:  # a network that only lives inside a named field: its class name and arguments
            return f"{self.names.get(('label', id(call.model)), 'net')}({', '.join(self(a) for a in args)})"
        if all(isinstance(a, self.T["Variable"]) for a in args):
            return name
        return f"{name}({', '.join(self(a) for a in args)})"

    def binary(self, b):
        st, op = self.st, b.op
        L, R = b.left, b.right
        if op in ("+", "-"):
            if self._is_zero(R):
                return self.go(L)
            if op == "+" and self._is_zero(L):
                return self.go(R)
            left = self.wrap(L, _SUM)
            rtext, rp = self.go(R)
            negative = rtext.startswith("-") or rtext.startswith("−")
            if op == "-":
                if negative and rp >= _PROD:  # a − (−c x) = a + c x
                    return (f"{left} + {rtext[1:]}", _SUM)
                rtext = f"({rtext})" if rp <= _SUM else rtext
                return (f"{left} {st.minus} {rtext}", _SUM)
            if negative:
                return (f"{left} {st.minus} {rtext[1:]}", _SUM)
            return (f"{left} + {rtext}", _SUM)
        if op == "*":
            lit_l, lit_r = self._is_number(L), self._is_number(R)
            if lit_r and not lit_l:
                L, R = R, L  # a number first: 0.5 u, not u 0.5
            left, right = self.wrap(L, _PROD), self.wrap(R, _PROD)
            if left in ("-1", "−1"):
                return (f"{st.minus}{right}", _NEG)
            if left == "1":
                return (right, self.go(R)[1])
            sep = f" {st.dot} " if self._is_number(R) else " "
            return (f"{left}{sep}{right}", _PROD)
        if op == "/":
            return (f"{self.wrap(L, _PROD)}/{self.wrap(R, _POW)}", _PROD)
        if op == "**":
            base = self.wrap(L, _ATOM)
            if self._is_number(R):
                e = float(np.asarray(R.value).reshape(())) if hasattr(R, "value") else float(R)
                if e == int(e) and 0 <= e <= 9:
                    return (base + st.sup(int(e)), _POW)
                if e == 0.5:
                    return (f"{st.sqrt}{base}" if not st.ascii else f"sqrt({self(L)})", _ATOM)
            return (f"{base}^{self.wrap(R, _ATOM)}", _POW)
        cmp = {"<": "<", ">": ">", "<=": st.le, ">=": st.ge, "@": "@"}.get(op, op)
        return (f"{self.wrap(L, _SUM)} {cmp} {self.wrap(R, _SUM)}", _SUM)

    def _is_zero(self, n) -> bool:
        if not self._is_number(n):
            return False
        v = n if isinstance(n, (int, float, np.floating, np.integer)) else np.asarray(n.value).reshape(())
        return float(v) == 0.0

    def _is_number(self, n) -> bool:
        if isinstance(n, (int, float, np.floating, np.integer)):
            return True
        return isinstance(n, self.T["Literal"]) and np.asarray(getattr(n, "value", np.nan)).size == 1

    def function(self, fc):
        st, name = self.st, _fn_name(fc)
        args = [a for a in fc.args]
        if name == "getitem":
            i = _component_index(fc)
            base = args[0]
            if i is not None and isinstance(base, self.T["ModelCall"]):
                sym = self.names.get(("component", id(base.model), i))
                if sym is None and ("model", id(base.model)) not in self.names:
                    return (self.model_call(base) + st.sub(i), _ATOM)
                if sym is not None:
                    sym = self._sub_symbol(sym)
                    return (
                        sym
                        if all(isinstance(a, self.T["Variable"]) for a in base.args)
                        else f"{sym}({', '.join(self(a) for a in base.args)})",
                        _ATOM,
                    )
            if i is not None:
                return (self.wrap(base, _ATOM) + st.sub(i), _ATOM)
            return (f"{self.wrap(base, _ATOM)}[{getattr(fc, 'getitem_key', '…')}]", _ATOM)
        if name in ("concat", "concatenate", "stack"):
            return ("(" + ", ".join(self(a) for a in args) + ")", _ATOM)
        if name in ("mse", "mae"):
            return (self(args[0]), _SUM)
        if name in ("inner", "dot", "vdot") and len(args) == 2:
            return (f"{self.wrap(args[0], _PROD)} {st.dot} {self.wrap(args[1], _PROD)}", _PROD)
        if name == "abs" and len(args) == 1:
            return (f"|{self(args[0])}|", _ATOM)
        if name == "sqrt" and len(args) == 1:
            return (f"{st.sqrt}({self(args[0])})" if not st.ascii else f"sqrt({self(args[0])})", _ATOM)
        if name in ("negative", "neg") and len(args) == 1:
            return (f"{st.minus}{self.wrap(args[0], _POW)}", _NEG)
        if name == "grad" and args:
            return (st.grad + self.wrap(args[0], _ATOM), _NEG)
        inner = ", ".join(self(a) for a in args if isinstance(a, self.T["Placeholder"])) or ", ".join(map(str, args))
        return (f"{name}({inner})", _ATOM)

    def _sub_symbol(self, sym: str) -> str:
        base, _, idx = sym.partition("#")
        return base + self.st.sub(idx) if idx else base

    def derivative(self, n):
        """A chain of single-coordinate derivatives collapsed: ∂u/∂x, ∂²u/∂x∂t, ∂³u/∂x³."""
        T, st = self.T, self.st
        coords, node = [], n
        while True:
            if isinstance(node, T["TemporalDerivative"]):
                coords.append("t")
                node = node.target
                continue
            if isinstance(node, T["Jacobian"]) and len(node.variables) == 1 and self._single_coord(node.variables[0]):
                coords.append(self.variable(node.variables[0]))
                node = node.target
                continue
            break
        if not coords:  # a gradient over several coordinates
            vars_ = getattr(n, "variables", [])
            return (
                st.grad + self.wrap(n.target, _ATOM)
                if len(vars_) != 1 or not self._single_coord(vars_[0])
                else self(n.target),
                _NEG,
            )
        order = len(coords)
        counts = {}
        for c in reversed(coords):
            counts[c] = counts.get(c, 0) + 1
        top = st.d + (st.sup(order) if order > 1 else "")
        bottom = "".join(st.d + c + (st.sup(k) if k > 1 else "") for c, k in counts.items())
        target, tp = self.go(node)
        if tp == _ATOM and len(target) <= 12:
            return (f"{top}{target}/{bottom}", _ATOM)
        return (f"{top}/{bottom} ({target})", _PROD)

    def _single_coord(self, v) -> bool:
        if getattr(v, "axis", "spatial") == "temporal":
            return True
        dim = getattr(v, "dim", None)
        return isinstance(dim, (list, tuple)) and len(dim) == 2 and dim[1] is not None and dim[1] - dim[0] == 1

    def constraint(self, c) -> str:
        sense = getattr(c, "sense", "eq")
        rel = {"le": self.st.le, "ge": self.st.ge}.get(sense, "=")
        return f"{self(c.residual)} {rel} 0"


def render(expr, names=None, *, ascii_only=None) -> str:
    """``expr`` in mathematical notation (see the module docstring)."""
    st = _Style(not unicode_ok() if ascii_only is None else ascii_only)
    names = names if names is not None else field_names([expr])[0]
    try:
        return _Renderer(names, st)(expr)
    except Exception as err:  # noqa: BLE001 -- a report must not fail on an expression it cannot spell
        return f"<{type(expr).__name__}: not printable ({type(err).__name__})>"


def equation(expr, names, *, ascii_only=None, measure=None) -> tuple[str, str]:
    """``(equation, loss)`` for one constraint: ``"∂u/∂t + … = 0"`` and ``"mse"`` (or ``""``).

    ``measure="dx"`` / ``"ds"`` writes a weak-form integrand as the integral it contributes, ``∫ (…) dx``.
    """
    from . import FunctionCall

    e = getattr(expr, "_expr", expr)
    loss = ""
    if isinstance(e, FunctionCall) and _fn_name(e) in ("mse", "mae"):
        loss, e = _fn_name(e), e.args[0]
    st = _Style(not unicode_ok() if ascii_only is None else ascii_only)
    try:
        text = _Renderer(names, st)(e)
    except Exception as err:  # noqa: BLE001
        return f"<not printable ({type(err).__name__})>", loss
    if getattr(e, "sense", None) in ("le", "ge", "eq"):  # a Constraint renders its own relation
        return text, loss
    if measure:
        return f"{st.int} ({text}) {measure}", loss
    return f"{text} = 0", loss


def where_rows(defs, names, *, ascii_only=None, shapes=None, width=58):
    """``[("where", "u = MLP(x, t)"), ...]`` from :func:`field_names` definitions. With ``shapes`` (``key ->
    [shape, ...]``, as :func:`tree` collects it) only the definitions it holds, each with its shapes."""
    st = _Style(not unicode_ok() if ascii_only is None else ascii_only)
    r = _Renderer(names, st)
    rows = []
    for symbols, rhs, key in defs:
        if shapes is not None and key not in shapes:
            continue
        syms = [r._sub_symbol(s) for s in symbols]
        lhs = syms[0] if len(syms) == 1 else "(" + ", ".join(syms) + ")"
        if isinstance(rhs, tuple):
            label, args = rhs
            text = f"{label}({', '.join(r(a) for a in args)})" if args else label
        else:
            r.defining = rhs
            text = r(rhs)
            r.defining = None
        text = f"{lhs} = {text}"
        if shapes is not None:
            known = [sh for sh in shapes[key] if sh != "per point"] or shapes[key]
            text = f"{text.ljust(width)}  {', '.join(known)}"
        rows.append(("where" if not rows else "", text))
    return rows


def _terms(n, T, op_set):
    """The operands of a chain of ``op_set`` operations (``a + b − c`` -> a, b, −c), with the sign each
    enters with; a chain is flattened so a sum of five terms is five siblings, not four nested levels."""
    if isinstance(n, T["BinaryOp"]) and n.op in op_set:
        left = _terms(n.left, T, op_set)
        right = _terms(n.right, T, op_set)
        if n.op == "-":
            right = [(c, not neg) for c, neg in right]
        return left + right
    return [(n, False)]


def _shown_children(n, T):
    """``[(child, negated)]`` a tree shows: sums and products flattened, derivative chains collapsed to the
    field they differentiate, wrappers looked through, and literals and coordinates (visible in their
    parent's label) left out."""
    deriv = (T["Jacobian"], T["Hessian"], T["TemporalDerivative"])
    skip = (T["Literal"], T["Constant"], T["Variable"])
    if isinstance(n, deriv):  # ∂²u/∂x∂t is one node over u
        base = n.target
        while isinstance(base, deriv):
            base = base.target
        return [] if isinstance(base, skip) else [(base, False)]
    if isinstance(n, T["BinaryOp"]) and n.op in ("+", "-"):
        kids = _terms(n, T, ("+", "-"))
    elif isinstance(n, T["BinaryOp"]) and n.op == "*":
        kids = _terms(n, T, ("*",))
    else:
        kids = [(c, False) for c in _children(n, T)]
    out = []
    for c, neg in kids:
        if isinstance(c, skip):
            continue
        if isinstance(c, (T["OperationDef"], T["Tracker"])):
            out += _shown_children(c, T)
            continue
        out.append((c, neg))
    return out


def _definition_key(n, names, T):
    """The ``where`` entry a node stands for -- a named field, a network, one of a network's outputs -- or
    ``None`` for a node drawn in the tree."""
    if ("node", id(n)) in names:
        return ("node", id(n))
    if isinstance(n, T["ModelCall"]):
        return ("model", id(n.model))
    if isinstance(n, T["FunctionCall"]) and _fn_name(n) == "getitem" and isinstance(n.args[0], T["ModelCall"]):
        return ("model", id(n.args[0].model))
    return None


def tree(expr, names, shapes, *, defined=None, ascii_only=None, width=58) -> list:
    """``expr`` as an indented tree, one node per line in mathematical notation with its shape.

    Nothing is drawn twice. A named node (a field, a network or one of its outputs) is not drawn at all: it is
    visible by name in every label that uses it and is defined once, in the ``where`` block (``defined``
    collects ``key -> [shapes]`` for it, see :func:`where_rows`). Any other node used again is drawn the
    first time only. ``shapes`` maps ``id(node)`` to a shape string; a node without one is evaluated only
    inside a derivative pass (``per point``).
    """
    st = _Style(not unicode_ok() if ascii_only is None else ascii_only)
    T = _node_types()
    r = _Renderer(names, st)
    tee, last, bar, gap = ("├─ ", "└─ ", "│  ", "   ") if not st.ascii else ("|- ", "`- ", "|  ", "   ")
    defined = {} if defined is None else defined
    lines, done = [], set()

    def label(n):
        try:
            text = r(n)
        except Exception as err:  # noqa: BLE001
            text = f"{type(n).__name__} ({type(err).__name__})"
        return text if len(text) <= width else text[: width - 1] + "…"

    def record(n, key):
        shape = shapes.get(id(n), "per point")
        seen = defined.setdefault(key, [])
        if shape not in seen:
            seen.append(shape)

    def visible(n):
        key = _definition_key(n, names, T)
        if key is not None:
            record(n, key)
            return False
        return id(n) not in done

    def visit(n, prefix, is_last, top=False, negated=False):
        branch = "" if top else (last if is_last else tee)
        text = label(n)
        if negated:  # a subtracted term: − a, and − (−a) = a
            text = text[1:] if text.startswith(("-", "−")) else st.minus + text
        lines.append(
            f"{prefix}{branch}{text.ljust(max(width - len(prefix) - len(branch), 0))}  {shapes.get(id(n), 'per point')}"
        )
        done.add(id(n))
        kids = [(c, neg) for c, neg in _shown_children(n, T) if visible(c)]
        child_prefix = prefix + ("" if top else (gap if is_last else bar))
        for i, (c, neg) in enumerate(kids):
            visit(c, child_prefix, i == len(kids) - 1, negated=neg)

    key = _definition_key(expr, names, T)
    if key is not None:
        record(expr, key)
    visit(expr, "", True, top=True)
    return lines
