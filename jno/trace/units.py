"""Lightweight physical-unit algebra for graph-time dimensional analysis.

Units are graph-level metadata: they live on jno's Python wrapper objects
(:class:`jno.trace.Placeholder`) and are inferred at *trace time*, before any
JAX computation, so a user can audit that a PDE formulation is dimensionally
consistent.  See :mod:`jno.trace.unit_log` for the inference walk and the
public ``jno.units.check(...)`` entry point.

``pint`` is intentionally not a dependency — jno minimises external deps and
the algebra needed here is small.  A :class:`Unit` is just a map from SI base
dimension to a :class:`fractions.Fraction` exponent (fractions so ``sqrt``
stays exact).  The seven SI bases plus a handful of common derived aliases are
recognised by :meth:`Unit.parse`.
"""

from __future__ import annotations

import re as _re
from fractions import Fraction as _Fraction
from typing import Dict as _Dict
from typing import Union as _Union

_Number = _Union[int, float, _Fraction]

# Seven SI base dimensions.  Order fixes the pretty-print order.
_SI_BASES = ("kg", "m", "s", "A", "K", "mol", "cd")

# Superscript digits for pretty exponents (e.g. m·s⁻²).
_SUPERSCRIPT = {
    "-": "⁻",
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "/": "ᐟ",
    ".": "·",
}


class Unit:
    """A physical unit as a map of SI base dimension → rational exponent.

    Instances are immutable value objects: equality and hashing are by the
    (zero-stripped) exponent map, so ``Unit.parse("m/s") == Unit.parse("m s^-1")``.
    """

    __slots__ = ("_exp",)

    def __init__(self, exponents: _Dict[str, _Number] | None = None):
        exp: _Dict[str, _Fraction] = {}
        for base, power in (exponents or {}).items():
            f = _Fraction(power)
            if f != 0:
                exp[base] = f
        self._exp = exp

    # -- construction ---------------------------------------------------------
    @classmethod
    def parse(cls, spec: "str | Unit") -> "Unit":
        """Parse a unit string such as ``'m'``, ``'m/s'``, ``'kg/m^3'``, ``'Pa'``.

        Grammar (informal): a ``numerator`` optionally followed by ``/`` and a
        ``denominator``; each is a juxtaposition of ``base[^power]`` factors
        separated by spaces, ``*``, or ``·``.  Powers may be negative or
        fractional (``m^1/2``).  Recognises the 7 SI bases plus the aliases in
        :data:`_ALIASES` (``T``→K, ``Pa``, ``N``, ``J``, ``W``, ``Hz``, …).
        Dimensionless is spelled ``''``, ``'1'``, or ``'-'``.
        """
        if isinstance(spec, Unit):
            return spec
        # Accept Python-style '**' as well as '^' for exponents.
        text = spec.strip().replace("**", "^")
        if text in ("", "1", "-", "dimensionless"):
            return DIMENSIONLESS

        # Split a single top-level '/' into numerator / denominator.
        if "/" in text:
            num, _, den = text.partition("/")
            return cls._parse_product(num) / cls._parse_product(den)
        return cls._parse_product(text)

    @classmethod
    def _parse_product(cls, text: str) -> "Unit":
        text = text.strip().strip("()")
        if text in ("", "1"):
            return DIMENSIONLESS
        result = DIMENSIONLESS
        # factors separated by space, '*' or '·'
        for factor in _re.split(r"[\s*·]+", text):
            if not factor:
                continue
            result = result * cls._parse_factor(factor)
        return result

    @classmethod
    def _parse_factor(cls, factor: str) -> "Unit":
        m = _re.fullmatch(r"([A-Za-zµ]+)(?:\^?(-?\d+(?:/\d+)?))?", factor.strip())
        if m is None:
            raise ValueError(f"Cannot parse unit factor {factor!r}")
        symbol, power = m.group(1), m.group(2)
        exponent = _Fraction(power) if power is not None else _Fraction(1)
        base = _ALIASES.get(symbol)
        if base is None:
            if symbol in _SI_BASES:
                base = Unit({symbol: 1})
            else:
                raise ValueError(f"Unknown unit symbol {symbol!r}")
        return base**exponent

    # -- algebra --------------------------------------------------------------
    def __mul__(self, other: "Unit") -> "Unit":
        exp = dict(self._exp)
        for base, power in other._exp.items():
            exp[base] = exp.get(base, _Fraction(0)) + power
        return Unit(exp)

    def __truediv__(self, other: "Unit") -> "Unit":
        exp = dict(self._exp)
        for base, power in other._exp.items():
            exp[base] = exp.get(base, _Fraction(0)) - power
        return Unit(exp)

    def __pow__(self, n: _Number) -> "Unit":
        f = _Fraction(n).limit_denominator(1_000_000)
        return Unit({base: power * f for base, power in self._exp.items()})

    # -- queries --------------------------------------------------------------
    def is_dimensionless(self) -> bool:
        return not self._exp

    def __eq__(self, other) -> bool:
        if not isinstance(other, Unit):
            return NotImplemented
        return self._exp == other._exp

    def __hash__(self) -> int:
        return hash(frozenset(self._exp.items()))

    # -- display --------------------------------------------------------------
    def __repr__(self) -> str:
        if not self._exp:
            return "1"
        order = list(_SI_BASES) + sorted(b for b in self._exp if b not in _SI_BASES)
        # Positive exponents first, then negative — the conventional reading
        # (e.g. K·m⁻¹ rather than m⁻¹·K), each group in SI-base order.
        positive = [b for b in order if self._exp.get(b, 0) > 0]
        negative = [b for b in order if self._exp.get(b, 0) < 0]
        parts = [self._format_factor(b, self._exp[b]) for b in positive + negative]
        return "·".join(parts)

    @staticmethod
    def _format_factor(base: str, power: _Fraction) -> str:
        if power == 1:
            return base
        sup = "".join(_SUPERSCRIPT.get(ch, ch) for ch in str(power))
        return f"{base}{sup}"


DIMENSIONLESS = Unit({})

# Derived-unit and convenience aliases → concrete Unit objects.
# Defined after the class so the algebra is available.
_ALIASES: _Dict[str, Unit] = {
    # SI bases (identity)
    **{b: Unit({b: 1}) for b in _SI_BASES},
    # common temperature / angle spellings
    "T": Unit({"K": 1}),  # temperature shorthand used in PDE write-ups
    "rad": DIMENSIONLESS,  # plane angle is dimensionless
    "sr": DIMENSIONLESS,  # solid angle is dimensionless
    "g": Unit({"kg": 1}),  # treat gram as a mass dimension (prefix ignored)
    # named derived units
    "N": Unit({"kg": 1, "m": 1, "s": -2}),  # newton
    "Pa": Unit({"kg": 1, "m": -1, "s": -2}),  # pascal
    "J": Unit({"kg": 1, "m": 2, "s": -2}),  # joule
    "W": Unit({"kg": 1, "m": 2, "s": -3}),  # watt
    "Hz": Unit({"s": -1}),  # hertz
    "C": Unit({"A": 1, "s": 1}),  # coulomb
    "V": Unit({"kg": 1, "m": 2, "s": -3, "A": -1}),  # volt
    "Ohm": Unit({"kg": 1, "m": 2, "s": -3, "A": -2}),  # ohm
}

# Public entry points live in unit_log; re-export here so the whole feature is
# reachable as ``jno.units.*`` (``jno.units.check``, ``jno.units.Unit``, …).
# Imported last so the algebra above is fully defined before unit_log loads it.
from .unit_log import NondimReport, UnitLogger, check, infer, nondimensionalize  # noqa: E402

__all__ = ["Unit", "DIMENSIONLESS", "UnitLogger", "check", "infer", "nondimensionalize", "NondimReport"]


def __dir__():
    # Present only the curated public surface as ``jno.units.*`` (PEP 562):
    # hides helper imports and the ``from __future__`` ``annotations`` artifact.
    return list(__all__)
