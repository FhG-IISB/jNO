"""Functions of a linear, constant-coefficient operator on a flat periodic boundary face.

``jno.np.sqrtm(L)`` with ``L`` a Python callable -- the operator written in strong form as a function of
the field, ``lambda w: k**2 * w + w.x.x + w.y.y`` -- returns a :class:`FaceOperator`. ``S ** p`` takes a
real power of it, and ``S(arg)`` applies it to a face trace of an unknown, giving a symbol that enters a
boundary term like any field (``inner(S(tE), tF)``, ``S(tE)[0].y``). The term stays LINEAR: the assembler
adds one dense block over the face's DOFs.

The operator is diagonal in the Fourier modes ``e_m(x) = exp(i k_m . x)`` of the periodic face,
``k_m = 2 pi m / P``, so ``f(L) u = sum_m f(sigma(k_m)) <e_m, u> e_m`` with ``sigma`` the operator's symbol.
The symbol is read off the callable by applying it to a plane wave (``.x -> i k_x``). This is the
exact non-reflecting (Dirichlet-to-Neumann) boundary of a periodic half-space: for the Helmholtz equation
``du/dn = i (k^2 + Delta_t)^(1/2) u`` (Givoli, "Non-reflecting boundary conditions", J. Comput. Phys. 94,
1991, sec. 3), and the vector Maxwell form follows the same way per TE/TM order.

Scope (each violation raises):
- the callable is linear with constant coefficients (numbers), built from ``w``, ``+ - *`` by numbers and
  tangential derivatives ``.x .y .z``; a derivative along the face normal, a product of two fields, a
  traced coefficient or a constant term are refused;
- the face is flat and normal to a coordinate axis, and the problem is periodic across it; a Bloch tie
  ``u(A) - phi u(B)`` shifts the face modes to ``exp(i (k_m + k_B) . x)`` (Floquet, oblique incidence);
- the block is a constant LINEAR operator in the spatial residual: steady, transient (real forms) and
  nonlinear forms carry it, never the mass;
- ``f`` is a real power, principal branch: ``sqrt`` of a negative symbol is ``+ i sqrt(|.|)``. That is the
  ``exp(-i w t)`` convention, where an outgoing wave is ``exp(+i k_z z)`` and an evanescent one decays;
- the modes are truncated at the face mesh's Nyquist limit, ``|m_a| <= P_a / (2 h)`` (logged).
"""

from __future__ import annotations

import itertools
import numbers

import numpy as np

from . import Placeholder, Variable


def _is_number(c) -> bool:
    if isinstance(c, Placeholder):
        return False
    if isinstance(c, numbers.Number):
        return True
    shape = getattr(c, "shape", None)
    return shape == () and np.issubdtype(np.asarray(c).dtype, np.number)


class _PlaneWave:
    """A plane wave ``sym * exp(i k . x)`` per Fourier mode: the probe that reads an operator's symbol."""

    def __init__(self, sym, k: dict):
        self.sym, self.k = sym, k

    def _d(self, axis: str):
        if axis not in self.k:
            raise ValueError(
                f"jno.np.sqrtm(operator): the operator takes a derivative .{axis} along the face normal; only "
                f"tangential derivatives ({', '.join('.' + a for a in self.k)}) are defined on the face."
            )
        return _PlaneWave(self.sym * 1j * self.k[axis], self.k)

    x = property(lambda self: self._d("x"))
    y = property(lambda self: self._d("y"))
    z = property(lambda self: self._d("z"))

    def _other(self, o, what):
        if isinstance(o, _PlaneWave):
            return o
        raise TypeError(
            f"jno.np.sqrtm(operator): the operator {what} a constant ({o!r}) to the field; it must be LINEAR in "
            "the field (every term carries w, e.g. `k**2 * w + w.x.x`)."
        )

    def __add__(self, o):
        return _PlaneWave(self.sym + self._other(o, "adds").sym, self.k)

    __radd__ = __add__

    def __sub__(self, o):
        return _PlaneWave(self.sym - self._other(o, "subtracts").sym, self.k)

    def __rsub__(self, o):
        return _PlaneWave(self._other(o, "subtracts").sym - self.sym, self.k)

    def __neg__(self):
        return _PlaneWave(-self.sym, self.k)

    def __mul__(self, o):
        if isinstance(o, _PlaneWave):
            raise TypeError("jno.np.sqrtm(operator): the operator multiplies the field by itself; it must be linear.")
        if not _is_number(o):
            raise TypeError(
                f"jno.np.sqrtm(operator): coefficient {o!r} is not a number. The operator must have CONSTANT "
                "coefficients on the face (a traced or space-varying coefficient has no Fourier symbol)."
            )
        return _PlaneWave(self.sym * o, self.k)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self * (1.0 / o)


class FaceOperator:
    """``f(L)`` for a linear constant-coefficient face operator ``L`` (see the module docstring).

    Args:
        fn: The operator in strong form, ``fn(w)`` -> ``L w``, linear in ``w`` (for example
            ``lambda w: k**2 * w + w.x.x``). It is evaluated on plane waves to read its symbol.
        power: The real power ``p`` in ``L ** p``; ``jno.np.sqrtm`` starts at ``0.5`` and
            ``S ** q`` multiplies it.
    """

    def __init__(self, fn, power: float = 1.0):
        if not callable(fn):
            raise TypeError(f"FaceOperator needs a callable operator, got {type(fn).__name__}.")
        self.fn, self.power = fn, float(power)

    def __pow__(self, p):
        if not _is_number(p) or np.iscomplexobj(p):
            raise TypeError(f"a face operator takes a REAL power, got {p!r}.")
        return FaceOperator(self.fn, self.power * float(p))

    def symbol(self, k: dict):
        """``sigma(k)`` per mode, ``k = {axis: (n_modes,) wavenumbers}`` over the face's tangential axes."""
        n = len(next(iter(k.values())))
        out = self.fn(_PlaneWave(np.ones(n, complex), k))
        if not isinstance(out, _PlaneWave):
            raise TypeError(
                "jno.np.sqrtm(operator): the callable must return the operator applied to its argument "
                f"(a linear expression in w), got {type(out).__name__}."
            )
        return out.sym

    def multiplier(self, k: dict):
        """``f(sigma(k)) = sigma(k) ** power``, principal branch; a zero symbol under a negative power raises."""
        s = np.asarray(self.symbol(k), complex)
        if self.power < 0 and np.any(np.abs(s) < 1e-12 * max(1.0, float(np.abs(s).max()))):
            bad = int(np.argmin(np.abs(s)))
            raise ValueError(
                "jno.np.sqrtm(operator) ** negative: the operator's symbol vanishes at Fourier mode "
                f"k = {[float(v[bad]) for v in k.values()]} (an order at grazing incidence, k_z = 0), so its inverse "
                "is infinite. Change the period or the wavelength by a little."
            )
        return np.power(s, self.power)

    def __call__(self, arg):
        return FaceOpField._create(self, arg)

    def __repr__(self):
        return f"FaceOperator(power={self.power:g})"


_COUNTER = itertools.count()


def _walk(node):
    from .._fem import _walk as w

    return w(node)


class FaceOpField(Variable):
    """``S(arg)``: a face operator applied to a face trace -- a placeholder the assembler fills per Fourier mode."""

    @classmethod
    def _create(cls, op: FaceOperator, arg):
        from ..jnp_ops import _u
        from . import TestFunction, TrialFunction

        view, arg = arg, _u(arg)  # a typed view (bound field, vector view) wraps the trace node
        if not isinstance(arg, Placeholder):
            raise TypeError(f"a face operator applies to a traced field expression, got {type(arg).__name__}.")
        nodes = list(_walk(arg))
        if any(isinstance(n, TestFunction) for n in nodes):
            raise TypeError("a face operator applies to the UNKNOWN's trace, not to the test function.")
        trials = {getattr(n, "field_key", None): n for n in nodes if isinstance(n, TrialFunction)}
        if len(trials) != 1:
            raise TypeError(f"a face operator applies to the trace of exactly ONE unknown; found {len(trials)}.")
        trial = next(iter(trials.values()))
        dom = getattr(trial, "_domain", None)
        if dom is None:
            raise ValueError("a face operator applies to the trace of an unknown made by `d.fem_symbols(...)`.")
        from .._fem import _region_and_support, _spatial_coord_vars

        # The face: from the trace's coordinates or normal when it names one, else (a `vec(...)` of
        # components drops the binding) from the boundary term the symbol is integrated in.
        region = None
        if _spatial_coord_vars(view) or _spatial_coord_vars(arg):
            support, region = _region_and_support(view, dom)
            if support != "boundary":
                raise ValueError(f"a face operator applies to a trace on a boundary FACE; this trace lives on {region!r}.")
        shape = tuple(getattr(trial, "value_shape", ()) or ())
        n = int(np.prod(shape)) if shape else 1
        base = dict(uid=next(_COUNTER), op=op, arg=arg, region=region, n=n, field_key=getattr(trial, "field_key", None))
        return cls._register(dom, base, (0, 0, 0))

    @classmethod
    def _register(cls, dom, base: dict, deriv: tuple, comp=None):
        key = f"faceop_{base['uid']}_{''.join(map(str, deriv))}"
        dom.__dict__.setdefault("_faceop_specs", {})[key] = {**base, "deriv": deriv}
        if key not in dom.context:  # placeholder so the Variable constructs; assembly packs the value
            dom.context[key] = np.zeros((1, base["n"]))
        dim = [0, base["n"]] if comp is None else [comp, comp + 1]
        v = cls(tag=key, dim=dim, domain=dom, axis="spatial")
        v._faceop_base, v._faceop_deriv, v._faceop_comp = base, deriv, comp
        return v

    def _d(self, axis: int):
        d = list(self._faceop_deriv)
        d[axis] += 1
        return FaceOpField._register(self._domain, self._faceop_base, tuple(d), self._faceop_comp)

    @property
    def x(self):
        """``d/dx`` of ``S(arg)``, taken per Fourier mode."""
        return self._d(0)

    @property
    def y(self):
        """``d/dy`` of ``S(arg)``, taken per Fourier mode."""
        return self._d(1)

    @property
    def z(self):
        """``d/dz`` of ``S(arg)``, taken per Fourier mode."""
        return self._d(2)

    def __getitem__(self, key):
        """``S(E)[i]``: component ``i`` (a per-point scalar), which keeps ``.x``/``.y`` -- so ``S(E).x[1]``
        and ``S(E)[1].x`` are the same symbol. (A plain index on a trace node would slice the points.)"""
        n = int(self._faceop_base["n"])
        if self._faceop_comp is None and isinstance(key, (int, np.integer)) and -n <= int(key) < n:
            return FaceOpField._register(self._domain, self._faceop_base, self._faceop_deriv, int(key) % n)
        raise IndexError(f"a face-operator symbol S(trace) takes one component index in [0, {n}); got {key!r}.")


def reads_faceop(node) -> bool:
    """Does a term read a face-operator symbol (a ``faceop_*`` Variable)?"""
    return any(isinstance(n, Variable) and str(getattr(n, "tag", "")).startswith("faceop_") for n in _walk(node))
