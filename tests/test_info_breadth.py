"""Breadth check for `jno.info`: every object type it claims to handle, in one parametrised test.

This began as an ad-hoc script and found six defects on its first run. It is a test now because the
failure mode it catches -- a handler that silently stops reporting once an attribute moves -- is
invisible to every other test in this file, each of which asserts on one object it builds itself.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

n = jno.np


def _rect(size=0.4, **kw):
    return jno.shape.rect(0, 0, 1, 1, size=size).domain(**kw)


def _fem(kind):
    d = _rect(time=(0.0, 0.5, 4)) if kind == "transient" else _rect()
    x, y, t = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    if kind == "transient":
        a, w = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
        c = d.variable("initial", split=True)
        return jno.fem([a.t * w + 0.1 * (a.x * w.x + a.y * w.y), u(b[0], b[1]) - 0.0,
                        u(c[0], c[1], c[2]) - n.sin(np.pi * c[0])])
    a, t_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
    base = a.x * t_.x + a.y * t_.y
    if kind == "linear":
        return jno.fem([base - 1.0 * t_, u(b[0], b[1]) - 0.0])
    if kind == "nonlinear":
        return jno.fem([base + (a * a * a) * t_ - 1.0 * t_, u(b[0], b[1]) - 0.0])
    if kind == "complex":
        return jno.fem([base - (4.0 + 0.5j) * a * t_, u(b[0], b[1]) - 0.0])
    raise ValueError(kind)


def _coupled():
    d = _rect()
    x, y, _ = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    T, s_ = d.fem_symbols(names=("T", "s"))
    C, r_ = d.fem_symbols(names=("C", "r"))
    Tb, sb = T.bind(x=x, y=y), s_.bind(x=x, y=y)
    Cb, rb = C.bind(x=x, y=y), r_.bind(x=x, y=y)
    return jno.fem([Tb.x * sb.x + Tb.y * sb.y - Cb * sb, Cb.x * rb.x + Cb.y * rb.y - Tb * rb,
                    T(b[0], b[1]) - 0.0, C(b[0], b[1]) - 1.0])


def _fdm(structured):
    sh = jno.shape.rect(0, 0, 1, 1, size=0.25)
    d = (sh.structured() if structured else sh).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=xi, y=yi)
    return jno.fdm([-ui.d2(xi) - ui.d2(yi) - 1.0, u(xb, yb) - 0.0])


def _expr(kind):
    d = _rect(time=(0.0, 1.0, 3)) if kind == "temporal" else _rect()
    x, y, t = d.variable("interior", split=True)
    u, v = d.fem_symbols()
    if kind == "temporal":
        return u.bind(x=x, y=y, t=t).t
    if kind == "weak":
        return u.bind(x=x, y=y).x * v.bind(x=x, y=y).x
    return n.sin(x) * n.cos(y)


CASES = [
    ("env", lambda: None),
    ("domain mesh-free plan", lambda: jno.shape.disk(0, 0, 1, size=0.3).domain()),
    ("domain lazy", lambda: _rect(size=0.5)),
    ("domain 3-D", lambda: jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()),
    ("domain time-dependent", lambda: _rect(time=(0.0, 1.0, 5))),
    ("domain structured", lambda: jno.shape.rect(0, 0, 1, 1, size=0.34).structured().domain()),
    ("shape primitive", lambda: jno.shape.rect(0, 0, 1, 1)),
    ("shape nested CSG", lambda: (jno.shape.rect(0, 0, 4, 4) - jno.shape.disk(2, 2, 1)) | jno.shape.disk(0, 0, 0.5)),
    ("shape regions + attach", lambda: jno.shape.rect(0, 0, 1, 1).name("a").attach(k=1.0)
                                       + jno.shape.rect(1, 0, 2, 1).name("b").attach(k=2.0)),
    ("shape 3-D", lambda: jno.shape.box(0, 0, 0, 1, 1, 1) - jno.shape.sphere(0.5, 0.5, 0.5, 0.2)),
    ("fem linear", lambda: _fem("linear")),
    ("fem nonlinear", lambda: _fem("nonlinear")),
    ("fem transient", lambda: _fem("transient")),
    ("fem complex", lambda: _fem("complex")),
    ("fem coupled", _coupled),
    ("fdm structured", lambda: _fdm(True)),
    ("fdm unstructured", lambda: _fdm(False)),
    ("solve.lu", jno.solve.lu),
    ("solve.cg", jno.solve.cg),
    ("solve.gmres", jno.solve.gmres),
    ("solve.bicgstab", jno.solve.bicgstab),
    ("solve.newton", jno.solve.newton),
    ("solve.picard", jno.solve.picard),
    ("solve.theta", lambda: jno.solve.theta(0.5)),
    ("precond.jacobi", jno.precond.jacobi),
    ("result 1d", lambda: np.linspace(0, 1, 10)),
    ("result 2d", lambda: np.ones((4, 3))),
    ("result scalar", lambda: np.array(3.0)),
    ("result nan/inf", lambda: np.array([1.0, np.nan, np.inf])),
    ("result empty", lambda: np.zeros(0)),
    ("result all zeros", lambda: np.zeros(5)),
    ("result complex", lambda: np.array([1 + 2j, 3 - 1j])),
    ("expr coordinate-only", lambda: _expr("plain")),
    ("expr weak-form term", lambda: _expr("weak")),
    ("expr temporal", lambda: _expr("temporal")),
    ("variable tuple", lambda: _rect().variable("boundary")),
    ("variable single", lambda: _rect().variable("interior", split=True)[0]),
]

# deep=True only differs for the forms and the expression tree.
DEEP = {"fem linear", "fem nonlinear", "fem transient", "fem complex", "fem coupled",
        "expr weak-form term"}


def _check(label, rep):
    filled = sum(1 for k, v in rep.as_dict().items() if k != "title" and v)
    assert filled, f"{label}: the report came back EMPTY — the handler's attributes have moved"
    assert len(str(rep).splitlines()) > 1


@pytest.mark.parametrize("label,fn", CASES, ids=[c[0] for c in CASES])
def test_info_reports_something(label, fn):
    _check(label, jno.info(fn()))


@pytest.mark.parametrize("label,fn", [c for c in CASES if c[0] in DEEP], ids=sorted(DEEP, key=[c[0] for c in CASES].index))
def test_info_deep_reports_something(label, fn):
    """`deep=True` is where the assembly-cost paths live (conditioning, sparsity, the tree walk)."""
    _check(label, jno.info(fn(), deep=True))
