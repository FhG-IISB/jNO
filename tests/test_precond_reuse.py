"""``reuse=``: one keyword, on every preconditioner, for keeping what its setup built across solves.

``jno.solve.lu(reuse=...)`` keeps a factorization and ``jno.solve.newton(reuse=...)`` a tangent, and a
preconditioner says it the same way: ``reuse=True`` freezes the setup after the first solve, ``reuse=k``
rebuilds it every k-th solve, ``reuse=False`` (the default) rebuilds every solve.

Oracle: how many times pyamg's setup runs over a sequence of solves.
"""

from __future__ import annotations

import inspect

import jax
import numpy as np
import pytest

import jno

pyamg = pytest.importorskip("pyamg")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson():
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([ub.x * vb.x + ub.y * vb.y - vb, u(xb, yb) - 0.0])


def _setups(monkeypatch, precond, n_solves=4):
    """pyamg setups over ``n_solves`` solves of one linear problem, and the solutions."""
    count = [0]
    real = pyamg.smoothed_aggregation_solver

    def counting(*a, **kw):
        count[0] += 1
        return real(*a, **kw)

    monkeypatch.setattr(pyamg, "smoothed_aggregation_solver", counting)
    fem = _poisson()
    sols = [np.asarray(fem.solve(linear=jno.solve.cg(), precond=precond)) for _ in range(n_solves)]
    return count[0], sols


@pytest.mark.parametrize("reuse, setups", [(False, 4), (True, 1), (2, 2)])
def test_reuse_decides_how_often_the_setup_runs(monkeypatch, reuse, setups):
    # Break: ignoring reuse= (4 setups always), or freezing without being asked (1 setup for reuse=False).
    n, sols = _setups(monkeypatch, jno.precond.amg(reuse=reuse))
    assert n == setups, f"reuse={reuse!r}: {n} setups over 4 solves"
    for s in sols[1:]:  # the preconditioner changes the iterations, never the answer
        np.testing.assert_allclose(s, sols[0], rtol=0, atol=1e-7 * np.abs(sols[0]).max())


def test_every_preconditioner_constructor_takes_reuse():
    # Break: a constructor without the keyword -- "one way of writing it" would then have exceptions.
    names = [n for n in jno.precond.__all__ if callable(getattr(jno.precond, n))]
    missing = [n for n in names if inspect.isfunction(getattr(jno.precond, n))
               and "reuse" not in inspect.signature(getattr(jno.precond, n)).parameters]  # fmt: skip
    assert not missing, f"constructors without reuse=: {missing}"


@pytest.mark.parametrize("bad", [-1, 0.5, "yes"])
def test_a_reuse_value_that_means_nothing_raises_by_name(bad):
    with pytest.raises(ValueError, match="reuse="):
        jno.precond.amg(reuse=bad)


def test_there_is_one_way_to_write_it():
    # Break: bringing back a second spelling (`.cached()` / `jno.precond.cached()`).
    assert not hasattr(jno.precond, "cached") and "cached" not in jno.precond.__all__
    assert not hasattr(jno.precond.amg(), "cached")
