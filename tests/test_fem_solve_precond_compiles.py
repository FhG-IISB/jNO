"""A nonlinear solve with a ``precond=`` compiles once and is reused, when nothing in the preconditioner can
go stale.

Every ``fem.solve(precond=...)`` ran eagerly: the Newton ``while_loop`` and ``custom_root`` re-staged on
each call (measured: a 2.7k-DOF repeat solve 326 ms with ``precond=jacobi()``, 43 ms without). That is
right for a preconditioner holding state built from CONCRETE values -- a plain ``amg()`` is rebuilt per
solve, a Picard-lagged form is refreshed from the entry iterate -- but not for one built inside the trace
(``jacobi``) or frozen on request (``amg(reuse=True)``, which builds on the first solve).

Oracle: how often the Newton driver's Python body runs (once per TRACE when compiled, once per CALL when
eager), and the answers, which must not change.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _problem():
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    x, y = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, v = d.fem_symbols()
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([(1.0 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - 4.0 * vb, u(xb, yb) - 0.0])


def _driver_runs(monkeypatch, n_solves, **slots):
    import jno.utils.solver.newton_krylov as nk

    runs = [0]
    real = nk.newton_direct

    def counting(*a, **kw):
        runs[0] += 1
        return real(*a, **kw)

    monkeypatch.setattr(nk, "newton_direct", counting)
    fem = _problem()
    sols = [np.asarray(jax.block_until_ready(jnp.asarray(fem.solve(**slots)))) for _ in range(n_solves)]
    return runs[0], sols


def test_a_traceable_preconditioner_compiles_once(monkeypatch):
    # Break: keeping every precond= solve eager (the driver body then runs on every call).
    runs, sols = _driver_runs(monkeypatch, 4, linear=jno.solve.cg(), precond=jno.precond.jacobi())
    assert runs == 1, f"the Newton driver ran {runs} times for 4 solves"
    _, ref = _driver_runs(monkeypatch, 1)
    np.testing.assert_allclose(sols[-1], ref[0], rtol=0, atol=1e-6 * np.abs(ref[0]).max())  # both meet Newton's tol


def test_a_kept_amg_builds_eagerly_once_then_compiles(monkeypatch):
    pytest.importorskip("pyamg")
    # Break: compiling the first solve (the hierarchy needs a CONCRETE tangent) or never compiling.
    spec = jno.precond.amg(reuse=True)
    runs, sols = _driver_runs(monkeypatch, 4, linear=jno.solve.cg(), precond=spec)
    assert runs <= 2, f"the Newton driver ran {runs} times for 4 solves"  # eager build + one trace
    # the same compiled program on the same frozen hierarchy (GPU scatter-add order: round-off only)
    np.testing.assert_allclose(sols[3], sols[1], rtol=0, atol=1e-12 * np.abs(sols[1]).max())


def test_a_preconditioner_rebuilt_per_solve_stays_eager(monkeypatch):
    pytest.importorskip("pyamg")
    # Break: caching a plain amg(): its hierarchy would be frozen at the first solve, against its contract
    # ("(re)built at each solve").
    runs, _ = _driver_runs(monkeypatch, 3, linear=jno.solve.cg(), precond=jno.precond.amg())
    assert runs == 3
