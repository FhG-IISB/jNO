"""A preconditioner built on one operator applies to another operator of the same problem.

A concrete operator drops its exact zeros, and a constant coefficient cancels some couplings exactly on a
mesh, so the operator of a uniform design or an intact plate stores fewer entries than one whose coefficient
varies. Built on the first, the Schwarz and coarse tables read the second at the problem's STRUCTURAL pattern
(:func:`deflation.structural_entries`); a nonzero entry outside the pattern still fails loudly.

Oracles: the stale spec's matrix on the varying operator equals a fresh spec's (the structural pattern gives
both the same partition); :func:`fsai.pattern_values` against a dense lookup.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

pytest.importorskip("pymetis", reason="Schwarz subdomains use METIS: the [metis] extra")

R = jno.shape.rect


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def plate():
    jax.config.update("jax_enable_x64", True)
    d = R(0.0, 0.0, 1.0, 1.0).sized(0.08).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    ε = lambda w: jno.np.symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    tr = jno.np.trace

    def problem(g):
        u, v = d.fem_symbols(value_shape=(2,))
        u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)
        return jno.fem([g * (2.0 * ddot(ε(u_), ε(v_)) + 1.5 * tr(ε(u_)) * tr(ε(v_))) - 0.1 * v_[1], u(xb, yb) - (0.0, 0.0)])

    return problem(1.0), problem(1.0 + x * y), (x, y)


def _matrix(spec, fem):
    spec.prepare(fem)
    A = fem._op[0]
    ap = materialize_precond(spec, PrecondContext(LinearOperator(A), fem))
    return np.asarray(jax.jit(jax.vmap(ap, in_axes=1, out_axes=1))(jnp.eye(A.shape[0])))


@pytest.mark.parametrize("kind", ["schwarz", "coarse"])
def test_a_spec_built_on_a_cancelling_operator_serves_a_varying_one(plate, kind):
    uniform, varying, (x, y) = plate
    assert uniform._op[0].nse < varying._op[0].nse  # the constant coefficient cancels entries exactly

    def make():
        S = jno.precond.schwarz(parts=6, coarse=False)
        return S if kind == "schwarz" else jno.precond.coarse([(1, 0), (0, 1), (-y, x)], parts=S)

    stale = make()
    _matrix(stale, uniform)  # built on the operator with the fewer entries
    B_stale, B_fresh = _matrix(stale, varying), _matrix(make(), varying)
    assert np.all(np.isfinite(B_stale))
    np.testing.assert_allclose(B_stale, B_fresh, rtol=1e-12, atol=1e-12 * np.abs(B_fresh).max())


def test_pattern_values_reads_absent_keys_as_zero_and_refuses_a_stray_entry():
    from jno.utils.solver.fsai import pattern_values

    n = 4
    keys = jnp.asarray(np.array([0 * n + 0, 0 * n + 1, 1 * n + 1, 2 * n + 2, 3 * n + 3], np.int64))
    idx = np.array([[0, 0], [1, 1], [0, 0], [2, 2], [3, 3]])  # a duplicate, no (0, 1): read as 0
    A = jsparse.BCOO((jnp.asarray([1.0, 2.0, 0.5, 3.0, 4.0]), jnp.asarray(idx)), shape=(n, n))
    np.testing.assert_array_equal(np.asarray(pattern_values(keys, A, n)), [1.5, 0.0, 2.0, 3.0, 4.0])
    zero_out = jsparse.BCOO((jnp.asarray([1.0, 0.0]), jnp.asarray([[0, 0], [3, 0]])), shape=(n, n))
    assert np.all(np.isfinite(np.asarray(pattern_values(keys, zero_out, n))))  # an exact zero outside: harmless
    stray = jsparse.BCOO((jnp.asarray([1.0, 0.25]), jnp.asarray([[0, 0], [3, 0]])), shape=(n, n))
    assert np.all(np.isnan(np.asarray(jax.jit(lambda A: pattern_values(keys, A, n))(stray))))
