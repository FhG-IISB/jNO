"""Shared factorisations on any solver (CPU): forms that differ only in their load are solved as one block of
right-hand sides on the first form's solver -- host SuperLU and the default solver here. Oracle: the same
values with sharing switched off."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import jno
from jno.utils.solver.linear import _pardiso_available

inner = jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _three_loads(linear, advect=0.0):
    """three loads on one operator; ``advect`` adds a convection term (a non-symmetric operator)"""
    d = jno.shape.rect(0.0, 0.0, 2.0, 1.0, size=0.15).domain()
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    k = jno.np.parameter(s, name="k")
    k.dtype(jnp.float64)
    k.initialize(lambda key, shape, dtype=None: jnp.asarray(np.random.default_rng(0).uniform(0.5, 2.0, shape)))
    U, V = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    grad = lambda f: jno.np.grad(f, [xi, yi])  # noqa: E731
    loads = [1.0 + 0.0 * xi, xi * yi, jno.np.sin(3.0 * xi)]
    fems = [jno.fem([k * inner(grad(U), grad(V)) + advect * grad(U)[..., 0] * V - f * V, u(xb, yb) - 0.0]) for f in loads]
    sols = [f.solve(linear=linear) for f in fems]
    F = jno.fn(lambda a, b, c: jnp.stack([jnp.sum(a * a), jnp.sum(a * b), jnp.sum(c)]), sols, name="F")
    return F, jno.core([F], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))


@pytest.mark.parametrize("linear", [jno.solve.lu(backend="host"), None], ids=["host-superlu", "default"])
def test_three_loads_are_solved_as_one_block(monkeypatch, linear):
    import jno.utils.solver.shared_solve as S

    grouped = []
    orig = S.solve_shared
    monkeypatch.setattr(S, "solve_shared", lambda *a, **kw: grouped.append(orig(*a, **kw)) or grouped[-1])
    F, crux = _three_loads(linear)
    got = np.asarray(crux.eval([F])).reshape(-1)
    assert grouped and grouped[0], "the three forms were not solved as one group"
    monkeypatch.setattr(S, "solve_shared", lambda *a, **kw: False)
    F2, crux2 = _three_loads(linear)
    want = np.asarray(crux2.eval([F2])).reshape(-1)
    np.testing.assert_allclose(got, want, rtol=1e-8)


@pytest.mark.parametrize("backend", ["host", "pardiso"])
def test_the_gradient_through_a_shared_group_is_that_of_separate_solves(monkeypatch, backend):
    if backend == "pardiso" and not _pardiso_available():
        pytest.skip("needs pypardiso (MKL)")
    import equinox as eqx

    import jno.utils.solver.shared_solve as S
    from jno.trace_compiler import TraceCompiler

    def grads():
        F, crux = _three_loads(jno.solve.lu(backend=backend))
        fn = TraceCompiler.compile_traced_expression(F, crux.all_ops)
        params, static = eqx.partition(eqx.tree_inference(crux._unwrapped_models), eqx.is_inexact_array)
        g = jax.grad(lambda p: jnp.sum(fn(eqx.combine(p, static), crux.domain_data.context)))(params)
        return [np.asarray(x) for x in jax.tree_util.tree_leaves(g)]

    shared = grads()
    monkeypatch.setattr(S, "solve_shared", lambda *a, **kw: False)
    alone = grads()
    for a, b in zip(shared, alone):
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12 * (np.abs(b).max() + 1e-300))
    assert any(np.abs(b).max() > 0 for b in alone)


@pytest.mark.skipif(not _pardiso_available(), reason="needs pypardiso (MKL)")
def test_pardiso_factors_the_group_once(monkeypatch):
    """PARDISO gets the group's block of right-hand sides in ONE call (one factorisation), not one per load"""
    import jno.utils.solver.linear as L

    calls = []
    orig = L._pardiso_host_solve
    monkeypatch.setattr(L, "_pardiso_host_solve", lambda d, i, r, shape, t: calls.append(np.shape(r)) or orig(d, i, r, shape, t))
    F, crux = _three_loads(jno.solve.lu(backend="pardiso"))
    got = np.asarray(crux.eval([F])).reshape(-1)
    assert len(calls) == 1 and calls[0][-1] == 3, f"expected one PARDISO call with a block of 3: {calls}"
    import jno.utils.solver.shared_solve as S

    monkeypatch.setattr(S, "solve_shared", lambda *a, **kw: False)
    F2, crux2 = _three_loads(jno.solve.lu(backend="pardiso"))
    np.testing.assert_allclose(got, np.asarray(crux2.eval([F2])).reshape(-1), rtol=1e-8)
