"""The developer stage markers: `jno.utils.profiling.stage` names the traced operations of a solve stage in the
compiled program (forward and transpose) without changing the result, and the backend-statistics collector is a
no-op until it is switched on."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils import profiling as P

jax.config.update("jax_enable_x64", True)


def test_stage_names_survive_jit_forward_and_transpose():
    def f(a):
        with P.stage("assemble"):
            K = jnp.sin(a) @ a.T
        with P.stage("solve"):
            return jnp.sum(jnp.linalg.solve(K + 100.0 * jnp.eye(8), jnp.ones(8)) ** 2)

    a = jnp.linspace(0, 1, 64).reshape(8, 8)
    hlo = jax.jit(jax.value_and_grad(f)).lower(a).compile().as_text()
    assert "jvp(jno/assemble)" in hlo and "jvp(jno/solve)" in hlo
    assert "transpose(jvp(jno/assemble))" in hlo and "transpose(jvp(jno/solve))" in hlo
    v, g = jax.jit(jax.value_and_grad(f))(a)
    v0, g0 = jax.value_and_grad(
        lambda a: jnp.sum(jnp.linalg.solve(jnp.sin(a) @ a.T + 100.0 * jnp.eye(8), jnp.ones(8)) ** 2)
    )(a)
    np.testing.assert_allclose(v, v0, rtol=1e-13)
    np.testing.assert_allclose(g, g0, rtol=1e-12)


def test_fem_assembly_is_marked():
    import jno

    d = jno.shape.rect(0, 0, 1, 1, size=0.2).domain()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi.trainable(name="X")
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    op = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0]).operator
    assert hasattr(op, "evaluate"), "the operator must be parametric in the coordinates for this test"
    x0 = jnp.asarray(d.mesh.points[:, 0])
    hlo = jax.jit(lambda x: op.evaluate({"X": x})[0].data).lower(x0).compile().as_text()
    assert "jno/assemble" in hlo


def test_stats_collector_off_by_default_then_records():
    P.collect_stats(False)
    P.record(backend="x", n=1)
    assert P.stats() == []
    P.collect_stats(True)
    P.record(backend="x", n=1)
    P.record(backend="y", n=2)
    assert P.stats() == [{"backend": "x", "n": 1}, {"backend": "y", "n": 2}]
    P.collect_stats(False)
    assert P.stats() == []


@pytest.mark.skipif(
    not __import__("jno.utils.solver.cudss_sym", fromlist=["available"]).available(), reason="cuDSS FFI not built"
)
def test_cudss_sym_stages_in_hlo():
    import jno.utils.solver.cudss_sym as cs

    n = 50
    i = np.arange(n)
    rows = np.concatenate([i, i[:-1], i[1:]])
    cols = np.concatenate([i, i[1:], i[:-1]])
    p = cs.plan(rows, cols, n)
    vals0 = jnp.concatenate([jnp.full(n, 4.0), jnp.full(n - 1, -1.0), jnp.full(n - 1, -1.0)])

    def f(vals):
        return jnp.sum(cs.solve(p, vals, jnp.ones(n)) ** 2)

    hlo = jax.jit(jax.value_and_grad(f)).lower(vals0).compile().as_text()
    paths = set(re.findall(r'op_name="([^"]*jno/[^"]*)"', hlo))
    assert any("jvp(jno/factor)" in q for q in paths)
    # inside custom_linear_solve the scope is nested as jvp()/jno/solve (forward) and transpose(jvp())/jno/solve
    assert any("jno/solve" in q and "transpose(" not in q for q in paths)
    assert any("jno/solve" in q and "transpose(" in q for q in paths)
