"""``crux.eval([e1, e2, ...])`` evaluates the list as one program per sample-batch size: what the expressions share (a
frozen field's solve) is computed once, not once per expression. Before, a list mixing a per-vertex
expression with integrals and per-cell fields fell back to one program per expression and repeated every shared solve.
"""

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
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    u, phi = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    k = jno.np.parameter(phi, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(jno.optimizers.mma(move=0.05, lower=0.5, upper=2.0))
    sol = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0]).solve()
    mean_u = ui.freeze(sol).integrate(d)
    traced = []

    def shared(v):  # a subexpression every list entry below reads: traced once per program that evaluates it
        traced.append(1)
        return 1.0 + jnp.sum(v)

    s = jno.fn(shared, [k.mean], name="shared")
    ops = [
        s * mean_u,
        s * k,
        s * k.mean,
        s * d.cell_volume(),
        s * (x * y).integrate(d),
        s * xb * 2.0,
        s * x,
    ]
    crux = jno.core([mean_u, jno.le(k.mean, 1.2).name("k mean")])
    crux.solve(2)
    return crux, ops, traced


def test_a_list_gives_each_expression_its_value_and_shape_alone():
    crux, ops, _ = _problem()
    together = crux.eval(ops)
    for i, op in enumerate(ops):
        alone = np.asarray(crux.eval([op]))
        got = np.asarray(together[i])
        assert got.shape == alone.shape, (i, got.shape, alone.shape)
        np.testing.assert_allclose(got, alone, rtol=1e-12, atol=1e-14)


def test_what_a_list_shares_is_traced_once_per_sample_batch_size():
    crux, ops, traced = _problem()
    traced.clear()
    crux.eval(ops)
    # every sampled entry has the same sample batch and the others join it: one program
    assert len(traced) == 1, f"the shared subexpression was traced {len(traced)} times for one eval of 7"
