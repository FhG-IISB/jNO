"""`jno.np.concat` broadcasts a bare temporal `t` onto the spatial point axis.

`dom.variable(tag)` returns x/y with a points axis but `t` as a *scalar* per time slice. Mixing
them in a concat raised nothing — the scalar was broadcast — but on a domain with two or more
spatial dimensions it produced a graph that never finished compiling, with no error to point at
(a PINN/DeepONet trunk `concat([x, y, t])` is the usual way in). concat now does the `t + 0*x`
alignment itself, so the natural spelling is also the fast one.
"""

import time

import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def test_bare_temporal_matches_manual_broadcast():
    """`concat([x, y, t])` == `concat([x, y, t + 0*x])` — the alignment is what users hand-wrote."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain(time=(0.0, 1.0, 3))
    x, y, t = d.variable("interior")

    bare = jno.np.concat([x, y, t], axis=-1)
    manual = jno.np.concat([x, y, t + 0.0 * x], axis=-1)

    core = jno.core([bare.mse], domain=d)
    got = core.eval(bare)
    want = core.eval(manual)
    assert got.shape == want.shape
    assert got.shape[-1] == 3  # x, y, t all present on the trailing axis


def test_bare_temporal_trunk_compiles_promptly():
    """The regression: a 2D spatiotemporal trunk with a bare `t` used to hang forever."""
    d = jno.Shape.rect(0.0, 0.0, 100.0, 100.0, size=25.0).domain(time=(0.0, 45.0, 4))
    x, y, t = d.variable("interior")

    net = jno.nn(foundax.mlp(in_features=3, output_dim=1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3))

    u = net(jno.np.concat([x, y, t], axis=-1)).scalar.bind(x=x, y=y, t=t)
    res = u.t - 0.1 * (u.xx + u.yy) + u

    t0 = time.time()
    jno.core([res.mse]).solve(epochs=2, batchsize=32)
    assert time.time() - t0 < 120.0, "bare-temporal trunk regressed to the pathological compile path"


def test_spatial_only_concat_unaffected():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    x, y, _ = d.variable("interior")
    out = jno.core([], domain=d).eval(jno.np.concat([x, y], axis=-1))
    assert jnp.asarray(out).shape[-1] == 2
