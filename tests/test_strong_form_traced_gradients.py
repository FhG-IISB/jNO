"""``jax.grad`` through a NONLINEAR strong-form solve whose residual closes over the differentiated value.

The structural probes (is the residual affine in u? is its operator symmetric?) compare JVPs as Python numbers,
which a traced coefficient forbids: they raised ``ConcretizationTypeError``. Undecidable there, they now answer
"not affine" / "not symmetric", the safe side (Newton solves an affine residual in one step), and the gradient
matches a central difference.
"""

import jax
import jax.numpy as jnp
import jno.numpy as jnn
import numpy as np
import pytest

import jno

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _central(loss, s, h=1e-4):
    return (float(loss(s + h)) - float(loss(s - h))) / (2 * h)


def test_a_nonlinear_spectral_solve_differentiates_by_a_coefficient():
    d = jno.shape.rect(0, 0, 1, 1).domain()
    U = d.unknown(space="spectral", order=10)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = U.bind(x=x, y=y)

    def loss(s):
        sol = jno.spectral([-(u.xx + u.yy) + s * u * u * u - 10.0, U(xb, yb) - 0.0]).solve()
        return jnp.sum(jnp.asarray(sol).reshape(-1) ** 2)

    g = float(jax.grad(loss)(2.0))
    assert abs(g - _central(loss, 2.0)) < 1e-5 * abs(g)


def test_a_coupled_solve_with_a_spectral_side_differentiates_by_its_conductivity():
    import equinox as eqx
    import optax

    from jno._coupling.driver import CouplingSpec, couple

    k = jno.np.parameter((1,), name="ka")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(optax.adam(1e-2))
    k.model.module = eqx.tree_at(lambda m: m.value, k.model.module, jnp.asarray([1.0]))
    kk = jnn.reshape(k, ())

    da = jno.shape.rect(0.0, 0.0, 1.0, 1.0).domain()
    U = da.unknown(space="spectral", order=(12, 12))
    x, y, _ = da.variable("interior", split=True)
    cons = [-kk * (U.bind(x=x, y=y).xx + U.bind(x=x, y=y).yy)]
    for tag in ("left", "top", "bottom"):
        xt, yt, _ = da.variable(tag, split=True)
        cons.append(U(xt, yt) - 0.0)
    xi, yi, _ = da.variable("right", split=True)
    ua = U.bind(x=xi, y=yi)
    fa = ua.d(da.variable("right", normals=True))

    db = jno.shape.rect(1.0, 0.0, 2.0, 1.0, size=0.1).domain()
    w, v = db.fem_symbols()
    xs, ys, _ = db.variable("interior", split=True)
    xr, yr, _ = db.variable("right", split=True)
    xt, yt, _ = db.variable("top", split=True)
    xb, yb, _ = db.variable("bottom", split=True)
    wi, vi = w.bind(x=xs, y=ys), v.bind(x=xs, y=ys)
    xl, yl, _ = db.variable("left", split=True)
    ub = w.bind(x=xl, y=yl)
    fb = ub.d(db.variable("left", normals=True))
    pb = jno.fem([4.0 * (wi.x * vi.x + wi.y * vi.y), w(xr, yr) - jnn.sin(PI * yr), w(xt, yt) - 0.0, w(xb, yb) - 0.0])

    c = couple([jno.spectral(cons), pb], [ua - ub, kk * fa + 4.0 * fb], CouplingSpec(tol=1e-12, max_iter=80))
    lid = k.model.layer_id

    def t(ka):
        sa, _ = c.solve(params={lid: jnp.reshape(ka, (1,))})
        return jnp.sum(jnp.asarray(sa) ** 2)

    g = float(jax.grad(t)(1.0))
    assert abs(g - _central(t, 1.0)) < 1e-6 * abs(g)
