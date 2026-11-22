"""``jno.fvm`` inside jNO: gradients through a solve, ``jno.core`` inverse problems, ``jno.info``, and the
segregated saddle solvers ``jno.solve.simple`` / ``jno.solve.piso``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _poisson(scale_or_param, d):
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * PI**2 * jnn.sin(PI * x) * jnn.sin(PI * y)
    return jno.fvm([-(ui.xx + ui.yy) - scale_or_param * f, u(xb, yb) - 0.0]).solve()


def test_a_gradient_flows_through_the_solve():
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    observed = jnp.asarray(_poisson(1.0, d))
    loss = lambda s: jnp.mean((jnp.asarray(_poisson(s, d)) - observed) ** 2)  # noqa: E731
    g = float(jax.grad(loss)(1.5))
    # the solution is linear in s, so the loss is (s - 1)^2 mean(u1^2) and its slope at 1.5 is mean(u1^2)
    assert g == pytest.approx(float(jnp.mean(observed**2)), rel=1e-8)


def test_jno_core_recovers_a_source_amplitude():
    import optax

    from jno.trace import FunctionCall

    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    observed = jnp.asarray(_poisson(1.0, d))
    s = jno.np.parameter((1,), name="s")
    s.dtype(jnp.float64)
    s.initialize(jax.nn.initializers.constant(2.5))
    s.optimizer(optax.adam(1e-1))
    node = _poisson(s, d)
    assert isinstance(node, FunctionCall)
    crux = jno.core([(node - observed).mse])
    assert crux.domain is d
    crux.solve(120)
    assert float(np.asarray(crux.eval([s])).reshape(-1)[0]) == pytest.approx(1.0, abs=1e-2)


def test_info_reports_how_each_equation_was_read():
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    text = str(jno.info(jno.fvm([-(ui.xx + ui.yy) + ui.d(x, scheme=jno.fv()) - 1.0, u(xb, yb) - 0.0])))
    assert "fvm (finite volume)" in text
    assert "diffusive fluxes" in text and "convective fluxes" in text and "rusanov" in text


def _cavity(n, time=None):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).quad()
    d = d.domain(time=time) if time else d.domain()
    d.tag("lid", lambda x, y: y > 1 - 1e-9)
    d.tag("wall", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9))
    d.point_region("ppin", (0.0, 0.0))
    x, y, t = d.variable("interior", split=True)
    xl, yl, _ = d.variable("lid", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    xp, yp, _ = d.variable("ppin", split=True)
    U, P = d.unknown.vector(2), d.unknown()
    Ui, pi = (U.bind(x=x, y=y, t=t), P.bind(x=x, y=y, t=t)) if time else (U.bind(x=x, y=y), P.bind(x=x, y=y))
    u, v = Ui[0], Ui[1]
    s = jno.fv()
    mom = (u * Ui).d(x, scheme=s) + (v * Ui).d(y, scheme=s) - 0.01 * (Ui.xx + Ui.yy) + jnn.stack([pi.x, pi.y], axis=-1)
    cons = [Ui.t + mom if time else mom, Ui[0].x + Ui[1].y]
    cons += [U(xl, yl) - jnn.stack([1.0 + 0.0 * xl, 0.0 * xl], axis=-1), U(xw, yw) - 0.0, P(xp, yp) - 0.0]
    if time:
        xi, yi, _ = d.variable("initial", split=True)
        cons.append(U(xi, yi) - 0.0)
    return cons


def test_simple_reaches_the_newton_solution():
    cons = _cavity(12)
    a = np.asarray(jno.fvm(cons).solve())
    b = np.asarray(jno.fvm(cons).solve(nonlinear=jno.solve.simple(max_iters=300)))
    assert np.abs(a - b).max() < 1e-6


@pytest.mark.slow
def test_piso_marches_like_the_implicit_newton_step():
    cons = _cavity(10, time=(0.0, 0.1, 11))
    a = np.asarray(jno.fvm(cons).solve())
    b = np.asarray(jno.fvm(cons).solve(nonlinear=jno.solve.piso(n_corr=3)))
    assert np.abs(a[-1, :2] - b[-1, :2]).max() < 1e-6


def test_simple_needs_a_saddle_pair():
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.raises(ValueError, match="saddle"):
        jno.fvm([-(ui.xx + ui.yy) + ui**3 - 1.0, u(xb, yb) - 0.0]).solve(nonlinear=jno.solve.simple())
