"""``jno.le(g, b).auglag(...)`` -- an inequality constraint any optimiser can honour.

The augmented Lagrangian (Powell-Hestenes-Rockafellar form, Nocedal & Wright, *Numerical
Optimization*, 2nd ed., Sec. 17.4) adds ``psi = (max(0, lam + mu g)**2 - lam**2) / (2 mu)`` to the
loss and, every ``every`` steps, sets ``lam <- max(0, lam + mu g)``. The tests use problems whose
optimum AND multiplier are known in closed form, so a wrongly scaled penalty (one averaged with the
objective rows, say) shows up as a wrong multiplier even where the point happens to come out right.

    minimise  sum_i (x_i - a_i)**2   subject to  sum_i x_i <= V

is a projection: ``x = a - (sum(a) - V)/n`` and ``lam = 2 (sum(a) - V)/n`` when the budget binds.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
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


A = np.array([1.0, 2.0, 3.0, 4.0])
N = A.size


def _projection(opt, sense="le", V=6.0, every=50, steps=600):
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    x = jno.np.parameter((N,), name="x")
    x.dtype(jnp.float64)
    x.initialize(jax.nn.initializers.constant(0.0))
    x.optimizer(opt)
    f0 = jno.fn(lambda v: jnp.sum((v - jnp.asarray(A)) ** 2), [x], name="f0")
    s = jno.fn(lambda v: jnp.sum(v), [x], name="sum_x")
    if sense == "le":
        c = jno.le(s, V).auglag(mu=10.0, every=every)
    else:  # sum(-x) >= -V is the same constraint written the other way round
        s = jno.fn(lambda v: -jnp.sum(v), [x], name="neg_sum_x")
        c = jno.ge(s, -V).auglag(mu=10.0, every=every)
    crux = jno.core([f0, c], domain=d)
    crux.solve(steps)
    return crux, x, c


def _lam(crux, c):
    return float(np.asarray(crux.eval([c.multiplier])).reshape(-1)[0])


@pytest.mark.parametrize("sense", ["le", "ge"])
def test_it_finds_the_projection_and_its_multiplier(sense):
    V = 6.0
    crux, x, c = _projection(jno.optimizers.ssbfgs(), sense=sense, V=V)
    xf = np.asarray(crux.eval([x])).reshape(-1)
    shift = (A.sum() - V) / N
    np.testing.assert_allclose(xf, A - shift, atol=1e-5)
    assert xf.sum() <= V + 1e-5, "feasible at the end, to the multiplier method's tolerance"
    assert _lam(crux, c) == pytest.approx(2.0 * shift, rel=1e-4), "the multiplier of THIS objective"


def test_it_works_with_a_first_order_optax_optimiser():
    """The point of the method: no constrained optimiser is needed.

    Adam at a fixed rate jitters about each inner minimum. Read at a single step, that jitter fed
    the multiplier update, kept mu growing (10 -> 1e4) and reset the multiplier to 0, and the design
    drifted to sum(x) = 2 against a budget of 6. The update now averages the violation over the
    second half of each window.
    """
    crux, x, c = _projection(optax.adam(5e-2), every=100, steps=1500)
    xf = np.asarray(crux.eval([x])).reshape(-1)
    shift = (A.sum() - 6.0) / N
    np.testing.assert_allclose(xf, A - shift, atol=1e-3)
    assert _lam(crux, c) == pytest.approx(2.0 * shift, rel=1e-3)


def test_an_inactive_constraint_keeps_a_zero_multiplier():
    crux, x, c = _projection(jno.optimizers.ssbfgs(), V=100.0)
    np.testing.assert_allclose(np.asarray(crux.eval([x])).reshape(-1), A, atol=1e-6)
    assert _lam(crux, c) == 0.0


def test_it_bounds_a_fem_inverse_problem():
    """Recover a Dirichlet value from data generated at g* = 2.5, under the bound g <= 2."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u(xi, yi), v(xi, yi)
    form = lambda gv: [ui.d(xi) * vi.d(xi) + ui.d(yi) * vi.d(yi) - 1.0 * vi, u(xb, yb) - gv]  # noqa: E731
    u_obs = np.asarray(jno.fem(form(2.5)).solve()).reshape(-1)  # no parameter in it: solved at once

    g = jno.np.parameter((1,), name="g")
    g.dtype(jnp.float64)
    g.initialize(jax.nn.initializers.constant(1.0))
    g.optimizer(jno.optimizers.ssbfgs())
    misfit = (jno.fem(form(g)).solve() - jnp.asarray(u_obs)).mse
    c = jno.le(g.mean, 2.0).auglag(mu=10.0, every=30)
    crux = jno.core([misfit, c])
    crux.solve(300)
    gf = float(np.asarray(crux.eval([g])).reshape(-1)[0])
    assert gf == pytest.approx(2.0, abs=1e-4), "the bound binds: the data pulls towards 2.5"
    assert _lam(crux, c) > 0.0


@pytest.mark.parametrize("write", ["le", "comparison", "comparison_reversed", "function"])
def test_a_comparison_node_is_the_same_constraint(write):
    """``(s <= V).auglag()`` and ``(V >= s).auglag()`` mean ``jno.le(s, V).auglag()``."""
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    x = jno.np.parameter((N,), name="x")
    x.dtype(jnp.float64)
    x.initialize(jax.nn.initializers.constant(0.0))
    x.optimizer(jno.optimizers.ssbfgs())
    f0 = jno.fn(lambda v: jnp.sum((v - jnp.asarray(A)) ** 2), [x], name="f0")
    s = jno.fn(lambda v: jnp.sum(v), [x], name="sum_x")
    c = {
        "le": lambda: jno.le(s, 6.0).auglag(every=50),
        "comparison": lambda: (s <= 6.0).auglag(every=50),
        "comparison_reversed": lambda: (6.0 >= s).auglag(every=50),
        "function": lambda: jno.auglag(s <= 6.0, every=50),
    }[write]()
    crux = jno.core([f0, c], domain=d)
    crux.solve(600)
    np.testing.assert_allclose(np.asarray(crux.eval([x])).reshape(-1), A - 1.0, atol=1e-5)


def test_a_strict_comparison_is_refused():
    x = jno.np.parameter((1,), name="x")
    with pytest.raises(ValueError, match="write `<=`"):
        (x.mean < 1.0).auglag()


@pytest.mark.parametrize("V", [6.0, 14.0])
def test_an_equality_binds_from_either_side_with_a_signed_multiplier(V):
    """``sum x == V``: the projection ``x = a - (sum(a) - V)/n`` whichever side ``V`` lies on.

    At V = 14 the inequality would be inactive; the equality pulls UP, with ``lam = -2``. A
    multiplier clipped at zero, as an inequality's is, cannot get there.
    """
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    x = jno.np.parameter((N,), name="x")
    x.dtype(jnp.float64)
    x.initialize(jax.nn.initializers.constant(0.0))
    x.optimizer(jno.optimizers.ssbfgs())
    f0 = jno.fn(lambda v: jnp.sum((v - jnp.asarray(A)) ** 2), [x], name="f0")
    s = jno.fn(lambda v: jnp.sum(v), [x], name="sum_x")
    c = jno.eq(s, V).auglag(mu=10.0, every=50)
    crux = jno.core([f0, c], domain=d)
    crux.solve(600)
    shift = (A.sum() - V) / N
    np.testing.assert_allclose(np.asarray(crux.eval([x])).reshape(-1), A - shift, atol=1e-5)
    assert _lam(crux, c) == pytest.approx(2.0 * shift, rel=1e-4)


def test_the_multiplier_is_logged_beside_its_constraint():
    crux, _x, c = _projection(jno.optimizers.ssbfgs())
    lam_log = crux.training_logs[-1]["multipliers"]
    assert lam_log.shape[1] == 1, "one column per constraint row"
    assert lam_log[-1, 0] == pytest.approx(_lam(crux, c), rel=1e-12)


class TestGuards:
    def test_a_plain_equality_is_refused(self):
        """MMA cannot take an equality; without .auglag() it would be evaluated and ignored."""
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((4,), name="x").optimizer(optax.adam(1e-2))
        f0 = jno.fn(lambda v: jnp.sum(v**2), [x], name="f0")
        vol = jno.fn(lambda v: jnp.sum(v), [x], name="vol")
        with pytest.raises(ValueError, match=r"jno\.eq\(g, b\)\.auglag\(\)"):
            jno.core([f0, jno.eq(vol, 1.0)], domain=d).solve(1)

    def test_mma_and_an_auglag_constraint_are_refused_together(self):
        """MMA would handle the row in its dual a second time."""
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((4,), name="x")
        x.optimizer(jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0))
        f0 = jno.fn(lambda v: jnp.sum(v**2), [x], name="f0")
        vol = jno.fn(lambda v: jnp.sum(v), [x], name="vol")
        with pytest.raises(ValueError, match="auglag"):
            jno.core([f0, jno.le(vol, 1.0).auglag()], domain=d).solve(1)

    def test_a_plain_constraint_without_a_constrained_optimiser_points_to_auglag(self):
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((4,), name="x").optimizer(optax.adam(1e-2))
        f0 = jno.fn(lambda v: jnp.sum(v**2), [x], name="f0")
        vol = jno.fn(lambda v: jnp.sum(v), [x], name="vol")
        with pytest.raises(ValueError, match=r"\.auglag\(\)"):
            jno.core([f0, jno.le(vol, 1.0)], domain=d).solve(1)

    def test_jno_auglag_refuses_what_is_not_a_constraint(self):
        x = jno.np.parameter((1,), name="x")
        with pytest.raises(TypeError, match="jno.auglag takes"):
            jno.auglag(1.0)
        with pytest.raises(TypeError, match="comparison"):
            jno.auglag(jno.np.sin(x.mean))

    @pytest.mark.parametrize("kw", [{"mu": 0.0}, {"mu": -1.0}, {"every": 0}])
    def test_nonsense_settings_are_refused(self, kw):
        x = jno.np.parameter((1,), name="x")
        with pytest.raises(ValueError):
            jno.le(x.mean, 1.0).auglag(**kw)
