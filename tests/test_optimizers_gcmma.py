"""GCMMA -- globally convergent MMA, the variant whose every step is feasible and a descent.

Plain MMA trusts its convex approximation. When a constraint curves faster than the approximation
does, the step it takes lands outside the feasible set. GCMMA (Svanberg 2002) evaluates the true
functions at the trial point and, wherever the approximation turned out optimistic, re-solves with a
larger convexity term ``rho`` until it is conservative. From a feasible start, every accepted iterate
is then feasible and the objective never rises.

The test problem has a closed-form optimum and a constraint curved enough to make plain MMA overshoot:

    minimise  sum_i c_i / x_i    subject to  sum_i x_i**4 <= R,   x in [xlo, xhi]

Stationarity gives ``c_i / x_i**2 = 4 lam x_i**3``, so ``x_i`` is proportional to ``c_i**(1/5)``,
scaled until the budget is spent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.optimizers.mma import gcmma_step, mma_subproblem
from jno.utils.adaptive.callbacks import Callback

P, R, XLO, XHI = 4, 3.0, 0.05, 5.0
# GCMMA accepts an approximation as conservative to within 1e-7 * max(1, |f_i|) (Svanberg's
# tolerance), so "feasible" and "never rises" hold to that, not to round-off. Plain MMA misses by
# five orders of magnitude more on this problem, which is what the tests separate.
ACCEPT = 1e-7


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _c(n=12, seed=0):
    return np.random.default_rng(seed).uniform(0.5, 4.0, n)


def _analytic(c):
    x = c ** (1.0 / (P + 1))
    x = x * (R / np.sum(x**P)) ** (1.0 / P)
    return x, float(np.sum(c / x))


class _Spec:
    """The subproblem parameters, as `_BlendedSpec` supplies them."""

    asy_init, asy_shrink, asy_grow, raa, dual_iters = 0.5, 0.7, 1.2, 1e-5, 400

    def __init__(self, n, move=0.2, max_inner=10):
        self.move = np.full(n, move)
        self.max_inner = max_inner


def _x0(n):
    """A feasible start well inside the constraint, so the first steps are long ones."""
    return np.full(n, 0.5 * (R / n) ** (1.0 / P))


def _march(c, step, iters):
    """Iterate ``step`` and record the objective and the constraint at every iterate."""
    n = c.size
    x = _x0(n)
    xmin, xmax = np.full(n, XLO), np.full(n, XHI)
    low = upp = xo1 = xo2 = None
    f_hist, g_hist, infos = [], [], []
    for k in range(iters):
        f, g = float(np.sum(c / x)), np.array([np.sum(x**P) - R])
        f_hist.append(f)
        g_hist.append(float(g[0]))
        df0, dg = -c / x**2, (P * x ** (P - 1))[None, :]
        xnew, low, upp, info = step(x, f, df0, g, dg, low, upp, xmin, xmax, xo1, xo2, k)
        infos.append(info)
        xo2, xo1 = xo1, x.copy()
        x = xnew
    return x, np.array(f_hist), np.array(g_hist), infos


def _gcmma(c, max_inner=10):
    spec = _Spec(c.size, max_inner=max_inner)

    def evaluate(xh):
        return float(np.sum(c / xh)), np.array([np.sum(xh**P) - R])

    def step(x, f, df0, g, dg, low, upp, xmin, xmax, xo1, xo2, k):
        return gcmma_step(x, f, df0, g, dg, low, upp, xmin, xmax, xo1, xo2, k, spec, evaluate)

    return step


def _mma(c):
    spec = _Spec(c.size)

    def step(x, f, df0, g, dg, low, upp, xmin, xmax, xo1, xo2, k):
        return (*mma_subproblem(x, f, df0, g, dg, low, upp, xmin, xmax, xo1, xo2, k, spec), None)

    return step


class TestStep:
    def test_plain_mma_overshoots_this_constraint(self):
        """The premise of the next test: without the inner loop, a step lands infeasible."""
        c = _c()
        _x, _f, g, _ = _march(c, _mma(c), 60)
        assert g.max() > 1e-2, "the problem must be curved enough to break plain MMA, or it tests nothing"

    def test_every_iterate_is_feasible_and_the_objective_never_rises(self):
        c = _c()
        _x, f, g, _ = _march(c, _gcmma(c), 60)
        assert g.max() <= ACCEPT, f"GCMMA left the feasible set: max g = {g.max():.3e}"
        assert np.all(np.diff(f) <= ACCEPT * f[:-1]), "a conservative approximation forbids an ascent"

    def test_reaches_the_analytic_optimum(self):
        c = _c()
        x_star, f_star = _analytic(c)
        assert np.all((x_star > XLO) & (x_star < XHI)), "the optimum must be interior to be a real test"
        x, _f, _g, _ = _march(c, _gcmma(c), 150)
        assert float(np.sum(c / x)) == pytest.approx(f_star, rel=1e-8)
        assert np.allclose(x, x_star, rtol=1e-6)

    def test_the_inner_loop_actually_runs(self):
        """Feasibility must come from re-solving, not from steps too short to overshoot."""
        c = _c()
        *_, infos = _march(c, _gcmma(c), 60)
        assert max(i["inner"] for i in infos) >= 1
        assert all(i["conservative"] for i in infos)

    def test_an_exhausted_inner_loop_is_reported(self):
        """With no re-solves allowed the step is plain MMA with GCMMA's rho, and must say it was optimistic."""
        c = _c()
        *_, infos = _march(c, _gcmma(c, max_inner=0), 60)
        assert not all(i["conservative"] for i in infos)
        assert all(i["inner"] == 0 for i in infos)


class _Rows(Callback):
    """Every step's row values. The training log keeps only every tenth or so, which skips the
    first few steps, where an overshoot happens."""

    def __init__(self):
        self.rows = []

    def on_before_update(self, *, individual_losses, **kw):
        self.rows.append(np.asarray(individual_losses, dtype=np.float64).reshape(-1))
        return None


class TestThroughCore:
    @staticmethod
    def _build(c, opt):
        n = c.size
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((n,), name="x")
        x.dtype(jnp.float64)
        x.initialize(jax.nn.initializers.constant(float(_x0(n)[0])))
        x.optimizer(opt)
        f0 = jno.fn(lambda xv: jnp.sum(jnp.asarray(c) / xv), [x], name="f0")
        g = jno.fn(lambda xv: jnp.sum(xv**P), [x], name="g")
        return d, x, f0, g

    @pytest.mark.parametrize("name", ["mma", "gcmma"])
    def test_only_gcmma_stays_feasible_end_to_end(self, name):
        """Through ``jno.core`` the values reach the inner loop from the trace, not from a closure."""
        c = _c()
        x_star, f_star = _analytic(c)
        d, x, f0, g = self._build(c, getattr(jno.optimizers, name)(move=0.2, lower=XLO, upper=XHI))
        rec = _Rows()
        crux = jno.core([f0, jno.le(g, R)], domain=d)
        crux.solve(150, callbacks=[rec])
        xf = np.asarray(crux.eval([x])).reshape(-1)
        rows = np.stack(rec.rows)  # (steps, 2): the objective, and g - R as `jno.le` hands it over

        assert float(np.sum(c / xf)) == pytest.approx(f_star, rel=1e-6), "both reach the optimum"
        assert np.allclose(xf, x_star, rtol=1e-5)
        if name == "mma":
            assert rows[:, 1].max() > 1e-2, "the premise: plain MMA overshoots here through core too"
        else:
            assert rows[:, 1].max() <= ACCEPT, f"GCMMA left the feasible set: max g = {rows[:, 1].max():.3e}"
            assert np.all(np.diff(rows[:, 0]) <= ACCEPT * rows[:-1, 0])

    @pytest.mark.parametrize("name", ["mma", "gcmma"])
    def test_the_logged_multiplier_is_the_kkt_multiplier(self, name):
        """Stationarity ``c_i / x_i**2 = 4 lam x_i**3`` gives ``lam = c_i / (4 x_i**5)`` at the optimum."""
        c = _c()
        x_star, _ = _analytic(c)
        lam_star = float(np.mean(c / (P * x_star ** (P + 1))))
        d, x, f0, g = self._build(c, getattr(jno.optimizers, name)(move=0.2, lower=XLO, upper=XHI))
        crux = jno.core([f0, jno.le(g, R)], domain=d)
        crux.solve(150)
        lam = crux.training_logs[-1]["multipliers"]
        assert lam[-1, 0] == pytest.approx(lam_star, rel=1e-4)

    def test_mixing_mma_and_gcmma_is_refused(self):
        """One joint subproblem cannot be both conservative and not."""
        c = _c(4)
        d, _x, f0, g = self._build(c, jno.optimizers.gcmma(move=0.2, lower=XLO, upper=XHI))
        y = jno.np.parameter((2,), name="y")
        y.optimizer(jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0))
        fy = jno.fn(lambda yv: jnp.sum(yv**2), [y], name="fy")
        with pytest.raises(ValueError, match="mix"):
            jno.core([f0 + fy, jno.le(g, R)], domain=d).solve(1)
