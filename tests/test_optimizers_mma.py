"""MMA — the Method of Moving Asymptotes as a constrained optimiser for ``jno.core``.

The subproblem is checked against a problem whose optimum is available in closed form, so these
are exactness tests rather than "it ran" tests:

    minimise  sum_i c_i / x_i    subject to  sum_i x_i <= V,   x in [xlo, xhi]

Stationarity of the Lagrangian gives ``x_i`` proportional to ``sqrt(c_i)``, scaled so the budget is
exactly spent. Both the optimal point and the optimal value are therefore known, which is what
makes this able to catch a wrong asymptote update or a sloppy dual solve — neither of which a
monotone-decrease assertion would notice.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.optimizers.mma import mma_subproblem


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


class _Spec:
    """The subproblem parameters, as `_BlendedSpec` supplies them."""

    asy_init, asy_shrink, asy_grow, raa, dual_iters = 0.5, 0.7, 1.2, 1e-5, 400

    def __init__(self, n, move=0.2):
        self.move = np.full(n, move)


def _problem(n=12, seed=0):
    c = np.random.default_rng(seed).uniform(0.5, 4.0, n)
    return c


def _analytic(c, V):
    x = np.sqrt(c)
    x = x / x.sum() * V
    return x, float(np.sum(c / x))


def _run(c, V, xlo, xhi, extra=None, iters=150):
    """Iterate the subproblem to convergence. ``extra`` adds a second constraint ``(vec, budget)``."""
    n = c.size
    x = np.full(n, V / n)
    xmin, xmax = np.full(n, xlo), np.full(n, xhi)
    low = upp = xo1 = xo2 = None
    spec = _Spec(n)
    for k in range(iters):
        df0 = -c / x**2
        if extra is None:
            g, dg = np.array([x.sum() - V]), np.ones((1, n))
        else:
            vec, budget = extra
            g = np.array([x.sum() - V, float(vec @ x) - budget])
            dg = np.stack([np.ones(n), vec])
        xnew, low, upp = mma_subproblem(x, None, df0, g, dg, low, upp, xmin, xmax, xo1, xo2, k, spec)
        # Shift the history BEFORE overwriting x — `xold1` is the previous iterate, and feeding it
        # the new one makes every asymptote update read a zero step.
        xo2, xo1 = xo1, x.copy()
        x = xnew
    return x


class TestSubproblem:
    def test_reaches_the_analytic_optimum(self):
        c, V, xlo, xhi = _problem(), 6.0, 0.05, 5.0
        x_star, f_star = _analytic(c, V)
        assert np.all((x_star > xlo) & (x_star < xhi)), "the optimum must be interior to be a real test"

        x = _run(c, V, xlo, xhi)
        f = float(np.sum(c / x))
        assert f == pytest.approx(f_star, rel=1e-8), "MMA must find the optimal VALUE"
        assert np.allclose(x, x_star, rtol=1e-6), "and the optimal POINT"

    def test_the_constraint_ends_active_and_satisfied(self):
        c, V = _problem(), 6.0
        x = _run(c, V, 0.05, 5.0)
        assert x.sum() <= V + 1e-9, "never infeasible"
        assert x.sum() == pytest.approx(V, abs=1e-8), "a budget worth spending is spent exactly"

    def test_the_box_is_never_violated(self):
        """A tight box makes it bind, which is the case a move limit can most easily overshoot."""
        c, V, xlo, xhi = _problem(), 6.0, 0.4, 0.6
        x = _run(c, V, xlo, xhi)
        assert x.min() >= xlo - 1e-12 and x.max() <= xhi + 1e-12
        assert np.any(np.isclose(x, xlo)) or np.any(np.isclose(x, xhi)), "the box should bind here"

    def test_two_binding_constraints(self):
        """Exercises the cyclic-bisection dual rather than the scalar bisection."""
        c, V, n = _problem(), 6.0, 12
        vec = np.zeros(n)
        vec[: n // 2] = 1.0
        budget = 2.0  # tight enough that this one genuinely binds
        x = _run(c, V, 0.05, 5.0, extra=(vec, budget), iters=250)
        g1, g2 = x.sum() - V, float(vec @ x) - budget
        assert g1 <= 1e-7 and g2 <= 1e-7, f"both must be feasible: g=({g1:.2e}, {g2:.2e})"
        assert g2 == pytest.approx(0.0, abs=1e-6), "the tight budget must end active"

    def test_an_unconstrained_subproblem_still_descends(self):
        c = _problem()
        n = c.size
        x = np.full(n, 0.5)
        f_before = float(np.sum(c / x))
        spec = _Spec(n)
        x, _, _ = mma_subproblem(
            x,
            None,
            -c / x**2,
            np.zeros(0),
            np.zeros((0, n)),
            None,
            None,
            np.full(n, 0.05),
            np.full(n, 5.0),
            None,
            None,
            0,
            spec,
        )
        assert float(np.sum(c / x)) < f_before


class TestThroughCore:
    @staticmethod
    def _build(n=12, V=6.0, seed=0, move=0.2):
        c = _problem(n, seed)
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((n,), name="x")
        x.dtype(jnp.float64)
        x.initialize(jax.nn.initializers.constant(V / n))
        x.optimizer(jno.optimizers.mma(move=move, lower=0.05, upper=5.0))
        f0 = jno.fn(lambda xv: jnp.sum(jnp.asarray(c) / xv), [x], name="f0")
        vol = jno.fn(lambda xv: jnp.sum(xv), [x], name="vol")
        return c, d, x, f0, vol

    def test_mma_reaches_the_optimum_end_to_end(self):
        """Values and gradients now arrive through the trace, and the step through the callback."""
        V = 6.0
        c, d, x, f0, vol = self._build(V=V)
        x_star, f_star = _analytic(c, V)

        crux = jno.core([f0, jno.le(vol, V)], domain=d)
        crux.solve(120)
        xf = np.asarray(crux.eval([x])).reshape(-1)

        assert float(np.sum(c / xf)) == pytest.approx(f_star, rel=1e-6)
        assert np.allclose(xf, x_star, rtol=1e-5)
        assert xf.sum() <= V + 1e-8
        assert xf.min() >= 0.05 - 1e-12 and xf.max() <= 5.0 + 1e-12

    def test_the_objective_decreases_monotonically(self):
        V = 6.0
        _c, d, _x, f0, vol = self._build(V=V)
        hist = jno.core([f0, jno.le(vol, V)], domain=d).solve(40).total_loss_history
        assert hist[-1] < hist[0]
        # MMA is a descent method on this problem; a rise means the asymptotes are mis-adapted.
        assert np.all(np.diff(hist) <= 1e-9), "compliance history must not rise"


class TestGuards:
    """Each failure mode here is one that would otherwise produce a plausible wrong answer."""

    def test_mma_without_any_constraint_is_refused(self):
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((4,), name="x")
        x.optimizer(jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0))
        f0 = jno.fn(lambda xv: jnp.sum(xv**2), [x], name="f0")
        with pytest.raises(ValueError, match="no inequality constraints"):
            jno.core([f0], domain=d).solve(1)

    def test_a_constraint_without_a_constrained_optimiser_is_refused(self):
        """Otherwise the constraint is evaluated every step and then quietly ignored."""
        import optax

        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((4,), name="x").optimizer(optax.adam(1e-2))
        f0 = jno.fn(lambda xv: jnp.sum(xv**2), [x], name="f0")
        vol = jno.fn(lambda xv: jnp.sum(xv), [x], name="vol")
        with pytest.raises(ValueError, match="no constrained optimiser"):
            jno.core([f0, jno.le(vol, 1.0)], domain=d).solve(1)

    def test_missing_bounds_are_refused(self):
        """MMA scales its asymptotes and move limit by the box width — there is no default."""
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((4,), name="x")
        x.optimizer(jno.optimizers.mma(move=0.2))  # no lower=/upper=
        f0 = jno.fn(lambda xv: jnp.sum(xv**2), [x], name="f0")
        vol = jno.fn(lambda xv: jnp.sum(xv), [x], name="vol")
        with pytest.raises(ValueError, match="lower= and upper= are required"):
            jno.core([f0, jno.le(vol, 1.0)], domain=d).solve(1)


class TestPerVariableBounds:
    """A nodal-movement design variable is bounded per node, about its OWN initial position.

    Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, eq. (34) and Sec. 3: the
    limit is +/-2 on the movement ``d_x,i``, while ``.trainable()`` seeds the parameter at the
    absolute coordinates — so the box is a different interval for each variable.
    """

    def test_an_array_bound_is_honoured_per_variable(self):
        n = 8
        x0 = np.linspace(0.0, 7.0, n)
        lo, hi = x0 - 0.5, x0 + 0.5

        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((n,), name="x")
        x.dtype(jnp.float64)
        x.initialize(lambda key, shape, dtype=None: jnp.asarray(x0))
        x.optimizer(jno.optimizers.mma(move=0.5, lower=lo, upper=hi))
        # Pulls every variable hard towards -inf, so the LOWER bound is what stops it.
        f0 = jno.fn(lambda xv: jnp.sum(xv), [x], name="f0")
        vol = jno.fn(lambda xv: jnp.sum(xv * 0.0), [x], name="vol")

        crux = jno.core([f0, jno.le(vol, 1.0)], domain=d)
        crux.solve(30)
        xf = np.asarray(crux.eval([x])).reshape(-1)

        assert np.all(xf >= lo - 1e-9) and np.all(xf <= hi + 1e-9), "the per-variable box must hold"
        np.testing.assert_allclose(xf, lo, atol=1e-6), "and a descent this strong must reach it"
        # A single scalar box could not express this: the intervals do not overlap.
        assert lo[-1] > hi[0]

    def test_a_wrong_length_bound_is_refused(self):
        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((4,), name="x")
        x.optimizer(jno.optimizers.mma(move=0.2, lower=np.zeros(3), upper=np.ones(4)))
        f0 = jno.fn(lambda xv: jnp.sum(xv**2), [x], name="f0")
        vol = jno.fn(lambda xv: jnp.sum(xv), [x], name="vol")
        with pytest.raises(ValueError, match="3 entries but the design variable has 4"):
            jno.core([f0, jno.le(vol, 1.0)], domain=d).solve(1)


class TestMoveLimitDamping:
    """``move_gamma`` — shrink the move limit as the run proceeds.

    MMA does not converge to a point: near the optimum the iterates chatter at the scale of the
    move limit. Harmless for the design, fatal for any relative-change convergence test built on
    top of it — on the deformable cantilever the SIMP continuation could never fire until this
    existed, and fired twice immediately after.
    """

    def test_the_spec_decays_and_floors_the_move_limit(self):
        from jno.optimizers.mma import _BlendedSpec

        spec = jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0, move_gamma=0.9, move_min=0.05)
        blocks = [(0, spec)]
        assert float(_BlendedSpec(blocks, [3], k=0).move[0]) == pytest.approx(0.2)
        assert float(_BlendedSpec(blocks, [3], k=5).move[0]) == pytest.approx(0.2 * 0.9**5)
        assert float(_BlendedSpec(blocks, [3], k=500).move[0]) == pytest.approx(0.05), "the floor holds"

    def test_the_default_is_the_classic_method(self):
        from jno.optimizers.mma import _BlendedSpec

        blocks = [(0, jno.optimizers.mma(move=0.2, lower=0.0, upper=1.0))]
        for k in (0, 10, 1000):
            assert float(_BlendedSpec(blocks, [3], k=k).move[0]) == pytest.approx(0.2)

    def test_damping_still_reaches_the_analytic_optimum(self):
        """A smaller late-stage step must not cost the answer — only the chatter around it."""
        c = _problem()
        V, n = 6.0, c.size
        x_star, f_star = _analytic(c, V)

        d = jno.domain.from_array({"_": np.zeros((1, 1))})
        x = jno.np.parameter((n,), name="x_damp")
        x.dtype(jnp.float64)
        x.initialize(jax.nn.initializers.constant(V / n))
        x.optimizer(jno.optimizers.mma(move=0.2, lower=0.05, upper=5.0, move_gamma=0.99, move_min=0.01))
        f0 = jno.fn(lambda xv: jnp.sum(jnp.asarray(c) / xv), [x], name="f0")
        vol = jno.fn(lambda xv: jnp.sum(xv), [x], name="vol")

        crux = jno.core([f0, jno.le(vol, V)], domain=d)
        crux.solve(150)
        xf = np.asarray(crux.eval([x])).reshape(-1)
        assert float(np.sum(c / xf)) == pytest.approx(f_star, rel=1e-5)
        assert xf.sum() <= V + 1e-8
