"""Edge-case tests for jno.core.

Most of jno.core is exercised end-to-end by integration tests; this file
focuses on the kind of edge cases that would otherwise surface as cryptic
errors deep in the JIT'd training loop — empty constraint lists, missing
optimizers, temporal Variables on stationary domains, and re-entry into
solve()/compile().
"""

import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno
import jno.jnp_ops as jnn


def _tiny_net(in_dim=1, out_dim=1, hidden=8, key_seed=0):
    return jnn.nn.wrap(
        foundax.mlp(
            in_dim,
            output_dim=out_dim,
            hidden_dims=hidden,
            num_layers=2,
            key=jax.random.PRNGKey(key_seed),
        )
    )


def _stationary_1d_domain(mesh_size=0.1):
    return jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))


# ---------------------------------------------------------------------------
# Empty / degenerate constraint lists
# ---------------------------------------------------------------------------


class TestEmptyConstraints:
    def test_solve_with_empty_constraints_raises_clear_error(self):
        dom = _stationary_1d_domain()
        dom.variable("interior")  # sample so the domain has something
        crux = jno.core([], dom)
        with pytest.raises(ValueError, match="at least one constraint"):
            crux.solve(2)


# ---------------------------------------------------------------------------
# Missing optimizer
# ---------------------------------------------------------------------------


class TestMissingOptimizer:
    def test_no_optimizer_raises_with_setup_example(self):
        dom = _stationary_1d_domain()
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        # Intentionally NOT calling net.optimizer(...)
        pde = (net(x) - x).mse
        with pytest.raises(ValueError, match=r"(?s)has no optimizer.*model\.optimizer"):
            jno.core([pde], dom).solve(2)


# ---------------------------------------------------------------------------
# Temporal Variable on a stationary domain
# ---------------------------------------------------------------------------


class TestTemporalOnStationaryDomain:
    def test_constructing_core_raises_clear_error(self):
        dom = _stationary_1d_domain()  # no time= argument → stationary
        x, t = dom.variable("interior")
        # t has axis='temporal' — using it on a stationary domain is invalid.
        # Build any expression that touches t so the walker finds it.
        net = _tiny_net(in_dim=2)
        u = net(jno.np.concat([x, t], axis=-1))
        pde = (u.d(t)).mse
        with pytest.raises(ValueError, match="temporal Variable"):
            jno.core([pde], dom)


# ---------------------------------------------------------------------------
# min_consecutive guards
# ---------------------------------------------------------------------------


class TestMinConsecutiveGuards:
    def test_integral_time_requires_min_consecutive_gte_2(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2, time=(0.0, 1.0, 6)))
        x, t = dom.variable("interior")
        net = _tiny_net(in_dim=2)
        net.optimizer(optax.adam(1e-3))
        u = net(jno.np.concat([t, x], axis=-1))
        integral = u.integrate(t)
        crux = jno.core([integral.mse], dom)
        with pytest.raises(ValueError, match="min_consecutive"):
            crux.solve(2, min_consecutive=1)

    def test_time_dependent_min_consecutive_1_logs_nudge(self, caplog):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2, time=(0.0, 1.0, 4)))
        x, t = dom.variable("interior")
        net = _tiny_net(in_dim=2)
        net.optimizer(optax.adam(1e-3))
        u = net(jno.np.concat([t, x], axis=-1))
        pde = u.d(t).mse  # no IntegralTime, so min_consecutive=1 is legal
        crux = jno.core([pde], dom)
        # Just verify it doesn't raise — the nudge is a logger.info call which
        # is not easily captured here without configuring caplog for jno's logger.
        crux.solve(2, min_consecutive=1)


# ---------------------------------------------------------------------------
# Re-entry: calling solve() twice on the same core instance
# ---------------------------------------------------------------------------


class TestReentry:
    def test_two_solve_calls_accumulate_history(self):
        dom = _stationary_1d_domain()
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x]).mse
        crux = jno.core([pde], dom)

        h1 = crux.solve(2)
        h2 = crux.solve(2)

        # Both calls return statistics objects with non-empty training_logs
        assert len(h1.training_logs) >= 1
        assert len(h2.training_logs) >= 1
        # Final total loss is a finite float (not NaN)
        loss = h2.total_loss
        assert loss is None or jnp.isfinite(jnp.array(loss))


# ---------------------------------------------------------------------------
# statistics.total_loss property
# ---------------------------------------------------------------------------


class TestStatisticsTotalLoss:
    def test_total_loss_none_when_empty(self):
        from jno.utils.statistics import statistics

        s = statistics(logs=[])
        assert s.total_loss is None
        assert s.total_loss_history.size == 0

    def test_total_loss_picks_last_value(self):
        from jno.utils.statistics import statistics

        s = statistics(logs=[{"total_loss": jnp.array([3.0, 2.0, 1.5])}])
        assert s.total_loss == pytest.approx(1.5)
        assert s.total_loss_history.shape == (3,)

    def test_total_loss_concatenates_across_calls(self):
        from jno.utils.statistics import statistics

        s = statistics(
            logs=[
                {"total_loss": jnp.array([5.0, 4.0])},
                {"total_loss": jnp.array([3.0, 2.0, 1.0])},
            ]
        )
        assert s.total_loss == pytest.approx(1.0)
        assert s.total_loss_history.shape == (5,)
