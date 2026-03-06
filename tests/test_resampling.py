"""Comprehensive tests for adaptive resampling strategies and solve() integration."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest

import jno
import jno.numpy as jnn
from jno import LearningRateSchedule as lrs
from jno import sampler
from jno.resampling import CR3, HA, PINNFluence, R3, RAD, RARD, RandomResampling, ResamplingStrategy


def _points_1d(n: int = 24) -> jnp.ndarray:
    return jnp.linspace(0.0, 1.0, n).reshape(n, 1)


def _points_xt(n: int = 24) -> jnp.ndarray:
    x = jnp.linspace(0.0, 1.0, n)
    t = jnp.linspace(0.0, 1.0, n)
    return jnp.stack([x, t], axis=1)


def _residuals(n: int = 24) -> jnp.ndarray:
    return jnp.linspace(0.01, 1.0, n)


def _domain_stub(points: jnp.ndarray, tag: str = "interior"):
    # Strategies look for `domain._mesh_points[tag]`.
    return SimpleNamespace(_mesh_points={tag: points})


class CountingStrategy(ResamplingStrategy):
    """Test strategy used to verify solve() integration paths."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0

    def resample(self, points, residuals, domain, tag, epoch, rng_key):
        self.call_count += 1
        return points


def _build_solver(strategy: ResamplingStrategy, *, time: tuple[float, float, int] | None = None):
    if time is None:
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    else:
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05), time=time)

    x, *_ = domain.variable(
        "interior",
        sample=(64, None),
        resampling_strategy=strategy,
    )

    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.mlp(1, hidden_dims=16, num_layers=2, key=key)
    u = u_net(x) * x * (1.0 - x)
    pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)

    # Keep this pointwise (not .mse) so solve()'s resampling pipeline can
    # consume per-point residual geometry.
    solver = jno.core([pde], domain)
    u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
    return solver, domain


def _build_solver_nd(
    strategy: ResamplingStrategy,
    *,
    spatial_dim: int,
    time: tuple[float, float, int] | None = None,
):
    if spatial_dim == 2:
        constructor = jno.domain.rect(mesh_size=0.2)
    elif spatial_dim == 3:
        constructor = jno.domain.cube(mesh_size=0.6)
    else:
        raise ValueError("spatial_dim must be 2 or 3")

    if time is None:
        domain = 1 * jno.domain(constructor=constructor)
    else:
        domain = 1 * jno.domain(constructor=constructor, time=time)

    vars_all = domain.variable(
        "interior",
        sample=(48, None),
        resampling_strategy=strategy,
    )
    coords = vars_all[:spatial_dim]

    key = jax.random.PRNGKey(7)
    u_net = jnn.nn.mlp(spatial_dim, hidden_dims=16, num_layers=2, key=key)
    u = u_net(*coords)
    for c in coords:
        u = u * c * (1.0 - c)

    pde = jnn.laplacian(u, list(coords)) - jnn.sin(jnn.pi * coords[0])
    solver = jno.core([pde], domain)
    u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
    return solver, domain


def test_sampler_factory_types():
    assert isinstance(sampler.random(), RandomResampling)
    assert isinstance(sampler.rad(), RAD)
    assert isinstance(sampler.rard(), RARD)
    assert isinstance(sampler.ha(), HA)
    assert isinstance(sampler.cr3(), CR3)
    assert isinstance(sampler.pinnfluence(), PINNFluence)


def test_base_should_resample_cadence_and_start_epoch():
    s = RandomResampling(resample_every=3, resample_fraction=0.2, start_epoch=5)

    # Before start_epoch -> never resamples.
    assert s.should_resample(0) is False
    assert s.should_resample(4) is False

    # At start_epoch -> due.
    assert s.should_resample(5) is True
    s.update_epoch(5)

    # Next due only after 3 epochs.
    assert s.should_resample(6) is False
    assert s.should_resample(7) is False
    assert s.should_resample(8) is True


@pytest.mark.parametrize(
    "strategy, points",
    [
        (RandomResampling(resample_every=1, resample_fraction=0.25, start_epoch=0), _points_1d()),
        (RAD(resample_every=1, resample_fraction=0.25, start_epoch=0, k=3), _points_1d()),
        (RARD(resample_every=1, resample_fraction=0.25, start_epoch=0, power=2.0), _points_1d()),
        (HA(resample_every=1, resample_fraction=0.5, start_epoch=0), _points_1d()),
        (R3(resample_every=1, resample_fraction=0.5, start_epoch=0), _points_1d()),
        (CR3(resample_every=1, resample_fraction=0.5, start_epoch=0), _points_xt()),
        (PINNFluence(resample_every=1, resample_fraction=0.25, start_epoch=0, candidate_factor=2.0), _points_1d()),
    ],
)
def test_strategy_resample_preserves_shape_and_finite_values(strategy, points):
    n = points.shape[0]
    residuals = _residuals(n)
    domain = _domain_stub(points, tag="interior")

    out = strategy.resample(points, residuals, domain, "interior", epoch=0, rng_key=jax.random.PRNGKey(0))

    assert out.shape == points.shape
    assert jnp.all(jnp.isfinite(out))


def test_strategy_residual_shape_mismatch_returns_input():
    points = _points_1d(16)
    wrong_residuals = _residuals(8)
    domain = _domain_stub(points)

    for strategy in [RAD(), RARD(), R3(), CR3(), PINNFluence()]:
        out = strategy.resample(points, wrong_residuals, domain, "interior", epoch=0, rng_key=jax.random.PRNGKey(0))
        assert jnp.array_equal(out, points)


def test_random_resampling_without_candidates_returns_input():
    points = _points_1d(16)
    residuals = _residuals(16)
    domain = SimpleNamespace()  # no _mesh_points

    s = RandomResampling(resample_every=1, resample_fraction=0.5, start_epoch=0)
    out = s.resample(points, residuals, domain, "interior", epoch=0, rng_key=jax.random.PRNGKey(0))
    assert jnp.array_equal(out, points)


def test_cr3_updates_gamma_history_on_resample():
    points = _points_xt(20)
    residuals = _residuals(20)
    domain = _domain_stub(points)

    s = CR3(resample_every=1, resample_fraction=0.5, start_epoch=0, gamma0=-0.5)
    old_gamma = s.gamma
    _ = s.resample(points, residuals, domain, "interior", epoch=0, rng_key=jax.random.PRNGKey(0))

    assert len(s.gamma_history) == 1
    assert s.gamma != old_gamma


def test_ha_alternating_phase_counter_advances():
    points = _points_1d(20)
    residuals = _residuals(20)
    domain = _domain_stub(points)

    s = HA(resample_every=1, resample_fraction=0.5, start_epoch=0, alternate=True, random_first=True)
    _ = s.resample(points, residuals, domain, "interior", epoch=0, rng_key=jax.random.PRNGKey(0))
    _ = s.resample(points, residuals, domain, "interior", epoch=1, rng_key=jax.random.PRNGKey(1))

    assert s._apply_count == 2


@pytest.mark.integration
def test_solve_invokes_resampling_strategy_each_epoch():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.5, start_epoch=0)
    solver, _ = _build_solver(strategy)

    stats = solver.solve(epochs=3)

    assert strategy.call_count > 0
    assert strategy._last_resample_epoch == 2
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


@pytest.mark.integration
def test_solve_resampling_with_offload_data():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.5, start_epoch=0)
    solver, _ = _build_solver(strategy)

    stats = solver.solve(epochs=4, offload_data=True, batchsize=16)

    assert strategy.call_count > 0
    assert strategy._last_resample_epoch == 3
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


@pytest.mark.integration
def test_time_dependent_tag_is_skipped_by_current_resampling_path():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.5, start_epoch=0)
    solver, _ = _build_solver(strategy, time=(0.0, 1.0, 3))

    stats = solver.solve(epochs=2)

    # Current solve() path only applies resampling to steady-state (B,1,N,D)
    # tags; time-dependent tags are skipped with a warning.
    assert strategy.call_count == 0
    assert strategy._last_resample_epoch == -1
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


@pytest.mark.integration
def test_unmapped_resampling_tag_does_not_crash_training():
    strategy_used = CountingStrategy(resample_every=1, resample_fraction=0.5, start_epoch=0)
    strategy_unused = CountingStrategy(resample_every=1, resample_fraction=0.5, start_epoch=0)

    solver, domain = _build_solver(strategy_used)
    domain._resampling_strategies["unused_tag"] = strategy_unused

    stats = solver.solve(epochs=3)

    assert strategy_used.call_count > 0
    assert strategy_unused.call_count == 0
    assert strategy_unused._last_resample_epoch == -1
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


@pytest.mark.integration
def test_solve_resampling_works_for_2d_steady_domain():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.3, start_epoch=0)
    solver, _ = _build_solver_nd(strategy, spatial_dim=2)

    stats = solver.solve(epochs=3)

    assert strategy.call_count > 0
    assert strategy._last_resample_epoch == 2
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


@pytest.mark.integration
def test_solve_resampling_works_for_3d_steady_domain():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.3, start_epoch=0)
    solver, _ = _build_solver_nd(strategy, spatial_dim=3)

    stats = solver.solve(epochs=3)

    assert strategy.call_count > 0
    assert strategy._last_resample_epoch == 2
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


@pytest.mark.integration
def test_time_dependent_2d_domain_training_path_remains_stable():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.3, start_epoch=0)
    solver, _ = _build_solver_nd(strategy, spatial_dim=2, time=(0.0, 1.0, 3))

    stats = solver.solve(epochs=2)

    # Current solve() resampling path focuses on steady-state point layouts;
    # time-dependent tags are skipped gracefully.
    assert strategy.call_count == 0
    assert strategy._last_resample_epoch == -1
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])
