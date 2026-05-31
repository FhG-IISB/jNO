"""Comprehensive tests for adaptive resampling strategies and solve() integration."""

from __future__ import annotations

from types import SimpleNamespace

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
import jno.jnp_ops as jnn
from jno import LearningRateSchedule as lrs
from jno import sampler
from jno.utils.adaptive.resampling import (
    CR3,
    HA,
    R3,
    RAD,
    RARD,
    PINNFluence,
    RandomResampling,
    ResamplingStrategy,
)


def _points_1d(n: int = 24) -> jnp.ndarray:
    return jnp.linspace(0.0, 1.0, n).reshape(n, 1)


def _points_xt(n: int = 24) -> jnp.ndarray:
    x = jnp.linspace(0.0, 1.0, n)
    t = jnp.linspace(0.0, 1.0, n)
    return jnp.stack([x, t], axis=1)


def _residuals(n: int = 24) -> jnp.ndarray:
    return jnp.linspace(0.01, 1.0, n)


def _domain_stub(points: jnp.ndarray, tag: str = "interior"):
    """Minimal domain stub exposing the draw_candidates() interface."""

    class _Stub:
        def draw_candidates(self, t):
            return (np.asarray(points), None) if t == tag else (None, None)

    return _Stub()


class CountingStrategy(ResamplingStrategy):
    """Test strategy used to verify solve() integration paths."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0

    def resample(self, points, residuals, domain, tag, epoch, rng_key, candidates=None):
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
    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=16, num_layers=2, key=key))
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
    u_net = jnn.nn.wrap(foundax.mlp(spatial_dim, hidden_dims=16, num_layers=2, key=key))
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


def test_sampler_factory_has_r3():
    assert isinstance(sampler.r3(), R3)


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
        (
            RandomResampling(resample_every=1, resample_fraction=0.25, start_epoch=0),
            _points_1d(),
        ),
        (
            RAD(resample_every=1, resample_fraction=0.25, start_epoch=0, k=3),
            _points_1d(),
        ),
        (
            RARD(resample_every=1, resample_fraction=0.25, start_epoch=0, power=2.0),
            _points_1d(),
        ),
        (HA(resample_every=1, resample_fraction=0.5, start_epoch=0), _points_1d()),
        (R3(resample_every=1, resample_fraction=0.5, start_epoch=0), _points_1d()),
        (CR3(resample_every=1, resample_fraction=0.5, start_epoch=0), _points_xt()),
        (
            PINNFluence(
                resample_every=1,
                resample_fraction=0.25,
                start_epoch=0,
                candidate_factor=2.0,
            ),
            _points_1d(),
        ),
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
        out = strategy.resample(
            points,
            wrong_residuals,
            domain,
            "interior",
            epoch=0,
            rng_key=jax.random.PRNGKey(0),
        )
        assert jnp.array_equal(out, points)


def test_random_resampling_without_candidates_returns_input():
    points = _points_1d(16)
    residuals = _residuals(16)
    domain = SimpleNamespace()  # no draw_candidates

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

    s = HA(
        resample_every=1,
        resample_fraction=0.5,
        start_epoch=0,
        alternate=True,
        random_first=True,
    )
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
def test_time_dependent_resampling_fires_for_1d_domain():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.5, start_epoch=0)
    solver, domain = _build_solver(strategy, time=(0.0, 1.0, 3))

    stats = solver.solve(epochs=2)

    assert strategy.call_count > 0
    assert strategy._last_resample_epoch == 1
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    # Spatial coordinates must be identical across all T timesteps after resampling.
    ctx = domain.context.get("interior")
    if ctx is not None and ctx.ndim == 4:
        T = ctx.shape[1]
        for t in range(1, T):
            assert np.allclose(ctx[:, 0, :, :], ctx[:, t, :, :], atol=1e-6)


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
def test_time_dependent_2d_domain_resampling_fires():
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.3, start_epoch=0)
    solver, domain = _build_solver_nd(strategy, spatial_dim=2, time=(0.0, 1.0, 3))

    stats = solver.solve(epochs=2)

    assert strategy.call_count > 0
    assert strategy._last_resample_epoch == 1
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    # Spatial coordinates identical across T.
    ctx = domain.context.get("interior")
    if ctx is not None and ctx.ndim == 4:
        T = ctx.shape[1]
        for t in range(1, T):
            assert np.allclose(ctx[:, 0, :, :], ctx[:, t, :, :], atol=1e-6)


def test_solve_resampling_works_with_adaptive_weight_wrapped_losses():
    """Resampling should work when constraints are ``w * loss.mse``.

    Adaptive weight balancers wrap the raw loss in a ``BinaryOp(*)`` which
    previously prevented ``_strip_reduction_for_resampling`` from reaching
    the inner ``.mse`` reduction to recover pointwise residuals.
    """
    strategy = CountingStrategy(resample_every=1, resample_fraction=0.3, start_epoch=0)

    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = domain.variable(
        "interior",
        sample=(64, None),
        resampling_strategy=strategy,
    )

    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=16, num_layers=2, key=key))
    u = u_net(x) * x * (1.0 - x)

    pde = (jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)).mse
    bcs = (u - 0).mse

    w0, w1 = jno.fn.adaptive.relobralo([pde, bcs])

    solver = jno.core([w0 * pde, w1 * bcs, w0.tracker(), w1.tracker()], domain)
    u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))

    stats = solver.solve(epochs=3)

    assert strategy.call_count > 0
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


def test_merged_domain_draw_candidates_covers_both_subdomains():
    """draw_candidates on a merged domain should return points from both sub-domains."""
    d1 = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05, x_range=(0.0, 0.5)))
    d2 = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05, x_range=(0.5, 1.0)))

    # Sample interior from each so _mesh_pool is populated.
    d1.variable("interior", sample=(40, None))
    d2.variable("interior", sample=(40, None))

    # Measure individual pool sizes BEFORE merge (__add__ mutates d1 in-place).
    pool1_pre, _ = d1.draw_candidates("interior")
    pool2_pre, _ = d2.draw_candidates("interior")
    n1, n2 = len(pool1_pre), len(pool2_pre)

    merged = d1 + d2  # d1 is now the merged domain

    pts, _ = merged.draw_candidates("interior")
    assert pts is not None
    assert len(pts) >= n1 + n2


def test_domain_draw_candidates_returns_spatial_slice_for_time_dep():
    """For a time-dependent domain the candidate pool should be (N, D_spatial)."""
    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05), time=(0.0, 1.0, 4))
    domain.variable("interior", sample=(50, None))

    pts, nrm = domain.draw_candidates("interior")
    assert pts is not None
    # Spatial coordinates only — should be (N, D_spatial), not (T, N, D_spatial).
    assert pts.ndim == 2, f"Expected 2-D spatial slice, got shape {pts.shape}"
    # normals may or may not be present depending on domain type; shape must match pts.
    if nrm is not None:
        assert nrm.shape == pts.shape


def test_merged_domain_resampling_draws_from_union():
    """Resampling on a merged domain can select points from either sub-domain.

    Two line domains cover non-overlapping x intervals ([0, 0.4] and [0.6, 1.0]).
    After merging, draw_candidates returns a pool that spans both intervals.
    Replacing all working-set points via RandomResampling must yield points from
    the second interval — proving the full merged pool is used, not just the first
    sub-domain's nodes.
    """
    d1 = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05, x_range=(0.0, 0.4)))
    d2 = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05, x_range=(0.6, 1.0)))
    d1.variable("interior", sample=(20, None))
    d2.variable("interior", sample=(20, None))
    merged = d1 + d2

    pool, _ = merged.draw_candidates("interior")
    assert pool is not None
    assert np.any(pool[:, 0] <= 0.45), "pool should cover left interval"
    assert np.any(pool[:, 0] >= 0.55), "pool should cover right interval"

    # Batch 0 starts entirely in [0.1, 0.35] — the left sub-domain.
    batch0_pts = jnp.array(merged.context["interior"][0, 0])
    assert float(batch0_pts[:, 0].max()) < 0.45, "batch 0 should start in left interval"

    strategy = RandomResampling(resample_every=1, resample_fraction=1.0, start_epoch=0)
    new_pts = strategy.resample(
        batch0_pts,
        jnp.ones(batch0_pts.shape[0]),
        merged,
        "interior",
        epoch=0,
        rng_key=jax.random.PRNGKey(42),
        candidates=pool,
    )
    # With full replacement from a balanced pool covering both halves, at least one
    # point from [0.6, 1.0] must appear; P(failure) < 0.1% per seed.
    assert np.any(np.asarray(new_pts[:, 0]) >= 0.55), (
        "After full-fraction resampling from merged pool, points from the right "
        f"sub-domain must appear; got max x = {float(new_pts[:, 0].max()):.3f}"
    )


def test_polygon_draw_candidates_exceeds_sample_size():
    """PolygonDomain.draw_candidates returns more candidates than the working sample.

    For a unit-square domain with 50 interior points, draw_candidates should
    generate max(10*50, 1000) = 1000 candidates so resampling strategies have
    genuine exploration room.
    """
    pytest.importorskip("shapely")
    SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    dom = jno.PolygonDomain(SQUARE, name="sq")
    dom.variable("interior", sample=(50, None))

    pts, nrm = dom.draw_candidates("interior")

    assert pts is not None, "draw_candidates must return a point array"
    assert len(pts) > 50, f"candidate pool ({len(pts)}) must exceed sample size (50)"
    assert nrm is None, "interior tags carry no normals"


def test_polygon_draw_candidates_returns_fresh_points():
    """Consecutive calls to PolygonDomain.draw_candidates produce different samples.

    Each call samples the geometry afresh (not from a fixed cached set), so two
    independent calls are extremely unlikely to be identical.
    """
    pytest.importorskip("shapely")
    SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    dom = jno.PolygonDomain(SQUARE, name="sq")
    dom.variable("interior", sample=(50, None))

    pts1, _ = dom.draw_candidates("interior")
    pts2, _ = dom.draw_candidates("interior")

    assert not np.allclose(pts1, pts2), (
        "Two consecutive draw_candidates calls returned identical arrays; "
        "the implementation should sample fresh points on each call."
    )


@pytest.mark.integration
def test_boundary_normals_updated_after_resample():
    """After resampling a boundary tag, n_{tag} normals stay consistent with new points.

    Samples the boundary with normals and attaches an RAD strategy to it.
    After solve(), the stored normals must:
      1. Have the same shape as the boundary-point array.
      2. All be unit vectors (‖n‖ ≈ 1.0), since they are recovered from the
         candidate pool where all normals are already normalised.
    """
    strategy = RAD(resample_every=1, resample_fraction=0.5, start_epoch=0, k=3)

    # mesh_size=0.1 → ~40 boundary nodes in pool, working set = 20 → 2× ratio.
    domain = 1 * jno.domain(constructor=jno.domain.rect(mesh_size=0.1))
    b_vars = domain.variable("boundary", sample=(20, None), normals=True, resampling_strategy=strategy)
    xb, yb = b_vars[0], b_vars[1]

    xi, yi = domain.variable("interior", sample=(40, None))[:2]

    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.wrap(foundax.mlp(2, hidden_dims=8, num_layers=2, key=key))

    pde = jnn.laplacian(u_net(xi, yi), [xi, yi])
    bc = (u_net(xb, yb) - 0.0).mse

    solver = jno.core([pde, bc], domain)
    u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
    solver.solve(epochs=3)

    pts = np.asarray(domain.context["boundary"])   # (B, T, N, 2)
    nrm = np.asarray(domain.context["n_boundary"]) # (B, T, N, 2)

    assert nrm.shape == pts.shape, (
        f"Normal shape {nrm.shape} must match point shape {pts.shape}"
    )
    magnitudes = np.linalg.norm(nrm.reshape(-1, 2), axis=-1)
    assert np.allclose(magnitudes, 1.0, atol=1e-4), (
        f"Normals must be unit vectors after resampling; "
        f"magnitudes range {magnitudes.min():.6f} – {magnitudes.max():.6f}"
    )


# ---------------------------------------------------------------------------
# Physics-motivated resampling tests
# ---------------------------------------------------------------------------
# The Burgers benchmark provides a ground truth for WHERE points should move:
# the steep spatial gradient near x=0 is the hardest region to satisfy,
# so residual-based strategies should migrate interior points there.
#
# Two levels of testing:
#   1. Oracle tests (no training): synthetic Gaussian residuals peaked at a
#      known location verify the concentration mechanism in isolation.
#   2. Integration test: a real Burgers solve confirms the mechanism fires
#      during training and points end up near the expected region.
# ---------------------------------------------------------------------------

_BURGERS_NU = 0.01 / np.pi  # classical PINN benchmark value


def _make_burgers_domain(strategy=None, mesh_size=0.15, n_sample=60):
    """Rectangle [-1,1] × [0,1]; x is spatial, y acts as time."""
    domain = 1 * jno.domain(
        constructor=jno.domain.rect(
            mesh_size=mesh_size,
            x_range=(-1.0, 1.0),
            y_range=(0.0, 1.0),
        )
    )
    if strategy is not None:
        domain.variable("interior", sample=(n_sample, None), resampling_strategy=strategy)
    else:
        domain.variable("interior", sample=(n_sample, None))
    return domain


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(
            RAD(resample_every=1, resample_fraction=0.2, start_epoch=0, k=5),
            id="rad",
        ),
        pytest.param(
            RARD(resample_every=1, resample_fraction=0.2, start_epoch=0, power=2.0),
            id="rard",
        ),
    ],
)
def test_oracle_residuals_concentrate_points_toward_burgers_shock(strategy):
    """RAD/RARD should concentrate points at a known high-residual strip.

    Uses Gaussian oracle residuals peaked at x=0 — the location where the
    Burgers solution develops its steepest gradient — to verify the
    concentration mechanism in isolation without any neural-network training.
    """
    domain = _make_burgers_domain()
    pts = jnp.array(np.asarray(domain.context["interior"])[0, 0])  # (N, 2)
    cand_pts, _ = domain.draw_candidates("interior")

    SHOCK_X = 0.0  # steep-gradient location for Burgers with IC = -sin(πx)
    HALF_WIDTH = 0.3  # |x| < 0.3 is the "shock strip"

    initial_frac = float(jnp.mean(jnp.abs(pts[:, 0] - SHOCK_X) < HALF_WIDTH))

    rng = jax.random.PRNGKey(0)
    for step in range(25):
        x_coords = pts[:, 0]
        # Oracle: mimics Burgers PDE residual — largest near x=0 where the
        # steep gradient lives, decaying smoothly toward the boundaries.
        residuals = jnp.exp(-8.0 * x_coords**2)
        rng, key = jax.random.split(rng)
        pts = strategy.resample(pts, residuals, domain, "interior", step, key, candidates=cand_pts)
        strategy.update_epoch(step)

    final_frac = float(jnp.mean(jnp.abs(pts[:, 0] - SHOCK_X) < HALF_WIDTH))

    assert final_frac > initial_frac + 0.10, (
        f"{type(strategy).__name__}: expected ≥10 pp more points in |x|<{HALF_WIDTH} "
        f"(initial {initial_frac:.2f} → final {final_frac:.2f})"
    )


@pytest.mark.integration
def test_burgers_rad_resampling_concentrates_near_steep_gradient():
    """After training on viscous Burgers, RAD interior points cluster near x≈0.

    Solves ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x² on [-1,1]×[0,1] with ν = 0.01/π.
    A hard-constraint architecture exactly satisfies:
        u(x, 0) = −sin(πx)   (initial condition)
        u(±1, t) = 0          (Dirichlet boundary conditions)

    The solution develops a steep spatial gradient near x = 0 at late times.
    Because that region is hardest to satisfy, PDE residuals concentrate there
    during training — so RAD should migrate collocation points toward |x| < 0.25.

    Mesh parameters are chosen so the candidate pool (421 nodes, mesh_size=0.08)
    is ~7× larger than the working set (60 points), giving RAD genuine room to
    move points into the high-residual strip.
    """
    strategy = RAD(resample_every=20, resample_fraction=0.25, start_epoch=0, k=5)

    domain = 1 * jno.domain(
        constructor=jno.domain.rect(
            mesh_size=0.08,
            x_range=(-1.0, 1.0),
            y_range=(0.0, 1.0),
        )
    )
    vars_all = domain.variable("interior", sample=(60, None), resampling_strategy=strategy)
    x, t = vars_all[0], vars_all[1]

    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.wrap(foundax.mlp(2, hidden_dims=20, num_layers=3, key=key))

    # Hard-constraint lift: satisfies IC and BCs analytically.
    #   u(x,0) = 0 - sin(πx)·1 = -sin(πx)   ✓
    #   u(±1,t) = 0 - sin(±π)·(1-t) = 0     ✓
    u = u_net(x, t) * (1.0 - x * x) * t - jnn.sin(jnn.pi * x) * (1.0 - t)

    # Use .mse so the optimizer minimises squared residuals (loss ≥ 0),
    # which ensures _strip_reduction_for_resampling can recover pointwise values
    # with correct spatial structure for the RAD scoring step.
    pde = (u.d(t) + u * u.d(x) - _BURGERS_NU * jnn.laplacian(u, [x])).mse

    solver = jno.core([pde], domain)
    u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))

    initial_pts = np.asarray(domain.context["interior"])[0, 0]  # (N, 2)
    # Mean |x| for a uniform distribution on [-1,1] ≈ 0.5.
    # As RAD clusters points near x=0 (the steep-gradient strip), mean |x| drops.
    initial_mean_abs_x = np.mean(np.abs(initial_pts[:, 0]))

    stats = solver.solve(epochs=120)

    final_pts = np.asarray(domain.context["interior"])[0, 0]  # (N, 2)
    final_mean_abs_x = np.mean(np.abs(final_pts[:, 0]))

    assert strategy._last_resample_epoch >= 0, "RAD was never triggered"
    # Consistently drops by ≥0.10 across different network seeds and mesh
    # realizations: stable physics-based signal.
    assert final_mean_abs_x < initial_mean_abs_x - 0.10, (
        f"Expected mean |x| to drop by ≥0.10 as RAD concentrates near the shock "
        f"(initial {initial_mean_abs_x:.3f} → final {final_mean_abs_x:.3f})"
    )
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])
