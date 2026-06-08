"""Unit tests for builders in ``jno.utils.explainability``.

These tests exercise each builder against a hand-crafted constraint
function with known residual values, asserting that the JIT-compiled
closure returns the analytically expected statistics. They are *not*
end-to-end solver tests — that integration lives in
``test_wandb_integration.py::TestExplainabilityCallbacksWandBLogging``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jno.utils.explainability import make_residual_stats_fn


def test_residual_stats_matches_handcomputed():
    """Residual stats closure returns mean/std/max/p99 matching numpy ground truth."""
    # Two synthetic constraints with known residual arrays of different shapes.
    r0 = jnp.linspace(0.0, 1.0, 11)  # shape (11,) — values 0.0, 0.1, ..., 1.0
    r1 = jnp.array([0.5, 0.5, 0.5, 4.0])  # shape (4,) — one big outlier

    def fake_constraints_fn(full_models, context, *, batchsize, key, min_consecutive):
        # Signature matches the real ``compiled_constraints_fn``; arguments unused
        # because the residuals are baked in for this unit test.
        del full_models, context, batchsize, key, min_consecutive
        return [r0, r1]

    fn = make_residual_stats_fn(
        compiled_constraints_fn=fake_constraints_fn,
        n_constraints=2,
        batchsize=None,
        frozen={},
        static={},
    )
    fn = jax.jit(fn)

    # Empty trainable / context are accepted because fake_constraints_fn ignores them.
    means, stds, maxes, p99, raw = fn(trainable={}, context={}, rng=jax.random.PRNGKey(0))
    means = np.asarray(means)
    stds = np.asarray(stds)
    maxes = np.asarray(maxes)
    p99 = np.asarray(p99)

    expected_means = np.array([np.mean(np.asarray(r0)), np.mean(np.asarray(r1))], dtype=np.float32)
    expected_stds = np.array([np.std(np.asarray(r0)), np.std(np.asarray(r1))], dtype=np.float32)
    expected_maxes = np.array([np.max(np.asarray(r0)), np.max(np.asarray(r1))], dtype=np.float32)
    expected_p99 = np.array(
        [np.percentile(np.asarray(r0), 99.0), np.percentile(np.asarray(r1), 99.0)],
        dtype=np.float32,
    )

    np.testing.assert_allclose(means, expected_means, atol=1e-6)
    np.testing.assert_allclose(stds, expected_stds, atol=1e-6)
    np.testing.assert_allclose(maxes, expected_maxes, atol=1e-6)
    np.testing.assert_allclose(p99, expected_p99, atol=1e-5)

    # Raw residuals are returned as flattened 1-D arrays per constraint.
    assert len(raw) == 2
    assert raw[0].shape == (11,)
    assert raw[1].shape == (4,)


def test_residual_stats_subset_selects_one_constraint():
    """End-to-end: passing a subset of constraints scopes stats and W&B keys
    to those constraints only.
    """
    import foundax
    import optax
    import pytest

    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs
    from jno.utils.adaptive.callbacks import ResidualStatsCallback

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    key = jax.random.PRNGKey(0)

    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    u_net.optimizer(optax.sgd, lr=lrs.exponential(0.0, 1.0, 1, 0.0))  # frozen
    u = u_net(x)
    c0 = (u * u).mse  # arbitrary residual
    c1 = u.mse  # second constraint, different residual

    solver = jno.core([c0, c1], domain)

    # Subset: only c0 → result should have 1 column, and only the
    # solver-side index 0 should appear in W&B keys (we can't inspect those
    # without mocking, but we can verify the .result["indices"] is [0]).
    cb_subset = ResidualStatsCallback(interval=1, constraints=[c0])
    # Full: both constraints
    cb_full = ResidualStatsCallback(interval=1)

    solver.solve(2, callbacks=[cb_subset, cb_full])

    # Subset result has K=1 column with index 0.
    assert cb_subset.result["means"].shape == (2, 1)
    np.testing.assert_array_equal(cb_subset.result["indices"], np.array([0]))

    # Full has K=2 columns.
    assert cb_full.result["means"].shape == (2, 2)
    np.testing.assert_array_equal(cb_full.result["indices"], np.array([0, 1]))

    # Subset's c0 column must equal full's c0 column (same compute, just sliced).
    np.testing.assert_allclose(cb_subset.result["means"][:, 0], cb_full.result["means"][:, 0], atol=1e-6)

    # Re-accessing .mse returns a fresh placeholder → identity match fails.
    cb_bad = ResidualStatsCallback(interval=1, constraints=[(u * u).mse])
    with pytest.raises(ValueError, match="constraint .* not found"):
        # Re-using the same already-built solver; on_solve_begin fires when
        # solve() is called again.
        solver.solve(1, callbacks=[cb_bad])


def test_input_saliency_records_finite_values_with_consistent_shape():
    """InputSensitivityCallback compiles ``u.d(x)`` and records finite
    per-point sensitivities of consistent shape across sampled epochs.
    Params are frozen via lr=0, so values should not change between epochs.
    """
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs
    from jno.utils.adaptive.callbacks import InputSensitivityCallback

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    key = jax.random.PRNGKey(0)

    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    u_net.optimizer(optax.sgd, lr=lrs.exponential(0.0, 1.0, 1, 0.0))  # lr=0
    u = u_net(x)
    pde = u

    solver = jno.core([pde.mse], domain)
    cb = InputSensitivityCallback(u.d(x), interval=1)
    solver.solve(2, callbacks=[cb])

    values = cb.result["values"]
    assert cb.result["epochs"].shape == (2,)
    assert values.shape[0] == 2
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[0], values[1], atol=1e-5)


def test_input_sensitivity_rejects_non_placeholder():
    """InputSensitivityCallback rejects non-Placeholder arguments."""
    import pytest

    from jno.utils.adaptive.callbacks import InputSensitivityCallback

    with pytest.raises(TypeError, match="Placeholder"):
        InputSensitivityCallback(expr=jnp.zeros(3), interval=1)


# ---------------------------------------------------------------------------
# NTK spectrum
# ---------------------------------------------------------------------------


def test_ntk_spectrum_records_finite_eigenvalues_with_consistent_shape():
    """NTKSpectrumCallback compiles u.grad(net) and records finite
    eigenvalues with the expected shape; spectrum is identical across
    epochs when params are frozen (lr=0) thanks to the fixed sample seed.
    """
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs
    from jno.utils.adaptive.callbacks import NTKSpectrumCallback

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    key = jax.random.PRNGKey(0)

    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    u_net.optimizer(optax.sgd, lr=lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)
    pde = u

    n_points = 8
    top_k = 5

    solver = jno.core([pde.mse], domain)
    cb = NTKSpectrumCallback(u.grad(u_net), n_points=n_points, top_k=top_k, interval=1)
    solver.solve(2, callbacks=[cb])

    res = cb.result
    assert res["epochs"].shape == (2,)
    assert res["eigvals_topk"].shape == (2, top_k)
    assert res["all_eigvals"].shape == (2, n_points)
    assert np.all(np.isfinite(res["all_eigvals"]))
    assert np.all(res["all_eigvals"] >= -1e-5)
    assert np.all(res["eigvals_topk"][:, :-1] >= res["eigvals_topk"][:, 1:] - 1e-6)
    np.testing.assert_allclose(res["eigvals_topk"][0], res["eigvals_topk"][1], atol=1e-5)


def test_ntk_spectrum_rejects_non_network_gradient():
    """NTKSpectrumCallback rejects non-NetworkGradient arguments."""
    import pytest

    from jno.utils.adaptive.callbacks import NTKSpectrumCallback

    with pytest.raises(TypeError, match="NetworkGradient"):
        NTKSpectrumCallback(grad_expr=jnp.zeros(3), n_points=4)


# ---------------------------------------------------------------------------
# Hessian spectrum
# ---------------------------------------------------------------------------


def test_hessian_spectrum_records_finite_eigenvalues_with_consistent_shape():
    """HessianSpectrumCallback runs Lanczos and records top-k eigenvalues +
    a sharpness scalar with the expected shape; identical across epochs
    when params are frozen (lr=0) thanks to the fixed Lanczos start seed.
    """
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs
    from jno.utils.adaptive.callbacks import HessianSpectrumCallback

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    key = jax.random.PRNGKey(0)

    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    u_net.optimizer(optax.sgd, lr=lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)
    pde = u

    k = 4
    n_iter = 8

    solver = jno.core([pde.mse], domain)
    cb = HessianSpectrumCallback(k=k, n_iter=n_iter, interval=1)
    solver.solve(2, callbacks=[cb])

    res = cb.result
    assert res["epochs"].shape == (2,)
    assert res["eigvals"].shape == (2, k)
    assert res["sharpness"].shape == (2,)
    assert np.all(np.isfinite(res["eigvals"]))
    assert np.all(np.isfinite(res["sharpness"]))
    np.testing.assert_allclose(res["sharpness"], res["eigvals"][:, 0], atol=1e-5)
    assert np.all(res["eigvals"][:, :-1] >= res["eigvals"][:, 1:] - 1e-5)
    np.testing.assert_allclose(res["eigvals"][0], res["eigvals"][1], atol=1e-5)


def test_hessian_lanczos_recovers_quadratic_top_eigenvalue():
    """For L = mean(p²) with N=4 params, Hessian = (2/N)·I → λ_max = 0.5."""
    from jno.utils.explainability import make_hessian_spectrum_fn

    theta = jnp.array([1.0, 2.0, 3.0, 4.0])
    trainable = {"theta": theta}

    def constraints_fn(full_models, context, *, batchsize, key, min_consecutive):
        del context, batchsize, key, min_consecutive
        params = full_models["theta"]
        return [params * params]

    fn = make_hessian_spectrum_fn(
        constraints_fn,
        batchsize=None,
        frozen={"theta": None},
        static={"theta": None},
        k=3,
        n_iter=10,
    )
    top, lambda_max, all_eigs = fn(trainable, context={}, rng=jax.random.PRNGKey(0))
    expected = 2.0 / 4.0
    assert abs(lambda_max - expected) < 1e-5
    assert abs(top[0] - expected) < 1e-5
    assert all_eigs.shape == (10,)


def test_hessian_constraint_subset_isolates_one_constraint():
    """Subsetting the Hessian to one of two constraints with different
    closed-form λ_max recovers that constraint's value, not the mean.
    """
    from jno.utils.explainability import make_hessian_spectrum_fn

    N = 4
    a = 3.0
    b = 1.0
    theta = jnp.zeros((N,))
    trainable = {"theta": theta}

    def constraints_fn(full_models, context, *, batchsize, key, min_consecutive):
        del context, batchsize, key, min_consecutive
        p = full_models["theta"]
        return [(a * p) * (a * p), (b * p) * (b * p)]

    fn_full = make_hessian_spectrum_fn(
        constraints_fn,
        batchsize=None,
        frozen={"theta": None},
        static={"theta": None},
        k=2,
        n_iter=6,
    )
    fn_sub = make_hessian_spectrum_fn(
        constraints_fn,
        batchsize=None,
        frozen={"theta": None},
        static={"theta": None},
        k=2,
        n_iter=6,
        constraint_indices=(0,),
    )

    _, lmax_full, _ = fn_full(trainable, context={}, rng=jax.random.PRNGKey(0))
    _, lmax_sub, _ = fn_sub(trainable, context={}, rng=jax.random.PRNGKey(0))

    expected_full = (a**2 + b**2) / N
    expected_sub = 2 * a**2 / N
    assert abs(lmax_full - expected_full) < 1e-5
    assert abs(lmax_sub - expected_sub) < 1e-5
    assert abs(lmax_sub - lmax_full) > 0.1


def test_hessian_spectrum_subset_rejects_unknown_constraint():
    """Re-accessing .mse after solver construction fails identity matching."""
    import foundax
    import optax
    import pytest

    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs
    from jno.utils.adaptive.callbacks import HessianSpectrumCallback

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    u_net.optimizer(optax.sgd, lr=lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)
    pde_loss = u.mse

    solver = jno.core([pde_loss], domain)

    cb_bad = HessianSpectrumCallback(k=2, n_iter=4, interval=1, constraints=[u.mse])
    with pytest.raises(ValueError, match="constraint .* not found"):
        solver.solve(1, callbacks=[cb_bad])


# ---------------------------------------------------------------------------
# Live-value channel (tracker.value / tracker.latest_epoch)
# ---------------------------------------------------------------------------


def test_tracker_value_starts_none_then_populates_after_first_interval():
    """tracker.value is None before any computation, then a dict with the
    advertised keys after the first interval fires."""
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    key = jax.random.PRNGKey(0)

    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    u_net.optimizer(optax.sgd, lr=lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)

    solver = jno.core([u.mse], domain)

    gn = jno.trackers.gradient_norms(interval=2)
    ntk = jno.trackers.ntk_spectrum(u.grad(u_net), n_points=4, top_k=2, interval=2)
    res = jno.trackers.residual_stats(interval=2)

    # Before solve(): both trackers are unfired.
    assert gn.value is None and gn.latest_epoch is None
    assert ntk.value is None
    assert res.value is None

    # One outer step: interval=2 hasn't fired yet (epoch 0 fires only when
    # 0 % 2 == 0 ✓ — so it DOES fire at epoch 0). Verify after solve.
    solver.solve(2, callbacks=[gn, ntk, res])

    assert gn.value is not None
    assert set(gn.value.keys()) == {"norms"}
    assert gn.value["norms"].shape == (1,)
    assert gn.latest_epoch is not None

    assert ntk.value is not None
    assert {"eigvals_topk", "lambda_max", "trace", "condition_number"}.issubset(ntk.value.keys())
    assert ntk.value["eigvals_topk"].shape == (2,)

    assert res.value is not None
    assert {"means", "stds", "maxes", "p99", "indices"}.issubset(res.value.keys())


# ---------------------------------------------------------------------------
# Tracker-driven weight schemes
# ---------------------------------------------------------------------------


def test_gradient_norm_balanced_returns_uniform_then_inverse_of_norms():
    """GradientNormBalanced returns ones() before the tracker fires and
    weights inversely proportional to tracker.value['norms'] afterwards."""
    import numpy as _np

    from jno.utils.adaptive.weights import GradientNormBalanced

    class _StubTracker:
        value = None

    tracker = _StubTracker()
    w = GradientNormBalanced(tracker)

    # Cold start — no value yet → uniform weights.
    w0, w1 = w(jnp.array(0.5), jnp.array(0.7))
    _np.testing.assert_allclose(_np.asarray([w0, w1]), _np.array([1.0, 1.0]), atol=1e-6)

    # Populate the tracker with known norms.
    tracker.value = {"norms": _np.array([1.0, 4.0], dtype=_np.float32)}
    w0, w1 = w(jnp.array(0.5), jnp.array(0.7))
    out = _np.asarray([w0, w1], dtype=_np.float32)
    # inv = [1.0, 0.25] → normalised to sum=2 → [1.6, 0.4]
    _np.testing.assert_allclose(out, _np.array([1.6, 0.4], dtype=_np.float32), atol=1e-5)


def test_ntk_balanced_returns_uniform_until_all_trackers_fire():
    """NTKBalanced waits until every tracker has a 'trace' value."""
    import numpy as _np

    from jno.utils.adaptive.weights import NTKBalanced

    class _StubTracker:
        def __init__(self):
            self.value = None

    t0, t1 = _StubTracker(), _StubTracker()
    w = NTKBalanced([t0, t1], ema=0.0)  # ema=0 ⇒ no smoothing, use latest

    # Both unset → uniform.
    out = _np.asarray(w(jnp.array(0.1), jnp.array(0.2)))
    _np.testing.assert_allclose(out, _np.array([1.0, 1.0]), atol=1e-6)

    # Only one set → still uniform.
    t0.value = {"trace": 1.0}
    out = _np.asarray(w(jnp.array(0.1), jnp.array(0.2)))
    _np.testing.assert_allclose(out, _np.array([1.0, 1.0]), atol=1e-6)

    # Both set → w_i = tr(K_total) / tr(K_i), re-normalised to sum to N.
    # traces = [1.0, 4.0]; total = 5.0; raw = [5, 1.25]; sum=6.25; *2/6.25 = [1.6, 0.4]
    t1.value = {"trace": 4.0}
    out = _np.asarray(w(jnp.array(0.1), jnp.array(0.2)))
    _np.testing.assert_allclose(out, _np.array([1.6, 0.4], dtype=_np.float32), atol=1e-5)


def test_ntk_balanced_end_to_end_against_live_solver():
    """Wire an NTK tracker + NTKBalanced through crux.solve() and verify
    the scheme emits non-uniform weights once the tracker has fired."""
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn
    from jno import LearningRateSchedule as lrs

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    key = jax.random.PRNGKey(0)

    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    u_net.optimizer(optax.sgd, lr=lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)

    # Two trackers — different network outputs so their traces differ
    # measurably. We use u and u*u as the scalar projections.
    ntk_a = jno.trackers.ntk_spectrum(u.grad(u_net), n_points=4, top_k=2, interval=1)
    ntk_b = jno.trackers.ntk_spectrum((u * 2.0).grad(u_net), n_points=4, top_k=2, interval=1)

    solver = jno.core([u.mse, (u * 2.0).mse], domain)
    solver.solve(2, callbacks=[ntk_a, ntk_b])

    # Both trackers should have fired.
    assert ntk_a.value is not None and ntk_b.value is not None
    # Traces differ by a factor proportional to the chain rule (scale on u → factor 4 on K).
    assert ntk_a.value["trace"] > 0.0
    assert ntk_b.value["trace"] > 0.0

    # Now use the trackers to drive a weight scheme. Disable EMA so the
    # cold-start uniform doesn't dominate.
    from jno.utils.adaptive.weights import ntk_balanced

    w = ntk_balanced([ntk_a, ntk_b], ema=0.0)
    w0, w1 = w(jnp.array(0.1), jnp.array(0.2))
    import numpy as _np

    out = _np.asarray([w0, w1], dtype=_np.float32)
    # Weights must sum to N=2 (within tolerance).
    _np.testing.assert_allclose(out.sum(), 2.0, atol=1e-4)
    # And non-uniform (traces differ).
    assert not _np.allclose(out, _np.array([1.0, 1.0]))
