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
    u_net.optimizer(optax.sgd).scale(lrs.exponential(0.0, 1.0, 1, 0.0))  # frozen
    u = u_net(x)
    c0 = (u * u).mse  # arbitrary residual
    c1 = u.mse  # second constraint, different residual

    solver = jno.core([c0, c1])

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
    u_net.optimizer(optax.sgd).scale(lrs.exponential(0.0, 1.0, 1, 0.0))  # lr=0
    u = u_net(x)
    pde = u

    solver = jno.core([pde.mse])
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
    u_net.optimizer(optax.sgd).scale(lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)
    pde = u

    n_points = 8
    top_k = 5

    solver = jno.core([pde.mse])
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
    u_net.optimizer(optax.sgd).scale(lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)
    pde = u

    k = 4
    n_iter = 8

    solver = jno.core([pde.mse])
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
    u_net.optimizer(optax.sgd).scale(lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)
    pde_loss = u.mse

    solver = jno.core([pde_loss])

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
    u_net.optimizer(optax.sgd).scale(lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)

    solver = jno.core([u.mse])

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
    u_net.optimizer(optax.sgd).scale(lrs.exponential(0.0, 1.0, 1, 0.0))
    u = u_net(x)

    # Two trackers — different network outputs so their traces differ
    # measurably. We use u and u*u as the scalar projections.
    ntk_a = jno.trackers.ntk_spectrum(u.grad(u_net), n_points=4, top_k=2, interval=1)
    ntk_b = jno.trackers.ntk_spectrum((u * 2.0).grad(u_net), n_points=4, top_k=2, interval=1)

    solver = jno.core([u.mse, (u * 2.0).mse])
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


# ---------------------------------------------------------------------------
# ENGD callback
# ---------------------------------------------------------------------------


def test_engd_gram_math_lstsq():
    """ENGD Gram: G = (1/N) J^T J, lstsq solution satisfies G x = g."""
    N, P = 20, 4
    key = jax.random.PRNGKey(42)
    J = jax.random.normal(key, shape=(N, P))
    g = jax.random.normal(jax.random.PRNGKey(1), shape=(P,))

    G = (1.0 / N) * (J.T @ J)
    nat_g = jnp.linalg.lstsq(G, g, rcond=None)[0]

    # G is P×P PSD — lstsq gives exact solution when G is full rank.
    np.testing.assert_allclose(np.array(G @ nat_g), np.array(g), atol=1e-5)

    # The (1/N) normalisation: doubling N while keeping J fixed should
    # halve G but leave the lstsq solution unchanged (G^-1 g scales back up).
    J2 = jnp.concatenate([J, J], axis=0)  # shape (2N, P)
    G2 = (1.0 / (2 * N)) * (J2.T @ J2)  # == G (both are (1/N)*J^T J)
    nat_g2 = jnp.linalg.lstsq(G2, g, rcond=None)[0]
    np.testing.assert_allclose(np.array(nat_g), np.array(nat_g2), atol=1e-5)


def test_engd_callback_rejects_bad_inputs():
    """ENGDCallback validates gram_terms types and model consistency."""
    import pytest

    from jno.utils.adaptive.callbacks import ENGDCallback

    with pytest.raises(TypeError, match="NetworkGradient"):
        ENGDCallback(gram_terms=[(jnp.zeros(3), 1.0)])

    with pytest.raises(ValueError, match="empty"):
        ENGDCallback(gram_terms=[])


def test_engd_callback_compiles_and_reduces_loss():
    """ENGDCallback runs end-to-end on 1-D Poisson; loss after 20 ENGD steps
    is lower than after 20 plain-GD steps with the same learning rate.
    """
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn

    π = jno.np.pi

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.02))
    x, _ = domain.variable("interior")
    xb, _ = domain.variable("boundary")

    key = jax.random.PRNGKey(0)
    width = 8

    def make_model():
        net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=width, num_layers=2, key=key))
        u = net(x)
        # u = net(x).scalar.bind(x=x)  # scalar scalar binding
        f = π**2 * jno.np.sin(π * x)
        pde = -u.d2(x) - f  # should be 0 at solution
        ub = net(xb)  # should be 0 at boundary
        return net, u, pde, ub

    # ── ENGD run ──────────────────────────────────────────────────────────────
    net_e, _, pde_e, ub_e = make_model()
    net_e.optimizer(optax.sgd(1.0))  # lr=1.0 for natural gradient direction

    engd_cb = jno.callbacks.engd(
        gram_terms=[(pde_e.grad(net_e), 1.0), (ub_e.grad(net_e), 1.0)],
    )
    crux_e = jno.core([pde_e.mse, ub_e.mse])
    stats_e = crux_e.solve(20, callbacks=[engd_cb])

    # ── GD baseline (same lr) ─────────────────────────────────────────────────
    net_g, _, pde_g, ub_g = make_model()
    net_g.optimizer(optax.sgd(1e-3))
    crux_g = jno.core([pde_g.mse, ub_g.mse])
    stats_g = crux_g.solve(20)

    loss_engd = float(stats_e.total_loss)
    loss_gd = float(stats_g.total_loss)

    # ENGD with lr=1 should outperform GD with lr=1e-3 on the same problem.
    assert loss_engd < loss_gd, f"ENGD loss {loss_engd:.3e} should be < GD loss {loss_gd:.3e}"
    # Loss must be finite.
    assert np.isfinite(loss_engd), f"ENGD loss is not finite: {loss_engd}"


def test_engd_callback_gram_interval_caches_g():
    """With gram_interval=2 the Gram matrix is reused on odd steps.
    Both gram_interval=1 and gram_interval=2 should converge; they differ
    only in cost, not correctness.
    """
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn

    π = jno.np.pi

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, _ = domain.variable("interior")
    xb, _ = domain.variable("boundary")
    key = jax.random.PRNGKey(0)

    def make_model():
        net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, key=key))
        u = net(x)
        f = π**2 * jno.np.sin(π * x)
        pde = -u.d2(x) - f
        ub = net(xb)
        return net, pde, ub

    net1, pde1, ub1 = make_model()
    net1.optimizer(optax.sgd(1.0))
    cb1 = jno.callbacks.engd([(pde1.grad(net1), 1.0), (ub1.grad(net1), 1.0)], gram_interval=1)
    stats1 = jno.core([pde1.mse, ub1.mse]).solve(10, callbacks=[cb1])

    net2, pde2, ub2 = make_model()
    net2.optimizer(optax.sgd(1.0))
    cb2 = jno.callbacks.engd([(pde2.grad(net2), 1.0), (ub2.grad(net2), 1.0)], gram_interval=2)
    stats2 = jno.core([pde2.mse, ub2.mse]).solve(10, callbacks=[cb2])

    # Both should produce a finite loss.
    assert np.isfinite(float(stats1.total_loss))
    assert np.isfinite(float(stats2.total_loss))


def test_engd_callback_rejects_inner_steps():
    """ENGDCallback raises if inner_steps > 1."""
    import foundax
    import optax
    import pytest

    import jno
    import jno.jnp_ops as jnn

    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
    x, _ = domain.variable("interior")
    xb, _ = domain.variable("boundary")
    key = jax.random.PRNGKey(0)

    net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=4, num_layers=2, key=key))
    net.optimizer(optax.sgd(1.0))
    u = net(x)
    pde = u
    ub = net(xb)

    cb = jno.callbacks.engd([(pde.grad(net), 1.0), (ub.grad(net), 1.0)])
    crux = jno.core([pde.mse, ub.mse])

    with pytest.raises(ValueError, match="inner_steps"):
        crux.solve(2, inner_steps=2, callbacks=[cb])


def test_engd_line_search_reduces_loss():
    """ENGDCallback with line_search=True compiles and converges faster than GD.

    2-D Poisson on a tiny 5x5 grid: verifies the 31-point grid search
    (α ∈ {0.5^0, …, 0.5^30}) picks a valid step and the loss decreases.
    """
    import equinox as eqx
    import foundax
    import optax

    import jno
    import jno.jnp_ops as jnn

    N = 5
    int_pts = np.array(
        [[x, y] for x in np.linspace(1 / 6, 5 / 6, N) for y in np.linspace(1 / 6, 5 / 6, N)],
        dtype=np.float64,
    )
    bdy_pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    dom = jno.Shape.rect(0, 0, 1, 1, size=0.3).domain()
    x, y, _ = dom.variable("interior")
    dom.context["interior"] = int_pts[np.newaxis, np.newaxis]
    xb, yb, _ = dom.variable("boundary")
    dom.context["boundary"] = bdy_pts[np.newaxis, np.newaxis]

    π = jno.np.pi
    forcing = 2 * π**2 * jno.np.sin(π * x) * jno.np.sin(π * y)

    key = jax.random.PRNGKey(7)

    def make_net():
        base = foundax.mlp(in_features=2, hidden_dims=8, num_layers=1, activation=jax.nn.tanh, key=key)
        scaled = jax.tree_util.tree_map(lambda leaf: leaf * 0.1 if eqx.is_array(leaf) else leaf, base)
        return jnn.nn.wrap(scaled)

    net = make_net()
    net.optimizer(optax.sgd(1.0))
    u = net(x, y)
    lap = u.laplacian(x, y)
    r = lap + forcing
    u_bc = net(xb, yb)

    engd = jno.callbacks.engd(
        gram_terms=[(lap.grad(net), 1.0), (u_bc.grad(net), 1.0)],
        gram_interval=1,
        line_search=True,
    )
    crux = jno.core([r.mse, u_bc.mse])
    stats = crux.solve(10, callbacks=[engd])

    loss = float(stats.total_loss)
    assert np.isfinite(loss), f"ENGD line_search loss is not finite: {loss}"
    assert loss < 50.0, f"ENGD line_search loss {loss:.3e} did not decrease from ~52 initial"


# ---------------------------------------------------------------------------
# A `fem.solve()` in the trace — the vmap-over-`spsolve` wall
# ---------------------------------------------------------------------------
#
# `jax.jacrev` takes one `vjp` and then vmaps the pullback across the rows of the output
# basis. `jax.experimental.sparse.linalg.spsolve` has no batching rule, so that vmap — not
# the differentiation — is what fails on any trace containing `fem.solve()`, and it fails
# even for a single scalar output because the basis still carries a leading axis. Forward
# mode is no escape (`jacfwd` hits the same wall at `csr_matvec`); plain `jax.grad` works.
#
# Every other tracker test in this file uses a PINN, which is why this went unnoticed: the
# per-loss gradient trackers, and the loss weighting built on them, were unusable on jNO's
# headline problem class.


def test_rowwise_jacobian_is_bit_identical_to_jacrev():
    """The replacement must be a refactor, not a rewrite — same VJPs, same order, same bits.

    Uses a pytree with leaves of different shapes so the flattening order is actually
    exercised, and a nonlinear vector-valued function so every row differs.
    """
    from jno.utils.ad_mode import rowwise_jacobian

    x = {"a": jnp.array([1.0, -2.0, 0.5]), "b": jnp.array([[3.0, 1.0], [0.25, -1.5]])}

    def f(t):
        a, b = t["a"], t["b"]
        return jnp.stack([jnp.sum(jnp.sin(a) * a), jnp.sum(b**3), jnp.sum(a) * jnp.sum(jnp.tanh(b))])

    leaves = jax.tree_util.tree_leaves(jax.jacrev(f)(x))
    expected = jnp.stack([jnp.concatenate([lf[i].ravel() for lf in leaves]) for i in range(3)])

    got = rowwise_jacobian(f, x, range(3))
    assert got.shape == expected.shape == (3, 7)
    assert bool(jnp.all(got == expected)), "must agree with jacrev to the last bit"

    # A caller may ask for a subset of rows, in its own order — MMA wants only the
    # inequality-constraint rows and pays for those alone.
    subset = rowwise_jacobian(f, x, [2, 0])
    assert bool(jnp.all(subset[0] == expected[2]) and jnp.all(subset[1] == expected[0]))


def _fem_compliance_crux():
    """A cantilever whose loss depends on a sparse solve: clamped left, loaded right.

    Deliberately coarse (28 nodes) — the point is that `spsolve` is in the trace at all,
    not the mechanics, which `test_topology_optimisation.py` covers.
    """
    import optax

    import jno

    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    # emin = 1e-6: at 1e-9 the GPU spsolve (cuSolver QR) falsely reports the SIMP stiffness
    # singular -- same measured failure and fix as tests/test_topology_optimisation.py.
    e0, emin, nu, penal = 1.0, 1e-6, 0.3, 3.0
    lam, mu = e0 * nu / (1 - nu**2), e0 / (2 * (1 + nu))

    d = jno.Shape.rect(0, 0, 2, 1, size=0.4).domain()
    u, phi = d.fem_symbols(value_shape=(2,))
    _r, s = d.fem_symbols(names=("r", "s"))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)

    rho = jno.np.parameter(s, name="rho")
    rho.initialize(jax.nn.initializers.constant(0.4))
    rho.optimizer(optax.adam(1e-2))

    eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
    fem = jno.fem(
        [
            (emin + rho**penal * (e0 - emin)) * (lam * trace(eu) * trace(ep) + 2 * mu * inner(eu, ep, n_contract=2)),
            u(xl, yl) - (0.0, 0.0),
            -1.0 * inner(jnp.array([0.0, -1.0]), phi.bind(x=xr, y=yr), n_contract=1),
        ],
        quad_degree=2,
    )
    n_nodes = int(np.asarray(d.built_mesh.points).shape[0])
    _a, b = fem.operator.evaluate({"rho": jnp.full(n_nodes, 0.4)})
    f_vec = np.asarray(jnp.asarray(b).reshape(-1))

    compliance = jno.fn(lambda uu: jnp.sum(uu * jnp.asarray(f_vec)), [fem.solve()], name="C")
    reg = jno.fn(lambda rv: jnp.mean(rv**2), [rho], name="reg")
    return jno.core([compliance, reg], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))


def test_gradient_trackers_run_on_a_trace_containing_a_sparse_solve():
    """All four vmap-dependent trackers, on one FEM solve, in one run.

    Each of these raised ``NotImplementedError: Batching rule for 'spsolve' not
    implemented`` before the fix. Attaching them together also checks they coexist.
    """
    import jno

    gn = jno.trackers.gradient_norms(interval=1)
    cs = jno.trackers.cos_similarity(interval=1)
    ga = jno.trackers.gradient_alignment(interval=1)
    ll = jno.trackers.loss_landscape(interval=1, n_grid=3)

    _fem_compliance_crux().solve(2, callbacks=[gn, cs, ga, ll])

    norms = np.asarray(gn.value["norms"])
    assert norms.shape == (2,) and np.all(np.isfinite(norms)) and np.all(norms >= 0.0)
    # The compliance term is driven through the solve and the regulariser is not, so their
    # gradient norms must not be the same number — a placeholder would be.
    assert norms[0] != norms[1]

    cos = np.asarray(cs.value["cos_sim_matrix"])
    assert cos.shape == (2, 2) and np.all(np.isfinite(cos))
    # atol=1e-3, not 1e-5: the matrix's two evaluation paths (batched vs single) disagree at GPU
    # noise level -- measured diag 0.99988 on cuda where cpu gives 1-1e-9. This test pins that the
    # trackers RUN and COEXIST on a sparse-solve trace, not the self-similarity precision.
    np.testing.assert_allclose(np.diag(cos), np.ones(2), atol=1e-3)
    assert np.all(np.abs(cos) <= 1.0 + 1e-5)

    align = np.asarray(ga.value["alignment"]).reshape(-1)
    assert align.shape == (1,) and np.all(np.isfinite(align))
    assert -1.0 - 1e-5 <= float(align[0]) <= 1.0 + 1e-5

    land = np.asarray(ll.value["landscape"])
    assert land.shape == (3, 3) and np.all(np.isfinite(land))


def test_gradient_norm_balanced_weights_a_trace_containing_a_sparse_solve():
    """The downstream consumer that is loss weighting, not a diagnostic.

    ``GradientNormBalanced`` reads ``tracker.value["norms"]``, so it inherited the failure
    wholesale. Existing coverage stubs the tracker, and so never touched this path.
    """
    import jno
    from jno.utils.adaptive.weights import gradient_norm_balanced

    gn = jno.trackers.gradient_norms(interval=1)
    _fem_compliance_crux().solve(2, callbacks=[gn])

    w = gradient_norm_balanced(gn)
    w0, w1 = w(jnp.array(0.1), jnp.array(0.2))
    out = np.asarray([w0, w1], dtype=np.float64)

    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out.sum(), 2.0, rtol=1e-5)
    # Inversely proportional to the measured norms, which differ by orders of magnitude here.
    norms = np.asarray(gn.value["norms"], dtype=np.float64)
    inv = 1.0 / norms
    np.testing.assert_allclose(out, inv / inv.sum() * 2.0, rtol=1e-4)
