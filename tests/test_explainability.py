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
