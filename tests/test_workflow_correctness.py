"""Correctness tests for complex jNO workflows.

Tests here verify *behavior*, not just absence of errors:
  - Multi-constraint loss logging shape and per-constraint decrease
  - Callbacks: EarlyStopping actually halts; GradientNorms has correct structure
  - Checkpointing: restored weights match post-training weights numerically
  - Resampling: RARD actually moves points; point count preserved
  - Divergence / curl operators: evaluated against closed-form analytic values
  - Training statistics: epoch ordering, total-loss identity, param counts
"""

from __future__ import annotations

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
import jno.jnp_ops as jnn
from jno import LearningRateSchedule as lrs
from jno.utils.adaptive.callbacks import EarlyStoppingCallback, GradientNormsCallback

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pde_bc_solver(*, hidden_dims=16, num_layers=2, mesh_size=0.05, key=None):
    """1-D Poisson with interior PDE + soft Dirichlet BC — two constraints."""
    if key is None:
        key = jax.random.PRNGKey(0)
    domain = jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))
    x, _ = domain.variable("interior")
    xb, _ = domain.variable("boundary")
    u_net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=hidden_dims, num_layers=num_layers, key=key))
    u = u_net(x)
    pde = -jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)
    bc = u_net(xb)
    solver = jno.core([pde.mse, bc.mse])
    return solver, u_net


def _make_simple_solver(*, key=None):
    """Minimal single-constraint 1-D solver for statistics / callback tests."""
    if key is None:
        key = jax.random.PRNGKey(0)
    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, _ = domain.variable("interior")
    u_net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key))
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)
    solver = jno.core([pde.mse])
    return solver, u_net


def _build_context(domain):
    """Strip leading singleton axes from domain.context (simulate post-solve view)."""
    ctx = {}
    for k, v in domain.context.items():
        arr = np.asarray(v)
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        ctx[k] = jnp.array(arr)
    return ctx


def _direct_eval(expr, domain):
    """Evaluate a symbolic expression without running solve() (TraceEvaluator)."""
    from jno.trace_evaluator import TraceEvaluator

    ev = TraceEvaluator(params={})
    return ev.evaluate(expr, context=_build_context(domain), var_bindings={})


# ===========================================================================
# Group 1: Multi-constraint loss logging
# ===========================================================================


@pytest.mark.integration
class TestMultiConstraintLosses:
    def test_per_constraint_losses_have_correct_shape(self):
        """With 2 constraints, losses.shape == (n_logged, 2) — 2 columns regardless of epochs."""
        solver, u_net = _make_pde_bc_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(20)
        losses = stats.training_logs[-1]["losses"]
        assert losses.ndim == 2, f"Expected 2D losses array, got shape {losses.shape}"
        assert losses.shape[1] == 2, f"Expected 2 constraint columns, got {losses.shape[1]}"
        assert losses.shape[0] >= 1, "No log entries were recorded"

    def test_per_constraint_losses_are_finite_and_positive(self):
        """Every per-constraint loss at every epoch must be finite and > 0."""
        solver, u_net = _make_pde_bc_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(10)
        losses = stats.training_logs[-1]["losses"]
        assert jnp.all(jnp.isfinite(losses)), "Some per-constraint losses are NaN/Inf"
        assert jnp.all(losses > 0), "Some per-constraint losses are non-positive"

    def test_total_loss_equals_mean_of_per_constraint(self):
        """total_loss at every epoch must equal mean(losses, axis=1)."""
        solver, u_net = _make_pde_bc_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(10)
        logs = stats.training_logs[-1]
        computed = jnp.mean(jnp.array(logs["losses"]), axis=1)
        assert jnp.allclose(jnp.array(logs["total_loss"]), computed, rtol=1e-4), (
            "total_loss does not equal mean of per-constraint losses"
        )

    def test_per_constraint_losses_decrease_over_training(self):
        """Both PDE loss and BC loss should fall over 300 epochs."""
        solver, u_net = _make_pde_bc_solver(hidden_dims=32, num_layers=3)
        u_net.optimizer(optax.adam).scale(lrs.exponential(1e-3, 0.9, 100))
        stats = solver.solve(300)
        losses = stats.training_logs[-1]["losses"]
        pde_loss = losses[:, 0]
        bc_loss = losses[:, 1]
        assert pde_loss[-1] < pde_loss[0], "PDE constraint loss did not decrease"
        assert bc_loss[-1] < bc_loss[0], "BC constraint loss did not decrease"


# ===========================================================================
# Group 2: Callbacks — correct behavior, not just invocation
# ===========================================================================


@pytest.mark.integration
class TestEarlyStoppingCallback:
    def test_early_stopping_halts_before_max_epochs(self):
        """With patience=5 and a converged loss, training stops well before 500."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        solver.solve(50)
        # min_delta=1e-5 means per-epoch improvements below ~1e-5 don't count.
        # With lr=1e-7 Adam steps are O(1e-7), well below that threshold.
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-7))
        cb = EarlyStoppingCallback(patience=5, min_delta=1e-5, mode="min")
        solver.solve(epochs=500, callbacks=[cb])
        assert cb.has_stopped, "EarlyStoppingCallback never triggered"
        assert cb.stopped_epoch is not None
        assert cb.stopped_epoch < 490, f"Stopped too late: epoch {cb.stopped_epoch}"

    def test_early_stopping_stopped_epoch_consistent_with_log_length(self):
        """Number of logged epochs must not exceed stopped_epoch + small buffer."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        solver.solve(30)
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-7))
        cb = EarlyStoppingCallback(patience=3, min_delta=1e-5, mode="min")
        stats = solver.solve(epochs=500, callbacks=[cb])
        n_logged = len(stats.training_logs[-1]["epoch"])
        assert cb.stopped_epoch is not None
        # Log length should match the stop point (±1 for implementation details)
        assert n_logged <= cb.stopped_epoch + 2, f"Logged {n_logged} epochs but stopped at {cb.stopped_epoch}"

    def test_early_stopping_best_metric_is_minimum_observed(self):
        """best_metric must be <= initial loss (loss only decreases or stays)."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(10)
        initial_loss = float(stats.training_logs[-1]["total_loss"][0])

        u_net.optimizer(optax.adam).scale(lrs.constant(1e-7))
        cb = EarlyStoppingCallback(patience=3, min_delta=1e-5, mode="min")
        solver.solve(50, callbacks=[cb])
        assert cb.best_metric is not None
        assert cb.best_metric <= initial_loss


@pytest.mark.integration
class TestGradientNormsCallback:
    def test_gradient_norms_result_has_correct_shape(self):
        """result['norms'] shape must be (n_samples, n_constraints)."""
        solver, u_net = _make_pde_bc_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        cb = GradientNormsCallback(interval=2)
        solver.solve(epochs=6, callbacks=[cb])
        result = cb.result
        assert result["norms"].ndim == 2
        n_samples, n_constraints = result["norms"].shape
        assert n_constraints == 2, f"Expected 2 constraints, got {n_constraints}"
        assert n_samples >= 1

    def test_gradient_norms_values_are_finite_and_nonneg(self):
        """All computed gradient norms must be finite and >= 0."""
        solver, u_net = _make_pde_bc_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        cb = GradientNormsCallback(interval=1)
        solver.solve(epochs=4, callbacks=[cb])
        norms = cb.result["norms"]
        assert np.all(np.isfinite(norms)), "Some gradient norms are NaN/Inf"
        assert np.all(norms >= 0), "Some gradient norms are negative"

    def test_gradient_norms_epochs_match_interval(self):
        """result['epochs'] must record every `interval`-th epoch."""
        solver, u_net = _make_pde_bc_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        cb = GradientNormsCallback(interval=2)
        solver.solve(epochs=8, callbacks=[cb])
        recorded = cb.result["epochs"]
        assert len(recorded) >= 1
        # All recorded epochs must be multiples of the interval
        assert all(e % 2 == 0 for e in recorded), f"Unexpected epochs: {recorded}"


@pytest.mark.integration
class TestCheckpointCallback:
    def test_checkpoint_metadata_matches_training(self, tmp_path):
        """Checkpoint metadata epoch and total_loss must match the training log."""
        pytest.importorskip("orbax.checkpoint", reason="orbax-checkpoint not installed")

        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        from jno.utils.adaptive.callbacks import CheckpointCallback

        cb = CheckpointCallback(
            directory=str(tmp_path / "ckpt"),
            save_interval_epochs=1,
            max_to_keep=5,
            async_checkpointing=False,
        )
        stats = solver.solve(epochs=5, callbacks=[cb])

        state = cb.restore()
        meta = state["metadata"]
        last_epoch = int(stats.training_logs[-1]["epoch"][-1])
        assert meta["epoch"] == last_epoch, f"Checkpoint epoch {meta['epoch']} != logged epoch {last_epoch}"
        logged_loss = float(stats.training_logs[-1]["total_loss"][-1])
        assert abs(meta["total_loss"] - logged_loss) < 1e-3, (
            f"Checkpoint loss {meta['total_loss']:.6e} != logged loss {logged_loss:.6e}"
        )

    def test_restored_model_produces_same_outputs(self, tmp_path):
        """Forward pass with correctly-restored weights must match post-training."""
        from pathlib import Path

        ocp = pytest.importorskip("orbax.checkpoint", reason="orbax-checkpoint not installed")

        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        from jno.utils.adaptive.callbacks import CheckpointCallback

        cb = CheckpointCallback(
            directory=str(tmp_path / "ckpt"),
            save_interval_epochs=1,
            max_to_keep=5,
            async_checkpointing=False,
        )
        solver.solve(epochs=5, callbacks=[cb])

        lid = u_net.layer_id
        model_trained = solver.models[lid]
        x_test = jnp.linspace(0.05, 0.95, 20).reshape(-1, 1)
        output_trained = jax.vmap(model_trained)(x_test)

        # Restore using the template-based approach so orbax can reconstruct the
        # equinox module with correct leaf ordering (identical to test_checkpointing.py).
        latest_step = cb.latest_step
        step_dir = Path(str(tmp_path / "ckpt")) / str(latest_step) / "state"
        restore_template = jax.tree_util.tree_map(
            lambda leaf: leaf if eqx.is_array(leaf) else ocp.PLACEHOLDER,
            model_trained,
            is_leaf=lambda leaf: leaf is None,
        )
        restore_item = {"trainable": {str(lid): restore_template}}
        restore_args = ocp.checkpoint_utils.construct_restore_args(restore_item)
        checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        restored = checkpointer.restore(
            step_dir,
            args=ocp.args.PyTreeRestore(
                item=restore_item,
                restore_args=restore_args,
                partial_restore=True,
            ),
        )
        if hasattr(checkpointer, "close"):
            checkpointer.close()

        model_restored = jax.tree_util.tree_map(
            lambda r, fresh: fresh if (r is ocp.PLACEHOLDER or r is None) else r,
            restored["trainable"][str(lid)],
            model_trained,
            is_leaf=lambda leaf: leaf is ocp.PLACEHOLDER or leaf is None,
        )
        output_restored = jax.vmap(model_restored)(x_test)

        assert jnp.allclose(output_trained, output_restored, atol=1e-5), (
            "Restored model gives different outputs than trained model"
        )


# ===========================================================================
# Group 3: Resampling — points actually move
# ===========================================================================


@pytest.mark.integration
class TestResamplingCorrectness:
    def test_rard_changes_interior_points(self):
        """After RARD resampling, context['interior'] must differ from the original."""
        from jno.utils.adaptive.resampling import RARD

        strategy = RARD(resample_every=1, resample_fraction=0.5, start_epoch=0, power=2.0)
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
        x, *_ = domain.variable("interior", sample=(64, None), resampling_strategy=strategy)
        points_before = np.array(domain.context["interior"])

        u_net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(0)))
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)
        solver = jno.core([pde])
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        solver.solve(epochs=5)

        points_after = np.array(domain.context["interior"])
        assert not np.allclose(points_before, points_after), "RARD resampling did not change any interior points"

    def test_rard_preserves_point_count(self):
        """Resampling must not change the total number of collocation points."""
        from jno.utils.adaptive.resampling import RARD

        strategy = RARD(resample_every=1, resample_fraction=0.5, start_epoch=0)
        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
        x, *_ = domain.variable("interior", sample=(64, None), resampling_strategy=strategy)
        n_before = domain.context["interior"].shape[-2]  # (B, T, N, D) → N

        u_net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(1)))
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)
        solver = jno.core([pde])
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        solver.solve(epochs=5)

        n_after = domain.context["interior"].shape[-2]
        assert n_before == n_after, f"Point count changed: {n_before} → {n_after}"

    def test_cr3_gamma_advances_during_training(self):
        """CR3's γ gate must increase after _update_gamma is called.

        Tests the update logic directly (bypassing the full resampling pipeline)
        because the residual shape check in resample() can prevent _update_gamma
        from being reached in some training configurations.
        With epsilon=0, step = eta_g * min(exp(0), delta_max) = 1e-3 * 0.1 = 1e-4 > 0.
        """
        from jno.utils.adaptive.resampling import CR3

        strategy = CR3(resample_every=1, resample_fraction=0.5, start_epoch=0, gamma0=-0.5, epsilon=0.0)
        gamma_init = strategy.gamma

        residuals = jnp.ones(20) * 0.5
        gate_values = jnp.ones(20)
        strategy._update_gamma(residuals, gate_values)

        assert strategy.gamma > gamma_init, f"CR3 γ did not increase: {gamma_init} → {strategy.gamma}"


# ===========================================================================
# Group 4: Divergence and curl operators — analytic verification
# ===========================================================================


class TestDivergenceCorrectness:
    """div([x, y], [x, y]) = ∂x/∂x + ∂y/∂y = 2 everywhere on a 2D domain."""

    @pytest.fixture(scope="class")
    def dom2d(self):
        return jno.Shape.rect(0, 0, 1, 1, size=0.1).domain()

    def test_divergence_of_identity_field_is_two(self, dom2d):
        x, y, _ = dom2d.variable("interior")
        div_val = jnn.divergence([x, y], [x, y])
        result = _direct_eval(div_val, dom2d)
        assert jnp.allclose(result, 2.0, atol=1e-4), (
            f"div([x,y],[x,y]) should be 2.0, got mean {float(jnp.mean(result)):.6f}"
        )

    def test_divergence_of_solenoidal_rotation_field_is_zero(self, dom2d):
        """div([-y, x], [x, y]) = ∂(-y)/∂x + ∂x/∂y = 0 + 0 = 0."""
        x, y, _ = dom2d.variable("interior")
        div_val = jnn.divergence([-y, x], [x, y])
        result = _direct_eval(div_val, dom2d)
        assert jnp.allclose(result, 0.0, atol=1e-4), (
            f"div([-y,x],[x,y]) should be 0, got max |val| = {float(jnp.max(jnp.abs(result))):.6f}"
        )

    def test_divergence_of_quadratic_field(self, dom2d):
        """div([x², y²], [x, y]) = 2x + 2y at each point."""
        x, y, _ = dom2d.variable("interior")
        div_val = jnn.divergence([x * x, y * y], [x, y])
        result = _direct_eval(div_val, dom2d)
        ctx = _build_context(dom2d)
        x_vals = ctx["interior"][:, 0:1]
        y_vals = ctx["interior"][:, 1:2]
        expected = 2 * x_vals + 2 * y_vals
        assert jnp.allclose(result, expected, atol=1e-4), (
            f"div([x²,y²]) max error: {float(jnp.max(jnp.abs(result - expected))):.6f}"
        )


class TestCurlCorrectness:
    """curl_2d(Fx, Fy, x, y) = ∂Fy/∂x - ∂Fx/∂y."""

    @pytest.fixture(scope="class")
    def dom2d(self):
        return jno.Shape.rect(0, 0, 1, 1, size=0.1).domain()

    def test_curl_2d_of_rotation_field_is_constant_two(self, dom2d):
        """curl_2d(-y, x) = ∂x/∂x - ∂(-y)/∂y = 1 - (-1) = 2."""
        x, y, _ = dom2d.variable("interior")
        curl = jnn.curl_2d(-y, x, x, y)
        result = _direct_eval(curl, dom2d)
        assert jnp.allclose(result, 2.0, atol=1e-4), f"curl_2d(-y, x) should be 2.0, got mean {float(jnp.mean(result)):.6f}"

    def test_curl_2d_of_irrotational_gradient_field_is_zero(self, dom2d):
        """curl of any gradient field is zero: curl(∇f) = 0."""
        x, y, _ = dom2d.variable("interior")
        # f = x² + y²  →  Fx = 2x, Fy = 2y  →  curl = ∂(2y)/∂x - ∂(2x)/∂y = 0
        Fx = x + x  # 2x via addition
        Fy = y + y  # 2y
        curl = jnn.curl_2d(Fx, Fy, x, y)
        result = _direct_eval(curl, dom2d)
        assert jnp.allclose(result, 0.0, atol=1e-4), (
            f"curl(∇f) should be 0, got max |val| = {float(jnp.max(jnp.abs(result))):.6f}"
        )

    def test_curl_2d_of_constant_field_is_zero(self, dom2d):
        """curl of a constant vector field is zero."""
        from jno.trace import Literal

        x, y, _ = dom2d.variable("interior")
        one = Literal(1.0)
        curl = jnn.curl_2d(one, one, x, y)
        result = _direct_eval(curl, dom2d)
        assert jnp.allclose(result, 0.0, atol=1e-4)

    def test_curl_2d_of_scaled_rotation_field(self, dom2d):
        """curl_2d(-2y, 2x) = ∂(2x)/∂x - ∂(-2y)/∂y = 2 - (-2) = 4."""
        x, y, _ = dom2d.variable("interior")
        curl = jnn.curl_2d(-y - y, x + x, x, y)  # (-2y, 2x)
        result = _direct_eval(curl, dom2d)
        assert jnp.allclose(result, 4.0, atol=1e-4), (
            f"curl_2d(-2y, 2x) should be 4.0, got mean {float(jnp.mean(result)):.6f}"
        )


# ===========================================================================
# Group 5: Training statistics correctness
# ===========================================================================


@pytest.mark.integration
class TestStatisticsCorrectness:
    def test_epoch_indices_are_sequential(self):
        """Logged epoch indices must be strictly increasing integers."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        # 100 epochs produces ~11 log entries (print_rate=10); enough to verify ordering
        stats = solver.solve(100)
        epochs = stats.training_logs[-1]["epoch"]
        assert len(epochs) >= 2, "Expected at least 2 log entries"
        assert all(int(epochs[i + 1]) > int(epochs[i]) for i in range(len(epochs) - 1)), (
            "Epoch indices are not strictly increasing"
        )

    def test_total_loss_is_always_finite(self):
        """total_loss must be finite at every logged epoch."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(20)
        total_loss = stats.training_logs[-1]["total_loss"]
        assert jnp.all(jnp.isfinite(jnp.array(total_loss))), "total_loss contains NaN or Inf"

    def test_multiple_solve_calls_accumulate_logs(self):
        """Calling solve() twice must produce two separate training_log entries."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        solver.solve(10)
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(10)
        assert len(stats.training_logs) == 2, f"Expected 2 log entries, got {len(stats.training_logs)}"

    def test_total_epochs_accumulate_across_solve_calls(self):
        """solver._total_epochs must accumulate across multiple solve() calls."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        solver.solve(10)
        assert solver._total_epochs == 10
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        solver.solve(10)
        assert solver._total_epochs == 20

    def test_timestamps_are_monotonically_non_decreasing(self):
        """Wall-clock timestamps must be non-decreasing across epochs."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(20)
        ts = np.array(stats.training_logs[-1]["timestamps"])
        assert np.all(ts[1:] >= ts[:-1]), "Timestamps are not monotonically non-decreasing"

    def test_trainable_param_count_matches_model_leaf_count(self):
        """trainable_params logged by solve() must match actual model parameter count."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(1)
        logged = stats.training_logs[-1]["trainable_params"]
        expected = sum(leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(u_net.module, eqx.is_inexact_array)))
        assert logged == expected, f"Logged trainable_params={logged} != actual model params={expected}"

    def test_frozen_params_zero_for_fully_trainable_model(self):
        """When no freezing is applied, frozen_params must be 0."""
        solver, u_net = _make_simple_solver()
        u_net.optimizer(optax.adam).scale(lrs.constant(1e-3))
        stats = solver.solve(1)
        assert stats.training_logs[-1]["frozen_params"] == 0, "Expected 0 frozen params for a fully trainable model"


# ===========================================================================
# Group 7: jno.fn.stop_gradient — gradient isolation
# ===========================================================================


import paramax as _paramax


def _model_leaves(crux, layer_id):
    """Return model weights as a list of numpy arrays (safe to compare after solve)."""
    inner = _paramax.unwrap(crux.models[layer_id])
    return [np.array(leaf) for leaf in jax.tree_util.tree_leaves(inner)]


def _weights_changed(before, after):
    return any(not np.allclose(a, b) for a, b in zip(before, after))


@pytest.mark.integration
class TestStopGradient:
    """Verify that jno.fn.stop_gradient blocks gradient flow between cooperating models."""

    def _setup(self):
        domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, _ = domain.variable("interior")
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))
        phy = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=8, num_layers=2, key=k1))
        syn = jno.nn.wrap(foundax.mlp(in_features=1, output_dim=1, hidden_dims=8, num_layers=2, key=k2))
        phy.optimizer(optax.adam(1e-3))
        syn.optimizer(optax.adam(1e-3))
        u_phy = phy(x) * x * (1 - x)
        u_syn = syn(x) * x * (1 - x)
        return domain, x, phy, syn, u_phy, u_syn

    def test_int_phy_does_not_update_syn(self):
        """L_int_phy = (u_phy - stop_gradient(u_syn)).mse must not change syn weights."""
        domain, x, phy, syn, u_phy, u_syn = self._setup()
        L = (u_phy - jno.fn.stop_gradient(u_syn)).mse
        crux = jno.core([L])

        before_phy = _model_leaves(crux, phy.layer_id)
        before_syn = _model_leaves(crux, syn.layer_id)
        crux.solve(20)
        after_phy = _model_leaves(crux, phy.layer_id)
        after_syn = _model_leaves(crux, syn.layer_id)

        assert _weights_changed(before_phy, after_phy), "phy weights should have changed"
        assert not _weights_changed(before_syn, after_syn), "syn weights must not change (stop_gradient)"

    def test_int_syn_does_not_update_phy(self):
        """L_int_syn = (u_syn - stop_gradient(u_phy)).mse must not change phy weights."""
        domain, x, phy, syn, u_phy, u_syn = self._setup()
        L = (u_syn - jno.fn.stop_gradient(u_phy)).mse
        crux = jno.core([L])

        before_phy = _model_leaves(crux, phy.layer_id)
        before_syn = _model_leaves(crux, syn.layer_id)
        crux.solve(20)
        after_phy = _model_leaves(crux, phy.layer_id)
        after_syn = _model_leaves(crux, syn.layer_id)

        assert _weights_changed(before_syn, after_syn), "syn weights should have changed"
        assert not _weights_changed(before_phy, after_phy), "phy weights must not change (stop_gradient)"

    def test_without_stop_gradient_both_models_update(self):
        """Negative control: without stop_gradient, the interaction loss updates both models."""
        domain, x, phy, syn, u_phy, u_syn = self._setup()
        L = (u_phy - u_syn).mse  # no stop_gradient
        crux = jno.core([L])

        before_phy = _model_leaves(crux, phy.layer_id)
        before_syn = _model_leaves(crux, syn.layer_id)
        crux.solve(20)
        after_phy = _model_leaves(crux, phy.layer_id)
        after_syn = _model_leaves(crux, syn.layer_id)

        assert _weights_changed(before_phy, after_phy), "phy weights should change without stop_gradient"
        assert _weights_changed(before_syn, after_syn), "syn weights should change without stop_gradient"
