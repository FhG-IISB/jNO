"""Integration tests — end-to-end traces through compile + evaluate."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from jno.trace import (
    Literal,
    Model,
    OperationDef,
    Jacobian,
    Hessian,
    Constant,
    ConstantNamespace,
    BinaryOp,
    FunctionCall,
    collect_operations,
)
from jno.trace_evaluator import TraceEvaluator
from tests.conftest import make_var


# ======================================================================
# Compile & evaluate a full expression graph
# ======================================================================
@pytest.mark.integration
class TestFullPipeline:
    def test_literal_arithmetic_pipeline(self):
        """Compile and evaluate: (2*3 + 4) / 2 = 5."""
        expr = (Literal(2.0) * Literal(3.0) + Literal(4.0)) / Literal(2.0)
        op = OperationDef(expr)
        all_ops = [op]

        compiled = TraceEvaluator.compile_traced_expression(expr, all_ops)
        result = compiled({}, {"x": jnp.ones((1, 1))})
        assert jnp.allclose(result, 5.0)

    def test_variable_expression(self):
        """Evaluate x^2 at x=3."""
        x = make_var("x")
        expr = x ** Literal(2.0)
        op = OperationDef(expr, [x])
        all_ops = [op]

        compiled = TraceEvaluator.compile_traced_expression(expr, all_ops)
        points = {"x": jnp.array([[3.0]])}
        result = compiled({}, points)
        assert jnp.allclose(result, 9.0, atol=0.1)

    def test_neural_network_in_expression(self):
        """Evaluate u_nn(x) where u_nn is a Dense layer."""
        key = jax.random.PRNGKey(0)
        module = eqx.nn.Linear(1, 1, key=key)
        fm = Model(module, name="u_nn")
        x = make_var("x")
        call = fm(x)

        op = OperationDef(call, [x])
        all_ops = [op]

        # The model is already initialized — just put it in the params dict
        layer_params = {fm.layer_id: module}

        compiled = TraceEvaluator.compile_traced_expression(call, all_ops)
        points = {"x": jnp.ones((5, 1))}
        result = compiled(layer_params, points)
        # Linear(1,1) output — check last dim or total elements
        assert result.size >= 1

    def test_gradient_in_pipeline(self):
        """d/dx(x^2) = 2x at x=2.0 using single-variable Jacobian."""
        x = make_var("x")
        u = x ** Literal(2.0)
        du_dx = Jacobian(u, [x], scheme="automatic_differentiation")

        op = OperationDef(du_dx, [x])
        all_ops = [op]

        compiled = TraceEvaluator.compile_traced_expression(du_dx, all_ops)
        points = {"x": jnp.array([[2.0]])}
        result = compiled({}, points)
        assert jnp.allclose(result, 4.0, atol=0.5)

    def test_constant_namespace_in_expression(self):
        """Use constants from a namespace in an expression."""
        ns = ConstantNamespace("params", {"alpha": 2.0, "beta": 3.0})
        x = make_var("x")
        alpha = ns.alpha
        beta = ns.beta
        # alpha * x + beta
        expr = alpha * x + beta

        op = OperationDef(expr, [x])
        all_ops = [op]

        compiled = TraceEvaluator.compile_traced_expression(expr, all_ops)
        points = {"x": jnp.array([[5.0]])}
        result = compiled({}, points)
        # 2.0 * 5.0 + 3.0 = 13.0
        assert jnp.allclose(result, 13.0, atol=0.1)

    def test_multi_variable_expression(self):
        """Evaluate x + y at (x=1, y=2) yielding 3."""
        x = make_var("x")
        y = make_var("y")
        expr = x + y

        op = OperationDef(expr, [x, y])
        all_ops = [op]

        compiled = TraceEvaluator.compile_traced_expression(expr, all_ops)
        points = {"x": jnp.array([[1.0]]), "y": jnp.array([[2.0]])}
        result = compiled({}, points)
        assert jnp.allclose(result, 3.0, atol=0.1)


# ======================================================================
# collect_* on complex trees
# ======================================================================
@pytest.mark.integration
class TestCollectorsIntegration:
    def test_collect_operations_nested(self):
        x = make_var("x")
        inner_expr = x ** Literal(2.0)
        inner_op = OperationDef(inner_expr, [x])
        inner_call = inner_op(x)

        outer_expr = inner_call + Literal(1.0)
        outer_op = OperationDef(outer_expr)

        ops = collect_operations(outer_op)
        # Should find at least the inner OperationDef
        assert len(ops) >= 1


# ======================================================================
# Bug-fix regression tests
# ======================================================================
@pytest.mark.integration
class TestRegressions:
    def test_placeholder_eq_identity_not_symbolic(self):
        """Regression: __eq__ must return bool, not FunctionCall."""
        a = make_var("x")
        b = make_var("y")
        result = a == b
        assert isinstance(result, bool)
        assert result is False

    def test_placeholder_eq_same_object(self):
        a = make_var("x")
        assert (a == a) is True

    def test_weight_schedule_accepts_list(self):
        """Regression: WeightSchedule must accept a list of floats."""
        from jno.utils.adaptive import WeightSchedule

        ws = WeightSchedule([1.0, 2.0])
        result = ws(jnp.array(0), jnp.array([1.0, 1.0]))
        assert result.shape == (2,)

    def test_weight_schedule_accepts_callable(self):
        from jno.utils.adaptive import WeightSchedule

        ws = WeightSchedule(lambda t, L: jnp.array([1.0]))
        result = ws(jnp.array(0), jnp.array([1.0]))
        assert result.shape == (1,)


# ======================================================================
# Memory management strategy tests (solve loop)
# ======================================================================


def _make_solver():
    """Build a minimal core solver for memory-management tests."""
    import jno
    import jno.numpy as jnn

    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.01))
    x, t = domain.variable("interior")

    key = jax.random.PRNGKey(0)
    u_net = jnn.nn.mlp(1, hidden_dims=32, num_layers=2, key=key)
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)

    solver = jno.core([pde.mse], domain)
    return solver, u_net


@pytest.mark.integration
class TestMemoryStrategies:
    """Tests for buffer donation, gradient checkpointing, and data offloading."""

    def test_default_solve(self):
        """Baseline: default solve (no new flags) still works."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(20)

        logs = stats.training_logs[-1]
        assert "epoch" in logs
        assert "losses" in logs
        assert logs["losses"].shape[0] > 0
        # Loss should be a finite number
        assert jnp.isfinite(logs["total_loss"][-1])

    def test_checkpoint_gradients(self):
        """Gradient checkpointing (activation rematerialisation) runs end-to-end."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(20, checkpoint_gradients=True)

        logs = stats.training_logs[-1]
        assert logs["losses"].shape == (20, 1)
        assert jnp.isfinite(logs["total_loss"][-1])

    def test_offload_data_with_batchsize(self):
        """Host-resident data: stream mini-batches each step."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(20, offload_data=True, batchsize=16)

        logs = stats.training_logs[-1]
        assert logs["losses"].shape == (20, 1)
        assert jnp.isfinite(logs["total_loss"][-1])

    def test_offload_data_without_batchsize_is_ignored(self):
        """offload_data=True without batchsize falls back to on-device."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        # Should not raise, just warn and ignore offload_data
        stats = solver.solve(20, offload_data=True)

        logs = stats.training_logs[-1]
        assert jnp.isfinite(logs["total_loss"][-1])

    def test_checkpoint_and_offload_combined(self):
        """Both checkpoint_gradients and offload_data active together."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(
            20,
            checkpoint_gradients=True,
            offload_data=True,
            batchsize=16,
        )

        logs = stats.training_logs[-1]
        assert logs["losses"].shape == (20, 1)
        assert jnp.isfinite(logs["total_loss"][-1])

    def test_loss_decreases(self):
        """Sanity: loss should decrease over enough epochs."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(100)

        losses = stats.training_logs[-1]["total_loss"]
        assert losses[-1] < losses[0], "Loss did not decrease"

    def test_multiple_solve_calls(self):
        """Calling solve() twice accumulates training logs."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        solver.solve(20)
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(20, checkpoint_gradients=True)

        assert len(stats.training_logs) == 2

    def test_log_keys_complete(self):
        """Log dict contains all expected keys."""
        import optax
        from jno import LearningRateSchedule as lrs

        solver, u_net = _make_solver()
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(20)

        logs = stats.training_logs[-1]
        for key in ("epoch", "total_loss", "losses", "weights", "timestamps", "training_time"):
            assert key in logs, f"Missing log key: {key}"
