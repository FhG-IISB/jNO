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
from jno.trace_compiler import TraceCompiler
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

        compiled = TraceCompiler.compile_traced_expression(expr, all_ops)
        result = compiled({}, {"x": jnp.ones((1, 1))})
        assert jnp.allclose(result, 5.0)

    def test_variable_expression(self):
        """Evaluate x^2 at x=3."""
        x = make_var("x")
        expr = x ** Literal(2.0)
        op = OperationDef(expr, [x])
        all_ops = [op]

        compiled = TraceCompiler.compile_traced_expression(expr, all_ops)
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

        compiled = TraceCompiler.compile_traced_expression(call, all_ops)
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

        compiled = TraceCompiler.compile_traced_expression(du_dx, all_ops)
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

        compiled = TraceCompiler.compile_traced_expression(expr, all_ops)
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

        compiled = TraceCompiler.compile_traced_expression(expr, all_ops)
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


# ======================================================================
# Param mask integration tests
# ======================================================================


@pytest.mark.integration
class TestParamMask:
    """End-to-end tests that verify param_mask is respected by core.solve()."""

    def _make_masked_solver(self, mask_fn):
        """Build a solver with a masked model.

        mask_fn receives the raw equinox module and must return a pytree
        of bools with the same structure.
        """
        import jax
        import jno
        import jno.numpy as jnn
        import equinox as eqx

        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
        x, *_ = domain.variable("interior")
        key = jax.random.PRNGKey(0)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=key)
        u = u_net(x) * x * (1 - x)
        pde = jnn.laplacian(u, [x])
        param_mask = mask_fn(u_net.module)
        u_net.mask(param_mask)
        return jno.core([pde.mse], domain), u_net

    def test_partial_mask_solve_produces_finite_loss(self):
        """Solve with a partial mask (first hidden layer only) runs without
        errors and produces a finite loss."""
        import jax
        import optax
        import equinox as eqx
        from jno import LearningRateSchedule as lrs

        def make_mask(module):
            all_false = jax.tree_util.tree_map(lambda _: False, module)
            return eqx.tree_at(
                lambda m: (m.hidden_layers[0].weight, m.hidden_layers[0].bias),
                all_false,
                (True, True),
            )

        solver, u_net = self._make_masked_solver(make_mask)
        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        stats = solver.solve(10)
        loss = stats.training_logs[-1]["total_loss"][-1]
        assert jnp.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_masked_out_params_do_not_change(self):
        """Weights masked to False must not be modified by the optimizer."""
        import jax
        import optax
        import equinox as eqx
        from jno import LearningRateSchedule as lrs

        def make_mask(module):
            # Train only the output layer; freeze everything else
            all_false = jax.tree_util.tree_map(lambda _: False, module)
            return eqx.tree_at(
                lambda m: (m.output_layer.weight, m.output_layer.bias),
                all_false,
                (True, True),
            )

        solver, u_net = self._make_masked_solver(make_mask)
        lid = u_net.layer_id

        # Save the first hidden layer weights before training
        pre_w0 = jnp.array(solver.models[lid].hidden_layers[0].weight)
        pre_b0 = jnp.array(solver.models[lid].hidden_layers[0].bias)

        u_net.optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        solver.solve(20)

        # Hidden layer 0 must be numerically identical (frozen by mask)
        post_w0 = jnp.array(solver.models[lid].hidden_layers[0].weight)
        post_b0 = jnp.array(solver.models[lid].hidden_layers[0].bias)
        assert jnp.allclose(pre_w0, post_w0), "Masked weight changed after solve"
        assert jnp.allclose(pre_b0, post_b0), "Masked bias changed after solve"

    def test_all_true_mask_trains_all_params(self):
        """An all-True mask behaves identically to no mask (all params train)."""
        import jax
        import jax.numpy as jnp
        import optax
        import equinox as eqx
        from jno import LearningRateSchedule as lrs
        import jno
        import jno.numpy as jnn

        def make_masked(seed):
            domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
            x, *_ = domain.variable("interior")
            key = jax.random.PRNGKey(seed)
            u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=key)
            return u_net, domain, jno.core([jnn.laplacian(u_net(x) * x * (1 - x), [x]).mse], domain)

        u_masked, dom_m, solver_m = make_masked(7)
        u_nomask, dom_n, solver_n = make_masked(7)

        all_true = jax.tree_util.tree_map(lambda _: True, u_masked.module)
        u_masked.mask(all_true)

        opt_kw = dict(lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))
        u_masked.optimizer(optax.adam, **opt_kw)
        u_nomask.optimizer(optax.adam, **opt_kw)

        stats_m = solver_m.solve(5)
        stats_n = solver_n.solve(5)

        # Both should converge to the same loss (identical seeds & arch)
        loss_m = stats_m.training_logs[-1]["total_loss"][-1]
        loss_n = stats_n.training_logs[-1]["total_loss"][-1]
        assert jnp.isclose(loss_m, loss_n, rtol=1e-4), f"Masked and unmasked runs diverged: {loss_m} vs {loss_n}"

    def test_mask_via_model_call_syntax(self):
        """The fluent call syntax model(x).mask(...).optimizer(...) works end-to-end."""
        import jax
        import optax
        import equinox as eqx
        from jno import LearningRateSchedule as lrs
        import jno
        import jno.numpy as jnn

        domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
        x, *_ = domain.variable("interior")
        key = jax.random.PRNGKey(0)
        u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=key)

        all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
        partial_mask = eqx.tree_at(
            lambda m: (m.hidden_layers[0].weight, m.hidden_layers[0].bias),
            all_false,
            (True, True),
        )

        # Chain via ModelCall (the result of u_net(x))
        u = u_net(x) * x * (1 - x)
        u_net.mask(partial_mask).optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5))

        pde = jnn.laplacian(u, [x])
        solver = jno.core([pde.mse], domain)
        stats = solver.solve(5)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


# ======================================================================
# Combined: mask + initialize(pytree) + different optimizers + freeze + LoRA
# ======================================================================


@pytest.mark.integration
def test_mask_initialize_freeze_lora_combined():
    """One comprehensive test exercising every piece of the mask API together.

    Network layout
    ~~~~~~~~~~~~~~
    u_net  — trained with a partial mask (only output layer) and pretrained
             weights loaded from a pytree.  Hidden layer is frozen by the mask.

    v_net  — uses mask().freeze().lora(): LoRA overrides the freeze flag, so
             LoRA adapter weights are trained while base weights stay fixed.

    Both networks contribute one MSE constraint in the same ``core`` solver
    (two different PDEs on the same 1-D domain).

    Assertions
    ~~~~~~~~~~
    * Loss is finite after training.
    * u_net's hidden layer did not change (frozen by mask).
    * u_net's output layer did change (trained by mask).
    * u_net's arrays match the pretrained weights exactly at solve start
      (initialize overwrote the random init).
    * v_net has LoRA structure (base weights frozen, adapters trained).
    """
    import jax
    import jax.numpy as jnp
    import optax
    import equinox as eqx
    import jno
    import jno.numpy as jnn
    from jno import LearningRateSchedule as lrs

    # ── Domain ───────────────────────────────────────────────
    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = domain.variable("interior")

    # ── u_net: mask (output layer only) + initialize from pytree ─────
    key_u = jax.random.PRNGKey(0)
    u_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=key_u)

    # Create "pretrained" weights from a different seed
    pretrained = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(99))

    # Mask: only output layer is trainable
    all_false_u = jax.tree_util.tree_map(lambda _: False, u_net.module)
    mask_u = eqx.tree_at(
        lambda m: (m.output_layer.weight, m.output_layer.bias),
        all_false_u,
        (True, True),
    )

    (u_net.mask(mask_u).initialize(pretrained.module).optimizer(optax.adam, lr=lrs.exponential(1e-3, 0.8, 1000, 1e-5)))  # load pretrained weights from pytree

    u = u_net(x) * x * (1 - x)
    pde_u = jnn.laplacian(u, [x])

    # ── v_net: mask + freeze + lora  (LoRA takes priority over freeze) ──
    key_v = jax.random.PRNGKey(1)
    v_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=key_v)

    all_false_v = jax.tree_util.tree_map(lambda _: False, v_net.module)
    mask_v = eqx.tree_at(
        lambda m: (m.hidden_layers[0].weight, m.hidden_layers[0].bias),
        all_false_v,
        (True, True),
    )

    (
        v_net.mask(mask_v)
        .freeze()  # would freeze whole model, but ...
        .lora(rank=2, alpha=1.0)  # ... LoRA takes priority: adapters trainable
        .optimizer(optax.sgd, lr=lrs.exponential(5e-4, 0.9, 1000, 1e-6))
    )

    v = v_net(x) * x * (1 - x)
    pde_v = jnn.laplacian(v, [x]) - jnn.sin(jnn.pi * x)

    # ── Solver ───────────────────────────────────────────────
    solver = jno.core([pde_u.mse, pde_v.mse], domain)

    lid_u = u_net.layer_id
    lid_v = v_net.layer_id

    # Capture u_net hidden layer weights right after compile (post-initialize)
    pre_hidden_w = jnp.array(solver.models[lid_u].hidden_layers[0].weight)
    pre_hidden_b = jnp.array(solver.models[lid_u].hidden_layers[0].bias)
    pre_output_w = jnp.array(solver.models[lid_u].output_layer.weight)

    # Capture v_net weights before solve — LoRA will train adapters and merge
    # them back, so we verify the merged result differs from the random init.
    pre_v_hidden0_w = jnp.array(solver.models[lid_v].hidden_layers[0].weight)

    # Verify initialize loaded the pretrained weights (not the original key_u init)
    assert jnp.allclose(
        solver.models[lid_u].hidden_layers[0].weight,
        pretrained.module.hidden_layers[0].weight,
    ), "initialize(pytree) did not load the pretrained weights"

    # ── Train ────────────────────────────────────────────────
    stats = solver.solve(15)

    # 1. Loss is finite
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    # 2. u_net frozen hidden layer must not have changed
    post_hidden_w = jnp.array(solver.models[lid_u].hidden_layers[0].weight)
    post_hidden_b = jnp.array(solver.models[lid_u].hidden_layers[0].bias)
    assert jnp.allclose(pre_hidden_w, post_hidden_w), "u_net hidden layer weight changed despite mask"
    assert jnp.allclose(pre_hidden_b, post_hidden_b), "u_net hidden layer bias changed despite mask"

    # 3. u_net output layer must have been updated
    post_output_w = jnp.array(solver.models[lid_u].output_layer.weight)
    assert not jnp.allclose(pre_output_w, post_output_w), "u_net output layer was not trained"

    # 4. v_net LoRA was applied and merged back into base weights by core.solve().
    # The merged weights must differ from the original random init, proving the
    # LoRA adapters were actually trained (freeze flag was correctly overridden).
    post_v_hidden0_w = jnp.array(solver.models[lid_v].hidden_layers[0].weight)
    assert not jnp.allclose(pre_v_hidden0_w, post_v_hidden0_w), "v_net base weights did not change — LoRA adapters were not trained/merged"


# =============================================================================
# flax.nnx model support
# =============================================================================


@pytest.mark.integration
def test_nnx_auto_wrap_and_train():
    """nn.wrap(nnx_module) auto-detects NNX and basic training works."""
    from flax import nnx
    import optax
    import jno
    import jno.numpy as jnn
    from jno import LearningRateSchedule as lrs
    from jno.architectures.common import FlaxNNXWrapper

    class TwoLayer(nnx.Module):
        def __init__(self, rngs):
            self.l1 = nnx.Linear(1, 8, rngs=rngs)
            self.l2 = nnx.Linear(8, 1, rngs=rngs)

        def __call__(self, x):
            return self.l2(nnx.relu(self.l1(x)))

    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = domain.variable("interior")

    net = jnn.nn.wrap(TwoLayer(nnx.Rngs(0)))
    # auto-wrap should have produced a FlaxNNXWrapper
    assert isinstance(net.module, FlaxNNXWrapper), "nn.wrap did not produce FlaxNNXWrapper"

    net.optimizer(optax.adam, lr=lrs(1e-3))
    u = net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) + 1.0

    solver = jno.core([pde.mse], domain)
    pre_w = jnp.array(jax.tree_util.tree_leaves(solver.models[net.layer_id].state)[0])

    stats = solver.solve(10)
    assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    post_w = jnp.array(jax.tree_util.tree_leaves(solver.models[net.layer_id].state)[0])
    assert not jnp.allclose(pre_w, post_w), "weights did not change after training"


@pytest.mark.integration
def test_nnx_freeze():
    """freeze() prevents NNX model weights from updating."""
    from flax import nnx
    import optax
    import jno
    import jno.numpy as jnn
    from jno import LearningRateSchedule as lrs

    class Net(nnx.Module):
        def __init__(self, rngs):
            self.l1 = nnx.Linear(1, 8, rngs=rngs)
            self.l2 = nnx.Linear(8, 1, rngs=rngs)

        def __call__(self, x):
            return self.l2(nnx.relu(self.l1(x)))

    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = domain.variable("interior")

    frozen_net = jnn.nn.wrap(Net(nnx.Rngs(0)))
    frozen_net.freeze()

    # need at least one trainable model → add a small equinox MLP
    train_net = jnn.nn.mlp(1, output_dim=1, hidden_dims=4, num_layers=1, key=jax.random.PRNGKey(99))
    train_net.optimizer(optax.adam, lr=lrs(1e-3))

    u_frozen = frozen_net(x) * x * (1 - x)
    u_train = train_net(x) * x * (1 - x)
    pde = jnn.laplacian(u_frozen + u_train, [x]) + 1.0

    solver = jno.core([pde.mse], domain)
    lid = frozen_net.layer_id
    pre_leaves = [jnp.array(l) for l in jax.tree_util.tree_leaves(solver.models[lid].state) if hasattr(l, "shape")]

    solver.solve(10)

    post_leaves = [jnp.array(l) for l in jax.tree_util.tree_leaves(solver.models[lid].state) if hasattr(l, "shape")]
    for pre, post in zip(pre_leaves, post_leaves):
        assert jnp.allclose(pre, post), "frozen NNX model weights changed"


@pytest.mark.integration
def test_nnx_mask():
    """mask() trains only the selected leaves of a NNX model."""
    from flax import nnx
    import optax
    import equinox as eqx
    import jno
    import jno.numpy as jnn
    from jno import LearningRateSchedule as lrs

    class Net(nnx.Module):
        def __init__(self, rngs):
            self.l1 = nnx.Linear(1, 8, rngs=rngs)
            self.l2 = nnx.Linear(8, 1, rngs=rngs)

        def __call__(self, x):
            return self.l2(nnx.relu(self.l1(x)))

    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = domain.variable("interior")

    net = jnn.nn.wrap(Net(nnx.Rngs(0)))

    # Build a mask: train only l2 kernel and bias, freeze l1.
    # Navigate the NNX State pytree with subscript access; use
    # tree_leaves(param)[0] to reach the raw array leaf (avoids
    # the deprecated .value attribute).
    all_false = jax.tree_util.tree_map(lambda _: False, net.module)
    mask = eqx.tree_at(
        lambda w: (
            jax.tree_util.tree_leaves(w.state["l2"]["kernel"])[0],
            jax.tree_util.tree_leaves(w.state["l2"]["bias"])[0],
        ),
        all_false,
        (True, True),
    )
    net.mask(mask).optimizer(optax.adam, lr=lrs(1e-3))

    u = net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) + 1.0
    solver = jno.core([pde.mse], domain)
    lid = net.layer_id

    # Capture l1 and l2 kernel weights before training
    pre_l1_k = jnp.array(jax.tree_util.tree_leaves(solver.models[lid].state["l1"]["kernel"])[0])
    pre_l2_k = jnp.array(jax.tree_util.tree_leaves(solver.models[lid].state["l2"]["kernel"])[0])

    solver.solve(15)

    post_l1_k = jnp.array(jax.tree_util.tree_leaves(solver.models[lid].state["l1"]["kernel"])[0])
    post_l2_k = jnp.array(jax.tree_util.tree_leaves(solver.models[lid].state["l2"]["kernel"])[0])

    assert jnp.allclose(pre_l1_k, post_l1_k), "l1 (masked False) changed"
    assert not jnp.allclose(pre_l2_k, post_l2_k), "l2 (masked True) did not change"


@pytest.mark.integration
def test_nnx_initialize_from_pytree():
    """initialize(pretrained_nnx_model) loads weights from another NNX instance."""
    from flax import nnx
    import optax
    import jno
    import jno.numpy as jnn
    from jno import LearningRateSchedule as lrs

    class Net(nnx.Module):
        def __init__(self, rngs):
            self.l1 = nnx.Linear(1, 8, rngs=rngs)
            self.l2 = nnx.Linear(8, 1, rngs=rngs)

        def __call__(self, x):
            return self.l2(nnx.relu(self.l1(x)))

    pretrained = Net(nnx.Rngs(42))  # different seed → different weights
    random_init = Net(nnx.Rngs(0))

    domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = domain.variable("interior")

    net = jnn.nn.wrap(random_init)
    net.initialize(pretrained)  # load weights from the pretrained NNX model
    net.optimizer(optax.adam, lr=lrs(1e-3))

    u = net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) + 1.0
    solver = jno.core([pde.mse], domain)
    lid = net.layer_id

    # After compile the weights must match the pretrained model, not random_init
    _, pretrained_state = nnx.split(pretrained)
    loaded_l1_k = jnp.array(jax.tree_util.tree_leaves(solver.models[lid].state["l1"]["kernel"])[0])
    pretrained_l1_k = jnp.array(jax.tree_util.tree_leaves(pretrained_state["l1"]["kernel"])[0])
    random_l1_k = jnp.array(jax.tree_util.tree_leaves(nnx.split(random_init)[1]["l1"]["kernel"])[0])

    assert jnp.allclose(loaded_l1_k, pretrained_l1_k), "initialize() did not load pretrained NNX weights"
    assert not jnp.allclose(loaded_l1_k, random_l1_k), "weights still match random init — initialize() had no effect"
