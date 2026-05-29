"""Tests that model controls are correctly attached to the right nodes.

These tests verify that optimizer(), lr(), lora(), constrain(), freeze(), and mask()
store their configurations on the right model objects — not just that training runs
without error. They also cover the new .constrain() feature, which has no prior tests.

Structure:
  - TestOptimizerAttachment   — pure attribute inspection, no training
  - TestLRScheduleAttachment  — schedule values callable and correct
  - TestLoRAAttachment        — lora config stored correctly
  - TestConstrainAttachment   — constrain() wraps the right leaves
  - TestWeightChangeVerification — post-solve weight delta checks
"""

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno
import jno.jnp_ops as jnn
from jno import LearningRateSchedule as lrs
from jno.architectures.models import nn

# Dummy losses array needed to call LR schedules in unit tests (no training)
_dummy_losses = jnp.array([0.5])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _mlp(key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    return foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key)


def _make_solver(key=None):
    """Minimal 1D Poisson solver for integration tests."""
    domain = 1 * jno.domain.line(mesh_size=0.05)
    x, *_ = domain.variable("interior")
    u_net = jnn.nn.wrap(_mlp(key or jax.random.PRNGKey(0)))
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x])
    return jno.core([pde.mse], domain), u_net


def _all_false(module):
    return jax.tree_util.tree_map(lambda _: False, module)


def _mask_hidden0(module):
    """Mask that selects only hidden_layers[0].weight and .bias."""
    return eqx.tree_at(
        lambda m: (m.hidden_layers[0].weight, m.hidden_layers[0].bias),
        _all_false(module),
        (True, True),
    )


def _mask_hidden1(module):
    """Mask that selects only hidden_layers[1].weight and .bias."""
    return eqx.tree_at(
        lambda m: (m.hidden_layers[1].weight, m.hidden_layers[1].bias),
        _all_false(module),
        (True, True),
    )


# ---------------------------------------------------------------------------
# Group 1: Optimizer attachment — pure attribute inspection (no training)
# ---------------------------------------------------------------------------


class TestOptimizerAttachment:
    def test_single_optimizer_stored_on_correct_attribute(self):
        net = nn.wrap(_mlp())
        net.optimizer(optax.adam)
        assert net._opt_fn is optax.adam
        assert len(net._param_groups) == 0

    def test_two_models_different_optimizers_do_not_cross_contaminate(self):
        net_a = nn.wrap(_mlp(jax.random.PRNGKey(0)))
        net_b = nn.wrap(_mlp(jax.random.PRNGKey(1)))
        net_a.optimizer(optax.adam)
        net_b.optimizer(optax.adamw)
        assert net_a._opt_fn is optax.adam
        assert net_b._opt_fn is optax.adamw
        assert net_a._opt_fn is not optax.adamw
        assert net_b._opt_fn is not optax.adam

    def test_masked_optimizer_creates_param_group(self):
        net = nn.wrap(_mlp())
        net.optimizer(optax.sgd)                                # global fallback first
        net.mask(_mask_hidden0(net.module)).optimizer(optax.adam)  # group
        assert net._opt_fn is optax.sgd                         # global unchanged
        assert len(net._param_groups) == 1
        assert net._param_groups[0]["opt_fn"] is optax.adam

    def test_bare_global_optimizer_clears_existing_groups(self):
        net = nn.wrap(_mlp())
        net.mask(_mask_hidden0(net.module)).optimizer(optax.adam)  # creates group
        assert len(net._param_groups) == 1
        net.optimizer(optax.adamw)                               # bare — resets groups
        assert len(net._param_groups) == 0
        assert net._opt_fn is optax.adamw

    def test_two_param_groups_stored_in_order(self):
        net = nn.wrap(_mlp())
        net.optimizer(optax.sgd)
        net.mask(_mask_hidden0(net.module)).optimizer(optax.adam)
        net.mask(_mask_hidden1(net.module)).optimizer(optax.adamw)
        assert len(net._param_groups) == 2
        assert net._param_groups[0]["opt_fn"] is optax.adam
        assert net._param_groups[1]["opt_fn"] is optax.adamw
        assert net._opt_fn is optax.sgd

    def test_three_independent_models_three_optimizers(self):
        net_a = nn.wrap(_mlp(jax.random.PRNGKey(0)))
        net_b = nn.wrap(_mlp(jax.random.PRNGKey(1)))
        net_c = nn.wrap(_mlp(jax.random.PRNGKey(2)))
        net_a.optimizer(optax.adam)
        net_b.optimizer(optax.adamw)
        net_c.optimizer(optax.sgd)
        assert net_a._opt_fn is optax.adam
        assert net_b._opt_fn is optax.adamw
        assert net_c._opt_fn is optax.sgd


# ---------------------------------------------------------------------------
# Group 2: LR schedule attachment and value correctness
# ---------------------------------------------------------------------------


class TestLRScheduleAttachment:
    def test_constant_lr_stored_and_returns_correct_value(self):
        net = nn.wrap(_mlp())
        sched = lrs.constant(3e-3)
        net.optimizer(optax.adam, lr=sched)
        assert net._lr is sched
        assert float(net._lr(0, _dummy_losses)) == pytest.approx(3e-3, rel=1e-4)

    def test_exponential_lr_starts_at_initial_value(self):
        net = nn.wrap(_mlp())
        sched = lrs.exponential(2e-3, decay_rate=0.9, decay_steps=100)
        net.optimizer(optax.adam, lr=sched)
        assert float(net._lr(0, _dummy_losses)) == pytest.approx(2e-3, rel=1e-4)

    def test_exponential_lr_decays_over_time(self):
        sched = lrs.exponential(1e-3, decay_rate=0.5, decay_steps=10)
        lr_early = float(sched(0, _dummy_losses))
        lr_later = float(sched(50, _dummy_losses))
        assert lr_later < lr_early

    def test_warmup_cosine_lr_increases_during_warmup_then_decays(self):
        sched = lrs.warmup_cosine(total_steps=200, warmup_steps=20, lr0=1e-3)
        lr_0 = float(sched(0, _dummy_losses))
        lr_10 = float(sched(10, _dummy_losses))
        lr_19 = float(sched(19, _dummy_losses))  # last warmup step
        lr_150 = float(sched(150, _dummy_losses))  # deep in cosine decay
        assert lr_0 < lr_10 < lr_19          # increasing during warmup
        assert lr_150 < lr_19                # decaying after warmup

    def test_per_group_lr_differs_from_global(self):
        net = nn.wrap(_mlp())
        sched_global = lrs.constant(1e-2)
        sched_group = lrs.constant(1e-4)
        net.optimizer(optax.adam, lr=sched_global)
        net.mask(_mask_hidden0(net.module)).optimizer(optax.adam, lr=sched_group)
        global_lr = float(net._lr(0, _dummy_losses))
        group_lr = float(net._param_groups[0]["lr"](0, _dummy_losses))
        assert global_lr == pytest.approx(1e-2, rel=1e-4)
        assert group_lr == pytest.approx(1e-4, rel=1e-4)
        assert global_lr != group_lr

    def test_two_models_independent_lr_schedules(self):
        net_a = nn.wrap(_mlp(jax.random.PRNGKey(0)))
        net_b = nn.wrap(_mlp(jax.random.PRNGKey(1)))
        net_a.optimizer(optax.adam, lr=lrs.constant(1e-3))
        net_b.optimizer(optax.adam, lr=lrs.constant(5e-5))
        assert float(net_a._lr(0, _dummy_losses)) == pytest.approx(1e-3, rel=1e-4)
        assert float(net_b._lr(0, _dummy_losses)) == pytest.approx(5e-5, rel=1e-4)

    def test_piecewise_constant_lr_returns_correct_segment_values(self):
        sched = lrs.piecewise_constant(boundaries=[10, 20], values=[1e-2, 1e-3, 1e-4])
        assert float(sched(5, _dummy_losses)) == pytest.approx(1e-2, rel=1e-4)
        assert float(sched(15, _dummy_losses)) == pytest.approx(1e-3, rel=1e-4)
        assert float(sched(25, _dummy_losses)) == pytest.approx(1e-4, rel=1e-4)


# ---------------------------------------------------------------------------
# Group 3: LoRA attachment
# ---------------------------------------------------------------------------


class TestLoRAAttachment:
    def test_lora_config_rank_and_alpha_stored(self):
        net = nn.wrap(_mlp())
        net.lora(rank=8, alpha=16)
        assert len(net._lora_config) == 1
        cfg = net._lora_config[0]
        assert cfg["rank"] == 8
        assert cfg["alpha"] == 16

    def test_lora_without_mask_sets_target_none_and_no_param_mask(self):
        net = nn.wrap(_mlp())
        net.lora(rank=4, alpha=1.0)
        assert net._lora_config[0]["target"] is None
        assert net._lora_param_mask is None

    def test_masked_lora_stores_lora_param_mask(self):
        net = nn.wrap(_mlp())
        mask = _mask_hidden0(net.module)
        net.mask(mask).lora(rank=4, alpha=8)
        assert net._lora_param_mask is not None
        assert len(net._lora_config) == 1

    def test_second_lora_call_overwrites_first(self):
        net = nn.wrap(_mlp())
        net.lora(rank=4, alpha=1.0)
        net.lora(rank=16, alpha=2.0)
        assert len(net._lora_config) == 1
        assert net._lora_config[0]["rank"] == 16
        assert net._lora_config[0]["alpha"] == 2.0

    def test_freeze_then_lora_sets_both_frozen_flag_and_config(self):
        net = nn.wrap(_mlp())
        net.freeze()
        net.lora(rank=4, alpha=1.0)
        assert net._frozen is True
        assert len(net._lora_config) == 1

    def test_lora_target_regex_stored_in_config(self):
        net = nn.wrap(_mlp())
        net.lora(rank=4, alpha=1.0, target="hidden_layers.0")
        assert net._lora_config[0]["target"] == "hidden_layers.0"

    def test_two_models_independent_lora_configs(self):
        net_a = nn.wrap(_mlp(jax.random.PRNGKey(0)))
        net_b = nn.wrap(_mlp(jax.random.PRNGKey(1)))
        net_a.lora(rank=4, alpha=1.0)
        net_b.lora(rank=8, alpha=2.0)
        assert net_a._lora_config[0]["rank"] == 4
        assert net_b._lora_config[0]["rank"] == 8


# ---------------------------------------------------------------------------
# Group 4: constrain() attachment — new feature with no prior tests
# ---------------------------------------------------------------------------


class TestConstrainAttachment:
    def _pm_leaves(self, pm, module):
        """Return leaves, stopping recursion at Parameterize nodes."""
        return jax.tree_util.tree_leaves(
            module, is_leaf=lambda x: isinstance(x, pm.Parameterize)
        )

    def test_constrain_wraps_all_inexact_leaves(self):
        pm = pytest.importorskip("paramax")
        net = nn.wrap(_mlp())
        n_float = sum(
            1 for l in jax.tree_util.tree_leaves(net.module) if eqx.is_inexact_array(l)
        )
        net.constrain(jax.nn.softplus)
        wrapped = [l for l in self._pm_leaves(pm, net.module) if isinstance(l, pm.Parameterize)]
        assert len(wrapped) == n_float

    def test_constrain_stores_correct_transform_fn(self):
        pm = pytest.importorskip("paramax")
        net = nn.wrap(_mlp())
        net.constrain(jax.nn.softplus)
        for leaf in self._pm_leaves(pm, net.module):
            if isinstance(leaf, pm.Parameterize):
                assert leaf.fn is jax.nn.softplus

    def test_constrain_sets_contains_unwrappables(self):
        pm = pytest.importorskip("paramax")
        net = nn.wrap(_mlp())
        assert not pm.contains_unwrappables(net.module)
        net.constrain(jax.nn.softplus)
        assert pm.contains_unwrappables(net.module)

    def test_masked_constrain_wraps_only_selected_leaves(self):
        pm = pytest.importorskip("paramax")
        net = nn.wrap(_mlp())
        mask = eqx.tree_at(
            lambda m: m.hidden_layers[0].weight,
            _all_false(net.module),
            True,
        )
        net.mask(mask).constrain(jax.nn.softplus)
        w0 = net.module.hidden_layers[0].weight
        w1 = net.module.hidden_layers[1].weight
        assert isinstance(w0, pm.Parameterize)
        assert not isinstance(w1, pm.Parameterize)

    def test_masked_constrain_consumes_mask_scope(self):
        pytest.importorskip("paramax")
        net = nn.wrap(_mlp())
        net.mask(_mask_hidden0(net.module)).constrain(jax.nn.softplus)
        assert net._mask_scope_pending is False

    def test_constrain_returns_self_for_chaining(self):
        pytest.importorskip("paramax")
        net = nn.wrap(_mlp())
        ret = net.constrain(jax.nn.softplus)
        assert ret is net

    def test_sigmoid_and_softplus_produce_different_wrapped_leaves(self):
        pm = pytest.importorskip("paramax")
        net_a = nn.wrap(_mlp(jax.random.PRNGKey(0)))
        net_b = nn.wrap(_mlp(jax.random.PRNGKey(0)))
        net_a.constrain(jax.nn.softplus)
        net_b.constrain(jax.nn.sigmoid)
        leaves_a = [l for l in self._pm_leaves(pm, net_a.module) if isinstance(l, pm.Parameterize)]
        leaves_b = [l for l in self._pm_leaves(pm, net_b.module) if isinstance(l, pm.Parameterize)]
        assert all(l.fn is jax.nn.softplus for l in leaves_a)
        assert all(l.fn is jax.nn.sigmoid for l in leaves_b)

    @pytest.mark.integration
    def test_constrain_trains_without_error(self):
        pytest.importorskip("paramax")
        solver, u_net = _make_solver()
        u_net.constrain(jax.nn.softplus)
        u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
        stats = solver.solve(5)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    @pytest.mark.integration
    def test_masked_constrain_trains_without_error(self):
        pytest.importorskip("paramax")
        solver, u_net = _make_solver()
        u_net.mask(_mask_hidden0(u_net.module)).constrain(jax.nn.softplus)
        u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
        stats = solver.solve(5)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])

    @pytest.mark.integration
    def test_constrain_with_freeze_and_lora(self):
        pytest.importorskip("paramax")
        solver, u_net = _make_solver()
        u_net.constrain(jax.nn.softplus)
        u_net.freeze()
        u_net.lora(rank=4, alpha=1.0)
        u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
        stats = solver.solve(5)
        assert jnp.isfinite(stats.training_logs[-1]["total_loss"][-1])


# ---------------------------------------------------------------------------
# Group 5: Post-solve weight-change verification
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWeightChangeVerification:
    def test_masked_freeze_frozen_leaves_unchanged_after_solve(self):
        """mask(layer0).freeze() → layer0 unchanged, other layers trained."""
        solver, u_net = _make_solver()
        lid = u_net.layer_id

        w0_before = jnp.array(solver.models[lid].hidden_layers[0].weight)
        b0_before = jnp.array(solver.models[lid].hidden_layers[0].bias)
        w1_before = jnp.array(solver.models[lid].hidden_layers[1].weight)

        u_net.mask(_mask_hidden0(u_net.module)).freeze()
        u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
        solver.solve(20)

        w0_after = jnp.array(solver.models[lid].hidden_layers[0].weight)
        b0_after = jnp.array(solver.models[lid].hidden_layers[0].bias)
        w1_after = jnp.array(solver.models[lid].hidden_layers[1].weight)

        assert jnp.allclose(w0_before, w0_after), "Frozen layer[0].weight changed"
        assert jnp.allclose(b0_before, b0_after), "Frozen layer[0].bias changed"
        assert not jnp.allclose(w1_before, w1_after), "Trainable layer[1].weight did not change"

    def test_two_models_frozen_and_trainable_update_correctly(self):
        """Frozen model stays put; trainable model updates. Both in one solve."""
        domain = 1 * jno.domain.line(mesh_size=0.05)
        x, *_ = domain.variable("interior")
        key_a, key_b = jax.random.split(jax.random.PRNGKey(99))
        net_frozen = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key_a))
        net_train = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=2, key=key_b))

        u_f = net_frozen(x)
        u_t = net_train(x)
        pde = jnn.laplacian(u_t, [x]) + u_f
        solver = jno.core([pde.mse], domain)

        lid_f = net_frozen.layer_id
        lid_t = net_train.layer_id
        w_f_before = jnp.array(solver.models[lid_f].hidden_layers[0].weight)
        w_t_before = jnp.array(solver.models[lid_t].hidden_layers[0].weight)

        net_frozen.freeze()
        net_train.optimizer(optax.adam, lr=lrs.constant(1e-3))
        solver.solve(30)

        w_f_after = jnp.array(solver.models[lid_f].hidden_layers[0].weight)
        w_t_after = jnp.array(solver.models[lid_t].hidden_layers[0].weight)

        assert jnp.allclose(w_f_before, w_f_after), "Frozen model weights changed"
        assert not jnp.allclose(w_t_before, w_t_after), "Trainable model weights did not change"

    def test_masked_group_optimizer_all_params_remain_trainable(self):
        """mask(...).optimizer() scopes the optimizer but does not freeze anything."""
        solver, u_net = _make_solver()
        mask = _mask_hidden0(u_net.module)
        u_net.optimizer(optax.adam, lr=lrs.constant(1e-3))
        u_net.mask(mask).optimizer(optax.adamw, lr=lrs.constant(1e-5))
        stats = solver.solve(1)
        logs = stats.training_logs[-1]
        assert logs["trainable_params"] == logs["total_params"]
