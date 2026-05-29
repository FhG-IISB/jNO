"""Tests that jNO model-control methods attach the right config to the right nodes.

These tests verify *attachment*, not training behavior:
  - The correct optimizer/LR/LoRA config is stored on the correct attribute
  - Masks target the intended parameter group
  - Two models don't cross-contaminate each other's state
"""

from __future__ import annotations

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno.jnp_ops as jnn
from jno import LearningRateSchedule as lrs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_net(*, hidden_dims=16, num_layers=2):
    """Return a fresh jNO-wrapped MLP with a 1-D line domain."""
    key = jax.random.PRNGKey(42)
    return jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=hidden_dims, num_layers=num_layers, key=key))


def _all_false_mask(module):
    """All-False pytree with the same structure as module."""
    return jax.tree_util.tree_map(lambda _: False, module)


def _mask_first_layer(module):
    """Mask that selects weight + bias of the first hidden linear layer only."""
    base = _all_false_mask(module)
    return eqx.tree_at(
        lambda m: (m.hidden_layers[0].weight, m.hidden_layers[0].bias),
        base,
        (True, True),
    )


# ===========================================================================
# Optimizer attachment
# ===========================================================================


class TestOptimizerAttachment:
    def test_single_optimizer_stored_on_correct_attribute(self):
        net = _make_net()
        net.optimizer(optax.adam)
        assert net._opt_fn is optax.adam
        assert len(net._param_groups) == 0

    def test_global_optimizer_call_clears_param_groups(self):
        net = _make_net()
        # First add a masked group, then override with a global call
        mask = _mask_first_layer(net.module)
        net.mask(mask).optimizer(optax.sgd)
        assert len(net._param_groups) == 1
        net.optimizer(optax.adam)  # global call must clear groups
        assert net._opt_fn is optax.adam
        assert len(net._param_groups) == 0

    def test_masked_optimizer_creates_param_group(self):
        net = _make_net()
        net.optimizer(optax.sgd)  # global fallback
        mask = _mask_first_layer(net.module)
        net.mask(mask).optimizer(optax.adam)
        # Global fallback unchanged
        assert net._opt_fn is optax.sgd
        # One masked group with adam
        assert len(net._param_groups) == 1
        assert net._param_groups[0]["opt_fn"] is optax.adam

    def test_masked_optimizer_stores_mask_in_group(self):
        net = _make_net()
        mask = _mask_first_layer(net.module)
        net.optimizer(optax.sgd)
        net.mask(mask).optimizer(optax.adam)
        stored_mask = net._param_groups[0]["mask"]
        # The stored mask should be a pytree with the same leaf count
        stored_leaves = jax.tree_util.tree_leaves(stored_mask)
        orig_leaves = jax.tree_util.tree_leaves(mask)
        assert len(stored_leaves) == len(orig_leaves)

    def test_two_models_different_optimizers_do_not_cross_contaminate(self):
        net_a = _make_net()
        net_b = _make_net()
        net_a.optimizer(optax.adam)
        net_b.optimizer(optax.adamw)
        assert net_a._opt_fn is optax.adam
        assert net_b._opt_fn is optax.adamw

    def test_two_masked_groups_stored_independently(self):
        net = _make_net()
        net.optimizer(optax.sgd)
        mask1 = _mask_first_layer(net.module)
        # Build a second mask for a different layer
        base = _all_false_mask(net.module)
        mask2 = eqx.tree_at(
            lambda m: (m.hidden_layers[1].weight, m.hidden_layers[1].bias),
            base,
            (True, True),
        )
        net.mask(mask1).optimizer(optax.adam)
        net.mask(mask2).optimizer(optax.adamw)
        assert len(net._param_groups) == 2
        opt_fns = {g["opt_fn"] for g in net._param_groups}
        assert optax.adam in opt_fns
        assert optax.adamw in opt_fns


# ===========================================================================
# Learning-rate attachment
# ===========================================================================


class TestLearningRateAttachment:
    def test_lr_stored_on_attribute(self):
        net = _make_net()
        sched = lrs.constant(1e-3)
        net.optimizer(optax.adam, lr=sched)
        assert net._lr is sched

    def test_lr_via_separate_call(self):
        net = _make_net()
        sched = lrs.exponential(1e-3, 0.9, 100)
        net.optimizer(optax.adam)
        net.lr(sched)
        assert net._lr is sched

    def test_masked_lr_stored_in_group(self):
        net = _make_net()
        sched_global = lrs.constant(1e-3)
        sched_layer0 = lrs.constant(1e-4)
        mask = _mask_first_layer(net.module)
        net.optimizer(optax.adam, lr=sched_global)
        net.mask(mask).lr(sched_layer0)
        # Global LR unchanged
        assert net._lr is sched_global
        # Group has its own LR
        group_lrs = [g["lr"] for g in net._param_groups if g["lr"] is not None]
        assert sched_layer0 in group_lrs

    def test_warmup_cosine_schedule_stored_correctly(self):
        net = _make_net()
        sched = lrs.warmup_cosine(total_steps=1000, warmup_steps=100, lr0=1e-3)
        net.optimizer(optax.adam, lr=sched)
        assert net._lr is sched

    def test_piecewise_constant_schedule_stored_correctly(self):
        net = _make_net()
        sched = lrs.piecewise_constant([100, 500], [1e-3, 1e-4, 1e-5])
        net.optimizer(optax.adam, lr=sched)
        assert net._lr is sched

    def test_two_models_different_lr_do_not_cross_contaminate(self):
        net_a = _make_net()
        net_b = _make_net()
        sched_a = lrs.constant(1e-3)
        sched_b = lrs.constant(1e-5)
        net_a.optimizer(optax.adam, lr=sched_a)
        net_b.optimizer(optax.adam, lr=sched_b)
        assert net_a._lr is sched_a
        assert net_b._lr is sched_b


# ===========================================================================
# Freeze attachment
# ===========================================================================


class TestFreezeAttachment:
    def test_freeze_sets_frozen_flag(self):
        net = _make_net()
        net.optimizer(optax.adam)
        net.freeze()
        assert net._frozen is True
        assert net._trainable_param_mask is None

    def test_unfreeze_clears_frozen_flag(self):
        net = _make_net()
        net.freeze()
        net.unfreeze()
        assert net._frozen is False
        assert net._trainable_param_mask is None

    def test_masked_freeze_sets_trainable_param_mask(self):
        net = _make_net()
        mask = _mask_first_layer(net.module)
        net.mask(mask).freeze()
        # Whole-model freeze flag should NOT be set
        assert net._frozen is False
        # But trainable_param_mask must be set (inverted mask: True=freeze → False in trainable)
        assert net._trainable_param_mask is not None

    def test_masked_freeze_inverts_mask_correctly(self):
        """mask(M).freeze() → trainable_param_mask has False where M was True."""
        net = _make_net()
        mask = _mask_first_layer(net.module)
        net.mask(mask).freeze()
        tm = net._trainable_param_mask
        # The leaves selected by mask (True) should be False (frozen) in trainable mask
        orig_leaves = jax.tree_util.tree_leaves(mask)
        trainable_leaves = jax.tree_util.tree_leaves(tm)
        for orig, trainable in zip(orig_leaves, trainable_leaves):
            if orig is True or orig is jnp.array(True):
                assert not trainable, "Masked-True leaves should be frozen (False in trainable mask)"

    def test_global_freeze_does_not_set_trainable_param_mask(self):
        net = _make_net()
        net.freeze()
        assert net._trainable_param_mask is None
        assert net._frozen is True

    def test_two_models_freeze_do_not_cross_contaminate(self):
        net_a = _make_net()
        net_b = _make_net()
        net_a.freeze()
        assert net_a._frozen is True
        assert net_b._frozen is False


# ===========================================================================
# LoRA attachment
# ===========================================================================


class TestLoraAttachment:
    def test_lora_sets_config_attribute(self):
        net = _make_net()
        net.lora(rank=4, alpha=1.0)
        assert net._lora_config is not None
        assert len(net._lora_config) == 1

    def test_lora_stores_correct_rank(self):
        net = _make_net()
        net.lora(rank=8)
        assert net._lora_config[0]["rank"] == 8

    def test_lora_stores_correct_alpha(self):
        net = _make_net()
        net.lora(rank=4, alpha=2.0)
        assert net._lora_config[0]["alpha"] == pytest.approx(2.0)

    def test_lora_specs_mode_stores_multiple_configs(self):
        net = _make_net()
        specs = [
            {"target": "layers.0", "rank": 4, "alpha": 1.0},
            {"target": "layers.1", "rank": 8, "alpha": 2.0},
        ]
        net.lora(specs=specs)
        assert net._lora_config is not None
        assert len(net._lora_config) == 2
        ranks = {c["rank"] for c in net._lora_config}
        assert ranks == {4, 8}

    def test_lora_clears_trainable_param_mask(self):
        net = _make_net()
        mask = _mask_first_layer(net.module)
        net.mask(mask).freeze()
        assert net._trainable_param_mask is not None
        net.lora(rank=4)  # lora() clears any stale freeze mask
        assert net._trainable_param_mask is None

    def test_two_models_different_lora_do_not_cross_contaminate(self):
        net_a = _make_net()
        net_b = _make_net()
        net_a.lora(rank=4)
        net_b.lora(rank=8)
        assert net_a._lora_config[0]["rank"] == 4
        assert net_b._lora_config[0]["rank"] == 8

    def test_lora_config_is_none_by_default(self):
        net = _make_net()
        assert net._lora_config is None

    def test_masked_lora_stores_lora_param_mask(self):
        """mask(M).lora() restricts which layers receive LoRA adapters."""
        net = _make_net()
        mask = _mask_first_layer(net.module)
        net.mask(mask).lora(rank=4)
        assert net._lora_param_mask is not None

    def test_global_lora_clears_lora_param_mask(self):
        """Global lora() (without mask) must clear any stale lora_param_mask."""
        net = _make_net()
        mask = _mask_first_layer(net.module)
        net.mask(mask).lora(rank=4)
        assert net._lora_param_mask is not None
        net.lora(rank=8)  # global call, no mask
        assert net._lora_param_mask is None


# ===========================================================================
# Integration: chaining multiple controls
# ===========================================================================


class TestChainedControls:
    def test_freeze_plus_lora_chains_correctly(self):
        """freeze().lora() → _frozen=True, _lora_config set, trainable_param_mask=None."""
        net = _make_net()
        net.freeze().lora(rank=4)
        # freeze() sets _frozen=True; lora() respects that and sets config
        assert net._lora_config is not None
        assert net._lora_config[0]["rank"] == 4

    def test_optimizer_then_masked_optimizer_coexist(self):
        """Global optimizer + masked group must both survive on the same model."""
        net = _make_net()
        net.optimizer(optax.sgd, lr=lrs.constant(1e-2))
        mask = _mask_first_layer(net.module)
        net.mask(mask).optimizer(optax.adam, lr=lrs.constant(1e-3))
        assert net._opt_fn is optax.sgd
        assert len(net._param_groups) == 1
        grp = net._param_groups[0]
        assert grp["opt_fn"] is optax.adam

    def test_reset_clears_all_controls(self):
        """reset() must restore the model to its default (no optimizer, not frozen)."""
        net = _make_net()
        net.optimizer(optax.adam, lr=lrs.constant(1e-3))
        net.lora(rank=4)
        mask = _mask_first_layer(net.module)
        net.mask(mask).freeze()
        net.reset()
        assert net._opt_fn is None
        assert net._frozen is False
        assert net._lora_config is None
        assert len(net._param_groups) == 0
        assert net._trainable_param_mask is None
