"""Unit tests for jno.utils.adaptive — LR and weight schedules."""

import pytest
import jax.numpy as jnp

from jno.utils.adaptive import LearningRateSchedule, WeightSchedule


# ======================================================================
# LearningRateSchedule
# ======================================================================
class TestLearningRateSchedule:
    def test_constant(self):
        schedule = LearningRateSchedule.constant(0.001)
        lr = schedule(jnp.array(0), jnp.array([1.0]))
        assert float(lr) == pytest.approx(0.001)

    def test_constant_at_different_steps(self):
        schedule = LearningRateSchedule.constant(0.01)
        for step in [0, 100, 1000]:
            lr = schedule(jnp.array(step), jnp.array([1.0]))
            assert float(lr) == pytest.approx(0.01)

    def test_cosine_endpoints(self):
        total = 1000
        schedule = LearningRateSchedule.cosine(total, lr0=0.1, lr_end=0.0)
        lr_start = float(schedule(jnp.array(0), jnp.array([1.0])))
        lr_end = float(schedule(jnp.array(total), jnp.array([1.0])))
        assert lr_start == pytest.approx(0.1, abs=0.01)
        # At end, should be close to lr_end
        assert lr_end < lr_start

    def test_exponential_decay(self):
        schedule = LearningRateSchedule.exponential(lr0=0.1, decay_rate=0.5, decay_steps=100)
        lr_0 = float(schedule(jnp.array(0), jnp.array([1.0])))
        lr_100 = float(schedule(jnp.array(100), jnp.array([1.0])))
        assert lr_0 > lr_100  # should decay

    def test_clipping_max(self):
        schedule = LearningRateSchedule(lambda t, L: jnp.array(10.0), max_lr=1.0)
        lr = float(schedule(jnp.array(0), jnp.array([1.0])))
        assert lr <= 1.0

    def test_clipping_min(self):
        schedule = LearningRateSchedule(lambda t, L: jnp.array(1e-20), min_lr=1e-10)
        lr = float(schedule(jnp.array(0), jnp.array([1.0])))
        assert lr >= 1e-10

    def test_warmup_cosine(self):
        schedule = LearningRateSchedule.warmup_cosine(total_steps=1000, warmup_steps=100, lr0=0.1)
        lr_0 = float(schedule(jnp.array(0), jnp.array([1.0])))
        lr_50 = float(schedule(jnp.array(50), jnp.array([1.0])))
        lr_100 = float(schedule(jnp.array(100), jnp.array([1.0])))
        # During warmup, lr should increase
        assert lr_50 > lr_0 or abs(lr_50 - lr_0) < 0.01  # some tolerance
        # At end of warmup, should be close to lr0
        assert lr_100 == pytest.approx(0.1, abs=0.02)


# ======================================================================
# WeightSchedule
# ======================================================================
class TestWeightSchedule:
    def test_constant_list(self):
        ws = WeightSchedule([1.0, 2.0, 3.0])
        weights = ws(jnp.array(0), jnp.array([1.0, 1.0, 1.0]))
        assert weights.shape == (3,)
        assert jnp.allclose(weights, jnp.array([1.0, 2.0, 3.0]))

    def test_callable(self):
        fn = lambda t, L: jnp.array([1.0, t * 0.01])
        ws = WeightSchedule(fn)
        weights = ws(jnp.array(100), jnp.array([1.0, 1.0]))
        assert weights.shape == (2,)
        assert float(weights[1]) == pytest.approx(1.0)

    def test_clipping_max(self):
        ws = WeightSchedule([1e10], max_weight=100.0)
        weights = ws(jnp.array(0), jnp.array([1.0]))
        assert float(weights[0]) <= 100.0

    def test_clipping_min(self):
        ws = WeightSchedule([-5.0], min_weight=0.0)
        weights = ws(jnp.array(0), jnp.array([1.0]))
        assert float(weights[0]) >= 0.0

    def test_single_float_list(self):
        ws = WeightSchedule([5.0])
        weights = ws(jnp.array(0), jnp.array([1.0]))
        assert float(weights[0]) == pytest.approx(5.0)
