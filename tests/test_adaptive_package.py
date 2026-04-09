"""Tests for the new ``jno.utils.adaptive`` package implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import jno.utils.adaptive as adaptive_pkg
from jno.utils.adaptive.lrscheduler import DLRS
from jno.utils.adaptive.weights import LbPINNsLossBalancing, ReLoBRaLo, SoftAdapt, DWA, RLW


def test_adaptive_import_resolves_to_package_directory():
    """Ensure ``jno.utils.adaptive`` points to the new package, not adaptive.py."""
    resolved = adaptive_pkg.__file__.replace("\\", "/")
    assert resolved.endswith("jno/utils/adaptive/__init__.py")


def test_dlrs_decreases_on_rising_losses_and_recovers_on_falling_losses():
    """DLRS should reduce LR for rising losses and increase LR for falling losses."""
    sched = DLRS(
        lr0=1e-3,
        window=3,
        decremental_factor=0.5,
        stagnation_factor=0.1,
        increment_factor=0.1,
        min_lr=1e-6,
        max_lr=1e-1,
    )

    lr0 = float(sched(0, jnp.array([1.0], dtype=jnp.float32)))
    _ = float(sched(1, jnp.array([2.0], dtype=jnp.float32)))
    lr_down = float(sched(2, jnp.array([3.5], dtype=jnp.float32)))
    assert lr_down < lr0

    lr_up = float(sched(3, jnp.array([0.5], dtype=jnp.float32)))
    assert lr_up > lr_down


def test_relobralo_outputs_valid_weights_and_prioritizes_larger_loss():
    """ReLoBRaLo should return positive weights and favor the larger loss term."""
    balancer = ReLoBRaLo(num_losses=2, alpha=0.5, tau=0.2, expected_rho=1.0, seed=0)

    w0_a, w0_b = balancer(jnp.array([1.0, 1.0], dtype=jnp.float32))
    assert jnp.allclose(w0_a, 1.0)
    assert jnp.allclose(w0_b, 1.0)

    w1_a, w1_b = balancer(jnp.array([8.0, 1.0], dtype=jnp.float32))
    assert bool(jnp.isclose(w1_a + w1_b, 2.0, atol=1e-5))
    assert float(w1_a) > float(w1_b)


def test_lbpinns_updates_weights_toward_large_loss_downweighting():
    """LbPINNs should learn smaller weight for persistently larger loss."""
    balancer = LbPINNsLossBalancing(num_losses=2, init_s=0.0, lr_s=0.1)

    w_a, w_b = None, None
    for _ in range(5):
        w_a, w_b = balancer(jnp.array([100.0, 1.0], dtype=jnp.float32))

    assert w_a is not None
    assert float(w_a) < float(w_b)


def test_softadapt_upweights_non_decreasing_loss():
    """SoftAdapt should give more weight to a loss that stops decreasing."""
    balancer = SoftAdapt(num_losses=2, beta=0.1)

    # First call (initialisation) → uniform weights
    w0_a, w0_b = balancer(jnp.array([1.0, 1.0], dtype=jnp.float32))
    assert jnp.allclose(w0_a, 1.0)
    assert jnp.allclose(w0_b, 1.0)

    # L0 stays constant, L1 drops → ratio0 > ratio1 → w0 > w1
    w1_a, w1_b = balancer(jnp.array([1.0, 0.1], dtype=jnp.float32))
    assert float(w1_a) > float(w1_b)


def test_dwa_upweights_increasing_loss():
    """DWA should give more weight to a loss whose rate of descent is poor."""
    balancer = DWA(num_losses=2, temperature=1.0)

    # Warm-up: two history steps return uniform
    balancer(jnp.array([1.0, 1.0], dtype=jnp.float32))
    balancer(jnp.array([2.0, 0.5], dtype=jnp.float32))

    # L0 went 1→2 (rate=2), L1 went 1→0.5 (rate=0.5) → w0 > w1
    w_a, w_b = balancer(jnp.array([3.0, 0.4], dtype=jnp.float32))
    assert float(w_a) > float(w_b)


def test_rlw_returns_correct_shape_and_positive_weights():
    """RLW should return positive weights summing to num_losses (in expectation)."""
    balancer = RLW(num_losses=3, alpha=1.0, seed=0)

    w = balancer(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    assert len(w) == 3
    assert all(float(wi) > 0 for wi in w)
    # Mean over many draws ≈ 1 per component, but single draw sum = N
    assert bool(jnp.isclose(sum(float(wi) for wi in w), 3.0, atol=1e-5))


def test_dwa_supports_jitted_value_and_grad_without_jvp_error():
    """Adaptive callbacks should behave as non-differentiable control signals."""
    balancer = DWA(num_losses=2, temperature=1.0)

    def objective(theta):
        losses = jnp.stack([theta**2, (theta - 1.0) ** 2]).astype(jnp.float32)
        w0, w1 = balancer(losses)
        return w0 * losses[0] + w1 * losses[1]

    value, grad = jax.jit(jax.value_and_grad(objective))(jnp.array(0.3, dtype=jnp.float32))

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(grad))
