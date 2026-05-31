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
