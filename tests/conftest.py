"""Shared fixtures and helpers for jNO tests."""

import os

# Prevent JAX from pre-allocating GPU memory in the pytest process so that
# subprocess-based smoke tests can still initialize their own CUDA contexts.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Use the platform allocator (eager cudaFree) instead of the caching BFC allocator, so GPU
# memory does not accumulate across the many in-process FEM/3D tests and exhaust a small (8 GB)
# GPU when the whole suite runs in one process. Slower per op, but it stops the false CUDA-OOMs.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# The session-wide default stays at float32 (JAX's default); keeping it off avoids breaking the
# many dtype-sensitive tests (bayesian/blackjax, pdeformer/equinox, the dtype defaults). FEM tests
# that genuinely need float64 opt in per-test via their local ``_x64`` autouse fixture.

import warnings

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(autouse=True)
def _restore_x64():
    """Put ``jax_enable_x64`` back the way the test found it.

    The flag is process-wide, and it decides the float width of every subsequent computation. A test
    that flips it and does not put it back silently changes the precision of every test that runs
    after it, in whatever order pytest happened to pick -- so the failure reproduces only in
    company, never alone, and lands on a file that did nothing wrong.

    Measured before this fixture existed::

        pytest tests/test_fdm.py                          ->  41 passed
        pytest tests/test_node_eval.py tests/test_fdm.py  ->  16 failed, 33 passed

    One unrelated 8-test file in front, and `newton_krylov` in test_fdm stops converging, because
    its solves dropped to float32. Restoring per test makes the suite order-independent; the warning
    keeps the offending test named rather than quietly healed.
    """
    prev = jax.config.jax_enable_x64
    yield
    if jax.config.jax_enable_x64 != prev:
        jax.config.update("jax_enable_x64", prev)
        warnings.warn(
            f"this test left jax_enable_x64 as {not prev} instead of {prev}; it has been restored, "
            "but set the flag through a save/restore fixture rather than in the test body",
            RuntimeWarning,
            stacklevel=2,
        )


@pytest.fixture(autouse=True)
def _free_jax_memory():
    """Drop JAX's compiled-executable caches after each test so they don't pile up across the
    suite (pairs with the no-prealloc + platform allocator above to keep the 8 GB GPU from
    OOMing on a full run)."""
    yield
    jax.clear_caches()


class MockDomain:
    """Minimal domain stub for creating Variables without a real mesh."""

    def __init__(self, tags=None, dim=2):
        self.context = {}
        self._param_tags = set()
        self.dimension = dim
        if tags:
            for tag in tags:
                # 10 sample points with `dim` columns
                self.context[tag] = jnp.zeros((10, dim))


@pytest.fixture(autouse=True)
def deterministic_rng():
    """Provide a deterministic JAX PRNG key for every test."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_domain():
    return MockDomain(tags=["x", "y", "z"])


def make_var(tag, dim=None, domain_tags=None):
    """Create a Variable backed by a mock domain."""
    from jno.trace import Variable

    if dim is None:
        dim = [0, 1]
    tags = domain_tags or [tag]
    d = MockDomain(tags=tags)
    return Variable(tag, dim, domain=d)
