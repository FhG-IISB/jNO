"""Shared fixtures and helpers for jNO tests."""

import os

# Prevent JAX from pre-allocating GPU memory in the pytest process so that
# subprocess-based smoke tests can still initialize their own CUDA contexts.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Use the platform allocator (eager cudaFree) instead of the caching BFC allocator, so GPU
# memory does not accumulate across the many in-process FEM/3D tests and exhaust a small (8 GB)
# GPU when the whole suite runs in one process. Slower per op, but it stops the false CUDA-OOMs.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# Keep the *session-wide* default at float32 (JAX's default). feax enables ``jax_enable_x64``
# globally at import time (its own default), which — because pytest imports every FEM test
# module during collection — would otherwise flip x64 on for the whole run and break the many
# dtype-sensitive tests (bayesian/blackjax, pdeformer/equinox, the dtype defaults). We use
# feax's own documented ``FEAX_X64`` override (not a feax patch) to keep the default off; FEM
# tests that genuinely need float64 opt in per-test via their local ``_x64`` autouse fixture.
# Must be set before feax is first imported (i.e. before any test module is collected).
os.environ.setdefault("FEAX_X64", "0")

import jax
import jax.numpy as jnp
import pytest


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
