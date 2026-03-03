"""Shared fixtures and helpers for jNO tests."""

import pytest
import jax
import jax.numpy as jnp


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
