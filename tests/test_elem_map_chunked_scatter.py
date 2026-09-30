"""The chunked element scatter slides its last chunk instead of padding every input.

Padding concatenated a filler tail onto each input -- a full copy of it, the scatter's index block
included (one int32 per raw tangent triplet): 81 MiB of index copies in a 24^3 P1/P1 Navier-Stokes
march, for 7 filler cells. Oracle: the chunked scatter equals the unchunked one, values and gradients,
for chunk sizes that divide the cell count and ones that do not.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("n,c", [(10, 3), (12, 4), (33, 8), (100, 32)])
def test_the_chunked_scatter_needs_no_padding(n, c):
    from jno.utils.solver.fem_utils import _chunked_scatter

    rng = np.random.default_rng(n + c)
    x = jnp.asarray(rng.standard_normal((n, 3)))
    idx = jnp.asarray(rng.integers(0, 20, (n, 3)))

    def fn(row):
        return row * row + 1.0

    def full(xx):
        return jnp.zeros(20).at[idx.reshape(-1)].add(jax.vmap(fn)(xx).reshape(-1))

    def chunked(xx):
        return _chunked_scatter(fn, [xx], c, jnp.zeros(20), idx)

    assert float(jnp.abs(chunked(x) - full(x)).max()) < 1e-14
    g1 = jax.grad(lambda xx: (chunked(xx) ** 2).sum())(x)
    g2 = jax.grad(lambda xx: (full(xx) ** 2).sum())(x)
    assert float(jnp.abs(g1 - g2).max()) < 1e-12
