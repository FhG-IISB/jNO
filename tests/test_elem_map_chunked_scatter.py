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

    # Relative: a scatter-add sums in an unspecified order (atomics on a GPU), so the last bits move.
    assert float(jnp.abs(chunked(x) - full(x)).max()) <= 1e-14 * float(jnp.abs(full(x)).max())
    g1 = jax.grad(lambda xx: (chunked(xx) ** 2).sum())(x)
    g2 = jax.grad(lambda xx: (full(xx) ** 2).sum())(x)
    assert float(jnp.abs(g1 - g2).max()) < 1e-12


@pytest.mark.parametrize("chunk", [None, 4], ids=["one_vmap", "chunked"])
def test_a_wrapped_kernel_whose_parameter_buffer_was_donated_is_recompiled(chunk):
    """The chunked tangent passes its element kernel as ``lambda c, la, _k=kernel: ...``, so the runtime
    parameter the kernel closes over sits one closure level down. ``elem_map``'s cache shares a compiled
    program between equal-content builds, and checks that the program's own baked buffers are still alive
    (an optimizer step donates the old parameter buffer). The check read only the top-level leaves -- one
    function -- so the second of two crux recoveries of one form ran a compilation whose parameter buffer
    was gone: "Array has been deleted with shape=float64[1]". Oracle: the second build runs, and agrees."""
    from jno.utils.solver.fem_utils import elem_map

    def build(p):
        def kernel(c, x):
            return x * p[0] + c

        return lambda c, x, _k=kernel: _k(c, x).reshape(-1)

    def run(p, n):  # n cells: a new size makes the shared program RETRACE, through its original closure
        xs = (jnp.arange(float(n)), jnp.ones((n, 1)))
        return np.asarray(elem_map(build(p), xs, chunk, scatter=(jnp.zeros((n,)), jnp.arange(n).reshape(n, 1))))

    p1 = jnp.array([2.0])
    assert np.array_equal(run(p1, 6), np.arange(6.0) + 2.0)
    p1.delete()  # what a donating optimizer step does to the old parameter buffer
    assert np.array_equal(run(jnp.array([2.0]), 8), np.arange(8.0) + 2.0)
