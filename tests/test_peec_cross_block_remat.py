"""`cross_block` under reverse mode keeps its INPUTS, not its pair blocks.

The block between a bond wire and the lattice it welds to is built in chunks of up to `budget` pairs,
precisely so that no (wire sub-points x lattice sub-points) array ever exists at once (see
`test_peec_cross_block_memory.py`). Under a gradient in the wire's SHAPE that bound was lost: reverse
mode saved every chunk's distance and kernel arrays for the backward pass, so the memory was the whole
pair count again. Measured on a 54,866-element power module with 12 traced source wires: 30 GB and the
OOM killer within four minutes, where the same solve with only the pads traced peaked at 11 GB.

Each chunk is now rematerialised in the backward pass, so what is saved is what it was computed from.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.ad_checkpoint import saved_residuals  # public only as print_saved_residuals in jax 0.10

from jno.utils.solver.peec import cross_block

jax.config.update("jax_enable_x64", True)

NA, NB, SA, SB = 40, 3000, 4, 12  # elements and sub-points per element on each side


def _sides():
    rng = np.random.default_rng(0)
    pa = rng.uniform(0, 1e-2, (NA * SA, 3))
    pb = rng.uniform(0, 1e-2, (NB * SB, 3)) + np.array([0, 0, 2e-3])
    ma = rng.normal(size=(NA * SA, 3)) * 1e-4
    mb = rng.normal(size=(NB * SB, 3)) * 1e-4
    ga, gb = np.repeat(np.arange(NA), SA), np.repeat(np.arange(NB), SB)
    return pa, ma, ga, pb, mb, gb


def _g(r):
    return 1.0 / r


def _total(pa, chunk, budget):
    _pa, ma, ga, pb, mb, gb = _sides()
    K = cross_block(pa, ma, ga, NA, pb, mb, gb, NB, _g, chunk=chunk, budget=budget)
    return jnp.sum(K * K)


def test_the_backward_pass_does_not_store_the_pair_blocks():
    """No (chunk_a, chunk_b) array is saved, and the total no longer scales with the pair count.

    What remains is the inputs, re-sliced once per `a` chunk, and the block itself -- so it grows
    with the number of WIRE chunks (one to three on a module), not with wire x lattice pairs.
    """
    pa = _sides()[0]
    chunk, budget = 16, 16 * 1500  # 10 a-chunks x 24 b-chunks of (16, 1500) pairs
    saved = saved_residuals(lambda p: _total(p, chunk, budget), pa)
    assert not [a for a, _src in saved if tuple(a.shape) == (chunk, budget // chunk)]
    floats = sum(int(np.prod(a.shape)) for a, _src in saved)
    pa_, ma, _ga, pb, mb, _gb = _sides()
    a_chunks = -(-pa_.shape[0] // chunk)
    assert floats < (a_chunks + 2) * (pa_.size + ma.size + pb.size + mb.size + NA * NB)
    assert floats < NA * SA * NB * SB  # was 6.7x the pair count before the remat


def test_the_gradient_is_unchanged_by_the_chunking():
    """Rematerialising is exact: the chunked gradient matches one unchunked block to rounding.

    Rounding, not bits: each chunk centres its own coordinates, so the r^2 cancellation differs
    between chunkings (measured 2.6e-12 of the largest entry).
    """
    pa = _sides()[0]
    g_chunked = jax.grad(lambda p: _total(p, 16, 16 * 1500))(pa)
    g_whole = jax.grad(lambda p: _total(p, 10**9, 10**12))(pa)
    a, b = np.asarray(g_chunked), np.asarray(g_whole)
    assert np.abs(a - b).max() < 1e-10 * np.abs(b).max()
