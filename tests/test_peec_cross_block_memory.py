"""The cross block between two welded discretisations must bound the PAIR block, not one side.

`cross_block` chunked only its `a` side. In a welded network the sides are wildly asymmetric and the
big one is `b`: the thin side is a bond wire, the wide one is the lattice it welds to. The
displacement array is then (chunk_a, ALL of b, 3), which on a 12,000-element power module reached
6.96 GB -- with the distances and the kernel on top, a 14.6 GB peak against 18 GB of machine. Every
attempt to solve that layout was killed, and the module could not be run at any pitch fine enough to
resolve its geometry.

Bounding the pair block instead: 14.6 GB -> 3.6 GB peak, and a 21,246-element solve that never
finished now takes 50 s. Chunking BOTH sides at 2048 also bounds it but tiles far too finely (27.7 s
against 4.5 s, all dispatch), which is why the budget is on the product.
"""

import resource

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.peec import cross_block

jax.config.update("jax_enable_x64", True)

G = lambda r: 1.0 / r


def _sides(na, nb, sub=4, seed=0):
    rng = np.random.default_rng(seed)
    pa = rng.standard_normal((na * sub, 3))
    pb = rng.standard_normal((nb * sub, 3)) + np.array([6.0, 0.0, 0.0])  # apart, so 1/r is tame
    ma = rng.standard_normal((na * sub, 3))
    mb = rng.standard_normal((nb * sub, 3))
    return pa, ma, np.repeat(np.arange(na), sub), pb, mb, np.repeat(np.arange(nb), sub)


def test_the_chunking_does_not_change_the_answer():
    """Every chunk/budget pair is a different tiling of one exact sum, so all must agree."""
    pa, ma, ga, pb, mb, gb = _sides(40, 90)
    ref = np.asarray(cross_block(pa, ma, ga, 40, pb, mb, gb, 90, G, chunk=10**9, budget=10**12))
    for chunk, budget in ((8, 64), (17, 1000), (2048, 32_000_000), (3, 9)):
        got = np.asarray(cross_block(pa, ma, ga, 40, pb, mb, gb, 90, G, chunk=chunk, budget=budget))
        assert np.allclose(got, ref, rtol=1e-11, atol=1e-13), (chunk, budget)


def test_it_is_the_transpose_the_other_way_round():
    """A partial inductance block is symmetric under swapping the two conductors."""
    pa, ma, ga, pb, mb, gb = _sides(25, 60, seed=3)
    ab = np.asarray(cross_block(pa, ma, ga, 25, pb, mb, gb, 60, G))
    ba = np.asarray(cross_block(pb, mb, gb, 60, pa, ma, ga, 25, G))
    assert np.allclose(ab, ba.T, rtol=1e-11, atol=1e-13)


@pytest.mark.filterwarnings("ignore")
def test_a_wide_b_side_does_not_allocate_in_proportion_to_it():
    """The regression itself: cost must follow the BLOCK, not the wide side.

    `a` is small and `b` is 200x larger, which is the shape a bond wire welded to a lattice has. The
    unbounded version allocated `chunk_a * n_b * 3` doubles here regardless of the budget asked for.
    """
    na, nb, sub = 60, 12_000, 4
    pa, ma, ga, pb, mb, gb = _sides(na, nb, sub=sub, seed=7)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = cross_block(pa, ma, ga, na, pb, mb, gb, nb, G, chunk=2048, budget=2_000_000)
    out.block_until_ready()
    grew = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * 1024

    # the pair block asked for is 2e6 doubles; the wide side, unbounded, would be 2048 * 48000 * 3
    unbounded = 2048 * nb * sub * 3 * 8
    assert grew < unbounded / 4, f"grew {grew / 1e6:.0f} MB; unbounded would be {unbounded / 1e6:.0f} MB"
    assert out.shape == (na, nb)
