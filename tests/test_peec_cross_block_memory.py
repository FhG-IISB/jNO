"""A pair-block builder must bound what it allocates by the CHUNK, not by the pair count.

This has now been the same defect three times in this module -- `cross_block` chunked one side and
left the other whole, `cross_block` then built a displacement where only a distance was wanted, and
`near_block` built the displacement for every near pair at once. Each produced a result of one scalar
per pair from an intermediate that scaled with all of them. These tests exist so the fourth instance
fails in CI instead of on a real model.

The cross block between two welded discretisations must bound the PAIR block, not one side.

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


# --------------------------------------------------------------------------- near_block
def _plate(pitch):
    """A lattice with enough near neighbours that the pair tensor is the dominant allocation."""
    import jno
    from jno.utils.solver.peec import bar_filaments

    return bar_filaments(jno.Shape.box(0, 0, 0, 0.060, 0.030, 0.001), size=(pitch, pitch, 0.001), sigma=5.8e7)


def _near_pairs(fil):
    from scipy.spatial import cKDTree

    grp, pos = np.asarray(fil.group), np.asarray(fil.pos)
    ne = int(grp.max()) + 1
    cen = np.zeros((ne, 3))
    np.add.at(cen, grp, pos)
    cen /= np.bincount(grp, minlength=ne)[:, None]
    reach = 2.0 * float(np.mean(np.asarray(fil.length)))
    return len(cKDTree(cen).query_pairs(reach)), len(grp) // ne


def test_the_near_field_chunking_does_not_change_the_answer():
    """Every budget is a different tiling of one exact sum over pairs, so all must agree."""
    from jno.utils.solver import peec as P

    fil = _plate(0.002)
    keep = P._NEAR_PAIR_BUDGET
    try:
        P._NEAR_PAIR_BUDGET = 10**12  # one chunk: the whole thing at once
        r0, c0, v0 = P.near_block(fil, G)
        for budget in (10**9, 400_000, 4_000, 200):
            P._NEAR_PAIR_BUDGET = budget
            r1, c1, v1 = P.near_block(fil, G)
            assert np.array_equal(r0, r1) and np.array_equal(c0, c1)
            assert np.allclose(v0, v1, rtol=1e-12, atol=1e-15), budget
    finally:
        P._NEAR_PAIR_BUDGET = keep


def test_the_near_field_does_not_allocate_in_proportion_to_the_pair_count():
    """The regression: cost must follow the CHUNK, not the number of near pairs.

    The unchunked form built (n_pairs, q, q, 3) doubles, and q is twelve once a lattice element
    samples its volume -- 1.1 GB of displacement here before the distances, the einsum output and
    numpy's temporaries, to produce one scalar per pair. Measured chunked: ~340 MB, 31 % of the
    displacement alone.
    """
    import resource

    from jno.utils.solver.peec import near_block

    fil = _plate(0.00035)
    npair, q = _near_pairs(fil)
    assert npair > 200_000 and q == 12  # the case is actually big enough to discriminate

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    near_block(fil, G)
    grew = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) * 1024

    unchunked = npair * q * q * 3 * 8  # the displacement alone, ignoring everything built from it
    assert grew < unchunked / 2, f"grew {grew / 1e6:.0f} MB; the displacement alone is {unchunked / 1e6:.0f} MB"
