"""`cross_block_nearfar`: the wire-lattice coupling with the far field on one point per element.

Oracle: the exact sub-point double sum, `cross_block`. With every pair inside the near radius the two
are the same sum; at a finite radius the one-point error falls with the radius (O((r/d)^2) per far
pair); a near set cut short by `knear` is a NaN block, never a quietly incomplete one.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.peec import cross_block, cross_block_nearfar


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


NA, NB, SA, SB = 12, 300, 4, 8  # elements and sub-points per element on each side


def _side(n, s, spread, lo, hi, seed):
    """`n` compact elements of `s` sub-points each, `spread` around centres drawn in [lo, hi)^3."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(lo, hi, (n, 3))
    pos = np.repeat(c, s, axis=0) + rng.uniform(-spread, spread, (n * s, 3))
    # a filament's sub-points share its direction, as a real element's do
    mom = np.repeat(rng.normal(size=(n, 3)), s, axis=0) * 1e-4 / s
    return pos, mom, np.repeat(np.arange(n), s)


def _sides():
    pa, ma, ga = _side(NA, SA, 1e-4, 4e-3, 6e-3, 0)  # a short wire's elements
    pb, mb, gb = _side(NB, SB, 3e-4, 0.0, 1e-2, 1)  # a lattice around it
    return pa, ma, ga, pb, mb, gb


def _g(r):
    return 1.0 / r


def _blocks(reach, knear=NB):
    pa, ma, ga, pb, mb, gb = _sides()
    exact = np.asarray(cross_block(pa, ma, ga, NA, pb, mb, gb, NB, _g))
    approx = np.asarray(cross_block_nearfar(pa, ma, ga, NA, pb, mb, gb, NB, _g, reach=reach, knear=knear, chunk=5))
    return exact, approx


def test_with_every_pair_near_it_is_the_exact_block():
    exact, approx = _blocks(reach=1e6)
    assert approx.shape == (NA, NB)
    assert np.abs(approx - exact).max() < 1e-12 * np.abs(exact).max()


def test_the_far_field_error_falls_with_the_near_radius():
    errs = []
    for reach in (2.0, 4.0, 8.0):
        exact, approx = _blocks(reach)
        errs.append(np.abs(approx - exact).max() / np.abs(exact).max())
    # measured 7.9e-3, 2.8e-3, 3.5e-4 of the largest entry
    assert errs[2] < 1e-3
    assert errs[0] > errs[1] > 3 * errs[2]


def test_a_truncated_near_set_is_nan_not_a_silent_miss():
    _exact, approx = _blocks(reach=1e6, knear=8)
    assert np.isnan(approx).all()


def test_the_gradient_matches_the_exact_block_when_every_pair_is_near():
    pa, ma, ga, pb, mb, gb = _sides()

    def tot(f, p, **kw):
        return jnp.sum(f(p, ma, ga, NA, pb, mb, gb, NB, _g, **kw) ** 2)

    g_exact = np.asarray(jax.grad(lambda p: tot(cross_block, p))(pa))
    g_nf = np.asarray(jax.grad(lambda p: tot(cross_block_nearfar, p, reach=1e6, knear=NB, chunk=5))(pa))
    assert np.abs(g_nf - g_exact).max() < 1e-10 * np.abs(g_exact).max()


def test_the_thin_side_may_come_second():
    pa, ma, ga, pb, mb, gb = _sides()
    ab = np.asarray(cross_block_nearfar(pa, ma, ga, NA, pb, mb, gb, NB, _g, reach=8.0, knear=NB))
    ba = np.asarray(cross_block_nearfar(pb, mb, gb, NB, pa, ma, ga, NA, _g, reach=8.0, knear=NB))
    assert np.allclose(ba, ab.T, rtol=1e-12, atol=0)


def test_the_counted_candidate_count_covers_the_near_set_and_no_more():
    """`_counted_knear`: the candidate count from the near set COUNTED on concrete geometry. With no
    margin it is the largest near set plus one (the truncation guard needs a candidate beyond it) -- the
    block equals the uncapped one, and one fewer candidate is the truncated NaN; with the margin it is
    that count times the margin."""
    from jno.utils.solver.peec import _counted_knear

    pa, ma, ga, pb, mb, gb = _sides()
    pos, grp = np.concatenate([pa, pb]), np.concatenate([ga, gb + NA])
    ri, rj = np.arange(len(pa)), np.arange(len(pa), len(pa) + len(pb))
    reach = 3.0
    k = _counted_knear(pos, grp, ri, rj, reach, margin=1.0)
    # brute force: centres, radii, count per thin-side element
    ca = pa.reshape(NA, SA, 3).mean(1)
    cb = pb.reshape(NB, SB, 3).mean(1)
    ra = np.linalg.norm(pa.reshape(NA, SA, 3) - ca[:, None], axis=-1).max(1)
    rb = np.linalg.norm(pb.reshape(NB, SB, 3) - cb[:, None], axis=-1).max(1)
    near = (np.linalg.norm(ca[:, None] - cb[None], axis=-1) < reach * (ra[:, None] + rb[None])).sum(1)
    assert k == near.max() + 1 and near.max() > 0
    full = np.asarray(cross_block_nearfar(pa, ma, ga, NA, pb, mb, gb, NB, _g, reach=reach, knear=NB, chunk=5))
    tight = np.asarray(cross_block_nearfar(pa, ma, ga, NA, pb, mb, gb, NB, _g, reach=reach, knear=k, chunk=5))
    assert np.abs(tight - full).max() <= 1e-13 * np.abs(full).max()  # the same pairs, summed in another order
    short = np.asarray(cross_block_nearfar(pa, ma, ga, NA, pb, mb, gb, NB, _g, reach=reach, knear=k - 1, chunk=5))
    assert np.any(np.isnan(short))  # loud: the chunk that holds the largest near set is NaN
    assert _counted_knear(pos, grp, ri, rj, reach, margin=1.5) == min(NB, int(np.ceil(1.5 * near.max())))
