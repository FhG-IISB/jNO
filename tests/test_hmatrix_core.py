"""The hierarchical operator against the dense one it replaces.

`pair_matrix` forms the exact element-by-element matrix, so it is not a "reference implementation"
in the weak sense -- it is the answer. That makes the accuracy claim directly checkable rather than
plausible: an H-matrix asked for `tol` must reproduce `pair_matrix @ x` to `tol`, on point sets
chosen to stress the cluster tree rather than to flatter it.

What is being tested is a COMPRESSION, so the tests that matter are the ones that would catch it
being quietly wrong: that tightening the tolerance actually buys accuracy (a compression that ignores
`tol` would pass a single loose check), that a block it cannot compress is stored densely rather than
approximated badly, and that the gradient survives -- `pair_matrix` is differentiable in the element
positions on purpose, and an accelerated path that lost that would be a silent regression for
inverse design.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.hmatrix import build, cluster, materialize
from jno.utils.solver.kernel import pair_matrix

jax.config.update("jax_enable_x64", True)

G = lambda r: 1.0 / r


def _elements(ne, nsub=4, spread="scattered", seed=0):
    """Elements as clouds of sub-points, in the layout `pair_matrix` expects."""
    rng = np.random.default_rng(seed)
    if spread == "scattered":
        cen = rng.uniform(-1.0, 1.0, (ne, 3))
    elif spread == "clustered":  # two tight balls far apart: every cross block is admissible
        half = ne // 2
        cen = np.concatenate([rng.normal(0, 0.05, (half, 3)), rng.normal(0, 0.05, (ne - half, 3)) + 5.0])
    elif spread == "sheet":  # a flat plate: clusters are wide and thin, the hard case for boxes
        cen = np.stack([rng.uniform(-1, 1, ne), rng.uniform(-1, 1, ne), rng.normal(0, 1e-3, ne)], 1)
    else:
        raise ValueError(spread)
    pos = (cen[:, None, :] + rng.normal(0, 0.01, (ne, nsub, 3))).reshape(-1, 3)
    mom = rng.normal(0, 1.0, (ne * nsub, 3))
    group = np.repeat(np.arange(ne), nsub)
    self_g = rng.uniform(0.5, 1.5, ne)
    return pos, mom, group, self_g


def _dense(pos, mom, group, self_g):
    return np.asarray(pair_matrix(pos, mom, G, self_g, group=group))


@pytest.mark.parametrize("spread", ["scattered", "clustered", "sheet"])
def test_it_reproduces_the_dense_operator(spread):
    """The whole claim, on three point distributions that stress the tree differently."""
    pos, mom, group, self_g = _elements(250, nsub=3, spread=spread)
    K = _dense(pos, mom, group, self_g)
    h = build(pos, mom, group, G, tol=1e-8, leaf=32)
    ap = materialize(h, pos, mom, self_g, G)
    x = np.random.default_rng(1).normal(size=K.shape[0])
    got, want = np.asarray(ap(jnp.asarray(x))), K @ x
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 1e-6, (spread, rel)


def test_a_tighter_tolerance_actually_buys_accuracy():
    """A compression that silently ignored `tol` would pass any single loose check.

    So the test is the TREND: the error falls as the tolerance is tightened, and the rank rises to
    pay for it. Either one alone can be faked; together they cannot.

    Run on well-separated clusters with a LARGE leaf, because that is the regime where compression
    actually happens. On small blocks it does not, and it should not: a rank of half the block size
    costs more to store than the block itself, so those fall back to dense and are then exact at
    every tolerance -- which is correct behaviour, and is pinned by the test below rather than being
    mistaken for the tolerance being ignored.
    """
    pos, mom, group, self_g = _elements(600, nsub=2, spread="clustered")
    K = _dense(pos, mom, group, self_g)
    x = np.random.default_rng(2).normal(size=K.shape[0])
    want = K @ x
    errs, ranks = [], []
    for tol in (1e-2, 1e-4, 1e-6):
        h = build(pos, mom, group, G, tol=tol, leaf=150)
        assert h.far, f"nothing compressed at tol={tol}; the test is not exercising ACA"
        got = np.asarray(materialize(h, pos, mom, self_g, G)(jnp.asarray(x)))
        errs.append(np.linalg.norm(got - want) / np.linalg.norm(want))
        ranks.append(float(np.mean(h.ranks)))
    assert errs[0] > errs[1] > errs[2], errs
    assert ranks[0] <= ranks[1] <= ranks[2], ranks


def test_a_block_that_will_not_compress_is_stored_exactly():
    """The economics: a rank of half the block size is not cheaper than the block, so ACA is
    abandoned and the entries are kept. That must be EXACT, not approximate -- a fallback that
    quietly kept a bad low-rank fit would be the worst of both.

    Scattered points on a small leaf are exactly that regime, so every block goes dense and the whole
    operator reduces to `pair_matrix` to round-off, whatever tolerance was asked for.
    """
    pos, mom, group, self_g = _elements(300, spread="scattered")
    K = _dense(pos, mom, group, self_g)
    x = np.random.default_rng(9).normal(size=K.shape[0])
    h = build(pos, mom, group, G, tol=1e-4, leaf=32)
    got = np.asarray(materialize(h, pos, mom, self_g, G)(jnp.asarray(x)))
    assert np.linalg.norm(got - K @ x) / np.linalg.norm(K @ x) < 1e-12


def test_the_operator_is_differentiable_in_the_positions():
    """`pair_matrix` is differentiable w.r.t. geometry deliberately -- its NaN guard exists because
    someone differentiated it. An accelerated path that lost that would break inverse design silently,
    since the forward value would stay perfectly correct.

    The pivots are frozen, so this differentiates the SKELETON at fixed indices, which is exactly the
    structure-frozen / values-traced split used everywhere else in this codebase.
    """
    pos, mom, group, self_g = _elements(120, nsub=2, spread="scattered")
    h = build(pos, mom, group, G, tol=1e-10, leaf=32)
    x = jnp.asarray(np.random.default_rng(3).normal(size=h.ne))

    def f(p):
        return jnp.sum(materialize(h, p, mom, self_g, G)(x) ** 2)

    g = jax.grad(f)(jnp.asarray(pos))
    assert np.all(np.isfinite(np.asarray(g)))
    # against a central difference on one coordinate
    k, eps = 17, 1e-6
    pp = np.array(pos)
    pp[k, 0] += eps
    hi = float(f(jnp.asarray(pp)))
    pp[k, 0] -= 2 * eps
    lo = float(f(jnp.asarray(pp)))
    assert abs(float(g[k, 0]) / ((hi - lo) / (2 * eps)) - 1) < 1e-4


def test_a_complex_vector_goes_through_by_parts():
    """The currents are complex at every frequency that matters, and the kernel is real."""
    pos, mom, group, self_g = _elements(150)
    ap = materialize(build(pos, mom, group, G, tol=1e-8, leaf=32), pos, mom, self_g, G)
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=150), rng.normal(size=150)
    got = np.asarray(ap(jnp.asarray(a + 1j * b)))
    assert np.allclose(got, np.asarray(ap(jnp.asarray(a))) + 1j * np.asarray(ap(jnp.asarray(b))))


def test_every_element_lands_in_exactly_one_leaf():
    """A cluster tree that dropped or duplicated an element would give a plausible wrong answer --
    the operator would simply be missing some coupling, which no norm check on a smooth field
    reliably catches."""
    cen = np.random.default_rng(5).uniform(-1, 1, (300, 3))
    nodes, children = cluster(cen, leaf=16)
    leaves = [nodes[i] for i, c in enumerate(children) if c is None]
    allidx = np.sort(np.concatenate(leaves))
    assert np.array_equal(allidx, np.arange(300))
    assert all(len(n) <= 16 for n in leaves)


def test_the_blocks_cover_the_whole_matrix_exactly_once():
    """The same completeness claim one level up: near plus far must tile the (ne, ne) matrix, with
    the diagonal carried by the self term. Overlap would double-count a coupling."""
    pos, mom, group, self_g = _elements(160, nsub=2)
    h = build(pos, mom, group, G, tol=1e-6, leaf=32)
    cover = np.zeros((h.ne, h.ne), dtype=int)
    for r, c in h.near:
        cover[np.ix_(r, c)] += 1
    for r, c, _i, _j in h.far:
        cover[np.ix_(r, c)] += 1
    assert cover.min() == 1 and cover.max() == 1, (cover.min(), cover.max())


def test_a_traced_geometry_is_refused():
    """The pivots ARE the structure. Choosing them per trace would mean a different operator on
    every call, so this raises rather than quietly rebuilding."""
    pos, mom, group, _s = _elements(60)

    def f(p):
        build(p, mom, group, G)
        return 0.0

    with pytest.raises(ValueError, match="tracer"):
        jax.jit(f)(jnp.asarray(pos))


def test_ragged_sub_points_are_refused():
    """A uniform sub-point count is what makes the inner sum one einsum; `near_block` imposes the
    same rule for the same reason."""
    pos, mom, _g, _s = _elements(10, nsub=4)
    bad = np.concatenate([np.repeat(np.arange(9), 4), np.array([9, 9, 9, 9])])[: len(pos)]
    bad[-1] = 8  # element 9 now has 3 sub-points, element 8 has 5
    with pytest.raises(ValueError, match="different numbers of sub-points"):
        build(pos, mom, bad, G)
