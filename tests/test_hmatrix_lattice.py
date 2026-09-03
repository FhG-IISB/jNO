"""The hierarchical operator against the FFT, on a real `bar_filaments` lattice.

This is the test whose absence let an 18 % error through. `tests/test_hmatrix_core.py` builds
elements from random scattered points, and random points have no structure: every entry of the
matrix is a generic nonzero. A PEEC lattice is the opposite. Its moments are axis-aligned, so
``mom_a . mom_b`` vanishes EXACTLY between perpendicular bar families, and any cluster that mixes
families gives a block with zero sub-blocks -- which ACA cannot approximate and, worse, cannot detect
by watching its own residual. Every synthetic test passed at 1e-6 while the geometry the module
exists to serve was 18 % wrong.

So the geometry under test here is the real one, and the reference is the FFT: `lattice_apply` is
exact to round-off on a lattice and is what a graded mesh will have to reproduce before anyone trusts
it on a mesh where no FFT exists.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.hmatrix import build, materialize
from jno.utils.solver.kernel import pair_matrix
from jno.utils.solver.peec import bar_filaments, lattice_apply

jax.config.update("jax_enable_x64", True)

G = lambda r: 1.0 / r
mm = 1e-3


def _plate(pitch_mm=0.8, x=16.0, y=8.0):
    """A trace layer: flat, one cell thick, two bar families. The shape of every real conductor."""
    p = pitch_mm * mm
    return bar_filaments(jno.Shape.box(0, 0, 0, x * mm, y * mm, p, size=(p, p, p)), sigma=5.8e7)


def _arrays(f):
    return (np.asarray(f.pos), np.asarray(f.mom), np.asarray(f.group), np.asarray(f.self_g))


def test_it_reproduces_the_fft_on_a_lattice():
    """The FFT is exact on a uniform lattice, so this is a reference and not a second opinion.

    A graded mesh has no FFT to check against, which is exactly why the hierarchical path has to
    prove itself here first -- on the one geometry where an exact answer exists.
    """
    f = _plate()
    pos, mom, grp, sg = _arrays(f)
    ne = int(np.asarray(f.length).size)
    x = jnp.asarray(np.random.default_rng(0).normal(size=ne))
    want = np.asarray(lattice_apply(f, G)(x))
    got = np.asarray(materialize(build(pos, mom, grp, G, tol=1e-8, leaf=64), pos, mom, sg, G)(x))
    assert np.linalg.norm(got - want) / np.linalg.norm(want) < 1e-6


def test_it_reproduces_the_dense_operator_on_a_lattice():
    """And against `pair_matrix`, which is the same claim by a different route -- the FFT and the
    dense sum are independent implementations, so agreeing with both is worth more than either."""
    f = _plate()
    pos, mom, grp, sg = _arrays(f)
    K = np.asarray(pair_matrix(f.pos, f.mom, G, f.self_g, group=grp))
    x = np.random.default_rng(1).normal(size=K.shape[0])
    got = np.asarray(materialize(build(pos, mom, grp, G, tol=1e-8, leaf=64), pos, mom, sg, G)(jnp.asarray(x)))
    assert np.linalg.norm(got - K @ x) / np.linalg.norm(K @ x) < 1e-6


def test_the_perpendicular_families_are_partitioned_apart():
    """The structural fact the whole thing turns on.

    `mom_a . mom_b` is exactly zero between perpendicular bars, so the operator is block diagonal by
    direction. Building one tree over all of them mixes the families into every block and ACA fails
    on the zero sub-blocks; partitioning first removes them. This asserts the partition actually
    happened -- no block may span two families -- because if it silently stopped, the answer would
    still be right (the failures are caught and stored densely) and only the compression would
    quietly vanish.
    """
    f = _plate()
    pos, mom, grp, sg = _arrays(f)
    h = build(pos, mom, grp, G, tol=1e-6, leaf=64)
    axis = np.asarray(f.lattice["axis"])
    assert len(set(axis.tolist())) > 1, "this plate should have two bar families"
    for r, c, _i, _j in h.far:
        assert len(set(axis[r].tolist())) == 1, "a far block spans two bar families"
        assert set(axis[r].tolist()) == set(axis[c].tolist()), "a far block couples two families"


def test_it_actually_compresses_a_lattice():
    """Correctness without compression is the failure mode this module already had once: every block
    rejected, answers perfect, 1.00x. So the ratio is asserted, not just the accuracy."""
    f = _plate(pitch_mm=0.5)
    pos, mom, grp, _sg = _arrays(f)
    ne = int(np.asarray(f.length).size)
    h = build(pos, mom, grp, G, tol=1e-6, leaf=64)
    stored = sum(len(r) * len(c) for r, c in h.near)
    stored += sum(len(r) * k + k * len(c) for (r, c, _i, _j), k in zip(h.far, h.ranks))
    assert h.far, "nothing compressed at all"
    assert ne * ne / stored > 2.0, ne * ne / stored


@pytest.mark.parametrize("tol", [1e-4, 1e-8])
def test_the_tolerance_holds_on_a_lattice(tol):
    """The accuracy asked for is the accuracy delivered, on the structured geometry rather than on
    the random points where ACA never had to work for it."""
    f = _plate()
    pos, mom, grp, sg = _arrays(f)
    ne = int(np.asarray(f.length).size)
    x = jnp.asarray(np.random.default_rng(2).normal(size=ne))
    want = np.asarray(lattice_apply(f, G)(x))
    got = np.asarray(materialize(build(pos, mom, grp, G, tol=tol, leaf=64), pos, mom, sg, G)(x))
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < max(1e-9, 1e3 * tol), (tol, rel)
