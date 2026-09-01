"""The BTTB generator between TWO families on one grid — the two current sheets of a slab.

A slab thick against the skin depth carries a sheet per face, so its partial inductance needs the
coupling between two lattices that share a grid and sit a thickness apart. That block is still
Toeplitz (both families are regular and the offset is constant), but it is NOT symmetric: the offset
has a sign, so K_BA is K_AB transposed. The full 2x2 is symmetric, as a partial inductance must be.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.kernel import lattice_operator

jax.config.update("jax_enable_x64", True)

G = lambda r: 1.0 / r
SHAPE, H = (4, 3, 2), (2e-3, 1.5e-3, 1e-3)
GAP = 0.6e-3  # the slab thickness the two sheets sit either side of


def _subs():
    """Two sub-point sets: one biased to each face, as the sheets of one slab are."""
    a = np.array([[-4e-4, 0.0, +GAP / 2], [+4e-4, 0.0, +GAP / 2]])
    b = np.array([[-4e-4, 0.0, -GAP / 2], [+4e-4, 0.0, -GAP / 2]])
    w = np.array([0.6, 0.4])
    return a, b, w


def _dense_cross(sub_a, sub_b, w_a, w_b):
    """The O(N^2) block the FFT replaces, written out."""
    ijk = np.array([(i, j, k) for i in range(SHAPE[0]) for j in range(SHAPE[1]) for k in range(SHAPE[2])])
    r = ijk * np.array(H)
    n = len(r)
    M = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            s = 0.0
            for a in range(len(sub_a)):
                for b in range(len(sub_b)):
                    d = np.linalg.norm(r[p] + sub_a[a] - r[q] - sub_b[b])
                    s += w_a[a] * w_b[b] * G(d)
            M[p, q] = s
    return M


def test_the_cross_operator_reproduces_the_dense_block():
    """To round-off, which is the claim that makes the FFT path exact rather than an approximation."""
    sa, sb, w = _subs()
    M = _dense_cross(sa, sb, w, w)
    op = lattice_operator(SHAPE, H, G, 0.0, sub=sa, w=w, sub_b=sb, w_b=w)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(M.shape[0])
    got = np.asarray(op(jnp.asarray(x.reshape(SHAPE)))).reshape(-1)
    assert np.allclose(got, M @ x, rtol=1e-10, atol=1e-12)


def test_the_transpose_flag_gives_the_transposed_block():
    """K_BA = K_AB^T. Reversing the generator is conjugation in Fourier space."""
    sa, sb, w = _subs()
    M = _dense_cross(sa, sb, w, w)
    opT = lattice_operator(SHAPE, H, G, 0.0, sub=sa, w=w, sub_b=sb, w_b=w, transpose=True)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(M.shape[0])
    got = np.asarray(opT(jnp.asarray(x.reshape(SHAPE)))).reshape(-1)
    assert np.allclose(got, M.T @ x, rtol=1e-10, atol=1e-12)


def test_the_cross_block_carries_no_self_term():
    """Two sheets of one cell are a THICKNESS apart, so the coincident entry is an ordinary mutual.

    Substituting an analytic self inductance there -- which the same-family generator must do, since
    that entry really is a singular integral -- would be badly wrong. A wildly different self_g must
    change nothing.
    """
    sa, sb, w = _subs()
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal(SHAPE))
    a = np.asarray(lattice_operator(SHAPE, H, G, 0.0, sub=sa, w=w, sub_b=sb, w_b=w)(x))
    b = np.asarray(lattice_operator(SHAPE, H, G, 1e6, sub=sa, w=w, sub_b=sb, w_b=w)(x))
    assert np.allclose(a, b, rtol=1e-12, atol=1e-14)


def test_the_same_family_generator_still_uses_its_self_term():
    """The other half of that: without sub_b this is the old path, and self_g must still land."""
    sa, _sb, w = _subs()
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal(SHAPE))
    a = np.asarray(lattice_operator(SHAPE, H, G, 0.0, sub=sa, w=w)(x))
    b = np.asarray(lattice_operator(SHAPE, H, G, 1e6, sub=sa, w=w)(x))
    assert not np.allclose(a, b)


def test_the_assembled_two_by_two_block_is_symmetric():
    """The invariant the solve rests on: a partial inductance matrix is symmetric.

    Neither cross block is symmetric on its own -- the offset between the sheets has a sign -- so
    this is the check that the pieces compose into something a Krylov solve and its adjoint can both
    trust. Assembled column by column from the operators themselves, not from the dense reference.
    """
    sa, sb, w = _subs()
    n = int(np.prod(SHAPE))
    ops = {
        "AA": lattice_operator(SHAPE, H, G, 3.0, sub=sa, w=w),
        "BB": lattice_operator(SHAPE, H, G, 3.0, sub=sb, w=w),
        "AB": lattice_operator(SHAPE, H, G, 0.0, sub=sa, w=w, sub_b=sb, w_b=w),
        "BA": lattice_operator(SHAPE, H, G, 0.0, sub=sa, w=w, sub_b=sb, w_b=w, transpose=True),
    }
    cols = {}
    for key, op in ops.items():
        M = np.zeros((n, n))
        for c in range(n):
            e = np.zeros(n)
            e[c] = 1.0
            M[:, c] = np.asarray(op(jnp.asarray(e.reshape(SHAPE)))).reshape(-1)
        cols[key] = M
    full = np.block([[cols["AA"], cols["AB"]], [cols["BA"], cols["BB"]]])
    assert np.allclose(full, full.T, rtol=1e-10, atol=1e-12)
    assert np.abs(cols["AB"] - cols["AB"].T).max() > 1e-6  # and the pieces really are asymmetric
