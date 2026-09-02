"""`K`, the electric-magnetic coupling -- the curl block, and the one that fails quietly.

An electric element drives a magnetomotive force around a magnetic one by Ampere's law, so the kernel
is the CURL of the Green function rather than the Green function itself:

    K[m, e] = (1 / 4 pi) L_a L_b < (e_a x r_hat) . e_b / r^2 >

averaged over both element volumes. pypeec reaches the same thing by differencing two half-cell
shifted Green evaluations; this builds the derivative kernel directly, which is why
`lattice_operator` grew a `generator=` argument -- and which also means the cancellation of two
nearly-equal numbers, the risk in the differenced form, never arises here.

A SIGN error in this block does not blow up. It produces a plausible, smooth, wrong-signed coupling
whose first symptom would be a negative or cancelled inductance three phases later, with nothing
pointing back here. So the sign is pinned in isolation, against Ampere's law.
"""

import jax
import numpy as np
import pytest

from jno.utils.solver.peec import _bar_rule, coupling_generator

jax.config.update("jax_enable_x64", True)

N, D, Q = (3, 3, 2), (0.002, 0.002, 0.002), 2


def _cells():
    return np.array([(i, j, k) for i in range(N[0]) for j in range(N[1]) for k in range(N[2])], float)


def _dense(a, b):
    """The coupling straight from Ampere's law, element by element -- O(N^2), no FFT, no generator."""
    sa, wa = _bar_rule(D, a, Q)
    sb, wb = _bar_rule(D, b, Q)
    ea, eb = np.eye(3)[a], np.eye(3)[b]
    pos = _cells() * np.array(D)
    m = len(pos)
    K = np.zeros((m, m))
    for u in range(m):
        for v in range(m):
            acc = 0.0
            for p in range(len(sa)):
                for w in range(len(sb)):
                    r = (pos[v] + sb[w]) - (pos[u] + sa[p])
                    rn = float(np.linalg.norm(r))
                    if rn == 0.0:
                        continue
                    acc += wa[p] * wb[w] * float(np.dot(np.cross(ea, r / rn), eb)) / rn**2
            K[v, u] = acc * D[a] * D[b] / (4.0 * np.pi)
    return K


def _from_generator(a, b):
    gen = coupling_generator(N, D, a, b, Q)
    cells = _cells()
    m = len(cells)
    K = np.zeros((m, m))
    wrap = 2 * np.array(N)
    for u in range(m):
        for v in range(m):
            o = (cells[v] - cells[u]).astype(int)
            K[v, u] = gen[tuple(o % wrap)]
    return K


def test_the_generator_is_the_biot_savart_coupling():
    """Against a direct element-by-element Ampere sum -- O(N^2), wrong in a different way."""
    for a, b in ((0, 1), (1, 2), (2, 0)):
        K, G = _dense(a, b), _from_generator(a, b)
        assert np.allclose(G, K, rtol=1e-12, atol=1e-18 + 1e-12 * np.abs(K).max()), (a, b)


def test_a_bar_drives_no_circulation_around_its_own_orientation():
    """e_a x r_hat is perpendicular to e_a, so its projection on e_a vanishes identically -- not
    merely to round-off. A non-zero diagonal pair would mean a current linking its own flux path."""
    for a in (0, 1, 2):
        assert np.all(coupling_generator(N, D, a, a, Q) == 0.0)


def test_reciprocity_relates_the_two_orientations():
    """gen_ba[-delta] = gen_ab[delta]: reversing the separation flips r_hat AND swaps the cross
    product's arguments, and the two sign changes cancel. Index reversal, no sign flip."""
    gab = coupling_generator(N, D, 0, 1, Q)
    gba = coupling_generator(N, D, 1, 0, Q)
    rev = gba[tuple(np.ix_(*[(-np.arange(2 * v)) % (2 * v) for v in N]))]
    assert np.allclose(rev, gab, rtol=1e-11, atol=1e-16 * np.abs(gab).max() + 1e-20)


def test_the_sign_follows_amperes_law():
    """A current in +x, seen from +z, produces H along -y: e_x x e_z = -e_y.

    This is the assertion that catches a transposed cross product or a flipped separation, both of
    which leave every other test in this file passing.
    """
    gen = coupling_generator(N, D, 0, 1, Q)  # electric along x, magnetic along y
    at_plus_z = gen[(0, 0, 1)]  # magnetic element one pitch along +z from the electric one
    assert at_plus_z < 0, at_plus_z
    at_minus_z = gen[(0, 0, (-1) % (2 * N[2]))]
    assert at_minus_z > 0  # and the other side is the mirror
    assert abs(at_plus_z + at_minus_z) < 1e-12 * abs(at_plus_z)


def test_it_stays_accurate_as_the_elements_separate():
    """The derivative kernel is built directly, so there is no difference of nearly-equal numbers to
    lose precision to -- unlike the shifted-Green form. Checked out to the far corner of the grid."""
    K, G = _dense(0, 1), _from_generator(0, 1)
    far = np.abs(K) < 0.05 * np.abs(K).max()
    assert far.sum() > 10  # there really are distant pairs in this grid
    assert np.abs(G[far] - K[far]).max() < 1e-12 * np.abs(K).max()
