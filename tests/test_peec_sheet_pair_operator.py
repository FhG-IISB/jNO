"""The operator carries a coupled element impedance, not just a diagonal one.

``Z = diag(R) + j w Lp`` gives every element one current. A slab thick against the skin depth needs
two -- one sheet per face -- coupled by the off-diagonal of the 2-port slab impedance. This is the
operator half of that: the pairing is made by hand here, because nothing emits one yet.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import _element_impedance, bar_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG, MU0 = 5.8e7, 4e-7 * np.pi
OMEGA = 2 * np.pi * 1e6


def _bar(ny=4):
    """A 40 x 4 x 2 mm bar, ONE cell through its thickness and ``ny`` across its width.

    span == 1 throughout, so `area / skin` really is the element's in-plane width and the thickness
    guard has nothing to say -- this file is about the operator, not about which conductors deserve
    a pair.
    """
    return bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=(0.002, 0.004 / ny, 0.002))


def _centres(fil):
    n = np.asarray(fil.length).size
    grp, pos = np.asarray(fil.group), np.asarray(fil.pos)
    return np.stack([pos[grp == k].mean(axis=0) for k in range(n)])


def _synthetic_pairs(fil):
    """Pair elements two by two.

    SYNTHETIC: it pairs neighbours across the width rather than the two faces of a slab, because
    nothing emits a real pairing until the discretisation does. It is a valid pairing -- symmetric,
    each element in at most one pair -- which is all the operator cares about.
    """
    cen = _centres(fil)
    n = len(cen)
    idx = -np.ones(n, dtype=int)
    rows = {}
    for k in range(n):
        rows.setdefault((round(cen[k, 0], 9), round(cen[k, 2], 9)), []).append(k)
    for ks in rows.values():
        ks = sorted(ks, key=lambda i: cen[i, 1])
        for a, b in zip(ks[::2], ks[1::2]):
            idx[a], idx[b] = b, a
    return idx


def test_an_unpaired_network_is_untouched():
    """The common case must pay nothing: no pairing, no coupling, and the same R as before."""
    fil = _bar()
    sig = jnp.full(np.asarray(fil.length).size, SIG)
    R, pidx, cz = _element_impedance(fil, OMEGA, sig, MU0)
    assert pidx is None and cz is None
    assert np.all(np.isfinite(np.asarray(R)))


def test_an_all_minus_one_pairing_is_also_untouched():
    """-1 means 'this element has no other face', so a network of them is the unpaired case."""
    fil = _bar()
    n = np.asarray(fil.length).size
    sig = jnp.full(n, SIG)
    base, _, _ = _element_impedance(fil, OMEGA, sig, MU0)
    R, pidx, cz = _element_impedance(fil._replace(pair=-np.ones(n, int)), OMEGA, sig, MU0)
    assert pidx is None and cz is None
    assert np.allclose(np.asarray(R), np.asarray(base))


def test_a_paired_element_takes_the_two_port_diagonal():
    """Its own impedance becomes z_self, and its partner supplies z_mutual."""
    fil = _bar()
    idx = _synthetic_pairs(fil)
    assert (idx >= 0).any()
    n = np.asarray(fil.length).size
    R, pidx, cz = _element_impedance(fil._replace(pair=idx), OMEGA, jnp.full(n, SIG), MU0)
    live = idx >= 0
    assert pidx is not None
    assert np.all(np.abs(np.asarray(cz))[live] > 0)  # coupled where paired
    assert np.allclose(np.asarray(cz)[~live], 0)  # and nowhere else
    assert np.all(np.asarray(pidx)[~live] == np.arange(n)[~live])  # unpaired point at themselves


def _port(fil, pair, matrix_free):
    p = np.asarray(fil.nodes)
    a = terminal_nodes(fil, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(fil, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)
    f = fil if pair is None else fil._replace(pair=pair)
    _c, _phi, inj = solve_network(
        f, SIG, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=OMEGA, matrix_free=matrix_free
    )
    return complex(1.0 / inj["A"])


def test_the_dense_and_matrix_free_operators_carry_the_same_coupling():
    """The FFT path and the assembled one must agree WITH a pairing, not only without it."""
    fil = _bar()
    idx = _synthetic_pairs(fil)
    assert abs(_port(fil, idx, True) / _port(fil, idx, False) - 1) < 1e-6


def test_the_coupling_is_actually_applied():
    """A guard against the term being threaded but silently dropped: it must change the answer."""
    fil = _bar()
    idx = _synthetic_pairs(fil)
    assert abs(_port(fil, idx, False) / _port(fil, None, False) - 1) > 1e-3
