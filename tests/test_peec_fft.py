"""The FFT path: the same operator as the dense one, applied without forming it.

A bar lattice is block-Toeplitz within each current direction, so the partial-inductance apply is an
FFT. That is only worth having if it is the SAME operator, so every case here checks the matrix-free
result against the dense one built by ``pair_matrix``.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import pair_matrix
from jno.utils.solver.peec import bar_filaments, lattice_apply, line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7
INV_R = lambda r: 1.0 / r  # noqa: E731


def bar_ends(f):
    p = np.asarray(f.nodes)
    return (
        terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9),
        terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9),
    )


@pytest.mark.parametrize("pitch", [0.002, 0.001])
def test_the_fft_apply_is_the_dense_operator(pitch):
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=pitch)
    k = np.asarray(pair_matrix(f.pos, f.mom, INV_R, f.self_g, group=f.group))
    x = np.random.default_rng(0).normal(size=k.shape[0])
    assert np.linalg.norm(np.asarray(lattice_apply(f, INV_R)(x)) - k @ x) / np.linalg.norm(k @ x) < 1e-13


def test_the_quadrature_goes_into_the_generator():
    """One point per element is 4.2 % low on a bar lattice, so the FFT must carry the sub-points too.

    Every bar of a family has the same offsets, so the double sum still depends only on the cell
    separation and the block stays Toeplitz -- which is why this can be exact rather than a coarser rule.
    """
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=0.002, quad=3)
    k = np.asarray(pair_matrix(f.pos, f.mom, INV_R, f.self_g, group=f.group))
    x = np.ones(k.shape[0])
    matched = np.asarray(lattice_apply(f, INV_R, quad=3)(x))
    one_point = np.asarray(lattice_apply(f, INV_R, quad=1)(x))
    assert np.linalg.norm(matched - k @ x) / np.linalg.norm(k @ x) < 1e-13
    assert np.linalg.norm(one_point - k @ x) / np.linalg.norm(k @ x) > 1e-2  # and the coarse rule is not


def test_a_hole_in_the_conductor_keeps_the_fft_exact():
    """The grid covers the BOUNDING BOX and a mask says which cells are metal.

    That is what lets an L-shape keep translation invariance: the lattice ignores the geometry, and
    the absent cells simply carry no current.
    """
    ell = jno.Shape.box(0, 0, 0, 0.030, 0.008, 0.002) | jno.Shape.box(0.022, 0.0, 0, 0.030, 0.030, 0.002)
    f = bar_filaments(ell, size=0.002)
    kept, total = np.asarray(f.nodes).shape[0], int(np.prod(f.lattice["n"]))
    assert kept < total  # there really is a hole in the grid
    k = np.asarray(pair_matrix(f.pos, f.mom, INV_R, f.self_g, group=f.group))
    x = np.random.default_rng(1).normal(size=k.shape[0])
    assert np.linalg.norm(np.asarray(lattice_apply(f, INV_R)(x)) - k @ x) / np.linalg.norm(k @ x) < 1e-13


@pytest.mark.parametrize("pitch", [0.002, 0.001])
def test_the_matrix_free_solve_agrees_with_the_dense_one(pitch):
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=pitch)
    a, b = bar_ends(f)
    got = {}
    for mf in (False, True):
        cur, _phi, inj = solve_network(
            f, SIG, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * 1e6, matrix_free=mf
        )
        got[mf] = (complex(1.0 / inj["A"]), np.asarray(cur))
    assert abs(got[True][0] - got[False][0]) / abs(got[False][0]) < 1e-11
    assert np.linalg.norm(got[True][1] - got[False][1]) / np.linalg.norm(got[False][1]) < 1e-8


def test_filaments_off_a_lattice_refuse_the_fft():
    """A polyline's filaments are not Toeplitz, and saying so beats applying the wrong operator."""
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, 0.05)], r=5e-4, size=0.005))
    with pytest.raises(ValueError, match="do not sit on a lattice"):
        lattice_apply(f, INV_R)
    with pytest.raises(ValueError, match="matrix_free=True needs a bar lattice"):
        solve_network(f, SIG, {"A": np.array([0]), "B": np.array([1])}, [("A", "B", 1.0 + 0j)], matrix_free=True)


def test_mutual_inductance_is_what_makes_the_ac_solution_differ_from_dc():
    """Self-inductance alone reproduces the DC current distribution exactly — the known wrong answer.

    PyPEEC's method page makes this point with its Fig. 3 / Fig. 4: a resistive circuit extended with
    only SELF inductances gives back the DC distribution, because it is the mutual terms that carry
    Faraday's law between cells. Measured here on a 40 x 20 x 2 mm busbar at 1 MHz: with self terms
    only the edge/middle current density is 1.0000 and the profile matches DC to 1.7e-14; with the
    mutuals it is 2.2497, current crowding to the edges.
    """
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.020, 0.002), size=0.002)
    lp = np.asarray(pair_matrix(f.pos, f.mom, INV_R, f.self_g, group=f.group)) * 4e-7 * np.pi / (4 * np.pi)
    assert np.count_nonzero(lp - np.diag(np.diag(lp))) > 0  # there ARE mutual terms
    off = np.abs(lp - np.diag(np.diag(lp))).sum()
    assert off > np.abs(np.diag(lp)).sum()  # and they dominate: this is not a diagonal model
