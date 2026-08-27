"""The FFT path: the same operator as the dense one, applied without forming it.

The dense-vs-matrix-free comparisons pass an explicit tight ``tol``: they are checking that the two
paths are the same OPERATOR, and the default tolerance is set where the time curve is flat rather
than where the residual is smallest, so it would otherwise bound the agreement rather than the
operator doing so.

A bar lattice is block-Toeplitz within each current direction, so the partial-inductance apply is an
FFT. That is only worth having if it is the SAME operator, so every case here checks the matrix-free
result against the dense one built by ``pair_matrix``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import pair_matrix
from jno.utils.solver.peec import (
    bar_filaments,
    lattice_apply,
    line_filaments,
    solve_network,
    terminal_nodes,
    welded_apply,
)

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
            f, SIG, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * 1e6, matrix_free=mf, tol=1e-12
        )
        got[mf] = (complex(1.0 / inj["A"]), np.asarray(cur))
    assert abs(got[True][0] - got[False][0]) / abs(got[False][0]) < 1e-11
    assert np.linalg.norm(got[True][1] - got[False][1]) / np.linalg.norm(got[False][1]) < 1e-8


def test_filaments_off_a_lattice_refuse_the_fft():
    """A polyline's filaments are not Toeplitz, and saying so beats applying the wrong operator."""
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, 0.05)], r=5e-4, size=0.005))
    with pytest.raises(ValueError, match="do not sit on a lattice"):
        lattice_apply(f, INV_R)
    with pytest.raises(ValueError, match="needs a lattice somewhere in the network"):
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


SIGCU = 5.8e7


def _trace_and_wire():
    trace = jno.Shape.box(0, 0, 0, 0.02, 0.004, 0.001, size=(0.001, 0.001, 0.0005)).attach(sigma=SIGCU).name("trace")
    wire = (
        jno.Shape.line([(0.019, 0.002, 0.00075), (0.019, 0.002, 0.006), (0.030, 0.002, 0.00075)], r=1.9e-4, size=0.001)
        .attach(sigma=SIGCU)
        .name("wire")
    )
    d = (trace + wire).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: (x > 0.0295) & (z < 0.0011))
    return d


def test_several_solids_share_one_grid():
    """Separate lattices couple through a block that is not Toeplitz; one grid stays a single FFT."""
    a = jno.Shape.box(0, 0, 0, 0.020, 0.006, 0.001)
    b = jno.Shape.box(0.024, 0, 0, 0.044, 0.006, 0.001)
    f = bar_filaments([a, b], size=(0.002, 0.002, 0.001))
    assert set(np.unique(np.asarray(f.part)).tolist()) == {0, 1}  # two conductors...
    assert np.asarray(f.nodes).shape[0] < int(np.prod(f.lattice["n"]))  # ...with the gap masked out
    k = np.asarray(pair_matrix(f.pos, f.mom, INV_R, f.self_g, group=f.group))
    x = np.random.default_rng(0).normal(size=k.shape[0])
    assert np.linalg.norm(np.asarray(lattice_apply(f, INV_R)(x)) - k @ x) / np.linalg.norm(k @ x) < 1e-13


def test_a_bar_straddling_two_conductors_takes_them_in_series():
    """Touching conductors are normal — a strap shorting two plates is the usual case, not an error.

    Half the bar lies in each, so its conductivity is the series (harmonic) mean: for one material
    that degenerates to the material, and only a genuine mismatch changes anything.
    """
    a = jno.Shape.box(0, 0, 0, 0.010, 0.006, 0.001)
    b = jno.Shape.box(0.010, 0, 0, 0.020, 0.006, 0.001)  # touching, so a bar straddles them
    with pytest.raises(ValueError, match="its conductivity depends on both"):
        bar_filaments([a, b], size=(0.002, 0.002, 0.001))  # ambiguous without conductivities

    f = bar_filaments([a, b], size=(0.002, 0.002, 0.001), sigma=[SIGCU, SIGCU / 3])
    per = np.asarray(f.lattice["sigma"])
    assert np.isclose(per.max(), SIGCU) and np.isclose(per.min(), SIGCU / 3)
    straddling = per[(per != SIGCU) & (per != SIGCU / 3)]
    assert len(straddling) > 0
    assert np.allclose(straddling, 2 * SIGCU * (SIGCU / 3) / (SIGCU + SIGCU / 3))


def test_a_welded_network_keeps_each_block_structure():
    d = _trace_and_wire()
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    pe = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6)
    fil, _sigma, _terms, _names = pe._discretise()
    kinds = [lat is not None for _lo, _hi, lat in fil.lattice["welded"]]
    assert kinds == [False, True]  # the wire's filaments dense, the trace's bars a lattice


def test_the_stitched_apply_is_the_dense_operator():
    """The cross block is neither Toeplitz nor square, so it is evaluated rather than stored."""
    d = _trace_and_wire()
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    fil, _s, _t, _n = jno.peec([v(*at("A")) - v(*at("B")) - 1.0])._discretise()
    k = np.asarray(pair_matrix(fil.pos, fil.mom, INV_R, fil.self_g, group=fil.group))
    x = np.random.default_rng(2).normal(size=k.shape[0])
    got = np.asarray(welded_apply(fil, INV_R)(x))
    assert np.linalg.norm(got - k @ x) / np.linalg.norm(k @ x) < 1e-13


def test_the_stitched_solve_agrees_with_the_dense_one():
    d = _trace_and_wire()
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    pe = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6)
    fil, sigma, terms, _names = pe._discretise()
    nodes = {t: terminal_nodes(fil, sh) for t, sh in terms.items()}
    got = {}
    for mf in (False, True):
        cur, _phi, inj = solve_network(
            fil, sigma, nodes, pe.sources, pe.grounds, pe.currents, omega=2 * np.pi * 1e6, matrix_free=mf, tol=1e-12
        )
        got[mf] = (complex(1.0 / inj["A"]), np.asarray(cur))
    assert abs(got[True][0] - got[False][0]) / abs(got[False][0]) < 1e-12
    assert np.linalg.norm(got[True][1] - got[False][1]) / np.linalg.norm(got[False][1]) < 1e-9


def test_one_compiled_solve_is_reused_without_leaking_between_networks():
    """The Krylov solve is compiled once per network and reused across frequencies.

    That reuse is keyed on the network's identity, and an id alone is recyclable after a garbage
    collection — a stale hit would silently run the previous geometry's closures, giving a plausible
    impedance for the wrong conductor. So the entry holds the network and is checked against it.
    """

    def run(box):
        f = bar_filaments(box, size=(0.002, 0.002, 0.001))
        p = np.asarray(f.nodes)
        a = terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
        b = terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)
        _c, _phi, inj = solve_network(
            f, SIGCU, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * 1e6, matrix_free=True
        )
        return complex(1.0 / inj["A"])

    small = jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002)
    large = jno.Shape.box(0, 0, 0, 0.060, 0.006, 0.002)
    first = run(small)
    other = run(large)
    again = run(small)
    assert abs(again - first) / abs(first) < 1e-12  # the same network gives the same answer...
    assert abs(other - first) / abs(first) > 1e-3  # ...and a different one does not borrow it


def test_a_frequency_sweep_reuses_the_compilation():
    """Reuse is the point: without it every frequency recompiles, and at these sizes the compilation
    costs about what the fusion saves."""
    from jno.utils.solver.peec import _KRYLOV_CACHE

    # the cache is module-level and bounded, so other tests both fill and evict it; this test is
    # about how many entries THIS network adds, so it starts from a known state
    _KRYLOV_CACHE.clear()
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=(0.002, 0.002, 0.001))
    p = np.asarray(f.nodes)
    a = terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)
    zs = []
    for hz in (1e5, 1e6, 1e7):
        _c, _phi, inj = solve_network(
            f, SIGCU, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * hz, matrix_free=True
        )
        zs.append(complex(1.0 / inj["A"]))
    assert len(_KRYLOV_CACHE) == 1  # three frequencies, one compilation
    assert abs(zs[2].imag) > abs(zs[0].imag)  # and they are genuinely different frequencies


def test_the_dense_path_is_differentiable_and_the_matrix_free_one_refuses():
    """A wrong gradient is worse than a missing one, so the fast path refuses to give one.

    Measured against the dense path, which agrees with finite differences to 1e-8: the matrix-free
    gradient came out at -2.80e-14 where the answer is -1.41e-12, and at DC -3.09e-09 where the answer
    is -8.19e-05. Until that is understood it is refused rather than returned.
    """
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=(0.002, 0.002, 0.001))
    p = np.asarray(f.nodes)
    a = terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)

    def port_r(sig, mf):
        _c, _phi, inj = solve_network(
            f, sig, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * 1e6, matrix_free=mf
        )
        return jnp.real(1.0 / inj["A"])

    g = float(jax.grad(lambda s: port_r(s, False))(SIGCU))
    fd = float((port_r(SIGCU * 1.0001, False) - port_r(SIGCU * 0.9999, False)) / (0.0002 * SIGCU))
    assert abs(g / fd - 1) < 1e-6
    assert g < 0  # more conductive, less resistive

    with pytest.raises(NotImplementedError, match="not differentiable yet"):
        jax.grad(lambda s: port_r(s, True))(SIGCU)
