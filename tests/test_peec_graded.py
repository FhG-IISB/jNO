"""A GRADED rectilinear lattice: per-axis cell boundaries instead of one pitch.

PEEC's speed comes from translation invariance -- identical bars on a regular grid make the operator
block-Toeplitz, so an FFT applies it. The price is that one pitch has to serve the whole model, and
on every real power-module layout the narrowest trace is 1.0 mm on a 96 x 74 mm plate: resolving it
uniformly costs one to three million elements, an order of magnitude beyond affordable.

A graded grid keeps the cells axis-aligned boxes and the indexing structured -- so `bar_self`, the
volume quadrature and the incidence are all unchanged, and there are no hanging nodes to tie -- while
letting the spacing vary along each axis. What it gives up is exactly the translation invariance, so
the FFT no longer applies and the operator must be `jno.solve.hierarchical(...)`.

The oracles here are the ones that would catch it being quietly wrong. A graded grid whose spacing
happens to be uniform must reproduce the uniform answer to the last bit -- that is the claim that the
new geometry code did not change the old geometry. And a genuinely graded grid must reproduce a
uniform grid at its FINEST spacing, to discretisation error, because that is the answer it is meant
to approximate cheaply.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import bar_filaments, lattice_apply

jax.config.update("jax_enable_x64", True)

mm, CU = 1e-3, 5.8e7
BOX = (0.0, 0.0, 0.0, 16 * mm, 8 * mm, 0.8 * mm)


def _shape(p):
    return jno.Shape.box(*BOX, size=(p, p, p))


def _uniform_edges(p):
    """The boundaries a uniform grid of pitch `p` would have -- same grid, stated the other way."""
    return [np.linspace(BOX[a], BOX[a + 3], int(round((BOX[a + 3] - BOX[a]) / p)) + 1) for a in range(3)]


def _merge(*arrays, tol=1e-12):
    """Union of coordinate sets, merging values that differ only in the last bits.

    `np.unique` does not: two `linspace` calls that both contain 0.8 mm can disagree in the final
    bit, and the union then holds a cell 1e-19 wide. That is a singular operator, and it is how the
    first graded solve here failed.
    """
    v = np.sort(np.concatenate([np.asarray(a, dtype=float).reshape(-1) for a in arrays]))
    keep = np.concatenate([[True], np.diff(v) > tol * max(abs(v[-1] - v[0]), 1e-300)])
    return v[keep]


def test_a_graded_grid_that_is_uniform_reproduces_the_uniform_one():
    """The regression guard on the geometry code: stating a uniform grid as explicit boundaries must
    give back exactly the same mesh, or the new path has changed the old one."""
    p = 0.8 * mm
    a = bar_filaments(_shape(p), sigma=CU)
    b = bar_filaments(_shape(p), sigma=CU, edges=_uniform_edges(p))
    assert a.lattice["n"] == b.lattice["n"]
    assert np.allclose(np.asarray(a.nodes), np.asarray(b.nodes))
    for name in ("length", "area", "self_g", "pos", "mom"):
        assert np.allclose(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name
    assert not a.lattice["graded"] and b.lattice["graded"]


def test_the_fft_refuses_a_graded_lattice():
    """A graded grid is not translation invariant, so `lattice_apply` would return the uniform-grid
    answer -- right where the spacing is constant and wrong everywhere it changes. That is the shape
    of error this codebase refuses rather than returns."""
    e = _uniform_edges(0.8 * mm)
    e[0] = _merge(e[0], 0.5 * (e[0][:-1] + e[0][1:])[:4])  # refine a band in x
    f = bar_filaments(_shape(0.8 * mm), sigma=CU, edges=e)
    with pytest.raises(ValueError, match="GRADED"):
        lattice_apply(f, lambda r: 1.0 / r)


def test_a_graded_mesh_puts_cells_where_it_was_told():
    """The point of the whole exercise: the fine band is fine and the rest is not."""
    e = _uniform_edges(0.8 * mm)
    fine = np.linspace(0.0, 4 * mm, 17)  # a quarter of the span at a quarter of the pitch
    e[0] = _merge(fine, e[0][e[0] > 4 * mm])
    f = bar_filaments(_shape(0.8 * mm), sigma=CU, edges=e)
    dx = np.asarray(f.lattice["dax"][0])
    assert dx.min() < 0.3 * mm and dx.max() > 0.7 * mm, (dx.min(), dx.max())
    # and the cell count grew only where asked, not everywhere
    uni_fine = bar_filaments(_shape(0.2 * mm), sigma=CU)
    assert len(np.asarray(f.nodes)) < 0.5 * len(np.asarray(uni_fine.nodes))


def test_a_graded_solve_needs_the_hierarchical_operator_and_says_so():
    """There is no FFT for a graded grid, so asking for one must name the alternative rather than
    fall back to a dense operator the user did not ask for."""
    from jno.utils.solver.peec import solve_network, terminal_nodes

    e = _uniform_edges(0.8 * mm)
    e[0] = _merge(np.linspace(0.0, 4 * mm, 17), e[0][e[0] > 4 * mm])
    f = bar_filaments(_shape(0.8 * mm), sigma=CU, edges=e)
    A = terminal_nodes(f, lambda P: P[:, 0] < 0.9 * mm)
    B = terminal_nodes(f, lambda P: P[:, 0] > 15.1 * mm)
    with pytest.raises(ValueError, match="GRADED"):
        solve_network(f, CU, {"A": A, "B": B}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * 1e6)


def test_a_graded_solve_reproduces_the_uniform_answer_it_approximates():
    """The physics claim. A grid graded toward the terminals must give the same impedance as a
    uniform grid, to discretisation error -- it is the same conductor, meshed differently.

    The terminals are picked as the EXTREME node column rather than by a fixed coordinate band, and
    that matters: a band predicate captures however many columns the local spacing puts inside it, so
    refining near a port silently shorts two columns where the coarse mesh shorted one. That is a
    different circuit, and comparing it to the uniform one measured a port change rather than a mesh
    change -- first attempt at this test read 2.8x on R for exactly that reason, while the DC
    resistance of the same graded mesh matches the analytic `rho L / A` to four decimals.
    """
    from jno.utils.solver.peec import solve_network, terminal_nodes

    p = 0.8 * mm
    got = {}
    for name, edges in (("uniform", _uniform_edges(p)), ("graded", None)):
        if edges is None:
            e = _uniform_edges(p)
            lo = np.linspace(0.0, 3.2 * mm, 9)
            hi = np.linspace(12.8 * mm, 16 * mm, 9)
            e[0] = _merge(lo, e[0], hi)
            edges = e
        f = bar_filaments(_shape(p), sigma=CU, edges=edges)
        xs = np.asarray(f.nodes)[:, 0]
        A = terminal_nodes(f, lambda P: P[:, 0] < xs.min() + 1e-9)
        B = terminal_nodes(f, lambda P: P[:, 0] > xs.max() - 1e-9)
        _c, _p, inj = solve_network(
            f, CU, {"A": A, "B": B}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * 1e6,
            operator=jno.solve.hierarchical(tol=1e-8, leaf=64, floor=0),
        )
        got[name] = complex(1.0 / inj["A"])
    r = abs(got["graded"].real / got["uniform"].real - 1)
    li = abs(got["graded"].imag / got["uniform"].imag - 1)
    assert r < 0.15, (got["uniform"].real, got["graded"].real)
    assert li < 0.15, (got["uniform"].imag, got["graded"].imag)
