"""`jno.peec(..., grid=jno.peec.graded(...))` -- grading, through the front door.

The graded mesher existed but only `bar_filaments(edges=...)` could reach it, so the measurement
that justified it had to be run by hand against the internals. That is the gap this closes.

What grading buys, measured on layout1's plate-less loop against a converged Ansys Q3D reference of
54.505 nH: 8,192 graded elements read 57.299 nH where 83,705 UNIFORM ones read 57.340, and 14,111
graded read 56.604 while the uniform sequence had flattened. The elements were never the problem;
their placement was.

The oracles here are the ones that catch it being quietly wrong. `grid=None` must be untouched, to
the last bit -- an opt-in that moved the default would be a silent regression across every existing
model. A graded grid must refine where the faces are and NOT elsewhere, or it is just a finer grid
wearing a new name. And a graded solve must reproduce the uniform answer it approximates.
"""

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

mm, CU = 1e-3, 5.8e7
P = 0.8 * mm


def _net(grid, freq=1e6):
    """Two coplanar traces with a gap: interior faces on y, none on x."""
    a = jno.Shape.box(0, 0, 0, 16 * mm, 2 * mm, P, size=(P,) * 3).attach(sigma=CU).name("t1")
    b = jno.Shape.box(0, 5 * mm, 0, 16 * mm, 7 * mm, P, size=(P,) * 3).attach(sigma=CU).name("t2")
    d = (a + b).domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 15.1 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq, grid=grid).build()


def test_the_default_grid_is_untouched():
    """The governing constraint of an opt-in: with no `grid=`, nothing changes.

    A uniform lattice carries ONE pitch per axis in `d` and leaves `dax` unset -- that absence is
    the uniform representation, and it is what keeps the operator Toeplitz and the apply an FFT.
    """
    e = _net(None)
    assert not e.fil.lattice.get("graded", False)
    assert e.fil.lattice["dax"] is None, "a default grid must not carry per-cell spacings at all"
    assert np.all(np.asarray(e.fil.lattice["d"]) > 0)


def test_grading_refines_toward_the_faces_and_nowhere_else():
    """The whole claim. These traces have interior faces on y (at 2 and 5 mm) and none on x, so y
    must gain cells and x must not -- a grid that refined both would just be a finer grid."""
    u, g = _net(None), _net(jno.peec.graded(fine=0.2 * mm, halo=1.0 * mm))
    nu, ng = tuple(u.fil.lattice["n"]), tuple(g.fil.lattice["n"])
    assert g.fil.lattice["graded"]
    assert ng[0] == nu[0], f"x has no interior face, so it must not be refined: {nu} -> {ng}"
    assert ng[1] > 2 * nu[1], f"y has two interior faces and must be refined: {nu} -> {ng}"
    dy = np.asarray(g.fil.lattice["dax"][1])
    assert dy.min() < 0.3 * mm < dy.max(), (dy.min(), dy.max())


def test_the_fine_cells_sit_AT_the_faces():
    """Refining is only worth anything if the fine cells are where the current crowds. The gap
    between the traces runs 2 mm to 5 mm, so cells near those coordinates must be the small ones."""
    g = _net(jno.peec.graded(fine=0.2 * mm, halo=1.0 * mm))
    dy = np.asarray(g.fil.lattice["dax"][1])
    edges = np.concatenate([[0.0], np.cumsum(dy)]) + float(np.asarray(g.fil.lattice["lo"])[1])
    centres = 0.5 * (edges[:-1] + edges[1:])
    gap = np.abs(centres[:, None] - np.array([2 * mm, 5 * mm])[None, :]).min(1)
    # Judged with one fine cell of slack either way: a band boundary falls between cell centres, so
    # the cell straddling it belongs to neither side and asserting on it would test the rounding.
    assert dy[gap < 1.0 * mm - 0.2 * mm].max() <= 0.2 * mm + 1e-12, "a cell well inside the halo is not fine"
    assert dy[gap > 1.0 * mm + 0.2 * mm].min() > 0.2 * mm, "a cell well outside the halo was refined for nothing"


def test_the_boundaries_never_collapse_to_a_degenerate_cell():
    """Unioning coordinate sets is how a graded grid is built, and `np.unique` will not merge two
    values that differ in the last bit -- which yields a cell 1e-19 wide and a singular operator.
    `bar_filaments` refuses such a cell by name; this asserts it never has to, at spacings chosen so
    the base grid and the refined bands land on nearly-equal coordinates."""
    for fine in (0.4 * mm, 0.2 * mm, P / 3):
        g = _net(jno.peec.graded(fine=fine, halo=1.6 * mm))
        for a, dax in enumerate(g.fil.lattice["dax"]):
            d = np.asarray(dax)
            assert d.min() > 0.01 * fine, (fine, a, d.min())


def test_a_graded_network_refuses_the_FFT_and_names_the_alternative():
    """A graded grid is not translation invariant, so the FFT would return the uniform-grid answer:
    right where the spacing is constant and wrong everywhere it changes."""
    g = _net(jno.peec.graded(fine=0.2 * mm, halo=1.0 * mm))
    with pytest.raises(ValueError, match="GRADED"):
        g.solve()


def _slotted(grid, freq=1e6):
    """ONE connected conductor with a slot: interior faces, but a single piece of metal.

    Deliberately not the two-trace pair above. A terminal spanning two disconnected traces SHORTS
    them, and how many nodes each pad captures depends on the local spacing -- so refining changes
    the circuit, and comparing the two meshes would measure a port change rather than a mesh change.
    """
    bar = jno.Shape.box(0, 0, 0, 16 * mm, 8 * mm, P, size=(P,) * 3)
    sh = (bar - jno.Shape.box(4 * mm, 3 * mm, -P, 12 * mm, 5 * mm, 2 * P)).attach(sigma=CU).name("t")
    d = sh.domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 15.1 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq, grid=grid).build()


def test_a_graded_solve_reproduces_the_uniform_answer_it_approximates():
    """The physics. It is the same conductor meshed differently, so the impedance must agree to
    discretisation error -- refining toward a face may not move the answer somewhere else."""
    u = _slotted(None).solve()
    g = _slotted(jno.peec.graded(fine=0.4 * mm, halo=0.8 * mm)).solve(
        operator=jno.solve.hierarchical(tol=1e-8, leaf=64, floor=0)
    )
    assert abs(complex(g.Z).real / complex(u.Z).real - 1) < 0.15, (u.Z, g.Z)
    assert abs(complex(g.Z).imag / complex(u.Z).imag - 1) < 0.15, (u.Z, g.Z)


@pytest.mark.parametrize(
    "kw, msg",
    [
        ({"fine": 0.0, "halo": 1e-3}, "positive lengths"),
        ({"fine": -1e-4, "halo": 1e-3}, "positive lengths"),
        ({"fine": 1e-3, "halo": 0.0}, "positive lengths"),
        ({"fine": 1e-3, "halo": 1e-4}, "smaller than fine"),
    ],
)
def test_a_grid_that_could_not_resolve_anything_is_refused(kw, msg):
    """A halo narrower than one refined cell resolves nothing, and a non-positive length is not a
    length. Both are stated at the point of the mistake rather than surfacing as a mesh nobody
    asked for."""
    with pytest.raises(ValueError, match=msg):
        jno.peec.graded(**kw)
