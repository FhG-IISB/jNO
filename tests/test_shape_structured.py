"""``Shape.structured()`` — a regular lattice as a property of the shape.

A regular box had four spellings, each with a different subset of the features: the cell choice on
one, the ``jno.fdm`` grid descriptor on another, the region names on a third, and hexahedra only on
the one that was not part of the ``Shape`` DSL at all — ``Shape.quad()``'s own 3-D refusal pointed
the reader at ``jno.domain.equi_distant_box(cell='hex')``, a different front door.

``.structured()`` is the single one, and it sits beside ``.quad()`` / ``.curved()`` / ``.sized()``
because structuredness is decided *before* meshing. The oracle throughout is that the new spelling
produces the **same mesh** as the lattice builder it replaces, node for node and cell for cell — not
merely a mesh that works.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.domain.geometries import Geometries
from jno.geometry import Shape

gmsh = pytest.importorskip("gmsh", reason="the unstructured control needs the Shape mesher")


def _dom(shape, **kw):
    return shape.domain(compute_mesh_connectivity=False, **kw)


def _blocks(d):
    return {c.type: len(c.data) for c in d.mesh.cells}


def _same_mesh(a, b):
    """Identical points, identical blocks, identical cell sets — the migration oracle."""
    if not np.array_equal(np.asarray(a.mesh.points), np.asarray(b.mesh.points)):
        return False
    if [c.type for c in a.mesh.cells] != [c.type for c in b.mesh.cells]:
        return False
    if not all(np.array_equal(x.data, y.data) for x, y in zip(a.mesh.cells, b.mesh.cells)):
        return False
    ca, cb = a.mesh.cell_sets, b.mesh.cell_sets
    if set(ca) != set(cb):
        return False
    return all(all(np.array_equal(u, v) for u, v in zip(ca[k], cb[k])) for k in ca)


# ------------------------------------------------------- the same mesh as the spelling it replaces


@pytest.mark.parametrize("cell", [None, "quad"])
def test_a_structured_rect_is_the_lattice_builders_own_mesh(cell):
    shape = Shape.rect(0, 0, 1, 1).structured(n=4)
    new = _dom(shape.quad() if cell else shape)
    old = jno.domain(
        constructor=Geometries.equi_distant_rect(nx=4, ny=4, cell=cell or "triangle"),
        compute_mesh_connectivity=False,
    )
    assert _same_mesh(new, old)


@pytest.mark.parametrize("cell", [None, "quad"])
def test_a_structured_box_is_the_lattice_builders_own_mesh(cell):
    shape = Shape.box(0, 0, 0, 1, 1, 1).structured(n=3)
    new = _dom(shape.quad() if cell else shape)
    old = jno.domain(
        constructor=Geometries.equi_distant_box(nx=3, ny=3, nz=3, cell="hex" if cell else "tetra"),
        compute_mesh_connectivity=False,
    )
    assert _same_mesh(new, old)


def test_the_foundation_model_grid_is_reproducible():
    """`jno.domain.poseidon(nx, ny)` is an nx x ny NODE grid, i.e. nx-1 cells — the one place the
    cells-vs-nodes convention is felt, since these models speak in pixel resolution."""
    d = _dom(Shape.rect(0, 0, 1, 1).structured(n=15))
    assert d.grid["shape"] == (16, 16)
    assert len(d.mesh.points) == 16 * 16


# ------------------------------------------------------------------------- hexahedra via the DSL


def test_structured_unlocks_quad_in_3d():
    d = _dom(Shape.box(0, 0, 0, 1, 1, 1).structured(n=3).quad())
    assert _blocks(d)["hexahedron"] == 27


def test_quad_and_structured_compose_in_either_order():
    """A chain should not care about order — which is why the 3-D refusal lives at build time rather
    than inside ``.quad()``, where it could only see half the plan."""
    a = _dom(Shape.box(0, 0, 0, 1, 1, 1).structured(n=3).quad())
    b = _dom(Shape.box(0, 0, 0, 1, 1, 1).quad().structured(n=3))
    assert _same_mesh(a, b)


def test_an_unstructured_3d_quad_is_refused_and_names_the_fix():
    """The refusal fires when the mesh is BUILT, which a deferred domain postpones to first use.

    ``.domain()`` no longer meshes on the spot, so the refusal surfaces on the first access to
    ``.mesh`` rather than out of the constructor. Asserted through ``_blocks`` -- the same accessor
    every other test in this file goes through -- so the timing this pins is the one a caller
    actually meets.
    """
    with pytest.raises(NotImplementedError, match=r"\.structured\(\)"):
        _blocks(_dom(Shape.box(0, 0, 0, 1, 1, 1, size=0.5).quad()))


def test_a_structured_hex_domain_solves():
    """The point of a mesh is that a weak form runs on it."""
    d = Shape.box(0, 0, 0, 1, 1, 1).structured(n=3).quad().domain()
    u, v = d.fem_symbols()
    ci, cb = d.variable("interior", split=True), d.variable("boundary", split=True)
    ui, vi = u.bind(x=ci[0], y=ci[1], z=ci[2]), v.bind(x=ci[0], y=ci[1], z=ci[2])
    lap = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z
    sol = np.asarray(jno.fem([lap, u(*cb[:3]) - cb[2]]).solve()).ravel()
    np.testing.assert_allclose(sol, np.asarray(d.mesh.points)[:, 2], atol=2e-4)  # u = z, to float32


# ----------------------------------------------------------------------------------- resolution


def test_the_cell_counts_come_from_size_when_n_is_absent():
    d = _dom(Shape.rect(0, 0, 1, 1, size=0.25).structured())
    assert d.grid["shape"] == (5, 5)
    np.testing.assert_allclose(d.grid["spacing"], (0.25, 0.25))


@pytest.mark.parametrize("n,shape", [(4, (5, 5, 5)), ((4, 2, 3), (5, 3, 4))])
def test_n_counts_cells_scalar_or_per_axis(n, shape):
    d = _dom(Shape.box(0, 0, 0, 1, 1, 1).structured(n=n))
    assert d.grid["shape"] == shape


def test_the_grid_descriptor_describes_a_non_unit_box():
    d = _dom(Shape.box(-1, -2, 0, 3, 1, 5).structured(n=(4, 2, 3)))
    assert d.grid["origin"] == (-1.0, -2.0, 0.0)
    np.testing.assert_allclose(d.grid["spacing"], (1.0, 1.5, 5.0 / 3.0))


def test_a_nodal_field_reshapes_to_the_grid():
    """What the descriptor is for: C-ordered nodes, so `u.reshape(grid["shape"])` is the image."""
    d = _dom(Shape.rect(0, 0, 1, 1).structured(n=5))
    x = np.asarray(d.mesh.points)[:, 0].reshape(d.grid["shape"])
    assert np.allclose(x, x[:, :1])  # x varies along axis 0 only


def test_an_unstructured_domain_has_no_grid():
    assert _dom(Shape.rect(0, 0, 1, 1, size=0.4)).grid is None


# -------------------------------------------------------------------- refusals, each naming a reason


@pytest.mark.parametrize(
    "plan,match",
    [
        pytest.param(lambda: Shape.rect(0, 0, 1, 1) - Shape.disk(0.5, 0.5, 0.2), "'cut' plan", id="csg"),
        pytest.param(lambda: Shape.disk(0, 0, 1), "this is a disk", id="disk"),
        pytest.param(lambda: Shape.rect(0, 0, 1, 1, size=lambda x, y: 0.1), "graded mesh", id="graded-size"),
    ],
)
def test_an_unstructurable_plan_is_refused_by_name(plan, match):
    with pytest.raises(NotImplementedError, match=match):
        _dom(plan().structured())


def test_structured_and_curved_is_refused():
    with pytest.raises(NotImplementedError, match="9-/27-node"):
        _dom(Shape.rect(0, 0, 1, 1).structured(n=4).curved())


def test_a_graded_size_is_accepted_when_the_counts_are_explicit():
    """The refusal is about deriving counts, not about the callable itself — so saying the counts
    outright is enough. Worth pinning: the two are easy to conflate into one blanket refusal."""
    d = _dom(Shape.rect(0, 0, 1, 1, size=lambda x, y: 0.1).structured(n=4))
    assert d.grid["shape"] == (5, 5)


@pytest.mark.parametrize("n", [(4, 4, 4), (4,)])
def test_the_wrong_number_of_axes_is_refused(n):
    with pytest.raises(ValueError, match="cell counts"):
        Shape.rect(0, 0, 1, 1).structured(n=n)


def test_a_zero_cell_axis_is_refused():
    with pytest.raises(ValueError, match="at least one cell"):
        Shape.rect(0, 0, 1, 1).structured(n=(4, 0))


# ------------------------------------------------------------------------------ the rest of the DSL


def test_a_named_region_survives_the_lattice_path():
    """The lattice constructor replaces the shape, and the region name and attachments are read off
    the shape — so they used to be dropped silently, `d.k` reporting "no attribute" on a plan that
    plainly attached one."""
    d = _dom(Shape.rect(0, 0, 1, 1).structured(n=3).name("steel").attach(k=2.0))
    assert "steel" in d.avaiable_mesh_tags
    assert d.k is not None
    assert len(np.atleast_1d(d.tag_indices["steel"])) == 16


def test_the_named_faces_come_with_the_lattice():
    """A structured domain gets its faces named without a gmsh model to classify."""
    d = _dom(Shape.box(0, 0, 0, 1, 1, 1).structured(n=2))
    assert {"left", "right", "bottom", "top", "front", "back", "boundary", "interior"} <= set(d.avaiable_mesh_tags)


@pytest.mark.parametrize("n", [2, (2, 2)])
def test_the_smallest_lattice_is_two_cells_per_axis(n):
    """Two is the floor, so the 3-point edge stencil is defined; a coarser request is raised to it."""
    assert _dom(Shape.rect(0, 0, 1, 1).structured(n=n)).grid["shape"] == (3, 3)


def test_one_cell_is_raised_to_the_floor_not_refused():
    assert _dom(Shape.rect(0, 0, 1, 1).structured(n=1)).grid["shape"] == (3, 3)


def test_a_strongly_anisotropic_slab_builds_and_solves():
    """The extreme end: 64 cells one way, 2 the others."""
    d = Shape.box(0, 0, 0, 1, 0.05, 0.05).structured(n=(64, 2, 2)).quad().domain()
    assert _blocks(d)["hexahedron"] == 64 * 2 * 2
    u, v = d.fem_symbols()
    ci, cb = d.variable("interior", split=True), d.variable("boundary", split=True)
    ui, vi = u.bind(x=ci[0], y=ci[1], z=ci[2]), v.bind(x=ci[0], y=ci[1], z=ci[2])
    sol = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(*cb[:3]) - cb[0]]).solve()).ravel()
    np.testing.assert_allclose(sol, np.asarray(d.mesh.points)[:, 0], atol=1e-4)
