"""Named cell regions must survive LOCAL REFINEMENT, not only a metric remesh.

jNO has two h-adaptive mechanisms and they are different algorithms, not settings on one: remeshing
(mmg, `domain.refine`) and local refinement with hanging nodes (`jno.solve.refine` ->
`refine_domain`), which splits each marked cell into 4 or 8. The split path is the one that "needs
neither a geometry to rebuild from nor a mesher", so it is *the* branch for a mesh loaded from a
file -- exactly the meshes whose materials live in named cell regions.

It rebuilt the domain with only `interior` / `boundary` cell-sets, so those regions were dropped.
The failure is loud rather than silent (a later `by_region` raises "unknown region"), but it makes
the region-coefficient route work for one round of adaptation and not the next.

A child cell has its parent's material by construction, so carrying the sets across is a matter of
knowing which parent each output cell came from.
"""

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
import jno  # noqa: E402
from jno.domain.mesh_utils import mesh_cell_region_membership  # noqa: E402


def _two_region_quad_mesh(tmp_path, n=4):
    pts = np.array([[i / n, j / n, 0.0] for i in range(n + 1) for j in range(n + 1)])
    quads, left = [], []
    for i in range(n):
        for j in range(n):
            quads.append([i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1])
            left.append(i < n // 2)
    quads, left = np.asarray(quads, np.int64), np.asarray(left)
    p = tmp_path / "q.inp"
    meshio.write(
        p,
        meshio.Mesh(
            pts,
            [("quad", quads)],
            cell_sets={
                "left": [np.flatnonzero(left).astype(np.int64)],
                "right": [np.flatnonzero(~left).astype(np.int64)],
            },
        ),
        file_format="abaqus",
    )
    return jno.domain(str(p), compute_mesh_connectivity=False)


def _areas(d):
    q = np.asarray(d.mesh.cells_dict["quad"])
    p = np.asarray(d.mesh.points)[:, :2][q]
    x, y = p[:, :, 0], p[:, :, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))


def _region_area(d, name):
    m = mesh_cell_region_membership(d.mesh, 2).get(name)
    return 0.0 if m is None else float(_areas(d)[m].sum())


def test_the_fixture_has_regions_before_refining(tmp_path):
    d = _two_region_quad_mesh(tmp_path)
    assert {"left", "right"} <= set(mesh_cell_region_membership(d.mesh, 2))
    assert np.isclose(_region_area(d, "left"), 0.5) and np.isclose(_region_area(d, "right"), 0.5)


def test_local_refinement_keeps_the_region_names(tmp_path):
    from jno.utils.solver.fem_refine import refine_domain

    d = _two_region_quad_mesh(tmp_path)
    marked = np.zeros(len(d.mesh.cells_dict["quad"]), bool)
    marked[:3] = True
    refine_domain(d, marked, copy=False)
    assert {"left", "right"} <= set(mesh_cell_region_membership(d.mesh, 2)), (
        f"regions dropped by the split; got {sorted(d.mesh.cell_sets)}"
    )


def test_children_inherit_their_parents_material(tmp_path):
    """The strong invariant: a child covers part of its parent, so each region's AREA is unchanged and
    the two still partition the domain. A mapping that is off by even one cell moves both."""
    from jno.utils.solver.fem_refine import refine_domain

    d = _two_region_quad_mesh(tmp_path)
    marked = np.zeros(len(d.mesh.cells_dict["quad"]), bool)
    marked[[0, 1, 5, 9]] = True
    refine_domain(d, marked, copy=False)

    a_l, a_r = _region_area(d, "left"), _region_area(d, "right")
    assert np.isclose(a_l, 0.5, rtol=1e-9), f"left area {a_l}"
    assert np.isclose(a_r, 0.5, rtol=1e-9), f"right area {a_r}"
    m = mesh_cell_region_membership(d.mesh, 2)
    n_cells = len(d.mesh.cells_dict["quad"])
    assert int(m["left"].sum()) + int(m["right"].sum()) == n_cells, "the regions no longer partition the cells"
    assert not (m["left"] & m["right"]).any(), "a cell is in both regions"


def test_a_mesh_with_no_named_regions_is_unaffected(tmp_path):
    """A mesh carrying no named volume region must behave exactly as before."""
    from jno.utils.solver.fem_refine import refine_domain

    n = 4
    pts = np.array([[i / n, j / n, 0.0] for i in range(n + 1) for j in range(n + 1)])
    quads = np.asarray(
        [
            [i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1]
            for i in range(n)
            for j in range(n)
        ],
        np.int64,
    )
    p = tmp_path / "plain.inp"
    meshio.write(p, meshio.Mesh(pts, [("quad", quads)]), file_format="abaqus")
    d = jno.domain(str(p), compute_mesh_connectivity=False)

    marked = np.zeros(len(d.mesh.cells_dict["quad"]), bool)
    marked[:2] = True
    refine_domain(d, marked, copy=False)
    assert {"interior", "boundary"} <= set(d.mesh.cell_sets)
    assert mesh_cell_region_membership(d.mesh, 2) == {}, "invented a region"
