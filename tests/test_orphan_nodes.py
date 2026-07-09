"""jno.domain drops mesh nodes that support no finite element (gmsh geometry-construction points
emitted as isolated 0-D ``vertex`` cells) -- they are zero rows in the assembled operator. Off by
default; ``keep_orphan_nodes=True`` opts out."""

import numpy as np
import pytest

meshio = pytest.importorskip("meshio")
import jno  # noqa: E402


def _mesh_with_orphan():
    """One tetrahedron on points 0-3, plus point 4 referenced only by a 0-D ``vertex`` cell
    (exactly what gmsh emits for an arc centre / spline control point)."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 5, 5]], dtype=float)
    cells = [("tetra", np.array([[0, 1, 2, 3]])), ("vertex", np.array([[4]]))]
    return meshio.Mesh(pts, cells)


def test_orphan_nodes_dropped_by_default():
    dom = jno.domain(lambda geo: (_mesh_with_orphan(), 3, 0.1), compute_mesh_connectivity=False)
    assert len(dom.mesh.points) == 4  # the orphan point is gone
    assert all(cb.type != "vertex" for cb in dom.mesh.cells)  # the 0-D block is gone
    # the surviving tetra still references its four corners, renumbered in place
    assert set(np.asarray(dom.mesh.cells_dict["tetra"]).reshape(-1)) == {0, 1, 2, 3}


def test_keep_orphan_nodes_opt_out():
    dom = jno.domain(lambda geo: (_mesh_with_orphan(), 3, 0.1), compute_mesh_connectivity=False, keep_orphan_nodes=True)
    assert len(dom.mesh.points) == 5  # nothing dropped


def test_clean_mesh_is_untouched():
    """A mesh whose every node supports an element is returned unchanged (no false drops)."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    dom = jno.domain(
        lambda geo: (meshio.Mesh(pts, [("tetra", np.array([[0, 1, 2, 3]]))]), 3, 0.1), compute_mesh_connectivity=False
    )
    assert len(dom.mesh.points) == 4
