"""`jno.geometry.cell_fractions`: exact area fractions of a rectangle region on a non-conforming mesh.

The oracle is the geometry. The fractions weighted by the cell areas must add up to the region's area, a cell
inside the region is 1 and one outside is 0, a single triangle against a box has a hand-computed overlap, and the
outline of a box is vertical on its sides and horizontal on its top.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.geometry.raster import closing_boxes, rect_boxes, tri_box_area


@pytest.fixture(scope="module")
def mesh():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.07).domain()
    m = d.mesh
    return np.asarray(m.points)[:, :2], np.asarray(m.cells_dict["triangle"])


def _areas(pts, tris):
    T = pts[tris]
    return 0.5 * np.abs(
        (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
    )


def test_fractions_add_up_to_the_area_of_the_region(mesh):
    region = jno.shape.rect(0.13, 0.21, 0.58, 0.47) | jno.shape.rect(0.4, 0.4, 0.77, 0.9)
    frac = jno.geometry.cell_fractions(mesh, region)
    area = 0.45 * 0.26 + 0.37 * 0.5 - (0.58 - 0.4) * (0.47 - 0.4)
    assert np.all((frac >= 0) & (frac <= 1))
    np.testing.assert_allclose(np.sum(frac * _areas(*mesh)), area, rtol=1e-12)


def test_cells_inside_are_one_and_cells_outside_are_zero(mesh):
    pts, tris = mesh
    frac = jno.geometry.cell_fractions(mesh, jno.shape.rect(0.2, 0.2, 0.8, 0.8))
    c = pts[tris]
    inside = np.all((c > 0.2) & (c < 0.8), axis=(1, 2))
    outside = np.all(c[..., 0] < 0.2, axis=1) | np.all(c[..., 0] > 0.8, axis=1)
    assert np.all(frac[inside] == 1.0) and np.all(frac[outside] == 0.0)


def test_triangle_box_overlap_in_closed_form_with_numpy_and_jax():
    T = np.array([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]])
    B = np.array([[0.0, 0.0, 1.0, 1.0]])  # the unit square lies inside the triangle
    C = np.array([[1.0, 0.0, 3.0, 3.0]])  # the triangle's corner x > 1: half of 1 x 1
    assert tri_box_area(T, B)[0] == pytest.approx(1.0)
    assert tri_box_area(T, C)[0] == pytest.approx(0.5)
    jax_area = jax.jit(lambda t, b: tri_box_area(t, b, jnp))(jnp.asarray(T), jnp.asarray(C))
    assert float(jax_area[0]) == pytest.approx(0.5)


def test_closing_fills_a_slit_narrower_than_twice_the_radius():
    boxes = np.array([[0.0, 0.0, 1.0, 1.0], [1.01, 0.0, 2.0, 1.0]])
    closed = closing_boxes(boxes, 0.01)
    assert np.sum((closed[:, 2] - closed[:, 0]) * (closed[:, 3] - closed[:, 1])) == pytest.approx(2.0)
    assert rect_boxes(jno.shape.rect(0, 0, 1, 1) - jno.shape.rect(0.2, 0.2, 0.4, 0.4)).shape[1] == 4


def test_outline_normals_are_vertical_on_the_sides_and_horizontal_on_top(mesh):
    pts, tris = mesh
    frac, nx2 = jno.geometry.cell_fractions(mesh, jno.shape.rect(0.31, 0.0, 0.62, 0.53), normals=True)
    c = pts[tris].mean(1)
    side = (np.abs(c[:, 0] - 0.31) < 0.02) & (c[:, 1] > 0.1) & (c[:, 1] < 0.4) & (frac > 0) & (frac < 1)
    top = (np.abs(c[:, 1] - 0.53) < 0.02) & (np.abs(c[:, 0] - 0.465) < 0.1) & (frac > 0) & (frac < 1)
    assert side.any() and top.any()
    np.testing.assert_allclose(nx2[side], 1.0)
    np.testing.assert_allclose(nx2[top], 0.0)
