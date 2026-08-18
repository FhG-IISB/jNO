"""``domain.cell_aspect()`` — per-cell distortion, dimension-generic and differentiable in the mesh.

``cell_size`` is ``|det J|^(1/dim)``, an isotropic SIZE that cannot see stretch: a sliver and a
regular element of the same area share it. ``cell_angles`` can see it but is 2-D only. This is the
quantity a mesh-quality condition is written on, on triangles and tetrahedra alike.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

meshio = pytest.importorskip("meshio")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _ref_tri(v):
    """Longest edge / inradius / (2 sqrt 3) — written out directly, no shared helper with the code."""
    e = [np.linalg.norm(v[j] - v[i]) for i, j in itertools.combinations(range(3), 2)]
    e1, e2 = v[1] - v[0], v[2] - v[0]
    area = abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2.0
    return max(e) / ((2 * area / sum(e)) * 2 * np.sqrt(3))


def _ref_tet(v):
    e = [np.linalg.norm(v[j] - v[i]) for i, j in itertools.combinations(range(4), 2)]
    vol = abs(np.linalg.det(np.stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]], axis=-1))) / 6.0
    surf = 0.0
    for k in range(4):
        keep = [i for i in range(4) if i != k]
        surf += np.linalg.norm(np.cross(v[keep[1]] - v[keep[0]], v[keep[2]] - v[keep[0]])) / 2.0
    return max(e) / ((3 * vol / surf) * 2 * np.sqrt(6))


def _eval(node, dom):
    return np.asarray(jno.core([node], domain=dom).eval([node], domain=dom)).reshape(-1)


def test_a_regular_simplex_reads_exactly_one_in_2d():
    """The normalisation is the whole point of the measure: 1.0 is 'as good as a triangle gets'."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2], [2.0, 0.0], [3.0, 0.0], [2.5, 0.05]])
    cells = np.array([[0, 1, 2], [3, 4, 5]])
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    d._apply_mesh(
        meshio.Mesh(
            np.c_[pts, np.zeros(len(pts))],
            [("triangle", cells)],
            cell_sets={"interior": [np.arange(2)], "boundary": [np.array([], dtype=np.int64)]},
        )
    )
    got = _eval(d.cell_aspect(), d)
    assert got[0] == pytest.approx(1.0, abs=1e-12), f"an equilateral triangle must read 1.0, got {got[0]}"
    assert got[1] > 10.0, f"a sliver must read much worse than a regular element: {got[1]}"
    np.testing.assert_allclose(got, [_ref_tri(pts[c]) for c in cells], rtol=1e-12)


def test_it_matches_a_direct_reference_in_3d():
    """Dimension-generic is the reason this exists rather than a 3-D `cell_angles`."""
    from jno.domain.geometries import Geometries

    mesh, _, _ = Geometries.equi_distant_box(nx=2, ny=2, nz=2)(None)
    d = jno.domain(lambda g: (mesh, 3, 0.5), compute_mesh_connectivity=True)
    pts = np.asarray(d.mesh.points)[:, :3]
    cells = np.asarray(d.mesh.cells_dict["tetra"])
    got = _eval(d.cell_aspect(), d)
    assert got.shape == (cells.shape[0],)
    np.testing.assert_allclose(got, [_ref_tet(pts[c]) for c in cells], rtol=1e-10)
    assert (got >= 1.0 - 1e-12).all(), "1.0 is the regular-simplex floor; nothing may read below it"
    # the reference normalisation, on a tetrahedron that is actually regular
    reg = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    assert _ref_tet(reg) == pytest.approx(1.0, abs=1e-12)


def test_the_gradient_in_the_vertices_matches_finite_differences():
    """It is a mesh-motion quantity, so the gradient is the half that has to be right -- and a
    nonzero gradient is not evidence, a wrong one is also nonzero."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.35).domain()
    _xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
    ym.trainable(name="iy")
    node = d.cell_aspect()
    sp = d._trainable_coords[0]
    base = jnp.asarray(np.asarray(d.mesh.points)[np.asarray(sp["ids"], int), int(sp["axis"])])
    fn = node.fn  # the closure the trace evaluates, over the trainable coordinates

    def total(v):
        return jnp.sum(fn(v))

    _val, grad = jax.value_and_grad(total)(base)
    grad = np.asarray(grad)
    assert np.linalg.norm(grad) > 1e-6, "no dependence on the vertices at all"
    h = 1e-6
    fd = np.array(
        [(float(total(base.at[j].add(h))) - float(total(base.at[j].add(-h)))) / (2 * h) for j in range(grad.size)]
    )
    assert np.linalg.norm(fd - grad) / np.linalg.norm(fd) < 1e-6


def test_1d_is_refused_by_name():
    """A 1-D cell has no shape to be bad, so there is nothing to measure -- say so rather than
    returning a number that means nothing."""
    d = jno.domain.line((0.0, 1.0), 0.2)
    with pytest.raises(NotImplementedError, match="simplices in 2-D or 3-D"):
        d.cell_aspect()
