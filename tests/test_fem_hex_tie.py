"""Tying two hexahedral domains across a **non-matching** interface.

A tie constrains each secondary node to a weighted combination of the main face's nodes. Those weights
came from barycentric (triangle) shape functions, so a hexahedron's *quadrilateral* facet had none —
the tie refused by name rather than interpolating a quad from three of its four nodes.

A quadrilateral facet's map is **bilinear** and has no closed-form inverse, which is why it could not
simply reuse the triangle formulas: the reference coordinates come from a Newton inversion
(``fem_lagrange._invert_tensor_map``, the same inverse the quad solution transfer uses).

The decisive assertion is the **linear patch test**: a field the main space can represent exactly must
come through the tie exactly. Weights that sum to 1 but sit at the wrong reference point still pass a
partition-of-unity check and fail this one.

This is also the constraint mechanism hanging nodes need — a hanging node is a secondary node lying on
a coarse facet — so the weights are exercised here before anything is built on them.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.domain.geometries import Geometries
from jno.utils.solver.fem_utils import _periodic_facet_weights

meshio = pytest.importorskip("meshio")


@pytest.fixture
def x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------- the weights, on their own


def _warped_quad_patch(nx=3, amp=0.22, seed=0):
    """A patch of quadrilateral facets in the interface plane, deliberately NON-parallelogram.

    Warping matters: on parallelograms the bilinear map degenerates to an affine one, and the triangle
    formulas this replaces would pass. The Jacobian has to genuinely vary within the facet.
    """
    rng = np.random.default_rng(seed)
    g = np.linspace(0, 1, nx + 1)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(float)
    inner = (pts[:, 0] > 1e-9) & (pts[:, 0] < 1 - 1e-9) & (pts[:, 1] > 1e-9) & (pts[:, 1] < 1 - 1e-9)
    pts[inner] += (amp / nx) * rng.uniform(-1, 1, (int(inner.sum()), 2))

    def nid(i, j):
        return i * (nx + 1) + j

    facets = np.array([[nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)] for i in range(nx) for j in range(nx)])
    return pts, facets


def test_quad_facet_weights_are_a_partition_of_unity():
    pts, facets = _warped_quad_patch()
    rng = np.random.default_rng(1)
    for q in rng.uniform(0.02, 0.98, (200, 2)):
        w = np.array([x for _, x in _periodic_facet_weights(q, facets, pts)])
        assert abs(w.sum() - 1.0) < 1e-12


def test_quad_facet_weights_reproduce_a_linear_field_exactly():
    """The patch test at weight level, on warped facets. Interpolating a quadrilateral from three of
    its four nodes — what the triangle branch would do — fails this by O(1)."""
    pts, facets = _warped_quad_patch()
    field = 1.3 + 2.0 * pts[:, 0] - 0.7 * pts[:, 1]
    rng = np.random.default_rng(2)
    worst = 0.0
    for q in rng.uniform(0.02, 0.98, (200, 2)):
        w = _periodic_facet_weights(q, facets, pts)
        got = sum(wt * field[i] for i, wt in w)
        worst = max(worst, abs(got - (1.3 + 2.0 * q[0] - 0.7 * q[1])))
    assert worst < 1e-12, f"linear reproduction off by {worst:.3e}"


def test_a_query_at_a_node_returns_that_node():
    """The interpolation must be nodal: at a facet vertex the weight there is 1 and the rest 0."""
    pts, facets = _warped_quad_patch()
    for c in range(len(pts)):
        w = dict(_periodic_facet_weights(pts[c], facets, pts))
        assert abs(w.get(c, 0.0) - 1.0) < 1e-10, f"node {c} did not recover itself: {w}"


def test_a_facet_arity_with_no_shape_functions_is_refused():
    """P3 quadrilateral (9-node) facets and the like: refused rather than interpolated with the Q1
    formulas, which would silently misplace the extra nodes."""
    pts, facets = _warped_quad_patch()
    nine = np.concatenate([facets, facets[:, :5]], axis=1)
    with pytest.raises(NotImplementedError, match="non-matching periodic/tied interface"):
        _periodic_facet_weights(np.array([0.5, 0.5]), nine, pts)


# ------------------------------------------------------------------------- end to end, two blocks


def _two_hex_blocks(tmp_path, na=4, nb=3):
    """Two hexahedral lattices meeting at x = 1 at DIFFERENT resolutions, with duplicated interface
    nodes — the configuration `Shape.regions(..., conforming=False)` produces, built directly because
    that path meshes through gmsh, which cannot hex-mesh."""

    def block(x0, x1, n):
        m, _, _ = Geometries.equi_distant_box(
            x_range=(x0, x1), y_range=(0, 1), z_range=(0, 1), nx=n, ny=n, nz=n, cell="hex"
        )(None)
        b = {c.type: np.asarray(c.data) for c in m.cells}
        return np.asarray(m.points), b["hexahedron"], b["quad"]

    pa, ha, qa = block(0.0, 1.0, na)
    pb, hb, qb = block(1.0, 2.0, nb)
    pts = np.vstack([pa, pb])
    hexes = np.vstack([ha, hb + len(pa)])
    quads = np.vstack([qa, qb + len(pa)])
    at_iface = np.all(np.abs(pts[quads][:, :, 0] - 1.0) < 1e-9, axis=1)
    is_a = np.arange(len(quads)) < len(qa)
    on_a, on_b = at_iface & is_a, at_iface & ~is_a
    empty = np.array([], dtype=np.int64)
    sets = {
        "interior": [np.arange(len(hexes), dtype=np.int64), empty],
        "boundary": [empty, np.where(~at_iface)[0].astype(np.int64)],
        "iface_a": [empty, np.where(on_a)[0].astype(np.int64)],
        "iface_b": [empty, np.where(on_b)[0].astype(np.int64)],
    }
    path = str(tmp_path / "hextie.inp")
    meshio.write(
        path,
        meshio.Mesh(points=pts, cells=[("hexahedron", hexes), ("quad", quads)], cell_sets=sets),
        file_format="abaqus",
    )
    return path, pts, int(on_a.sum()), int(on_b.sum())


def test_a_non_matching_hex_tie_passes_the_linear_patch_test(tmp_path, x64):
    """The deliverable. Two hex blocks at different resolutions, tied, must reproduce ``u = x``
    exactly — the field is linear, the main face can represent it, and a correct constraint carries it
    across. Measured 5.6e-16; before this the tie refused outright.

    Solved directly: the default Jacobi-BiCGStab stops at its own tolerance (4.1e-04 in float32 here),
    which would hide a constraint error of the same size behind solver noise.
    """
    path, pts, n_a, n_b = _two_hex_blocks(tmp_path)
    assert n_a != n_b, f"the interface is matching ({n_a} vs {n_b}); nothing non-conforming is tested"

    d = jno.domain(path, compute_mesh_connectivity=False)
    u, v = d.fem_symbols()
    ci = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ca = d.variable("iface_a", split=True)
    cbb = d.variable("iface_b", split=True)
    ui, vi = u.bind(x=ci[0], y=ci[1], z=ci[2]), v.bind(x=ci[0], y=ci[1], z=ci[2])
    lap = ui.x * vi.x + ui.y * vi.y + ui.z * vi.z
    fem = jno.fem([lap, u(*cb[:3]) - cb[0], u(*ca[:3]) - u(*cbb[:3])])
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).ravel()
    assert np.abs(sol[: len(pts)] - pts[:, 0]).max() < 1e-10


def test_the_integrated_mortar_coupling_still_refuses_on_a_quad_facet():
    """Only the COLLOCATED coupling gained quadrilateral support. The integrated (mortar) rows clip
    triangles and have no quad analogue, so they must still refuse — and the availability check keeps
    routing quad facets away from them, which is why the tie above reaches the collocated path."""
    from jno.utils.solver.fem_utils import _tri_dual_available, _tri_shape

    assert _tri_dual_available(3) is True
    assert _tri_dual_available(4) is False  # a quad facet: unavailable, not an error
    with pytest.raises(NotImplementedError, match="HEXAHEDRAL facet"):
        _tri_shape(np.zeros((1, 3)), 4)
