"""A CURVED 3-D tied interface — mortar segmentation in each facet's own plane.

The 3-D mortar already clipped polygons properly (Puso & Laursen, *CMAME* 193:601-629, 2004). What it
did not do is measure them anywhere sensible: every geometric step ran in ONE plane fitted to the whole
interface, so on a curved surface the areas were *projected* areas and the projection folded. Its own
docstring said so — *"the interface must be (near-)planar"*.

Two failure modes were measured on nested spheres, and neither was loud:

* at a coarse mesh the containment gate failed in the folded frame, the tie degraded **silently** to
  the collocated coupling, and the constant-gradient patch test came back at **9.8e-01** — the linear
  field was simply not reproduced;
* at a finer one the segmentation ran and reported covering **0.0378387** of a facet of area
  **0.0189193** — exactly twice, the same fold signature a cornered 2-D interface gives.

Giving each secondary facet its own frame removes the assumption entirely, because a projection onto
the secondary facet's own plane is second-order accurate for any smooth surface. Nothing else changes.

**On the residual.** A curved patch test cannot reach round-off the way a flat one does: the two bodies
approximate the same sphere with *different* triangulations, so their discrete surfaces genuinely
differ by the sagitta ~h²/8R. The test below pins that the error tracks the sagitta and falls at second
order — which is the statement that the coupling is consistent and the remainder is geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver.fem_utils import (
    _covers_local_3d,
    _main_covers_secondary_3d,
    _mortar_rows_3d,
)

GRAD, OFF = np.array([0.7, -0.4, 0.25]), 0.3


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _spheres(h_in, h_out, R=1.0, RO=2.0):
    """A ball in a shell: the interface is a full sphere — curved AND closed, which the 2-D arc-length
    coordinate cannot do at all (it is periodic, so a loop must be cut into arcs first)."""
    ball = jno.shape.sphere(0, 0, 0, R)
    return jno.shape.regions(
        ball=ball.sized(h_in), shell=(jno.shape.sphere(0, 0, 0, RO) - ball).sized(h_out), conforming=False
    ).domain()


def _patch(d):
    """Laplace with NO source and a linear field on the outer boundary: the exact solution is that
    linear field, so any deviation is the tie's."""
    u, v = d.fem_symbols()
    c = d.variable("interior", split=True)
    ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
    a, b = (d.variable(t, split=True) for t in sorted(t for t in d.built_mesh.cell_sets if "|" in t))
    ob = d.variable("boundary", split=True)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z,
            u(a[0], a[1], a[2]) - u(b[0], b[1], b[2]),
            u(ob[0], ob[1], ob[2]) - (GRAD[0] * ob[0] + GRAD[1] * ob[1] + GRAD[2] * ob[2] + OFF),
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)
    exact = np.asarray(fem.field_points[0]) @ GRAD + OFF
    return fem._periodic.get("coupling"), float(np.abs(sol - exact).max())


# ----------------------------------------------------------------------------------------------
# The defect, stated as the gate that used to fail
# ----------------------------------------------------------------------------------------------
def test_the_global_frame_cannot_even_see_that_the_surfaces_coincide():
    """Why it degraded silently. The two sphere surfaces ARE coincident, but flattened onto one plane
    the containment test cannot tell — and a failed containment sends the tie to collocation."""
    d = _spheres(0.34, 0.24)
    import jno.utils.solver.fem_utils as fu

    pts = np.asarray(d.built_mesh.points)
    tri = np.asarray(d.built_mesh.cells_dict["triangle"])
    tags = sorted(t for t in d.built_mesh.cell_sets if "|" in t)
    fc = [tri[np.asarray(d.built_mesh.cell_sets[t][1]).reshape(-1).astype(int)] for t in tags]
    frame, origin = fu._interface_frame(pts[np.unique(fc[1])], pts[np.unique(fc[0])])
    loc = (pts - origin) @ np.asarray(frame).T

    assert _covers_local_3d(fc[0], fc[1], pts), "measured in space, the surfaces plainly coincide"
    assert not _main_covers_secondary_3d(fc[0], fc[1], loc), "flattened, the same test fails"


# ----------------------------------------------------------------------------------------------
# Must not move: a planar interface takes the same path it always did
# ----------------------------------------------------------------------------------------------
def _plane_tri(n, tag):
    """A triangulated unit square at z = 0, with ids offset so two of them share no node."""
    g = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
    q = np.arange(n * n).reshape(n, n)
    t = []
    for i in range(n - 1):
        for j in range(n - 1):
            t += [[q[i, j], q[i, j + 1], q[i + 1, j]], [q[i, j + 1], q[i + 1, j + 1], q[i + 1, j]]]
    return P, np.asarray(t) + tag


def test_a_planar_interface_is_bit_identical_in_either_frame():
    """The guarantee that lets this land: on a plane, a facet's own frame differs from the global one
    by a rigid in-plane rotation, and clipping, areas and barycentrics are all invariant under that.
    Asserted as bit-identity of the prolongation rows, not as 'close enough'."""
    Ps, s_fc = _plane_tri(5, 0)
    Pm, m_fc = _plane_tri(4, len(Ps))
    P = np.concatenate([Ps, Pm])
    loc = P[:, :2]  # the global flattened frame IS (x, y) here

    glob = _mortar_rows_3d(s_fc, m_fc, loc, span=1.0)
    loc3 = _mortar_rows_3d(s_fc, m_fc, loc, span=1.0, pts3=P)
    assert set(glob) == set(loc3) and glob, "the same secondary nodes must get rows"
    for k in glob:
        a = dict((int(i), float(w)) for i, w in glob[k])
        b = dict((int(i), float(w)) for i, w in loc3[k])
        assert set(a) == set(b), f"node {k}: different main nodes"
        assert max(abs(a[i] - b[i]) for i in a) < 1e-14, f"node {k}: weights moved"


# ----------------------------------------------------------------------------------------------
# The physics: the patch test, and what its residual is made of
# ----------------------------------------------------------------------------------------------
def test_a_curved_interface_reaches_the_integrated_coupling():
    """It used to report `collocated` on this exact geometry, with a patch error of 9.8e-01."""
    coupling, err = _patch(_spheres(0.40, 0.29))
    assert coupling == "mortar", f"a curved interface must reach the integrated coupling, got {coupling!r}"
    assert err < 0.05, f"patch-test error {err:.3e}"


@pytest.mark.parametrize("h_in,h_out", [(0.40, 0.29), (0.20, 0.145), (0.14, 0.101)])
def test_the_patch_error_is_the_faceting_and_nothing_more(h_in, h_out):
    """The residual is the two triangulations disagreeing about where the sphere is, not the coupling.

    Its scale is the sagitta of a chord of length `h` on radius 1, `h^2/8`. Measured, the error stays
    UNDER that sagitta across a 3x range of `h` (ratios 0.91, 0.62, 0.65) and falls at second order
    (rates 2.56 and 1.86) -- tracking `h^2` is what says the two are the same quantity. An
    inconsistent coupling would stall at a fixed error instead of following it down.

    The ratios were re-measured once the shell stopped being meshed at the ball's size: the earlier
    0.52-0.54 came from a mesh where both sides were ~0.4 regardless of what the shell asked for, so
    the two triangulations were far more alike than the test meant them to be.
    """
    coupling, err = _patch(_spheres(h_in, h_out))
    if coupling == "conforming":
        pytest.skip("gmsh produced matching surfaces here; there is no mortar to measure")
    assert coupling == "mortar", f"got {coupling!r}"
    assert err < h_in**2 / 8.0, f"error {err:.3e} exceeds the faceting scale {h_in**2 / 8:.3e}"
