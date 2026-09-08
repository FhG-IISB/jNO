"""A graded mesh: ``shape(..., size=f(x, y, z))`` -> a gmsh ``setSizeCallback``.

``Size = Union[float, Callable, None]`` (``jno/geometry/shape.py``), and a callable becomes a
per-position mesh-size callback that composes with every other size control via ``min``
(``jno/geometry/emit.py``). It is the "denser here" knob, and the only way to resolve a thin feature
without paying for a uniform mesh everywhere.

Nothing tested it. The callable appeared in exactly two places in the suite --
``tests/test_shape_structured.py`` and ``tests/test_fdm_structured.py`` -- and in both it is a
*refusal* test for a path that rejects graded sizing before any meshing happens, so no test ever
built a graded mesh. Both of those write ``lambda x, y:``, which is the wrong arity: the callback
invokes ``f(x, y, z)`` in 2-D as well as 3-D.

Measured on a 1.2 x 0.4 mm rectangle graded from 4 um at the top to 30 um at the bottom: 2,035 nodes
against 35,226 for the uniform 4 um mesh it replaces, at the same near-surface resolution.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

LX, LY = 1.2e-3, 0.4e-3
H_FINE, H_COARSE, BAND = 4e-6, 30e-6, 100e-6


def _graded(x, y, z):
    """Fine in a band under the top edge, coarsening downward. Three arguments -- see the module note."""
    return H_FINE + (H_COARSE - H_FINE) * min(1.0, max(0.0, (LY - y) / BAND))


def _element_sizes(d):
    """Representative element size per cell, and each cell's centroid height."""
    pts = np.asarray(d.mesh.points)[:, :2]
    v = pts[np.asarray(d._cells_p1())]
    e1, e2 = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    return np.sqrt(2.0 * area), v[:, :, 1].mean(axis=1), len(pts)


def test_a_graded_size_callable_actually_grades_the_mesh():
    """The oracle is the callable itself: element size near an edge must track ``f`` there, not some
    average. Without this, `size=` could be silently ignored and the mesh would still look fine."""
    size, yc, _n = _element_sizes(jno.shape.rect(0.0, 0.0, LX, LY, size=_graded).domain())
    top = size[yc > LY - 0.25 * BAND].mean()
    bot = size[yc < 0.25 * BAND].mean()
    assert top == pytest.approx(H_FINE, rel=0.5), f"top band should be near {H_FINE * 1e6:.0f} um, got {top * 1e6:.2f}"
    assert bot == pytest.approx(H_COARSE, rel=0.5), f"bottom should be near {H_COARSE * 1e6:.0f} um, got {bot * 1e6:.2f}"
    assert bot > 3.0 * top, f"the mesh is not graded: bottom/top = {bot / top:.2f}"


def test_grading_buys_the_resolution_far_cheaper_than_a_uniform_mesh():
    """The whole point. Same size at the fine edge, a fraction of the nodes -- this is what makes a
    thin feature (a melt pool 48 um deep in a 400 um domain) affordable at all."""
    _s, _y, n_graded = _element_sizes(jno.shape.rect(0.0, 0.0, LX, LY, size=_graded).domain())
    _s2, _y2, n_uniform = _element_sizes(jno.shape.rect(0.0, 0.0, LX, LY, size=H_FINE).domain())
    assert n_graded < n_uniform / 5, f"graded {n_graded} vs uniform {n_uniform} -- grading bought little"


def test_a_size_callable_of_the_wrong_arity_is_refused_by_name():
    """``f(x, y)`` is the natural thing to write in 2-D and it is WRONG -- the callback passes three
    coordinates whatever the dimension.

    Unguarded, the TypeError is raised inside gmsh's C callback, where it surfaces as
    ``Wrong mesh element size lc = 0 (lcmin = 0, lcmax = 1e+22)`` -- a message that names neither the
    callback, nor its signature, nor the shape it came from. ``emit.py`` already guards the sibling
    case (a size function returning a 1-element array) for exactly this reason; this is the other half.
    """
    two_arg = lambda x, y: H_FINE  # noqa: E731
    with pytest.raises(TypeError, match=r"f\(x, y, z\)"):
        # `.mesh` forces the build: meshing is LAZY, so `.domain()` alone never reaches the callback
        # and the wrong signature would sail through until something first asked for a mesh.
        jno.shape.rect(0.0, 0.0, LX, LY, size=two_arg).domain().mesh


def test_a_graded_mesh_composes_with_a_time_grid():
    """Grading is geometry and the time grid is not, but they meet on the domain -- and a melt-pool
    model needs both at once."""
    d = jno.shape.rect(0.0, 0.0, LX, LY, size=_graded).domain(time=(0.0, 1.0e-3, 5))
    size, yc, n = _element_sizes(d)
    assert n > 0 and np.isfinite(size).all()
    assert size[yc < 0.25 * BAND].mean() > 3.0 * size[yc > LY - 0.25 * BAND].mean()
