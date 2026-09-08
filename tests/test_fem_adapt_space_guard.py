"""The ZZ / Hessian estimator reads VERTEX values, so it must know which spaces actually have them.

``_vertex_view`` takes the vertex part of a solution vector. For a scalar field with more DOFs than
vertices it returns ``sol[:n_vert]``, whose comment says "higher-order scalar: the first n_vert DOFs
are the vertex (nodal) values". That is true for higher-order **Lagrange** -- P2/P3 keep the P1
vertices as ids ``0..n_vert-1`` -- and false for every other scalar space jNO has:

  * **Hermite** carries value and both derivatives per vertex, with the VALUE at ``3*v``. The first
    ``n_vert`` DOFs are the value and two derivatives of the first ``n_vert/3`` vertices.
  * **Argyris** is the same with 6 DOFs per vertex.
  * **P0** has one DOF per CELL and no vertex values at all.

The guard was on the DOF-count SHAPE, not on the space, so those cases fell through the ``>``
branch and the estimator refined on a reinterpretation of unrelated DOFs -- silently, since the
array has the right length and finite values.
"""

import numpy as np
import pytest

import jno  # noqa: E402
from jno.utils.solver.fem_adapt import _vertex_view  # noqa: E402


def _hermite_fem():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols(space="Hermite")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - (-6.0 * (xi + yi)) * vi])


def test_a_hermite_field_is_refused_not_reinterpreted():
    d, fem = _hermite_fem()
    n_vert = len(d.mesh.points)
    sol = np.arange(3 * n_vert, dtype=float)  # Hermite: value+2 derivatives per vertex
    with pytest.raises(NotImplementedError, match="Hermite"):
        _vertex_view(sol, fem)


def test_a_lagrange_field_still_works():
    """P1 and higher-order Lagrange are the case the branch was written for and must be untouched."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi])
    n_vert = len(d.mesh.points)
    got = _vertex_view(np.arange(n_vert, dtype=float), fem)
    assert got.shape == (n_vert,)
    # a P2 vector is longer than n_vert and its first n_vert entries ARE the vertices
    got2 = _vertex_view(np.arange(n_vert + 7, dtype=float), fem)
    assert np.array_equal(got2, np.arange(n_vert, dtype=float))
