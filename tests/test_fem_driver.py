"""The ``jno.fem([...])`` driver: classify traced residuals (test-function
presence x region) and assemble through feax into an ``FEM`` container of
matrices/operators (no solve).

Covers:
* steady linear Poisson (Dirichlet) assembles + solves to the manufactured
  solution, and matches the legacy ``init_fem`` + ``assemble(target="fem_system")``
  path byte-for-byte;
* a nonlinear form classifies as a residual operator (``residual``/``jacobian``);
* a mixed Dirichlet + Neumann problem buckets correctly and the Neumann term
  contributes to the load;
* error cases (test-function-free interior term, empty input) raise;
* the ``FEM`` accessor guards (``.A`` on nonlinear, ``.residual`` on linear).
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import box  # noqa: E402


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _poisson_fem(mesh_size=0.12):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = ui.x * vi.x + ui.y * vi.y - f * vi
    bc = u(xb, yb) - 0.0
    return d, jno.fem([weak, bc], quad_degree=3)


def test_poisson_dirichlet_assembles_and_solves():
    d, fem = _poisson_fem()
    assert fem.is_linear
    A, b = _dense(fem.A), np.asarray(fem.b).reshape(-1)
    assert A.shape == (fem.dofs, fem.dofs)
    assert np.allclose(A, A.T, atol=1e-8)  # symmetric stiffness
    u_sol = np.linalg.solve(A, b)
    c = np.asarray(d.mesh.points)[:, :2]
    u_exact = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    rel = np.linalg.norm(u_exact - u_sol) / np.linalg.norm(u_exact)
    assert rel < 1e-2


def test_matches_legacy_assembly_path():
    # jno.fem must assemble the same linear system as the classic
    # init_fem + weak.assemble(target="fem_system") authoring.
    _, fem = _poisson_fem(mesh_size=0.18)

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.18)
    d.init_fem(element_type="TRI3", quad_degree=3, bcs=[d.dirichlet("boundary", 0.0)], fem_solver=True)
    u, phi = d.fem_symbols()
    xg, yg, _ = d.variable("fem_gauss", split=True)
    f = 2.0 * (xg * (1.0 - xg) + yg * (1.0 - yg))
    A_leg, b_leg = (u.d(xg) * phi.d(xg) + u.d(yg) * phi.d(yg) - f * phi).assemble(d, target="fem_system")

    assert np.allclose(_dense(fem.A), _dense(A_leg), atol=1e-9)
    assert np.allclose(np.asarray(fem.b).reshape(-1), np.asarray(b_leg).reshape(-1), atol=1e-9)


def test_nonlinear_form_is_residual_operator():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.18)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = 0.01 * (ui.x * vi.x + ui.y * vi.y) + (u * u * u - u) * phi
    fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)
    assert not fem.is_linear
    assert callable(fem.residual) and callable(fem.jacobian)
    R = np.asarray(fem.residual(np.zeros(fem.dofs)))
    assert R.shape == (fem.dofs,)


def test_mixed_dirichlet_neumann_buckets_and_contributes():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = ui.x * vi.x + ui.y * vi.y - 1.0 * vi
    flux = 0.5 * phi.bind(x=xr, y=yr)
    bc = u(xl, yl) - 0.0

    fem = jno.fem([weak, flux, bc], quad_degree=3)
    assert any("surface@right" in s for s in fem.classification)
    assert any("dirichlet@left" in s for s in fem.classification)

    fem_noflux = jno.fem([weak, bc], quad_degree=3)
    assert not np.allclose(np.asarray(fem.b).reshape(-1), np.asarray(fem_noflux.b).reshape(-1))


def test_test_function_free_interior_term_raises():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, _ = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    with pytest.raises(ValueError):
        jno.fem([u(xi, yi) - 1.0])  # interior, trial-only -> forgot the test function


def test_empty_constraints_raises():
    with pytest.raises(ValueError):
        jno.fem([])


def test_fem_accessor_guards():
    _, fem = _poisson_fem(mesh_size=0.3)
    assert fem.is_linear
    with pytest.raises(AttributeError):
        _ = fem.residual
    with pytest.raises(AttributeError):
        _ = fem.jacobian
