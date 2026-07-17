"""Higher-order (P2, P3, P4, …) Lagrange element coverage for ``jno.fem``.

jno's domain machinery assumes linear cells, so the domain mesh stays P1 and the FEM *assembly* mesh is
promoted to P{order} (the element's basix interpolation points placed on each cell and deduplicated by
coordinate; vertices preserved) only for assembly. A P{k} element captures polynomials up to degree
``k`` exactly, which lower orders cannot — the manufactured harmonic-polynomial recovery is the check,
and it also validates the node generation (a wrong multi-node-edge orientation gives an O(1) error, not
machine precision).

Order is per-field via ``fem_symbols(order=k)``; the element is inferred from ``(dimension, order)``.
The solution lives on the P{k} nodes exposed by ``fem.points``.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64, so these tests opt into x64 per-test. The session default is
    x64-off (see tests/conftest.py); save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _solve(fem):
    return np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))


def _poisson_xy(order):
    # -lap u = 0, u = xy on the boundary -> u = xy (xy is harmonic, degree 2).
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    nverts = int(np.asarray(d.mesh.points).shape[0])
    u, phi = d.fem_symbols(order=order)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - xb * yb])
    pts = np.asarray(fem.points)
    sol = _solve(fem)
    rel = np.linalg.norm(pts[:, 0] * pts[:, 1] - sol) / np.linalg.norm(pts[:, 0] * pts[:, 1])
    return fem, nverts, pts, rel


def test_p2_poisson_recovers_quadratic_exactly():
    fem, nverts, pts, rel = _poisson_xy(order=2)
    # the P2 space was actually built (vertices + edge nodes), not a silent P1 fallback
    assert fem.dofs > nverts
    assert pts.shape[0] == fem.dofs
    A = _dense(fem.A)
    assert np.allclose(A, A.T, atol=1e-10)
    assert rel < 1e-9  # P2 captures xy exactly


def test_p1_cannot_capture_quadratic():
    # control: the same problem at P1 is NOT exact (xy is not piecewise linear),
    # so P2's machine-precision recovery is meaningful, not trivially true.
    fem, nverts, pts, rel = _poisson_xy(order=1)
    assert fem.dofs == nverts  # P1: DOFs are just the vertices
    # xy is bilinear (not affine), so P1 has a real approximation error (~1e-3 here)
    # -- many orders above P2's machine-precision recovery, which is the point.
    assert rel > 1e-5


def _poisson_harmonic(order, u_fn, mesh_size=0.34):
    """-Δu = 0 with Dirichlet u = u_fn on the boundary -> u = u_fn (a harmonic polynomial). A P{order}
    element captures any polynomial up to degree ``order`` exactly, so a harmonic polynomial of degree
    ``order`` is recovered to machine precision iff the P{order} node generation is correct (a wrong
    multi-node-edge orientation would scramble cross-cell continuity and give an O(1) error)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    nverts = int(np.asarray(d.mesh.points).shape[0])
    u, phi = d.fem_symbols(order=order)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - u_fn(xb, yb)])
    pts = np.asarray(fem.points)
    sol = _solve(fem)
    exact = np.asarray(u_fn(pts[:, 0], pts[:, 1]))
    return fem, nverts, np.linalg.norm(exact - sol) / np.linalg.norm(exact)


def test_p3_recovers_cubic_exactly():
    # u = Re((x+iy)^3) = x^3 - 3xy^2 is harmonic and cubic. P3 has 2 nodes per edge, so this is the
    # patch test that catches a wrong edge-node orientation across shared cells.
    fem, nverts, rel = _poisson_harmonic(3, lambda x, y: x**3 - 3 * x * y**2)
    assert fem.dofs > nverts  # P3 nodes (vertices + 2/edge + interior) were built
    assert rel < 1e-9, f"P3 did not recover the cubic exactly: rel={rel:.2e}"


def test_p4_recovers_quartic_exactly():
    # u = Re((x+iy)^4) = x^4 - 6x^2y^2 + y^4 is harmonic and quartic; P4 has 3 nodes per edge + 3 interior.
    fem, nverts, rel = _poisson_harmonic(4, lambda x, y: x**4 - 6 * x**2 * y**2 + y**4)
    assert fem.dofs > nverts
    assert rel < 1e-9, f"P4 did not recover the quartic exactly: rel={rel:.2e}"


def test_p3_3d_recovers_cubic_exactly():
    pytest.importorskip("pygmsh", reason="pygmsh required for cube meshing")
    # 3D P3 (TET20 via promotion): harmonic cubic u = x^3 - 3x y^2 (independent of z, still harmonic),
    # recovered exactly -- exercises the tet edge + face node placement.
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.6).domain()
    nverts = int(np.asarray(d.mesh.points).shape[0])
    u, phi = d.fem_symbols(order=3)
    co = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
    bc = u(cb[0], cb[1], cb[2]) - (cb[0] ** 3 - 3 * cb[0] * cb[1] ** 2)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, bc])
    assert fem.dofs > nverts
    pts = np.asarray(fem.points)
    sol = _solve(fem)
    exact = pts[:, 0] ** 3 - 3 * pts[:, 0] * pts[:, 1] ** 2
    assert np.linalg.norm(exact - sol) / np.linalg.norm(exact) < 1e-9


def test_p2_3d_recovers_quadratic_exactly():
    pytest.importorskip("pygmsh", reason="pygmsh required for cube meshing")
    # 3D P2 (TET10 via promotion of the linear cube). u = xy + yz + xz (degree 2,
    # harmonic) recovered exactly; exercises all three edge directions.
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    nverts = int(np.asarray(d.mesh.points).shape[0])
    u, phi = d.fem_symbols(order=2)
    co = d.variable("interior", split=True)
    xi, yi, zi = co[0], co[1], co[2]
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    bc = u(cb[0], cb[1], cb[2]) - (cb[0] * cb[1] + cb[1] * cb[2] + cb[0] * cb[2])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, bc])
    assert fem.dofs > nverts  # TET10 built (vertices + edge nodes)
    pts = np.asarray(fem.points)
    sol = _solve(fem)
    exact = pts[:, 0] * pts[:, 1] + pts[:, 1] * pts[:, 2] + pts[:, 0] * pts[:, 2]
    assert np.linalg.norm(exact - sol) / np.linalg.norm(exact) < 1e-9
