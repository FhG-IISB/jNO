"""Coupled (multi-field) FEM robustness: domain/mesh-size independence + the harder
scenario combinations, all through ``jno.fem``.

The block-assembly machinery is geometry-agnostic — it only ever reads the auto-tagged
``interior``/``boundary`` regions and the mesh nodes — so the manufactured solution
``u* = x``, ``p* = y`` (linear, hence exact in P1 on *any* domain, with the cross terms
populating both off-diagonal blocks) must recover to machine precision regardless of the
domain shape or mesh size. These tests pin that explicitly across a rectangle (two mesh
sizes), an L-shaped polygon (vertex list), a disk, and a 3-D cube, and then exercise the
genuinely combined cases: nonlinear+coupled, nonlinear+coupled+transient, and mixed
Dirichlet/Neumann/Robin boundary conditions on one coupled problem.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import Point, box  # noqa: E402


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


# --------------------------------------------------------------------------------------
# domain / mesh-size independence (steady linear coupled)
# --------------------------------------------------------------------------------------
# Each factory returns a freshly meshed 2-D domain. The manufactured recovery below is
# identical across all of them — only the geometry and mesh size change.
_DOMAINS_2D = {
    "rect_h0.10": lambda: jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.10),
    "rect_h0.20": lambda: jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.20),
    "Lshape": lambda: jno.domain([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], mesh_size=0.25),
    "disk": lambda: jno.domain(Point(0.5, 0.5).buffer(0.5, resolution=40), mesh_size=0.12),
}


def _coupled_poisson(d):
    """-lap u + p = y ; -lap p + u = x ; u = x, p = y on the boundary.

    Solution u* = x, p* = y (linear -> exact in P1 on any domain); the +p and +u cross
    terms populate both off-diagonal blocks. Returns ``(fem, node_xy)``."""
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + pi * vi - yi * vi,
            pi.x * qi.x + pi.y * qi.y + ui * qi - xi * qi,
            u(xb, yb) - xb,
            p(xb, yb) - yb,
        ]
    )
    return fem, np.asarray(d.mesh.points)[:, :2]


@pytest.mark.parametrize("name", list(_DOMAINS_2D))
def test_coupled_linear_recovers_across_domains(name):
    # Same coupled Poisson, four geometries / mesh sizes -> all recover u=x, p=y exactly.
    d = _DOMAINS_2D[name]()
    fem, c = _coupled_poisson(d)
    n = c.shape[0]
    assert fem.is_linear and fem.dofs == 2 * n
    A = _dense(fem.A)
    assert np.any(np.abs(A[:n, n:]) > 1e-12) and np.any(np.abs(A[n:, :n]) > 1e-12)  # genuine coupling
    sol = np.linalg.solve(A, np.asarray(fem.b).reshape(-1))
    assert np.linalg.norm(sol[:n] - c[:, 0]) / np.linalg.norm(c[:, 0]) < 1e-8
    assert np.linalg.norm(sol[n:] - c[:, 1]) / np.linalg.norm(c[:, 1]) < 1e-8


def test_coupled_3d_cube_recovers():
    # The same block machinery on a 3-D tetrahedral cube: -lap u + p = y, -lap p + u = x.
    pytest.importorskip("pygmsh", reason="pygmsh required for cube meshing")
    d = jno.domain(constructor=jno.domain.cube(mesh_size=0.4))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, yi, zi = co[0], co[1], co[2]
    cb = d.variable("boundary", split=True)
    xb, yb, zb = cb[0], cb[1], cb[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    pi, qi = p.bind(x=xi, y=yi, z=zi), q.bind(x=xi, y=yi, z=zi)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z + pi * vi - yi * vi,
            pi.x * qi.x + pi.y * qi.y + pi.z * qi.z + ui * qi - xi * qi,
            u(xb, yb, zb) - xb,
            p(xb, yb, zb) - yb,
        ]
    )
    n = int(np.asarray(d.mesh.points).shape[0])
    assert fem.is_linear and fem.dofs == 2 * n
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    c = np.asarray(d.mesh.points)[:, :3]
    assert np.linalg.norm(sol[:n] - c[:, 0]) / np.linalg.norm(c[:, 0]) < 1e-8
    assert np.linalg.norm(sol[n:] - c[:, 1]) / np.linalg.norm(c[:, 1]) < 1e-8


# --------------------------------------------------------------------------------------
# combined scenarios: nonlinear+coupled, nonlinear+coupled+transient, mixed BCs
# --------------------------------------------------------------------------------------
def test_coupled_nonlinear_recovers_on_disk():
    # Nonlinear coupled on a non-box domain (disk): -lap u + u*p = x*y, -lap p + u^2 = x^2,
    # u=x, p=y on the boundary. u*=x, p*=y is the exact solution (the nonlinear terms equal
    # their sources at the solution), recovered by a Newton solve via feax's block residual/
    # Jacobian -> shows the nonlinear coupled path is domain-independent too.
    spo = pytest.importorskip("scipy.optimize")
    d = jno.domain(Point(0.5, 0.5).buffer(0.5, resolution=28), mesh_size=0.22)
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + (u * p) * vi - (xi * yi) * vi,  # -lap u + u*p = x*y
            pi.x * qi.x + pi.y * qi.y + (u * u) * qi - (xi * xi) * qi,  # -lap p + u^2 = x^2
            u(xb, yb) - xb,
            p(xb, yb) - yb,
        ]
    )
    n = int(np.asarray(d.mesh.points).shape[0])
    assert not fem.is_linear and fem.dofs == 2 * n
    sol = spo.root(
        lambda w: np.asarray(fem.residual(w)),
        np.zeros(fem.dofs),
        jac=lambda w: _dense(fem.jacobian(w)),
        method="hybr",
    )
    assert sol.success
    c = np.asarray(d.mesh.points)[:, :2]
    assert np.linalg.norm(sol.x[:n] - c[:, 0]) / np.linalg.norm(c[:, 0]) < 1e-7
    assert np.linalg.norm(sol.x[n:] - c[:, 1]) / np.linalg.norm(c[:, 1]) < 1e-7


def test_coupled_nonlinear_transient_recovers_manufactured():
    # The full triple: nonlinear + coupled + transient. Zero-flux (natural Neumann) so a
    # spatially-uniform field is exact in space; the system reduces to the ODEs
    #   u_t = -u ,  p_t = u^2   (the u^2 coupling makes it nonlinear: dR/du carries 2u),
    # with u(0)=1, p(0)=0 -> u = e^{-t}, p = (1 - e^{-2t})/2. feax autodiffs the block
    # residual/Jacobian; a Newton backward-Euler march recovers the analytic solution.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25, time=(0.0, 0.1, 11))
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pi, qi = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    u_eq = ui.t * vi + (ui.x * vi.x + ui.y * vi.y) + ui * vi  # u_t - lap u + u = 0 -> u_t = -u (uniform)
    uu = u.bind(x=xi, y=yi, t=ti)
    p_eq = pi.t * qi + (pi.x * qi.x + pi.y * qi.y) - (uu * uu) * qi  # p_t = u^2 (uniform)
    fem = jno.fem([u_eq, p_eq, u(ci[0], ci[1]) - 1.0, p(ci[0], ci[1]) - 0.0])

    assert fem.is_transient and not fem.is_linear
    n = int(np.asarray(d.mesh.points).shape[0])
    assert fem.dofs == 2 * n
    op = fem.operator
    M = _dense(op.mass(0.0, None))
    w = np.asarray(fem.state0).copy()
    assert abs(w[:n].mean() - 1.0) < 1e-12 and np.allclose(w[n:], 0.0)  # uniform IC
    dt = float(fem.dt)
    for _ in range(round((fem.t1 - fem.t0) / dt)):  # Newton backward-Euler: M(w-w_old)/dt + R(w) = 0
        w_old = w.copy()
        for _ in range(30):
            G = M @ (w - w_old) / dt + np.asarray(op.residual(w, 0.0, None))
            if np.linalg.norm(G) < 1e-11:
                break
            w = w - np.linalg.solve(M / dt + _dense(op.jacobian(w, 0.0, None)), G)

    u_ex = np.exp(-fem.t1)
    p_ex = (1.0 - np.exp(-2.0 * fem.t1)) / 2.0
    assert w[:n].std() < 1e-10 and w[n:].std() < 1e-10  # stays spatially uniform
    assert abs(w[:n].mean() - u_ex) / u_ex < 1e-2
    assert abs(w[n:].mean() - p_ex) / p_ex < 3e-2
    assert w[n:].mean() > 0.05  # p grew from zero through the nonlinear coupling


def test_coupled_mixed_bcs_recovers():
    # One coupled problem carrying every BC type at once: u has Dirichlet (left, bottom),
    # Neumann (right), and Robin (top); p is Dirichlet on the whole boundary. Manufactured
    # u*=x, p*=y. du*/dn = 1 on the right (Neumann flux), and on the top du*/dn = 0 so the
    # Robin condition du/dn + alpha*u = g gives g = alpha*x.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12)
    alpha = 3.0
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xbo, ybo, _ = d.variable("bottom", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xt, yt, _ = d.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    ut, vt = u.bind(x=xt, y=yt), v.bind(x=xt, y=yt)
    vr = v.bind(x=xr, y=yr)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + pi * vi - yi * vi,  # -lap u + p = y
            pi.x * qi.x + pi.y * qi.y + ui * qi - xi * qi,  # -lap p + u = x
            u(xl, yl) - xl,  # Dirichlet (left)
            u(xbo, ybo) - xbo,  # Dirichlet (bottom)
            -1.0 * vr,  # Neumann du/dn = 1 (right)
            alpha * ut * vt - alpha * xt * vt,  # Robin du/dn + alpha*u = alpha*x (top)
            p(xb, yb) - yb,  # Dirichlet p = y (whole boundary)
        ]
    )
    cls = fem.classification
    assert "surface@right" in cls and "surface@top" in cls
    assert "dirichlet@left" in cls and "dirichlet@bottom" in cls
    n = int(np.asarray(d.mesh.points).shape[0])
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    c = np.asarray(d.mesh.points)[:, :2]
    assert np.linalg.norm(sol[:n] - c[:, 0]) / np.linalg.norm(c[:, 0]) < 1e-8
    assert np.linalg.norm(sol[n:] - c[:, 1]) / np.linalg.norm(c[:, 1]) < 1e-8
