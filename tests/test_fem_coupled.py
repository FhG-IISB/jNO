"""Coupled / mixed multi-field FEM through ``jno.fem`` (Phase 2).

Each field is its own ``(trial, test)`` pair from a ``fem_symbols()`` call (they
share a ``field_key``); ``jno.fem`` detects several fields and assembles a block
(multi-variable) system via feax. Cross-field weak terms populate the off-diagonal
blocks, and feax autodiffs the universal kernel into the full block matrix. The
single-field path is unchanged (one field → existing assembly).
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


def test_coupled_p1_recovers_manufactured_with_offdiagonal():
    # Two coupled scalar fields on the unit square:
    #   -lap u + p = f1 ,  -lap p + u = f2 ,  u = p = 0 on the boundary.
    # Manufactured u* = g, p* = 2g with g = x(1-x)y(1-y); f1 = -lap(u*) + p*,
    # f2 = -lap(p*) + u*. The cross terms (+p in u's eq, +u in p's eq) make the
    # off-diagonal blocks non-zero, so recovery genuinely exercises coupling.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    g = xi * (1 - xi) * yi * (1 - yi)
    lg = 2 * (xi * (1 - xi) + yi * (1 - yi))  # -lap(g)
    f1 = lg + 2 * g
    f2 = 2 * lg + g
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + p * vi - f1 * vi,  # u-equation (test v), couples to p
            pi.x * qi.x + pi.y * qi.y + u * qi - f2 * qi,  # p-equation (test q), couples to u
            u(xb, yb) - 0.0,
            p(xb, yb) - 0.0,
        ]
    )
    assert fem.is_linear
    assert fem.dofs == 2 * n  # block system over the two P1 fields

    A = _dense(fem.A)
    assert np.allclose(A, A.T, atol=1e-9)
    # the off-diagonal block (u-rows x p-cols) must be populated -> real coupling,
    # not two independent problems that happen to recover.
    assert np.any(np.abs(A[:n, n:]) > 1e-12)
    assert np.any(np.abs(A[n:, :n]) > 1e-12)

    sol = np.linalg.solve(A, np.asarray(fem.b).reshape(-1))
    uu, pp = sol[:n], sol[n:]
    c = np.asarray(d.mesh.points)[:, :2]
    gg = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    assert np.linalg.norm(uu - gg) / np.linalg.norm(gg) < 1e-2
    assert np.linalg.norm(pp - 2 * gg) / np.linalg.norm(2 * gg) < 1e-2


def test_taylor_hood_stokes_recovers_manufactured():
    # Taylor-Hood Stokes (inf-sup stable): P2 velocity, P1 pressure on the same
    # triangulation (the pressure mesh is the velocity P2 mesh's vertex block).
    # Manufactured u = (x, -y), p = x, body force f = (1, 0); div u = 0. All live in
    # the P2/P1 spaces, so the discrete solution is exact. Velocity is unique;
    # pressure is determined only up to a constant (pure-Dirichlet velocity -> pressure
    # null space). A single-node pin (domain.point_region) fixes the pressure at one
    # vertex, removing the null space so the saddle system is solved DIRECTLY (no
    # lstsq/zero-mean) and recovers BOTH fields exactly. The pin sits at (0, 0) where
    # p* = x = 0, so pinning to 0 is consistent and the recovered p equals p* exactly.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    nv = int(np.asarray(d.mesh.points).shape[0])
    d.point_region("ppin", (0.0, 0.0))  # pin the pressure at a single vertex
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
    p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    gu, gv = jno.np.grad(u, [xi, yi]), jno.np.grad(v, [xi, yi])
    pp, qq, vv = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    weak_mom = jno.np.inner(gu, gv, n_contract=2) - pp * jno.np.trace(gv) - 1.0 * vv[0]  # f = (1, 0)
    weak_cont = -qq * jno.np.trace(gu)
    fem = jno.fem(
        [
            weak_mom,
            weak_cont,
            u(xb, yb)[0] - xb,
            u(xb, yb)[1] - (-1.0 * yb),
            p(xpn, ypn) - 0.0,  # single-node pressure pin
        ]
    )
    assert "dirichlet@ppin" in fem.classification

    prob = fem.problem
    off = prob.offset
    assert len(off) == 2  # two coupled fields
    assert (off[1] - off[0]) > 2 * nv  # velocity carries edge dofs -> genuinely P2

    # pinned -> non-singular -> direct solve recovers both fields to machine precision
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    pts_v = np.asarray(prob.mesh[0].points)
    pts_p = np.asarray(prob.mesh[1].points)
    uu = sol[off[0] : off[1]].reshape(-1, 2)
    ppres = sol[off[1] :]
    u_ex = np.stack([pts_v[:, 0], -pts_v[:, 1]], axis=-1)
    p_ex = pts_p[:, 0]
    assert np.linalg.norm(uu - u_ex) / np.linalg.norm(u_ex) < 1e-9
    assert np.linalg.norm(ppres - p_ex) / np.linalg.norm(p_ex) < 1e-9  # exact, not up to mean


def _coupled_robin_fem(with_robin=True, order_u=1, mesh_size=0.1):
    """Coupled manufactured problem with a Neumann + Robin condition on one field.

    Two coupled scalar fields on the unit square, manufactured u* = x, p* = y (both
    linear, so exact in P1 *and* P2). Cross terms (+p in u's eq, +u in p's eq) populate
    both off-diagonal blocks. On u's right edge the Robin condition du/dn + alpha*u = g_R
    contributes a trial-dependent ``alpha*u*v`` surface term (-> stiffness block A) and a
    pure ``-g_R*v`` load (-> b); u is Dirichlet (=x) on the other three edges, p is
    Dirichlet (=y) on the whole boundary. ``order_u`` promotes u to P2 (mixed-order vs
    the P1 pressure) to exercise feax's per-field face-shape concat across orders."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, v = d.fem_symbols(names=("u", "v"), order=order_u)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xbo, ybo, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    ur, vr = u.bind(x=xr, y=yr), v.bind(x=xr, y=yr)
    alpha = 2.0
    g_robin = 1.0 + alpha  # du*/dn + alpha*u* on the right edge (du*/dn = 1, u* = 1 there)
    cons = [
        ui.x * vi.x + ui.y * vi.y + pi * vi - yi * vi,  # u-eq: grad u.grad v + p v - f_u v (f_u = p* = y)
        -g_robin * vr,  # Neumann part of the Robin condition -> load b
        pi.x * qi.x + pi.y * qi.y + ui * qi - xi * qi,  # p-eq: grad p.grad q + u q - f_p v (f_p = u* = x)
        u(xl, yl) - xl,  # u = x on left / bottom / top (right edge is Robin)
        u(xbo, ybo) - xbo,
        u(xt, yt) - xt,
        p(xb, yb) - yb,  # p = y on the whole boundary
    ]
    if with_robin:
        cons.insert(1, alpha * ur * vr)  # Robin (trial-dependent) surface term -> stiffness A
    return d, jno.fem(cons)


def _check_up_recovery(fem, tol=1e-9):
    """Direct-solve a coupled (u*, p*) = (x, y) system and assert per-field recovery."""
    prob = fem.problem
    off = prob.offset
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    uu, pp = sol[off[0] : off[1]], sol[off[1] :]
    x_u = np.asarray(prob.mesh[0].points)[:, 0]  # u nodes (P2 carries edge nodes)
    y_p = np.asarray(prob.mesh[1].points)[:, 1]  # p nodes (P1 vertices)
    assert np.linalg.norm(uu - x_u) / np.linalg.norm(x_u) < tol
    assert np.linalg.norm(pp - y_p) / np.linalg.norm(y_p) < tol


def test_coupled_neumann_robin_equal_order_recovers_manufactured():
    # Equal-order (P1/P1) coupled problem with a Neumann + Robin condition on one field.
    # Isolates the multi-field SURFACE kernel from any mixed-order concern: exact recovery
    # requires the Robin term in A (block stiffness) and the Neumann term in b (load) to
    # both be correct, and the cross terms to populate the off-diagonal blocks.
    d, fem = _coupled_robin_fem(with_robin=True, order_u=1, mesh_size=0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    assert fem.is_linear
    assert fem.dofs == 2 * n
    assert "surface@right" in fem.classification
    A = _dense(fem.A)
    assert np.any(np.abs(A[:n, n:]) > 1e-12)  # off-diagonal coupling, both blocks
    assert np.any(np.abs(A[n:, :n]) > 1e-12)
    _check_up_recovery(fem)


def test_coupled_robin_surface_term_enters_stiffness_block():
    # The trial-dependent (Robin) surface term alpha*u*v must land in the stiffness block
    # A, not only the load b -- that is the part a constant Neumann flux never exercises.
    # Dropping it changes A; its presence is what makes the recovery above well-posed.
    _, fem_robin = _coupled_robin_fem(with_robin=True, order_u=1, mesh_size=0.2)
    _, fem_neumann = _coupled_robin_fem(with_robin=False, order_u=1, mesh_size=0.2)
    A_r, A_n = _dense(fem_robin.A), _dense(fem_neumann.A)
    assert not np.allclose(A_r, A_n)  # the Robin surface term modifies the matrix
    assert float(np.sum(np.abs(A_r - A_n))) > 1e-9


def test_coupled_neumann_robin_mixed_order_recovers_manufactured():
    # Escalation: same coupled Neumann/Robin problem but u is P2 and p is P1, so the two
    # fields have different face-node counts on the shared right edge. The multi-field
    # surface kernel must slice the per-field face shape data feax concatenates by order;
    # if that mixed-order concat were wrong, recovery fails here while the equal-order
    # surface test still passes -- localizing the bug to feax rather than the kernel.
    d, fem = _coupled_robin_fem(with_robin=True, order_u=2, mesh_size=0.15)
    assert fem.is_linear
    assert "surface@right" in fem.classification
    off = fem.problem.offset
    assert (off[1] - off[0]) > int(np.asarray(d.mesh.points).shape[0])  # u carries P2 edge dofs
    _check_up_recovery(fem)


def test_coupled_nonlinear_newton_recovers_manufactured():
    # Nonlinear coupled: -lap u + u*p = f1 ; -lap p + u^3 = f2 ; u=p=0 on boundary.
    # Manufactured u*=g, p*=2g (g=x(1-x)y(1-y)). feax autodiffs the block residual/
    # Jacobian on the multi-field problem, so a scipy Newton solve recovers both.
    spo = pytest.importorskip("scipy.optimize")
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pi, qi = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    g = xi * (1 - xi) * yi * (1 - yi)
    lg = 2 * (xi * (1 - xi) + yi * (1 - yi))
    f1 = lg + g * (2 * g)  # -lap(u*) + u* p*
    f2 = 2 * lg + g**3  # -lap(p*) + u*^3
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + (u * p) * vi - f1 * vi,  # nonlinear u*p coupling
            pi.x * qi.x + pi.y * qi.y + (u * u * u) * qi - f2 * qi,  # nonlinear u^3
            u(xb, yb) - 0.0,
            p(xb, yb) - 0.0,
        ]
    )
    assert not fem.is_linear
    assert fem.dofs == 2 * n
    sol = spo.root(
        lambda w: np.asarray(fem.residual(w)),
        np.zeros(fem.dofs),
        jac=lambda w: _dense(fem.jacobian(w)),
        method="hybr",
    )
    assert sol.success
    c = np.asarray(d.mesh.points)[:, :2]
    gg = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    assert np.linalg.norm(sol.x[:n] - gg) / np.linalg.norm(gg) < 1e-2
    assert np.linalg.norm(sol.x[n:] - 2 * gg) / np.linalg.norm(2 * gg) < 1e-2
