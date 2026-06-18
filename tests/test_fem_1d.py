"""1D ("segment" / ``LINE2``) FEM coverage through ``jno.fem``.

feax has no 1D volume element, so 1D is assembled by a small native ``LINE2``
assembler (``jno/utils/solver/fem_1d.py``) that reuses the same integrand
evaluator as the 2D/3D path. These tests mirror ``test_fem_3d.py`` on a line
domain (``jno.domain.line`` -> pygmsh): steady (linear + nonlinear, all BCs) and
transient (1D+time). Same matrices-only contract — no solve.

1D linear FEM is *nodally exact* for ``-u'' = f`` (the discrete Green's function
reproduces nodal values), so the linear Dirichlet/Neumann/Robin cases recover to
machine precision; the nonlinear and transient cases use mesh-appropriate tols.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("pygmsh", reason="pygmsh required for line meshing")


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _solve(fem):
    return np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))


def _line(mesh_size=0.05, **kwargs):
    return jno.domain(constructor=jno.domain.line(mesh_size=mesh_size), **kwargs)


def _x(d):
    return np.asarray(d.mesh.points)[:, 0]


# ==========================================================================
# structure
# ==========================================================================
def test_line_domain_is_1d():
    d = _line(0.2)
    assert d.dimension == 1
    assert "line" in d.mesh.cells_dict
    assert {"left", "right", "boundary"} <= set(getattr(d, "_boundary_regions", {}))


def test_vec_gt_1_rejected():
    d = _line(0.3)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    weak = jno.np.inner(jno.np.grad(u, [xi]), jno.np.grad(phi, [xi]), n_contract=2)
    with pytest.raises(NotImplementedError):
        jno.fem([weak, u(xb) - 0.0])


# ==========================================================================
# steady scalar — linear, all BC kinds, recovered exactly
# ==========================================================================
def test_poisson_dirichlet_recovers_linear():
    # -u'' = 0, u(0)=0, u(1)=1 -> u = x (LINE2-exact).
    d = _line(0.1)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x, u(xl) - 0.0, u(xr) - 1.0])
    assert fem.is_linear
    sol = _solve(fem)
    c = _x(d)
    assert np.linalg.norm(sol - c) / np.linalg.norm(c) < 1e-9
    A = _dense(fem.A)
    assert np.allclose(A, A.T, atol=1e-12)  # symmetric (Dirichlet elimination keeps it so)


def test_poisson_dirichlet_bubble_nodally_exact():
    # -u'' = 2, u(0)=u(1)=0 -> u = x(1-x). 1D P1 FEM is nodally exact for -u''=f.
    d = _line(0.05)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x - 2.0 * vi, u(xl) - 0.0, u(xr) - 0.0])
    sol = _solve(fem)
    c = _x(d)
    exact = c * (1 - c)
    assert np.linalg.norm(sol - exact) / np.linalg.norm(exact) < 1e-9


def test_poisson_neumann_recovers_linear():
    # -u'' = 0, u(0)=0, du/dn=1 on the right endpoint -> u = x. Boundary term -g*phi.
    d = _line(0.1)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    fem = jno.fem([ui.x * vi.x, -1.0 * phi.bind(x=xr), u(xl) - 0.0])
    assert "surface@right" in fem.classification
    sol = _solve(fem)
    assert np.linalg.norm(sol - _x(d)) / np.linalg.norm(_x(d)) < 1e-8


def test_poisson_robin_recovers_linear():
    # du/dn + a u = 1 + a on the right endpoint, u(0)=0 -> u = x. The a*u term must
    # land in the matrix (unified boundary path), not just the load.
    a = 2.0
    d = _line(0.1)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    robin = (a * u.bind(x=xr) - (1.0 + a)) * phi.bind(x=xr)
    fem = jno.fem([ui.x * vi.x, robin, u(xl) - 0.0])
    assert "surface@right" in fem.classification
    sol = _solve(fem)
    assert np.linalg.norm(sol - _x(d)) / np.linalg.norm(_x(d)) < 1e-8


# ==========================================================================
# steady nonlinear
# ==========================================================================
def test_nonlinear_reaction_newton_recovers_manufactured():
    spo = pytest.importorskip("scipy.optimize")
    # -u'' + u^3 = f, u_exact = x(1-x), f = 2 + (x(1-x))^3, u(0)=u(1)=0.
    d = _line(0.02)
    u, phi = d.fem_symbols()
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    ui, vi = u.bind(x=xi), phi.bind(x=xi)
    f = 2.0 + (xi * (1 - xi)) ** 3
    fem = jno.fem([ui.x * vi.x + (u * u * u) * vi - f * vi, u(xl) - 0.0, u(xr) - 0.0])
    assert not fem.is_linear
    sol = spo.root(
        lambda v: np.asarray(fem.residual(v)),
        np.zeros(fem.dofs),
        jac=lambda v: _dense(fem.jacobian(v)),
        method="hybr",
    )
    assert sol.success
    c = _x(d)
    exact = c * (1 - c)
    assert np.linalg.norm(sol.x - exact) / np.linalg.norm(exact) < 1e-2


# ==========================================================================
# transient (1D + time)
# ==========================================================================
def test_transient_heat_decays_to_analytic():
    nu = 1.0
    d = _line(0.02, time=(0.0, 0.02, 21))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    u, phi = d.fem_symbols()
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    ic = u(ci[0]) - jno.fn(lambda x: jnp.sin(jnp.pi * x), [ci[0]])
    fem = jno.fem([ui.t * vi + nu * (ui.x * vi.x), u(xb) - 0.0, ic])
    assert fem.is_transient and fem.is_linear

    M, A = _dense(fem.M), _dense(fem.operator.A)
    assert np.allclose(M, M.T) and np.allclose(A, A.T)
    w, dt = np.asarray(fem.state0), float(fem.dt)
    nsteps = round((fem.t1 - fem.t0) / dt)
    for _ in range(nsteps):  # backward Euler
        w = np.linalg.solve(M + dt * A, M @ w)

    c = _x(d)
    analytic = np.exp(-nu * np.pi**2 * fem.t1) * np.sin(np.pi * c)
    assert np.linalg.norm(w - analytic) / np.linalg.norm(analytic) < 1e-2
    assert 0.0 < np.linalg.norm(w) < np.linalg.norm(np.asarray(fem.state0))  # decays


def test_transient_nonlinear_assembles_residual_block():
    # 1D Allen-Cahn-style reaction: u_t*phi + u_x*phi_x + (u^3 - u)*phi.
    d = _line(0.05, time=(0.0, 0.1, 6))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    u, phi = d.fem_symbols()
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    fem = jno.fem([ui.t * vi + (ui.x * vi.x) + (u * u * u - u) * vi, u(xb) - 0.0, u(ci[0]) - 0.0])
    assert fem.is_transient and not fem.is_linear
    block = fem.operator
    assert block.residual is not None and block.jacobian is not None and block.mass is not None
    R0 = np.asarray(block.residual(np.asarray(fem.state0), float(fem.t0), None))
    assert R0.shape == (fem.dofs,)


# ==========================================================================
# coupled / mixed multi-field 1D (native block assembly; no feax)
# ==========================================================================
# feax has no LINE2 element, so coupled 1D is assembled by a native block residual
# (jno/utils/solver/fem_1d.py::assemble_fem_1d_multifield). There is no feax problem
# here (fem.problem is None), so the block layout is hand-computed: field i occupies
# sol[i*n : (i+1)*n] for scalar fields. The manufactured pairs use ASYMMETRIC cross-
# coupling so a transposed/mis-scattered block would change the solution.
def test_coupled_linear_recovers():
    # -u'' + p = 2x ; -p'' + 3u = 3x ; u=x, p=2x on the boundary (asymmetric: the p->u
    # coupling coeff is 1, the u->p coeff is 3). u*=x, p*=2x are LINE2-exact.
    d = _line(0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    fem = jno.fem(
        [
            ui.x * vi.x + 1.0 * pi * vi - 2.0 * xi * vi,
            pi.x * qi.x + 3.0 * ui * qi - 3.0 * xi * qi,
            u(xb) - xb,
            p(xb) - 2.0 * xb,
        ]
    )
    assert fem.is_linear and fem.dofs == 2 * n
    A = _dense(fem.A)
    # off-diagonal blocks present and DIFFERENT (asymmetric coupling -> not transposed)
    assert np.any(np.abs(A[:n, n:]) > 1e-12) and np.any(np.abs(A[n:, :n]) > 1e-12)
    assert not np.allclose(A[:n, n:], A[n:, :n])
    sol = np.linalg.solve(A, np.asarray(fem.b).reshape(-1))
    c = _x(d)
    assert np.linalg.norm(sol[:n] - c) / np.linalg.norm(c) < 1e-9  # u = x
    assert np.linalg.norm(sol[n:] - 2 * c) / np.linalg.norm(2 * c) < 1e-9  # p = 2x


def test_coupled_nonlinear_recovers():
    # Nonlinear coupled: -u'' + u*p = 2x^2 ; -p'' + u^2 = x^2 ; u=x, p=2x. u*=x, p*=2x
    # solve it (the nonlinear terms equal their sources at the solution); Newton recovers.
    spo = pytest.importorskip("scipy.optimize")
    d = _line(0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    fem = jno.fem(
        [
            ui.x * vi.x + (u * p) * vi - 2.0 * (xi * xi) * vi,
            pi.x * qi.x + (u * u) * qi - (xi * xi) * qi,
            u(xb) - xb,
            p(xb) - 2.0 * xb,
        ]
    )
    assert not fem.is_linear and fem.dofs == 2 * n
    sol = spo.root(
        lambda w: np.asarray(fem.residual(w)),
        np.zeros(fem.dofs),
        jac=lambda w: _dense(fem.jacobian(w)),
        method="hybr",
    )
    assert sol.success
    c = _x(d)
    assert np.linalg.norm(sol.x[:n] - c) / np.linalg.norm(c) < 1e-7
    assert np.linalg.norm(sol.x[n:] - 2 * c) / np.linalg.norm(2 * c) < 1e-7


def test_coupled_mixed_bc_recovers():
    # Coupled with mixed BCs: u is Dirichlet at the left and Neumann at the right
    # (du/dn = 1), p is Dirichlet at both ends. -u'' + p = 2x, -p'' + u = x; u*=x, p*=2x.
    d = _line(0.1)
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    xi = d.variable("interior", split=True)[0]
    xl = d.variable("left", split=True)[0]
    xr = d.variable("right", split=True)[0]
    xb = d.variable("boundary", split=True)[0]
    ui, vi = u.bind(x=xi), v.bind(x=xi)
    pi, qi = p.bind(x=xi), q.bind(x=xi)
    vr = v.bind(x=xr)
    fem = jno.fem(
        [
            ui.x * vi.x + pi * vi - 2.0 * xi * vi,
            pi.x * qi.x + ui * qi - xi * qi,
            u(xl) - 0.0,  # Dirichlet (left)
            -1.0 * vr,  # Neumann du/dn = 1 (right)
            p(xb) - 2.0 * xb,  # Dirichlet p = 2x (both ends)
        ]
    )
    assert "surface@right" in fem.classification and "dirichlet@left" in fem.classification
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    c = _x(d)
    assert np.linalg.norm(sol[:n] - c) / np.linalg.norm(c) < 1e-9
    assert np.linalg.norm(sol[n:] - 2 * c) / np.linalg.norm(2 * c) < 1e-9


def test_coupled_transient_decays_to_analytic():
    # Coupled 1D transient, asymmetric: u_t = u'' (heat), p_t = p'' + c*u (driven by u).
    # IC u0 = sin(pi x), p0 = 0; u = e^{-pi^2 t} sin(pi x), p = c*t*e^{-pi^2 t} sin(pi x).
    cc = 4.0
    d = _line(0.02, time=(0.0, 0.05, 51))
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    xb = d.variable("boundary", split=True)[0]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x,
            pi.t * qi + pi.x * qi.x - cc * u.bind(x=xi, t=ti) * qi,
            u(xb) - 0.0,
            p(xb) - 0.0,
            u(ci) - jno.fn(lambda x: jnp.sin(jnp.pi * x), [ci]),
            p(ci) - 0.0,
        ]
    )
    assert fem.is_transient and fem.is_linear and fem.dofs == 2 * n
    M, A = _dense(fem.M), _dense(fem.operator.A)
    assert np.allclose(M[:n, n:], 0.0)  # mass block-diagonal
    w = np.asarray(fem.state0).copy()
    assert np.allclose(w[n:], 0.0)  # p starts at 0
    dt = float(fem.dt)
    for _ in range(round((fem.t1 - fem.t0) / dt)):  # backward Euler
        w = np.linalg.solve(M + dt * A, M @ w)
    cx = _x(d)
    decay = np.exp(-(np.pi**2) * fem.t1)
    u_ex, p_ex = decay * np.sin(np.pi * cx), cc * fem.t1 * decay * np.sin(np.pi * cx)
    assert np.linalg.norm(w[:n] - u_ex) / np.linalg.norm(u_ex) < 1e-2
    assert np.linalg.norm(w[n:] - p_ex) / np.linalg.norm(p_ex) < 2e-2
    assert np.linalg.norm(w[n:]) > 1e-3  # p grew from zero via the coupling


def test_coupled_nonlinear_transient_recovers():
    # The full triple in 1D: nonlinear + coupled + transient. Zero-flux (natural Neumann),
    # spatially-uniform: u_t = -u, p_t = u^2 (the u^2 makes it nonlinear), u(0)=1, p(0)=0
    # -> u = e^{-t}, p = (1 - e^{-2t})/2. Newton backward-Euler recovers it.
    d = _line(0.1, time=(0.0, 0.1, 11))
    n = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(names=("u", "v"))
    p, q = d.fem_symbols(names=("p", "q"))
    co = d.variable("interior", split=True)
    xi, ti = co[0], co[1]
    ci = d.variable("initial", split=True)[0]
    ui, vi = u.bind(x=xi, t=ti), v.bind(x=xi, t=ti)
    pi, qi = p.bind(x=xi, t=ti), q.bind(x=xi, t=ti)
    uu = u.bind(x=xi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui * vi, pi.t * qi + pi.x * qi.x - (uu * uu) * qi, u(ci) - 1.0, p(ci) - 0.0])
    assert fem.is_transient and not fem.is_linear and fem.dofs == 2 * n
    op = fem.operator
    M = _dense(op.mass(0.0, None))
    w = np.asarray(fem.state0).copy()
    dt = float(fem.dt)
    for _ in range(round((fem.t1 - fem.t0) / dt)):  # Newton backward-Euler
        w_old = w.copy()
        for _ in range(30):
            G = M @ (w - w_old) / dt + np.asarray(op.residual(w, 0.0, None))
            if np.linalg.norm(G) < 1e-11:
                break
            w = w - np.linalg.solve(M / dt + _dense(op.jacobian(w, 0.0, None)), G)
    u_ex = np.exp(-fem.t1)
    p_ex = (1.0 - np.exp(-2.0 * fem.t1)) / 2.0
    assert w[:n].std() < 1e-10 and w[n:].std() < 1e-10  # spatially uniform
    assert abs(w[:n].mean() - u_ex) / u_ex < 1e-2
    assert abs(w[n:].mean() - p_ex) / p_ex < 3e-2
