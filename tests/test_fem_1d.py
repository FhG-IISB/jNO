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
