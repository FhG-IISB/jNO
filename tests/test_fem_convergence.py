"""Order-of-accuracy verification for ``jno.fem`` (the "hard" suite).

The other FEM test modules check one of two things: *consistency* -- machine-precision
recovery of a solution the element represents exactly (affine for P1, quadratics for P2)
-- or a single coarse-mesh tolerance. Neither pins the thing that actually proves the
assembly + quadrature are correct: the **observed order of accuracy** under mesh
refinement. A subtle assembly bug (a missing Jacobian factor, a wrong quadrature weight,
a transposed gradient) sails through a single-mesh tolerance but wrecks a convergence rate.

This module measures that rate. For each problem we manufacture a *transcendental* exact
solution (so nothing is captured exactly -- the discretization error is genuinely nonzero),
solve on a sequence of meshes, and fit the log-log slope of the error vs. ``h``. The error
is a **true integrated norm**, not a nodal difference: nodal FE values superconverge, so a
discrete nodal "L2" can show a rate that is right by luck or wrong in a way that mimics a
bug. We integrate with the FE mass / stiffness operators assembled through the *same public
``jno.fem`` entry* used everywhere else:

    ||e||_L2^2 = e^T M e ,   |e|_H1^2 = e^T K e ,   M = int phi_i phi_j , K = int grad.grad

where ``e`` is the nodal coefficient vector of ``u_h - I(u_exact)`` (``I`` the nodal
interpolant). ``M`` is a genuine mass matrix -- ``M.sum() == |domain|`` -- so ``e^T M e`` is
the exact L2 norm of the interpolated error field and does not superconverge.

Method of Manufactured Solutions: Roache, "Code Verification by the Method of Manufactured
Solutions", J. Fluids Eng. 124(1), 2002. Theory orders (smooth solution, simplices):
P1 -> L2 O(h^2), H1 O(h); P2 -> L2 O(h^3), H1 O(h^2); Taylor-Hood P2/P1 -> velocity H1
O(h^2), pressure L2 O(h^2). Brackets below are deliberately generous (the coarse meshes a
fast test can afford sit above the asymptotic regime); the assertions that carry the
verification are the *lower* bounds (the method achieves its design order, not rate ~0) plus
strictly monotone error decrease under refinement.

The five scalar/spectral tests are cheap enough to **gate CI** (they run under
``pytest -m "not slow"``). Only the Kovasznay Navier-Stokes test is marked
``@pytest.mark.slow`` -- its cold-start Newton builds a dense Jacobian (a few thousand DOFs),
which is both the slowest case and a memory risk on a small GPU; run it deliberately via
``pytest tests/test_fem_convergence.py`` (or ``-m slow``).
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.optimize as spo  # noqa: E402
from shapely.geometry import box  # noqa: E402

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace


@pytest.fixture(autouse=True)
def _x64():
    """feax assembly is float64, so these tests opt into x64 per-test. The session default is
    x64-off (see tests/conftest.py); save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ============================================================
# Convergence harness
# ============================================================


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _observed_order(hs, errs):
    """Least-squares slope of log(err) vs log(h) -> the observed convergence rate."""
    return float(np.polyfit(np.log(np.asarray(hs)), np.log(np.asarray(errs)), 1)[0])


def _scalar_mass_stiffness(d, order):
    """Raw Galerkin mass M = int phi_i phi_j and stiffness K = int grad phi_i . grad phi_j on
    ``d``'s ``order`` space, no BCs -- assembled through the public ``jno.fem`` entry so the DOF
    numbering matches a same-order solve's ``fem.points``."""
    u, phi = d.fem_symbols(order=order, names=("_mu", "_mv"))
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    M = _dense(jno.fem([ui * vi]).A)
    K = _dense(jno.fem([ui.x * vi.x + ui.y * vi.y]).A)
    return M, K


def _study(builder, sizes, order):
    """Run ``builder(mesh_size) -> (domain, u_h, u_exact_fn)`` over ``sizes`` and return
    ``(hs, l2_errs, h1_errs)`` using the integrated norms on each mesh."""
    hs, l2s, h1s = [], [], []
    for ms in sizes:
        d, u_h, u_exact_fn, pts = builder(ms)
        M, K = _scalar_mass_stiffness(d, order)
        assert M.shape[0] == u_h.shape[0] == pts.shape[0], "mass matrix / solution DOF mismatch"
        e = np.asarray(u_h) - u_exact_fn(pts[:, 0], pts[:, 1])
        hs.append(ms)
        l2s.append(float(np.sqrt(e @ M @ e)))
        h1s.append(float(np.sqrt(e @ K @ e)))
    return hs, l2s, h1s


def _assert_decreasing(errs, what):
    errs = np.asarray(errs)
    assert np.all(np.diff(errs) < 0), f"{what} did not decrease monotonically under refinement: {errs}"


# ============================================================
# 1. Poisson, transcendental MMS:  -lap u = f,  u = sin(pi x) sin(pi y)
# ============================================================

_PI = np.pi


def _poisson_exact(x, y):
    return np.sin(_PI * x) * np.sin(_PI * y)


def _build_poisson(order):
    def builder(mesh_size):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
        u, phi = d.fem_symbols(order=order)
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        f = 2.0 * _PI**2 * jno.np.sin(_PI * xi) * jno.np.sin(_PI * yi)  # -lap(sin sin) = 2 pi^2 sin sin
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=4)
        u_h = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
        return d, u_h, _poisson_exact, np.asarray(fem.points)

    return builder


def test_poisson_mms_p1_convergence_orders():
    # P1: theory L2 O(h^2), H1 O(h). The mass matrix sanity check (M.sum == area) confirms the
    # integrated norm is a real L2 norm, not a nodal RMS that would superconverge.
    d0 = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    M, _ = _scalar_mass_stiffness(d0, 1)
    assert np.isclose(M.sum(), 1.0, atol=1e-6), "P1 mass matrix does not integrate to the domain area"

    hs, l2, h1 = _study(_build_poisson(1), [0.2, 0.1, 0.05], order=1)
    _assert_decreasing(l2, "P1 L2 error")
    _assert_decreasing(h1, "P1 H1 error")
    o_l2, o_h1 = _observed_order(hs, l2), _observed_order(hs, h1)
    assert 1.6 <= o_l2 <= 2.8, f"P1 L2 order {o_l2:.2f} outside [1.6, 2.8] (theory 2); errs={l2}"
    assert 0.8 <= o_h1 <= 1.8, f"P1 H1 order {o_h1:.2f} outside [0.8, 1.8] (theory 1); errs={h1}"


def test_poisson_mms_p2_convergence_orders():
    # P2: theory L2 O(h^3), H1 O(h^2). Asserting L2 order > 2.5 proves super-quadratic accuracy,
    # which P1 *cannot* reach -> the P2 space is genuinely built and delivering, not a P1 fallback.
    hs, l2, h1 = _study(_build_poisson(2), [0.25, 0.125, 0.0625], order=2)
    _assert_decreasing(l2, "P2 L2 error")
    _assert_decreasing(h1, "P2 H1 error")
    o_l2, o_h1 = _observed_order(hs, l2), _observed_order(hs, h1)
    assert 2.5 <= o_l2 <= 3.8, f"P2 L2 order {o_l2:.2f} outside [2.5, 3.8] (theory 3); errs={l2}"
    assert 1.7 <= o_h1 <= 2.9, f"P2 H1 order {o_h1:.2f} outside [1.7, 2.9] (theory 2); errs={h1}"


# ============================================================
# 2. Variable-coefficient diffusion:  -div(kappa(x) grad u) = f,  kappa = 1 + x + y
# ============================================================


def _build_variable_coeff():
    # u = sin(pi x) sin(pi y), kappa = 1 + x + y.  f = -div(kappa grad u)
    #   = -pi (cos(pi x) sin(pi y) + sin(pi x) cos(pi y)) + 2 pi^2 (1+x+y) sin(pi x) sin(pi y)
    # (the first group is grad(kappa).grad(u); the second is -kappa lap u). Verified numerically.
    def builder(mesh_size):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        s1, s2 = jno.np.sin(_PI * xi), jno.np.sin(_PI * yi)
        c1, c2 = jno.np.cos(_PI * xi), jno.np.cos(_PI * yi)
        kappa = 1.0 + xi + yi
        f = -_PI * (c1 * s2 + s1 * c2) + 2.0 * _PI**2 * kappa * s1 * s2
        fem = jno.fem([kappa * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=4)
        u_h = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
        return d, u_h, _poisson_exact, np.asarray(fem.points)

    return builder


def test_variable_coefficient_diffusion_convergence():
    # A spatially varying scalar coefficient adds the grad(kappa).grad(u) chain-rule term to the
    # forcing -- the part a constant-coefficient Poisson never exercises. P1 -> L2 O(h^2), H1 O(h).
    hs, l2, h1 = _study(_build_variable_coeff(), [0.2, 0.1, 0.05], order=1)
    _assert_decreasing(l2, "variable-coeff L2 error")
    o_l2, o_h1 = _observed_order(hs, l2), _observed_order(hs, h1)
    assert 1.6 <= o_l2 <= 2.8, f"variable-coeff L2 order {o_l2:.2f} outside [1.6, 2.8]; errs={l2}"
    assert 0.8 <= o_h1 <= 1.8, f"variable-coeff H1 order {o_h1:.2f} outside [0.8, 1.8]; errs={h1}"


# ============================================================
# 3. Anisotropic diffusion:  -div(K grad u) = f,  K = [[2,1],[1,3]] (constant SPD tensor)
# ============================================================


def _build_anisotropic():
    # Constant anisotropic tensor K = [[2,1],[1,3]]; the off-diagonal couples d/dx and d/dy.
    # u = sin(pi x) sin(pi y).  f = -div(K grad u) = pi^2 (5 sin sin - 2 cos cos). Verified numerically.
    def builder(mesh_size):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        s1, s2 = jno.np.sin(_PI * xi), jno.np.sin(_PI * yi)
        c1, c2 = jno.np.cos(_PI * xi), jno.np.cos(_PI * yi)
        f = _PI**2 * (5.0 * s1 * s2 - 2.0 * c1 * c2)
        # weak form: int (K grad u) . grad v.  (K grad u) = (2 u_x + u_y, u_x + 3 u_y)
        weak = (2.0 * ui.x + ui.y) * vi.x + (ui.x + 3.0 * ui.y) * vi.y - f * vi
        fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=4)
        u_h = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
        return d, u_h, _poisson_exact, np.asarray(fem.points)

    return builder


def test_anisotropic_diffusion_convergence():
    # The cross term (u_y v_x + u_x v_y) populates the stiffness in a way an isotropic Laplacian
    # never does; a wrong tensor contraction would change the solution, not just its accuracy.
    hs, l2, h1 = _study(_build_anisotropic(), [0.2, 0.1, 0.05], order=1)
    _assert_decreasing(l2, "anisotropic L2 error")
    o_l2, o_h1 = _observed_order(hs, l2), _observed_order(hs, h1)
    assert 1.6 <= o_l2 <= 2.8, f"anisotropic L2 order {o_l2:.2f} outside [1.6, 2.8]; errs={l2}"
    assert 0.8 <= o_h1 <= 1.8, f"anisotropic H1 order {o_h1:.2f} outside [0.8, 1.8]; errs={h1}"


# ============================================================
# 4. Kovasznay flow: closed-form steady Navier-Stokes, Taylor-Hood P2/P1
# ============================================================


def _kovasznay_fields(nu):
    """Kovasznay (1948), 'Laminar flow behind a two-dimensional grid', Proc. Camb. Phil. Soc.
    44(1). Exact steady-NS solution (zero body force) at Re = 1/nu."""
    Re = 1.0 / nu
    lam = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4.0 * _PI**2)
    ue = lambda x, y: 1.0 - np.exp(lam * x) * np.cos(2.0 * _PI * y)  # noqa: E731
    ve = lambda x, y: lam / (2.0 * _PI) * np.exp(lam * x) * np.sin(2.0 * _PI * y)  # noqa: E731
    pe = lambda x, y: 0.5 * (1.0 - np.exp(2.0 * lam * x))  # noqa: E731
    return lam, ue, ve, pe


def _solve_kovasznay(mesh_size, nu=0.025):
    lam, ue, ve, pe = _kovasznay_fields(nu)
    d = jno.domain(box(-0.5, -0.5, 1.0, 1.5), mesh_size=mesh_size)
    x0, y0 = -0.5, -0.5  # pin pressure at a corner vertex (exact p known there)
    d.point_region("ppin", (x0, y0))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
    p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure (inf-sup-stable pair)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    ub = u.bind(x=xi, y=yi)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq, vv = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    conv = inner(gu, ub, n_contract=1)  # (u.grad)u -- the convective nonlinearity -> Newton
    momentum = inner(conv, vv, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * trace(gv)
    bx = 1.0 - jno.np.exp(lam * xb) * jno.np.cos(2.0 * _PI * yb)
    by = lam / (2.0 * _PI) * jno.np.exp(lam * xb) * jno.np.sin(2.0 * _PI * yb)
    fem = jno.fem(
        [
            momentum,
            -qq * trace(gu),
            u(xb, yb)[0] - bx,
            u(xb, yb)[1] - by,
            p(xpn, ypn) - float(pe(x0, y0)),
        ]
    )
    assert not fem.is_linear and not fem.is_transient, "Kovasznay must be a steady nonlinear system"

    # cold-start Newton from rest (no interpolant cheat) -> converges to the discrete solution u_h
    res = spo.root(
        lambda w: np.asarray(fem.residual(w)),
        np.zeros(fem.dofs),
        jac=lambda w: _dense(fem.jacobian(w)),
        method="hybr",
        tol=1e-11,
    )
    resid = float(np.linalg.norm(np.asarray(fem.residual(res.x))))
    assert resid < 1e-7, f"Newton did not converge at h={mesh_size}: |residual|={resid:.2e}"

    prob = fem.problem
    off = prob.offset
    pts_v = np.asarray(prob.mesh[0].points)
    pts_p = np.asarray(prob.mesh[1].points)
    uu = res.x[off[0] : off[1]].reshape(-1, 2)
    ppres = res.x[off[1] :]
    u_ex = np.stack([ue(pts_v[:, 0], pts_v[:, 1]), ve(pts_v[:, 0], pts_v[:, 1])], axis=-1)
    p_ex = pe(pts_p[:, 0], pts_p[:, 1])

    # velocity H1 seminorm (vector P2) and pressure L2 (P1) via assembled operators
    u2, v2 = d.fem_symbols(value_shape=(2,), names=("eu", "ev"), order=2)
    Kv = _dense(jno.fem([inner(grad(u2, [xi, yi]), grad(v2, [xi, yi]), n_contract=2)]).A)
    p2, q2 = d.fem_symbols(names=("ep", "eq"), order=1)
    Mp = _dense(jno.fem([p2.bind(x=xi, y=yi) * q2.bind(x=xi, y=yi)]).A)
    ev = (uu - u_ex).reshape(-1)
    ep = ppres - p_ex
    return float(np.sqrt(ev @ Kv @ ev)), float(np.sqrt(ep @ Mp @ ep))


@pytest.mark.slow  # cold-start Newton on a dense few-thousand-DOF Jacobian: slowest case + GPU-memory risk
def test_kovasznay_navier_stokes_convergence():
    # Canonical closed-form Navier-Stokes verification: exercises the convective nonlinearity
    # (autodiff Jacobian + Newton) AND inf-sup stability of the Taylor-Hood pair simultaneously.
    # Theory: velocity H1 O(h^2), pressure L2 O(h^2).
    sizes = [0.2, 0.12, 0.08]
    h1v, l2p = [], []
    for ms in sizes:
        a, b = _solve_kovasznay(ms)
        h1v.append(a)
        l2p.append(b)
    _assert_decreasing(h1v, "Kovasznay velocity H1 error")
    _assert_decreasing(l2p, "Kovasznay pressure L2 error")
    o_v, o_p = _observed_order(sizes, h1v), _observed_order(sizes, l2p)
    assert 1.7 <= o_v <= 3.4, f"Kovasznay velocity H1 order {o_v:.2f} outside [1.7, 3.4] (theory 2); errs={h1v}"
    assert 1.5 <= o_p <= 2.7, f"Kovasznay pressure L2 order {o_p:.2f} outside [1.5, 2.7] (theory 2); errs={l2p}"


# ============================================================
# 5. Laplacian eigenvalues:  -lap phi = lambda phi,  phi = 0 on the boundary
# ============================================================


def _laplacian_eigs(mesh_size, k=5):
    """Smallest ``k`` Dirichlet-Laplacian eigenvalues on the unit square via the generalized
    eigenproblem K_II x = lambda M_II x restricted to interior DOFs."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem_m = jno.fem([ui * vi])
    M = _dense(fem_m.A)
    K = _dense(jno.fem([ui.x * vi.x + ui.y * vi.y]).A)
    pts = np.asarray(fem_m.points)
    eps = 1e-9  # interior = strictly inside the unit square (all boundary nodes lie on an edge)
    inside = (pts[:, 0] > eps) & (pts[:, 0] < 1 - eps) & (pts[:, 1] > eps) & (pts[:, 1] < 1 - eps)
    w = sla.eigh(K[np.ix_(inside, inside)], M[np.ix_(inside, inside)], eigvals_only=True)
    return np.sort(w)[:k]


def test_laplacian_eigenvalues_spectral():
    # A spectral check that forcing-based tests cannot make: it verifies the mass AND stiffness
    # matrices *together* against the analytic spectrum lambda = pi^2 (m^2 + n^2). The double
    # eigenvalue 5 pi^2 (modes (1,2) and (2,1)) must appear as a degenerate pair.
    exact = np.array([2, 5, 5, 8, 10]) * _PI**2  # (1,1),(1,2),(2,1),(2,2),(1,3)/(3,1)
    sizes = [0.15, 0.1, 0.07]
    spectra = [_laplacian_eigs(ms) for ms in sizes]

    finest = spectra[-1]
    # discrete Galerkin eigenvalues bound the true ones from above (min-max principle)
    assert np.all(finest >= exact - 1e-6), f"eigenvalues fell below the analytic spectrum: {finest} vs {exact}"
    rel = np.abs(finest - exact) / exact
    assert np.all(rel < 0.04), f"finest-mesh eigenvalues off by >4%: {finest} vs {exact} (rel={rel})"
    # the degenerate pair is resolved as (near-)equal
    assert abs(finest[1] - finest[2]) / finest[1] < 0.02, f"5 pi^2 pair not degenerate: {finest[1]}, {finest[2]}"

    # first eigenvalue converges to 2 pi^2 at O(h^2)
    err1 = [abs(s[0] - 2 * _PI**2) for s in spectra]
    _assert_decreasing(err1, "first-eigenvalue error")
    o1 = _observed_order(sizes, err1)
    assert 1.6 <= o1 <= 2.5, f"first-eigenvalue convergence order {o1:.2f} outside [1.6, 2.5] (theory 2); errs={err1}"
