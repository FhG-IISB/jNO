"""Order-of-accuracy verification matrix for ``jno.fem`` (the "hard" suite).

The other FEM test modules check *consistency* (machine-precision recovery of a solution the
element represents exactly) or a single coarse-mesh tolerance. Neither pins the **observed order
of accuracy** -- the thing that actually proves assembly + quadrature are correct. This module
measures it across a matrix of dimensions, problem types, boundary conditions, field kinds, and
physics, against **transcendental** manufactured solutions (nothing is captured exactly, so the
discretization error is genuinely nonzero).

The error is the **true** ``L2`` norm ``||u_h - u||`` -- not a nodal difference (nodal FE values
superconverge and fake the rate) and not an interpolant-difference ``e^T M e`` (which omits the
dominant interpolation-error part and mis-orders the H1 seminorm). It is evaluated exactly from
assembled load vectors via the mass identity

    ||u_h - u||^2 = u_h^T M u_h - 2 u_h . b + C,   M = int phi_i phi_j,
    b_i = int u phi_i,   C = int u^2,

where ``M``, ``b`` (load of the manufactured ``u``), and ``C`` (load of ``u^2``, summed) all come
from the public ``jno.fem`` entry. This gives the genuine ``O(h^{p+1})`` rate (clean ~2.0 for P1,
~3.0 for P2), independent of any nodal superconvergence.

Convergence order between two meshes is the per-interval slope (Roache 2002, "Code Verification by
the Method of Manufactured Solutions"):  ``p = log(e_i / e_{i+1}) / log(h_i / h_{i+1})``.
Theory (smooth solution, simplices, L2 norm): P1 O(h^2), P2 O(h^3); Taylor-Hood P2/P1 -> velocity
O(h^3), pressure O(h^2). The asserts that carry the verification are a band around the theoretical
order plus strictly monotone error decrease.

The cheap scalar/spectral rows gate CI (run under ``-m "not slow"``); the expensive rows (3D, all
nonlinear, transient, vector, complex, multiphysics) are ``@pytest.mark.slow`` -- run deliberately
via ``pytest tests/test_fem_convergence.py`` or ``-m slow``. Running the module as a script
(``python tests/test_fem_convergence.py``) prints the LaTeX table rows from real measured errors;
that output is the single source of truth for the whitepaper's FEM appendix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.optimize as spo  # noqa: E402
from shapely.geometry import box  # noqa: E402

inner, grad, trace, symgrad = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.symgrad
sin, cos, exp = jno.np.sin, jno.np.cos, jno.np.exp
PI = np.pi


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


@dataclass
class Row:
    """One measured convergence study -> one table row."""

    label: str
    dim: str
    kind: str  # linear | nonlinear | transient
    bc: str
    field: str  # scalar | vector | complex | 3-field | spectral
    elem: str  # P1 | P2 | P2/P1
    norm: str  # L2 | vel-L2 | p-L2 | eig | reflection
    hs: list
    errs: list
    expected: float | None = None  # theoretical order (None for non-rate metrics, e.g. PML)
    physics: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def orders(self):
        h, e = np.asarray(self.hs, float), np.asarray(self.errs, float)
        return [float(np.log(e[i] / e[i + 1]) / np.log(h[i] / h[i + 1])) for i in range(len(h) - 1)]

    @property
    def fitted_order(self):
        return float(np.polyfit(np.log(self.hs), np.log(self.errs), 1)[0])


def _assert_row(row, lo, hi, *, monotone=True):
    """Pass/fail gate: fitted order in [lo, hi] and (optionally) strictly decreasing error."""
    if monotone:
        assert np.all(np.diff(np.asarray(row.errs)) < 0), f"{row.label} ({row.norm}) error not decreasing: {row.errs}"
    p = row.fitted_order
    assert lo <= p <= hi, (
        f"{row.label} ({row.norm}) order {p:.2f} outside [{lo}, {hi}] (theory {row.expected}); errs={row.errs}"
    )


def _bind(sym, co, dim):
    """Bind a symbol to the interior coordinate tuple for the given spatial dimension."""
    if dim == 1:
        return sym.bind(x=co[0])
    if dim == 2:
        return sym.bind(x=co[0], y=co[1])
    return sym.bind(x=co[0], y=co[1], z=co[2])


def _true_l2(d, order, uh, gfuns, dim=2, quad=6):
    """True ``||u_h - u||_L2`` via the mass identity (no interpolant -> genuine O(h^{p+1}) rate).

    ``gfuns`` is a callable ``coords -> jno expr`` (scalar field) or a tuple of such callables
    (vector field). ``uh`` is the flat nodal solution (interleaved for a vector field), ordered to
    match ``fem.points`` of a same-(domain, order) solve.
    """
    scalar = callable(gfuns)
    ncomp = 1 if scalar else len(gfuns)
    if scalar:
        u, w = d.fem_symbols(order=order, names=("_eu", "_ev"))
    else:
        u, w = d.fem_symbols(value_shape=(ncomp,), order=order, names=("_eu", "_ev"))
    co = d.variable("interior", split=True)
    ui, vi = _bind(u, co, dim), _bind(w, co, dim)
    if scalar:
        g = gfuns(co)
        femM = jno.fem([ui * vi - g * vi], quad_degree=quad)
        M, b = _dense(femM.A), np.asarray(femM.b).reshape(-1)
        C = float(np.sum(np.asarray(jno.fem([ui * vi - (g * g) * vi], quad_degree=quad).b)))
    else:
        gs = [gf(co) for gf in gfuns]
        gv = jno.np.vector(*gs)
        femM = jno.fem([inner(ui, vi, n_contract=1) - inner(gv, vi, n_contract=1)], quad_degree=quad)
        M, b = _dense(femM.A), np.asarray(femM.b).reshape(-1)
        gmag2 = gs[0] * gs[0]
        for gk in gs[1:]:
            gmag2 = gmag2 + gk * gk
        us, ws = d.fem_symbols(order=order, names=("_cu", "_cv"))  # scalar load for C = int |g|^2
        usi, wsi = _bind(us, co, dim), _bind(ws, co, dim)
        C = float(np.sum(np.asarray(jno.fem([usi * wsi - gmag2 * wsi], quad_degree=quad).b)))
    uh = np.asarray(uh).reshape(-1)
    return float(np.sqrt(max(uh @ M @ uh - 2.0 * uh @ b + C, 0.0)))


def _mass_stiffness(d, order=1):
    """Assembled mass M and stiffness K (for the eigenvalue row and the mass sanity check)."""
    u, w = d.fem_symbols(order=order, names=("_mu", "_mv"))
    xi, yi = d.variable("interior", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
    return _dense(jno.fem([ui * vi]).A), _dense(jno.fem([ui.x * vi.x + ui.y * vi.y]).A)


def _solve_linear(fem):
    return np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))


def _newton(fem, guess=None, tol=1e-11):
    x0 = np.zeros(fem.dofs) if guess is None else np.asarray(guess)
    sol = spo.root(lambda v: np.asarray(fem.residual(v)), x0, jac=lambda v: _dense(fem.jacobian(v)), tol=tol)
    r = float(np.linalg.norm(np.asarray(fem.residual(sol.x))))
    assert r < 1e-7, f"Newton did not converge: |residual|={r:.2e}"
    return sol.x


def _rate(hs, errs):
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


# ============================================================
# Scalar: reaction-diffusion (1D), Poisson (2D/3D), variable-coeff, anisotropic, Robin, periodic
# ============================================================


def study_reaction_1d():
    # -u'' + u = f on [0,1]; u = sin(pi x); f = (pi^2 + 1) sin(pi x). Mixed BC: Dirichlet u(0)=0,
    # Neumann u'(1) = -pi. (The reaction term breaks the 1D nodal exactness of the pure -u'' operator.)
    g1 = lambda co: sin(PI * co[0])  # noqa: E731

    def solve(ms, order):
        d = jno.domain(constructor=jno.domain.line(mesh_size=ms))
        u, w = d.fem_symbols(order=order)
        xi = d.variable("interior", split=True)[0]
        xl, xr = d.variable("left", split=True)[0], d.variable("right", split=True)[0]
        ui, vi = u.bind(x=xi), w.bind(x=xi)
        f = (PI**2 + 1) * sin(PI * xi)
        fem = jno.fem([ui.x * vi.x + ui * vi - f * vi, -(-PI) * w.bind(x=xr), u(xl) - 0.0], quad_degree=4)
        return _true_l2(d, order, _solve_linear(fem), g1, dim=1)

    # P1 only: jno's P2 promotion inserts edge-midpoint nodes for triangles/tets, not for 1D line
    # cells, so an order=2 line element silently stays P1. (P2 is exercised on the 2D Poisson row.)
    sizes = [0.1, 0.05, 0.025]
    errs = [solve(h, 1) for h in sizes]
    return [Row("Reaction-diffusion", "1D", "linear", "mixed (D+N)", "scalar", "P1", "L2", sizes, errs, 2, "reaction")]


def study_poisson_2d():
    # -Delta u = 2 pi^2 sin sin; u = sin(pi x) sin(pi y); Dirichlet (baseline). P1 and P2.
    g = lambda co: sin(PI * co[0]) * sin(PI * co[1])  # noqa: E731

    def solve(ms, order):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, w = d.fem_symbols(order=order)
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        f = 2 * PI**2 * sin(PI * xi) * sin(PI * yi)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=4)
        return _true_l2(d, order, _solve_linear(fem), g)

    rows = []
    for order, sizes, el in ((1, [0.2, 0.1, 0.05], "P1"), (2, [0.25, 0.125, 0.0625], "P2")):
        errs = [solve(h, order) for h in sizes]
        rows.append(Row("Poisson", "2D", "linear", "Dirichlet", "scalar", el, "L2", sizes, errs, order + 1, "elliptic"))
    return rows


def study_poisson_3d():
    # -Delta u = 3 pi^2 sin sin sin on the unit cube; u = sin sin sin; Dirichlet (u = 0 on every face).
    g = lambda co: sin(PI * co[0]) * sin(PI * co[1]) * sin(PI * co[2])  # noqa: E731

    def solve(ms):
        d = jno.domain(constructor=jno.domain.cube(mesh_size=ms))
        u, w = d.fem_symbols()
        xi, yi, zi = d.variable("interior", split=True)[:3]
        ui, vi = u.bind(x=xi, y=yi, z=zi), w.bind(x=xi, y=yi, z=zi)
        f = 3 * PI**2 * sin(PI * xi) * sin(PI * yi) * sin(PI * zi)
        cb = d.variable("boundary", split=True)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - f * vi, u(cb[0], cb[1], cb[2]) - 0.0])
        return _true_l2(d, 1, _solve_linear(fem), g, dim=3, quad=6)

    # 3D convergence on unstructured tets is noisy/pre-asymptotic at affordable resolutions: the
    # per-interval order swings (~1.2-2.1) but the overall slope (~1.6) trends to 2 and the error
    # decreases monotonically. Four meshes (smooth ~2x dof growth: 234 -> 1866) fit through the noise.
    sizes = [0.22, 0.16, 0.115, 0.085]
    errs = [solve(h) for h in sizes]
    return [Row("Poisson", "3D", "linear", "Dirichlet", "scalar", "P1", "L2", sizes, errs, 2, "elliptic")]


def study_variable_coeff():
    # -div((1+x+y) grad u) = f; u = sin sin.  f = -pi(c1 s2 + s1 c2) + 2 pi^2 (1+x+y) s1 s2.
    g = lambda co: sin(PI * co[0]) * sin(PI * co[1])  # noqa: E731

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, w = d.fem_symbols()
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        s1, s2, c1, c2 = sin(PI * xi), sin(PI * yi), cos(PI * xi), cos(PI * yi)
        kappa = 1.0 + xi + yi
        f = -PI * (c1 * s2 + s1 * c2) + 2 * PI**2 * kappa * s1 * s2
        fem = jno.fem([kappa * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=4)
        return _true_l2(d, 1, _solve_linear(fem), g)

    sizes = [0.2, 0.1, 0.05]
    errs = [solve(h) for h in sizes]
    return [Row("Variable-coeff diffusion", "2D", "linear", "Dirichlet", "scalar", "P1", "L2", sizes, errs, 2, "diffusion")]


def study_anisotropic():
    # -div(K grad u) = f, K = [[2,1],[1,3]]; u = sin sin.  f = pi^2 (5 s1 s2 - 2 c1 c2).
    # Tensor Neumann flux (K grad u).n on +x/+y; Dirichlet on left/bottom.
    g = lambda co: sin(PI * co[0]) * sin(PI * co[1])  # noqa: E731

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, w = d.fem_symbols()
        xi, yi = d.variable("interior", split=True)[:2]
        xl, yl = d.variable("left", split=True)[:2]
        xbo, ybo = d.variable("bottom", split=True)[:2]
        xr, yr = d.variable("right", split=True)[:2]
        xt, yt = d.variable("top", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        s1, s2, c1, c2 = sin(PI * xi), sin(PI * yi), cos(PI * xi), cos(PI * yi)
        f = PI**2 * (5 * s1 * s2 - 2 * c1 * c2)
        weak = (2 * ui.x + ui.y) * vi.x + (ui.x + 3 * ui.y) * vi.y - f * vi
        gr = -2 * PI * sin(PI * yr)  # (K grad u).n on +x = 2 u_x + u_y
        gt = -3 * PI * sin(PI * xt)  # (K grad u).n on +y = u_x + 3 u_y
        fem = jno.fem(
            [weak, -gr * w.bind(x=xr, y=yr), -gt * w.bind(x=xt, y=yt), u(xl, yl) - 0.0, u(xbo, ybo) - 0.0],
            quad_degree=4,
        )
        return _true_l2(d, 1, _solve_linear(fem), g)

    sizes = [0.2, 0.1, 0.05]
    errs = [solve(h) for h in sizes]
    return [
        Row("Anisotropic diffusion", "2D", "linear", "Neumann", "scalar", "P1", "L2", sizes, errs, 2, "tensor diffusion")
    ]


def study_robin():
    # -Delta u + sigma u = f, u = sin sin.  Robin du/dn + a u = g on right & top; Dirichlet on left & bottom.
    sigma, a_r, a_t = 4.0, 2.0, 3.0
    g = lambda co: sin(PI * co[0]) * sin(PI * co[1])  # noqa: E731

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, w = d.fem_symbols()
        xi, yi = d.variable("interior", split=True)[:2]
        xl, yl = d.variable("left", split=True)[:2]
        xbo, ybo = d.variable("bottom", split=True)[:2]
        xr, yr = d.variable("right", split=True)[:2]
        xt, yt = d.variable("top", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        f = (2 * PI**2 + sigma) * sin(PI * xi) * sin(PI * yi)
        gR, gT = -PI * sin(PI * yr), -PI * sin(PI * xt)  # du/dn (= -pi sin) + a*0
        fem = jno.fem(
            [
                ui.x * vi.x + ui.y * vi.y + sigma * ui * vi - f * vi,
                (a_r * u.bind(x=xr, y=yr) - gR) * w.bind(x=xr, y=yr),
                (a_t * u.bind(x=xt, y=yt) - gT) * w.bind(x=xt, y=yt),
                u(xl, yl) - 0.0,
                u(xbo, ybo) - 0.0,
            ],
            quad_degree=4,
        )
        return _true_l2(d, 1, _solve_linear(fem), g)

    sizes = [0.2, 0.1, 0.05]
    errs = [solve(h) for h in sizes]
    return [Row("Reaction-diffusion", "2D", "linear", "Robin", "scalar", "P1", "L2", sizes, errs, 2, "reaction")]


def study_periodic():
    # -Delta u = f, periodic in x, Dirichlet in y.  u = (cos 2pi x + 0.5 sin 2pi x) sin(pi y).
    g = lambda co: (cos(2 * PI * co[0]) + 0.5 * sin(2 * PI * co[0])) * sin(PI * co[1])  # noqa: E731

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        d.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
        d.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
        d.tag("bot", lambda x, y: y < 1e-6)
        d.tag("top", lambda x, y: y > 1 - 1e-6)
        u, w = d.fem_symbols()
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("bot", split=True)[:2]
        xt, yt = d.variable("top", split=True)[:2]
        xl, yl = d.variable("left", split=True)[:2]
        xr, yr = d.variable("right", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        hh = cos(2 * PI * xi) + 0.5 * sin(2 * PI * xi)
        f = 5 * PI**2 * hh * sin(PI * yi)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0, u(xt, yt) - 0.0, u(xl, yl) - u(xr, yr)])
        return _true_l2(d, 1, np.asarray(fem.solve()), g)

    sizes = [0.16, 0.08, 0.04]
    errs = [solve(h) for h in sizes]
    return [Row("Poisson", "2D", "linear", "periodic", "scalar", "P1", "L2", sizes, errs, 2, "elliptic")]


# ============================================================
# Nonlinear scalar: Bratu (exponential).  Transient: heat (time-linear MMS).
# ============================================================


def study_bratu():
    # -Delta u - lam e^u = g,  g = 2 pi^2 sin sin - lam e^{sin sin} (coordinate forcing); u = sin sin.
    # Mixed Neumann (+x/+y) / Dirichlet. lam=1: cold-start Newton finds the manufactured branch.
    lam = 1.0
    gex = lambda co: sin(PI * co[0]) * sin(PI * co[1])  # noqa: E731

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, w = d.fem_symbols()
        xi, yi = d.variable("interior", split=True)[:2]
        xl, yl = d.variable("left", split=True)[:2]
        xbo, ybo = d.variable("bottom", split=True)[:2]
        xr, yr = d.variable("right", split=True)[:2]
        xt, yt = d.variable("top", split=True)[:2]
        ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        sc = sin(PI * xi) * sin(PI * yi)
        src = 2 * PI**2 * sc - lam * exp(sc)  # coordinate forcing
        weak = ui.x * vi.x + ui.y * vi.y - lam * exp(ui) * vi - src * vi  # exp(u) -> nonlinear
        gr, gt = -PI * sin(PI * yr), -PI * sin(PI * xt)
        fem = jno.fem([weak, -gr * w.bind(x=xr, y=yr), -gt * w.bind(x=xt, y=yt), u(xl, yl) - 0.0, u(xbo, ybo) - 0.0])
        assert not fem.is_linear
        pts = np.asarray(fem.points)
        guess = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])  # manufactured branch
        return _true_l2(d, 1, _newton(fem, guess), gex)

    sizes = [0.2, 0.1, 0.05]
    errs = [solve(h) for h in sizes]
    return [Row("Bratu", "2D", "nonlinear", "Neumann+Dirichlet", "scalar", "P1", "L2", sizes, errs, 2, "exp nonlinearity")]


def _march(fem):
    M, A = _dense(fem.M), _dense(fem.operator.A)
    c = np.asarray(fem.operator.affine_bias).reshape(-1)
    f = fem.operator.forcing_vector_fn
    w = np.asarray(fem.state0).copy()
    dt, t = float(fem.dt), float(fem.t0)
    for _ in range(round((fem.t1 - fem.t0) / dt)):
        t += dt
        rhs = M @ w + dt * c
        if f is not None:
            rhs = rhs + dt * np.asarray(f(t)).reshape(-1)
        w = np.linalg.solve(M + dt * A, rhs)
    return w


def study_transient_heat():
    # u_t = alpha lap u + s, time-linear MMS u = (1+t) sin sin -> backward Euler is temporally exact,
    # so refining h gives a clean SPATIAL O(h^2).  s = sin sin (1 + 2 alpha pi^2 (1+t)).
    AL = 1.0

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms, time=(0.0, 0.5, 11))
        u, w = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb = d.variable("boundary", split=True)[:2]
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
        s = sin(PI * xi) * sin(PI * yi) * (1.0 + 2 * AL * PI**2 * (1.0 + ti))
        weak = ui.t * vi + AL * (ui.x * vi.x + ui.y * vi.y) - s * vi
        icf = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [ci[0], ci[1]])
        fem = jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - icf])
        w_final = _march(fem)
        T = float(fem.t1)
        d_space = jno.domain(box(0, 0, 1, 1), mesh_size=ms)  # steady twin for the spatial L2 norm
        return _true_l2(d_space, 1, w_final, lambda co: (1.0 + T) * sin(PI * co[0]) * sin(PI * co[1]))

    sizes = [0.2, 0.1, 0.05]
    errs = [solve(h) for h in sizes]
    return [Row("Heat", "2D", "transient", "Dirichlet", "scalar", "P1", "L2", sizes, errs, 2, "parabolic")]


# ============================================================
# Vector: linear elasticity (traction), nonlinear Ginzburg-Landau
# ============================================================


def study_elasticity():
    # lambda-mu elasticity, u = (s, 0.5 s), s = sin sin.  f hand-derived (FD-verified). Traction
    # t = sigma.n on +x, Dirichlet (u = 0) on the other three edges.
    LL, MM = 1.0, 1.0
    gvec = (lambda co: sin(PI * co[0]) * sin(PI * co[1]), lambda co: 0.5 * sin(PI * co[0]) * sin(PI * co[1]))

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, w = d.fem_symbols(value_shape=(2,))
        xi, yi = d.variable("interior", split=True)[:2]
        xl, yl = d.variable("left", split=True)[:2]
        xbo, ybo = d.variable("bottom", split=True)[:2]
        xt, yt = d.variable("top", split=True)[:2]
        xr, yr = d.variable("right", split=True)[:2]
        eu, ev = symgrad(u, [xi, yi]), symgrad(w, [xi, yi])
        vv = w.bind(x=xi, y=yi)
        S1, S2, C1, C2 = sin(PI * xi), sin(PI * yi), cos(PI * xi), cos(PI * yi)
        f1 = PI**2 * ((LL + 3 * MM) * S1 * S2 - 0.5 * (LL + MM) * C1 * C2)
        f2 = PI**2 * ((0.5 * LL + 1.5 * MM) * S1 * S2 - (LL + MM) * C1 * C2)
        weak = LL * trace(eu) * trace(ev) + 2 * MM * inner(eu, ev, n_contract=2) - (f1 * vv[0] + f2 * vv[1])
        tr1, tr2 = -(LL + 2 * MM) * PI * sin(PI * yr), -0.5 * MM * PI * sin(PI * yr)  # sigma.n on +x
        trac = -(tr1 * w.bind(x=xr, y=yr)[0] + tr2 * w.bind(x=xr, y=yr)[1])
        fem = jno.fem([weak, trac, u(xl, yl) - (0.0, 0.0), u(xbo, ybo) - (0.0, 0.0), u(xt, yt) - (0.0, 0.0)])
        return _true_l2(d, 1, _solve_linear(fem), gvec)

    sizes = [0.2, 0.1, 0.05]
    errs = [solve(h) for h in sizes]
    return [Row("Linear elasticity", "2D", "linear", "traction+Dirichlet", "vector", "P1", "L2", sizes, errs, 2, "solid")]


def study_ginzburg_landau():
    # Vector Ginzburg-Landau -Delta u + (|u|^2 - 1) u = f, u = (sin sin, sin2pix sin piy). Dirichlet.
    gvec = (
        lambda co: sin(PI * co[0]) * sin(PI * co[1]),
        lambda co: sin(2 * PI * co[0]) * sin(PI * co[1]),
    )

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        ub, vv = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        U1, U2 = sin(PI * xi) * sin(PI * yi), sin(2 * PI * xi) * sin(PI * yi)
        m2 = U1 * U1 + U2 * U2
        f1 = 2 * PI**2 * U1 + (m2 - 1.0) * U1
        f2 = 5 * PI**2 * U2 + (m2 - 1.0) * U2
        react = (inner(ub, ub, n_contract=1) - 1.0) * inner(ub, vv, n_contract=1)
        weak = inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=2) + react - (f1 * vv[0] + f2 * vv[1])
        fem = jno.fem([weak, u(xb, yb) - (0.0, 0.0)])
        assert not fem.is_linear
        return _true_l2(d, 1, _newton(fem), gvec)

    sizes = [0.18, 0.1, 0.06]
    errs = [solve(h) for h in sizes]
    return [Row("Ginzburg-Landau", "2D", "nonlinear", "Dirichlet", "vector", "P1", "L2", sizes, errs, 2, "phase-field")]


# ============================================================
# Fluid (Kovasznay), multiphysics (Boussinesq), complex (Helmholtz)
# ============================================================


def study_kovasznay():
    # Closed-form steady Navier-Stokes (Kovasznay 1948), Taylor-Hood P2/P1.  velocity L2 O(h^3),
    # pressure L2 O(h^2).  Cold-start Newton from rest.
    nu = 0.025
    Re = 1.0 / nu
    lam = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4.0 * PI**2)
    uex = lambda co: 1.0 - exp(lam * co[0]) * cos(2 * PI * co[1])  # noqa: E731
    vex = lambda co: lam / (2 * PI) * exp(lam * co[0]) * sin(2 * PI * co[1])  # noqa: E731
    pex = lambda co: 0.5 * (1.0 - exp(2 * lam * co[0]))  # noqa: E731

    def solve(ms):
        d = jno.domain(box(-0.5, -0.5, 1.0, 1.5), mesh_size=ms)
        x0, y0 = -0.5, -0.5
        d.point_region("ppin", (x0, y0))
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
        p, q = d.fem_symbols(names=("p", "q"), order=1)
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        xpn, ypn = d.variable("ppin", split=True)[:2]
        ub = u.bind(x=xi, y=yi)
        gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
        pp, qq, vv = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        conv = inner(gu, ub, n_contract=1)
        momentum = inner(conv, vv, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * trace(gv)
        bx = 1.0 - exp(lam * xb) * cos(2 * PI * yb)
        by = lam / (2 * PI) * exp(lam * xb) * sin(2 * PI * yb)
        fem = jno.fem(
            [
                momentum,
                -qq * trace(gu),
                u(xb, yb)[0] - bx,
                u(xb, yb)[1] - by,
                p(xpn, ypn) - float(0.5 * (1 - np.exp(2 * lam * x0))),
            ]
        )
        assert not fem.is_linear and not fem.is_transient
        sol = _newton(fem)
        off = fem.offsets  # [0, n_vel, n_total]
        l2v = _true_l2(d, 2, sol[off[0] : off[1]], (uex, vex))  # same domain -> matching P2 node order
        l2p = _true_l2(d, 1, sol[off[1] :], pex)
        return l2v, l2p

    sizes = [0.14, 0.095, 0.065]  # finer: Taylor-Hood pressure is pre-asymptotic on coarse meshes
    res = [solve(h) for h in sizes]
    v = [r[0] for r in res]
    p = [r[1] for r in res]
    return [
        Row("Kovasznay flow", "2D", "nonlinear", "Dirichlet", "vector", "P2/P1", "vel-L2", sizes, v, 3, "fluid"),
        Row("Kovasznay flow", "2D", "nonlinear", "Dirichlet", "pressure", "P2/P1", "p-L2", sizes, p, 2, "fluid"),
    ]


def study_complex_helmholtz():
    # -Delta u - c u = f, c = 1 + 0.5i, manufactured u_r = sin sin, u_i = sin2pix sin piy (Re != Im).
    # complex=True -> coupled real system; L2 on |e| = sqrt(|e_r|^2 + |e_i|^2).
    cr, ci = 1.0, 0.5
    c = cr + 1j * ci

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        u, wt = d.fem_symbols(complex=True)
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        ub, wb = u.bind(x=xi, y=yi), wt.bind(x=xi, y=yi)
        URi = sin(PI * xi) * sin(PI * yi)
        UIi = sin(2 * PI * xi) * sin(PI * yi)
        fr = 2 * PI**2 * URi - (cr * URi - ci * UIi)
        fi = 5 * PI**2 * UIi - (cr * UIi + ci * URi)
        f = jno.complex(fr, fi)
        weak = (ub.x * wb.x + ub.y * wb.y) - c * (ub * wb) - f * wb
        fem = jno.fem([weak.real, u.real(xb, yb) - 0.0, u.imag(xb, yb) - 0.0])
        assert fem._mode == "linear"
        sol = _solve_linear(fem)
        n = int(fem.offsets[1])  # [0, n_re, n_total] -- the real/imag field split (native, no feax problem)
        er = _true_l2(d, 1, sol[:n], lambda co: sin(PI * co[0]) * sin(PI * co[1]))
        ei = _true_l2(d, 1, sol[n:], lambda co: sin(2 * PI * co[0]) * sin(PI * co[1]))
        return float(np.hypot(er, ei))

    sizes = [0.2, 0.1, 0.05]
    errs = [solve(h) for h in sizes]
    return [Row("Helmholtz (complex)", "2D", "linear", "Dirichlet", "complex", "P1", "L2", sizes, errs, 2, "wave")]


def study_boussinesq():
    # Steady Boussinesq (Rayleigh-Benard), 3 coupled fields P2/P1/P2. Manufactured div-free u,
    # pressure, temperature (sin/cos), analytic forcing -> per-field L2 convergence.
    Pr, Ra = 1.0, 100.0
    uex = lambda co: PI * sin(PI * co[0]) * cos(PI * co[1])  # noqa: E731
    vex = lambda co: -PI * cos(PI * co[0]) * sin(PI * co[1])  # noqa: E731
    Tex = lambda co: sin(PI * co[0]) * cos(PI * co[1])  # noqa: E731

    def solve(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        d.point_region("ppin", (0.0, 0.0))
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
        p, q = d.fem_symbols(names=("p", "q"), order=1)
        T, sT = d.fem_symbols(names=("T", "sT"), order=2)
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        xpn, ypn = d.variable("ppin", split=True)[:2]
        ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        pb, qb = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
        Tb, sb = T.bind(x=xi, y=yi), sT.bind(x=xi, y=yi)
        S1, C1, S2, C2 = sin(PI * xi), cos(PI * xi), sin(PI * yi), cos(PI * yi)
        fmx = PI**3 * S1 * C1 - PI * S1 * C2 + 2 * PI**3 * Pr * S1 * C2
        fmy = PI**3 * S2 * C2 - PI * C1 * S2 - 2 * PI**3 * Pr * C1 * S2 - Ra * Pr * S1 * C2
        fe = PI**2 * S1 * C1 + 2 * PI**2 * S1 * C2
        ux, uy = ub[0], ub[1]
        uxx, uxy, uyx, uyy = ub.x[0], ub.y[0], ub.x[1], ub.y[1]
        vxx, vxy, vyx, vyy = vb.x[0], vb.y[0], vb.x[1], vb.y[1]
        mom = (
            ((ux * uxx + uy * uxy) * vb[0] + (ux * uyx + uy * uyy) * vb[1])
            + Pr * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)
            - pb * (vxx + vyy)
            - Ra * Pr * Tb * vb[1]
            - (fmx * vb[0] + fmy * vb[1])
        )
        cont = qb * (uxx + uyy)
        ener = (ux * Tb.x + uy * Tb.y) * sb + (Tb.x * sb.x + Tb.y * sb.y) - fe * sb
        UXb, UYb = PI * sin(PI * xb) * cos(PI * yb), -PI * cos(PI * xb) * sin(PI * yb)
        Tmb = sin(PI * xb) * cos(PI * yb)
        fem = jno.fem([mom, cont, ener, u(xb, yb)[0] - UXb, u(xb, yb)[1] - UYb, T(xb, yb) - Tmb, p(xpn, ypn) - 1.0])
        assert not fem.is_linear
        w = _newton(fem)
        off = fem.offsets  # [0, n_vel, n_vel+n_p, n_total]
        l2v = _true_l2(d, 2, w[off[0] : off[1]], (uex, vex))  # same domain -> matching P2 node order
        l2T = _true_l2(d, 2, w[off[2] : off[3]], Tex)
        return l2v, l2T

    sizes = [0.12, 0.08, 0.055]
    res = [solve(h) for h in sizes]
    return [
        Row(
            "Boussinesq (velocity)",
            "2D",
            "nonlinear",
            "Dirichlet",
            "3-field",
            "P2/P1",
            "vel-L2",
            sizes,
            [r[0] for r in res],
            3,
            "multiphysics",
        ),
        Row(
            "Boussinesq (temperature)",
            "2D",
            "nonlinear",
            "Dirichlet",
            "3-field",
            "P2/P1",
            "T-L2",
            sizes,
            [r[1] for r in res],
            3,
            "multiphysics",
        ),
    ]


# ============================================================
# Spectral (eigenvalues) and PML (reflection metric, not a rate)
# ============================================================


def study_eigenvalues():
    # -Delta phi = lambda phi, Dirichlet; lambda = pi^2 (m^2 + n^2). Generalized eig K x = lambda M x.
    def eigs(ms, k=5):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        M, K = _mass_stiffness(d)
        u, w = d.fem_symbols(names=("_pu", "_pv"))
        xi, yi = d.variable("interior", split=True)[:2]
        pts = np.asarray(jno.fem([u.bind(x=xi, y=yi) * w.bind(x=xi, y=yi)]).points)
        eps = 1e-9
        inside = (pts[:, 0] > eps) & (pts[:, 0] < 1 - eps) & (pts[:, 1] > eps) & (pts[:, 1] < 1 - eps)
        wv = sla.eigh(K[np.ix_(inside, inside)], M[np.ix_(inside, inside)], eigvals_only=True)
        return np.sort(wv)[:k]

    sizes = [0.15, 0.1, 0.07]
    spectra = [eigs(h) for h in sizes]
    err1 = [abs(s[0] - 2 * PI**2) for s in spectra]
    return [
        Row(
            "Dirichlet-Laplacian eigenvalues",
            "2D",
            "linear",
            "Dirichlet",
            "spectral",
            "P1",
            "eig",
            sizes,
            err1,
            2,
            "spectral",
            extra={"finest": spectra[-1].tolist(), "exact": (np.array([2, 5, 5, 8, 10]) * PI**2).tolist()},
        )
    ]


def study_pml():
    # Helmholtz + PML (complex coordinate stretch s = 1 + i sigma/k). No analytic solution: verified
    # by *reflection-freedom* -- a converged PML's physical-core field is insensitive to the absorber
    # strength sigma0 (a poor / absent PML reflects and changes with sigma0). Reported as a metric.
    Lb, wpml, k = 1.0, 0.25, 25.0
    relu = lambda z: jno.np.maximum(z, 0.0)  # noqa: E731

    def solve_pml(sigma0, ms=0.025):
        d = jno.domain(box(0.0, 0.0, Lb, Lb), mesh_size=ms)
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        sx = sigma0 * (relu(wpml - xi) ** 2 + relu(xi - (Lb - wpml)) ** 2) / wpml**2
        sy = sigma0 * (relu(wpml - yi) ** 2 + relu(yi - (Lb - wpml)) ** 2) / wpml**2
        Sx, Sy = 1.0 + 1j * sx / k, 1.0 + 1j * sy / k
        src = exp(-(((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / (2 * 0.025**2)))
        weak = (Sy / Sx) * (ui.x * vi.x) + (Sx / Sy) * (ui.y * vi.y) - k**2 * Sx * Sy * (u * vi) - src * vi
        fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)
        return fem, np.asarray(fem.solve())

    fem, u40 = solve_pml(40.0)
    _, u60 = solve_pml(60.0)  # 1.5x absorber: physical core must be unchanged
    _, u0 = solve_pml(0.0)  # no PML: the wave reflects off the walls
    pts = np.asarray(fem.points)
    core = (pts[:, 0] > wpml) & (pts[:, 0] < Lb - wpml) & (pts[:, 1] > wpml) & (pts[:, 1] < Lb - wpml)
    insens = float(np.linalg.norm(u40[core] - u60[core]) / np.linalg.norm(u40[core]))
    sens_off = float(np.linalg.norm(u40[core] - u0[core]) / np.linalg.norm(u40[core]))
    return [
        Row(
            "Helmholtz + PML",
            "2D",
            "linear",
            "PML",
            "complex",
            "P1",
            "reflection",
            [0.025],
            [insens],
            None,
            "wave",
            extra={"insensitivity": insens, "no_pml_sensitivity": sens_off},
        )
    ]


# ============================================================
# Per-row pass/fail gates
# ============================================================


def test_reaction_1d():
    _assert_row(study_reaction_1d()[0], 1.7, 2.4)  # P1


def test_poisson_2d():
    rows = study_poisson_2d()
    d0 = jno.domain(box(0, 0, 1, 1), mesh_size=0.2)
    M, _ = _mass_stiffness(d0)
    assert np.isclose(M.sum(), 1.0, atol=1e-6), "P1 mass matrix does not integrate to the domain area"
    _assert_row(rows[0], 1.7, 2.4)
    _assert_row(rows[1], 2.6, 3.4)


def test_variable_coeff():
    _assert_row(study_variable_coeff()[0], 1.7, 2.4)


def test_anisotropic():
    _assert_row(study_anisotropic()[0], 1.7, 2.4)


def test_robin():
    _assert_row(study_robin()[0], 1.7, 2.4)


def test_periodic():
    _assert_row(study_periodic()[0], 1.7, 2.4)


def test_eigenvalues():
    row = study_eigenvalues()[0]
    finest, exact = np.array(row.extra["finest"]), np.array(row.extra["exact"])
    assert np.all(finest >= exact - 1e-6), f"eigenvalues below the analytic spectrum: {finest} vs {exact}"
    assert np.all(np.abs(finest - exact) / exact < 0.04), f"finest eigenvalues off >4%: {finest} vs {exact}"
    assert abs(finest[1] - finest[2]) / finest[1] < 0.02, f"5 pi^2 pair not degenerate: {finest[1]}, {finest[2]}"
    _assert_row(row, 1.6, 2.5)


@pytest.mark.slow
def test_poisson_3d():
    _assert_row(study_poisson_3d()[0], 1.3, 2.4)  # noisy on coarse unstructured tets (overall ~1.6)


@pytest.mark.slow
def test_bratu():
    _assert_row(study_bratu()[0], 1.6, 2.5)


@pytest.mark.slow
def test_transient_heat():
    _assert_row(study_transient_heat()[0], 1.6, 2.6)


@pytest.mark.slow
def test_elasticity():
    _assert_row(study_elasticity()[0], 1.6, 2.5)


@pytest.mark.slow
def test_ginzburg_landau():
    _assert_row(study_ginzburg_landau()[0], 1.6, 2.6)


@pytest.mark.slow
def test_kovasznay():
    rows = study_kovasznay()
    _assert_row(rows[0], 2.3, 3.6)  # velocity L2 (O(h^3), pre-asymptotic high)
    _assert_row(rows[1], 1.5, 2.6)  # pressure L2 (O(h^2))


@pytest.mark.slow
def test_complex_helmholtz():
    _assert_row(study_complex_helmholtz()[0], 1.6, 2.5)


@pytest.mark.slow
def test_boussinesq():
    rows = study_boussinesq()
    _assert_row(rows[0], 2.2, 3.6)  # velocity
    _assert_row(rows[1], 2.2, 3.6)  # temperature


@pytest.mark.slow
def test_pml():
    row = study_pml()[0]
    assert row.extra["insensitivity"] < 1e-2, f"PML not reflection-free: {row.extra['insensitivity']:.3e}"
    # without the PML the core reflects -> strongly sensitive: confirms the absorber does real work
    assert row.extra["no_pml_sensitivity"] > 0.1, f"no-PML core should reflect: {row.extra['no_pml_sensitivity']:.3e}"


# ============================================================
# Table generator -- run as a script to emit LaTeX rows from real measured errors
# ============================================================

_ALL_STUDIES = [
    study_reaction_1d,
    study_poisson_2d,
    study_poisson_3d,
    study_variable_coeff,
    study_anisotropic,
    study_robin,
    study_periodic,
    study_bratu,
    study_transient_heat,
    study_elasticity,
    study_ginzburg_landau,
    study_kovasznay,
    study_complex_helmholtz,
    study_boussinesq,
    study_eigenvalues,
    study_pml,
]


def _emit_table():  # pragma: no cover - manual table regeneration
    jax.config.update("jax_enable_x64", True)
    print("\n% --- jno.fem convergence matrix (auto-generated; do not hand-edit numbers) ---")
    for study in _ALL_STUDIES:
        for r in study():
            o = " & ".join(f"{p:.2f}" for p in r.orders) if r.expected is not None else "--"
            errs = " & ".join(f"{e:.2e}" for e in r.errs)
            exp = "--" if r.expected is None else f"{r.expected}"
            print(f"{r.label} & {r.dim} & {r.kind} & {r.bc} & {r.field} & {r.elem} & {r.norm} & {errs} & {o} & {exp} \\\\")


if __name__ == "__main__":  # pragma: no cover
    _emit_table()
