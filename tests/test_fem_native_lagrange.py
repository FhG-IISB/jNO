"""Validation of the native 2D Lagrange assembler (:func:`assemble_fem_native`).

The assembler is exercised here *directly* (it is not yet routed through
``jno.fem``); the per-constraint classification that ``jno.fem`` performs is
replicated by :func:`_classify` so the assembler receives exactly the
``(volume_terms, boundary_terms, dirichlet_raw, ic_residuals)`` contract it
expects.

Two oracles are used:

* **Matrix-level vs feax** (machine precision).  feax is the known-good engine;
  for P1 single- and vector-fields the native global DOF numbering matches
  feax's, so the assembled ``A``/``b`` can be compared entry-for-entry.  This
  catches push-forward, scatter and DOF-map bugs that a convergence-rate check
  can hide on symmetric problems.
* **Analytic solution + convergence** for P2.  The native P2 element numbers its
  edge nodes in basix's element-DOF order (required: the gradients come from
  basix tabulation), which differs from feax's edge numbering by a benign
  permutation — so the raw matrices are not comparable, but the solution at the
  native nodes is, and must match the manufactured field and converge at O(h^3).
"""

from __future__ import annotations

import numpy as np
import pytest

basix = pytest.importorskip("basix", reason="basix required for element tabulation")
pytest.importorskip("feax", reason="feax required as the assembly oracle")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno._fem import (  # noqa: E402
    _bare,
    _contains,
    _dirichlet_spec,
    _field_key_of,
    _region_and_support,
    _retag_coords_for_quadrature,
)
from jno.trace import TestFunction, TrialFunction  # noqa: E402
from jno.utils.solver.fem_native import _get_mesh, assemble_fem_native  # noqa: E402

inner, grad, trace, symgrad = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.symgrad
sin, exp = jno.np.sin, jno.np.exp
PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    """The native assembler is compared to feax's float64 matrices, so these tests
    opt into x64 per-test (the session default is x64-off; save/restore keeps the
    flag from leaking to other modules)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dense(M):
    return np.asarray(M.todense() if hasattr(M, "todense") else M)


def _classify(domain, constraints):
    """Replicate ``jno.fem``'s constraint classification (the steady subset).

    Returns ``(volume_terms, boundary_terms, dirichlet_raw, ic_residuals)`` — the
    exact inputs ``jno.fem`` feeds to its element assembler.  Mutates the
    constraint coordinate tags in place (``_retag_coords_for_quadrature``), so the
    caller must build a *fresh* constraint list for every domain it classifies.
    """
    volume_terms: list = []
    boundary_terms: dict = {}
    dirichlet_raw: list = []
    ic_residuals: list = []
    for c in constraints:
        if _contains(c, TestFunction):
            support, region = _region_and_support(c, domain)
            _retag_coords_for_quadrature(c, support, region)
            bare = _bare(c)
            if support == "volume":
                volume_terms.append(bare)
            else:
                boundary_terms.setdefault(region, []).append(bare)
        elif _contains(c, TrialFunction):
            _, region = _region_and_support(c, domain)
            comp, value, value_node = _dirichlet_spec(_bare(c))
            dirichlet_raw.append((_field_key_of(c), region, comp, value, value_node))
        else:  # pragma: no cover - defensive
            raise AssertionError("constraint has neither test nor trial function")
    return volume_terms, boundary_terms, dirichlet_raw, ic_residuals


def _native_linear(build, *, mesh_size, vec, order, quad):
    """Assemble ``(A, b)`` for a steady-linear problem through the native path."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    vt, bt, dr, ic = _classify(d, build(d))
    op, mode, _offs = assemble_fem_native(d, vt, bt, dr, ic, vec=vec, quad_degree=max(quad, 2 * order))
    assert mode == "linear"
    A, b = op
    return d, _dense(A), np.asarray(b)


def _native_op(build, *, mesh_size, vec, order, quad):
    """Assemble the residual/Jacobian operator for a steady-nonlinear problem."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    vt, bt, dr, ic = _classify(d, build(d))
    op, mode, _offs = assemble_fem_native(d, vt, bt, dr, ic, vec=vec, quad_degree=max(quad, 2 * order))
    assert mode == "nonlinear"
    return d, op


def _feax_linear(build, *, mesh_size, quad):
    """Assemble ``(A, b)`` for the same problem through feax (the oracle)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    fem = jno.fem(build(d), quad_degree=quad)
    assert fem.is_linear
    return _dense(fem.A), np.asarray(fem.b)


# ---------------------------------------------------------------------------
# P1 matrix-level oracle (native global numbering == feax's)
# ---------------------------------------------------------------------------


def test_p1_scalar_poisson_matches_feax():
    """-Δu = f, u = sin(πx)sin(πy), homogeneous Dirichlet."""

    def build(d):
        u, w = d.fem_symbols(names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        vv = w.bind(x=xi, y=yi)
        f = 2 * PI**2 * sin(PI * xi) * sin(PI * yi)
        return [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - f * vv, u(xb, yb) - 0.0]

    _d, A_n, b_n = _native_linear(build, mesh_size=0.2, vec=1, order=1, quad=2)
    A_f, b_f = _feax_linear(build, mesh_size=0.2, quad=2)
    assert A_n.shape == A_f.shape
    assert np.abs(A_n - A_f).max() < 1e-11
    assert np.abs(b_n - b_f).max() < 1e-11


def test_p1_vector_elasticity_matches_feax():
    """Linear elasticity (vec=2): λ tr(ε(u)) tr(ε(v)) + 2μ ε(u):ε(v) = f·v."""

    def build(d):
        u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        eu, ev = symgrad(u, [xi, yi]), symgrad(w, [xi, yi])
        vv = w.bind(x=xi, y=yi)
        lam, mu = 1.2, 0.8
        weak = lam * trace(eu) * trace(ev) + 2 * mu * inner(eu, ev, n_contract=2) - (1.0 * vv[0] + 0.5 * vv[1])
        return [weak, u(xb, yb) - (0.0, 0.0)]

    _d, A_n, b_n = _native_linear(build, mesh_size=0.25, vec=2, order=1, quad=2)
    A_f, b_f = _feax_linear(build, mesh_size=0.25, quad=2)
    assert A_n.shape == A_f.shape
    assert np.abs(A_n - A_f).max() < 1e-11
    assert np.abs(b_n - b_f).max() < 1e-11


def test_p1_neumann_plus_dirichlet_matches_feax():
    """Mixed BC: Dirichlet on the whole boundary set is replaced by a Neumann
    (natural) flux term on the 'right' edge plus Dirichlet elsewhere — exercises the
    surface integral path alongside the volume assembly."""

    def build(d):
        u, w = d.fem_symbols(names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        vv = w.bind(x=xi, y=yi)
        f = 2 * PI**2 * sin(PI * xi) * sin(PI * yi)
        # Neumann load on the right edge (a boundary test term)
        xr, yr = d.variable("right", split=True)[:2]
        wr = w.bind(x=xr, y=yr)
        g = sin(PI * yr)
        xb, yb = d.variable("left", split=True)[:2]
        return [
            inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - f * vv,
            g * wr,  # Neumann flux on 'right'
            u(xb, yb) - 0.0,  # Dirichlet on 'left'
        ]

    _d, A_n, b_n = _native_linear(build, mesh_size=0.25, vec=1, order=1, quad=2)
    A_f, b_f = _feax_linear(build, mesh_size=0.25, quad=2)
    assert A_n.shape == A_f.shape
    assert np.abs(A_n - A_f).max() < 1e-11
    assert np.abs(b_n - b_f).max() < 1e-11


def _newton(residual, jacobian, n, *, iters=30, tol=1e-12):
    """Damped-free Newton for the small dense steady-nonlinear systems here."""
    u = jnp.zeros(n)
    for _ in range(iters):
        r = np.asarray(residual(u)).reshape(-1)
        du = np.linalg.solve(_dense(jacobian(u)), r)
        u = u - jnp.asarray(du)
        if np.abs(du).max() < tol:
            break
    return np.asarray(u)


def test_p1_nonlinear_bratu_matches_feax():
    """Bratu -Δu - λ e^u = 0 (nonlinear).

    The residual matches feax entry-for-entry at a probe state.  The raw Jacobian
    differs only on Dirichlet *columns* (native uses row-replacement elimination,
    feax symmetric elimination) — a convention that leaves the Newton step
    unchanged — so correctness is asserted by driving both operators through the
    *same* Newton iteration and comparing the converged solutions.
    """
    lam = 1.5

    def build(d):
        u, w = d.fem_symbols(names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        ub, vv = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
        weak = inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - lam * exp(ub) * vv
        return [weak, u(xb, yb) - 0.0]

    d, op = _native_op(build, mesh_size=0.25, vec=1, order=1, quad=2)
    df = jno.domain(box(0, 0, 1, 1), mesh_size=0.25)
    fem = jno.fem(build(df), quad_degree=2)
    assert not fem.is_linear

    n = int(np.asarray(d.mesh.points).shape[0])  # P1: one DOF per vertex
    u_probe = jnp.asarray(0.3 * np.sin(np.arange(n, dtype=float)))  # reproducible probe state
    r_n = np.asarray(op.residual(u_probe)).reshape(-1)
    r_f = np.asarray(fem.residual(u_probe)).reshape(-1)
    assert np.abs(r_n - r_f).max() < 1e-10

    u_native = _newton(op.residual, op.jacobian, n)
    u_feax = _newton(fem.residual, fem.jacobian, n)
    assert np.abs(np.asarray(op.residual(jnp.asarray(u_native))).reshape(-1)).max() < 1e-9
    assert np.abs(u_native - u_feax).max() < 1e-10


def test_p1_coupled_two_field_matches_feax():
    """Coupled reaction-diffusion (two scalar P1 fields a, b with cross terms) —
    exercises the multi-field block assembly; native and feax both order the fields
    by first appearance with node-major DOFs, so the block matrix matches."""

    def build(d):
        a, p = d.fem_symbols(names=("a", "p"))
        b, q = d.fem_symbols(names=("b", "q"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        pa, qb = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
        ab, bb = a.bind(x=xi, y=yi), b.bind(x=xi, y=yi)
        eq_a = inner(grad(a, [xi, yi]), grad(p, [xi, yi]), n_contract=1) + bb * pa - 1.0 * pa
        eq_b = inner(grad(b, [xi, yi]), grad(q, [xi, yi]), n_contract=1) + ab * qb - 0.5 * qb
        return [eq_a, eq_b, a(xb, yb) - 0.0, b(xb, yb) - 0.0]

    _d, A_n, b_n = _native_linear(build, mesh_size=0.3, vec=1, order=1, quad=2)
    A_f, b_f = _feax_linear(build, mesh_size=0.3, quad=2)
    assert A_n.shape == A_f.shape
    assert np.abs(A_n - A_f).max() < 1e-11
    assert np.abs(b_n - b_f).max() < 1e-11


def test_p1_homogeneous_source_is_trivial():
    """Edge case: zero source + homogeneous Dirichlet -> the only solution is u≡0,
    and the native load vector is exactly zero."""

    def build(d):
        u, w = d.fem_symbols(names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        return [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - 0.0 * w.bind(x=xi, y=yi), u(xb, yb) - 0.0]

    _d, A_n, b_n = _native_linear(build, mesh_size=0.3, vec=1, order=1, quad=2)
    assert np.abs(b_n).max() < 1e-12
    sol = np.linalg.solve(A_n, b_n)
    assert np.abs(sol).max() < 1e-12


# ---------------------------------------------------------------------------
# P2: analytic-solution + convergence oracle (matrix numbering differs from feax)
# ---------------------------------------------------------------------------


def _p2_poisson_l2_error(mesh_size):
    def build(d):
        u, w = d.fem_symbols(names=("u", "w"), order=2)
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        vv = w.bind(x=xi, y=yi)
        f = 2 * PI**2 * sin(PI * xi) * sin(PI * yi)
        return [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - f * vv, u(xb, yb) - 0.0]

    d, A_n, b_n = _native_linear(build, mesh_size=mesh_size, vec=1, order=2, quad=4)
    sol = np.linalg.solve(A_n, b_n)
    _, _, pts_f, _ = _get_mesh(d, 2, 2)
    exact = np.sin(PI * pts_f[:, 0]) * np.sin(PI * pts_f[:, 1])
    return float(np.sqrt(np.mean((sol - exact) ** 2)))


def test_p2_scalar_poisson_converges():
    """Native P2 Poisson solves the manufactured problem and converges super-linearly
    (rate > 2; P2 nodal L2 is ~O(h^3))."""
    e_coarse = _p2_poisson_l2_error(0.2)
    e_fine = _p2_poisson_l2_error(0.1)
    assert e_fine < 1e-3
    rate = np.log(e_coarse / e_fine) / np.log(2.0)
    assert rate > 2.0


# ---------------------------------------------------------------------------
# Public-path wiring: jno.fem must route 2D Lagrange to the native assembler and
# expose the correct DOF coordinates via fem.points (esp. the promoted P2 nodes).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_public_jno_fem_routes_native_and_points_match(order):
    """Through the public ``jno.fem`` API: a 2D Lagrange Poisson is assembled by the
    native path (no feax problem), and ``fem.points`` returns the coordinates the flat
    solution actually lives on — vertices for P1, vertices+edge-midpoints for P2 — so
    evaluating the manufactured field at ``fem.points`` recovers the solution.
    """
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.15)
    u, w = d.fem_symbols(names=("u", "w"), order=order)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    vv = w.bind(x=xi, y=yi)
    f = 2 * PI**2 * sin(PI * xi) * sin(PI * yi)
    fem = jno.fem([inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - f * vv, u(xb, yb) - 0.0])

    assert fem.problem is None  # routed natively (no feax problem object)
    sol = np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))
    pts = np.asarray(fem.points)
    assert pts.shape[0] == sol.shape[0] == fem.dofs  # points index the solution one-to-one

    exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    # P2 (more DOFs / higher order) resolves the field far better than P1.
    tol = 1e-2 if order == 1 else 1e-3
    assert np.abs(sol - exact).max() < tol


@pytest.mark.parametrize("vec", [1, 2])
def test_large_mesh_assembles_without_blowup(vec):
    """The Jacobian is assembled per element (``jacfwd`` of each element residual), so a fine mesh
    assembles in O(n_cells × n_local²) memory. A regression to a single global ``jacfwd(residual)``
    materialises an O(n_dofs × n_cells) tangent tensor and OOMs here (this mesh has ~thousands of DOFs
    — large enough that the global form allocated multiple GB and failed). Guards that scaling.
    """
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.03)  # ~1.4k P1 nodes / ~2.8k cells
    if vec == 1:
        u, w = d.fem_symbols(names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        vv = w.bind(x=xi, y=yi)
        f = 2 * PI**2 * sin(PI * xi) * sin(PI * yi)
        cons = [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - f * vv, u(xb, yb) - 0.0]
    else:
        u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        eu, ev = symgrad(u, [xi, yi]), symgrad(w, [xi, yi])
        vv = w.bind(x=xi, y=yi)
        cons = [
            1.0 * trace(eu) * trace(ev) + 2 * inner(eu, ev, n_contract=2) - (1.0 * vv[0] + 0.5 * vv[1]),
            u(xb, yb) - (0.0, 0.0),
        ]
    fem = jno.fem(cons)
    assert fem.problem is None  # native path
    n = fem.dofs
    # 1.4k+ DOFs over ~2.8k cells: the global-jacfwd intermediate (n_dofs × n_cells × n_local) was
    # multiple GB here and OOM'd the 8 GB GPU (elasticity blew up at ~940 DOFs); per-element
    # assembly stays in megabytes.
    assert n > 1400
    sol = np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))
    assert np.all(np.isfinite(sol))


# ---------------------------------------------------------------------------
# Transient: native assembles the semidiscrete block (M, A, time-dependent
# forcing, state0); a backward-Euler march recovers the manufactured solution.
# ---------------------------------------------------------------------------


def _march(fem):
    """Backward-Euler march of a native FeaxTimeBlock (M u̇ + A u = c + f(t))."""
    M, A = np.asarray(fem.M), np.asarray(fem.operator.A)
    c = np.asarray(fem.operator.affine_bias).reshape(-1)
    f = fem.operator.forcing_vector_fn
    w = np.asarray(fem.state0).reshape(-1).copy()
    dt, t = float(fem.dt), float(fem.t0)
    for _ in range(round((fem.t1 - fem.t0) / dt)):
        t += dt
        rhs = M @ w + dt * c
        if f is not None:
            rhs = rhs + dt * np.asarray(f(t)).reshape(-1)
        w = np.linalg.solve(M + dt * A, rhs)
    return w


def test_transient_heat_time_dependent_source_routes_native():
    """Heat ``u_t = α Δu + s(x,t)`` with the MMS ``u = (1+t) sin(πx) sin(πy)`` (backward Euler is
    temporally exact for this time-linear field). Routes natively, assembles the time-dependent
    forcing, and recovers the manufactured field at the final time."""
    AL = 1.0
    T = 0.5
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.1, time=(0.0, T, 26))
    u, w = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb = d.variable("boundary", split=True)[:2]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
    s = sin(PI * xi) * sin(PI * yi) * (1.0 + 2 * AL * PI**2 * (1.0 + ti))
    weak = ui.t * vi + AL * (ui.x * vi.x + ui.y * vi.y) - s * vi
    icf = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [ci[0], ci[1]])
    fem = jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - icf])

    assert fem.is_transient and fem.is_linear
    assert fem.problem is None  # native path
    assert fem.operator.forcing_vector_fn is not None  # time-dependent source carried per step

    w_final = _march(fem)
    pts = np.asarray(fem.points)
    exact = (1.0 + T) * np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    assert np.sqrt(np.mean((w_final - exact) ** 2)) < 2e-3  # spatial O(h^2) at mesh 0.1


def test_transient_nonhomogeneous_dirichlet_relaxes_to_one():
    """Autonomous heat ``u_t = Δu`` with ``u = 1`` held on the boundary and zero IC relaxes to
    ``u ≡ 1``. Exercises constant non-homogeneous Dirichlet on the native transient path (zero mass
    on Dirichlet rows, the lift carried in the affine bias)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2, time=(0.0, 0.5, 51))
    u, w = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb = d.variable("boundary", split=True)[:2]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 1.0, u(ci[0], ci[1]) - 0.0])

    assert fem.is_transient and fem.problem is None
    # Autonomous operator + no source: the time-dependent forcing increment is identically zero.
    fvf = fem.operator.forcing_vector_fn
    assert fvf is None or np.abs(np.asarray(fvf(0.3))).max() < 1e-12
    w_final = _march(fem)
    assert np.abs(w_final - 1.0).max() < 5e-3
