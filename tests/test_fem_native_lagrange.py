"""Validation of the native 2D Lagrange assembler (:func:`assemble_fem_native`).

The assembler is exercised here *directly* (it is not yet routed through
``jno.fem``); the per-constraint classification that ``jno.fem`` performs is
replicated by :func:`_classify` so the assembler receives exactly the
``(volume_terms, boundary_terms, dirichlet_raw, ic_residuals)`` contract it
expects.

Two oracles are used:

* **Matrix-level vs the ``jno.fem`` route** (machine precision).  The high-level
  ``jno.fem`` route is the reference; for P1 single- and vector-fields the direct
  global DOF numbering matches the route's, so the assembled ``A``/``b`` can be
  compared entry-for-entry.  This catches push-forward, scatter and DOF-map bugs
  that a convergence-rate check can hide on symmetric problems.
* **Analytic solution + convergence** for P2.  The native P2 element numbers its
  edge nodes in basix's element-DOF order (required: the gradients come from basix
  tabulation), and is validated against the manufactured field at the native nodes,
  which must converge at O(h^3).
"""

from __future__ import annotations

import numpy as np
import pytest

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
    """The native assembler is compared at float64 precision, so these tests opt
    into x64 per-test (the session default is x64-off; save/restore keeps the flag
    from leaking to other modules)."""
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


def _route_linear(build, *, mesh_size, quad):
    """Assemble ``(A, b)`` for the same problem through the high-level ``jno.fem`` route (the oracle)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    fem = jno.fem(build(d), quad_degree=quad)
    assert fem.is_linear
    return _dense(fem.A), np.asarray(fem.b)


# ---------------------------------------------------------------------------
# P1 matrix-level oracle (direct global numbering == the jno.fem route's)
# ---------------------------------------------------------------------------


def test_p1_scalar_poisson_matches_jno_fem():
    """-Δu = f, u = sin(πx)sin(πy), homogeneous Dirichlet."""

    def build(d):
        u, w = d.fem_symbols(names=("u", "w"))
        xi, yi = d.variable("interior", split=True)[:2]
        xb, yb = d.variable("boundary", split=True)[:2]
        vv = w.bind(x=xi, y=yi)
        f = 2 * PI**2 * sin(PI * xi) * sin(PI * yi)
        return [inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - f * vv, u(xb, yb) - 0.0]

    _d, A_n, b_n = _native_linear(build, mesh_size=0.2, vec=1, order=1, quad=2)
    A_f, b_f = _route_linear(build, mesh_size=0.2, quad=2)
    assert A_n.shape == A_f.shape
    assert np.abs(A_n - A_f).max() < 1e-11
    assert np.abs(b_n - b_f).max() < 1e-11


def test_p1_vector_elasticity_matches_jno_fem():
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
    A_f, b_f = _route_linear(build, mesh_size=0.25, quad=2)
    assert A_n.shape == A_f.shape
    assert np.abs(A_n - A_f).max() < 1e-11
    assert np.abs(b_n - b_f).max() < 1e-11


def test_p1_neumann_plus_dirichlet_matches_jno_fem():
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
    A_f, b_f = _route_linear(build, mesh_size=0.25, quad=2)
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


def test_p1_nonlinear_bratu_matches_jno_fem():
    """Bratu -Δu - λ e^u = 0 (nonlinear).

    The residual matches the ``jno.fem`` route entry-for-entry at a probe state;
    correctness is then asserted by driving both operators through the *same* Newton
    iteration and comparing the converged solutions.
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
    u_dsl = _newton(fem.residual, fem.jacobian, n)
    assert np.abs(np.asarray(op.residual(jnp.asarray(u_native))).reshape(-1)).max() < 1e-9
    assert np.abs(u_native - u_dsl).max() < 1e-10


def test_p1_coupled_two_field_matches_jno_fem():
    """Coupled reaction-diffusion (two scalar P1 fields a, b with cross terms) —
    exercises the multi-field block assembly; the direct path and the ``jno.fem`` route both order
    the fields by first appearance with node-major DOFs, so the block matrix matches."""

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
    A_f, b_f = _route_linear(build, mesh_size=0.3, quad=2)
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
# P2: analytic-solution + convergence oracle (manufactured field at the native nodes)
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
    native path (``fem.problem`` is None), and ``fem.points`` returns the coordinates the flat
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

    assert fem.problem is None  # routed natively (problem is None)
    sol = np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))
    pts = np.asarray(fem.points)
    assert pts.shape[0] == sol.shape[0] == fem.dofs  # points index the solution one-to-one

    exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    # P2 (more DOFs / higher order) resolves the field far better than P1.
    tol = 1e-2 if order == 1 else 1e-3
    assert np.abs(sol - exact).max() < tol


# ---------------------------------------------------------------------------
# 3D (tetrahedral) Lagrange: the assembler is dimension-generic (the cell Jacobian,
# element factory and facet machinery all key off `dim`), and handles both Dirichlet
# and Neumann/Robin terms (the latter via tet-face surface quadrature).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_3d_tet_poisson_routes_native_and_solves(order):
    """A 3D tet Poisson (Dirichlet on every cube face) is assembled natively and solves the
    manufactured ``sin·sin·sin`` problem; ``fem.points`` indexes the flat solution one-to-one
    (vertices for P1, vertices+edge-midpoints for P2)."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain()
    u, w = d.fem_symbols(names=("u", "w"), order=order)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    vv = w.bind(x=xi, y=yi, z=zi)
    f = 3 * PI**2 * sin(PI * xi) * sin(PI * yi) * sin(PI * zi)
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    fem = jno.fem([inner(grad(u, [xi, yi, zi]), grad(w, [xi, yi, zi]), n_contract=1) - f * vv, u(xb, yb, zb) - 0.0])

    assert fem.problem is None  # routed natively (problem is None)
    sol = np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))
    pts = np.asarray(fem.points)
    assert pts.shape[0] == sol.shape[0] == fem.dofs
    exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1]) * np.sin(PI * pts[:, 2])
    rel = np.linalg.norm(sol - exact) / np.linalg.norm(exact)
    assert rel < (0.12 if order == 1 else 0.06)


def test_3d_tet_p2_reproduces_quadratic_exactly():
    """A P2 tet must represent a quadratic field exactly (patch test). This pins both the basix
    tet edge-DOF ordering used by ``_promote_to_quadratic`` (a permutation bug silently scrambles
    the local DOFs) and the robust facet-based boundary-node detection (the geometric containment
    test misses P2 edge-midpoints sitting exactly on a cube face, leaving them unconstrained)."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.4).domain()
    u, w = d.fem_symbols(names=("u", "w"), order=2)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    vv = w.bind(x=xi, y=yi, z=zi)
    f = -12.0 + 0.0 * xi  # u = x^2 + 2y^2 + 3z^2 + xy  =>  -lap u = -(2 + 4 + 6) = -12
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    gb = xb**2 + 2 * yb**2 + 3 * zb**2 + xb * yb
    fem = jno.fem([inner(grad(u, [xi, yi, zi]), grad(w, [xi, yi, zi]), n_contract=1) - f * vv, u(xb, yb, zb) - gb])

    assert fem.problem is None
    sol = np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))
    pts = np.asarray(fem.points)
    exact = pts[:, 0] ** 2 + 2 * pts[:, 1] ** 2 + 3 * pts[:, 2] ** 2 + pts[:, 0] * pts[:, 1]
    assert np.abs(sol - exact).max() < 1e-9  # exact up to the linear solve (x64)


def test_3d_tet_neumann_routes_native_and_recovers_flux():
    """A 3D tet problem with a Neumann (surface flux) term assembles natively over the tet faces (4
    triangular faces, 2-D triangle quadrature). Manufactured ``u = x``: ``-Δu = 0``, Dirichlet ``u = 0``
    on the left face, ``∂u/∂n = +1`` on the right (written ``-1·v`` in the residual convention), and the
    y/z faces left natural (zero flux, which ``u = x`` satisfies). The native solution recovers ``u = x``
    to the linear-solve tolerance -- pinning the tet-face area element, the parent-basis face
    tabulation and the local-face indexing against ``build_facet_connectivity``."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain()
    u, w = d.fem_symbols(names=("u", "w"))
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xr, yr, zr = d.variable("right", split=True)[:3]
    xl, yl, zl = d.variable("left", split=True)[:3]
    vv = w.bind(x=xi, y=yi, z=zi)
    vr = w.bind(x=xr, y=yr, z=zr)
    fem = jno.fem(
        [
            inner(grad(u, [xi, yi, zi]), grad(w, [xi, yi, zi]), n_contract=1) - 0.0 * vv,
            -1.0 * vr,  # ∂u/∂n = +1 on the right face (3D tet-face surface integral)
            u(xl, yl, zl) - 0.0,
        ]
    )
    assert fem.problem is None  # routed natively (tet-face surface quadrature)
    sol = np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))
    pts = np.asarray(fem.points)
    assert np.abs(sol - pts[:, 0]).max() < 1e-7  # recovers u = x (x64)


def test_3d_tet_multifield_routes_native_and_recovers():
    """A coupled (two-field) steady problem on a 3D tet mesh assembles natively -- the multifield
    gate admits 3D now that the assembler is dimension-generic. Manufactured ``u = p = sin·sin·sin``
    with the symmetric coupling ``-Δu + p = (3π²+1)g`` / ``-Δp + u = (3π²+1)g`` (which ``u = p = g``
    solves) is recovered in both blocks."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.25).domain()
    u, wu = d.fem_symbols(names=("u", "wu"))
    p, wp = d.fem_symbols(names=("p", "wp"))
    xi, yi, zi = d.variable("interior", split=True)[:3]
    ui, vi = u.bind(x=xi, y=yi, z=zi), wu.bind(x=xi, y=yi, z=zi)
    pp, qi = p.bind(x=xi, y=yi, z=zi), wp.bind(x=xi, y=yi, z=zi)
    g = sin(PI * xi) * sin(PI * yi) * sin(PI * zi)
    fac = 3 * PI**2 + 1.0
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    fem = jno.fem(
        [
            inner(grad(u, [xi, yi, zi]), grad(wu, [xi, yi, zi]), n_contract=1) + pp * vi - fac * g * vi,
            inner(grad(p, [xi, yi, zi]), grad(wp, [xi, yi, zi]), n_contract=1) + ui * qi - fac * g * qi,
            u(xb, yb, zb) - 0.0,
            p(xb, yb, zb) - 0.0,
        ]
    )
    assert fem.problem is None  # 3D multifield routed natively
    sol = np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))
    offs = fem.offsets
    for i in range(2):
        pts = np.asarray(fem.field_points[i])
        block = sol[offs[i] : offs[i + 1]]
        exact = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1]) * np.sin(PI * pts[:, 2])
        assert np.linalg.norm(block - exact) / np.linalg.norm(exact) < 0.08


# ---------------------------------------------------------------------------
# Periodic ties: the prolongation reduction (_build_periodic_reduction +
# build_periodic_prolongation) is fed the native assembly cells, so a scalar
# single-field periodic problem reduces and solves natively in both the steady and
# transient cases. Vector / runtime-parametric periodic raise NotImplementedError.
# ---------------------------------------------------------------------------


def test_periodic_steady_scalar_routes_native_and_reduces():
    """A steady scalar Poisson, periodic in x (``u(left) - u(right)``) + Dirichlet in y, assembles
    natively (``fem.problem`` is None) and the periodic tie still reduces the system (slave DOFs eliminated)
    and recovers the manufactured solution -- the reduction reads the native assembly cells."""
    pi = np.pi
    dom = jno.domain({"fine": box(0, 0, 0.5, 1), "coarse": box(0.5, 0, 1, 1)}).build_mesh(0.12, sizes={"fine": 0.06})
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)

    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    hh = jno.np.cos(2 * pi * xi) + 0.5 * jno.np.sin(2 * pi * xi)
    f = 5 * pi**2 * hh * sin(PI * yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0, u(xt, yt) - 0.0, u(xl, yl) - u(xr, yr)])

    assert fem.problem is None  # routed natively
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]
    uh = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    ex = (np.cos(2 * pi * pts[:, 0]) + 0.5 * np.sin(2 * pi * pts[:, 0])) * np.sin(pi * pts[:, 1])
    assert float(np.linalg.norm(uh - ex) / np.linalg.norm(ex)) < 0.05


def test_periodic_transient_routes_native_and_reduces():
    """A transient periodic problem routes natively: the prolongation reduction is pre-built into the
    assembly context at assembly time, so the semidiscrete block carries the reduced periodic system."""
    dom = jno.domain(box(0, 0, 1, 1), mesh_size=0.2, time=(0.0, 0.01, 2))
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)
    u, phi = dom.fem_symbols()
    xi, yi, ti = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ci = dom.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ic = sin(PI * ci[0]) * sin(PI * ci[1])
    fem = jno.fem(
        [
            ui.t * vi + ui.x * vi.x + ui.y * vi.y,  # u_t = Δu
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
            u(xl, yl) - u(xr, yr),  # periodic in x
            u(ci[0], ci[1]) - ic,
        ]
    )
    assert fem.problem is None  # routed natively
    assert fem.is_transient
    assert fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]


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
    """Backward-Euler march of a native SemidiscreteTimeBlock (M u̇ + A u = c + f(t))."""
    M, A = _dense(fem.M), _dense(fem.operator.A)
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


def test_transient_scalar_parametric_routes_native():
    """A transient with an unknown scalar coefficient ``u_t = alpha Δu`` routes natively to a
    parametric SemidiscreteTimeBlock whose ``operator_fn(t, args)`` re-assembles A(alpha) per step (used by the
    differentiable inverse solve). The free-row operator scales linearly with alpha here."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2, time=(0.0, 0.3, 16))
    u, w = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb = d.variable("boundary", split=True)[:2]
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
    alpha = jno.np.parameter((1,), name="alpha")
    icf = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [ci[0], ci[1]])
    fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - icf])

    assert fem.is_transient and fem.problem is None
    blk = fem.operator
    assert blk.operator_fn is not None and list(blk.runtime_parameter_exprs) == ["alpha"]
    a1 = np.asarray(blk.operator_fn(0.0, {"alpha": 1.0}).todense())
    a2 = np.asarray(blk.operator_fn(0.0, {"alpha": 2.0}).todense())
    free = ~np.isclose(np.abs(a1).sum(axis=1), 1.0)  # interior rows (Dirichlet rows -> unit diagonal)
    assert free.any() and np.abs(a2[free] - 2.0 * a1[free]).max() < 1e-10


# ---------------------------------------------------------------------------
# Runtime-parametric (inverse): the operator is re-assembled at the runtime args and
# stays differentiable in them. A finite-difference check guards the gradient itself
# (a forward-value oracle would pass even with a silently wrong/severed gradient).
# ---------------------------------------------------------------------------


def _parametric_poisson(mesh_size=0.2):
    """Parametric Poisson ``-alpha Δu = f`` (recovers exact ``u`` at alpha=1)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    alpha = jno.np.parameter((1,), name="alpha")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    weak = alpha * (ui.x * vi.x + ui.y * vi.y) - f * vi
    return jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)


def test_native_parametric_routes_native_and_is_parametric():
    fem = _parametric_poisson()
    assert fem.problem is None  # native path
    sys = fem.operator
    assert sys.is_parametric and list(sys.runtime_parameter_exprs) == ["alpha"]
    # The operator genuinely depends on alpha: the free (non-Dirichlet) rows scale by alpha for
    # -alpha*lap (Dirichlet rows are the identity regardless, so compare only the free block).
    a1 = _dense(sys.evaluate({"alpha": 1.0})[0])
    a2 = _dense(sys.evaluate({"alpha": 2.0})[0])
    free = ~np.isclose(np.abs(a1).sum(axis=1), 1.0)  # interior rows (Dirichlet rows have a unit diagonal only)
    assert free.any()
    assert np.abs(a2[free] - 2.0 * a1[free]).max() < 1e-10


def test_native_parametric_gradient_matches_finite_difference():
    """The key inverse property: ∂(solve)/∂alpha must flow through the native re-assembly.
    A forward A-vs-A oracle cannot see a severed gradient — finite difference can."""
    fem = _parametric_poisson()
    sys = fem.operator

    def loss(a):
        A, b = sys.evaluate({"alpha": a})
        u = jnp.linalg.solve(A.todense(), jnp.asarray(b).reshape(-1))
        return jnp.sum(u**2)

    a0 = 1.3
    g_ad = float(jax.grad(loss)(a0))
    eps = 1e-6
    g_fd = float((loss(a0 + eps) - loss(a0 - eps)) / (2 * eps))
    assert abs(g_ad) > 1e-8  # the parameter genuinely moves the solution
    assert abs(g_ad - g_fd) <= 1e-5 * max(1.0, abs(g_fd))


# ---------------------------------------------------------------------------
# Native fem_context (VPINN / grouped-weak-form): the quadrature, shape values &
# gradients, JxW and surface data must satisfy the basic FEM invariants (partition
# of unity, exact area/measure sums), so the network-trial test-projection is sound.
# ---------------------------------------------------------------------------


def test_native_field_parameter_routes_native_and_gradient_flows():
    """A nodal FIELD parameter k(x) = jno.np.parameter(phi) routes natively: its per-cell nodal
    values are gathered and interpolated to the quad points. A linear field equals the same
    coordinate-function coefficient (P1 interpolation is exact) -> catches gather/node-order bugs;
    and ∂(solve)/∂k flows through the gather (finite-difference check)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.25)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    k = jno.np.parameter(phi, name="k")
    fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem.problem is None  # native path
    sys = fem.operator
    assert sys.is_parametric and list(sys.runtime_parameter_exprs) == ["k"]

    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_true = jnp.asarray(0.6 + 0.8 * nodes[:, 0] + 0.5 * nodes[:, 1])  # smooth, exactly P1-representable
    fem_ref = jno.fem([(0.6 + 0.8 * xi + 0.5 * yi) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    A_field = _dense(sys.evaluate({"k": k_true})[0])
    A_ref = np.asarray(fem_ref.A)
    assert np.abs(A_field - A_ref).max() < 1e-9  # nodal interpolation/gather is exact for a linear k

    def loss(kv):
        A, b = sys.evaluate({"k": kv})
        return jnp.sum(jnp.linalg.solve(A.todense(), jnp.asarray(b).reshape(-1)) ** 2)

    g_ad = np.asarray(jax.grad(loss)(k_true))
    j = int(np.argmax(np.abs(g_ad)))  # check the most-sensitive nodal component
    eps = 1e-6
    kp = k_true.at[j].add(eps)
    km = k_true.at[j].add(-eps)
    g_fd = float((loss(kp) - loss(km)) / (2 * eps))
    assert abs(g_ad[j]) > 1e-8 and abs(g_ad[j] - g_fd) <= 1e-4 * max(1.0, abs(g_fd))


def test_native_fem_context_satisfies_fem_invariants():
    """The native fem_context (quadrature, shape values & gradients, JxW, surface data) used by the
    VPINN / grouped-weak-form evaluator must satisfy the basic FEM invariants on a P1 unit-square
    mesh: a partition of unity, gradients of the partition summing to zero, and exact area /
    edge-length measures. These pin the tabulation, push-forward and facet quadrature without an
    external oracle."""
    from jno.utils.solver.fem_native import build_native_fem_context

    dn = jno.domain(box(0, 0, 1, 1), mesh_size=0.2)
    fc, _qp, _sq, _sn = build_native_fem_context(dn, element_type="TRI3", quad_degree=2, vec=1, neumann_tags=["right"])

    n_nodes = int(fc["num_total_nodes"])
    cells = np.asarray(fc["cells"])
    assert cells.shape[1] == 3  # P1 triangles
    assert int(cells.max()) == n_nodes - 1  # the cells index every node
    assert np.asarray(fc["global_areas"]).shape[0] == n_nodes

    # Volume: P1 shape functions are a partition of unity at every quad point ...
    assert np.allclose(np.asarray(fc["N_flat"]).sum(axis=-1), 1.0, atol=1e-12)
    # ... and their physical gradients sum to zero (the gradient of a constant is zero).
    assert np.abs(np.asarray(fc["dN_dx_flat"]).sum(axis=1)).max() < 1e-10
    # JxW integrates the unit square exactly; the nodal area partition also sums to the area.
    assert abs(float(np.asarray(fc["JxW"]).sum()) - 1.0) < 1e-12
    assert abs(float(np.asarray(fc["global_areas"]).sum()) - 1.0) < 1e-12

    # Surface ('right' edge, length 1): parent shape values restricted to the face are a partition
    # of unity at every face quad point, and the boundary measure recovers the edge length.
    sd = fc["surface_data"]["right"]
    assert np.allclose(np.asarray(sd["face_shape_vals"]).sum(axis=-1), 1.0, atol=1e-12)
    assert abs(float(np.asarray(sd["global_boundary_areas"]).sum()) - 1.0) < 1e-12
