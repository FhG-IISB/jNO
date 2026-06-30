"""Spatial-Hessian (second-derivative) assembly for 4th-order weak forms (biharmonic / plate / Cahn-Hilliard).

The ``Hessian`` trace node and ``jno.np.laplacian``/``hessian`` already lower for an FE trial/test field;
this module pins the FE **assembly** of those terms (previously a `NotImplementedError` at the evaluator).

Validation is by **exact per-element energy identities**: a P{k} element represents polynomials up to
degree ``k`` exactly, so for a global quadratic ``u`` the discrete ``Δu`` / ``D²u`` are exact and the
assembled energy ``cᵀ K c`` equals ``∫(Δu)²`` / ``∫(D²u:D²u)`` to machine precision. (A wrong
``basix.index`` mapping or pushforward fails these.) P1 is excluded -- its reference Hessian is
identically zero.

**Conformity note:** standard Lagrange is C⁰, so ``∫Δu·Δv`` over P2 is *non-conforming* and does NOT give
a convergent biharmonic discretisation. A convergent solve needs a C¹ element (forthcoming) or the mixed
(Ciarlet–Raviart) method (validated here as the baseline). These tests check the *assembly* is exact, not
that a biharmonic BVP converges over P2.

Reference: the affine reference→physical Hessian map ``∂²φ/∂x_a∂x_b = K_ia K_jb ∂²φ/∂ξ_i∂ξ_j`` (K=J⁻¹),
exact because the simplex geometry is P1 (constant Jacobian) — Ciarlet, *The Finite Element Method for
Elliptic Problems* (2002), §2.
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
laplacian, hessian, inner = jno.np.laplacian, jno.np.hessian, jno.np.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _laplacian_K(dim, order, mesh_size):
    """Assemble ``K_ij = ∫ Δφ_i Δφ_j`` and return (K, node coords)."""
    if dim == 2:
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
        co = d.variable("interior", split=True)
        u, phi = d.fem_symbols(order=order)
        ui, vi = u.bind(x=co[0], y=co[1]), phi.bind(x=co[0], y=co[1])
        var = [co[0], co[1]]
    else:
        d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
        co = d.variable("interior", split=True)
        u, phi = d.fem_symbols(order=order)
        ui, vi = u.bind(x=co[0], y=co[1], z=co[2]), phi.bind(x=co[0], y=co[1], z=co[2])
        var = [co[0], co[1], co[2]]
    fem = jno.fem([laplacian(ui, var) * laplacian(vi, var)])
    return _dense(fem.A), np.asarray(fem.points)


def test_laplacian_energy_2d():
    """``cᵀK c = ∫(Δu)²`` for global quadratics: x²⇒Δ=2⇒4, xy⇒Δ=0 (null), x²+y²⇒Δ=4⇒16."""
    K, pts = _laplacian_K(2, 2, 0.4)
    assert np.allclose(K, K.T, atol=1e-10), "Laplacian stiffness must be symmetric"
    x, y = pts[:, 0], pts[:, 1]
    for c, exp, name in [(x**2, 4.0, "x^2"), (x * y, 0.0, "xy(null)"), (x**2 + y**2, 16.0, "x^2+y^2")]:
        got = float(c @ K @ c)
        assert abs(got - exp) < 1e-9, f"∫(Δu)² for {name}: got {got:.3e}, expect {exp}"


def test_hessian_energy_2d():
    """Full Hessian ``inner(hessian(u),hessian(v))`` ⇒ ``cᵀK c = ∫(D²u:D²u)``: xy⇒2, x²⇒4."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4)
    xi, yi, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols(order=2)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([inner(hessian(ui, [xi, yi]), hessian(vi, [xi, yi]), n_contract=2)])
    K, pts = _dense(fem.A), np.asarray(fem.points)
    assert np.allclose(K, K.T, atol=1e-10)
    x, y = pts[:, 0], pts[:, 1]
    for c, exp, name in [(x * y, 2.0, "xy"), (x**2, 4.0, "x^2"), (y**2, 4.0, "y^2")]:
        got = float(c @ K @ c)
        assert abs(got - exp) < 1e-9, f"∫(D²u:D²u) for {name}: got {got:.3e}, expect {exp}"


def test_laplacian_energy_3d():
    pytest.importorskip("pygmsh", reason="pygmsh required for cube meshing")
    K, pts = _laplacian_K(3, 2, 0.6)
    assert np.allclose(K, K.T, atol=1e-10)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    for c, exp, name in [(x**2, 4.0, "x^2"), (x * y, 0.0, "xy(null)"), (x**2 + y**2 + z**2, 36.0, "r^2")]:
        got = float(c @ K @ c)
        assert abs(got - exp) < 1e-9, f"3D ∫(Δu)² for {name}: got {got:.3e}, expect {exp}"


def test_p1_hessian_is_identically_zero():
    """P1 reference Hessians are 0 (linear basis), so ``∫Δu·Δv`` over P1 assembles a zero operator -- the
    assembly must not error, and the clear 'use order>=2' message only fires if shape_hess is missing."""
    K, _ = _laplacian_K(2, 1, 0.4)
    assert float(np.max(np.abs(K))) < 1e-12, "P1 Laplacian-squared operator must be identically zero"


def test_temporal_second_derivative_still_routes_to_time_path():
    """A temporal second derivative (`u_tt`, wave equation) must route to the second-order-in-time
    augmented block, NOT the spatial-Hessian assembly -- the temporal gate keeps them separate."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 1.0, 11))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.tt * vi + (ui.x * vi.x + ui.y * vi.y),  # u_tt = Δu
            u(xb, yb) - 0.0,
            u(ci[0], ci[1]) - 0.0,
            u.bind(x=ci[0], y=ci[1], t=d.variable("initial", split=True)[2]).t - 0.0,
        ]
    )
    assert fem.is_transient, "a u_tt form must build a transient (second-order-time) problem, not a spatial Hessian"


def test_biharmonic_mixed_method_recovers():
    """Baseline (no Hessian path, no C¹ element): the Ciarlet–Raviart mixed method solves the simply-
    supported biharmonic ``Δ²u = f`` with coupled C⁰ Lagrange. Introduce ``w = Δu``; with ``u=w=0`` on
    ∂Ω (simply supported, since ``w*=Δu*=0`` there) the weak system is ``∫w p + ∫∇u·∇p = 0`` (w=Δu) and
    ``∫∇w·∇q + ∫f q = 0`` (Δw=f). Manufactured ``u* = sin(πx)sin(πy)`` ⇒ ``Δu* = -2π²u*`` (=0 on ∂Ω) and
    ``Δ²u* = 4π⁴u* = f``. This is the reference the forthcoming C¹ element matches."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, p = d.fem_symbols(order=2, names=("u", "p"))
    w, q = d.fem_symbols(order=2, names=("w", "q"))
    ui, pi = u.bind(x=xi, y=yi), p.bind(x=xi, y=yi)
    wi, qi = w.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    f = 4.0 * PI**4 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    fem = jno.fem(
        [
            (ui.x * pi.x + ui.y * pi.y) + wi * pi,  # ∫∇u·∇p + ∫w p = 0  (w=Δu weak; u-trial first ⇒ u=field 0)
            wi.x * qi.x + wi.y * qi.y + f * qi,  # ∫∇w·∇q + ∫f q = 0   (Δw = f, weak)
            u(xb, yb) - 0.0,
            w(xb, yb) - 0.0,
        ]
    )
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    off = fem.offsets  # [0, n_u, n_total]
    uh = sol[off[0] : off[1]]
    pts_u = np.asarray(fem.field_points[0])
    u_exact = np.sin(PI * pts_u[:, 0]) * np.sin(PI * pts_u[:, 1])
    rel = float(np.linalg.norm(uh - u_exact) / np.linalg.norm(u_exact))
    assert rel < 2e-2, f"mixed-method biharmonic did not recover u*: rel-L2 {rel:.3e}"
