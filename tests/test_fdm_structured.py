"""``jno.fdm`` on a **structured grid** — ``jno.domain(Shape.rect(...), structured=True)``.

A structured request builds a regular right-triangulation and records a grid descriptor on
``mesh_connectivity["grid"]``; ``jno.fdm`` then takes the assembly-free direct finite-difference
stencils (the 5-point Laplacian) instead of the cotangent operator. On a uniform right-triangulation
those coincide *exactly*, so the grid path is validated against the cotangent path as an oracle, and
the full solve/gradient paths are validated against analytic solutions.

Run with x64 (the solve accumulates in float64)."""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the unstructured comparison domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _nodes(d):
    return np.asarray(d.mesh_connectivity["points"])[:, :2]


def _structured(x0=0.0, y0=0.0, x1=1.0, y1=1.0, size=0.1, **kw):
    return jno.domain(jno.Shape.rect(x0, y0, x1, y1, size=size), structured=True, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# grid metadata + structure
# ─────────────────────────────────────────────────────────────────────────────


def test_grid_descriptor_present_and_correct():
    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.1)
    grid = d.mesh_connectivity.get("grid")
    assert grid is not None, "structured=True must stamp mesh_connectivity['grid']"
    assert grid["shape"] == (11, 11)
    assert np.allclose(grid["spacing"], (0.1, 0.1))
    assert np.allclose(grid["origin"], (0.0, 0.0))
    assert _nodes(d).shape[0] == 11 * 11  # N = Nx * Ny


def test_point_ordering_is_meshgrid():
    """The FD grid stencil relies on node order idx(i, j) = i·Ny + j (reshape → meshgrid). Guard it."""
    d = _structured(0.0, 0.0, 2.0, 1.0, size=0.1)  # non-square: Nx != Ny
    Nx, Ny = d.mesh_connectivity["grid"]["shape"]
    assert (Nx, Ny) == (21, 11)
    P = _nodes(d).reshape(Nx, Ny, 2)
    assert np.all(np.diff(P[:, 0, 0]) > 0)  # x increases along axis 0
    assert np.all(np.diff(P[0, :, 1]) > 0)  # y increases along axis 1
    assert np.allclose(P[:, :, 0], P[:, :1, 0])  # x constant along axis 1
    assert np.allclose(P[:, :, 1], P[:1, :, 1])  # y constant along axis 0


def test_offset_origin_and_nonsquare():
    d = _structured(1.0, 2.0, 3.0, 4.0, size=0.25)
    grid = d.mesh_connectivity["grid"]
    assert grid["origin"] == (1.0, 2.0)
    assert grid["shape"] == (9, 9)  # 2.0 / 0.25 = 8 cells → 9 nodes per axis
    p = _nodes(d)
    assert np.isclose(p[:, 0].min(), 1.0) and np.isclose(p[:, 0].max(), 3.0)
    assert np.isclose(p[:, 1].min(), 2.0) and np.isclose(p[:, 1].max(), 4.0)


def test_coarse_size_clamps_to_two_cells():
    """A huge size must not collapse the grid below the 2-cell minimum (edge stencil needs ≥3 nodes)."""
    d = _structured(0.0, 0.0, 1.0, 1.0, size=5.0)
    assert d.mesh_connectivity["grid"]["shape"] == (3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# stencil correctness — grid path ≡ cotangent path on the same mesh
# ─────────────────────────────────────────────────────────────────────────────


def test_grid_laplacian_matches_cotangent_exactly():
    """On a uniform right-triangulation the 5-point stencil == the cotangent P1 operator. The grid
    fast path must reproduce the cotangent path to machine precision (the oracle)."""
    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.1)
    p = _nodes(d)
    u = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.cos(2 * np.pi * p[:, 1]) + 0.3 * p[:, 0] ** 2)
    lap_grid = np.asarray(jno.fdm.laplacian(u, d))  # grid fast path (grid descriptor present)
    lap_cot = np.asarray(jno.fdm.laplacian(u, d, method="cotangent"))
    interior = (p[:, 0] > 1e-9) & (p[:, 0] < 1 - 1e-9) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
    assert np.max(np.abs(lap_grid[interior] - lap_cot[interior])) < 1e-10


def test_grid_laplacian_second_order_vs_analytic():
    """Δ(sin πx · sin πy) = −2π²·u; the interior error is the expected 2nd-order truncation and
    shrinks ≈4× per halving of h."""

    def err(size):
        d = _structured(0.0, 0.0, 1.0, 1.0, size=size)
        p = _nodes(d)
        u = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
        lap = np.asarray(jno.fdm.laplacian(u, d))
        analytic = -2.0 * np.pi**2 * np.asarray(u)
        interior = (p[:, 0] > 1e-9) & (p[:, 0] < 1 - 1e-9) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
        return np.max(np.abs(lap[interior] - analytic[interior]))

    e_coarse, e_fine = err(0.05), err(0.025)
    assert e_fine < e_coarse
    assert e_fine / e_coarse < 0.4  # ~4× reduction (2nd order)


def test_grid_gradient_matches_analytic():
    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.04)
    p = _nodes(d)
    u = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    g = np.asarray(jno.fdm.gradient(u, d))  # (N, 2), grid fast path
    gx = np.pi * np.cos(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    gy = np.pi * np.sin(np.pi * p[:, 0]) * np.cos(np.pi * p[:, 1])
    interior = (p[:, 0] > 1e-9) & (p[:, 0] < 1 - 1e-9) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
    assert np.max(np.abs(g[interior, 0] - gx[interior])) < 5e-2
    assert np.max(np.abs(g[interior, 1] - gy[interior])) < 5e-2


# ─────────────────────────────────────────────────────────────────────────────
# full solve — function form and constraint-list form
# ─────────────────────────────────────────────────────────────────────────────


def test_structured_poisson_function_form():
    """-Δu = f on a structured grid, u = sin(πx)sin(πy). Function-form FDMSystem → grid stencil."""
    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.05)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * exact)
    sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - f, dirichlet={"boundary": 0.0})
    u = np.asarray(sys.solve()).reshape(-1)
    assert float(np.linalg.norm(u - exact) / np.linalg.norm(exact)) < 1e-2


def test_structured_matches_unstructured():
    """Same Poisson problem solved on a structured grid and an unstructured mesh both hit the analytic
    solution — the structured backend is not a different physics, only a faster discretisation."""

    def rel_err_structured(size):
        d = _structured(0.0, 0.0, 1.0, 1.0, size=size)
        p = _nodes(d)
        exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
        f = jnp.asarray(2 * np.pi**2 * exact)
        u = np.asarray(jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - f, dirichlet={"boundary": 0.0}).solve())
        return float(np.linalg.norm(u.reshape(-1) - exact) / np.linalg.norm(exact))

    def rel_err_unstructured(size):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=size)
        p = _nodes(d)
        exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
        f = jnp.asarray(2 * np.pi**2 * exact)
        u = np.asarray(jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - f, dirichlet={"boundary": 0.0}).solve())
        return float(np.linalg.norm(u.reshape(-1) - exact) / np.linalg.norm(exact))

    assert rel_err_structured(0.05) < 1e-2
    assert rel_err_unstructured(0.05) < 1e-2


def test_structured_constraint_list_solve():
    """The canonical fem-style authoring — jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(bnd) - 0]).solve() —
    JUST WORKS on a structured grid: the strong-form ∂² routes through the grid Hessian, and the
    structured solve uses GMRES so the nonsymmetric reduced-Dirichlet operator does not break down."""
    import jno.jnp_ops as jnn

    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.05)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    sol = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve()
    assert float(np.all(np.isfinite(np.asarray(sol))))
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 3e-2


def test_structured_constraint_list_inhomogeneous():
    """Inhomogeneous Dirichlet on a structured grid via the constraint list: u = x²+y², -Δu = -4."""
    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.06)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    exact = p[:, 0] ** 2 + p[:, 1] ** 2
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    sol = jno.fdm([-ui.d2(x) - ui.d2(y) + 4.0, u(xb, yb) - (xb**2 + yb**2)]).solve()
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 1e-2


@pytest.mark.slow
def test_structured_constraint_list_differentiable():
    """The structured constraint-list solve is reverse-mode differentiable (inverse-problem readiness) —
    GMRES keeps gradients flowing through the driver's custom_linear_solve firewall. (Slow: the GMRES
    reverse pass compiles a large graph — the fast function-form differentiability is covered above.)"""
    import jno.jnp_ops as jnn

    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.12)  # small grid keeps the grad compile fast
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    obs = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    u = d.unknown()
    ui = u.bind(x=x, y=y)

    def loss(scale):
        f = scale * 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
        sol = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve()
        return jnp.mean((sol.reshape(-1) - obs) ** 2)

    g = float(jax.grad(loss)(1.5))
    assert np.isfinite(g) and g > 0.0  # at scale=1.5 (> true 1.0) the loss increases with scale
    assert float(loss(1.0)) < float(loss(1.5))


def test_structured_differentiable_for_inverse():
    """The structured solve is reverse-mode differentiable w.r.t. a residual parameter (inverse-problem
    readiness) — the grid stencils keep gradients flowing."""
    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.08)
    p = _nodes(d)
    fbase = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    obs = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))

    def loss(scale):
        sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - scale * fbase, dirichlet={"boundary": 0.0})
        return jnp.mean((sys.solve() - obs) ** 2)

    g = float(jax.grad(loss)(1.5))
    assert np.isfinite(g)
    assert g > 0.0  # at scale=1.5 (> true 1.0) the loss increases with scale
    assert float(loss(1.0)) < float(loss(1.5))


def test_structured_transient_heat():
    """Method-of-lines transient on a structured grid: u_t = ν Δu, homogeneous Dirichlet → decays as
    e^(−2νπ²t)·u0 for u0 = sin(πx)sin(πy)."""
    nu, T = 0.05, 0.4
    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.06)
    d = jno.domain(d, time=(0.0, T, 160))  # add time (grid descriptor carries through the clone)
    assert d.mesh_connectivity.get("grid") is not None
    p = _nodes(d)
    u0 = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    sys = jno.fdm(d, residual=lambda u: -nu * jno.fdm.laplacian(u, d), dirichlet={"boundary": 0.0})
    traj = np.asarray(sys.solve_transient(jnp.asarray(u0), (0.0, T), 160))
    final = traj[-1]
    expected = np.exp(-2 * nu * np.pi**2 * T) * u0
    interior = (p[:, 0] > 1e-9) & (p[:, 0] < 1 - 1e-9) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
    assert float(np.linalg.norm(final[interior] - expected[interior]) / np.linalg.norm(expected[interior])) < 5e-2


def test_structured_constraint_list_transient():
    """fem-style transient authoring on a structured grid: ui.t = ν Δu, IC as a constraint, t_span from
    domain.time. Composes structured=True *with* time= directly. The march's backward-Euler operator
    (I − dt·ν·Δ) is diagonally dominant, so it is robust with the default inner solve (no GMRES needed)."""
    import jno.jnp_ops as jnn

    nu, T = 0.05, 0.3
    d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1), structured=True, time=(0.0, T, 100))
    assert d.mesh_connectivity.get("grid") is not None
    p = _nodes(d)
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    u0 = jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    traj = np.asarray(jno.fdm([ui.t - nu * (ui.d2(x) + ui.d2(y)), u(xb, yb) - 0.0, u(xi, yi) - u0]).solve())
    final = traj[-1]
    expected = np.exp(-2 * nu * np.pi**2 * T) * (np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    interior = (p[:, 0] > 1e-9) & (p[:, 0] < 1 - 1e-9) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
    assert float(np.linalg.norm(final[interior] - expected[interior]) / np.linalg.norm(expected[interior])) < 3e-2


def test_grid_stencil_preserves_complex():
    """The grid stencil must preserve a complex field (like the triangle path) — not silently drop the
    imaginary part. FDM has no first-class complex solve, but the operator itself must stay honest:
    Δ(sin πx + i cos πy) = −π²(sin πx + i cos πy)."""
    from jno.differential_operators import DifferentialOperators as D

    d = _structured(0.0, 0.0, 1.0, 1.0, size=0.04)
    grid = d.mesh_connectivity["grid"]
    pts = jnp.asarray(np.asarray(d.mesh_connectivity["points"]))
    tris = jnp.asarray(d.mesh_connectivity["triangles"])
    p = np.asarray(d.mesh_connectivity["points"])[:, :2]
    uc = jnp.asarray(np.sin(np.pi * p[:, 0]) + 1j * np.cos(np.pi * p[:, 1]), dtype=jnp.complex128)
    lap = np.asarray(D.compute_fd_laplacian_2d_simple(uc, pts, tris, (0, 1), grid=grid))
    assert np.iscomplexobj(lap)  # imaginary part not dropped
    analytic = -(np.pi**2) * np.asarray(uc)
    interior = (p[:, 0] > 1e-9) & (p[:, 0] < 1 - 1e-9) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
    assert np.max(np.abs(lap[interior] - analytic[interior])) < 5e-2  # both real and imag parts correct


# ─────────────────────────────────────────────────────────────────────────────
# fail-loud scope limits (v1: 2-D axis-aligned Shape.rect only)
# ─────────────────────────────────────────────────────────────────────────────


def test_structured_rejects_disk():
    with pytest.raises((ValueError, NotImplementedError)):
        jno.domain(jno.Shape.disk(0.0, 0.0, 1.0, size=0.1), structured=True)


def test_structured_rejects_3d_box():
    with pytest.raises((ValueError, NotImplementedError)):
        jno.domain(jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.2), structured=True)


def test_structured_rejects_composite():
    shape = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1) - jno.Shape.disk(0.5, 0.5, 0.2)
    with pytest.raises((ValueError, NotImplementedError)):
        jno.domain(shape, structured=True)


def test_structured_rejects_callable_size():
    shape = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=lambda x, y: 0.1)
    with pytest.raises(NotImplementedError):
        jno.domain(shape, structured=True)


def test_structured_rejects_non_shape():
    with pytest.raises(ValueError):
        jno.domain(box(0.0, 0.0, 1.0, 1.0), structured=True)
