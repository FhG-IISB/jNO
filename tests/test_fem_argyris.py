"""Argyris C¹ quintic triangle (21 DOF) — the conforming biharmonic element — through ``jno.fem``.

Argyris is the first **C¹-conforming** element in jno.fem: across a shared edge both the trace ``u`` and
the normal derivative ``∂u/∂n`` are continuous, so ``∫Δu·Δv`` is a *convergent* biharmonic discretisation
(unlike C⁰ Lagrange/Hermite, where it is non-conforming). It routes through the non-nodal assembler with
a 21-DOF layout (6 per vertex: value + gradient + Hessian; 1 per edge: normal derivative) and the
affine-equivalence ``M(cell)`` DOF-transform (Kirby 2018), the globally-oriented edge-normal DOF being the
ingredient that makes it C¹ on an *unstructured* mesh (the reduced-quintic Bell element fails exactly here).

The decisive gate is the **solve**, not energy identities: a wrong cross-cell transform passes every
per-element energy check to machine precision while being globally broken (the Bell failure mode). So the
primary test recovers an exact biharmonic solution on an **unstructured** mesh; energy identities are kept
only as a fast necessary-condition sanity layer.

References: Argyris–Fried–Scharpf 1968 (the TUBA-6 element); R.C. Kirby, *A general approach to transforming
finite elements*, SMAI J. Comput. Math. 4 (2018).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
laplacian = jno.np.laplacian


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


_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


def _eval(out):
    """A jno.fem solve to a concrete array (linear is already an array; nonlinear/transient is a node)."""
    if isinstance(out, jax.Array):
        return np.asarray(out)
    crux = jno.core([out.mean], domain=_DUMMY)
    return np.asarray(crux.eval([out]))


def _symbols(d, **bind):
    u, phi = d.fem_symbols(space="Argyris")
    return u.bind(**bind), phi.bind(**bind), u, phi


def _topology(d):
    from jno.utils.solver.fem_topology import BASIX_TRIANGLE_EDGES, build_edge_topology

    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    top = build_edge_topology(cells, BASIX_TRIANGLE_EDGES)
    return pts, cells, top


def _argyris_dofs(d, fns):
    """Global Argyris DOF vector of a function. ``fns = (u, ux, uy, uxx, uxy, uyy)`` callables of (x, y).
    Layout: per vertex ``[u, ∂x, ∂y, ∂xx, ∂xy, ∂yy]`` (``6·v+k``), then ``∂u/∂n`` per global edge at base
    ``6·n_verts`` (global normal ``n = R90·(P[hi]-P[lo])``). Returns ``(c, pts, top, nv, ne)``."""
    pts, _cells, top = _topology(d)
    nv, ne = pts.shape[0], top.n_edges
    u, ux, uy, uxx, uxy, uyy = fns
    c = np.zeros(6 * nv + ne)
    for v in range(nv):
        x, y = pts[v]
        c[6 * v : 6 * v + 6] = [u(x, y), ux(x, y), uy(x, y), uxx(x, y), uxy(x, y), uyy(x, y)]
    for e in range(ne):
        lo, hi = (int(x) for x in top.edge_vertices[e])
        mx, my = 0.5 * (pts[lo] + pts[hi])
        dvec = pts[hi] - pts[lo]
        n = np.array([-dvec[1], dvec[0]])
        n = n / np.linalg.norm(n)
        c[6 * nv + e] = n[0] * ux(mx, my) + n[1] * uy(mx, my)
    return c, pts, top, nv, ne


def _value_dofs(arr, nv):
    """Field values at the mesh vertices = the value DOFs (every 6th DOF in the vertex block)."""
    return np.asarray(arr)[..., 6 * np.arange(nv)]


def _boundary(pts, top):
    """Boundary vertices (on the unit-square edge) and boundary edges (single-incidence)."""
    on_b = (pts[:, 0] < 1e-9) | (pts[:, 0] > 1 - 1e-9) | (pts[:, 1] < 1e-9) | (pts[:, 1] > 1 - 1e-9)
    bverts = set(np.where(on_b)[0].tolist())
    counts = np.bincount(np.asarray(top.cell_edges).reshape(-1), minlength=top.n_edges)
    bedges = [int(e) for e in np.where(counts == 1)[0]]
    return bverts, bedges


# ---------------------------------------------------------------------------
# Fast sanity: assembly energy identity (necessary, NOT sufficient — see module docstring).
# ---------------------------------------------------------------------------


def test_argyris_biharmonic_energy_quartic():
    """``cᵀK c = ∫(Δu)²`` for ``u = x⁴`` (⇒ ``Δu = 12x²``, ``∫(12x²)² = 144/5 = 28.8`` on the unit square).
    Exact iff the assembly + M(cell) transform are correct. (A fast necessary check; the solve is the gate.)"""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi, *_ = _symbols(d, x=xi, y=yi)
    K = _dense(jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi])]).A)
    assert np.allclose(K, K.T, atol=1e-9), "biharmonic stiffness must be symmetric"
    c, _pts, _top, nv, ne = _argyris_dofs(
        d,
        (
            lambda x, y: x**4,
            lambda x, y: 4 * x**3,
            lambda x, y: 0.0 * x,
            lambda x, y: 12 * x**2,
            lambda x, y: 0.0 * x,
            lambda x, y: 0.0 * x,
        ),
    )
    assert K.shape == (6 * nv + ne, 6 * nv + ne), "ndof must be 6*n_vertices + n_edges"
    assert abs(float(c @ K @ c) - 28.8) < 1e-7, "Argyris biharmonic energy != ∫(Δu)² (wrong assembly/M(cell)?)"


# ---------------------------------------------------------------------------
# THE GATE: exact biharmonic recovery on an UNSTRUCTURED mesh (the Bell-killer at the solve level).
# ---------------------------------------------------------------------------


def _recover_exact(u_exact_dofs, f_node, mesh_size):
    """Assemble the conforming biharmonic ``∫Δu·Δv = ∫f v``, pin EVERY boundary DOF (all 6 per boundary
    vertex + the normal-derivative per boundary edge) to the exact solution, solve, return ``(sol, c_exact)``.
    A non-C¹ (Bell-style) transform makes ``c_exact`` NOT satisfy the discrete equation → recovery fails."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi, *_ = _symbols(d, x=xi, y=yi)
    f = f_node(xi, yi)
    fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi])
    K, b = _dense(fem.A).copy(), np.asarray(fem.b).reshape(-1).copy()
    c_exact, pts, top, nv, _ = _argyris_dofs(d, u_exact_dofs)
    bverts, bedges = _boundary(pts, top)
    pinned = [6 * v + k for v in bverts for k in range(6)] + [6 * nv + e for e in bedges]
    for dof in pinned:  # symmetric elimination
        b -= K[:, dof] * c_exact[dof]
    for dof in pinned:
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        b[dof] = c_exact[dof]
    return np.linalg.solve(K, b), c_exact


def test_argyris_biharmonic_recovers_quartic_exactly():
    """PRIMARY GATE. ``u* = x⁴ + y⁴`` ⇒ ``Δ²u* = 48``. ``u*`` lies in the Argyris space, so with every
    boundary DOF clamped to the exact data the conforming solve recovers ``u*`` to machine precision on an
    unstructured mesh. This is exactly what the (non-C¹) Bell attempt could NOT do."""
    fns = (
        lambda x, y: x**4 + y**4,
        lambda x, y: 4 * x**3,
        lambda x, y: 4 * y**3,
        lambda x, y: 12 * x**2,
        lambda x, y: 0.0 * x,
        lambda x, y: 12 * y**2,
    )
    sol, c_exact = _recover_exact(fns, lambda x, y: 48.0 + 0.0 * x, 0.32)
    rel = float(np.linalg.norm(sol - c_exact) / np.linalg.norm(c_exact))
    assert rel < 1e-8, f"Argyris biharmonic did not recover x⁴+y⁴ exactly (C¹ broken?): rel {rel:.2e}"


def test_argyris_biharmonic_recovers_biharmonic_polynomial_exactly():
    """Harder discriminator: ``u* = x⁴ − 6x²y² + y⁴`` is HARMONIC (``Δu*=0`` ⇒ ``Δ²u*=0``, ``f=0``), so the
    solution is driven ENTIRELY by the clamped boundary data (value + normal derivative). A wrong cross-cell
    transform or BC has nothing to hide behind here. Exact recovery on an unstructured mesh."""
    fns = (
        lambda x, y: x**4 - 6 * x**2 * y**2 + y**4,
        lambda x, y: 4 * x**3 - 12 * x * y**2,
        lambda x, y: -12 * x**2 * y + 4 * y**3,
        lambda x, y: 12 * x**2 - 12 * y**2,
        lambda x, y: -24 * x * y,
        lambda x, y: -12 * x**2 + 12 * y**2,
    )
    sol, c_exact = _recover_exact(fns, lambda x, y: 0.0 * x, 0.32)
    rel = float(np.linalg.norm(sol - c_exact) / np.linalg.norm(c_exact))
    assert rel < 1e-8, f"Argyris biharmonic did not recover the harmonic quartic exactly: rel {rel:.2e}"


# ---------------------------------------------------------------------------
# Real-world feature combinations through fem.solve(): DSL clamped BC, convergence, nonlinear, transient.
# ---------------------------------------------------------------------------


def test_argyris_clamped_dsl_recovers_exactly():
    """Argyris + the **explicit** clamped DSL BC: the deflection ``u(region)-g`` pins the value trace and the
    rotation ``u.dn(region)-h`` pins ``∂u/∂n``. With ``u* = x⁴+y⁴`` clamped to its exact trace (``∂u*/∂n`` is
    the constant 0 on the ``x=0``/``y=0`` edges, 4 on ``x=1``/``y=1``) the solve recovers ``u*`` exactly —
    validating that the value + rotation pins together reproduce a full clamped trace."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.32)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xbo, ybo, _ = d.variable("bottom", split=True)
    ui, vi, u, _phi = _symbols(d, x=xi, y=yi)
    f = 48.0 + 0.0 * xi
    fem = jno.fem(
        [
            laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi,
            u(xb, yb) - (xb**4 + yb**4),  # deflection on the full boundary
            u.dn(xl, yl) - 0.0,
            u.dn(xbo, ybo) - 0.0,  # ∂u*/∂n = 0 on x=0 and y=0
            u.dn(xr, yr) - 4.0,
            u.dn(xt, yt) - 4.0,  # ∂u*/∂n = 4 on x=1 and y=1
        ]
    )
    sol = _eval(fem.solve()).reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :2]
    nv = pts.shape[0]
    uh = _value_dofs(sol, nv)
    ue = pts[:, 0] ** 4 + pts[:, 1] ** 4
    rel = float(np.linalg.norm(uh - ue) / np.linalg.norm(ue))
    assert rel < 1e-7, f"Argyris DSL clamped solve did not recover x⁴+y⁴: rel {rel:.2e}"


def test_argyris_biharmonic_converges():
    """Argyris + clamped DSL BC on a smooth (non-polynomial) solution: ``Δ²u = 4π⁴ sin(πx)sin(πy)`` with
    ``u* = sin(πx)sin(πy)``, clamped to the exact ``g = u*``. The vertex error decreases under refinement on
    unstructured meshes — the convergence the C⁰ Hessian assembly cannot deliver and the mixed method
    matches (test_fem_hessian)."""
    errs = []
    for ms in (0.34, 0.22, 0.15):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi, u, _phi = _symbols(d, x=xi, y=yi)
        f = 4.0 * PI**4 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
        g = jno.np.sin(PI * xb) * jno.np.sin(PI * yb)
        fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi, u(xb, yb) - g])
        sol = _eval(fem.solve()).reshape(-1)
        pts = np.asarray(d.mesh.points)[:, :2]
        nv = pts.shape[0]
        uh = _value_dofs(sol, nv)
        ue = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
        errs.append(float(np.linalg.norm(uh - ue) / np.linalg.norm(ue)))
    assert errs[0] < 5e-3, f"coarse Argyris biharmonic too inaccurate: {errs[0]:.3e}"
    assert np.all(np.diff(errs) < 0), f"Argyris biharmonic error not decreasing under refinement: {errs}"


def test_argyris_nonlinear_biharmonic_recovers():
    """Argyris + nonlinear Newton: ``Δ²u + u³ = f`` with manufactured ``u* = sin(πx)sin(πy)``,
    ``f = 4π⁴u* + u*³``, clamped to ``g = u*``. The Newton solve recovers ``u*`` — the nonlinear path
    (matrix-free residual/Jacobian) carries the C¹ element for free."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi, u, _phi = _symbols(d, x=xi, y=yi)
    ss = jno.np.sin(PI * xi) * jno.np.sin(PI * yi)
    f = 4.0 * PI**4 * ss + ss**3
    g = jno.np.sin(PI * xb) * jno.np.sin(PI * yb)
    fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) + (ui * ui * ui) * vi - f * vi, u(xb, yb) - g])
    assert not fem.is_linear, "u³ term must make the biharmonic problem nonlinear"
    sol = _eval(fem.solve()).reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :2]
    nv = pts.shape[0]
    uh = _value_dofs(sol, nv)
    ue = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    rel = float(np.linalg.norm(uh - ue) / np.linalg.norm(ue))
    assert rel < 1e-2, f"Argyris nonlinear biharmonic did not recover u*: rel {rel:.3e}"


def test_argyris_transient_biharmonic_dissipates():
    """Argyris + transient (a real fem.solve() time-stepping combination): the biharmonic heat flow
    ``u_t + Δ²u = 0`` with homogeneous clamped BC is energy-dissipative — ``½ d/dt‖u‖² = -‖Δu‖² ≤ 0`` — so
    the discrete vertex energy must decay monotonically from the IC and stay finite. Exercises the transient
    block (mass split + IC projection) over the C¹ element."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.16, time=(0.0, 2.0e-4, 11))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi, u, _phi = _symbols(d, x=xi, y=yi, t=ti)
    psi0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.t * vi + laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]), u(xb, yb) - 0.0, u(ci[0], ci[1]) - psi0])
    assert fem.is_transient, "a u_t + Δ²u form must build a transient problem"
    traj = _eval(fem.solve())  # (n_steps, ndof)
    pts = np.asarray(d.mesh.points)[:, :2]
    nv = pts.shape[0]
    norms = np.array([float(np.linalg.norm(_value_dofs(traj[k], nv))) for k in range(traj.shape[0])])
    assert np.all(np.isfinite(norms)), "transient biharmonic blew up (non-finite)"
    # monotone non-increasing energy (allow a tiny numerical slack), and genuine decay from the IC
    assert np.all(np.diff(norms) < 1e-9), f"biharmonic heat flow energy must dissipate, got {norms}"
    assert norms[-1] < 0.999 * norms[0], f"transient did not evolve/decay: {norms[0]:.3e} -> {norms[-1]:.3e}"


def test_argyris_clamped_bc_frees_boundary_curvature():
    """The **proper clamped BC**: `u(region) - g` pins value + ∂u/∂n but leaves the boundary curvature
    ∂²u/∂n² *free* (a natural BC). Manufactured `u* = sin²(πx)sin²(πy)` is homogeneous-clamped
    (`u* = ∂u*/∂n = 0` on ∂Ω) but has **nonzero** boundary curvature. So the proper clamped BC converges to
    `u*`, whereas pinning the *full trace* (`∂²u/∂n² = 0`, the old over-constraint) solves a different,
    over-stiff problem and does **not** — this is exactly what distinguishes a true clamped plate."""

    def _setup(ms):
        d = jno.domain(box(0, 0, 1, 1), mesh_size=ms)
        xi, yi, _ = d.variable("interior", split=True)
        a = 2 * PI
        c2x, c2y = jno.np.cos(a * xi), jno.np.cos(a * yi)
        px, py = (1 - c2x) / 2, (1 - c2y) / 2
        f = 8 * PI**4 * (c2x * (c2y - py) - px * c2y)  # Δ²(sin²πx·sin²πy)
        u, phi = d.fem_symbols(space="Argyris")
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        pts = np.asarray(d.mesh.points)[:, :2]
        ue = (np.sin(PI * pts[:, 0]) ** 2) * (np.sin(PI * pts[:, 1]) ** 2)
        return d, xi, yi, u, ui, vi, f, pts, ue

    def _err_proper(ms):  # clamped BC via the DSL: deflection u(reg)-g + rotation u.dn(reg)-h (frees ∂²u/∂n²)
        d, xi, yi, u, ui, vi, f, pts, ue = _setup(ms)
        xb, yb, _ = d.variable("boundary", split=True)
        fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi, u(xb, yb) - 0.0, u.dn(xb, yb) - 0.0])
        uh = _value_dofs(_eval(fem.solve()).reshape(-1), pts.shape[0])
        return float(np.linalg.norm(uh - ue) / np.linalg.norm(ue))

    def _err_full_trace(ms):  # manual: pin ALL boundary DOFs to 0 (the old over-constraint, incl. ∂²/∂n²)
        d, xi, yi, u, ui, vi, f, pts, ue = _setup(ms)
        fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi])
        K, b = _dense(fem.A).copy(), np.asarray(fem.b).reshape(-1).copy()
        _p, _c, top = _topology(d)
        nv = pts.shape[0]
        bverts, bedges = _boundary(pts, top)
        pinned = [6 * v + k for v in bverts for k in range(6)] + [6 * nv + e for e in bedges]
        for dof in pinned:
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            b[dof] = 0.0
        return float(np.linalg.norm(np.linalg.solve(K, b)[6 * np.arange(nv)] - ue) / np.linalg.norm(ue))

    ep_coarse, ep_fine = _err_proper(0.3), _err_proper(0.2)
    ef_fine = _err_full_trace(0.2)
    assert ep_fine < ep_coarse, f"proper clamped must converge to sin²·sin²: {ep_coarse:.3e} -> {ep_fine:.3e}"
    assert ep_fine < 0.05, f"proper clamped too inaccurate: {ep_fine:.3e}"
    # the over-constrained full-trace pin solves a DIFFERENT (over-stiff) problem -> clearly larger error
    assert ef_fine > 3 * ep_fine, f"proper clamped ({ep_fine:.3e}) should clearly beat the full-trace pin ({ef_fine:.3e})"


def test_argyris_dynamic_plate_conserves_energy():
    """Argyris + **second-order-in-time** (a vibrating Kirchhoff plate): ``w_tt + Δ²w = 0``, clamped, released
    from rest. The augmented [w, v] block integrates by the trapezoidal (Newmark average-acceleration) rule,
    which conserves energy for the undamped system — so the discrete energy ``E = ½(vᵀM₂v + wᵀKw)`` stays
    constant. This is the BC-robust check (a manufactured solution would expose the clamped-BC over-pinning);
    it validates the non-nodal ``u_tt`` (dynamic plate) path, previously nodal-Lagrange only."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.24, time=(0.0, 0.03, 13))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    w0 = (jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])) ** 2  # a clamped initial deflection, at rest
    fem = jno.fem([ui.tt * vi + laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]), u(xb, yb) - 0.0, u(ci[0], ci[1]) - w0])
    assert fem.is_transient, "a w_tt + Δ²w form must build a (second-order) transient problem"

    def direct_theta(block, args, save_ts):
        """Trapezoidal θ-method with a DENSE DIRECT solve (the h⁻⁴-conditioned biharmonic defeats the
        default matrix-free Krylov step, just like Cahn–Hilliard). Factors (M + θΔt A) once."""
        import jax.numpy as jnp
        from jax import lax

        M = jnp.asarray(block.M.todense() if hasattr(block.M, "todense") else block.M)
        A = jnp.asarray(block.A.todense() if hasattr(block.A, "todense") else block.A)
        c = jnp.zeros((M.shape[0],)) if block.affine_bias is None else jnp.asarray(block.affine_bias).reshape(-1)
        th, dt = float(block.metadata.get("theta", 1.0)), float(block.dt)
        lhs, rhs_mat = M + th * dt * A, M - (1.0 - th) * dt * A
        grid = jnp.asarray(save_ts)

        def step(y, _t):
            yn = jnp.linalg.solve(lhs, rhs_mat @ y + dt * c)
            return yn, yn

        _, ys = lax.scan(step, jnp.asarray(block.state0).reshape(-1), grid[1:])
        return ys

    traj = _eval(fem.solve(direct_theta))  # (n_steps, 2N), state = [w; v]

    # raw mass M₂ and biharmonic stiffness K (fresh space-only symbols) for the discrete energy
    u2, p2 = d.fem_symbols(space="Argyris")
    ux, vx = u2.bind(x=xi, y=yi), p2.bind(x=xi, y=yi)
    M2 = _dense(jno.fem([ux * vx]).A)
    K = _dense(jno.fem([laplacian(ux, [xi, yi]) * laplacian(vx, [xi, yi])]).A)
    n = M2.shape[0]
    assert traj.shape[1] == 2 * n, "second-order state must be the augmented [w; v] of size 2N"
    w, v = traj[:, :n], traj[:, n:]
    E = 0.5 * (np.einsum("ki,ij,kj->k", v, M2, v) + np.einsum("ki,ij,kj->k", w, K, w))
    assert np.all(np.isfinite(E)) and E[0] > 0, "plate energy must be finite and positive"
    drift = float(np.max(np.abs(E - E[0])) / E[0])
    assert drift < 1e-3, f"undamped plate energy must be conserved (trapezoidal): drift {drift:.3e}, E={E}"
    assert float(np.max(np.abs(v))) > 1e-6, "the plate must actually vibrate (nonzero velocity)"
