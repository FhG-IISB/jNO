"""``jno.fdm`` — finite-difference PDE solver (strong-form sibling of ``jno.fem``).

Run with x64 (the solve accumulates in float64)."""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


_COT = "finite_difference:cotangent"  # whole-Laplacian FD stencil; one .laplacian term


def _nodes(d):
    return np.asarray(d.mesh_connectivity["points"])[:, :2]


def _poisson_homogeneous(mesh_size):
    """-Δu = f on [0,1]², u=0 on ∂Ω, exact u = sin(πx)sin(πy). Returns rel-L2 error."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    sol = jno.fdm([-ui.laplacian(x, y, scheme=_COT) - f, u(xb, yb) - 0.0]).solve()
    return float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))


def test_poisson_homogeneous_dirichlet():
    assert _poisson_homogeneous(0.06) < 1e-2


def test_poisson_convergence_under_refinement():
    """Refining the mesh reduces the FD error (consistency)."""
    errs = [_poisson_homogeneous(h) for h in (0.10, 0.06, 0.035)]
    assert errs[0] > errs[1] > errs[2], f"not monotonically converging: {errs}"
    assert errs[2] < 3e-3


def test_matches_fem_on_same_mesh():
    """The FD solution agrees with the FE solution to FD-discretization accuracy."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    import jno.jnp_ops as jnn

    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    u_fd = np.asarray(jno.fdm([-ui.laplacian(x, y, scheme=_COT) - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
    # both solve the same BVP; the FD field is in the analytic ballpark
    assert float(np.linalg.norm(u_fd - exact) / np.linalg.norm(exact)) < 1e-2


def test_differentiable_for_inverse_problems():
    """The solve is differentiable w.r.t. a parameter in the residual (source scale), and the
    gradient points toward the true value — the requirement for composing into jno.core."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    p = _nodes(d)
    obs = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))

    def loss(scale):
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        f_base = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
        sol = jno.fdm([-ui.d2(x) - ui.d2(y) - scale * f_base, u(xb, yb) - 0.0]).solve()
        return jnp.mean((jnp.asarray(sol).reshape(-1) - obs) ** 2)

    g = float(jax.grad(loss)(1.5))
    assert np.isfinite(g)
    assert g > 0.0, "at scale=1.5 (> true 1.0) the loss must increase with scale"
    assert float(loss(1.0)) < float(loss(1.5)), "scale=1.0 (truth) should beat an off value"


def test_nonlinear_reaction_diffusion():
    """Nonlinear MMS: -Δu + u³ = f with exact u = sin(πx)sin(πy). Reuses jno.solve.newton via
    the same .solve() call — a linear residual would converge in one step; this one iterates."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    import jno.jnp_ops as jnn

    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    s_xy = jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    f = 2 * np.pi**2 * s_xy + s_xy**3  # -Δ(sin sin) + (sin sin)³
    sol = jno.fdm([-ui.laplacian(x, y, scheme=_COT) + ui**3 - f, u(xb, yb) - 0.0]).solve()
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 1e-2


def test_transient_differentiable_for_inverse():
    """The transient march differentiates w.r.t. a parameter (diffusivity) — time-dependent inverse."""
    import jno.jnp_ops as jnn

    T = 0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08, time=(0.0, T, 200))
    p = _nodes(d)
    target = jnp.asarray(np.exp(-2 * 0.05 * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)

    def loss(nu):
        traj = jno.fdm(
            [
                ui.t - nu * (ui.d2(x) + ui.d2(y)),
                u(xb, yb) - 0.0,
                u(xi, yi) - jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi),
            ]
        ).solve()
        return jnp.mean((jnp.asarray(traj)[-1] - target) ** 2)

    g = float(jax.grad(loss)(0.05))
    assert np.isfinite(g)
    assert float(loss(0.05)) < float(loss(0.07)), "true diffusivity should beat an off value"


def test_domain_unknown_is_valued_nodal_field():
    """`domain.unknown()` → a valued P1 nodal field sized to the mesh (the strong-form counterpart to
    the symbolic `fem_symbols()` trial); supports strong-form derivatives and `.bind()` like a fem trial."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    n_nodes = _nodes(d).shape[0]
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    assert u.model.module.value.shape == (n_nodes,)  # one DOF per mesh node
    assert hasattr(u, "d") and hasattr(u, "d2")  # supports strong-form derivatives (u.d(x), u.d2(x))
    ui = u.bind(x=x, y=y)  # .bind like fem symbols (u.bind(x=xi, y=yi))
    assert hasattr(ui, "x") and hasattr(ui, "d2")  # bound view supports fem-style authoring


# ==========================================================================
# constraint-list front-end (fem-style: jno.fdm([...]) with u = domain.unknown())
# ==========================================================================


def test_constraint_list_poisson():
    """fem-style authoring: jno.fdm([-Δu - f, u(xb,yb) - 0]) with u = domain.unknown(). No `scheme=`
    anywhere — a nodal field's `.d`/`.d2` default to finite differences (autodiff is meaningless on a
    discrete field), so `ui.d2(x)` is the FD second derivative."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    sol = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve()  # no scheme= → FD by default
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 3e-2


def test_constraint_list_inhomogeneous_dirichlet():
    """u = x²+y², -Δu = -4, with inhomogeneous Dirichlet g(x,y)=x²+y² as a constraint (validates g-eval)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    exact = p[:, 0] ** 2 + p[:, 1] ** 2
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    sch = "finite_difference"
    sol = jno.fdm([-ui.d2(x, scheme=sch) - ui.d2(y, scheme=sch) + 4.0, u(xb, yb) - (xb**2 + yb**2)]).solve()
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 1e-2


def test_constraint_list_transient_heat():
    """fem-style transient authoring: the IC is a `u(xi, yi) - u0` constraint (NOT a config arg), and
    t_span/step-count come from domain.time. u_t = ν Δu with homogeneous Dirichlet → e^(−2νπ²t)·u0."""
    import jno.jnp_ops as jnn

    nu, T = 0.05, 0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06, time=(0.0, T, 200))
    p = _nodes(d)
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    sch = "finite_difference"
    traj = np.asarray(
        jno.fdm(
            [
                ui.t - nu * (ui.d2(x, scheme=sch) + ui.d2(y, scheme=sch)),  # u_t = ν Δu
                u(xb, yb) - 0.0,  # Dirichlet
                u(xi, yi) - jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi),  # IC
            ]
        ).solve()
    )
    exact = np.exp(-2 * nu * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert traj.shape[1] == p.shape[0]
    assert float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact)) < 2e-2


def test_constraint_list_transient_requires_ic():
    """Guard: a `u.t` term in the PDE with no `u(initial) - u0` condition is a clear ValueError
    (the IC is found from the constraints the same way jno.fem does it, never a config flag)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15, time=(0.0, 0.5, 50))
    x, y, t = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    with pytest.raises(ValueError, match="no initial condition"):
        jno.fdm([ui.t - 0.05 * (ui.d2(x, scheme="finite_difference") + ui.d2(y, scheme="finite_difference"))])


# ==========================================================================
# constraint-list Neumann flux BCs (ui.d(n, scheme) - h; n = domain.variable(reg, normals=True))
# ==========================================================================


def _mixed_dirichlet_neumann(mesh_size, exact_fn, du_dn_right):
    """Solve -Δu = 0 with Dirichlet on left/bottom/top (u = exact) and Neumann ∂u/∂n = h on the right
    edge, authored fem-style: the flux is `ui.d(n, scheme) - h` with n = domain.variable(reg, normals=True).
    Returns rel-L2 vs the (harmonic) exact solution."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    p = _nodes(d)
    exact = exact_fn(p[:, 0], p[:, 1])
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    nr = d.variable("right", normals=True)  # single outward-normal Variable for the right edge
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    sch = "finite_difference"
    sol = jno.fdm(
        [
            -ui.d2(x, scheme=sch) - ui.d2(y, scheme=sch),  # -Δu = 0 (harmonic exact)
            u(xl, yl) - exact_fn(xl, yl),  # Dirichlet left
            u(xb, yb) - exact_fn(xb, yb),  # Dirichlet bottom
            u(xt, yt) - exact_fn(xt, yt),  # Dirichlet top
            ui.d(nr, scheme=sch) - du_dn_right,  # Neumann right: ∂u/∂n = h
        ]
    ).solve()
    return float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))


def test_neumann_linear_exact():
    """u = x + 2y is linear ⇒ the FD gradient and Laplacian are exact ⇒ mixed D+N recovers it to solver
    tolerance. This pins the flux-row correctness (normal orientation, unit-normalization, ∇u·n = h)."""
    err = _mixed_dirichlet_neumann(0.1, lambda x, y: x + 2 * y, du_dn_right=1.0)  # ∂u/∂n = ∂u/∂x = 1 on x=1
    assert err < 1e-4, f"linear mixed D+N should be near-exact, got {err}"


def test_neumann_convergence_harmonic():
    """u = x² − y² is harmonic (−Δu = 0), ∂u/∂n = 2x = 2 on the right edge; the error decreases under
    refinement."""
    errs = [_mixed_dirichlet_neumann(h, lambda x, y: x**2 - y**2, du_dn_right=2.0) for h in (0.1, 0.06, 0.035)]
    assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
    assert errs[2] < 5e-3


def _sin_neumann(h, interior):
    """−Δu = f with u = sin(πx/2) sin(πy): Dirichlet 0 on left/bottom/top, ∂u/∂n = 0 on the right."""
    import jno.jnp_ops as jnn

    π = np.pi
    if interior == "structured":
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    else:
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h)
    x, y, _ = d.variable("interior", split=True)
    (xl, yl, _), (xo, yo, _), (xt, yt, _), (xr, yr, _) = (
        d.variable(r, split=True) for r in ("left", "bottom", "top", "right")
    )
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    Δu = ui.laplacian(x, y, scheme=_COT) if interior == "cotangent" else ui.d2(x) + ui.d2(y)
    f = 1.25 * π**2 * jnn.sin(π * x / 2) * jnn.sin(π * y)
    sol = jno.fdm([-Δu - f, u(xl, yl) - 0.0, u(xo, yo) - 0.0, u(xt, yt) - 0.0, ur.d(nr) - 0.0]).solve()
    p = _nodes(d)
    exact = np.sin(π * p[:, 0] / 2) * np.sin(π * p[:, 1])
    return float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))


@pytest.mark.parametrize("interior", ["d2", "cotangent", "structured"])
def test_neumann_is_second_order(interior):
    """The flux row's gradient matches the interior stencil. The cotangent Laplacian never reads the
    area-weighted boundary gradient, so a flux row built on it (first order at a boundary node) capped the
    solve at first order: 2.2e-3 at h = 0.025, rate 0.7. It now uses a quadratic least-squares fit over
    the node's two-ring: 2.3e-3 → 7.4e-4. The default `.d2` (a gradient of that same gradient) keeps it,
    because there it is the consistent closure: 7.5e-3 → 1.8e-3. The structured grid, which used to drop
    the flux row altogether, gives 1.7e-3 → 4.3e-4."""
    e = [_sin_neumann(h, interior) for h in (0.05, 0.025)]
    assert e[1] < 2e-3, f"error at h = 0.025: {e[1]:.2e}"
    assert e[0] / e[1] > 2.8, f"expected close to O(h²): {e}"


def test_robin_linear_exact():
    """Robin ∂u/∂n + α(u − u∞) = 0 on the right edge, α=1, u∞=2: for u = x this reads 1 + (1 − 2) = 0.
    The whole edge equation is written with that edge's boundary tags (`ur = u.bind(x=xr, y=yr)`) — no
    mixing with the interior. Linear ⇒ recovered to solver tolerance, pinning the two-probe
    (a·∇u·n + b) coefficient extraction and the boundary field-value evaluation."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    p = _nodes(d)
    exact = p[:, 0]  # u = x
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    ur = u.bind(x=xr, y=yr)  # edge-bound field for the flux + value terms of the Robin condition
    sol = jno.fdm(
        [
            -ui.d2(x) - ui.d2(y),  # −Δu = 0
            u(xl, yl) - xl,
            u(xb, yb) - xb,
            u(xt, yt) - xt,  # Dirichlet on three edges
            ur.d(nr) + 1.0 * (ur - 2.0),  # Robin on the right: ∂u/∂n + (u − 2) = 0
        ]
    ).solve()
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 1e-4


def test_mixed_dirichlet_neumann_robin():
    """Any mix of BCs composes: Dirichlet (left, top), Neumann (bottom), Robin (right), all on u = x.
    Reports the corner error honestly — the flux/flux corner falls back to the PDE (exact for a linear
    field here)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    p = _nodes(d)
    exact = p[:, 0]  # u = x
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    nb = d.variable("bottom", normals=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    urb = u.bind(x=xb, y=yb)
    ur = u.bind(x=xr, y=yr)
    sol = jno.fdm(
        [
            -ui.d2(x) - ui.d2(y),  # −Δu = 0
            u(xl, yl) - xl,  # Dirichlet left
            u(xt, yt) - xt,  # Dirichlet top
            urb.d(nb) - 0.0,  # Neumann bottom: ∂u/∂n = −∂u/∂y = 0
            ur.d(nr) + 1.0 * (ur - 2.0),  # Robin right
        ]
    ).solve()
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 1e-3


def test_nonlinear_flux_condition():
    """A flux condition need not be affine in ∂u/∂n: the boundary row is evaluated as written and Newton
    solves it. ∂u/∂n + (∂u/∂n)³ = g + g³ (slope 1 + 3s² > 0) with u = 2x + 3y is recovered to rounding.
    It used to raise ("must be affine in the normal derivative")."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    xl, yl, _ = d.variable("left", split=True)
    terms = [-ui.d2(x) - ui.d2(y), u(xl, yl) - (2.0 * xl + 3.0 * yl)]
    for r in ("right", "bottom", "top"):
        xr, yr, _, nx, ny = d.variable(r, normals=True, split=True)
        s, g = u.bind(x=xr, y=yr).d(d.variable(r, normals=True)), 2.0 * nx + 3.0 * ny
        terms.append(s + s**3 - (g + g**3))
    sol = np.asarray(jno.fdm(terms).solve()).reshape(-1)
    p = _nodes(d)
    assert np.abs(sol - (2.0 * p[:, 0] + 3.0 * p[:, 1])).max() < 1e-10


@pytest.mark.slow
def test_transient_neumann_bc():
    """Transient Neumann (insulated) BC via the algebraic-flux march: u = cos(πx)·sin(πy)·e^{−2νπ²t} has
    homogeneous Neumann ∂u/∂n = 0 on left/right (∂u/∂x = 0 there) and Dirichlet u = 0 on top/bottom. The
    Neumann boundary nodes EVOLVE (a zero mass row + the flux constraint, an index-1 DAE) and track the
    analytic solution — the flux + transient combination that jno.fdm used to reject."""
    import jno.jnp_ops as jnn

    nu, T = 0.05, 0.2
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05, time=(0.0, T, 100))
    p = _nodes(d)
    x, y, t = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    nl = d.variable("left", normals=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    ul = u.bind(x=xl, y=yl)
    ur = u.bind(x=xr, y=yr)
    u0 = jnn.cos(np.pi * xi) * jnn.sin(np.pi * yi)
    traj = np.asarray(
        jno.fdm(
            [
                ui.t - nu * (ui.d2(x) + ui.d2(y)),
                ul.d(nl) - 0.0,
                ur.d(nr) - 0.0,  # Neumann (insulated) left + right
                u(xb, yb) - 0.0,
                u(xt, yt) - 0.0,  # Dirichlet bottom + top
                u(xi, yi) - u0,  # initial condition
            ]
        ).solve()
    )
    final = traj[-1]
    expected = np.cos(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) * np.exp(-2 * nu * np.pi**2 * T)
    assert np.all(np.isfinite(final))
    interior = (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)  # all non-Dirichlet nodes (incl. left/right Neumann)
    left_right = ((p[:, 0] < 1e-9) | (p[:, 0] > 1 - 1e-9)) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
    assert float(np.linalg.norm(final[interior] - expected[interior]) / np.linalg.norm(expected[interior])) < 2e-2
    # the Neumann boundary nodes evolve (not pinned) and match the analytic decay
    assert float(np.linalg.norm(final[left_right] - expected[left_right]) / np.linalg.norm(expected[left_right])) < 2e-2


def test_coupled_two_field():
    """A coupled 2-field system — −Δu + v = f_u, −Δv + u = f_v, u = v = 0 on ∂Ω — authored as one PDE
    equation per unknown (equation k drives unknown k). `.solve()` returns `(nf, N)` with each field
    recovered. MMS: u = sin(πx)sin(πy), v = sin(2πx)sin(πy)."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.04)
    p = _nodes(d)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    v = d.unknown()
    ui = u.bind(x=x, y=y)
    vi = v.bind(x=x, y=y)
    u_ex = jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    v_ex = jnn.sin(2 * np.pi * x) * jnn.sin(np.pi * y)
    f_u = 2 * np.pi**2 * u_ex + v_ex  # −Δu_ex = 2π²·u_ex
    f_v = 5 * np.pi**2 * v_ex + u_ex  # −Δv_ex = 5π²·v_ex
    sol = np.asarray(
        jno.fdm(
            [
                -ui.d2(x) - ui.d2(y) + vi - f_u,  # equation for u (block 0)
                -vi.d2(x) - vi.d2(y) + ui - f_v,  # equation for v (block 1)
                u(xb, yb) - 0.0,  # Dirichlet u
                v(xb, yb) - 0.0,  # Dirichlet v
            ]
        ).solve()
    )
    assert sol.shape == (2, p.shape[0])  # (nf, N), one row per field
    uex = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    vex = np.sin(2 * np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert float(np.linalg.norm(sol[0] - uex) / np.linalg.norm(uex)) < 2e-2
    assert float(np.linalg.norm(sol[1] - vex) / np.linalg.norm(vex)) < 5e-2  # v is higher-frequency


def test_coupled_guards():
    """A coupled system needs one PDE equation per unknown, a flux condition belongs to one unknown, and it marches
    first order in time with equation k carrying only its own unknown's `u.t`."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    v = d.unknown()
    ui = u.bind(x=x, y=y)
    vi = v.bind(x=x, y=y)
    with pytest.raises(ValueError, match="one PDE equation per unknown"):  # 2 unknowns, 1 equation
        jno.fdm([-ui.d2(x) - ui.d2(y) + vi, u(xb, yb) - 0.0, v(xb, yb) - 0.0]).solve()

    dt = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 0.5, 50))
    xt, yt, tt = dt.variable("interior", split=True)
    xit, yit, _ = dt.variable("initial", split=True)
    ut = dt.unknown()
    vt = dt.unknown()
    uit = ut.bind(x=xt, y=yt, t=tt)
    vit = vt.bind(x=xt, y=yt, t=tt)
    with pytest.raises(NotImplementedError, match="time derivative of unknown 1"):  # off-diagonal mass
        jno.fdm([vit.t - (uit.d2(xt) + uit.d2(yt)), uit.t - (vit.d2(xt) + vit.d2(yt)), ut(xit, yit) - 0.0])
    with pytest.raises(NotImplementedError, match="first order in time"):  # coupled u.tt
        jno.fdm([uit.t.t - (uit.d2(xt) + uit.d2(yt)) + vit, vit.t - uit, ut(xit, yit) - 0.0, vt(xit, yit) - 0.0])
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    xl, yl, _ = d.variable("left", split=True)
    with pytest.raises(ValueError, match="cannot tell which field"):  # two free fields: whose row is it?
        jno.fdm(
            [
                -ui.d2(x) - ui.d2(y) + vi,
                -vi.d2(x) - vi.d2(y) + ui,
                u(xl, yl) - 0.0,
                v(xl, yl) - 0.0,
                u.bind(x=xr, y=yr).d(nr) + v.bind(x=xr, y=yr).d(nr) - 1.0,
            ]
        ).solve()


@pytest.mark.slow
def test_general_mass_coefficient():
    """A general `c(x)·u.t` mass coefficient (variable material) — extracted via the two-probe
    `c = F(u.t=1) − F(u.t=0)` and carried as `M = diag(c)`. Extraction is exact (constant + coordinate);
    a constant `a·u.t` rescales the effective diffusivity: `a·u̇ = νΔu ⇒ u̇ = (ν/a)Δu`."""
    import jno.jnp_ops as jnn

    nu, T = 0.1, 0.3
    # exact extraction of a coordinate-dependent coefficient
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1, time=(0.0, T, 40))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    p = _nodes(d)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    u0 = jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    s = jno.fdm([(1.0 + 0.5 * jnn.sin(np.pi * x)) * ui.t - nu * (ui.d2(x) + ui.d2(y)), u(xb, yb) - 0.0, u(xi, yi) - u0])
    assert np.max(np.abs(np.asarray(s._mass_coefficient()()) - (1.0 + 0.5 * np.sin(np.pi * p[:, 0])))) < 1e-9

    # constant a=2, ν=0.1 ⇒ effective diffusivity ν/a = 0.05
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06, time=(0.0, T, 60))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    p = _nodes(d)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    u0 = jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    traj = np.asarray(jno.fdm([2.0 * ui.t - nu * (ui.d2(x) + ui.d2(y)), u(xb, yb) - 0.0, u(xi, yi) - u0]).solve())
    exact = np.exp(-2 * (nu / 2.0) * np.pi**2 * T) * (np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    interior = (p[:, 0] > 1e-9) & (p[:, 0] < 1 - 1e-9) & (p[:, 1] > 1e-9) & (p[:, 1] < 1 - 1e-9)
    assert float(np.linalg.norm(traj[-1][interior] - exact[interior]) / np.linalg.norm(exact[interior])) < 2e-2


def test_nonlinear_mass_rejected():
    """A nonlinear mass `c(u)·u.t` (here `u·u.t`) is not supported — the two-probe detects u-dependence
    and fails loud."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15, time=(0.0, 0.5, 50))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    u0 = jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)
    with pytest.raises(ValueError, match="nonlinear mass"):
        jno.fdm([ui * ui.t - (ui.d2(x) + ui.d2(y)), u(xb, yb) - 0.0, u(xi, yi) - u0]).solve()


@pytest.mark.slow
@pytest.mark.parametrize("flux", [2.0, 5.0])
def test_structured_grid_flux_value_is_applied(flux):
    """`boundary_edges` indexes the boundary-node list, but the flux normals read it as global node
    numbers. A gmsh mesh numbers its boundary nodes first, so it worked there by accident; on a structured
    grid no edge node got a normal, and the flux row was silently dropped: ∂u/∂n = 2 and ∂u/∂n = 5 gave
    bit-identical answers. Oracle: u = x² + y + (g − 2)·x, with −Δu = −2 and ∂u/∂n = g at x = 1."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    (xl, yl, _), (xo, yo, _), (xt, yt, _), (xr, yr, _) = (
        d.variable(r, split=True) for r in ("left", "bottom", "top", "right")
    )
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    exact = lambda x, y: x**2 + y + (flux - 2.0) * x  # noqa: E731
    sol = jno.fdm(
        [
            -(ui.d2(x) + ui.d2(y)) + 2.0,
            u(xl, yl) - exact(xl, yl),
            u(xo, yo) - exact(xo, yo),
            u(xt, yt) - exact(xt, yt),
            ur.d(nr) - flux,
        ]
    ).solve()
    p = _nodes(d)
    ref = exact(p[:, 0], p[:, 1])
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - ref) / np.linalg.norm(ref)) < 1e-8


def test_periodic_poisson():
    """A periodic tie `u(left) - u(right)` wraps the structured x-axis (the Nx-node periodic 5-point
    stencil), authored exactly as in jno.fem. MMS: -Δu = 5π²·sin(2πx)sin(πy), periodic in x with
    Dirichlet u=0 in y ⇒ u = sin(2πx)sin(πy). The tie holds to machine precision."""
    import jno.jnp_ops as jnn

    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).structured())
    p = _nodes(d)
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 5 * np.pi**2 * jnn.sin(2 * np.pi * x) * jnn.sin(np.pi * y)
    sol = np.asarray(
        jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xl, yl) - u(xr, yr), u(xb, yb) - 0.0, u(xt, yt) - 0.0]).solve()
    ).reshape(-1)
    exact = np.sin(2 * np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)) < 3e-2
    sx, sy = d.mesh_connectivity["grid"]["shape"]  # the tie holds exactly: left face == right face
    grid_sol = sol.reshape(sx, sy)
    assert float(np.max(np.abs(grid_sol[0, :] - grid_sol[-1, :]))) < 1e-9


def test_a_periodic_problem_does_not_leak_into_the_next_one():
    """A periodic tie belongs to its problem, not to the domain. It used to be written into the domain's shared
    grid descriptor, so a Dirichlet problem built afterwards on the same grid wrapped its stencils too: with
    u = x on the boundary (0 on the left face, 1 on the right) its max error went from 2.1e-3 to 0.90, silently.
    Oracle: -Δu = 2π² sin(πx) sin(πy), u = x on the boundary ⇒ u = sin(πx) sin(πy) + x."""
    import jno.jnp_ops as jnn

    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured())
    p = _nodes(d)
    x, y, _ = d.variable("interior", split=True)
    xa, ya, _ = d.variable("boundary", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)

    def dirichlet_error():
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
        sol = np.asarray(jno.fdm([-ui.xx - ui.yy - f, u(xa, ya) - xa]).solve()).reshape(-1)
        return float(np.abs(sol - (np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) + p[:, 0])).max())

    before = dirichlet_error()
    w = d.unknown()
    wi = w.bind(x=x, y=y)
    f = 5 * np.pi**2 * jnn.sin(2 * np.pi * x) * jnn.sin(np.pi * y)
    jno.fdm([-wi.xx - wi.yy - f, w(xl, yl) - w(xr, yr), w(xb, yb) - 0.0, w(xt, yt) - 0.0]).solve()
    assert not any(d.mesh_connectivity["grid"].get("periodic") or ())
    assert before < 5e-3
    assert dirichlet_error() == pytest.approx(before, rel=1e-9)


def test_periodic_requires_structured():
    """A periodic tie on an unstructured mesh raises — the FD stencil must wrap the grid, which only a
    structured grid can do."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)  # unstructured
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.raises(NotImplementedError, match="STRUCTURED"):
        jno.fdm([-ui.d2(x) - ui.d2(y), u(xl, yl) - u(xr, yr), u(xb, yb) - 0.0, u(xt, yt) - 0.0])


def test_constraint_list_inverse_via_crux():
    """A trainable jno.np.parameter in the constraint list makes jno.fdm([...]).solve() a deferred trace
    node (like fem.solve()) that composes into jno.core — recover a source amplitude from an observed
    field through crux + the parameter's attached optimizer (never a hand-rolled jax.grad loop)."""
    import optax

    import jno.jnp_ops as jnn
    from jno.trace import FunctionCall

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f_base = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    observed = jnp.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - 1.0 * f_base, u(xb, yb) - 0.0]).solve()).reshape(-1)

    s = jno.np.parameter((1,), name="s")
    s.dtype(jnp.float64)
    s.initialize(jax.nn.initializers.constant(2.5))
    s.optimizer(optax.adam(1e-1))
    node = jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()
    assert isinstance(node, FunctionCall), "a trainable parameter must make .solve() a deferred crux node"

    crux = jno.core([(node - observed).mse])  # NO domain= — jno.core infers it from the solve node's graph
    assert crux.domain is d, "jno.core must infer the domain from the solve node in the graph"
    crux.solve(120)
    rec = float(np.asarray(crux.eval([s])).reshape(-1)[0])
    assert abs(rec - 1.0) < 2e-2, f"crux did not recover the source amplitude: s={rec:.4f}"


def test_dirichlet_value_from_nodal_field():
    """A Dirichlet value can be a **known nodal field** (a `jno.np.parameter` carrying data, no
    optimizer) — its per-node values are gathered at the boundary. This is the symbolic path a coupled
    /domain-decomposition solve uses to pin a region to a neighbour's field (no raw arrays). Because the
    field has no optimizer it is data, so `.solve()` stays eager (not a deferred crux node).
    u = x²+y², -Δu = -4, Dirichlet = the field on ∂Ω."""
    import equinox as eqx

    from jno.trace import FunctionCall

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    n = p.shape[0]
    exact = p[:, 0] ** 2 + p[:, 1] ** 2

    g = jno.np.parameter((n,), name="g")  # a nodal data-field (no optimizer)
    g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray(exact))
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    sol = jno.fdm([-ui.d2(x) - ui.d2(y) + 4.0, u(xb, yb) - g]).solve()

    assert not isinstance(sol, FunctionCall), "a data-field Dirichlet value must stay an eager solve"
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 1e-2


# ==========================================================================
# 3-D tetrahedral meshes (interior operators — Tier 1)
# ==========================================================================


def _nodes3(d):
    return np.asarray(d.mesh_connectivity["points"])[:, :3]


def _cube(mesh_size):
    """Unit cube meshed by jno.shape (gmsh tets) — no shapely."""
    return jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=mesh_size).domain()


def _poisson3d(mesh_size, method="cotangent"):
    """-Δu = f on [0,1]³, u=0 on ∂Ω, exact u = sin(πx)sin(πy)sin(πz) ⇒ f = 3π²u. Returns rel-L2 error."""
    d = _cube(mesh_size)
    p = _nodes3(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) * np.sin(np.pi * p[:, 2])
    import jno.jnp_ops as jnn

    x, y, z, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    f = 3 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y) * jnn.sin(np.pi * z)
    # "cotangent" is the whole-Laplacian stencil (one term); "gradient_of_gradient" is the nested per-axis
    # stencil, named explicitly — the plain default sum is now fused into the cotangent Laplacian.
    gog = "finite_difference:area_weighted"
    lap = (
        ui.laplacian(x, y, z, scheme=_COT)
        if method == "cotangent"
        else ui.d2(x, scheme=gog) + ui.d2(y, scheme=gog) + ui.d2(z, scheme=gog)
    )
    sol = jno.fdm([-lap - f, u(xb, yb, zb) - 0.0]).solve()
    return float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))


def test_gradient_3d_shape():
    """The FD gradient on a tet mesh is (N, 3) — the flux dot-product ∇u·n stays dimension-agnostic."""
    d = _cube(0.25)
    g = jno.fdm.gradient(jnp.asarray(_nodes3(d)[:, 0]), d)  # ∇x = (1,0,0)
    assert g.shape == (_nodes3(d).shape[0], 3)


def test_poisson_3d_dirichlet():
    assert _poisson3d(0.1) < 3e-2  # cotangent (P1 Laplace–Beltrami) default


def test_poisson_3d_convergence_under_refinement():
    """Refining the tet mesh reduces the FD error and does so at ~2nd order — the default cotangent
    stencil is the P1 tetrahedral Laplace–Beltrami operator (a small-constant Galerkin solve, unlike the
    first-order gradient-of-gradient)."""
    errs = [_poisson3d(h) for h in (0.20, 0.14, 0.10)]
    assert errs[0] > errs[1] > errs[2], f"not monotonically converging: {errs}"
    assert errs[2] < 3e-2


def test_laplacian_3d_cotangent_beats_grad_of_grad():
    """The 3-D cotangent (P1 FEM) Laplacian is a distinct, materially more accurate operator than the
    local gradient-of-gradient double-difference — on the same tet mesh its Poisson error is several
    times smaller (this is the whole point of wiring it up)."""
    h = 0.12
    cot = _poisson3d(h, method="cotangent")
    gog = _poisson3d(h, method="gradient_of_gradient")
    assert cot < 0.4 * gog, f"cotangent ({cot:.3e}) should be << gradient_of_gradient ({gog:.3e})"


def test_laplacian_on_nodal_field_defaults_to_finite_differences():
    """`.laplacian` on a nodal unknown must take the SAME finite-difference default `.d`/`.d2`/`.dd`
    already take. It used to fall through to Placeholder.laplacian (AD default), and the AD Hessian
    branch has no points to differentiate at, so it died with an opaque AttributeError."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    lap = np.asarray(jno.fdm([-ui.laplacian(x, y) - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
    split = np.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
    assert np.allclose(lap, split), "laplacian(x, y) must equal d2(x) + d2(y) on a nodal field"
    assert float(np.linalg.norm(lap - exact) / np.linalg.norm(exact)) < 3e-2


@pytest.mark.parametrize("method", ["d2", "dd"])
def test_whole_laplacian_subscheme_rejected_on_per_axis_derivative(method):
    """`finite_difference:cotangent` returns the WHOLE Laplacian for any requested dimension, so
    `d2(x, ...) + d2(y, ...)` silently computed 2∇²u and converged to half the right answer. Per-axis
    second derivatives now refuse the sub-scheme and point at `.laplacian`."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    x, y, _ = d.variable("interior", split=True)
    ui = d.unknown().bind(x=x, y=y)
    with pytest.raises(ValueError, match="WHOLE Laplacian|laplacian"):
        getattr(ui, method)(x, scheme="finite_difference:cotangent")


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("sub", ["upwind", "bogus"])
def test_unknown_fd_subscheme_raises(structured, sub):
    """An unknown `finite_difference:<sub>` used to reach the gradient kernel as `method=sub`, whose
    final branch is area-weighted: `":upwind"` silently solved with the central default (bit-identical
    answers on both grids). It now raises and lists the known sub-schemes."""
    if structured:
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured().domain()
    else:
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.raises(ValueError, match=f"Unknown finite-difference sub-scheme '{sub}'"):
        jno.fdm([-ui.d2(x) - ui.d2(y) + ui.d(x, scheme=f"finite_difference:{sub}") - 1.0, u(xb, yb) - 0.0]).solve()


def test_whole_laplacian_subscheme_allowed_on_laplacian():
    """The same sub-scheme is legitimate on `.laplacian`, which takes every coordinate at once so it
    cannot be double-counted — and it is markedly more accurate than the nested gradient-of-gradient
    stencil (named explicitly: the plain default sum is fused into the cotangent Laplacian)."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    rel = lambda sol: float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))
    cot = rel(jno.fdm([-ui.laplacian(x, y, scheme="finite_difference:cotangent") - f, u(xb, yb) - 0.0]).solve())
    gog = "finite_difference:area_weighted"
    nested = rel(jno.fdm([-ui.d2(x, scheme=gog) - ui.d2(y, scheme=gog) - f, u(xb, yb) - 0.0]).solve())
    assert cot < nested / 3, f"cotangent should be several times better: {cot:.3e} vs {nested:.3e}"
    default = rel(jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve())
    assert abs(default - cot) < 1e-8, "the default ui.d2(x) + ui.d2(y) must be fused into the cotangent Laplacian"


def test_every_stencil_adjoint_matches_the_closed_form():
    """The oracle the old test lacked: an adjoint must be RIGHT, not merely finite and positive.

    `u` is linear in the source scale `s`, so `d(Σu)/ds == Σu(s=1)` exactly, for every stencil. Mesh
    0.08 is the one that used to fail: `jno.np.parameter` hardcoded `float32` and `_pde_residual_fn`
    casts the DOF vector to the unknown's dtype, so under x64 the residual rounded to single
    precision on every evaluation. That is a NON-LINEAR operator (measured 6e-08 relative), which
    capped the forward Krylov solve near 1e-05 while its recursive residual claimed 1e-10, and broke
    the adjoint BiCGStab outright: `d(Σu)/ds` came back -1.887e+22 against an exact +8.172e+01. The
    coarser 0.12 mesh happened to survive it, which is why this pins the mesh."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)

    for lap in (
        ui.laplacian(x, y, scheme=_COT),
        ui.d2(x) + ui.d2(y),
        ui.laplacian(x, y, scheme="finite_difference:lsq"),
    ):
        total = lambda s, lap=lap: jnp.sum(  # noqa: E731
            jnp.asarray(jno.fdm([-lap - s * f, u(xb, yb) - 0.0]).solve()).reshape(-1)
        )
        exact = float(total(1.0))
        assert abs(float(jax.grad(total)(1.5)) - exact) / abs(exact) < 1e-6


def test_strong_form_residual_is_exactly_linear():
    """The property whose violation caused the above, asserted directly and cheaply.

    `-Δu - f` with Dirichlet rows is affine in the DOF vector, so `F(a+b) - F(a) - F(b) + F(0)` is
    zero in exact arithmetic. A float32 round trip inside the evaluation is not a small error here:
    ROUNDING IS NON-LINEAR, so it showed up as a 6e-08 defect and drove the Krylov solvers off. The
    raw operators (`jno.fdm.laplacian`) were always clean — only the traced path was not, which is
    why this measures the residual `jno.fdm` actually hands to Newton."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)

    for lap in (
        ui.laplacian(x, y, scheme=_COT),
        ui.d2(x) + ui.d2(y),
        ui.laplacian(x, y, scheme="finite_difference:lsq"),
        ui.d(x) + ui.d(y),  # first derivatives share the same evaluation path
    ):
        sysobj = jno.fdm([-lap - f, u(xb, yb) - 0.0])
        residual = sysobj._pde_residual_fn()
        n = sysobj._Ntot
        a = jax.random.normal(jax.random.PRNGKey(0), (n,))
        b = jax.random.normal(jax.random.PRNGKey(1), (n,))
        ev = lambda v: jnp.asarray(residual(v)).reshape(-1)  # noqa: E731
        defect = ev(a + b) - ev(a) - ev(b) + ev(jnp.zeros((n,)))
        rel = float(jnp.linalg.norm(defect) / jnp.linalg.norm(ev(a)))
        assert rel < 1e-14, f"the strong-form residual must be affine in the DOFs; defect {rel:.2e}"


def test_constraint_list_cotangent_3d():
    """The constraint-list path reaches the 3-D cotangent stencil too, written as the whole Laplacian
    `ui.laplacian(x, y, z, scheme="finite_difference:cotangent")`. The cotangent sub-scheme returns
    ∇²u for every requested dimension, so it must be ONE term — the split −d2(x)−d2(y)−d2(z) would
    triple it, which is why `d2`/`dd` now reject this sub-scheme outright. Matches function-form
    accuracy."""
    import jno.jnp_ops as jnn

    d = _cube(0.14)
    p = _nodes3(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) * np.sin(np.pi * p[:, 2])
    x, y, z, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    f = 3 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y) * jnn.sin(np.pi * z)
    sol = jno.fdm([-ui.laplacian(x, y, z, scheme="finite_difference:cotangent") - f, u(xb, yb, zb) - 0.0]).solve()
    err = float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))
    # ~0.064 at h=0.14 — the function-form cotangent value, and far below the per-direction
    # gradient_of_gradient (~0.27), proving the whole-Laplacian cotangent stencil took effect.
    assert err < 0.1, f"whole-cotangent constraint list should match function-form accuracy, got {err}"


def test_constraint_list_poisson_3d():
    """fem-style authoring in 3-D: jno.fdm([-u.d2(x)-u.d2(y)-u.d2(z)-f, u(xb,yb,zb)-0]). `split=True`
    yields (x, y, z, t) on a 3-D domain — the trailing coord is temporal."""
    import jno.jnp_ops as jnn

    d = _cube(0.14)
    x, y, z, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    p = _nodes3(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) * np.sin(np.pi * p[:, 2])
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    f = 3 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y) * jnn.sin(np.pi * z)
    sol = jno.fdm([-ui.d2(x) - ui.d2(y) - ui.d2(z) - f, u(xb, yb, zb) - 0.0]).solve()
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 0.35


def test_mesh_nodes_in_3d_shape_box():
    """Keystone 3-D containment: `jno.fdm._mesh_nodes_in` resolves a `jno.shape.box` sub-region to the
    exact tetrahedral-mesh node subset via the analytic 3-D `shape.contains` — the production path both
    `_TraceFDM._region_nodes` and `jno.dd._region_mask` route through to turn a geometric sub-region into
    a node set. This is what shapely could never do (it is 2-D only). (Wiring such a region into a *3-D
    coupled solve* additionally needs region-tag support on the base 3-D domain — a separate feature.)"""
    from jno.fdm import _mesh_nodes_in

    d = _cube(0.12)
    p = _nodes3(d)
    core = jno.shape.box(0.3, 0.3, 0.3, 0.7, 0.7, 0.7)
    idx = _mesh_nodes_in(p, core)
    # resolves exactly the analytic 3-D containment ...
    assert np.array_equal(np.sort(idx), np.sort(np.nonzero(core.contains(p))[0]))
    # ... which is the hand-checkable "all three coords in [0.3, 0.7]" box (inclusive within tol)
    hand = np.nonzero(np.all((p >= 0.3 - 1e-9) & (p <= 0.7 + 1e-9), axis=1))[0]
    assert np.array_equal(np.sort(idx), np.sort(hand))
    assert len(idx) > 0, "the central box must capture interior tet nodes"


# ---- 3-D flux BCs (Neumann / Robin on a face — Tier 2) ----


def test_node_normals_3d_face_is_axis():
    """The 3-D flux normals of a cube face are the exact outward axis normal — apex orientation +
    coplanar averaging give (+1,0,0) on the right face and (0,0,1) on the top, for every face node."""
    from jno.fdm import _TraceFDM

    d = _cube(0.2)
    x, y, z, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    sysm = _TraceFDM([-ui.d2(x) - ui.d2(y) - ui.d2(z), u(xb, yb, zb) - 0.0])  # a valid system to reach the method
    for face, axis in (("right", [1.0, 0.0, 0.0]), ("top", [0.0, 0.0, 1.0]), ("front", [0.0, -1.0, 0.0])):
        idx, n = sysm._node_normals(face)
        assert len(idx) > 0
        assert np.allclose(np.asarray(n), np.array(axis), atol=1e-9), f"{face} normal off: {np.asarray(n)[0]}"


def _cube_mixed_flux_3d(mesh_size, exact_fn, du_dn_right):
    """-Δu = 0 on the cube, Dirichlet u = exact on five faces, Neumann ∂u/∂n = h on the right face
    (x=1). Returns rel-L2 vs the exact solution."""
    d = _cube(mesh_size)
    p = _nodes3(d)
    exact = exact_fn(p[:, 0], p[:, 1], p[:, 2])
    x, y, z, _ = d.variable("interior", split=True)
    nr = d.variable("right", normals=True)  # outward normal on the right face (x=1)
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    cons = [-ui.d2(x) - ui.d2(y) - ui.d2(z), ui.d(nr) - du_dn_right]
    for face in ("left", "front", "back", "bottom", "top"):
        xf, yf, zf, _ = d.variable(face, split=True)
        cons.append(u(xf, yf, zf) - exact_fn(xf, yf, zf))  # Dirichlet on the other five faces
    sol = jno.fdm(cons).solve()
    return float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))


def test_neumann_3d_linear_exact():
    """u = x + 2y − z is linear ⇒ the tet FD gradient/Laplacian are exact ⇒ mixed Dirichlet+Neumann on
    the cube recovers it. Right face (x=1) carries ∂u/∂n = ∂u/∂x = 1. Pins the 3-D flux row: apex-
    oriented face normals, unit-normalization, ∇u·n = h."""
    err = _cube_mixed_flux_3d(0.2, lambda x, y, z: x + 2 * y - z, du_dn_right=1.0)
    assert err < 1e-3, f"linear 3-D mixed D+N should be near-exact, got {err}"


def test_robin_3d_linear_exact():
    """Robin ∂u/∂n + α(u − u∞) = 0 on the right face, α=1, u∞=2: for u = x this reads 1 + (1 − 2) = 0.
    The whole face equation is written with that face's tags (ur = u.bind(x=xr,y=yr,z=zr)) — pins the
    two-probe (a·∇u·n + b) extraction and the boundary value evaluation in 3-D."""
    d = _cube(0.2)
    p = _nodes3(d)
    exact = p[:, 0]  # u = x
    x, y, z, _ = d.variable("interior", split=True)
    xr, yr, zr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    ur = u.bind(x=xr, y=yr, z=zr)  # face-bound field for the flux + value terms of the Robin condition
    cons = [-ui.d2(x) - ui.d2(y) - ui.d2(z), ur.d(nr) + 1.0 * (ur - 2.0)]
    for face in ("left", "front", "back", "bottom", "top"):
        xf, yf, zf, _ = d.variable(face, split=True)
        cons.append(u(xf, yf, zf) - xf)  # Dirichlet u = x
    sol = jno.fdm(cons).solve()
    assert float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact)) < 1e-3


def test_flux_dirichlet_shared_edge_precedence_3d():
    """A right-face (Neumann) node that also lies on an adjacent Dirichlet face gets BOTH a flux row and
    a Dirichlet row. The assembly applies Dirichlet last, so Dirichlet wins (the well-posed choice — the
    3-D flux path keeps region-edge nodes rather than dropping them, unlike a 2-D corner). Use an
    inconsistent flux (∂u/∂n = 3 while Dirichlet pins u = x) and confirm the shared right-face edge nodes
    take the Dirichlet value x = 1, not the flux-driven value."""
    d = _cube(0.25)
    p = _nodes3(d)
    x, y, z, _ = d.variable("interior", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    cons = [-ui.d2(x) - ui.d2(y) - ui.d2(z), ui.d(nr) - 3.0]  # deliberately inconsistent flux on the right
    for face in ("left", "front", "back", "bottom", "top"):
        xf, yf, zf, _ = d.variable(face, split=True)
        cons.append(u(xf, yf, zf) - xf)  # Dirichlet u = x
    sol = np.asarray(jno.fdm(cons).solve()).reshape(-1)
    on_right = np.isclose(p[:, 0], 1.0)
    on_adj = np.isclose(p[:, 1], 0) | np.isclose(p[:, 1], 1) | np.isclose(p[:, 2], 0) | np.isclose(p[:, 2], 1)
    shared = on_right & on_adj  # right-face nodes shared with an adjacent Dirichlet face
    assert shared.sum() > 0, "expected right-face edge nodes shared with adjacent faces"
    assert np.max(np.abs(sol[shared] - 1.0)) < 1e-9, "Dirichlet must win at a shared flux/Dirichlet edge node"


# --------------------------------------------------------------------------------------------------
# Newton must not demand a tolerance the FD operator cannot deliver
# --------------------------------------------------------------------------------------------------


def test_every_strong_form_stencil_is_exact_under_x64():
    """This test used to assert the OPPOSITE — that the nested `gradient_of_gradient` second
    derivative carries a 5e-08 precision floor "by construction", nine orders worse than the
    cotangent stencil. That was wrong, and it pinned a bug as a feature: the floor was
    `jno.np.parameter`'s hardcoded `float32`, which `_pde_residual_fn` casts the unknown to on every
    evaluation. Called directly, BOTH stencils always measured ~2e-16; only the traced path was
    noisy, and both were equally noisy there. With the dtype following `jax_enable_x64` every
    stencil is exact, so `_fd_newton_tolerances` leaves Newton's own 1e-8 gate alone."""
    from jno.fdm import _fd_newton_tolerances, _fd_operator_noise, laplacian

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    n = int(np.asarray(d.mesh.points).shape[0])
    probe = jnp.asarray(np.random.default_rng(0).normal(size=n))

    assert _fd_operator_noise(lambda z: laplacian(z, d, method="cotangent"), probe) < 1e-14
    assert _fd_operator_noise(lambda z: laplacian(z, d, method="gradient_of_gradient"), probe) < 1e-14

    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    uu = d.unknown()
    ui = uu.bind(x=x, y=y)
    for lap in (ui.d2(x) + ui.d2(y), ui.laplacian(x, y, scheme=_COT)):
        prob = jno.fdm([-lap - 1.0, uu(xb, yb) - 0.0])
        rf = prob._pde_residual_fn()
        noise = _fd_operator_noise(rf, jnp.zeros(prob._Ntot))
        assert noise < 1e-14, f"the traced residual must be exact under x64; measured {noise:.2e}"
        assert _fd_newton_tolerances(rf, jnp.zeros(prob._Ntot)) == {}, "an exact operator keeps 1e-8"


def test_fd_operator_noise_is_immune_to_nonlinearity_and_to_tracing():
    """The probe leans on a JVP being exactly LINEAR IN ITS TANGENT for any differentiable residual,
    so a genuinely nonlinear residual must still measure ~0 noise (otherwise the rule would loosen
    Newton on every nonlinear problem). And it is eager-only: under a trace it returns 0.0, leaving
    the driver's own defaults untouched rather than inventing a floor."""
    import jax

    from jno.fdm import _fd_newton_tolerances, _fd_operator_noise

    u = jnp.asarray(np.random.default_rng(1).normal(size=64))
    assert _fd_operator_noise(lambda z: z**3, u) < 1e-14, "nonlinearity must not read as noise"
    assert _fd_operator_noise(lambda z: jnp.sin(z) * jnp.exp(0.1 * z), u) < 1e-14

    # an exact operator keeps the driver's defaults (no override dict at all)
    assert _fd_newton_tolerances(lambda z: 2.0 * z, u) == {}

    # under a trace the probe must not raise (the parametric / crux inverse path hits this)
    out = jax.jit(lambda z: jnp.sum(jnp.asarray(list(_fd_newton_tolerances(lambda w: 2.0 * w, z).values()) or [0.0])))(u)
    assert np.isfinite(float(out))


def test_fdm_poisson_converges_without_raising_on_the_default_stencil():
    """End-to-end: the default (nested-FD) Poisson solve must converge and be accurate. Before the
    floor-aware gate this raised `newton_krylov did not converge` — residual 7.0e-05 against a
    1.07e-07 request — while the ANSWER was fine, which is why the suite passed until the Newton
    convergence guard landed."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    sol = jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(xb, yb) - 0.0]).solve()  # must not raise
    rel = float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))
    assert rel < 3e-2, f"the answer must still be accurate: rel {rel:.3e}"


# ---- second order in time: `ui.tt` -------------------------------------------------------------------
# `u.tt` used to be probed as if it were `u.t`, so a wave equation was silently solved as a heat equation
# (the centre of a standing wave decayed to 8e-5 by t = 0.5 instead of swinging to -0.61). The oracle for
# the structured 5-point grid is the SEMIDISCRETE mode: sin(πx)sin(πy) is an exact eigenvector with
# λ_h = 2·(4/h²)·sin²(πh/2), so the spatial error drops out and only the time integration is tested.

_H = 0.1


def _wave(n_steps, T=0.5, *, damping=None, velocity=False, time=None):
    """u_tt [+ c u_t] = Δu on the unit square, u = 0 on ∂Ω, returns (nodes, trajectory)."""
    import jno.jnp_ops as jnn

    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=_H).structured(), time=(0.0, T, n_steps))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, ti = d.variable("initial", split=True)
    u = d.unknown()
    ui, ui0 = u.bind(x=x, y=y, t=t), u.bind(x=xi, y=yi, t=ti)
    Δu = ui.d2(x) + ui.d2(y)
    mode0 = jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi)

    pde = ui.tt - Δu if damping is None else ui.tt + damping * ui.t - Δu
    terms = [pde, u(xb, yb) - 0.0]
    terms += [u(xi, yi) - 0.0, ui0.t - mode0] if velocity else [u(xi, yi) - mode0]
    prob = jno.fdm(terms)
    return _nodes(d), np.asarray(prob.solve() if time is None else prob.solve(time=time))


def _mode(p):
    return np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])


_OMEGA_H = np.sqrt(2 * (4 / _H**2) * np.sin(np.pi * _H / 2) ** 2)  # semidiscrete frequency of the mode


def _err(sol, amplitude, p):
    return float(np.linalg.norm(sol - amplitude * _mode(p)) / np.linalg.norm(_mode(p)))


def test_wave_standing_mode_oscillates():
    p, traj = _wave(51)
    assert traj.shape == (51, len(p))
    ts = np.linspace(0.0, 0.5, 51)
    worst = max(_err(traj[k], np.cos(_OMEGA_H * ts[k]), p) for k in range(len(ts)))
    assert worst < 1e-3, f"standing wave off the exact cos(ω_h t) mode by {worst:.2e}"
    assert traj[-1][np.argmax(_mode(p))] < -0.5, "the centre must swing negative, not decay like heat"


def test_wave_time_error_is_second_order():
    """θ = ½ (trapezoidal / Newmark average acceleration) is the default: halving dt quarters the error."""
    e = [_err(traj[-1], np.cos(_OMEGA_H * 0.5), p) for p, traj in (_wave(n) for n in (26, 51))]
    assert e[0] / e[1] > 3.5, f"expected O(dt²): errors {e}"


def test_wave_initial_velocity():
    p, traj = _wave(51, velocity=True)
    assert _err(traj[-1], np.sin(_OMEGA_H * 0.5) / _OMEGA_H, p) < 1e-3


def test_wave_damped():
    c, T = 2.0, 0.5
    wd = np.sqrt(_OMEGA_H**2 - c**2 / 4)
    amplitude = np.exp(-c * T / 2) * (np.cos(wd * T) + c / (2 * wd) * np.sin(wd * T))
    p, traj = _wave(51, damping=c)
    assert _err(traj[-1], amplitude, p) < 1e-3


def test_wave_time_scheme_slot_composes():
    """`time=jno.solve.theta(1.0)` swaps in backward Euler, which visibly damps an undamped wave."""
    p, traj = _wave(51, time=jno.solve.theta(1.0))
    exact = abs(np.cos(_OMEGA_H * 0.5))
    assert np.abs(traj[-1]).max() < 0.99 * exact


def test_wave_neumann_edge():
    """A flux edge composes with u_tt: u0 = sin(πx/2) sin(πy), insulated at x = 1."""
    import jno.jnp_ops as jnn

    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured(), time=(0.0, 0.5, 101))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    (xl, yl, _), (xo, yo, _), (xt, yt, _) = (d.variable(r, split=True) for r in ("left", "bottom", "top"))
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y, t=t), u.bind(x=xr, y=yr)
    traj = np.asarray(
        jno.fdm(
            [
                ui.tt - ui.d2(x) - ui.d2(y),
                u(xl, yl) - 0.0,
                u(xo, yo) - 0.0,
                u(xt, yt) - 0.0,
                ur.d(nr) - 0.0,
                u(xi, yi) - jnn.sin(np.pi * xi / 2) * jnn.sin(np.pi * yi),
            ]
        ).solve()
    )
    p = _nodes(d)
    mode = np.sin(np.pi * p[:, 0] / 2) * np.sin(np.pi * p[:, 1])
    exact = np.cos(np.pi * np.sqrt(1.25) * 0.5) * mode
    assert float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(mode)) < 1e-2


@pytest.mark.parametrize(
    "case", ["velocity_on_first_order", "velocity_without_displacement", "nonlinear_inertia", "third_order"]
)
def test_wave_guards(case):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).structured(), time=(0.0, 0.1, 5))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, ti = d.variable("initial", split=True)
    u = d.unknown()
    ui, ui0 = u.bind(x=x, y=y, t=t), u.bind(x=xi, y=yi, t=ti)
    Δu = ui.d2(x) + ui.d2(y)
    terms, match = {
        "velocity_on_first_order": ([ui.t - Δu, u(xi, yi) - 1.0, ui0.t - 1.0], "no `u.tt` term"),
        "velocity_without_displacement": ([ui.tt - Δu, ui0.t - 1.0], "without an initial displacement"),
        "nonlinear_inertia": ([(1.0 + ui) * ui.tt - Δu, u(xi, yi) - 1.0], "nonlinear inertia"),
        "third_order": ([ui.tt.t - Δu, u(xi, yi) - 1.0], "order 3"),
    }[case]
    with pytest.raises((ValueError, NotImplementedError), match=match):
        jno.fdm(terms + [u(xb, yb) - 0.0]).solve()


# ---- variable coefficients in divergence form: `(κ * ui.x).x` ------------------------------------------


@pytest.mark.parametrize("nonlinear", [False, True])
def test_divergence_form_coefficient_converges(nonlinear):
    """−∇·(κ∇u) = f is written with the bound field's partials, `(κ * ui.x).x + (κ * ui.y).y`, for a
    coordinate coefficient κ = 1 + x and for a nonlinear κ = 1 + u. Manufactured u = sin(πx)sin(πy);
    both converge at second order (measured 4.9e-2 → 1.2e-2 → 3.1e-3 and 5.1e-2 → 1.2e-2 → 3.0e-3)."""
    import jno.jnp_ops as jnn

    π = np.pi
    errs = []
    for h in (0.1, 0.05):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h)
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        ue = jnn.sin(π * x) * jnn.sin(π * y)
        if nonlinear:  # −∇·((1+u)∇u) = −(1+u)Δu − |∇u|²
            κ = 1.0 + ui
            grad2 = π**2 * ((jnn.cos(π * x) * jnn.sin(π * y)) ** 2 + (jnn.sin(π * x) * jnn.cos(π * y)) ** 2)
            f = (1.0 + ue) * 2 * π**2 * ue - grad2
        else:  # −∇·((1+x)∇u) = −u_x − (1+x)Δu
            κ = 1.0 + x
            f = -π * jnn.cos(π * x) * jnn.sin(π * y) + κ * 2 * π**2 * ue
        sol = np.asarray(jno.fdm([-(κ * ui.x).x - (κ * ui.y).y - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
        p = _nodes(d)
        exact = np.sin(π * p[:, 0]) * np.sin(π * p[:, 1])
        errs.append(float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)))
    assert errs[1] < 2e-2 and errs[0] / errs[1] > 3.0, f"expected O(h²): {errs}"


# ---- `domain.cell_size` in the strong form: the node spacing, so upwinding is written as math ----------


@pytest.mark.parametrize("dim", [2, 3])
def test_cell_size_is_the_grid_spacing_on_a_structured_grid(dim):
    """FDM resolves `cell_size` per node as the mean of (d!·|K|)^(1/d) over incident cells: exactly the
    grid spacing on the 2-D right-triangulation and the 3-D Kuhn tets (FEM's |K|^(1/d) is h/√2 in 2-D)."""
    import importlib

    fdm_mod = importlib.import_module("jno.fdm")
    h = 0.05 if dim == 2 else 0.25
    shape = jno.shape.rect(0, 0, 1, 1, size=h) if dim == 2 else jno.shape.box(0, 0, 0, 1, 1, 1, size=h)
    d = shape.structured().domain()
    coords = d.variable("interior", split=True)[:dim]
    u = d.unknown()
    ui = u.bind(**dict(zip("xyz", coords)))
    prob = fdm_mod._TraceFDM([-ui.d2(coords[0]) - 1.0, u(*d.variable("boundary", split=True)[:dim]) - 0.0])
    assert np.allclose(np.asarray(prob._node_spacing()), h, rtol=1e-12)


def test_upwinding_written_with_cell_size_matches_the_upwind_matrix():
    """−εΔu + b·u_x = 1 at cell Péclet bh/2ε = 2.5. Upwinding is the identity
    (u_i − u_{i−1})/h = central − (h/2)·(second difference), so it is written as the math,
    `b*ui.x - abs(b)*h/2*ui.xx` with h = d.cell_size. Oracle: the upwind system assembled by hand
    with numpy. Central differences alone overshoot the exact bound u ≤ 1 (max 1.38 here)."""
    import jno.jnp_ops as jnn

    ε, b, h = 1e-2, 1.0, 0.05
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    Δu = ui.xx + ui.yy
    sol = np.asarray(jno.fdm([-ε * Δu + b * ui.x - jnn.abs(b) * d.cell_size / 2 * ui.xx - 1.0, u(xb, yb) - 0.0]).solve())

    n = round(1 / h) - 1  # interior nodes per axis
    eye = np.eye(n)
    D2 = (np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) / h**2
    Dm = (np.eye(n) - np.diag(np.ones(n - 1), -1)) / h  # backward difference: the upwind side for b > 0
    A = -ε * (np.kron(D2, eye) + np.kron(eye, D2)) + b * np.kron(Dm, eye)  # x is the first (slow) index
    ref_int = np.linalg.solve(A, np.ones(n * n)).reshape(n, n)

    p = _nodes(d)
    ix, iy = np.rint(p[:, 0] / h).astype(int), np.rint(p[:, 1] / h).astype(int)
    inner = (ix > 0) & (ix < n + 1) & (iy > 0) & (iy < n + 1)
    ref = ref_int[ix[inner] - 1, iy[inner] - 1]
    assert np.allclose(sol[inner], ref, atol=1e-8), np.abs(sol[inner] - ref).max()
    assert sol.max() < 1.0


# ---- the compiled steady solve -------------------------------------------------------------------------


def _structured_poisson(h=0.05):
    import jno.jnp_ops as jnn

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    return d, u, jno.fdm([-(ui.xx + ui.yy) - f, u(xb, yb) - 0.0])


def test_compiled_residual_has_no_all_pairs_distance():
    """Under `jit` the mesh points used to become tracers, which sent every mesh derivative to the
    in-graph nearest-node fallback: an N×N×dim distance tensor per residual call. That was 0.2 s per
    residual at 16k nodes and 3.8 s at 66k, and `origin/main` was killed at 66k. No intermediate of the
    compiled residual may be quadratic in N."""
    d, _, prob = _structured_poisson(0.05)
    n = prob._N
    jaxpr = jax.make_jaxpr(prob._pde_residual_fn())(jnp.ones(n))
    biggest = max(int(np.prod(v.aval.shape)) for e in jaxpr.jaxpr.eqns for v in e.outvars if hasattr(v.aval, "shape"))
    assert biggest < 16 * n, f"an intermediate of size {biggest} for N = {n}"


def test_steady_solve_is_compiled_once_and_reused():
    """A linear problem on a structured grid takes the one-Krylov-solve path; it compiles once."""
    d, _, prob = _structured_poisson(0.05)
    first = np.asarray(prob.solve())
    fn = prob._grid_linear_cache["fn"]
    second = np.asarray(prob.solve())
    assert prob._grid_linear_cache["fn"] is fn
    np.testing.assert_array_equal(first, second)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert float(np.linalg.norm(second - exact) / np.linalg.norm(exact)) < 5e-3


def test_compiled_solve_sees_a_changed_data_field():
    """A known nodal field is baked into the compiled solve, so its values are part of the cache key:
    changing them must re-solve with the new data, not silently reuse the old."""
    import equinox as eqx

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    p = _nodes(d)
    g = jno.np.parameter((p.shape[0],), name="g")  # data: no optimizer
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    prob = jno.fdm([-ui.d2(x) - ui.d2(y) + 4.0, u(xb, yb) - g])
    for shift in (0.0, 1.0):
        exact = p[:, 0] ** 2 + p[:, 1] ** 2 + shift  # −Δu = −4, u = g on ∂Ω
        g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray(exact))
        sol = np.asarray(prob.solve()).reshape(-1)
        assert float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)) < 1e-2, shift


def test_compiled_solve_still_raises_on_a_stalled_newton():
    """Newton's own guard is blind under `jit`; the compiled path re-checks the concrete result against
    the spec's tolerances and raises, as the uncompiled path did."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    prob = jno.fdm([-ui.laplacian(x, y, scheme=_COT) - 5.0 * jnn.exp(ui), u(xb, yb) - 0.0])
    with pytest.raises(RuntimeError, match="did not converge"):
        prob.solve(nonlinear=jno.solve.newton(max_steps=1))


# ---- fem.solve's solver slots on jno.fdm: linear= / precond= --------------------------------------------
# Setting either assembles the strong-form operator as a sparse matrix once (coloured JVPs, verified
# against the matrix-free action), then composes exactly as `fem.solve` does. The oracle throughout is
# the unchanged matrix-free default: every slot must reach the same answer.


def _slot_problem(kind="unstructured", time=None, order=1, nonlinear=False):
    import jno.jnp_ops as jnn

    kw = {} if time is None else {"time": time}
    if kind == "structured":
        d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured(), **kw)
    else:
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05, **kw)
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t) if time else u.bind(x=x, y=y)
    Δu = ui.d2(x) + ui.d2(y) if kind == "structured" else ui.laplacian(x, y, scheme=_COT)
    if time:
        xi, yi, _ = d.variable("initial", split=True)
        u0 = 16 * xi * (1 - xi) * yi * (1 - yi) * jnn.exp(3 * xi)
        return lambda: jno.fdm([(ui.tt if order == 2 else ui.t) - Δu, u(xb, yb) - 0.0, u(xi, yi) - u0])
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    return lambda: jno.fdm([-Δu - f - (jnn.exp(ui) if nonlinear else 0.0), u(xb, yb) - 0.0])


_SLOTS = {
    "lu": lambda: dict(linear=jno.solve.lu()),
    "bicgstab+jacobi": lambda: dict(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi()),
    "gmres+amg": lambda: dict(linear=jno.solve.gmres(), precond=jno.precond.amg()),
}


def _max_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize("slot", list(_SLOTS))
@pytest.mark.parametrize("case", ["steady", "steady-nonlinear", "heat", "wave"])
def test_solver_slots_reach_the_default_answer(slot, case):
    if slot == "gmres+amg":
        pytest.importorskip("pyamg")
    time = None if case.startswith("steady") else (0.0, 0.1, 21)
    make = _slot_problem(time=time, order=2 if case == "wave" else 1, nonlinear=case == "steady-nonlinear")
    kw = _SLOTS[slot]()
    ref = make().solve()
    assert _max_rel(make().solve(**kw), ref) < 1e-7


def test_gmg_slot_on_a_structured_grid():
    for case, time in (("steady", None), ("heat", (0.0, 0.1, 21))):
        make = _slot_problem("structured", time=time)
        got = make().solve(linear=jno.solve.gmres(), precond=jno.precond.gmg())
        assert _max_rel(got, make().solve()) < 1e-7, case


def test_wave_slots_on_the_newmark_step():
    """The default u.tt march solves for the new displacement alone (Newmark), a single scalar field. So
    gmg now preconditions it, and cg applies: both used to be refused or fail on the non-symmetric,
    twice-as-large [u; v] system."""
    make = _slot_problem("structured", time=(0.0, 0.1, 21), order=2)
    ref = make().solve()
    for kw in (
        dict(linear=jno.solve.gmres(), precond=jno.precond.gmg()),
        dict(linear=jno.solve.cg(), precond=jno.precond.jacobi()),
    ):
        assert _max_rel(make().solve(**kw), ref) < 1e-7, kw


def test_gmg_refuses_the_augmented_wave_state():
    """An explicit time scheme keeps the augmented [u; v] march, which gmg cannot precondition."""
    make = _slot_problem("structured", time=(0.0, 0.1, 21), order=2)
    with pytest.raises(ValueError, match="single scalar field"):
        make().solve(linear=jno.solve.gmres(), precond=jno.precond.gmg(), time=jno.solve.theta(0.5))


@pytest.mark.parametrize("case", ["steady", "heat", "wave"])
def test_cg_refuses_a_nonsymmetric_operator(case):
    """Unstructured FDM operators are not symmetric (cotangent rows are divided by nodal areas), and CG on
    them converged to an answer ~2e-6 off, inside the residual gate. It now refuses up front."""
    time = None if case == "steady" else (0.0, 0.1, 21)
    make = _slot_problem(time=time, order=2 if case == "wave" else 1)
    with pytest.raises(ValueError, match="needs a symmetric operator"):
        make().solve(linear=jno.solve.cg(), precond=jno.precond.jacobi())


def test_cg_on_a_structured_grid_after_the_dirichlet_lift():
    """On a structured grid the operator is symmetric once the Dirichlet columns are lifted to the
    right-hand side, so CG applies and reaches the default answer."""
    for case, time in (("steady", None), ("heat", (0.0, 0.1, 21))):
        make = _slot_problem("structured", time=time)
        got = make().solve(linear=jno.solve.cg(), precond=jno.precond.jacobi())
        assert _max_rel(got, make().solve()) < 1e-7, case


def test_nonlinear_slot_on_a_linear_problem_raises():
    with pytest.raises(ValueError, match="this problem is linear"):
        _slot_problem()().solve(linear=jno.solve.lu(), nonlinear=jno.solve.newton())


def test_transient_nonlinear_slot_is_used():
    """`nonlinear=` on a transient FDM problem used to be accepted and silently ignored. One Newton step
    cannot converge a cubic reaction, so the march must now refuse."""
    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1, time=(0.0, 0.1, 6))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    prob = jno.fdm(
        [ui.t - ui.laplacian(x, y, scheme=_COT) - 5.0 * ui**3, u(xb, yb) - 0.0, u(xi, yi) - 2.0 * jnn.sin(np.pi * xi)]
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        prob.solve(nonlinear=jno.solve.newton(max_steps=1))


def test_inverse_through_a_solver_slot():
    """The assembled operator is traceable, so a crux-driven inverse runs through `linear=` too."""
    import optax

    import jno.jnp_ops as jnn

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    Δu = ui.laplacian(x, y, scheme=_COT)
    observed = jnp.asarray(jno.fdm([-Δu - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
    s = jno.np.parameter((1,), name="s")
    s.dtype(jnp.float64)
    s.initialize(jax.nn.initializers.constant(2.5))
    s.optimizer(optax.adam(1e-1))
    node = jno.fdm([-Δu - s * f, u(xb, yb) - 0.0]).solve(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi())
    crux = jno.core([(node - observed).mse])
    crux.solve(120)
    assert abs(float(np.asarray(crux.eval([s])).reshape(-1)[0]) - 1.0) < 2e-2


def test_exponential_scheme_refused_even_with_a_linear_slot():
    """With a slot the block is linear and the exponential scheme used to run, 4.8e-3 off a converged
    reference (Crank-Nicolson at the same step: 2.0e-5): an FDM march is a DAE whose boundary rows have
    zero mass, and the scheme forms M⁻¹A. It now refuses in both cases."""
    make = _slot_problem(time=(0.0, 0.1, 21))
    with pytest.raises(NotImplementedError, match="invertible mass"):
        make().solve(time=jno.solve.exponential(), linear=jno.solve.gmres())


def test_coupled_fields_return_in_declaration_order():
    """The fields come back in the order they were declared, whatever order the equations are listed in.
    They used to come back in first-appearance order, so listing the v-equation first swapped u and v."""
    import jno.jnp_ops as jnn

    π = np.pi
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.unknown(), d.unknown()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    U, V = jnn.sin(π * x) * jnn.sin(π * y) + x, x * y + jnn.cos(x)
    eu = -(ui.xx + ui.yy) + vi - (2 * π**2 * jnn.sin(π * x) * jnn.sin(π * y) + V)
    ev = -(vi.xx + vi.yy) + ui - (jnn.cos(x) + U)
    bcs = [u(xb, yb) - (jnn.sin(π * xb) * jnn.sin(π * yb) + xb), v(xb, yb) - (xb * yb + jnn.cos(xb))]
    p = _nodes(d)
    Ue = np.sin(π * p[:, 0]) * np.sin(π * p[:, 1]) + p[:, 0]
    Ve = p[:, 0] * p[:, 1] + np.cos(p[:, 0])
    for order in ([eu, ev], [ev, eu]):
        uh, vh = np.asarray(jno.fdm(order + bcs).solve())
        assert np.linalg.norm(uh - Ue) / np.linalg.norm(Ue) < 5e-3
        assert np.linalg.norm(vh - Ve) / np.linalg.norm(Ve) < 5e-3


def test_structured_box_faces_are_named_like_shape_box():
    """front/back at y = 0/1 and bottom/top at z = 0/1, on the tet mesh and the structured grid alike.
    The structured grid used to swap the two pairs, so a condition on "top" moved faces with `.structured()`."""
    import importlib

    fdm_mod = importlib.import_module("jno.fdm")
    expected = {"front": (1, 0.0), "back": (1, 1.0), "bottom": (2, 0.0), "top": (2, 1.0)}
    for structured in (False, True):
        shape = jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.25)
        d = shape.structured().domain() if structured else shape.domain()
        x, y, z, _ = d.variable("interior", split=True)
        u = d.unknown()
        prob = fdm_mod._TraceFDM([-u.bind(x=x, y=y, z=z).d2(x) - 1.0, u(*d.variable("boundary", split=True)[:3]) - 0.0])
        pts = np.asarray(d.mesh_connectivity["points"])[:, :3]
        for face, (axis, value) in expected.items():
            d.variable(face, split=True)
            face_pts = pts[prob._region_nodes(face)]
            assert np.allclose(face_pts[:, axis], value), (structured, face)


@pytest.mark.parametrize("spelling", ["d2", "xx", "laplacian", "scaled"])
def test_default_laplacian_is_fused_into_cotangent(spelling):
    """On an unstructured mesh the per-axis default (a gradient of the area-weighted gradient) has a
    spurious oscillating mode: its lowest Dirichlet eigenvalue on the unit square is ~5.4, not 2π², and it
    does not refine away. An advection–diffusion solve came out 2.09 off, and Helmholtz at c = 5.41 blew up.
    Every spelling of the plain Laplacian now fuses into the cotangent one, which has no such mode.
    Oracle: −Δu − 5.41 u = f with u = sin(πx) sin(πy), sitting right on the old spurious eigenvalue."""
    import jno.jnp_ops as jnn

    π, c = np.pi, 5.4126
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    Δu = {
        "d2": ui.d2(x) + ui.d2(y),
        "xx": ui.xx + ui.yy,
        "laplacian": ui.laplacian(x, y),
        "scaled": None,
    }[spelling]
    f = (2 * π**2 - c) * jnn.sin(π * x) * jnn.sin(π * y)
    pde = (-2.0 * ui.xx - 2.0 * ui.yy) / 2.0 - c * ui - f if spelling == "scaled" else -Δu - c * ui - f
    sol = np.asarray(jno.fdm([pde, u(xb, yb) - 0.0]).solve()).reshape(-1)
    p = _nodes(d)
    exact = np.sin(π * p[:, 0]) * np.sin(π * p[:, 1])
    assert float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)) < 1e-2


# ---- time-dependent data: sources f(x, t), boundary values g(x, t), coefficients κ(t) ------------------
# A source containing t raised KeyError('__time__'); a boundary value containing t was accepted and then
# held at its start value (the boundary stayed at 1.0 while g decayed to 0.14; error 3.5 at T).


def _heat_with_time_data(n_steps, *, kind, slots=None, structured=True):
    """u_t − κ(t)Δu = f(x, t) with an exact solution, every datum written with the time variable."""
    import jno.jnp_ops as jnn

    π = np.pi
    shape = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05)
    d = jno.domain(shape.structured() if structured else shape, time=(0.0, 0.2, n_steps))
    x, y, t = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    Δu = ui.xx + ui.yy
    if kind == "source":  # u = e^{-t} sin πx sin πy
        terms = [
            ui.t - Δu - (2 * π**2 - 1) * jnn.exp(-t) * jnn.sin(π * x) * jnn.sin(π * y),
            u(xb, yb) - 0.0,
            u(xi, yi) - jnn.sin(π * xi) * jnn.sin(π * yi),
        ]
        exact = lambda p, T: np.exp(-T) * np.sin(π * p[:, 0]) * np.sin(π * p[:, 1])  # noqa: E731
    elif kind == "dirichlet":  # u = e^{-π² t} cos πx
        terms = [ui.t - Δu, u(xb, yb) - jnn.exp(-(π**2) * tb) * jnn.cos(π * xb), u(xi, yi) - jnn.cos(π * xi)]
        exact = lambda p, T: np.exp(-(π**2) * T) * np.cos(π * p[:, 0])  # noqa: E731
    else:  # "coefficient": κ(t) = 1 + t, u = e^{-t} sin πx sin πy
        terms = [
            ui.t - (1.0 + t) * Δu - (2 * π**2 * (1.0 + t) - 1) * jnn.exp(-t) * jnn.sin(π * x) * jnn.sin(π * y),
            u(xb, yb) - 0.0,
            u(xi, yi) - jnn.sin(π * xi) * jnn.sin(π * yi),
        ]
        exact = lambda p, T: np.exp(-T) * np.sin(π * p[:, 0]) * np.sin(π * p[:, 1])  # noqa: E731
    traj = np.asarray(jno.fdm(terms).solve(time=jno.solve.theta(0.5), **(slots or {})))
    p = _nodes(d)
    ref = exact(p, 0.2)
    return float(np.linalg.norm(traj[-1] - ref) / np.linalg.norm(ref))


@pytest.mark.parametrize("kind", ["source", "dirichlet", "coefficient"])
def test_time_dependent_data_converges(kind):
    """Crank–Nicolson, h = 0.05: the error falls under Δt refinement to the spatial floor."""
    coarse, fine = _heat_with_time_data(11, kind=kind), _heat_with_time_data(41, kind=kind)
    # Both sit at or near the spatial error floor (the Dirichlet case is already there at 11 steps).
    assert fine < 5e-3 and coarse < 2e-2, (coarse, fine)


@pytest.mark.parametrize("kind", ["source", "dirichlet", "coefficient"])
def test_time_dependent_data_through_the_solver_slots(kind):
    """With linear=/precond= the operator is assembled once: time-dependent data must ride the forcing
    (it used to be frozen as a constant bias), and a time-varying κ(t) must fall back to the Newton step."""
    ref = _heat_with_time_data(21, kind=kind, structured=False)
    got = _heat_with_time_data(
        21, kind=kind, structured=False, slots=dict(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi())
    )
    assert abs(got - ref) < 1e-6 * max(1.0, ref) and got < 1e-2, (got, ref)


def test_forced_wave_with_time_dependent_source():
    """u_tt − Δu = f(x, t) with u = sin(t) sin πx sin πy (u0 = 0, v0 = sin πx sin πy), Newmark default."""
    import jno.jnp_ops as jnn

    π = np.pi
    errs = []
    for n in (21, 41):
        d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured(), time=(0.0, 1.0, n))
        x, y, t = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        xi, yi, ti = d.variable("initial", split=True)
        u = d.unknown()
        ui, ui0 = u.bind(x=x, y=y, t=t), u.bind(x=xi, y=yi, t=ti)
        f = (2 * π**2 - 1) * jnn.sin(t) * jnn.sin(π * x) * jnn.sin(π * y)
        traj = np.asarray(
            jno.fdm(
                [ui.tt - ui.xx - ui.yy - f, u(xb, yb) - 0.0, u(xi, yi) - 0.0, ui0.t - jnn.sin(π * xi) * jnn.sin(π * yi)]
            ).solve()
        )
        p = _nodes(d)
        ref = np.sin(1.0) * np.sin(π * p[:, 0]) * np.sin(π * p[:, 1])
        errs.append(float(np.linalg.norm(traj[-1] - ref) / np.linalg.norm(ref)))
    assert errs[1] < errs[0] and errs[1] < 1e-2, errs


def _time_everywhere(kind, h):
    """u_t − Δu = f with a datum written with t in a flux condition or on the mass; exact solutions:
    u = e^{-t} x² sin πy (flux on the right edge), u = e^{-t} sin πx sin πy (mass (1 + t)·u_t)."""
    import jno.jnp_ops as jnn

    π = np.pi
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured(), time=(0.0, 0.2, 41))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    p = _nodes(d)
    if kind == "mass":
        xb, yb, _ = d.variable("boundary", split=True)
        f = (2 * π**2 - (1 + t)) * jnn.exp(-t) * jnn.sin(π * x) * jnn.sin(π * y)
        terms = [(1 + t) * ui.t - ui.xx - ui.yy - f, u(xb, yb) - 0.0, u(xi, yi) - jnn.sin(π * xi) * jnn.sin(π * yi)]
        exact = np.exp(-0.2) * np.sin(π * p[:, 0]) * np.sin(π * p[:, 1])
    else:
        (xl, yl, _), (xo, yo, _), (xt, yt, _) = (d.variable(r, split=True) for r in ("left", "bottom", "top"))
        xr, yr, tr = d.variable("right", split=True)
        nr = d.variable("right", normals=True)
        ur = u.bind(x=xr, y=yr)
        f = jnn.exp(-t) * ((π**2 - 1) * x**2 - 2) * jnn.sin(π * y)
        flux = {  # ∂u/∂n = u_x = 2 e^{-t} sin πy at x = 1
            "neumann": ur.d(nr) - 2 * jnn.exp(-tr) * jnn.sin(π * yr),
            "robin": ur.d(nr) + (1 + tr) * ur - (3 + tr) * jnn.exp(-tr) * jnn.sin(π * yr),
        }[kind]
        terms = [
            ui.t - ui.xx - ui.yy - f,
            u(xl, yl) - 0.0,
            u(xo, yo) - 0.0,
            u(xt, yt) - 0.0,
            flux,
            u(xi, yi) - xi**2 * jnn.sin(π * yi),
        ]
        exact = np.exp(-0.2) * p[:, 0] ** 2 * np.sin(π * p[:, 1])
    traj = np.asarray(jno.fdm(terms).solve(time=jno.solve.theta(0.5)))
    return float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact))


@pytest.mark.parametrize("kind", ["neumann", "robin", "mass"])
def test_time_dependent_flux_data_and_mass_coefficient(kind):
    """A datum written with t works wherever it appears: a Neumann value h(t), a Robin α(t), a mass
    (1 + t)·u_t. They were evaluated once at the start and held: 0.10, 0.12 and 5.8e-3 at T, against
    1.5e-3, 1.2e-3 and 2.2e-3 now at h = 0.05 — and second order under refinement."""
    e = [_time_everywhere(kind, h) for h in (0.1, 0.05)]
    assert e[1] < 1e-2 and e[0] / e[1] > 3.0, e


@pytest.mark.parametrize("structured", [False, True])
def test_inverse_recovers_a_robin_coefficient(structured):
    """A trainable α inside a Robin condition, ∂u/∂n + α(u − 0.5) = 0, recovered through jno.core from the
    field it produces. It crashed: the flux rows' host-side mesh work and their affine check ran inside the
    crux trace (and on a structured grid, the multigrid setup too)."""
    import optax

    shape = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1)
    d = shape.structured().domain() if structured else jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    x, y, _ = d.variable("interior", split=True)
    (xl, yl, _), (xo, yo, _), (xr, yr, _), (xt, yt, _) = (
        d.variable(r, split=True) for r in ("left", "bottom", "right", "top")
    )
    nt = d.variable("top", normals=True)
    u = d.unknown()
    ui, ut = u.bind(x=x, y=y), u.bind(x=xt, y=yt)

    def problem(alpha):
        return jno.fdm(
            [-(ui.xx + ui.yy) - 1.0, u(xl, yl) - 0.0, u(xo, yo) - 1.0, u(xr, yr) - 0.0, ut.d(nt) + alpha * (ut - 0.5)]
        )

    observed = jnp.asarray(problem(2.0).solve()).reshape(-1)
    a = jno.np.parameter((1,), name="alpha")
    a.dtype(jnp.float64)
    a.initialize(jax.nn.initializers.constant(0.5))
    a.optimizer(optax.adam(5e-2))
    crux = jno.core([(problem(a).solve() - observed).mse])
    crux.solve(300)
    recovered = float(np.asarray(crux.eval([a])).reshape(-1)[0])
    assert abs(recovered - 2.0) < 5e-2, recovered


@pytest.mark.parametrize("where", ["source", "dirichlet", "neumann"])
def test_a_trainable_parameter_is_differentiable_wherever_it_appears(where):
    """Recover s = 1.5 from the field it produces, with s in the source, a Dirichlet value, or a Neumann
    value. The Dirichlet value used to be read from the parameter's STORED value, so its gradient was zero
    and the inverse never moved (0.5 stayed 0.5), with no error."""
    import optax

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    (xl, yl, _), (xo, yo, _), (xr, yr, _), (xt, yt, _) = (
        d.variable(r, split=True) for r in ("left", "bottom", "right", "top")
    )
    nt = d.variable("top", normals=True)
    u = d.unknown()
    ui, ut = u.bind(x=x, y=y), u.bind(x=xt, y=yt)
    Δu = ui.xx + ui.yy

    def problem(s):
        if where == "source":
            return jno.fdm([-Δu - s, u(xb, yb) - 0.0])
        if where == "dirichlet":
            return jno.fdm([-Δu - 1.0, u(xb, yb) - s])
        return jno.fdm([-Δu - 1.0, u(xl, yl) - 0.0, u(xo, yo) - 0.0, u(xr, yr) - 0.0, ut.d(nt) - s])

    observed = jnp.asarray(problem(1.5).solve()).reshape(-1)
    s = jno.np.parameter((1,), name="s")
    s.dtype(jnp.float64)
    s.initialize(jax.nn.initializers.constant(0.5))
    s.optimizer(optax.adam(5e-2))
    crux = jno.core([(problem(s).solve() - observed).mse])
    crux.solve(300)
    assert abs(float(np.asarray(crux.eval([s])).reshape(-1)[0]) - 1.5) < 1e-3


@pytest.mark.parametrize("kind", ["diffusivity", "source", "dirichlet", "robin", "wave_speed", "slots"])
def test_transient_inverse_recovers_the_parameter(kind):
    """A trainable parameter in a TIME-DEPENDENT problem, recovered through jno.core from the trajectory it
    produces — in the diffusivity, a source f(x, t), a boundary value g(x, t), a Robin coefficient, a wave
    speed (the Newmark march), and through linear=/precond= slots. This raised "No model for Model N": the
    march evaluated the residual without the parameter. Every evaluator in the march now reads the injected
    value, and the march's structural decisions are made once on concrete values."""
    import optax

    import jno.jnp_ops as jnn

    π = np.pi
    if kind == "slots":
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1, time=(0.0, 0.1, 11))
    else:
        d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured(), time=(0.0, 0.1, 11))
    x, y, t = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    (xl, yl, _), (xo, yo, _), (xr, yr, _), (xt, yt, _) = (
        d.variable(r, split=True) for r in ("left", "bottom", "right", "top")
    )
    nt = d.variable("top", normals=True)
    u = d.unknown()
    ui, ut = u.bind(x=x, y=y, t=t), u.bind(x=xt, y=yt)
    Δu = ui.xx + ui.yy
    u0 = 16 * xi * (1 - xi) * yi * (1 - yi)

    def problem(s):
        if kind in ("diffusivity", "slots"):
            return jno.fdm([ui.t - s * Δu, u(xb, yb) - 0.0, u(xi, yi) - u0])
        if kind == "source":
            return jno.fdm(
                [ui.t - Δu - s * jnn.exp(-t) * jnn.sin(π * x) * jnn.sin(π * y), u(xb, yb) - 0.0, u(xi, yi) - 0.0]
            )
        if kind == "dirichlet":
            return jno.fdm([ui.t - Δu, u(xb, yb) - s * jnn.exp(-tb) * xb, u(xi, yi) - s * xi])
        if kind == "robin":
            return jno.fdm(
                [ui.t - Δu, u(xl, yl) - 0.0, u(xo, yo) - 0.0, u(xr, yr) - 0.0, ut.d(nt) + s * (ut - 1.0), u(xi, yi) - u0]
            )
        return jno.fdm([ui.tt - s * Δu, u(xb, yb) - 0.0, u(xi, yi) - u0])

    slots = dict(linear=jno.solve.bicgstab(), precond=jno.precond.jacobi()) if kind == "slots" else {}
    true = 3.0 if kind == "robin" else 1.5
    observed = jnp.asarray(problem(true).solve(**slots))
    s = jno.np.parameter((1,), name="s")
    s.dtype(jnp.float64)
    s.initialize(jax.nn.initializers.constant(1.0))
    s.optimizer(optax.adam(5e-2))
    crux = jno.core([(problem(s).solve(**slots) - observed).mse])
    crux.solve(300)
    assert abs(float(np.asarray(crux.eval([s])).reshape(-1)[0]) - true) < 1e-3


# θ-steps on the DAE: the Dirichlet and flux rows carry zero mass, so they are constraints, not ODEs.
# Forward Euler used to evaluate them only at the OLD state (a singular step: NaN at step 38 once the
# decaying field reached ~1e-8), and Crank-Nicolson averaged them (a boundary started off its value
# flipped sign every step and never decayed). They are now imposed at the new time.


def _heat_mode(k, n, *, ic_one=False, T=0.02, h=0.1):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured(), time=(0.0, T, n))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    ic = 1.0 + 0.0 * xi if ic_one else jno.np.sin(k * np.pi * xi) * jno.np.sin(k * np.pi * yi)
    return d, jno.fdm([ui.t - ui.xx - ui.yy, u(xb, yb) - 0.0, u(xi, yi) - ic])


@pytest.mark.parametrize("k, slots", [(1, {}), (9, {}), (1, {"linear": "gmres"}), (1, {"linear": "lu"})])
def test_forward_euler_matches_the_discrete_decay(k, slots):
    """θ = 0 on a grid mode of the 5-point Laplacian decays by exactly (1 − Δt·λ_h) per step. The k = 9
    mode (Δt·λ_h = 0.39, stable) is the one that used to abort with a NaN."""
    n, T, h = 201, 0.02, 0.1
    d, problem = _heat_mode(k, n, T=T, h=h)
    slots = {name: getattr(jno.solve, solver)() for name, solver in slots.items()}
    traj = np.asarray(problem.solve(time=jno.solve.theta(0.0), **slots)).reshape(n, -1)
    p = _nodes(d)
    mode = np.sin(k * np.pi * p[:, 0]) * np.sin(k * np.pi * p[:, 1])
    factor = (1.0 - T / (n - 1) * 8.0 / h**2 * np.sin(k * np.pi * h / 2) ** 2) ** np.arange(n)
    for i in (10, 50, n - 1):
        # the step's Newton converges to an absolute tolerance, so a state decayed to 1e-7 is exact to
        # ~1e-15 absolutely, not to 1e-8 relative to itself
        assert np.abs(traj[i] - factor[i] * mode).max() < 1e-8 * abs(factor[i]) + 1e-12, (i, factor[i])


@pytest.mark.parametrize("theta", [0.0, 0.5])
def test_theta_step_imposes_the_boundary_at_the_new_time(theta):
    """An initial state that violates the Dirichlet value (u0 = 1, g = 0) is pulled onto it by the first
    step, as backward Euler does. Crank–Nicolson held the boundary at |u| = 1 for the whole march."""
    d, problem = _heat_mode(0, 11, ic_one=True, T=1e-4)
    traj = np.asarray(problem.solve(time=jno.solve.theta(theta))).reshape(11, -1)
    p = _nodes(d)
    on_boundary = (np.minimum(p[:, 0], 1.0 - p[:, 0]) < 1e-9) | (np.minimum(p[:, 1], 1.0 - p[:, 1]) < 1e-9)
    assert np.abs(traj[1:, on_boundary]).max() < 1e-12


def test_nonlinear_crank_nicolson_assembled_tangent():
    """With a solver slot a nonlinear march uses the assembled step tangent M/Δt + θ·J. It dropped the θ,
    so Crank–Nicolson's Newton converged on a wrong tangent: 3.3e-11 off the matrix-free march, now 4e-17."""

    def solve(**slots):
        d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured(), time=(0.0, 0.2, 21))
        x, y, t = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        xi, yi, _ = d.variable("initial", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y, t=t)
        ic = 3.0 * jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi)
        terms = [ui.t - ui.xx - ui.yy + 30.0 * ui**3, u(xb, yb) - 0.0, u(xi, yi) - ic]
        return np.asarray(jno.fdm(terms).solve(time=jno.solve.theta(0.5), **slots))[-1]

    assert np.abs(solve(linear=jno.solve.lu()) - solve()).max() < 1e-13


# cg / minres on a Newton path: the assembled tangent keeps the Dirichlet identity rows, whose columns the
# interior rows still reference, so it is not symmetric and CG returned NaN. The Newton path now solves
# through the same Dirichlet elimination the linear path uses, exactly, for J and for Jᵀ (the adjoint).


def test_dirichlet_elimination_is_exact_for_the_tangent_and_its_transpose():
    import jax.experimental.sparse as jsp

    from jno.fdm import _TraceFDM

    rng = np.random.default_rng(0)
    n, is_d = 12, np.zeros(12, dtype=bool)
    is_d[[0, 5, 11]] = True
    J = rng.standard_normal((n, n)) + 8.0 * np.eye(n)
    J[is_d] = 0.0
    J[is_d, is_d] = rng.uniform(1.0, 3.0, 3)  # pure constraint rows
    b = rng.standard_normal(n)
    dense = lambda A_s, rhs: jnp.linalg.solve(A_s.todense(), rhs)  # noqa: E731
    for A in (J, J.T):
        _, solve = _TraceFDM._eliminate(jsp.BCOO.fromdense(jnp.asarray(A)), is_d)
        assert np.abs(np.asarray(solve(dense, jnp.asarray(b))) - np.linalg.solve(A, b)).max() < 1e-12


def _bratu(d):
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    S = jno.np.sin(np.pi * x) * jno.np.sin(np.pi * y)
    f = 2 * np.pi**2 * S - 2.0 * jno.np.exp(S + 0.5 * x * y)
    return jno.fdm([-(ui.xx + ui.yy) - 2.0 * jno.np.exp(ui) - f, u(xb, yb) - 0.5 * xb * yb])


def test_cg_on_a_nonlinear_structured_problem():
    """Bratu MMS, u = sin πx sin πy + xy/2 on the structured grid: cg matches lu (it returned NaN)."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured().domain()
    got = np.asarray(_bratu(d).solve(linear=jno.solve.cg(), precond=jno.precond.jacobi())).reshape(-1)
    ref = np.asarray(_bratu(d).solve(linear=jno.solve.lu())).reshape(-1)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) + 0.5 * p[:, 0] * p[:, 1]
    assert np.abs(got - ref).max() < 1e-12 and np.abs(got - exact).max() < 5e-3


def test_cg_on_a_nonlinear_march():
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured(), time=(0.0, 0.1, 11))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    ic = jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi) + 0.2
    terms = [ui.t - ui.xx - ui.yy + ui**3, u(xb, yb) - 0.2, u(xi, yi) - ic]
    got = np.asarray(jno.fdm(terms).solve(linear=jno.solve.cg()))
    assert np.abs(got - np.asarray(jno.fdm(terms).solve(linear=jno.solve.lu()))).max() < 1e-12


def test_cg_on_a_nonlinear_unstructured_problem_refuses():
    """The cotangent rows are divided by nodal areas, so the eliminated tangent is still not symmetric."""
    with pytest.raises(ValueError, match="needs a symmetric operator"):
        _bratu(jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)).solve(linear=jno.solve.cg())


@pytest.mark.parametrize("linear", ["lu", "gmres", "bicgstab", "cg"])
def test_gradient_through_a_nonlinear_solve_with_any_linear_slot(linear):
    """d(mean u²)/dg for a Dirichlet value g·xy on Bratu, against a central difference. With an iterative
    slot `jax.grad` raised "Reverse-mode differentiation does not work for lax.while_loop": the direct
    Newton's forward loop closed over the parameter, and the Krylov residual gate kept its tangents alive."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured().domain()

    def solve(g, **slots):
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        S = jno.np.sin(np.pi * x) * jno.np.sin(np.pi * y)
        f = 2 * np.pi**2 * S - 2.0 * jno.np.exp(S + 0.5 * x * y)
        terms = [-(ui.xx + ui.yy) - 2.0 * jno.np.exp(ui) - f, u(xb, yb) - g * xb * yb]
        return jnp.mean(jnp.asarray(jno.fdm(terms).solve(**slots)).reshape(-1) ** 2)

    fd = (float(solve(0.5 + 1e-5)) - float(solve(0.5 - 1e-5))) / 2e-5
    got = float(jax.grad(lambda g: solve(g, linear=getattr(jno.solve, linear)()))(0.5))
    assert abs(got - fd) < 1e-8 * abs(fd), (got, fd)


def test_all_neumann_structured_grid_keeps_the_mean():
    """−Δu + u = f with ∂u/∂n = 0 on all four sides, u = cos πx cos πy + ½. The PDE fixes the mean of u only
    through the reaction term, so a flux error ε shifts the whole solution by ∮ε. The quadratic boundary
    fit left the mean 0.31 off (0.43 relative error at h = 0.1); a box face's ∂u/∂n is the three-point
    one-sided difference, which gives 2.4e-3."""
    import jno.jnp_ops as jnn

    π = np.pi
    errs = []
    for h in (0.1, 0.05):
        d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
        x, y, _ = d.variable("interior", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        flux = []
        for r in ("left", "right", "bottom", "top"):
            xr, yr, _ = d.variable(r, split=True)
            flux.append(u.bind(x=xr, y=yr).d(d.variable(r, normals=True)) - 0.0)
        f = (2 * π**2 + 1) * jnn.cos(π * x) * jnn.cos(π * y) + 0.5
        sol = np.asarray(jno.fdm([-(ui.xx + ui.yy) + ui - f, *flux]).solve()).reshape(-1)
        p = _nodes(d)
        exact = np.cos(π * p[:, 0]) * np.cos(π * p[:, 1]) + 0.5
        assert abs(np.mean(sol - exact)) < 1e-3
        errs.append(float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)))
    assert errs[0] < 5e-3 and errs[0] / errs[1] > 3.5, errs


# Coupled time-dependent systems: the march works on the blocked vector [u_0; …; u_{nf-1}], equation k
# carrying u_k.t (a diagonal mass); an equation without a time derivative makes its field algebraic.


def _coupled_rotation(h, n, *, structured=True, T=0.1, **slots):
    """u_t = Δu − v, v_t = Δv + u: u = e^{−2π²t} S cos t, v = e^{−2π²t} S sin t, S = sin πx sin πy."""
    shape = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h)
    d = jno.domain(shape.structured() if structured else shape, time=(0.0, T, n))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u, v = d.unknown(), d.unknown()
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    terms = [
        ui.t - (ui.xx + ui.yy) + vi,
        vi.t - (vi.xx + vi.yy) - ui,
        u(xb, yb) - 0.0,
        v(xb, yb) - 0.0,
        u(xi, yi) - jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi),
        v(xi, yi) - 0.0,
    ]
    traj = np.asarray(jno.fdm(terms).solve(time=jno.solve.theta(0.5), **slots))
    p = _nodes(d)
    S = np.exp(-2 * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    exact = np.stack([S * np.cos(T), S * np.sin(T)])
    assert traj.shape == (n, 2, len(p))  # (step, field, node), fields in declaration order
    return float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact))


@pytest.mark.parametrize("structured", [True, False])
def test_coupled_march_converges(structured):
    """Crank–Nicolson with Δt ∝ h: 1.0e-2 → 2.5e-3 structured, 1.7e-2 → 4.4e-3 unstructured (rate 2)."""
    e = [_coupled_rotation(h, n, structured=structured) for h, n in ((0.1, 11), (0.05, 21))]
    assert e[1] < 5e-3 and e[0] / e[1] > 3.5, e


def test_coupled_march_through_the_solver_slots():
    ref = _coupled_rotation(0.1, 11)
    assert abs(_coupled_rotation(0.1, 11, linear=jno.solve.gmres(), precond=jno.precond.jacobi()) - ref) < 1e-10


def test_coupled_march_with_an_algebraic_field():
    """u_t = Δu + w with −Δw = 2π²u: w has no time derivative, so it is a constraint at every step (a
    DAE). u = w = e^{(1−2π²)t} S. Crank–Nicolson, 1.7e-3 at h = 0.05."""
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured(), time=(0.0, 0.1, 21))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u, w = d.unknown(), d.unknown()
    ui, wi = u.bind(x=x, y=y, t=t), w.bind(x=x, y=y, t=t)
    terms = [
        ui.t - (ui.xx + ui.yy) - wi,
        -(wi.xx + wi.yy) - 2 * np.pi**2 * ui,
        u(xb, yb) - 0.0,
        w(xb, yb) - 0.0,
        u(xi, yi) - jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi),
    ]
    traj = np.asarray(jno.fdm(terms).solve(time=jno.solve.theta(0.5)))
    p = _nodes(d)
    exact = np.exp((1 - 2 * np.pi**2) * 0.1) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert np.abs(traj[-1] - exact).max() / np.abs(exact).max() < 3e-3


def test_coupled_nonlinear_march():
    """A Gray–Scott reaction–diffusion pair: the matrix-free Newton march and the lu slot agree."""
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).structured(), time=(0.0, 0.5, 26))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u, v = d.unknown(), d.unknown()
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    bump = jno.np.exp(-40.0 * ((xi - 0.5) ** 2 + (yi - 0.5) ** 2))
    terms = [
        ui.t - 0.02 * (ui.xx + ui.yy) + ui * vi**2 - 0.04 * (1.0 - ui),
        vi.t - 0.01 * (vi.xx + vi.yy) - ui * vi**2 + 0.1 * vi,
        u(xb, yb) - 1.0,
        v(xb, yb) - 0.0,
        u(xi, yi) - (1.0 - 0.5 * bump),
        v(xi, yi) - 0.25 * bump,
    ]
    a = np.asarray(jno.fdm(terms).solve())
    b = np.asarray(jno.fdm(terms).solve(linear=jno.solve.lu()))
    assert np.isfinite(a).all() and np.abs(a - b).max() < 1e-12


def test_coupled_march_inverse_recovers_the_coupling():
    """A trainable coupling strength ω in u_t = Δu − ωv, v_t = Δv + ωu, recovered through jno.core."""
    import optax

    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured(), time=(0.0, 0.1, 11))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u, v = d.unknown(), d.unknown()
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)

    def problem(omega):
        return jno.fdm(
            [
                ui.t - (ui.xx + ui.yy) + omega * vi,
                vi.t - (vi.xx + vi.yy) - omega * ui,
                u(xb, yb) - 0.0,
                v(xb, yb) - 0.0,
                u(xi, yi) - jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi),
                v(xi, yi) - 0.0,
            ]
        )

    observed = jnp.asarray(problem(6.0).solve())
    w = jno.np.parameter((1,), name="omega")
    w.dtype(jnp.float64)
    w.initialize(jax.nn.initializers.constant(4.0))
    w.optimizer(optax.adam(5e-2))
    crux = jno.core([(problem(w).solve() - observed).mse])
    crux.solve(300)
    assert abs(float(np.asarray(crux.eval([w])).reshape(-1)[0]) - 6.0) < 1e-2


@pytest.mark.parametrize("order", [1, 2])
def test_save_ts_samples_the_march(order):
    """`save_ts=` as in fem.solve: the march keeps its own Δt, and the trajectory is sampled at the given
    times. Every 5th step is exactly those rows of the full trajectory, for heat and for the Newmark wave;
    a time between steps is the linear interpolation of its two neighbours."""
    n = 41
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured(), time=(0.0, 0.2, n))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    lhs = ui.t.t if order == 2 else ui.t
    problem = jno.fdm([lhs - ui.xx - ui.yy, u(xb, yb) - 0.0, u(xi, yi) - jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi)])
    ts = np.linspace(0.0, 0.2, n)
    full = np.asarray(problem.solve())
    assert np.abs(np.asarray(problem.solve(save_ts=ts[::5])) - full[::5]).max() < 1e-12
    mid = np.asarray(problem.solve(save_ts=[0.5 * (ts[3] + ts[4])]))[0]
    assert np.abs(mid - 0.5 * (full[3] + full[4])).max() < 1e-12


def test_save_ts_on_a_coupled_march_and_a_steady_refusal():
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured(), time=(0.0, 0.1, 11))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u, v = d.unknown(), d.unknown()
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    ic = jno.np.sin(np.pi * xi) * jno.np.sin(np.pi * yi)
    terms = [ui.t - (ui.xx + ui.yy) + vi, vi.t - (vi.xx + vi.yy) - ui, u(xb, yb) - 0.0, v(xb, yb) - 0.0, u(xi, yi) - ic]
    full = np.asarray(jno.fdm(terms).solve())
    got = np.asarray(jno.fdm(terms).solve(save_ts=np.linspace(0.0, 0.1, 11)[::2]))
    assert got.shape == (6, 2, full.shape[2]) and np.abs(got - full[::2]).max() < 1e-12
    ds = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured().domain()
    xs, ys_, _ = ds.variable("interior", split=True)
    xsb, ysb, _ = ds.variable("boundary", split=True)
    w = ds.unknown()
    wi = w.bind(x=xs, y=ys_)
    with pytest.raises(ValueError, match="steady"):
        jno.fdm([-(wi.xx + wi.yy) - 1.0, w(xsb, ysb) - 0.0]).solve(save_ts=[0.0])


# Incompressible Navier–Stokes in primitive variables. On one collocated grid, central differences leave
# the pressure in four decoupled sub-lattices; the continuity equation is written with an O(h²) pressure
# Laplacian (pressure stabilisation, Brezzi & Pitkäranta 1984) — ∇·u − ε h² Δp — which vanishes as h → 0.
# Measured on Kovasznay flow: without it the pressure stalls at 1e-1; with ε = 0.05 it converges.


def _kovasznay(h, pressure):
    """Kovasznay flow, Re = 40 (Kovasznay 1948), on [-0.5, 1] × [-0.5, 1.5]. ``pressure``: the pressure
    boundary condition — ``"dirichlet"`` everywhere, or ``"wall"``: p on the left edge and the momentum
    balance ∂p/∂n = n·(νΔu − u·∇u) on the other three (a flux condition reading the velocity)."""
    import jno.jnp_ops as jnn

    Re, π = 40.0, np.pi
    nu, lam = 1.0 / Re, Re / 2 - np.sqrt(Re**2 / 4 + 4 * π**2)
    U = lambda x, y, m: 1 - m.exp(lam * x) * m.cos(2 * π * y)  # noqa: E731
    V = lambda x, y, m: lam / (2 * π) * m.exp(lam * x) * m.sin(2 * π * y)  # noqa: E731
    P = lambda x, y, m: 0.5 * (1 - m.exp(2 * lam * x))  # noqa: E731
    d = jno.shape.rect(-0.5, -0.5, 1.0, 1.5, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v, p = d.unknown(), d.unknown(), d.unknown()
    ui, vi, pi = u.bind(x=x, y=y), v.bind(x=x, y=y), p.bind(x=x, y=y)
    terms = [
        ui * ui.x + vi * ui.y + pi.x - nu * (ui.xx + ui.yy),
        ui * vi.x + vi * vi.y + pi.y - nu * (vi.xx + vi.yy),
        ui.x + vi.y - 0.05 * d.cell_size**2 * (pi.xx + pi.yy),
        u(xb, yb) - U(xb, yb, jnn),
        v(xb, yb) - V(xb, yb, jnn),
    ]
    if pressure == "dirichlet":
        terms.append(p(xb, yb) - P(xb, yb, jnn))
    else:
        xl, yl, _ = d.variable("left", split=True)
        terms.append(p(xl, yl) - P(xl, yl, jnn))
        for r in ("right", "bottom", "top"):
            X, Y, _ = d.variable(r, split=True)
            ub, vb, pb = u.bind(x=X, y=Y), v.bind(x=X, y=Y), p.bind(x=X, y=Y)
            mx = nu * (ub.xx + ub.yy) - (ub * ub.x + vb * ub.y)
            my = nu * (vb.xx + vb.yy) - (ub * vb.x + vb * vb.y)
            terms.append(pb.d(d.variable(r, normals=True)) - {"right": mx, "bottom": -my, "top": my}[r])
    sol = np.asarray(jno.fdm(terms).solve())
    pts = _nodes(d)
    exact_u, exact_p = U(pts[:, 0], pts[:, 1], np), P(pts[:, 0], pts[:, 1], np)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    return rel(sol[0], exact_u), rel(sol[2], exact_p)


@pytest.mark.parametrize("pressure", ["dirichlet", "wall"])
def test_navier_stokes_kovasznay(pressure):
    """Velocity second order, pressure converging, with either pressure boundary condition. The wall
    condition is a flux condition on a coupled system whose value reads another field's derivatives."""
    (u0, p0), (u1, p1) = _kovasznay(0.1, pressure), _kovasznay(0.05, pressure)
    assert u1 < 3e-3 and u0 / u1 > 3.4, (u0, u1)
    assert p1 < 3e-2 and p0 / p1 > 2.3, (p0, p1)


def _cavity(n, Re=100.0):
    """Lid-driven cavity: u = (1, 0) on the lid, no slip elsewhere, ∂p/∂n from the momentum balance on
    every wall. The pressure is then defined up to a constant, and `domain.point_region` fixes it at one
    interior node — that row replaces the continuity equation there, which is the dependent one."""
    nu = 1.0 / Re
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    u, v, p = d.unknown(), d.unknown(), d.unknown()
    ui, vi, pi = u.bind(x=x, y=y), v.bind(x=x, y=y), p.bind(x=x, y=y)
    terms = [
        ui * ui.x + vi * ui.y + pi.x - nu * (ui.xx + ui.yy),
        ui * vi.x + vi * vi.y + pi.y - nu * (vi.xx + vi.yy),
        ui.x + vi.y - 0.05 * d.cell_size**2 * (pi.xx + pi.yy),
    ]
    d.point_region("gauge", (0.5, 0.5))
    xg, yg, _ = d.variable("gauge", split=True)
    terms.append(p(xg, yg) - 0.0)
    for r in ("left", "right", "bottom", "top"):
        X, Y, _, nx, ny = d.variable(r, normals=True, split=True)
        ub, vb, pb = u.bind(x=X, y=Y), v.bind(x=X, y=Y), p.bind(x=X, y=Y)
        mx = nu * (ub.xx + ub.yy) - (ub * ub.x + vb * ub.y)  # ν Δu − (u·∇)u, the wall momentum balance
        my = nu * (vb.xx + vb.yy) - (ub * vb.x + vb * vb.y)
        lid = 1.0 if r == "top" else 0.0
        terms += [u(X, Y) - lid, v(X, Y) - 0.0, pb.d(d.variable(r, normals=True)) - (nx * mx + ny * my)]
    sol = np.asarray(jno.fdm(terms).solve())
    nx, ny = d.mesh_connectivity["grid"]["shape"]
    centre = sol[0].reshape(nx, ny)[nx // 2]  # u(0.5, y)
    # Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982), Table I, Re = 100
    gy = np.array([0.0547, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9766])
    gu = np.array([-0.03717, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.84123])
    return float(np.abs(np.interp(gy, np.linspace(0.0, 1.0, ny), centre) - gu).max()), sol


def test_navier_stokes_lid_driven_cavity():
    """Re = 100 on 33²: within 0.02 of Ghia et al. (0.0019 on 65², the slow test)."""
    err, sol = _cavity(32)
    assert err < 0.02 and np.isfinite(sol).all(), err


@pytest.mark.slow
def test_navier_stokes_lid_driven_cavity_fine():
    err, _ = _cavity(64)
    assert err < 3e-3, err


def test_unknown_region_tag_raises_and_a_point_region_pins_one_node():
    """An unrecognised region tag used to resolve to the WHOLE boundary, so a `point_region` pin fixed
    every wall node (the cavity came out 0.016 off Ghia instead of 0.0019)."""
    from jno.fdm import _TraceFDM

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    d.point_region("pin", (0.5, 0.5))
    xp, yp, _ = d.variable("pin", split=True)
    f = jno.fdm([-(ui.xx + ui.yy) - 1.0, u(xb, yb) - 0.0, u(xp, yp) - 0.0])
    assert list(f._region_nodes("pin")) == [int(np.argmin(np.linalg.norm(_nodes(d) - [0.5, 0.5], axis=1)))]
    with pytest.raises(ValueError, match="no mesh nodes found"):
        _TraceFDM._region_nodes(f, "no-such-region")


@pytest.mark.parametrize("structured", [True, False])
def test_normal_components_in_a_flux_value(structured):
    """`nx, ny` from `d.variable(region, normals=True, split=True)` are the outward normal at the flux
    nodes, so `ub.d(n) - (a nx + b ny)` is exactly ∂u/∂n for u = a x + b y. They were missing from the
    evaluation context (KeyError)."""
    shape = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25 if structured else 0.15)
    d = shape.structured().domain() if structured else jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    xl, yl, _ = d.variable("left", split=True)
    terms = [-(ui.xx + ui.yy), u(xl, yl) - (2.0 * xl + 3.0 * yl)]
    for r in ("right", "bottom", "top"):
        xr, yr, _, nx, ny = d.variable(r, normals=True, split=True)
        terms.append(u.bind(x=xr, y=yr).d(d.variable(r, normals=True)) - (2.0 * nx + 3.0 * ny))
    sol = np.asarray(jno.fdm(terms).solve()).reshape(-1)
    p = _nodes(d)
    assert np.abs(sol - (2.0 * p[:, 0] + 3.0 * p[:, 1])).max() < (1e-10 if structured else 1e-6)


@pytest.mark.parametrize("spelling", ["d(n)", "d((nx, ny))", "components", "oblique"])
@pytest.mark.parametrize("structured", [True, False])
def test_every_spelling_of_a_flux_condition(spelling, structured):
    """∂u/∂n written three ways — `ub.d(n)`, `ub.d((nx, ny))`, `nx*ub.x + ny*ub.y` — and an oblique
    condition mixing ∂u/∂x in, all recover u = 2x + 3y. The component spelling used to be read as a second
    PDE and summed into the first."""
    d = (
        jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
        if structured
        else jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    )
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    xl, yl, _ = d.variable("left", split=True)
    terms = [-(ui.xx + ui.yy), u(xl, yl) - (2.0 * xl + 3.0 * yl)]
    for r in ("right", "bottom", "top"):
        xr, yr, _, nx, ny = d.variable(r, normals=True, split=True)
        n, ub, g = d.variable(r, normals=True), u.bind(x=xr, y=yr), 2.0 * nx + 3.0 * ny
        terms.append(
            {
                "d(n)": ub.d(n) - g,
                "d((nx, ny))": ub.d((nx, ny)) - g,
                "components": nx * ub.x + ny * ub.y - g,
                "oblique": (ub.x - 2.0) + 0.5 * (ub.d(n) - g),
            }[spelling]
        )
    sol = np.asarray(jno.fdm(terms).solve()).reshape(-1)
    p = _nodes(d)
    assert np.abs(sol - (2.0 * p[:, 0] + 3.0 * p[:, 1])).max() < (1e-10 if structured else 1e-6)


def test_a_direction_needs_one_component_per_coordinate():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
    xr, yr, _, nx, ny = d.variable("right", normals=True, split=True)
    with pytest.raises(ValueError, match="one component"):
        d.unknown().bind(x=xr, y=yr).d((nx,))


# Vector unknowns: `domain.unknown(value_shape=(2,))` is one field with two components, differentiated
# with the vector views (`.grad()`, `.div()`, `.laplacian()`, `@`). It used to allocate one value per node
# (silently scalar-sized), and every vector-view derivative of a nodal field took automatic
# differentiation and came back as zeros.


def test_vector_nodal_field_derivatives():
    """Every vector-view derivative of a nodal field, on U = (x², xy), against its exact value."""
    from jno.fdm import _unwrap
    from jno.trace_evaluator import TraceEvaluator

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    U, p = d.unknown(value_shape=(2,)), d.unknown()
    assert np.shape(U.model.module.value) == (25, 2)
    Ui, pb = U.vector.bind(x=x, y=y), p.bind(x=x, y=y)
    f = jno.fdm([Ui.laplacian(), pb.laplacian()])
    P = _nodes(d)
    X, Y = P[:, 0], P[:, 1]
    dofs = jnp.concatenate([jnp.asarray(X**2), jnp.asarray(X * Y), jnp.asarray(X + 2 * Y)])
    ev = TraceEvaluator(params={**f._params_scope(), **f._inject(dofs)})
    ctx = f._eval_context({"interior"})
    at = lambda e: np.asarray(ev.evaluate(_unwrap(e), context=ctx, var_bindings={})).reshape(25, -1)[12]  # (0.5, 0.5)
    np.testing.assert_allclose(at(Ui.x), [1.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(at(Ui.xx), [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(at(Ui.grad()), [1.0, 0.0, 0.5, 0.5], atol=1e-12)  # J[i, j] = ∂u_i/∂x_j
    np.testing.assert_allclose(at(Ui.div()), [1.5], atol=1e-12)
    np.testing.assert_allclose(at(Ui.laplacian()), [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(at(Ui.grad() @ Ui), [0.25, 0.25], atol=1e-12)  # (u·∇)u
    np.testing.assert_allclose(at(Ui[0].x), [1.0], atol=1e-12)
    # a component of a nodal vector field keeps its axis, (N, 1), so `u[0] * x` stays (N, 1), not (N, N)
    assert np.asarray(ev.evaluate(_unwrap(Ui[0] * x), context=ctx, var_bindings={})).shape == (25, 1)
    np.testing.assert_allclose(at(pb.grad()), [1.0, 2.0], atol=1e-12)


def _kovasznay_vector(h):
    import jno.jnp_ops as jnn

    Re, π = 40.0, np.pi
    nu, lam = 1.0 / Re, Re / 2 - np.sqrt(Re**2 / 4 + 4 * π**2)
    Ux = lambda x, y, m: 1 - m.exp(lam * x) * m.cos(2 * π * y)  # noqa: E731
    Uy = lambda x, y, m: lam / (2 * π) * m.exp(lam * x) * m.sin(2 * π * y)  # noqa: E731
    P = lambda x, y, m: 0.5 * (1 - m.exp(2 * lam * x))  # noqa: E731
    d = jno.shape.rect(-0.5, -0.5, 1.0, 1.5, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    U, p = d.unknown(value_shape=(2,)), d.unknown()
    Ui, pi = U.vector.bind(x=x, y=y), p.bind(x=x, y=y)
    sol = np.asarray(
        jno.fdm(
            [
                Ui.grad() @ Ui + pi.grad() - nu * Ui.laplacian(),  # (u·∇)u + ∇p − νΔu
                Ui.div() - 0.05 * d.cell_size**2 * pi.laplacian(),
                U(xb, yb) - jnn.stack([Ux(xb, yb, jnn), Uy(xb, yb, jnn)], axis=-1),
                p(xb, yb) - P(xb, yb, jnn),
            ]
        ).solve()
    )
    pts = _nodes(d)
    return sol, (Ux(pts[:, 0], pts[:, 1], np), Uy(pts[:, 0], pts[:, 1], np), P(pts[:, 0], pts[:, 1], np))


def test_navier_stokes_with_a_vector_velocity():
    """The vector form is the same discretisation as three scalar unknowns, written once: it converges the
    same way (u 8.1e-3 → 2.1e-3, p 3.4e-2 → 1.0e-2), and the solution rows are [u_x, u_y, p]."""
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    (s0, e0), (s1, e1) = _kovasznay_vector(0.1), _kovasznay_vector(0.05)
    assert s1.shape == (3, len(e1[0]))
    u0, u1, p0, p1 = rel(s0[0], e0[0]), rel(s1[0], e1[0]), rel(s0[2], e0[2]), rel(s1[2], e1[2])
    assert u1 < 3e-3 and u0 / u1 > 3.4 and p1 < 2e-2 and p0 / p1 > 2.3, (u0, u1, p0, p1)
    assert rel(s1[1], e1[1]) < 2e-2


def test_vector_taylor_green_march():
    """A vector unknown in a march: Taylor–Green with BDF2, second order in the velocity."""
    import jno.jnp_ops as jnn

    nu, T = 0.1, 1.0

    def run(n):
        d = jno.domain(jno.shape.rect(0.0, 0.0, np.pi, np.pi, size=np.pi / n).structured(), time=(0.0, T, n + 1))
        x, y, t = d.variable("interior", split=True)
        xb, yb, tb = d.variable("boundary", split=True)
        x0, y0, _ = d.variable("initial", split=True)
        U, p = d.unknown(value_shape=(2,)), d.unknown()
        Ui, pi = U.vector.bind(x=x, y=y, t=t), p.bind(x=x, y=y, t=t)
        E = lambda s, k=2: jnn.exp(-k * nu * s)  # noqa: E731
        Uex = lambda X, Y, s: jnn.stack([-jnn.cos(X) * jnn.sin(Y) * E(s), jnn.sin(X) * jnn.cos(Y) * E(s)], axis=-1)  # noqa: E731
        Pex = lambda X, Y, s: -0.25 * (jnn.cos(2 * X) + jnn.cos(2 * Y)) * E(s, 4)  # noqa: E731
        traj = np.asarray(
            jno.fdm(
                [
                    Ui.t + Ui.grad() @ Ui + pi.grad() - nu * Ui.laplacian(),
                    Ui.div() - 0.05 * d.cell_size**2 * pi.laplacian(),
                    U(xb, yb) - Uex(xb, yb, tb),
                    p(xb, yb) - Pex(xb, yb, tb),
                    U(x0, y0) - Uex(x0, y0, 0.0),
                ]
            ).solve(time=jno.solve.bdf2())
        )
        P = _nodes(d)
        exact = -np.cos(P[:, 0]) * np.sin(P[:, 1]) * np.exp(-2 * nu * T)
        assert traj.shape == (n + 1, 3, len(P))
        return float(np.linalg.norm(traj[-1, 0] - exact) / np.linalg.norm(exact))

    e0, e1 = run(10), run(20)
    assert e1 < 2e-3 and e0 / e1 > 3.4, (e0, e1)


def test_vector_equation_component_mismatch_raises():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    U = d.unknown(value_shape=(2,))
    Ui = U.vector.bind(x=x, y=y)
    with pytest.raises(ValueError, match="component"):  # a scalar equation for a 2-component unknown
        jno.fdm([Ui.div(), U(xb, yb) - jno.np.stack([0.0 * xb, 0.0 * xb], axis=-1)]).solve()


def test_vector_attribute_form_equals_the_shorthand():
    """The term-by-term spelling and the vector-calculus shorthand are the same stencils: (u·∇)u, ∇p, Δu
    and ∇·u agree exactly on the Kovasznay field. `u.xx` on a vector field used to chain two first
    derivatives (the wide stencil, 24 off the Laplacian at the boundary) and Newton diverged."""
    import jno.jnp_ops as jnn
    from jno.fdm import _unwrap
    from jno.trace_evaluator import TraceEvaluator

    lam = 20.0 - np.sqrt(400.0 + 4 * np.pi**2)
    d = jno.shape.rect(-0.5, -0.5, 1.0, 1.5, size=0.1).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    U, p = d.unknown(value_shape=(2,)), d.unknown()
    u, pi = U.vector.bind(x=x, y=y), p.bind(x=x, y=y)
    ux, uy = u[0], u[1]
    f = jno.fdm([u.laplacian(), pi.laplacian()])
    P = _nodes(d)
    X, Y = P[:, 0], P[:, 1]
    fields = [1 - np.exp(lam * X) * np.cos(2 * np.pi * Y), lam / (2 * np.pi) * np.exp(lam * X) * np.sin(2 * np.pi * Y)]
    dofs = jnp.concatenate([jnp.asarray(fields[0]), jnp.asarray(fields[1]), jnp.asarray(0.5 * (1 - np.exp(2 * lam * X)))])
    ev = TraceEvaluator(params={**f._params_scope(), **f._inject(dofs)})
    ctx = f._eval_context({"interior"})
    val = lambda e: np.asarray(ev.evaluate(_unwrap(e), context=ctx, var_bindings={}))  # noqa: E731
    pairs = [
        (ux * u.x + uy * u.y, u.grad() @ u),
        (jnn.stack([pi.x, pi.y], axis=-1), pi.grad()),
        (u.xx + u.yy, u.laplacian()),
        (ux.x + uy.y, u.div()),
    ]
    for full, short in pairs:
        a, b = val(full), val(short)
        assert np.abs(a.reshape(b.shape) - b).max() < 1e-12


def test_point_region_on_a_long_time_dependent_domain():
    """A time-dependent domain stores a `point_region` once per time step; with 101 steps the node match
    used to give up (it only handled pools under 64 points) and the gauge raised "no mesh nodes"."""
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured(), time=(0.0, 1.0, 101))
    x, y, t = d.variable("interior", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    d.point_region("pin", (0.5, 0.5))
    xp, yp, _ = d.variable("pin", split=True)
    f = jno.fdm([ui.t - (ui.xx + ui.yy), u(xp, yp) - 0.0, u(xi, yi) - 1.0])
    assert list(f._region_nodes("pin")) == [int(np.argmin(np.linalg.norm(_nodes(d) - [0.5, 0.5], axis=1)))]


@pytest.mark.parametrize("spelling", ["u - g", "g - u", "u + v"])
def test_value_conditions_in_every_natural_form(spelling):
    """`u(xb, yb) + 1.0` used to be imposed as u = 0: only the `-` form was read, anything else silently
    gave g = 0. Each natural form now reads u = g; a factor on the unknown raises."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    bc = {"u - g": u(xb, yb) - (-1.0), "g - u": -1.0 - u(xb, yb), "u + v": u(xb, yb) + 1.0}[spelling]
    sol = np.asarray(jno.fdm([ui.xx + ui.yy, bc]).solve()).reshape(-1)
    assert np.abs(sol + 1.0).max() < 1e-10  # harmonic with u = −1 on the boundary: u ≡ −1
    with pytest.raises(ValueError, match="Divide out"):
        jno.fdm([ui.xx + ui.yy, 2.0 * u(xb, yb) - 2.0]).solve()


def test_a_data_field_as_a_pde_coefficient():
    """A known nodal field (a `jno.np.parameter` holding data, no optimizer) inside the PDE failed with
    "No model for Model N": only trainable parameters were in the evaluation scope. −∇·(κ∇u) = f with
    κ = 1 + x² given as nodal data reproduces the formula-κ solve."""
    import equinox as eqx

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    P = _nodes(d)
    K = jno.np.parameter((len(P),), name="kappa")
    K.model.module = eqx.tree_at(lambda m: m.value, K.model.module, jnp.asarray(1 + P[:, 0] ** 2))
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 1.0 + 0.0 * x
    data = np.asarray(jno.fdm([-(K * ui.x).x - (K * ui.y).y - f, u(xb, yb) - 0.0]).solve()).reshape(-1)
    k = 1 + x**2
    formula = np.asarray(
        jno.fdm([-(k * ui.x).d(x, scheme=jno.fd(average="arithmetic")) - (k * ui.y).y - f, u(xb, yb) - 0.0]).solve()
    ).reshape(-1)
    assert np.isfinite(data).all() and np.abs(data - formula).max() < 1e-10 and np.abs(data).max() > 1e-3


# ---------------------------------------------------------------------------------------------------
# a linear problem on a structured grid: one Krylov solve, no Newton
# ---------------------------------------------------------------------------------------------------
def _grid_problem(n, terms_of, dim=2):
    d = (
        (
            jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n)
            if dim == 2
            else jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=1.0 / n)
        )
        .structured()
        .domain()
    )
    c = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(**dict(zip("xyz", c[:dim])))
    return d, jno.fdm(terms_of(u, ui, c, cb))


def test_a_linear_grid_problem_is_one_conjugate_gradient_solve():
    """-Δu = f with u = 0: symmetric after the Dirichlet rows are eliminated, so conjugate gradients with the
    multigrid V-cycle -- the same second-order answer the Newton path gave."""
    import jno.jnp_ops as jnn

    d, prob = _grid_problem(
        32,
        lambda u, ui, c, cb: [
            -ui.d2(c[0]) - ui.d2(c[1]) - 2 * np.pi**2 * jnn.sin(np.pi * c[0]) * jnn.sin(np.pi * c[1]),
            u(cb[0], cb[1]) - 0.0,
        ],
    )
    sol = np.asarray(prob.solve()).reshape(-1)
    assert prob._grid_linear_ok() and prob._grid_linear_symmetric_flag
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)) < 1e-3


def test_an_advection_diffusion_grid_problem_takes_gmres():
    """-Δu + b·∇u = f is not symmetric: the probe sees it, and GMRES (checking every iteration) solves it.
    Manufactured u = sin(πx) sin(πy) with b = (1, 2)."""
    import jno.jnp_ops as jnn

    def terms(u, ui, c, cb):
        s, co = jnn.sin, jnn.cos
        f = (
            2 * np.pi**2 * s(np.pi * c[0]) * s(np.pi * c[1])
            + np.pi * co(np.pi * c[0]) * s(np.pi * c[1])
            + 2 * np.pi * s(np.pi * c[0]) * co(np.pi * c[1])
        )
        return [-ui.d2(c[0]) - ui.d2(c[1]) + 1.0 * ui.d(c[0]) + 2.0 * ui.d(c[1]) - f, u(cb[0], cb[1]) - 0.0]

    d, prob = _grid_problem(32, terms)
    sol = np.asarray(prob.solve()).reshape(-1)
    assert prob._grid_linear_ok() and not prob._grid_linear_symmetric_flag
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert float(np.linalg.norm(sol - exact) / np.linalg.norm(exact)) < 2e-3


def test_the_grid_linear_path_is_differentiable_in_the_source():
    """d/da of a misfit through -Δu = a·f on a structured grid: the adjoint through custom_linear_solve
    against a central difference of the same solve."""
    import jax

    import jno.jnp_ops as jnn

    obs = None

    def loss(a):
        _, prob = _grid_problem(
            16,
            lambda u, ui, c, cb: [
                -ui.d2(c[0]) - ui.d2(c[1]) - a * 2 * np.pi**2 * jnn.sin(np.pi * c[0]) * jnn.sin(np.pi * c[1]),
                u(cb[0], cb[1]) - 0.0,
            ],
        )
        assert prob._grid_linear_ok()
        sol = jnp.asarray(prob.solve()).reshape(-1)
        return jnp.mean((sol - obs) ** 2)

    d, prob0 = _grid_problem(16, lambda u, ui, c, cb: [-ui.d2(c[0]) - ui.d2(c[1]) - 1.0, u(cb[0], cb[1]) - 0.0])
    p = _nodes(d)
    obs = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    g = float(jax.grad(loss)(1.5))
    fd = (float(loss(1.5 + 1e-4)) - float(loss(1.5 - 1e-4))) / 2e-4
    assert g == pytest.approx(fd, rel=1e-6)


def test_a_grid_problem_with_a_flux_boundary_keeps_the_newton_path():
    """Eliminating the Dirichlet rows leaves the V-cycle's interior only when the whole ring is Dirichlet."""
    _, prob = _grid_problem(
        16,
        lambda u, ui, c, cb: [-ui.d2(c[0]) - ui.d2(c[1]) - 1.0, u(cb[0], cb[1]) - 0.0],
    )
    assert prob._grid_linear_ok()
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / 16).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _, nx, ny = d.variable("right", normals=True, split=True)
    ur = u.bind(x=xr, y=yr)
    terms = [-ui.d2(x) - ui.d2(y) - 1.0, u(xl, yl) - 0.0, ur.d(d.variable("right", normals=True)) - 0.0]
    assert not jno.fdm(terms)._grid_linear_ok()


def test_krylov_slots_on_a_grid_stay_matrix_free():
    """linear=cg / gmres with precond=gmg on a structured linear problem runs on the JVP, not on an assembled
    matrix (which cost 3x the memory and 25x the time at 1M nodes, measured) -- same answer as the default."""
    import jno.jnp_ops as jnn

    def terms(u, ui, c, cb):
        return [
            -ui.d2(c[0]) - ui.d2(c[1]) - 2 * np.pi**2 * jnn.sin(np.pi * c[0]) * jnn.sin(np.pi * c[1]),
            u(cb[0], cb[1]) - 0.0,
        ]

    _, ref = _grid_problem(32, terms)
    base = np.asarray(ref.solve()).reshape(-1)
    for linear in (jno.solve.cg(tol=1e-10), jno.solve.gmres(tol=1e-10)):
        _, prob = _grid_problem(32, terms)
        sol = np.asarray(prob.solve(linear=linear, precond=jno.precond.gmg())).reshape(-1)
        assert "_sparsity_cache" not in prob.__dict__, "the slot route assembled a matrix"
        np.testing.assert_allclose(sol, base, atol=1e-8)


def test_cg_on_a_nonsymmetric_grid_problem_raises():
    import jno.jnp_ops as jnn

    _, prob = _grid_problem(
        16,
        lambda u, ui, c, cb: [-ui.d2(c[0]) - ui.d2(c[1]) + 3.0 * ui.d(c[0]) - jnn.sin(np.pi * c[0]), u(cb[0], cb[1]) - 0.0],
    )
    with pytest.raises(ValueError, match="symmetric"):
        prob.solve(linear=jno.solve.cg(), precond=jno.precond.gmg())
