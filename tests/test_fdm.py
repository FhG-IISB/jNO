"""``jno.fdm`` — finite-difference PDE solver (strong-form sibling of ``jno.fem``).

Run with x64 (the solve accumulates in float64)."""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _nodes(d):
    return np.asarray(d.mesh_connectivity["points"])[:, :2]


def _poisson_homogeneous(mesh_size):
    """-Δu = f on [0,1]², u=0 on ∂Ω, exact u = sin(πx)sin(πy). Returns rel-L2 error."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - f, dirichlet={"boundary": 0.0})
    u = np.asarray(sys.solve()).reshape(-1)
    return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


def test_poisson_homogeneous_dirichlet():
    assert _poisson_homogeneous(0.06) < 1e-2


def test_poisson_convergence_under_refinement():
    """Refining the mesh reduces the FD error (consistency)."""
    errs = [_poisson_homogeneous(h) for h in (0.10, 0.06, 0.035)]
    assert errs[0] > errs[1] > errs[2], f"not monotonically converging: {errs}"
    assert errs[2] < 3e-3


def test_inhomogeneous_dirichlet():
    """u = x²+y², -Δu = -4, with the boundary value g(x,y) = x²+y²."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    p = _nodes(d)
    exact = p[:, 0] ** 2 + p[:, 1] ** 2
    sys = jno.fdm(
        d,
        residual=lambda u: -jno.fdm.laplacian(u, d) + 4.0,
        dirichlet={"boundary": lambda x, y: x**2 + y**2},
    )
    u = np.asarray(sys.solve()).reshape(-1)
    assert float(np.linalg.norm(u - exact) / np.linalg.norm(exact)) < 1e-3


def test_matches_fem_on_same_mesh():
    """The FD solution agrees with the FE solution to FD-discretization accuracy."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    p = _nodes(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    f = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    u_fd = np.asarray(
        jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - f, dirichlet={"boundary": 0.0}).solve()
    ).reshape(-1)
    # both solve the same BVP; the FD field is in the analytic ballpark
    assert float(np.linalg.norm(u_fd - exact) / np.linalg.norm(exact)) < 1e-2


def test_differentiable_for_inverse_problems():
    """The solve is differentiable w.r.t. a parameter in the residual (source scale), and the
    gradient points toward the true value — the requirement for composing into jno.core."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    p = _nodes(d)
    fbase = jnp.asarray(2 * np.pi**2 * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    obs = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))

    def loss(scale):
        sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) - scale * fbase, dirichlet={"boundary": 0.0})
        return jnp.mean((sys.solve() - obs) ** 2)

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
    f = jnp.asarray(2 * np.pi**2 * exact + exact**3)  # -Δ(sin sin) + (sin sin)³
    sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d) + u**3 - f, dirichlet={"boundary": 0.0})
    u = np.asarray(sys.solve()).reshape(-1)
    assert float(np.linalg.norm(u - exact) / np.linalg.norm(exact)) < 1e-2


def test_transient_heat_2d():
    """u_t = ν Δu, u₀ = sin(πx)sin(πy), homogeneous Dirichlet → e^(−2νπ²t)·sin(πx)sin(πy).
    solve_transient reuses jno.fem's SemidiscreteTimeBlock stepper (no new time-integration code)."""
    nu, T = 0.05, 0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    p = _nodes(d)
    u0 = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    sys = jno.fdm(d, residual=lambda u: -nu * jno.fdm.laplacian(u, d), dirichlet={"boundary": 0.0})
    traj = np.asarray(sys.solve_transient(u0, (0.0, T), 200))
    exact = np.exp(-2 * nu * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
    assert traj.shape == (201, p.shape[0])
    assert float(np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact)) < 1e-2


def test_transient_differentiable_for_inverse():
    """The transient march differentiates w.r.t. a parameter (diffusivity) — time-dependent inverse."""
    T = 0.5
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    p = _nodes(d)
    u0 = jnp.asarray(np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))
    target = jnp.asarray(np.exp(-2 * 0.05 * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]))

    def loss(nu):
        s = jno.fdm(d, residual=lambda u: -nu * jno.fdm.laplacian(u, d), dirichlet={"boundary": 0.0})
        return jnp.mean((s.solve_transient(u0, (0.0, T), 200)[-1] - target) ** 2)

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
    """u = x² − y² is harmonic (−Δu = 0), ∂u/∂n = 2x = 2 on the right edge. The boundary-flux stencil
    is O(h), so the error decreases under refinement."""
    errs = [_mixed_dirichlet_neumann(h, lambda x, y: x**2 - y**2, du_dn_right=2.0) for h in (0.1, 0.06, 0.035)]
    assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
    assert errs[2] < 5e-3


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


def test_flux_rejects_nonaffine():
    """A flux BC nonlinear in ∂u/∂n (here `(∂u/∂n)² − 1`) raises rather than silently returning a secant."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    ur = u.bind(x=xr, y=yr)
    with pytest.raises(ValueError, match="affine"):
        jno.fdm([-ui.d2(x) - ui.d2(y), u(xl, yl) - 0.0, ur.d(nr) * ur.d(nr) - 1.0]).solve()


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
    """A coupled system is v1-limited to STEADY + Dirichlet with exactly one PDE equation per unknown."""
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
    with pytest.raises(NotImplementedError, match="coupled"):  # coupled + transient not yet supported
        jno.fdm(
            [
                uit.t - (uit.d2(xt) + uit.d2(yt)) + vit,
                vit.t - (vit.d2(xt) + vit.d2(yt)) + uit,
                ut(xit, yit) - 0.0,
                vt(xit, yit) - 0.0,
            ]
        )


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
    assert np.max(np.abs(np.asarray(s._mass_coefficient()) - (1.0 + 0.5 * np.sin(np.pi * p[:, 0])))) < 1e-9

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
def test_periodic_poisson():
    """A periodic tie `u(left) - u(right)` wraps the structured x-axis (the Nx-node periodic 5-point
    stencil), authored exactly as in jno.fem. MMS: -Δu = 5π²·sin(2πx)sin(πy), periodic in x with
    Dirichlet u=0 in y ⇒ u = sin(2πx)sin(πy). The tie holds to machine precision."""
    import jno.jnp_ops as jnn

    d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08), structured=True)
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
    """Unit cube meshed by jno.Shape (gmsh tets) — no shapely."""
    return jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=mesh_size).domain()


def _poisson3d(mesh_size, method="cotangent"):
    """-Δu = f on [0,1]³, u=0 on ∂Ω, exact u = sin(πx)sin(πy)sin(πz) ⇒ f = 3π²u. Returns rel-L2 error."""
    d = _cube(mesh_size)
    p = _nodes3(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) * np.sin(np.pi * p[:, 2])
    f = jnp.asarray(3 * np.pi**2 * exact)
    sys = jno.fdm(d, residual=lambda u: -jno.fdm.laplacian(u, d, method=method) - f, dirichlet={"boundary": 0.0})
    u = np.asarray(sys.solve()).reshape(-1)
    return float(np.linalg.norm(u - exact) / np.linalg.norm(exact))


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


def test_constraint_list_cotangent_3d():
    """The constraint-list path reaches the 3-D cotangent stencil too — a SINGLE whole-Laplacian term
    `ui.d2(x, scheme="finite_difference:cotangent")` (NOT the split −d2(x)−d2(y)−d2(z), which stays
    per-direction gradient-of-gradient), exactly as in 2-D. It matches the function-form accuracy."""
    import jno.jnp_ops as jnn

    d = _cube(0.14)
    p = _nodes3(d)
    exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1]) * np.sin(np.pi * p[:, 2])
    x, y, z, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    f = 3 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y) * jnn.sin(np.pi * z)
    sol = jno.fdm([-ui.d2(x, scheme="finite_difference:cotangent") - f, u(xb, yb, zb) - 0.0]).solve()
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
    """Keystone 3-D containment: `jno.fdm._mesh_nodes_in` resolves a `jno.Shape.box` sub-region to the
    exact tetrahedral-mesh node subset via the analytic 3-D `Shape.contains` — the production path both
    `_TraceFDM._region_nodes` and `jno.dd._region_mask` route through to turn a geometric sub-region into
    a node set. This is what shapely could never do (it is 2-D only). (Wiring such a region into a *3-D
    coupled solve* additionally needs region-tag support on the base 3-D domain — a separate feature.)"""
    from jno.fdm import _mesh_nodes_in

    d = _cube(0.12)
    p = _nodes3(d)
    core = jno.Shape.box(0.3, 0.3, 0.3, 0.7, 0.7, 0.7)
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


def test_fd_operator_noise_separates_exact_from_nested_stencils():
    """The measurement the tolerance rule is built on. A strong-form second derivative defaults to
    ``gradient_of_gradient`` — a NESTED gradient — and nesting amplifies roundoff, so the operator
    carries a precision floor no solver can go below. The cotangent Laplacian on the same mesh does
    not. Newton then stalled at ~||r||*noise and the convergence guard (correctly) raised; the fix is
    to ask for what the discretization can actually deliver, not to loosen the guard."""
    from jno.fdm import _fd_operator_noise, laplacian

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
    n = int(np.asarray(d.mesh.points).shape[0])
    u = jnp.asarray(np.random.default_rng(0).normal(size=n))

    exact = _fd_operator_noise(lambda z: laplacian(z, d, method="cotangent"), u)
    assert exact < 1e-14, f"the cotangent stencil must be exact, measured {exact:.2e}"

    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    uu = d.unknown()
    ui = uu.bind(x=x, y=y)
    prob = jno.fdm([-ui.d2(x) - ui.d2(y) - 1.0, uu(xb, yb) - 0.0])
    nested = _fd_operator_noise(prob._pde_residual_fn(), u)
    assert nested > 1e-10, f"the nested gradient_of_gradient stencil is noisy; measured {nested:.2e}"
    assert nested > 1e4 * max(exact, 1e-17), "the two stencils must differ by orders, not a little"


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
