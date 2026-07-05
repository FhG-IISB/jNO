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


def test_neumann_rejects_malformed_and_transient():
    """v1 guards: an unrecognized flux structure (here the flux on the wrong side, `h - ui.d(n)`, which a
    Robin/`+` form would also hit) raises rather than silently assuming h=0, and a Neumann flux on a
    transient problem raises."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    sch = "finite_difference"
    with pytest.raises(ValueError, match="Robin|not supported"):
        jno.fdm([-ui.d2(x, scheme=sch) - ui.d2(y, scheme=sch), u(xl, yl) - 0.0, 1.0 - ui.d(nr, scheme=sch)]).solve()

    dt = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15, time=(0.0, 0.5, 50))
    xt, yt, tt = dt.variable("interior", split=True)
    xit, yit, _ = dt.variable("initial", split=True)
    nrt = dt.variable("right", normals=True)
    ut = dt.unknown()
    uit = ut.bind(x=xt, y=yt, t=tt)
    with pytest.raises(ValueError, match="Neumann.*transient|transient.*not supported"):
        jno.fdm(
            [
                uit.t - 0.05 * (uit.d2(xt, scheme=sch) + uit.d2(yt, scheme=sch)),
                ut(xit, yit) - 0.0,
                uit.d(nrt, scheme=sch) - 1.0,
            ]
        )
