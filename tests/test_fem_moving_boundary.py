"""Moving-boundary driver — `FEM.solve(move=MovingBoundary(...))` on a transient problem.

The domain boundary moves each step by a prescribed velocity; the mesh deforms to follow it
(`harmonic_extension` + `move_mesh`), the physics marches on the moving mesh, and the state is carried
across each move. Headline oracle: the **method of manufactured solutions** on a *deforming* domain.

A unit square is stretched vertically to [0,1]×[0,L(t)], L(t)=1+ct, by the prescribed boundary velocity
v_y = c·y/(1+ct). We manufacture the exact field

    u*(x, y, t) = sin(πx) · sin(π y / L(t))        (≡ 0 on the whole moving boundary, all t)

feed the required source f = u*_t − κΔu* (computed by autodiff, so no hand-derivation to get wrong) into
u_t = κΔu + f with homogeneous Dirichlet, and require the driver's trajectory to reproduce u* at the
(moved) node positions. That single test exercises the mesh motion, the state transfer across moves, and
the physics march on a genuinely changing domain — if any were wrong, u* would not come back. Plus the
fail-loud scope guards.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # FEM assembly/solves are float64
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_moving_boundary_matches_manufactured_solution_on_a_deforming_domain():
    C, KAPPA, T_END, NT = 0.5, 0.05, 0.4, 41  # stretch rate, diffusivity, horizon, steps (dt=0.01)

    def height(t):
        return 1.0 + C * t

    def u_star(x, y, t):  # ≡ 0 on x∈{0,1} and y∈{0, L(t)} for all t (broadcasts; also scalar-safe)
        return jnp.sin(PI * x) * jnp.sin(PI * y / height(t))

    def f_src(x, y, t):  # manufactured forcing f = u*_t − κΔu*, hand-derived (broadcasts over jno.fn's arrays)
        lt = height(t)
        phi = PI * y / lt  # Δu* = −π²(1 + 1/L²)u* ;  u*_t = sin(πx)cos(φ)·(−πy c/L²)  (φ time-dep via L)
        ustar = jnp.sin(PI * x) * jnp.sin(phi)
        u_t = jnp.sin(PI * x) * jnp.cos(phi) * (-PI * y * C / lt**2)
        lap = -(PI**2) * (1.0 + 1.0 / lt**2) * ustar
        return u_t - KAPPA * lap

    d = jno.Shape.rect(0, 0, 1, 1, size=0.06).domain(time=(0.0, T_END, NT))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    weak = ui.t * vi + KAPPA * (ui.x * vi.x + ui.y * vi.y) - jno.fn(f_src, [xi, yi, ti]) * vi  # u_t = κΔu + f
    ic = jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])  # u*(·, 0), L(0)=1
    fem = jno.fem([weak, u(xb, yb) - 0.0, u(xi0, yi0) - ic])

    def velocity(t, x):  # v_y = c·y/(1+ct) makes y(t)=y0·(1+ct) exactly under forward-Euler moves; v_x=0
        v = np.zeros_like(x)
        v[:, 1] = C * x[:, 1] / (1.0 + C * t)
        return v

    traj = fem.solve(move=jno.MovingBoundary(velocity=velocity, every=1))

    # the mesh actually deformed to track [0,1]×[0,1+cT]
    ymax = np.asarray(traj.meshes[-1][0])[:, 1].max()
    assert abs(ymax - height(T_END)) < 0.02, f"final domain height {ymax:.3f} did not track L(T)={height(T_END):.3f}"

    worst = 0.0
    for k, t in enumerate(np.asarray(traj.times)):
        pts = np.asarray(traj.meshes[k][0])  # this frame's moved node positions
        uk = np.asarray(traj.states[k])
        exact = np.asarray(jax.vmap(lambda p, tt=float(t): u_star(p[0], p[1], tt))(jnp.asarray(pts)))
        worst = max(worst, np.linalg.norm(uk - exact) / max(np.linalg.norm(exact), 1e-12))
    assert worst < 0.08, (
        f"moving-boundary trajectory did not reproduce the manufactured solution: worst rel L2 = {worst:.3f}"
    )


def test_moving_boundary_resample_and_trajectory_shape():
    """The result is an AdaptiveTrajectory whose frames live on different (moved) meshes; resample onto a
    fixed reference gives a uniform array."""
    from jno.utils.solver.fem_adapt import AdaptiveTrajectory

    d = jno.Shape.rect(0, 0, 1, 1, size=0.1).domain(time=(0.0, 0.2, 11))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0]),
        ]
    )

    def velocity(t, x):
        v = np.zeros_like(x)
        v[:, 1] = 0.3 * x[:, 1]  # stretch upward
        return v

    traj = fem.solve(move=jno.MovingBoundary(velocity=velocity, every=2))
    assert isinstance(traj, AdaptiveTrajectory)
    assert len(traj) == 11 and len(traj.states) == len(traj.meshes) == 11
    assert np.asarray(traj.meshes[-1][0])[:, 1].max() > 1.05  # the top actually rose

    ref = jno.Shape.rect(0, 0, 1, 1.2, size=0.1).domain()
    ys = np.asarray(traj.resample(ref))
    assert ys.shape == (len(traj), len(np.asarray(ref.mesh.points)))


# ── fail-loud scope guards ────────────────────────────────────────────────────
def _mini_transient():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain(time=(0.0, 0.1, 6))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - 1.0])
    return fem


def test_steady_problem_with_move_raises():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.2).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    with pytest.raises(NotImplementedError, match="transient"):
        fem.solve(move=jno.MovingBoundary(velocity=lambda t, x: np.zeros_like(x)))


def test_move_with_solver_slots_raises():
    with pytest.raises(NotImplementedError, match="does not compose"):
        _mini_transient().solve(move=jno.MovingBoundary(velocity=lambda t, x: np.zeros_like(x)), linear=jno.solve.gmres())


def test_velocity_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"\(n_boundary, dim\)"):
        _mini_transient().solve(move=jno.MovingBoundary(velocity=lambda t, x: np.zeros((x.shape[0], 3))))


def test_non_callable_velocity_raises():
    with pytest.raises(TypeError, match="callable"):
        _mini_transient().solve(move=jno.MovingBoundary(velocity=np.zeros((3, 2))))


# ── state-dependent (physics-driven) velocity ────────────────────────────────
def _heat_blob(mesh=0.12, t_end=0.2, nt=11, u0=1.0, insulated=True):
    d = jno.Shape.rect(0, 0, 1, 1, size=mesh).domain(time=(0.0, t_end, nt))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    terms = [ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y), u(xi0, yi0) - u0]  # insulated (natural) ⇒ u stays u0
    return jno.fem(terms), d


def test_state_dependent_velocity_receives_the_current_field():
    """The 4-arg form velocity(t, x, state, domain) is handed the CURRENT nodal field on the CURRENT mesh
    (the solution before this move) — the prerequisite for any physics-driven law."""
    fem, _ = _heat_blob(t_end=0.2, nt=11)
    seen = []

    def vel(t, x, state, dom):
        seen.append((np.asarray(state).copy(), int(np.asarray(dom.mesh.points).shape[0])))
        return np.zeros_like(x)  # no motion → an ordinary transient march

    traj = fem.solve(move=jno.MovingBoundary(velocity=vel, every=1))
    assert len(seen) == len(traj) - 1  # velocity called once before each step
    for k, (st, nv) in enumerate(seen):
        assert st.shape[0] == nv  # state aligned with the (unchanged) mesh
        assert np.allclose(st, np.asarray(traj.states[k]), atol=1e-8)  # it IS the field at that step


def test_state_dependent_velocity_drives_motion():
    """The payoff: a velocity that reads the current field off `state` and moves the boundary with it.
    With a constant field u≡1 (insulated), the top rises at RATE·u = RATE, so its height reaches
    1 + RATE·T exactly — the motion is genuinely driven by the state (double u ⇒ double the rate)."""
    RATE, T_END = 0.5, 0.4
    fem, _ = _heat_blob(mesh=0.1, t_end=T_END, nt=21, u0=1.0)

    def vel(t, x, state, dom):
        u_bdry = float(np.asarray(state).mean())  # ≈ 1 — the current field sets the rate (state-dependent)
        v = np.zeros_like(x)
        top = x[:, 1] > x[:, 1].max() - 1e-6
        v[top, 1] = RATE * u_bdry
        return v

    traj = fem.solve(move=jno.MovingBoundary(velocity=vel, every=1))
    top_final = np.asarray(traj.meshes[-1][0])[:, 1].max()
    assert abs(top_final - (1.0 + RATE * T_END)) < 0.05, f"top reached {top_final:.3f}, expected {1.0 + RATE * T_END:.3f}"
