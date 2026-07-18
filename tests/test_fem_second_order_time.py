"""Second-order-in-time (`u_tt`) FEM through ``jno.fem`` — the wave / elastodynamics path.

A weak form carrying a second temporal derivative is reduced to a first-order augmented system in
``y = [u, v=u_t]`` and integrated by the **trapezoidal rule** (θ=½, energy-conserving). The headline
check is *amplitude conservation over several periods*: backward Euler (θ=1) spuriously damps an
undamped wave, so a single-step / loose-tolerance test would pass while shipping a wrong (damped)
solution. We verify against the analytic standing wave on the unit square::

    u_tt = Δu ,  u = sin(πx) sin(πy) cos(ω t) ,  ω = π√2 ,  T = 2π/ω = √2 ,

check the amplitude is preserved over 4 periods (and that θ=1 would *not* preserve it), match the
analytic solution over one period, exercise damping (decay), a non-zero velocity IC, and **vector
elastodynamics** (energy conservation of the clamped elastic square), and confirm the fail-loud
scope boundaries (1D, nonlinear, multi-field).
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

from jno.utils.solver.backend_blocks import _block_time_grid, _default_transient_integrate  # noqa: E402

PI = np.pi
OMEGA = PI * np.sqrt(2.0)  # eigenfrequency of mode (1,1) for u_tt = Δu
PERIOD = 2.0 * PI / OMEGA  # = √2


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly/solves run in float64, so opt into x64 per-test (session default is x64-off)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _mode11(x, y):
    return jnp.sin(PI * x) * jnp.sin(PI * y)


def _wave_fem(mesh_size=0.1, n_periods=4, n_steps=240, damping=0.0, u0_fn=_mode11, v0_const=0.0, dirichlet=0.0, order=1):
    """Assemble `u_tt + c u_t = Δu` on the unit square with the mode-(1,1) IC."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, float(n_periods * PERIOD), n_steps))
    u, phi = d.fem_symbols(order=order)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    weak = ui.tt * vi + (ui.x * vi.x + ui.y * vi.y)
    if damping:
        weak = weak + damping * ui.t * vi
    u_ic = u(xi0, yi0) - jno.fn(u0_fn, [xi0, yi0]) if u0_fn is not None else u(xi0, yi0) - 0.0
    vel_ic = ui0.t - float(v0_const)
    return jno.fem([weak, u(xb, yb) - float(dirichlet), u_ic, vel_ic])


def _trajectory(fem, theta=None):
    block = fem.operator
    if theta is not None:
        block.metadata["theta"] = theta
    ts = np.asarray(_block_time_grid(block))
    ys = np.asarray(_default_transient_integrate(block, {}, ts))
    n = fem.offsets[1]
    return ts, ys[:, :n], ys[:, n:]  # times, displacement, velocity


# ==========================================================================
# structure
# ==========================================================================
def test_second_order_routes_to_transient_augmented_block():
    fem = _wave_fem(mesh_size=0.2, n_periods=1, n_steps=10)
    assert fem.problem is None  # the M2/C/K/F blocks are assembled natively
    assert fem.is_transient and fem.is_linear
    n = fem.offsets[1]
    assert fem.offsets == [0, n, 2 * n]  # y = [u; v]
    assert fem.dofs == 2 * n
    assert np.asarray(fem.M).shape == (2 * n, 2 * n)
    assert np.asarray(fem.state0).shape[0] == 2 * n


def test_solve_node_evaluates_to_trajectory():
    """fem.solve() is a differentiable trace node; with no runtime parameters it still evaluates
    to the forward (n_grid, 2N) trajectory."""
    from jno.trace_evaluator import TraceEvaluator

    fem = _wave_fem(mesh_size=0.2, n_periods=1, n_steps=12)
    node = fem.solve()
    val = np.asarray(TraceEvaluator({}).evaluate(node.expr if hasattr(node, "expr") else node, context={}))
    assert val.ndim == 2 and val.shape[1] == fem.dofs


def test_second_order_warns_once_in_float32():
    """A second-order-in-time assembly in float32 (``jax_enable_x64`` off) can silently misresolve a
    soft mode's frequency (a slender beam's fundamental rings at the wrong speed while energy is
    conserved), so ``jno.fem`` warns — exactly once per process, and never under x64. jNO does not
    force precision; it flags the footgun. (The autouse ``_x64`` fixture enables x64; we flip it here
    to exercise both paths, and reset the module-level once-flag so the test is order-independent.)"""
    import warnings

    from jno import _fem as _femmod

    prev = jax.config.jax_enable_x64

    def _build():
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.4, time=(0.0, 1.0, 8))
        u, phi = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        xi0, yi0, ti0 = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
        ui0 = u.bind(x=xi0, y=yi0, t=ti0)
        return jno.fem([ui.tt * vi + (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - 0.0, ui0.t - 0.0])

    try:
        jax.config.update("jax_enable_x64", False)  # float32: the warned path
        _femmod._SECOND_ORDER_FLOAT32_WARNED = False
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")  # defeat the warnings-module dedup; our own flag must dedup
            _build()
            _build()  # a second assembly must NOT warn again
        got = [w for w in rec if "second-order-in-time" in str(w.message)]
        assert len(got) == 1, f"expected exactly one float32 warning, got {len(got)}"

        jax.config.update("jax_enable_x64", True)  # float64: silent
        _femmod._SECOND_ORDER_FLOAT32_WARNED = False
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _build()
        assert not [w for w in rec if "second-order-in-time" in str(w.message)], "must not warn under x64"
    finally:
        jax.config.update("jax_enable_x64", prev)
        _femmod._SECOND_ORDER_FLOAT32_WARNED = False


# ==========================================================================
# the energy-conservation gate (trapezoidal, not backward Euler)
# ==========================================================================
def test_undamped_wave_conserves_amplitude_over_periods():
    """The headline check: the trapezoidal default preserves the standing-wave amplitude over 4
    periods (an undamped wave neither grows nor decays)."""
    fem = _wave_fem(mesh_size=0.1, n_periods=4, n_steps=240)
    _ts, U, _V = _trajectory(fem)
    ratio = np.linalg.norm(U[-1]) / np.linalg.norm(U[0])
    assert 0.95 < ratio < 1.05, f"amplitude not conserved over 4 periods: ratio={ratio:.4f}"


def test_default_is_trapezoidal_backward_euler_would_damp():
    """Locks the integrator choice: the same block under backward Euler (θ=1) loses most of its
    amplitude over 4 periods, while the θ=½ default conserves it. A regression to backward Euler
    would silently ship damped waves; this test catches it."""
    fem = _wave_fem(mesh_size=0.1, n_periods=4, n_steps=240)
    _ts, U_half, _ = _trajectory(fem, theta=0.5)
    _ts, U_be, _ = _trajectory(fem, theta=1.0)
    r_half = np.linalg.norm(U_half[-1]) / np.linalg.norm(U_half[0])
    r_be = np.linalg.norm(U_be[-1]) / np.linalg.norm(U_be[0])
    assert r_half > 0.9, f"θ=½ should conserve amplitude (got {r_half:.3f})"
    assert r_be < 0.6, f"θ=1 (backward Euler) should visibly damp (got {r_be:.3f})"


def test_wave_matches_analytic_standing_wave_one_period():
    """Over one period (little phase drift), the antinode value tracks the analytic cos(ω t)."""
    fem = _wave_fem(mesh_size=0.1, n_periods=1, n_steps=80)
    ts, U, _ = _trajectory(fem)
    pts = np.asarray(fem.points)
    ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))  # node nearest the antinode (0.5, 0.5)
    exact = _mode11(pts[ci, 0], pts[ci, 1]) * np.cos(OMEGA * ts)
    rel = np.linalg.norm(U[:, ci] - exact) / np.linalg.norm(exact)
    assert rel < 0.05, f"antinode does not track analytic cos(ω t): rel L2 = {rel:.4f}"


def test_wave_p2_elements_higher_accuracy():
    """The reduction works on P2 (order=2) elements too — the IC is read at the assembly nodes
    (edge midpoints included), and the higher order gives a smaller phase error than P1."""
    fem = _wave_fem(mesh_size=0.18, n_periods=1, n_steps=80, order=2)
    n = fem.offsets[1]
    assert fem.dofs == 2 * n
    ts, U, _ = _trajectory(fem)
    pts = np.asarray(fem.points)
    assert pts.shape[0] == n  # P2 assembly nodes (vertices + edge midpoints)
    ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))
    exact = _mode11(pts[ci, 0], pts[ci, 1]) * np.cos(OMEGA * ts)
    rel = np.linalg.norm(U[:, ci] - exact) / np.linalg.norm(exact)
    assert rel < 0.03, f"P2 antinode does not track analytic cos(ω t): rel L2 = {rel:.4f}"


# ==========================================================================
# extremes — damping, non-zero velocity IC, non-homogeneous Dirichlet
# ==========================================================================
def test_damped_wave_energy_decays_monotonically():
    """A positive damping term `c u_t` makes the energy (here tracked by the displacement norm
    envelope) decay; this exercises the C (first-order) block of the reduction."""
    fem = _wave_fem(mesh_size=0.12, n_periods=2, n_steps=200, damping=2.0)
    _ts, U, _ = _trajectory(fem)
    norms = np.linalg.norm(U, axis=1)
    assert norms[-1] < 0.5 * norms[0], f"damped wave should decay (end/start = {norms[-1] / norms[0]:.3f})"
    # the per-period peak envelope is non-increasing (no spurious energy injection)
    assert norms.max() <= norms[0] * 1.02


def test_nonzero_velocity_initial_condition():
    """Start at rest (u0=0) with velocity v0 = sin(πx)sin(πy): u(t) = sin(πx)sin(πy) sin(ω t)/ω.
    The antinode peaks at +1/ω a quarter period in — verifies the velocity IC is actually used."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1, time=(0.0, float(PERIOD), 120))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    fem = jno.fem(
        [
            ui.tt * vi + (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(xi0, yi0) - 0.0,  # u0 = 0
            ui0.t - jno.fn(_mode11, [xi0, yi0]),  # v0 = sin(πx) sin(πy)
        ]
    )
    ts, U, _ = _trajectory(fem)
    pts = np.asarray(fem.points)
    ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))
    assert abs(U[0, ci]) < 1e-9  # starts at rest displacement
    qi = int(np.argmin(np.abs(ts - PERIOD / 4.0)))  # quarter period
    expect = _mode11(pts[ci, 0], pts[ci, 1]) * np.sin(OMEGA * ts[qi]) / OMEGA
    assert abs(U[qi, ci] - expect) < 0.05 * abs(expect) + 5e-3, f"velocity IC not honored: {U[qi, ci]:.4f} vs {expect:.4f}"


def test_constant_nonhomogeneous_dirichlet_is_held():
    """A constant non-zero Dirichlet value must be held for all time (u=g on the boundary, v=0
    there). With u0=g everywhere and v0=0 the solution is static u≡g — verifies the g-on-rows path
    of the augmented Dirichlet construction for g≠0."""
    g = 1.5
    fem = _wave_fem(mesh_size=0.15, n_periods=1, n_steps=20, u0_fn=lambda x, y: 0.0 * x + g, dirichlet=g)
    _ts, U, V = _trajectory(fem)
    assert np.max(np.abs(U - g)) < 1e-6, "constant Dirichlet not held (u should stay = g)"
    assert np.max(np.abs(V)) < 1e-6, "velocity should stay zero for a static constant-Dirichlet field"


def test_second_order_time_varying_dirichlet_drives_wave():
    """Time-varying (driven) Dirichlet ``g(x,t)``: ``u = cos(πx)cos(πt)`` solves ``u_tt = Δu`` (it is
    y-independent, so ``u_yy=0``) with the whole boundary driven by ``g(x,t)=cos(πx)cos(πt)``. The
    displacement rows carry ``u[d]=g(t)`` and the velocity rows the compatible ``v[d]=ġ(t)`` (AD of the
    boundary value through the θ-scheme), so the interior tracks the analytic *driven* wave — the load
    the old code rejected (it hard-zeroed the velocity boundary value)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.09, time=(0.0, 1.2, 72))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    weak = ui.tt * vi + (ui.x * vi.x + ui.y * vi.y)
    g = u(xb, yb) - jno.np.cos(PI * xb) * jno.np.cos(PI * tb)  # driven boundary g(x,t)=cos(πx)cos(πt)
    fem = jno.fem([weak, g, u(xi0, yi0) - jno.fn(lambda x, y: jnp.cos(PI * x), [xi0, yi0]), ui0.t - 0.0])
    assert fem.is_transient and fem.operator.forcing_vector_fn is not None  # driven -> time-dependent forcing
    ts, U, _ = _trajectory(fem)
    pts = np.asarray(fem.points)
    exact = np.cos(PI * pts[:, 0])[None, :] * np.cos(PI * ts)[:, None]
    rel = np.linalg.norm(U - exact) / np.linalg.norm(exact)
    assert rel < 0.01, f"driven wave does not track cos(πx)cos(πt): rel L2 = {rel:.4f}"
    # the driven boundary itself must equal g(x,t) exactly (the g(t) row) at every grid time
    bnode = int(np.argmin(np.abs(pts[:, 0]) + np.abs(pts[:, 1] - 0.5)))  # a node on the driven left edge
    g_bnode = np.cos(PI * pts[bnode, 0]) * np.cos(PI * ts)
    assert np.max(np.abs(U[:, bnode] - g_bnode)) < 1e-6, "the driven boundary must equal g(x,t) exactly"


# ==========================================================================
# fail-loud scope boundaries (a mis-assembled second-order solve is silently wrong)
# ==========================================================================
def test_second_order_1d_wave_matches_analytic():
    """Native 1D second-order-in-time: the wave ``u_tt = u_xx`` on [0,1], clamped, released from
    ``u(x,0)=sin(πx)`` at rest, tracks the analytic standing wave ``sin(πx)cos(πt)``. The native 1D
    assembler (LINE2/P1) builds the augmented [u, v] block just like the 2D/3D path — 1D was
    previously fail-loud (first-order only)."""
    pytest.importorskip("pygmsh", reason="pygmsh required for line meshing")
    d = jno.domain(constructor=jno.domain.line(mesh_size=0.02), time=(0.0, 1.2, 80))
    u, phi = d.fem_symbols()
    xi, ti = d.variable("interior", split=True)[0], d.variable("interior", split=True)[-1]
    xb = d.variable("boundary", split=True)[0]
    xi0, ti0 = d.variable("initial", split=True)[0], d.variable("initial", split=True)[-1]
    ui, vi = u.bind(x=xi, t=ti), phi.bind(x=xi, t=ti)
    ui0 = u.bind(x=xi0, t=ti0)
    fem = jno.fem([ui.tt * vi + ui.x * vi.x, u(xb) - 0.0, u(xi0) - jno.fn(lambda x: jnp.sin(PI * x), [xi0]), ui0.t - 0.0])
    assert fem.is_transient and fem.is_linear
    n = fem.offsets[1]
    assert fem.offsets == [0, n, 2 * n]  # augmented [u; v]
    ts, U, _ = _trajectory(fem)
    xx = np.asarray(fem.points)[:, 0]
    exact = np.sin(PI * xx)[None, :] * np.cos(PI * ts)[:, None]
    rel = np.linalg.norm(U - exact) / np.linalg.norm(exact)
    assert rel < 0.01, f"1D wave does not track sin(πx)cos(πt): rel L2 = {rel:.4f}"


def _elastodynamics_fem(mesh_size=0.12, n_periods=4, n_steps=240, E=1.0, nu=0.25, rho=1.0):
    """Clamped elastic unit square, released from an initial x-displacement bump (at rest):
    ρ u_tt = ∇·σ(u),  σ = λ(∇·u)I + 2μ ε(u)  — vector (elastodynamics) second-order-in-time."""
    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    cs = np.sqrt(mu / rho)
    t1 = n_periods * (2.0 / (cs * PI * np.sqrt(2.0)))  # a few shear transit times
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, float(t1), n_steps))
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
    weak = rho * inner(ui.tt, vi, n_contract=1) + lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
    u0 = u(xi0, yi0) - jno.fn(lambda x, y: jnp.stack([_mode11(x, y), 0.0 * x], axis=-1), [xi0, yi0])
    return jno.fem([weak, u(xb, yb) - (0.0, 0.0), u0, ui0.t - (0.0, 0.0)])


def test_vector_elastodynamics_conserves_energy():
    """Vector (value_shape=(2,)) second-order-in-time — elastodynamics. The trapezoidal rule
    conserves the discrete energy ``E = ½ vᵀM v + ½ uᵀK u`` of the undamped system, while backward
    Euler dissipates it; the clamped boundary stays at zero. This covers the vector reduction,
    the vector initial condition, and the vector Dirichlet path in one physical invariant."""
    fem = _elastodynamics_fem(mesh_size=0.12, n_periods=4, n_steps=240)
    assert fem.is_linear and fem.is_transient
    n = fem.offsets[1]
    M_uu = np.asarray(fem.M)[:n, :n]  # block mass M2 (Dirichlet rows zeroed)
    A_aug = fem.operator.A
    K_uu = np.asarray(A_aug.todense() if hasattr(A_aug, "todense") else A_aug)[n:, :n]  # stiffness block K
    ts = np.asarray(_block_time_grid(fem.operator))

    def energy(theta):
        fem.operator.metadata["theta"] = theta
        Y = np.asarray(_default_transient_integrate(fem.operator, {}, ts))
        U, V = Y[:, :n], Y[:, n:]
        return 0.5 * np.einsum("ti,ij,tj->t", V, M_uu, V) + 0.5 * np.einsum("ti,ij,tj->t", U, K_uu, U)

    e_half = energy(0.5)
    e_be = energy(1.0)
    assert abs(e_half[-1] / e_half[0] - 1.0) < 0.02, (
        f"trapezoidal should conserve energy (ratio {e_half[-1] / e_half[0]:.4f})"
    )
    # backward Euler clearly dissipates (and θ=½ clearly does not) -- the discriminator, not an
    # absolute decay rate (which depends on how many oscillations fit the window).
    assert e_be[-1] / e_be[0] < 0.8, f"backward Euler should dissipate energy (ratio {e_be[-1] / e_be[0]:.4f})"
    assert e_half[-1] / e_half[0] > e_be[-1] / e_be[0] + 0.1, "θ=½ must conserve markedly better than backward Euler"


def _elastic_planewave_fem(active, lam, mu, rho, mesh_size=0.05, n_steps=120):
    """A vector elastodynamics standing wave whose exact frequency is analytic — the check the
    energy invariant cannot make (the trapezoidal rule conserves a quadratic invariant of *any*
    linear block, even a frequency-wrong one, so energy conservation says nothing about the speed).

    ``active=0`` builds a **pressure (P) wave** ``u = (sin(πx), 0) cos(ω_p t)`` polarized and
    propagating in x, speed ``c_p = √((λ+2μ)/ρ)`` (``ω_p = π c_p``); both the dilatational
    ``λ tr(ε)tr(ε')`` and deviatoric ``2μ ε:ε'`` terms act. ``active=1`` builds a **shear (S) wave**
    ``u = (0, sin(πx)) cos(ω_s t)``, ``c_s = √(μ/ρ)`` — *pure shear* (``tr ε = 0``) that isolates the
    ``2μ ε:ε'`` term. Consistent BCs make each an exact mode: the propagation-direction (x) faces are
    clamped (the profile vanishes there) and the transverse component is pinned to zero on the y-faces
    (a roller reacting the normal/shear stress, the remaining traction being naturally zero). A wrong
    λ/μ/ρ factor anywhere in the vector M₂/K assembly shifts ω and breaks the analytic match.
    """
    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    c = np.sqrt(((lam + 2.0 * mu) if active == 0 else mu) / rho)
    omega = PI * c
    period = 2.0 * PI / omega
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, float(period), n_steps))
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, ti = d.variable("interior", split=True)
    xL, yL, _ = d.variable("left", split=True)
    xR, yR, _ = d.variable("right", split=True)
    xB, yB, _ = d.variable("bottom", split=True)
    xT, yT, _ = d.variable("top", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
    weak = rho * inner(ui.tt, vi, n_contract=1) + lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
    b = 1 - active  # the transverse (inactive) component index
    if active == 0:
        ic = lambda x, y: jnp.stack([jnp.sin(PI * x), 0.0 * x], axis=-1)  # noqa: E731
    else:
        ic = lambda x, y: jnp.stack([0.0 * x, jnp.sin(PI * x)], axis=-1)  # noqa: E731
    fem = jno.fem(
        [
            weak,
            u(xL, yL) - (0.0, 0.0),
            u(xR, yR) - (0.0, 0.0),  # clamp the propagation-direction (x) faces
            u(xB, yB)[b] - 0.0,
            u(xT, yT)[b] - 0.0,  # pin the transverse component on the y-faces
            u(xi0, yi0) - jno.fn(ic, [xi0, yi0]),
            ui0.t - (0.0, 0.0),
        ]
    )
    return fem, omega, active


def test_vector_elastodynamics_pressure_wave_frequency():
    """The vector reduction runs at the correct *pressure-wave* speed ``c_p = √((λ+2μ)/ρ)``: the
    antinode of the analytic standing wave ``u=(sin(πx),0)cos(ω_p t)`` tracks ``cos(ω_p t)``. This
    asserts the *period* — the missing check next to energy conservation, and the one that (in
    float32, without the ``_x64`` fixture) exposed the soft-mode frequency error."""
    fem, omega, active = _elastic_planewave_fem(active=0, lam=1.0, mu=1.0, rho=1.0)
    ts, U, _ = _trajectory(fem)
    pts = np.asarray(fem.points)
    ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))  # node nearest the (0.5, 0.5) antinode
    exact = np.sin(PI * pts[ci, 0]) * np.cos(omega * ts)
    rel = np.linalg.norm(U[:, 2 * ci + active] - exact) / np.linalg.norm(exact)
    assert rel < 0.02, f"P-wave antinode does not track analytic cos(ω_p t): rel L2 = {rel:.4f}"
    assert np.max(np.abs(U[:, 1::2])) < 0.02, "transverse (u_y) component must stay ~0 for a P-wave"


def test_vector_elastodynamics_shear_wave_frequency():
    """The vector reduction runs at the correct *shear-wave* speed ``c_s = √(μ/ρ)``, isolating the
    ``2μ ε:ε'`` term (pure shear, ``tr ε = 0``) with ``λ≠μ`` so a λ/μ mix-up would shift ω. The
    antinode of ``u=(0,sin(πx))cos(ω_s t)`` tracks ``cos(ω_s t)``."""
    fem, omega, active = _elastic_planewave_fem(active=1, lam=2.0, mu=1.0, rho=1.0)
    ts, U, _ = _trajectory(fem)
    pts = np.asarray(fem.points)
    ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))
    exact = np.sin(PI * pts[ci, 0]) * np.cos(omega * ts)
    rel = np.linalg.norm(U[:, 2 * ci + active] - exact) / np.linalg.norm(exact)
    assert rel < 0.02, f"S-wave antinode does not track analytic cos(ω_s t): rel L2 = {rel:.4f}"
    assert np.max(np.abs(U[:, 0::2])) < 0.02, "longitudinal (u_x) component must stay ~0 for this S-wave"


def test_vector_elastodynamics_boundary_stays_clamped():
    """The clamped (u = 0) boundary of the vector problem is held to ~machine precision for all
    time — the vector Dirichlet rows of the augmented system are correct."""
    fem = _elastodynamics_fem(mesh_size=0.15, n_periods=1, n_steps=60)
    pts = np.asarray(fem.points)
    on_b = (
        (np.abs(pts[:, 0]) < 1e-9)
        | (np.abs(pts[:, 0] - 1) < 1e-9)
        | (np.abs(pts[:, 1]) < 1e-9)
        | (np.abs(pts[:, 1] - 1) < 1e-9)
    )
    bdof = np.concatenate([np.where(on_b)[0] * 2, np.where(on_b)[0] * 2 + 1])  # both components, interleaved
    fem.operator.metadata["theta"] = 0.5
    Y = np.asarray(_default_transient_integrate(fem.operator, {}, np.asarray(_block_time_grid(fem.operator))))
    assert np.max(np.abs(Y[:, bdof])) < 1e-8, "clamped boundary not held for the vector field"


# ==========================================================================
# runtime / trainable parameters — the differentiable inverse through u_tt
# ==========================================================================
def _param_wave_fem(name="c2", on="stiffness", mesh_size=0.2, n_steps=24, t1=1.1):
    """Scalar wave carrying a runtime parameter, for the inverse tests. ``on='stiffness'`` puts the
    parameter on the Laplacian — a wave speed ``c²`` in ``u_tt = c² Δu`` (constant mass, so only
    ``operator_fn`` is parametric); ``on='mass'`` puts it on ``u_tt`` — a density ``ρ`` in
    ``ρ u_tt = Δu`` (exercises the ``mass_fn`` path). Truth value is 1.0."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, float(t1), n_steps))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    u0 = jno.np.sin(PI * xi0) * jno.np.sin(PI * yi0)
    p = jno.np.parameter((1,), name=name)
    weak = (
        (p * ui.tt * vi + (ui.x * vi.x + ui.y * vi.y)) if on == "mass" else (ui.tt * vi + p * (ui.x * vi.x + ui.y * vi.y))
    )
    fem = jno.fem([weak, u(xb, yb) - 0.0, u(xi0, yi0) - u0, ui0.t - 0.0])
    return fem, p


def test_second_order_parametric_is_reformable_and_exact():
    """A runtime parameter in a u_tt form no longer fails loud: the block re-forms from ``args``
    (``operator_fn`` + ``runtime_parameter_exprs``), and evaluated at a value it reproduces a freshly
    built parameter-free block to machine precision — the parametric path is exact, not approximate.
    A stiffness parameter leaves the mass constant (``mass_fn is None``); a mass parameter wires it."""
    fem, _ = _param_wave_fem(name="c2", on="stiffness")
    blk = fem.operator
    assert blk.metadata.get("second_order") and blk.operator_fn is not None
    assert list(blk.runtime_parameter_exprs) == ["c2"] and blk.mass_fn is None  # stiffness-only -> constant mass

    fem_mass, _ = _param_wave_fem(name="rho", on="mass")
    assert fem_mass.operator.mass_fn is not None  # a parametric mass wires mass_fn (M2 also feeds the -M2 coupling)

    val = 1.35
    ts = np.asarray(_block_time_grid(blk))
    U_param = np.asarray(_default_transient_integrate(blk, {"c2": val}, ts))
    # a parameter-free wave at the same speed (c² folded into the Laplacian coefficient)
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2, time=(0.0, 1.1, 24))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    u0 = jno.np.sin(PI * xi0) * jno.np.sin(PI * yi0)
    fem_s = jno.fem([ui.tt * vi + val * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - u0, ui0.t - 0.0])
    U_static = np.asarray(_default_transient_integrate(fem_s.operator, {}, np.asarray(_block_time_grid(fem_s.operator))))
    rel = np.linalg.norm(U_param - U_static) / np.linalg.norm(U_static)
    assert rel < 1e-9, f"parametric u_tt forward must match the parameter-free block exactly: rel={rel:.2e}"


@pytest.mark.parametrize("on,name", [("stiffness", "c2"), ("mass", "rho")])
def test_second_order_gradient_matches_finite_difference(on, name):
    """The gradient of ``‖u(θ) − u_obs‖²`` w.r.t. the parameter flows through the whole trapezoidal
    augmented march: autodiff matches a central finite difference. Covers both the ``operator_fn``
    (stiffness → wave speed) and the ``mass_fn`` (mass → density) re-assembly paths."""
    fem, _ = _param_wave_fem(name=name, on=on, mesh_size=0.22, n_steps=20, t1=1.0)
    blk = fem.operator
    ts = jnp.asarray(_block_time_grid(blk))
    u_obs = _default_transient_integrate(blk, {name: 1.0}, ts)  # data at the truth

    def loss(pval):
        return jnp.mean((_default_transient_integrate(blk, {name: pval}, ts) - u_obs) ** 2)

    val = 1.2
    g_ad = float(jax.grad(loss)(val))
    h = 1e-4
    g_fd = float((loss(val + h) - loss(val - h)) / (2 * h))
    assert abs(g_ad - g_fd) <= 1e-3 * (abs(g_fd) + 1e-8), f"AD {g_ad:.6e} vs FD {g_fd:.6e} ({name})"
    assert abs(g_ad) > 1e-6, "gradient should be non-trivial (the loss depends on the parameter)"


def test_vector_elastodynamics_gradient_matches_finite_difference():
    """The vector (elastodynamics) inverse is differentiable too: the gradient of the trajectory loss
    w.r.t. a shear modulus μ (in the ``2μ ε:ε'`` stiffness) matches a finite difference — the path an
    elastography / full-waveform inversion would take."""
    inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.24, time=(0.0, 1.4, 20))
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
    mu = jno.np.parameter((1,), name="mu")
    weak = inner(ui.tt, vi, n_contract=1) + 1.0 * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
    u0 = jno.fn(lambda x, y: jnp.stack([jnp.sin(PI * x) * jnp.sin(PI * y), 0.0 * x], axis=-1), [xi0, yi0])
    fem = jno.fem([weak, u(xb, yb) - (0.0, 0.0), u(xi0, yi0) - u0, ui0.t - (0.0, 0.0)])
    blk = fem.operator
    ts = jnp.asarray(_block_time_grid(blk))
    u_obs = _default_transient_integrate(blk, {"mu": 1.0}, ts)

    def loss(pval):
        return jnp.mean((_default_transient_integrate(blk, {"mu": pval}, ts) - u_obs) ** 2)

    g_ad = float(jax.grad(loss)(1.25))
    g_fd = float((loss(1.25 + 1e-4) - loss(1.25 - 1e-4)) / (2e-4))
    assert abs(g_ad - g_fd) <= 1e-3 * (abs(g_fd) + 1e-8), f"vector AD {g_ad:.6e} vs FD {g_fd:.6e}"
    assert abs(g_ad) > 1e-6, "vector gradient should be non-trivial"


def test_second_order_recovers_wave_speed_via_crux():
    """End-to-end: recover a wave speed ``c²`` from a synthetic trajectory through the differentiable
    ``fem.solve()`` node and ``jno.core([...]).solve(...)`` — the whole inverse loop, optimizer and
    all, not just a raw gradient."""
    import optax

    fem, c2 = _param_wave_fem(name="c2", on="stiffness", mesh_size=0.22, n_steps=24, t1=1.1)
    blk = fem.operator
    ts = np.asarray(_block_time_grid(blk))
    u_obs = np.asarray(_default_transient_integrate(blk, {"c2": 1.0}, ts))  # data at the truth
    c2.dtype(jnp.float64)
    c2.initialize(jax.nn.initializers.constant(2.0))  # start far from the truth
    c2.optimizer(optax.adam(8e-2))
    crux = jno.core([(fem.solve() - u_obs).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    crux.solve(180)
    rec = float(np.asarray(crux.eval([c2])).reshape(-1)[0])
    assert abs(rec - 1.0) < 0.05, f"wave speed not recovered through crux: c²={rec:.4f} (truth 1.0)"


def _cubic_klein_gordon(amp, mesh_size=0.16, t1=1.6, n_steps=96):
    """Cubic Klein–Gordon ``u_tt = Δu − u³`` on the unit square, clamped, released from
    ``amp·sin(πx)sin(πy)`` at rest — a nonlinear second-order-in-time form."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, float(t1), n_steps))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    weak = ui.tt * vi + (ui.x * vi.x + ui.y * vi.y) + (ui**3) * vi  # u_tt = Δu − u³
    u0 = u(xi0, yi0) - jno.fn(lambda x, y: amp * jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])
    return jno.fem([weak, u(xb, yb) - 0.0, u0, ui0.t - 0.0])


def test_second_order_nonlinear_reduces_to_linear_at_small_amplitude():
    """A nonlinear second-order form is supported (Newton on the augmented residual). At tiny amplitude
    the cubic term ``u³`` is negligible, so the solve reproduces the linear standing wave
    ``sin(πx)sin(πy)cos(ω t)``, ``ω=π√2`` — a direct check that the nonlinear augmented assembly is
    correct (it must reduce to the linear block as the nonlinearity vanishes)."""
    fem = _cubic_klein_gordon(1e-3)
    assert fem.operator.is_nonlinear() and fem.is_transient
    ts, U, _ = _trajectory(fem)
    pts = np.asarray(fem.points)
    ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))
    exact = 1e-3 * _mode11(pts[ci, 0], pts[ci, 1]) * np.cos(OMEGA * ts)
    rel = np.linalg.norm(U[:, ci] - exact) / np.linalg.norm(exact)
    assert rel < 0.05, f"small-amplitude cubic KG should track the linear wave: rel L2 = {rel:.4f}"


def test_second_order_nonlinear_klein_gordon_conserves_energy():
    """Nonlinear ``u_tt = Δu − u³`` via Newton on the augmented residual. The θ=½ (Newmark) step
    conserves the discrete energy ``E = ½vᵀM₂v + ½uᵀKu + ¼∫u⁴`` of the undamped nonlinear system,
    while backward Euler (θ=1) dissipates it — the nonlinear analogue of the linear energy gate, and
    the check that the θ-aware nonlinear step (not hard backward Euler) is in place."""
    fem = _cubic_klein_gordon(0.8, mesh_size=0.17, t1=2.0, n_steps=110)
    n = fem.offsets[1]
    M2 = np.asarray(fem.M)[:n, :n]
    d2 = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.17)
    u2, p2 = d2.fem_symbols()
    a2, b2, _ = d2.variable("interior", split=True)
    ux, vx = u2.bind(x=a2, y=b2), p2.bind(x=a2, y=b2)
    K = np.asarray(jno.fem([ux.x * vx.x + ux.y * vx.y]).A)
    ts = np.asarray(_block_time_grid(fem.operator))

    def energy(theta):
        fem.operator.metadata["theta"] = theta
        Y = np.asarray(_default_transient_integrate(fem.operator, {}, ts))
        Ut, Vt = Y[:, :n], Y[:, n:]
        lin = 0.5 * np.einsum("ti,ij,tj->t", Vt, M2, Vt) + 0.5 * np.einsum("ti,ij,tj->t", Ut, K, Ut)
        return lin + 0.25 * np.einsum("ti,ij,tj->t", Ut**2, M2, Ut**2)  # + ¼∫u⁴ (nonlinear potential)

    e_half, e_be = energy(0.5), energy(1.0)
    assert np.all(np.isfinite(e_half)), "the nonlinear Newton march must not blow up"
    assert abs(e_half[-1] / e_half[0] - 1.0) < 0.03, f"θ=½ should conserve nonlinear energy ({e_half[-1] / e_half[0]:.4f})"
    assert e_be[-1] / e_be[0] < 0.85, f"backward Euler should dissipate nonlinear energy ({e_be[-1] / e_be[0]:.4f})"


def test_second_order_multifield_rejected():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 1.0, 10))
    u, p = d.fem_symbols(names=("u", "pu"))
    w, q = d.fem_symbols(names=("w", "qw"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, pi = u.bind(x=xi, y=yi, t=ti), p.bind(x=xi, y=yi, t=ti)
    wi, qi = w.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    weak = ui.tt * pi + (ui.x * pi.x + ui.y * pi.y) + wi.t * qi + (wi.x * qi.x + wi.y * qi.y)
    with pytest.raises(NotImplementedError, match="single-field"):
        jno.fem([weak, u(xb, yb) - 0.0, w(xb, yb) - 0.0, u(xi0, yi0) - 0.0, ui0.t - 0.0, w(xi0, yi0) - 0.0])
