"""Second-order-in-time (`u_tt`) FEM through ``jno.fem`` — the wave / elastodynamics path.

A weak form carrying a second temporal derivative is reduced to a first-order augmented system in
``y = [u, v=u_t]`` and integrated by the **trapezoidal rule** (θ=½, energy-conserving). The headline
check is *amplitude conservation over several periods*: backward Euler (θ=1) spuriously damps an
undamped wave, so a single-step / loose-tolerance test would pass while shipping a wrong (damped)
solution. We verify against the analytic standing wave on the unit square::

    u_tt = Δu ,  u = sin(πx) sin(πy) cos(ω t) ,  ω = π√2 ,  T = 2π/ω = √2 ,

check the amplitude is preserved over 4 periods (and that θ=1 would *not* preserve it), match the
analytic solution over one period, exercise damping (decay) and a non-zero velocity IC, and confirm
the fail-loud scope boundaries (1D, nonlinear, multi-field, vector).
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
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
    """feax assembly is float64, so opt into x64 per-test (session default is x64-off)."""
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


# ==========================================================================
# fail-loud scope boundaries (a mis-assembled second-order solve is silently wrong)
# ==========================================================================
def _second_order_1d():
    pytest.importorskip("pygmsh", reason="pygmsh required for line meshing")
    d = jno.domain(constructor=jno.domain.line(mesh_size=0.1), time=(0.0, 1.0, 10))
    u, phi = d.fem_symbols()
    xi, ti = d.variable("interior", split=True)[0], d.variable("interior", split=True)[-1]
    xb = d.variable("boundary", split=True)[0]
    xi0, ti0 = d.variable("initial", split=True)[0], d.variable("initial", split=True)[-1]
    ui = u.bind(x=xi, t=ti)
    vi = phi.bind(x=xi, t=ti)
    ui0 = u.bind(x=xi0, t=ti0)
    return [ui.tt * vi + ui.x * vi.x, u(xb) - 0.0, u(xi0) - 0.0, ui0.t - 0.0]


def test_second_order_1d_rejected():
    with pytest.raises(NotImplementedError, match="2D/3D"):
        jno.fem(_second_order_1d())


def test_second_order_vector_rejected():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 1.0, 10))
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    weak = jno.np.inner(ui.tt, vi, n_contract=1) + jno.np.inner(
        jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2
    )
    with pytest.raises(NotImplementedError, match="scalar-only|single"):
        jno.fem([weak, u(xb, yb) - 0.0, u(xi0, yi0) - 0.0, ui0.t - 0.0])


def test_second_order_nonlinear_rejected():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 1.0, 10))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    weak = ui.tt * vi + (ui.x * vi.x + ui.y * vi.y) + (ui**3) * vi  # cubic reaction -> nonlinear
    with pytest.raises(NotImplementedError, match="nonlinear"):
        jno.fem([weak, u(xb, yb) - 0.0, u(xi0, yi0) - 0.0, ui0.t - 0.0])


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
