"""State-dependent (nonlinear) transient mass:  ``c(u)·u_t = κΔu + f``.

The transient mass term may carry a coefficient that depends on the *unknown* — e.g. an
apparent heat capacity ``c(u)=1+u``, or a phase-dependent density ``ρ(φ)`` on ``∂_t u`` in a
coupled system.  The semidiscrete residual is then ``M(u) u̇ + K u = f`` with a genuinely
state-dependent mass, and its exact step Jacobian gains the ``∂M/∂u`` coupling term.

Oracle: the **method of manufactured solutions**.  On the unit square with homogeneous Dirichlet,

    u*(x, y, t) = g(t)·φ(x, y),   g(t) = 1 + A t,   φ = sin(πx) sin(πy)   (≡ 0 on ∂Ω)

is fed into ``c(u) u_t = κΔu + f`` with ``c(u)=1+u`` by supplying the required forcing
``f = c(u*) u*_t − κΔu*`` (hand-derived below).  If the mass coefficient is honored at the
current state the solve reproduces u*; if it is frozen at ``u=0`` (c≡1, plain heat) the
trajectory is wrong by an O(1) amount — c(u*) ranges over [1, 2.2] here.

Regression pin: on the pre-feature assembler this FAILS (the mass is assembled once at zeros,
so ``c(u)·u_t`` silently reduces to ``u_t``); see jno-fem-hard-limits #1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
A, KAPPA, T_END, NT = 0.5, 0.05, 0.4, 41  # capacity slope, diffusivity, horizon, steps (dt=0.01)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _phi(x, y):
    return jnp.sin(PI * x) * jnp.sin(PI * y)


def _u_star(x, y, t):
    return (1.0 + A * t) * _phi(x, y)


def _f_src(x, y, t):
    # c(u*) u*_t − κΔu*,  with c(u)=1+u, Δφ = −2π²φ, u*_t = A·φ, u* = g·φ, g = 1+At
    phi = _phi(x, y)
    g = 1.0 + A * t
    return (1.0 + g * phi) * (A * phi) + 2.0 * PI**2 * KAPPA * g * phi


def _build():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05, time=(0.0, T_END, NT))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    # c(u)=1+u on the time term; the whole point is that c is read at the current state.
    weak = (1.0 + ui) * ui.t * vi + KAPPA * (ui.x * vi.x + ui.y * vi.y) - jno.fn(_f_src, [xi, yi, ti]) * vi
    ic = jno.fn(lambda x, y: _phi(x, y), [xi0, yi0])
    fem = jno.fem([weak, u(xb, yb) - 0.0, u(xi0, yi0) - ic])
    return d, fem


def _rel_err(d, sol):
    pts = np.asarray(d.mesh.points)[:, :2]
    exact = np.asarray(jax.vmap(lambda p: _u_star(p[0], p[1], T_END))(jnp.asarray(pts)))
    got = np.asarray(sol)
    got = got[-1] if got.ndim == 2 else got  # transient .fn() is (n_times, n_nodes); take t=T_END
    return np.linalg.norm(got.reshape(-1) - exact) / max(np.linalg.norm(exact), 1e-12)


def test_nonlinear_mass_matches_manufactured_solution():
    """The state-dependent capacity c(u)=1+u is honored: default solve reproduces u*."""
    d, fem = _build()
    err = _rel_err(d, fem.solve().fn())
    assert err < 0.05, f"nonlinear-mass trajectory did not reproduce u*: rel L2 = {err:.3f}"


def test_nonlinear_mass_direct_newton_matches_default():
    """The sparse-direct step Newton (exact assembled ∂M/∂u tangent) matches the matrix-free path."""
    d, fem = _build()
    err = _rel_err(d, fem.solve(nonlinear=jno.solve.newton(direct=True)).fn())
    assert err < 0.05, f"direct-Newton nonlinear-mass trajectory wrong: rel L2 = {err:.3f}"


def test_nonlinear_mass_theta_half_raises():
    """A state-dependent mass is backward-Euler only; Crank–Nicolson must fail loud, not silently wrong."""
    _, fem = _build()
    with pytest.raises((ValueError, Exception), match="backward Euler|theta"):
        fem.solve(time=jno.solve.theta(0.5)).fn()


# --- coupled cross-coefficient: mass of field `a` depends on another field `b` (the ∂M/∂b coupling that a
#     phase-field ρ(φ)·u_t needs). One-way manufactured pair a*(1+At)φ, b*(1+Bt)φ, φ=sin πx sin πy. ---
B = 0.7


def _build_coupled():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05, time=(0.0, T_END, NT))
    a, pa = d.fem_symbols(names=("a", "pa"))
    b, pb = d.fem_symbols(names=("b", "pb"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ai, pai = a.bind(x=xi, y=yi, t=ti), pa.bind(x=xi, y=yi, t=ti)
    bi, pbi = b.bind(x=xi, y=yi, t=ti), pb.bind(x=xi, y=yi, t=ti)

    def f_a(x, y, t):
        phi = _phi(x, y)
        return (1.0 + (1.0 + B * t) * phi) * (A * phi) + 2.0 * PI**2 * KAPPA * (1.0 + A * t) * phi

    def f_b(x, y, t):
        phi = _phi(x, y)
        return B * phi + 2.0 * PI**2 * KAPPA * (1.0 + B * t) * phi

    # mass of `a` carries the state-dependent coefficient (1 + b); `b`'s mass is plain.
    a_eq = (1.0 + bi) * ai.t * pai + KAPPA * (ai.x * pai.x + ai.y * pai.y) - jno.fn(f_a, [xi, yi, ti]) * pai
    b_eq = bi.t * pbi + KAPPA * (bi.x * pbi.x + bi.y * pbi.y) - jno.fn(f_b, [xi, yi, ti]) * pbi
    ic = jno.fn(lambda x, y: _phi(x, y), [xi0, yi0])
    fem = jno.fem([a_eq, b_eq, a(xb, yb) - 0.0, b(xb, yb) - 0.0, a(xi0, yi0) - ic, b(xi0, yi0) - ic])
    return d, fem


def test_nonlinear_mass_cross_field_coupling():
    """Mass coefficient of `a` depends on field `b`: the exact ∂M/∂b coupling reproduces both fields."""
    d, fem = _build_coupled()
    traj = np.asarray(fem.solve().fn())  # (n_times, 2*n_nodes) — fields concatenated
    pts = np.asarray(d.mesh.points)[:, :2]
    n = pts.shape[0]
    a_exact = np.asarray(jax.vmap(lambda p: _u_star(p[0], p[1], T_END))(jnp.asarray(pts)))
    b_exact = np.asarray(jax.vmap(lambda p: (1.0 + B * T_END) * _phi(p[0], p[1]))(jnp.asarray(pts)))
    final = traj[-1].reshape(-1)
    a_got, b_got = final[:n], final[n : 2 * n]
    ea = np.linalg.norm(a_got - a_exact) / np.linalg.norm(a_exact)
    eb = np.linalg.norm(b_got - b_exact) / np.linalg.norm(b_exact)
    # 8% = the backward-Euler + P1 discretization floor at this dt/mesh (matches the moving-boundary MMS
    # threshold): note the LINEAR-mass field `b` sits at the same ~6%, so the state-dependent mass of `a`
    # (and its ∂M/∂b coupling) is exact — a wrong coupling would make `a` diverge from `b`, not track it.
    assert ea < 0.08 and eb < 0.08, f"cross-coupled nonlinear mass wrong: rel L2 a={ea:.3f}, b={eb:.3f}"


def test_nonlinear_mass_vector_field():
    """State-dependent mass on a VECTOR field (CHNS velocity shape): c(w)=1+w₀ on the vector u_t term.
    Manufactured w* = (1+At)(φ, φ); both components must be reproduced. Exercises the (n_nodes, vec)
    previous-state delivery and the field's own vector basis (no P1 aliasing)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06, time=(0.0, T_END, NT))
    w, v = d.fem_symbols(value_shape=(2,), names=("w", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    wi, vi = w.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)

    def f_src(x, y, t):  # same forcing for both components (c(w*)=1+(1+At)φ, w*_i,t = Aφ)
        phi = _phi(x, y)
        return (1.0 + (1.0 + A * t) * phi) * (A * phi) + 2.0 * PI**2 * KAPPA * (1.0 + A * t) * phi

    f = jno.fn(f_src, [xi, yi, ti])
    mass = (1.0 + wi[0]) * (wi.t[0] * vi[0] + wi.t[1] * vi[1])  # coefficient depends on component 0
    diff = KAPPA * (wi.x[0] * vi.x[0] + wi.y[0] * vi.y[0] + wi.x[1] * vi.x[1] + wi.y[1] * vi.y[1])
    weak = mass + diff - f * (vi[0] + vi[1])
    ic = jno.fn(lambda x, y: _phi(x, y), [xi0, yi0])
    fem = jno.fem([weak, w(xb, yb) - 0.0, w(xi0, yi0)[0] - ic, w(xi0, yi0)[1] - ic])

    final = np.asarray(fem.solve().fn())[-1].reshape(-1)  # node-major interleaved (node·2 + comp)
    pts = np.asarray(d.mesh.points)[:, :2]
    exact = np.asarray(jax.vmap(lambda p: _u_star(p[0], p[1], T_END))(jnp.asarray(pts)))
    e0 = np.linalg.norm(final[0::2] - exact) / np.linalg.norm(exact)
    e1 = np.linalg.norm(final[1::2] - exact) / np.linalg.norm(exact)
    assert e0 < 0.08 and e1 < 0.08, f"vector nonlinear-mass wrong: rel L2 comp0={e0:.3f}, comp1={e1:.3f}"


def _heat_nlmass(coeff, mesh_size=0.25, time=(0.0, 0.2, 11)):
    """Transient heat whose u_t coefficient is 1 + coeff·u (state-dependent mass), sine-bump IC."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=time)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    weak = (1.0 + coeff * ui) * ui.t * vi + KAPPA * (ui.x * vi.x + ui.y * vi.y)
    return jno.fem([weak, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def test_nonlinear_mass_inverse_adjoint():
    """Reverse-mode adjoint flows through the state-dependent mass: recover a coefficient parameter
    (α in c(u)=1+α·u on the u_t term) from an observed trajectory. Guards jNO differentiability (rule #3):
    the per-step Newton's ``custom_root`` and the mass reassembly must be transposable end-to-end."""
    import optax

    u_obs = jnp.asarray(_heat_nlmass(1.0).solve().fn())  # α_true = 1

    alpha = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="alpha_mass")
    alpha.initialize(jax.nn.initializers.constant(2.5))
    alpha.dtype(jnp.float64)
    alpha.optimizer(optax.adam(5e-2))

    node = _heat_nlmass(alpha).solve()
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    crux = jno.core([(node - u_obs).mse], domain=dummy)
    crux.solve(150)
    a = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
    assert abs(a - 1.0) < 0.1, f"mass-coefficient α not recovered through the nonlinear-mass adjoint: {a}"
