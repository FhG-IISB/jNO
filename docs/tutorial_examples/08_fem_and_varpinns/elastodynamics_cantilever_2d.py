"""10 - Elastodynamics + modal analysis: a ringing cantilever (vector, 2nd-order time) via ``jno.fem``.

The transient sibling of the static cantilever (tutorial 06). A plane-stress elastic beam clamped at
the root obeys Newton's second law for a continuum,

    ρ u_tt = ∇·σ(u) ,    σ(u) = λ (∇·u) I + 2μ ε(u) ,    ε(u) = ½(∇u + ∇uᵀ) ,

with the **vector** displacement u = (u_x, u_y) carrying a *second* time derivative ``ui.tt`` --
``jno.fem`` auto-reduces this to the first-order system in y = [u, v=u_t] and integrates it with the
energy-conserving trapezoidal rule (θ=½; backward Euler would bleed the vibration away).

To prove the *dynamics* (not just that the integrator is conservative -- the trapezoidal rule
conserves a quadratic invariant of any linear block, even a wrong one) we do a small **modal
analysis**: the assembled mass/stiffness give a generalized eigenproblem  K φ = ω² M φ  whose lowest
mode (ω₁, φ₁) is the fundamental bending shape. Released from that mode at rest, the exact solution is
u(t) = φ₁ cos(ω₁ t), so the tip traces a clean cosine -- a direct check that the augmented [u, v]
block reproduces M ü + K u = 0 at the right frequency. As a bonus the modal frequency matches
Euler-Bernoulli beam theory  ω₁ ≈ (1.875/L)² √(E I / ρ A)  (Timoshenko & Goodier, *Theory of
Elasticity*).
"""

import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
from shapely.geometry import box

import jno

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

E, nu, rho = 1000.0, 0.3, 1.0
lam, mu = E * nu / (1.0 - nu**2), E / (2.0 * (1.0 + nu))  # plane-stress Lamé parameters
L, H = 8.0, 1.0  # slender 8:1 beam (Euler-Bernoulli is a good reference)
Iz, Acs = H**3 / 12.0, H  # second moment / cross-section area (unit thickness)
omega_eb = (1.875 / L) ** 2 * np.sqrt(E * Iz / (rho * Acs))  # EB fundamental frequency (sets the window)
T_eb = 2.0 * np.pi / omega_eb

# integrate over two fundamental periods so the beam visibly rings (not a quarter swing)
d = jno.domain(box(0.0, 0.0, L, H), mesh_size=0.5, time=(0.0, float(2.0 * T_eb), 200))
u, phi = d.fem_symbols(value_shape=(2,), order=2)  # P2 vector displacement (TRI3 is too stiff in bending)
xi, yi, ti = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)  # clamped root
xi0, yi0, ti0 = d.variable("initial", split=True)
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
ui0 = u.bind(x=xi0, y=yi0, t=ti0)

eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = rho * inner(ui.tt, vi, n_contract=1) + lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
# Assemble the elastodynamics operators; the initial state is set from the eigenmode below, so the
# trivial (at-rest) IC here is just a placeholder.
fem = jno.fem([weak, u(xl, yl) - (0.0, 0.0), u(xi0, yi0) - (0.0, 0.0), ui0.t - (0.0, 0.0)])
assert fem.is_transient and fem.is_linear and fem.offsets == [0, fem.dofs // 2, fem.dofs]

M, A = np.asarray(dense(fem.M)), np.asarray(dense(fem.operator.A))
N = fem.offsets[1]  # state y = [u; v]; displacement = first N (node-major interleaved [n0x, n0y, ...])
M_uu, K_uu = M[:N, :N], A[N:, :N]  # mass and stiffness blocks of the augmented system

# --- modal analysis: fundamental vibration mode of the assembled operators (free dofs only) ---
pts = np.asarray(fem.points)
root = pts[:, 0] < 1e-9
clamped = np.sort(np.concatenate([np.where(root)[0] * 2, np.where(root)[0] * 2 + 1]))
free = np.setdiff1d(np.arange(N), clamped)
evals, evecs = sla.eigh(K_uu[np.ix_(free, free)], M_uu[np.ix_(free, free)])  # K φ = ω² M φ
omega1 = float(np.sqrt(evals[0]))
phi1 = np.zeros(N)
phi1[free] = evecs[:, 0]
phi1 /= np.max(np.abs(phi1))  # unit max displacement (a mode's amplitude is arbitrary)

# --- release from the fundamental mode at rest and march with the trapezoidal (θ=½) rule ---
dt = float(fem.dt)
lhs, rhs_op = M + 0.5 * dt * A, M - 0.5 * dt * A  # (M+½dtA) y⁺ = (M−½dtA) y   [c = 0]
y = np.concatenate([phi1, np.zeros(N)])  # u(0) = φ₁, v(0) = 0
disp, vel = [y[:N].copy()], [y[N:].copy()]
for _ in range(round((fem.t1 - fem.t0) / dt)):
    y = np.linalg.solve(lhs, rhs_op @ y)
    disp.append(y[:N].copy())
    vel.append(y[N:].copy())
disp, vel = np.asarray(disp), np.asarray(vel)
ts = np.linspace(fem.t0, fem.t1, disp.shape[0])

# --- verify: the tip rings as the exact modal cosine, energy is conserved, the root stays clamped ---
tip = int(np.argmax(pts[:, 0]))
tip_y = disp[:, 2 * tip + 1]
analytic = tip_y[0] * np.cos(omega1 * ts)  # exact modal solution u(t) = φ₁ cos(ω₁ t)
rel = np.linalg.norm(tip_y - analytic) / np.linalg.norm(analytic)
energy = 0.5 * np.einsum("ti,ij,tj->t", vel, M_uu, vel) + 0.5 * np.einsum("ti,ij,tj->t", disp, K_uu, disp)
drift = float(np.max(np.abs(energy / energy[0] - 1.0)))
clamp = float(np.max(np.abs(disp[:, clamped])))

print(f"\nRinging cantilever (elastodynamics + modal analysis): dofs={fem.dofs}  P2 vector")
print(f"  fundamental ω₁(FEM) = {omega1:.4f}   Euler-Bernoulli ω₁ = {omega_eb:.4f}   ratio = {omega1 / omega_eb:.3f}")
print(f"  tip vs analytic φ₁·cos(ω₁ t) over 2 periods:  rel L2 = {rel:.4f}")
print(f"  energy drift = {drift:.4f}   clamped-root max |u| = {clamp:.2e}")

assert rel < 0.05, f"tip does not ring as the modal cosine: rel L2 = {rel:.4f}"  # correct dynamics + frequency
assert drift < 0.02, f"undamped elastodynamics must conserve energy (drift {drift:.4f})"
assert clamp < 1e-8, "clamped root must stay fixed"
assert abs(omega1 - omega_eb) / omega_eb < 0.15, "FEM fundamental frequency should match beam theory"
