"""09 - Vibrating membrane: the 2-D wave equation (second-order in time) via ``jno.fem``.

A square drum head clamped on all four edges, plucked into the fundamental mode and released:

    u_tt = c^2 Δu ,    u = 0 on ∂Ω ,    u(t=0) = sin(πx) sin(πy) ,   u_t(t=0) = 0 .

The exact solution is the standing wave  u(x, y, t) = sin(πx) sin(πy) cos(ω t)  with the modal
frequency  ω = c π √2  (so  -Δ(sin πx sin πy) = 2π² · sin πx sin πy).  This is a **second-order**
weak form -- the unknown carries a *second* time derivative ``ui.tt`` -- which ``jno.fem`` auto-reduces
to the first-order system in y = [u, v=u_t] and exposes as the usual transient block ``fem.M`` /
``fem.operator.A`` / ``fem.state0``.

Time integration uses the **trapezoidal rule** (θ=½, the energy-conserving member of the Newmark
average-acceleration family -- Newmark 1959, *J. Eng. Mech. Div. ASCE* 85(3)). For a *second-order*
block this matters: backward Euler would spuriously damp an undamped membrane, so we step with θ=½
rather than the backward-Euler pattern used for parabolic (first-order) problems.

Verification: the centre-node displacement tracks the analytic cos(ω t) over a full period, and the
discrete energy E = ½ vᵀM v + ½ uᵀK u is conserved (a drum does not lose energy on its own).
"""

import jax.numpy as jnp
import numpy as np
from shapely.geometry import box

import jno

dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

PI = np.pi
C = 1.0  # wave speed
OMEGA = C * PI * np.sqrt(2.0)  # fundamental modal frequency
PERIOD = 2.0 * PI / OMEGA  # = √2 / C

# One full period, resolved with 120 steps; a moderate mesh keeps the example quick.
d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08, time=(0.0, float(PERIOD), 120))
u, phi = d.fem_symbols()
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
xi0, yi0, ti0 = d.variable("initial", split=True)  # the t=0 slice carries its coords AND time ti0
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
ui0 = u.bind(x=xi0, y=yi0, t=ti0)

# Weak form of u_tt = c² Δu :  ∫ u_tt φ + c² ∫ ∇u·∇φ = 0 .
weak = ui.tt * vi + C**2 * (ui.x * vi.x + ui.y * vi.y)
u0 = u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [xi0, yi0])  # plucked shape
v0 = ui0.t - 0.0  # released from rest (note: velocity IC binds the initial-slice time ti0)
fem = jno.fem([weak, u(xb, yb) - 0.0, u0, v0])
assert fem.is_transient and fem.is_linear

# --- march in time with the trapezoidal (θ=½) rule on the augmented block M ẏ + A y = c ---
#   (M + ½dt A) y_next = (M − ½dt A) y + dt c        [θ=½: energy-conserving]
# Do NOT use backward Euler (M + dt A) here -- it damps the wave (see the module docstring).
M, A = dense(fem.M), dense(fem.operator.A)
c_vec = np.zeros(M.shape[0]) if fem.operator.affine_bias is None else np.asarray(fem.operator.affine_bias).reshape(-1)
dt = float(fem.dt)
N = fem.offsets[1]  # state is y = [u; v]; displacement is the first N entries
lhs = M + 0.5 * dt * A
rhs_op = M - 0.5 * dt * A

y = np.asarray(fem.state0)
traj = [y[:N].copy()]
for _ in range(round((fem.t1 - fem.t0) / dt)):
    y = np.linalg.solve(lhs, rhs_op @ y + dt * c_vec)
    traj.append(y[:N].copy())
traj = np.asarray(traj)  # (n_steps+1, N) displacement history
ts = np.linspace(fem.t0, fem.t1, traj.shape[0])

# --- verify against the analytic standing wave + energy conservation ---
pts = np.asarray(fem.points)
ci = int(np.argmin(np.sum((pts - 0.5) ** 2, axis=1)))  # node nearest the centre antinode
u_center = traj[:, ci]
u_exact = np.sin(PI * pts[ci, 0]) * np.sin(PI * pts[ci, 1]) * np.cos(OMEGA * ts)
rel = np.linalg.norm(u_center - u_exact) / np.linalg.norm(u_exact)

M_uu, K_uu = M[:N, :N], A[N:, :N]  # mass and stiffness blocks of the augmented system
V = np.gradient(traj, ts, axis=0)  # velocity ~ d/dt of the displacement history
energy = 0.5 * np.einsum("ti,ij,tj->t", V, M_uu, V) + 0.5 * np.einsum("ti,ij,tj->t", traj, K_uu, traj)
amp = np.linalg.norm(traj[-1]) / np.linalg.norm(traj[0])

print(f"\nVibrating membrane (2-D wave, second-order in time): dofs={fem.dofs} (= 2N, N={N})")
print(f"  modal frequency ω = c·π·√2 = {OMEGA:.4f}   period T = {PERIOD:.4f}")
print(f"  centre-node vs analytic cos(ω t) over one period:  rel L2 = {rel:.4f}")
print(f"  amplitude after one period ||u(T)|| / ||u(0)|| = {amp:.4f}   (≈ 1: energy-conserving)")

assert rel < 0.05, f"membrane does not track the analytic standing wave: rel L2 = {rel:.4f}"
assert 0.95 < amp < 1.05, f"amplitude not conserved over a period: {amp:.4f}"  # θ=½, not backward Euler
assert abs(energy[len(energy) // 2] / energy[1] - 1.0) < 0.05, "discrete energy should be conserved"
