"""08 - Operator-splitting Allen-Cahn via ``block.step`` (the one-step primitive).

    u_t = D Lap u + r (u - u^3)      on the unit square,  zero-flux (natural) boundaries.

The Allen-Cahn equation phase-separates a noisy field into +/-1 domains. It splits cleanly into a
LINEAR diffusion part (global, a FEM solve) and a LOCAL nonlinear reaction ``r(u - u^3)`` (pointwise,
a per-node ODE). Strang splitting advances one step as

    react(dt/2)  ->  diffuse(dt)  ->  react(dt/2)

where the diffusion advance is jNO's one-step primitive ``block.step(u, t, dt)`` on a *linear*
transient block -- no hand-rolled Crank-Nicolson or CG. The reaction is a vmapped midpoint step.

Two checks: (1) ``block.step`` reproduces the textbook backward-Euler advance exactly (the primitive
is correct), and (2) the split solve separates the field into +/-1 phases (it works end to end).
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jno
import jno.jnp_ops as jnn

dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

D, R, N = 1e-3, 4.0, 48
T_END, N_STEPS = 1.5, 150
DT = T_END / N_STEPS

dom = jno.domain(
    constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=N, ny=N),
    time=(0.0, T_END, N_STEPS + 1),
)
u, phi = dom.fem_symbols()
xi, yi, ti = dom.variable("interior", split=True)
ci = dom.variable("initial", split=True)
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)

# deterministic low-mode initial condition (in the unstable band -> seeds phase separation)
ic = 0.15 * jnn.sin(2 * np.pi * ci[0]) * jnn.cos(2 * np.pi * 2 * ci[1]) + 0.15 * jnn.cos(2 * np.pi * 2 * ci[0]) * jnn.sin(
    2 * np.pi * ci[1]
)
fem = jno.fem([ui.t * vi + D * (ui.x * vi.x + ui.y * vi.y), u(ci[0], ci[1]) - ic])  # linear diffusion block
block = fem.operator

# --- (1) block.step reproduces the textbook backward-Euler diffusion step exactly ---
M, A = dense(block.M), dense(block.A)
u0 = jnp.asarray(block.state0)
be_direct = jnp.linalg.solve(M + DT * A, M @ u0)  # (M + dt A) u1 = M u0
be_blockstep = block.step(u0, 0.0, DT, theta=1.0)
step_err = float(jnp.linalg.norm(be_blockstep - be_direct) / jnp.linalg.norm(be_direct))
print(f"\nOperator-splitting Allen-Cahn via block.step  (N={N})")
print(f"  block.step vs direct backward-Euler:  rel error = {step_err:.2e}")


# --- (2) Strang split: react(dt/2) -> block.step(dt) -> react(dt/2) ---
def reaction(w):  # du/dt = r (u - u^3), pointwise
    return R * (w - w**3)


def react_half(w):  # midpoint (2nd order) over dt/2
    hh = 0.5 * DT
    return w + hh * reaction(w + 0.5 * hh * reaction(w))


u_split = jnp.asarray(block.state0)
t = float(block.t0)
for _ in range(N_STEPS):
    u_split = react_half(u_split)
    u_split = block.step(u_split, t, DT)  # jNO advances the diffusion
    u_split = react_half(u_split)
    t += DT
u_split = np.asarray(u_split)

sep = float(np.mean(np.abs(u_split)))
print(f"  phase separation  mean|u| = {sep:.3f}  (range [{u_split.min():+.2f}, {u_split.max():+.2f}])")

# --- figure ---
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(fem.points)
    ix, iy = np.round(pts[:, 0] * N).astype(int), np.round(pts[:, 1] * N).astype(int)
    g = np.full((N + 1, N + 1), np.nan)
    g[iy, ix] = u_split
    figm, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1, extent=[0, 1, 0, 1])
    ax.set(title="Allen-Cahn $u$ via Strang splitting (block.step)", xlabel="x", ylabel="y")
    figm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = Path(__file__).parents[2] / "assets" / "operator_splitting_allen_cahn_2d.png"
    out.parent.mkdir(exist_ok=True)
    figm.savefig(out, bbox_inches="tight")
    print(f"  figure -> {out}")
except Exception as e:  # pragma: no cover
    print(f"  (plot skipped: {e})")

# --- verification ---
assert step_err < 1e-6, f"block.step must match the direct backward-Euler step (rel {step_err:.2e})"
assert sep > 0.7, f"field must separate into +/-1 phases (mean|u| = {sep:.3f})"
assert u_split.min() < -0.5 and u_split.max() > 0.5, "both phases must be present"
assert u_split.min() > -1.05 and u_split.max() < 1.05, "the split solution must stay bounded"
print("  OK: block.step matches backward Euler; the split solve separates into phases.")
