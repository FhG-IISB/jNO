"""2D time-dependent Schrodinger equation: a Gaussian wave packet scattering off a potential barrier.

    i d(psi)/dt = H psi,   H = -1/2 lap + V(x),     psi complex, psi = 0 far away.

The wavefunction is genuinely complex -- the momentum lives in the phase ``exp(i k.x)``. jno.fem
assembles the (real, symmetric) mass ``M`` and Hamiltonian ``H = 1/2 stiffness + V``; the imaginary
unit enters through the time stepping. Schrodinger evolution is *unitary* (the norm is conserved
exactly), and the default backward-Euler is strongly dissipative for it -- so we bring our own
norm-preserving **Crank-Nicolson** stepper (a Cayley transform of ``H``):

    (M + i dt/2 H) psi_{n+1} = (M - i dt/2 H) psi_n.

A Gaussian packet is launched toward a tall thin barrier; part tunnels through and part reflects,
with interference fringes where the incoming and reflected waves overlap. The animation is |psi|^2,
the probability density (the actual computed field at each frame).
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")  # play nice on a shared GPU

import jax

jax.config.update("jax_enable_x64", True)  # the assembler builds in float64; psi is complex128

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

Lx, Ly, xbar = 2.0, 0.9, 1.0  # box, barrier x-position
sigma, k, V0 = 0.1, 16.0, 220.0  # packet width, momentum (+x); barrier height (E = k^2/2 = 128 < V0)
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

# assemble the REAL mass M and Hamiltonian H = 1/2 stiffness + V(x) mass with jno.fem
d = jno.domain(box(0, 0, Lx, Ly), mesh_size=0.025, time=(0.0, 0.05, 2))
u, phi = d.fem_symbols()
xi, yi, ti = d.variable("interior", split=True)
ci = d.variable("initial", split=True)
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
V = V0 * jno.np.exp(-((xi - xbar) ** 2) / (2 * 0.04**2))  # a tall, thin potential ridge
block = jno.fem([ui.t * vi + 0.5 * (ui.x * vi.x + ui.y * vi.y) + V * (u * vi), u(ci[0], ci[1]) - 0.0]).operator
M, H = dense(block.M), dense(block.A)
pts = np.asarray(jnp.asarray(d.mesh.points)[:, :2])
print(f"\n2D Schrodinger wavepacket (Crank-Nicolson, unitary): nodes={pts.shape[0]}")

# complex Gaussian wave packet, launched toward the barrier
psi0 = np.exp(-((pts[:, 0] - 0.45) ** 2 + (pts[:, 1] - 0.45) ** 2) / (2 * sigma**2)) * np.exp(1j * k * pts[:, 0])
psi = jnp.asarray(psi0, dtype=jnp.complex128)
T, nsteps, nframes = 0.08, 320, 40
dt = T / nsteps
P = jnp.linalg.solve(M + 0.5j * dt * H, M - 0.5j * dt * H)  # CN propagator (factor once -> matvec per step)
mnorm = lambda p: float(jnp.real(jnp.vdot(p, M @ p)))  # noqa: E731  conserved norm psi^dag M psi

frames, n0 = [np.asarray(psi)], mnorm(psi)
every = nsteps // nframes
for step in range(nsteps):
    psi = P @ psi
    if (step + 1) % every == 0:
        frames.append(np.asarray(psi))
frames = np.stack(frames)  # (n_frames, n_nodes) complex
dens = np.abs(frames) ** 2  # |psi|^2 probability density
left = pts[:, 0] < xbar
refl = float((dens[-1, left]).sum() / dens[-1].sum())
print(f"  norm ratio (CN is unitary): {mnorm(jnp.asarray(frames[-1])) / n0:.4f}")
print(f"  reflected {refl:.2f} / transmitted {1 - refl:.2f}  |  max|Im psi|={float(np.abs(frames.imag).max()):.2f}")

# ---- animate the probability density |psi|^2 -> a looping GIF ----
tris = np.asarray(d.built_mesh.cells_dict["triangle"])
triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
vmax = 0.3 * float(dens[0].max())  # clip below the incoming peak so the reflected/transmitted parts stay vivid
fig, ax = plt.subplots(figsize=(8.6, 4.0))
tpc = ax.tripcolor(triang, dens[0], cmap="inferno", shading="gouraud", vmin=0.0, vmax=vmax)
ax.axvline(xbar, color="cyan", lw=1.2, alpha=0.6, ls="--")  # the barrier
fig.colorbar(tpc, ax=ax, shrink=0.85, label="$|\\psi|^2$")
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])


def _frame(j):
    tpc.set_array(dens[j])
    ax.set_title(f"2D Schrödinger wave packet (|ψ|²) — frame {j}/{len(dens) - 1}", fontsize=11)
    return (tpc,)


ani = animation.FuncAnimation(fig, _frame, frames=len(dens), interval=80, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "schrodinger_wavepacket_2d.gif", writer="pillow", fps=11, dpi=82)

assert mnorm(jnp.asarray(frames[-1])) / n0 > 0.999, "Crank-Nicolson should conserve the norm"
assert refl > 0.1 and (1 - refl) > 0.1, "expected a partial reflection AND transmission (a genuine split)"
assert float(np.abs(frames.imag).max()) > 1e-3, "psi must be genuinely complex"
