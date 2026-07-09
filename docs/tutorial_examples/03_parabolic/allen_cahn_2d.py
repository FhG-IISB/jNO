# --8<-- [start:code]
"""03 — 2-D Allen–Cahn equation (manufactured-solution verification)"""

import foundax
import jax
import optax

import jno

π = jno.np.pi
ε = 0.1
T_end = 1.0

# Time-dependent unit square: the Shape one-liner forwards ``time=`` to the domain.
domain = jno.Shape.rect(0, 0, 1, 1, size=0.05).domain(time=(0, T_end, 4))
x, y, t = domain.variable("interior")

S = jno.np.sin(π * x) * jno.np.sin(π * y)
u_exact = jno.np.exp(-t) * S
source = jno.np.exp(-t) * S * (2 * ε**2 * π**2 - 2) + jno.np.exp(-3 * t) * S**3

# Network with hard Dirichlet BCs in space; t is fed via the trunk input.
net = jno.nn(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=2,
        n_outputs=1,
        n_layers=3,
        basis_functions=64,
        hidden_dim=40,
        key=jax.random.PRNGKey(42),
    )
)
# Loss-adaptive LR: the DLRS schedule raises/lowers the step size from the recent loss
# slope, so training slows when the stiff ε-thin Allen–Cahn interface makes the loss
# stagnate and speeds up when it can descend — no hand-tuned decay curve required.
net.optimizer(optax.adam(1)).scale(jno.fn.adaptive.dlrs(lr0=1e-3, window=10))

xy = jno.np.concat([x, y])
# Bind names so partials read like the math:  u.t, u.xx, u.yy, u.xy, ...
u = (net(t, xy) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y, t=t)

pde = u.t - ε**2 * (u.xx + u.yy) - u + u**3 - source

# Initial condition  (t=0 via 0*t trick)
u_at_0 = net(0 * t, xy) * x * (1 - x) * y * (1 - y)
ini = u_at_0 - S

grad_norms = jno.trackers.gradient_norms(interval=500)
crux = jno.core([pde.mse, ini.mse])

print(f"Allen–Cahn 2-D  (ε = {ε})")
crux.solve(5000, callbacks=[grad_norms])

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
if grad_norms.value is not None:
    print(f"Final ∇L norms (pde, ini): {grad_norms.value['norms']}")

assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:code]

# ---------------------------------------------------------------------------
# Figures (hidden from the rendered docs): a time-lapse GIF of the trained
# network's OWN field u(x,y,t) (consistent colour scale), and a pred-vs-exact
# panel at the final time against the manufactured solution.
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

# Re-evaluate the trained model on a finer time grid (the network is continuous
# in t, so this is still its own output) for a smooth animation.
n_frames = 16
_dg = jno.Shape.rect(0, 0, 1, 1, size=0.05).domain(time=(0, T_end, n_frames))
_uf, _uef, _xf, _yf, _tf = crux.eval([u, u_exact, x, y, t], domain=_dg, min_consecutive=n_frames)
uf = np.asarray(_uf)[0, :, :, 0]  # (frame, node)
uef = np.asarray(_uef)[0, :, :, 0]
xn = np.asarray(_xf)[0, 0, :, 0]
yn = np.asarray(_yf)[0, 0, :, 0]
tvals = np.linspace(0, T_end, n_frames)

vmin = float(min(uf.min(), uef.min()))
vmax = float(max(uf.max(), uef.max()))
levels = np.linspace(vmin, vmax, 40)

assets = Path(__file__).parents[2] / "assets"
figf, axf = plt.subplots(figsize=(4.4, 4))
axf.set_aspect("equal")
axf.set_axis_off()
_im = axf.tricontourf(xn, yn, uf[0], levels=levels, cmap="cividis", extend="both")
figf.colorbar(_im, ax=axf, shrink=0.8)


def _frame(k):
    axf.clear()
    axf.set_aspect("equal")
    axf.set_axis_off()
    axf.tricontourf(xn, yn, uf[k], levels=levels, cmap="cividis", extend="both")
    axf.set_title(f"jNO  $u(x,y,t)$   t = {tvals[k]:.2f}")
    return []


ani = animation.FuncAnimation(figf, _frame, frames=n_frames, interval=200, blit=False)
gif_path = assets / "allen_cahn_2d.gif"
ani.save(gif_path, writer="pillow", fps=5, dpi=90)
plt.close(figf)
print(f"Saved GIF to {gif_path}")

# Pred-vs-exact panel at the final time (own colour range for contrast; the
# field has decayed by e^-1, so the GIF's global scale would render it dim).
up, ue = uf[-1], uef[-1]
err = up - ue
elim = float(np.abs(err).max())
flevels = np.linspace(float(min(up.min(), ue.min())), float(max(up.max(), ue.max())), 40)
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
im0 = axs[0].tricontourf(xn, yn, up, levels=flevels, cmap="cividis", extend="both")
axs[0].set_title(f"jNO  $u$  (t={T_end:.0f})")
fig.colorbar(im0, ax=axs[0], shrink=0.8)
im1 = axs[1].tricontourf(xn, yn, ue, levels=flevels, cmap="cividis", extend="both")
axs[1].set_title(r"exact  $e^{-t}\sin\pi x\,\sin\pi y$")
fig.colorbar(im1, ax=axs[1], shrink=0.8)
im2 = axs[2].tricontourf(xn, yn, err, levels=40, cmap="RdBu_r", vmin=-elim, vmax=elim)
axs[2].set_title(f"error  (rel-$L^2$={rel_l2:.1e})")
fig.colorbar(im2, ax=axs[2], shrink=0.8)
for ax in axs:
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.tight_layout()
png_path = assets / "allen_cahn_2d.png"
fig.savefig(png_path)
print(f"Saved figure to {png_path}")
