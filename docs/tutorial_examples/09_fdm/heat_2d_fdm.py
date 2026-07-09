# --8<-- [start:code]
"""03 - Transient heat equation through ``jno.fdm`` (method of lines).

    u_t = nu * Delta u  on the unit square,  u = 0 on the boundary,
    u0(x, y) = sin(pi x) sin(pi y)   ->   u(x, y, t) = e^{-2 nu pi^2 t} sin(pi x) sin(pi y).

A problem is **transient** exactly when it carries an initial condition -- and, as in ``jno.fem``,
the IC is *found from the constraints* (``u(xi, yi) - u0``, with ``xi, yi`` the ``"initial"`` region),
never passed as a config flag. The time window and step count come from ``domain.time = (t0, t1, n)``.
The ``u.t`` term marks the time derivative; ``jno.fdm`` marches by the method of lines, reusing the
same semidiscrete time-stepper ``jno.fem`` uses. ``.solve()`` returns the trajectory ``(n_steps, N)``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

nu, T = 0.05, 0.5
d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.06).domain(time=(0.0, T, 200))
x, y, t = d.variable("interior", split=True)  # note the temporal Variable t
xb, yb, _ = d.variable("boundary", split=True)
xi, yi, _ = d.variable("initial", split=True)  # the t = t0 slice
u = d.unknown()
ui = u.bind(x=x, y=y, t=t)

traj = jno.fdm(
    [
        ui.t - nu * (ui.d2(x) + ui.d2(y)),  # u_t = nu * Delta u
        u(xb, yb) - 0.0,  # Dirichlet u = 0
        u(xi, yi) - jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi),  # initial condition u0
    ]
).solve()

p = np.asarray(d.mesh_connectivity["points"])[:, :2]
exact = np.exp(-2 * nu * np.pi**2 * T) * np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
rel_l2 = float(np.linalg.norm(np.asarray(traj)[-1] - exact) / np.linalg.norm(exact))
print(f"\nTransient heat via jno.fdm: steps=200  rel_L2(t={T})={rel_l2:.3e}")
assert rel_l2 < 2e-2, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:code]

# ---- figures: a GIF over the trajectory + a PNG (final field | error | convergence) --------
import os  # noqa: E402

os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

plt.rcParams.update(
    {
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "figure.dpi": 120,
    }
)

tri = mtri.Triangulation(p[:, 0], p[:, 1], triangles=np.asarray(d.mesh_connectivity["triangles"]))
traj_np = np.asarray(traj)  # (n_steps, N); traj_np[0] is the first marched step
ts = np.linspace(0.0, T, traj_np.shape[0])

# GIF: subsample the trajectory to ~40 frames, fixed colour scale (the peak decays 1 -> ~0.6).
frame_idx = np.linspace(0, traj_np.shape[0] - 1, 40).round().astype(int)
figg, axg = plt.subplots(figsize=(5.2, 4.6))


def _frame(k):
    axg.clear()
    j = frame_idx[k]
    im = axg.tripcolor(tri, traj_np[j], cmap="cividis", shading="gouraud", vmin=0.0, vmax=1.0)
    axg.set_aspect("equal")
    axg.set_axis_off()
    axg.set_title(f"u(x, y, t),  t = {ts[j]:.3f}")
    return (im,)


im0 = axg.tripcolor(tri, traj_np[frame_idx[0]], cmap="cividis", shading="gouraud", vmin=0.0, vmax=1.0)
figg.colorbar(im0, ax=axg, shrink=0.8)
ani = animation.FuncAnimation(figg, _frame, frames=len(frame_idx), interval=90, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "heat_2d_fdm.gif", writer="pillow", fps=11, dpi=90)
plt.close(figg)


def _final_rel_l2(size):
    """Re-run the transient solve at a given mesh size; return (h, rel_L2 at t=T)."""
    dm = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, T, 200))
    xx, yy, tt = dm.variable("interior", split=True)
    xxb, yyb, _ = dm.variable("boundary", split=True)
    xxi, yyi, _ = dm.variable("initial", split=True)
    uu = dm.unknown()
    uui = uu.bind(x=xx, y=yy, t=tt)
    tr = jno.fdm(
        [
            uui.t - nu * (uui.d2(xx) + uui.d2(yy)),
            uu(xxb, yyb) - 0.0,
            uu(xxi, yyi) - jnn.sin(np.pi * xxi) * jnn.sin(np.pi * yyi),
        ]
    ).solve()
    pp = np.asarray(dm.mesh_connectivity["points"])[:, :2]
    ex = np.exp(-2 * nu * np.pi**2 * T) * np.sin(np.pi * pp[:, 0]) * np.sin(np.pi * pp[:, 1])
    r = float(np.linalg.norm(np.asarray(tr)[-1] - ex) / np.linalg.norm(ex))
    h = float(np.sqrt(dm.mesh_connectivity["p1_area"].mean()))
    return h, r


sizes = [0.12, 0.09, 0.06]
conv = [_final_rel_l2(sz) for sz in sizes]
hs = np.array([c[0] for c in conv])
errs = np.array([c[1] for c in conv])
print("convergence (h, rel_L2 @t=T):", [(f"{h:.3f}", f"{e:.2e}") for h, e in conv])
slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
print(f"fitted spatial order p = {slope:.2f}")

final = traj_np[-1]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
im0 = ax[0].tripcolor(tri, final, cmap="cividis", shading="gouraud")
ax[0].set_title(f"jno.fdm  u(t={T})")
ax[0].set_axis_off()
ax[0].set_aspect("equal")
fig.colorbar(im0, ax=ax[0], shrink=0.8)

err = final - exact
vmax = float(np.abs(err).max())
im1 = ax[1].tripcolor(tri, err, cmap="RdBu_r", shading="gouraud", vmin=-vmax, vmax=vmax)
ax[1].set_title(r"error  $u - u^*$ at $t=T$")
ax[1].set_axis_off()
ax[1].set_aspect("equal")
fig.colorbar(im1, ax=ax[1], shrink=0.8)

ax[2].loglog(hs, errs, "o-", label="rel-$L^2$ at $t=T$")
ax[2].loglog(hs, errs[0] * (hs / hs[0]) ** 2, "k--", alpha=0.6, label=r"$O(h^2)$")
ax[2].set_title(f"spatial convergence (order ≈ {slope:.2f})")
ax[2].set_xlabel("mean element size $h$")
ax[2].set_ylabel(r"relative $L^2$ error")
ax[2].grid(True, which="both", alpha=0.3)
ax[2].legend()

fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "heat_2d_fdm.png")
