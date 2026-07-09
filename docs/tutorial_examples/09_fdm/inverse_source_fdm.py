# --8<-- [start:code]
"""04 - Differentiable inverse through ``jno.fdm`` + ``jno.core``: recover an unknown source amplitude.

When the constraint list carries a trainable ``jno.np.parameter``, ``jno.fdm([...]).solve()`` returns
a differentiable **trace node** (not an array) -- exactly as ``fem.solve()`` does -- so it composes
straight into ``jno.core``. We run a twin experiment: generate a synthetic observation from the
forward solve at the true amplitude ``s = 1``, then recover ``s`` from a deliberately wrong start by
minimising the data misfit, with the parameter's own attached optimizer driving the fit.

    -Delta u = s * f_base,  u = 0 on the boundary.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).domain()
x, y, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
f_base = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
u = d.unknown()
ui = u.bind(x=x, y=y)

# Synthetic observation: the forward solve at the true amplitude s = 1 (a plain float -> eager array).
observed = jnp.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - 1.0 * f_base, u(xb, yb) - 0.0]).solve()).reshape(-1)

# Recover s: a trainable parameter with an attached optimizer, driven by crux through the data misfit.
s = jno.np.parameter((1,), name="s")
s.dtype(jnp.float64)
s.initialize(jax.nn.initializers.constant(2.5))  # deliberately wrong start
# A single, well-scaled scalar over a convex (quadratic) misfit: plain gradient descent converges
# straight to the minimum. Adam's per-parameter moment adaptation is counter-productive here — it
# oscillates and can settle at a spurious fixed point away from the true amplitude.
s.optimizer(optax.sgd(1.0))
solve = jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()  # a differentiable trace node
crux = jno.core([(solve - observed).mse])  # domain inferred from the graph — no explicit domain= needed
crux.solve(150)

rec = float(np.asarray(crux.eval([s])).reshape(-1)[0])  # the recovered amplitude (do NOT index [0] on a field)
print(f"\nInverse via jno.fdm + jno.core: recovered source amplitude s={rec:.4f}  (true 1.0)")
assert abs(rec - 1.0) < 1e-2, f"did not recover the source amplitude: s={rec:.4f}"
# --8<-- [end:code]

# ---- figure: observed field | fit residual at s_rec | parameter convergence s -> 1 ----------
import os  # noqa: E402

os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

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

# The forward field at the recovered amplitude (the model's OWN computed output at s = s_rec).
u_rec = np.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - rec * f_base, u(xb, yb) - 0.0]).solve()).reshape(-1)
obs = np.asarray(observed).reshape(-1)

# An independent re-run (same SGD(1.0) optimizer) to capture the recovered scalar's convergence.
# |s - 1| falls geometrically until it saturates at the iterative forward solver's tolerance floor
# (~1e-4); we plot the clean descent to that floor (beyond it the residual noise makes SGD(1.0) bounce
# off the floor — a solver-tolerance artifact, not a failure to recover; the 150-epoch solve above
# lands at s = 1.0000).
s2 = jno.np.parameter((1,), name="s_conv")
s2.dtype(jnp.float64)
s2.initialize(jax.nn.initializers.constant(2.5))
s2.optimizer(optax.sgd(1.0))
solve2 = jno.fdm([-ui.d2(x) - ui.d2(y) - s2 * f_base, u(xb, yb) - 0.0]).solve()
crux2 = jno.core([(solve2 - observed).mse])
s_hist = [2.5]
n_epochs = 18  # descent to the ~1e-4 tolerance floor (before the noise-floor limit cycle)
for _ in range(n_epochs):
    crux2.solve(1)
    s_hist.append(float(np.asarray(crux2.eval([s2])).reshape(-1)[0]))
s_hist = np.array(s_hist)
err_s = np.abs(s_hist - 1.0)
print("|s - 1| trajectory (first 6):", [f"{e:.2e}" for e in err_s[:6]], "... final:", f"{err_s[-1]:.2e}")

p = np.asarray(d.mesh_connectivity["points"])[:, :2]
tri = mtri.Triangulation(p[:, 0], p[:, 1], triangles=np.asarray(d.mesh_connectivity["triangles"]))
fig, ax = plt.subplots(1, 3, figsize=(13, 4))

im0 = ax[0].tripcolor(tri, obs, cmap="cividis", shading="gouraud")
ax[0].set_title("observed field  $u^*$\n(data: forward solve at true $s=1$)")
ax[0].set_axis_off()
ax[0].set_aspect("equal")
fig.colorbar(im0, ax=ax[0], shrink=0.8)

resid = u_rec - obs
vmax = float(np.abs(resid).max()) or 1e-16
im1 = ax[1].tripcolor(tri, resid, cmap="RdBu_r", shading="gouraud", vmin=-vmax, vmax=vmax)
ax[1].set_title(f"fit residual  $u(s_{{rec}}) - u^*$\n(max |·| = {vmax:.1e})")
ax[1].set_axis_off()
ax[1].set_aspect("equal")
fig.colorbar(im1, ax=ax[1], shrink=0.8)

ax[2].semilogy(np.arange(len(err_s)), err_s, "o-")
ax[2].set_title(f"parameter convergence\n$s: 2.5 \\to {rec:.4f}$  (true 1.0)")
ax[2].set_xlabel("gradient step")
ax[2].set_ylabel(r"$|s - 1|$")
ax[2].grid(True, which="both", alpha=0.3)

fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "inverse_source_fdm.png")
