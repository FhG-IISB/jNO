# --8<-- [start:code]
"""02 — 2-D variable-coefficient Poisson  (Shape domain + named partials + tracker)"""

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi

# --8<-- [start:setup]
domain = jno.Shape.rect(0, 0, 1, 1, size=0.05).domain()
x, y, _ = domain.variable("interior")

κ = 1 + x + y
u_exact = jno.np.sin(π * x) * jno.np.sin(π * y)
# --8<-- [end:setup]

# --8<-- [start:residual]
net = jno.nn(foundax.mlp(in_features=2, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(13)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 1000, 0.5, end_value=1e-5)))

# Multiplicative hard BC — the x(1-x)y(1-y) factor's derivatives flow through.
u = (net(x, y) * x * (1 - x) * y * (1 - y)).scalar.bind(x=x, y=y)
# −∂_x(κ ∂_x u) − ∂_y(κ ∂_y u) = forcing.  Named-partial syntax reads
# component-by-component as if from the math, including the product rule.
pde = -((κ * u.x).d(x) + (κ * u.y).d(y)) - (
    2 * π**2 * κ * u_exact - π * jno.np.cos(π * x) * jno.np.sin(π * y) - π * jno.np.sin(π * x) * jno.np.cos(π * y)
)
# --8<-- [end:residual]

# --8<-- [start:solve]
grad_norms = jno.trackers.gradient_norms(interval=500)
crux = jno.core([pde.mse])
crux.solve(5_000, callbacks=[grad_norms])
# --8<-- [end:solve]

# --8<-- [start:eval]
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
if grad_norms.value is not None:
    print(f"Final ∇L norm: {grad_norms.value['norms']}")
# --8<-- [end:eval]

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"02_elliptic/variable_coefficient_poisson_2d.py | epochs=5000 | rel_L2={rel_l2:.6e}\n")

# --8<-- [start:assert]
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:assert]
# --8<-- [end:code]

# ---------------------------------------------------------------------------
# Figure (hidden from the rendered docs): the trained network's OWN output on
# the mesh nodes, the manufactured exact field, and their signed error.
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

_u_n, _ue_n, _x_n, _y_n = crux.eval([u, u_exact, x, y])
xn = jax.numpy.asarray(_x_n).reshape(-1)
yn = jax.numpy.asarray(_y_n).reshape(-1)
up = jax.numpy.asarray(_u_n).reshape(-1)
ue = jax.numpy.asarray(_ue_n).reshape(-1)
err = up - ue

vmin = float(min(up.min(), ue.min()))
vmax = float(max(up.max(), ue.max()))
elim = float(jax.numpy.abs(err).max())

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
im0 = axs[0].tricontourf(xn, yn, up, levels=40, cmap="cividis", vmin=vmin, vmax=vmax)
axs[0].set_title("jNO  $u$")
fig.colorbar(im0, ax=axs[0], shrink=0.8)
im1 = axs[1].tricontourf(xn, yn, ue, levels=40, cmap="cividis", vmin=vmin, vmax=vmax)
axs[1].set_title(r"exact  $\sin\pi x\,\sin\pi y$")
fig.colorbar(im1, ax=axs[1], shrink=0.8)
im2 = axs[2].tricontourf(xn, yn, err, levels=40, cmap="RdBu_r", vmin=-elim, vmax=elim)
axs[2].set_title(f"error  (rel-$L^2$={rel_l2:.1e})")
fig.colorbar(im2, ax=axs[2], shrink=0.8)
for ax in axs:
    ax.set_aspect("equal")
    ax.set_axis_off()

fig.tight_layout()
_out = Path(__file__).parents[2] / "assets" / "variable_coefficient_poisson_2d.png"
fig.savefig(_out)
print(f"Saved figure to {_out}")
