# --8<-- [start:code]
"""01 — 1-D Laplace equation (simplest possible PINN)"""

import foundax
import jax
import optax

import jno

domain = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.1).domain()  # x in [0, 1]
x, _ = domain.variable("interior")

u_exact = x

net = jno.nn(foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

u = (x + x * (1 - x) * net(x)).scalar.bind(x=x)
pde = u.xx  # Laplace: u'' = 0

crux = jno.core([pde.mse])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}")
assert rel_l2 < 1e-1, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:code]

# ---------------------------------------------------------------------------
# Figure (hidden from the rendered docs): predicted-vs-exact line + a real
# convergence study (re-solve at several mesh sizes, same seed/epochs so only
# the collocation density varies).
# ---------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

# Predicted vs exact, evaluated on a dense grid = the trained network's OWN output.
_dense = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.005).domain()
_u_dense, _x_dense = crux.eval([u, x], domain=_dense)
xs = jax.numpy.asarray(_x_dense).reshape(-1)
up = jax.numpy.asarray(_u_dense).reshape(-1)


def _solve_at(size):
    """Re-run the identical problem at a given mesh size; return (n_points, rel_L2)."""
    d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=size).domain()
    xd, _ = d.variable("interior")
    n = jno.nn(foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0)))
    n.optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))
    ud = (xd + xd * (1 - xd) * n(xd)).scalar.bind(x=xd)
    c = jno.core([ud.xx.mse])
    c.solve(5000)
    _p, _e = c.eval([ud, xd])
    err = float(jax.numpy.linalg.norm(_p - _e) / (jax.numpy.linalg.norm(_e) + 1e-8))
    return int(jax.numpy.asarray(_p).size), err


sizes = [0.2, 0.1, 0.05, 0.025]
conv = [_solve_at(s) for s in sizes]
npts = [c[0] for c in conv]
errs = [c[1] for c in conv]

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

ax0.plot(xs, up, label="jNO", lw=2)
ax0.plot(xs, xs, "--", label="exact  $u=x$", lw=1.5)
ax0.set_xlabel("x")
ax0.set_ylabel("u(x)")
ax0.set_title(f"Predicted vs exact  (rel-$L^2$={rel_l2:.1e})")
ax0.legend()
ax0.grid(True, alpha=0.3)

ax1.loglog(npts, errs, "o-")
ax1.set_xlabel("collocation points")
ax1.set_ylabel("relative $L^2$ error")
ax1.set_title("Convergence (training-limited)")
ax1.grid(True, which="both", alpha=0.3)

fig.tight_layout()
_out = Path(__file__).parents[2] / "assets" / "laplace_1d.png"
fig.savefig(_out)
print(f"Saved figure to {_out}")
print("convergence (n_points, rel_L2):", list(zip(npts, errs)))
