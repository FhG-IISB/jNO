# --8<-- [start:code]
"""06 — Fredholm integral equation of the second kind"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.01).domain()
x, _ = domain.variable("interior")

domain.summary()

# ── Forcing term  f(x) = sin(πx) − x/π ───────────────────────────────────────
pi_val = float(jnp.pi)
f = jno.np.sin(π * x) - x / pi_val

# ── Model ──────────────────────────────────────────────────────────────────────
net = jno.nn(
    foundax.mlp(
        in_features=1,
        hidden_dims=64,
        num_layers=4,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(
    optax.adam(
        optax.exponential_decay(
            init_value=1e-3,
            transition_steps=5_000,
            decay_rate=0.5,
            end_value=1e-5,
        )
    )
)

u = net(x)

# ── Fredholm residual ──────────────────────────────────────────────────────────
# C = ∫₀¹ t · u(t) dt  — scalar, independent of x.
# .integrate() evaluates the integrand over all mesh nodes and sums with
# nodal volume weights.  Here x is the integration variable (dummy variable t).
C = (x * u).integrate()

# Pointwise residual  R(xᵢ) = u(xᵢ) − f(xᵢ) − xᵢ · C
residual = u - f - x * C

# ── Solve ──────────────────────────────────────────────────────────────────────
EPOCHS = 50_000
crux = jno.core([residual.mse]).print_shapes()
_history = crux.solve(EPOCHS)

# ── Evaluate ───────────────────────────────────────────────────────────────────
u_exact = jno.np.sin(π * x)
u_pred, u_ref = crux.eval([u, u_exact])

rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}   (exact solution: u(x) = sin(πx))")

# ── Record result ──────────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f_out:
    f_out.write(f"06_integration/fredholm_integral_equation.py | epochs={EPOCHS} | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 0.05, f"Relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:code]

# ── Figure: computed solution vs analytic u*(x) = sin(pi x) + pointwise error ─
# (No convergence panel: this is a PINN, so the error is optimisation-limited,
#  not discretisation-limited — a mesh-refinement line would be meaningless.)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

_x = np.asarray(crux.eval([x])).ravel()
_up = np.asarray(u_pred).ravel()
_ur = np.asarray(u_ref).ravel()
order = np.argsort(_x)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(_x[order], _up[order], color="#4c72b0", label="jNO")
axes[0].plot(_x[order], _ur[order], "--", color="k", label=r"exact  $\sin(\pi x)$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("u(x)")
axes[0].set_title("computed vs analytic solution")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(_x[order], np.abs(_up[order] - _ur[order]), color="#c44e52")
axes[1].set_xlabel("x")
axes[1].set_ylabel("|u − u*|")
axes[1].set_title(f"pointwise error (rel-L2 = {rel_l2:.2e})")
axes[1].set_yscale("log")
axes[1].grid(True, which="both", alpha=0.3)

fig.savefig(Path(__file__).parents[2] / "assets" / "fredholm_integral_equation.png")
