"""06 — Integro-Differential Equation (IDE)"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.05)

x, _ = domain.variable("interior")

domain.summary()

# ── Forcing term  g(x) = π cos(πx) + sin(πx) − 2/π ──────────────────────────
pi_val = float(jnp.pi)
g = π * jno.np.cos(π * x) + jno.np.sin(π * x) - 2.0 / pi_val

# ── Model ──────────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=32,
        num_layers=3,
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

# Hard boundary condition: u(0) = 0 — multiplying by x enforces it for all x
u = net(x) * x

# ── IDE residual ──────────────────────────────────────────────────────────────
# ∫₀¹ u(t) dt is a scalar (not a function of x).
# u.d(x) is the pointwise derivative — both shapes are (N, 1).
C = u.integrate()  # scalar: ∫₀¹ u(t) dt
du = u.d(x)  # (N, 1): u'(x) at every collocation point
residual = du + u - g - C  # IDE residual at every collocation point

# ── Solve ──────────────────────────────────────────────────────────────────────
EPOCHS = 30_000
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
    f_out.write(f"06_integration/integro_differential.py | epochs={EPOCHS} | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 0.10, f"Relative L2 error too large: {rel_l2:.3e}"
