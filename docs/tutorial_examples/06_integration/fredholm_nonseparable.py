"""06 — Fredholm integral equation with non-separable kernel"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.05)

x, _ = domain.variable("interior")  # outer collocation variable
t, _ = domain.variable("interior")  # inner integration dummy

domain.summary()

# ── Forcing term  f(x) = sin(πx) − 2x/π − 1/π ────────────────────────────────
pi_val = float(jnp.pi)
f = jno.np.sin(π * x) - 2.0 * x / pi_val - 1.0 / pi_val

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

u_x = net(x)  # network evaluated at collocation points  (N, 1)
u_t = net(t)  # same network, evaluated at integration points  (N, 1)

# ── Non-separable Fredholm residual ───────────────────────────────────────────
# ∫₀¹ (x + t) · u(t) dt  — result is (N, 1): depends on x, not a scalar.
# var=x tells the evaluator: keep x fixed, sweep t over the full mesh.
integral_term = ((x + t) * u_t).integrate(var=x)

residual = u_x - f - integral_term

# ── Solve ──────────────────────────────────────────────────────────────────────
EPOCHS = 30_000
crux = jno.core([residual.mse]).print_shapes()
_history = crux.solve(EPOCHS)

# ── Evaluate ───────────────────────────────────────────────────────────────────
u_exact = jno.np.sin(π * x)
u_pred, u_ref = crux.eval([u_x, u_exact])

rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
print(f"Relative L2 error: {rel_l2:.4e}   (exact solution: u(x) = sin(πx))")

# ── Record result ──────────────────────────────────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f_out:
    f_out.write(f"06_integration/fredholm_nonseparable.py | epochs={EPOCHS} | rel_L2={rel_l2:.6e}\n")

assert rel_l2 < 0.10, f"Relative L2 error too large: {rel_l2:.3e}"
