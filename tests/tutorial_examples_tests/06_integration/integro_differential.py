"""06 — Integro-Differential Equation

Equation:  u'(x) + u(x) = g(x) + ∫₀¹ u(t) dt,   u(0) = 0
Exact:     u*(x) = sin(πx)
Forcing:   g(x) = π cos(πx) + sin(πx) − 2/π

Hard BC ansatz: u = net(x) · x  ensures u(0) = 0.
Combines .d(x) and .integrate() in the same residual.
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.005))

x, _ = domain.variable("interior")

# ── Forcing term ───────────────────────────────────────────────────────────────
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

u = net(x) * x  # hard BC: u(0) = 0

C = u.integrate()
du = u.d(x)
residual = du + u - g - C

# ── Solve ──────────────────────────────────────────────────────────────────────
EPOCHS = 30_000
crux = jno.core([residual.mse], domain)
_history = crux.solve(EPOCHS)

# ── Evaluate ───────────────────────────────────────────────────────────────────
u_exact = jno.np.sin(π * x)
u_pred, u_ref = crux.eval([u, u_exact])

rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
assert rel_l2 < 1e-3, f"Relative L2 error too large: {rel_l2:.3e}"
