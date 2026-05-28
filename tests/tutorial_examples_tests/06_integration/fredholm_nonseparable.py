"""06 — Fredholm integral equation with non-separable kernel

Equation:  u(x) = f(x) + ∫₀¹ (x + t) · u(t) dt
Exact:     u*(x) = sin(πx)
Forcing:   f(x) = sin(πx) − 2x/π − 1/π

Demonstrates .integrate(var=x) for a kernel that depends on both the
outer collocation variable x and the inner dummy t.
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.05))

x, _ = domain.variable("interior")  # outer collocation variable
t, _ = domain.variable("interior")  # inner integration dummy

# ── Forcing term ───────────────────────────────────────────────────────────────
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

u_x = net(x)
u_t = net(t)

integral_term = ((x + t) * u_t).integrate(var=x)
residual = u_x - f - integral_term

# ── Solve ──────────────────────────────────────────────────────────────────────
EPOCHS = 30_000
crux = jno.core([residual.mse], domain)
_history = crux.solve(EPOCHS)

# ── Evaluate ───────────────────────────────────────────────────────────────────
u_exact = jno.np.sin(π * x)
u_pred, u_ref = crux.eval([u_x, u_exact])

rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
assert rel_l2 < 0.10, f"Relative L2 error too large: {rel_l2:.3e}"
