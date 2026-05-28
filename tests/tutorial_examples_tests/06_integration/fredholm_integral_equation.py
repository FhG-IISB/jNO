"""06 — Fredholm integral equation of the second kind

Equation
--------
    u(x) = f(x) + ∫₀¹ x·t · u(t) dt,   x ∈ [0, 1]

with forcing term chosen so that the exact solution is

    u*(x) = sin(πx)

Derivation of f
---------------
    ∫₀¹ t · sin(πt) dt = 1/π   (integration by parts)
    f(x) = sin(πx) − x/π

Network residual
----------------
    R(x) = u(x) − f(x) − x · ∫₀¹ t · u(t) dt = 0

The integral ∫₀¹ t · u(t) dt is a scalar C computed once via .integrate().
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")

# ── Forcing term  f(x) = sin(πx) − x/π ───────────────────────────────────────
pi_val = float(jnp.pi)
f = jno.np.sin(π * x) - x / pi_val

# ── Model ──────────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
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
C = (x * u).integrate()
residual = u - f - x * C

# ── Solve ──────────────────────────────────────────────────────────────────────
EPOCHS = 50_000
crux = jno.core([residual.mse], domain)
_history = crux.solve(EPOCHS)

# ── Evaluate ───────────────────────────────────────────────────────────────────
u_exact = jno.np.sin(π * x)
u_pred, u_ref = crux.eval([u, u_exact])

rel_l2 = float(jnp.linalg.norm(u_pred - u_ref) / (jnp.linalg.norm(u_ref) + 1e-8))
assert rel_l2 < 0.05, f"Relative L2 error too large: {rel_l2:.3e}"
