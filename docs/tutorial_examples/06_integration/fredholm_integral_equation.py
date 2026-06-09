"""06 — Fredholm integral equation of the second kind

Equation
--------
    u(x) = f(x) + ∫₀¹ x·t · u(t) dt,   x ∈ [0, 1]

with forcing term chosen so that the exact solution is

    u*(x) = sin(πx)

Derivation of f
---------------
Substituting u* into the equation:

    sin(πx) = f(x) + ∫₀¹ x·t · sin(πt) dt

The integral evaluates to

    ∫₀¹ t · sin(πt) dt = 1/π   (integration by parts)

so  x · ∫₀¹ t · sin(πt) dt = x/π, giving

    f(x) = sin(πx) − x/π

Network residual
----------------
    R(x) = u(x) − f(x) − x · ∫₀¹ t · u(t) dt = 0

The key insight: ∫₀¹ t · u(t) dt is a single scalar C that does not depend
on x.  We compute it with .integrate(), which evaluates the integrand over
the full mesh and returns a JAX scalar.  This scalar is then multiplied by
the pointwise sampled x, giving a term of shape (N_pts,) in the residual.

The same jno expression u(x) appears both inside the integral (evaluated at
all mesh nodes by the integrator) and in the outer residual (evaluated at
the sampled collocation points).  Both evaluations go through the same
trained network weights.
"""

from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain ─────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.01)
x, _ = domain.variable("interior")

domain.summary()

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
