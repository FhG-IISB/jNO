"""06 — Fredholm integral equation with non-separable kernel

Equation
--------
    u(x) = f(x) + ∫₀¹ (x + t) · u(t) dt,   x ∈ [0, 1]

The kernel K(x, t) = x + t depends on both the evaluation point x and
the integration dummy t.  It cannot be collapsed to a single scalar
without evaluating it for every x simultaneously — exactly what
.integrate(var=x) enables.

Exact solution:  u*(x) = sin(πx)

Derivation of f
---------------
    ∫₀¹ (x + t) sin(πt) dt
        = x · ∫₀¹ sin(πt) dt  +  ∫₀¹ t · sin(πt) dt
        = x · (2/π)            +  1/π

    f(x) = sin(πx) − 2x/π − 1/π

Network residual
----------------
    R(x) = u(x) − f(x) − ∫₀¹ (x + t) · u(t) dt  =  0

API used
--------
Two variables are obtained from the same domain tag — one acting as the
outer collocation variable, one as the inner integration dummy.  The outer
variable is passed to .integrate(var=x), so the result is an (N, 1) array
(a function of x) rather than a scalar:

    x, _ = domain.variable("interior")   # outer / collocation
    t, _ = domain.variable("interior")   # inner / dummy  — no flag needed!

    integral_term = ((x + t) * u(t)).integrate(var=x)
"""

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
