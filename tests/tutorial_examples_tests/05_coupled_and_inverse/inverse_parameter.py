"""05 — Inverse parameter identification

Problem
-------
Given the 1D equation

    a · sin(πx) + b · cos(πx) + c · x(1−x) = d(x)

where d(x) is a noisy measurement of the analytical expression
    d(x) = A·sin(πx) + B·cos(πx) + C·x(1−x)

with hidden ground truth  A = 3.14,  B = −2.71,  C = 42.0

the goal is to recover A, B, C from the residual alone.

Technique
---------
Three scalar parameters (a, b, c) are declared as trainable jno parameters.
A small MLP evaluates sin/cos/x(1-x) at interior points, and the residual
between the parametric expression and the "measured" data is minimised.
"""

import jax
import jno

import optax
from pathlib import Path

π = jno.np.pi
# ── Ground-truth and "measured" data ─────────────────────────────────────────
A_true, B_true, C_true = 3.14, -2.71, 42.0

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.01))
x, _ = domain.variable("interior")

# Target (observed) field — no noise here, but you can add jnp.normal(...) * σ
target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x) + C_true * x * (1 - x)

# ── Trainable scalar parameters ───────────────────────────────────────────────
key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)

a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")
c = jno.np.parameter((1,), key=k3, name="c")

# ── Residual: parametric model − observations ─────────────────────────────────
residual = (a * jno.np.sin(π * x) + b * jno.np.cos(π * x) + c * x * (1 - x)) - target

# ── Optimizers ────────────────────────────────────────────────────────────────
for net in [a, b, c]:
    # Constant LR converges more reliably here: sin(pi*x) and x(1-x)
    # are highly correlated on [0, 1], so aggressive early decay stalls.
    net.optimizer(optax.adam(1e-2))

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([residual.mse], domain)
history = crux.solve(30000)

_a, _b, _c = crux.eval([a, b, c])

print(f"Recovered parameters: a={_a[0]:.3f}, b={_b[0]:.3f}, c={_c[0]:.3f}")

rel_l2_a = float(jax.numpy.linalg.norm(_a - A_true) / (jax.numpy.linalg.norm(A_true) + 1e-8))
rel_l2_b = float(jax.numpy.linalg.norm(_b - B_true) / (jax.numpy.linalg.norm(B_true) + 1e-8))
rel_l2_c = float(jax.numpy.linalg.norm(_c - C_true) / (jax.numpy.linalg.norm(C_true) + 1e-8))

# Write result to tracking file
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"05_coupled_and_inverse/inverse_parameter.py | epochs=30000 | rel_L2_a={rel_l2_a:.6e} | rel_L2_b={rel_l2_b:.6e} | rel_L2_c={rel_l2_c:.6e}\n")

assert rel_l2_a < 1e-1, f"a relative L2 error too large: {rel_l2_a:.3e}"
assert rel_l2_b < 1e-1, f"b relative L2 error too large: {rel_l2_b:.3e}"
assert rel_l2_c < 1e-1, f"c relative L2 error too large: {rel_l2_c:.3e}"
