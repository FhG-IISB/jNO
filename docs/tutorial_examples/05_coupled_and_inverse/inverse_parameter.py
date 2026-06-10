"""05 — Inverse parameter identification"""

from pathlib import Path

import jax
import optax

import jno

π = jno.np.pi
A_true, B_true, C_true = 3.14, -2.71, 42.0

domain = jno.domain.line(mesh_size=0.01)
x, _ = domain.variable("interior")

target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x) + C_true * x * (1 - x)

k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
a = jno.np.parameter((1,), key=k1, name="a")
b = jno.np.parameter((1,), key=k2, name="b")
c = jno.np.parameter((1,), key=k3, name="c")

residual = (a * jno.np.sin(π * x) + b * jno.np.cos(π * x) + c * x * (1 - x)) - target

for param in (a, b, c):
    param.optimizer(optax.adam(1e-2))

crux = jno.core([residual.mse])
crux.solve(30_000)

_a, _b, _c = crux.eval([a, b, c])
print(f"Recovered: a={_a[0]:.3f}  b={_b[0]:.3f}  c={_c[0]:.3f}    (truth: {A_true}, {B_true}, {C_true})")

rel_l2_a = float(jax.numpy.linalg.norm(_a - A_true) / (jax.numpy.linalg.norm(A_true) + 1e-8))
rel_l2_b = float(jax.numpy.linalg.norm(_b - B_true) / (jax.numpy.linalg.norm(B_true) + 1e-8))
rel_l2_c = float(jax.numpy.linalg.norm(_c - C_true) / (jax.numpy.linalg.norm(C_true) + 1e-8))

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"05_coupled_and_inverse/inverse_parameter.py | epochs=30000"
        f" | rel_L2_a={rel_l2_a:.6e} | rel_L2_b={rel_l2_b:.6e} | rel_L2_c={rel_l2_c:.6e}\n"
    )

assert rel_l2_a < 1e-1, f"a rel_L2 too large: {rel_l2_a:.3e}"
assert rel_l2_b < 1e-1, f"b rel_L2 too large: {rel_l2_b:.3e}"
assert rel_l2_c < 1e-1, f"c rel_L2 too large: {rel_l2_c:.3e}"
