# --8<-- [start:code]
"""05 — Inverse parameter identification"""

from pathlib import Path

import jax
import optax

import jno

π = jno.np.pi
A_true, B_true, C_true = 3.14, -2.71, 42.0

domain = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.01).domain()
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
# --8<-- [end:code]

# ── Figure: recovered coefficients vs truth + fitted field vs data ────────────
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

# Fitted field = recovered a·sin + b·cos + c·x(1-x); target = the synthetic data.
fitted = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) + c * x * (1 - x)
_x, _fit, _tgt = crux.eval([x, fitted, target])
_x = np.asarray(_x).ravel()
_fit = np.asarray(_fit).ravel()
_tgt = np.asarray(_tgt).ravel()
order = np.argsort(_x)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

labels = ["a", "b", "c"]
truth = [A_true, B_true, C_true]
recov = [float(_a[0]), float(_b[0]), float(_c[0])]
xpos = np.arange(3)
axes[0].bar(xpos - 0.18, truth, width=0.36, label="true", color="#4c72b0")
axes[0].bar(xpos + 0.18, recov, width=0.36, label="recovered", color="#dd8452")
for i, (tv, rv) in enumerate(zip(truth, recov)):
    axes[0].text(i, max(tv, rv) + 1.5, f"{rv:.2f}", ha="center", fontsize=8)
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("coefficient value")
axes[0].set_title("recovered vs true coefficients")
axes[0].legend(fontsize=8)
axes[0].axhline(0, color="k", lw=0.6)

axes[1].plot(_x[order], _tgt[order], "k--", lw=2, label="target data")
axes[1].plot(_x[order], _fit[order], color="#dd8452", label="fitted field")
axes[1].set_xlabel("x")
axes[1].set_ylabel("value")
axes[1].set_title("fitted field vs data")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.savefig(Path(__file__).parents[2] / "assets" / "inverse_parameter.png")
