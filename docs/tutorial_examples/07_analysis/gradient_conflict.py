# --8<-- [start:code]
"""07 — Gradient and sensitivity analysis with u.grad(net)"""

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Domain (hard-BC ansatz ⇒ no boundary sampling needed) ──────────────────────
domain = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.001).domain()  # the unit interval
x, _ = domain.variable("interior")

# ── Exact solution (for validation only, not used in training) ─────────────────
u_exact = jno.np.sin(π * x) / π**2

# ── Network with a hard-enforced Dirichlet BC:  u(0) = u(1) = 0 ────────────────
u_net = jno.nn(
    foundax.mlp(
        in_features=1,
        hidden_dims=32,
        num_layers=3,
        key=jax.random.PRNGKey(0),
    )
).optimizer(optax.adam(1e-3))

u = u_net(x) * x * (1 - x)  # ansatz vanishes at x=0 and x=1
pde = -u.d2(x) - jno.np.sin(π * x)  # residual — should be 0

# ── In-training cosine similarity tracker ─────────────────────────────────────
# Build a boolean mask selecting only the output-layer weight matrix.
# This makes the Jacobian fast to compute — P_out_weight ≪ P_total.
all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
output_mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

# Symbolic Jacobian restricted to the masked parameters; shape (N, P_out) at eval.
J = u.grad(u_net.mask(output_mask))


# Cosine similarity between the LEFT and RIGHT halves of the domain: do the two
# regions push the shared parameters the same way, or fight each other?
def _cos_sim_halves(J):
    mid = J.shape[0] // 2
    g_left, g_right = J[:mid].mean(0), J[mid:].mean(0)
    return jnp.dot(g_left, g_right) / (jnp.linalg.norm(g_left) * jnp.linalg.norm(g_right) + 1e-12)


cos_tracker = jno.np.function(_cos_sim_halves, [J]).tracker(200)  # logged every 200 epochs

# ── Solve ──────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, cos_tracker])
crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(_u - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))
print(f"Relative L² error: {rel_l2:.3e}")
assert rel_l2 < 1e-1, f"solution error too large: {rel_l2:.3e}"

# ── Post-training: final cosine similarity (left vs right halves) ─────────────
# crux.eval([single_expr]) returns the raw array without a batch dimension.
[J_sparse] = crux.eval([J])  # (N, P_out_weight)
mid = J_sparse.shape[0] // 2
g_left, g_right = J_sparse[:mid].mean(0), J_sparse[mid:].mean(0)
cos_sim = float(jnp.dot(g_left, g_right) / (jnp.linalg.norm(g_left) * jnp.linalg.norm(g_right) + 1e-12))
print(f"\ncos_sim (left vs right halves) = {cos_sim:.4f}")
assert -1.0 <= cos_sim <= 1.0, f"cos_sim out of range: {cos_sim:.4f}"

# ── Post-training: full Jacobian + Neural Tangent Kernel ──────────────────────
# Clear the output-layer mask so we get the full (N, P_total) Jacobian.
[J_full] = crux.eval([u.grad(u_net.mask(None))])  # (N, P_total)
N, P_total = J_full.shape
print(f"\nFull Jacobian  J  shape: {J_full.shape}  ({P_total} parameters)")

K = J_full @ J_full.T  # (N, N)
# Clip small negative eigenvalues (numerical noise from semi-definite K).
eigvals = jnp.maximum(jnp.sort(jnp.linalg.eigvalsh(K))[::-1], 0.0)

eff_rank = float(jnp.sum(eigvals) ** 2 / (jnp.sum(eigvals**2) + 1e-12))
cond = float(eigvals[0] / (eigvals[-1] + 1e-12))

print(f"\nNeural Tangent Kernel  K  ({N}×{N})")
print(f"  λ_max        = {float(eigvals[0]):.4f}")
print(f"  λ_min        = {float(eigvals[-1]):.4f}")
print(f"  Eff. rank    = {eff_rank:.2f}  (trace² / ‖K‖²_F)")
print(f"  Cond. number = {cond:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Scale analysis: units & non-dimensionalization  (jno.units)
# ───────────────────────────────────────────────────────────────────────────────
# Gradient conflict has a twin — *scale* conflict. When the additive terms of ONE
# residual differ in magnitude by orders, the loss is ill-conditioned no matter how
# well the collocation points align. jno.units makes that structure explicit: you
# annotate the coordinates and the field with .unit(...)/.scale(...), and it (A)
# audits dimensional consistency and (B) reports — and rewrites away — the
# dimensionless group each term carries (the Fourier/Péclet-type numbers you would
# otherwise derive by hand).

# A thin, anisotropic domain (Lx=1, Ly=1/20). Geometry ALONE puts the two diffusion
# terms of the Laplacian at very different scales — no material coefficient needed.
Lx, Ly, U = 1.0, 0.05, 3.0
adom = jno.Shape.rect(0.0, 0.0, Lx, Ly, size=0.1).domain()
ax, ay, _ = adom.variable("interior", split=True)
ax = ax.unit("m").scale(Lx)  # characteristic length along x
ay = ay.unit("m").scale(Ly)  # 20× shorter characteristic length along y
au = jno.nn(foundax.mlp(in_features=2, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(1)))(ax, ay)
au = au.unit("K").scale(U)  # the field carries a temperature scale U
aniso = au.d2(ax) + au.d2(ay)  # anisotropic Laplacian — two terms in ONE residual

# Phase A — audit + report. check() confirms both terms share a unit (K·m⁻²);
# nondimensionalize() gives each term's dimensionless magnitude πᵢ = Sᵢ / S_ref.
audit = jno.units.check(aniso)
assert not audit.warnings, f"dimensional inconsistency: {audit.warnings}"
terms = jno.units.nondimensionalize(aniso).residuals[0].terms
scale_sep = terms[1].pi / terms[0].pi
print("\nScale analysis (anisotropic Laplacian)")
print(f"  term units       = {[str(t.unit) for t in terms]}")
print(f"  dimensionless πᵢ  = {[round(t.pi, 2) for t in terms]}")
print(f"  scale separation = {scale_sep:.0f}×   (= (Lx/Ly)² = {(Lx / Ly) ** 2:.0f})")
assert abs(scale_sep - (Lx / Ly) ** 2) < 1e-6

# Phase B — the transform. rescale() rewrites the residual to its O(1) dimensionless
# form: the hidden scale separation resurfaces as an explicit leading coefficient,
# and the returned Rescaler maps coordinates to the unit domain and a solution back
# to physical units (u_physical = U · û). This is the non-dimensionalization itself,
# and `transformed` is an ordinary residual you can hand to jno.core on rescaler's
# rescaled_domain(adom).
transformed, rescaler = jno.units.rescale(aniso)
print(f"  rescaler         = {rescaler}")
assert rescaler.field_scale == U
assert float(rescaler.to_physical(1.0)) == U  # û = 1 ↦ U in physical units
# --8<-- [end:code]

# --- figure: the trained solution vs exact, and the NTK eigenvalue spectrum ---
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)
# The network's OWN output on the interior, and the analytic solution, sorted for a clean line.
_xg, _u, _ue = (np.asarray(a).reshape(-1) for a in crux.eval([x, u, u_exact]))
order = np.argsort(_xg)
_ntk = np.asarray(eigvals)  # the NTK spectrum computed above (descending, clipped >= 0)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
ax[0].plot(_xg[order], _u[order], color="#0072B2", lw=2.2, label="jNO (trained)")
ax[0].plot(_xg[order], _ue[order], "--", color="0.25", lw=1.4, label=r"exact  $\sin(\pi x)/\pi^2$")
ax[0].set_title(f"solution   ·   rel $L^2$ = {rel_l2:.1e}")
ax[0].set_xlabel("x")
ax[0].set_ylabel("u")
ax[0].legend(frameon=False)
ax[0].grid(True, alpha=0.25, ls="--")
pos = _ntk[_ntk > 0]
ax[1].semilogy(np.arange(1, pos.size + 1), pos, "o-", ms=3, color="#D55E00")
ax[1].set_title(f"NTK spectrum   ·   eff. rank {eff_rank:.2f},  κ ≈ {cond:.0e}")
ax[1].set_xlabel("index")
ax[1].set_ylabel(r"eigenvalue $\lambda_i(K)$")
ax[1].grid(True, which="both", alpha=0.25, ls="--")
fig.suptitle(f"Gradient/NTK analysis — left/right cos-sim = {cos_sim:.2f}", fontweight="bold")
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "gradient_conflict.png")
