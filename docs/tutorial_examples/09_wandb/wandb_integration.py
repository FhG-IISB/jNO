"""09 — W&B + Weave integration with explainability callbacks

Problem
-------
    −u''(x) = sin(πx),   x ∈ [0, 1]
    u(0) = u(1) = 0   (soft boundary condition as second constraint)

Exact solution
--------------
    u(x) = sin(πx) / π²

Techniques shown
----------------
* jno.setup(..., wandb=True)  — starts a W&B + Weave run automatically
* GradientNormsCallback        — ‖∇Lᵢ‖₂ per constraint, logged to W&B
* CosSimilarityCallback        — pairwise cosine similarity matrix (heatmap)
* GradientAlignmentCallback    — total alignment scalar ‖Σgᵢ‖ / Σ‖gᵢ‖
* LossLandscapeCallback        — 2-D loss landscape around current params
* ResidualStatsCallback        — per-constraint residual mean/std/max/p99 + histogram
* InputSensitivityCallback     — |∂u/∂x| at collocation points (saliency)
* NTKSpectrumCallback          — empirical NTK eigenvalue spectrum
* HessianSpectrumCallback      — top-k Hessian eigenvalues + sharpness
* CheckpointCallback           — periodic Orbax save + W&B artifact upload

Run this script directly to see all metrics appear in your W&B dashboard
under the project configured in jno.setup() (default: the script stem).
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Setup: creates the run directory and initialises W&B + Weave ──────────────
# Set wandb=True to push metrics to Weights & Biases.
# The project name defaults to the script filename stem ("wandb_integration").
# Override with:  jno.setup(__file__, name="my_run", wandb={"project": "jNO"})
dire = jno.setup(__file__, wandb=True)

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain.line(mesh_size=0.05)
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Exact solution (validation only) ──────────────────────────────────────────
u_exact = jno.np.sin(π * x) / π**2

# ── Network ───────────────────────────────────────────────────────────────────
u_net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=32,
        num_layers=3,
        key=jax.random.PRNGKey(0),
    )
).optimizer(optax.adam(optax.exponential_decay(1e-3, 1_000, 0.5, end_value=1e-5)))

u = u_net(x)
ub = u_net(xb)

# ── Constraints ───────────────────────────────────────────────────────────────
pde = -u.d2(x) - jno.np.sin(π * x)  # PDE residual
bc = ub  # u = 0 on boundary

# ── Explainability callbacks ───────────────────────────────────────────────────
# All four are logged to W&B when the run is active.
# Use interval= to control how often each is computed.
# The loss landscape is the most expensive — keep its interval large for real runs.

cb_norms = jno.callbacks.gradient_norms(
    interval=50,
    # mask=...  # optional: restrict to a subset of parameters (faster)
)

cb_cos = jno.callbacks.cos_similarity(
    interval=50,
)

cb_align = jno.callbacks.gradient_alignment(
    interval=50,
)

cb_landscape = jno.callbacks.loss_landscape(
    interval=200,  # expensive — n_grid² forward passes per call
    n_grid=11,
    alpha_range=0.5,
)

cb_residual = jno.callbacks.residual_stats(
    interval=50,
)

# Input saliency: |∂u/∂x| at the interior collocation points.
# Any jno placeholder expression compiles — try Jacobian(u, [x, y]) for 2-D problems.
cb_saliency = jno.callbacks.input_sensitivity(
    u.d(x),
    interval=50,
)

# Empirical NTK spectrum: diagnoses spectral bias.
cb_ntk = jno.callbacks.ntk_spectrum(
    u.grad(u_net),
    n_points=32,
    top_k=5,
    interval=200,
)

# Hessian eigenspectrum: top-k eigenvalues + sharpness via Lanczos w/ HVPs.
cb_hess = jno.callbacks.hessian_spectrum(
    k=5,
    n_iter=15,
    interval=400,
)

# ── Checkpoint callback ────────────────────────────────────────────────────────
# Saves to disk every 500 epochs; uploads a versioned artifact to W&B.
# Artifact metadata includes epoch, total_loss, individual_losses, checkpoint_dir.
cb_ckpt = jno.callbacks.checkpoint(
    directory=f"{dire}/checkpoints",
    save_interval_epochs=500,
    max_to_keep=3,
    best_fn=lambda m: m["total_loss"],
)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, bc.mse])
crux.solve(
    2_000,
    callbacks=[cb_norms, cb_cos, cb_align, cb_landscape, cb_residual, cb_saliency, cb_ntk, cb_hess, cb_ckpt],
)
cb_ckpt.close()

# ── Inspect callback results locally ──────────────────────────────────────────
norms_result = cb_norms.result
print("\n── Gradient norms ────────────────────────────────────────────────────")
print(f"  sampled epochs : {norms_result['epochs']}")
print(f"  norms shape    : {norms_result['norms'].shape}   (samples × constraints)")
print(f"  final norms    : {norms_result['norms'][-1]}")

cos_result = cb_cos.result
print("\n── Cosine similarity (final sample) ──────────────────────────────────")
print(f"  matrix:\n{cos_result['cos_sim_matrix'][-1]}")

align_result = cb_align.result
print("\n── Gradient alignment ────────────────────────────────────────────────")
print(f"  values : {align_result['alignment']}")
print("  (1.0 = perfect alignment, 0.0 = destructive interference)")

land_result = cb_landscape.result
print("\n── Loss landscape ────────────────────────────────────────────────────")
print(f"  grid shape : {land_result['landscapes'].shape}")
print(f"  final min  : {land_result['landscapes'][-1].min():.4e}")
print(f"  final max  : {land_result['landscapes'][-1].max():.4e}")

residual_result = cb_residual.result
print("\n── Residual statistics ───────────────────────────────────────────────")
print(f"  means shape : {residual_result['means'].shape}   (samples × constraints)")
print(f"  final mean  : {residual_result['means'][-1]}")
print(f"  final max   : {residual_result['maxes'][-1]}")
print(f"  final p99   : {residual_result['p99'][-1]}")

saliency_result = cb_saliency.result
print("\n── Input sensitivity ─────────────────────────────────────────────────")
print(f"  values shape : {saliency_result['values'].shape}")
final_abs = jnp.abs(saliency_result["values"][-1])
print(f"  final mean|∂u/∂x| : {float(final_abs.mean()):.4e}")
print(f"  final max |∂u/∂x| : {float(final_abs.max()):.4e}")

ntk_result = cb_ntk.result
print("\n── NTK spectrum ──────────────────────────────────────────────────────")
print(f"  top-{ntk_result['eigvals_topk'].shape[1]} eigvals (final): {ntk_result['eigvals_topk'][-1]}")
print(f"  λ_max (final)        : {ntk_result['lambda_max'][-1]:.4e}")
print(f"  condition (final)    : {ntk_result['condition_number'][-1]:.4e}")

hess_result = cb_hess.result
print("\n── Hessian eigenspectrum ─────────────────────────────────────────────")
print(f"  top-{hess_result['eigvals'].shape[1]} eigvals (final): {hess_result['eigvals'][-1]}")
print(f"  sharpness (final)    : {hess_result['sharpness'][-1]:.4e}")

# ── Validate solution ──────────────────────────────────────────────────────────
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jnp.linalg.norm(_u - _u_exact) / (jnp.linalg.norm(_u_exact) + 1e-8))
print(f"\nRelative L² error: {rel_l2:.3e}")
