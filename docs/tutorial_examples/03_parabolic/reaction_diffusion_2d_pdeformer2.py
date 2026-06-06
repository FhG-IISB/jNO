"""03 — 2-D reaction-diffusion equation with a PDEformer-2 backbone

Problem
-------
    ∂u/∂t = α ∇²u − λ u,   (x,y) ∈ [0,1]²,  t ∈ [0, 1]
    u = 0 on ∂Ω  (homogeneous Dirichlet BCs)
    u(x,y,0) = sin(πx) sin(πy)

Analytical solution
-------------------
    u(x,y,t) = exp(−(2 α π² + λ) t) · sin(πx) · sin(πy)

The reaction-diffusion family is one of the primary 2-D PDE benchmarks in
the PDEformer-2 paper (Shi et al., *PDEformer-2: A Foundation Model for
Two-Dimensional PDEs*, arXiv:2502.14844, 2025, §4).  Here we use the
linear-damping case so the analytical solution is in closed form, which
makes verification trivial.

Notes on the residual
---------------------
PDEformer-2 was trained on a fixed operator vocabulary
{add, mul, neg, square, dt, dx, dy, sin, cos, exp10, log10}.  The PDE
residual below uses only ``dt``, ``∇²`` (= dx·dx + dy·dy), and a linear
``λ u`` reaction term — all in vocabulary.

The analytical solution ``u_exact`` does contain ``exp`` but is **not**
walked through the PDEformer graph builder.  It is only used by
``crux.eval(...)`` for verification, which uses jNO's regular trace
evaluator.

Boundary conditions — why soft, not ansatz
------------------------------------------
The standard PINN trick of writing
``u = net(t, x, y) * x(1-x) y(1-y)`` to enforce Dirichlet BCs is
incompatible with a foundation-model backbone:

1. *Vocabulary mismatch.* Expanding ``∇²(NN · ansatz)`` via the product
   rule produces ``∇²ansatz`` and ``∇ansatz`` terms — i.e. raw
   ``Variable(x), Variable(y)`` nodes that PDEformer-2's operator
   vocabulary {add, mul, neg, square, dt, dx, dy, sin, cos, exp10,
   log10} does not contain.

2. *Pre-training distribution mismatch.* Even if (1) were patched, the
   trainer would ask ``NN · ansatz`` to satisfy the PDE — forcing the
   foundation model to learn ``NN ≈ u / ansatz`` instead of ``u``.  The
   pre-trained weights were optimised to predict ``u`` directly, so they
   become a poor warm start.

Soft BCs are the natural fit:

* the PDE graph the encoder sees is the canonical equation, exactly the
  form it was pre-trained on;
* the IC node values (``sin(πx)sin(πy)`` here) already vanish on ∂Ω, so
  the encoder has indirect information about where the boundary is;
* the BC residual ``ub.mse`` is trained as a normal soft loss but is
  excluded from the DAG by the bridge (it carries the ``boundary`` tag).

For genuinely stiff BC problems, a *residual-style* ansatz
``u(x,y,t) = u_BC(x,y) + NN(t,x,y) · χ(x,y)`` (with ``u_BC`` a known
function meeting the BCs and ``χ`` vanishing on ∂Ω) keeps ``NN`` in the
same units as ``u`` and preserves the foundation model's prior — at the
cost of a more complex residual.  For the canonical reaction-diffusion
problem here, plain soft BCs are simpler and sufficient.
"""

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi
sin = jno.np.sin

α = 0.05  # diffusivity
λ = 1.0  # reaction rate
T_end = 1.0
N_t = 5

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, N_t),
    compute_mesh_connectivity=False,
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")
xb, yb, tb = domain.variable("boundary")
domain.summary()

# ── Analytical solution (for verification only) ───────────────────────────────
u_exact = jno.np.exp(-(2 * α * π**2 + λ) * t) * sin(π * x) * sin(π * y)

# ── Network: PDEformer-2 small ────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.pdeformer2.small(
        num_encoder_layers=3,
        embed_dim=64,
        ffn_embed_dim=128,
        num_heads=4,
        inr_dim_hidden=64,
        inr_num_layers=4,
        hyper_num_layers=2,
        scalar_num_layers=2,
    )
)
net.optimizer(
    optax.adam(
        optax.warmup_cosine_decay_schedule(
            init_value=0,
            peak_value=1e-3,
            warmup_steps=200,
            decay_steps=10_000 - 200,
            end_value=1e-5,
        )
    )
)
net.summary()

# ── Constraints ───────────────────────────────────────────────────────────────
u = net(t, x, y)
u0 = net(t0, x0, y0)
ub = net(tb, xb, yb)

pde = u.d(t) - α * jno.np.laplacian(u, [x, y]) + λ * u
ini = u0 - sin(π * x0) * sin(π * y0)
bc = ub  # u = 0 on ∂Ω (soft)

# ── Solve — jno.core auto-detects PDEformer-2 and builds the DAG ──────────────
crux = jno.core([pde.mse, ini.mse, bc.mse], domain).print_shapes()
history = crux.solve(10_000)

# ── Evaluation ────────────────────────────────────────────────────────────────
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"03_parabolic/reaction_diffusion_2d_pdeformer2.py | epochs=10000 | rel_L2={rel_l2:.6e}\n")

print(f"Reaction-diffusion 2D (PDEformer-2 backbone): relative L2 = {rel_l2:.4e}")
