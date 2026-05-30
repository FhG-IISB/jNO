"""03 — 2-D heat equation with a PDEformer-2 backbone

Problem
-------
    ∂u/∂t = α ∇²u,   (x,y) ∈ [0,1]²,  t ∈ [0, 0.5]
    u = 0 on ∂Ω  (homogeneous Dirichlet BCs)
    u(x,y,0) = sin(πx) sin(πy)

Analytical solution
-------------------
    u(x,y,t) = exp(−2απ²t) sin(πx) sin(πy)

This is the same physics as ``heat_2d.py`` but with PDEformer-2 as the
neural backbone instead of a hand-rolled DeepONet.  The program is identical
in shape to any other jNO PINN — there is **no PDEformer-specific code**.

How it works
------------
The PDEformer-2 graph encoder expects the *canonical* PDE form expressed in
its operator vocabulary {add, mul, neg, square, dt, dx, dy, sin, cos,
exp10, log10}.  ``jno.core`` detects the backbone via
``isinstance(model.module, jax_pdeformer2.PDEformer)``, walks the symbolic
loss tree, and auto-builds the DAG.  Boundary terms (anything not tagged
``interior`` or ``initial``) are passed through to the trainer but are not
folded into the PDEformer-2 graph.

Why soft BCs (and not the usual ansatz)
---------------------------------------
The standard PINN trick of enforcing Dirichlet BCs by writing
``u = net(t, x, y) * x*(1-x)*y*(1-y)`` is **incompatible with a
foundation-model backbone**, for two reasons:

1. *Vocabulary mismatch.* PDEformer-2 was trained on PDE graphs built from
   a fixed operator set {add, mul, neg, square, dt, dx, dy, sin, cos,
   exp10, log10} plus uf / coef / ic nodes — there is no node type for a
   raw spatial coordinate.  Expanding ``∇²(NN · ansatz)`` via the product
   rule introduces ``∇²ansatz`` and ``∇ansatz`` terms, i.e. raw
   ``Variable(x), Variable(y)`` nodes that the bridge cannot translate.

2. *Pre-training distribution mismatch.* Even if (1) could be patched
   (e.g. by hiding the ansatz inside the wrapper), the trainer would be
   asking ``NN · ansatz`` to satisfy the PDE — so the foundation model
   would have to learn ``NN ≈ u / ansatz``, not ``u``.  But every
   pre-training sample taught it to predict ``u`` directly, so its
   pre-trained weights become a poor warm start and most of the
   foundation-model benefit is lost.

The natural fit for a foundation backbone is therefore **soft BCs**:

* the PDE graph the encoder sees is the canonical equation, exactly the
  form it was pre-trained on;
* the IC node values for ``sin(πx)sin(πy)`` already vanish on ∂Ω, so the
  encoder has indirect information about where the boundary is;
* the BC residual ``ub.mse`` is trained as a normal soft loss but is
  excluded from the DAG by the bridge.

For genuinely stiff BC problems, a residual-style ansatz of the form
``u(x,y,t) = u_BC(x,y) + NN(t,x,y) · ansatz`` (with ``u_BC`` a known
function satisfying the BC) keeps ``NN`` in the same units as ``u`` and
preserves the foundation model's prior — but for the canonical examples
PDEformer-2 was trained on, plain soft BCs are simpler and sufficient.
"""

from pathlib import Path

import foundax
import jax
import optax

import jno

π = jno.np.pi
α = 0.1
T_end = 0.5
N_t = 4

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

# ── Analytical solution (used for evaluation only) ───────────────────────────
u_exact = jno.np.exp(-2 * α * π**2 * t) * jno.np.sin(π * x) * jno.np.sin(π * y)

# ── Network: PDEformer-2 small (CPU-friendly tiny variant) ────────────────────
net = jno.nn.wrap(
    foundax.pdeformer2.small(
        num_encoder_layers=2,
        embed_dim=64,
        ffn_embed_dim=128,
        num_heads=4,
        inr_dim_hidden=64,
        inr_num_layers=3,
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

# ── Constraints (no ansatz; BC is a soft loss) ───────────────────────────────
u = net(t, x, y)  # interior
u0 = net(t0, x0, y0)  # IC
ub = net(tb, xb, yb)  # boundary

pde = jno.np.grad(u, t) - α * jno.np.laplacian(u, [x, y])
ini = u0 - jno.np.sin(π * x0) * jno.np.sin(π * y0)
bc = ub  # u = 0 on ∂Ω

# ── Solve — jno.core detects PDEformer-2 and auto-builds the graph ───────────
crux = jno.core([pde.mse, ini.mse, bc.mse], domain).print_shapes()
history = crux.solve(10_000)

# ── Evaluation ───────────────────────────────────────────────────────────────
_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(f"03_parabolic/heat_2d_pdeformer2.py | epochs=10000 | rel_L2={rel_l2:.6e}\n")

print(f"Heat 2D (PDEformer-2 backbone): relative L2 = {rel_l2:.4e}")
