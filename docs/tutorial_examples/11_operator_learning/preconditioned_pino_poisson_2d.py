"""Preconditioned physics-informed FNO for Poisson in jNO (Wen, Mishra, Zeinhofer, arXiv:2609.36216, Poisson part).

    -Laplace u = rho on (0,1)^2, u = 0 on the boundary, Q1 finite elements on a (N+1)^2 grid.
    loss(u) = 1/2 || P (A u - M rho) ||^2,   P = one multigrid V(2,2) cycle ~ A^-1   (jno.precond.gmg)

No labels: the exact solution is only used to score the result.

Measured (2026-11-05, 16000 updates, 1024 fixed forcings, 1 seed, 37 min on an RTX 3070): held-out relative L2 error
6.13 % on 256 unseen forcings; the paper reports 5.1 +- 0.8 % (preconditioned) and 4.7 % (supervised) for the same setup.
One forcing pool, not a stream: jno.core trains on the domain it is given, and batched domains copy the mesh per sample.
"""

import os

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

import jno

N, K = 64, 10  # cells per side, sine-series modes of the forcing
POOL, TEST = int(os.environ.get("POOL", 1024)), 256  # forcings drawn up front (no solutions), held-out forcings
EPOCHS, BATCH, LR = int(os.environ.get("EPOCHS", 4000)), 32, 3e-4
WIDTH, MODES, LAYERS = 64, 16, 5
KEY = jax.random.PRNGKey(0)

# ── Finite elements: stiffness A, mass M, and a V-cycle built from A's own stencil ────────────────
base = jno.shape.rect(0, 0, 1, 1).structured(n=N).quad().domain(compute_mesh_connectivity=True)
U, V = base.fem_symbols()
xi, yi, _ = base.variable("interior", split=True)
xb, yb, _ = base.variable("boundary", split=True)
ui, vi = U.bind(x=xi, y=yi), V.bind(x=xi, y=yi)
fem_A = jno.fem([ui.x * vi.x + ui.y * vi.y, U(xb, yb) - 0.0])  # weak form of -Laplace, u = 0 on the boundary
fem_M = jno.fem([ui * vi, U(xb, yb) - 0.0])  # mass matrix: the load vector of rho is M rho
A, M = LinearOperator(fem_A.operator[0]), LinearOperator(fem_M.operator[0])
P = materialize_precond(jno.precond.gmg(), PrecondContext(A, fem_A))

pts = np.asarray(fem_A.points)  # the FEM's node order -> positions on the FNO's (N+1)x(N+1) grid
gx, gy = np.rint(pts[:, 0] * N).astype(int), np.rint(pts[:, 1] * N).astype(int)
inner = jnp.asarray((gx > 0) & (gx < N) & (gy > 0) & (gy < N), jnp.float32)
to_nodes = lambda field: field.reshape(N + 1, N + 1)[gy, gx]


# ── Forcings (and, for scoring only, their exact solutions): random truncated sine series ────────
def sine_series(n, seed):
    a = np.random.default_rng(seed).uniform(-1, 1, (n, K, K)).astype(np.float32)
    s = np.arange(1, K + 1)
    w = np.sqrt(s[:, None] ** 2 + s[None, :] ** 2).astype(np.float32)
    g = np.linspace(0, 1, N + 1, dtype=np.float32)
    sx = np.sin(np.pi * s[:, None] * g[None, :])  # (K, N+1)
    synth = lambda c: np.einsum("nij,ix,jy->nyx", c, sx, sx)[..., None]  # (n, y, x, 1)
    return synth(a * w * np.pi / K**2), synth(a / w / (np.pi * K**2))


def batched(rho, u=None, interior=True):
    dom = rho.shape[0] * base
    if interior:
        dom.variable("interior")
    dom.variable("_rho", rho[:, None, None])
    if u is not None:
        dom.variable("_u", u[:, None, None])
    return dom


rho_train, _ = sine_series(POOL, seed=1)
train = batched(rho_train)

# ── Model: FNO, boundary nodes hard-set to zero ───────────────────────────────────────────────────
net = jno.nn(
    foundax.fno2d(
        in_features=1, hidden_channels=WIDTH, n_modes=MODES, d_vars=1, n_layers=LAYERS,
        d_model=(N + 1, N + 1), norm=None, key=KEY,
    )
)
net.optimizer(optax.adam(optax.cosine_decay_schedule(LR, EPOCHS, alpha=0.1)))
rho = train.variable("_rho")
u_h = jno.fn(lambda g: to_nodes(g[..., 0]) * inner, [net(rho)])  # (nodes,) in the FEM's order

# ── The preconditioned least-squares loss ─────────────────────────────────────────────────────────
precond_residual = jno.fn(lambda u, f: P(A.mv(u) - M.mv(to_nodes(f[..., 0]))), [u_h, rho], name="P(Au - Mrho)")

crux = jno.core([precond_residual.mse])
crux.solve(epochs=EPOCHS, batchsize=BATCH)

# ── Held-out forcings, scored against the exact solutions (built after training: domains multiply) ─
rho_test, u_test = sine_series(TEST, seed=2)
chunks = range(0, TEST, BATCH)  # the whole test set at once does not fit next to the training state
predict = lambda dom: np.array(crux.eval([net(dom.variable("_rho"))], domain=dom))
pred = np.concatenate([predict(batched(rho_test[i : i + BATCH], interior=False)) for i in chunks])
pred = pred.reshape(TEST, N + 1, N + 1)
pred[:, [0, -1], :] = pred[:, :, [0, -1]] = 0.0  # boundary nodes are hard zeros
ref = u_test[..., 0]
rel = np.linalg.norm((pred - ref).reshape(TEST, -1), axis=1) / np.linalg.norm(ref.reshape(TEST, -1), axis=1)
print(f"held-out relative L2 error: {rel.mean() * 100:.2f} % (paper: 5.1 +- 0.8 % at K=10)")
