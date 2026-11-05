"""Preconditioned physics-informed FNO for Poisson (Wen, Mishra & Zeinhofer, arXiv:2609.36216, Poisson part).

    -Laplace u = rho on (0,1)^2, u = 0 on the boundary, Q1 finite elements on a (N+1)^2 grid.
    loss(u) = 1/2 || P (A u - b(rho)) ||^2,   P = one multigrid V(2,2) cycle ~ A^-1

The FNO's output, declared a field on the mesh (`.field.bind`), is written into the weak form where the
unknown goes, like the FE trial; jno.fem returns the FE residual of that output, per sample, and
`.solve(precond=...)` preconditions it. No labels: the exact solutions are
only used to score the result.
"""

import os

import foundax
import jax
import numpy as np
import optax

import jno

N, K = 64, 10  # cells per side, sine-series modes of the forcing
POOL, TEST = int(os.environ.get("POOL", 1024)), 256  # forcings drawn up front (no solutions), held-out forcings
EPOCHS, BATCH, LR = int(os.environ.get("EPOCHS", 4000)), 32, 3e-4


def sine_series(n, seed):
    """Random forcings rho on the node grid, indexed [ix, iy], and (for scoring only) their exact solutions."""
    a = np.random.default_rng(seed).uniform(-1, 1, (n, K, K)).astype(np.float32)
    s = np.arange(1, K + 1)
    w = np.hypot(s[:, None], s[None, :]).astype(np.float32)
    sx = np.sin(np.pi * s[:, None] * np.linspace(0, 1, N + 1, dtype=np.float32))  # (K, N+1)
    synth = lambda c: np.einsum("nij,ix,jy->nxy", c, sx, sx)[..., None]  # noqa: E731   (n, x, y, 1)
    return synth(a * w * np.pi / K**2), synth(a / w / (np.pi * K**2))


def forcings(rho):
    dom = rho.shape[0] * jno.shape.rect(0, 0, 1, 1).structured(n=N).quad().domain(compute_mesh_connectivity=True)
    dom.variable("_rho", rho[:, None, None])
    return dom


train = forcings(sine_series(POOL, seed=1)[0])
U, V = train.fem_symbols()
xi, yi, _ = train.variable("interior", split=True)
xb, yb, _ = train.variable("boundary", split=True)
vi, rho = V.bind(x=xi, y=yi), train.variable("_rho")

net = jno.nn(
    foundax.fno2d(
        in_features=1,
        hidden_channels=64,
        n_modes=16,
        d_vars=1,
        n_layers=5,
        d_model=(N + 1, N + 1),
        norm=None,
        key=jax.random.PRNGKey(0),
    )
)
net.optimizer(optax.adam(optax.cosine_decay_schedule(LR, EPOCHS, alpha=0.1)))
u = net(rho).field.bind(x=xi, y=yi)  # the FNO's output is a field on the mesh, not a function of (x, y)

pde = jno.fem([u.x * vi.x + u.y * vi.y - rho * vi, U(xb, yb) - 0.0])
crux = jno.core([pde.solve(precond=jno.precond.gmg()).mse])
crux.solve(epochs=EPOCHS, batchsize=BATCH)

# Held-out forcings, scored against the exact solutions at the nodes, in chunks of BATCH samples.
rho_test, u_test = sine_series(TEST, seed=2)
pred = np.concatenate(
    [np.asarray(crux.eval([pde.field], domain=forcings(rho_test[i : i + BATCH]))) for i in range(0, TEST, BATCH)]
).reshape(TEST, -1)
pts = np.asarray(pde.fem.points)
ref = u_test[:, *np.rint(pts * N).astype(int).T, 0]  # exact solution at the FE nodes
rel = np.linalg.norm(pred - ref, axis=1) / np.linalg.norm(ref, axis=1)
print(f"held-out relative L2 error: {rel.mean() * 100:.2f} % (paper: 5.1 +- 0.8 % at K=10)")
