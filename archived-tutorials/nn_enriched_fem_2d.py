"""FE-basis enrichment: correct a neural-network prior with a coarse finite-element solve.

A network prior ``u_NN`` (fast, but only approximate) is *certified and sharpened* by a finite-element
correction ``u_h`` on a **coarse** mesh: seek ``u ≈ u_NN + u_h`` where the correction solves

    a(u_h, v) = (f, v) − a(u_NN, v)      ∀ v ∈ V_h ,

i.e. ``u_h`` is the Galerkin projection of the prior's error ``e = u_exact − u_NN``. Enrichment beats
standard FEM on the same mesh **iff the prior captures sub-grid content the coarse space cannot** —
so the prior's gradient must enter the weak form *continuously at the quadrature points*, which is the
capability ``jnn.grad(frozen_net, x)`` adds to the assembler (a P1-nodal ``FrozenField`` would carry
no sub-grid information and give exactly standard FEM back).

Method: **NN-enriched finite elements** — Barucq, Faucher, Pham & Tonnoir, "Enriching continuous
Lagrange finite element approximation spaces using neural networks", 2025 (arXiv:2502.04947). The
prior uses **random Fourier features** (Tancik et al., NeurIPS 2020, arXiv:2006.10739) to defeat the
spectral bias that stops a plain MLP from representing high-frequency content.

Test problem: Poisson ``-Δu = f`` on ``[0,1]²`` with a two-scale exact solution
``u = sin(πx)sin(πy) + 0.3 sin(5πx)sin(5πy)`` — the high-frequency term is what a coarse mesh
misses and the network prior supplies.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path  # noqa: E402

import equinox as eqx  # noqa: E402
import foundax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

jax.config.update("jax_enable_x64", True)  # the FEM assembly runs in float64

K = 5.0  # high-frequency wavenumber the coarse mesh cannot resolve


class FourierMLP(eqx.Module):
    """Random Fourier Features → MLP (Tancik et al. 2020): the feature map ``x ↦ [sin(2πBx),
    cos(2πBx)]`` lets the network represent high frequencies a plain tanh MLP cannot (spectral bias)."""

    B: jax.Array
    mlp: object

    def __init__(self, in_dim, n_freq, sigma, hidden, layers, key):
        kB, km = jax.random.split(key)
        self.B = sigma * jax.random.normal(kB, (n_freq, in_dim))
        self.mlp = foundax.mlp(2 * n_freq, hidden_dims=hidden, num_layers=layers, activation=jax.nn.tanh, key=km)

    def __call__(self, *coords):
        x = jnp.concatenate([jnp.atleast_2d(c) for c in coords], axis=-1)
        proj = 2 * jnp.pi * (x @ self.B.T)
        return self.mlp(jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1))


# ---- 1. train the Fourier-feature prior to the high-frequency part, on a dense grid --------
# Sobolev fit: the enrichment integrates ∇u_NN, so we match the network's VALUE *and* GRADIENT to
# the target — matching values alone leaves the gradient (amplified ~Kπ×) too inaccurate to help.
d_fit = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.03)
xf, yf, _ = d_fit.variable("interior", split=True)
net = jno.nn.wrap(FourierMLP(in_dim=2, n_freq=96, sigma=4.0, hidden=64, layers=3, key=jax.random.PRNGKey(0)))
net.dtype(jnp.float64)
tgt = 0.3 * jnn.sin(K * np.pi * xf) * jnn.sin(K * np.pi * yf)  # the sub-grid content
tgt_x = 0.3 * K * np.pi * jnn.cos(K * np.pi * xf) * jnn.sin(K * np.pi * yf)
tgt_y = 0.3 * K * np.pi * jnn.sin(K * np.pi * xf) * jnn.cos(K * np.pi * yf)
net.optimizer(optax.adam(optax.exponential_decay(2e-3, 3500, 0.5, end_value=1e-5)))
jno.core([(net(xf, yf) - tgt).mse, (jnn.grad(net(xf, yf), xf) - tgt_x).mse, (jnn.grad(net(xf, yf), yf) - tgt_y).mse]).solve(
    7000
)

# ---- 2. coarse Poisson: standard FEM vs NN-enriched FEM on the SAME coarse mesh -------------
d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.16)  # coarse: under-resolves the high-freq term
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
f = 2 * np.pi**2 * jnn.sin(np.pi * xi) * jnn.sin(np.pi * yi) + 0.3 * 2 * (K * np.pi) ** 2 * jnn.sin(
    K * np.pi * xi
) * jnn.sin(K * np.pi * yi)

pts = np.asarray(d.mesh.points)[:, :2]
exact = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1]) + 0.3 * np.sin(K * np.pi * pts[:, 0]) * np.sin(
    K * np.pi * pts[:, 1]
)
rel = lambda v: float(np.linalg.norm(v - exact) / np.linalg.norm(exact))  # noqa: E731

u_std = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=6).solve()).reshape(-1)

# enriched: the frozen prior's CONTINUOUS gradient enters the weak form → RHS (Part-1 capability)
fnet = net.freeze()
gx = ui.x + jnn.grad(fnet(xi, yi), xi)  # ∇(u_NN + u_h)·x̂
gy = ui.y + jnn.grad(fnet(xi, yi), yi)
u_h = np.asarray(jno.fem([gx * vi.x + gy * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=6).solve()).reshape(-1)
prior_nodal = np.asarray(net.module(jnp.asarray(pts[:, 0:1]), jnp.asarray(pts[:, 1:2]))).reshape(
    -1
)  # trained prior at DOFs
u_enr = prior_nodal + u_h

print(f"\nNN-enriched FEM (two-scale Poisson, coarse mesh, dofs={d.mesh.points.shape[0]}):")
print(f"  standard coarse FEM     rel-L2 = {rel(u_std):.3e}")
print(f"  NN-enriched coarse FEM  rel-L2 = {rel(u_enr):.3e}   ({rel(u_std) / rel(u_enr):.1f}x better)")

# ---- plot: standard vs enriched pointwise error on the same coarse mesh ---------------------
import matplotlib.tri as mtri  # noqa: E402

tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
for a, err, ttl in (
    (ax[0], np.abs(u_std - exact), "standard coarse FEM"),
    (ax[1], np.abs(u_enr - exact), "NN-enriched (same mesh)"),
):
    tp = a.tripcolor(tri, err, cmap="magma", shading="gouraud")
    fig.colorbar(tp, ax=a, shrink=0.85)
    a.set_title(f"|error| — {ttl}")
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "nn_enriched_fem_2d.png", dpi=90)

assert rel(u_enr) < 0.35 * rel(u_std), f"enrichment did not beat standard FEM: {rel(u_enr):.3e} vs {rel(u_std):.3e}"
assert rel(u_enr) < 7e-3, f"enriched solution not accurate: rel-L2={rel(u_enr):.3e}"
