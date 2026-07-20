"""Deep Ritz / Deep Energy Method: solve a PDE by **minimising an energy functional** with a network.

Where a VPINN (`vpinn_poisson_2d.py`) minimises the *weak residual* test-projected onto the FE basis,
the Deep Ritz method minimises the **energy functional** directly — no test functions, no assembly.
For Poisson ``-Δu = f`` with ``u = 0`` on ``∂Ω`` the governing energy is

    J[u] = ∫_Ω ( ½ |∇u|² − f·u ) dx ,

whose minimiser over ``H¹₀`` is the weak solution. The neural network *is* the trial ``u``; we integrate
the energy density over the mesh with ``expr.integrate()`` and hand the single scalar ``J`` to
``jno.core`` as the loss. Lower derivative order and no linear solve, at the price of a
non-convex optimisation.

Method: **Deep Ritz** — E & Yu, *Commun. Math. Stat.* 6:1 (2018), §2 (arXiv:1710.00211); the solid-
mechanics form is the **Deep Energy Method**, Samaniego et al., *CMAME* 362 (2020) and Nguyen-Thanh,
Zhuang & Rabczuk, *Eur. J. Mech. A/Solids* 80 (2020).

We solve ``-Δu = f`` on the unit square with exact ``u = x(1-x)y(1-y)`` (so ``f = 2[x(1-x)+y(1-y)]``).

Quadrature consistency (why we use Gauss): the default ``.integrate()`` uses the P1 nodal-volume
(vertex) rule — the energy is sampled *only at mesh nodes*. A network expressive enough to develop
structure *between* nodes can then drive the discrete energy *below* the true minimum (a variational
crime: the quadrature is no longer a faithful estimate) and the solution degrades even as the
reported loss keeps dropping. ``.integrate(quadrature="gauss")`` samples the per-element Gauss points
instead, which is far harder to alias — it lets a *capable* network minimise the energy safely and
converge to ``rel-L2 ≈ 1e-4`` here. (Swap in the default nodal rule with this same network and the
energy sinks past the true minimum while the error blows up.)
"""

import os

os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path  # noqa: E402

import foundax  # noqa: E402
import jax  # noqa: E402  (jax.nn / jax.random for the network)
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

jax.config.update("jax_enable_x64", True)  # the mesh integral accumulates in float64

# ---- domain, network trial, energy functional ----------------------------------------------
dom = jno.Shape.rect(0, 0, 1, 1, size=0.05).domain()
xi, yi, _ = dom.variable("interior", split=True)  # .integrate() re-evaluates at the quadrature points

# a capable network — the Gauss quadrature below keeps its energy honest (see the consistency note)
net = jnn.nn.wrap(foundax.mlp(2, hidden_dims=48, num_layers=4, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
ansatz = xi * (1 - xi) * yi * (1 - yi)  # hard-BC ansatz: vanishes on the [0,1]^2 boundary → u ∈ H¹₀
u = net(xi, yi) * ansatz  # the network trial
ux, uy = jnn.grad(u, xi), jnn.grad(u, yi)
f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))  # -Δ[x(1-x)y(1-y)]

# energy functional J[u] = ∫ (½|∇u|² − f u) dx, integrated with element GAUSS quadrature
# (alias-resistant — a nodal rule would let this network cheat the integral; see the docstring)
energy = (0.5 * (ux**2 + uy**2) - f * u).integrate(quadrature="gauss")

# ---- train the network by minimising J[u] through jno.core ---------------------------------
net.optimizer(optax.adam(3e-3))
crux = jno.core([energy], domain=dom)  # the *signed* energy is the loss (it converges to J_min < 0)
crux.solve(4000)

# ---- verify the trained network against the analytic solution (on a fresh grid) ------------
test_dom = jno.Shape.rect(0, 0, 1, 1, size=0.035).domain()
xt, yt, _ = test_dom.variable("interior", split=True)
exact_expr = xt * (1 - xt) * yt * (1 - yt)
pred = np.asarray(crux.eval([net(xt, yt) * exact_expr], domain=test_dom)).reshape(-1)
exact = np.asarray(crux.eval([exact_expr], domain=test_dom)).reshape(-1)
rel = float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
print(f"\nDeep Ritz Poisson 2D: minimised energy functional; dofs={dom.mesh.points.shape[0]}")
print(f"  trained Deep-Ritz vs analytic x(1-x)y(1-y):  rel-L2 = {rel:.3e}")

# ---- plot the learned field and the error (the actual computed prediction) -----------------
pts = np.asarray(test_dom.mesh.points)[:, :2]
tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
tp0 = ax[0].tripcolor(tri, pred, cmap="viridis", shading="gouraud")
ax[0].set_title("Deep Ritz solution  u = net·x(1-x)y(1-y)")
tp1 = ax[1].tripcolor(tri, np.abs(pred - exact), cmap="magma", shading="gouraud")
ax[1].set_title("|Deep Ritz − analytic|")
for a, tp in zip(ax, (tp0, tp1)):
    fig.colorbar(tp, ax=a, shrink=0.85)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "deep_ritz_poisson_2d.png", dpi=90)

assert rel < 1e-3, f"Deep Ritz did not solve Poisson: rel-L2={rel:.3e}"
