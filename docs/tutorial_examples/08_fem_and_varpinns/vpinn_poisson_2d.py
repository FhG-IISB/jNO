"""Variational PINN (VPINN): the trial is a **neural network**, the test functions are the **FE
basis**. ``jno.fem`` detects the network trial written into the weak form and test-projects it onto
the FE test space -- a trainable residual loss, trained through ``jno.core``. No ``init_fem``, no
``weak.assemble``: a VPINN is authored exactly like any other ``jno.fem`` problem.

Poisson  -Δu = f  on the unit square, exact ``u = x(1-x)y(1-y)``  (so ``f = 2[x(1-x)+y(1-y)]``).
The network trial uses a hard-BC ansatz  ``u = net(x,y) · x(1-x)y(1-y)``  that vanishes on the
boundary; the Dirichlet condition ``u(boundary) - 0`` tells ``jno.fem`` which test functions vanish
on the boundary, so their (irreducible ``∂u/∂n``-flux) residual is masked -- without it the loss
minimum is not the PDE solution and training would diverge from it.
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

jax.config.update("jax_enable_x64", True)  # feax assembly is float64

# ---- domain, network trial, weak form -------------------------------------------------------
dom = jno.domain(constructor=jno.domain.rect(mesh_size=0.07))
u, phi = dom.fem_symbols()
xi, yi, _ = dom.variable("interior", split=True)
xb, yb, _ = dom.variable("boundary", split=True)

net = jnn.nn.wrap(foundax.mlp(2, hidden_dims=32, num_layers=3, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
ansatz = xi * (1 - xi) * yi * (1 - yi)  # hard-BC ansatz: vanishes on the [0,1]^2 boundary
u_net = net(xi, yi) * ansatz  # the network trial
vi = phi.bind(x=xi, y=yi)  # FE test function
f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))  # -Δ[x(1-x)y(1-y)]

# weak form with the NETWORK trial + the Dirichlet declaration (masks the boundary test functions)
pde = jno.fem([jnn.grad(u_net, xi) * jnn.grad(vi, xi) + jnn.grad(u_net, yi) * jnn.grad(vi, yi) - f * vi, u(xb, yb) - 0.0])
print(f"\nVPINN Poisson 2D: {type(pde).__name__} (test-projected residual); dofs={dom.mesh.points.shape[0]}")

# ---- train the network through jno.core (minimise the test-projected residual) --------------
net.optimizer(optax.adam(1e-2))
crux = jno.core([pde.mse], domain=dom)
crux.solve(2500)

# ---- verify the trained network against the analytic solution (on a fresh grid) ------------
test_dom = jno.domain(constructor=jno.domain.rect(mesh_size=0.04))
xt, yt, _ = test_dom.variable("interior", split=True)
exact_expr = xt * (1 - xt) * yt * (1 - yt)
pred = np.asarray(crux.eval([net(xt, yt) * exact_expr], domain=test_dom)).reshape(-1)
exact = np.asarray(crux.eval([exact_expr], domain=test_dom)).reshape(-1)
rel = float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
print(f"  trained VPINN vs analytic x(1-x)y(1-y):  rel-L2 = {rel:.3e}")

# ---- plot the learned field and the error (the actual computed prediction) ------------------
pts = np.asarray(test_dom.mesh.points)[:, :2]
tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))
tp0 = ax[0].tripcolor(tri, pred, cmap="viridis", shading="gouraud")
ax[0].set_title("VPINN solution  u = net·x(1-x)y(1-y)")
tp1 = ax[1].tripcolor(tri, np.abs(pred - exact), cmap="magma", shading="gouraud")
ax[1].set_title("|VPINN − analytic|")
for a, tp in zip(ax, (tp0, tp1)):
    fig.colorbar(tp, ax=a, shrink=0.85)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "vpinn_poisson_2d.png", dpi=90)

assert rel < 1e-2, f"VPINN did not solve Poisson: rel-L2={rel:.3e}"
