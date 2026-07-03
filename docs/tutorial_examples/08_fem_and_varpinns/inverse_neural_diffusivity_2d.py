"""Inverse problem: recover a hidden diffusivity k(x) with a NEURAL coefficient.

    Forward:  -div(k(x) grad u) = f,   u = 0 on the boundary,   with unknown k(x) > 0.

Same identification problem as the nodal-field tutorial (``inverse_diffusivity_field.py``), but
the unknown is parametrised by a coordinate MLP instead of one value per mesh node:
``k(x, y) = 1 + net(x, y)`` written directly into the weak form. The network is a trainable
*coefficient* on an assembled FE system — the kernel re-evaluates it at the quadrature points on
every re-assembly, and ``crux`` trains the weights through the differentiable ``fem.solve()``.

Why prefer a net over a nodal field? The parametrisation is mesh-independent (remeshing never
touches the weights), smooth by architecture (no explicit H1 prior needed here), and it extends
to constitutive laws ``net(u)`` / ``net(∇u)``. This is the unsupervised coefficient-recovery
setting of NN-EUCLID (Flaschel, Kumar, De Lorenzis, J. Mech. Phys. Solids 165 (2022) 105076,
§2.2-2.3) and Tartakovsky et al. (Water Resour. Res. 56, 2020, §2).
"""

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from shapely.geometry import box

import jno

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
f = 30.0 * (xi * (1 - xi) + yi * (1 - yi))  # strong source so u is sensitive to k

# Hidden truth: background + a smooth high-diffusivity inclusion.
k_true_expr = 1.0 + 0.8 * jno.np.exp(-((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / (2 * 0.18**2))
fem_ref = jno.fem([k_true_expr * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
A_true = fem_ref.A.todense() if hasattr(fem_ref.A, "todense") else jnp.asarray(fem_ref.A)
u_obs = jnp.linalg.solve(A_true, jnp.asarray(fem_ref.b).reshape(-1))  # clean full-field data

# The unknown coefficient: k = 1 + net(x, y). The offset keeps A nonsingular at the
# (near-zero) net init — the same practice as starting a nodal field at k = 1.
net = jno.nn.wrap(foundax.mlp(2, hidden_dims=32, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
net.dtype(jnp.float64)  # match the float64 assembly
net.optimizer(optax.adam(2e-2))

fem = jno.fem([(1.0 + net(xi, yi)) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
crux = jno.core(
    [(fem.solve() - u_obs).mse],  # no explicit prior: the architecture regularises
    domain=jno.domain.from_array({"_": np.zeros((1, 1))}),
)
crux.solve(2500)

# Verify the TRAINED network itself against the hidden truth on the mesh nodes.
trained = crux.eval([jno.trace.ModelWeights(net)])  # the trained module (current weights)
nodes = np.asarray(d.built_mesh.points)[:, :2]
k_rec = 1.0 + np.asarray(trained(jnp.asarray(nodes))).reshape(-1)
k_true = 1.0 + 0.8 * np.exp(-((nodes[:, 0] - 0.5) ** 2 + (nodes[:, 1] - 0.5) ** 2) / (2 * 0.18**2))
rel = float(np.linalg.norm(k_rec - k_true) / np.linalg.norm(k_true))
print(
    f"\nInverse neural diffusivity: nodes={len(k_true)}  k(x) rel_L2={rel:.3e}  "
    f"peak rec/true={k_rec.max():.3f}/{k_true.max():.3f}"
)
assert rel < 0.1
