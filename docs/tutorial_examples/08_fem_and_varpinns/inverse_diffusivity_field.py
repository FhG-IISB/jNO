"""09 - Inverse problem: recover a hidden diffusivity field k(x) through a differentiable FEM solve.

    Forward:  -div(k(x) grad u) = f,   u = 0 on the boundary,   with unknown k(x) > 0.

Given the measured response ``u`` to a known source, recover the *entire* nodal diffusivity
field ``k(x)`` -- a hidden high-conductivity inclusion -- by differentiating the FEM solve end
to end. ``k = jno.np.parameter(phi)`` is a trainable P1 field on the trial space;
``fem.solve()`` is the differentiable forward solve; ``crux`` minimises the data misfit plus an
H1-seminorm smoothness prior ``k.regularize("h1seminorm")`` (field inversion is ill-posed
without regularisation). This is the FEM flavour of parameter-field identification / tomography.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

d = jno.domain(jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1))
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
f = 30.0 * (xi * (1 - xi) + yi * (1 - yi))  # strong source so u is sensitive to k

nodes = np.asarray(d.built_mesh.points)[:, :2]
k_true = 1.0 + 0.8 * np.exp(
    -((nodes[:, 0] - 0.5) ** 2 + (nodes[:, 1] - 0.5) ** 2) / (2 * 0.12**2)
)  # background + inclusion

# Generate clean full-field data with one forward solve at the true k -- a known coefficient
# frozen from its coordinate function, so it is a plain non-parametric fem.solve() (no manual solve).
k_fn = lambda x, y: 1.0 + 0.8 * jnp.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2 * 0.12**2))  # noqa: E731
k_known = jno.np.parameter(phi).initialize(k_fn).freeze()
u_obs = jnp.asarray(jno.fem([k_known * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3).solve())

# ... then recover k(x) from u_obs through the differentiable solve + an H1 smoothness prior.
k = jno.np.parameter(phi, name="k")
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
k.dtype(jnp.float64)
k.initialize(jax.nn.initializers.constant(1.0))  # start from a uniform field
k.optimizer(optax.adam(2e-2))
crux = jno.core(
    [(fem.solve() - u_obs).mse, 1e-3 * k.regularize("h1seminorm").mean],
    domain=jno.domain.from_array({"_": np.zeros((1, 1))}),
)
crux.solve(500)

rec = np.asarray(crux.eval([k])).reshape(-1)  # the recovered nodal field (do NOT index [0])
rel = float(np.linalg.norm(rec - k_true) / np.linalg.norm(k_true))
print(
    f"\nInverse diffusivity field: nodes={k_true.shape[0]}  k(x) rel_L2={rel:.3e}  peak rec/true={rec.max():.3f}/{k_true.max():.3f}"
)
assert rel < 0.1
