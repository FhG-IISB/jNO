"""Inverse problem: learn a nonlinear constitutive law k(T) from a measured field (NN-EUCLID style).

    Forward:  -div(k(T) grad T) = f,   T = 0 on the boundary,   with unknown law k(T) > 0.

Here the unknown is *not* a spatial map k(x) but a **material law**: the conductivity depends on
the solution itself, ``k = k(T)`` (temperature-dependent conduction). A network takes the trial
value as its input, so the weak form is nonlinear in the unknown and ``jno.fem`` routes it to the
matrix-free Newton path automatically; the network's T-dependence enters the element Jacobians
through per-element forward AD. Physical positivity of the conductivity is enforced *by the
architecture* — a softplus output, ``k(T) = softplus(net(T)) > 0`` — so the operator can never go
indefinite during training (the same idea as input-convex nets for hyperelastic potentials).
Training the weights through the differentiable solve recovers the law from a single observed
temperature field, with no stress/flux labels — the unsupervised constitutive-learning setting of
NN-EUCLID (M. Flaschel, S. Kumar, L. De Lorenzis, J. Mech. Phys. Solids 165 (2022) 105076,
§2.2-2.3; A. Tartakovsky et al., Water Resour. Res. 56 (2020), §2).

Unlike a spatial map, a learned law k(T) is *transferable*: reuse it on any geometry or load.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # heavy per-step Newton inverse: run on CPU (no GPU contention/OOM)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # FEM assembly / solves are float64

import foundax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.newton_krylov import newton_krylov  # noqa: E402

PI = np.pi

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
u, phi = d.fem_symbols()  # ``u`` is the temperature field T
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
f = 12.0 * jno.np.sin(PI * xi) * jno.np.sin(PI * yi)  # heat source (drives a wide T-range)


# Hidden truth: conductivity rises with temperature, k(T) = 1 + 0.6 T^2. Generate the observed
# field by solving the (nonlinear) forward problem written symbolically.
def k_true(T):
    return 1.0 + 0.6 * T**2


fem_ref = jno.fem([k_true(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
T_obs = newton_krylov(lambda w: fem_ref.operator(w), jnp.zeros(fem_ref.operator.size))
T_max = float(jnp.max(T_obs))

# The unknown law: k(T) = softplus(net(T)). A 1-input MLP; the softplus output guarantees k > 0
# (so the operator stays positive-definite for *any* weights the optimiser visits) and gives a
# nonsingular k = softplus(0) ~ 0.69 at the near-zero network init.
net = jno.nn.wrap(foundax.mlp(1, hidden_dims=32, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0)))
net.dtype(jnp.float64)
net.optimizer(optax.adam(5e-3))


def k_model(T):
    return jno.np.log1p(jno.np.exp(net(T)))  # softplus(net(T)) > 0


fem = jno.fem([k_model(ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)
crux = jno.core([(fem.solve() - T_obs).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
crux.solve(1500)

# Verify the LEARNED law against the truth over the observed temperature range [0, T_max]:
# the net only sees k(T) where T was actually sampled, so we score it there.
trained = crux.eval([jno.trace.ModelWeights(net)])
T_grid = jnp.linspace(0.0, T_max, 100).reshape(-1, 1)
k_learned = np.asarray(jnp.log1p(jnp.exp(trained(T_grid)))).reshape(-1)  # softplus(net) on the grid
k_ref = np.asarray(k_true(T_grid)).reshape(-1)
rel = float(np.linalg.norm(k_learned - k_ref) / np.linalg.norm(k_ref))
print(
    f"\nLearned constitutive law k(T): T_range=[0, {T_max:.3f}]  "
    f"rel_L2={rel:.3e}  k(T_max) learned/true = {k_learned[-1]:.3f}/{k_ref[-1]:.3f}"
)
assert rel < 0.05
