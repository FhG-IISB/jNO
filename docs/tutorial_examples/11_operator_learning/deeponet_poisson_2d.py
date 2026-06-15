"""11 — DeepONet 2D for parametric Poisson"""

import foundax
import jax
import optax
from shapely.geometry import box

import jno

KEY = jax.random.PRNGKey(0)
N_SAMPLES = 50
EPOCHS = 2_000

# ── Parametric domain — replicate one mesh across N_SAMPLES random k values ──
dom = N_SAMPLES * jno.domain(box(0, 0, 2, 1), mesh_size=0.05)
x, y, _ = dom.variable("interior")

k_values = jax.random.uniform(KEY, shape=(N_SAMPLES, 1, 1), minval=0.5, maxval=1.5)
k = dom.variable("k", k_values)

# ── Network ──────────────────────────────────────────────────────────────────
net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,  # branch input is the scalar k
        coord_dim=2,  # trunk input is (x, y)
        basis_functions=32,
        hidden_dim=128,
        activation=jax.numpy.tanh,
        key=KEY,
    )
)
net.optimizer(optax.adam(optax.cosine_decay_schedule(1e-3, EPOCHS, alpha=1e-5 / 1e-3)))

# ── Hard BC ansatz + PDE residual ────────────────────────────────────────────
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.d2(x) + u.d2(y)) + 1.0

# ── Solve ────────────────────────────────────────────────────────────────────
crux = jno.core(constraints=[pde.mse])
crux.solve(epochs=EPOCHS, batchsize=32)
