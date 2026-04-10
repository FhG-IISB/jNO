import jno
import jax
import optax

dir = jno.setup("./runs/test")


# Training and validation data
train_k = jax.random.uniform(jax.random.PRNGKey(0), (800, 1, 1), minval=1.0, maxval=5.0)
train_a = jax.numpy.repeat(jax.numpy.array([1, 2, 3, 4])[:, None, None], jax.numpy.array([500, 100, 100, 100]), axis=0)
test_k = jax.random.uniform(jax.random.PRNGKey(1), (16, 1, 1), minval=0.0, maxval=6.0)
test_a = jax.numpy.ones((16, 1, 1))

# Domains
dom = 500 * jno.domain.polygon([(0, 0), (0, 1), (1, 1), (1, 0)], mesh_size=0.025)
dom += 100 * jno.domain.polygon([(0, 0), (1, 0), (0, 1)], mesh_size=0.025)
dom += 100 * jno.domain.polygon([(0, 0), (1, 0), (1, 1)], mesh_size=0.025)
dom += 100 * jno.domain.disk(mesh_size=0.025, center=(1, 0.5), radius=0.5)


# Variables
x, y, _ = dom.variable("interior")
xb, yb, _ = dom.variable("boundary")
k, a = dom.variable("k", train_k), dom.variable("a", train_a)

xy = jno.np.concat([x, y], axis=-1)
xyb = jno.np.concat([xb, yb], axis=-1)
ka = jno.np.concat([k, a], axis=-1)


# Neural Network
net = jno.nn.deeponet(n_sensors=2, coord_dim=2, basis_functions=8, hidden_dim=16, activation=jax.numpy.tanh)
net.optimizer(optax.adam(learning_rate=optax.schedules.cosine_decay_schedule(init_value=1e-3, decay_steps=20_000, alpha=1e-5)))


# Forward pass
u = net(ka, xy)
ub = net(ka, xyb)

pde = (u.dd(x) + u.dd(y) - k).mse  # PDE Loss
bcs = (ub - 0.0).mse  # Boundary Conditions Loss

w0, w1 = jno.fn.adaptive.relobralo([pde, bcs])  # Dynamic Weight Averaging for balancing losses

# Checkpointing
cb = jno.callback.checkpoint(save_interval_epochs=5000, best_fn=lambda m: m["total_loss"])
es = jno.callback.early_stopping(patience=1000, min_delta=1e-6, mode="rel")

# Create -> Train -> Save
crux = jno.core(constraints=[w0 * pde, w1 * bcs, w0.tracker(), w1.tracker()], domain=dom).print_shapes()
crux.solve(epochs=5_000, batchsize=6, callbacks=[cb, es], accumulation_steps=1, profile=False).plot(f"{dir}/training.png")
jno.save(crux, f"{dir}/model.pkl")


pred_v, x_v, y_v, k_v, a_v = crux.eval([u, x, y, k, a])
print(pred_v.shape, x_v.shape, y_v.shape, k_v.shape, a_v.shape)

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    idx = i * (pred_v.shape[0] // 32)
    xi = np.array(x_v[idx, :, 0])
    yi = np.array(y_v[idx, :, 0])
    zi = np.array(pred_v[idx, :, 0])

    tri = mtri.Triangulation(xi, yi)
    # Mask long-edge triangles that span across concavities
    xy_pts = np.column_stack([xi, yi])
    edge_thresh = np.median(np.linalg.norm(np.diff(np.sort(xi)) + 1j * np.diff(np.sort(yi)))) * 4
    triangles = tri.triangles
    p0, p1, p2 = xy_pts[triangles[:, 0]], xy_pts[triangles[:, 1]], xy_pts[triangles[:, 2]]
    max_edge = np.max(
        np.stack(
            [
                np.linalg.norm(p1 - p0, axis=1),
                np.linalg.norm(p2 - p1, axis=1),
                np.linalg.norm(p0 - p2, axis=1),
            ],
            axis=0,
        ),
        axis=0,
    )
    tri.set_mask(max_edge > edge_thresh)

    ax.tricontourf(tri, zi, levels=32, cmap="viridis", vmin=pred_v.min(), vmax=pred_v.max())
    ax.text(0.5, 0.05, f"k={float(k_v[idx, 0]):.2f}", transform=ax.transAxes, ha="center", va="bottom", fontsize=6, color="white", bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, lw=0))
    ax.set_aspect("equal")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{dir}/solutions.png", dpi=150)
plt.close()


del crux

# Inference via test domain on a finer mesh
crux = jno.load(f"{dir}/model.pkl")
tst_dom = 16 * jno.domain.rect(mesh_size=0.01)
tst_dom.variable("k", jax.random.uniform(jax.random.PRNGKey(0), (16, 1, 1), minval=0.1, maxval=1.9))
tst_dom.variable("a", jax.numpy.ones((16, 1, 1)))

pred, x, y, k, a = crux.eval([u, x, y, k, a], domain=tst_dom)
print(pred.shape, x.shape, y.shape, k.shape, a.shape)
