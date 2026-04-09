import jno
import jax
import optax

dir = jno.setup("./runs/test")

# Domain
dom = 500 * jno.domain.rect(mesh_size=0.05, x_range=(0, 2), y_range=(0, 1))
x, y, _ = dom.variable("interior")
xb, yb, _ = dom.variable("boundary")

random_k = jax.random.uniform(jax.random.PRNGKey(0), shape=(500, 1, 1), minval=0.5, maxval=1.5)
k = dom.variable("k", random_k)

# Neural Network
net = jno.nn.deeponet(n_sensors=1, coord_dim=2, basis_functions=32, hidden_dim=128, activation=jax.numpy.tanh)
net.optimizer(optax.adam(learning_rate=optax.schedules.cosine_decay_schedule(init_value=1e-3, decay_steps=20_000, alpha=1e-5)))

# Forward pass and hard enforcement of BCs via output transformation
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.dd(x) + u.dd(y)) + 1.0  # PDE Loss

# Checkpointing (saves every 5000 epochs, keeps best 3)
cb = jno.callback.checkpoint(save_interval_epochs=5000, best_fn=lambda m: m["total_loss"])

# Create -> Train -> Save
crux = jno.core(constraints=[pde.mse], domain=dom).print_shapes()
crux.solve(epochs=20_000, batchsize=32, callbacks=[cb], accumulation_steps=3, profile=True).plot(f"{dir}/training.png")
jno.save(crux, f"{dir}/model.pkl")

# Inference via test domain on a finer mesh
tst_dom = 16 * jno.domain.rect(mesh_size=0.01, x_range=(0, 2), y_range=(0, 1))
tst_dom.variable("k", jax.random.uniform(jax.random.PRNGKey(0), shape=(16, 1, 1), minval=0.1, maxval=1.9))

pred, x, y, k = crux.eval([u, x, y, k], domain=tst_dom)
print(pred.shape, x.shape, y.shape, k.shape)
