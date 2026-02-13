import jno
import jno.numpy as jnn
import flax.linen as nn
import optax

π = jnn.pi
sin = jnn.sin

# Logging
dire = "./runs/laplace1D"
jno.logger(dire)

# Domain
tst_domain = jno.domain(constructor=jno.domain.line(mesh_size=0.001))
domain = 1 * jno.domain(constructor=jno.domain.line(mesh_size=0.01))
(x,) = domain.variable("interior")

# Analytical
_u = -(1 / (π**2)) * sin(π * x)

# Neural Network
u = jnn.nn.mlp(hidden_dims=64, num_layers=3)(x) * x * (1 - x)

# Constraints
pde = jnn.laplacian(u, [x]) - sin(π * x)  # 2D heat equation
con = jnn.tracker(jnn.mean(u - _u))


# Solve
crux = jno.core([pde, con], domain)
crux.solve(10_000, optax.adam, jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5)).plot(f"{dire}/training_history.png")

crux.plot(operation=u - _u, test_pts=tst_domain).savefig(f"{dire}/u_erro.png", dpi=300)
crux.errors.all([u], [_u], test_pts=tst_domain)

# Finetune using LoRA
crux.solve(
    10_000,
    optax.adam,
    jno.schedule.learning_rate.exponential(1e-3, 0.8, 10_000, 1e-5),
    lora=jno.create_rank_dict(crux.params, rank=1, alpha=1.0),
).plot(f"{dire}/training_history_lora.png")

# pred = crux.eval(u)
crux.plot(operation=u - _u, test_pts=tst_domain).savefig(f"{dire}/u_erro_lora.png", dpi=300)
crux.plot(operation=u, test_pts=tst_domain).savefig(f"{dire}/u_pred.png", dpi=300)
crux.errors.all([u], [_u], test_pts=tst_domain)

# Save
crux.save(f"{dire}/crux.pkl")
