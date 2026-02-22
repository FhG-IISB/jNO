import jax
import jno
import jno.numpy as pnp
from jno import LearningRateSchedule as lrs
import optax
import equinox as eqx
from jno.architectures.linear import Linear
import jax.numpy as jnp
import numpy as np
from fem_solution import fem_solver


dire = "./runs/operator_learning"
jno.logger(dire)

π = pnp.pi
sin = pnp.sin
cos = pnp.cos
exp = pnp.exp  # if pnp.exp is unavailable, replace with (1.0 + 0.2*...) and clamp positivity via offsets

# ------------------------------------------------------
# Training data: κ-parameter vectors (batch dimension first)
# Each row defines one κ field via a parametric form κ(x,y;θ)
# ------------------------------------------------------
B_train = 40
B_test = 3

# θ = [θ0, θ1, θ2, θ3]  (simple deterministic spread; you can randomize with jax.random if you want)
key = jnp.arange(B_train + B_test)[:, None]
theta_all = jnp.concatenate([0.2 * jnp.sin(0.3 * key), 0.8 * jnp.sin(0.7 * key), 0.8 * jnp.cos(0.5 * key), 0.6 * jnp.sin(0.9 * key + 0.2)], axis=1)
theta_train = theta_all[:B_train, :]  # (B_train, 4)
theta_test = theta_all[B_train:, :]  # (B_test, 4)

# ------------------------------------------------------
# domain
# ------------------------------------------------------
domain = B_train * jno.domain(constructor=jno.domain.rect(mesh_size=0.05), compute_mesh_connectivity=True)
x, y = domain.variable("interior", (None, None))
xb, yb = domain.variable("boundary", (None, None))
θ = domain.variable("θ", theta_train)
domain.plot(f"{dire}/domain.png")


# ------------------------------------------------------
# DeepONet
# branch takes θ (per-sample), trunk takes (x,y) (per-point)
# output u(x,y;θ)
# ------------------------------------------------------
class DeepONet(eqx.Module):
    width: int
    depth: int
    p: int
    trunk_layers: list
    trunk_out: Linear
    branch_layers: list
    branch_out: Linear

    def __init__(self, width: int, depth: int, p: int, *, key):
        self.width = width
        self.depth = depth
        self.p = p

        keys = jax.random.split(key, 2 * depth + 2)

        # Trunk: first layer takes 2 inputs (x, y)
        self.trunk_layers = []
        in_dim = 2
        for i in range(depth):
            self.trunk_layers.append(Linear(in_dim, width, key=keys[i]))
            in_dim = width
        self.trunk_out = Linear(width, p, key=keys[depth])

        # Branch: first layer takes 4 inputs (θ)
        self.branch_layers = []
        in_dim = 4
        for i in range(depth):
            self.branch_layers.append(Linear(in_dim, width, key=keys[depth + 1 + i]))
            in_dim = width
        self.branch_out = Linear(width, p, key=keys[2 * depth + 1])

    def __call__(self, x, y, θ):
        # x:  (N, 1), y: (N, 1), θ: (N, 4)

        # Trunk
        t = jnp.concatenate([x, y], axis=-1)  # (N, 2)
        for layer in self.trunk_layers:
            t = jnp.tanh(layer(t))
        t = self.trunk_out(t)

        # Branch
        b = θ
        for layer in self.branch_layers:
            b = jnp.tanh(layer(b))
        b = self.branch_out(b)

        return jnp.sum(t * b, axis=-1)  # (N,1) -> but this will get squeezed


net = pnp.nn.wrap(DeepONet(width=64, depth=4, p=32, key=jax.random.PRNGKey(0)))
u = net(x, y, θ) * x * (1 - x) * y * (1 - y)

# ------------------------------------------------------
# Spatially varying diffusivity κ(x,y;θ)
# Ensure positivity using exp (log κ parameterization)
# κ = exp(θ0 + θ1 sin(2πx) + θ2 cos(2πy) + θ3 sin(2π(x+y)))
# ------------------------------------------------------
κ = exp(θ[0] + θ[1] * sin(2 * π * x) + θ[2] * cos(2 * π * y) + θ[3] * sin(2 * π * (x + y)))

# ------------------------------------------------------
# PDE: -div(κ grad u) = f
# Choose a fixed forcing f that does NOT depend on θ.
# ------------------------------------------------------
ux = pnp.grad(u(x, y, θ), x)
uy = pnp.grad(u(x, y, θ), y)
pde = -(pnp.grad(κ * ux, x) + pnp.grad(κ * uy, y)) - (2 * π**2 * sin(π * x) * sin(π * y))

crux = jno.core([pde.mse], domain)
net.optimizer(optax.adam, lr=lrs.warmup_cosine(1_000, 1_00, 1e-3, 1e-5))
crux.solve(1_000).plot(f"{dire}/training_history.png")

# ------------------------------------------------------
# Inference / testing
# (No closed-form u here because κ varies in space; instead, sanity-check PDE residual and visualize u)
# ------------------------------------------------------
tst_domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.01), compute_mesh_connectivity=False)
tst_domain.variable("θ", theta_test)
crux.save(f"{dire}/crux.pkl")


# Inference
pred = np.array(crux.predict(points=np.tile(tst_domain.points[None, ...], (B_test, 1, 1)), operation=u, context=tst_domain.context))
true = fem_solver(tst_domain)(theta_test)


def plot_fem_vs_pinn(domain, u_fem, u_pinn, save_path=f"{dire}/comparison.png"):
    """Quick plot:  FEM vs PINN with error for batch of solutions.

    Args:
        domain: pino.domain with mesh
        u_fem: (B, N) FEM reference solutions
        u_pinn: (B, N) PINN predictions
        save_path: Path to save figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # Extract mesh from domain
    points = np.array(domain.mesh.points[:, :2])
    for cell_block in domain.mesh.cells:
        if cell_block.type == "triangle":
            triangles = np.array(cell_block.data)
            break

    # Handle single solution vs batch
    if u_fem.ndim == 1:
        u_fem = u_fem[None, :]
        u_pinn = u_pinn[None, :]

    B = u_fem.shape[0]
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    fig, axes = plt.subplots(B, 3, figsize=(12, 3 * B), squeeze=False)

    for i in range(B):
        err = np.abs(u_pinn[i] - u_fem[i])
        l2 = np.sqrt(np.mean(err**2))
        rel_l2 = l2 / np.sqrt(np.mean(u_fem[i] ** 2))

        vmin = min(u_fem[i].min(), u_pinn[i].min())
        vmax = max(u_fem[i].max(), u_pinn[i].max())

        # FEM
        tcf = axes[i, 0].tricontourf(triang, u_fem[i], levels=50, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(tcf, ax=axes[i, 0])
        axes[i, 0].set_title("FEM Reference")
        axes[i, 0].set_aspect("equal")

        # PINN
        tcf = axes[i, 1].tricontourf(triang, u_pinn[i], levels=50, vmin=vmin, vmax=vmax, cmap="viridis")
        plt.colorbar(tcf, ax=axes[i, 1])
        axes[i, 1].set_title("PINN Prediction")
        axes[i, 1].set_aspect("equal")

        # Error
        tcf = axes[i, 2].tricontourf(triang, err, levels=50, cmap="Reds")
        plt.colorbar(tcf, ax=axes[i, 2])
        axes[i, 2].set_title(f"|Error| (L2={l2:.2e}, Rel={rel_l2:.2%})")
        axes[i, 2].set_aspect("equal")

        axes[i, 0].set_ylabel(f"Sample {i}", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved:  {save_path}")

    # Print summary
    print("\nL2 Errors:")
    for i in range(B):
        err = np.abs(u_pinn[i] - u_fem[i])
        l2 = np.sqrt(np.mean(err**2))
        rel = l2 / np.sqrt(np.mean(u_fem[i] ** 2))
        print(f"  Sample {i}: L2 = {l2:.6e}, Relative = {rel:.4%}")


plot_fem_vs_pinn(tst_domain, true, pred, save_path=f"{dire}/fem_vs_pinn.png")
