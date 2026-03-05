import jno
import jno.numpy as jnn
import optax
import jax
import jax.numpy as jnp

# from aFunctionGenerator import aFunctionGeneratorHelmholtz
import argparse


@jax.jit
def compute_helmholtz_residuals_pi(_u: jnp.ndarray, bc: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    omega = 5 * jnp.pi / 2
    h = (1.0 / 127.0) ** 2
    p = jnp.pad(_u, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant", constant_values=jnp.squeeze(bc))
    laplacian = (p[:, 2:, 1:-1, :] + p[:, :-2, 1:-1, :] + p[:, 1:-1, 2:, :] + p[:, 1:-1, :-2, :] - 4 * _u) / h  # 5-point Laplacian stencil
    return laplacian + (omega**2 * _u * a)


def parse_args():
    parser = argparse.ArgumentParser(description="Poseidon Helmholtz training")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples")
    parser.add_argument("--mode", type=str, choices=["data", "phys", "hybrid"], default="data", help="Training mode")
    return parser.parse_args()


if __name__ == "__main__":

    # SETUP
    args = parse_args()
    name = f"poseidon_pi_helmholtz_{args.samples}_{args.mode}"
    dire = jno.setup(__file__, name)
    # gen_ic = aFunctionGeneratorHelmholtz(shape=(128, 128))
    EPOCHS = 100
    PHYS_CONSTANT = 1.93
    constraints = []

    # DOMAIN
    domain = jno.load(f"/home/b8cl/projects/jNO/domain_1.pkl")
    t = domain.variable("t")
    bc = domain.variable("bc")
    a = domain.variable("a")
    u = domain.variable("u")

    # NEURAL NETWORK
    NN = jnn.nn.poseidonT(num_in_channels=1, num_out_channels=1)
    NN.dont_show()
    NN.initialize("/home/b8cl/projects/DATA/poseidon/poseidonT.msgpack")
    NN.optimizer(opt_fn=optax.adamw(1))
    NN.lr(jno.schedule.learning_rate.exponential(1e-3, 0.8, EPOCHS, 1e-6))

    # CONSTRAINTS
    if args.mode in ["data", "hybrid"]:
        constraints.append(NN(a, t) - u)
    # if args.mode in ["phys", "hybrid"]:
    #    gen_ic_fun = jnn.function(gen_ic.generate_random, [])
    #    constraints.append(jnn.function(compute_helmholtz_residuals_pi, [NN(a, t), bc * PHYS_CONSTANT, gen_ic_fun * PHYS_CONSTANT]))

    # TRAINING
    crux = jno.core(constraints=[const.mse for const in constraints], domain=domain, mesh=(1, 1)).print_shapes()

    with jax.profiler.trace("/tmp/xprof", create_perfetto_trace=False):
        stats = crux.solve(epochs=EPOCHS, batchsize=2)

    stats.plot(f"./{dire}/training_history.png")

    # SAVE
    jno.save(crux, f"./{dire}/crux.pkl")
