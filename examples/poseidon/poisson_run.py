import jno
import optax
import argparse
import jno.numpy as jnn
import jax
from source_generator import SourceTermGeneratorPoisson
import jax.numpy as jnp

model_choices = [
    "poseidonT",
    "poseidonB",
    "poseidonL",
    "walrusL",
    "cno",
    "pit",
    "fno",
    "unet",
    "pointnet",
    "pdeformer2_small",
    "pdeformer2_base",
    "pdeformer2_fast",
    "morphTi",
    "morphS",
    "morphM",
    "morphL",
    "mppTi",
    "mppS",
    "mppB",
    "mppL",
]

MODEL_PATH = "/home/users/armbrust/projects/DATA/models"
DATA_PATH = "/home/users/armbrust/projects/jno/examples/poseidon"
RUN_PATH = "/home/users/armbrust/projects/jno"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune foundation models for physics-informed learning")
    parser.add_argument("--model", "-m", type=str, choices=model_choices)
    parser.add_argument("--samples", "-s", type=int, help="Number of samples [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]")
    parser.add_argument("--training", "-t", type=str, choices=["data", "phys", "both"])
    parser.add_argument("--lora_rank", "-l", type=int, default=0, choices=[0, 1, 2, 4, 8, 16, 32, 64, 128])
    args = parser.parse_args()

    MODEL = args.model
    SAMPLES = args.samples
    TYPE = args.training
    LORA = True if args.lora_rank != 0 else False
    LORARANK = int(args.lora_rank)
    DATA_DRIVEN = True if TYPE in ["data", "both"] else False
    PHYS_INFORM = True if TYPE in ["phys", "both"] else False
    BATCHSIZE = SAMPLES if SAMPLES <= 32 else 40
    EPOCHS = int(200 * (SAMPLES / BATCHSIZE))
    constraints = []

    l_path = f"{RUN_PATH}/runs/{MODEL}_{TYPE}_{SAMPLES}_lora{LORARANK}" if LORA else f"{RUN_PATH}/runs/{MODEL}_{TYPE}_{SAMPLES}"
    d_path = f"{DATA_PATH}/domain_{SAMPLES}.pkl"
    w_path = f"{MODEL_PATH}/{MODEL}.msgpack"

    jno.logger(l_path)

    domain = jno.domain.load(d_path)
    x, y, t = domain.variable("interior")
    _u = domain.variable("_u")  # (S, 1, 1, 128, 128, 1)
    _f = domain.variable("_f")  # (S, 1, 1, 128, 128, 1)
    import numpy as np

    GENF = SourceTermGeneratorPoisson(max_amplitude=np.max(domain.context["_f"]))

    key = jax.random.PRNGKey(0)

    if "pdeformer2" in MODEL:
        import numpy as np
        import sys

        sys.path.append("/home/users/armbrust/projects/jax_pdeformer2/scripts")
        from compare import PDENodesCollector, x_fenc, y_fenc

        # Build static Poisson DAG:  u_xx + u_yy + f = 0,  u=0 on boundary
        pde = PDENodesCollector()
        u_var = pde.new_uf()
        # Source term f (coefficient function — placeholder zeros; structure is what matters)
        pde._add_func(np.zeros_like(x_fenc), x=x_fenc, y=y_fenc)
        f_node = pde._add_node("cf")
        # Zero Dirichlet BC
        pde._add_func(np.zeros_like(x_fenc), x=x_fenc, y=y_fenc)
        pde._add_node("bv", [u_var])
        # PDE equation:  u_xx + u_yy + f = 0
        u_xx = pde.dx(pde.dx(u_var))
        u_yy = pde.dy(pde.dy(u_var))
        pde.sum_eq0(u_xx, u_yy, f_node)
        dag = pde.gen_dag(uf_num_mod=11)

        pdeformer2_factory = {
            "pdeformer2_small": jnn.nn.pdeformer2_small,
            "pdeformer2_base": jnn.nn.pdeformer2_base,
            "pdeformer2_fast": jnn.nn.pdeformer2_fast,
        }
        u = pdeformer2_factory[MODEL](dag_inputs=dag).dont_show()
        u.initialize(w_path)
        input = [jnn.concat([x * 0.0, x, y, x * 0.0])]  # coordinate: [t=0, x, y, z=0] → (N, 4)

    elif MODEL == "fno":
        u = jnn.nn.fno2d(1, hidden_channels=48, n_modes=24, d_vars=1, n_layers=4, n_steps=1, d_model=(128, 128), key=key).dont_show()
        input = [_f]

    elif MODEL == "cno":
        u = jnn.nn.cno2d(size=128, key=key)
        input = [_f]

    elif MODEL == "pit":
        u = jnn.nn.pit(1, 1, n_head=2, input_res=(128, 128), output_res=(128, 128), key=key)
        input = [_f[0, ...].reshape((128 * 128, 1))]
        _u = _u.reshape((1, 128 * 128, 1))

    elif MODEL == "pointnet":
        u = jnn.nn.pointnet(1, 1, hidden_dims=400 * [32, 16, 8, 4, 2, 2, 4, 8, 8], key=key)
        input = [_f[0, ...].reshape((1, 128 * 128, 1))]
        _u = _u.reshape((1, 128 * 128, 1))

    elif MODEL == "unet":
        u = jnn.nn.unet2d(in_channels=1, out_channels=1, depth=4, wf=6, key=key)
        input = [_f[0, ...]]

    elif MODEL == "walrusL":
        u = jnn.nn.walrus((1, 1, 128, 128, 1), num_out_channels=1).dont_show()
        u.initialize(w_path)
        input = [_f]

    elif "morph" in MODEL:
        # MorphAdapter (built into the wrapper) reshapes jNO's per-sample
        # (1, H, W, 1) → MORPH's (B, t, F, C, D, H, W) = (1,1,1,1,1,128,128)
        # and maps output (1,F,C,D,H,W) back to (1, H, W, 1).
        morph_factory = {
            "morphTi": jnn.nn.morphTi,
            "morphS": jnn.nn.morphS,
            "morphM": jnn.nn.morphM,
            "morphL": jnn.nn.morphL,
        }
        u = morph_factory[MODEL](spatial_size=128).dont_show()
        u.initialize(w_path)
        input = [_f]

    elif "mpp" in MODEL:
        mpp_factory = {
            "mppTi": jnn.nn.mppTi,
            "mppS": jnn.nn.mppS,
            "mppB": jnn.nn.mppB,
            "mppL": jnn.nn.mppL,
        }
        u = mpp_factory[MODEL](spatial_size=128).dont_show()
        u.initialize(w_path)
        input = [_f]

    elif "poseidon" in MODEL:
        poseidon_factory = {"poseidonT": jnn.nn.poseidonT, "poseidonB": jnn.nn.poseidonB, "poseidonL": jnn.nn.poseidonL}
        u = poseidon_factory[MODEL](num_in_channels=1, num_out_channels=1).dont_show()
        u.initialize(w_path)
        input = [_f, t]

    # u.dtype(jnp.bfloat16)

    if DATA_DRIVEN:
        constraints.append(_u - u(*input))

    if PHYS_INFORM:

        @jax.jit
        def fd_kernel_poisson(upred, f):
            p = jnp.pad(upred, ((1, 1), (1, 1)))  # Pad with zeros (Dirichlet BC)
            return ((p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * upred) / 6.200012400024799e-05) + f  # 5-point Laplacian stencil

        gen_f = jnn.function(GENF.generate_random, [])
        constraints.append(jnn.function(fd_kernel_poisson, [u(gen_f), gen_f]))

    crux = jno.core([con.mse for con in constraints], domain, 42, (len(jax.devices()), 1))
    crux.print_shapes()

    # Per-model optimizer + optional LoRA
    if LORA:
        u.lora(rank=LORARANK, alpha=LORARANK * 2.0)

    u.optimizer(
        optax.chain(optax.clip_by_global_norm(1e-3), optax.adamw(1.0, weight_decay=1e-6)),
        lr=jno.schedule.learning_rate.cosine(EPOCHS, 5e-4, 1e-7),
    )

    crux.solve(epochs=EPOCHS, constraint_weights=jno.schedule.constraint([1]), batchsize=BATCHSIZE, checkpoint_gradients=False, offload_data=False).plot(f"{l_path}/training_history.png")

    crux.domain = None
    crux.save(f"{l_path}/crux.pkl")

    return None


if __name__ == "__main__":
    main()
