import numpy as np

"""
FNO2D Poisson Operator Learning With Reference Comparison
=========================================================

Problem:
    -Δu = f  on [0,1]^2,
    u = 0   on ∂Ω.

This script starts with FNO2D because the problem lives on a fixed regular
Cartesian grid. For this architecture the natural input is just the forcing
field f(x, y): geometry is fixed, the boundary condition is fixed, and the
only sample-varying quantity is the right-hand side.

What this script does:
    1. Generates a supervised operator-learning dataset from truncated sine
       series with known exact solutions.
    2. Trains an FNO2D to learn the map f -> u.
    3. Evaluates on a held-out forcing term.
    4. Compares the neural operator prediction against both the analytical
       solution and a simple structured-grid conjugate-gradient reference
       solve for -Δu = f with homogeneous Dirichlet data.

The file name is left unchanged so it can serve as a scratchpad while we work
through the operator architectures one at a time.
"""

import jax
import jax.numpy as jnp
import jno
import foundax
import numpy as np
import optax


KEY = jax.random.PRNGKey(0)
GRID = 64
SAMPLES = 200
TRAIN_MODES = 5
TEST_MODES = 7
ALPHA = 1.5
TRAIN_SEED = 42
TEST_SEED = 314
EPOCHS = 5_000
BATCH = 40
CG_TOL = 1e-8
CG_MAXITER = 4_000


def generate_poisson_data(n_samples: int, grid_size: int, n_modes: int, alpha: float, seed: int):
    """Return forcing/solution pairs with shape (N, H, W, 1)."""
    rng = np.random.default_rng(seed)

    x1d = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    xg, yg = np.meshgrid(x1d, x1d, indexing="ij")

    sin_x = np.stack([np.sin((k + 1) * np.pi * xg) for k in range(n_modes)], axis=0)
    sin_y = np.stack([np.sin((m + 1) * np.pi * yg) for m in range(n_modes)], axis=0)

    kk, mm = np.meshgrid(np.arange(1, n_modes + 1), np.arange(1, n_modes + 1), indexing="ij")
    denom = (kk**2 + mm**2).astype(np.float32) * np.pi**2
    scale = (kk**2 + mm**2).astype(np.float32) ** (-alpha / 2)

    f_arr = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    u_arr = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)

    for sample_idx in range(n_samples):
        coeffs = rng.standard_normal((n_modes, n_modes)).astype(np.float32) * scale
        for k in range(n_modes):
            for m in range(n_modes):
                mode = sin_x[k] * sin_y[m]
                f_arr[sample_idx] += coeffs[k, m] * mode
                u_arr[sample_idx] += coeffs[k, m] / denom[k, m] * mode

    return f_arr[..., None], u_arr[..., None]


def build_training_domain(samples: int, grid_size: int) -> jno.domain:
    f_train, u_train = generate_poisson_data(samples, grid_size, TRAIN_MODES, ALPHA, TRAIN_SEED)

    base_domain = jno.domain(
        constructor=jno.domain.poseidon(nx=grid_size, ny=grid_size),
        compute_mesh_connectivity=True,
    )
    domain = samples * base_domain
    domain.variable("interior")
    domain.variable("_f", f_train[:, None, None, :, :, :])
    domain.variable("_u", u_train[:, None, None, :, :, :])
    return domain


def apply_dirichlet_poisson(u: np.ndarray, h: float) -> np.ndarray:
    padded = np.pad(u, 1, mode="constant")
    return (4.0 * padded[1:-1, 1:-1] - padded[:-2, 1:-1] - padded[2:, 1:-1] - padded[1:-1, :-2] - padded[1:-1, 2:]) / (h * h)


def solve_poisson_reference(forcing: np.ndarray, tol: float = CG_TOL, maxiter: int = CG_MAXITER):
    """Matrix-free structured-grid reference solve for -Δu = f with zero BCs."""
    rhs = forcing.astype(np.float64)
    grid_size = rhs.shape[0]
    h = 1.0 / (grid_size + 1)

    u = np.zeros_like(rhs)
    residual = rhs - apply_dirichlet_poisson(u, h)
    direction = residual.copy()
    residual_sq = float(np.sum(residual * residual))

    if residual_sq == 0.0:
        return u.astype(np.float32), 0

    for iteration in range(1, maxiter + 1):
        ap = apply_dirichlet_poisson(direction, h)
        alpha = residual_sq / float(np.sum(direction * ap))
        u = u + alpha * direction
        residual = residual - alpha * ap
        new_residual_sq = float(np.sum(residual * residual))

        if np.sqrt(new_residual_sq) < tol:
            return u.astype(np.float32), iteration

        beta = new_residual_sq / residual_sq
        direction = residual + beta * direction
        residual_sq = new_residual_sq

    raise RuntimeError(f"CG reference solve did not converge in {maxiter} iterations")


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    target_norm = float(np.linalg.norm(target.ravel()))
    if target_norm == 0.0:
        return float(np.linalg.norm(prediction.ravel()))
    return float(np.linalg.norm((prediction - target).ravel()) / target_norm)


def main():
    domain = build_training_domain(SAMPLES, GRID)
    forcing_train = domain.variable("_f")
    solution_train = domain.variable("_u")

    model = jno.nn.wrap(foundax.fno2d(
        in_features=1,
        hidden_channels=48,
        n_modes=24,
        d_vars=1,
        n_layers=2,
        n_steps=1,
        d_model=(GRID, GRID),
        norm="layer",
        activation=jnp.tanh,
        key=KEY,
    ))
    model.optimizer(optax.chain(optax.clip_by_global_norm(1e-3), optax.adamw(1.0, weight_decay=1e-6)))
    model.lr(jno.schedule.learning_rate.cosine(EPOCHS, 5e-4, 1e-7))

    crux = jno.core([(solution_train - model(forcing_train)).mse], domain)

    crux.solve(
        epochs=EPOCHS,
        batchsize=BATCH,
        checkpoint_gradients=False,
        offload_data=False,
    )

    forcing_test, solution_exact = generate_poisson_data(1, GRID, TEST_MODES, ALPHA, TEST_SEED)
    forcing_test = forcing_test[0]
    solution_exact = solution_exact[0]

    solution_reference, cg_iterations = solve_poisson_reference(forcing_test[..., 0])
    solution_reference = solution_reference[..., None]

    solution_prediction = np.asarray(jax.device_get(model.module(jnp.asarray(forcing_test))))

    err_pred_exact = relative_l2(solution_prediction, solution_exact)
    err_pred_ref = relative_l2(solution_prediction, solution_reference)
    err_ref_exact = relative_l2(solution_reference, solution_exact)

    print("FNO2D input choice: forcing field only (1 channel).")
    print(f"Prediction vs exact relative L2:     {err_pred_exact:.6e}")
    print(f"Prediction vs reference relative L2: {err_pred_ref:.6e}")
    print(f"Reference vs exact relative L2:      {err_ref_exact:.6e}")
    print(f"Reference solver CG iterations:      {cg_iterations}")


if __name__ == "__main__":
    main()
