"""Shared Poisson data generation for the operator-learning architecture zoo."""

import jno
import numpy as np
GRID = 128
N_MODES = 5
ALPHA = 1.5
SAMPLES = 3000
SEED = 42
FORCING_SCALE = 4.0
def cell_centered_grid(grid_size: int):
    """Return 1D and 2D cell-centered coordinates on [0, 1]."""
    x1d = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
    xg, yg = np.meshgrid(x1d, x1d, indexing="ij")
    return x1d, xg, yg
def grid_coordinates(grid_size: int) -> np.ndarray:
    """Return flattened coordinate pairs with shape (grid_size^2, 2)."""
    _, xg, yg = cell_centered_grid(grid_size)
    return np.stack([xg.ravel(), yg.ravel()], axis=-1).astype(np.float32)
def generate_poisson_data(
    n_samples: int,
    grid_size: int,
    n_modes: int,
    alpha: float,
    seed: int,
    forcing_scale: float = FORCING_SCALE,
):
    """Return exact forcing/solution pairs with shape (N, H, W, 1)."""
    rng = np.random.default_rng(seed)
    _, xg, yg = cell_centered_grid(grid_size)

    sin_x = np.stack([np.sin((k + 1) * np.pi * xg) for k in range(n_modes)], axis=0)
    sin_y = np.stack([np.sin((m + 1) * np.pi * yg) for m in range(n_modes)], axis=0)

    kk, mm = np.meshgrid(np.arange(1, n_modes + 1), np.arange(1, n_modes + 1), indexing="ij")
    denom = (kk**2 + mm**2).astype(np.float32) * np.pi**2
    scale = (kk**2 + mm**2).astype(np.float32) ** (-alpha / 2)

    f_arr = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    u_arr = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)

    for sample_idx in range(n_samples):
        coeffs = rng.standard_normal((n_modes, n_modes)).astype(np.float32) * scale * forcing_scale
        for k in range(n_modes):
            for m in range(n_modes):
                mode = sin_x[k] * sin_y[m]
                f_arr[sample_idx] += coeffs[k, m] * mode
                u_arr[sample_idx] += coeffs[k, m] / denom[k, m] * mode

    return f_arr[..., None], u_arr[..., None]
def build_domain_from_arrays(forcing: np.ndarray, solution: np.ndarray, grid_size: int):
    """Build a batched jNO domain with stored `_f` and `_u` tensors."""
    if forcing.shape != solution.shape:
        raise ValueError(f"forcing and solution must have identical shapes, got {forcing.shape} and {solution.shape}")

    samples = int(forcing.shape[0])
    base_domain = jno.domain(
        constructor=jno.domain.poseidon(nx=grid_size, ny=grid_size),
        compute_mesh_connectivity=True,
    )
    domain = samples * base_domain
    domain.variable("interior")
    domain.variable("_f", forcing[:, np.newaxis, np.newaxis, :, :, :])
    domain.variable("_u", solution[:, np.newaxis, np.newaxis, :, :, :])
    return domain
def build_training_domain(
    samples: int = SAMPLES,
    grid_size: int = GRID,
    n_modes: int = N_MODES,
    alpha: float = ALPHA,
    seed: int = SEED,
    forcing_scale: float = FORCING_SCALE,
):
    forcing, solution = generate_poisson_data(samples, grid_size, n_modes, alpha, seed, forcing_scale=forcing_scale)
    domain = build_domain_from_arrays(forcing, solution, grid_size)
    return domain, forcing, solution
def save_domain(
    out_path: str | None = None,
    samples: int = SAMPLES,
    grid_size: int = GRID,
    n_modes: int = N_MODES,
    alpha: float = ALPHA,
    seed: int = SEED,
    forcing_scale: float = FORCING_SCALE,
):
    """Generate the training dataset and persist it as a jNO domain."""
    domain, forcing, solution = build_training_domain(samples, grid_size, n_modes, alpha, seed, forcing_scale=forcing_scale)
    output_path = out_path or f"domain_{samples}_{grid_size}.pkl"
    jno.save(domain, output_path)
    return output_path, domain, forcing, solution
def main():
    print(f"Generating {SAMPLES} samples on {GRID}x{GRID} grid ...")
    out_path, domain, forcing, solution = save_domain()
    print(f"  forcing scale: {FORCING_SCALE:.2f}")
    print(f"  f: {forcing.shape}, u: {solution.shape}")
    print(f"  f range [{forcing.min():.3f}, {forcing.max():.3f}]  u range [{solution.min():.3f}, {solution.max():.3f}]")

