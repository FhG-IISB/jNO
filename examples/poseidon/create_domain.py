import jno
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# Data paths — adjust as needed
DATA_DIR = os.environ.get("DATA_DIR", "/home/users/armbrust/projects/DATA")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# Normalize as in https://github.com/camlab-ethz/poseidon/blob/main/scOT/problems/elliptic/poisson.py
with h5py.File(f"{DATA_DIR}/Poisson-Gauss.nc", "r") as file:
    _F = file["source"][...]  # (20000, 128, 128) or # (20000, 128, 128, 1)
    _U = file["solution"][...]  # (20000, 128, 128) or # (20000, 128, 128, 1)

_F = (_F - np.mean(_F)) / np.std(_F)
_U = (_U - np.mean(_U)) / np.std(_U)


_domain_ = jno.domain(jno.domain.poseidon(), compute_mesh_connectivity=True)


def create_training_domain(SAMPLES):

    time = np.ones((SAMPLES, 1))  # All 1's since it is a time-indepenent problem

    _f = _F[240 : 240 + SAMPLES][..., np.newaxis]
    _u = _U[240 : 240 + SAMPLES][..., np.newaxis]

    domain = SAMPLES * _domain_
    x, y, t = domain.variable("interior")
    domain.variable("_t", time[:, None, :])
    domain.variable("_f", _f[:, None, None, :, :, :])
    domain.variable("_u", _u[:, None, None, :, :, :])

    domain.save(f"{OUTPUT_DIR}/domain_{SAMPLES}.pkl")

    print(f"Created domain with {SAMPLES} samples -> _f.shape = {domain.context['_f'].shape} and _u.shape = {domain.context['_u'].shape} and _t.shape = {domain.context['_t'].shape}")
    print(f"                                      -> _f.min-max = {_f.min()}-{_f.max()} and _u.min-max = {_u.min()}-{_u.max()}")
    del domain
    return None


def plot_sources_grid(_f, figsize=(20, 20), cmap="viridis", save_path=None):
    """
    Plot a 16x16 grid of source images.

    Parameters:
    -----------
    _f : np.ndarray
        Source array of shape (B, 128, 128, 1) where B >= 256
    figsize : tuple
        Figure size in inches (width, height)
    cmap : str
        Colormap for imshow
    save_path : str, optional
        If provided, saves the figure to this path
    """
    fig, axes = plt.subplots(16, 16, figsize=figsize)
    fig.subplots_adjust(hspace=0.02, wspace=0.02)

    for idx, ax in enumerate(axes.flat):
        if idx < _f.shape[0]:
            # Remove channel dimension and plot
            img = _f[idx, :, :, 0]
            ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")
    return None


def create_inference_domain():

    _f1 = _F[:240][..., np.newaxis]
    _u1 = _U[:240][..., np.newaxis]

    _f2 = np.load(f"/home/users/armbrust/projects/physics_informed_poseidon_finetuning/data/poisson/poisson_dataset_sources_used.npy")[..., np.newaxis]
    _u2 = np.load(f"/home/users/armbrust/projects/physics_informed_poseidon_finetuning/data/poisson/poisson_dataset_solutions_used.npy")[..., np.newaxis]

    _f2 = (_f2 - np.mean(_f2)) / np.std(_f2)
    _u2 = (_u2 - np.mean(_u2)) / np.std(_u2)

    _f = np.concatenate([_f1, _f2], axis=0)
    _u = np.concatenate([_u1, _u2], axis=0)

    SAMPLES = _f.shape[0]  # (B, 128, 128, 1)

    # plot_sources_grid(_f1, save_path=f"{OUTPUT_DIR}/sources_grid_interpolation.png")
    # plot_sources_grid(_f2, save_path=f"{OUTPUT_DIR}/sources_grid_extrapolation.png")
    # plot_sources_grid(_u1, save_path=f"{OUTPUT_DIR}/solutions_grid_interpolation.png")
    # plot_sources_grid(_u2, save_path=f"{OUTPUT_DIR}/solutions_grid_extrapolation.png")

    time = np.ones((SAMPLES, 1))  # All 1's since it is a time-indepenent problem

    domain = SAMPLES * _domain_
    x, y, t = domain.variable("interior")
    domain.variable("_t", time[:, None, :])
    domain.variable("_f", _f[:, None, None, :, :, :])
    domain.variable("_u", _u[:, None, None, :, :, :])

    domain.save(f"{OUTPUT_DIR}/inference_domain_{SAMPLES}.pkl")

    print(f"Created domain with {SAMPLES} samples -> _f.shape = {domain.context['_f'].shape} and _u.shape = {domain.context['_u'].shape} and t.shape = {domain.context['_t'].shape}")
    print(f"                                      -> _f1.min-max = {_f1.min()} / {_f1.mean()} / {_f1.max()} and _u1.min-max = {_u1.min()} / {_u1.mean()} / {_u1.max()}")
    print(f"                                      -> _f2.min-max = {_f2.min()} / {_f2.mean()} / {_f2.max()} and _u1.min-max = {_u2.min()} / {_u2.mean()} / {_u2.max()}")
    del domain
    return None


for S in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    create_training_domain(S)

create_inference_domain()
