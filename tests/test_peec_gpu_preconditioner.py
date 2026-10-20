"""Where a welded network's preconditioner is factorised on a GPU, and that it solves what it is given.

cuDSS pivots statically. On a welded power-module network (6,590 unknowns, a saddle point: the KCL rows
carry a zero diagonal block) its factorisation -- equilibrated, with diagonal matching, as jNO fed it --
solved its own matrix to a relative residual of 3e17, and GMRES preconditioned by it diverged (jNO #168).
SuperLU on the host solved the same matrix to 2e-13 in 1.6 ms per apply; cuDSS only matched that with
global pivoting, at 9-12 ms per apply. So on a GPU the host factorisation is the default, and the device
paths are opt-in (``JNO_PEEC_DEVICE_LU=1``).
"""

import jax
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jno.utils.solver.peec as PS

gpu = pytest.mark.skipif(jax.default_backend() != "gpu", reason="the device paths exist only on a GPU")


def saddle(n=40, m=12, seed=0):
    """[[Z, -A^T], [A, 0]]: a complex impedance block and an incidence-like constraint block."""
    rng = np.random.default_rng(seed)
    Z = sp.diags(rng.uniform(1e-3, 1e-2, n) + 1j * rng.uniform(1e-2, 1.0, n))
    Z = Z + sp.random(n, n, density=0.1, random_state=1) * 1e-3j
    A = sp.random(
        m, n, density=0.2, random_state=2, data_rvs=lambda k: np.sign(np.random.default_rng(3).standard_normal(k))
    )
    A = A + sp.eye(m, n)  # full row rank
    return sp.bmat([[Z, -A.T], [A, None]], format="csr").astype(complex)


@gpu
def test_on_a_gpu_the_preconditioner_is_the_host_factorisation_by_default(monkeypatch):
    monkeypatch.delenv("JNO_PEEC_DEVICE_LU", raising=False)
    lu = PS._factor_preconditioner(saddle())
    assert isinstance(lu, spla.SuperLU), type(lu)


def test_the_default_factorisation_solves_its_matrix_and_a_batch(monkeypatch):
    monkeypatch.delenv("JNO_PEEC_DEVICE_LU", raising=False)
    M = saddle()
    PS._LU_HOLDER["lu"] = PS._factor_preconditioner(M)
    rhs = np.random.default_rng(4).standard_normal((3, M.shape[0])) + 0j
    x = PS._batched_lu(rhs, "N")
    assert np.abs(M @ x.T - rhs.T).max() < 1e-10 * np.abs(rhs).max()


@gpu
def test_the_device_lu_solves_a_batch_of_right_hand_sides(monkeypatch):
    """The callback hands the factorisation an (n, k) batch; it used to assume one vector and raised
    cupy's 'Out shape is mismatched'."""
    pytest.importorskip("cupy")
    pytest.importorskip("nvmath")
    M = saddle()
    dev = PS._DeviceLU(M)
    rhs = np.random.default_rng(5).standard_normal((M.shape[0], 3)) + 0j
    ref = spla.splu(M.tocsc()).solve(rhs)
    got = dev.solve(rhs)
    dev.free()
    assert got.shape == rhs.shape
    # cuDSS pivots statically, so only an approximate inverse is promised: compare loosely
    assert np.linalg.norm(got - ref) <= 1e-6 * np.linalg.norm(ref)
