"""Tests for ``jno._rcwa.matrices``: Fourier-factorization matrices.

The fmmax-independent anchors run always (Toeplitz/uniform structure, jit- and
grad-cleanliness, formulation dispatch, grad-decoupling of the tangent field);
the fmmax-oracle tests compare our matrices against fmmax 1.7.1 numerically and
skip when fmmax is not installed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno._rcwa import basis as obasis
from jno._rcwa import fft as offt
from jno._rcwa import matrices

# -----------------------------------------------------------------------------
# Shared fixtures.
# -----------------------------------------------------------------------------


SQUARE = (0.6, 0.6)
NON_ORTHOGONAL = (0.6, 0.4)


def _lattice(period, dtype=jnp.complex64):
    return obasis.LatticeVectors(u=jnp.asarray([period[0], 0.0]), v=jnp.asarray([period[1] * 0.7, period[1]]))


def _disk(grid=32, period=(0.6, 0.6), eps_disk=11.0, r=0.2):
    x = (np.arange(grid) + 0.5) / grid
    xx, yy = np.meshgrid(x, x, indexing="ij")
    inside = (xx - 0.5) ** 2 + (yy - 0.5) ** 2 < r**2
    return np.where(inside, eps_disk, 1.0).astype(np.float32)


@pytest.fixture(params=[SQUARE, NON_ORTHOGONAL], ids=["square", "non-orthogonal"])
def lattice(request):
    return _lattice(request.param)


@pytest.fixture
def disk_grid():
    return _disk()


@pytest.fixture
def expansion(lattice):
    return obasis.generate_expansion(lattice, approximate_num_terms=41)


# -----------------------------------------------------------------------------
# Anchors (no fmmax needed).
# -----------------------------------------------------------------------------


def test_formulation_names_match_seam():
    """The Formulation names (string lookup) match what ``jno.rcwa`` passes."""
    for name in ("FFT", "JONES_DIRECT_FOURIER", "JONES", "NORMAL", "POL"):
        assert matrices.Formulation[name].value == name.lower()
    assert matrices.Formulation["JONES_DIRECT_FOURIER"].value == "jones_direct_fourier"


def test_fft_uniform_grid_collapses_to_scalar(expansion, lattice):
    """A constant ``(ng, ng)`` grid gives ``E_t = eps * I`` (the exact limit)."""
    eps = 4.5
    grid = jnp.full((32, 32), eps, dtype=jnp.complex64)
    e_t = matrices.transverse_permittivity_fft(grid, expansion)
    n = expansion.num_terms
    assert e_t.shape == (2 * n, 2 * n)
    np.testing.assert_allclose(e_t, eps * np.eye(2 * n), rtol=1e-6, atol=1e-6)


def test_toeplitz_zero_order_is_mean(expansion, lattice):
    """The (0,0) entry of the FFT convolution matrix is the spatial mean."""
    grid = _disk(24)
    conv = offt.fourier_convolution_matrix(jnp.asarray(grid, jnp.complex64), expansion)
    i0 = int(np.where(np.all(expansion.basis_coefficients == [0, 0], axis=-1))[0][0])
    np.testing.assert_allclose(conv[i0, i0], np.mean(grid), rtol=1e-5)


def test_k_matrix_patterned_closed_form(expansion, lattice):
    """Nonmagnetic ``K`` equals the explicit ``K_E^T K_E`` dyadic."""
    twv = obasis.transverse_wavevectors(jnp.asarray([0.0, 0.0]), lattice, expansion).astype(jnp.complex64)
    k_mat = matrices.k_matrix_patterned(twv)
    kx = jnp.diag(twv[:, 0])
    ky = jnp.diag(twv[:, 1])
    expected = jnp.block([[kx @ kx, kx @ ky], [ky @ kx, ky @ ky]])
    np.testing.assert_allclose(k_mat, expected, rtol=1e-6)


def test_script_k_is_hermitian_patterned(expansion, lattice):
    """``script_K`` is hermitian for real frequencies and real eps."""
    grid = jnp.asarray(_disk(24), jnp.complex64)
    z_eps = offt.fourier_convolution_matrix(grid, expansion)
    twv = obasis.transverse_wavevectors(jnp.asarray([0.2, -0.1]), lattice, expansion).astype(jnp.complex64)
    s = matrices.script_k_matrix_patterned(z_eps, twv)
    # Exact symmetry up to the complex64 inverse/solve noise floor (~1e-4).
    np.testing.assert_allclose(s, np.conj(s.T), rtol=1e-4, atol=1e-4)


def test_formulation_dispatch_and_hook(disk_grid, expansion, lattice):
    """FFT -> no tangent field; JONES -> one; a callable hook is used directly."""
    g2 = jnp.asarray(disk_grid, jnp.complex64)
    _, _, e_fft, tangent_fft = matrices.transverse_matrices_isotropic(g2, expansion, matrices.Formulation.FFT, lattice)
    assert tangent_fft is None

    _, _, e_jon, tangent_jon = matrices.transverse_matrices_isotropic(
        g2, expansion, matrices.Formulation.JONES_DIRECT_FOURIER, lattice
    )
    assert tangent_jon is not None

    def hook(g, ex, lv):
        return jnp.ones_like(g) * 0.5, jnp.ones_like(g) * 0.3

    _, _, e_hook, tangent_hook = matrices.transverse_matrices_isotropic(g2, expansion, hook, lattice)
    assert tangent_hook is not None
    tx, ty = tangent_hook
    np.testing.assert_allclose(tx, 0.5 * np.ones_like(g2))
    np.testing.assert_allclose(ty, 0.3 * np.ones_like(g2))

    # Unimplemented schemes raise loudly (stage 2 wires them).
    with pytest.raises(NotImplementedError, match="stage 2"):
        matrices.transverse_matrices_isotropic(g2, expansion, matrices.Formulation.JONES, lattice)


def test_tangent_field_is_grad_decoupled(disk_grid, expansion, lattice):
    """The tangent field is a stop-gradient constant of the density."""
    g = jnp.asarray(disk_grid, jnp.float32)

    def tangent_of_grid(g):
        tx, _ = matrices.compute_tangent_field(
            g,
            expansion,
            lattice,
            use_jones_direct=True,
            fourier_loss_weight=0.05,
            smoothness_loss_weight=0.0,
        )
        return jnp.sum(jnp.abs(tx))

    grad = jax.grad(tangent_of_grid)(g)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_transverse_matrices_jit_clean(disk_grid, expansion, lattice):
    """The whole factorization compiles under jit and matches the eager run."""
    g = jnp.asarray(disk_grid, jnp.complex64)
    fn = jax.jit(
        matrices.transverse_matrices_isotropic,
        static_argnames=("formulation",),
    )
    eager = matrices.transverse_matrices_isotropic(g, expansion, matrices.Formulation.FFT, lattice)
    jitted = fn(g, expansion, matrices.Formulation.FFT, lattice)
    for a, b in zip(eager[:3], jitted[:3]):
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_et_grad_flows_through_fft_formulation(disk_grid, expansion, lattice):
    """jacrev of ``E_t`` (FFT) w.r.t. the grid is finite and nonzero."""
    g = jnp.asarray(disk_grid, jnp.float32)

    def loss(g):
        e_t = matrices.transverse_permittivity_fft(g, expansion)
        return jnp.real(jnp.sum(e_t))

    grad = jax.grad(loss)(g)
    assert jnp.any(grad != 0)
    assert jnp.isfinite(grad).all()


def test_smoothness_weight_guard(disk_grid, expansion, lattice):
    with pytest.raises(NotImplementedError, match="smoothness"):
        matrices.compute_tangent_field(
            jnp.asarray(disk_grid, jnp.float32),
            expansion,
            lattice,
            use_jones_direct=False,
            fourier_loss_weight=0.0,
            smoothness_loss_weight=1.0,
        )


# -----------------------------------------------------------------------------
# fmmax oracle parity (skips when fmmax is absent).
# -----------------------------------------------------------------------------


def _fmmax():
    import fmmax

    return fmmax


def test_fft_convolution_matrix_matches_fmmax(disk_grid, expansion, lattice):
    """z-eps Toeplitz matrix: bit-level (float32) parity with fmmax."""
    fmmax = _fmmax()
    g = jnp.asarray(disk_grid, jnp.complex64)
    ours = offt.fourier_convolution_matrix(g, expansion)
    lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(lv, 41)
    fmm = fmmax.fft.fourier_convolution_matrix(g, f_ex)
    np.testing.assert_allclose(ours, fmm, rtol=1e-6)


def test_tangent_field_matches_fmmax_jones_direct(disk_grid, expansion, lattice):
    """The JONES_DIRECT_FOURIER tangent field matches fmmax's (rel ~1e-4)."""
    fmmax = _fmmax()
    g = jnp.asarray(disk_grid, jnp.complex64)
    lv = obasis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, 41)

    tx, ty = matrices.compute_tangent_field(
        g,
        expansion,
        lv,
        use_jones_direct=True,
        fourier_loss_weight=0.05,
        smoothness_loss_weight=0.0,
    )
    ftx, fty = fmmax.vector.compute_field_jones_direct(g, f_ex, f_lv, fourier_loss_weight=0.05, smoothness_loss_weight=0.0)
    scale = np.max(np.abs(ftx))
    assert np.max(np.abs(tx - ftx)) / scale < 1e-4
    assert np.max(np.abs(ty - fty)) / scale < 1e-4


def test_et_vector_matches_fmmax(disk_grid, expansion, lattice):
    """Vector E_t (with the fmmax-computed field) matches fmmax's E_t."""
    fmmax = _fmmax()
    g = jnp.asarray(disk_grid, jnp.complex64)
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, 41)
    ftx, fty = fmmax.vector.compute_field_jones_direct(g, f_ex, f_lv, fourier_loss_weight=0.05, smoothness_loss_weight=0.0)
    ours = matrices.transverse_permittivity_vector(g, ftx, fty, expansion)
    fmm = fmmax.fmm_matrices.transverse_permittivity_vector(g, ftx, fty, f_ex)
    np.testing.assert_allclose(ours, fmm, rtol=1e-5, atol=1e-6)


def test_script_k_and_k_match_fmmax(disk_grid, expansion, lattice):
    fmmax = _fmmax()
    g = jnp.asarray(disk_grid, jnp.complex64)
    lv = obasis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, 41)
    twv = obasis.transverse_wavevectors(jnp.asarray([0.1, 0.0]), lv, expansion).astype(jnp.complex64)
    f_twv = fmmax.basis.transverse_wavevectors(jnp.asarray([0.1, 0.0]), f_lv, f_ex).astype(jnp.complex64)

    z_eps = offt.fourier_convolution_matrix(g, expansion)
    f_z_eps = fmmax.fft.fourier_convolution_matrix(g, f_ex)
    np.testing.assert_allclose(
        matrices.script_k_matrix_patterned(z_eps, twv),
        fmmax.fmm_matrices.script_k_matrix_patterned(f_z_eps, f_twv),
        rtol=1e-5,
        atol=1e-6,
    )
    ns = len(twv)
    np.testing.assert_allclose(
        matrices.k_matrix_patterned(twv),
        fmmax.fmm_matrices.k_matrix_patterned(jnp.eye(ns, dtype=jnp.complex64), f_twv),
        rtol=1e-5,
        atol=1e-6,
    )


def test_omega_script_k_matches_fmmax(disk_grid, expansion, lattice):
    fmmax = _fmmax()
    g = jnp.asarray(disk_grid, jnp.complex64)
    lv = obasis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, 41)
    twv = obasis.transverse_wavevectors(jnp.asarray([0.0, 0.0]), lv, expansion).astype(jnp.complex64)
    f_twv = fmmax.basis.transverse_wavevectors(jnp.asarray([0.0, 0.0]), f_lv, f_ex).astype(jnp.complex64)
    z_eps = offt.fourier_convolution_matrix(g, expansion)
    f_z_eps = fmmax.fft.fourier_convolution_matrix(g, f_ex)
    n = expansion.num_terms
    omega = jnp.asarray(2 * np.pi, jnp.complex64)

    ours = matrices.omega_script_k_matrix(omega, z_eps, jnp.eye(2 * n, dtype=jnp.complex64), twv)
    fmm = fmmax.fmm_matrices.omega_script_k_matrix_patterned(
        jnp.asarray(1.0),
        f_z_eps,
        jnp.eye(2 * n, dtype=jnp.complex64),
        f_twv,
    )
    np.testing.assert_allclose(ours, fmm, rtol=1e-5, atol=1e-6)
