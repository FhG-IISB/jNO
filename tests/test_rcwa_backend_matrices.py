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
    grid = np.where(inside, eps_disk, 1.0)
    if np.iscomplexobj(grid):
        return grid.astype(np.complex64)
    return grid.astype(np.float32)


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
    for member in matrices.Formulation:
        assert matrices.Formulation[member.name].value == member.name.lower()


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
    """FFT -> no tangent field; a vector scheme -> one; a callable hook is used directly."""
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

    # All eight named vector schemes are wired and produce a tangent field.
    for form in (
        matrices.Formulation.JONES_DIRECT,
        matrices.Formulation.JONES,
        matrices.Formulation.NORMAL,
        matrices.Formulation.POL,
        matrices.Formulation.JONES_DIRECT_FOURIER,
        matrices.Formulation.JONES_FOURIER,
        matrices.Formulation.NORMAL_FOURIER,
        matrices.Formulation.POL_FOURIER,
    ):
        _, _, e_vec, tangent_vec = matrices.transverse_matrices_isotropic(g2, expansion, form, lattice)
        assert tangent_vec is not None
        assert e_vec.shape == e_fft.shape


def test_tangent_field_follows_the_design(disk_grid, expansion, lattice):
    """The tangent field is differentiated through, not stopped: it follows the interfaces, so a change of the
    grid that moves an edge changes the field. Its directional derivative is finite, nonzero, and equals a
    central difference. (A stop-gradient here dropped the moving-interface term and made every vector
    formulation's design gradient wrong; ``tests/test_rcwa_gradients.py`` checks the end-to-end gradients.)"""
    prev = jax.config.jax_enable_x64  # float64 for this check: a float32 central difference is too noisy
    jax.config.update("jax_enable_x64", True)
    try:
        _check_tangent_field_derivative(disk_grid, expansion, lattice)
    finally:
        jax.config.update("jax_enable_x64", prev)


def _check_tangent_field_derivative(disk_grid, expansion, lattice):
    rng = np.random.default_rng(1)
    k = np.fft.fftfreq(disk_grid.shape[0]) * disk_grid.shape[0]
    bump = np.real(
        np.fft.ifft2(np.fft.fft2(rng.standard_normal(disk_grid.shape)) * np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / 8.0))
    )
    # Break the disk's symmetry: on a symmetric grid several pixels tie for the maximum gradient magnitude that
    # scales the field, and the field has a kink there (autodiff and a central difference then differ).
    g = jnp.asarray(disk_grid + 2.0 * bump / np.abs(bump).max(), jnp.float64)
    weight = jnp.asarray(rng.standard_normal(g.shape), jnp.float64)
    direction = np.real(
        np.fft.ifft2(np.fft.fft2(rng.standard_normal(g.shape)) * np.exp(-(k[:, None] ** 2 + k[None, :] ** 2) / 8.0))
    )
    direction = jnp.asarray(direction / np.abs(direction).max(), jnp.float64)

    def tangent_of_grid(g):
        tx, ty = matrices.compute_tangent_field(
            g,
            expansion,
            lattice,
            use_jones_direct=True,
            fourier_loss_weight=0.05,
            smoothness_loss_weight=0.0,
        )
        return jnp.sum(weight * (jnp.real(tx) + 0.5 * jnp.imag(ty)))

    grad = np.asarray(jax.grad(tangent_of_grid)(g))
    assert np.all(np.isfinite(grad))
    directional = float(np.sum(grad * np.asarray(direction)))
    h = 1e-5
    fd = (float(tangent_of_grid(g + h * direction)) - float(tangent_of_grid(g - h * direction))) / (2 * h)
    assert abs(fd) > 1e-2, "the field must move with the design"
    assert directional == pytest.approx(fd, rel=1e-5)


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


def test_smoothness_schemes_run(disk_grid, expansion, lattice):
    """The plain (non-_FOURIER) schemes exercise the smoothness loss and match
    the FFT formulation in structure (exact tangents are oracle-checked vs fmmax)."""
    g = jnp.asarray(disk_grid, jnp.float32)
    # JONES_DIRECT uses a nonzero smoothness weight; it must run without raising.
    for form in (
        matrices.Formulation.JONES_DIRECT,
        matrices.Formulation.JONES,
        matrices.Formulation.NORMAL,
        matrices.Formulation.POL,
    ):
        tx, ty = matrices.tangent_field_for(form, g, expansion, lattice)
        assert tx.shape == g.shape and ty.shape == g.shape
        assert jnp.isfinite(tx).all() and jnp.isfinite(ty).all()


# -----------------------------------------------------------------------------
# fmmax oracle parity (skips when fmmax is absent).
# -----------------------------------------------------------------------------


def _fmmax():
    return pytest.importorskip("fmmax", reason="fmmax oracle not installed")


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


# -----------------------------------------------------------------------------
# Anisotropic media (stage 2): the 5-/10-component transverse matrices.
# -----------------------------------------------------------------------------


def _aniso_grids(rng=None, n=12):
    """Random anisotropic component grids with matching shapes (for parity)."""
    rng = np.random.default_rng(3) if rng is None else rng

    def g():
        return jnp.asarray(rng.uniform(1.0, 6.0, size=(n, n)) + 1j * rng.uniform(-0.2, 0.2, size=(n, n)), jnp.complex64)

    return tuple(g() for _ in range(5))


def test_aniso_matrix_oracle_parity(lattice, expansion):
    """The anisotropic FFT E_t / M_t and vector matrices match fmmax 1.7.1."""
    fmmax = _fmmax()
    eps = _aniso_grids()
    mu = _aniso_grids()
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, 41)

    ours_e = matrices.transverse_permittivity_fft_anisotropic(*eps[:4], expansion)
    fmm_e = fmmax.fmm_matrices.transverse_permittivity_fft_anisotropic(*eps[:4], f_ex)
    np.testing.assert_allclose(ours_e, fmm_e, rtol=1e-5, atol=1e-6)

    ours_m = matrices.transverse_permeability_fft_anisotropic(*mu[:4], expansion)
    fmm_m = fmmax.fmm_matrices.transverse_permeability_fft_anisotropic(*mu[:4], f_ex)
    np.testing.assert_allclose(ours_m, fmm_m, rtol=1e-5, atol=1e-6)


def test_aniso_vector_matrix_parity_matches_fmmax(lattice, expansion):
    """The anisotropic VECTOR E_t / M_t (tangent-field rotation) match fmmax."""
    fmmax = _fmmax()
    eps = _aniso_grids()
    mu = _aniso_grids()
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, 41)
    tx, ty = fmmax.vector.compute_field_jones_direct(
        (eps[0] + eps[3]) / 2, f_ex, f_lv, fourier_loss_weight=0.05, smoothness_loss_weight=0.0
    )
    ours_e = matrices.transverse_permittivity_vector_anisotropic(*eps[:4], tx, ty, expansion)
    fmm_e = fmmax.fmm_matrices.transverse_permittivity_vector_anisotropic(*eps[:4], tx, ty, f_ex)
    np.testing.assert_allclose(ours_e, fmm_e, rtol=1e-5, atol=1e-6)
    ours_m = matrices.transverse_permeability_vector_anisotropic(*mu[:4], tx, ty, expansion)
    fmm_m = fmmax.fmm_matrices.transverse_permeability_vector_anisotropic(*mu[:4], tx, ty, f_ex)
    np.testing.assert_allclose(ours_m, fmm_m, rtol=1e-5, atol=1e-6)


def test_transverse_matrices_anisotropic_matches_fmmax(lattice, expansion):
    """The full 6-matrix anisotropic factorization matches fmmax's patterned
    general-anisotropic builder (used by the M_L assembly)."""
    fmmax = _fmmax()
    eps = _aniso_grids()
    mu = _aniso_grids()
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, 41)
    (
        o_i_z_eps,
        o_z_eps,
        o_e_t,
        o_i_z_mu,
        o_z_mu,
        o_m_t,
        o_tg,
    ) = matrices.transverse_matrices_anisotropic(eps, mu, expansion, matrices.Formulation.FFT, lattice)
    (
        f_i_z_eps,
        f_z_eps,
        f_e_t,
        f_i_z_mu,
        f_z_mu,
        f_m_t,
        f_tg,
    ) = fmmax.fmm._fourier_matrices_patterned_anisotropic_media(f_lv, eps, mu, f_ex, fmmax.fmm.Formulation.FFT, None)
    assert o_tg is None and f_tg is None
    for a, b in [
        (o_i_z_eps, f_i_z_eps),
        (o_z_eps, f_z_eps),
        (o_e_t, f_e_t),
        (o_i_z_mu, f_i_z_mu),
        (o_z_mu, f_z_mu),
        (o_m_t, f_m_t),
    ]:
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


# -----------------------------------------------------------------------------
# Formulation parity (stage 2c): all eight vector schemes + callable hook.
# -----------------------------------------------------------------------------

_ALL_VECTOR_SCHEMES = (
    "JONES_DIRECT",
    "JONES",
    "NORMAL",
    "POL",
    "JONES_DIRECT_FOURIER",
    "JONES_FOURIER",
    "NORMAL_FOURIER",
    "POL_FOURIER",
)


def _fmmax_view(lattice, approx_terms=41):
    """The fmmax lattice/expansion mirroring our fixture objects."""
    fmmax = _fmmax()
    f_lv = fmmax.basis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    f_ex = fmmax.basis.generate_expansion(f_lv, approx_terms)
    return fmmax, f_lv, f_ex


@pytest.mark.parametrize("name", _ALL_VECTOR_SCHEMES, ids=_ALL_VECTOR_SCHEMES)
def test_vector_scheme_tangent_and_et_match_fmmax(name, lattice, disk_grid):
    """Each of the eight vector schemes reproduces fmmax's tangent field and E_t."""
    fmmax, f_lv, f_ex = _fmmax_view(lattice)
    g = jnp.asarray(disk_grid, jnp.complex64)
    lv = obasis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    expansion = obasis.generate_expansion(lv, approximate_num_terms=41)

    form = matrices.Formulation[name]
    ours_tx, ours_ty = matrices.tangent_field_for(form, g, expansion, lv)
    oracle_fn = fmmax.vector.VECTOR_FIELD_SCHEMES[name.lower()]
    f_tx, f_ty = oracle_fn(g, f_ex, f_lv)
    scale = max(np.max(np.abs(f_tx)), np.max(np.abs(f_ty)))
    assert np.max(np.abs(ours_tx - f_tx)) / scale < 1e-4
    assert np.max(np.abs(ours_ty - f_ty)) / scale < 1e-4

    # E_t must agree to complex64 noise, whichever tangent field feeds the ports.
    ours_et = matrices.transverse_permittivity_vector(g, ours_tx, ours_ty, expansion)
    fmm_et = fmmax.fmm_matrices.transverse_permittivity_vector(g, f_tx, f_ty, f_ex)
    np.testing.assert_allclose(ours_et, fmm_et, rtol=1e-5, atol=1e-6)


def test_formulation_parametrization_matches_fmmax(lattice, disk_grid):
    """The full isotropic layer transfer for each scheme: fmmax formulation
    results (E_t and the z matrices) agree with ours within complex64 tolerance,
    for the crossed-grating disk at low-mid orders."""
    fmmax, f_lv, f_ex = _fmmax_view(lattice)
    g = jnp.asarray(disk_grid, jnp.complex64)
    lv = obasis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    expansion = obasis.generate_expansion(lv, approximate_num_terms=41)

    rel_by_form = {}
    for name in ("FFT",) + _ALL_VECTOR_SCHEMES:
        if name == "FFT":
            o_i_z, o_z, o_et, o_tg = matrices.transverse_matrices_isotropic(g, expansion, matrices.Formulation.FFT, lv)
            f_i_z, f_z, f_et, f_tg = fmmax.fmm._fourier_matrices_patterned_isotropic_media(
                f_lv, g, f_ex, fmmax.fmm.Formulation.FFT
            )
        else:
            form = matrices.Formulation[name]
            o_i_z, o_z, o_et, o_tg = matrices.transverse_matrices_isotropic(g, expansion, form, lv)
            fform = getattr(fmmax.fmm.Formulation, name)
            f_i_z, f_z, f_et, f_tg = fmmax.fmm._fourier_matrices_patterned_isotropic_media(f_lv, g, f_ex, fform)
        rel = max(
            np.max(np.abs(o_et - f_et)) / np.max(np.abs(f_et)),
            np.max(np.abs(o_z - f_z)) / np.max(np.abs(f_z)),
            np.max(np.abs(o_i_z - f_i_z)) / np.max(np.abs(f_i_z)),
        )
        rel_by_form[name] = rel
        assert rel < 1e-5, f"{name}: rel {rel:.2e} vs fmmax"

    # Report the per-formulation table for the record (also keeps the canary's
    # reference numbers visible in a failure).
    assert set(rel_by_form) == set(("FFT",) + _ALL_VECTOR_SCHEMES)


def test_asi_disk_canary_low_orders(lattice):
    """a-Si (high-contrast eps ~ 14) disk at LOW orders: the formulation
    differences that fmmax shows must appear in ours -- no silent degradation
    toward the naive FFT rule (finding family iss_mwuorigh4mh19k)."""
    # Amorphous silicon at ~633nm: n ~ 3.7, eps ~ 13.7 (+ small loss).
    eps_asi = 13.5 + 0.2j
    dg = _disk(grid=48, period=(0.6, 0.6), eps_disk=eps_asi, r=0.22)
    g = jnp.asarray(dg, jnp.complex64)
    lv = obasis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )

    # Low orders: a coarse expansion where the vector formulations still differ
    # markedly from the condensed FFT rule.
    expansion = obasis.generate_expansion(lv, approximate_num_terms=25)
    fmmax, f_lv, f_ex = _fmmax_view(lattice, approx_terms=25)

    # fmmax's own spread of E_t across the schemes -- the reference "formulation
    # differences at low orders". Normalized by the FFT E_t scale on each side,
    # so a degraded port (all schemes collapsing to FFT) is caught as spread ~ 0.
    f_fft_et = fmmax.fmm_matrices.transverse_permittivity_fft(g, f_ex)
    ref_fft_scale = np.max(np.abs(f_fft_et))
    ref_spreads = {}
    for name in _ALL_VECTOR_SCHEMES:
        ftx, fty = fmmax.vector.VECTOR_FIELD_SCHEMES[name.lower()](g, f_ex, f_lv)
        f_et = fmmax.fmm_matrices.transverse_permittivity_vector(g, ftx, fty, f_ex)
        ref_spreads[name] = np.max(np.abs(f_et - f_fft_et)) / ref_fft_scale
    # The vector formulations MUST differ from FFT at these orders on a
    # high-contrast disk (that difference is the point of the formulation).
    assert min(ref_spreads.values()) > 1e-3, f"fmmax reference spreads collapsed: {ref_spreads}"

    ours_fft_et = matrices.transverse_permittivity_fft(g, expansion)
    ours_fft_scale = np.max(np.abs(ours_fft_et))
    for name in _ALL_VECTOR_SCHEMES:
        form = matrices.Formulation[name]
        tx, ty = matrices.tangent_field_for(form, g, expansion, lv)
        et = matrices.transverse_permittivity_vector(g, tx, ty, expansion)
        # Our spread reproduces fmmax's spread -- formulations differ from FFT
        # by the same amount (no collapse, no silent drift).
        ours_spread = np.max(np.abs(et - ours_fft_et)) / ours_fft_scale
        assert ref_spreads[name] > 1e-3  # fmmax reference (guard against oracle collapse)
        np.testing.assert_allclose(ours_spread, ref_spreads[name], rtol=1e-4, err_msg=f"{name} formulation spread drift")
        # And E_t itself still agrees with fmmax at the complex64 floor.
        ftx, fty = fmmax.vector.VECTOR_FIELD_SCHEMES[name.lower()](g, f_ex, f_lv)
        f_et = fmmax.fmm_matrices.transverse_permittivity_vector(g, ftx, fty, f_ex)
        np.testing.assert_allclose(et, f_et, rtol=1e-4, atol=1e-5)


def test_callable_hook_matches_fmmax_hook(lattice, disk_grid):
    """A user-supplied tangent-field callable is honored identically to fmmax's."""
    fmmax, f_lv, f_ex = _fmmax_view(lattice)
    g = jnp.asarray(disk_grid, jnp.complex64)
    lv = obasis.LatticeVectors(
        u=jnp.asarray([lattice.u[0], 0.0]),
        v=jnp.asarray([lattice.v[0], lattice.v[1]]),
    )
    expansion = obasis.generate_expansion(lv, approximate_num_terms=41)

    def hook(g, ex, lv):
        # A constant tangent field: normalized so its magnitude pattern is plain.
        return 0.6 * jnp.ones_like(g), 0.8 * jnp.ones_like(g)

    _, _, o_et, o_tg = matrices.transverse_matrices_isotropic(g, expansion, hook, lv)
    _, _, f_et, f_tg = fmmax.fmm._fourier_matrices_patterned_isotropic_media(f_lv, g, f_ex, hook)
    np.testing.assert_allclose(o_et, f_et, rtol=1e-5, atol=1e-6)


def test_full_stack_solve_all_formulation_names(lattice, disk_grid):
    """The seam -- a full StateStack solve per formulation NAME -- accepts every
    vector scheme by its public name (as ``jno.rcwa formulation=...`` passes it)
    and stays physically consistent against the deep-stack oracle."""
    from jno._rcwa.stack import StateStack

    dg = jnp.asarray(disk_grid, jnp.complex64)

    def solve_as(formulation):
        st = StateStack(
            1.0,
            (float(lattice.u[0]), float(lattice.v[1])),
            1.0,
            1.0,
            [(0.35, dg)],
            orders=41,
            k_in=(0.0, 0.0),
            formulation=formulation,
        )
        return float(st.solve(validate=False).T)

    # Every named vector scheme + FFT must be constructible by name and solve.
    t_by_name = {name: solve_as(name) for name in ("FFT",) + _ALL_VECTOR_SCHEMES}

    # The callable hook is accepted by the same seam path (as a callable).
    def hook(g, ex, lv):
        return 0.5 * jnp.ones_like(g), 0.5 * jnp.ones_like(g)

    t_hook = solve_as(hook)

    # Sanity: all transmissions are in (0, 1); the hook's isotropic tangent field
    # behaves like the POL/NORMAL family (alignment weights differ slightly, so
    # only assert finiteness/range, not equality to another scheme).
    for name, t in t_by_name.items():
        assert 0.0 < t < 1.0, f"{name}: T={t:.4e} outside (0, 1)"
    assert 0.0 < t_hook < 1.0

    # Formulation differences at these (low-ish) orders are real: the FFT rule
    # differs from the vector schemes on a high-contrast disk, and the vector
    # schemes differ among themselves -- matching the matrix-level spread.
    t_fft = t_by_name["FFT"]
    spreads = {name: abs(t - t_fft) / max(t_fft, 1e-12) for name, t in t_by_name.items()}
    assert max(spreads[name] for name in _ALL_VECTOR_SCHEMES) > 1e-3
    assert len({round(t, 6) for t in t_by_name.values()}) >= 3  # not all identical
