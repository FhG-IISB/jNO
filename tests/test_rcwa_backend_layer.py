"""Tests for ``jno._rcwa.layer``: the state-space operator, its eig-free expm,
the uniform-isotropic analytic shortcut, and the ambient split.

All anchors run without fmmax. expm accuracy is anchored against
``jax.scipy.linalg.expm`` (itself eig-based, but used here only as a test
oracle), and the uniform shortcut is anchored against the full block expm and
the analytic uniform spectrum ``eig(M_L) = +-q``.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

from jno._rcwa import basis as obasis
from jno._rcwa import fft as offt
from jno._rcwa import layer, matrices


def _lattice(period=(0.6, 0.6)):
    return obasis.LatticeVectors(u=jnp.asarray([period[0], 0.0]), v=jnp.asarray([0.0, period[1]]))


@pytest.fixture(scope="module")
def expansion():
    return obasis.generate_expansion(_lattice(), approximate_num_terms=85)


@pytest.fixture(scope="module")
def twv(expansion):
    return obasis.transverse_wavevectors(
        in_plane_wavevector=jnp.asarray([0.1, 0.0]),
        primitive_lattice_vectors=_lattice(),
        expansion=expansion,
    ).astype(jnp.complex64)


# -----------------------------------------------------------------------------
# expm: fixed-degree Pade + static scaling-squaring.
# -----------------------------------------------------------------------------


def test_expm_pade_matches_jax_expm():
    """[13/13] Pade with a static squaring count reproduces ``expm`` (c64)."""
    key = jax.random.PRNGKey(0)
    x = (jax.random.normal(key, (12, 12)) + 1j * jax.random.normal(jax.random.PRNGKey(1), (12, 12))).astype(jnp.complex64)
    squarings = layer.squarings_for_spectral_norm(float(np.max(np.abs(x))))
    ours = layer.expm_pade_scaling_squaring(x, squarings)
    ref = jsp.linalg.expm(x)
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-4)


def test_expm_pade_scaling_squaring_static(twv):
    """The squaring count is a static Python int; jit compiles the loop away."""
    key = jax.random.PRNGKey(3)
    x = (jax.random.normal(key, (16, 16)) + 0j).astype(jnp.complex64)
    squarings = 2
    traced = jax.jit(lambda x: layer.expm_pade_scaling_squaring(x, squarings))
    np.testing.assert_allclose(traced(x), layer.expm_pade_scaling_squaring(x, squarings), rtol=1e-6)
    assert isinstance(squarings, int)


def test_squarings_guard_raises():
    with pytest.raises(ValueError, match="squarings"):
        layer.squarings_for_spectral_norm(layer.THETA_13 * 2 ** (layer.MAX_SQUARINGS + 1))


def test_layer_transfer_exp_guard_on_concrete():
    """The eager never-silent guard fires when the bound is exceeded."""
    m = (jnp.eye(6, dtype=jnp.complex64) * 1000).at[0, 1].set(1000)
    with pytest.raises(ValueError, match="expm argument bound"):
        layer.layer_transfer_exp(m, 1.0, squarings=1)


def test_layer_transfer_exp_eager_guard_uses_inf_norm():
    """The eager expm-bound guard must use the INF-norm (the norm of the static
    squaring count, ||.||_max <= ||.||_inf): a matrix whose largest entry stays
    under the Pade budget but whose row-sum exceeds it must still raise -- the
    max-entry check silently steps aside exactly there."""
    n = 8
    m = jnp.zeros((n, n), dtype=jnp.complex64).at[0].set(1.0 + 1j)
    # ||x||_max ~ 1.41 < THETA_13, but ||x||_inf = 8*sqrt(2) ~ 11.3 > THETA_13.
    assert layer.matrix_inf_norm(1j * m) > layer.THETA_13  # the danger is real
    assert float(np.max(np.abs(np.asarray(1j * m)))) < layer.THETA_13  # max-entry misses it
    with pytest.raises(ValueError, match="expm argument bound"):
        layer.layer_transfer_exp(m, 1.0, squarings=0)


def test_layer_transfer_exp_fallback_squarings_from_inf_norm():
    """Without guard_norm/squarings, the eager fallback derives the squaring count
    from the SAME inf-norm as the build-time static count (never the weaker max
    entry norm), and the guard then passes consistently for the row-concentrated
    matrix."""
    n = 8
    m = jnp.zeros((n, n), dtype=jnp.complex64).at[0].set(1.0)
    s = layer.squarings_for_spectral_norm(layer.matrix_inf_norm(1j * m))
    assert s == 1  # ||1j m||_inf = 8 in (THETA_13, 2*THETA_13]
    out = layer.layer_transfer_exp(m, 1.0)
    assert jnp.isfinite(out).all()


# -----------------------------------------------------------------------------
# M_L assembly and the uniform-isotropic analytic shortcut.
# -----------------------------------------------------------------------------


def test_assemble_ml_uniform_spectrum_is_q(expansion, twv):
    """``eig(M_L) = +-q`` with ``q^2 = eps w^2 - |k+G|^2`` (test-only eig)."""
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    eps = 4.0
    z_eps = offt.fourier_convolution_matrix(jnp.full((16, 16), eps, jnp.complex64), expansion)
    e_t = matrices.transverse_permittivity_fft(jnp.full((16, 16), eps, jnp.complex64), expansion)
    m_l = layer.assemble_M_L(omega, twv, z_eps, e_t)
    q2 = eps * omega**2 - (twv[:, 0] ** 2 + twv[:, 1] ** 2)
    expected = jnp.concatenate([jnp.sqrt(q2), jnp.sqrt(q2), -jnp.sqrt(q2), -jnp.sqrt(q2)])
    # The eigensolver may return eigenvalues in any order (and eig itself only
    # reproduces the ±q pairs to rounding), so compare the sorted |lambda| sets:
    # each eigenvalue has modulus |q|, distinct per order for a generic k_in.
    got = np.sort(np.abs(np.asarray(jnp.linalg.eigvals(m_l))))
    want = np.sort(np.abs(np.asarray(expected)))
    err = np.max(np.abs(got - want))
    scale = np.max(want)
    assert err < 1e-3 * max(scale, 1.0)


def _small_twv():
    """A modest truncation (moderate |q d|) for the exactness checks below."""
    lv = _lattice()
    ex = obasis.generate_expansion(lv, approximate_num_terms=25)
    return obasis.transverse_wavevectors(
        in_plane_wavevector=jnp.asarray([0.1, 0.0]),
        primitive_lattice_vectors=lv,
        expansion=ex,
    ).astype(jnp.complex64)


def test_uniform_transfer_matches_block_expm():
    """The analytic shortcut equals expm of the assembled M_L blocks (c64)."""
    twv = _small_twv()
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    eps = 2.5 + 0.3j
    d = 0.23
    n = twv.shape[0]

    # Uniform collapse: an n x n Toeplitz of a constant grid is eps * I.
    z_eps = jnp.eye(n, dtype=jnp.complex64) * eps
    e_t = jnp.eye(2 * n, dtype=jnp.complex64) * eps

    ours = layer.uniform_layer_transfer(omega, twv, eps, d)
    m_l = layer.assemble_M_L(omega, twv, z_eps, e_t)
    squarings = layer.squarings_for_spectral_norm(float(np.max(np.abs(1j * m_l * d))))
    ref = layer.expm_pade_scaling_squaring(1j * m_l * d, squarings)
    np.testing.assert_allclose(ours, ref, rtol=1e-3, atol=1e-3)


def test_uniform_transfer_semigroup():
    """``T(d) == T(d/8)^8`` up to roundoff (same-material chunking is exact)."""
    twv = _small_twv()
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    eps = 4.0
    d = 0.4
    n = twv.shape[0]
    full = layer.uniform_layer_transfer(omega, twv, eps, d)
    chunk = layer.uniform_layer_transfer(omega, twv, eps, d / 8)
    eighth = jnp.eye(4 * n, dtype=jnp.complex64)
    for _ in range(8):
        eighth = eighth @ chunk
    np.testing.assert_allclose(full, eighth, rtol=1e-3, atol=1e-3)


def test_uniform_transfer_finite_and_lossless_det():
    """The analytic transfer is finite and unit-determinant for a lossless layer
    (small truncation, so the determinant stays in complex64 range)."""
    twv = _small_twv()
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    d = 0.2
    t = layer.uniform_layer_transfer(omega, twv, 4.0, d)
    assert jnp.isfinite(t).all()
    sign, logdet = jnp.linalg.slogdet(t)
    assert float(logdet.real) == pytest.approx(0.0, abs=1e-2)


# -----------------------------------------------------------------------------
# Ambient split and readouts.
# -----------------------------------------------------------------------------


def test_ambient_admittance_analytic(twv):
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    kz = layer.ambient_kz(twv, 1.0, omega)
    y = layer.ambient_admittance(twv, kz, omega)
    kx, ky = twv[:, 0], twv[:, 1]
    np.testing.assert_allclose(y[:, 0, 0], (ky**2 / (omega * kz) + kz / omega), rtol=1e-6)
    np.testing.assert_allclose(y[:, 1, 1], (kz / omega + kx**2 / (omega * kz)), rtol=1e-6)
    # the off-diagonal is symmetric
    np.testing.assert_allclose(y[:, 0, 1], y[:, 1, 0], rtol=1e-6)


def test_ambient_admittance_matrix_block_diag(twv):
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    kz = layer.ambient_kz(twv, 1.0, omega)
    ym = layer.ambient_admittance_matrix(twv, kz, omega)
    n = twv.shape[0]
    y = layer.ambient_admittance(twv, kz, omega)
    np.testing.assert_allclose(jnp.diag(ym[:n, :n]), y[:, 0, 0], rtol=1e-6)
    np.testing.assert_allclose(jnp.diag(ym[n:, n:]), y[:, 1, 1], rtol=1e-6)


def test_plane_wave_incident_state(twv):
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    n = twv.shape[0]
    u, v = layer.plane_wave_incident_state((0.0, 1.0), twv, 1.0, omega)
    assert u.shape == (2 * n,)
    # only the zeroth order is excited: u = (-Ey, Ex) = (-1, 0) at order 0
    np.testing.assert_allclose(u[0], -1.0, atol=1e-6)
    np.testing.assert_allclose(u[n], 0.0, atol=1e-6)
    assert jnp.all(u[1:n] == 0) and jnp.all(u[n + 1 :] == 0)
    # v = Y u
    kz = layer.ambient_kz(twv, 1.0, omega)
    y = layer.ambient_admittance_matrix(twv, kz, omega)
    np.testing.assert_allclose(v, y @ u, rtol=1e-6)


def test_sz_flux_convention(twv):
    n = twv.shape[0]
    # u = (-Ey, Ex), v = (Hx, Hy); a forward TE (Ex, Hy) state has positive flux.
    u_f = jnp.zeros(2 * n, jnp.complex64)
    u_f = u_f.at[n].set(1.0)  # u2 = Ex
    v_f = jnp.zeros(2 * n, jnp.complex64)
    v_f = v_f.at[n].set(1.0)  # v2 = Hy (same order-0 slot as u2)
    flux = layer.sz_flux(u_f, v_f, n)
    assert flux[0] > 0
    assert jnp.all(flux[1:] == 0)
    flux_b = layer.sz_flux(u_f, -v_f, n)
    assert flux_b[0] < 0


def test_uniform_kz_branch(twv):
    omega = jnp.asarray(2 * np.pi, jnp.complex64)
    kz = layer.uniform_kz(twv, 1.0, omega)
    # all orders decay or propagate; none should grow in -z (Im(kz) >= 0)
    assert jnp.all(jnp.imag(kz) >= -1e-6)
