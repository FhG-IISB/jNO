"""Tests for ``jno._rcwa.layer``: the state-space operator, its eig-free thin-slice
transfer, the uniform-isotropic analytic shortcut, and the ambient split.

All anchors run without fmmax. The transfer is anchored against
``jax.scipy.linalg.expm`` of the full ``4n`` operator (used here only as a test
oracle), and the uniform shortcut against the same expm and the analytic
uniform spectrum ``eig(M_L) = +-q``.
"""

import contextlib

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
# The thin-slice transfer: series in P = B C plus double-angle steps.
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def _x64():
    """float64 for one block; the flag is process-wide, so it is restored afterwards (see tests/conftest.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _random_bc(m, seed, dtype=jnp.complex128):
    """A non-normal ``(B, C)`` pair: random complex blocks, so ``P = B C`` has no structure to lean on."""
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    rnd = lambda a, b: (jax.random.normal(a, (m, m)) + 1j * jax.random.normal(b, (m, m))) / np.sqrt(m)  # noqa: E731
    return (3.0 * jnp.eye(m) + rnd(k[0], k[1])).astype(dtype), (2.0 * jnp.eye(m) + rnd(k[2], k[3])).astype(dtype)


def _full_transfer(blocks):
    a, b, c, d = (np.asarray(x) for x in blocks)
    return np.block([[a, b], [c, d]])


def _expm_ref(b, c, h):
    z = jnp.zeros_like(b)
    return np.asarray(jsp.linalg.expm(1j * h * jnp.block([[z, b], [c, z]])))


@pytest.mark.parametrize("h", [1e-3, 0.1, 0.7, 2.0])
def test_slice_transfer_equals_the_exponential_of_the_full_operator(h):
    """All four blocks equal ``expm(i M_L h)`` of the ``4n`` operator, in float64, from thin to several growth
    lengths thick, with the build rule's doubling count."""
    with _x64():
        b, c = _random_bc(12, 0)
        p = b @ c
        k = layer.series_doublings(layer.matrix_inf_norm(p), h)
        ours = _full_transfer(layer.slice_transfer(b, c, p, h, k, layer.series_degree(jnp.complex128)))
        ref = _expm_ref(b, c, h)
        assert np.max(np.abs(ours - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_slice_transfer_complex64_reaches_single_precision():
    with _x64():
        b, c = _random_bc(12, 1)
        ref = _expm_ref(b, c, 0.5)
    b64, c64 = b.astype(jnp.complex64), c.astype(jnp.complex64)
    p = b64 @ c64
    k = layer.series_doublings(layer.matrix_inf_norm(p), 0.5)
    ours = _full_transfer(layer.slice_transfer(b64, c64, p, 0.5, k, layer.series_degree(jnp.complex64)))
    assert np.max(np.abs(ours - ref)) <= 2e-6 * np.max(np.abs(ref))


@pytest.mark.parametrize("doublings", [0, 6, 14])
def test_extra_doublings_do_not_amplify_roundoff(doublings):
    """The steps are carried in the ``1 - cos`` form (``s2``), not ``cos(2x) = 2 cos^2 x - 1``: in the cosine form
    each step multiplies the error by about 4, so 14 steps would cost 4^14 ~ 3e8 of float64's 1e-16. Here the
    result stays at roundoff however many steps a thin slice is doubled by."""
    with _x64():
        b, c = _random_bc(10, 2)
        p = b @ c
        h = 0.3
        k = max(doublings, layer.series_doublings(layer.matrix_inf_norm(p), h))
        ours = _full_transfer(layer.slice_transfer(b, c, p, h, k, layer.series_degree(jnp.complex128)))
        ref = _expm_ref(b, c, h)
        assert np.max(np.abs(ours - ref)) <= 1e-13 * np.max(np.abs(ref))


def test_the_doubling_count_is_static_and_the_thickness_traced():
    """One compiled program serves every thickness: ``h`` is an input, ``k`` a Python int."""
    with _x64():
        b, c = _random_bc(10, 3)
        p = b @ c
        k = layer.series_doublings(layer.matrix_inf_norm(p), 0.4)
        deg = layer.series_degree(jnp.complex128)
        f = jax.jit(lambda h: layer.slice_transfer(b, c, p, h, k, deg))
        for h in (0.1, 0.4, 0.8):
            np.testing.assert_allclose(
                _full_transfer(f(h)), _expm_ref(b, c, h), rtol=0, atol=1e-12 * np.max(np.abs(_expm_ref(b, c, h)))
            )
        # d/dh of the transfer is i M_L T(h): autodiff through the series and the doublings
        dA = jax.jacfwd(lambda h: f(h)[0])(0.4)
        z = jnp.zeros_like(b)
        m = jnp.block([[z, b], [c, z]])
        ref = np.asarray(1j * m @ jnp.asarray(_expm_ref(b, c, 0.4)))[:10, :10]
        np.testing.assert_allclose(np.asarray(dA), ref, rtol=0, atol=1e-11 * np.max(np.abs(ref)))


def test_series_doublings_is_the_smallest_count_within_the_bound():
    for p_norm, h in [(1.0, 0.1), (13.2, 1.0), (4e4, 0.3), (2.0, 1e-4)]:
        k = layer.series_doublings(p_norm, h)
        x = lambda k: p_norm * (layer.THICKNESS_MARGIN * h / 2**k) ** 2  # noqa: E731
        assert x(k) <= layer.SERIES_THETA * (1 + 1e-12)
        assert k == 0 or x(k - 1) > layer.SERIES_THETA


def test_series_load_is_a_quarter_at_the_build_thickness():
    """The doubling count is sized for twice the build thickness, so the build itself sits at load <= 1/4 and
    the load reaches 1 only at the margin."""
    with _x64():
        b, c = _random_bc(10, 4)
        p = b @ c
        h = 0.9
        k = layer.series_doublings(layer.matrix_inf_norm(p), h)
        assert float(layer.series_load(p, h, k)) <= 0.25 + 1e-12
        assert float(layer.series_load(p, layer.THICKNESS_MARGIN * h, k)) <= 1.0 + 1e-12


def test_series_doublings_refuses_a_non_finite_operator():
    with pytest.raises(ValueError, match="non-finite"):
        layer.series_doublings(float("inf"), 1.0)


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
    ref = jsp.linalg.expm(1j * m_l * d)
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
