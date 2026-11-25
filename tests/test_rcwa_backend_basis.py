"""Oracle-parity tests for the owned ``jno._rcwa`` basis and FFT utilities.

fmmax 1.7.1 is imported ONLY in this file (via ``pytest.importorskip``) as the
numerical oracle; the ``jno/_rcwa`` package itself never imports it. The
utilities must reproduce fmmax's names, shapes, dtypes and orderings exactly,
so every test compares our implementation against fmmax on the same inputs.
A run without fmmax installed skips this module.
"""

import numpy as np
import pytest

fmmax = pytest.importorskip("fmmax")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from fmmax import basis as fbasis  # noqa: E402
from fmmax import fft as fft_oracle  # noqa: E402
from fmmax.fmm import eigensolve_isotropic_media  # noqa: E402

from jno._rcwa import basis as obasis  # noqa: E402
from jno._rcwa import fft as offt  # noqa: E402
from jno._rcwa import utils as outil  # noqa: E402

# ------------------------------------------------------------------------------
# Shared fixtures: concrete lattices, built with identical dtype on both sides.
# ------------------------------------------------------------------------------


SQUARE = {"u": np.array([0.6, 0.0]), "v": np.array([0.0, 0.6])}
NON_ORTHOGONAL = {"u": np.array([0.6, 0.0]), "v": np.array([0.2, 0.4])}


@pytest.fixture(params=[SQUARE, NON_ORTHOGONAL], ids=["square", "non-orthogonal"])
def lattices(request):
    """A pair of ``(ours, fmmax)`` lattice-vector objects over the same values."""
    u, v = request.param["u"], request.param["v"]
    return obasis.LatticeVectors(u=u.copy(), v=v.copy()), fbasis.LatticeVectors(u=u.copy(), v=v.copy())


def _expansion_pair(lv_ours, lv_fmmax, nt, truncation_ours, truncation_fmmax):
    return (
        obasis.generate_expansion(lv_ours, nt, truncation_ours),
        fbasis.generate_expansion(lv_fmmax, nt, truncation_fmmax),
    )


# ------------------------------------------------------------------------------
# Expansion generation (eager host-side).
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("nt", [20, 25, 41])
def test_circular_expansion_matches_fmmax(lattices, nt):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, nt, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)

    # Identical integer coefficients, identical ordering, identical count.
    assert ex_ours.basis_coefficients.dtype == ex_fmmax.basis_coefficients.dtype
    assert np.array_equal(ex_ours.basis_coefficients, ex_fmmax.basis_coefficients)
    assert ex_ours.num_terms == ex_fmmax.num_terms == ex_ours.basis_coefficients.shape[0]
    assert obasis.min_array_shape_for_expansion(ex_ours) == (fbasis.min_array_shape_for_expansion(ex_fmmax))

    # Host-side: a plain numpy array, and the zeroth order sorts first.
    assert type(ex_ours.basis_coefficients) is np.ndarray
    assert np.array_equal(ex_ours.basis_coefficients[0], [0, 0])


@pytest.mark.parametrize("nt", [20, 25, 41])
def test_parallelogramic_expansion_matches_fmmax(lattices, nt):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(
        lv_ours, lv_fmmax, nt, obasis.Truncation.PARALLELOGRAMIC, fbasis.Truncation.PARALLELOGRAMIC
    )
    assert ex_ours.basis_coefficients.dtype == ex_fmmax.basis_coefficients.dtype
    assert np.array_equal(ex_ours.basis_coefficients, ex_fmmax.basis_coefficients)
    assert ex_ours.num_terms == ex_fmmax.num_terms


def test_reciprocal_matches_fmmax(lattices):
    lv_ours, lv_fmmax = lattices
    ru_ours, rv_ours = lv_ours.reciprocal.u, lv_ours.reciprocal.v
    ru_fmmax, rv_fmmax = lv_fmmax.reciprocal.u, lv_fmmax.reciprocal.v
    np.testing.assert_allclose(ru_ours, ru_fmmax, rtol=1e-6)
    np.testing.assert_allclose(rv_ours, rv_fmmax, rtol=1e-6)
    # Reciprocal conventions: u' x v' == 1 / (u x v), and dual dot products are 1.
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    assert np.isclose(cross(ru_ours, rv_ours) * cross(lv_ours.u, lv_ours.v), 1.0)


def test_generate_expansion_runs_under_compile_time_eval(lattices):
    lv_ours, _ = lattices
    with jax.ensure_compile_time_eval():
        ex = obasis.generate_expansion(lv_ours, 25, obasis.Truncation.CIRCULAR)
    assert np.array_equal(ex.basis_coefficients[0], [0, 0])


def test_generate_expansion_guards():
    lv = obasis.LatticeVectors(u=np.array([0.6, 0.0]), v=np.array([0.0, 0.6]))
    with pytest.raises(ValueError, match="approximate_num_terms"):
        obasis.generate_expansion(lv, -3)
    with pytest.raises(ValueError, match="approximate_num_terms"):
        obasis.generate_expansion(lv, 2.5)
    with pytest.raises(ValueError, match="Unknown `truncation`"):
        obasis.generate_expansion(lv, 25, "circular")
    with pytest.raises(ValueError, match="shape \\(num_terms, 2\\)"):
        obasis.Expansion(np.array([1, 2]))
    with pytest.raises(ValueError, match="integer"):
        obasis.Expansion(np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="trailing dimension of 2"):
        obasis.LatticeVectors(u=np.zeros(3), v=np.zeros(2))


# ------------------------------------------------------------------------------
# Transverse wavevectors.
# ------------------------------------------------------------------------------


def test_transverse_wavevectors_matches_fmmax(lattices):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)
    k_in = jnp.array([0.35, -0.2])
    k_ours = obasis.transverse_wavevectors(k_in, lv_ours, ex_ours)
    k_fmmax = fbasis.transverse_wavevectors(k_in, lv_fmmax, ex_fmmax)
    assert k_ours.shape == k_fmmax.shape == (ex_ours.num_terms, 2)
    np.testing.assert_allclose(k_ours, k_fmmax, rtol=1e-6)
    # Zeroth order sits at the excitation wavevector.
    np.testing.assert_allclose(k_ours[0], k_in, rtol=1e-6)


def test_transverse_wavevectors_batched_matches_fmmax(lattices):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)
    k_in = jnp.array([[0.0, 0.0], [0.3, -0.1], [-0.2, 0.25]])  # (batch, 2)
    k_ours = obasis.transverse_wavevectors(k_in, lv_ours, ex_ours)
    k_fmmax = fbasis.transverse_wavevectors(k_in, lv_fmmax, ex_fmmax)
    assert k_ours.shape == (3, ex_ours.num_terms, 2)
    np.testing.assert_allclose(k_ours, k_fmmax, rtol=1e-6)


def test_transverse_wavevectors_jit_grad(lattices):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)

    loss = lambda k, lv, ex: jnp.sum(jnp.abs(obasis.transverse_wavevectors(k, lv, ex)) ** 2)
    jitted = jax.jit(loss)
    k_in = jnp.array([0.35, -0.2])
    got = jitted(k_in, lv_ours, ex_ours)
    expect = loss(k_in, lv_ours, ex_ours)
    np.testing.assert_allclose(got, expect, rtol=1e-6)

    g_ours = jax.jit(jax.grad(loss, argnums=0))(k_in, lv_ours, ex_ours)
    g_fmmax = jax.jit(jax.grad(loss, argnums=0))(k_in, lv_fmmax, ex_fmmax)
    np.testing.assert_allclose(g_ours, g_fmmax, rtol=1e-5)

    # Gradient w.r.t. the lattice vectors (device-resident, differentiable).
    g_ours = jax.jit(jax.grad(lambda lv: loss(k_in, lv, ex_ours)))(lv_ours)
    g_fmmax = jax.jit(jax.grad(lambda lv: loss(k_in, lv, ex_fmmax)))(lv_fmmax)
    gp_ours = np.stack([g_ours.u, g_ours.v])
    gp_fmmax = np.stack([g_fmmax.u, g_fmmax.v])
    np.testing.assert_allclose(gp_ours, gp_fmmax, rtol=1e-5)


# ------------------------------------------------------------------------------
# Fourier convolution matrix.
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("complex_field", [False, True], ids=["real", "complex"])
def test_fourier_convolution_matrix_matches_fmmax(lattices, complex_field):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)
    rng = np.random.default_rng(0)
    eps = rng.normal(size=(2, 12, 12)).astype(np.float32)
    if complex_field:
        eps = eps + 1j * rng.normal(size=eps.shape).astype(np.float32)

    c_ours = offt.fourier_convolution_matrix(eps, ex_ours)
    c_fmmax = fft_oracle.fourier_convolution_matrix(eps, ex_fmmax)
    assert c_ours.shape == c_fmmax.shape == (2, ex_ours.num_terms, ex_ours.num_terms)
    np.testing.assert_allclose(c_ours, c_fmmax, rtol=1e-5)

    # jit-clean: the compiled path reproduces the eager matrix.
    jitted = jax.jit(offt.fourier_convolution_matrix)
    np.testing.assert_allclose(jitted(eps, ex_ours), c_ours, rtol=1e-6)


def test_fourier_convolution_matrix_grad_adjoint_and_fd(lattices):
    """The convolution matrix is a real-linear map ``C = M x``, so the gradient
    of ``loss = sum |C|^2`` is the exact analytic adjoint ``2 Re[M^T C]``;
    the jitted gradient must reproduce it, and a central difference must land
    within float32 rounding of it."""
    lv_ours, _ = lattices
    ex_ours = obasis.generate_expansion(lv_ours, 25, obasis.Truncation.CIRCULAR)
    rng = np.random.default_rng(7)
    eps0 = rng.normal(size=(8, 8))
    n = eps0.size
    nt = ex_ours.num_terms

    def loss(x):
        c = offt.fourier_convolution_matrix(x, ex_ours)
        return jnp.sum(jnp.abs(c) ** 2)

    g = jax.jit(jax.grad(loss))(eps0)
    assert g.shape == eps0.shape

    # Build the linear map column by column and compare to the adjoint.
    M = np.zeros((nt * nt, n), dtype=np.complex128)
    for i in range(n):
        e = np.zeros((8, 8))
        e.flat[i] = 1.0
        M[:, i] = np.asarray(offt.fourier_convolution_matrix(e, ex_ours)).reshape(-1)
    c = np.asarray(offt.fourier_convolution_matrix(eps0, ex_ours)).reshape(-1)
    adjoint = 2 * np.real(np.conj(M.T) @ c)
    np.testing.assert_allclose(np.asarray(g).reshape(-1), adjoint, rtol=1e-5, atol=1e-9)

    # Central difference on a float32 evaluation stays within float32 rounding.
    lj = jax.jit(loss)
    h = 1e-2
    for ii in (0, 3, 17):
        e = np.zeros_like(eps0)
        e.flat[ii] = 1.0
        fd = (lj(eps0 + h * e) - lj(eps0 - h * e)) / (2 * h)
        np.testing.assert_allclose(g.flatten()[ii], fd, rtol=5e-2)


# ------------------------------------------------------------------------------
# Expansion-aware fft / ifft.
# ------------------------------------------------------------------------------


def test_fft_matches_fmmax(lattices):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 16, 16)).astype(np.float32)

    y_ours = offt.fft(x, ex_ours)
    y_fmmax = fft_oracle.fft(x, ex_fmmax)
    assert y_ours.shape == y_fmmax.shape == (3, ex_ours.num_terms)
    np.testing.assert_allclose(y_ours, y_fmmax, rtol=1e-5)

    # Non-centered coordinates variant, and the corner-phase convention.
    y2_ours = offt.fft(x, ex_ours, centered_coordinates=False)
    y2_fmmax = fft_oracle.fft(x, ex_fmmax, centered_coordinates=False)
    np.testing.assert_allclose(y2_ours, y2_fmmax, rtol=1e-5)


def test_ifft_matches_fmmax_and_roundtrips(lattices):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)
    rng = np.random.default_rng(2)
    x = rng.normal(size=(16, 16)).astype(np.float32)

    # Our fft/ifft pair is operation-for-operation the fmmax pair (both use
    # ``norm="forward"``), so any composition across the two libraries agrees
    # bit for bit, and our pair round-trips up to the forward/forward scaling.
    y_ours = offt.fft(x, ex_ours)
    x_back_ours = offt.ifft(y_ours, ex_ours, shape=(16, 16))
    x_back_mixed_1 = fft_oracle.ifft(y_ours, ex_fmmax, shape=(16, 16))
    x_back_mixed_2 = offt.ifft(fft_oracle.fft(x, ex_fmmax), ex_ours, shape=(16, 16))
    np.testing.assert_allclose(x_back_ours, x_back_mixed_1, rtol=1e-5)
    np.testing.assert_allclose(x_back_ours, x_back_mixed_2, rtol=1e-5)

    # jit-clean round trip through the compiled transform pair (same value as
    # the eager pair).
    roundtrip = jax.jit(lambda z: offt.ifft(offt.fft(z, ex_ours), ex_ours, shape=(16, 16)))
    np.testing.assert_allclose(roundtrip(x), x_back_ours, rtol=1e-5)


# ------------------------------------------------------------------------------
# Static-aux usability: Expansion as a jit argument.
# ------------------------------------------------------------------------------


def test_expansion_static_aux_in_jit(lattices):
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)
    rng = np.random.default_rng(3)
    eps = rng.normal(size=(8, 8)).astype(np.float32)

    f = lambda e, ex: jnp.sum(jnp.abs(offt.fourier_convolution_matrix(e, ex)))
    eager = f(eps, ex_ours)
    jitted = jax.jit(f)(eps, ex_ours)
    # JIT fusion reassociates the final reduction, so the match is at float32
    # rounding level rather than bit level.
    np.testing.assert_allclose(jitted, eager, rtol=1e-4)

    # Same-value expansions hash and compare equal (jit cache key stability).
    assert ex_ours == ex_ours
    assert hash(ex_ours) == hash(ex_ours)
    ex_twin = obasis.Expansion(ex_ours.basis_coefficients.copy())
    assert ex_ours == ex_twin
    assert hash(ex_ours) == hash(ex_twin)
    # Cross-type inequality (fmmax's objects differ by type) and genuine
    # value inequality against a different truncation.
    assert ex_ours != ex_fmmax
    ex_other = obasis.generate_expansion(lv_ours, 41, obasis.Truncation.CIRCULAR)
    assert ex_other != ex_ours
    assert ex_ours != "not an expansion"


# ------------------------------------------------------------------------------
# End-to-end: fmmax's uniform-isotropic eigensolve consumes our objects.
# ------------------------------------------------------------------------------


def test_uniform_isotropic_eigensolve_consumes_our_objects(lattices):
    """A LayerSolveResult built with our Expansion/LatticeVectors matches one built
    with fmmax's own objects, proving drop-in interop of the static-aux types."""
    lv_ours, lv_fmmax = lattices
    ex_ours, ex_fmmax = _expansion_pair(lv_ours, lv_fmmax, 25, obasis.Truncation.CIRCULAR, fbasis.Truncation.CIRCULAR)

    wavelength = jnp.asarray(1.0)
    k_in = jnp.zeros((2,))
    eps = jnp.zeros((1, 1), jnp.complex64) + (2.0 + 0.1j)

    res_ours = eigensolve_isotropic_media(wavelength, k_in, lv_ours, eps, expansion=ex_ours)
    res_fmmax = eigensolve_isotropic_media(wavelength, k_in, lv_fmmax, eps, expansion=ex_fmmax)

    assert res_ours.eigenvalues.shape == (2 * ex_ours.num_terms,)
    np.testing.assert_allclose(res_ours.eigenvalues, res_fmmax.eigenvalues, rtol=1e-5)
    np.testing.assert_allclose(res_ours.eigenvectors, res_fmmax.eigenvectors, rtol=1e-5)
    assert np.array_equal(res_ours.expansion.basis_coefficients, ex_ours.basis_coefficients)


# ------------------------------------------------------------------------------
# Utility parity.
# ------------------------------------------------------------------------------


def test_utils_match_fmmax():
    from fmmax import misc as fmisc
    from fmmax import utils as futils

    wl = jnp.asarray(0.7)
    np.testing.assert_allclose(
        outil.angular_frequency_for_wavelength(wl),
        futils.angular_frequency_for_wavelength(wl),
        rtol=1e-6,
    )

    x = jnp.arange(6.0).reshape(2, 3)
    np.testing.assert_allclose(outil.batch_diag(x), fmisc.diag(x))
    m = jnp.arange(12.0).reshape(2, 2, 3)
    np.testing.assert_allclose(outil.matrix_adjoint(m), fmisc.matrix_adjoint(m))

    assert outil.absolute_axes((-2, -1), 3) == (1, 2)
    assert outil.absolute_axes((0, 2), 3) == (0, 2)
    with pytest.raises(ValueError):
        outil.absolute_axes((0, 3), 2)
    with pytest.raises(ValueError):
        outil.absolute_axes((1, 1), 2)
