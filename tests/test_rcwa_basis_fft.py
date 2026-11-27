"""The Fourier basis and the Fourier coefficients of ``jno._rcwa``, against their definitions (no other library).

* the reciprocal basis satisfies ``a_i . b_j = delta_ij``;
* the circular truncation keeps exactly the orders inside its circle, symmetric under ``(m, n) -> (-m, -n)``, sorted
  by ``|G|`` with the zeroth order first, in the same order in single and double precision;
* the parallelogramic truncation is the full ``(2M + 1)(2N + 1)`` block;
* Laurent's matrix ``[[eps]]`` equals the direct DFT sum with pixel-centred samples to roundoff (the centring phase
  used to be single precision, which capped it at 1e-8);
* ``fft`` and ``ifft`` invert each other on band-limited grids.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jno._rcwa import basis, fft  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """float64; the flag is process-wide, so it is restored afterwards (see tests/conftest.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


LATTICES = {
    "square": ([0.6, 0.0], [0.0, 0.6]),
    "rect": ([0.8, 0.0], [0.0, 0.7]),
    "skew": ([0.6, 0.0], [0.2, 0.4]),
    "hex": ([1.0, 0.0], [0.5, np.sqrt(3) / 2]),
}


def _lattice(name):
    u, v = LATTICES[name]
    return basis.LatticeVectors(u=jnp.asarray(u), v=jnp.asarray(v))


@pytest.mark.parametrize("name", LATTICES)
def test_reciprocal_basis_is_dual(name):
    lv = _lattice(name)
    rec = lv.reciprocal
    a = np.stack([np.asarray(lv.u), np.asarray(lv.v)])
    b = np.stack([np.asarray(rec.u), np.asarray(rec.v)])
    np.testing.assert_allclose(a @ b.T, np.eye(2), atol=1e-15)


@pytest.mark.parametrize("name", LATTICES)
@pytest.mark.parametrize("count", [9, 25, 121, 441])
def test_circular_truncation_is_the_full_disk_sorted_and_symmetric(name, count):
    lv = _lattice(name)
    c = basis.generate_expansion(lv, count).basis_coefficients
    b = np.stack([np.asarray(lv.reciprocal.u), np.asarray(lv.reciprocal.v)])
    mag = np.linalg.norm(c @ b, axis=-1)
    radius = np.sqrt(count * abs(np.linalg.det(b)) / np.pi)
    # every kept order inside, every order inside kept (search a box that surely holds the disk)
    k = int(np.ceil(radius / np.min(np.linalg.svd(b, compute_uv=False)))) + 1
    mm, nn = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1), indexing="ij")
    grid = np.stack([mm.ravel(), nn.ravel()], -1)
    inside = {tuple(g) for g in grid[np.linalg.norm(grid @ b, axis=-1) < radius]}
    assert set(map(tuple, c)) == inside
    assert tuple(c[0]) == (0, 0)
    assert np.all(np.diff(mag) >= -1e-6 * mag.max())  # sorted up to the tie tolerance
    assert set(map(tuple, c)) == set(map(tuple, -c))
    assert abs(len(c) - count) <= 0.25 * count + 8  # area-matched up to the lattice-point fluctuation


@pytest.mark.parametrize("name", LATTICES)
def test_the_order_does_not_depend_on_the_precision(name):
    """Symmetry-equivalent orders tie exactly; the tie order is the enumeration order, the same in float32."""
    c64 = basis.generate_expansion(_lattice(name), 441).basis_coefficients
    jax.config.update("jax_enable_x64", False)
    u, v = LATTICES[name]
    lv32 = basis.LatticeVectors(u=jnp.asarray(u, jnp.float32), v=jnp.asarray(v, jnp.float32))
    c32 = basis.generate_expansion(lv32, 441).basis_coefficients
    np.testing.assert_array_equal(c32, c64)


@pytest.mark.parametrize("count", [9, 25, 121])
def test_parallelogramic_truncation_is_a_full_block(count):
    c = basis.generate_expansion(_lattice("square"), count, basis.Truncation.PARALLELOGRAMIC).basis_coefficients
    m, n = np.max(np.abs(c), axis=0)
    assert len(c) == (2 * m + 1) * (2 * n + 1)
    assert len(set(map(tuple, c))) == len(c)
    assert tuple(c[0]) == (0, 0)


def test_transverse_wavevectors_are_k_in_plus_2pi_g():
    lv = _lattice("skew")
    ex = basis.generate_expansion(lv, 25)
    k_in = jnp.asarray([0.3, -0.2])
    k = np.asarray(basis.transverse_wavevectors(k_in, lv, ex))
    b = np.stack([np.asarray(lv.reciprocal.u), np.asarray(lv.reciprocal.v)])
    np.testing.assert_allclose(k, np.asarray(k_in) + 2 * np.pi * ex.basis_coefficients @ b, atol=1e-14)


def _direct_coefficient(g, m, n):
    nx, ny = g.shape
    i, j = np.arange(nx)[:, None], np.arange(ny)[None, :]
    return np.sum(g * np.exp(-2j * np.pi * (m * (i + 0.5) / nx + n * (j + 0.5) / ny))) / (nx * ny)


def test_convolution_matrix_is_the_direct_dft_sum():
    rng = np.random.default_rng(0)
    g = 1.0 + 4.0 * (rng.random((24, 20)) > 0.6) + 0.3j * rng.random((24, 20))
    ex = basis.generate_expansion(_lattice("rect"), 49)
    conv = np.asarray(fft.fourier_convolution_matrix(jnp.asarray(g), ex))
    c = ex.basis_coefficients
    ref = np.array([[_direct_coefficient(g, *(ci - cj)) for cj in c] for ci in c])
    assert np.max(np.abs(conv - ref)) <= 1e-15 * np.max(np.abs(ref)) * len(c)


def test_convolution_matrix_of_a_real_grid_is_hermitian():
    rng = np.random.default_rng(1)
    g = 1.0 + 5.0 * (rng.random((32, 32)) > 0.5)
    conv = np.asarray(fft.fourier_convolution_matrix(jnp.asarray(g + 0j), basis.generate_expansion(_lattice("hex"), 121)))
    np.testing.assert_allclose(conv, conv.conj().T, atol=1e-15)


@pytest.mark.parametrize("centered", [True, False])
def test_fft_and_ifft_invert_each_other(centered):
    ex = basis.generate_expansion(_lattice("square"), 49)
    rng = np.random.default_rng(2)
    y = jnp.asarray(rng.standard_normal((3, ex.num_terms)) + 1j * rng.standard_normal((3, ex.num_terms)))
    x = fft.ifft(y, ex, shape=(16, 18), axis=-1, centered_coordinates=centered)
    assert x.shape == (3, 16, 18)
    back = fft.fft(x, ex, axes=(-2, -1), centered_coordinates=centered)
    np.testing.assert_allclose(np.asarray(back), np.asarray(y), atol=1e-14)


def test_a_grid_too_coarse_for_the_orders_raises():
    ex = basis.generate_expansion(_lattice("square"), 121)
    with pytest.raises(ValueError, match="insufficient"):
        fft.fourier_convolution_matrix(jnp.ones((8, 8)), ex)
