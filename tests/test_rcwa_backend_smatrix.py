"""Tests for ``jno._rcwa.smatrix``: layer S-matrices in one lossy reference medium, doublings, star products and
the ambient closure, against the transfer matrices they replace."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """float64; the flag is process-wide, so it is restored afterwards (see tests/conftest.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


from jno._rcwa import layer, smatrix  # noqa: E402
from jno._rcwa import stack as smod  # noqa: E402

N = 24
_X = (np.arange(N) + 0.5) / N - 0.5
GX, GY = np.meshgrid(_X, _X, indexing="ij")
DISK = np.where(GX**2 + GY**2 < 0.3**2, 6.0, 1.0) + 0.0j


@pytest.fixture
def patterned():
    """A patterned layer's material record, its stack and the reference medium."""
    st = smod.StateStack(0.9, (0.8, 0.7), 1.0, 2.25, [(0.6, DISK)], orders=25, k_in=(0.7, 0.3), dtype=jnp.complex128)
    b, c, _, _ = st._patterned_bc(st._prototype[0], st._omega, st._twv)
    p = b @ c

    def transfer(h):  # the exact transfer over h, its doubling count sized for h itself
        k = layer.series_doublings(layer.matrix_inf_norm(p), h)
        return layer.slice_transfer(b, c, p, h, k, layer.series_degree(jnp.complex128))

    y_r, z_r = smatrix.reference(st._twv, st._omega, jnp.complex128)
    return st, transfer, y_r, z_r


def _full_smatrix(blocks, y_r, z_r):
    """All four blocks of the chunk S-matrix from the dense change of basis (the reference for the O(n^2) path)."""
    a, b, c, d = (np.asarray(x) for x in blocks)
    n = y_r.shape[0]
    y = np.asarray(
        jnp.block([[jnp.diag(y_r[:, 0, 0]), jnp.diag(y_r[:, 0, 1])], [jnp.diag(y_r[:, 1, 0]), jnp.diag(y_r[:, 1, 1])]])
    )
    eye = np.eye(2 * n)
    w = np.block([[eye, eye], [y, -y]])
    tp = np.linalg.solve(w, np.block([[a, b], [c, d]]) @ w)
    t11, t12, t21, t22 = tp[: 2 * n, : 2 * n], tp[: 2 * n, 2 * n :], tp[2 * n :, : 2 * n], tp[2 * n :, 2 * n :]
    s12 = np.linalg.inv(t22)
    s11 = -s12 @ t21
    return s11, s12, t11 + t12 @ s11, t12 @ s12


def test_a_layer_smatrix_is_mirror_symmetric(patterned):
    """``S11 = S22`` and ``S12 = S21`` for a z-invariant layer, and the O(n^2) basis change equals the dense one."""
    st, transfer, y_r, z_r = patterned
    blocks = transfer(0.6 / 8)
    s11, s12, s21, s22 = _full_smatrix(blocks, y_r, z_r)
    scale = np.max(np.abs(s12))
    assert np.max(np.abs(s11 - s22)) <= 1e-12 * scale
    assert np.max(np.abs(s12 - s21)) <= 1e-12 * scale
    r, t = smatrix.chunk_smatrix(blocks, y_r, z_r)
    assert np.max(np.abs(np.asarray(r) - s11)) <= 1e-12 * scale
    assert np.max(np.abs(np.asarray(t) - s12)) <= 1e-12 * scale


@pytest.mark.parametrize("chunks", [2, 3, 8, 13])
def test_doublings_equal_the_smatrix_of_the_whole_thickness(patterned, chunks):
    """``S(h)^c`` (doublings, or the binary expansion for a count that is not a power of two) equals the S-matrix
    built directly from the transfer over ``c h`` (thin enough here for that transfer to be well conditioned)."""
    st, transfer, y_r, z_r = patterned
    h = 0.02
    r, t = smatrix.layer_smatrix(*smatrix.chunk_smatrix(transfer(h), y_r, z_r), chunks)
    r_ref, t_ref = smatrix.chunk_smatrix(transfer(chunks * h), y_r, z_r)
    np.testing.assert_allclose(np.asarray(r), np.asarray(r_ref), rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.asarray(t), np.asarray(t_ref), rtol=0, atol=1e-11)


def test_star_with_the_identity_and_associativity(patterned):
    st, transfer, y_r, z_r = patterned
    n2 = 2 * st._n
    eye, zero = jnp.eye(n2, dtype=jnp.complex128), jnp.zeros((n2, n2), dtype=jnp.complex128)
    ident = (zero, eye, eye, zero)
    ra, ta = smatrix.chunk_smatrix(transfer(0.05), y_r, z_r)
    rb, tb = smatrix.chunk_smatrix(transfer(0.11), y_r, z_r)
    sa, sb = (ra, ta, ta, ra), (rb, tb, tb, rb)
    for got, want in zip(smatrix.star(ident, sa), sa):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-13)
    left = smatrix.star(smatrix.star(sa, sb), sa)
    right = smatrix.star(sa, smatrix.star(sb, sa))
    for got, want in zip(left, right):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), atol=1e-12)


def test_a_thick_evanescent_layer_stays_bounded():
    """Two hundred growth lengths of a patterned layer: the doubled S-matrix stays finite and the stack still
    conserves energy, where a transfer matrix over the same thickness would overflow (e^200)."""
    # FFT (Laurent) factorization: its truncated operator conserves energy exactly for a lossless grid, so T + R
    # is a test of the stack algebra alone (an unconverged vector factorization loses energy per unit thickness).
    st = smod.StateStack(
        0.9, (0.8, 0.7), 1.0, 2.25, [(30.0, DISK)], orders=25, k_in=(0.7, 0.3), formulation="FFT", dtype=jnp.complex128
    )
    assert st._chunks[0] >= 128
    res = st.solve()
    assert np.isfinite(float(res.T)) and np.isfinite(float(res.R))
    assert float(res.T + res.R) == pytest.approx(1.0, abs=1e-9)


def test_block_diagonal_products_match_dense():
    n = 7
    rng = np.random.default_rng(0)
    y = jnp.asarray(rng.standard_normal((n, 2, 2)) + 1j * rng.standard_normal((n, 2, 2)))
    m = jnp.asarray(rng.standard_normal((2 * n, 2 * n)) + 1j * rng.standard_normal((2 * n, 2 * n)))
    dense = jnp.block([[jnp.diag(y[:, 0, 0]), jnp.diag(y[:, 0, 1])], [jnp.diag(y[:, 1, 0]), jnp.diag(y[:, 1, 1])]])
    np.testing.assert_allclose(np.asarray(smatrix.right(m, y)), np.asarray(m @ dense), atol=1e-13)
    np.testing.assert_allclose(np.asarray(smatrix.left(y, m)), np.asarray(dense @ m), atol=1e-13)
    np.testing.assert_allclose(np.asarray(smatrix.per_order(dense, n)), np.asarray(y), atol=0)
    np.testing.assert_allclose(np.asarray(smatrix.right(dense, smatrix.inverse_2x2(y))), np.eye(2 * n), atol=1e-12)


def test_the_reference_medium_has_no_grazing_order():
    """kz of the lossy reference is never 0, even for an order exactly on the vacuum light cone."""
    om = 2 * np.pi
    twv = jnp.asarray([[om, 0.0], [0.0, om], [0.3, 0.1]], dtype=jnp.complex128)
    kz = layer.ambient_kz(twv, smatrix.EPS_REFERENCE, om, dtype=jnp.complex128)
    assert np.min(np.abs(np.asarray(kz))) > 0.5 * om
