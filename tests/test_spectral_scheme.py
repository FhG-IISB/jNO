"""``scheme="spectral"`` — FFT differentiation along a uniform grid axis.

The claim spectral makes and finite differences cannot: on a band-limited periodic field the
derivative is **exact**, not merely high-order. Measured here rather than asserted — and the same
comparison is what caught the one real bug during development (jNO's structured grids span the
interval inclusive of both ends, so the last node duplicates the first; without dropping it the
transform assumes a period of `(n+1)h` and is *worse* than a 2nd-order stencil).

It also reaches where automatic differentiation cannot: a field with no analytic dependence on the
coordinate (a stored tensor, an operator's output) differentiates fine here, because the FFT works
on the values along the grid axis rather than on a path from `x`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.differential_operators import DifferentialOperators as D
from jno.trace_evaluator import _spectral_diff, _uniform_grid_spec


@pytest.fixture(autouse=True)
def _x64():
    """The exactness claim resolves errors near 1e-14, far below float32."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _grid(n=16, x1=1.0, y1=1.0):
    d = jno.domain(jno.Shape.rect(0.0, 0.0, x1, y1, size=min(x1, y1) / n), structured=True)
    shape, spacing = _uniform_grid_spec(d, d.mesh_connectivity["points"].shape[0])
    P = np.asarray(d.mesh_connectivity["points"])[:, :2]
    return d, shape, spacing, P


class TestExactness:
    def test_band_limited_field_is_differentiated_exactly(self):
        _, shape, spacing, P = _grid()
        x, y = P[:, 0], P[:, 1]
        u = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
        ux = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * y)
        uy = -4 * np.pi * np.sin(2 * np.pi * x) * np.sin(4 * np.pi * y)
        assert np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 1)) - ux).max() < 1e-11
        assert np.abs(np.asarray(_spectral_diff(u, shape, spacing, 1, 1)) - uy).max() < 1e-11

    def test_second_derivative_is_exact_too(self):
        _, shape, spacing, P = _grid()
        x, y = P[:, 0], P[:, 1]
        u = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
        assert np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 2)) + (2 * np.pi) ** 2 * u).max() < 1e-9

    def test_it_beats_finite_differences_by_orders(self):
        """The measurement that justifies the backend existing."""
        d, shape, spacing, P = _grid()
        x, y = P[:, 0], P[:, 1]
        u = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
        ux = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * y)
        spec = np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 1)) - ux).max()
        fd = np.abs(
            np.asarray(
                D.compute_fd_gradient_2d_simple(
                    u, P, np.asarray(d.mesh_connectivity["triangles"]), 0, grid=d.mesh_connectivity["grid"]
                )
            )
            - ux
        ).max()
        assert spec < fd / 1e6, f"spectral {spec:.2e} should crush FD {fd:.2e}"

    def test_a_constant_field_has_zero_derivative(self):
        _, shape, spacing, P = _grid()
        assert np.abs(np.asarray(_spectral_diff(np.full(P.shape[0], 2.5), shape, spacing, 0, 1))).max() < 1e-12


class TestGridConventions:
    def test_non_square_grid(self):
        _, shape, spacing, P = _grid(n=8, x1=1.0, y1=2.0)
        assert shape[0] != shape[1]
        x, y = P[:, 0], P[:, 1]
        u = np.sin(2 * np.pi * x) * np.sin(np.pi * y)  # period 1 in x, period 2 in y
        ux = 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(np.pi * y)
        assert np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 1)) - ux).max() < 1e-10

    def test_odd_and_even_mode_counts_both_work(self):
        for n in (8, 9, 16, 17):
            _, shape, spacing, P = _grid(n=n)
            x = P[:, 0]
            u = np.sin(2 * np.pi * x)
            err = np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 1)) - 2 * np.pi * np.cos(2 * np.pi * x)).max()
            assert err < 1e-10, f"n={n}: {err:.2e}"

    def test_the_duplicated_endpoint_is_preserved(self):
        """Node N-1 is the periodic image of node 0; the result must line up with the mesh order."""
        _, shape, spacing, P = _grid(n=8)
        u = np.sin(2 * np.pi * P[:, 0])
        out = np.asarray(_spectral_diff(u, shape, spacing, 0, 1)).reshape(shape)
        assert out.shape == shape
        np.testing.assert_allclose(out[0, :], out[-1, :], atol=1e-12)

    def test_complex_field_stays_complex(self):
        _, shape, spacing, P = _grid(n=8)
        u = np.exp(2j * np.pi * P[:, 0])
        out = _spectral_diff(jnp.asarray(u), shape, spacing, 0, 1)
        assert jnp.iscomplexobj(out)
        np.testing.assert_allclose(np.asarray(out), 2j * np.pi * u, atol=1e-10)


class TestRefusals:
    def test_unstructured_domain_raises_and_names_the_fix(self):
        d = jno.domain(jno.Shape.disk(0.0, 0.0, 1.0, size=0.3))
        with pytest.raises(ValueError, match="structured=True"):
            _uniform_grid_spec(d, d.mesh_connectivity["points"].shape[0])

    def test_wrong_value_count_raises(self):
        d, shape, _, _ = _grid(n=8)
        with pytest.raises(ValueError, match="nodes"):
            _uniform_grid_spec(d, 7)

    def test_unknown_family_still_raises(self):
        with pytest.raises(ValueError, match="Unknown differentiation scheme family"):
            from jno.utils.schemes import scheme_family

            scheme_family("spectrall")


class TestDocumentedLimitation:
    def test_a_non_periodic_field_rings(self):
        """Gibbs is the price of assuming periodicity. Pinned as behaviour, not left as a sentence."""
        _, shape, spacing, P = _grid(n=16)
        x = P[:, 0]
        u = x  # ramp: the periodic extension has a jump at the seam
        err = np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 1)) - 1.0)
        assert err.max() > 1.0, "a non-periodic field should ring — if it does not, this test is stale"
