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


class TestSecondDerivatives:
    def _field(self, n=16):
        _, shape, spacing, P = _grid(n=n)
        x, y = P[:, 0], P[:, 1]
        u = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
        return shape, spacing, P, u

    def test_laplacian_is_exact(self):
        from jno.trace_evaluator import _spectral_second_moments

        shape, spacing, _, u = self._field()
        analytic = -((2 * np.pi) ** 2 + (4 * np.pi) ** 2) * u
        got = np.asarray(_spectral_second_moments(u, shape, spacing, [(0, 0), (1, 1)], trace=True))
        assert np.abs(got - analytic).max() < 1e-9

    def test_laplacian_beats_finite_differences_by_orders(self):
        from jno.trace_evaluator import _spectral_second_moments

        d, shape, spacing, P = _grid()
        x, y = P[:, 0], P[:, 1]
        u = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
        analytic = -((2 * np.pi) ** 2 + (4 * np.pi) ** 2) * u
        spec = np.abs(
            np.asarray(_spectral_second_moments(u, shape, spacing, [(0, 0), (1, 1)], trace=True)) - analytic
        ).max()
        fd = np.abs(
            np.asarray(
                D.compute_fd_laplacian_2d_simple(
                    u, P, np.asarray(d.mesh_connectivity["triangles"]), (0, 1), grid=d.mesh_connectivity["grid"]
                )
            )
            - analytic
        ).max()
        assert spec < fd / 1e6, f"spectral {spec:.2e} vs FD {fd:.2e}"

    def test_mixed_partial_is_exact(self):
        from jno.trace_evaluator import _spectral_second_moments

        shape, spacing, P, u = self._field()
        x, y = P[:, 0], P[:, 1]
        uxy = -8 * np.pi**2 * np.cos(2 * np.pi * x) * np.sin(4 * np.pi * y)
        comps = _spectral_second_moments(u, shape, spacing, [(0, 0), (0, 1), (1, 0), (1, 1)], trace=False)
        assert np.abs(np.asarray(comps[1]) - uxy).max() < 1e-9

    def test_hessian_is_exactly_symmetric(self):
        """The multiplier -k_a k_b is symmetric in (a, b), so this is structural, not approximate."""
        from jno.trace_evaluator import _spectral_second_moments

        shape, spacing, _, u = self._field()
        c = _spectral_second_moments(u, shape, spacing, [(0, 0), (0, 1), (1, 0), (1, 1)], trace=False)
        np.testing.assert_array_equal(np.asarray(c[1]), np.asarray(c[2]))

    def test_fusing_the_laplacian_halves_the_transform_count(self):
        """The cost argument, measured on the jaxpr rather than asserted.

        `fftn` over d axes lowers to d transforms each way, so this is a 2x saving in 2-D, not the
        literal "one pair" the design sketch claimed.
        """
        from jno.trace_evaluator import _spectral_diff, _spectral_second_moments

        _, shape, spacing, P = _grid(n=8)
        u = P[:, 0]

        def count(f):
            return str(jax.make_jaxpr(f)(u)).count("fft")

        fused = count(lambda z: _spectral_second_moments(z, shape, spacing, [(0, 0), (1, 1)], trace=True))
        apart = count(lambda z: _spectral_diff(z, shape, spacing, 0, 2) + _spectral_diff(z, shape, spacing, 1, 2))
        assert fused * 2 == apart, f"fused {fused}, separate {apart}"

    def test_three_dimensional_laplacian(self):
        from jno.trace_evaluator import _spectral_second_moments, _uniform_grid_spec

        d = jno.domain(jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=1 / 6), structured=True)
        shape, spacing = _uniform_grid_spec(d, d.mesh_connectivity["points"].shape[0])
        P = np.asarray(d.mesh_connectivity["points"])[:, :3]
        u = np.sin(2 * np.pi * P[:, 0]) * np.sin(2 * np.pi * P[:, 1]) * np.sin(2 * np.pi * P[:, 2])
        analytic = -3 * (2 * np.pi) ** 2 * u
        got = np.asarray(_spectral_second_moments(u, shape, spacing, [(0, 0), (1, 1), (2, 2)], trace=True))
        assert np.abs(got - analytic).max() < 1e-8


class TestCosineVariant:
    """``spectral:cosine`` — an even (mirror) extension instead of assuming periodicity.

    Exact for fields whose odd derivatives vanish at both ends (Neumann-like). That is a real but
    **narrower** class than "non-periodic": a field with u' != 0 at an end still has a kink in the
    mirrored extension and still rings, just far less. Both cases are pinned below so the scope is
    a measurement rather than a claim.

    Implemented by mirroring and reusing the periodic FFT, not by a DCT: JAX implements only DCT-2
    and has no DST, and differentiating a cosine series yields a SINE series — so the transform-pair
    route would have meant hand-rolling a DST.
    """

    def _line(self, n=32):
        d = jno.Shape.rect(0, 0, 1, 1, size=1 / n).domain(structured=True)
        shape, spacing = _uniform_grid_spec(d, d.mesh_connectivity["points"].shape[0])
        P = np.asarray(d.mesh_connectivity["points"])[:, :2]
        return shape, spacing, P

    def test_exact_on_a_neumann_field_where_periodic_fails(self):
        shape, spacing, P = self._line()
        x = P[:, 0]
        u, du = np.cos(np.pi * x), -np.pi * np.sin(np.pi * x)  # u'(0) = u'(1) = 0, but u(0) != u(1)
        per = np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 1)) - du).max()
        cos = np.abs(np.asarray(_spectral_diff(u, shape, spacing, 0, 1, mirror=True)) - du).max()
        assert cos < 1e-10, f"cosine should be exact here, got {cos:.2e}"
        assert per > 1.0, "plain spectral should fail badly on a non-periodic field"

    def test_laplacian_too(self):
        from jno.trace_evaluator import _spectral_second_moments

        shape, spacing, P = self._line()
        u = np.cos(np.pi * P[:, 0])
        got = np.asarray(_spectral_second_moments(u, shape, spacing, [(0, 0)], trace=True, mirror=True))
        assert np.abs(got - (-(np.pi**2) * u)).max() < 1e-9

    def test_a_field_that_is_neither_periodic_nor_neumann_is_better_but_not_exact(self):
        """The honest scope limit: a ramp has u' != 0 at the ends, so the mirror has a kink."""
        shape, spacing, P = self._line()
        x = P[:, 0]
        per = np.abs(np.asarray(_spectral_diff(x, shape, spacing, 0, 1)) - 1.0).max()
        cos = np.abs(np.asarray(_spectral_diff(x, shape, spacing, 0, 1, mirror=True)) - 1.0).max()
        assert cos < per / 10, f"cosine should still help a lot ({per:.2e} -> {cos:.2e})"
        assert cos > 1e-3, "…but it is NOT exact here — if this ever passes, the scope note is stale"

    def test_reachable_through_the_scheme_string(self):
        from jno.utils.schemes import scheme_family

        assert scheme_family("spectral:cosine") == "spectral"

    def test_shape_is_preserved(self):
        shape, spacing, P = self._line(n=8)
        out = _spectral_diff(np.cos(np.pi * P[:, 0]), shape, spacing, 0, 1, mirror=True)
        assert out.shape == (P.shape[0],)


def test_finite_difference_also_reaches_a_stored_field():
    """Guards a claim that was overstated once: only AD is broken on a dataset-fed operator.

    `finite_difference` evaluates the target on the mesh and stencils the VALUES, so it never needed
    a path from x either — a physics residual on a stored input was writable before the spectral
    backend existed. Spectral's contribution there is accuracy, not capability, and this pins the
    distinction so the docs cannot drift back to the stronger claim.
    """
    _, shape, spacing, P = _grid(n=16)
    d = jno.Shape.rect(0, 0, 1, 1, size=1 / 16).domain(structured=True)
    x, y = P[:, 0], P[:, 1]
    u = np.sin(2 * np.pi * x) * np.cos(4 * np.pi * y)
    ux = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(4 * np.pi * y)

    fd = np.asarray(
        D.compute_fd_gradient_2d_simple(
            u, P, np.asarray(d.mesh_connectivity["triangles"]), 0, grid=d.mesh_connectivity["grid"]
        )
    )
    spec = np.asarray(_spectral_diff(u, shape, spacing, 0, 1))

    fd_err, spec_err = np.abs(fd - ux).max(), np.abs(spec - ux).max()
    assert fd_err < 1.0, "finite differences must give a REAL derivative here, not zero"
    assert spec_err < fd_err / 1e6, "…and spectral must be far more accurate on a periodic field"
