"""Tests for the typed semantic views on :class:`Placeholder`.

Covers ``ScalarView``, ``VectorView``, ``ComplexView``, ``MatrixView``,
``NamedMatrixView``, and ``VoigtView`` from ``jno.trace.views``.

Strategy: build a small ``(N, D)`` coordinate context, expose it as a
multi-component Variable, then evaluate the traced expression directly via
``TraceEvaluator`` and compare to numpy ground truth. This keeps tests fast
and free of training-loop overhead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.trace import (
    ComplexView,
    Jacobian,
    MatrixView,
    NamedComplexViewWithPartials,
    NamedMatrixView,
    NamedMatrixViewWithPartials,
    NamedScalarViewWithPartials,
    NamedVectorView,
    NamedVectorViewWithPartials,
    NamedVoigtViewWithPartials,
    NetworkGradient,
    Placeholder,
    ScalarView,
    Variable,
    VectorView,
    VoigtView,
)
from jno.trace_evaluator import TraceEvaluator
from tests.conftest import MockDomain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval(expr, context):
    """Evaluate a Placeholder expression directly via TraceEvaluator."""
    ev = TraceEvaluator({})
    return ev.evaluate(expr, context, {}, key=jax.random.PRNGKey(0))


def _domain_with(*tags_and_dims):
    """Build a MockDomain that allocates (10, dim_i) context arrays per tag."""
    d = MockDomain()
    for tag, dim in tags_and_dims:
        d.context[tag] = jnp.zeros((10, dim))
    return d


def _vec_field_2d(values):
    """Helper: create (Variable as 2-D vector field, x-var, y-var) sharing tag 'xy'.

    The context for 'xy' is set to `values` (shape (N, 2)).
    """
    d = _domain_with(("xy", 2))
    u = Variable("xy", [0, 2], domain=d)  # full 2-vector
    x = Variable("xy", [0, 1], domain=d)
    y = Variable("xy", [1, 2], domain=d)
    return u, x, y, {"xy": values}


# ---------------------------------------------------------------------------
# ScalarView
# ---------------------------------------------------------------------------


class TestScalarView:
    def test_property_returns_scalarview(self):
        u, x, _, _ = _vec_field_2d(jnp.array([[1.0, 2.0]]))
        s = x.scalar
        assert isinstance(s, ScalarView)
        assert s.expr is x

    def test_basic_ops(self):
        d = _domain_with(("p", 1))
        p = Variable("p", [0, 1], domain=d)
        ctx = {"p": jnp.array([[4.0], [9.0]])}

        np.testing.assert_allclose(_eval(p.scalar.abs().expr, ctx), [[4.0], [9.0]])
        np.testing.assert_allclose(_eval(p.scalar.sqrt().expr, ctx), [[2.0], [3.0]])
        np.testing.assert_allclose(_eval(p.scalar.log().expr, ctx), np.log([[4.0], [9.0]]))
        np.testing.assert_allclose(_eval(p.scalar.exp().expr, ctx), np.exp([[4.0], [9.0]]))
        np.testing.assert_allclose(_eval(p.scalar.pow(2).expr, ctx), [[16.0], [81.0]])

    def test_cross_type_multiply(self):
        # scalar * vector → VectorView; scalar * matrix → MatrixView; etc.
        d = _domain_with(("p", 1), ("v", 2))
        p = Variable("p", [0, 1], domain=d)
        v = Variable("v", [0, 2], domain=d)

        r1 = p.scalar * v.vector
        r2 = p.scalar * v.matrix.from_diag()
        r3 = p.scalar * v.voigt
        r4 = p.scalar * v.complex
        assert isinstance(r1, VectorView)
        assert isinstance(r2, MatrixView)
        assert isinstance(r3, VoigtView)
        assert isinstance(r4, ComplexView)

    def test_arithmetic_returns_scalarview(self):
        d = _domain_with(("p", 1))
        p = Variable("p", [0, 1], domain=d)
        s = p.scalar
        assert isinstance(s + s, ScalarView)
        assert isinstance(s - 1.0, ScalarView)
        assert isinstance(2.0 - s, ScalarView)
        assert isinstance(-s, ScalarView)
        assert isinstance(s / 2.0, ScalarView)


# ---------------------------------------------------------------------------
# VectorView
# ---------------------------------------------------------------------------


class TestVectorView:
    def test_property_returns_vectorview(self):
        u, *_ = _vec_field_2d(jnp.zeros((3, 2)))
        assert isinstance(u.vector, VectorView)

    def test_component_is_scalarview(self):
        u, *_ = _vec_field_2d(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        c0 = u.vector.component(0)
        c1 = u.vector.component(1)
        assert isinstance(c0, ScalarView)
        assert isinstance(c1, ScalarView)
        np.testing.assert_allclose(_eval(c0.expr, {"xy": jnp.array([[1.0, 2.0]])}), [1.0])
        np.testing.assert_allclose(_eval(c1.expr, {"xy": jnp.array([[1.0, 2.0]])}), [2.0])

    def test_div_of_identity_field_equals_two(self):
        """div((x, y)) = ∂x/∂x + ∂y/∂y = 2 everywhere."""
        u, x, y, _ = _vec_field_2d(None)
        ctx = {"xy": jnp.array([[0.3, 0.7], [0.5, 0.2], [0.1, 0.9]])}
        res = u.vector.div(x, y)
        assert isinstance(res, ScalarView)
        out = jnp.asarray(_eval(res.expr, ctx)).squeeze()
        np.testing.assert_allclose(out, [2.0, 2.0, 2.0], atol=1e-6)

    def test_curl_2d_returns_scalarview(self):
        """For u = (x, y), curl = ∂u_y/∂x - ∂u_x/∂y = 0."""
        u, x, y, _ = _vec_field_2d(None)
        ctx = {"xy": jnp.array([[0.3, 0.7]])}
        res = u.vector.curl(x, y)
        assert isinstance(res, ScalarView)
        out = jnp.asarray(_eval(res.expr, ctx)).squeeze()
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_curl_2d_rotation_field(self):
        """For u = (-y, x) the curl is 2."""
        d = _domain_with(("xy", 2))
        x = Variable("xy", [0, 1], domain=d)
        y = Variable("xy", [1, 2], domain=d)
        # u = (-y, x) — assemble via concat
        u = jno.np.concat([-y, x])
        ctx = {"xy": jnp.array([[0.3, 0.7], [0.5, 0.2]])}
        curl = u.vector.curl(x, y)
        out = jnp.asarray(_eval(curl.expr, ctx)).squeeze()
        np.testing.assert_allclose(out, [2.0, 2.0], atol=1e-6)

    def test_curl_3d_returns_vectorview(self):
        d = _domain_with(("xyz", 3))
        x = Variable("xyz", [0, 1], domain=d)
        y = Variable("xyz", [1, 2], domain=d)
        z = Variable("xyz", [2, 3], domain=d)
        # u = (x, y, z) — curl is (0, 0, 0)
        u = Variable("xyz", [0, 3], domain=d)
        res = u.vector.curl(x, y, z)
        assert isinstance(res, VectorView)
        ctx = {"xyz": jnp.array([[0.3, 0.7, 0.1]])}
        out = jnp.asarray(_eval(res.expr, ctx))
        np.testing.assert_allclose(out.reshape(-1, 3), [[0.0, 0.0, 0.0]], atol=1e-6)

    def test_norm_returns_scalarview(self):
        u, *_ = _vec_field_2d(jnp.array([[3.0, 4.0], [5.0, 12.0]]))
        res = u.vector.norm()
        assert isinstance(res, ScalarView)
        out = _eval(res.expr, {"xy": jnp.array([[3.0, 4.0], [5.0, 12.0]])})
        np.testing.assert_allclose(out, [5.0, 13.0])

    def test_dot_returns_scalarview(self):
        d = _domain_with(("xy", 2), ("ab", 2))
        u = Variable("xy", [0, 2], domain=d)
        v = Variable("ab", [0, 2], domain=d)
        res = u.vector.dot(v)
        assert isinstance(res, ScalarView)
        out = _eval(
            res.expr,
            {"xy": jnp.array([[1.0, 2.0]]), "ab": jnp.array([[3.0, 4.0]])},
        )
        np.testing.assert_allclose(out, [11.0])  # 1*3 + 2*4

    def test_outer_returns_matrixview(self):
        d = _domain_with(("xy", 2), ("ab", 2))
        u = Variable("xy", [0, 2], domain=d)
        v = Variable("ab", [0, 2], domain=d)
        res = u.vector.outer(v.vector)
        assert isinstance(res, MatrixView)
        out = _eval(
            res.expr,
            {"xy": jnp.array([[1.0, 2.0]]), "ab": jnp.array([[3.0, 4.0]])},
        )
        # outer([1,2], [3,4]) = [[3,4],[6,8]]
        np.testing.assert_allclose(out, [[[3.0, 4.0], [6.0, 8.0]]])

    def test_normalize_returns_vectorview(self):
        u, *_ = _vec_field_2d(None)
        res = u.vector.normalize()
        assert isinstance(res, VectorView)
        out = _eval(res.expr, {"xy": jnp.array([[3.0, 4.0]])})
        np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-6)

    def test_arithmetic_returns_vectorview(self):
        u, *_ = _vec_field_2d(None)
        v = u
        assert isinstance(u.vector + v.vector, VectorView)
        assert isinstance(u.vector - 1.0, VectorView)
        assert isinstance(-u.vector, VectorView)
        assert isinstance(2.0 * u.vector, VectorView)


# ---------------------------------------------------------------------------
# ComplexView
# ---------------------------------------------------------------------------


class TestComplexView:
    def test_real_imag_are_scalarviews(self):
        d = _domain_with(("z", 2))
        z = Variable("z", [0, 2], domain=d)
        assert isinstance(z.complex.real, ScalarView)
        assert isinstance(z.complex.imag, ScalarView)
        ctx = {"z": jnp.array([[3.0, 4.0]])}
        np.testing.assert_allclose(_eval(z.complex.real.expr, ctx), [3.0])
        np.testing.assert_allclose(_eval(z.complex.imag.expr, ctx), [4.0])

    def test_abs_equals_norm_of_vectorview(self):
        d = _domain_with(("z", 2))
        z = Variable("z", [0, 2], domain=d)
        ctx = {"z": jnp.array([[3.0, 4.0]])}
        assert isinstance(z.complex.abs, ScalarView)
        np.testing.assert_allclose(_eval(z.complex.abs.expr, ctx), [5.0])
        np.testing.assert_allclose(_eval(z.complex.abs.expr, ctx), _eval(z.vector.norm().expr, ctx))

    def test_angle(self):
        d = _domain_with(("z", 2))
        z = Variable("z", [0, 2], domain=d)
        ctx = {"z": jnp.array([[1.0, 1.0], [0.0, 1.0]])}
        out = _eval(z.complex.angle.expr, ctx)
        np.testing.assert_allclose(out, [np.pi / 4, np.pi / 2], atol=1e-6)

    def test_conj_returns_complexview(self):
        d = _domain_with(("z", 2))
        z = Variable("z", [0, 2], domain=d)
        res = z.complex.conj
        assert isinstance(res, ComplexView)
        ctx = {"z": jnp.array([[3.0, 4.0]])}
        out = _eval(res.expr, ctx)
        np.testing.assert_allclose(out, [[3.0, -4.0]])

    def test_to_native_complex_dtype(self):
        d = _domain_with(("z", 2))
        z = Variable("z", [0, 2], domain=d)
        native = z.complex.to_native()
        ctx = {"z": jnp.array([[3.0, 4.0]])}
        out = _eval(native, ctx)
        assert jnp.iscomplexobj(out)
        np.testing.assert_allclose(out, [3.0 + 4.0j])

    def test_complex_mul(self):
        # (1+2i)(3+4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        d = _domain_with(("a", 2), ("b", 2))
        a = Variable("a", [0, 2], domain=d)
        b = Variable("b", [0, 2], domain=d)
        res = a.complex.mul(b.complex)
        assert isinstance(res, ComplexView)
        ctx = {"a": jnp.array([[1.0, 2.0]]), "b": jnp.array([[3.0, 4.0]])}
        out = _eval(res.expr, ctx)
        np.testing.assert_allclose(out, [[-5.0, 10.0]], atol=1e-6)


# ---------------------------------------------------------------------------
# MatrixView
# ---------------------------------------------------------------------------


def _make_2x2(values_per_point):
    """Create a [N, 2, 2] MatrixView from a [N, 4] flat field."""
    d = _domain_with(("m", 4))
    flat = Variable("m", [0, 4], domain=d)
    return flat.matrix.from_flat(2), {"m": jnp.asarray(values_per_point)}


class TestMatrixView:
    def test_from_flat_shape(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        out = _eval(A.expr, ctx)
        assert out.shape == (1, 2, 2)
        np.testing.assert_allclose(out, [[[1.0, 2.0], [3.0, 4.0]]])

    def test_trace_returns_scalarview(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        res = A.trace()
        assert isinstance(res, ScalarView)
        np.testing.assert_allclose(_eval(res.expr, ctx), [5.0])

    def test_det_returns_scalarview(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        res = A.det()
        assert isinstance(res, ScalarView)
        np.testing.assert_allclose(_eval(res.expr, ctx), [1 * 4 - 2 * 3], atol=1e-6)

    def test_inv_returns_matrixview(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        res = A.inv()
        assert isinstance(res, MatrixView)
        out = _eval(res.expr, ctx)
        expected = jnp.linalg.inv(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_allclose(out[0], expected, atol=1e-6)

    def test_eigvals_returns_vectorview(self):
        # Symmetric matrix [[2, 1], [1, 2]] → eigenvalues [1, 3]
        A, ctx = _make_2x2([[2.0, 1.0, 1.0, 2.0]])
        res = A.eigvals()
        assert isinstance(res, VectorView)
        out = _eval(res.expr, ctx)
        np.testing.assert_allclose(out, [[1.0, 3.0]], atol=1e-6)

    def test_sym_skew_decomposition(self):
        A, ctx = _make_2x2([[1.0, 2.0, 4.0, 5.0]])  # asymmetric
        sym = _eval(A.sym().expr, ctx)
        skew = _eval(A.skew().expr, ctx)
        recon = sym + skew
        np.testing.assert_allclose(recon, _eval(A.expr, ctx), atol=1e-6)

    def test_diag_returns_vectorview(self):
        A, ctx = _make_2x2([[5.0, 0.0, 0.0, 7.0]])
        res = A.diag()
        assert isinstance(res, VectorView)
        np.testing.assert_allclose(_eval(res.expr, ctx), [[5.0, 7.0]])

    def test_norm_frobenius(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        res = A.norm()
        assert isinstance(res, ScalarView)
        # ||A||_F = sqrt(1+4+9+16) = sqrt(30)
        np.testing.assert_allclose(_eval(res.expr, ctx), [np.sqrt(30)], atol=1e-5)

    def test_transpose(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        At = A.transpose()
        assert isinstance(At, MatrixView)
        np.testing.assert_allclose(_eval(At.expr, ctx), [[[1.0, 3.0], [2.0, 4.0]]])

    def test_matpow_squared_equals_matmul_self(self):
        # Use symmetric matrix (pow uses eigh)
        A, ctx = _make_2x2([[2.0, 1.0, 1.0, 2.0]])
        A_pow2 = _eval(A.pow(2).expr, ctx)
        A_eval = _eval(A.expr, ctx)
        np.testing.assert_allclose(A_pow2, A_eval @ A_eval, atol=1e-5)

    def test_logexp_roundtrip(self):
        # SPD matrix [[2, 0.5], [0.5, 2]]
        A, ctx = _make_2x2([[2.0, 0.5, 0.5, 2.0]])
        roundtrip = _eval(A.log().exp().expr, ctx)
        np.testing.assert_allclose(roundtrip, _eval(A.expr, ctx), atol=1e-5)

    def test_from_upper_tri_roundtrip(self):
        # [..., 3] upper tri → [..., 2, 2] symmetric → back to [..., 3]
        d = _domain_with(("v", 3))
        v = Variable("v", [0, 3], domain=d)
        ctx = {"v": jnp.array([[1.0, 2.0, 3.0]])}  # [a, b, c] → [[a, b], [b, c]]
        A = v.matrix.from_upper_tri()
        out = _eval(A.expr, ctx)
        np.testing.assert_allclose(out, [[[1.0, 2.0], [2.0, 3.0]]])
        packed = _eval(A.to_upper_tri(), ctx)
        np.testing.assert_allclose(packed, [[1.0, 2.0, 3.0]])

    def test_from_lower_tri(self):
        # [a, b, c] lower tri → [[a, b], [b, c]] (same symmetric matrix)
        d = _domain_with(("v", 3))
        v = Variable("v", [0, 3], domain=d)
        ctx = {"v": jnp.array([[1.0, 2.0, 3.0]])}
        A = v.matrix.from_lower_tri()
        out = _eval(A.expr, ctx)
        # tril_indices for 2x2 = ([0,1,1], [0,0,1]) → entries [a, b, c] go to (0,0), (1,0), (1,1)
        # Then symmetrize: A = [[a, b], [b, c]]
        np.testing.assert_allclose(out, [[[1.0, 2.0], [2.0, 3.0]]])

    def test_from_flat_rectangular(self):
        d = _domain_with(("m", 6))
        v = Variable("m", [0, 6], domain=d)
        A = v.matrix.from_flat(2, 3)
        ctx = {"m": jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])}
        out = _eval(A.expr, ctx)
        assert out.shape == (1, 2, 3)
        np.testing.assert_allclose(out, [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    def test_from_diag(self):
        d = _domain_with(("v", 2))
        v = Variable("v", [0, 2], domain=d)
        A = v.matrix.from_diag()
        ctx = {"v": jnp.array([[3.0, 5.0]])}
        out = _eval(A.expr, ctx)
        np.testing.assert_allclose(out, [[[3.0, 0.0], [0.0, 5.0]]])

    def test_arithmetic(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        assert isinstance(A + A, MatrixView)
        assert isinstance(A - A, MatrixView)
        assert isinstance(2.0 * A, MatrixView)
        assert isinstance(A * A, MatrixView)
        assert isinstance(-A, MatrixView)
        assert isinstance(A**2, MatrixView)
        # elementwise sum: 2A
        np.testing.assert_allclose(_eval((A + A).expr, ctx), 2 * _eval(A.expr, ctx))

    def test_matmul_matrix_matrix(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        AA = A @ A
        assert isinstance(AA, MatrixView)
        Aev = _eval(A.expr, ctx)
        np.testing.assert_allclose(_eval(AA.expr, ctx), Aev @ Aev, atol=1e-6)

    def test_matmul_matrix_vector(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        d2 = _domain_with(("v", 2))
        v = Variable("v", [0, 2], domain=d2)
        ctx2 = {**ctx, "v": jnp.array([[1.0, 1.0]])}
        res = A @ v.vector
        assert isinstance(res, VectorView)
        out = _eval(res.expr, ctx2)
        # [[1,2],[3,4]] @ [1,1] = [3, 7]
        np.testing.assert_allclose(out, [[3.0, 7.0]])

    def test_vecmat_via_vector_matmul(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        d2 = _domain_with(("v", 2))
        v = Variable("v", [0, 2], domain=d2)
        ctx2 = {**ctx, "v": jnp.array([[1.0, 1.0]])}
        res = v.vector @ A
        assert isinstance(res, VectorView)
        out = _eval(res.expr, ctx2)
        # [1, 1] @ [[1,2],[3,4]] = [4, 6]
        np.testing.assert_allclose(out, [[4.0, 6.0]])


# ---------------------------------------------------------------------------
# NamedMatrixView
# ---------------------------------------------------------------------------


class TestNamedMatrixView:
    def test_coords_returns_named(self):
        A, _ = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        N = A.coords(["x", "y"])
        assert isinstance(N, NamedMatrixView)
        assert isinstance(N, MatrixView)

    def test_single_char_components(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        N = A.coords(["x", "y"])
        # A = [[1, 2], [3, 4]] → xx=1, xy=2, yx=3, yy=4
        assert isinstance(N.xx, ScalarView)
        np.testing.assert_allclose(_eval(N.xx.expr, ctx), [1.0])
        np.testing.assert_allclose(_eval(N.xy.expr, ctx), [2.0])
        np.testing.assert_allclose(_eval(N.yx.expr, ctx), [3.0])
        np.testing.assert_allclose(_eval(N.yy.expr, ctx), [4.0])

    def test_multi_char_underscore(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        N = A.coords(["r", "theta"])
        # A.rr → [0,0] = 1; A.r_theta → [0,1] = 2; A.theta_r → [1,0] = 3
        assert isinstance(N.rr, ScalarView)
        np.testing.assert_allclose(_eval(N.rr.expr, ctx), [1.0])
        np.testing.assert_allclose(_eval(N.r_theta.expr, ctx), [2.0])
        np.testing.assert_allclose(_eval(N.theta_r.expr, ctx), [3.0])
        np.testing.assert_allclose(_eval(N.theta_theta.expr, ctx), [4.0])

    def test_explicit_component_method(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        N = A.coords(["x", "y"])
        c = N.component("x", "y")
        assert isinstance(c, ScalarView)
        np.testing.assert_allclose(_eval(c.expr, ctx), [2.0])

    def test_matrix_ops_still_work(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        N = A.coords(["x", "y"])
        # Inherited from MatrixView
        np.testing.assert_allclose(_eval(N.trace().expr, ctx), [5.0])
        np.testing.assert_allclose(_eval(N.det().expr, ctx), [-2.0], atol=1e-6)


# ---------------------------------------------------------------------------
# VoigtView
# ---------------------------------------------------------------------------


def _make_voigt_2d(values_per_point):
    """Create a 2D Voigt VoigtView ([N, 3] = [σ_xx, σ_yy, σ_xy])."""
    d = _domain_with(("s", 3))
    s = Variable("s", [0, 3], domain=d)
    return s.voigt, {"s": jnp.asarray(values_per_point)}


def _make_voigt_3d(values_per_point):
    d = _domain_with(("s", 6))
    s = Variable("s", [0, 6], domain=d)
    return s.voigt, {"s": jnp.asarray(values_per_point)}


class TestVoigtView:
    def test_trace_2d(self):
        sigma, ctx = _make_voigt_2d([[10.0, 5.0, 3.0]])
        np.testing.assert_allclose(_eval(sigma.trace().expr, ctx), [15.0])

    def test_trace_3d(self):
        sigma, ctx = _make_voigt_3d([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        np.testing.assert_allclose(_eval(sigma.trace().expr, ctx), [6.0])

    def test_von_mises_2d(self):
        # sigma_xx=10, sigma_yy=0, sigma_xy=0 → vm = sqrt(100 - 0 + 0 + 0) = 10
        sigma, ctx = _make_voigt_2d([[10.0, 0.0, 0.0]])
        np.testing.assert_allclose(_eval(sigma.von_mises().expr, ctx), [10.0], atol=1e-6)

    def test_von_mises_3d_positive(self):
        sigma, ctx = _make_voigt_3d([[2.0, 1.0, 0.0, 0.5, 0.3, 0.2]])
        out = _eval(sigma.von_mises().expr, ctx)
        assert np.all(out > 0)

    def test_to_full_2d_is_symmetric(self):
        sigma, ctx = _make_voigt_2d([[10.0, 5.0, 3.0]])
        A = sigma.to_full()
        assert isinstance(A, MatrixView)
        out = _eval(A.expr, ctx)
        # [[10, 3], [3, 5]]
        np.testing.assert_allclose(out, [[[10.0, 3.0], [3.0, 5.0]]])

    def test_to_full_3d_shape_and_symmetry(self):
        sigma, ctx = _make_voigt_3d([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        A = sigma.to_full()
        out = _eval(A.expr, ctx)
        assert out.shape == (1, 3, 3)
        # Symmetric
        np.testing.assert_allclose(out, out.swapaxes(-2, -1))

    def test_principal_returns_vectorview(self):
        # Diagonal Voigt sigma = [3, 1, 0] → matrix [[3, 0], [0, 1]] → eigenvalues [1, 3]
        sigma, ctx = _make_voigt_2d([[3.0, 1.0, 0.0]])
        res = sigma.principal()
        assert isinstance(res, VectorView)
        np.testing.assert_allclose(_eval(res.expr, ctx), [[1.0, 3.0]], atol=1e-6)

    def test_invariants_2d(self):
        # sigma_xx=10, sigma_yy=5, sigma_xy=2 → I1=15, I2=10*5 - 4 = 46
        sigma, ctx = _make_voigt_2d([[10.0, 5.0, 2.0]])
        res = sigma.invariants()
        assert isinstance(res, VectorView)
        out = _eval(res.expr, ctx)
        np.testing.assert_allclose(out, [[15.0, 46.0]])

    def test_max_shear(self):
        # eigenvalues [1, 3] → max_shear = (3-1)/2 = 1
        sigma, ctx = _make_voigt_2d([[3.0, 1.0, 0.0]])
        res = sigma.max_shear()
        assert isinstance(res, ScalarView)
        np.testing.assert_allclose(_eval(res.expr, ctx), [1.0], atol=1e-6)

    def test_voigt_arithmetic(self):
        s, ctx = _make_voigt_2d([[10.0, 5.0, 3.0]])
        total = s + s
        assert isinstance(total, VoigtView)
        np.testing.assert_allclose(_eval(total.expr, ctx), [[20.0, 10.0, 6.0]])

    def test_hydrostatic(self):
        sigma, ctx = _make_voigt_2d([[10.0, 6.0, 0.0]])
        np.testing.assert_allclose(_eval(sigma.hydrostatic().expr, ctx), [8.0])


# ---------------------------------------------------------------------------
# Native complex Placeholder support
# ---------------------------------------------------------------------------


class TestNativeComplex:
    def test_real_imag_on_complex_array(self):
        # Stuff a native complex array into context for Placeholder.real / .imag
        d = MockDomain()
        d.context["z"] = jnp.array([[1.0 + 2.0j, 3.0 - 4.0j]])  # (1, 2) complex
        z = Variable("z", [0, 1], domain=d)
        # z is shape (N, 1) of complex
        re = _eval(z.real, {"z": d.context["z"]})
        im = _eval(z.imag, {"z": d.context["z"]})
        np.testing.assert_allclose(re, [[1.0]])
        np.testing.assert_allclose(im, [[2.0]])


# ---------------------------------------------------------------------------
# __getattr__ fallthrough + .expr property
# ---------------------------------------------------------------------------


class TestFallthrough:
    def test_expr_property_returns_placeholder(self):
        u, *_ = _vec_field_2d(None)
        v = u.vector
        assert v.expr is u
        # All view types
        d = _domain_with(("p", 1))
        p = Variable("p", [0, 1], domain=d)
        assert p.scalar.expr is p
        assert p.complex.expr is p
        assert p.matrix.expr is p
        assert p.voigt.expr is p

    def test_mse_fallthrough(self):
        u, *_ = _vec_field_2d(None)
        # u.vector.mse forwards to u.mse — both return a Placeholder (FunctionCall)
        a = u.vector.mse
        b = u.mse
        assert isinstance(a, Placeholder)
        assert isinstance(b, Placeholder)

    def test_d_method_preserves_view_type(self):
        u, x, y, _ = _vec_field_2d(None)
        # ``u.vector.d(x)`` is now first-class on VectorView and returns a VectorView.
        j = u.vector.d(x)
        assert isinstance(j, VectorView)


# ---------------------------------------------------------------------------
# Cross-type interaction smoke test
# ---------------------------------------------------------------------------


class TestCrossType:
    def test_full_chain(self):
        """Build a stress tensor via Voigt → to_full → coords → component → ScalarView."""
        sigma, ctx = _make_voigt_2d([[10.0, 5.0, 3.0]])
        named = sigma.to_full().coords(["x", "y"])
        # σ_xy = 3 (off-diagonal Voigt entry, position 2)
        np.testing.assert_allclose(_eval(named.xy.expr, ctx), [3.0])

    def test_scalar_times_voigt_then_von_mises(self):
        d = _domain_with(("p", 1), ("s", 3))
        p = Variable("p", [0, 1], domain=d)
        s = Variable("s", [0, 3], domain=d)
        # 2.0 * voigt → new voigt; then .von_mises() → scalar
        scaled = p.scalar * s.voigt
        vm = scaled.von_mises()
        assert isinstance(scaled, VoigtView)
        assert isinstance(vm, ScalarView)
        ctx = {"p": jnp.array([[2.0]]), "s": jnp.array([[10.0, 0.0, 0.0]])}
        out = _eval(vm.expr, ctx)
        # Original vm = 10, scaled by 2 → vm = 20
        np.testing.assert_allclose(out, [20.0], atol=1e-5)

    def test_unwrap_pairwise(self):
        """Mix of view types and plain Placeholders in arithmetic."""
        u, *_ = _vec_field_2d(None)
        # ScalarView + plain Placeholder → ScalarView (the underlying Placeholder is unwrapped)
        s = u.vector.norm() + 1.0
        assert isinstance(s, ScalarView)

    def test_placeholder_op_view_mixed_direction(self):
        """``placeholder + view`` preserves the view type via NotImplemented dispatch.

        Phase 2: ``Placeholder.__add__/__sub__/__mul__/__truediv__`` return
        ``NotImplemented`` when the other operand is a view, so Python falls
        back to ``View.__radd__/...`` which returns the matching view type.
        """
        d = _domain_with(("p", 1))
        p = Variable("p", [0, 1], domain=d)
        s = p.scalar
        # placeholder + scalarview: NotImplemented dispatch → ScalarView.__radd__
        result = p + s
        assert isinstance(result, ScalarView)
        ctx = {"p": jnp.array([[3.0]])}
        np.testing.assert_allclose(_eval(result.expr, ctx), [[6.0]])  # 3 + 3

        # Same for matrix
        A, ctxA = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        mixed = A.expr + A
        assert isinstance(mixed, MatrixView)
        np.testing.assert_allclose(_eval(mixed.expr, ctxA), _eval((A + A).expr, ctxA))


# ===========================================================================
# Phase 2 — ergonomic additions
#   * Placeholder.grad(*vars)         → VectorView
#   * jno.np.vector(*components)      → VectorView
#   * NotImplemented dispatch across all view types
#   * VectorView.jacobian / .grad     → MatrixView
#   * .integrate() preserves view type
#   * Unified .coords(**vars) with higher-order partial parsing
# ===========================================================================


def _domain_xy():
    """Single domain with tag ``xy`` (10 points × 2 dims), x and y Variables."""
    d = _domain_with(("xy", 2))
    x = Variable("xy", [0, 1], domain=d)
    y = Variable("xy", [1, 2], domain=d)
    return d, x, y


class TestPlaceholderGradDispatch:
    def test_grad_with_variables_returns_vectorview(self):
        d, x, y = _domain_xy()
        u = x * y  # scalar field
        g = u.grad(x, y)
        assert isinstance(g, VectorView)
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        # ∂u/∂x = y = 0.7, ∂u/∂y = x = 0.5
        out = jnp.asarray(_eval(g.expr, ctx))
        np.testing.assert_allclose(out.reshape(2), [0.7, 0.5], atol=1e-6)

    def test_grad_no_args_raises(self):
        d, x, _ = _domain_xy()
        with pytest.raises(TypeError):
            x.grad()  # neither Variable nor Model

    def test_grad_with_model_still_returns_network_gradient(self):
        """Backward-compatibility: the existing parameter-gradient form."""
        import foundax

        net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=4, num_layers=2, key=jax.random.PRNGKey(0)))
        # Build a Placeholder that depends on `net`
        d, x, y = _domain_xy()
        u = net(x, y)
        ng = u.grad(net)
        assert isinstance(ng, NetworkGradient)


class TestVectorConstructor:
    def test_basic(self):
        d, x, y = _domain_xy()
        v = jno.np.vector(x, y)
        assert isinstance(v, VectorView)
        ctx = {"xy": jnp.array([[0.3, 0.7]])}
        out = _eval(v.expr, ctx)
        np.testing.assert_allclose(out, [[0.3, 0.7]])

    def test_three_components(self):
        d, x, y = _domain_xy()
        v = jno.np.vector(x, y, x + y)
        ctx = {"xy": jnp.array([[0.3, 0.7]])}
        out = _eval(v.expr, ctx)
        np.testing.assert_allclose(out, [[0.3, 0.7, 1.0]])


class TestMixedArithmeticPreservation:
    """Placeholder OP View → View (NotImplemented dispatch fires).

    Sweeps each operator (+, -, *, /) and each view type.
    """

    @pytest.mark.parametrize(
        "make_view, expected_cls",
        [
            (lambda v: v.scalar, ScalarView),
            (lambda v: v.vector, VectorView),
            (lambda v: v.complex, ComplexView),
            (lambda v: v.matrix.from_diag(), MatrixView),
            (lambda v: v.voigt, VoigtView),
        ],
        ids=["Scalar", "Vector", "Complex", "Matrix", "Voigt"],
    )
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"], ids=["+", "-", "*", "/"])
    def test_placeholder_op_view(self, make_view, expected_cls, op):
        d, x, y = _domain_xy()
        v = Variable("xy", [0, 2], domain=d)  # generic 2-component carrier
        view = make_view(v)
        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "truediv": lambda a, b: a / b,
        }
        # x is a plain Placeholder; the OP should yield the view type
        result = ops[op](x, view)
        assert isinstance(result, expected_cls)

    def test_literal_division_via_rtruediv(self):
        """``1 / vec_view`` exercises the new VectorView.__rtruediv__."""
        d, x, y = _domain_xy()
        v = Variable("xy", [0, 2], domain=d)
        for vw, cls in (
            (v.vector, VectorView),
            (v.complex, ComplexView),
            (v.matrix.from_diag(), MatrixView),
            (v.voigt, VoigtView),
        ):
            assert isinstance(1 / vw, cls)


class TestVectorViewJacobian:
    def test_jacobian_shape_and_values(self):
        d, x, y = _domain_xy()
        # v = (x*y, x+y) → J = [[y, x], [1, 1]]
        v = jno.np.vector(x * y, x + y)
        J = v.jacobian(x, y)
        assert isinstance(J, MatrixView)
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        out = jnp.asarray(_eval(J.expr, ctx))
        # Result shape: [1, n_components=2, n_vars=2]
        assert out.shape == (1, 2, 2)
        np.testing.assert_allclose(out[0], [[0.7, 0.5], [1.0, 1.0]], atol=1e-6)

    def test_grad_alias(self):
        d, x, y = _domain_xy()
        v = jno.np.vector(x * y, x + y)
        assert isinstance(v.grad(x, y), MatrixView)
        # Same values as .jacobian
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        np.testing.assert_allclose(_eval(v.grad(x, y).expr, ctx), _eval(v.jacobian(x, y).expr, ctx))


class TestIntegratePreservation:
    """Each view's .integrate() returns the same view type wrapping the result."""

    def _setup(self):
        d, x, y = _domain_xy()
        v2 = Variable("xy", [0, 2], domain=d)
        return d, v2

    def test_scalar_integrate(self):
        d, v2 = self._setup()
        # use first component to make a scalar
        s = v2.vector.component(0)
        assert isinstance(s.integrate(), ScalarView)

    def test_vector_integrate(self):
        d, v2 = self._setup()
        assert isinstance(v2.vector.integrate(), VectorView)

    def test_matrix_integrate(self):
        d, v2 = self._setup()
        A = v2.matrix.from_diag()
        assert isinstance(A.integrate(), MatrixView)

    def test_complex_integrate(self):
        d, v2 = self._setup()
        assert isinstance(v2.complex.integrate(), ComplexView)

    def test_voigt_integrate(self):
        d = _domain_with(("s", 3))
        s = Variable("s", [0, 3], domain=d)
        assert isinstance(s.voigt.integrate(), VoigtView)


class TestBindKwargsForm:
    """`view.bind(x=x_var, ...)` returns a Named<View>WithPartials.

    Attribute access by registered name yields the partial derivative in
    the same view type. Up-to-4th-order parsing supported.
    """

    def test_each_view_partial_returns_same_type(self):
        d, x, y = _domain_xy()
        v2 = Variable("xy", [0, 2], domain=d)
        # ScalarView → ScalarView
        s = (x * y).scalar.bind(x=x, y=y)
        assert isinstance(s, NamedScalarViewWithPartials)
        assert isinstance(s.x, ScalarView)
        # VectorView → VectorView
        vec = v2.vector.bind(x=x, y=y)
        assert isinstance(vec, NamedVectorViewWithPartials)
        assert isinstance(vec.x, VectorView)
        # ComplexView → ComplexView
        cplx = v2.complex.bind(x=x, y=y)
        assert isinstance(cplx, NamedComplexViewWithPartials)
        assert isinstance(cplx.x, ComplexView)
        # MatrixView → MatrixView
        mat = v2.matrix.from_diag().bind(x=x, y=y)
        assert isinstance(mat, NamedMatrixViewWithPartials)
        assert isinstance(mat.x, MatrixView)
        # VoigtView → VoigtView
        dV = _domain_with(("sigma", 3))
        sig = Variable("sigma", [0, 3], domain=dV)
        voi = sig.voigt.bind(x=x, y=y)
        assert isinstance(voi, NamedVoigtViewWithPartials)
        assert isinstance(voi.x, VoigtView)

    def test_higher_order_single_char(self):
        d, x, y = _domain_xy()
        # u = x²y → ∂u/∂x = 2xy, ∂²u/∂x² = 2y, ∂²u/∂x∂y = 2x, ∂³u/∂x²∂y = 2
        u = x * x * y
        u_named = u.scalar.bind(x=x, y=y)
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        np.testing.assert_allclose(_eval(u_named.x.expr, ctx), [[2 * 0.5 * 0.7]], atol=1e-6)
        np.testing.assert_allclose(_eval(u_named.xx.expr, ctx), [[2 * 0.7]], atol=1e-6)
        np.testing.assert_allclose(_eval(u_named.xy.expr, ctx), [[2 * 0.5]], atol=1e-6)
        np.testing.assert_allclose(_eval(u_named.xxy.expr, ctx), [[2.0]], atol=1e-6)
        # 4th order: u.xxxy = 0
        np.testing.assert_allclose(_eval(u_named.xxxy.expr, ctx), [[0.0]], atol=1e-6)

    def test_fifth_order_falls_through(self):
        """5th+-order names aren't parsed as partial sequences."""
        d, x, _ = _domain_xy()
        u = (x * x).scalar.bind(x=x)
        with pytest.raises(AttributeError):
            _ = u.xxxxx  # 5 chars > max_order=4 → falls through, then _expr has no `xxxxx`

    def test_multi_char_underscore_regime(self):
        d = _domain_with(("rt", 2))
        r = Variable("rt", [0, 1], domain=d)
        t = Variable("rt", [1, 2], domain=d)
        u = r * t
        u_named = u.scalar.bind(r=r, theta=t)
        # ∂u/∂r = theta = t
        ctx = {"rt": jnp.array([[0.4, 0.6]])}
        np.testing.assert_allclose(_eval(u_named.r.expr, ctx), [[0.6]], atol=1e-6)
        # ∂²u/∂r∂theta = 1
        np.testing.assert_allclose(_eval(u_named.r_theta.expr, ctx), [[1.0]], atol=1e-6)
        # ∂²u/∂theta∂r = 1 (same)
        np.testing.assert_allclose(_eval(u_named.theta_r.expr, ctx), [[1.0]], atol=1e-6)

    def test_xy_symmetry_with_smooth_field(self):
        """∂²/∂x∂y = ∂²/∂y∂x for smooth fields."""
        d, x, y = _domain_xy()
        u = x * x * y * y * y  # x²y³
        u_named = u.scalar.bind(x=x, y=y)
        ctx = {"xy": jnp.array([[0.4, 0.6], [0.2, 0.9]])}
        a = jnp.asarray(_eval(u_named.xy.expr, ctx))
        b = jnp.asarray(_eval(u_named.yx.expr, ctx))
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_unknown_coord_falls_through(self):
        d, x, _ = _domain_xy()
        u = (x * x).scalar.bind(x=x)
        with pytest.raises(AttributeError):
            _ = u.z  # not a registered coord, and _expr has no .z

    def test_explicit_partials_match_laplacian(self):
        """``u.xx + u.yy`` reads like the math and equals ``Δu``."""
        d, x, y = _domain_xy()
        u_named = (x * x + y * y).scalar.bind(x=x, y=y)
        lap = u_named.xx + u_named.yy  # = 4 everywhere
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        np.testing.assert_allclose(_eval(lap.expr, ctx), [[4.0]], atol=1e-6)

    def test_mixing_kwargs_and_positionals_raises(self):
        d, x, _ = _domain_xy()
        v2 = Variable("xy", [0, 2], domain=d)
        with pytest.raises(TypeError):
            v2.vector.coords("x", x=x)  # positional + kwarg

    def test_vector_positional_string_form_unchanged(self):
        """VectorView.coords("x", "y") → NamedVectorView with component access."""
        d, x, y = _domain_xy()
        v2 = Variable("xy", [0, 2], domain=d)
        nv = v2.vector.coords("x", "y")
        assert isinstance(nv, NamedVectorView)
        ctx = {"xy": jnp.array([[0.3, 0.7]])}
        np.testing.assert_allclose(_eval(nv.x.expr, ctx), [0.3])
        np.testing.assert_allclose(_eval(nv.y.expr, ctx), [0.7])

    def test_matrix_positional_string_form_still_works(self):
        """MatrixView.coords(["x", "y"]) → NamedMatrixView with element access."""
        A, ctxA = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        nA = A.coords(["x", "y"])
        assert isinstance(nA, NamedMatrixView)
        np.testing.assert_allclose(_eval(nA.xy.expr, ctxA), [2.0])

    def test_matrix_kwargs_form_returns_partial(self):
        d, x, y = _domain_xy()
        # A = diag(x, y) → ∂A/∂x = diag(1, 0)
        v2 = Variable("xy", [0, 2], domain=d)
        A = v2.matrix.from_diag().bind(x=x, y=y)
        assert isinstance(A, NamedMatrixViewWithPartials)
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        out = jnp.asarray(_eval(A.x.expr, ctx))
        # shape (1, 2, 2); ∂(diag(x, y))/∂x = diag(1, 0)
        np.testing.assert_allclose(out[0], [[1.0, 0.0], [0.0, 0.0]], atol=1e-6)

    def test_indexing_strips_named_view_type(self):
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x, y=y)
        result = u.x  # ∂u/∂x
        # ScalarView, not NamedScalarViewWithPartials
        assert isinstance(result, ScalarView)
        assert type(result) is ScalarView

    def test_arithmetic_merges_disjoint_bindings(self):
        """``a.bind(x=x) + b.bind(y=y)`` → result carries both bindings."""
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x)
        v = (x * y).scalar.bind(y=y)
        combined = u + v
        assert sorted(object.__getattribute__(combined, "_coord_vars").keys()) == ["x", "y"]
        # Both partials are reachable on the result
        assert type(combined.x).__name__ == "ScalarView"
        assert type(combined.y).__name__ == "ScalarView"

    def test_arithmetic_with_matching_bindings_succeeds(self):
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x, y=y)
        v = (x * y).scalar.bind(x=x, y=y)  # same Variables
        combined = u + v
        # Still a Named*WithPartials, named partials still work
        assert type(combined).__name__ == "NamedScalarViewWithPartials"
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        # ∂(2xy)/∂x = 2y → 1.4
        np.testing.assert_allclose(_eval(combined.x.expr, ctx), [[1.4]], atol=1e-6)

    def test_conflicting_bindings_raise(self):
        """Same name → different Variable must error rather than silently picking one."""
        d, x, y = _domain_xy()
        x_alt = Variable("xy", [0, 1], domain=d)  # different Python object, same role
        u = (x * y).scalar.bind(x=x, y=y)
        v = (x * y).scalar.bind(x=x_alt, y=y)
        with pytest.raises(ValueError, match="coord binding conflict for 'x'"):
            _ = u + v
        with pytest.raises(ValueError, match="coord binding conflict for 'x'"):
            _ = u - v
        with pytest.raises(ValueError, match="coord binding conflict for 'x'"):
            _ = u * v
        with pytest.raises(ValueError, match="coord binding conflict for 'x'"):
            _ = u / v


class TestStopGradientMethod:
    """``.stop_gradient`` should work as a property on Placeholder and every view,
    return the same view type, and preserve named-partial bindings."""

    def test_placeholder_property_form(self):
        d, x, y = _domain_xy()
        p = (x * y).stop_gradient
        # FunctionCall is a Placeholder subclass
        assert hasattr(p, "_user_name") or hasattr(p, "name")

    def test_scalar_view_preserves_type(self):
        d, x, y = _domain_xy()
        sv = (x * y).scalar.stop_gradient
        assert type(sv).__name__ == "ScalarView"

    def test_named_view_preserves_bindings(self):
        """``u.bind(x=x).stop_gradient.x`` still works → AD on stop-grad'd output."""
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x, y=y)
        sg = u.stop_gradient
        assert type(sg).__name__ == "NamedScalarViewWithPartials"
        # Named partials still reachable through the stop-grad'd view
        assert type(sg.x).__name__ == "ScalarView"

    def test_hyco_style_arithmetic(self):
        """``(u_phy - u_syn.stop_gradient).mse`` flows through cleanly."""
        d, x, y = _domain_xy()
        u_phy = (x * y).scalar.bind(x=x, y=y)
        u_syn = (x * y).scalar.bind(x=x, y=y)
        L = (u_phy - u_syn.stop_gradient).mse
        # Result is a Placeholder (FunctionCall from .mse)
        assert L is not None

    def test_vector_view_preserves_type(self):
        d = _domain_with(("xy", 2))
        v = Variable("xy", [0, 2], domain=d)
        sg = v.vector.stop_gradient
        assert isinstance(sg, VectorView)

    def test_complex_view_preserves_type(self):
        d = _domain_with(("z", 2))
        v = Variable("z", [0, 2], domain=d)
        sg = v.complex.stop_gradient
        assert isinstance(sg, ComplexView)

    def test_matrix_view_preserves_type(self):
        A, _ = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        sg = A.stop_gradient
        assert isinstance(sg, MatrixView)

    def test_voigt_view_preserves_type(self):
        d = _domain_with(("s", 3))
        s = Variable("s", [0, 3], domain=d)
        sg = s.voigt.stop_gradient
        assert isinstance(sg, VoigtView)


class TestSchemeNamespace:
    """``u.fd.x`` / ``u.fd.xx`` mirror the AD attribute syntax but thread
    ``scheme="finite_difference"`` through every partial-derivative call."""

    def test_fd_first_order_matches_explicit_kwarg(self):
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x, y=y)
        proxy_result = u.fd.x
        explicit = u.d(x, scheme="finite_difference")
        # Both should produce a Jacobian with the FD scheme.
        assert isinstance(proxy_result, ScalarView)
        assert isinstance(proxy_result._expr, Jacobian)
        assert proxy_result._expr.scheme == "finite_difference"
        assert proxy_result._expr.scheme == explicit._expr.scheme
        # Same variable bound on both paths.
        assert proxy_result._expr.variables == explicit._expr.variables

    def test_fd_second_order_chains_two_jacobians(self):
        """``u.fd.xx`` chains ``.d(x).d(x)`` with FD on each step — same construction
        as the AD attribute path (``u.xx``), just with the scheme threaded through."""
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x, y=y)
        result = u.fd.xx
        assert isinstance(result, ScalarView)
        # Outer Jacobian — second derivative.
        assert isinstance(result._expr, Jacobian)
        assert result._expr.scheme == "finite_difference"
        # Inner Jacobian — first derivative, also FD.
        inner = result._expr.target
        assert isinstance(inner, Jacobian)
        assert inner.scheme == "finite_difference"

    def test_fd_mixed_partial(self):
        """``u.fd.xy`` registers as two distinct variables, both FD."""
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x, y=y)
        result = u.fd.xy
        assert isinstance(result, ScalarView)
        assert isinstance(result._expr, Jacobian)
        assert result._expr.scheme == "finite_difference"
        # Variables consumed left-to-right: outer call differentiates against the
        # last-seen name 'y'.
        assert result._expr.variables == [y]

    def test_fd_unknown_name_raises(self):
        d, x, y = _domain_xy()
        u = (x * y).scalar.bind(x=x, y=y)
        with pytest.raises(AttributeError, match="not a registered partial-name sequence"):
            _ = u.fd.z

    def test_fd_on_named_vector_partials(self):
        """``v.bind(x=x, y=y).fd.x`` returns the base VectorView (no Named wrapper)."""
        d, x, y = _domain_xy()
        v = Variable("xy", [0, 2], domain=d).vector.bind(x=x, y=y)
        result = v.fd.x
        assert isinstance(result, VectorView)
        assert isinstance(result._expr, Jacobian)
        assert result._expr.scheme == "finite_difference"

    def test_fd_on_named_matrix_partials(self):
        d, x, y = _domain_xy()
        A = Variable("xy", [0, 2], domain=d).matrix.from_diag().bind(x=x, y=y)
        result = A.fd.x
        assert isinstance(result, MatrixView)
        assert isinstance(result._expr, Jacobian)
        assert result._expr.scheme == "finite_difference"

    def test_fd_on_named_complex_partials(self):
        d = _domain_with(("z", 2))
        x = Variable("z", [0, 1], domain=d)
        y = Variable("z", [1, 2], domain=d)
        z = Variable("z", [0, 2], domain=d).complex.bind(x=x, y=y)
        result = z.fd.x
        assert isinstance(result, ComplexView)
        assert isinstance(result._expr, Jacobian)
        assert result._expr.scheme == "finite_difference"

    def test_fd_on_named_voigt_partials(self):
        # Voigt requires last-dim 3 (2-D symmetric tensor). Use coord vars from
        # a separate spatial domain so `.bind` doesn't collide with the Voigt tag.
        d2 = _domain_with(("s", 3), ("xy", 2))
        s = Variable("s", [0, 3], domain=d2)
        x = Variable("xy", [0, 1], domain=d2)
        y = Variable("xy", [1, 2], domain=d2)
        sv = s.voigt.bind(x=x, y=y)
        result = sv.fd.x
        assert isinstance(result, VoigtView)
        assert isinstance(result._expr, Jacobian)
        assert result._expr.scheme == "finite_difference"


class TestCruxEvalAcceptsViews:
    """``crux.eval(view)`` and ``crux.eval([view, ...])`` should unwrap views."""

    def test_eval_scalar_view(self):
        import foundax

        net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=4, num_layers=2, key=jax.random.PRNGKey(0)))
        dom = jno.domain.rect(mesh_size=0.5)
        x, y, _ = dom.variable("interior")
        u = net(x, y)
        crux = jno.core([u.mse])
        # No solve() — just verify eval accepts a ScalarView
        sv = u.scalar
        out = crux.eval(sv)  # single ScalarView
        assert out is not None

        # And as part of a list
        outs = crux.eval([sv, u])
        assert len(outs) == 2


# ===========================================================================
# Complex multi-step interactions ACROSS view types.
#
# The classes above mostly assert the *return type* of a single op.  These
# verify the numerical *value* of chained cross-type pipelines (matrix @ vector,
# scalar · (A @ v), Voigt → full → eigen, divergence = trace Jacobian, …) and
# exercise 3-D, which the rest of the file barely touches.  A wrong answer here
# would survive the type-only checks, so these are the higher-yield regression
# tests for the views' interactions.
# ===========================================================================


def _vec3(values):
    """([N, 3] VectorView, ctx) on tag ``v3``."""
    d = _domain_with(("v3", 3))
    return Variable("v3", [0, 3], domain=d).vector, {"v3": jnp.asarray(values)}


def _make_3x3(values):
    """([N, 3, 3] MatrixView, ctx) from a [N, 9] flat field on tag ``m9``."""
    d = _domain_with(("m9", 9))
    return Variable("m9", [0, 9], domain=d).matrix.from_flat(3), {"m9": jnp.asarray(values)}


def _make_voigt_3d(values):
    """([N, 6] VoigtView = [xx, yy, zz, yz, xz, xy], ctx) on tag ``s6``."""
    d = _domain_with(("s6", 6))
    return Variable("s6", [0, 6], domain=d).voigt, {"s6": jnp.asarray(values)}


class TestComplexInteractions:
    # -- VectorView ⇄ ScalarView / MatrixView -------------------------------
    def test_outer_trace_equals_dot_equals_norm_squared(self):
        v, ctx = _vec3([[1.0, 2.0, 3.0]])
        o_tr = _eval(v.outer(v).trace().expr, ctx)  # tr(v⊗v) = Σ vᵢ²
        dot = _eval(v.dot(v).expr, ctx)
        nrm2 = np.asarray(_eval(v.norm().expr, ctx)) ** 2
        np.testing.assert_allclose(o_tr, dot, atol=1e-6)
        np.testing.assert_allclose(o_tr, nrm2, atol=1e-6)
        np.testing.assert_allclose(np.asarray(o_tr).reshape(-1), [14.0], atol=1e-6)

    def test_cross_is_orthogonal_to_both_operands(self):
        d = _domain_with(("v3", 3), ("w3", 3))
        v = Variable("v3", [0, 3], domain=d).vector
        w = Variable("w3", [0, 3], domain=d).vector
        ctx = {"v3": jnp.array([[1.0, 2.0, 3.0]]), "w3": jnp.array([[2.0, 0.0, -1.0]])}
        c = v.cross(w)
        np.testing.assert_allclose(_eval(c.dot(v).expr, ctx), 0.0, atol=1e-5)
        np.testing.assert_allclose(_eval(c.dot(w).expr, ctx), 0.0, atol=1e-5)

    def test_matrix_at_vector_matches_numpy(self):
        d = _domain_with(("m", 4), ("v", 2))
        A = Variable("m", [0, 4], domain=d).matrix.from_flat(2)
        v = Variable("v", [0, 2], domain=d).vector
        ctx = {"m": jnp.array([[1.0, 2.0, 3.0, 4.0]]), "v": jnp.array([[5.0, 6.0]])}
        out = np.asarray(_eval((A @ v).expr, ctx)).reshape(-1)
        np.testing.assert_allclose(out, np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([5.0, 6.0]))

    def test_vector_at_matrix_matches_numpy(self):
        d = _domain_with(("m", 4), ("v", 2))
        A = Variable("m", [0, 4], domain=d).matrix.from_flat(2)
        v = Variable("v", [0, 2], domain=d).vector
        ctx = {"m": jnp.array([[1.0, 2.0, 3.0, 4.0]]), "v": jnp.array([[5.0, 6.0]])}
        out = np.asarray(_eval((v @ A).expr, ctx)).reshape(-1)
        np.testing.assert_allclose(out, np.array([5.0, 6.0]) @ np.array([[1.0, 2.0], [3.0, 4.0]]))

    # -- MatrixView algebra -------------------------------------------------
    def test_sym_plus_skew_reconstructs_matrix(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        recon = np.asarray(_eval((A.sym() + A.skew()).expr, ctx)).reshape(2, 2)
        np.testing.assert_allclose(recon, [[1.0, 2.0], [3.0, 4.0]], atol=1e-6)

    def test_trace_is_cyclic(self):
        d = _domain_with(("m", 4), ("mb", 4))
        A = Variable("m", [0, 4], domain=d).matrix.from_flat(2)
        B = Variable("mb", [0, 4], domain=d).matrix.from_flat(2)
        ctx = {"m": jnp.array([[1.0, 2.0, 3.0, 4.0]]), "mb": jnp.array([[5.0, 6.0, 7.0, 8.0]])}
        np.testing.assert_allclose(_eval((A @ B).trace().expr, ctx), _eval((B @ A).trace().expr, ctx), atol=1e-5)

    def test_inverse_times_self_is_identity_2d(self):
        A, ctx = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        ii = np.asarray(_eval((A.inv() @ A).expr, ctx)).reshape(2, 2)
        np.testing.assert_allclose(ii, np.eye(2), atol=1e-5)

    def test_inverse_times_self_is_identity_3d(self):
        A, ctx = _make_3x3([[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]])
        ii = np.asarray(_eval((A.inv() @ A).expr, ctx)).reshape(3, 3)
        np.testing.assert_allclose(ii, np.eye(3), atol=1e-4)

    def test_scalar_matrix_vector_linearity(self):
        d = _domain_with(("p", 1), ("m", 4), ("v", 2))
        p = Variable("p", [0, 1], domain=d)
        A = Variable("m", [0, 4], domain=d).matrix.from_flat(2)
        v = Variable("v", [0, 2], domain=d).vector
        ctx = {"p": jnp.array([[3.0]]), "m": jnp.array([[1.0, 2.0, 3.0, 4.0]]), "v": jnp.array([[5.0, 6.0]])}
        lhs = _eval((p.scalar * (A @ v)).expr, ctx)
        mid = _eval(((p.scalar * A) @ v).expr, ctx)
        rhs = _eval((A @ (p.scalar * v)).expr, ctx)
        np.testing.assert_allclose(lhs, mid, atol=1e-5)
        np.testing.assert_allclose(lhs, rhs, atol=1e-5)

    def test_from_diag_times_vector_is_elementwise(self):
        d = _domain_with(("dg", 2), ("v", 2))
        D = Variable("dg", [0, 2], domain=d).matrix.from_diag()  # diag([2, 3])
        v = Variable("v", [0, 2], domain=d).vector
        ctx = {"dg": jnp.array([[2.0, 3.0]]), "v": jnp.array([[5.0, 6.0]])}
        out = np.asarray(_eval((D @ v).expr, ctx)).reshape(-1)
        np.testing.assert_allclose(out, [10.0, 18.0], atol=1e-6)

    # -- divergence = trace of the Jacobian (VectorView → MatrixView → Scalar)
    def test_divergence_equals_trace_of_jacobian(self):
        d, x, y = _domain_xy()
        F = jno.np.vector(x * x, x * y)  # F = (x², xy);  ∇·F = 2x + x = 3x
        ctx = {"xy": jnp.array([[0.5, 0.7]])}
        # div → (N, 1), jacobian.trace → (N,): equal values, flatten to compare.
        jtr = np.asarray(_eval(F.jacobian(x, y).trace().expr, ctx)).reshape(-1)
        dv = np.asarray(_eval(F.div(x, y).expr, ctx)).reshape(-1)
        np.testing.assert_allclose(jtr, dv, atol=1e-6)
        np.testing.assert_allclose(dv, [3 * 0.5], atol=1e-5)

    # -- VoigtView 3-D (6-component) ----------------------------------------
    def test_voigt3d_to_full_is_symmetric(self):
        sig, ctx = _make_voigt_3d([[10.0, 8.0, 6.0, 1.0, 2.0, 3.0]])
        full = np.asarray(_eval(sig.to_full().expr, ctx)).reshape(3, 3)
        np.testing.assert_allclose(full, full.T, atol=1e-6)

    def test_voigt3d_principal_equals_eigvalsh(self):
        sig, ctx = _make_voigt_3d([[10.0, 8.0, 6.0, 1.0, 2.0, 3.0]])
        full = np.asarray(_eval(sig.to_full().expr, ctx)).reshape(3, 3)
        prin = np.sort(np.asarray(_eval(sig.principal().expr, ctx)).reshape(-1))
        np.testing.assert_allclose(prin, np.sort(np.linalg.eigvalsh(full)), atol=1e-5)

    def test_voigt3d_von_mises_value(self):
        sig, ctx = _make_voigt_3d([[10.0, 8.0, 6.0, 1.0, 2.0, 3.0]])
        # √(½[(10−8)²+(8−6)²+(6−10)²] + 3[1²+2²+3²]) = √54
        vm = float(np.asarray(_eval(sig.von_mises().expr, ctx)).reshape(-1)[0])
        np.testing.assert_allclose(vm, np.sqrt(54.0), atol=1e-4)


# ===========================================================================
# Views on a COMPLEX, unstructured domain.
#
# Every class above runs on MockDomain synthetic arrays or a structured grid.
# These build a real CSG mesh (concave L-shape, unstructured triangulation) and
# a 3-D tetrahedral cube, then verify the view API end-to-end on it: AD
# differential ops are exact at every node regardless of geometry, and the FD
# scheme is checked (interior-node L²) on the unstructured triangulation.
# ===========================================================================


def _interior_idx(mc):
    bnd = np.asarray(mc["boundary_indices"], dtype=np.int64)
    return np.setdiff1d(np.arange(int(mc["n_points"])), bnd)


def _rel_l2_at(computed, analytic, idx):
    c = np.asarray(computed).reshape(-1)[idx]
    a = np.asarray(analytic).reshape(-1)[idx]
    return float(np.sqrt(np.mean((c - a) ** 2)) / (np.sqrt(np.mean(a**2)) + 1e-12))


@pytest.fixture(scope="module")
def lshape():
    """Concave L-shape (area 3) on an unstructured triangular mesh — built once."""
    dom = jno.domain.csg(
        [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)],
        name="L",
    )
    dom.build_mesh(mesh_size=0.12)
    x, y, _ = dom.variable("interior")
    pts = np.asarray(dom.mesh_connectivity["points"])
    return dom, x, y, {"interior": jnp.asarray(pts)}, _interior_idx(dom.mesh_connectivity), pts[:, 0], pts[:, 1]


@pytest.fixture(scope="module")
def cube3d():
    """3-D unit cube on a tetrahedral mesh — built once."""
    dom = jno.domain(constructor=jno.domain.cube(mesh_size=0.25), compute_mesh_connectivity=True)
    v = dom.variable("interior")
    x, y, z = v[0], v[1], v[2]
    pts = np.asarray(dom.mesh_connectivity["points"])
    return dom, x, y, z, {"interior": jnp.asarray(pts)}, _interior_idx(dom.mesh_connectivity), pts


class TestViewsOnComplexDomain:
    """Typed views exercised on a concave unstructured (2-D) and tetrahedral
    (3-D) mesh — AD ops are exact; FD is checked in L²."""

    # -- 2-D concave L-shape, automatic differentiation (exact at every node) --
    def test_vector_div_ad(self, lshape):
        _, x, y, ctx, idx, X, _Y = lshape
        out = _eval(jno.np.vector(x * x, x * y).div(x, y).expr, ctx)  # ∇·(x², xy) = 3x
        assert _rel_l2_at(out, 3 * X, idx) < 1e-4

    def test_vector_curl_ad(self, lshape):
        _, x, y, ctx, idx, X, _Y = lshape
        out = _eval(jno.np.vector(-y, x).curl(x, y).expr, ctx)  # curl(−y, x) = 2
        assert _rel_l2_at(out, np.full_like(X, 2.0), idx) < 1e-4

    def test_scalar_laplacian_ad(self, lshape):
        _, x, y, ctx, idx, X, _Y = lshape
        # ScalarView has no .laplacian → falls through to Placeholder.laplacian (a Hessian).
        out = _eval((x * x + y * y).scalar.laplacian(x, y), ctx)  # Δ(x²+y²) = 4
        assert _rel_l2_at(out, np.full_like(X, 4.0), idx) < 1e-4

    def test_voigt_von_mises_ad(self, lshape):
        _, x, y, ctx, idx, X, Y = lshape
        s = jno.np.concat([x, y, 0.0 * x]).voigt  # [σ_xx, σ_yy, σ_xy] = [x, y, 0]
        out = _eval(s.von_mises().expr, ctx)  # √(x² − xy + y²)
        assert _rel_l2_at(out, np.sqrt(X**2 - X * Y + Y**2), idx) < 1e-4

    def test_matrix_jacobian_trace_equals_div_ad(self, lshape):
        _, x, y, ctx, idx, X, _Y = lshape
        F = jno.np.vector(x * x, x * y)
        jtr = _eval(F.jacobian(x, y).trace().expr, ctx)  # tr ∂Fᵢ/∂xⱼ = ∇·F = 3x
        assert _rel_l2_at(jtr, 3 * X, idx) < 1e-4

    # -- 2-D, finite-difference scheme on the unstructured triangulation --
    def test_scalar_fd_gradient_on_triangulation(self, lshape):
        dom, x, y, ctx, idx, X, Y = lshape
        uvals = (X**2 + Y**2).astype(np.float32)  # ∂/∂x = 2x
        uvar = dom.variable("uf", uvals[None, :, None])
        ctx2 = dict(ctx)
        ctx2["uf"] = uvals[:, None]
        fdx = _eval(uvar.scalar.d(x, scheme="finite_difference").expr, ctx2)
        assert _rel_l2_at(fdx, 2 * X, idx) < 0.05  # FD on a coarse unstructured mesh

    # -- 3-D tetrahedral cube, automatic differentiation --
    def test_3d_vector_div_ad(self, cube3d):
        _, x, y, z, ctx, idx, pts = cube3d
        out = _eval(jno.np.vector(x * x, y * y, z * z).div(x, y, z).expr, ctx)  # 2(x+y+z)
        analytic = 2.0 * (pts[:, 0] + pts[:, 1] + pts[:, 2])
        assert _rel_l2_at(out, analytic, idx) < 1e-4

    def test_3d_scalar_laplacian_ad(self, cube3d):
        _, x, y, z, ctx, idx, pts = cube3d
        out = _eval((x * x + y * y + z * z).scalar.laplacian(x, y, z), ctx)  # Δ = 6
        assert _rel_l2_at(out, np.full(pts.shape[0], 6.0), idx) < 1e-4
