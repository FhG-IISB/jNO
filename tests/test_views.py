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

import jno
from jno.trace import (
    ComplexView,
    MatrixView,
    NamedMatrixView,
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

    def test_d_method_fallthrough(self):
        u, x, y, _ = _vec_field_2d(None)
        # u.vector.d(x) forwards to u.d(x) — Jacobian of the full vector
        j = u.vector.d(x)
        assert isinstance(j, Placeholder)


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
        """``placeholder + view`` must work, not just ``view + placeholder``.

        Python dispatches `placeholder + view` to ``Placeholder.__add__``, so
        ``Placeholder._wrap`` must recognize views and pull out ``._expr``.
        Otherwise the view would be embedded as a ``Literal`` and crash at eval.
        """
        d = _domain_with(("p", 1))
        p = Variable("p", [0, 1], domain=d)
        s = p.scalar
        # placeholder + scalarview: dispatches to Placeholder.__add__
        result = p + s
        assert isinstance(result, Placeholder)  # raw, since LHS is Placeholder
        ctx = {"p": jnp.array([[3.0]])}
        out = _eval(result, ctx)
        np.testing.assert_allclose(out, [[6.0]])  # 3 + 3

        # Same for matrix
        A, ctxA = _make_2x2([[1.0, 2.0, 3.0, 4.0]])
        # A.expr + A (Placeholder + MatrixView) — both yield equal results
        mixed = A.expr + A
        out = _eval(mixed, ctxA)
        out_direct = _eval((A + A).expr, ctxA)
        np.testing.assert_allclose(out, out_direct)
