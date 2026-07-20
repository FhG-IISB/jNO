"""Unit tests for jno.jnp_ops — JAX NumPy wrappers for the tracing DSL."""

import jax
import jax.numpy as jnp
import pytest

import jno
import jno.jnp_ops as np
from jno.trace import FunctionCall, Placeholder
from tests.conftest import make_var


# ======================================================================
# Constants
# ======================================================================
class TestConstants:
    def test_top_level_np_alias_matches_submodule(self):
        assert jno.np is np

    def test_top_level_numpy_module_export(self):
        assert jno.jnp_ops is np

    def test_pi(self):
        assert float(np.pi) == pytest.approx(3.141592653589793)

    def test_e(self):
        assert float(np.e) == pytest.approx(2.718281828459045)


# ======================================================================
# Placeholder → FunctionCall passthrough
# ======================================================================
class TestTrigWrappers:
    def test_sin_placeholder(self):
        x = make_var("x")
        result = np.sin(x)
        assert isinstance(result, FunctionCall)

    def test_cos_placeholder(self):
        x = make_var("x")
        result = np.cos(x)
        assert isinstance(result, FunctionCall)

    def test_tan_placeholder(self):
        x = make_var("x")
        result = np.tan(x)
        assert isinstance(result, FunctionCall)


class TestExpLogWrappers:
    def test_exp_placeholder(self):
        x = make_var("x")
        assert isinstance(np.exp(x), FunctionCall)

    def test_log_placeholder(self):
        x = make_var("x")
        assert isinstance(np.log(x), FunctionCall)

    def test_sqrt_placeholder(self):
        x = make_var("x")
        assert isinstance(np.sqrt(x), FunctionCall)


class TestActivationWrappers:
    def test_tanh_placeholder(self):
        x = make_var("x")
        assert isinstance(np.tanh(x), FunctionCall)


class TestReductionWrappers:
    def test_sum_placeholder(self):
        x = make_var("x")
        result = np.sum(x)
        assert isinstance(result, FunctionCall)

    def test_mean_placeholder(self):
        x = make_var("x")
        result = np.mean(x)
        assert isinstance(result, FunctionCall)


class TestArrayManipulation:
    def test_concat_placeholder(self):
        a = make_var("x")
        b = make_var("y")
        result = np.concat([a, b])
        # concat with Placeholder args should return a trace node
        assert isinstance(result, Placeholder)

    def test_reshape_placeholder(self):
        x = make_var("x")
        result = np.reshape(x, (2, 3))
        assert isinstance(result, Placeholder)

    def test_squeeze_placeholder(self):
        x = make_var("x")
        result = np.squeeze(x)
        assert isinstance(result, FunctionCall)


class TestComparisonWrappers:
    def test_where_placeholder(self):
        x = make_var("x")
        y = make_var("y")
        cond = x > y
        result = np.where(cond, x, y)
        assert isinstance(result, FunctionCall)

    def test_maximum_placeholder(self):
        x = make_var("x")
        y = make_var("y")
        result = np.maximum(x, y)
        assert isinstance(result, FunctionCall)

    def test_minimum_placeholder(self):
        x = make_var("x")
        y = make_var("y")
        result = np.minimum(x, y)
        assert isinstance(result, FunctionCall)


class TestDifferentialWrappers:
    def test_grad_returns_jacobian(self):
        from jno.trace import Jacobian

        x = make_var("x")
        u = x**2
        result = np.grad(u, x)
        assert isinstance(result, Jacobian)
        assert len(result.variables) == 1

    def test_laplacian_returns_hessian_with_trace(self):
        from jno.trace import Hessian

        x = make_var("x")
        u = x**2
        result = np.laplacian(u, [x])
        assert isinstance(result, Hessian)
        assert result.trace is True

    def test_jacobian_returns_jacobian(self):
        from jno.trace import Jacobian

        x = make_var("x")
        u = x**2
        result = np.jacobian(u, [x])
        assert isinstance(result, Jacobian)


class TestCreationWrappers:
    def test_zeros(self):
        result = np.zeros((3, 2))
        assert result.shape == (3, 2)
        assert jnp.allclose(result, 0.0)

    def test_ones(self):
        result = np.ones((2,))
        assert result.shape == (2,)
        assert jnp.allclose(result, 1.0)

    def test_linspace(self):
        result = np.linspace(0, 1, 5)
        assert result.shape == (5,)

    def test_arange(self):
        result = np.arange(0, 5, 1)
        assert result.shape == (5,)


class TestLinalgTensorWrappers:
    """Second-order-tensor linear algebra: inv / det / eigvalsh / logm / expm / sqrtm.

    These act on the last two axes and broadcast over leading (quadrature) axes.
    The deep numerical + FD-gradient verification of the shared spectral helper
    lives in tests/test_views.py; here we cover the ``jno.np`` free-function
    surface, the broadcast contract, and that the degeneracy-stable gradient is
    inherited (the enabler for a finite-strain-plasticity return map).
    """

    def test_return_types(self):
        x = make_var("x")
        for op in (np.inv, np.det, np.eigvalsh, np.logm, np.expm, np.sqrtm):
            assert isinstance(op(x), FunctionCall)

    def test_inv_det_numerical(self):
        A = jnp.array([[4.0, 1.0], [2.0, 3.0]])
        assert jnp.allclose(np.inv(make_var("x")).fn(A), jnp.linalg.inv(A))
        assert jnp.allclose(np.det(make_var("x")).fn(A), jnp.linalg.det(A))

    def test_eigvalsh_numerical(self):
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])  # eigenvalues 1, 3
        assert jnp.allclose(np.eigvalsh(make_var("x")).fn(A), jnp.linalg.eigvalsh(A))

    def test_logm_expm_roundtrip(self):
        A = jnp.array([[3.0, 1.0], [1.0, 2.0]])  # SPD
        logA = np.logm(make_var("x")).fn(A)
        assert jnp.allclose(np.expm(make_var("x")).fn(logA), A, atol=1e-5)

    def test_sqrtm_squares_back(self):
        A = jnp.array([[3.0, 1.0], [1.0, 2.0]])  # SPD
        S = np.sqrtm(make_var("x")).fn(A)
        assert jnp.allclose(S @ S, A, atol=1e-5)

    def test_broadcasts_over_leading_axis(self):
        # (Q, n, n): broadcast over the leading quadrature axis, one entry degenerate
        batch = jnp.stack([2.0 * jnp.eye(3), jnp.diag(jnp.array([1.0, 2.0, 3.0]))])  # (2, 3, 3)
        assert np.inv(make_var("x")).fn(batch).shape == (2, 3, 3)
        assert np.det(make_var("x")).fn(batch).shape == (2,)
        assert np.eigvalsh(make_var("x")).fn(batch).shape == (2, 3)
        assert np.logm(make_var("x")).fn(batch).shape == (2, 3, 3)

    def test_matrix_function_gradient_stable_at_degeneracy(self):
        # logm gradient must stay finite at repeated eigenvalues (equal principal
        # stretches) — the enabler for a differentiable finite-strain return map.
        prev = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        try:
            logm = np.logm(make_var("x")).fn
            g = jax.grad(lambda A: logm(A)[0, 0])(2.0 * jnp.eye(3))  # eigenvalues [2, 2, 2]
            assert bool(jnp.all(jnp.isfinite(g))), "jno.np.logm gradient is NaN at repeated eigenvalues"
            assert jnp.allclose(g[0, 0], 0.5, atol=1e-6)  # d/dλ log λ = 1/λ = 1/2
        finally:
            jax.config.update("jax_enable_x64", prev)


class TestDtypes:
    def test_float32(self):
        assert np.float32 is jnp.float32

    def test_float64(self):
        assert np.float64 is jnp.float64

    def test_int32(self):
        assert np.int32 is jnp.int32
