"""Unit tests for jno.numpy — JAX NumPy wrappers for the tracing DSL."""

import pytest
import jax.numpy as jnp
import jno

from jno.trace import Placeholder, FunctionCall, Literal, BinaryOp
from tests.conftest import make_var
import jno.numpy as np


# ======================================================================
# Constants
# ======================================================================
class TestConstants:
    def test_top_level_np_alias_matches_submodule(self):
        assert jno.np is np

    def test_top_level_numpy_module_export(self):
        assert jno.numpy is np

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


class TestDtypes:
    def test_float32(self):
        assert np.float32 is jnp.float32

    def test_float64(self):
        assert np.float64 is jnp.float64

    def test_int32(self):
        assert np.int32 is jnp.int32
