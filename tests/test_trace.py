"""Unit tests for jno.trace — the symbolic tracing DSL."""

import pytest
import jax.numpy as jnp

from jno.trace import (
    NewAxis,
    Placeholder,
    Reshape,
    Slice,
    Concat,
    FunctionCall,
    Literal,
    ConstantNamespace,
    Constant,
    Variable,
    TensorTag,
    BinaryOp,
    Model,
    ModelCall,
    OperationDef,
    OperationCall,
    Hessian,
    Jacobian,
    collect_operations,
)
from tests.conftest import make_var


# ======================================================================
# Placeholder identity-based equality & hashing
# ======================================================================
class TestPlaceholderIdentity:
    def test_eq_identity(self):
        a = make_var("x")
        b = make_var("y")
        assert a == a
        assert not (a == b)
        assert a != b

    def test_hashable_dict_key(self):
        a = make_var("x")
        b = make_var("y")
        d = {a: "first", b: "second"}
        assert d[a] == "first"
        assert d[b] == "second"

    def test_hashable_set(self):
        a = make_var("x")
        s = {a, a, a}
        assert len(s) == 1

    def test_distinct_in_set(self):
        a = make_var("x")
        b = make_var("x")  # same tag but different object
        s = {a, b}
        assert len(s) == 2


# ======================================================================
# Symbolic comparison methods
# ======================================================================
class TestSymbolicComparison:
    def test_equal_returns_function_call(self):
        a = make_var("x")
        b = make_var("y")
        result = a.equal(b)
        assert isinstance(result, FunctionCall)
        assert result.fn is jnp.equal

    def test_not_equal_returns_function_call(self):
        a = make_var("x")
        b = make_var("y")
        result = a.not_equal(b)
        assert isinstance(result, FunctionCall)
        assert result.fn is jnp.not_equal

    def test_gt_lt_ge_le(self):
        a = make_var("x")
        b = make_var("y")
        assert isinstance(a > b, FunctionCall)
        assert isinstance(a < b, FunctionCall)
        assert isinstance(a >= b, FunctionCall)
        assert isinstance(a <= b, FunctionCall)


# ======================================================================
# Arithmetic operators
# ======================================================================
class TestArithmetic:
    def test_add(self):
        a = make_var("x")
        b = make_var("y")
        result = a + b
        assert isinstance(result, BinaryOp)
        assert result.op == "+"

    def test_sub(self):
        a = make_var("x")
        b = make_var("y")
        result = a - b
        assert isinstance(result, BinaryOp)
        assert result.op == "-"

    def test_mul(self):
        a = make_var("x")
        result = a * 2
        assert isinstance(result, BinaryOp)
        assert result.op == "*"

    def test_rmul(self):
        a = make_var("x")
        result = 3 * a
        assert isinstance(result, BinaryOp)
        assert result.op == "*"

    def test_truediv(self):
        a = make_var("x")
        result = a / 2
        assert isinstance(result, BinaryOp)
        assert result.op == "/"

    def test_rtruediv(self):
        a = make_var("x")
        result = 1 / a
        assert isinstance(result, BinaryOp)
        assert result.op == "/"

    def test_pow(self):
        a = make_var("x")
        result = a**2
        assert isinstance(result, BinaryOp)
        assert result.op == "**"

    def test_neg(self):
        a = make_var("x")
        result = -a
        assert isinstance(result, BinaryOp)
        assert result.op == "*"
        assert isinstance(result.left, Literal)

    def test_radd(self):
        a = make_var("x")
        result = 5 + a
        assert isinstance(result, BinaryOp)
        assert result.op == "+"


# ======================================================================
# Indexing / Slicing
# ======================================================================
class TestSlicing:
    def test_getitem_int(self):
        a = make_var("x")
        result = a[0]
        assert isinstance(result, Slice)

    def test_getitem_slice(self):
        a = make_var("x")
        result = a[1:3]
        assert isinstance(result, Slice)

    def test_getitem_none_becomes_newaxis(self):
        a = make_var("x")
        result = a[None, :]
        assert isinstance(result, Slice)
        # Check NewAxis is in the key
        key = result.key
        assert any(isinstance(k, NewAxis) for k in key)


# ======================================================================
# Reshape
# ======================================================================
class TestReshape:
    def test_reshape(self):
        a = make_var("x")
        result = a.reshape(2, 3)
        assert isinstance(result, Reshape)
        assert result.target_shape == (2, 3)


# ======================================================================
# Concat
# ======================================================================
class TestConcat:
    def test_concat(self):
        a = make_var("x")
        b = make_var("y")
        result = Concat([a, b])
        assert isinstance(result, Concat)
        assert len(result.items) == 2


# ======================================================================
# Literal & Constant
# ======================================================================
class TestLiteralConstant:
    def test_literal_value(self):
        lit = Literal(3.14)
        assert float(lit.value) == pytest.approx(3.14)

    def test_literal_array(self):
        lit = Literal([1, 2, 3])
        assert lit.value.shape == (3,)

    def test_constant(self):
        c = Constant("data", "key", 42.0)
        assert c.tag == "data"
        assert c.key == "key"
        assert c.value == 42.0


# ======================================================================
# ConstantNamespace
# ======================================================================
class TestConstantNamespace:
    def test_from_dict(self):
        ns = ConstantNamespace("test", {"a": 1.0, "b": 2.0})
        assert "a" in ns
        assert "b" in ns

    def test_getattr(self):
        ns = ConstantNamespace("test", {"a": 1.0, "b": 2.0})
        a = ns.a
        assert isinstance(a, Constant)
        assert a.key == "a"

    def test_keys_values_items(self):
        ns = ConstantNamespace("test", {"x": 1.0, "y": 2.0})
        assert set(ns.keys()) == {"x", "y"}
        assert len(list(ns.values())) == 2
        assert len(list(ns.items())) == 2

    def test_len(self):
        ns = ConstantNamespace("test", {"a": 1, "b": 2, "c": 3})
        assert len(ns) == 3

    def test_nested_dict(self):
        ns = ConstantNamespace("test", {"sub": {"x": 1.0}})
        sub = ns.sub
        assert isinstance(sub, ConstantNamespace)

    def test_to_dict(self):
        data = {"a": 1.0, "b": 2.0}
        ns = ConstantNamespace("test", data)
        result = ns.to_dict()
        assert set(result.keys()) == {"a", "b"}


# ======================================================================
# Variable & TensorTag
# ======================================================================
class TestVariableTensorTag:
    def test_variable_tag_dim(self):
        v = make_var("x")
        assert v.tag == "x"
        assert v.dim == [0, 1]

    def test_variable_unbounded_dim(self):
        v = make_var("x", [0, None])
        assert v.dim == [0, None]

    def test_tensor_tag(self):
        tt = TensorTag("coeff")
        assert tt.tag == "coeff"

    def test_tensor_tag_dim_index(self):
        tt = TensorTag("coeff", dim_index=2)
        assert tt.dim_index == 2


# ======================================================================
# Model / ModelCall
# ======================================================================
class TestFlaxModule:
    def test_flax_module_unique_id(self):
        from flax import linen as ln

        m1 = Model(ln.Dense(3))
        m2 = Model(ln.Dense(3))
        assert m1.layer_id != m2.layer_id

    def test_call_returns_module_call(self):
        from flax import linen as ln

        m = Model(ln.Dense(3))
        a = make_var("x")
        result = m(a)
        assert isinstance(result, ModelCall)

    def test_dont_show(self):
        from flax import linen as ln

        m = Model(ln.Dense(3), name="test")
        result = m.dont_show()
        assert result is m
        assert m.show is False


# ======================================================================
# OperationDef / OperationCall
# ======================================================================
class TestOperations:
    def test_operation_def_collects_vars(self):
        x = make_var("x")
        expr = x + Literal(1.0)
        op = OperationDef(expr)
        # Should collect variable x
        assert len(op._collected_vars) > 0

    def test_operation_call(self):
        x = make_var("x")
        expr = x * Literal(2.0)
        op = OperationDef(expr, [x])
        result = op(x)
        assert isinstance(result, OperationCall)


# ======================================================================
# Differential operators (type checking only — no evaluation)
# ======================================================================
class TestDifferentialOperators:
    def test_gradient_creation(self):
        """grad() now returns a single-variable Jacobian."""
        x = make_var("x")
        u = x**2
        g = Jacobian(u, [x])
        assert isinstance(g, Jacobian)
        assert len(g.variables) == 1

    def test_laplacian_creation(self):
        """laplacian() now returns a Hessian with trace=True."""
        x = make_var("x")
        u = x**2
        lap = Hessian(u, [x], trace=True)
        assert isinstance(lap, Hessian)
        assert lap.trace is True

    def test_hessian_creation(self):
        x = make_var("x")
        u = x**2
        h = Hessian(u, [x])
        assert isinstance(h, Hessian)
        assert h.trace is False

    def test_jacobian_creation(self):
        x = make_var("x")
        u = x**2
        j = Jacobian(u, [x])
        assert isinstance(j, Jacobian)


# ======================================================================
# collect_operations / collect_tags / get_primary_tag
# ======================================================================
class TestCollectors:
    def test_collect_operations_empty(self):
        x = make_var("x")
        ops = collect_operations(x + Literal(1.0))
        assert isinstance(ops, list)

    def test_collect_operations_with_op(self):
        x = make_var("x")
        expr = x * Literal(2.0)
        op_def = OperationDef(expr, [x])
        call = op_def(x)
        ops = collect_operations(call)
        assert len(ops) >= 1
        assert any(isinstance(o, OperationDef) for o in ops)


# ======================================================================
# Properties (.shape, .mean, .sum, etc.)
# ======================================================================
class TestPlaceholderProperties:
    def test_mean_property(self):
        x = make_var("x")
        result = x.mean
        assert isinstance(result, FunctionCall)

    def test_sum_property(self):
        x = make_var("x")
        result = x.sum
        assert isinstance(result, FunctionCall)

    def test_mse_property(self):
        x = make_var("x")
        result = x.mse
        assert isinstance(result, FunctionCall)

    def test_T_property(self):
        x = make_var("x")
        result = x.T
        assert isinstance(result, FunctionCall)
