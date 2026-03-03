"""Unit tests for jno.trace — the symbolic tracing DSL."""

import pytest
import jax.numpy as jnp

from jno.trace import (
    Placeholder,
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
        assert isinstance(result, FunctionCall)

    def test_getitem_slice(self):
        a = make_var("x")
        result = a[1:3]
        assert isinstance(result, FunctionCall)

    def test_getitem_none_becomes_newaxis(self):
        a = make_var("x")
        result = a[None, :]
        # None is inlined as None in the concrete key (no NewAxis wrapper)
        assert isinstance(result, FunctionCall)


# ======================================================================
# Reshape
# ======================================================================
class TestReshape:
    def test_reshape(self):
        a = make_var("x")
        result = a.reshape(2, 3)
        assert isinstance(result, FunctionCall)


# ======================================================================
# Concat
# ======================================================================
class TestConcat:
    def test_concat(self):
        import jno.numpy as pnp

        a = make_var("x")
        b = make_var("y")
        result = pnp.concat([a, b])
        assert isinstance(result, FunctionCall)
        assert len(result.args) == 2


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
# Model.mask / ModelCall.mask
# ======================================================================
class TestModelMask:
    """Tests for the param_mask feature on Model and ModelCall."""

    def _make_eqx_model(self):
        import jax
        import jno.numpy as jnn

        key = jax.random.PRNGKey(0)
        return jnn.nn.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=key)

    def test_mask_stores_param_mask(self):
        """mask() stores the pytree on _param_mask."""
        import jax
        import equinox as eqx

        u_net = self._make_eqx_model()
        all_true = jax.tree_util.tree_map(lambda _: True, u_net.module)
        result = u_net.mask(all_true)
        assert u_net._param_mask is all_true

    def test_mask_returns_self(self):
        """mask() returns the Model for chaining."""
        import jax

        u_net = self._make_eqx_model()
        all_true = jax.tree_util.tree_map(lambda _: True, u_net.module)
        assert u_net.mask(all_true) is u_net

    def test_mask_then_optimizer_chains(self):
        """mask().optimizer() sets both _param_mask and _opt_fn."""
        import jax, optax

        u_net = self._make_eqx_model()
        all_true = jax.tree_util.tree_map(lambda _: True, u_net.module)
        u_net.mask(all_true).optimizer(optax.adam, lr=1e-3)
        assert u_net._param_mask is all_true
        assert u_net._opt_fn is optax.adam

    def test_mask_then_freeze_chains(self):
        """mask().freeze() sets both _param_mask and _frozen."""
        import jax

        u_net = self._make_eqx_model()
        all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
        u_net.mask(all_false).freeze()
        assert u_net._param_mask is all_false
        assert u_net._frozen is True

    def test_mask_then_lora_chains(self):
        """mask().lora() sets both _param_mask and _lora_config."""
        import jax

        u_net = self._make_eqx_model()
        all_true = jax.tree_util.tree_map(lambda _: True, u_net.module)
        u_net.mask(all_true).lora(rank=4, alpha=1.0)
        assert u_net._param_mask is all_true
        assert u_net._lora_config == (4, 1.0)

    def test_mask_reset_clears_param_mask(self):
        """reset() clears _param_mask back to None."""
        import jax

        u_net = self._make_eqx_model()
        all_true = jax.tree_util.tree_map(lambda _: True, u_net.module)
        u_net.mask(all_true)
        assert u_net._param_mask is not None
        u_net.reset()
        assert u_net._param_mask is None

    def test_mask_via_model_call_proxies_to_model(self):
        """ModelCall.mask() delegates to the underlying Model."""
        import jax

        u_net = self._make_eqx_model()
        x = make_var("x")
        call = u_net(x)  # returns ModelCall
        assert isinstance(call, ModelCall)
        all_true = jax.tree_util.tree_map(lambda _: True, u_net.module)
        result = call.mask(all_true)
        # mask() on ModelCall returns the ModelCall (for chaining)
        assert result is call
        # but the mask is stored on the underlying Model
        assert u_net._param_mask is all_true

    def test_model_call_mask_chains_with_optimizer(self):
        """ModelCall.mask().optimizer() sets mask and optimizer on the model."""
        import jax, optax

        u_net = self._make_eqx_model()
        x = make_var("x")
        all_true = jax.tree_util.tree_map(lambda _: True, u_net.module)
        u_net(x).mask(all_true).optimizer(optax.adam, lr=1e-3)
        assert u_net._param_mask is all_true
        assert u_net._opt_fn is optax.adam

    def test_default_param_mask_is_none(self):
        """A freshly created Model has _param_mask == None (train everything)."""
        u_net = self._make_eqx_model()
        assert u_net._param_mask is None

    def test_partial_mask_structure(self):
        """A partial mask (some True, some False) is stored as-is."""
        import jax
        import equinox as eqx

        u_net = self._make_eqx_model()
        # Build a mask: only first hidden layer trainable
        all_false = jax.tree_util.tree_map(lambda _: False, u_net.module)
        partial_mask = eqx.tree_at(
            lambda m: (m.hidden_layers[0].weight, m.hidden_layers[0].bias),
            all_false,
            (True, True),
        )
        u_net.mask(partial_mask)
        assert u_net._param_mask is partial_mask
        # Verify the mask has the expected boolean leaves
        leaves = jax.tree_util.tree_leaves(u_net._param_mask)
        assert leaves.count(True) == 2
        assert leaves.count(False) == 4


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
