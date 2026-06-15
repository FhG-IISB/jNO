"""Unit tests for jno.trace_evaluator — the dispatch-table evaluator."""

import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import pytest

from jno.trace import (
    Constant,
    FunctionCall,
    Jacobian,
    Literal,
    Model,
    OperationDef,
    TensorTag,
)
from jno.trace_compiler import TraceCompiler
from jno.trace_evaluator import TraceEvaluator
from tests.conftest import make_var


# ======================================================================
# Helpers
# ======================================================================
def make_evaluator(params=None):
    """Create a TraceEvaluator with empty params."""
    return TraceEvaluator(params or {})


def make_points(tag="x", n=10, d=1):
    """Create a simple context dict with spatial points."""
    return {tag: jnp.linspace(0, 1, n).reshape(n, d)}


# ======================================================================
# _EvalCtx
# ======================================================================
class TestEvalCtx:
    def test_ctx_creation(self):
        ctx = TraceEvaluator._EvalCtx(
            context={"x": jnp.ones((5, 1))},
            var_bindings={},
            key=jax.random.PRNGKey(0),
        )
        assert ctx.context["x"].shape == (5, 1)
        assert ctx.var_bindings == {}


# ======================================================================
# Dispatch table
# ======================================================================
class TestDispatchTable:
    def test_handlers_count(self):
        assert len(TraceEvaluator._HANDLERS) == 23  # Update this if we add more node types

    def test_handlers_are_strings(self):
        for node_type, method_name in TraceEvaluator._HANDLERS:
            assert isinstance(method_name, str)
            assert hasattr(TraceEvaluator, method_name)


# ======================================================================
# Literal evaluation
# ======================================================================
class TestEvalLiteral:
    def test_scalar(self):
        ev = make_evaluator()
        lit = Literal(3.14)
        result = ev.evaluate(lit, make_points())
        assert float(result) == pytest.approx(3.14)

    def test_array(self):
        ev = make_evaluator()
        lit = Literal([1.0, 2.0, 3.0])
        result = ev.evaluate(lit, make_points())
        assert result.shape == (3,)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))


# ======================================================================
# Constant evaluation
# ======================================================================
class TestEvalConstant:
    def test_constant(self):
        ev = make_evaluator()
        c = Constant("data", "key", jnp.array(42.0))
        result = ev.evaluate(c, make_points())
        assert float(result) == pytest.approx(42.0)


# ======================================================================
# Variable evaluation
# ======================================================================
class TestEvalVariable:
    def test_variable_from_points(self):
        ev = make_evaluator()
        v = make_var("x")
        points = {"x": jnp.linspace(0, 1, 10).reshape(5, 2)}
        result = ev.evaluate(v, points)
        # dim=[0,1] means slice columns 0:1; evaluator may squeeze trailing dim
        assert result.shape[0] == 5


# ======================================================================
# TensorTag evaluation
# ======================================================================
class TestEvalTensorTag:
    def test_tensor_tag(self):
        ev = make_evaluator()
        tt = TensorTag("coeff")
        context = {**make_points(), "coeff": jnp.array([1.0, 2.0, 3.0])}
        result = ev.evaluate(tt, context)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    def test_tensor_tag_dim_index(self):
        ev = make_evaluator()
        tt = TensorTag("coeff", dim_index=1)
        data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        context = {**make_points(), "coeff": data}
        result = ev.evaluate(tt, context)
        # dim_index slicing returns column 1 (shape depends on evaluator impl)
        assert result.shape[0] == 2


# ======================================================================
# BinaryOp evaluation
# ======================================================================
class TestEvalBinaryOp:
    def test_add(self):
        ev = make_evaluator()
        a = Literal(2.0)
        b = Literal(3.0)
        expr = a + b
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(5.0)

    def test_sub(self):
        ev = make_evaluator()
        a = Literal(5.0)
        b = Literal(2.0)
        expr = a - b
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(3.0)

    def test_mul(self):
        ev = make_evaluator()
        a = Literal(3.0)
        b = Literal(4.0)
        expr = a * b
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(12.0)

    def test_div(self):
        ev = make_evaluator()
        a = Literal(10.0)
        b = Literal(2.0)
        expr = a / b
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(5.0)

    def test_pow(self):
        ev = make_evaluator()
        a = Literal(2.0)
        expr = a ** Literal(3.0)
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(8.0)

    def test_neg(self):
        ev = make_evaluator()
        a = Literal(5.0)
        expr = -a
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(-5.0)


# ======================================================================
# FunctionCall evaluation
# ======================================================================
class TestEvalFunctionCall:
    def test_sin(self):
        ev = make_evaluator()
        x = Literal(0.0)
        expr = FunctionCall(jnp.sin, [x])
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(0.0, abs=1e-6)

    def test_cos(self):
        ev = make_evaluator()
        x = Literal(0.0)
        expr = FunctionCall(jnp.cos, [x])
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_sum_reduces_axis(self):
        ev = make_evaluator()
        x = Literal([1.0, 2.0, 3.0])
        expr = FunctionCall(jnp.sum, [x], reduces_axis=True)
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(6.0)


# ======================================================================
# Concat evaluation
# (now via jno.jnp_ops.concat which returns a FunctionCall)
# ======================================================================
class TestEvalConcat:
    def test_concat_literals(self):
        import jno.jnp_ops as pnp

        ev = make_evaluator()
        a = Literal(jnp.array([[1.0], [2.0]]))
        b = Literal(jnp.array([[3.0], [4.0]]))
        expr = pnp.concat([a, b])
        result = ev.evaluate(expr, make_points())
        assert result.shape[1] == 2  # concatenated along last axis


# ======================================================================
# Reshape evaluation
# (now via Placeholder.reshape() which returns a FunctionCall)
# ======================================================================
class TestEvalReshape:
    def test_reshape(self):
        ev = make_evaluator()
        x = Literal(jnp.arange(6.0))
        expr = x.reshape((2, 3))
        result = ev.evaluate(expr, make_points())
        assert result.shape == (2, 3)


# ======================================================================
# Slice evaluation
# ======================================================================
class TestEvalSlice:
    def test_slice_int(self):
        ev = make_evaluator()
        x = Literal(jnp.array([10.0, 20.0, 30.0]))
        expr = x[1]
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(20.0)

    def test_slice_range(self):
        ev = make_evaluator()
        x = Literal(jnp.array([10.0, 20.0, 30.0, 40.0]))
        expr = x[1:3]
        result = ev.evaluate(expr, make_points())
        assert result.shape == (2,)
        assert jnp.allclose(result, jnp.array([20.0, 30.0]))


# ======================================================================
# Chained expression evaluation
# ======================================================================
class TestChainedExpressions:
    def test_complex_expression(self):
        """Test (2 * x + 1) where x = Literal(3.0)."""
        ev = make_evaluator()
        x = Literal(3.0)
        expr = Literal(2.0) * x + Literal(1.0)
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(7.0)

    def test_nested_ops(self):
        """Test (a + b) * (a - b) = a^2 - b^2."""
        ev = make_evaluator()
        a = Literal(5.0)
        b = Literal(3.0)
        expr = (a + b) * (a - b)
        result = ev.evaluate(expr, make_points())
        assert float(result) == pytest.approx(16.0)  # 25 - 9


# ======================================================================
# OperationDef / OperationCall evaluation
# ======================================================================
class TestEvalOperations:
    def test_operation_passthrough(self):
        """OperationDef wrapping a literal should evaluate to the literal."""
        ev = make_evaluator()
        lit = Literal(42.0)
        op = OperationDef(lit)
        result = ev.evaluate(op, make_points())
        assert float(result) == pytest.approx(42.0)


# ======================================================================
# ModelCall evaluation
# ======================================================================
class TestEvalFlaxModule:
    def test_dense_layer(self):
        """Evaluate a simple Dense layer (batched Linear) through the trace evaluator."""
        import jax

        from jno.architectures.linear import Linear

        module = Linear(1, 2, key=jax.random.PRNGKey(0))
        fm = Model(module, name="dense")
        x_var = make_var("x")
        call = fm(x_var)

        layer_params = {fm.layer_id: module}

        ev = TraceEvaluator(layer_params)
        points = {"x": jnp.ones((5, 1))}
        result = ev.evaluate(call, points)
        assert result.shape == (5, 2)  # Linear(1, 2) output


# ======================================================================
# compile_traced_expression
# ======================================================================
class TestCompileTracedExpression:
    def test_compile_literal(self):
        """Compile a trivial expression (literal) and call it."""
        lit = Literal(99.0)
        op = OperationDef(lit)
        all_ops = [op]

        compiled = TraceCompiler.compile_traced_expression(lit, all_ops)
        # compiled(params, context, batchsize, key)
        result = compiled({}, make_points())
        assert jnp.allclose(result, 99.0)

    def test_compile_binary_op(self):
        """Compile a + b expression."""
        a = Literal(3.0)
        b = Literal(4.0)
        expr = a + b
        op = OperationDef(expr)
        all_ops = [op]

        compiled = TraceCompiler.compile_traced_expression(expr, all_ops)
        result = compiled({}, make_points())
        assert jnp.allclose(result, 7.0)


# ======================================================================
# Gradient evaluation via Jacobian (AD scheme)
# ======================================================================
class TestEvalGradient:
    def test_gradient_of_square(self):
        """d/dx(x^2) = 2x, evaluated at x=3.0 via single-variable Jacobian."""
        x = make_var("x")
        u = x ** Literal(2.0)
        grad_u = Jacobian(u, [x], scheme="automatic_differentiation")

        # We need to evaluate this through compile_traced_expression
        # because gradient requires JAX tracing
        op = OperationDef(grad_u, [x])
        all_ops = [op]

        compiled = TraceCompiler.compile_traced_expression(grad_u, all_ops)
        points = {"x": jnp.array([[3.0]])}
        result = compiled({}, points)
        assert jnp.allclose(result, 6.0, atol=0.1)


# ======================================================================
# NetworkGradient evaluation and stop_gradient
# ======================================================================
def _make_tiny_net():
    """Return a small MLP (foundax) and its layer_id, for evaluator tests."""
    import jno.jnp_ops as jnn

    key = jax.random.PRNGKey(0)
    net = jnn.nn.wrap(foundax.mlp(1, output_dim=1, hidden_dims=8, num_layers=2, key=key))
    return net


class _LinearWB(eqx.Module):
    """Tiny linear model ``u(x) = w · x + b`` with two scalar parameters.

    Used by ``test_grad_matches_analytic_parameter_jacobian`` below — the
    parameter Jacobian is closed-form: ``∂u/∂w = x``, ``∂u/∂b = 1``.
    """

    w: jax.Array
    b: jax.Array

    def __call__(self, x):
        return self.w * x + self.b


class TestNetworkGradientEval:
    """Evaluate NetworkGradient through TraceEvaluator directly (no crux.core)."""

    def _ctx(self, net, N=5):
        """Build evaluator + context for N spatial points."""
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        params = {net.layer_id: net.module}
        ev = TraceEvaluator(params)
        ctx = ev._EvalCtx(context, {}, None)
        return ev, ctx

    def test_shape_is_N_x_P(self):
        """J = u.grad(net) must have shape (N, P) after direct evaluation."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)
        J = u.grad(net)

        N = 5
        ev, ctx = self._ctx(net, N=N)
        result = ev._dispatch(J, ctx)

        trainable, _ = eqx.partition(net.module, eqx.is_array)
        P = sum(leaf.size for leaf in jax.tree_util.tree_leaves(trainable))
        assert result.shape == (N, P)

    def test_grad_matches_analytic_parameter_jacobian(self):
        """Anchor ``_eval_network_gradient`` to a closed-form parameter
        Jacobian. For ``u(x) = w·x + b`` the parameter Jacobian is
        ``∂u/∂w = x`` and ``∂u/∂b = 1`` at every spatial point, so the
        flattened (N, P=2) output of ``u.grad(net)`` must equal
        ``[[x_0, 1], [x_1, 1], ..., [x_{N-1}, 1]]``.

        This is the only test that anchors the NetworkGradient values
        against ground truth — existing tests only check shape and the
        ``stop_gradient`` equivalence.
        """
        import jno.jnp_ops as jnn

        w0 = jnp.array([2.5])
        b0 = jnp.array([-0.7])
        net = jnn.nn.wrap(_LinearWB(w=w0, b=b0))
        x = make_var("interior")
        u = net(x)
        J = u.grad(net)

        N = 6
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        params = {net.layer_id: net.module}
        ev = TraceEvaluator(params)
        result = ev._dispatch(J, ev._EvalCtx(context, {}, None))

        # eqx.partition with eqx.is_array preserves the field order
        # declared on _LinearWB: w first, then b. So column 0 of J is
        # ∂u/∂w = x, column 1 is ∂u/∂b = 1.
        assert result.shape == (N, 2), f"expected (N=6, P=2), got {result.shape}"
        x_col = context["interior"][:, 0]
        assert jnp.allclose(result[:, 0], x_col, atol=1e-6), (
            f"∂u/∂w column should be x; got {result[:, 0]}, expected {x_col}"
        )
        assert jnp.allclose(result[:, 1], jnp.ones(N), atol=1e-6), f"∂u/∂b column should be ones; got {result[:, 1]}"

    def test_stop_gradient_values_unchanged(self):
        """J and J.stop_gradient must produce identical numerical values."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)
        J = u.grad(net)
        J_sg = J.stop_gradient

        ev, ctx = self._ctx(net, N=4)
        j_val = ev._dispatch(J, ctx)
        j_sg_val = ev._dispatch(J_sg, ctx)
        assert jnp.allclose(j_val, j_sg_val)

    def test_stop_gradient_blocks_second_order_grad(self):
        """jax.grad through J.stop_gradient must return all-zero leaves."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)
        J = u.grad(net)
        J_sg = J.stop_gradient

        N = 3
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        trainable, static = eqx.partition(net.module, eqx.is_array)

        def loss_sg(tp):
            full = eqx.combine(tp, static)
            ev = TraceEvaluator({net.layer_id: full})
            ctx = ev._EvalCtx(context, {}, None)
            return jnp.sum(ev._dispatch(J_sg, ctx))

        grad_sg = jax.grad(loss_sg)(trainable)
        leaves = jax.tree_util.tree_leaves(grad_sg)
        assert all(jnp.allclose(leaf, jnp.zeros_like(leaf)) for leaf in leaves)

    def test_without_stop_gradient_has_nonzero_second_order_grad(self):
        """jax.grad through J (no stop_gradient) must be non-zero for a nonlinear net."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)
        J = u.grad(net)

        N = 3
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        trainable, static = eqx.partition(net.module, eqx.is_array)

        def loss(tp):
            full = eqx.combine(tp, static)
            ev = TraceEvaluator({net.layer_id: full})
            ctx = ev._EvalCtx(context, {}, None)
            return jnp.sum(ev._dispatch(J, ctx))

        grad_full = jax.grad(loss)(trainable)
        leaves = jax.tree_util.tree_leaves(grad_full)
        assert any(jnp.any(leaf != 0) for leaf in leaves)

    # ------------------------------------------------------------------
    # Selector tests
    # ------------------------------------------------------------------

    def test_mask_reduces_P(self):
        """net.mask(bool_pytree) limits the Jacobian to selected parameters."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)

        all_false = jax.tree_util.tree_map(lambda _: False, net.module)
        mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)

        J_all = u.grad(net)
        J_sel = u.grad(net.mask(mask))

        N = 5
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        params = {net.layer_id: net.module}
        ev = TraceEvaluator(params)
        ctx = ev._EvalCtx(context, {}, None)

        j_all = ev._dispatch(J_all, ctx)
        j_sel = ev._dispatch(J_sel, ctx)

        P_sel_expected = net.module.output_layer.weight.size
        assert j_all.shape[0] == N
        assert j_sel.shape == (N, P_sel_expected)
        assert j_sel.shape[1] < j_all.shape[1]

    def test_mask_two_tensors(self):
        """Mask selecting weight + bias gives P = weight.size + bias.size."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)

        all_false = jax.tree_util.tree_map(lambda _: False, net.module)
        mask = eqx.tree_at(
            lambda m: (m.output_layer.weight, m.output_layer.bias),
            all_false,
            (True, True),
        )
        J = u.grad(net.mask(mask))

        N = 4
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        ev = TraceEvaluator({net.layer_id: net.module})
        ctx = ev._EvalCtx(context, {}, None)
        result = ev._dispatch(J, ctx)

        w = net.module.output_layer.weight
        b = net.module.output_layer.bias
        P_expected = w.size + b.size
        assert result.shape == (N, P_expected)

    def test_mask_stop_gradient_blocks_grad(self):
        """stop_gradient on a masked Jacobian blocks all gradient flow."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)

        all_false = jax.tree_util.tree_map(lambda _: False, net.module)
        mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)
        J_sg = u.grad(net.mask(mask)).stop_gradient

        N = 3
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        trainable, static = eqx.partition(net.module, eqx.is_array)

        def loss(tp):
            full = eqx.combine(tp, static)
            ev = TraceEvaluator({net.layer_id: full})
            ctx = ev._EvalCtx(context, {}, None)
            return jnp.sum(ev._dispatch(J_sg, ctx))

        grad_sg = jax.grad(loss)(trainable)
        leaves = jax.tree_util.tree_leaves(grad_sg)
        assert all(jnp.allclose(leaf, jnp.zeros_like(leaf)) for leaf in leaves)


# ======================================================================
# Gradient cosine similarity via NetworkGradient
#
# The Jacobian J = u.grad(net) with shape (N, P) lets you compute
# "virtual gradients" for any loss term without running an actual
# backward pass:
#
#   g = J.T @ residual        # (P,) — gradient direction for that term
#
# Cosine similarity between two loss gradients reveals whether the
# losses are aligned (≈1), orthogonal (≈0), or conflicting (≈-1).
# Use net.mask(sparse_mask) to restrict to a fast sparse subset of
# parameters while preserving the qualitative direction information.
#
# Example (post-training analysis):
#
#   J = crux.eval([u.grad(net.mask(sparse_mask))])[0]  # (N, P_sparse)
#   g_pde = J.T @ pde_residual[:, 0]
#   g_bc  = J.T @ bc_residual[:, 0]
#   cos_sim = jnp.dot(g_pde, g_bc) / (
#       jnp.linalg.norm(g_pde) * jnp.linalg.norm(g_bc)
#   )
# ======================================================================
def _grad_direction(J, residual):
    """Virtual gradient: J^T @ residual → (P,) gradient direction."""
    return J.T @ residual  # (P,)


def _cosine_similarity(g1, g2):
    """Cosine similarity between two gradient vectors."""
    return jnp.dot(g1, g2) / (jnp.linalg.norm(g1) * jnp.linalg.norm(g2))


class TestGradientCosineSimilarity:
    """Use NetworkGradient to compare loss-gradient directions."""

    def _setup(self, N=8):
        """Build a tiny net, its Jacobian, and a context."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)
        J_expr = u.grad(net)

        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        ev = TraceEvaluator({net.layer_id: net.module})
        ctx = ev._EvalCtx(context, {}, None)
        J = ev._dispatch(J_expr, ctx)  # (N, P)
        return J, N

    def test_same_loss_twice_has_cosine_similarity_one(self):
        """Writing the same residual twice must give cos_sim = 1."""
        J, N = self._setup()
        residual = jnp.ones(N)

        g1 = _grad_direction(J, residual)
        g2 = _grad_direction(J, residual)  # identical expression

        cos_sim = _cosine_similarity(g1, g2)
        assert jnp.isclose(cos_sim, 1.0)

    def test_opposite_residuals_have_cosine_similarity_minus_one(self):
        """Flipping the sign of a residual gives cos_sim = -1."""
        J, N = self._setup()
        residual = jnp.ones(N)

        g_pos = _grad_direction(J, residual)
        g_neg = _grad_direction(J, -residual)

        cos_sim = _cosine_similarity(g_pos, g_neg)
        assert jnp.isclose(cos_sim, -1.0)

    def test_sparse_mask_same_loss_twice(self):
        """Same result holds when using a sparse parameter mask."""
        net = _make_tiny_net()
        x = make_var("interior")
        u = net(x)

        all_false = jax.tree_util.tree_map(lambda _: False, net.module)
        mask = eqx.tree_at(lambda m: m.output_layer.weight, all_false, True)
        J_expr = u.grad(net.mask(mask))

        N = 8
        context = {"interior": jnp.linspace(0.1, 0.9, N).reshape(N, 1)}
        ev = TraceEvaluator({net.layer_id: net.module})
        ctx = ev._EvalCtx(context, {}, None)
        J = ev._dispatch(J_expr, ctx)  # (N, P_sparse)

        residual = jnp.linspace(-1.0, 1.0, N)
        g1 = _grad_direction(J, residual)
        g2 = _grad_direction(J, residual)

        assert jnp.isclose(_cosine_similarity(g1, g2), 1.0)
        assert J.shape[1] == net.module.output_layer.weight.size
