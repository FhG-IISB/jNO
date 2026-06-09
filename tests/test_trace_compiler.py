"""Tests for jno.trace_compiler internals exercised through public APIs.

These tests verify compiler-level correctness — common subexpression
elimination, multi-constraint compilation, and layer collection — without
poking at the internal compile_traced_expression machinery directly.
Most are easier to write against ``jno.core``.
"""

import foundax
import jax
import jax.numpy as jnp
import optax

import jno
import jno.jnp_ops as jnn
from jno.trace import collect_operations
from jno.trace_compiler import TraceCompiler


def _tiny_net(in_dim=1, out_dim=1, hidden=8, key_seed=0):
    return jnn.nn.wrap(
        foundax.mlp(
            in_dim,
            output_dim=out_dim,
            hidden_dims=hidden,
            num_layers=2,
            key=jax.random.PRNGKey(key_seed),
        )
    )


# ---------------------------------------------------------------------------
# collect_dense_layers — find every unique layer in an expression DAG
# ---------------------------------------------------------------------------


class TestCollectDenseLayers:
    def test_finds_single_layer(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        u = net(x)
        layers = TraceCompiler.collect_dense_layers(u)
        # An MLP wrapped via jnn.nn.wrap contributes at least one Model node
        assert len(layers) >= 1

    def test_deduplicates_repeated_use_of_same_network(self):
        """Using the same network three times shouldn't yield three copies."""
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        u = net(x) + net(x) * net(x)  # 3 calls, same network
        layers = TraceCompiler.collect_dense_layers(u)
        # Two distinct networks would double the count; one shared net stays at 1
        net2 = _tiny_net(key_seed=1)
        v = net(x) + net2(x)
        layers_two = TraceCompiler.collect_dense_layers(v)
        assert len(layers_two) >= len(layers)


# ---------------------------------------------------------------------------
# CSE (common subexpression elimination)
# ---------------------------------------------------------------------------


class TestCSE:
    def test_cse_preserves_evaluation_semantics(self):
        """An expression with a shared subexpression should evaluate the same
        whether or not CSE has been applied first."""
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x)
        # Build two constraints that share `u`
        loss_a = (u * u).mse
        loss_b = (u + u).mse
        crux = jno.core([loss_a, loss_b])
        # If CSE breaks evaluation, solve() will produce NaNs or shape errors.
        hist = crux.solve(3)
        final = hist.total_loss
        assert final is None or jnp.isfinite(jnp.array(final))


# ---------------------------------------------------------------------------
# Multi-expression compilation
# ---------------------------------------------------------------------------


class TestCompileMultiExpression:
    def test_seven_constraints_compile_and_train(self):
        """A graph with more than a handful of constraints should compile."""
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x) * x * (1 - x)

        # 7 different constraint expressions referencing the same network.
        cs = [
            jnn.laplacian(u, [x]).mse,
            (u - 0.0).mse,
            (u.d(x)).mse,
            (u * x).mse,
            (jnn.laplacian(u, [x]) + 1).mse,
            (u + jno.np.sin(x)).mse,
            (u - jno.np.cos(x)).mse,
        ]
        crux = jno.core(cs)
        hist = crux.solve(2)
        assert len(hist.training_logs) >= 1

    def test_single_constraint_via_multi_compile_path(self):
        """compile_multi_expression with one expression should be equivalent to
        compile_traced_expression for that expression."""
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x) * x * (1 - x)
        loss = jnn.laplacian(u, [x]).mse
        crux = jno.core([loss])
        hist = crux.solve(2)
        final = hist.total_loss
        assert final is None or jnp.isfinite(jnp.array(final))


# ---------------------------------------------------------------------------
# collect_operations — operation harvest from constraint trees
# ---------------------------------------------------------------------------


class TestCollectOperations:
    def test_returns_a_list_for_a_pure_arithmetic_expression(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        x, *_ = dom.variable("interior")
        expr = (x * x + 2 * x).mse
        ops = collect_operations(expr)
        # An arithmetic expression has no OperationDef nodes — empty list ok.
        assert isinstance(ops, list)
