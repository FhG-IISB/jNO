"""Tests for FlaxModelWrapper — wrapping Flax models as Equinox modules."""

import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import flax.linen as flax_nn

from jno.architectures.common import FlaxModelWrapper
from jno.architectures.models import nn
from jno.trace import Model


# ======================================================================
# A simple Flax module for testing (avoids heavy ScOT dependency)
# ======================================================================


class _FlaxMLP(flax_nn.Module):
    """Tiny Flax MLP for test purposes."""

    hidden: int = 8
    out_dim: int = 1

    @flax_nn.compact
    def __call__(self, x, deterministic=True):
        x = flax_nn.Dense(self.hidden)(x)
        x = flax_nn.relu(x)
        x = flax_nn.Dense(self.out_dim)(x)
        return x


# ======================================================================
# FlaxModelWrapper unit tests
# ======================================================================


class TestFlaxModelWrapper:

    def test_is_eqx_module(self):
        """Wrapper must be a valid Equinox module."""
        model = _FlaxMLP(hidden=8, out_dim=2)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        assert isinstance(wrapped, eqx.Module)

    def test_forward_pass_shape(self):
        """Forward pass must produce correct output shape."""
        model = _FlaxMLP(hidden=16, out_dim=4)
        rng = jax.random.PRNGKey(1)
        params = model.init(rng, jnp.ones((1, 5)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)

        x = jnp.ones((3, 5))
        y = wrapped(x)
        assert y.shape == (3, 4)

    def test_forward_pass_values(self):
        """Wrapper output must match direct model.apply output."""
        model = _FlaxMLP(hidden=8, out_dim=2)
        rng = jax.random.PRNGKey(2)
        x = jnp.ones((2, 3))
        params = model.init(rng, x)

        direct = model.apply(params, x, deterministic=True)
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        via_wrapper = wrapped(x)
        assert jnp.allclose(direct, via_wrapper)

    def test_key_kwarg_ignored(self):
        """The jNO evaluator passes key= to every model; wrapper must not choke."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(3)
        params = model.init(rng, jnp.ones((1, 2)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)

        y = wrapped(jnp.ones((1, 2)), key=jax.random.PRNGKey(99))
        assert y.shape == (1, 1)

    def test_eqx_partition_combine(self):
        """Params must survive eqx.partition / eqx.combine round-trip."""
        model = _FlaxMLP(hidden=8, out_dim=2)
        rng = jax.random.PRNGKey(4)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)

        trainable, static = eqx.partition(wrapped, eqx.is_array)
        recombined = eqx.combine(trainable, static)

        x = jnp.ones((2, 3))
        assert jnp.allclose(wrapped(x), recombined(x))

    def test_jit_compatible(self):
        """Wrapped model must be JIT-compilable."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(5)
        params = model.init(rng, jnp.ones((1, 4)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)

        @jax.jit
        def f(m, x):
            return m(x)

        x = jnp.ones((2, 4))
        y = f(wrapped, x)
        assert y.shape == (2, 1)
        assert jnp.allclose(y, wrapped(x))

    def test_grad_through_params(self):
        """Gradients must flow through the Flax params."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(6)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)

        @eqx.filter_grad
        def loss_grad(m):
            return jnp.sum(m(jnp.ones((1, 3))))

        grads = loss_grad(wrapped)
        # The gradient pytree must have the same shape and contain non-zero arrays
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        assert len(grad_leaves) > 0
        assert any(jnp.any(g != 0) for g in grad_leaves)

    def test_optax_update(self):
        """Training step with optax must work end-to-end."""
        import optax

        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(7)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)

        opt = optax.adam(1e-3)
        opt_state = opt.init(eqx.filter(wrapped, eqx.is_array))

        @eqx.filter_value_and_grad
        def loss_fn(m):
            return jnp.mean(m(jnp.ones((4, 3))) ** 2)

        loss_before, grads = loss_fn(wrapped)
        updates, opt_state = opt.update(
            eqx.filter(grads, eqx.is_array),
            opt_state,
            eqx.filter(wrapped, eqx.is_array),
        )
        new_wrapped = eqx.apply_updates(wrapped, updates)
        loss_after, _ = loss_fn(new_wrapped)

        # Loss should change after one step (very likely to decrease)
        assert not jnp.allclose(loss_before, loss_after)

    def test_default_kwargs_override(self):
        """Per-call kwargs must override defaults."""

        class _FlaxWithFlag(flax_nn.Module):
            @flax_nn.compact
            def __call__(self, x, flag=False):
                if flag:
                    return x * 2
                return x

        model = _FlaxWithFlag()
        rng = jax.random.PRNGKey(8)
        params = model.init(rng, jnp.ones((1, 2)))

        wrapped = FlaxModelWrapper(model.apply, params, flag=False)
        x = jnp.ones((1, 2))
        assert jnp.allclose(wrapped(x), x)
        assert jnp.allclose(wrapped(x, flag=True), x * 2)


# ======================================================================
# Integration: nn.wrap + FlaxModelWrapper
# ======================================================================


class TestFlaxWrapperInPipeline:

    def test_nn_wrap_returns_flax_module(self):
        """nn.wrap(FlaxModelWrapper(...)) must yield a Model."""
        model = _FlaxMLP(hidden=8, out_dim=2)
        rng = jax.random.PRNGKey(10)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped, name="test_flax")
        assert isinstance(fm, Model)
        assert fm.name == "test_flax"

    def test_wrapped_module_callable_through_flax_module(self):
        """The inner module should be callable via .module(...)."""
        model = _FlaxMLP(hidden=8, out_dim=2)
        rng = jax.random.PRNGKey(11)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped)

        x = jnp.ones((5, 3))
        y = fm.module(x)
        assert y.shape == (5, 2)

    def test_build_single_layer_params_equinox_path(self):
        """FlaxModelWrapper should pass the isinstance(module, eqx.Module) check."""
        from jno.trace_evaluator import TraceEvaluator
        from jno.utils import get_logger

        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(12)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped)

        logger = get_logger(log_print=(False, False))
        built = TraceEvaluator.build_single_layer_params(
            fm,
            arg_shapes=[(5, 3)],
            rng=jax.random.PRNGKey(0),
            logger=logger,
        )
        assert isinstance(built, FlaxModelWrapper)
        # Must preserve callable behaviour
        y = built(jnp.ones((5, 3)))
        assert y.shape == (5, 1)


# ======================================================================
# ScOT integration (lightweight: just check wrapping, not full forward)
# ======================================================================


# class TestScotWrapper:
#
#    @pytest.mark.slow
#    def test_nn_scot_returns_flax_module(self):
#        """nn.scot() should return a Model wrapping a FlaxModelWrapper."""
#        m = nn.scot(
#            name="test_scot",
#            image_size=32,
#            patch_size=4,
#            num_channels=1,
#            num_out_channels=1,
#            embed_dim=24,
#            depths=(1, 1, 1, 1),
#            num_heads=(3, 3, 3, 3),
#            window_size=8,
#        )
#        assert isinstance(m, Model)
#        assert isinstance(m.module, FlaxModelWrapper)
#
#    @pytest.mark.slow
#    def test_nn_scot_forward_pass(self):
#        """nn.scot() model must produce correct output shape on forward pass."""
#        m = nn.scot(
#            name="test_scot",
#            image_size=32,
#            patch_size=4,
#            num_channels=2,
#            num_out_channels=1,
#            embed_dim=24,
#            depths=(1, 1, 1, 1),
#            num_heads=(3, 3, 3, 3),
#            window_size=8,
#        )
#        x = jnp.ones((1, 32, 32, 2))
#        t = jnp.zeros((1,))
#        out = m.module(pixel_values=x, time=t)
#        assert out.shape == (1, 32, 32, 1)
#
#    @pytest.mark.slow
#    def test_nn_scot_gradient_flow(self):
#        """Gradients must flow through ScOT wrapper."""
#        m = nn.scot(
#            name="test_scot",
#            image_size=32,
#            patch_size=4,
#            num_channels=1,
#            num_out_channels=1,
#            embed_dim=24,
#            depths=(1, 1, 1, 1),
#            num_heads=(3, 3, 3, 3),
#            window_size=8,
#        )
#        model = m.module
#
#        @eqx.filter_grad
#        def loss_grad(model):
#            out = model(pixel_values=jnp.ones((1, 32, 32, 1)), time=jnp.zeros((1,)))
#            return jnp.mean(out)
#
#        grads = loss_grad(model)
#        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
#        assert len(grad_leaves) > 0
#        assert any(jnp.any(g != 0) for g in grad_leaves)
# ======================================================================
# Model.initialize() tests
# ======================================================================


class TestInitialize:

    def test_initialize_sets_weight_path(self):
        """initialize() must store the path and return self for chaining."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped)

        result = fm.initialize("/tmp/fake_weights.msgpack")
        assert result is fm
        assert fm.weight_path == "/tmp/fake_weights.msgpack"

    def test_initialize_chains_with_dont_show(self):
        """initialize() should be chainable with other Model methods."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped).dont_show().initialize("/tmp/w.msgpack")

        assert fm.show is False
        assert fm.weight_path == "/tmp/w.msgpack"


# ======================================================================
# dtype casting tests
# ======================================================================


class TestDtype:

    def test_dtype_sets_attribute(self):
        """dtype() must store the target dtype and return self."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped)

        result = fm.dtype(jnp.bfloat16)
        assert result is fm
        assert fm._dtype == jnp.bfloat16

    def test_dtype_chains_with_initialize(self):
        """dtype() should chain with initialize() and dont_show()."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped).dont_show().initialize("/tmp/w.msgpack").dtype(jnp.bfloat16)

        assert fm.show is False
        assert fm.weight_path == "/tmp/w.msgpack"
        assert fm._dtype == jnp.bfloat16

    def test_reset_clears_dtype(self):
        """reset() must clear _dtype back to None."""
        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        fm = nn.wrap(wrapped)
        fm.dtype(jnp.bfloat16)
        fm.reset()
        assert fm._dtype is None

    def test_cast_model_dtype_flax_wrapper(self):
        """_cast_model_dtype should convert float arrays, leave ints alone."""
        import logging
        from jno.trace_evaluator import TraceEvaluator

        model = _FlaxMLP(hidden=8, out_dim=1)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 3)))
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)

        logger = logging.getLogger("test_dtype")
        cast = TraceEvaluator._cast_model_dtype(wrapped, jnp.bfloat16, logger)

        # All float leaves should be bfloat16
        leaves = jax.tree_util.tree_leaves(cast.params)
        for leaf in leaves:
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
                assert leaf.dtype == jnp.bfloat16, f"Expected bfloat16, got {leaf.dtype}"

    def test_cast_model_dtype_equinox(self):
        """_cast_model_dtype should work on plain Equinox modules."""
        import logging
        from jno.trace_evaluator import TraceEvaluator

        model = eqx.nn.Linear(3, 4, key=jax.random.PRNGKey(0))
        logger = logging.getLogger("test_dtype")
        cast = TraceEvaluator._cast_model_dtype(model, jnp.bfloat16, logger)

        assert cast.weight.dtype == jnp.bfloat16
        assert cast.bias.dtype == jnp.bfloat16

    def test_dtype_preserves_int_arrays(self):
        """Integer arrays in params must not be cast."""
        import logging
        from jno.trace_evaluator import TraceEvaluator

        params = {"w": jnp.ones((3, 4), dtype=jnp.float32), "idx": jnp.array([1, 2, 3], dtype=jnp.int32)}
        model = FlaxModelWrapper(lambda p, x: x, params)
        logger = logging.getLogger("test_dtype")
        cast = TraceEvaluator._cast_model_dtype(model, jnp.bfloat16, logger)

        assert cast.params["w"].dtype == jnp.bfloat16
        assert cast.params["idx"].dtype == jnp.int32


# ======================================================================
# nn.poseidon API tests (removed — poseidon now requires weights_path)
# ======================================================================
