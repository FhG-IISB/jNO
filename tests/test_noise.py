"""Tests for jno.noise — symbolic noise terms used in training and eval."""

import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno
import jno.jnp_ops as jnn


def _tiny_net(in_dim=1, hidden=8, key_seed=0):
    return jnn.nn.wrap(
        foundax.mlp(
            in_dim,
            output_dim=1,
            hidden_dims=hidden,
            num_layers=2,
            key=jax.random.PRNGKey(key_seed),
        )
    )


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


class TestConstructors:
    def test_gaussian_returns_noise_node(self):
        from jno.trace import Noise

        n = jno.noise.gaussian(std=0.1)
        assert isinstance(n, Noise)
        assert n.distribution == "gaussian"
        assert n.params["std"] == pytest.approx(0.1)
        assert n.params["ndim"] == 1

    def test_uniform_returns_noise_node_with_bounds(self):
        from jno.trace import Noise

        n = jno.noise.uniform(low=-0.5, high=0.5, ndim=2)
        assert isinstance(n, Noise)
        assert n.distribution == "uniform"
        assert n.params["low"] == pytest.approx(-0.5)
        assert n.params["high"] == pytest.approx(0.5)
        assert n.params["ndim"] == 2

    def test_laplace_returns_noise_node(self):
        from jno.trace import Noise

        n = jno.noise.laplace(std=0.3, ndim=3)
        assert isinstance(n, Noise)
        assert n.distribution == "laplace"
        assert n.params["std"] == pytest.approx(0.3)
        assert n.params["ndim"] == 3

    def test_unknown_distribution_raises_at_eval_time(self):
        from jno.trace import Noise

        # Constructing with a bogus distribution is allowed; evaluation with a
        # key (training mode) should fail clearly.
        bogus = Noise("not_a_distribution")
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.2))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x)
        crux = jno.core([(u + 0.0).mse], dom)
        with pytest.raises(ValueError, match="Unknown noise distribution"):
            crux.eval([u + bogus], key=jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# Eval semantics: zeros when key=None
# ---------------------------------------------------------------------------


class TestEvalModeReturnsZeros:
    def test_eval_without_key_returns_zeros(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x)
        crux = jno.core([(u + 0.0).mse], dom)

        # gaussian(std=10) is large; in eval mode it should still return zeros.
        n = jno.noise.gaussian(std=10.0)
        (val,) = crux.eval([n])
        assert jnp.allclose(val, 0.0)


# ---------------------------------------------------------------------------
# Output shape (N, ndim)
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_shape_trailing_dim_matches_ndim(self, ndim):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x)
        crux = jno.core([(u + 0.0).mse], dom)

        n = jno.noise.gaussian(std=0.01, ndim=ndim)
        (val,) = crux.eval([n])
        # eval returns (B, N, ndim)
        assert val.shape[-1] == ndim
