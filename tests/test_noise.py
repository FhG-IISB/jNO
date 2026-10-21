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
        crux = jno.core([(u + 0.0).mse])
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
        crux = jno.core([(u + 0.0).mse])

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
        crux = jno.core([(u + 0.0).mse])

        n = jno.noise.gaussian(std=0.01, ndim=ndim)
        (val,) = crux.eval([n])
        # eval returns (B, N, ndim)
        assert val.shape[-1] == ndim


# ---------------------------------------------------------------------------
# like= — shape borrowed from another traced expression
# ---------------------------------------------------------------------------


class TestLike:
    def test_like_matches_the_reference_shape(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net(hidden=8)
        net.optimizer(optax.adam(1e-3))
        u = net(x)  # (N, 1)

        n = jno.noise.gaussian(std=1.0, like=u)
        crux = jno.core([(u + 0.0).mse])
        (val, ref) = crux.eval([n, u], key=jax.random.PRNGKey(0))
        assert val.shape == ref.shape

    def test_like_tracks_a_field_shaped_reference(self):
        # The motivating case: the prior of a flow-matching path takes the DATA's shape, not a
        # pointwise (N, ndim) shape — here a (2, 3) tensor per sample rather than a scalar column.
        import numpy as np

        arr = np.random.default_rng(0).normal(size=(4, 2, 3)).astype(np.float32)
        dom = jno.domain.from_array({"pt": np.zeros((1, 2), dtype=np.float32)})
        dom = 4 * dom
        dom.variable("pt")
        field = dom.variable("_f", arr[:, None, None, :, :])

        n = jno.noise.gaussian(std=1.0, like=field)
        crux = jno.core([(field + 0.0).mse])
        (val, ref) = crux.eval([n, field], key=jax.random.PRNGKey(1))
        assert val.shape == ref.shape == (4, 1, 2, 3)

    def test_like_follows_points_subsampling(self):
        # The property that motivates `like=` over a shape written down by hand: it must track
        # solve(points=...) automatically, since a literal shape would silently go stale under it.
        import numpy as np

        x = np.random.default_rng(0).normal(size=(64, 2)).astype(np.float32)
        dom = jno.domain.from_array({"data": x})
        d0, d1, _ = dom.variable("data")
        target = jno.np.concatenate([d0, d1])

        n = jno.noise.gaussian(std=1.0, like=target)
        crux = jno.core([(0.0 * target + n).mse])
        out = crux.compiled_resample_constraints_fn(
            crux.models, crux.domain_data.context, batchsize=None, key=jax.random.PRNGKey(0), points=8
        )
        assert out[0].shape == (1, 8, 2)

    def test_like_and_ndim_together_raises(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        with pytest.raises(ValueError, match=r"either ndim= or like="):
            jno.noise.gaussian(ndim=2, like=x)

    def test_like_rejects_a_non_placeholder(self):
        import numpy as np

        with pytest.raises(TypeError, match=r"like= expects a traced jNO expression"):
            jno.noise.gaussian(like=np.zeros(3))

    def test_like_works_for_uniform_and_laplace(self):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x)
        crux = jno.core([(u + 0.0).mse])

        for node in (jno.noise.uniform(0.0, 1.0, like=u), jno.noise.laplace(std=1.0, like=u)):
            (val, ref) = crux.eval([node, u], key=jax.random.PRNGKey(2))
            assert val.shape == ref.shape

    def test_two_like_draws_stay_independent(self):
        # The same collapse risk the CSE fix (584058db) guards against: two Noise nodes sharing a
        # `like=` reference must still be two independent draws, not one shared realisation.
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
        x, *_ = dom.variable("interior")
        net = _tiny_net()
        net.optimizer(optax.adam(1e-3))
        u = net(x)
        crux = jno.core([(u + 0.0).mse])

        a = jno.noise.gaussian(std=1.0, like=u)
        b = jno.noise.gaussian(std=1.0, like=u)
        (val,) = crux.eval([a - b], key=jax.random.PRNGKey(3))
        assert float(jnp.abs(val).max()) > 0.0
