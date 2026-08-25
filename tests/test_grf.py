"""``jno.noise.grf`` — a spatially correlated random field, redrawn every step.

The point of shipping this rather than letting users hand-roll it with ``jno.fn`` is not the fifteen
lines of spectral math; it is that the hand-rolled version has two silent failure modes. Every
``jno.fn`` node receives the SAME ``ctx.key``, so two "independent" fields come out identical, and a
DIY function crashes under ``crux.eval()`` where the key is ``None``. Both are pinned below,
alongside a statistical oracle for the covariance.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


def _line(mesh_size=0.02):
    dom = jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))
    x, *_ = dom.variable("interior")
    return dom, x, jnp.asarray(np.asarray(dom.context["interior"]).reshape(-1, 1))


def _draws(node, xs, n=1500, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return np.asarray(jax.vmap(lambda k: node.fn(xs, key=k))(keys))


class TestStatistics:
    def test_covariance_matches_the_analytic_matern(self):
        """The oracle: empirical covariance vs k(r) = σ²(1 + √3r/ℓ)e^{-√3r/ℓ}, not a restatement."""
        _, x, xs = _line()
        ls, var = 0.15, 2.0
        d = _draws(jno.noise.grf(x, length_scale=ls, variance=var, nu=1.5, modes=512), xs, n=3000)[:, :, 0]

        xc = d - d.mean(0)
        emp = (xc.T @ xc) / (d.shape[0] - 1)
        r = np.abs(np.asarray(xs)[:, 0][:, None] - np.asarray(xs)[:, 0][None, :])
        a = np.sqrt(3.0) * r / ls
        analytic = var * (1.0 + a) * np.exp(-a)

        # O(M^-1/2) at M=512 is ~4.4% of the marginal variance; allow 10% and no bias.
        assert np.abs(emp - analytic).max() < 0.10 * var
        assert abs(float(np.mean(emp - analytic))) < 0.02 * var

    def test_marginal_variance(self):
        _, x, xs = _line()
        d = _draws(jno.noise.grf(x, length_scale=0.2, variance=3.0, modes=512), xs, n=3000)
        assert float(np.var(d)) == pytest.approx(3.0, rel=0.05)

    def test_longer_length_scale_is_more_correlated(self):
        """Monotonicity — the property a user actually reasons about."""
        _, x, xs = _line()

        def corr_at_lag(ls, lag=5):
            d = _draws(jno.noise.grf(x, length_scale=ls, modes=512), xs)[:, :, 0]
            xc = d - d.mean(0)
            c = (xc.T @ xc) / (d.shape[0] - 1)
            return float(np.mean(np.diag(c, k=lag)) / np.mean(np.diag(c)))

        assert corr_at_lag(0.02) < corr_at_lag(0.10) < corr_at_lag(0.50)


class TestExtremes:
    def test_length_scale_far_above_the_domain_is_near_constant(self):
        _, x, xs = _line()
        d = _draws(jno.noise.grf(x, length_scale=1e4, modes=256), xs, n=200)[:, :, 0]
        assert float(np.mean(d.std(axis=1))) < 1e-2  # flat within each realisation

    def test_length_scale_far_below_the_spacing_decorrelates(self):
        _, x, xs = _line()
        d = _draws(jno.noise.grf(x, length_scale=1e-4, modes=512), xs, n=1500)[:, :, 0]
        xc = d - d.mean(0)
        c = (xc.T @ xc) / (d.shape[0] - 1)
        assert abs(float(np.mean(np.diag(c, k=1))) / float(np.mean(np.diag(c)))) < 0.2

    def test_zero_variance_is_exactly_zero(self):
        _, x, xs = _line()
        assert np.all(_draws(jno.noise.grf(x, variance=0.0), xs, n=8) == 0.0)

    def test_single_mode_is_finite_and_correctly_scaled(self):
        _, x, xs = _line()
        d = _draws(jno.noise.grf(x, variance=1.0, modes=1), xs, n=3000)
        assert np.all(np.isfinite(d))
        assert float(np.var(d)) == pytest.approx(1.0, rel=0.15)

    def test_rbf_kernel_and_two_dimensions(self):
        dom = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=(7, 7)).domain()
        x, y, _ = dom.variable("interior")
        pts = jnp.asarray(np.asarray(dom.context["interior"]).reshape(-1, 2))
        node = jno.noise.grf(x, y, kernel="rbf", length_scale=0.3)
        out = node.fn(pts[:, :1], pts[:, 1:], key=jax.random.PRNGKey(0))
        assert out.shape == (pts.shape[0], 1) and np.all(np.isfinite(np.asarray(out)))

    def test_ndim_gives_independent_components(self):
        _, x, xs = _line()
        d = _draws(jno.noise.grf(x, length_scale=0.2, ndim=3, modes=512), xs, n=2000)
        assert d.shape[-1] == 3
        c01 = np.corrcoef(d[:, :, 0].ravel(), d[:, :, 1].ravel())[0, 1]
        assert abs(float(c01)) < 0.1


class TestValidation:
    @pytest.mark.parametrize(
        "kw, match",
        [
            ({"length_scale": 0.0}, "length_scale"),
            ({"length_scale": -1.0}, "length_scale"),
            ({"variance": -1.0}, "variance"),
            ({"modes": 0}, "modes"),
            ({"kernel": "cauchy"}, "unknown kernel"),
            ({"nu": 0.0}, "nu must be"),
        ],
    )
    def test_bad_arguments_raise(self, kw, match):
        _, x, _ = _line(mesh_size=0.2)
        with pytest.raises(ValueError, match=match):
            jno.noise.grf(x, **kw)

    def test_no_coordinates_raises(self):
        with pytest.raises(ValueError, match="coordinate Variables"):
            jno.noise.grf()


class TestTheReasonThisIsNotHandRolled:
    def test_two_nodes_are_independent_fields(self):
        """A `jno.fn` GRF gets the same ctx.key for every node, so two of them are IDENTICAL."""
        _, x, xs = _line()
        k = jax.random.PRNGKey(0)
        f1 = np.asarray(jno.noise.grf(x, length_scale=0.2).fn(xs, key=k))
        f2 = np.asarray(jno.noise.grf(x, length_scale=0.2).fn(xs, key=k))
        assert not np.allclose(f1, f2)

    def test_same_node_same_key_is_reproducible(self):
        _, x, xs = _line()
        node = jno.noise.grf(x, length_scale=0.2)
        k = jax.random.PRNGKey(3)
        np.testing.assert_array_equal(np.asarray(node.fn(xs, key=k)), np.asarray(node.fn(xs, key=k)))

    def test_eval_without_a_key_returns_zeros_instead_of_crashing(self):
        """A DIY function raises `unexpected PRNG key type NoneType` here."""
        dom, x, _ = _line(mesh_size=0.1)
        net = jno.nn(__import__("foundax").mlp(1, output_dim=1, key=jax.random.PRNGKey(0)))
        net.optimizer(optax.adam(1e-3))
        crux = jno.core([(net(x) + 0.0).mse])
        (val,) = crux.eval([jno.noise.grf(x, length_scale=0.2)])
        assert np.allclose(np.asarray(val), 0.0)
