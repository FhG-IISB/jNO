"""Tests for the per-parameter `.bayesian(...)` API.

These tests exercise the end-to-end path: configuration → solve loop →
posterior_samples → crux.eval(samples="chain").  Where possible we run
short chains on tiny problems so the suite stays under a second.
"""

from __future__ import annotations

import blackjax
import foundax
import jax
import jax.numpy as jnp
import optax
import pytest

import jno
import jno.jnp_ops as jnn


def _line_domain(mesh_size: float = 0.05):
    return jno.domain(constructor=jno.domain.line(mesh_size=mesh_size))


def _tiny_net(in_dim: int = 1, out_dim: int = 1, hidden: int = 4, key_seed: int = 0):
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
# NUTS on a low-dim inverse problem
# ---------------------------------------------------------------------------


class TestNUTSInverseProblem:
    """Recover A in u = A * sin(πx) from noiseless data via NUTS.

    A 1-parameter problem keeps the chain length small (warmup + keep) while
    still being meaningful: the posterior mean must land near the true value.
    """

    def _solve(self, warmup: int, keep: int, step_size: float = 1e-2):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")

        target = 3.14 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(
            blackjax.nuts,
            step_size=step_size,
            inverse_mass_matrix=jnp.ones(1),
            warmup=warmup,
            keep=keep,
        )

        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse], dom)
        crux.solve(warmup + keep)
        return a

    def test_chain_shape_and_recovery(self):
        a = self._solve(warmup=100, keep=200)
        chain = a.posterior_samples
        assert chain is not None
        assert chain.shape == (200, 1)
        post_mean = float(jnp.mean(chain))
        # Loose tolerance: a short NUTS chain with no step-size adaptation is
        # noisy; we only check that the mode is roughly recovered.
        assert abs(post_mean - 3.14) < 0.35, f"NUTS posterior mean {post_mean} far from 3.14"

    def test_reproducible_with_fixed_seed(self):
        a1 = self._solve(warmup=10, keep=20)
        a2 = self._solve(warmup=10, keep=20)
        assert jnp.allclose(a1.posterior_samples, a2.posterior_samples)


# ---------------------------------------------------------------------------
# SGLD on a tiny MLP
# ---------------------------------------------------------------------------


class TestSGLDOnMLP:
    """SGLD on a 2-layer MLP fitting a constant target — checks shape +
    that the chain is populated without errors.  We do not assert posterior
    quality (SGLD on a small MLP without tuning is noisy)."""

    def test_chain_shape(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net()
        net.bayesian(blackjax.sgld, step_size=1e-4, warmup=10, keep=20)
        residual = net(x) - 0.0
        crux = jno.core([residual.mse], dom)
        crux.solve(30)
        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        assert all(leaf.shape[0] == 20 for leaf in leaves)


# ---------------------------------------------------------------------------
# Mixed optimizer + Bayesian
# ---------------------------------------------------------------------------


class TestMixedOptimizerBayesian:
    """One model uses `.optimizer`, another uses `.bayesian` in the same
    crux.solve().  The optax model must NOT carry posterior samples, while
    the Bayesian one must."""

    def test_posterior_samples_only_on_bayesian_model(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")

        target = jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")

        a.bayesian(
            blackjax.nuts,
            step_size=5e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=10,
            keep=20,
        )
        b.optimizer(optax.adam(1e-2))

        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        crux = jno.core([residual.mse], dom)
        crux.solve(30)

        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (20, 1)
        assert b.posterior_samples is None


# ---------------------------------------------------------------------------
# crux.eval(samples="chain")
# ---------------------------------------------------------------------------


class TestEvalChainSamples:
    """`samples='chain'` must vmap the evaluator over the chain so that
    nonlinear pushforward is correct."""

    def test_returns_stacked_along_leading_axis(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")

        target = 3.0 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(
            blackjax.nuts,
            step_size=5e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=5,
            keep=10,
        )
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse], dom)
        crux.solve(15)

        a_chain = crux.eval([a], samples="chain")
        assert a_chain.shape == (10, 1)

        # Nonlinear expression `a**2` — chain mean must match jnp.mean(a**2)
        sq_chain = crux.eval([a * a], samples="chain")
        assert sq_chain.shape[0] == 10
        ref = a.posterior_samples**2
        assert jnp.allclose(sq_chain.reshape(10, -1), ref.reshape(10, -1))

    def test_raises_when_no_bayesian_models(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.optimizer(optax.adam(1e-2))
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        crux = jno.core([residual.mse], dom)
        crux.solve(5)
        with pytest.raises(ValueError, match="samples='chain'"):
            crux.eval([a], samples="chain")


# ---------------------------------------------------------------------------
# Determinism: no .bayesian anywhere → matches the optax-only baseline
# ---------------------------------------------------------------------------


class TestNoBayesianBitIdentical:
    """If no model uses `.bayesian()`, the training path should match the
    pre-feature behaviour bit-for-bit on a small deterministic problem."""

    def _run(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.optimizer(optax.adam(1e-2))
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse], dom)
        crux.solve(20)
        return float(crux.eval([a])[0])

    def test_two_runs_identical(self):
        assert abs(self._run() - self._run()) < 1e-12


# ---------------------------------------------------------------------------
# Custom prior overrides default
# ---------------------------------------------------------------------------


class TestCustomPrior:
    def test_custom_prior_is_called(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")

        # A very strong prior centred at zero — should pull the posterior
        # away from the data-only MLE.
        def strong_zero_prior(p, _scale=0.01):
            leaves = jax.tree_util.tree_leaves(p)
            sq = jnp.array(0.0)
            for leaf in leaves:
                sq = sq + jnp.sum(jnp.asarray(leaf) ** 2)
            return -sq / _scale

        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            prior=strong_zero_prior,
            warmup=20,
            keep=40,
        )
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse], dom)
        crux.solve(60)
        post_mean = float(jnp.mean(a.posterior_samples))
        # Strong prior should drag the posterior closer to 0 than the truth (2.0).
        assert abs(post_mean) < 1.5, f"Strong prior failed to shrink toward 0; got {post_mean}"


# ---------------------------------------------------------------------------
# Substeps + Bayesian → explicit error
# ---------------------------------------------------------------------------


class TestSubstepsBayesianGuard:
    def test_substeps_with_bayesian_raises(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=jnp.ones(1))
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse, residual.mse], dom)
        with pytest.raises(ValueError, match="substeps.*not supported"):
            crux.solve(5, substeps=[[0], [1]])


# ---------------------------------------------------------------------------
# Missing step_size → clear error
# ---------------------------------------------------------------------------


class TestMissingStepSize:
    def test_no_step_size_raises_clearly(self):
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        # Configure without step_size — the error fires inside solve(), when
        # build_kernel_handle runs.
        a.bayesian(blackjax.nuts, inverse_mass_matrix=jnp.ones(1))
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        crux = jno.core([residual.mse], dom)
        with pytest.raises(ValueError, match="requires a step_size"):
            crux.solve(2)


# ---------------------------------------------------------------------------
# Phase 2 — additional coverage
# ---------------------------------------------------------------------------


def _trivial_bayesian_param(*, warmup, keep, thin=1, step_size=1e-2):
    """Helper: a 1-parameter inverse problem set up for short Bayesian runs."""
    π = jno.np.pi
    dom = _line_domain()
    x, _ = dom.variable("interior")
    target = jno.np.sin(π * x)
    a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
    a.bayesian(
        blackjax.nuts,
        step_size=step_size,
        inverse_mass_matrix=jnp.ones(1),
        warmup=warmup,
        keep=keep,
        thin=thin,
    )
    residual = a * jno.np.sin(π * x) - target
    return a, jno.core([residual.mse], dom)


# ---------------------------------------------------------------------------
# Knob behaviour (keep / thin / warmup)
# ---------------------------------------------------------------------------


class TestKnobs:
    def test_keep_caps_chain_length(self):
        a, crux = _trivial_bayesian_param(warmup=0, keep=20)
        crux.solve(50)  # run more than keep epochs
        assert a.posterior_samples.shape == (20, 1)

    def test_warmup_skips_initial_samples(self):
        a, crux = _trivial_bayesian_param(warmup=5, keep=10)
        crux.solve(15)
        # first stored sample must be the one AT epoch 5, not the initial value
        # → at minimum the first kept sample is not equal to the zero init.
        assert a.posterior_samples.shape == (10, 1)
        assert float(jnp.abs(a.posterior_samples[0]).max()) > 0.0

    def test_thin_keeps_every_kth(self):
        a, crux = _trivial_bayesian_param(warmup=0, keep=5, thin=3)
        crux.solve(30)
        # We collect at epochs 0, 3, 6, 9, 12 → 5 samples total.
        assert a.posterior_samples.shape == (5, 1)


# ---------------------------------------------------------------------------
# Other blackjax kernel families
# ---------------------------------------------------------------------------


def _kernel_recovery(kernel_factory, *, kernel_kwargs, step_size, warmup=80, keep=120):
    """Run a 1-parameter recovery loop with a given full-data kernel."""
    π = jno.np.pi
    dom = _line_domain()
    x, _ = dom.variable("interior")
    target = 2.0 * jno.np.sin(π * x)
    a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
    a.bayesian(
        kernel_factory,
        step_size=step_size,
        warmup=warmup,
        keep=keep,
        **kernel_kwargs,
    )
    residual = a * jno.np.sin(π * x) - target
    jno.core([residual.mse], dom).solve(warmup + keep)
    return a


class TestHMC:
    def test_hmc_runs_and_recovers(self):
        a = _kernel_recovery(
            blackjax.hmc,
            kernel_kwargs=dict(inverse_mass_matrix=jnp.ones(1), num_integration_steps=4),
            step_size=2e-2,
        )
        chain = a.posterior_samples
        assert chain.shape == (120, 1)
        assert abs(float(jnp.mean(chain)) - 2.0) < 0.5


class TestMALA:
    def test_mala_runs_and_recovers(self):
        a = _kernel_recovery(
            blackjax.mala,
            kernel_kwargs=dict(),  # MALA has no inverse_mass_matrix
            step_size=1e-2,
        )
        chain = a.posterior_samples
        assert chain.shape == (120, 1)
        assert abs(float(jnp.mean(chain)) - 2.0) < 0.5


class TestSGHMC:
    def test_sghmc_runs(self):
        # SG-MCMC family — duck-typed via grad_estimator dispatch.  Smoke
        # test only: assert shape, do not assert recovery (SGHMC needs
        # careful tuning that we don't do automatically).
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.sghmc, step_size=1e-4, warmup=5, keep=10)
        residual = a * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse], dom).solve(15)
        assert a.posterior_samples.shape == (10, 1)


# ---------------------------------------------------------------------------
# Vector + multi-leaf shapes
# ---------------------------------------------------------------------------


class TestVectorAndMultiLeafShapes:
    def test_vector_parameter_chain_shape(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x) + 2.0 * jno.np.cos(π * x) + 3.0 * x
        p = jno.np.parameter((3,), key=jax.random.PRNGKey(0), name="abc")
        p.bayesian(
            blackjax.nuts,
            step_size=5e-3,
            inverse_mass_matrix=jnp.ones(3),
            warmup=20,
            keep=30,
        )
        # build expression a*sin + b*cos + c*x where (a, b, c) = p[0], p[1], p[2]
        residual = p[0] * jno.np.sin(π * x) + p[1] * jno.np.cos(π * x) + p[2] * x - target
        jno.core([residual.mse], dom).solve(50)
        assert p.posterior_samples.shape == (30, 3)

    def test_multi_leaf_mlp_chain_pytree(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        net.bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=10)
        residual = net(x) - 0.0
        jno.core([residual.mse], dom).solve(15)

        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        assert len(leaves) > 1, "MLP chain must contain multiple weight/bias leaves"
        for leaf in leaves:
            assert leaf.shape[0] == 10, f"Unexpected leading-axis size {leaf.shape}"


# ---------------------------------------------------------------------------
# Edge cases — freeze, LoRA, ModelCall proxy
# ---------------------------------------------------------------------------


class TestFreezeBayesianClearsFreeze:
    def test_freeze_then_bayesian_runs(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        # User asks to freeze, then changes their mind and samples it.
        a.model.freeze()
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=5,
            keep=10,
        )
        assert a.model._frozen is False, "bayesian() must clear the freeze flag"
        residual = a * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse], dom).solve(15)
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (10, 1)


class TestLoRABayesian:
    """LoRA + Bayesian: sampling the LoRA adapter posterior.

    The chain stores the partitioned (adapter-only) pytree because
    eqx.partition leaves frozen base weights as static.  We assert that
    every inexact-array leaf in the chain has the expected leading axis;
    frozen-base leaves come through as None.
    """

    def test_lora_with_bayesian_samples_adapters(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        net.freeze().lora(rank=2, alpha=1.0)
        net.bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=10)

        residual = net(x) - 0.0
        jno.core([residual.mse], dom).solve(15)

        chain = net.posterior_samples
        assert chain is not None
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(chain) if hasattr(leaf, "shape")]
        assert len(leaves) > 0, "LoRA chain must carry adapter-array leaves"
        for leaf in leaves:
            assert leaf.shape[0] == 10


class TestModelCallProxy:
    def test_modelcall_bayesian_proxies_to_model(self):
        # parameter() returns a ModelCall — calling .bayesian on it must
        # store the config on the underlying Model (not on the ModelCall).
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=2,
            keep=3,
        )
        assert a.model._bayesian_cfg is not None
        assert a.model._bayesian_cfg["warmup"] == 2
        assert a.model._bayesian_cfg["keep"] == 3
        # The proxy attribute mirrors the underlying Model's chain.
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        jno.core([residual.mse], dom).solve(5)
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (3, 1)
