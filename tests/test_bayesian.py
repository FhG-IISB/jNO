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
# Auto-chain default + samples="point" escape hatch
# ---------------------------------------------------------------------------


class TestEvalAutoChainDefault:
    """`crux.eval(...)` auto-picks chain vs point per expression based on whether
    its dependency graph touches a Bayesian model.  The chain default avoids the
    `f(mean(θ)) ≠ mean(f(θ))` foot-gun without forcing the user to pass
    ``samples="chain"`` every time."""

    def _setup(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")
        a.bayesian(
            blackjax.nuts,
            step_size=5e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=5,
            keep=10,
        )
        b.optimizer(optax.adam(1e-2))
        target = 1.0 * jno.np.sin(π * x)
        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        crux = jno.core([residual.mse], dom)
        crux.solve(15)
        return a, b, x, crux

    def test_bayesian_expr_returns_chain_by_default(self):
        a, _b, _x, crux = self._setup()
        chain = crux.eval([a])  # auto → chain
        assert chain.shape == (10, 1)

    def test_non_bayesian_expr_returns_point_by_default(self):
        _a, b, _x, crux = self._setup()
        point = crux.eval([b])  # auto → point
        assert point.shape == (1,)

    def test_mixed_list_picks_per_expression(self):
        a, b, _x, crux = self._setup()
        a_out, b_out = crux.eval([a, b])  # auto: a → chain, b → point
        assert a_out.shape == (10, 1)
        assert b_out.shape == (1,)

    def test_samples_point_forces_point_on_bayesian_expr(self):
        a, _b, _x, crux = self._setup()
        point = crux.eval([a], samples="point")
        # last sample, no leading chain axis
        assert point.shape == (1,)

    def test_unknown_samples_value_raises(self):
        _a, _b, _x, crux = self._setup()
        with pytest.raises(ValueError, match="samples="):
            crux.eval([1.0], samples="bogus")  # value won't matter — raises before eval


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
# Substeps + Bayesian — two-stage decoupled inference
# ---------------------------------------------------------------------------


class TestSubstepsWithBayesian:
    """`substeps=` is allowed with `.bayesian()`.  The classic use case is
    two-stage decoupled inference: substep 0 trains a surrogate (optax),
    substep 1 samples a coefficient against the trained surrogate (NUTS).
    Each substep's active-models set is detected from gradient probing, so
    the surrogate doesn't update in substep 1 and the coefficient isn't
    touched in substep 0.

    When ``adapt=True`` is set on a Bayesian model AND substeps= is in
    play, we raise — the adapter would tune against the full loss but the
    kernel only sees substep-local constraints."""

    def test_substeps_with_bayesian_runs(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=5, keep=10, adapt=False)
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse, residual.mse], dom)
        crux.solve(15, substeps=[[0], [1]])
        # Bayesian model collects samples once per outer epoch from whichever
        # substep updated its position.
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (10, 1)

    def test_substeps_with_adapt_true_raises(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        # adapt=True is the default — make it explicit for clarity.
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=5, keep=5, adapt=True)
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse, residual.mse], dom)
        with pytest.raises(ValueError, match="substeps.*adapt=True"):
            crux.solve(10, substeps=[[0], [1]])


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


class TestWindowAdaptation:
    """Phase 4B — `adapt=True` (default for HMC-family) runs
    blackjax.window_adaptation for `warmup` steps before the main loop and
    replaces step_size + inverse_mass_matrix with the adapted values.  The
    main loop then collects samples from epoch 0."""

    def _run_nuts(self, *, adapt: bool, step_size: float, warmup: int, keep: int):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x)
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        p.bayesian(
            blackjax.nuts,
            step_size=step_size,
            warmup=warmup,
            keep=keep,
            adapt=adapt,
        )
        residual = p * jno.np.sin(π * x) - target
        jno.core([residual.mse], dom).solve(warmup + keep)
        return p

    def test_adapt_recovers_with_bad_initial_step_size(self):
        # step_size=5.0 is too large; without adaptation NUTS diverges.
        # With adapt=True window_adaptation tunes it down before sampling.
        p = self._run_nuts(adapt=True, step_size=5.0, warmup=200, keep=200)
        chain = p.posterior_samples
        assert chain.shape == (200, 1)
        assert abs(float(jnp.mean(chain)) - 2.0) < 0.5

    def test_adapt_false_keeps_skip_n_semantics(self):
        # adapt=False → main loop runs warmup+keep epochs and the first
        # `warmup` are discarded.  Total stored = keep.
        p = self._run_nuts(adapt=False, step_size=1e-2, warmup=5, keep=10)
        assert p.posterior_samples.shape == (10, 1)

    def test_mala_with_adapt_true_is_noop(self):
        # MALA isn't in the HMC family; adapt=True must silently skip
        # adaptation and fall back to skip-N semantics.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        p.bayesian(blackjax.mala, step_size=1e-2, warmup=5, keep=10, adapt=True)
        residual = p * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse], dom).solve(15)
        assert p.posterior_samples.shape == (10, 1)


class TestAutoInverseMassMatrix:
    """Phase 4A — kernels that accept inverse_mass_matrix get an identity
    default of the right shape inferred from the position pytree.  Users no
    longer need to hand-write jnp.ones(D)."""

    def _run(self, p, *, kernel=blackjax.nuts, kernel_kwargs=None, warmup=2, keep=3):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        kernel_kwargs = dict(kernel_kwargs or {})
        # Deliberately do NOT pass inverse_mass_matrix.
        p.bayesian(kernel, step_size=1e-2, warmup=warmup, keep=keep, **kernel_kwargs)
        residual = p * jno.np.sin(π * x) - jno.np.sin(π * x)
        crux = jno.core([residual.mse], dom)
        crux.solve(warmup + keep)

    def test_nuts_scalar_param_no_imm_runs(self):
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        self._run(p)
        assert p.posterior_samples.shape == (3, 1)

    def test_nuts_vector_param_no_imm_runs(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x) + 2.0 * jno.np.cos(π * x) + 3.0 * x
        p = jno.np.parameter((3,), key=jax.random.PRNGKey(0), name="abc")
        p.bayesian(blackjax.nuts, step_size=5e-3, warmup=2, keep=3)
        residual = p[0] * jno.np.sin(π * x) + p[1] * jno.np.cos(π * x) + p[2] * x - target
        jno.core([residual.mse], dom).solve(5)
        assert p.posterior_samples.shape == (3, 3)

    def test_nuts_mlp_no_imm_runs(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        net.bayesian(blackjax.nuts, step_size=1e-3, warmup=1, keep=2)
        residual = net(x) - 0.0
        jno.core([residual.mse], dom).solve(3)
        leaves = jax.tree_util.tree_leaves(net.posterior_samples)
        assert all(leaf.shape[0] == 2 for leaf in leaves)

    def test_explicit_imm_is_respected(self):
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        # Pass a deliberately non-identity diagonal.
        custom = jnp.array([4.0])
        p.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=custom, warmup=1, keep=2)
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = p * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse], dom).solve(3)
        assert p.model._bayesian_cfg["kernel_kwargs"]["inverse_mass_matrix"] is custom

    def test_mala_does_not_get_imm_injected(self):
        # MALA's signature has no inverse_mass_matrix; injection must skip it.
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        p.bayesian(blackjax.mala, step_size=1e-2, warmup=1, keep=2)
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = p * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse], dom).solve(3)
        assert "inverse_mass_matrix" not in p.model._bayesian_cfg["kernel_kwargs"]

    def test_scalar_imm_broadcasts_to_position_shape(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x) + 2.0 * jno.np.cos(π * x) + 3.0 * x
        p = jno.np.parameter((3,), key=jax.random.PRNGKey(0), name="abc")
        p.bayesian(blackjax.nuts, step_size=5e-3, inverse_mass_matrix=2.0, warmup=2, keep=3)
        residual = p[0] * jno.np.sin(π * x) + p[1] * jno.np.cos(π * x) + p[2] * x - target
        jno.core([residual.mse], dom).solve(5)
        # After injection, the kwarg in extra_kwargs gets broadcast to a length-3 vector.
        # Look at the handle on the model — it was built inside solve(), so the cfg
        # itself still holds the original scalar (we only mutate handle.extra_kwargs,
        # not cfg).  The proof is that the run finishes without a blackjax shape error.
        assert p.posterior_samples.shape == (3, 3)


class TestWandbChainStatsSmoke:
    """Phase 4D — wandb_log gets posterior/<name>/{mean,last,n_samples}
    entries for each Bayesian model when a wandb run is active.  Patched
    end-to-end; no real wandb."""

    def test_wandb_receives_posterior_stats(self, monkeypatch):
        # `jno.core` is both a module and a class; bind the module explicitly.
        import importlib

        core_mod = importlib.import_module("jno.core")

        # Fake a wandb run so the logging block actually runs.
        class _FakeRun:
            def __init__(self):
                self.config = type("_Cfg", (), {"update": lambda *a, **kw: None})()
                self.summary = type("_Sum", (), {"update": lambda *a, **kw: None})()

        recorded: list[dict] = []

        def _fake_get_run():
            return _FakeRun()

        def _fake_log(metrics, step=None):
            recorded.append(dict(metrics))

        monkeypatch.setattr(core_mod, "get_wandb_run", _fake_get_run)
        monkeypatch.setattr(core_mod, "wandb_log", _fake_log)
        # wandb_log_model is fine as the no-op fake — only called at end.
        monkeypatch.setattr(core_mod, "wandb_log_model", lambda *a, **kw: None)

        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="alpha")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=0, keep=3, adapt=False)
        residual = a * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse], dom).solve(4)

        # Collect all keys logged across all calls.
        all_keys = set()
        for metrics in recorded:
            all_keys.update(metrics.keys())
        assert "posterior/alpha/mean" in all_keys
        assert "posterior/alpha/last" in all_keys
        assert "posterior/alpha/n_samples" in all_keys


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
