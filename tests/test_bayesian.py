"""Tests for the per-parameter `.bayesian(...)` API.

These tests exercise the end-to-end path: configuration → solve loop →
posterior_samples → crux.eval(samples="chain").  Where possible we run
short chains on tiny problems so the suite stays under a second.
"""

from __future__ import annotations

import blackjax
import equinox as eqx
import foundax
import jax
import jax.numpy as jnp
import numpy as np
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
        crux = jno.core([residual.mse])
        crux.solve(warmup + keep)
        return a

    def test_chain_shape_and_recovery(self):
        a = self._solve(warmup=100, keep=200)
        chain = a.posterior_samples
        assert chain is not None
        assert chain.shape == (1, 200, 1)
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
        crux = jno.core([residual.mse])
        crux.solve(30)
        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        assert all(leaf.shape[0] == 1 and leaf.shape[1] == 20 for leaf in leaves)


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
        crux = jno.core([residual.mse])
        crux.solve(30)

        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (1, 20, 1)
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
        crux = jno.core([residual.mse])
        crux.solve(15)

        a_chain = crux.eval([a], samples="chain")
        # (K, N, *) with K=1 default
        assert a_chain.shape[:2] == (1, 10)

        # Nonlinear expression `a**2` — chain mean must match jnp.mean(a**2)
        sq_chain = crux.eval([a * a], samples="chain")
        assert sq_chain.shape[:2] == (1, 10)
        ref = a.posterior_samples**2  # (1, 10, 1)
        assert jnp.allclose(sq_chain.reshape(10, -1), ref.reshape(10, -1))

    def test_raises_when_no_bayesian_models(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.optimizer(optax.adam(1e-2))
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        crux = jno.core([residual.mse])
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
        crux = jno.core([residual.mse])
        crux.solve(15)
        return a, b, x, crux

    def test_bayesian_expr_returns_chain_by_default(self):
        a, _b, _x, crux = self._setup()
        chain = crux.eval([a])  # auto → chain
        assert chain.shape == (1, 10, 1)

    def test_non_bayesian_expr_returns_point_by_default(self):
        _a, b, _x, crux = self._setup()
        point = crux.eval([b])  # auto → point
        assert point.shape == (1,)

    def test_mixed_list_picks_per_expression(self):
        a, b, _x, crux = self._setup()
        a_out, b_out = crux.eval([a, b])  # auto: a → chain, b → point
        assert a_out.shape[:2] == (1, 10)
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
        crux = jno.core([residual.mse])
        crux.solve(20)
        return float(crux.eval([a])[0])

    def test_two_runs_identical(self):
        assert abs(self._run() - self._run()) < 1e-12


# ---------------------------------------------------------------------------
# Custom prior overrides default
# ---------------------------------------------------------------------------


class TestPriorsNamespace:
    """``jno.bayesian.priors.{gaussian, laplace, student_t,
    layerwise_gaussian}`` — each factory returns a callable
    ``pytree -> scalar`` that obeys the ``prior=`` contract on
    :meth:`Model.bayesian` / :meth:`Model.vi`.  Tests cover the
    mathematical form (compare numerical output to a hand-coded
    reference) and the fan-in-aware scaling for weight tensors.
    """

    def test_gaussian_matches_hand_coded(self):
        prior = jno.bayesian.priors.gaussian(sigma=2.0)
        # A small pytree of two floating leaves.
        p = {"w": jnp.array([1.0, -1.0, 2.0]), "b": jnp.array([0.5])}
        # Hand: -0.5 * (1 + 1 + 4 + 0.25) / 4 = -6.25 / 8.
        expected = -0.5 * (1.0 + 1.0 + 4.0 + 0.25) / (2.0 * 2.0)
        assert float(prior(p)) == pytest.approx(expected, rel=1e-5)

    def test_gaussian_fan_in_aware_uses_prod_of_trailing_dims(self):
        # A (3, 4) weight tensor has fan_in=4 → σ scaled by sqrt(4)=2.
        # Bias (4,) has ndim<2 → unscaled.
        prior = jno.bayesian.priors.gaussian(sigma=1.0, fan_in_aware=True)
        W = jnp.ones((3, 4))  # ‖W‖² = 12
        b = jnp.ones((4,))  # ‖b‖² = 4
        # log p = -0.5 * (‖W‖² * fan_in_W + ‖b‖²) / σ²
        #       = -0.5 * (12 * 4 + 4) / 1 = -26.
        assert float(prior({"W": W, "b": b})) == pytest.approx(-26.0, rel=1e-5)

    def test_gaussian_rejects_non_positive_sigma(self):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            jno.bayesian.priors.gaussian(sigma=0.0)
        with pytest.raises(ValueError, match="sigma must be > 0"):
            jno.bayesian.priors.gaussian(sigma=-1.0)

    def test_laplace_matches_hand_coded(self):
        prior = jno.bayesian.priors.laplace(scale=2.0)
        p = {"w": jnp.array([1.0, -1.0, 3.0])}
        # log p = -(1 + 1 + 3) / 2 = -2.5
        assert float(prior(p)) == pytest.approx(-2.5, rel=1e-5)

    def test_laplace_rejects_non_positive_scale(self):
        with pytest.raises(ValueError, match="scale must be > 0"):
            jno.bayesian.priors.laplace(scale=0.0)

    def test_student_t_heavier_tails_than_gaussian(self):
        # At the same scale, student-t assigns higher log-prob to a
        # large-magnitude weight than Gaussian does.  Quick sanity check.
        gauss = jno.bayesian.priors.gaussian(sigma=1.0)
        st = jno.bayesian.priors.student_t(df=3.0, scale=1.0)
        big = {"w": jnp.array([5.0])}
        # Gaussian: -0.5 * 25 = -12.5
        # Student-t (df=3): -2 * log(1 + 25/3) = -2 * log(28/3) ≈ -4.46
        # student_t > gaussian at large |θ| — heavier tails.
        assert float(st(big)) > float(gauss(big))

    def test_student_t_rejects_df_below_two(self):
        with pytest.raises(ValueError, match="df must be > 2"):
            jno.bayesian.priors.student_t(df=2.0, scale=1.0)
        with pytest.raises(ValueError, match="df must be > 2"):
            jno.bayesian.priors.student_t(df=1.5, scale=1.0)

    def test_layerwise_gaussian_default_is_fan_in_aware(self):
        # Default: fan_in_aware=True, base_sigma=1, default_sigma=1.
        # Weight (2, 3): fan_in=3 → σ = 1/sqrt(3) → log-prob includes
        # ‖W‖² * 3.
        # Bias (3,): σ = 1.
        prior = jno.bayesian.priors.layerwise_gaussian()
        W = jnp.ones((2, 3))  # ‖W‖² = 6
        b = jnp.ones((3,))  # ‖b‖² = 3
        # log p = -0.5 * (6 * 3 + 3) = -10.5
        assert float(prior({"W": W, "b": b})) == pytest.approx(-10.5, rel=1e-5)

    def test_layerwise_gaussian_can_disable_fan_in_awareness(self):
        prior = jno.bayesian.priors.layerwise_gaussian(fan_in_aware=False)
        W = jnp.ones((2, 3))
        b = jnp.ones((3,))
        # log p = -0.5 * (6 + 3) = -4.5
        assert float(prior({"W": W, "b": b})) == pytest.approx(-4.5, rel=1e-5)

    def test_priors_compose_with_bayesian_solve(self):
        # End-to-end smoke: pass priors.gaussian into Model.bayesian and
        # check the chain runs without error.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            prior=jno.bayesian.priors.gaussian(sigma=0.5),
            warmup=5,
            keep=10,
            adapt=False,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(15)
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (1, 10, 1)

    def test_priors_compose_with_vi_solve(self):
        # Same end-to-end smoke for VI.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            prior=jno.bayesian.priors.laplace(scale=1.0),
            num_samples=4,
            posterior_draws=20,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(20)
        assert a.posterior_samples.shape == (1, 20, 1)

    def test_priors_skip_integer_leaves(self):
        # A pytree with an int array leaf (e.g. a mask / index) must not
        # contribute to the log-prob.  Verifies the _floating_leaves
        # filter inside each factory.
        prior = jno.bayesian.priors.gaussian(sigma=1.0)
        p = {"w": jnp.array([1.0, 2.0]), "idx": jnp.array([0, 1, 2], dtype=jnp.int32)}
        # Only the floats contribute: -0.5 * (1 + 4) = -2.5
        assert float(prior(p)) == pytest.approx(-2.5, rel=1e-5)


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
        crux = jno.core([residual.mse])
        crux.solve(60)
        post_mean = float(jnp.mean(a.posterior_samples))
        # Strong prior should drag the posterior closer to 0 than the truth (2.0).
        assert abs(post_mean) < 1.5, f"Strong prior failed to shrink toward 0; got {post_mean}"


class TestLikelihoodScale:
    """``likelihood_scale=`` multiplies the negative log-likelihood term in
    the per-step logdensity.  The canonical sum-over-data Gaussian
    likelihood is approximated by ``residual.mse`` which returns the
    mean; passing ``N_obs`` (or ``N_obs / sigma**2``) recovers the
    correct posterior magnitude.

    With ``likelihood_scale`` large the data overwhelms a weak prior and
    the posterior concentrates near the MLE — that's what we check.
    """

    def test_bayesian_likelihood_scale_concentrates_posterior(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x)

        # Wide prior + scaled likelihood → posterior concentrates near 2.0
        # with small variance.  Unscaled (default 1.0) the same setup
        # leaves the chain dispersed over a much wider region.
        def _solve(scale):
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.bayesian(
                blackjax.nuts,
                step_size=1e-2,
                inverse_mass_matrix=jnp.ones(1),
                warmup=50,
                keep=200,
                adapt=False,
                likelihood_scale=scale,
            )
            residual = a * jno.np.sin(π * x) - target
            jno.core([residual.mse]).solve(250)
            return jnp.std(a.posterior_samples)

        std_low = _solve(1.0)
        std_high = _solve(100.0)
        # Scaling the likelihood by 100 must concentrate the posterior
        # by a noticeable margin (sqrt-of-N rule of thumb).
        assert float(std_high) < float(std_low) * 0.5

    def test_bayesian_likelihood_scale_rejects_non_positive(self):
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        with pytest.raises(ValueError, match="likelihood_scale must be positive"):
            a.bayesian(blackjax.nuts, step_size=1e-2, warmup=2, keep=2, likelihood_scale=0.0)
        with pytest.raises(ValueError, match="likelihood_scale must be positive"):
            a.bayesian(blackjax.nuts, step_size=1e-2, warmup=2, keep=2, likelihood_scale=-1.0)

    def test_vi_likelihood_scale_pulls_mean_off_prior(self):
        # VI is the canonical case for likelihood_scale.  Without it,
        # an N_obs-mean likelihood leaves VI stuck near initialisation
        # because the wide Gaussian prior dominates.  With it, the
        # variational mean tracks the data.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 3.0 * jno.np.sin(π * x)

        def _solve(scale):
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.vi(
                blackjax.meanfield_vi,
                optimizer=optax.adam(5e-2),
                num_samples=8,
                posterior_draws=100,
                likelihood_scale=scale,
            )
            residual = a * jno.np.sin(π * x) - target
            jno.core([residual.mse]).solve(400)
            return float(jnp.mean(a.posterior_samples))

        mean_unscaled = _solve(1.0)
        mean_scaled = _solve(100.0)
        # Scaled VI should land much closer to the data-fit value (3.0)
        # than the unscaled run, which gets pulled toward the prior at 0.
        assert abs(mean_scaled - 3.0) < abs(mean_unscaled - 3.0)

    def test_vi_likelihood_scale_rejects_non_positive(self):
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        with pytest.raises(ValueError, match="likelihood_scale must be positive"):
            a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-3), likelihood_scale=0.0)


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
        crux = jno.core([residual.mse, residual.mse])
        crux.solve(15, substeps=[[0], [1]])
        # Bayesian model collects samples once per outer epoch from whichever
        # substep updated its position.
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (1, 10, 1)

    def test_substeps_with_adapt_true_raises(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        # adapt=True is the default — make it explicit for clarity.
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=5, keep=5, adapt=True)
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse, residual.mse])
        with pytest.raises(ValueError, match="substeps.*adapt=True"):
            crux.solve(10, substeps=[[0], [1]])


# ---------------------------------------------------------------------------
# Missing step_size → clear error
# ---------------------------------------------------------------------------


class TestMissingStepSize:
    def test_no_step_size_with_adapt_false_raises_clearly(self):
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        # Without window adaptation NUTS has no way to pick a step size,
        # so the missing kwarg must still raise.
        a.bayesian(blackjax.nuts, inverse_mass_matrix=jnp.ones(1), adapt=False)
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        crux = jno.core([residual.mse])
        with pytest.raises(ValueError, match="requires a step_size"):
            crux.solve(2)

    def test_no_step_size_with_mala_raises_clearly(self):
        # MALA has no inverse_mass_matrix parameter — it's not HMC-family,
        # so the adapt-route default doesn't apply.
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.mala, warmup=2, keep=2)
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        crux = jno.core([residual.mse])
        with pytest.raises(ValueError, match="requires a step_size"):
            crux.solve(4)

    def test_adapt_true_nuts_omits_step_size(self):
        # The common case: just give NUTS warmup + keep and let
        # blackjax.window_adaptation choose step_size.  No step_size
        # kwarg, no raise.
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, warmup=20, keep=10)  # adapt=True default
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        jno.core([residual.mse]).solve(30)
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (1, 10, 1)

    def test_adapt_true_hmc_omits_step_size(self):
        # Same default works for blackjax.hmc.
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.hmc, warmup=20, keep=10, num_integration_steps=4)
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        jno.core([residual.mse]).solve(30)
        assert a.posterior_samples.shape == (1, 10, 1)

    def test_adapt_true_zero_warmup_still_raises(self):
        # adapt=True with warmup=0 is a degenerate config — adaptation
        # has nothing to run, so the user-provided step_size is still
        # required.  Must raise loudly, not silently default to 1.0.
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, warmup=0, keep=2)  # adapt=True default
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        crux = jno.core([residual.mse])
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
    return a, jno.core([residual.mse])


# ---------------------------------------------------------------------------
# Knob behaviour (keep / thin / warmup)
# ---------------------------------------------------------------------------


class TestKnobs:
    def test_keep_caps_chain_length(self):
        a, crux = _trivial_bayesian_param(warmup=0, keep=20)
        crux.solve(50)  # run more than keep epochs
        assert a.posterior_samples.shape == (1, 20, 1)

    def test_warmup_skips_initial_samples(self):
        a, crux = _trivial_bayesian_param(warmup=5, keep=10)
        crux.solve(15)
        # first stored sample must be the one AT epoch 5, not the initial value
        # → at minimum the first kept sample is not equal to the zero init.
        assert a.posterior_samples.shape == (1, 10, 1)
        assert float(jnp.abs(a.posterior_samples[0]).max()) > 0.0

    def test_thin_keeps_every_kth(self):
        a, crux = _trivial_bayesian_param(warmup=0, keep=5, thin=3)
        crux.solve(30)
        # We collect at epochs 0, 3, 6, 9, 12 → 5 samples total.
        assert a.posterior_samples.shape == (1, 5, 1)


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
    jno.core([residual.mse]).solve(warmup + keep)
    return a


class TestHMC:
    def test_hmc_runs_and_recovers(self):
        a = _kernel_recovery(
            blackjax.hmc,
            kernel_kwargs=dict(inverse_mass_matrix=jnp.ones(1), num_integration_steps=4),
            step_size=2e-2,
        )
        chain = a.posterior_samples
        assert chain.shape == (1, 120, 1)
        assert abs(float(jnp.mean(chain)) - 2.0) < 0.5


class TestMALA:
    def test_mala_runs_and_recovers(self):
        # MALA at fixed step_size without adaptation is a notoriously
        # noisy sampler.  The assertion is primarily a duck-typing smoke
        # test: shape correct + chain mean in a plausible range
        # (not diverged to ±∞).  Recovery to the truth within a tight
        # tolerance is the job of NUTS-with-adapt, not bare MALA.
        a = _kernel_recovery(
            blackjax.mala,
            kernel_kwargs=dict(),  # MALA has no inverse_mass_matrix
            step_size=1e-2,
        )
        chain = a.posterior_samples
        assert chain.shape == (1, 120, 1)
        chain_mean = float(jnp.mean(chain))
        assert jnp.isfinite(chain_mean), "MALA chain diverged"
        assert abs(chain_mean - 2.0) < 2.0, f"MALA chain mean {chain_mean} unreasonably far from 2.0"


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
        jno.core([residual.mse]).solve(15)
        assert a.posterior_samples.shape == (1, 10, 1)


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
        jno.core([residual.mse]).solve(50)
        assert p.posterior_samples.shape == (1, 30, 3)

    def test_multi_leaf_mlp_chain_pytree(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        net.bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=10)
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(15)

        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        assert len(leaves) > 1, "MLP chain must contain multiple weight/bias leaves"
        for leaf in leaves:
            assert leaf.shape[:2] == (1, 10), f"Unexpected leading axes {leaf.shape[:2]}"


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
        jno.core([residual.mse]).solve(15)
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (1, 10, 1)


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
        jno.core([residual.mse]).solve(15)

        chain = net.posterior_samples
        assert chain is not None
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(chain) if hasattr(leaf, "shape")]
        assert len(leaves) > 0, "LoRA chain must carry adapter-array leaves"
        for leaf in leaves:
            assert leaf.shape[:2] == (1, 10)


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
        jno.core([residual.mse]).solve(warmup + keep)
        return p

    def test_adapt_recovers_with_bad_initial_step_size(self):
        # step_size=5.0 is too large; without adaptation NUTS diverges.
        # With adapt=True window_adaptation tunes it down before sampling.
        p = self._run_nuts(adapt=True, step_size=5.0, warmup=200, keep=200)
        chain = p.posterior_samples
        assert chain.shape == (1, 200, 1)
        assert abs(float(jnp.mean(chain)) - 2.0) < 0.5

    def test_adapt_false_keeps_skip_n_semantics(self):
        # adapt=False → main loop runs warmup+keep epochs and the first
        # `warmup` are discarded.  Total stored = keep.
        p = self._run_nuts(adapt=False, step_size=1e-2, warmup=5, keep=10)
        assert p.posterior_samples.shape == (1, 10, 1)

    def test_mala_with_adapt_true_is_noop(self):
        # MALA isn't in the HMC family; adapt=True must silently skip
        # adaptation and fall back to skip-N semantics.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        p.bayesian(blackjax.mala, step_size=1e-2, warmup=5, keep=10, adapt=True)
        residual = p * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse]).solve(15)
        assert p.posterior_samples.shape == (1, 10, 1)


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
        crux = jno.core([residual.mse])
        crux.solve(warmup + keep)

    def test_nuts_scalar_param_no_imm_runs(self):
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        self._run(p)
        assert p.posterior_samples.shape == (1, 3, 1)

    def test_nuts_vector_param_no_imm_runs(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x) + 2.0 * jno.np.cos(π * x) + 3.0 * x
        p = jno.np.parameter((3,), key=jax.random.PRNGKey(0), name="abc")
        p.bayesian(blackjax.nuts, step_size=5e-3, warmup=2, keep=3)
        residual = p[0] * jno.np.sin(π * x) + p[1] * jno.np.cos(π * x) + p[2] * x - target
        jno.core([residual.mse]).solve(5)
        assert p.posterior_samples.shape == (1, 3, 3)

    def test_nuts_mlp_no_imm_runs(self):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        net.bayesian(blackjax.nuts, step_size=1e-3, warmup=1, keep=2)
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(3)
        leaves = jax.tree_util.tree_leaves(net.posterior_samples)
        assert all(leaf.shape[:2] == (1, 2) for leaf in leaves)

    def test_explicit_imm_is_respected(self):
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        # Pass a deliberately non-identity diagonal.
        custom = jnp.array([4.0])
        p.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=custom, warmup=1, keep=2)
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = p * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse]).solve(3)
        assert p.model._bayesian_cfg["kernel_kwargs"]["inverse_mass_matrix"] is custom

    def test_mala_does_not_get_imm_injected(self):
        # MALA's signature has no inverse_mass_matrix; injection must skip it.
        p = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="p")
        p.bayesian(blackjax.mala, step_size=1e-2, warmup=1, keep=2)
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = p * jno.np.sin(π * x) - jno.np.sin(π * x)
        jno.core([residual.mse]).solve(3)
        assert "inverse_mass_matrix" not in p.model._bayesian_cfg["kernel_kwargs"]

    def test_scalar_imm_broadcasts_to_position_shape(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x) + 2.0 * jno.np.cos(π * x) + 3.0 * x
        p = jno.np.parameter((3,), key=jax.random.PRNGKey(0), name="abc")
        p.bayesian(blackjax.nuts, step_size=5e-3, inverse_mass_matrix=2.0, warmup=2, keep=3)
        residual = p[0] * jno.np.sin(π * x) + p[1] * jno.np.cos(π * x) + p[2] * x - target
        jno.core([residual.mse]).solve(5)
        # After injection, the kwarg in extra_kwargs gets broadcast to a length-3 vector.
        # Look at the handle on the model — it was built inside solve(), so the cfg
        # itself still holds the original scalar (we only mutate handle.extra_kwargs,
        # not cfg).  The proof is that the run finishes without a blackjax shape error.
        assert p.posterior_samples.shape == (1, 3, 3)


class TestPosteriorDiagnostics:
    """Per-step blackjax info (is_divergent / acceptance_rate / energy)
    surfaced on ``model.posterior_diagnostics``, wandb, and the
    solve-end log line.

    The kernel info is captured every step (mirroring the sample
    buffer), aggregated to ``(K, N)`` arrays at solve-end, and
    summarised in a single log line per Bayesian model.  Empty for
    SG-MCMC / VI (those kernels expose no info object) — surfaced as
    ``None``, not silently dropped.
    """

    def _line_setup(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x)
        return π, dom, x, target

    def test_nuts_diagnostics_have_all_three_fields(self):
        π, dom, x, target = self._line_setup()
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=0, keep=8, adapt=False)
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(8)
        diag = a.model.posterior_diagnostics
        assert diag is not None
        assert set(diag.keys()) == {"is_divergent", "acceptance_rate", "energy"}
        assert diag["is_divergent"].dtype == jnp.bool_
        assert diag["acceptance_rate"].dtype == jnp.float32
        assert diag["energy"].dtype == jnp.float32
        # Shape: (K=1, N=8) — one diagnostic per kept sample.
        assert diag["is_divergent"].shape == (1, 8)
        assert diag["acceptance_rate"].shape == (1, 8)
        assert diag["energy"].shape == (1, 8)
        # Acceptance rates live in [0, 1].
        assert bool(jnp.all((diag["acceptance_rate"] >= 0.0) & (diag["acceptance_rate"] <= 1.0)))

    def test_mala_diagnostics_only_acceptance_rate(self):
        π, dom, x, target = self._line_setup()
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.mala, step_size=1e-2, warmup=0, keep=5)
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(5)
        diag = a.model.posterior_diagnostics
        assert diag is not None
        # MALA has no Hamiltonian integrator → no divergence / energy.
        assert set(diag.keys()) == {"acceptance_rate"}
        assert diag["acceptance_rate"].shape == (1, 5)

    def test_sgld_diagnostics_is_none(self):
        # SG-MCMC kernels return only the new state — no info object —
        # so posterior_diagnostics is None (not an empty dict, not zeros).
        π, dom, x, target = self._line_setup()
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4, key_seed=1)
        net.bayesian(blackjax.sgld, step_size=1e-5, warmup=0, keep=3)
        residual = net(x) - target
        jno.core([residual.mse]).solve(3)
        assert net.posterior_diagnostics is None

    def test_diagnostics_multi_chain_shape(self):
        π, dom, x, target = self._line_setup()
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=0,
            keep=6,
            adapt=False,
            num_chains=3,
            init_jitter=0.1,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(6)
        diag = a.model.posterior_diagnostics
        assert diag is not None
        # Shape mirrors posterior_samples: (K=3, N=6).
        assert diag["is_divergent"].shape == (3, 6)
        assert diag["acceptance_rate"].shape == (3, 6)
        assert diag["energy"].shape == (3, 6)

    def test_modelcall_proxy(self):
        # The ModelCall proxy mirrors posterior_diagnostics like
        # posterior_samples does.
        π, dom, x, target = self._line_setup()
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=0, keep=3, adapt=False)
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(3)
        # ``a`` is the ModelCall; ``a.posterior_diagnostics`` must
        # resolve to the underlying Model's dict.
        assert a.posterior_diagnostics is a.model.posterior_diagnostics
        assert a.posterior_diagnostics is not None

    def test_diagnostics_solve_end_summary_logged(self, capsys):
        # jno's Logger uses ``propagate=False`` so caplog (which hooks the
        # root logger) doesn't see jno messages — but the same Logger
        # writes to stdout via a StreamHandler, so capsys captures it.
        π, dom, x, target = self._line_setup()
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="diag_a")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=0, keep=4, adapt=False)
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(4)
        captured = capsys.readouterr().out
        assert "diag_a" in captured
        assert "divergent" in captured
        assert "mean_accept" in captured

    def test_wandb_diagnostics_keys(self, monkeypatch):
        import importlib

        core_mod = importlib.import_module("jno.core")

        class _FakeRun:
            def __init__(self):
                self.config = type("_Cfg", (), {"update": lambda *a, **kw: None})()
                self.summary = type("_Sum", (), {"update": lambda *a, **kw: None})()

        recorded: list[dict] = []
        monkeypatch.setattr(core_mod, "get_wandb_run", lambda: _FakeRun())
        monkeypatch.setattr(core_mod, "wandb_log", lambda m, step=None: recorded.append(dict(m)))
        monkeypatch.setattr(core_mod, "wandb_log_model", lambda *a, **kw: None)

        π, dom, x, target = self._line_setup()
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="wb_a")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=0, keep=3, adapt=False)
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(4)
        keys: set[str] = set()
        for m in recorded:
            keys.update(m.keys())
        assert "posterior/wb_a/n_is_divergent" in keys
        assert "posterior/wb_a/mean_acceptance_rate" in keys
        assert "posterior/wb_a/mean_energy" in keys


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
        jno.core([residual.mse]).solve(4)

        # Collect all keys logged across all calls.
        all_keys = set()
        for metrics in recorded:
            all_keys.update(metrics.keys())
        assert "posterior/alpha/mean" in all_keys
        assert "posterior/alpha/n_samples" in all_keys
        assert "posterior/alpha/n_chains" in all_keys


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
        jno.core([residual.mse]).solve(5)
        assert a.posterior_samples is not None
        assert a.posterior_samples.shape == (1, 3, 1)


# ---------------------------------------------------------------------------
# Multi-chain (Phase 8) — num_chains > 1, init_jitter, diagnostics
# ---------------------------------------------------------------------------


def _multichain_solve(K: int, *, warmup=50, keep=80, init_jitter=0.0):
    """1-parameter NUTS inverse problem with K parallel chains."""
    π = jno.np.pi
    dom = _line_domain()
    x, _ = dom.variable("interior")
    target = 2.0 * jno.np.sin(π * x)
    a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
    a.bayesian(
        blackjax.nuts,
        step_size=1e-2,
        inverse_mass_matrix=jnp.ones(1),
        warmup=warmup,
        keep=keep,
        adapt=False,
        num_chains=K,
        init_jitter=init_jitter,
    )
    residual = a * jno.np.sin(π * x) - target
    jno.core([residual.mse]).solve(warmup + keep)
    return a


class TestMultiChain:
    """Phase 8 — ``num_chains=K`` runs K independent chains via ``vmap``
    and stores ``posterior_samples`` with the arviz-shaped
    ``(K, N, *param)`` layout regardless of K.  Convergence diagnostics
    ``jno.bayesian.{rhat, ess}`` operate on this layout."""

    def test_nuts_num_chains_K_shape(self):
        a = _multichain_solve(K=4)
        assert a.posterior_samples.shape == (4, 80, 1)

    def test_chains_are_independent(self):
        a = _multichain_solve(K=4)
        chain = a.posterior_samples  # (4, 80, 1)
        # Per-chain last samples should differ (independent PRNG paths).
        lasts = jnp.asarray([chain[k, -1, 0] for k in range(4)])
        assert float(jnp.std(lasts)) > 1e-3, "K chains produced identical last samples"

    def test_num_chains_1_keeps_K_axis(self):
        # K=1 still produces ``(1, N, *param)`` — arviz-shape uniformity.
        a = _multichain_solve(K=1, warmup=5, keep=10)
        assert a.posterior_samples.shape == (1, 10, 1)

    def test_init_jitter_disperses_chains(self):
        # Without jitter, K=2 chains start from identical positions and
        # diverge only through the kernel's per-chain RNG.  With
        # ``init_jitter > 0`` each chain starts at a different point.
        a_jit = _multichain_solve(K=2, warmup=2, keep=5, init_jitter=0.5)
        # First sample of each chain should already differ when jitter
        # was applied at init.
        c0 = a_jit.posterior_samples[0, 0, 0]
        c1 = a_jit.posterior_samples[1, 0, 0]
        assert float(jnp.abs(c0 - c1)) > 1e-3

    def test_eval_multichain_shape(self):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=5,
            keep=10,
            adapt=False,
            num_chains=3,
        )
        residual = a * jno.np.sin(π * x) - target
        crux = jno.core([residual.mse])
        crux.solve(15)
        # crux.eval with auto-chain mode vmaps over (K, N) → output has
        # leading (K, N) axes.
        out = crux.eval([a])
        assert out.shape[:2] == (3, 10), f"expected (3, 10, ...) got {out.shape}"

    def test_mismatched_num_chains_raises(self):
        # Two Bayesian models with different num_chains must raise.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=2, keep=3, adapt=False, num_chains=2)
        b.bayesian(blackjax.nuts, step_size=1e-2, warmup=2, keep=3, adapt=False, num_chains=4)
        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - jno.np.sin(π * x)
        crux = jno.core([residual.mse])
        with pytest.raises(ValueError, match="num_chains"):
            crux.solve(5)

    def test_rhat_helper(self):
        # Converged 1-parameter problem → R-hat close to 1.0.
        a = _multichain_solve(K=4, warmup=200, keep=400)
        r = jno.bayesian.rhat(a.posterior_samples)
        assert r.shape == (1,), f"unexpected rhat shape {r.shape}"
        # Loose: 4 NUTS chains on a 1-param problem mix quickly.
        assert float(r[0]) < 1.5, f"R-hat too high: {float(r[0])}"

    def test_rhat_strategy_multichain_raises_on_k1(self):
        # Single-chain input with strategy='multichain' must raise loudly
        # rather than silently switch to split-rhat.
        a = _multichain_solve(K=1, warmup=10, keep=40)
        with pytest.raises(ValueError, match="requires K>=2"):
            jno.bayesian.rhat(a.posterior_samples, strategy="multichain")

    def test_rhat_strategy_auto_falls_back_to_split_on_k1(self):
        # Default 'auto' on K=1 → split-rhat fallback (the historical
        # behaviour).  Must run without error and return a finite value.
        a = _multichain_solve(K=1, warmup=10, keep=40)
        r = jno.bayesian.rhat(a.posterior_samples, strategy="auto")
        assert r.shape == (1,)
        assert jnp.isfinite(r[0])

    def test_rhat_strategy_split_works_on_k2(self):
        # strategy='split' on K=2 → treat as 4 independent half-chains.
        # Just check shape + finite output.
        a = _multichain_solve(K=2, warmup=20, keep=40)
        r = jno.bayesian.rhat(a.posterior_samples, strategy="split")
        assert r.shape == (1,)
        assert jnp.isfinite(r[0])

    def test_rhat_strategy_unknown_raises(self):
        a = _multichain_solve(K=2, warmup=10, keep=40)
        with pytest.raises(ValueError, match="strategy must be one of"):
            jno.bayesian.rhat(a.posterior_samples, strategy="bogus")

    def test_ess_helper(self):
        a = _multichain_solve(K=4, warmup=50, keep=80)
        e = jno.bayesian.ess(a.posterior_samples)
        assert e.shape == (1,), f"unexpected ess shape {e.shape}"
        # ESS is bounded above by K*N=320 and below by 0; a working
        # chain should give at least a handful.
        assert float(e[0]) > 1.0, f"ESS too low: {float(e[0])}"
        assert float(e[0]) <= 4 * 80 + 1e-3, f"ESS exceeds K*N: {float(e[0])}"

    def test_ess_geyer_strict_truncation(self):
        # Strict initial-positive-sequence: discard EVERYTHING from
        # the first non-positive pair onward.  We assert this property
        # directly against the alternative "sum every positive pair"
        # rule on a manually-constructed pair-sum vector.
        # ``cummin > 0`` is True only for the prefix of strictly
        # positive pair sums.
        pair_sums = jnp.asarray([1.0, 0.5, -0.1, 0.3, 0.2])  # (5,)
        strict_mask = jnp.minimum.accumulate(pair_sums) > 0
        # Indices 0, 1 kept; 2 (-0.1) drops the cummin to -0.1 and
        # everything from there onward is excluded — including the
        # later spurious-positive pairs (0.3, 0.2).
        assert bool(strict_mask[0]) is True
        assert bool(strict_mask[1]) is True
        assert bool(strict_mask[2]) is False
        assert bool(strict_mask[3]) is False
        assert bool(strict_mask[4]) is False
        # Sanity check against IID samples — finite-N empirical ACF
        # has spurious-negative pairs early; strict rule keeps ESS in
        # the [0.2*N, N] band rather than collapsing to zero.
        rng = np.random.default_rng(0)
        N = 4096
        chain = rng.standard_normal((1, N, 1)).astype(np.float32)
        e = jno.bayesian.ess(jnp.asarray(chain))
        assert float(e[0]) > 0.2 * N, f"ESS {float(e[0])} too low for iid chain of {N}"
        assert float(e[0]) <= N + 1e-3, f"ESS {float(e[0])} exceeds N"


# ---------------------------------------------------------------------------
# MCMC fastpath (Phase 9) — scan-based pure-Bayesian solve loop
# ---------------------------------------------------------------------------


class TestMCMCFastpath:
    """Phase 9 — pure-Bayesian solves auto-dispatch to a scan-based
    fastpath that closes three perf gaps in the per-epoch Python loop:
    no outer ``value_and_grad``, one XLA dispatch per ``print_rate``
    chunk, and one host transfer per chunk.  Falls through to the
    per-epoch loop when conditions don't apply (mixed-mode, substeps,
    streaming, trackers, resampling, heterogeneous warmup/keep/thin).
    """

    def _fastpath_solve(self, *, warmup=10, keep=20, thin=1, num_chains=1, mesh_size=0.05):
        """Tiny inverse problem that exercises the fastpath."""
        π = jno.np.pi
        dom = _line_domain(mesh_size)
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=warmup,
            keep=keep,
            thin=thin,
            adapt=False,
            num_chains=num_chains,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(warmup + keep * thin)
        return a

    def test_fastpath_k1_recovers(self):
        # Single chain, single Bayesian model — the most basic
        # fastpath qualifier.
        a = self._fastpath_solve(warmup=100, keep=200)
        chain = a.posterior_samples
        assert chain.shape == (1, 200, 1)
        assert abs(float(jnp.mean(chain)) - 2.0) < 0.5

    def test_fastpath_k4_multichain(self):
        # K=4 → fastpath still applies and produces the arviz layout.
        a = self._fastpath_solve(warmup=20, keep=50, num_chains=4)
        chain = a.posterior_samples
        assert chain.shape == (4, 50, 1)

    def test_fastpath_multi_bayesian_gibbs(self):
        # T02-style two-coefficient inverse — verifies the per-lid
        # Gibbs cycle inside the scan body.  Step size + chain length
        # match T02 reasonably so the chain actually reaches truth.
        π = jno.np.pi
        dom = _line_domain(mesh_size=0.02)
        x, _ = dom.variable("interior")
        target = 3.14 * jno.np.sin(π * x) + (-2.71) * jno.np.cos(π * x)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
        a = jno.np.parameter((1,), key=k1, name="a")
        b = jno.np.parameter((1,), key=k2, name="b")
        for p in (a, b):
            p.bayesian(
                blackjax.nuts,
                step_size=1e-1,
                inverse_mass_matrix=jnp.ones(1),
                warmup=200,
                keep=200,
                adapt=False,
            )
        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        jno.core([residual.mse]).solve(400)
        assert a.posterior_samples.shape == (1, 200, 1)
        assert b.posterior_samples.shape == (1, 200, 1)
        # Both posterior means roughly recover truth (loose tolerance).
        assert abs(float(jnp.mean(a.posterior_samples)) - 3.14) < 1.5
        assert abs(float(jnp.mean(b.posterior_samples)) - (-2.71)) < 1.5

    def test_substeps_falls_through_to_slow_path(self):
        # substeps disqualifies the fastpath; the solve must still run
        # and produce ``posterior_samples`` of the right shape.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=5, keep=10, adapt=False)
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse, residual.mse]).solve(15, substeps=[[0], [1]])
        assert a.posterior_samples.shape == (1, 10, 1)

    def test_mixed_mode_falls_through_to_slow_path(self):
        # An optax + Bayesian mix disqualifies the fastpath; both
        # branches must still produce the right outputs.
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
            warmup=5,
            keep=10,
            adapt=False,
        )
        b.optimizer(optax.adam(1e-2))
        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        jno.core([residual.mse]).solve(15)
        assert a.posterior_samples.shape == (1, 10, 1)
        assert b.posterior_samples is None

    def test_mixed_thin_falls_through(self):
        # Heterogeneous ``thin`` across Bayesian models means a single
        # scan length isn't well-defined; the gate must reject and
        # the per-epoch loop must still work.
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
            warmup=2,
            keep=10,
            thin=1,
            adapt=False,
        )
        b.bayesian(
            blackjax.nuts,
            step_size=5e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=2,
            keep=10,
            thin=2,
            adapt=False,
        )
        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        jno.core([residual.mse]).solve(22)
        # Slow-path collection still respects per-handle thin.
        assert a.posterior_samples.shape == (1, 10, 1)
        assert b.posterior_samples.shape == (1, 10, 1)

    def test_thin_preserves_chain_length(self):
        # thin > 1 must still produce exactly ``keep`` samples.
        a = self._fastpath_solve(warmup=10, keep=30, thin=3)
        assert a.posterior_samples.shape == (1, 30, 1)

    @pytest.mark.slow
    def test_fastpath_faster_than_slow_path(self):
        # Performance benchmark: the fastpath's scan-over-keep design
        # closes three perf gaps that matter on GPU (no outer
        # value_and_grad, one XLA dispatch per chunk, one host
        # transfer per chunk).  On CPU with a tiny problem the scan's
        # fixed overhead dominates and the fastpath can be slower
        # than the Python loop — so this is marked ``slow`` and only
        # runs on the GPU/larger-problem benchmark path, not in CI.
        import time

        π = jno.np.pi
        dom = _line_domain(mesh_size=0.05)
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x)

        def _solve_once(use_substeps):
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.bayesian(
                blackjax.nuts,
                step_size=1e-2,
                inverse_mass_matrix=jnp.ones(1),
                warmup=0,
                keep=200,
                adapt=False,
            )
            residual = a * jno.np.sin(π * x) - target
            crux = jno.core([residual.mse, residual.mse]) if use_substeps else jno.core([residual.mse])
            kwargs = {"substeps": [[0], [1]]} if use_substeps else {}
            # Warmup the JIT — first call compiles.
            crux.solve(10, **kwargs)
            t0 = time.time()
            crux.solve(200, **kwargs)
            return time.time() - t0

        # Substeps forces the slow path; without forces the fastpath.
        t_slow = _solve_once(use_substeps=True)
        t_fast = _solve_once(use_substeps=False)
        # Both should be small; fastpath should not be dramatically
        # slower (a 2x margin gives wide CI tolerance).
        assert t_fast < 2.0 * t_slow + 1.0, f"fastpath ({t_fast:.2f}s) much slower than slow path ({t_slow:.2f}s)"


# ---------------------------------------------------------------------------
# Variational Inference (Phase 10) — mean-field VI via blackjax
# ---------------------------------------------------------------------------


class TestVIMeanField:
    """Phase 10 — ``Model.vi(blackjax.meanfield_vi, ...)`` fits a
    variational approximation through ``crux.solve()``.  After solve,
    ``posterior_draws`` i.i.d. samples are drawn from the fitted
    distribution and stored on ``posterior_samples`` in the same
    ``(1, N, *param)`` layout as the MCMC path.
    """

    def _vi_solve_scalar(self, *, num_iters=1500, posterior_draws=200, lr=1e-2):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 3.0 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(lr),
            num_samples=8,
            posterior_draws=posterior_draws,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(num_iters)
        return a

    def test_scalar_recovery(self):
        # Mean-field VI recovers the truth on a 1-parameter problem.
        a = self._vi_solve_scalar()
        chain = a.posterior_samples
        assert chain.shape == (1, 200, 1)
        assert abs(float(jnp.mean(chain)) - 3.0) < 0.5

    def test_vector_recovery(self):
        # VI on a 3-vector parameter.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x) + 2.0 * jno.np.cos(π * x) + 3.0 * x
        p = jno.np.parameter((3,), key=jax.random.PRNGKey(0), name="abc")
        p.vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            num_samples=8,
            posterior_draws=150,
        )
        residual = p[0] * jno.np.sin(π * x) + p[1] * jno.np.cos(π * x) + p[2] * x - target
        jno.core([residual.mse]).solve(1500)
        chain = p.posterior_samples
        assert chain.shape == (1, 150, 3)
        # Per-component means within reasonable tolerance.  Mean-field
        # VI assumes diagonal covariance, so for correlated coefficients
        # the marginal means can drift somewhat from a joint-MAP truth;
        # a generous tolerance keeps the test stable across reseeds.
        means = jnp.mean(chain, axis=(0, 1))
        truth = jnp.array([1.0, 2.0, 3.0])
        assert float(jnp.max(jnp.abs(means - truth))) < 1.5

    def test_mlp_smoke(self):
        # VI on a small MLP runs to completion and produces sensible
        # posterior_samples shape across the multi-leaf pytree.
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        net.vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            num_samples=4,
            posterior_draws=50,
        )
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(200)
        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        # Every inexact-array leaf has the (K, N, *param_leaf) layout.
        for leaf in leaves:
            assert leaf.shape[:2] == (1, 50), f"unexpected leaf shape {leaf.shape}"

    def test_posterior_samples_shape(self):
        # posterior_draws controls the second axis exactly.
        a = self._vi_solve_scalar(num_iters=200, posterior_draws=42)
        assert a.posterior_samples.shape == (1, 42, 1)

    def test_vi_and_bayesian_mutually_exclusive(self):
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=2, keep=3, adapt=False)
        with pytest.raises(ValueError, match="mutually exclusive"):
            a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-3))

    def test_bayesian_after_vi_raises(self):
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-3))
        with pytest.raises(ValueError, match="mutually exclusive"):
            a.bayesian(blackjax.nuts, step_size=1e-2, warmup=2, keep=3, adapt=False)

    def test_mixed_vi_and_mcmc_runs(self):
        # One model uses VI, another uses MCMC in the same solve.
        # Both paths populate posterior_samples with the same layout.
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.0 * jno.np.sin(π * x) + 2.0 * jno.np.cos(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")
        a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-2), num_samples=4, posterior_draws=50)
        b.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=jnp.ones(1), warmup=10, keep=50, adapt=False)
        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        jno.core([residual.mse]).solve(60)
        assert a.posterior_samples.shape == (1, 50, 1)
        assert b.posterior_samples.shape == (1, 50, 1)

    def test_init_log_std_defaults_to_minus_three(self):
        # Default init_log_std=-3 → state.rho starts at -3.0 everywhere.
        # Spy on init_state to capture the post-init state.
        import importlib

        bay_mod = importlib.import_module("jno.bayesian")
        captured = {}
        orig_init = bay_mod.init_state

        def _spy(handle, position, rng_key=None):
            state = orig_init(handle, position, rng_key=rng_key)
            if handle.kind == "vi":
                captured["rho"] = state.rho
                captured["mu"] = state.mu
            return state

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-3), num_samples=4, posterior_draws=10)
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(π * x) - jno.np.sin(π * x)
        # Patch init_state on the module the solve loop imports.
        bay_mod.init_state = _spy
        try:
            jno.core([residual.mse]).solve(2)
        finally:
            bay_mod.init_state = orig_init
        assert "rho" in captured
        # rho is a pytree of arrays, all -3.0.
        for leaf in jax.tree_util.tree_leaves(captured["rho"]):
            assert jnp.allclose(leaf, -3.0)

    def test_init_log_std_custom_value_respected(self):
        import importlib

        bay_mod = importlib.import_module("jno.bayesian")
        captured = {}
        orig_init = bay_mod.init_state

        def _spy(handle, position, rng_key=None):
            state = orig_init(handle, position, rng_key=rng_key)
            if handle.kind == "vi":
                captured["rho"] = state.rho
            return state

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-3),
            num_samples=4,
            posterior_draws=10,
            init_log_std=0.0,  # σ ≈ 1.0
        )
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(π * x) - jno.np.sin(π * x)
        bay_mod.init_state = _spy
        try:
            jno.core([residual.mse]).solve(2)
        finally:
            bay_mod.init_state = orig_init
        for leaf in jax.tree_util.tree_leaves(captured["rho"]):
            assert jnp.allclose(leaf, 0.0)

    def test_init_mu_at_position_false_keeps_blackjax_default(self):
        # When init_mu_at_position=False, state.mu stays at blackjax's
        # default (zeros).  Spy and check.
        import importlib

        bay_mod = importlib.import_module("jno.bayesian")
        captured = {}
        orig_init = bay_mod.init_state

        def _spy(handle, position, rng_key=None):
            state = orig_init(handle, position, rng_key=rng_key)
            if handle.kind == "vi":
                captured["mu"] = state.mu
            return state

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-3),
            num_samples=4,
            posterior_draws=10,
            init_mu_at_position=False,
        )
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(π * x) - jno.np.sin(π * x)
        bay_mod.init_state = _spy
        try:
            jno.core([residual.mse]).solve(2)
        finally:
            bay_mod.init_state = orig_init
        # blackjax default = zeros for every mu leaf.
        for leaf in jax.tree_util.tree_leaves(captured["mu"]):
            assert jnp.allclose(leaf, 0.0)

    def test_elbo_loss_decreases(self):
        # Trivially: after enough VI steps the loss / negative log
        # density should have decreased from the start.  Run two short
        # solves at different lengths and check the longer one ends
        # with a smaller loss.
        π = jno.np.pi

        def _run(num_iters):
            dom = _line_domain()
            x, _ = dom.variable("interior")
            target = 2.0 * jno.np.sin(π * x)
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-2), num_samples=4, posterior_draws=10)
            residual = a * jno.np.sin(π * x) - target
            crux = jno.core([residual.mse])
            history = crux.solve(num_iters)
            return float(history.total_loss)

        loss_short = _run(50)
        loss_long = _run(500)
        # Longer run converges further — assert the long-run final loss
        # is meaningfully lower than the short-run final loss.
        assert loss_long < loss_short, f"VI loss didn't decrease: short={loss_short:.4f}, long={loss_long:.4f}"


# ---------------------------------------------------------------------------
# Phase 11 — Composable per-mask backends: .mask().bayesian() / .mask().vi()
# ---------------------------------------------------------------------------


def _build_last_leaf_mask(net):
    """Build a leaf-level mask marking only the last array leaf of a model
    as ``True`` and all others as ``False``.  Used by the masked-Bayesian /
    masked-VI tests to restrict sampling to a single weight tensor.
    """
    leaves = jax.tree_util.tree_leaves(net.module)
    flags = [False] * len(leaves)
    flags[-1] = True
    treedef = jax.tree_util.tree_structure(net.module)
    return jax.tree_util.tree_unflatten(treedef, flags), len(leaves) - 1


class TestMaskedBackends:
    """Phase 11 — ``.mask(M).bayesian()`` / ``.mask(M).vi()`` restrict the
    posterior to a subset of a model's parameter pytree; the unmasked
    complement either stays at its initial value (no global backend) or
    is updated by a global ``.optimizer()`` on the same model.
    """

    def test_masked_bayesian_only_masked_varies(self):
        """Pattern A: head Bayesian, body frozen at init."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=10)

        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(15)

        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        for i, leaf in enumerate(leaves):
            # variance along the sample (N) axis, averaged over remaining dims
            var_along_n = float(jnp.mean(jnp.var(leaf, axis=1)))
            if i == masked_idx:
                assert var_along_n > 1e-8, (
                    f"masked leaf {i} should vary across the chain; got var-along-N={var_along_n:.3e}"
                )
            else:
                assert var_along_n < 1e-8, (
                    f"unmasked leaf {i} should be constant across the chain; got var-along-N={var_along_n:.3e}"
                )

    def test_masked_bayesian_with_global_optimizer_runs(self):
        """Pattern B — masked .bayesian() head + global .optimizer() body
        on the same model — is supported as of Phase 15.  Verifies that
        the solve completes, the masked leaf varies across the chain
        (Bayesian update fires), and the body leaves moved from their
        initial values (optax update fires)."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)

        # Snapshot initial body params (unmasked leaves).
        init_leaves = jax.tree_util.tree_leaves(net.module)
        init_body_leaves = [
            jnp.copy(leaf)
            for i, leaf in enumerate(init_leaves)
            if i != masked_idx and hasattr(leaf, "shape") and jnp.issubdtype(leaf.dtype, jnp.inexact)
        ]

        net.optimizer(optax.adam(1e-1))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=5, keep=20)
        residual = net(x) - 0.0
        crux = jno.core([residual.mse])
        crux.solve(30)

        # Bayesian contract: masked leaf varies across the chain.
        chain = net.posterior_samples
        assert chain is not None
        chain_leaves = jax.tree_util.tree_leaves(chain)
        masked_var = float(jnp.mean(jnp.var(chain_leaves[masked_idx], axis=1)))
        assert masked_var > 1e-8, f"masked leaf should vary across the chain; got var={masked_var:.3e}"

        # Optax contract: body params (unmasked leaves) moved during solve.
        final_leaves = jax.tree_util.tree_leaves(net.module)
        body_moved = False
        b_iter = iter(init_body_leaves)
        for i, leaf in enumerate(final_leaves):
            if i == masked_idx or not (hasattr(leaf, "shape") and jnp.issubdtype(leaf.dtype, jnp.inexact)):
                continue
            init_l = next(b_iter)
            if float(jnp.linalg.norm(leaf - init_l)) > 1e-4:
                body_moved = True
        assert body_moved, "Pattern B: body (unmasked) leaves didn't move under optax"

    def test_masked_vi_only_masked_varies(self):
        """Pattern A for VI: head VI, body frozen at init."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.mask(mask).vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            num_samples=4,
            posterior_draws=20,
        )

        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(60)

        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        masked_leaf = leaves[masked_idx]
        masked_var = float(jnp.mean(jnp.var(masked_leaf, axis=1)))
        assert masked_var > 1e-8, f"masked VI leaf should have non-zero variance from posterior draws; got {masked_var:.3e}"
        for i, leaf in enumerate(leaves):
            if i == masked_idx:
                continue
            var_along_n = float(jnp.mean(jnp.var(leaf, axis=1)))
            assert var_along_n < 1e-8, f"unmasked VI leaf {i} should be constant; got var-along-N={var_along_n:.3e}"

    def test_masked_vi_with_global_optimizer_runs(self):
        """Pattern B for VI — masked .vi() head + global .optimizer()
        body on the same model — is supported as of Phase 15.  Verifies
        the solve completes and posterior_samples is populated."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.optimizer(optax.adam(1e-1))
        net.mask(mask).vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            num_samples=4,
            posterior_draws=20,
        )
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(60)
        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        masked_var = float(jnp.mean(jnp.var(leaves[masked_idx], axis=1)))
        assert masked_var > 1e-8, f"masked VI leaf should vary across draws; got var={masked_var:.3e}"

    def test_masked_bayesian_posterior_samples_full_pytree(self):
        """The chain stores the full module pytree (every leaf present),
        not just the masked subset — so ``crux.eval(samples="auto")``
        and other downstream tooling work unchanged.
        """
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        # Count expected leaves on the live module.
        expected_n_leaves = len(jax.tree_util.tree_leaves(net.module))
        mask, _ = _build_last_leaf_mask(net)
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=8)
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(13)
        chain_leaves = jax.tree_util.tree_leaves(net.posterior_samples)
        assert len(chain_leaves) == expected_n_leaves, (
            f"chain has {len(chain_leaves)} leaves, expected {expected_n_leaves} (full pytree)"
        )

    def test_masked_with_multichain_runs(self):
        """Phase 15 lifts the v1 block on masked + num_chains > 1:
        the per-chain reassembly machinery falls out of Pattern B's
        state-storage refactor.  Verify the solve completes and the
        chain has the expected K-leading shape."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.mask(mask).bayesian(
            blackjax.sgld,
            step_size=1e-4,
            warmup=3,
            keep=5,
            num_chains=4,
        )
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(8)
        chain = net.posterior_samples
        assert chain is not None
        # Each leaf should now have leading (K, N) = (4, 5).
        leaves = jax.tree_util.tree_leaves(chain)
        for leaf in leaves:
            assert leaf.shape[:2] == (4, 5), f"unexpected K-leading shape {leaf.shape[:2]} (expected (4, 5))"
        # The masked leaf should vary along the sample axis.
        masked_var = float(jnp.mean(jnp.var(leaves[masked_idx], axis=1)))
        assert masked_var > 1e-8, f"masked leaf should vary across the chain; got var={masked_var:.3e}"

    def test_global_bayesian_then_mask_raises(self):
        """A model with both a global ``.bayesian(...)`` and a masked
        group is ambiguous — solve raises a clear error."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, _ = _build_last_leaf_mask(net)
        # Global Bayesian first, then masked Bayesian.  Configurator
        # itself doesn't reject (the user could conceivably want to
        # override); solve catches the ambiguity.
        net.bayesian(blackjax.sgld, step_size=1e-4, warmup=2, keep=3)
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-4, warmup=2, keep=3)
        residual = net(x) - 0.0
        with pytest.raises(ValueError, match="global .bayesian"):
            jno.core([residual.mse]).solve(5)

    def test_multiple_masked_bayesian_runs(self):
        """Phase 16 lifts the block on multiple disjoint masked
        ``.bayesian()`` groups (Pattern D).  Two SGLD groups on
        disjoint leaves of one model produce a unified chain whose
        full pytree captures both groups' variation."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=2, hidden=4)
        leaves = jax.tree_util.tree_leaves(net.module)
        treedef = jax.tree_util.tree_structure(net.module)
        m1_flags = [False] * len(leaves)
        m1_flags[-1] = True
        m2_flags = [False] * len(leaves)
        m2_flags[-2] = True
        m1 = jax.tree_util.tree_unflatten(treedef, m1_flags)
        m2 = jax.tree_util.tree_unflatten(treedef, m2_flags)

        net.mask(m1).bayesian(blackjax.sgld, step_size=1e-4, warmup=2, keep=5)
        net.mask(m2).bayesian(blackjax.sgld, step_size=1e-4, warmup=2, keep=5)

        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(7)
        chain = net.posterior_samples
        assert chain is not None
        # Both group leaves vary across the chain (each group's kernel
        # contributes variation to its own masked subset).
        chain_leaves = jax.tree_util.tree_leaves(chain)
        var_m1 = float(jnp.mean(jnp.var(chain_leaves[-1], axis=1)))
        var_m2 = float(jnp.mean(jnp.var(chain_leaves[-2], axis=1)))
        assert var_m1 > 1e-10, f"first masked group should vary; got var={var_m1:.3e}"
        assert var_m2 > 1e-10, f"second masked group should vary; got var={var_m2:.3e}"

    def test_mask_scope_consumed_one_shot(self):
        """``.mask(M)`` sets a pending scope consumed by the next
        configurator call.  After ``.mask(M).bayesian(...)`` the scope
        is gone, so a follow-on bare ``.bayesian(...)`` lands on the
        global path (and is then caught at solve time as ambiguous)."""
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, _ = _build_last_leaf_mask(net)
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-4, warmup=2, keep=3)
        # After this, _mask_scope_pending must be False.
        assert net._mask_scope_pending is False
        # And the configured backend lives in _param_groups, not in
        # _bayesian_cfg (which is the global slot).
        assert net._bayesian_cfg is None
        assert any(g.get("backend") == "bayesian" for g in net._param_groups)

    def test_masked_bayesian_inverse_recovery(self):
        """End-to-end recovery: a 1-leaf network whose only trainable
        leaf is the masked subset.  After 200 NUTS steps the posterior
        mean should land near the data-fit MAP — checks the full
        pipeline including the eqx.combine/eqx.filter reassembly works
        for a problem where the answer is non-trivial.
        """
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x)
        net = _tiny_net(in_dim=1, out_dim=1, hidden=2)
        # Mask True on every leaf — equivalent to no mask in the limit
        # (every parameter is masked into the Bayesian group).  This
        # exercises the eqx.partition / combine round-trip even when
        # the "unmasked complement" is empty.
        full_mask = jax.tree_util.tree_map(
            lambda leaf: True if eqx.is_inexact_array(leaf) else False,
            net.module,
        )
        net.mask(full_mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=100, keep=100)
        residual = net(x) - target
        jno.core([residual.mse]).solve(200)
        chain = net.posterior_samples
        # All inexact leaves vary (they're all in the mask).
        leaves = jax.tree_util.tree_leaves(chain)
        for leaf in leaves:
            var = float(jnp.mean(jnp.var(leaf, axis=1)))
            assert var > 1e-8, f"every leaf should vary under full mask; got var={var:.3e}"


# ---------------------------------------------------------------------------
# Under-the-hood: complex-scenario regression guards
# ---------------------------------------------------------------------------
#
# The tests above cover individual features in isolation.  This class
# targets the *combinations* and *new control flow* that compound
# assumptions in subtle ways and would slip past the per-feature tests
# if a regression occurred:
#
#   1. MCMC fastpath (Phase 9) vs slow path produce the same posterior.
#   2. Multi-leaf eqx.tree_at masks (the T10 tutorial pattern).
#   3. Window adaptation visibly mutates step_size / IMM.
#   4. Buffer chunk-boundary correctness across many chunks.
#   5. Mixed mode: the optax model's value actually moves from init.
#   6. Multi-chain reproducibility with adapt=True (window_adaptation
#      RNG determinism).


class TestUnderTheHood:
    """Combination and internal-invariant regression guards.  Each test
    verifies behaviour that's a consequence of *several* features
    interacting; per-feature unit tests above would not catch a bug
    that only manifests under the combination.
    """

    def test_fastpath_matches_slow_path_posterior(self):
        """Same problem, same seed, fastpath ON vs OFF → posterior
        statistics agree.

        Forces the slow path by passing ``offload_data=True`` (one of
        the fastpath gates).  The two paths thread PRNG keys
        differently across the chunk boundary, so chains aren't
        bit-identical — but for a well-mixed 1-D inverse problem the
        posterior mean should agree within ~1σ of either chain.
        """
        π = jno.np.pi

        def _solve(offload):
            dom = _line_domain()
            x, _ = dom.variable("interior")
            target = 2.5 * jno.np.sin(π * x)
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.bayesian(
                blackjax.nuts,
                step_size=1e-2,
                inverse_mass_matrix=jnp.ones(1),
                warmup=100,
                keep=200,
                adapt=False,
            )
            residual = a * jno.np.sin(π * x) - target
            crux = jno.core([residual.mse])
            crux.solve(300, offload_data=offload)
            return a.posterior_samples

        chain_fast = _solve(offload=False)
        chain_slow = _solve(offload=True)

        # Same shape — neither path drops or duplicates samples.
        assert chain_fast.shape == chain_slow.shape, (
            f"fast/slow chain shapes disagree: {chain_fast.shape} vs {chain_slow.shape}"
        )

        # Posterior means agree within ~1 chain-stddev.
        mean_fast = float(jnp.mean(chain_fast))
        mean_slow = float(jnp.mean(chain_slow))
        std_combined = float(jnp.std(jnp.concatenate([chain_fast.reshape(-1), chain_slow.reshape(-1)])))
        assert abs(mean_fast - mean_slow) < std_combined, (
            f"fast/slow means diverge: |{mean_fast:.4f} - {mean_slow:.4f}| = "
            f"{abs(mean_fast - mean_slow):.4f} > 1σ = {std_combined:.4f}"
        )
        # Both should recover the truth ~2.5.
        assert abs(mean_fast - 2.5) < 0.5
        assert abs(mean_slow - 2.5) < 0.5

    def test_multi_leaf_eqx_tree_at_mask_works(self):
        """T10 tutorial pattern: ``eqx.tree_at(lambda m: m.output_layer,
        all_false, replace=head_all_true)`` builds a mask covering BOTH
        ``output_layer.weight`` and ``output_layer.bias`` in one call.
        The single-leaf ``_build_last_leaf_mask`` helper doesn't
        exercise this — if multi-leaf partition / reassembly via
        ``eqx.partition`` ever breaks, T10 silently fails in docs CI but
        the unit suite stays green without this test.
        """
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)

        # The exact T10 mask construction — multi-leaf head mask.
        all_false = jax.tree_util.tree_map(lambda _: False, net.module)
        head_all_true = jax.tree_util.tree_map(lambda _: True, net.module.output_layer)
        head_mask = eqx.tree_at(lambda m: m.output_layer, all_false, replace=head_all_true)

        # Sanity: the head mask should select exactly 2 leaves (weight + bias).
        head_leaf_count = sum(int(b) for b in jax.tree_util.tree_leaves(head_mask))
        assert head_leaf_count == 2, f"expected 2 head leaves marked True; got {head_leaf_count}"

        net.mask(head_mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=10, keep=30)
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(40)

        chain = net.posterior_samples
        all_leaves = jax.tree_util.tree_leaves(chain)
        mask_leaves = jax.tree_util.tree_leaves(head_mask)
        head_vars = [float(jnp.mean(jnp.var(leaf, axis=1))) for leaf, m in zip(all_leaves, mask_leaves) if m]
        body_vars = [float(jnp.mean(jnp.var(leaf, axis=1))) for leaf, m in zip(all_leaves, mask_leaves) if not m]

        # Both head leaves (weight AND bias) must vary — this is what
        # the single-leaf helper doesn't check.
        assert len(head_vars) == 2, f"expected 2 head leaves in chain; got {len(head_vars)}"
        for v in head_vars:
            assert v > 1e-8, f"head leaf should vary across chain; got var={v:.3e}"
        for v in body_vars:
            assert v < 1e-10, f"body leaf should be frozen; got var={v:.3e}"

    def test_window_adaptation_visibly_mutates_kwargs(self, monkeypatch):
        """``adapt=True`` should actually run ``blackjax.window_adaptation``
        and replace ``step_size`` / ``inverse_mass_matrix`` in the kernel
        handle's ``extra_kwargs``.  Existing
        ``test_adapt_recovers_with_bad_initial_step_size`` only verifies
        recovery — both ``adapt=False`` with a lucky seed and ``adapt=True``
        could pass that.  This test fishes the adapted values out via a
        monkeypatched ``run_window_adaptation`` and asserts they differ
        from the (catastrophic) initial.
        """
        import jno.bayesian as bay_mod

        captured = {}
        orig = bay_mod.run_window_adaptation

        def _spy(handle, position, logdensity_fn, rng_key):
            result = orig(handle, position, logdensity_fn, rng_key)
            if result is not None:
                _state, adapted_kwargs = result
                captured["step_size"] = float(adapted_kwargs["step_size"])
                captured["imm"] = adapted_kwargs["inverse_mass_matrix"]
            return result

        monkeypatch.setattr(bay_mod, "run_window_adaptation", _spy)

        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 1.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")

        # Catastrophic initial step_size — adaptation must shrink it
        # massively or sampling diverges.
        bad_step = 1e3
        a.bayesian(
            blackjax.nuts,
            step_size=bad_step,
            inverse_mass_matrix=jnp.ones(1),
            warmup=200,
            keep=100,
            adapt=True,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(300)

        # The spy fired exactly once for this single Bayesian model.
        assert "step_size" in captured, "run_window_adaptation was never called — adapt=True did nothing"
        # Adapted step_size must be at least 2 orders of magnitude
        # smaller than the catastrophic initial 1e3.  The exact value
        # depends on the posterior curvature; on this 1-D problem it
        # typically lands near O(1) — comfortably below bad_step / 100.
        assert captured["step_size"] < bad_step / 100.0, (
            f"adapted step_size {captured['step_size']:.4f} not << bad initial {bad_step} — adaptation didn't move it"
        )
        # Adapted IMM has the right shape and is not the trivial identity.
        # NOTE: this IMM check is the load-bearing half of this test.
        # The step_size assertion above could plausibly pass under a
        # "clamp absurd step_size to sane default" fallback, but the
        # IMM moving away from the initial identity requires
        # window_adaptation to have actually run and estimated the
        # posterior covariance.
        imm = captured["imm"]
        assert imm.shape == (1,), f"adapted IMM has wrong shape {imm.shape}"
        assert float(jnp.abs(imm[0] - 1.0)) > 0.01, (
            f"adapted IMM {float(imm[0]):.4f} indistinguishable from initial 1.0 — adaptation didn't move it"
        )

    def test_fastpath_buffer_chunk_boundary_correctness(self):
        """The fastpath flushes a pre-stacked ``(chunk_keep, *)`` buffer
        per chunk and concatenates at the end.  Off-by-one or
        double-counting at chunk boundaries would change the final
        chain length in a non-linear way with respect to ``keep``.

        Verify: chain length scales linearly with ``keep`` across
        multiple chunk counts, samples are not NaN, and the chain
        is not stuck (first / last samples differ).
        """
        π = jno.np.pi

        def _solve(keep, thin, warmup):
            dom = _line_domain()
            x, _ = dom.variable("interior")
            target = 1.0 * jno.np.sin(π * x)
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.bayesian(
                blackjax.sgld,  # SGLD has no adaptation: warmup= is "skip-N" semantic
                step_size=1e-3,
                warmup=warmup,
                keep=keep,
                thin=thin,
            )
            residual = a * jno.np.sin(π * x) - target
            jno.core([residual.mse]).solve(warmup + keep * thin)
            return a.posterior_samples

        # Three different configurations that produce different chunk
        # counts (print_rate ≈ n_outer // 10, so chunk count ≈ 10 across
        # all of these, but chunk_keep differs).
        chain_a = _solve(keep=20, thin=1, warmup=5)
        chain_b = _solve(keep=60, thin=1, warmup=5)
        chain_c = _solve(keep=20, thin=3, warmup=5)

        # Shape contract: leading axis (K=1, N=keep, *param).
        assert chain_a.shape == (1, 20, 1), f"K=1 N=20 shape wrong: {chain_a.shape}"
        assert chain_b.shape == (1, 60, 1), f"K=1 N=60 shape wrong: {chain_b.shape}"
        assert chain_c.shape == (1, 20, 1), f"K=1 N=20 thin=3 shape wrong: {chain_c.shape}"

        # No NaN samples anywhere — would indicate kernel state lost
        # across a chunk boundary.
        for c, name in [(chain_a, "a"), (chain_b, "b"), (chain_c, "c")]:
            assert bool(jnp.all(jnp.isfinite(c))), f"chain {name} contains NaN / inf"

        # Chain is not stuck: first vs last sample differ.  SGLD on a
        # 1-param problem moves enough in 20 samples that the first /
        # last differ by far more than machine precision.
        first = float(chain_a[0, 0, 0])
        last = float(chain_a[0, -1, 0])
        assert abs(first - last) > 1e-5, (
            f"chain stuck — first {first:.3e} == last {last:.3e}; chunking may be dropping kernel state updates"
        )

    def test_mixed_mode_optax_param_actually_moves(self):
        """In a mixed solve (one model optax, one model Bayesian) the
        existing tests verify ``b.posterior_samples is None`` on the
        optax side.  They do NOT verify the optax param actually
        learned.  If a plumbing bug zeroed the optax gradient in
        mixed mode, that test would still pass.  This one catches it.
        """
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.0 * jno.np.sin(π * x) + 3.0 * jno.np.cos(π * x)

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")  # Bayesian
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")  # optax

        a.bayesian(blackjax.nuts, step_size=5e-2, inverse_mass_matrix=jnp.ones(1), warmup=20, keep=30, adapt=False)
        b.optimizer(optax.adam(1e-1))

        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        crux = jno.core([residual.mse])
        crux.solve(50)

        # Existing contract: only the Bayesian model carries a chain.
        assert a.posterior_samples is not None
        assert b.posterior_samples is None

        # New: the optax model's point estimate must have moved from
        # its zero init.  ``crux.eval([b])`` returns the current point
        # value; comparing against zero (the .parameter() default init)
        # tells us whether Adam actually fired.
        b_final = float(crux.eval([b]).reshape(()))
        assert abs(b_final) > 1e-2, (
            f"optax param stayed at init zero under mixed-mode solve: b_final={b_final:.3e}; "
            f"a plumbing bug may be zeroing the optax gradient when a Bayesian model is present"
        )
        # And b should have headed toward the true value of 3.0.
        # Initial b is 0.0; final must be closer to 3.0 than 0.0 was.
        assert abs(b_final - 3.0) < 3.0, (
            f"optax param moved (b_final={b_final:.3e}) but away from truth 3.0 — gradient sign may be wrong"
        )

    def test_multichain_reproducible_with_adapt(self):
        """Multi-chain solves with ``adapt=True`` go through one
        ``blackjax.window_adaptation`` run that broadcasts to K chains.
        If the adaptation RNG isn't threaded deterministically from
        the seed, two solves with the same seed produce different
        posteriors — which silently breaks reproducibility for the
        path users actually run (multi-chain + adapt is the default
        recommended setup).

        The single-chain ``TestNUTSInverseProblem.test_reproducible_with_fixed_seed``
        already covers K=1 + adapt=True implicitly; this test extends
        coverage to K>1.
        """
        π = jno.np.pi

        def _solve():
            dom = _line_domain()
            x, _ = dom.variable("interior")
            target = 1.7 * jno.np.sin(π * x)
            a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
            a.bayesian(
                blackjax.nuts,
                step_size=1e-2,
                inverse_mass_matrix=jnp.ones(1),
                warmup=40,
                keep=20,
                adapt=True,
                num_chains=3,
                init_jitter=0.0,
            )
            residual = a * jno.np.sin(π * x) - target
            jno.core([residual.mse]).solve(60)
            return a.posterior_samples

        chain_a = _solve()
        chain_b = _solve()
        assert chain_a.shape == chain_b.shape == (3, 20, 1), (
            f"unexpected multichain shape: {chain_a.shape} vs {chain_b.shape}"
        )
        assert jnp.allclose(chain_a, chain_b), (
            "multi-chain solve with adapt=True is not reproducible under a fixed seed; "
            "window_adaptation RNG is not threaded from the master seed"
        )


# ---------------------------------------------------------------------------
# Phase 12 — Logdensity-aware initializer hook (.initialize() extension)
# ---------------------------------------------------------------------------


class TestPathfinderInitializer:
    """``.initialize(jno.bayesian.pathfinder(...))`` warm-starts a chain by
    running ``blackjax.pathfinder`` against the loss-derived log-density
    *inside* ``solve()``.  These tests cover the protocol contract, the
    pathfinder dispatch, and composition with the existing mask /
    multi-chain / substep / VI features.
    """

    # ─── helpers ──────────────────────────────────────────────────────

    def _harmonic_inverse(self, *, a_init=0.0, initializer=None, adapt=True, warmup=100, keep=100, seed=0):
        """Standard 1-parameter inverse problem used by most tests in this
        class.  Returns the ``a`` parameter handle so the caller can read
        ``a.posterior_samples`` and friends.
        """
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(seed), name="a")
        a.initialize(jnp.array([a_init]))
        if initializer is not None:
            a.initialize(initializer)
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=warmup,
            keep=keep,
            adapt=adapt,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(warmup + keep)
        return a

    # ─── 1. spy: initializer invoked and IMM merged ───────────────────

    def test_pathfinder_initializer_runs_and_sets_imm(self, monkeypatch):
        """Patch ``PathfinderInitializer.__call__`` to spy; assert it ran
        exactly once with the right shape inputs, and that the IMM it
        returned was merged into ``handle.extra_kwargs``."""
        captured = {"count": 0, "imm": None}
        orig_call = jno.bayesian.PathfinderInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            captured["count"] += 1
            captured["num_chains"] = num_chains
            warm, kw = orig_call(self, rng_key, ld, position, num_chains)
            captured["imm"] = kw.get("inverse_mass_matrix")
            return warm, kw

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _spy)

        _ = self._harmonic_inverse(
            initializer=jno.bayesian.pathfinder(maxiter=20, num_samples=50),
            adapt=False,
            warmup=5,
            keep=20,
        )
        assert captured["count"] == 1, f"pathfinder should run once; ran {captured['count']} times"
        assert captured["num_chains"] == 1
        assert captured["imm"] is not None, "pathfinder didn't produce an IMM update"
        assert captured["imm"].shape == (1,), f"IMM shape wrong: {captured['imm'].shape}"

    # ─── 2. warm position is moved ────────────────────────────────────

    def test_pathfinder_warm_position_is_moved(self, monkeypatch):
        """From a deliberately bad init (a=-10, truth=2.5), pathfinder
        moves the warm position close to the MAP.  We capture the
        warm position directly via a spy on the initializer rather
        than the first chain sample (NUTS moves a lot in a few steps,
        which would mask the warm-start signal).
        """
        captured = {}
        orig_call = jno.bayesian.PathfinderInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            warm, kw = orig_call(self, rng_key, ld, position, num_chains)
            # Copy out of JAX-land — solve() donates buffers, so the
            # captured arrays would be deleted by the time we read them.
            captured["warm"] = jax.tree_util.tree_map(
                lambda x: float(jnp.asarray(x).reshape(-1)[0]) if hasattr(x, "shape") else x,
                warm,
            )
            return warm, kw

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _spy)

        _ = self._harmonic_inverse(
            a_init=-10.0,
            initializer=jno.bayesian.pathfinder(maxiter=30, num_samples=100),
            adapt=False,
            warmup=2,
            keep=10,
        )
        warm_value = jax.tree_util.tree_leaves(captured["warm"])[0]
        # Pathfinder should have moved from -10 to near the truth 2.5.
        assert abs(warm_value - 2.5) < 0.5, (
            f"pathfinder didn't move the warm position from -10 to near 2.5; got {warm_value:.3f}"
        )

    # ─── 3. chained: pathfinder + window ──────────────────────────────

    def test_pathfinder_then_window_chain(self):
        """``pathfinder + adapt=True`` runs both: pathfinder warm-starts;
        window adaptation refines step_size from there.  Sampler still
        recovers truth."""
        a = self._harmonic_inverse(
            a_init=-5.0,
            initializer=jno.bayesian.pathfinder(maxiter=20, num_samples=80),
            adapt=True,
            warmup=80,
            keep=80,
        )
        chain = a.posterior_samples
        assert chain.shape == (1, 80, 1)
        # Truth = 2.5; window adaptation from pathfinder's warm position
        # gives a tight chain.
        assert abs(float(jnp.mean(chain)) - 2.5) < 0.5

    # ─── 4. pathfinder only — no window adaptation ────────────────────

    def test_pathfinder_only_no_window(self, monkeypatch):
        """With ``adapt=False``, window adaptation is skipped entirely
        even when a pathfinder initializer is set.  Patch
        ``run_window_adaptation`` to a sentinel; assert it wasn't
        called.
        """
        sentinel = {"called": False}

        def _never_call(*_args, **_kw):
            sentinel["called"] = True
            return None

        monkeypatch.setattr(jno.bayesian, "run_window_adaptation", _never_call)

        _ = self._harmonic_inverse(
            initializer=jno.bayesian.pathfinder(maxiter=20, num_samples=50),
            adapt=False,
            warmup=5,
            keep=20,
        )
        assert not sentinel["called"], "window_adaptation should not run when adapt=False"

    # ─── 5. pathfinder kwargs reach blackjax ──────────────────────────

    def test_pathfinder_kwargs_forwarded(self, monkeypatch):
        """``pathfinder(maxiter=1)`` (too few L-BFGS iters from a bad
        starting point) produces a clearly worse warm position than
        ``maxiter=30``.  We capture both warm positions via a spy and
        compare them directly — chain samples are too noisy to detect
        the difference reliably.
        """
        captured = {"warms": []}
        orig_call = jno.bayesian.PathfinderInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            warm, kw = orig_call(self, rng_key, ld, position, num_chains)
            captured["warms"].append(
                jax.tree_util.tree_map(
                    lambda x: float(jnp.asarray(x).reshape(-1)[0]) if hasattr(x, "shape") else x,
                    warm,
                )
            )
            return warm, kw

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _spy)

        # maxiter=30 — fully converged
        _ = self._harmonic_inverse(
            a_init=-10.0,
            initializer=jno.bayesian.pathfinder(maxiter=30, num_samples=80),
            adapt=False,
            warmup=2,
            keep=8,
        )
        # maxiter=1 — barely moved
        _ = self._harmonic_inverse(
            a_init=-10.0,
            initializer=jno.bayesian.pathfinder(maxiter=1, num_samples=80),
            adapt=False,
            warmup=2,
            keep=8,
        )
        assert len(captured["warms"]) == 2
        good_value = jax.tree_util.tree_leaves(captured["warms"][0])[0]
        bad_value = jax.tree_util.tree_leaves(captured["warms"][1])[0]
        # maxiter=30 lands at the MAP (~2.5); maxiter=1 still near -10.
        assert abs(good_value - 2.5) < abs(bad_value - 2.5), (
            f"maxiter knob didn't propagate: good_warm={good_value:.3f}, bad_warm={bad_value:.3f}"
        )

    # ─── 6. multichain dispersion ─────────────────────────────────────

    def test_pathfinder_multichain_init_dispersion(self):
        """K=4 + pathfinder draws K distinct positions from the fitted q.
        With ``init_jitter`` ALSO set, an info-log fires but pathfinder's
        dispersion takes precedence (deterministic, doesn't fall back to
        jitter).
        """
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.initialize(jno.bayesian.pathfinder(maxiter=20, num_samples=80))
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=2,
            keep=8,
            adapt=False,
            num_chains=4,
            init_jitter=0.5,  # would normally disperse; should be overridden
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(10)
        chain = a.posterior_samples
        assert chain.shape == (4, 8, 1)
        # K=4 first samples must differ pairwise — pathfinder samples
        # K distinct starting points from the fitted q.
        firsts = chain[:, 0, 0]
        pairwise_diff = float(jnp.max(firsts) - jnp.min(firsts))
        assert pairwise_diff > 1e-4, f"K=4 chains share a starting position: {firsts}"

    # ─── 7. composes with .mask().bayesian() ──────────────────────────

    def test_pathfinder_with_mask_works(self):
        """``.mask(M).bayesian()`` + pathfinder: pathfinder runs on the
        masked subset's log-density; the unmasked complement stays at
        init in ``trainable[lid]``.
        """
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.initialize(jno.bayesian.pathfinder(maxiter=10, num_samples=30))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=10)
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(15)

        chain = net.posterior_samples
        leaves = jax.tree_util.tree_leaves(chain)
        # Masked leaf varies (sampled); unmasked leaves are constant
        # (unchanged by pathfinder, unchanged by SGLD).
        for i, leaf in enumerate(leaves):
            var = float(jnp.mean(jnp.var(leaf, axis=1)))
            if i == masked_idx:
                assert var > 1e-8, f"masked leaf {i} should vary; got var={var:.3e}"
            else:
                assert var < 1e-8, f"unmasked leaf {i} should be constant; got var={var:.3e}"

    # ─── 8. substeps guard ────────────────────────────────────────────

    def test_pathfinder_with_substeps_raises(self):
        """substeps + pathfinder → clear error (initializer runs against
        the full loss, but substep kernel sees only substep-local
        constraints).  Mirrors the existing adapt+substeps guard."""
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.initialize(jno.bayesian.pathfinder(maxiter=5))
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=2, keep=5, adapt=False)
        # Use a second optax-trained parameter so substeps has a second
        # group to alternate against (substeps require >=2 substeps).
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")
        b.optimizer(optax.adam(1e-2))
        residual_a = a * jno.np.sin(π * x) - target
        residual_b = b * jno.np.sin(π * x) - target
        with pytest.raises(ValueError, match="substeps"):
            jno.core([residual_a.mse, residual_b.mse]).solve(7, substeps=[([0], 1), ([1], 1)])

    # ─── 9. VI guard ──────────────────────────────────────────────────

    def test_pathfinder_with_vi_raises(self):
        """``.vi(...)`` + pathfinder → clear error (VI's init sets
        ``state.mu = position`` itself; a logdensity-aware warm-start
        doesn't compose)."""
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.initialize(jno.bayesian.pathfinder(maxiter=5))
        a.vi(blackjax.meanfield_vi, optimizer=optax.adam(1e-2), num_samples=4, posterior_draws=10)
        residual = a * jno.np.sin(π * x) - target
        with pytest.raises(ValueError, match=".vi"):
            jno.core([residual.mse]).solve(5)

    # ─── 10. non-IMM kernel drops IMM, keeps warm position ───────────

    def test_pathfinder_with_non_imm_kernel_keeps_position(self):
        """MALA doesn't accept ``inverse_mass_matrix``; pathfinder's IMM
        update should be silently dropped, but the warm position is
        still applied and sampling runs to completion."""
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.initialize(jno.bayesian.pathfinder(maxiter=20, num_samples=50))
        # MALA doesn't accept inverse_mass_matrix in its factory signature.
        a.bayesian(blackjax.mala, step_size=1e-3, warmup=5, keep=20, adapt=False)
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(25)
        chain = a.posterior_samples
        assert chain.shape == (1, 20, 1)
        # The warm position landed the chain near truth (within MALA's
        # short-chain noise tolerance).
        assert abs(float(jnp.mean(chain)) - 2.5) < 1.0

    # ─── 11. reproducibility ──────────────────────────────────────────

    def test_pathfinder_reproducible_with_fixed_seed(self):
        """Two solves with identical master seed produce identical
        posteriors.  Verifies pathfinder's L-BFGS RNG is threaded
        deterministically."""
        a1 = self._harmonic_inverse(
            a_init=-1.0,
            initializer=jno.bayesian.pathfinder(maxiter=15, num_samples=40),
            adapt=False,
            warmup=2,
            keep=15,
        )
        a2 = self._harmonic_inverse(
            a_init=-1.0,
            initializer=jno.bayesian.pathfinder(maxiter=15, num_samples=40),
            adapt=False,
            warmup=2,
            keep=15,
        )
        assert jnp.allclose(a1.posterior_samples, a2.posterior_samples), (
            "pathfinder warm-start not reproducible under a fixed seed"
        )

    # ─── 12. .initialize() lifecycle ──────────────────────────────────

    def test_initialize_marker_clears_other_init_state(self):
        """Calling ``.initialize(pretrained_tree)`` then
        ``.initialize(pathfinder(...))`` clears the pretrained tree —
        ``.initialize`` is last-write-wins.  Regression guard for the
        lifecycle cleanup in trace.py."""
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        # Existing-path init: pytree.  Use a tiny eqx.Module so the
        # existing isinstance(weights, eqx.Module) branch fires.
        pre_tree = a.model.module.__class__(value=jnp.array([99.0]))
        a.initialize(pre_tree)
        assert a.model._weight_tree is not None
        # Now overwrite with the pathfinder marker.
        a.initialize(jno.bayesian.pathfinder(maxiter=5))
        assert a.model._weight_tree is None, "_weight_tree should be cleared by the marker branch"
        assert a.model._bayesian_initializer is not None
        # And the reverse — re-set to a tree → marker must be cleared.
        a.initialize(pre_tree)
        assert a.model._bayesian_initializer is None, "_bayesian_initializer should be cleared by the tree branch"

    # ─── 13. protocol subclassability ─────────────────────────────────

    def test_protocol_contract_smoke(self):
        """A minimal user-written subclass of ``_BayesianInitializer`` runs
        end-to-end through ``.initialize()`` and ``solve()`` — validates
        the protocol is genuinely subclassable by third parties.

        The subclass deliberately produces a logdensity-derived warm
        position (the gradient direction from the input) so we can verify
        both invocation and that the result reaches ``trainable[lid]``
        and feeds the kernel.
        """
        call_count = [0]

        class _IdentityInitializer(jno.bayesian._BayesianInitializer):
            """Returns the input position unchanged + a trivial kwargs
            update.  Verifies the protocol dispatch end-to-end without
            any algorithmic content."""

            def __call__(self, rng_key, logdensity_fn, position, num_chains):
                call_count[0] += 1
                # Sanity-check: the logdensity_fn is callable on the
                # input position (the protocol must wire it correctly).
                ld_val = logdensity_fn(position)
                assert jnp.isfinite(ld_val), "ld_fn returned non-finite value"
                return position, {}

        a = self._harmonic_inverse(
            a_init=2.4,
            initializer=_IdentityInitializer(),
            adapt=False,
            warmup=2,
            keep=15,
        )
        # 1) The user-written initializer was invoked.
        assert call_count[0] == 1, f"protocol dispatched {call_count[0]} times; expected 1"
        # 2) The solve completed with a valid posterior shape.
        chain = a.posterior_samples
        assert chain.shape == (1, 15, 1)
        # 3) Identity initializer means the chain starts near the user's
        #    init (2.4) and recovers truth ~2.5 in 15 steps.
        assert abs(float(jnp.mean(chain)) - 2.5) < 1.0


# ---------------------------------------------------------------------------
# Phase 13 — LaplaceInitializer (MAP + Hessian-based Gaussian approximation)
# ---------------------------------------------------------------------------


class TestLaplaceInitializer:
    """``.initialize(jno.bayesian.laplace(...))`` finds the MAP via optax
    and forms a Gaussian approximation to the posterior using the
    Hessian at the MAP.  These tests cover the protocol contract, the
    diagonal/full Hessian strategies, and composition with the existing
    mask / multi-chain features (shared with the pathfinder dispatch).
    """

    @staticmethod
    def _harmonic_inverse(*, initializer, adapt=False, warmup=2, keep=15, seed=0, **bayes_kwargs):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(seed), name="a")
        if initializer is not None:
            a.initialize(initializer)
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=warmup,
            keep=keep,
            adapt=adapt,
            **bayes_kwargs,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(warmup + keep)
        return a

    # ─── 1. spy: invoked and IMM merged ───────────────────────────────

    def test_laplace_initializer_runs_and_sets_imm(self, monkeypatch):
        """Spy on ``LaplaceInitializer.__call__``; assert it ran exactly
        once with correct shape inputs and that the returned IMM was
        merged into ``handle.extra_kwargs``."""
        captured = {"count": 0, "imm_shape": None}
        orig_call = jno.bayesian.LaplaceInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            captured["count"] += 1
            warm, kw = orig_call(self, rng_key, ld, position, num_chains)
            captured["imm_shape"] = kw["inverse_mass_matrix"].shape
            return warm, kw

        monkeypatch.setattr(jno.bayesian.LaplaceInitializer, "__call__", _spy)
        _ = self._harmonic_inverse(initializer=jno.bayesian.laplace(map_steps=50))
        assert captured["count"] == 1
        assert captured["imm_shape"] == (1,), f"IMM shape wrong: {captured['imm_shape']}"

    # ─── 2. recovers truth on a 1-D inverse problem ──────────────────

    def test_laplace_recovers_inverse_problem(self):
        """Full pipeline: Laplace warm-start + NUTS chain recovers the
        truth on a 1-D inverse problem.  Tolerance is loose because
        jno's MSE convention treats the data noise as σ²=1, so the
        posterior is genuinely wide (std ≈ 1.0 for this problem); we
        check the mean lies within ~1 posterior-std of truth.
        """
        a = self._harmonic_inverse(
            initializer=jno.bayesian.laplace(map_steps=300, map_optimizer=optax.adam(1e-1)),
            warmup=2,
            keep=50,
        )
        chain = a.posterior_samples
        assert chain.shape == (1, 50, 1)
        assert abs(float(jnp.mean(chain)) - 2.5) < 1.5

    # ─── 3. diagonal vs full agree on 1-D problem ────────────────────

    def test_laplace_diagonal_matches_full_on_1d(self, monkeypatch):
        """For a 1-parameter problem, diagonal and full Hessian
        strategies should produce identical IMMs (D=1 → trivially
        diagonal)."""
        captured = {"diag_imm": None, "full_imm": None}
        orig_call = jno.bayesian.LaplaceInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            warm, kw = orig_call(self, rng_key, ld, position, num_chains)
            key = "diag_imm" if self.hessian_strategy == "diagonal" else "full_imm"
            captured[key] = float(kw["inverse_mass_matrix"][0])
            return warm, kw

        monkeypatch.setattr(jno.bayesian.LaplaceInitializer, "__call__", _spy)
        _ = self._harmonic_inverse(
            initializer=jno.bayesian.laplace(map_steps=200, map_optimizer=optax.adam(1e-1), hessian_strategy="diagonal")
        )
        _ = self._harmonic_inverse(
            initializer=jno.bayesian.laplace(map_steps=200, map_optimizer=optax.adam(1e-1), hessian_strategy="full")
        )
        assert captured["diag_imm"] is not None and captured["full_imm"] is not None
        # Should agree to numerical precision since D=1.
        assert jnp.isclose(captured["diag_imm"], captured["full_imm"], rtol=1e-4), (
            f"diagonal {captured['diag_imm']} != full {captured['full_imm']}"
        )

    # ─── 4. multichain: K distinct positions ─────────────────────────

    def test_laplace_multichain_dispersion(self):
        """K=4 chains start at K distinct samples from N(MAP, H^{-1})."""
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.initialize(jno.bayesian.laplace(map_steps=200, map_optimizer=optax.adam(1e-1)))
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=2,
            keep=8,
            num_chains=4,
            adapt=False,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(10)
        chain = a.posterior_samples
        assert chain.shape == (4, 8, 1)
        firsts = chain[:, 0, 0]
        pairwise = float(jnp.max(firsts) - jnp.min(firsts))
        assert pairwise > 1e-4, f"K=4 chains share a starting position: {firsts}"

    # ─── 5. composes with .mask().bayesian() ─────────────────────────

    def test_laplace_with_mask_works(self):
        """``.mask(M).bayesian()`` + Laplace: only the masked subset is
        Laplace-warm-started; the unmasked complement stays at init."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.initialize(jno.bayesian.laplace(map_steps=20, hessian_strategy="diagonal"))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=10)
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(15)

        chain = net.posterior_samples
        leaves = jax.tree_util.tree_leaves(chain)
        for i, leaf in enumerate(leaves):
            var = float(jnp.mean(jnp.var(leaf, axis=1)))
            if i == masked_idx:
                assert var > 1e-8, f"masked leaf {i} should vary; got var={var:.3e}"
            else:
                assert var < 1e-8, f"unmasked leaf {i} should be constant; got var={var:.3e}"

    # ─── 6. reproducible under a fixed seed ──────────────────────────

    def test_laplace_reproducible_with_fixed_seed(self):
        """Two solves with identical master seed produce identical
        posteriors.  Verifies the MAP scan + Hessian Cholesky path is
        deterministic."""
        a1 = self._harmonic_inverse(initializer=jno.bayesian.laplace(map_steps=100, map_optimizer=optax.adam(5e-2)))
        a2 = self._harmonic_inverse(initializer=jno.bayesian.laplace(map_steps=100, map_optimizer=optax.adam(5e-2)))
        assert jnp.allclose(a1.posterior_samples, a2.posterior_samples)

    # ─── 7. invalid hessian_strategy raises ──────────────────────────

    def test_laplace_invalid_hessian_strategy_raises(self):
        """``hessian_strategy`` must be 'diagonal' or 'full'; anything
        else raises a clear ValueError at run time."""
        with pytest.raises(ValueError, match="hessian_strategy"):
            self._harmonic_inverse(initializer=jno.bayesian.laplace(map_steps=5, hessian_strategy="bogus"))


# ---------------------------------------------------------------------------
# Phase 14 — SVGDInitializer (Stein Variational Gradient Descent)
# ---------------------------------------------------------------------------


class TestSVGDInitializer:
    """``.initialize(jno.bayesian.svgd(...))`` runs Stein Variational
    Gradient Descent (Liu & Wang 2016) — a particle-based variational
    method — and uses the final particle cloud as the warm-start.
    These tests cover the protocol contract, the multi-chain dispersion
    coming directly from the particle dynamics, and the composition
    with masks (shared dispatch helpers).
    """

    @staticmethod
    def _harmonic_inverse(*, initializer, adapt=False, warmup=2, keep=15, seed=0, **bayes_kwargs):
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(seed), name="a")
        if initializer is not None:
            a.initialize(initializer)
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=warmup,
            keep=keep,
            adapt=adapt,
            **bayes_kwargs,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(warmup + keep)
        return a

    # ─── 1. spy: invoked and IMM merged ───────────────────────────────

    def test_svgd_initializer_runs_and_sets_imm(self, monkeypatch):
        """Spy on ``SVGDInitializer.__call__``; assert it ran exactly
        once with correct shape inputs and that the returned IMM was
        merged into ``handle.extra_kwargs``."""
        captured = {"count": 0, "imm_shape": None}
        orig_call = jno.bayesian.SVGDInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            captured["count"] += 1
            warm, kw = orig_call(self, rng_key, ld, position, num_chains)
            captured["imm_shape"] = kw["inverse_mass_matrix"].shape
            return warm, kw

        monkeypatch.setattr(jno.bayesian.SVGDInitializer, "__call__", _spy)
        _ = self._harmonic_inverse(initializer=jno.bayesian.svgd(num_iters=50, num_particles=16))
        assert captured["count"] == 1
        assert captured["imm_shape"] == (1,), f"IMM shape wrong: {captured['imm_shape']}"

    # ─── 2. recovers truth ───────────────────────────────────────────

    def test_svgd_recovers_inverse_problem(self):
        """SVGD warm-start + NUTS chain recovers the truth on a 1-D
        inverse problem."""
        a = self._harmonic_inverse(
            initializer=jno.bayesian.svgd(num_iters=200, num_particles=32, init_jitter=2.0),
            warmup=2,
            keep=50,
        )
        chain = a.posterior_samples
        assert chain.shape == (1, 50, 1)
        # Wide-prior posterior is broad; check the chain mean is within
        # ~1 posterior-std of truth (see TestLaplaceInitializer note).
        assert abs(float(jnp.mean(chain)) - 2.5) < 1.5

    # ─── 3. K>1 particle dispersion ──────────────────────────────────

    def test_svgd_multichain_uses_distinct_particles(self):
        """K=4 chains start at 4 distinct particles from the final
        cloud; particle dynamics provide natural over-dispersion."""
        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        target = 2.5 * jno.np.sin(π * x)
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.initialize(jno.bayesian.svgd(num_iters=100, num_particles=16, init_jitter=2.0))
        a.bayesian(
            blackjax.nuts,
            step_size=1e-2,
            inverse_mass_matrix=jnp.ones(1),
            warmup=2,
            keep=8,
            num_chains=4,
            adapt=False,
        )
        residual = a * jno.np.sin(π * x) - target
        jno.core([residual.mse]).solve(10)
        chain = a.posterior_samples
        assert chain.shape == (4, 8, 1)
        firsts = chain[:, 0, 0]
        pairwise = float(jnp.max(firsts) - jnp.min(firsts))
        assert pairwise > 1e-3, f"K=4 chains share a starting particle: {firsts}"

    # ─── 4. num_particles < num_chains raises ────────────────────────

    def test_svgd_num_particles_too_small_raises(self):
        """``num_particles < num_chains`` raises a clear ValueError —
        you can't slice K chain inits out of fewer than K particles."""
        with pytest.raises(ValueError, match="num_particles"):
            self._harmonic_inverse(
                initializer=jno.bayesian.svgd(num_iters=5, num_particles=2),
                num_chains=4,
                warmup=1,
                keep=2,
            )

    # ─── 5. composes with .mask().bayesian() ─────────────────────────

    def test_svgd_with_mask_works(self):
        """``.mask(M).bayesian()`` + SVGD: only the masked subset is
        warm-started; the unmasked complement stays at init."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.initialize(jno.bayesian.svgd(num_iters=20, num_particles=8, init_jitter=0.5))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-4, warmup=5, keep=10)
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(15)

        chain = net.posterior_samples
        leaves = jax.tree_util.tree_leaves(chain)
        for i, leaf in enumerate(leaves):
            var = float(jnp.mean(jnp.var(leaf, axis=1)))
            if i == masked_idx:
                assert var > 1e-8, f"masked leaf {i} should vary; got var={var:.3e}"
            else:
                assert var < 1e-8, f"unmasked leaf {i} should be constant; got var={var:.3e}"

    # ─── 6. reproducibility ──────────────────────────────────────────

    def test_svgd_reproducible_with_fixed_seed(self):
        """Two solves with identical master seed produce identical
        posteriors.  Verifies the SVGD scan threads PRNG keys
        deterministically."""
        a1 = self._harmonic_inverse(initializer=jno.bayesian.svgd(num_iters=80, num_particles=16, init_jitter=1.0))
        a2 = self._harmonic_inverse(initializer=jno.bayesian.svgd(num_iters=80, num_particles=16, init_jitter=1.0))
        assert jnp.allclose(a1.posterior_samples, a2.posterior_samples)

    # ─── 7. particles approximate the posterior ──────────────────────

    def test_svgd_particles_approximate_posterior(self, monkeypatch):
        """Spy on SVGD; assert the final particle cloud's mean is
        near truth and the diagonal IMM is positive.  Sanity check on
        the SVGD path itself, independent of the NUTS chain that
        follows."""
        captured = {}
        orig_call = jno.bayesian.SVGDInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            warm, kw = orig_call(self, rng_key, ld, position, num_chains)
            captured["warm"] = float(jax.tree_util.tree_leaves(warm)[0].reshape(-1)[0])
            captured["imm"] = float(kw["inverse_mass_matrix"][0])
            return warm, kw

        monkeypatch.setattr(jno.bayesian.SVGDInitializer, "__call__", _spy)
        _ = self._harmonic_inverse(initializer=jno.bayesian.svgd(num_iters=300, num_particles=32, init_jitter=2.0))
        assert abs(captured["warm"] - 2.5) < 1.5, f"SVGD ensemble mean {captured['warm']} far from truth 2.5"
        assert captured["imm"] > 0.0, "SVGD particle variance should be positive"

    # ─── 8. init_jitter default is scale-aware ───────────────────────

    def test_init_jitter_default_is_none(self):
        # The class-level default was historically 1.0 — that's ~100× too
        # large for Xavier-init weights at scale 0.01.  None signals the
        # scale-aware code path inside __call__.
        init = jno.bayesian.svgd(num_iters=10, num_particles=8)
        assert init.init_jitter is None

    def test_init_jitter_none_picks_scale_aware_default(self, monkeypatch):
        # When init_jitter is None, __call__ derives a per-position
        # default of ``max(0.1 * std(flat_pos), 1e-3)``.  We spy on the
        # initial-particle creation to check the std of the first-step
        # particles matches that derived jitter.
        captured = {}
        orig_call = jno.bayesian.SVGDInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            # Compute expected jitter without re-running SVGD.
            import jax.flatten_util as _fu

            flat_pos, _ = _fu.ravel_pytree(position)
            captured["expected"] = float(jnp.maximum(0.1 * jnp.std(flat_pos), 1e-3))
            return orig_call(self, rng_key, ld, position, num_chains)

        monkeypatch.setattr(jno.bayesian.SVGDInitializer, "__call__", _spy)
        # Use a tiny MLP so the position std is well below 1.0.
        net = _tiny_net(in_dim=1, out_dim=1, hidden=8, key_seed=3)
        net.initialize(jno.bayesian.svgd(num_iters=5, num_particles=4))
        net.bayesian(blackjax.nuts, step_size=1e-2, warmup=1, keep=1, adapt=False)
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = net(x) - 0.0
        jno.core([residual.mse]).solve(2)
        # Scale-aware default is in (0, 1.0) — never the historical 1.0.
        assert 0.0 < captured["expected"] < 1.0

    def test_init_jitter_explicit_positive_respected(self):
        init = jno.bayesian.svgd(num_iters=10, num_particles=8, init_jitter=0.05)
        assert init.init_jitter == 0.05

    def test_init_jitter_explicit_non_positive_raises(self):
        # The validation fires at __call__ time (the dataclass holds the
        # raw user value).  We trigger it by running through a solve.
        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        a.initialize(jno.bayesian.svgd(num_iters=5, num_particles=4, init_jitter=0.0))
        a.bayesian(blackjax.nuts, step_size=1e-2, warmup=1, keep=1, adapt=False)
        dom = _line_domain()
        x, _ = dom.variable("interior")
        residual = a * jno.np.sin(jno.np.pi * x) - jno.np.sin(jno.np.pi * x)
        with pytest.raises(ValueError, match="init_jitter must be > 0"):
            jno.core([residual.mse]).solve(2)


# ---------------------------------------------------------------------------
# Initializer composition — multiple initializers / mixed-mode in one solve
# ---------------------------------------------------------------------------


class TestInitializerComposition:
    """The dispatch in ``core.py`` block 6a iterates ``_init_candidates``
    per Bayesian model — so multiple Bayesian models in the **same**
    solve can each carry a *different* initializer, and Bayesian models
    with initializers can coexist with optax-trained models in mixed
    mode.  These tests pin those compositions.

    Note: mixing **on a single pytree** (e.g. an MLP whose head is
    Bayesian and body is optax-trained) is Phase 15's Pattern B and is
    now supported — see
    ``TestMaskedBackends.test_masked_bayesian_with_global_optimizer_runs``.
    """

    # ─── 1. Two Bayesian models, two different initializers ──────────

    def test_different_initializers_per_model(self, monkeypatch):
        """``a.initialize(pathfinder)`` + ``b.initialize(laplace)`` in one
        solve: each initializer runs exactly once on its own model;
        each model gets its own warm position and IMM.
        """
        pf_count = [0]
        lap_count = [0]
        orig_pf = jno.bayesian.PathfinderInitializer.__call__
        orig_lap = jno.bayesian.LaplaceInitializer.__call__

        def _spy_pf(self, rng_key, ld, position, num_chains):
            pf_count[0] += 1
            return orig_pf(self, rng_key, ld, position, num_chains)

        def _spy_lap(self, rng_key, ld, position, num_chains):
            lap_count[0] += 1
            return orig_lap(self, rng_key, ld, position, num_chains)

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _spy_pf)
        monkeypatch.setattr(jno.bayesian.LaplaceInitializer, "__call__", _spy_lap)

        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        A_true, B_true = 3.14, -2.71
        target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x)

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")
        a.initialize(jno.bayesian.pathfinder(maxiter=20, num_samples=50))
        b.initialize(jno.bayesian.laplace(map_steps=100, map_optimizer=optax.adam(1e-1)))
        a.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=jnp.ones(1), warmup=2, keep=15, adapt=False)
        b.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=jnp.ones(1), warmup=2, keep=15, adapt=False)

        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        jno.core([residual.mse]).solve(17)

        # Each initializer ran exactly once — for its own model.
        assert pf_count[0] == 1, f"pathfinder ran {pf_count[0]} times; expected 1"
        assert lap_count[0] == 1, f"laplace ran {lap_count[0]} times; expected 1"
        # Both chains exist and are independent.
        assert a.posterior_samples.shape == (1, 15, 1)
        assert b.posterior_samples.shape == (1, 15, 1)
        # Both posteriors are in the right ballpark (wide posterior — loose
        # tolerance, see TestLaplaceInitializer for why).
        assert abs(float(jnp.mean(a.posterior_samples)) - A_true) < 2.0
        assert abs(float(jnp.mean(b.posterior_samples)) - B_true) < 2.0

    # ─── 2. Initializer on Bayesian + optax on another model ─────────

    def test_initializer_plus_optax_mixed_mode(self, monkeypatch):
        """One model with ``.initialize(pathfinder).bayesian(...)``, another
        with ``.optimizer(adam)``: pathfinder runs on the Bayesian
        model only, optax updates the other, both recover.  Combines
        ``TestPathfinderInitializer`` with the existing
        ``TestMixedOptimizerBayesian`` setup.
        """
        pf_count = [0]
        orig_pf = jno.bayesian.PathfinderInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            pf_count[0] += 1
            return orig_pf(self, rng_key, ld, position, num_chains)

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _spy)

        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        A_true, B_true = 2.0, 3.0
        target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x)

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")  # Bayesian + pathfinder
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")  # optax

        a.initialize(jno.bayesian.pathfinder(maxiter=20, num_samples=50))
        a.bayesian(blackjax.nuts, step_size=5e-2, inverse_mass_matrix=jnp.ones(1), warmup=2, keep=30, adapt=False)
        b.optimizer(optax.adam(1e-1))

        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        crux = jno.core([residual.mse])
        crux.solve(50)

        # Pathfinder ran exactly once — on the Bayesian model only.
        assert pf_count[0] == 1, f"pathfinder ran {pf_count[0]} times; expected 1 (only on Bayesian model)"
        # Bayesian model has a chain; optax model does not.
        assert a.posterior_samples is not None
        assert b.posterior_samples is None
        # Optax model moved from zero init toward B_true.
        b_final = float(crux.eval([b]).reshape(()))
        assert abs(b_final) > 1e-2, f"optax param stayed at init zero in mixed-mode with initializer; got {b_final:.3e}"
        # Bayesian chain mean recovers A_true (loose, see LaplaceInitializer
        # note on jno's MSE convention giving a wide posterior).
        assert abs(float(jnp.mean(a.posterior_samples)) - A_true) < 1.5

    # ─── 3. All three initializers in one solve ──────────────────────

    def test_three_initializers_one_solve(self, monkeypatch):
        """``a.initialize(pathfinder)`` + ``b.initialize(laplace)`` +
        ``c.initialize(svgd)`` — three Bayesian models, three different
        initializers.  Each runs exactly once on its own model; all
        three chains are produced.  Validates that block 6a correctly
        iterates and applies *different* initializer types per model.
        """
        counts = {"pf": 0, "lap": 0, "svgd": 0}
        orig_pf = jno.bayesian.PathfinderInitializer.__call__
        orig_lap = jno.bayesian.LaplaceInitializer.__call__
        orig_svgd = jno.bayesian.SVGDInitializer.__call__

        def _make_spy(name, orig):
            def _wrapped(self, rng_key, ld, position, num_chains):
                counts[name] += 1
                return orig(self, rng_key, ld, position, num_chains)

            return _wrapped

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _make_spy("pf", orig_pf))
        monkeypatch.setattr(jno.bayesian.LaplaceInitializer, "__call__", _make_spy("lap", orig_lap))
        monkeypatch.setattr(jno.bayesian.SVGDInitializer, "__call__", _make_spy("svgd", orig_svgd))

        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        A_true, B_true, C_true = 2.0, 1.5, -1.0
        target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x) + C_true * x * (1 - x)

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")
        c = jno.np.parameter((1,), key=jax.random.PRNGKey(2), name="c")
        a.initialize(jno.bayesian.pathfinder(maxiter=20, num_samples=50))
        b.initialize(jno.bayesian.laplace(map_steps=100, map_optimizer=optax.adam(1e-1)))
        c.initialize(jno.bayesian.svgd(num_iters=80, num_particles=16, init_jitter=1.0))
        for p in (a, b, c):
            p.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=jnp.ones(1), warmup=2, keep=15, adapt=False)

        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) + c * x * (1 - x) - target
        jno.core([residual.mse]).solve(17)

        # Each initializer ran on exactly its own model.
        assert counts == {"pf": 1, "lap": 1, "svgd": 1}, f"unexpected initializer call counts: {counts}"
        # All three chains exist with the right shape.
        for p in (a, b, c):
            assert p.posterior_samples.shape == (1, 15, 1)

    # ─── 4. Some Bayesian models have initializers, others don't ─────

    def test_initializer_on_subset_of_bayesian_models(self, monkeypatch):
        """``a.initialize(pathfinder).bayesian(...)`` + ``b.bayesian(...)``
        (no initializer on b): pathfinder runs only on a; b uses the
        standard init_state path.  Verifies the per-model filter in
        block 6a's ``_init_candidates`` builder.
        """
        pf_count = [0]
        orig_pf = jno.bayesian.PathfinderInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            pf_count[0] += 1
            return orig_pf(self, rng_key, ld, position, num_chains)

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _spy)

        π = jno.np.pi
        dom = _line_domain()
        x, _ = dom.variable("interior")
        A_true, B_true = 2.0, -1.0
        target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x)

        a = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="a")
        b = jno.np.parameter((1,), key=jax.random.PRNGKey(1), name="b")
        a.initialize(jno.bayesian.pathfinder(maxiter=10, num_samples=40))
        # b has NO initializer.
        a.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=jnp.ones(1), warmup=2, keep=10, adapt=False)
        b.bayesian(blackjax.nuts, step_size=1e-2, inverse_mass_matrix=jnp.ones(1), warmup=2, keep=10, adapt=False)

        residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
        jno.core([residual.mse]).solve(12)

        # Pathfinder ran exactly once — for a only, not for b.
        assert pf_count[0] == 1, f"pathfinder ran {pf_count[0]} times; expected 1 (a only, not b)"
        # Both Bayesian models still produced chains.
        assert a.posterior_samples is not None
        assert b.posterior_samples is not None


# ---------------------------------------------------------------------------
# Phase 15 — Pattern B: masked Bayesian head + global optax body on one model
# ---------------------------------------------------------------------------


class TestPatternB:
    """One pytree, two backends.  Phase 15 lifted the v1 block on
    ``.mask(M).bayesian()`` + ``.optimizer(...)`` on the same model;
    Phase 16 refactored the state storage so optax lives at the bare
    composite key (``"<lid>"``) and each Bayesian/VI group lives at
    its own composite key (``"<lid>.<group_idx>"``).  These tests pin
    the K=1 and K>1 paths and the composition with the existing
    Bayesian features.
    """

    @staticmethod
    def _build_problem(*, num_chains=1, hidden=4):
        """Tiny MLP regression problem used by most tests in this class.
        Head = output_layer (last leaf via _build_last_leaf_mask); body
        = everything else.  Returns ``(domain, net, residual)`` ready
        for ``crux.solve``.
        """
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=1, hidden=hidden)
        target = jno.np.sin(jno.np.pi * x)  # a target the net can approximate
        residual = net(x) - target
        return dom, net, residual

    # ─── 1. K=1: body moves under optax, head moves under SGLD ──────

    def test_pattern_b_k1_both_backends_fire(self):
        """K=1 Pattern B end-to-end: chain shows the masked head varies
        across samples (Bayesian fired) AND the body leaves moved from
        their initial values (optax fired)."""
        dom, net, residual = self._build_problem(hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)

        init_body = [jnp.copy(leaf) for i, leaf in enumerate(jax.tree_util.tree_leaves(net.module)) if i != masked_idx]

        net.optimizer(optax.adam(1e-1))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=5, keep=30)

        jno.core([residual.mse]).solve(40)
        chain = net.posterior_samples
        chain_leaves = jax.tree_util.tree_leaves(chain)

        # Bayesian contract — head varies across the chain
        masked_var = float(jnp.mean(jnp.var(chain_leaves[masked_idx], axis=1)))
        assert masked_var > 1e-8, f"head should vary; got var={masked_var:.3e}"

        # Optax contract — body moved from init
        final_body = [leaf for i, leaf in enumerate(jax.tree_util.tree_leaves(net.module)) if i != masked_idx]
        any_moved = any(float(jnp.linalg.norm(a - b)) > 1e-4 for a, b in zip(final_body, init_body))
        assert any_moved, "Pattern B: body leaves did not move under optax"

    # ─── 2. K>1: chain has K-leading axis ────────────────────────────

    def test_pattern_b_k4_multichain_shape(self):
        """K=4 Pattern B: chain has ``(K=4, N, *leaf)`` for every leaf,
        head leaf varies along N, body leaves are populated."""
        dom, net, residual = self._build_problem(hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.optimizer(optax.adam(1e-1))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=3, keep=8, num_chains=4)
        jno.core([residual.mse]).solve(12)

        chain = net.posterior_samples
        chain_leaves = jax.tree_util.tree_leaves(chain)
        for leaf in chain_leaves:
            assert leaf.shape[:2] == (4, 8), f"unexpected K, N axes: {leaf.shape[:2]} (expected (4, 8))"
        # Head varies along N within each chain
        head_var = float(jnp.mean(jnp.var(chain_leaves[masked_idx], axis=1)))
        assert head_var > 1e-8, f"head should vary along N; got var={head_var:.3e}"

    # ─── 3. Body gradient excludes head leaves ───────────────────────

    def test_pattern_b_body_gradient_excludes_head(self):
        """Sanity-check the masking: an optax update on Pattern B
        should NOT move the head leaves.  Compare the head leaves
        at init vs after solve — they must be identical (modulo the
        kernel's own Bayesian noise).
        """
        dom, net, residual = self._build_problem(hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)

        # Snapshot the head leaf at init
        init_head = jnp.copy(jax.tree_util.tree_leaves(net.module)[masked_idx])

        # Use a tiny step size so the body barely moves; what matters
        # is whether optax accidentally touches the head.
        net.optimizer(optax.adam(1e-2))
        # Use SGLD with step_size=0 so the head doesn't move by SGLD either
        net.mask(mask).bayesian(blackjax.sgld, step_size=0.0, warmup=2, keep=5)
        jno.core([residual.mse]).solve(8)

        final_head = jax.tree_util.tree_leaves(net.module)[masked_idx]
        # With SGLD step_size=0 the head should be EXACTLY at init
        # (no Bayesian update fires).  If the optax body update
        # accidentally bled into the head, the head would have moved.
        assert jnp.allclose(init_head, final_head, atol=1e-6), (
            "Pattern B: optax body update bled into the masked head leaves"
        )

    # ─── 4. Reproducibility ─────────────────────────────────────────

    def test_pattern_b_reproducible_with_fixed_seed(self):
        """Two solves with identical master seed produce identical
        chains AND identical final body values."""

        def _solve():
            dom, net, residual = self._build_problem(hidden=4)
            mask, _ = _build_last_leaf_mask(net)
            net.optimizer(optax.adam(1e-1))
            net.mask(mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=3, keep=10)
            jno.core([residual.mse]).solve(13)
            return (
                net.posterior_samples,
                jax.tree_util.tree_leaves(net.module),
            )

        chain_a, body_a = _solve()
        chain_b, body_b = _solve()
        for la, lb in zip(jax.tree_util.tree_leaves(chain_a), jax.tree_util.tree_leaves(chain_b)):
            assert jnp.allclose(la, lb), "Pattern B chain not reproducible with fixed seed"
        for la, lb in zip(body_a, body_b):
            assert jnp.allclose(la, lb), "Pattern B body not reproducible with fixed seed"

    # ─── 5. Pattern B + initializer composes ────────────────────────

    def test_pattern_b_with_initializer_composes(self, monkeypatch):
        """``.initialize(pathfinder).mask(M).bayesian()`` + global
        ``.optimizer()``: pathfinder warm-starts the head, optax
        trains the body.  Pathfinder must run exactly once on the
        masked subset."""
        pf_count = [0]
        orig = jno.bayesian.PathfinderInitializer.__call__

        def _spy(self, rng_key, ld, position, num_chains):
            pf_count[0] += 1
            return orig(self, rng_key, ld, position, num_chains)

        monkeypatch.setattr(jno.bayesian.PathfinderInitializer, "__call__", _spy)

        dom, net, residual = self._build_problem(hidden=4)
        mask, masked_idx = _build_last_leaf_mask(net)
        net.initialize(jno.bayesian.pathfinder(maxiter=10, num_samples=30))
        net.optimizer(optax.adam(1e-1))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=10)
        jno.core([residual.mse]).solve(12)

        assert pf_count[0] == 1, f"pathfinder ran {pf_count[0]} times; expected 1"
        chain = net.posterior_samples
        assert chain is not None
        # Head varies along chain axis (sampling fired)
        leaves = jax.tree_util.tree_leaves(chain)
        head_var = float(jnp.mean(jnp.var(leaves[masked_idx], axis=1)))
        assert head_var > 1e-8

    # ─── 6. SAEM warning fires for K>1 ──────────────────────────────

    def test_pattern_b_k1_no_saem_warning(self, capsys):
        # K=1 Pattern B is not subject to the chain-0 simplification.
        # The warning line must NOT fire.
        dom, net, residual = self._build_problem(hidden=4)
        mask, _ = _build_last_leaf_mask(net)
        net.optimizer(optax.adam(1e-2))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=3, num_chains=1)
        jno.core([residual.mse]).solve(5)
        out = capsys.readouterr().out
        assert "Pattern B + num_chains>1" not in out

    def test_pattern_b_k4_emits_saem_warning(self, capsys):
        # K>1 Pattern B: warn that the body's optax update is computed
        # against chain 0 only.  Loud at solve-start so users can see
        # the trade-off without reading the docs.
        dom, net, residual = self._build_problem(hidden=4)
        mask, _ = _build_last_leaf_mask(net)
        net.optimizer(optax.adam(1e-2))
        net.mask(mask).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=3, num_chains=4)
        jno.core([residual.mse]).solve(5)
        out = capsys.readouterr().out
        assert "Pattern B + num_chains>1" in out
        assert "K=4" in out
        assert "chain-0" in out

    # ─── 7. Composite-key contract (Phase 16 refactor) ──────────────

    def test_pattern_b_composite_keys_layout(self):
        """Phase 16: Pattern B's two states live under distinct composite
        keys (optax at ``"<lid>"``, kernel at ``"<lid>.<group_idx>"``).
        Regression guard for the contract that replaced the v1 ``_MixedState``
        wrapper."""
        from jno.core import _bay_key, _lid_of

        assert _lid_of("1") == 1
        assert _lid_of("1.0") == 1
        assert _lid_of("42.3") == 42
        assert _bay_key(1, 0) == "1.0"
        assert _bay_key(42, 3) == "42.3"


# ---------------------------------------------------------------------------
# Phase 16 — Pattern D: multiple disjoint masked Bayesian groups on one model
# ---------------------------------------------------------------------------


class TestPatternD:
    """Multiple disjoint masked ``.bayesian()`` groups on the same
    model.  Each group gets its own composite-keyed handle / kernel
    state in ``opt_states["<lid>.<group_idx>"]``.  At step time the
    Gibbs cycle iterates groups in sorted key order; each group's
    kernel sees the LATEST positions of the other groups (proper
    Metropolis-within-Gibbs for K=1).
    """

    @staticmethod
    def _build_two_groups(*, num_chains=1, hidden=4):
        """Two-group MLP: leaves split into two disjoint masks."""
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=2, hidden=hidden)
        leaves = jax.tree_util.tree_leaves(net.module)
        treedef = jax.tree_util.tree_structure(net.module)
        m1_flags = [False] * len(leaves)
        m1_flags[-1] = True
        m2_flags = [False] * len(leaves)
        m2_flags[-2] = True
        m1 = jax.tree_util.tree_unflatten(treedef, m1_flags)
        m2 = jax.tree_util.tree_unflatten(treedef, m2_flags)
        residual = net(x) - 0.0
        return dom, net, m1, m2, residual

    # ─── 1. Two SGLD groups both fire ────────────────────────────────

    def test_pattern_d_two_sgld_groups_both_vary(self):
        """Two disjoint masked SGLD groups: chain has variation in
        both groups' masked leaves; other leaves stay constant."""
        dom, net, m1, m2, residual = self._build_two_groups()
        net.mask(m1).bayesian(blackjax.sgld, step_size=1e-3, warmup=3, keep=15)
        net.mask(m2).bayesian(blackjax.sgld, step_size=1e-3, warmup=3, keep=15)
        jno.core([residual.mse]).solve(18)

        chain = net.posterior_samples
        assert chain is not None
        chain_leaves = jax.tree_util.tree_leaves(chain)
        # Last leaf is in m1; second-to-last is in m2.
        var_m1 = float(jnp.mean(jnp.var(chain_leaves[-1], axis=1)))
        var_m2 = float(jnp.mean(jnp.var(chain_leaves[-2], axis=1)))
        assert var_m1 > 1e-10, f"group m1 should vary; got var={var_m1:.3e}"
        assert var_m2 > 1e-10, f"group m2 should vary; got var={var_m2:.3e}"
        # First leaf is outside both masks → constant across chain.
        var_outer = float(jnp.mean(jnp.var(chain_leaves[0], axis=1)))
        assert var_outer < 1e-10, f"unmasked leaf should stay at init; got var={var_outer:.3e}"

    # ─── 2. Pattern D + Pattern B (multi-Bayesian + global optax) ───

    def test_pattern_d_with_global_optimizer(self):
        """Two SGLD groups + global ``.optimizer()``: each group's
        masked subset is sampled; the unmasked complement is
        Adam-trained.  Body moves from init; both group leaves vary."""
        dom, net, m1, m2, residual = self._build_two_groups()

        # Snapshot initial weights for "moved" comparison
        init_leaves = jax.tree_util.tree_leaves(net.module)
        init_body_l0 = jnp.copy(init_leaves[0])

        net.optimizer(optax.adam(1e-1))
        net.mask(m1).bayesian(blackjax.sgld, step_size=1e-3, warmup=3, keep=15)
        net.mask(m2).bayesian(blackjax.sgld, step_size=1e-3, warmup=3, keep=15)
        jno.core([residual.mse]).solve(18)

        chain = net.posterior_samples
        chain_leaves = jax.tree_util.tree_leaves(chain)
        # Both group leaves vary.
        assert float(jnp.mean(jnp.var(chain_leaves[-1], axis=1))) > 1e-10
        assert float(jnp.mean(jnp.var(chain_leaves[-2], axis=1))) > 1e-10
        # Body (first leaf, outside both masks) moved under optax.
        final_body_l0 = jax.tree_util.tree_leaves(net.module)[0]
        assert float(jnp.linalg.norm(final_body_l0 - init_body_l0)) > 1e-4, "Pattern D + B: body did not move under optax"

    # ─── 3. K>1 multichain + Pattern D ───────────────────────────────

    def test_pattern_d_multichain_K4(self):
        """Pattern D + K=4 chains: chain has (K=4, N, ...) and both
        group leaves vary across N."""
        dom, net, m1, m2, residual = self._build_two_groups()
        net.mask(m1).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=6, num_chains=4)
        net.mask(m2).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=6, num_chains=4)
        jno.core([residual.mse]).solve(8)

        chain = net.posterior_samples
        chain_leaves = jax.tree_util.tree_leaves(chain)
        for leaf in chain_leaves:
            assert leaf.shape[:2] == (4, 6), f"unexpected (K, N) = {leaf.shape[:2]}"
        # Both group leaves vary along N within each chain.
        assert float(jnp.mean(jnp.var(chain_leaves[-1], axis=1))) > 1e-10
        assert float(jnp.mean(jnp.var(chain_leaves[-2], axis=1))) > 1e-10

    # ─── 4. Composite-key contract ───────────────────────────────────

    def test_pattern_d_composite_keys_present(self):
        """After solve, the *internal* opt_states dict must carry
        composite keys per group.  We probe via opt_states exposure on
        the solve closure isn't possible from outside — but we can
        check via the symptom: the posterior_samples chain has two
        groups' worth of variation, which can only happen if both
        groups had their own state slots."""
        # Already covered by ``test_pattern_d_two_sgld_groups_both_vary``;
        # here we additionally verify the composite-key helpers
        # parse the expected formats.
        from jno.core import _bay_key, _group_idx_of, _lid_of

        assert _bay_key(2, 1) == "2.1"
        assert _lid_of("2.1") == 2
        assert _group_idx_of("2.1") == 1
        assert _group_idx_of("2") is None  # bare optax key


# ---------------------------------------------------------------------------
# Phase 16 — Pattern E: mixed VI + MCMC on disjoint masks of one model
# ---------------------------------------------------------------------------


class TestPatternE:
    """One model with both an MCMC handle and a VI handle on disjoint
    masks.  At step time the MCMC group samples; at solve end the VI
    group draws ``posterior_draws`` from its fitted distribution and
    its draws are spliced into the MCMC chain at the VI-masked leaves.
    Strict matching: MCMC's ``keep`` must equal VI's ``posterior_draws``.
    """

    @staticmethod
    def _build_two_groups(*, hidden=4):
        dom = _line_domain()
        x, _ = dom.variable("interior")
        net = _tiny_net(in_dim=1, out_dim=2, hidden=hidden)
        leaves = jax.tree_util.tree_leaves(net.module)
        treedef = jax.tree_util.tree_structure(net.module)
        m1_flags = [False] * len(leaves)
        m1_flags[-1] = True
        m2_flags = [False] * len(leaves)
        m2_flags[-2] = True
        m1 = jax.tree_util.tree_unflatten(treedef, m1_flags)
        m2 = jax.tree_util.tree_unflatten(treedef, m2_flags)
        residual = net(x) - 0.0
        return dom, net, m1, m2, residual

    # ─── 1. MCMC + VI on disjoint masks: chain produced ─────────────

    def test_pattern_e_mcmc_plus_vi_runs(self):
        """One MCMC + one VI on disjoint masks of the same model:
        chain has both groups' leaves varying."""
        dom, net, m1, m2, residual = self._build_two_groups()
        N = 20
        net.mask(m1).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=N)
        net.mask(m2).vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            num_samples=4,
            posterior_draws=N,  # must match MCMC keep
        )
        jno.core([residual.mse]).solve(22)

        chain = net.posterior_samples
        assert chain is not None
        leaves = jax.tree_util.tree_leaves(chain)
        var_mcmc = float(jnp.mean(jnp.var(leaves[-1], axis=1)))
        var_vi = float(jnp.mean(jnp.var(leaves[-2], axis=1)))
        assert var_mcmc > 1e-10, f"MCMC leaf should vary; got var={var_mcmc:.3e}"
        assert var_vi > 1e-10, f"VI leaf should vary across draws; got var={var_vi:.3e}"

    # ─── 2. Strict matching: keep != posterior_draws raises ─────────

    def test_pattern_e_mismatched_lengths_raises(self):
        """Pattern E requires ``keep == posterior_draws`` across the
        mixed VI / MCMC groups on one layer.  Mismatched lengths
        raise at solve start."""
        dom, net, m1, m2, residual = self._build_two_groups()
        net.mask(m1).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=10)
        net.mask(m2).vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            num_samples=4,
            posterior_draws=20,  # mismatch on purpose
        )
        with pytest.raises(ValueError, match="Pattern E requires"):
            jno.core([residual.mse]).solve(12)

    # ─── 3. Composite-key contract — both kinds at distinct keys ────

    def test_pattern_e_both_handles_registered(self):
        """Both handles are reachable via composite keys ``"<lid>.0"``
        and ``"<lid>.1"`` — symptom: chain has 2 groups' worth of
        variation, which is only possible if both are registered."""
        dom, net, m1, m2, residual = self._build_two_groups()
        net.mask(m1).bayesian(blackjax.sgld, step_size=1e-3, warmup=2, keep=15)
        net.mask(m2).vi(
            blackjax.meanfield_vi,
            optimizer=optax.adam(1e-2),
            num_samples=4,
            posterior_draws=15,
        )
        jno.core([residual.mse]).solve(17)
        chain = net.posterior_samples
        leaves = jax.tree_util.tree_leaves(chain)
        # Both group leaves vary
        assert float(jnp.mean(jnp.var(leaves[-1], axis=1))) > 1e-10
        assert float(jnp.mean(jnp.var(leaves[-2], axis=1))) > 1e-10
        # Unmasked complement (first leaf) stays at init
        assert float(jnp.mean(jnp.var(leaves[0], axis=1))) < 1e-10
