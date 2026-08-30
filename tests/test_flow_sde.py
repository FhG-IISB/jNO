"""``Model.flow(noise=...)`` -- the march as an SDE.

The property that needs pinning hardest is the **per-step redraw**. A ``jno.noise`` node is a
realisation everywhere else in jNO, keyed once per evaluation; if the march reused one draw across
its steps the increments would be perfectly correlated, the endpoint variance would come out a
factor of the STEP COUNT too large, and it would get worse as the step count -- the accuracy knob --
is raised. Nothing would raise. So the tests below assert the variance against the analytic
Ito isometry and assert that it does *not* move with the step count.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


class ZeroDrift(eqx.Module):
    """dx = sigma dW exactly -- the only velocity whose SDE has a closed-form marginal."""

    def __call__(self, x, t):
        return jnp.zeros_like(x)


class LinearDrift(eqx.Module):
    a: jax.Array

    def __call__(self, x, t):
        return self.a * x


@pytest.fixture(scope="module")
def zero():
    return jno.nn(ZeroDrift())


KEY = jax.random.PRNGKey(0)
B = 20000


# ---------------------------------------------------------------------------
# The marginal, against the analytic answer
# ---------------------------------------------------------------------------


class TestMarginal:
    def test_zero_drift_reproduces_the_brownian_variance(self, zero):
        # dx = sigma dW on [0, 1]  =>  Var[x_1] = sigma^2 T, exactly.
        sigma = 0.5
        x = zero.flow(jnp.zeros((B, 1)), time=(0.0, 1.0, 21), noise=jno.noise.gaussian(std=sigma), key=KEY)
        assert abs(float(jnp.var(x)) - sigma**2) < 0.02, "endpoint variance is not sigma^2 * T"
        assert abs(float(jnp.mean(x))) < 0.02, "a zero-drift SDE must stay zero-mean"

    @pytest.mark.parametrize("steps", [10, 40, 80])
    def test_the_variance_does_not_grow_with_the_step_count(self, zero, steps):
        # THE regression. One draw reused across the march would give sigma^2 * T * steps, so this
        # assertion fails by 10x / 40x / 80x rather than subtly.
        sigma = 0.5
        x = zero.flow(jnp.zeros((B, 1)), time=(0.0, 1.0, steps + 1), noise=jno.noise.gaussian(std=sigma), key=KEY)
        assert abs(float(jnp.var(x)) - sigma**2) < 0.02, f"variance moved with steps={steps}"

    def test_a_schedule_integrates_as_the_discrete_ito_isometry(self, zero):
        # std = sigma*sqrt(t) => Var = sum_i sigma^2 t_i dt over the LEFT endpoints, which is what
        # Euler-Maruyama actually integrates (not the continuum sigma^2 T^2 / 2).
        sigma, steps = 0.5, 20
        ts = np.linspace(0.0, 1.0, steps + 1)
        expected = float(np.sum(sigma**2 * ts[:-1] * np.diff(ts)))
        x = zero.flow(
            jnp.zeros((B, 1)),
            time=(0.0, 1.0, steps + 1),
            noise=jno.noise.gaussian(std=lambda t: sigma * jnp.sqrt(t)),
            key=KEY,
        )
        assert abs(float(jnp.var(x)) - expected) < 0.01, f"schedule variance {float(jnp.var(x))} != {expected}"

    def test_increments_are_independent_across_steps(self, zero):
        # Directly: consecutive increments of a saved path must be uncorrelated. A reused draw makes
        # this correlation exactly 1.
        path = np.asarray(
            zero.flow(jnp.zeros((2000, 1)), time=(0.0, 1.0, 21), noise=jno.noise.gaussian(std=0.5), key=KEY, save=True)
        )
        d = np.diff(path[:, :, 0], axis=0)  # (steps, batch)
        c = np.corrcoef(d[0], d[1])[0, 1]
        assert abs(c) < 0.1, f"consecutive increments correlate at {c:.3f}; they must be independent"


# ---------------------------------------------------------------------------
# Composition with what flow already does
# ---------------------------------------------------------------------------


class TestComposition:
    def test_noise_none_is_bit_identical_to_the_plain_ode(self):
        v = jno.nn(LinearDrift(a=jnp.array(-0.5)))
        x0 = jax.random.normal(KEY, (64, 2))
        assert jnp.array_equal(v.flow(x0, time=(0.0, 1.0, 11)), v.flow(x0, time=(0.0, 1.0, 11), noise=None))

    def test_a_pytree_state_gets_an_independent_draw_per_leaf(self, zero):
        class PyZero(eqx.Module):
            def __call__(self, x, t):
                return {k: jnp.zeros_like(v) for k, v in x.items()}

        v = jno.nn(PyZero())
        out = v.flow(
            {"u": jnp.zeros((4000, 1)), "p": jnp.zeros((4000, 1))},
            time=(0.0, 1.0, 11),
            noise=jno.noise.gaussian(std=0.5),
            key=KEY,
        )
        c = np.corrcoef(np.asarray(out["u"])[:, 0], np.asarray(out["p"])[:, 0])[0, 1]
        assert abs(c) < 0.1, f"leaves share a draw (corr {c:.3f}); each must be independent"

    def test_a_backward_march_is_a_real_sde_not_a_nan(self, zero):
        # dt < 0: the increment scales with sqrt(|dt|). Taking sqrt of the signed step would NaN.
        x = zero.flow(jnp.zeros((4000, 1)), time=(1.0, 0.0, 21), noise=jno.noise.gaussian(std=0.5), key=KEY)
        assert bool(jnp.all(jnp.isfinite(x))), "backward SDE produced non-finite values"
        assert abs(float(jnp.var(x)) - 0.25) < 0.02

    def test_the_same_key_repeats_and_a_different_key_does_not(self, zero):
        a = zero.flow(
            jnp.zeros((256, 1)), time=(0.0, 1.0, 11), noise=jno.noise.gaussian(std=0.5), key=jax.random.PRNGKey(3)
        )
        b = zero.flow(
            jnp.zeros((256, 1)), time=(0.0, 1.0, 11), noise=jno.noise.gaussian(std=0.5), key=jax.random.PRNGKey(3)
        )
        c = zero.flow(
            jnp.zeros((256, 1)), time=(0.0, 1.0, 11), noise=jno.noise.gaussian(std=0.5), key=jax.random.PRNGKey(4)
        )
        assert jnp.array_equal(a, b), "same key must reproduce the march exactly"
        assert not jnp.allclose(a, c), "a different key must give a different march"

    def test_uniform_and_laplace_drivers_run(self, zero):
        for spec in (jno.noise.uniform(low=-0.5, high=0.5), jno.noise.laplace(std=0.4)):
            x = zero.flow(jnp.zeros((512, 1)), time=(0.0, 1.0, 11), noise=spec, key=KEY)
            assert bool(jnp.all(jnp.isfinite(x)))

    def test_the_cse_pass_does_not_drop_the_noise(self):
        # `cse` REBUILDS a Flow node when a child changes. If the rebuild forgot noise=/key=, the SDE
        # would silently become an ODE -- so assert the rebuilt node still carries them.
        from jno.trace import Flow, Placeholder, cse

        v = jno.nn(ZeroDrift())
        spec = jno.noise.gaussian(std=0.5)
        # The rebuild only fires when a CHILD is canonicalised away, so the Flow's x0 has to be a
        # genuine duplicate of something cse sees first. `BinaryOp` is keyed by child *identity*,
        # so `p + p` twice matches while `p * 2.0` twice does not (two distinct Literal children).
        par = jno.np.parameter((4, 1))
        first = par + par
        node = Flow(v, par + par, [], jnp.linspace(0.0, 1.0, 5), False, noise=spec, key=KEY)
        out = cse(first + node)

        found, seen = [], set()

        def walk(n):
            if id(n) in seen or not hasattr(n, "__dict__"):
                return
            seen.add(id(n))
            if isinstance(n, Flow):
                found.append(n)
            for value in vars(n).values():
                for item in value if isinstance(value, (list, tuple)) else [value]:
                    if isinstance(item, Placeholder):
                        walk(item)

        walk(out)
        assert found, "no Flow node survived cse"
        rebuilt = [f for f in found if f is not node]
        assert rebuilt, "cse did not rebuild the Flow, so this test would pass with the bug present"
        assert all(f.noise is not None and f.key is not None for f in rebuilt), "cse dropped noise=/key="


# ---------------------------------------------------------------------------
# Fail loud
# ---------------------------------------------------------------------------


class TestGuards:
    def test_eager_noise_without_a_key_raises(self, zero):
        # The trap this closes: the eager path builds its context with key=None, and a Noise node
        # evaluated with no key returns ZEROS. Silently a deterministic ODE labelled as an SDE.
        with pytest.raises(ValueError, match=r"needs key= on the eager path"):
            zero.flow(jnp.zeros((4, 1)), time=(0.0, 1.0, 3), noise=jno.noise.gaussian(std=0.1))

    def test_a_key_without_noise_raises(self, zero):
        with pytest.raises(ValueError, match=r"key=.*without noise="):
            zero.flow(jnp.zeros((4, 1)), time=(0.0, 1.0, 3), key=KEY)

    @pytest.mark.parametrize("spec", [{"ndim": 2}, {"ndim": 1}])
    def test_a_shape_argument_on_the_noise_spec_raises(self, zero, spec):
        # Shape comes from the state; ndim= would be ignored, so it is refused rather than dropped.
        with pytest.raises(ValueError, match=r"takes the DISTRIBUTION only"):
            zero.flow(jnp.zeros((4, 1)), time=(0.0, 1.0, 3), noise=jno.noise.gaussian(std=0.1, **spec), key=KEY)

    def test_a_like_reference_on_the_noise_spec_raises(self, zero):
        ref = jno.np.parameter((4, 1))
        with pytest.raises(ValueError, match=r"takes the DISTRIBUTION only"):
            zero.flow(jnp.zeros((4, 1)), time=(0.0, 1.0, 3), noise=jno.noise.gaussian(std=0.1, like=ref), key=KEY)

    def test_a_non_noise_object_raises(self, zero):
        with pytest.raises(TypeError, match=r"expects a jno.noise spec"):
            zero.flow(jnp.zeros((4, 1)), time=(0.0, 1.0, 3), noise=0.1, key=KEY)

    def test_a_schedule_outside_a_march_raises(self):
        # A callable std has no march time at an ordinary noise node, so it must raise rather than
        # be coerced or silently called with the wrong argument.
        import foundax
        import optax

        data = np.random.default_rng(0).normal(size=(32, 2)).astype(np.float32)
        dom = jno.domain.from_array({"data": data})
        d0, d1, _ = dom.variable("data")
        net = jno.nn(foundax.mlp(2, output_dim=1, key=KEY))
        net.optimizer(optax.adam(1e-3))
        n = jno.noise.gaussian(std=lambda t: t, like=d0)
        with pytest.raises(ValueError, match=r"was given a schedule"):
            jno.core([(net(d0, d1) + n).mse]).solve(epochs=2)

    def test_a_bad_parameter_type_names_the_two_accepted_forms(self):
        with pytest.raises(TypeError, match=r"a number or a schedule"):
            jno.noise.gaussian(std="loud")
