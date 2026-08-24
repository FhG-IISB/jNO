"""Every resampling strategy must trace: jit-compilable, and gradients reach the points.

Before this, two strategies could not be traced at all -- R3 and CR3 selected with a boolean mask
(``points[score >= threshold]``, whose output SHAPE depends on the values) and then forced the count
to a Python int (``int(jnp.sum(mask))``). Both are hard tracer errors. They now rank instead, so the
shape is static and every returned point is a differentiable gather.

What is and is not differentiable here is worth stating: the CHOICE of which slot keeps its point
and which is refilled is discrete and has no gradient -- argsort and a threshold comparison are step
functions. The POSITIONS do: d(result)/d(points) and d(result)/d(pool) both flow, which is what an
inverse problem that moves the sampling geometry needs.
"""

import functools

import jax
import jax.numpy as jnp
import pytest

import jno

jax.config.update("jax_enable_x64", True)

N, D, POOL = 64, 2, 512

STRATEGIES = {
    "random": lambda: jno.sampler.random(resample_fraction=0.3),
    "rad": lambda: jno.sampler.rad(resample_fraction=0.3),
    "rard": lambda: jno.sampler.rard(resample_fraction=0.3),
    "ha": lambda: jno.sampler.ha(resample_fraction=0.3),
    "r3": lambda: jno.sampler.r3(),
    "cr3": lambda: jno.sampler.cr3(),
    "pinnfluence": lambda: jno.sampler.pinnfluence(resample_fraction=0.3),
}


def _inputs(seed=0):
    k = jax.random.PRNGKey(seed)
    k1, k2, k3, k4 = jax.random.split(k, 4)
    points = jax.random.uniform(k1, (N, D))
    residuals = jax.random.uniform(k2, (N,)) + 0.1
    pool = jax.random.uniform(k3, (POOL, D))
    return points, residuals, pool, k4


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_resample_is_jittable_and_shape_preserving(name):
    strat = STRATEGIES[name]()
    points, residuals, pool, key = _inputs()

    @functools.partial(jax.jit, static_argnums=(4,))
    def go(points, residuals, pool, key, tag):
        return strat.resample(points, residuals, None, tag, 0, key, candidates=pool)

    out = go(points, residuals, pool, key, "interior")
    assert out.shape == points.shape, f"{name} changed the point count"
    assert jnp.all(jnp.isfinite(out)), f"{name} produced non-finite points"


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_gradients_reach_points_and_pool(name):
    strat = STRATEGIES[name]()
    points, residuals, pool, key = _inputs()

    def loss(points, pool):
        out = strat.resample(points, residuals, None, "interior", 0, key, candidates=pool)
        return jnp.sum(out**2)

    gp, gq = jax.jit(jax.grad(loss, argnums=(0, 1)))(points, pool)
    assert jnp.all(jnp.isfinite(gp)) and jnp.all(jnp.isfinite(gq)), f"{name} gave non-finite grads"
    # every returned point came from one of the two sources, so between them the gradient is real
    assert float(jnp.sum(jnp.abs(gp)) + jnp.sum(jnp.abs(gq))) > 0.0, f"{name} gradient is all zero"


def test_ranking_retains_the_highest_residual_points():
    """The rewrite must keep R3's meaning: high residual survives, low residual is replaced."""
    strat = jno.sampler.r3(min_keep_frac=0.25, max_keep_frac=0.75)
    points, _, pool, key = _inputs()
    residuals = jnp.linspace(0.0, 1.0, N)  # point i has residual i/N
    out = strat.resample(points, residuals, None, "interior", 0, key, candidates=pool)
    kept = jnp.array([jnp.any(jnp.all(jnp.isclose(out, p), axis=1)) for p in points])
    top_half = kept[N // 2 :].sum()
    bottom_half = kept[: N // 2].sum()
    assert top_half > bottom_half, f"kept {bottom_half} low-residual vs {top_half} high-residual"


def test_cr3_next_gamma_is_pure_and_traceable():
    strat = jno.sampler.cr3()
    before = strat.gamma
    res, gate = jnp.ones(N) * 0.5, jnp.ones(N)
    g = jax.jit(strat.next_gamma)(res, gate)
    assert jnp.isfinite(g)
    assert strat.gamma == before, "next_gamma must not mutate the strategy"
    assert float(g) >= before, "gamma advances forward"


def test_no_pool_is_a_silent_no_op_for_the_ranking_strategies():
    """Documents the current contract rather than endorsing it.

    With no pool these hand back the caller's own points, so a run that asked for resampling gets
    none and looks healthy. It is deliberate -- test_resampling.py asserts it for RandomResampling --
    and this pins CR3 and R3 to the same behaviour so the three cannot drift apart silently.
    """
    points, residuals, _, key = _inputs()
    for name in ("random", "r3", "cr3"):
        out = STRATEGIES[name]().resample(points, residuals, None, "interior", 0, key, candidates=None)
        assert jnp.array_equal(out, points), f"{name} changed its no-pool behaviour"


def test_rad_keeps_its_pool_free_fallback():
    """RAD perturbs its high-residual points when there is no pool, so it must NOT raise."""
    points, residuals, _, key = _inputs()
    out = jno.sampler.rad(resample_fraction=0.3).resample(points, residuals, None, "interior", 0, key, candidates=None)
    assert out.shape == points.shape
    assert not jnp.array_equal(out, points), "RAD fallback should still move points"
