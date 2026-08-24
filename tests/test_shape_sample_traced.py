"""Sampling a shape inside jit, with no candidate pool and O(n) memory.

The pool route costs O(pool) memory and, once traced, freezes: a jitted strategy with
candidates=None reselects from one baked-in cloud forever (25 600 draws yielded 2559 distinct
points against a pool of 2560). `sample_inside` keeps only the answer -- rejection runs as a
while_loop over a fixed buffer -- so nothing is retained between calls and nothing is stale.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.geometry.shape import sample_inside


@pytest.fixture(autouse=True)
def _x64():
    """x64 for this module's tests only, restored after.

    At module scope this ran at IMPORT, so it leaked into every other file in the same pytest
    process and could not be undone; `tests/test_x64_isolation.py` guards against exactly that.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


S = jno.Shape
SHAPES = {
    "rect - disk": (S.rect(0, 0, 2, 1) - S.disk(0.7, 0.5, 0.28), 2),
    "polygon": (S.polygon([(0, 0), (2, 0), (1.6, 1), (0.5, 1.2)]), 2),
    "box - sphere": (S.box(0, 0, 0, 1, 1, 1) - S.sphere(0.5, 0.5, 0.5, 0.25), 3),
    "extrude": ((S.rect(0, 0, 1, 1) - S.disk(0.5, 0.5, 0.2)).extrude(0.6), 3),
}


def _bounds(shape, dim):
    lo, hi = shape.bounds()
    return np.array(lo[:dim], dtype=float), np.array(hi[:dim], dtype=float)


@pytest.mark.parametrize("name", list(SHAPES))
def test_samples_are_inside_and_the_draw_is_jitted(name):
    shape, dim = SHAPES[name]
    lo, hi = _bounds(shape, dim)
    n = 2000
    f = jax.jit(functools.partial(sample_inside, shape.contains, n=n))
    pts, filled = f(lo, hi, jax.random.PRNGKey(0))
    assert int(filled) == n, f"{name} only filled {int(filled)}/{n}"
    assert bool(jnp.all(jnp.asarray(shape.contains(pts), dtype=bool))), f"{name} returned outside points"


@pytest.mark.parametrize("name", list(SHAPES))
def test_every_call_is_a_fresh_draw(name):
    """The pool route saturates at the pool size; this must not saturate at all."""
    shape, dim = SHAPES[name]
    lo, hi = _bounds(shape, dim)
    f = jax.jit(functools.partial(sample_inside, shape.contains, n=256))
    seen = set()
    for i in range(20):
        pts, _ = f(lo, hi, jax.random.PRNGKey(i))
        seen.update(map(tuple, np.round(np.asarray(pts), 12)))
    assert len(seen) > 0.99 * 20 * 256, f"{name} repeated points across calls: {len(seen)}"


def test_gradients_flow_to_the_bounds():
    """lo + u*(hi-lo) is a reparametrisation, so an objective can move the sampling box."""
    shape, dim = SHAPES["rect - disk"]
    lo, hi = _bounds(shape, dim)

    def loss(lo, hi):
        pts, _ = sample_inside(shape.contains, lo, hi, jax.random.PRNGKey(0), 512)
        return jnp.sum(pts**2)

    glo, ghi = jax.jit(jax.grad(loss, argnums=(0, 1)))(lo, hi)
    assert jnp.all(jnp.isfinite(glo)) and jnp.all(jnp.isfinite(ghi))
    assert float(jnp.sum(jnp.abs(glo)) + jnp.sum(jnp.abs(ghi))) > 0.0


def test_memory_does_not_grow_with_rejection():
    """A thin sliver rejects most proposals; the compiled cost must still be O(n + batch).

    40 rounds at 2% acceptance is what it takes to fill 512 slots from batches of 1024. The point is
    that those rounds cost TIME and not MEMORY: the buffer is reused, so the working set is the same
    as for a shape that fills on the first round.
    """
    thin = S.rect(0, 0, 1, 1) & S.rect(0.49, 0, 0.51, 1)  # 2% of its bounding box
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    n, batch = 512, 1024
    f = jax.jit(functools.partial(sample_inside, thin.contains, n=n, batch=batch, max_rounds=40))
    live = f.lower(lo, hi, jax.random.PRNGKey(0)).compile().memory_analysis().temp_size_in_bytes
    assert live < 40 * (n + batch) * 8, f"temp memory {live} B is not O(n + batch)"
    pts, filled = f(lo, hi, jax.random.PRNGKey(0))
    assert int(filled) == n
    assert bool(jnp.all(jnp.asarray(thin.contains(pts), dtype=bool)))


def test_unfilled_is_reported_not_hidden():
    """An impossible region must report how many it managed, not silently hand back zeros."""
    empty = S.disk(0.5, 0.5, 0.05) & S.disk(5.0, 5.0, 0.05)  # disjoint -> empty
    lo, hi = np.array([0.0, 0.0]), np.array([6.0, 6.0])
    _, filled = jax.jit(functools.partial(sample_inside, empty.contains, n=100, max_rounds=4))(
        lo, hi, jax.random.PRNGKey(0)
    )
    assert int(filled) == 0
