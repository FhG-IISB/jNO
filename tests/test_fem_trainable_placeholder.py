"""Generic ``.trainable()`` placeholder promotion (``jno/trace``: ``Placeholder.trainable``).

``placeholder.trainable()`` mints a :func:`jno.np.parameter` of the **same shape and dtype** and seeds it
at the placeholder's current values, so an existing coefficient / data tag becomes a design variable in one
call and trains through ``jno.core`` exactly like a hand-written parameter.

Fail-loud scope: a **spatial coordinate** placeholder (``domain.variable(region)`` components) is a mesh
*geometry* design variable — routed through ``jno.fem`` (see ``plans/differentiable-r-adaptivity.md``
Feature 2) — and the **temporal** variable is not a design variable; the generic promotion raises for both
so a partial (coefficient-only) gradient can never be produced silently.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # float64 for the seed-exactness / solve asserts
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rect(size=0.5):
    return jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()


def _trivial():
    return jno.domain.from_array({"_": np.zeros((1, 1))})


def test_trainable_seeds_at_current_values():
    """The core contract: the promoted parameter starts at the placeholder's current value."""
    d = _rect()
    k = d.variable("kappa", sample=jnp.asarray([[[5.0]]]))
    kp = k.trainable(name="kp")
    crux = jno.core([kp], domain=_trivial())
    seed = np.asarray(crux.eval([kp])[0])
    assert seed.size == np.asarray(k.eval()).size, "promoted parameter size must match the placeholder"
    assert abs(float(seed.reshape(-1)[0]) - 5.0) < 1e-6, f"not seeded at current value: {seed.reshape(-1)}"


def test_trainable_preserves_dtype_float64():
    d = _rect()
    k = d.variable("k64", sample=jnp.asarray([[[3.0]]], dtype=jnp.float64))
    kp = k.trainable(name="k64p")
    crux = jno.core([kp], domain=_trivial())
    got = np.asarray(crux.eval([kp])[0])
    assert got.dtype == np.float64, f"dtype not preserved: {got.dtype}"
    assert abs(float(got.reshape(-1)[0]) - 3.0) < 1e-9


def test_trainable_coordinate_raises():
    """A spatial coordinate is a geometry design variable (Feature 2), not a plain promotion."""
    d = _rect()
    xi, yi, _ = d.variable("interior", split=True)
    with pytest.raises(NotImplementedError, match="geometry design variable"):
        xi.trainable()


def test_trainable_promoted_coefficient_is_recoverable():
    """Integration: a promoted coefficient is a genuine design variable — gradients reach it and an
    inverse solve recovers a known value, exactly as a hand-written ``jno.np.parameter`` would."""
    d = _rect(0.33)
    # attach a diffusion coefficient at the (wrong) initial guess 2.0, then promote it to trainable
    alpha = d.variable("alpha", sample=jnp.asarray([[[2.0]]])).trainable(name="alpha")
    alpha.optimizer(optax.adam(5e-2))

    def _poisson_with(coeff):
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
        return jno.fem([coeff * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    fem_inv = _poisson_with(alpha)
    a1, b1 = fem_inv.operator.evaluate({"alpha": 1.0})  # observation at the TRUE coefficient 1.0
    u_obs = jnp.linalg.solve(a1.todense(), jnp.asarray(b1).reshape(-1))

    crux = jno.core([(fem_inv.solve() - u_obs).mse], domain=_trivial())
    crux.solve(150)
    rec = float(np.asarray(crux.eval([alpha])[0]).reshape(-1)[0])
    assert abs(rec - 2.0) > 0.3, "promoted coefficient did not move — gradient did not reach it"
    assert abs(rec - 1.0) < 0.05, f"recovered alpha={rec:.4f} (want 1.0)"
