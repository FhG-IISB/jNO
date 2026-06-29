"""``jno.optimizers`` — the optax-compatible optimizer namespace (optax re-export + custom
second-order methods ssbroyden / ssbfgs / soap). Each custom optimizer is an optax
``GradientTransformation``, so it converges in a plain optax loop and composes with ``optax.chain``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

import jno

TARGET = jnp.array([3.0, -2.0, 0.5])


def _loss(x):
    return jnp.sum((x - TARGET) ** 2)


def _minimise_linesearch(opt, steps):
    """Loop for line-search optimizers (ssbroyden / ssbfgs): they consume the loss value + grad."""
    value_and_grad = optax.value_and_grad_from_state(_loss)
    x = jnp.zeros(3)
    state = opt.init(x)
    for _ in range(steps):
        value, grad = value_and_grad(x, state=state)
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=_loss)
        x = optax.apply_updates(x, updates)
    return np.asarray(x)


def _minimise_grad(opt, steps):
    """Loop for plain gradient optimizers (soap)."""
    gfn = jax.grad(_loss)
    x = jnp.zeros(3)
    state = opt.init(x)
    for _ in range(steps):
        updates, state = opt.update(gfn(x), state, x)
        x = optax.apply_updates(x, updates)
    return np.asarray(x)


def test_namespace_is_just_the_optimizers():
    o = jno.optimizers
    # exactly the custom optimizers — each an optax GradientTransformation
    assert set(o.__all__) == {"ssbroyden", "ssbfgs", "scale_by_ss_quasi_newton", "soap", "scale_by_soap", "engd", "ENGDOptimizer"}
    for factory in (o.ssbroyden, o.ssbfgs, o.soap):
        opt = factory()
        assert hasattr(opt, "init") and hasattr(opt, "update")
    # NOT a re-export of optax — chain/clipping/schedules live in optax, not here
    for n in ("chain", "adam", "clip_by_global_norm", "cosine_decay_schedule"):
        assert not hasattr(o, n), f"jno.optimizers should not re-export optax.{n}"


def test_ssbroyden_converges():
    x = _minimise_linesearch(jno.optimizers.ssbroyden(), steps=30)
    assert np.allclose(x, np.asarray(TARGET), atol=1e-3)


def test_ssbfgs_converges():
    x = _minimise_linesearch(jno.optimizers.ssbfgs(), steps=30)
    assert np.allclose(x, np.asarray(TARGET), atol=1e-3)


def test_soap_converges():
    x = _minimise_grad(jno.optimizers.soap(learning_rate=0.2), steps=400)
    assert np.allclose(x, np.asarray(TARGET), atol=1e-2)


def test_works_inside_optax_chain():
    # a custom optimizer composed via optax directly (optax.chain + optax clipping)
    opt = optax.chain(optax.clip_by_global_norm(10.0), jno.optimizers.ssbfgs())
    x = _minimise_linesearch(opt, steps=30)
    assert np.allclose(x, np.asarray(TARGET), atol=1e-2)
