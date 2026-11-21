"""``jno.optimizers`` — the optax-compatible optimizer namespace (optax re-export + custom
second-order methods ssbroyden / ssbfgs / soap). Each custom optimizer is an optax
``GradientTransformation``, so it converges in a plain optax loop and composes with ``optax.chain``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

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
    # exactly the custom optimizers. Most are optax GradientTransformations; `engd`, `md`, `mma`,
    # `gcmma`, `bounded` and the residual optimizers (`lm`, `dngd`, `nystrom_ngd`, `dsgnar`) are
    # sentinels that jno.core.solve() expands, because each needs
    # information the optax update() contract does not carry. `simp_continuation` is neither -- it is a schedule over a
    # parameter that is not a design variable, and it lives here because it drives the optimisation
    # rather than diagnosing it.
    assert set(o.__all__) == {
        "ssbroyden",
        "ssbfgs",
        "scale_by_ss_quasi_newton",
        "soap",
        "scale_by_soap",
        "ss_esoap",
        "scale_by_ss_esoap",
        "engd",
        "ENGDOptimizer",
        "lm",
        "LMOptimizer",
        "dngd",
        "DNGDOptimizer",
        "nystrom_ngd",
        "NystromNGDOptimizer",
        "dsgnar",
        "DSGNAROptimizer",
        "nncg",
        "score",
        "md",
        "md_decouple",
        "MDOptimizer",
        "mma",
        "MMAOptimizer",
        "gcmma",
        "GCMMAOptimizer",
        "bounded",
        "BoundedOptimizer",
        "simp_continuation",
        "SIMPContinuation",
        "geometric_decay",
        "GeometricDecay",
        "heaviside_continuation",
        "HeavisideContinuation",
    }
    for factory in (o.ssbroyden, o.ssbfgs, o.soap, o.nncg, o.score, o.ss_esoap):
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


def test_nncg_converges():
    # a damped Newton step on a quadratic: the error shrinks by mu / (2 + mu) per step
    opt, x = jno.optimizers.nncg(), jnp.zeros(3)
    state = opt.init(x)
    for _ in range(10):  # nncg keeps no value in its state, so plain value_and_grad, not optax's helper
        value, grad = jax.value_and_grad(_loss)(x)
        updates, state = opt.update(grad, state, x, value=value, value_fn=_loss)
        x = optax.apply_updates(x, updates)
    assert np.allclose(np.asarray(x), np.asarray(TARGET), atol=1e-10)


def test_soap_converges():
    x = _minimise_grad(jno.optimizers.soap(learning_rate=0.2), steps=400)
    assert np.allclose(x, np.asarray(TARGET), atol=1e-2)


def test_works_inside_optax_chain():
    # a custom optimizer composed via optax directly (optax.chain + optax clipping)
    opt = optax.chain(optax.clip_by_global_norm(10.0), jno.optimizers.ssbfgs())
    x = _minimise_linesearch(opt, steps=30)
    assert np.allclose(x, np.asarray(TARGET), atol=1e-2)


# ---------------------------------------------------------------------------------------------
# ssbroyden(pairs=m) / ssbfgs(pairs=m): the metric over the last m pairs, without the dense matrix
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _smooth_loss(x):
    """Not quadratic, with a spread of curvatures: its Hessian changes along the path."""
    scales = jnp.logspace(0.0, 3.0, x.shape[0], dtype=x.dtype)
    return jnp.sum(scales * (x - 1.0) ** 2) + jnp.sum((x[1:] - x[:-1] ** 2) ** 2)


def _run_linesearch(opt, loss, x0, steps):
    """Plain optax loop; returns the iterates and the final state."""

    @jax.jit
    def step(x, state):
        value, grad = jax.value_and_grad(loss)(x)
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=loss)
        return optax.apply_updates(x, updates), state

    x, state, xs = x0, opt.init(x0), []
    for _ in range(steps):
        x, state = step(x, state)
        xs.append(np.asarray(x))
    return np.stack(xs), state


def _dense_metric(metric):
    """``H = c (I - Q^T Q) + Q^T H_Q Q`` of a limited-memory state as a matrix, and ``Q``."""
    Q = np.asarray(metric.basis)[: int(metric.rank)]
    c = float(metric.scale)
    H_Q = np.asarray(metric.hessian_inv)[: len(Q), : len(Q)]
    return c * (np.eye(Q.shape[1]) - Q.T @ Q) + Q.T @ H_Q @ Q, Q


@pytest.mark.parametrize("factory", ["ssbroyden", "ssbfgs"])
def test_pairs_matches_dense_while_the_memory_is_not_full(x64, factory):
    # the dense update on the span of the pairs seen so far: the same iterates as the dense solver
    make = getattr(jno.optimizers, factory)
    x0 = jnp.zeros(12)
    dense, _ = _run_linesearch(make(), _smooth_loss, x0, 30)
    limited, state = _run_linesearch(make(pairs=64), _smooth_loss, x0, 30)
    assert int(state.rejected) == 0
    np.testing.assert_allclose(limited, dense, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("factory", ["ssbroyden", "ssbfgs"])
def test_pairs_metric_stays_positive_definite_on_an_orthonormal_basis(x64, factory):
    m, n = 8, 30
    _, state = _run_linesearch(getattr(jno.optimizers, factory)(pairs=m), _smooth_loss, jnp.zeros(n), 60)
    metric = state.inner[0]
    assert int(state.rejected) == 0
    assert m - 2 <= int(metric.rank) + 2 <= m  # the memory filled up and was projected
    H, Q = _dense_metric(metric)
    np.testing.assert_allclose(Q @ Q.T, np.eye(len(Q)), atol=1e-12)
    assert np.linalg.eigvalsh(0.5 * (H + H.T)).min() > 0


def test_projection_keeps_the_directions_of_highest_curvature(x64):
    from jno.optimizers.ssbroyden import _limited_project

    rng = np.random.default_rng(0)
    m, n, rank, keep, c = 6, 20, 5, 3, 0.5
    Q = np.zeros((m, n))
    Q[:rank] = np.linalg.qr(rng.standard_normal((n, rank)))[0].T
    Q[rank:] = rng.standard_normal((m - rank, n))  # an unused row holds anything; it is not read
    eig = np.array([1.1, 1 / 400.0, 30.0, 1 / 1.2, 9.0])  # of the inverse metric: small = high curvature
    R = np.linalg.qr(rng.standard_normal((rank, rank)))[0]
    H_Q = np.zeros((m, m))
    H_Q[:rank, :rank] = R @ np.diag(eig) @ R.T
    H_old = c * (np.eye(n) - Q[:rank].T @ Q[:rank]) + Q[:rank].T @ H_Q[:rank, :rank] @ Q[:rank]

    new_Q, new_H_Q, new_c = _limited_project(jnp.asarray(Q), jnp.asarray(H_Q), jnp.asarray(c), jnp.asarray(rank), keep)
    new_Q, new_H_Q, new_c = np.asarray(new_Q)[:keep], np.asarray(new_H_Q)[:keep, :keep], float(new_c)
    np.testing.assert_allclose(new_Q @ new_Q.T, np.eye(keep), atol=1e-12)
    np.testing.assert_allclose(np.sort(np.diag(new_H_Q)), np.sort(eig[[1, 3, 0]]), rtol=1e-10)
    np.testing.assert_allclose(new_c, np.sqrt(30.0 * 9.0), rtol=1e-10)  # geometric mean of the dropped
    H_new = new_c * (np.eye(n) - new_Q.T @ new_Q) + new_Q.T @ new_H_Q @ new_Q
    np.testing.assert_allclose(H_new @ new_Q.T, H_old @ new_Q.T, atol=1e-10)  # unchanged on what is kept
    dropped = (R[:, [2, 4]].T @ Q[:rank]).T  # the eigenvectors of 30 and 9
    np.testing.assert_allclose(H_new @ dropped, new_c * dropped, atol=1e-10)


def test_pairs_state_holds_no_dense_matrix(x64):
    m, n = 5, 4000
    state = jno.optimizers.ssbroyden(pairs=m).init(jnp.zeros(n))
    sizes = [np.size(leaf) for leaf in jax.tree.leaves(state)]
    assert max(sizes) == m * n  # the basis; nothing of size n * n
    assert sum(sizes) < m * n + 10 * n


def test_pairs_converges_with_less_memory_than_parameters(x64):
    n = 60
    xs, state = _run_linesearch(jno.optimizers.ssbroyden(pairs=10), _smooth_loss, jnp.zeros(n), 400)
    grad_norm = float(jnp.linalg.norm(jax.grad(_smooth_loss)(jnp.asarray(xs[-1]))))
    assert grad_norm < 1e-6 * float(jnp.linalg.norm(jax.grad(_smooth_loss)(jnp.zeros(n))))


def test_pairs_works_with_the_optax_cached_gradient_loop(x64):
    # optax.value_and_grad_from_state reads the value and gradient the line search cached
    opt = jno.optimizers.ssbfgs(pairs=4)
    value_and_grad = optax.value_and_grad_from_state(_loss)
    x = jnp.zeros(3)
    state = opt.init(x)
    for _ in range(30):
        value, grad = value_and_grad(x, state=state)
        updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=_loss)
        x = optax.apply_updates(x, updates)
    assert np.allclose(np.asarray(x), np.asarray(TARGET), atol=1e-8)


def test_step_check_rejects_a_step_that_raises_the_loss(x64):
    from jno.optimizers.ssbroyden import _reject_loss_increase, scale_by_ss_quasi_newton

    def loss(x):
        return jnp.sum(x**2)

    # no line search and a step 1000 times too long: every step overshoots
    inner = optax.chain(scale_by_ss_quasi_newton("ssbroyden", pairs=4), optax.scale(-1e3))
    opt = _reject_loss_increase(inner)
    x = jnp.ones(3)
    state = opt.init(x)
    value, grad = jax.value_and_grad(loss)(x)
    updates, state = opt.update(grad, state, x, value=value, grad=grad, value_fn=loss)
    assert np.all(np.asarray(updates) == 0.0)  # the parameters stay where they are
    assert int(state.rejected) == 1
    metric = state.inner[0]
    assert int(metric.rank) == 0 and not np.any(np.asarray(metric.hessian_inv))
    assert np.isclose(float(metric.scale), min(1.0, 1.0 / float(jnp.linalg.norm(grad))))


@pytest.mark.parametrize("bad", [0, 3, -3, 2.5, True, "10"])
def test_pairs_must_be_an_integer_of_at_least_four(bad):
    with pytest.raises(ValueError, match="pairs must be None"):
        jno.optimizers.ssbroyden(pairs=bad)
