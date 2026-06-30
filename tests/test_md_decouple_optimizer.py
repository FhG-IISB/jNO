"""Tests for Magnitude–Direction Decoupling (``jno.optimizers.md`` / ``md_decouple``).

MD decoupling (Hägele et al. 2026, arXiv:2606.25971, Algorithm 2) reparameterizes each 2-D weight
matrix ``W = diag(γ_row) Ŵ diag(γ_col)`` with the direction ``Ŵ`` on a fixed-Frobenius-norm sphere
and learnable per-row/per-column gains, wrapping any optax base optimizer on the direction. These
tests cover the bare ``md_decouple`` transform in a plain optax loop (init identity, sphere
projection, bias pass-through, convergence, axis knobs, deferred-feature guards) and the ``md``
sentinel end-to-end through ``crux.solve`` (host LR-scale neutralization + idempotent re-solve).

Run: ``pixi run -e default pytest tests/test_md_decouple_optimizer.py``.
"""

from __future__ import annotations

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
import jno.jnp_ops as jnn
from jno.optimizers import MDOptimizer, md, md_decouple


@pytest.fixture(autouse=True)
def _x64():
    """Tight round-trip / projection assertions need float64. Save/restore so the flag never leaks
    into co-run modules (the x64 test-isolation footgun)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _params(seed=0):
    k = jax.random.PRNGKey(seed)
    kw, kb = jax.random.split(k)
    return {"weight": jax.random.normal(kw, (4, 3)), "bias": jax.random.normal(kb, (4,))}


# --------------------------------------------------------------------------------------------
# Bare md_decouple transform — plain optax loop
# --------------------------------------------------------------------------------------------


def test_init_step_is_identity_with_zero_lrs():
    """At init γ=1 and Ŵ=W, so a step with both LRs at 0 reproduces W exactly (zero update)."""
    p = _params()
    opt = md_decouple(optax.sgd(0.0), gain_lr=0.0)
    state = opt.init(p)
    g = {"weight": jnp.ones((4, 3)), "bias": jnp.ones((4,))}
    updates, _ = opt.update(g, state, p)
    assert float(jnp.max(jnp.abs(updates["weight"]))) < 1e-12
    assert float(jnp.max(jnp.abs(updates["bias"]))) < 1e-12


def test_sphere_norm_is_preserved_after_a_step():
    """After a real step the direction stays pinned to its initialization Frobenius norm c."""
    p = _params()
    opt = md_decouple(optax.sgd(0.3), gain_lr=0.0)  # gains frozen -> W_new == Ŵ_new (γ=1)
    state = opt.init(p)
    c = float(state.sphere_c["weight"])
    g = {"weight": jax.random.normal(jax.random.PRNGKey(1), (4, 3)), "bias": jnp.zeros((4,))}
    updates, _ = opt.update(g, state, p)
    w_new = p["weight"] + updates["weight"]  # == Ŵ_new since γ=1
    assert float(jnp.linalg.norm(w_new)) == pytest.approx(c, rel=1e-9)


def test_bias_and_1d_leaves_pass_through_base_optimizer():
    """Non-2-D leaves get the plain base step, untouched by the MD machinery."""
    p = _params()
    eta = 0.1
    opt = md_decouple(optax.sgd(eta), gain_lr=1e-2)
    state = opt.init(p)
    g = {"weight": jax.random.normal(jax.random.PRNGKey(2), (4, 3)), "bias": jnp.array([1.0, -2.0, 3.0, 0.5])}
    updates, _ = opt.update(g, state, p)
    np.testing.assert_allclose(np.asarray(updates["bias"]), -eta * np.asarray(g["bias"]), atol=1e-12)


def test_decoupled_run_minimizes_a_scale_invariant_objective():
    """Fitting only the direction of a target matrix (a scale-invariant loss) converges."""
    p = _params(3)
    target = jax.random.normal(jax.random.PRNGKey(9), (4, 3))
    target = target / jnp.linalg.norm(target)

    def loss(pp):
        wd = pp["weight"] / jnp.linalg.norm(pp["weight"])
        return jnp.sum((wd - target) ** 2)

    opt = md_decouple(optax.adam(1e-1), gain_lr=1e-2)
    state = opt.init(p)
    l0 = float(loss(p))
    for _ in range(200):
        g = jax.grad(loss)(p)
        updates, state = opt.update(g, state, p)
        p = optax.apply_updates(p, updates)
    assert float(loss(p)) < l0 * 1e-3


@pytest.mark.parametrize("gain_axis", [("row", "col"), ("row",), ("col",)])
def test_gain_axis_variants_build_and_step(gain_axis):
    p = _params()
    opt = md_decouple(optax.sgd(0.05), gain_lr=1e-2, gain_axis=gain_axis)
    state = opt.init(p)
    g = {"weight": jnp.ones((4, 3)), "bias": jnp.ones((4,))}
    updates, _ = opt.update(g, state, p)
    assert updates["weight"].shape == (4, 3)
    assert jnp.all(jnp.isfinite(updates["weight"]))


# --------------------------------------------------------------------------------------------
# Deferred-feature / misuse guards
# --------------------------------------------------------------------------------------------


def test_scalar_gain_axis_is_deferred():
    with pytest.raises(NotImplementedError, match="scalar"):
        md_decouple(optax.sgd(0.1), gain_axis="scalar")


def test_non_softplus_gain_map_is_deferred():
    with pytest.raises(NotImplementedError, match="softplus"):
        md_decouple(optax.sgd(0.1), gain_map="exp")


def test_md_sentinel_rejects_non_gradient_transformation_base():
    with pytest.raises(TypeError, match="GradientTransformation"):
        md("not-an-optimizer")


# --------------------------------------------------------------------------------------------
# md sentinel — end-to-end through crux.solve
# --------------------------------------------------------------------------------------------


def _poisson_solver(seed=0):
    """1-D Poisson u''=sin(πx) with a hard-BC ansatz; a net with real 2-D weight matrices."""
    domain = jno.domain.line(mesh_size=0.05)
    x, _ = domain.variable("interior")
    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(seed)))
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)
    return jno.core([pde.mse]), u_net


def test_md_sentinel_trains_through_crux_solve():
    """The md() sentinel drives a real net: host builds the engine, forces scale(1.0), loss drops."""
    solver, u_net = _poisson_solver()
    u_net.optimizer(md(optax.adam(3e-3), gain_lr=1e-3))
    stats = solver.solve(400)
    losses = stats.training_logs[-1]["total_loss"]
    assert jnp.isfinite(losses[-1])
    assert float(losses[-1]) < 0.5 * float(losses[0])  # MD made real progress


def test_md_sentinel_is_not_mutated_and_re_solves():
    """fm._opt_fn stays the sentinel across solves (idempotent host hook, no write-back)."""
    solver, u_net = _poisson_solver(seed=1)
    u_net.optimizer(md(optax.adam(3e-3), gain_lr=1e-3))
    solver.solve(50)
    assert isinstance(u_net._opt_fn, MDOptimizer)  # unchanged after the first solve
    solver.solve(50)  # second solve must still run
    assert isinstance(u_net._opt_fn, MDOptimizer)
