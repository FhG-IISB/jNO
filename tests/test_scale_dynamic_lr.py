"""`.scale(dlrs(...))` dynamically adapts the learning rate during training.

The optimizer/LR cleanup renamed ``.lr`` -> ``.scale`` and removed ``.optimizer(lr=)``.
This guards the one property that had to be preserved: a loss-adaptive DLRS attached via
``.scale`` is still invoked every step and actually *moves* the effective learning rate
(``state.hyperparams["learning_rate"]`` is refreshed from the schedule each step in core).
"""

from __future__ import annotations

import jax
import optax
import pytest

import jno
import jno.jnp_ops as jnn

pytest.importorskip("foundax")
import foundax  # noqa: E402


def test_scale_dlrs_moves_lr_during_training():
    domain = 1 * jno.domain.line(mesh_size=0.02)
    x, _ = domain.variable("interior")
    u_net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=16, num_layers=2, key=jax.random.PRNGKey(0)))
    u = u_net(x) * x * (1 - x)
    pde = jnn.laplacian(u, [x]) - jnn.sin(jnn.pi * x)

    sched = jno.fn.adaptive.dlrs(lr0=1e-3, window=2)
    # optax carries a placeholder rate of 1; `.scale` supplies the (dynamic) effective LR
    u_net.optimizer(optax.adam(1)).scale(sched)

    jno.core([pde.mse]).solve(8)

    # DLRS was wired in by `.scale` and called every step (host-side state advanced)...
    assert sched.initialized, ".scale(dlrs()) was not invoked during training"
    assert len(sched.loss_hist) > 1, "DLRS did not receive a per-step loss stream"
    # ...and it dynamically changed the learning rate off its starting lr0.
    assert float(sched.lr) != pytest.approx(1e-3), "effective LR never changed — not dynamic"


def test_scale_is_a_multiplier_on_the_optimizer_rate():
    """A static `.scale(c)` multiplies a pre-built optimizer's rate (the optax.scale path)."""
    net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(1e-3)).scale(0.5)
    # `.scale` stores onto the same internal slot the old `.lr` used
    assert float(net._lr) == 0.5


def test_optimizer_no_longer_accepts_lr_kwarg():
    """The removed `lr=` shorthand must raise rather than silently no-op."""
    net = jnn.nn.wrap(foundax.mlp(1, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(0)))
    with pytest.raises(TypeError):
        net.optimizer(optax.adam, lr=1e-3)
    assert not hasattr(net, "lr")  # the `.lr` method is gone
