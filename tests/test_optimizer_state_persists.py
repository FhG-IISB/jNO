"""A second ``solve()`` continues the first, rather than restarting the optimizer.

optax carries its step count inside the optimizer state. Re-initialising that state on every
``solve()`` restarts every optax schedule, so a loop that trains in chunks -- to checkpoint, to
diagnose, to advance a curriculum -- never leaves the first few steps of its schedule. Nothing
raises; the run just trains at the wrong rate for its whole life.

jNO's own :class:`LearningRateSchedule` is immune already: it reads the persistent
``self._total_epochs``. These tests are about the optax side.
"""

import foundax
import jax
import numpy as np
import optax
import pytest

import jno


def _fit(chunks, epochs, lr_schedule):
    """Train `sum(chunks)` epochs, split as given, and return the final parameter."""
    # A MESHED domain, so `variable("interior")` is the fixed node set. A mesh-free domain draws
    # fresh points on every call, and that alone changes the answer between runs -- which would
    # mask exactly the effect these tests measure.
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.35).domain()
    x, y, _ = d.variable("interior", split=True)
    net = jno.nn(foundax.mlp(in_features=2, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(0)))
    net.optimizer(optax.adam(lr_schedule))
    u = net(x, y).scalar
    crux = jno.core([(u - 1.0).mse])
    for n in chunks:
        crux.solve(n)
    return float(np.asarray(crux.eval([(u - 1.0).mse])[0]).ravel()[0])


def test_a_chunked_run_matches_one_long_run():
    """The headline contract: HOW the epochs are split must not change what training does.

    Under a decaying schedule a restart holds the rate at its initial value, so a chunked run
    trains FASTER than it should -- the two losses come out different, and neither is a warning.
    """
    sched = optax.exponential_decay(1e-2, transition_steps=40, decay_rate=0.3, end_value=1e-6)
    one = _fit([200], 200, sched)
    many = _fit([50, 50, 50, 50], 200, sched)
    assert one == pytest.approx(many, rel=1e-4), (
        f"splitting 200 epochs into 4 chunks changed the result: {one:.6e} vs {many:.6e}"
    )


def test_a_warmup_boundary_is_crossed_by_chunked_training():
    """A schedule that is ZERO until step N must still release when the epochs arrive in chunks.

    This is the shape of a curriculum -- hold a parameter while something else is solved, then let
    it move. With the state restarted per call, and chunks shorter than the boundary, the release
    never happens and the held parameter stays at its initial value forever.
    """
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.35).domain()
    x, y, _ = d.variable("interior", split=True)
    net = jno.nn(foundax.mlp(in_features=2, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(1)))
    held = jno.np.parameter((1,), key=jax.random.PRNGKey(2), name="held")

    frozen_then_free = optax.join_schedules([optax.constant_schedule(0.0), optax.constant_schedule(1e-1)], boundaries=[100])
    net.optimizer(optax.adam(1e-3))
    held.optimizer(optax.adam(frozen_then_free))

    crux = jno.core([(net(x, y).scalar + held[0] - 1.0).mse])
    read = lambda: float(np.asarray(crux.eval([held])[0]).ravel()[0])  # noqa: E731

    start = read()
    for _ in range(4):  # 4 x 40 = 160 epochs, crossing the boundary at 100
        crux.solve(40)
    assert abs(read() - start) > 1e-3, "the held parameter never moved: the boundary was never crossed"


def test_a_fresh_core_starts_the_optimizer_over():
    """The escape hatch. States live on the core, so a new one is a genuine restart -- which is
    what makes the carry-over safe to have by default."""
    sched = optax.exponential_decay(1e-2, transition_steps=40, decay_rate=0.3, end_value=1e-6)
    a = _fit([200], 200, sched)
    b = _fit([200], 200, sched)
    assert a == pytest.approx(b, rel=1e-9), "two independent cores must train identically"


def test_changing_the_optimizer_reinitialises_rather_than_reusing():
    """A carried state is reused only when it FITS. Swap the optimizer for one with a different
    state and the fresh one is taken, instead of a shape error deep in the update."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.35).domain()
    x, y, _ = d.variable("interior", split=True)
    net = jno.nn(foundax.mlp(in_features=2, hidden_dims=8, num_layers=2, key=jax.random.PRNGKey(3)))
    net.optimizer(optax.adam(1e-3))
    u = net(x, y).scalar
    crux = jno.core([(u - 1.0).mse])
    crux.solve(20)

    net.optimizer(optax.sgd(1e-2))  # different state tree
    crux2 = jno.core([(u - 1.0).mse])
    crux2.solve(20)  # must not raise
    assert np.isfinite(float(np.asarray(crux2.eval([(u - 1.0).mse])[0]).ravel()[0]))
