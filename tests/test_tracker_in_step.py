"""A tracker (``expr.tracker(k)`` in the core's list) is evaluated in the step's own forward pass, with the rows: it
shares the step's solves and needs no program of its own. Before, every solve compiled a second program
(``track_fn``) that ran the forward pass again -- as long to compile as the step itself on a FEM problem -- and
evaluated it at the design after the update, one step off the losses logged beside it."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


class _Compiles(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.names = []

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("Compiling "):
            self.names.append(msg.split(" with ")[0])


def _compiled(run):
    h = _Compiles()
    loggers = [logging.getLogger(n) for n in ("jax._src.dispatch", "jax._src.interpreters.pxla")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.DEBUG)
        lg.addHandler(h)
    prev = jax.config.jax_log_compiles
    jax.config.update("jax_log_compiles", True)
    try:
        run()
    finally:
        jax.config.update("jax_log_compiles", prev)
        for lg, lv in zip(loggers, levels):
            lg.removeHandler(h)
            lg.setLevel(lv)
    return h.names


def _param(v):
    x = jno.np.parameter((1,), name="x")
    x.dtype(jnp.float64)
    x.initialize(jax.nn.initializers.constant(v))
    return x


def _one():
    return jno.domain.from_array({"_": np.zeros((1, 1))})


@pytest.mark.parametrize("opt", ["adam", "mma", "gcmma"])
def test_a_tracker_rides_in_the_step_and_reads_the_design_its_losses_read(opt):
    x = _param(1.0)
    if opt == "adam":
        x.optimizer(optax.adam(0.1))
    else:
        x.optimizer(getattr(jno.optimizers, opt)(move=0.2, lower=-5.0, upper=5.0))
    f0 = jno.fn(lambda v: (jnp.sum(v) - 3.0) ** 2, [x], name="f0")
    rows = [f0] if opt == "adam" else [f0, jno.le(jno.fn(lambda v: jnp.sum(v) ** 2, [x], name="g"), 4.0)]
    crux = jno.core(rows + [(2.0 * f0).name("twice").tracker(1)], domain=_one())
    names = _compiled(lambda: crux.solve(6))
    assert not any("track_fn" in n for n in names), f"the trackers compiled a program of their own: {names}"
    logs = crux.training_logs[-1]
    tracked = np.asarray(logs["track_stats"]).reshape(len(logs["epoch"]), -1)[:, 0]
    # the tracker of a step is evaluated at the design its rows were: twice the objective row of that step
    np.testing.assert_allclose(tracked, 2.0 * np.asarray(logs["losses"])[:, 0], rtol=1e-12)


def test_a_tracker_shares_the_steps_solve():
    """The FEM solve a tracker reads is the rows' solve: one solve per step, not one more for the trackers."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    k = jno.np.parameter(phi, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(jno.optimizers.mma(move=0.05, lower=0.5, upper=2.0))
    traced = []

    def load(v):
        traced.append(1)
        return v

    src = jno.fn(load, [1.0 + 0.0 * xi], name="load")
    sol = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - src * vi, u(xb, yb) - 0.0]).solve()
    mean_u = ui.freeze(sol).integrate(d)
    crux = jno.core([mean_u, jno.le(k.mean, 1.2).name("k mean"), (2.0 * mean_u).name("twice").tracker(1)])
    crux.solve(2)
    crux.solve(2)  # a carried optimizer state's copy compiles once more
    traced.clear()
    names = _compiled(lambda: crux.solve(3))
    assert names == [], f"a solve again compiled {sorted(set(names))}"
    crux2 = jno.core([mean_u, jno.le(k.mean, 1.2).name("k mean")])
    traced.clear()
    crux2.solve(1)
    alone = len(traced)
    traced.clear()
    crux3 = jno.core([mean_u, jno.le(k.mean, 1.2).name("k mean"), (2.0 * mean_u).name("twice").tracker(1)])
    crux3.solve(1)
    assert len(traced) == alone, f"the tracker traced the load {len(traced)} times, the rows alone {alone}"


def test_a_tracker_with_no_samples_rides_beside_rows_that_have_them():
    """A row reads the mesh's points (a sample batch); the tracker is an integral (none): the step maps over the batch,
    and the tracker, the same in every sample, keeps the value and the shape it has alone."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    k = jno.np.parameter(phi, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(jno.optimizers.mma(move=0.05, lower=0.5, upper=2.0))
    sol = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0]).solve()
    mean_u = ui.freeze(sol).integrate(d)
    reach = jno.fn(lambda v: jnp.max(v), [xi], name="reach")  # reads the interior points: a sample batch
    crux = jno.core([mean_u, jno.le(k.mean, 1.2).name("k mean"), jno.le(reach, 2.0), (2.0 * mean_u).name("t").tracker(1)])
    names = _compiled(lambda: crux.solve(3))
    assert not any("track_fn" in n for n in names), "the tracker compiled a program of its own"
    logs = crux.training_logs[-1]
    tracked = [np.asarray(v) for row in np.asarray(logs["track_stats"], dtype=object) for v in np.atleast_1d(row)]
    assert all(v.shape == () for v in tracked), [v.shape for v in tracked]
    np.testing.assert_allclose(np.asarray(tracked, float), 2.0 * np.asarray(logs["losses"])[:, 0], rtol=1e-12)
