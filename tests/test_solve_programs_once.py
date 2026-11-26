"""A solve again of the same problem compiles nothing: the step and every program made with it (trackers, MMA's
constraint Jacobian, GCMMA's value program) live in a program set kept per problem structure. Before, each `solve`
made those as new closures and compiled them again, which on a FEM problem costs several steps' worth per solve."""

import logging
import sys

import jax
import jax.numpy as jnp
import numpy as np
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


def _param(name, v=0.0):
    p = jno.np.parameter((1,), name=name)
    p.dtype(jnp.float64)
    p.initialize(jax.nn.initializers.constant(v))
    return p


def _one():
    return jno.domain.from_array({"_": np.zeros((1, 1))})


def _fem_problem(opt):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    k = jno.np.parameter(phi, name="k")
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(getattr(jno.optimizers, opt)(move=0.05, lower=0.5, upper=2.0))
    sol = jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0]).solve()
    mean_u = ui.freeze(sol).integrate(d)
    return jno.core([mean_u, jno.le(k.mean, 1.2).name("k mean"), (2.0 * mean_u).name("twice").tracker(1)])


@pytest.mark.parametrize("opt", ["mma", "gcmma"])
def test_a_solve_again_compiles_nothing(opt):
    crux = _fem_problem(opt)
    crux.solve(2)
    crux.solve(2)  # a carried optimizer state's copy compiles once more
    names = _compiled(lambda: crux.solve(3))
    assert names == [], f"a solve again compiled {len(names)} programs: {sorted(set(names))}"


def test_alternating_stages_compile_each_stage_once():
    """freeze a / train b, then the other way round, then again: the second round finds both stages' programs (the
    core keeps a program set per problem structure, a few at most), instead of compiling at every switch."""
    a, b = _param("a"), _param("b")
    for p_ in (a, b):
        p_.optimizer(jno.optimizers.mma(move=0.1, lower=-2.0, upper=2.0))
    f = jno.fn(lambda u, v: jnp.sum((u - 1.0) ** 2 + (v + 1.0) ** 2), [a, b], name="f")
    crux = jno.core([f, jno.le(jno.fn(lambda u: jnp.sum(u), [a], name="ga"), 0.8)], domain=_one())

    def round_():
        for which in (a, b):
            which.freeze()
            crux.solve(3)
            which.unfreeze()

    round_()
    round_()  # the optimizer states carried in both stages once
    names = _compiled(round_)
    assert names == [], f"switching stages compiled {len(names)} programs again: {sorted(set(names))}"


def test_a_reused_step_that_compiles_anyway_is_reported(monkeypatch):
    """The guard: a solve that reuses its compiled step yet spends time compiling is a program made anew per solve.
    A callback that jits a fresh function every step stands in for one."""
    import jno.utils.logger as jlog
    from jno.utils.adaptive.callbacks import Callback

    jcore = sys.modules["jno.core"]  # `jno.core` the attribute is the class
    monkeypatch.setattr(jcore, "RECOMPILE_REPORT_SECONDS", 0.0)
    a = _param("a", 1.0)
    a.optimizer(jno.optimizers.mma(move=0.1, lower=-2.0, upper=2.0))
    crux = jno.core(
        [jno.fn(lambda v: jnp.sum(v**2), [a], name="f"), jno.le(jno.fn(jnp.sum, [a], name="g"), 1.5)], domain=_one()
    )
    crux.solve(2)

    class Fresh(Callback):
        def on_epoch_end(self, **kw):
            jax.jit(lambda v: v * 2.0)(1.0)  # a new function each time: traced and compiled each time

    said = []

    class _Log(jlog.PrintFallback):
        def warning(self, message):
            said.append(message)

    crux.log = _Log()
    crux.solve(2, callbacks=[Fresh()])
    assert any("compiled" in m and "again" in m for m in said), said
