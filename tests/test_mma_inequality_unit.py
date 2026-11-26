"""A constrained step computes the objective's gradient and the bound rows' Jacobian in one program only when they
share work (a solve, an assembly). Rows that read nothing but the design take their own unit: the joint program would
pull back through the objective's tape once per row with a zero cotangent, for nothing."""

import logging

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


def _problem(row_reads_the_solve):
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
    row = jno.le(10.0 * mean_u, 1.0).name("u bound") if row_reads_the_solve else jno.le(k.mean, 1.2).name("k mean")
    return jno.core([mean_u, row])


@pytest.mark.parametrize("shares", [False, True])
def test_the_joint_program_is_used_only_when_the_rows_share_the_objectives_work(shares):
    crux = _problem(shares)
    names = _compiled(lambda: crux.solve(2))
    assert crux._ineq_shares_objective is shares
    joint = any("grad_jac_fn" in n for n in names)
    assert joint is shares, sorted(set(names))


def test_both_programs_take_the_same_steps(monkeypatch):
    """the choice is a cost choice: the losses of a few steps are the same either way"""
    import sys

    own = _problem(False)
    own.solve(4)
    monkeypatch.setattr(sys.modules["jno.core"], "_rows_share_work", lambda rows, objectives: True)
    joint = _problem(False)
    joint.solve(4)
    assert joint._ineq_shares_objective is True
    np.testing.assert_allclose(own.training_logs[-1]["losses"], joint.training_logs[-1]["losses"], rtol=1e-10)
