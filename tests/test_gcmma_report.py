"""GCMMA's summary names what held its steps back: the rows its approximation kept underestimating when max_inner ran
out, and how many re-solves a step took. Without the names a run that pays ten re-solves a step says only "N of K
steps ran out", and the row to reformulate (a kink, a sharp soft maximum) has to be found by hand."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.utils.logger as jlog


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_the_summary_names_the_rows_that_kept_a_step_from_being_conservative():
    x = jno.np.parameter((1,), name="x")
    x.dtype(jnp.float64)
    x.initialize(jax.nn.initializers.constant(0.0))
    x.optimizer(
        jno.optimizers.gcmma(move=0.2, lower=-2.0, upper=2.0, max_inner=0)
    )  # no re-solve: every step with an optimistic row runs out
    f0 = jno.fn(lambda v: -jnp.sum(v), [x], name="f0")  # pushes x up
    # no slope at the start and strongly curved: the first-order model misses its whole change there
    bowl = jno.fn(lambda v: 50.0 * jnp.sum(v**2), [x], name="bowl")
    crux = jno.core([f0, jno.le(bowl, 10.0).name("the bowl")], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    said = []

    class _Log(jlog.PrintFallback):
        def warning(self, message):
            said.append(message)

    prev, jlog._default_logger = jlog._default_logger, _Log()
    try:
        crux.solve(3)
    finally:
        jlog._default_logger = prev
    summary = [m for m in said if "steps ran out of max_inner" in m]
    assert len(summary) == 1, said
    assert "Held back by: the bowl" in summary[0], summary[0]
    assert "re-solves per step" in summary[0], summary[0]
