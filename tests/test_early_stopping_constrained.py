"""Early stopping on a constrained problem stops only at a feasible design.

``early_stopping`` watches the total loss, which with ``jno.le`` entries is the objective alone.
While MMA pulls an infeasible design back into the feasible set, the objective RISES, so "no
improvement for ``patience`` steps" fired in the middle of that restoration. The solve then
returned a design over its constraint budget with nothing said: on the problem below, three times
the budget after three steps. The rule now: only a feasible iterate can be an improvement or count
towards ``patience``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import pytest

import jno

sys.path.insert(0, str(Path(__file__).parent))
import test_optimizers_gcmma as gc  # noqa: E402  (the module, so pytest does not collect its classes here)
from test_topology_optimisation import VOLFRAC, _cantilever  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("patience", [2, 5])
def test_mma_is_not_stopped_while_restoring_feasibility(patience):
    """``sum x**4 <= R``: plain MMA's first steps overshoot it by up to 9x, then come back."""
    R = gc.R
    d, x, f0, g = gc.TestThroughCore._build(gc._c(), jno.optimizers.mma(move=0.2, lower=gc.XLO, upper=gc.XHI))
    es = jno.callbacks.early_stopping(patience=patience, verbose=False)
    crux = jno.core([f0, jno.le(g, R)], domain=d)
    crux.solve(150, callbacks=[es])
    xf = np.asarray(crux.eval([x])).reshape(-1)

    assert es.has_stopped, "the run converges well inside 150 steps, so patience must still fire"
    # MMA's own chatter at the optimum is round-off; an early stop mid-restoration is O(1).
    assert float(np.sum(xf**4)) - R <= 1e-6, f"stopped at an infeasible design: g = {np.sum(xf**4) - R:+.3e}"


def test_an_unconstrained_run_stops_exactly_as_before():
    """No ``jno.le`` rows: the gate must not exist, so the stop epoch is the plain patience rule's."""
    import optax

    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    x = jno.np.parameter((1,), name="x")
    x.initialize(jax.nn.initializers.constant(1.0))
    x.optimizer(optax.sgd(0.0))  # the loss never moves, so patience fires after exactly `patience` steps
    es = jno.callbacks.early_stopping(patience=4, verbose=False)
    jno.core([jno.fn(lambda v: v[0] ** 2, [x], name="f")], domain=d).solve(50, callbacks=[es])
    assert es.stopped_epoch == 4


def test_it_stops_a_fem_topology_optimisation_at_a_feasible_design():
    """Through a differentiable ``fem.solve()``: the stop comes early and the volume budget holds."""
    crux, rho, _ = _cantilever()
    es = jno.callbacks.early_stopping(patience=3, min_delta=1e-3, mode="rel", verbose=False)
    crux.solve(200, callbacks=[es])
    rho_f = np.asarray(crux.eval([rho])).reshape(-1)

    assert es.has_stopped and es.stopped_epoch < 199, "a 200-step budget must be cut short"
    assert rho_f.mean() <= VOLFRAC + 1e-6
