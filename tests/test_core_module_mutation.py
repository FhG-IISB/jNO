"""A model's module swapped after ``jno.core(...)`` was built — the core keeps its own copy.

``jno.core`` reads ``layer.module`` once, at construction, and everything after that runs on the
core's copy: training writes it, ``eval`` reads it. The eager idiom that a bare
``jno.fem([...]).solve()`` honours,

    g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray(vals))

is therefore invisible to a core built before it — and the sweep it is meant to drive read the same
number at every value, silently. It is refused by name now.

Honouring the live module instead would be wrong in the other direction: after ``solve()`` the core's
copy holds the TRAINED weights while the module still holds the ones it was built from, so re-reading
would quietly undo the training. Both halves are pinned below.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


def _poisson(size=0.25):
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return d, u, v, ui, vi, (xb, yb)


def _set(g, val):
    g.model.module = eqx.tree_at(lambda m: m.value, g.model.module, jnp.asarray([float(val)]))


def test_eval_refuses_a_module_swapped_after_the_core_was_built():
    """The reported shape: a parametric ``fem.solve()`` evaluated through a core, swept by mutation.
    Both values used to come back equal to the last bit — the value at construction."""
    d, u, v, ui, vi, (xb, yb) = _poisson()
    g = jno.np.parameter((1,), name="g", key=jax.random.PRNGKey(0))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
    crux = jno.core([fem.solve().mse], domain=d)

    _set(g, 5.0)
    with pytest.raises(RuntimeError, match=r"crux\.eval: g .*after this core was built"):
        crux.eval([fem.solve()])


def test_the_refusal_names_the_parameter_and_the_two_paths_that_work():
    d, u, v, ui, vi, (xb, yb) = _poisson()
    g = jno.np.parameter((1,), name="amplitude", key=jax.random.PRNGKey(0))
    crux = jno.core([(g * 1.0 - 3.0).mse], domain=d)
    _set(g, 7.0)
    with pytest.raises(RuntimeError) as e:
        crux.eval([g * 1.0])
    msg = str(e.value)
    assert "amplitude" in msg, "the refusal must name the parameter that moved"
    assert "jno.core" in msg and "solve()" in msg, "the refusal must name a path that works"


def test_a_value_set_before_the_core_is_what_eval_reads():
    """The supported spelling: the module is read at construction, so set it first. This is also the
    control for the test above — without it, a core that ignored the parameter entirely would pass."""
    d, u, v, ui, vi, (xb, yb) = _poisson()
    g = jno.np.parameter((1,), name="g", key=jax.random.PRNGKey(0))
    _set(g, 7.0)
    crux = jno.core([(g * 1.0).mse], domain=d)
    got = np.asarray(crux.eval([g * 1.0])).reshape(-1)
    assert np.allclose(got, 7.0), f"the core did not read the value it was built from: {got}"


def test_a_swept_parameter_is_read_per_core():
    """One core per value — the shape the refusal points at — does respond, and monotonically: the
    Dirichlet value IS the solution's level, so the mean scales with g."""
    d, u, v, ui, vi, (xb, yb) = _poisson()

    def mean_at(val):
        g = jno.np.parameter((1,), name="g", key=jax.random.PRNGKey(0))
        _set(g, val)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g])
        crux = jno.core([fem.solve().mse], domain=d)
        return float(np.asarray(crux.eval([fem.solve()])).reshape(-1).mean())

    a, b = mean_at(1.0), mean_at(5.0)
    assert b - a > 3.0, f"the swept parameter did not move the solution: {a:.6f} -> {b:.6f}"


def test_training_does_not_trip_the_guard():
    """``solve()`` writes the core's copy and syncs it back onto the module — that sync IS the core
    reading the module again, not a swap behind its back, so the guard must stay quiet after it."""
    import optax

    d, u, v, ui, vi, (xb, yb) = _poisson()
    g = jno.np.parameter((1,), name="g", key=jax.random.PRNGKey(0))
    _set(g, 0.0)
    g.optimizer(optax.adam(1e-1))
    crux = jno.core([(g * 1.0 - 3.0).mse], domain=d)
    crux.solve(200)

    trained = np.asarray(crux.eval([g * 1.0])).reshape(-1)  # must not raise
    assert abs(float(trained[0]) - 3.0) < 0.2, f"training was not visible to eval: {trained}"
    synced = float(np.asarray(g.model.module.value).reshape(-1)[0])
    assert abs(synced - float(trained[0])) < 1e-9, f"solve() left the module out of sync: {synced} vs {trained[0]}"


def test_the_eager_solve_path_still_reads_the_module_live():
    """The domain-decomposition idiom: mutate the neighbour field, re-solve, no core in the path. The
    refusal above must not reach it — this is the loop the mutation exists for."""
    d, u, v, ui, vi, (xb, yb) = _poisson(size=0.3)
    g = jno.np.parameter((1,), name="g", key=jax.random.PRNGKey(0))
    form = [ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - g]

    _set(g, 1.0)
    lo = np.asarray(jno.fem(form).solve()).reshape(-1)
    _set(g, 5.0)
    hi = np.asarray(jno.fem(form).solve()).reshape(-1)
    assert hi.mean() - lo.mean() > 3.0, f"the eager solve stopped reading the module: {lo.mean()} -> {hi.mean()}"
